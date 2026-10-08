"""Public-only bounded replay diagnostic for the actual client particle adapter.

No teacher snapshots, opponent recipes inferred from truth, or private response
windows enter this module. A tape contains one seat's received bytes and own
recorded decisions. Replayed-public is deliberately not called captured-wire.
This does not admit online search or compare hypothetical futures to truth.
"""
from __future__ import annotations

from collections import Counter

from dataclasses import asdict, replace
import hashlib
import math
import struct
import time

from ..netduel import constants as C
from ..netduel.actions import parse_select
from ..netduel.client import NetDuelClient
from ..netduel.wmplay import snapshot_from_shadow
from ..puzzle.single import RESPONSE_REQUIRED
from ..search.banish_origin import BanishOriginTracker
from ..search.belief import ledger_claims
from ..worldmodel.engine import DeckList
from ..worldmodel.state import public_view
from .client_continuity import CONTINUED_LAW
from .client_hidden_target import root_claims
from .client_history_registry import PROFILE as HISTORY_PROFILE
from .client_particle_adapter import (ClientParticleAdapter, ParticleRefused, facedown_field_claims,
                                     merge_field_claims, monster_sset_codes)
from .client_root import ClientRootController, RootTimeout
from .client_response_classes import response_class_bank
from .client_root_menu import RootMenuBinding
from .client_shadow import BlankClientSync
from .sidecar_io import canonical, digest
from .stage_a_joint_belief import JointRecipe
from .stage_a_joint_belief_runtime import JointParticleBank, belief_joint_bank, uniform_joint_bank

INFORMATION_SET_SEARCH = True
SCHEMA = "client-midgame-public-tape/v1"
SELECTION_LAW = "public-first-prompt-after-event/v1"
_FIELDS = {"schema", "kind", "viewer", "own_main", "own_extra", "public_prior",
           "opponent_counts", "rules", "command_selections", "packets_hex", "prompts",
           "winner", "source"}
_TRIGGERS = {C.MSG_CHAIN_END: "chain_end", C.MSG_SHUFFLE_SET_CARD: "set_shuffle",
             C.MSG_DRAW: "draw", C.MSG_CONFIRM_CARDS: "public_confirm",
             C.MSG_CONFIRM_DECKTOP: "decktop_confirm"}


def validate_tape(tape):
    """Closed boundary: provenance is hashes, not a privileged metadata bag."""
    if set(tape) != _FIELDS or tape["schema"] != SCHEMA or tape["kind"] != "replayed-public":
        raise ValueError("unknown public tape schema/fields")
    if type(tape["viewer"]) is not int or tape["viewer"] not in (0, 1):
        raise ValueError("invalid public observer")
    prior = tape["public_prior"]
    if set(prior) != {"authority", "main", "extra"} or prior["authority"] != "public_fixed_room_recipe":
        raise ValueError("opponent hypothesis recipe needs an explicit public room prior")
    if set(tape["source"]) != {"game_sha256", "replay_sha256", "core_sha256", "cards_db_sha256", "scripts_sha256"}:
        raise ValueError("source must contain only frozen hashes")
    for sha in tape["source"].values():
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("invalid source hash")
    if set(tape["rules"]) != {"start_lp", "start_hand", "draw_count", "duel_options"} \
            or any(type(v) is not int for v in tape["rules"].values()) \
            or type(tape["command_selections"]) is not bool:
        raise ValueError("invalid public rules")
    for deck in (tape["own_main"], tape["own_extra"], prior["main"], prior["extra"]):
        if not isinstance(deck, list) or any(type(c) is not int or c <= 0 for c in deck):
            raise ValueError("invalid entitled/public recipe")
    if tape["opponent_counts"] != [len(prior["main"]), len(prior["extra"])]:
        raise ValueError("public recipe and public counts differ")
    packets = tuple(bytes.fromhex(p) for p in tape["packets_hex"])
    if not packets or packets[0][0] != C.MSG_START or any(not p for p in packets):
        raise ValueError("public replay must begin with START")
    actual = [i for i, p in enumerate(packets) if p[0] in RESPONSE_REQUIRED]
    if actual != [r["packet"] for r in tape["prompts"]]:
        raise ValueError("prompt ledger dropped, duplicated or reordered a received private window")
    for row in tape["prompts"]:
        if set(row) != {"packet", "response_hex", "choices"}:
            raise ValueError("invalid own prompt record")
        packet = packets[row["packet"]]
        if packet[2 if packet[0] == C.MSG_SELECT_SUM else 1] != tape["viewer"]:
            raise ValueError("other player's private prompt in public tape")
        response = bytes.fromhex(row["response_hex"])
        if not 0 < len(response) <= 255:
            raise ValueError("invalid own response")
        for choice in row["choices"]:
            if set(choice) != {"msg", "menu_size", "round_index", "chosen_index"} \
                    or any(type(v) is not int for v in choice.values()):
                raise ValueError("invalid original selector path")
    return packets


def scheduled_roots(tape):
    """Select solely from received events; no successful-prefix filtering."""
    packets = validate_tape(tape)
    pending, roots, turn = [], [], 0
    prompts = {r["packet"]: i for i, r in enumerate(tape["prompts"])}
    for index, packet in enumerate(packets):
        msg, body = packet[0], packet[1:]
        if msg == C.MSG_NEW_TURN:
            turn += 1
            pending.append((index, "turn_start"))
        if msg in _TRIGGERS:
            pending.append((index, _TRIGGERS[msg]))
        if msg == C.MSG_CHAINING and len(body) >= 4 and struct.unpack_from("<I", body)[0] in (10045474, 24224830):
            pending.append((index, "persistent_card_activation"))
        if index in prompts and pending:
            roots.append({"prompt": prompts[index], "packet": index, "turn": turn,
                          "events": [{"packet": p, "kind": k} for p, k in pending]})
            pending.clear()
    return roots


def with_ledger_slots(snapshot, viewer: int, ledger):
    """A client's reconstruction of the board (``snapshot_from_shadow``) with the identity the seat's ledger proved at
    each field slot its packets left blank: a card set face down from a public zone, or confirmed while face down, is
    known to the client though the move that put it there hid its code. An engine capture holds every code, so
    ``public_view`` keeps such a card at its slot there; this gives the client's the same."""
    cards = list(snapshot.cards)
    for index, card in enumerate(cards):
        if card.location not in (C.LOCATION_MZONE, C.LOCATION_SZONE):
            continue
        code = ledger.known_slots(viewer, card.controller, card.location).get(card.sequence)
        if code is None or card.code == code:
            continue
        if card.code:
            raise ValueError("the client's board holds %d at %s where the ledger proved %d"
                             % (card.code, (card.controller, card.location, card.sequence), code))
        cards[index] = replace(card, code=code)
    return replace(snapshot, cards=cards)


def client_public_view(client):
    """The seat's public view of the board a network client holds (its shadow board, completed by its ledger), as
    root admission reads it."""
    if client.board_problems:
        raise ValueError("public board drift: " + str(client.board_problems[-1]))
    viewer = client.result.our_player
    full = snapshot_from_shadow(client.board, viewer, tuple(client._lp), client.result.turns, client.card_pool)
    full = with_ledger_slots(replace(full, view=None, revealed=frozenset()), viewer, client.board.disclosure)
    return public_view(full, viewer, client.board.disclosure)


def admit_root(root, client, prior, *, proposals, seed, context, extra_origins, adapter_options=None, belief=None,
               response_classes=False, continued=None, class_allocation="largest_remainder", direct_ar_proposal=None):
    """The particle adapter at a captured root: the public view, the claims every particle must honor (the ledger's,
    the follower's category facts still true there, what each face-down field card's arrival allows), and a bank of
    ``proposals`` public draws, uniform, or weighed by ``belief`` = (a checkpoint's own ``ParticleBelief``, the root's
    public latent from the seat's own forward, power) (``belief_joint_bank``). With ``response_classes`` the bank is
    ``proposals`` exact draws stratified by the opponent's hand-trap classes, weighted by the classes' exact
    probabilities under ``belief``'s tilt (``client_response_classes.response_class_bank``). ``continued`` =
    ``(worlds, weights)``: the worlds an earlier root kept (``client_continuity``) come first in the bank, weighted
    by their shares of their part, the fresh draws by the fresh bank's (the parts in proportion to their counts).
    ParticleRefused when the root is not admitted."""
    public = client_public_view(client)
    viewer = client.result.our_player
    sync = root.host["sync"]
    core = root.owner.core
    ledger = ledger_claims(client.board.disclosure, viewer, 1 - viewer)
    field = facedown_field_claims(sync["packets"][:sync["cursor"]], viewer, prior, core.card_pool().cards, public,
                                  monster_sset_codes(core, prior.codes))
    categories = merge_field_claims(ledger, field) + root_claims(sync, viewer)
    record = None
    direct_proof = None
    if direct_ar_proposal is not None:
        if belief is not None or response_classes or continued is not None or not callable(direct_ar_proposal):
            raise ValueError("direct current AR banks cannot mix with legacy weighting/continuation modes")
        bank, direct_proof = direct_ar_proposal(public, prior, viewer, categories)
        if not isinstance(bank, JointParticleBank):
            raise ValueError("direct current AR proposal did not produce a complete immutable bank")
    elif response_classes:
        table = None
        if belief is not None:
            head, latent, power = belief
            table = (head.recipe.codes, head.adjustments(latent), power, head.sha256)
        bank, record = response_class_bank(public, prior, viewer=viewer, count=proposals, seed=seed,
                                           extra_origin_slots=extra_origins, categories=categories, belief=table,
                                           allocation=class_allocation)
    elif belief is None:
        bank = uniform_joint_bank(public, prior, viewer=viewer, count=proposals, seed=seed,
                                  extra_origin_slots=extra_origins, categories=categories)
    else:
        head, latent, power = belief
        bank = belief_joint_bank(public, prior, head, latent, viewer=viewer, count=proposals, seed=seed, power=power,
                                 extra_origin_slots=extra_origins, categories=categories)
    if continued is None:
        return ClientParticleAdapter(root, public, bank, context, extra_origin_slots=extra_origins,
                                     categories=categories, response_classes=record, direct_public_proof=direct_proof,
                                     **(adapter_options or {}))
    worlds, weights = tuple(continued[0]), [float(value) for value in continued[1]]
    share, total = len(worlds) / (len(worlds) + proposals), math.fsum(weights)
    probabilities = tuple(share * value / total for value in weights) + tuple(
        (1 - share) * value for value in bank.probabilities)
    combined = JointParticleBank(bank.viewer, bank.recipe, worlds + bank.draws, probabilities,
                                 tuple(math.log(value) for value in probabilities), bank.field_keys, bank.zone_sizes,
                                 bank.seed, bank.power, bank.head_sha256, proposal_law=CONTINUED_LAW)
    kept = {"law": CONTINUED_LAW, "worlds": len(worlds), "weights": weights, "fresh_law": bank.proposal_law,
            "fresh_count": proposals, "classes": record}
    return ClientParticleAdapter(root, public, combined, context, extra_origin_slots=extra_origins,
                                 categories=categories, continued=kept, **(adapter_options or {}))


class _PublicClient(NetDuelClient):
    """Reuse the real client tracker; no socket and no autonomous response."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.banish_origin = BanishOriginTracker()

    def _game_msg(self, packet):
        self.banish_origin.observe(packet[0], packet[1:])
        super()._game_msg(packet)

    def connect(self):
        raise RuntimeError("offline public replay cannot connect")

    def _answer(self, msg, body):
        self.original_context = RootMenuBinding._clone_context(self.ctx)
        self.pending = parse_select(msg, body, self.ctx)
        if self.pending.cards:
            self.board_problems.extend(self.board.cross_check(self.pending.cards))

    def consume(self, row):
        result, encoded = self.pending, self.pending.auto_response
        if encoded is not None and row["choices"]:
            raise ValueError("automatic original prompt has policy choices")
        for round_index, choice in enumerate(row["choices"]):
            if encoded is not None or result.selector is None:
                raise ValueError("original choices continue after response completion")
            actions = result.selector.options()
            if (choice["msg"], choice["menu_size"], choice["round_index"]) != (result.msg, len(actions), round_index):
                raise ValueError("published original menu differs from recorded choice")
            index = choice["chosen_index"]
            if not 0 <= index < len(actions):
                raise ValueError("original choice out of range")
            encoded = result.selector.choose(index)
        if encoded is None or bytes(encoded) != bytes.fromhex(row["response_hex"]):
            raise ValueError("original published path does not encode the original response")
        return bytes(encoded)

    def public(self):
        return client_public_view(self)


def _error(exc):
    result = {"type": type(exc).__name__, "message": str(exc)}
    for field in ("reason", "index", "real", "local"):
        value = getattr(exc, field, None)
        if value is not None:
            result[field] = value.hex() if isinstance(value, bytes) else value
    return result


def _attempt(root, client, prior, *, proposals, seed, successor_steps, adapter_options=None):
    adapter = admit_root(root, client, prior, proposals=proposals, seed=seed, context=client.original_context,
                         extra_origins=client.banish_origin.extra_origin_slots(1 - client.result.our_player),
                         adapter_options=adapter_options)
    bank = adapter.bank
    rows = []
    for index in range(proposals):
        row = {"index": index, "probability": bank.probabilities[index], "steps": [], "status": "not_started"}
        rows.append(row)
        try:
            with adapter.particle(index) as session:
                actions, automatic = session.menu()
                row["root_server_rows"] = [asdict(session.server_choice(a)) for a in actions]
                for choice in row["root_server_rows"]:
                    if choice["response"] is not None:
                        choice["response"] = choice["response"].hex()
                for _ in range(successor_steps):
                    if session.finished:
                        break
                    actions, automatic = session.menu()
                    # Deliberately bounded deterministic rule successor, not a
                    # model policy or a claimed replay of the real future.
                    start = len(session._branch.driver.messages)
                    actor = session.actor
                    if automatic is not None:
                        session.respond_automatic()
                    elif actions:
                        session.step(actions[0])
                    else:
                        break
                    messages = session._branch.driver.messages[start:]
                    row["steps"].append({"actor": actor, "action": None if automatic is not None else 0,
                                         "messages": [[m.msg, m.payload.hex()] for m in messages]})
                row["status"] = "bounded_successors_executed"
        except Exception as exc:
            row.update(status="particle_refused" if isinstance(exc, ParticleRefused) else "particle_error", error=_error(exc))
        root.snapshot.verify()
    return {"status": "adapter_connected", "profile": adapter.profile, "particles": rows,
            "attempts": [asdict(r) for r in adapter.attempts], "completed_mass": adapter.completed_mass,
            "missing_mass": adapter.missing_mass, "same_hypothesis_oracle": "not_evaluated_no_history_witness",
            "materialized_particles": sum("root_server_rows" in row for row in rows),
            "native_successor_steps": sum(len(row["steps"]) for row in rows),
            "own_deck_order_realized": adapter.own_deck_order}


def follow_tape(core, tape, *, search, seconds=60., max_tries=256, proposals=2, seed=8519,
                successor_steps=2, root_seconds=10., phase_pass_audit=False, public_action_rebind=False,
                public_action_target_witness=False, hidden_hand_rebind=False, own_deck_order=False,
                history_domain="opening", hidden_target_deferral=False, hidden_target_pool=None,
                declined_prompt_audit=False, public_identities_only=False, read_answers=False,
                single_pass=False):
    """One independent arm, opening through original future or first failure.

    Deadline checks never alter the follower's snapshot schema; closures hold
    only an external clock. The CLI additionally isolates each arm in a child
    process with a hard deadline, since native process() is not interruptible.
    """
    packets = validate_tape(tape)
    if proposals < 1 or successor_steps < 0 or seconds <= 0 or root_seconds <= 0:
        raise ValueError("invalid bounded diagnostic budget")
    schedule = scheduled_roots(tape)
    roots = [dict(row, status="not_reached") for row in schedule]
    root_by_prompt = {row["prompt"]: row for row in roots}
    viewer = tape["viewer"]
    prior = JointRecipe.from_decks(tape["public_prior"]["main"], tape["public_prior"]["extra"])
    deadline = time.monotonic() + seconds

    class BoundedSync(BlankClientSync):
        def _run(self):
            if time.monotonic() >= deadline:
                raise TimeoutError("public follower deadline")
            return super()._run()

        def _try(self, choice):
            if time.monotonic() >= deadline:
                raise TimeoutError("public follower deadline")
            return super()._try(choice)

    sync = BoundedSync(viewer=viewer, own_deck=DeckList("entitled-own", tuple(tape["own_main"]), tuple(tape["own_extra"])),
                       opponent_main=tape["opponent_counts"][0], opponent_extra=tape["opponent_counts"][1],
                       core=core, seed=19, record_origins=True, phase_pass_audit=phase_pass_audit,
                       public_action_rebind=public_action_rebind, history_registry=HISTORY_PROFILE,
                       public_action_target_witness=public_action_target_witness,
                       public_action_witness_main=(tape["public_prior"]["main"] if public_action_target_witness else None),
                       hidden_hand_rebind=hidden_hand_rebind, hidden_target_deferral=hidden_target_deferral,
                       hidden_target_pool=hidden_target_pool, declined_prompt_audit=declined_prompt_audit,
                       public_identities_only=public_identities_only, read_answers=read_answers,
                       single_pass=single_pass, max_tries=max_tries, max_seed_tries=16, **tape["rules"])
    client = _PublicClient("", 0, "offline", tape["own_main"], tape["own_extra"], None,
                           card_pool=core.card_pool(), max_options=max(128, len(core.card_pool().cards)))
    client.ctx.full_phase_menu, client.ctx.auto_end_phase_discard = True, False
    client.ctx.commit_command_selections = tape["command_selections"]
    owner = ClientRootController(sync, quiesce=lambda: None, native_idle=lambda: True, on_packet=client._game_msg)
    fed, matched, committed, trace = 0, 0, 0, []
    result = {"search": bool(search), "viewer": viewer, "tape_sha256": digest(canonical(tape)),
              "selection_law": SELECTION_LAW, "roots": roots, "status": "prefix_only",
              "configuration": {"phase_pass_audit": sync.phase_pass_audit,
                  "public_action_rebind": sync.public_action_rebind, "history_registry": sync.history_registry,
                  "public_action_target_witness": sync.public_action_target_witness,
                  "hidden_hand_rebind": sync.hidden_hand_rebind, "hidden_target_deferral": sync.hidden_target_deferral,
                  "hidden_target_pool": list(sync.hidden_target_pool) if sync.hidden_target_pool else None,
                  "declined_prompt_audit": sync.declined_prompt_audit,
                  "public_identities_only": sync.public_identities_only, "read_answers": sync.read_answers,
                  "single_pass": sync.single_pass, "own_deck_order": own_deck_order, "history_domain": history_domain,
                  "max_tries": sync.max_tries, "max_seed_tries": sync.max_seed_tries, "seconds": seconds,
                  "root_seconds": root_seconds, "proposals": proposals, "seed": seed, "successor_steps": successor_steps}}
    try:
        for index, prompt in enumerate(tape["prompts"]):
            while fed <= prompt["packet"]:
                owner.receive(packets[fed])
                fed += 1
            message = owner.advance()
            if message is None:
                raise ValueError("local terminal before original prompt")
            matched += 1
            opaque = sync.deferred_action is not None
            with (owner.capture_opaque_root() if opaque else owner.capture_root(max_seconds=root_seconds)) as root:
                row = root_by_prompt.get(index)
                if row is not None:
                    row["receipt_markers"] = sorted({m for r in sync.receipt_state.accepted for m in r.markers})
                    row["status"] = "opaque_control_no_search" if opaque else "control_no_search"
                    if opaque and search:
                        row.update(status="opaque_not_connected", error={"type": "ParticleRefused",
                            "reason": "OPAQUE_ROOT_FORBIDS_SEARCH",
                            "message": "deferred public prompt has no native/model/search capability"})
                    elif search:
                        try:
                            row.update(_attempt(root, client, prior, proposals=proposals, seed=seed + index,
                                                successor_steps=successor_steps,
                                                adapter_options={"own_deck_order": own_deck_order,
                                                                 "history_domain": history_domain}))
                        except Exception as exc:
                            row.update(status="not_connected" if isinstance(exc, ParticleRefused) else "adapter_error",
                                       error=_error(exc))
                # The original public selector, not native hidden menu indices,
                # owns the real response. No hypothetical result replaces it.
                response = client.consume(prompt)
            commit = owner.commit_opaque_response if opaque else owner.commit_real_response
            commit(root, response, validate_original=lambda raw, path, data:
                   not path and raw == packets[prompt["packet"]] and data == response, send=lambda _data: None)
            committed += 1
            trace.append({"prompt": index, "packet": prompt["packet"], "response_hex": response.hex(), "opaque": opaque,
                          "filtered_cursor": sync.cursor,
                          "public_prefix_sha256": hashlib.sha256(b"".join(packets[:fed])).hexdigest()})
        for packet in packets[fed:]:
            owner.receive(packet)
            fed += 1
        if owner.advance() is not None or not sync.local.finished or sync.cursor != len(sync.packets):
            raise ValueError("original terminal not fully consumed")
        if sync.local.winner != tape["winner"]:
            raise ValueError("original terminal winner differs")
        result["status"] = "original_terminal_matched"
    except Exception as exc:
        result.update(status="budget_exhausted" if isinstance(exc, (TimeoutError, RootTimeout)) else "public_follow_failed",
                      failure=_error(exc))
    finally:
        result.update(fed_packets=fed, matched_prompts=matched, committed_responses=committed,
                      filtered_cursor=sync.cursor, own_trace=trace, original_future_sha256=digest(canonical(trace)),
                      follower_stats={name: dict(value) if isinstance(value, Counter) else value
                                      for name, value in vars(sync.stats).items() if name != "counts"})
        try:
            owner.close()
        except Exception as exc:
            result.update(status="cleanup_failed", cleanup_error=_error(exc))
    return result


def compare_arms(control, searched):
    """Only actual recorded continuation, never particle-vs-real strength."""
    if control["tape_sha256"] != searched["tape_sha256"] or control["viewer"] != searched["viewer"]:
        raise ValueError("paired arms used different public input")
    follower_keys = ("phase_pass_audit", "public_action_rebind", "public_action_target_witness",
                     "hidden_hand_rebind", "hidden_target_deferral", "hidden_target_pool", "declined_prompt_audit",
                     "public_identities_only", "read_answers", "single_pass",
                     "history_registry", "max_tries",
                     "max_seed_tries")
    # Adapter options only change what the searched arm attempts at a root.
    if any(control.get("configuration", {}).get(key) != searched.get("configuration", {}).get(key)
           for key in follower_keys):
        raise ValueError("paired arms used different follower/search configuration")
    if any(row["status"] == "budget_exhausted" for row in (control, searched)):
        return {"same_original_future": None, "status": "comparison_incomplete_budget",
                "scope": "unequal_wallclock_work", "behavior_equivalence": "not_evaluated", "search_ready": False}
    fields = ("own_trace", "status", "fed_packets", "matched_prompts", "committed_responses", "filtered_cursor", "failure")
    mismatches = [name for name in fields if control.get(name) != searched.get(name)]
    return {"same_original_future": not mismatches, "mismatches": mismatches,
            "status": "same_recorded_continuation" if not mismatches else "recorded_continuation_differs",
            "scope": "through_first_refusal_or_terminal", "behavior_equivalence": "not_evaluated",
            "search_ready": False}
