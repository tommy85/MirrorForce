"""Option-A history replay primitives, using a client's accepted local journal.

Never replay an old raw menu index on a newly hydrated engine: menus may gain
or reorder rows. ``rebind_response`` keeps the selected local entity and exact
effect/command semantics, and refuses missing or ambiguous matches. The bounded
producer rebuilds from opening, proves the full received prefix/root menu and
keeps that same hypothetical engine for continuation. Unsupported mutations or
new private prompts are coverage failures, never silently invented responses.
"""
from __future__ import annotations

import struct
import hashlib
import json
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

from . import constants as C
from ..common.client_replay_journal import ReplayEvent
from ..common.client_shadow import BLANK_CODES

INFORMATION_SET_SEARCH = True
ACTIVATION_LEGALITY = "natural-menu-only; blank-follower-force-not-installed/v1"


class OpponentStreamError(RuntimeError):
    """A rejected hypothesis, optionally with client/synthetic-only replay context.

    Context is diagnostic evidence, never a repaired state or permission to
    drop the failed particle. No host object or host hidden layout is accepted.
    """
    _FIELDS = frozenset({"event_ordinal", "event_kind", "matched_cursor", "own_answered", "mutation_kind",
                         "uid", "received_code", "sampled_code", "particle_index", "stream_seed",
                         "packet_index", "received_packet_hex", "synthetic_packet_hex", "received_packets",
                         "synthetic_packets", "prompt_msg", "received_envelope_hex", "synthetic_envelope_hex"})

    def __init__(self, message, **context):
        super().__init__(message)
        if set(context) - self._FIELDS or any(type(value) not in (str, int, bool, type(None))
                                             for value in context.values()):
            raise ValueError("replay diagnostics accept only declared client/synthetic scalar fields")
        self._context_json = json.dumps(context, sort_keys=True, allow_nan=False)

    @property
    def diagnostic(self):
        return {"schema": "mirrorforce_rejected_hypothesis/v1", "training_eligible": False,
                "source": "received-client-prefix-and-synthetic-particle-only/v1",
                "reason": str(self), "context": json.loads(self._context_json)}

    def with_context(self, **context):
        # Preserve the most specific inner event when an outer caller adds
        # particle coordinates. Return a fresh exception/JSON-owned context.
        return type(self)(str(self), **{**context, **json.loads(self._context_json)})


@dataclass(frozen=True)
class _CardChoice:
    group: int
    uid: int
    controller: int
    location: int
    code: int
    detail: bytes


def _uid(entities, controller, location, sequence):
    if location & C.LOCATION_OVERLAY:
        raise OpponentStreamError("overlay response rebinding needs an explicit parent/ordinal certificate")
    found = [e.uid for e in entities if (e.controller, e.location, e.sequence) ==
             (controller, location, sequence) and not e.overlay_parent]
    if len(found) != 1:
        raise OpponentStreamError("the selected menu coordinate does not name a unique local UID")
    return found[0]


def _choice(record, entities, group, *, chain=False):
    start = 2 if chain else 0
    code = struct.unpack_from("<I", record, start)[0] & 0x7fffffff
    controller, location, sequence = record[start + 4:start + 7]
    # Flags (chain), position/direct-attack bits, and the FULL effect description
    # are kept. A same-card choice with another effect is never equivalent.
    detail = record[:start] + record[start + 7:]
    return _CardChoice(group, _uid(entities, controller, location, sequence), controller, location, code, detail)


def _records(raw):
    """Return (envelope, groups of raw records, non-card response values)."""
    if len(raw) < 2 or raw[1] not in (0, 1):
        raise OpponentStreamError("malformed replay prompt")
    msg = raw[0]
    groups = []
    try:
        if msg in (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD):
            widths = (7, 7, 7, 7, 7, 11) if msg == C.MSG_SELECT_IDLECMD else (11, 8)
            offset = 2
            for width in widths:
                count = raw[offset]
                offset += 1
                end = offset + count * width
                if end > len(raw):
                    raise ValueError()
                groups.append(tuple(raw[i:i + width] for i in range(offset, end, width)))
                offset = end
            flags = raw[offset:]
            if len(flags) != (3 if msg == C.MSG_SELECT_IDLECMD else 2):
                raise ValueError()
            base = 6 if msg == C.MSG_SELECT_IDLECMD else 2
            return raw[:2], tuple(groups), {base + i for i, flag in enumerate(flags) if flag}
        if msg == C.MSG_SELECT_CHAIN:
            count, offset, width = raw[2], 12, 14
            if len(raw) != offset + count * width:
                raise ValueError()
            entries = tuple(raw[i:i + width] for i in range(offset, len(raw), width))
            # Like total count, spe_count describes this menu's candidate
            # population, which may grow in a complete hidden hypothesis.
            # Core select_chain validates cancel from per-entry forced flags,
            # not spe_count. Keep both timing words and all chosen-row flags.
            return raw[:2] + raw[4:12], (entries,), set() if any(r[1] for r in entries) else {-1}
        if msg == C.MSG_SELECT_CARD:
            count, offset, width = raw[5], 6, 8
            if len(raw) != offset + count * width:
                raise ValueError()
            entries = tuple(raw[i:i + width] for i in range(offset, len(raw), width))
            return raw[:5], (entries,), {-1} if raw[2] else set()
        if msg == C.MSG_SELECT_UNSELECT_CARD:
            offset = 7
            count = raw[6]
            end = offset + count * 8
            selected = tuple(raw[i:i + 8] for i in range(offset, end, 8))
            n_old = raw[end]
            if len(raw) != end + 1 + n_old * 8:
                raise ValueError()
            old = tuple(raw[i:i + 8] for i in range(end + 1, len(raw), 8))
            return raw[:6], (selected, old), {-1} if raw[2] or raw[3] else set()
    except (IndexError, ValueError, struct.error) as exc:
        raise OpponentStreamError("malformed replay menu records") from exc
    return None


def _matching(source, records, entities, msg):
    matches = []
    for index, record in enumerate(records):
        target = _choice(record, entities, source.group, chain=msg == C.MSG_SELECT_CHAIN)
        if (source.group, source.uid, source.controller, source.location, source.detail) != (
                target.group, target.uid, target.controller, target.location, target.detail):
            continue
        if source.code not in BLANK_CODES and source.code != 0 and source.code != target.code:
            continue
        matches.append(index)
    if len(matches) != 1:
        raise OpponentStreamError("the historical response has no unique same-UID semantic row")
    return matches[0]


def rebind_response(event: ReplayEvent, prompt: bytes, entities: tuple) -> bytes:
    """Map an accepted local response to this particle's prompt without trial steps.

    Non-card prompts are reusable only with identical raw bytes AND identical
    local entity coordinates. Unsupported schemas never silently use raw indices
    after hydration or a UID permutation. This primitive reads no server duel.
    """
    if not isinstance(event, ReplayEvent) or event.kind != "response" or not event.validated \
            or len(event.payload) != 2:
        raise OpponentStreamError("an accepted response journal event is required")
    old, response = event.payload
    if not isinstance(prompt, bytes) or prompt[:2] != old[:2]:
        raise OpponentStreamError("the replay prompt changed type or responding player")
    source, target = _records(old), _records(prompt)
    if source is None:
        def coordinates(rows):
            return tuple((e.uid, e.overlay_parent, e.controller, e.location, e.sequence, e.overlay_ordinal)
                         for e in rows)
        if old != prompt or coordinates(event.before.entities) != coordinates(entities):
            raise OpponentStreamError("this response schema lacks a semantic rebinding certificate")
        return bytes(response)
    if target is None or source[0] != target[0]:
        raise OpponentStreamError("historical response envelope changed: "
                                  f"msg={old[0]} old={source[0].hex()} new={target[0].hex() if target else None}",
                                  event_ordinal=event.ordinal, prompt_msg=int(old[0]),
                                  received_envelope_hex=source[0].hex(),
                                  synthetic_envelope_hex=target[0].hex() if target else None)
    old_groups, new_groups, msg = source[1], target[1], old[0]
    scalar = struct.unpack("<i", response)[0] if len(response) == 4 else None
    if scalar in source[2]:
        if scalar not in target[2]:
            raise OpponentStreamError("the historical pass/phase/cancel is not legal in this particle")
        return bytes(response)
    menu = msg in (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD)
    if menu or msg == C.MSG_SELECT_CHAIN:
        if scalar is None or scalar < 0:
            raise OpponentStreamError("a command/chain response must be a nonnegative integer")
        group, index = (scalar & 0xffff, scalar >> 16) if menu else (0, scalar)
        if group >= len(old_groups) or index >= len(old_groups[group]):
            raise OpponentStreamError("historical response index is outside the recorded menu")
        action = _choice(old_groups[group][index], event.before.entities, group, chain=msg == C.MSG_SELECT_CHAIN)
        new = _matching(action, new_groups[group], entities, msg)
        return struct.pack("<i", new << 16 | group if menu else new)
    if not response or response[0] != len(response) - 1 or len(set(response[1:])) != response[0]:
        raise OpponentStreamError("malformed historical card selection")
    # UNSELECT's response indices count both lists. Preserve which list a card
    # belongs to: selecting and deselecting the same UID are different actions.
    flat_old = [(group, index, row) for group, rows in enumerate(old_groups) for index, row in enumerate(rows)]
    chosen = []
    for index in response[1:]:
        if index >= len(flat_old):
            raise OpponentStreamError("historical card selection is outside its menu")
        group, _, row = flat_old[index]
        action = _choice(row, event.before.entities, group)
        new = _matching(action, new_groups[group], entities, msg) + sum(map(len, new_groups[:group]))
        if new > 255:
            raise OpponentStreamError("rebound card index exceeds the response encoding")
        chosen.append(new)
    return bytes([len(chosen), *chosen])


def _opening_config(events, uid_codes, declaration):
    """Map the proposed root identities back to the same local opening UIDs.

    Only the opponent's initially allocated deck objects are assigned. Tokens
    born later and the owner's own private cards cannot be supplied as guesses.
    """
    from ..worldmodel.engine import DeckList, DuelConfig
    from .agent_public_recipe import checked, declare
    opening = events[0]
    if len(opening.payload) != 6:
        raise OpponentStreamError("opening journal lacks exact local deck orders")
    viewer, seed, rules, decks, orders, registry = opening.payload
    public = checked(declaration)
    if public != declare(*decks[viewer]):
        raise OpponentStreamError("asymmetric known decks need a dual-perspective service binding")
    opponent = 1 - viewer
    origin = [e for e in opening.after.entities if e.owner == opponent]
    if any(e.controller != opponent or e.location not in (C.LOCATION_DECK, C.LOCATION_EXTRA)
           or e.overlay_parent for e in origin):
        raise OpponentStreamError("opponent opening contains unsupported generated or displaced entities")
    if set(uid_codes) != {e.uid for e in origin} or any(type(uid) is not int or type(code) is not int
            or not 0 < code < 2 ** 32 or code in BLANK_CODES for uid, code in uid_codes.items()):
        raise OpponentStreamError("a particle must assign exactly every opponent opening UID")
    main = sorted((e for e in origin if e.location == C.LOCATION_DECK), key=lambda e: e.sequence)
    extra = sorted((e for e in origin if e.location == C.LOCATION_EXTRA), key=lambda e: e.uid, reverse=True)
    guessed_main, guessed_extra = tuple(uid_codes[e.uid] for e in main), tuple(uid_codes[e.uid] for e in extra)
    if Counter(guessed_main) != Counter(public["main"]) or Counter(guessed_extra) != Counter(public["extra"]):
        raise OpponentStreamError("the opening UID assignment differs from the public recipe")
    # A public reveal is a constraint, not a later overwrite of the particle.
    for item in events:
        if item.kind == "mutation":
            mutation = item.payload[0]
            if mutation.uid in uid_codes and mutation.kind in ("hydrate", "public_align") \
                    and mutation.code not in BLANK_CODES and mutation.code is not None \
                    and uid_codes[mutation.uid] != mutation.code:
                raise OpponentStreamError("particle conflicts with a previously accepted public identity",
                    event_ordinal=item.ordinal, event_kind=item.kind, matched_cursor=item.matched_cursor,
                    own_answered=item.own_answered, mutation_kind=mutation.kind, uid=mutation.uid,
                    received_code=mutation.code, sampled_code=uid_codes[mutation.uid])
    resolved_decks = [DeckList("client-history", tuple(main), tuple(extra)) for main, extra in decks]
    resolved_decks[opponent] = DeckList("hypothetical-history", guessed_main, guessed_extra)
    resolved_orders = list(orders)
    resolved_orders[opponent] = guessed_main
    config = DuelConfig(tuple(resolved_decks), seed=seed, forced_deck_orders=tuple(resolved_orders),
                        full_phase_menu=True, full_card_sort_menu=True, auto_end_phase_discard=False, **dict(rules))
    return viewer, config, public, registry


def _coordinates(rows):
    return tuple((e.uid, e.overlay_parent, e.owner, e.controller, e.location, e.sequence, e.overlay_ordinal)
                 for e in rows)


def _apply_mutation(driver, event, viewer, uid_codes):
    """Replay only explicit supported public native operations, never guessed repairs."""
    from ..common.client_shadow import order_cards, reorder_deck, reorder_hand
    from ..common.client_sync import ClientSync
    if len(event.payload) != 2:
        raise OpponentStreamError("accepted mutation lacks exact native arguments")
    mutation, request = event.payload
    kind = mutation.kind
    shim = SimpleNamespace(core=driver.core, local=driver, viewer=viewer)
    if kind in ("hydrate", "public_align"):
        # The card was initialized at opening. Public blanking is bookkeeping,
        # not an instruction to erase the hypothesis or its private history.
        if mutation.uid not in uid_codes:
            raise OpponentStreamError("identity operation lacks a proposed opening UID")
        if mutation.code not in BLANK_CODES and mutation.code != uid_codes[mutation.uid]:
            raise OpponentStreamError("public identity disagrees with the initialized particle")
    elif kind == "own_place":
        # These are the searching client's already-received own identities,
        # never a server hidden order. Reproduce its exact accepted local donor
        # operation at the same point; recursive deck-top placement is scoped
        # to this detached hypothetical duel.
        if len(request) != 4 or request[0] != viewer:
            raise OpponentStreamError("own-card placement lacks its public client request")
        shim._place = lambda *args, **kwargs: ClientSync._place(shim, *args, **kwargs)
        if bool(shim._place(*request)) != mutation.changed:
            raise OpponentStreamError("own-card placement diverged from its accepted operation")
    elif kind == "force_shuffle":
        order = None if request == ("clear",) else request
        if order is not None and any(code in BLANK_CODES for code in order[-1]):
            raise OpponentStreamError("a blank-code shuffle has no exact particle UID order")
        ClientSync._force_shuffle(shim, order)
    elif kind == "force_random":
        ClientSync._force_random(shim, None if request == ("clear",) else request)
    elif kind in ("hand_rebind", "deck_rebind"):
        player, location, uids = request
        if location != (C.LOCATION_HAND if kind == "hand_rebind" else C.LOCATION_DECK):
            raise OpponentStreamError("zone rebind mutation has the wrong location")
        (reorder_hand if kind == "hand_rebind" else reorder_deck)(driver.core, driver.pduel, player, uids)
    elif kind == "group_order":
        order_cards(driver.core, driver.pduel, request)
    elif kind == "force_activation":
        if len(request) != 4 or request[0] not in (0, 1):
            raise OpponentStreamError("malformed blank-follower activation aid")
        # This is a BLANK follower aid, not a legal operation of a complete
        # hypothetical game: the ABI bypasses condition/cost/target checks.
        # Never install it here. The following actual response MUST bind to
        # a naturally offered same-UID/effect row, and all public wire must
        # still match. A hypothesis lacking that legal activation is refused.
    else:
        raise OpponentStreamError("unsupported accepted native mutation: " + kind)


@dataclass
class SyntheticRoot:
    """Owned only by ``synthetic_roots``; its exact engine also drives continuations."""
    driver: object
    prompt: object
    stream: dict
    proof: dict


def _produce(root, events, uid_codes, declaration, *, seed):
    from ..puzzle.messages import split_messages
    from ..puzzle.single import RESPONSE_REQUIRED
    from ..common.client_entity_map import _read
    from ..worldmodel.engine import DuelDriver, PROCESSOR_BUFFER_LEN
    from .wire_projection import InitialRefreshCore, project_message
    from .agent_public_recipe import declare
    viewer, config, public, registry = _opening_config(events, uid_codes, declaration)
    opponent = 1 - viewer
    # The hypothesis is complete from opening: normal initial_effect installs
    # its own global watchers. A blank-follower watcher registry is NOT installed
    # a second time. Public-stream/root checks below must still pass exactly.
    driver = DuelDriver(config, InitialRefreshCore(root.owner.core))
    expected = tuple(root.host["sync"]["receipt_wire_packets"])
    if not expected or expected[0][:2] != bytes([C.MSG_START, viewer]):
        raise OpponentStreamError("a complete client history must begin with its own MSG_START")
    frames, actual, outbox = [], [[], []], [[], []]
    prompt = None
    event = None
    try:
        driver.build()
        if _coordinates(_read(driver.core._lib, driver.pduel)) != _coordinates(events[0].after.entities):
            raise OpponentStreamError("hypothesis initialization changed the local UID allocation/order")
        for seat in (0, 1):
            start = expected[0][:1] + bytes([seat]) + expected[0][2:]
            initial = [start, *driver.core.initial_packets[seat]]
            outbox[seat].extend(initial)
            actual[seat].extend(initial)
        for event in events[1:]:
            root._check()
            if event.kind == "mutation":
                _apply_mutation(driver, event, viewer, uid_codes)
            elif event.kind == "query":
                from ..common.client_shadow import card_position, card_reason
                name, args, _recorded_value = event.payload
                if name not in ("position", "reason"):
                    raise OpponentStreamError("unsupported accepted native query: " + str(name))
                # The query cache side effect is replayed on this hypothetical
                # engine. Its private reason value need not equal the blank
                # follower's; exact received-wire/root checks remain mandatory.
                (card_position if name == "position" else card_reason)(driver.core, driver.pduel, *args)
            elif event.kind == "response":
                if prompt is None:
                    raise OpponentStreamError("accepted response has no prompt in the particle replay")
                responding = prompt.payload[1] if prompt.msg == C.MSG_SELECT_SUM else prompt.payload[0]
                response = rebind_response(event, bytes([prompt.msg]) + prompt.payload,
                                           _read(driver.core._lib, driver.pduel))
                if responding == opponent:
                    frames.append({"messages": [[p[0], p[1:].hex()] for p in outbox[opponent]],
                                   "response": response.hex()})
                    outbox[opponent] = []
                driver._answering_msg, driver._answering_payload = prompt.msg, prompt.payload
                driver._respond(response)
                prompt = None
            elif event.kind == "process":
                if prompt is not None:
                    raise OpponentStreamError("particle has an additional private prompt; no answer is invented")
                raw = driver.core.process(driver.pduel)
                driver.steps += 1
                if raw & PROCESSOR_BUFFER_LEN:
                    length = driver.core.get_message(driver.pduel, driver._msgbuf)
                    for message in split_messages(driver._msgbuf.raw[:length]):
                        driver._observe(message)
                        projected = project_message(driver.core, driver.pduel, message.msg, message.payload)
                        for seat in (0, 1):
                            actual[seat].extend(projected[seat])
                            outbox[seat].extend(projected[seat])
                        if message.msg in RESPONSE_REQUIRED:
                            if prompt is not None:
                                raise OpponentStreamError("multiple native prompts in one replay batch")
                            prompt = message
                if driver.core.log:
                    raise OpponentStreamError("particle replay emitted a native script error")
            else:
                raise OpponentStreamError("unsupported replay journal event")
        if tuple(actual[viewer]) != expected:
            at = next((i for i, pair in enumerate(zip(actual[viewer], expected)) if pair[0] != pair[1]),
                      min(len(actual[viewer]), len(expected)))
            raise OpponentStreamError("particle replay changes the client's received stream at packet " + str(at),
                packet_index=at, received_packets=len(expected), synthetic_packets=len(actual[viewer]),
                received_packet_hex=expected[at].hex() if at < len(expected) else None,
                synthetic_packet_hex=actual[viewer][at].hex() if at < len(actual[viewer]) else None)
        if prompt is None or (prompt.payload[1] if prompt.msg == C.MSG_SELECT_SUM else prompt.payload[0]) != viewer:
            raise OpponentStreamError("particle replay did not finish at the searching player's prompt")
        from .host_view import deliver
        if deliver(prompt.msg, prompt.payload).payloads.get(viewer) != root.server_prompt:
            raise OpponentStreamError("particle replay root menu differs from the live client")
        stream = {"op": "open_stream", "seat": opponent, "main": public["main"], "extra": public["extra"],
                  "seed": seed, "opponent_recipe_mode": "mirror",
                  "public_opponent_recipe": declare(config.decks[viewer].main, config.decks[viewer].extra),
                  "frames": frames, "messages": [[p[0], p[1:].hex()] for p in outbox[opponent]]}
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        proof = {"schema": "mirrorforce_synthetic_root/v1", "public_stream_equal": True,
                 "root_menu_equal": True, "training_eligible": False, "journal_events": len(events),
                 "hypothesis_sha256": digest(sorted(uid_codes.items())), "stream_sha256": digest(stream),
                 "public_prefix_sha256": digest([p.hex() for p in expected]), "blank_registry": registry,
                 "initialization": "complete-particle-from-opening/v1"}
        proof["activation_legality"] = ACTIVATION_LEGALITY
        driver._answering_msg, driver._answering_payload = prompt.msg, prompt.payload
        return SyntheticRoot(driver, prompt, stream, proof)
    except OpponentStreamError as exc:
        driver.close()
        if event is None:
            raise
        raise exc.with_context(event_ordinal=event.ordinal, event_kind=event.kind,
                               matched_cursor=event.matched_cursor, own_answered=event.own_answered) from exc
    except BaseException:
        driver.close()
        raise


@contextmanager
def synthetic_roots(root, uid_assignments, public_opponent_recipe, *, seed=0):
    """Admit ALL particles before root-action budget allocation, or reject the root.

    One root lease protects shared core callbacks/readers. Every detached duel
    is created here, never accepted from a caller; every handle closes before
    the lease restores the real follower, even on partial construction/error.
    Unsupported history is a coverage failure, not a pass action or a particle
    silently dropped after looking at a future root action's return.
    """
    from ..common.client_root import RootEnvelope
    from ..common.client_replay_journal import at_root
    if type(root) is not RootEnvelope or not uid_assignments:
        raise OpponentStreamError("synthetic replay requires an owned root and declared particles")
    root._check()
    events = at_root(root)
    admitted = []
    with root.branch():
        try:
            for index, assignment in enumerate(uid_assignments):
                try:
                    admitted.append(_produce(root, events, dict(assignment), public_opponent_recipe,
                                             seed=int(seed) + index))
                except OpponentStreamError as exc:
                    raise exc.with_context(particle_index=index, stream_seed=int(seed) + index) from exc
            yield tuple(admitted)
        finally:
            for item in reversed(admitted):
                item.driver.close()
