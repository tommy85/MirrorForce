"""Bounded public joint-bank realization in the permanent blank client VM.

No true-hidden donor pool, model forward, network send or global search gate.
Hand proposals are multisets; native physical coordinates are joined by local
UID only inside an owned root. A supported realization is not full MD/client
search admission. By default our unknown Deck order is NOT realized; the
``own_deck_order`` opt-in samples a complete order of our remaining deck per
particle and installs it by exact local UID (``duel_reorder_deck_uids``).
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import copy
import ctypes
import dataclasses
from dataclasses import dataclass
import hashlib
import json
import math
import random
import struct
import time
from types import SimpleNamespace

from ..netduel import constants as C
from ..netduel.actions import SelectContext, spec_to_ls
from ..netduel.cards import load_ydk
from ..netduel.disclosure import CategoryConstraint
from ..puzzle.single import SinglePuzzle
from ..search.particles import sample_future_core_seeds
from ..worldmodel.state import leaks
from .client_entity_map import capture_entities
from .client_current_root import materializer as current_root_materializer, hand_prompt_projection
from .client_continuity import CONTINUED_LAW, kept_world_breach
from .client_hidden_target import root_claims
from .client_response_classes import RESPONSE_CLASS_LAW, response_class_bank
from .client_root import RootEnvelope, RootError, _digest
from .client_root_menu import RootMenuBinding, MenuBindingError
from .client_shadow import BLANK_CODES, card_code
from .stage_a_joint_belief_runtime import JointParticleBank, SCHEMA as BANK_SCHEMA, _public_draws
from .stage_a_joint_proposals import PROPOSAL_LAW
from ..netduel.current_root_protocol import AR_BANK_LAW, AR_NATIVE_PROOF, AR_HAND_ASSIGNMENT, COUNT_BANK_LAW
from ..agent.search.belief_hand_scope import PublicHandScope, WorkBudget, MAX_NODES, UID_LAW, digest as hand_digest

INFORMATION_SET_SEARCH = True
SCHEMA = "client-public-joint-native-adapter/v1"
_EXTRA = 0x40 | 0x2000 | 0x800000 | 0x4000000
_NORMAL = 1 | 0x10 | 0x1000
_HELPERS = {
    "constant.lua": "01c949dad2eff92dde457258086f1ec0117888e794703cc679b64d0b6f1a8a6f",
    "utility.lua": "65fc51fa6e1f37581a4c04e4b7c5cd3b4c0dc0ba78119c01d30241bc41cb86fc",
    "procedure.lua": "aae9067432e47ab1082ae27ff5dbf5a2d9688950d3af40f333d0859b3493d3b0",
}
_BASIC_SCRIPTS = {
    5318639: "2ca56eae0bb525ff2459c186254d86de6c33783078c79fb63c868206cbf8e652",
    8267140: "7f933a2c8274e2a290bae25e1da46a4f7de88b9f90524d3da7f8d20a2582fd2d",
    23995346: "746ebff343ba524e27fba93e1ea5c7529644d8ae3f6f22b978c4539e18ae44d4",
    35261759: "8570ae233cca22516eaa79667365d636ffb9c3f243d9c194c081d32029223085",
    99550630: "60f4244316467ad69513fadc2a0e354f207af6fc08b603ff19f532699c1f3893",
    16435215: "71364747e1995fad0c328575265e7544a84650ed6f37a1fc625ea677bde42de1",
}
_SKY_BUNDLE = "1939b28a9291d873f5d64c17c54241a1a74d7f9478a6d2e3d69c36c317c3b742"
OWN_DECK_LAW = "client-own-deck-uniform-uid-order/v1"
HISTORY_DOMAINS = ("opening", "all_turns")
#: Receipt markers a realized own deck order answers: the follower's own draw
#: donor is the exact local object the branch re-installs on top.
_OWN_DECK_MARKERS = frozenset(("own_donor_unvalidated",))
#: Receipt markers the all-turns history domain answers: declining an optional
#: window and the core's empty-window pass leave the same duel state, as do an
#: opponent's declined prompt and the local core not asking it. A menu rebound
#: on public evidence initialized a revealed card late, just before its public
#: action; every particle initializes its hidden cards at the root the same way,
#: so a card the stream showed is no weaker than one the particle assumes. A
#: set proxy stands for a face-down card whose identity every particle assigns
#: at the root (as for any placeholder); setting it had the same effect for
#: any spell or trap, and its set-turn status stays on the object. A card
#: hydrated with an unproven source carries a code a public packet showed;
#: only which local object carries it was inferred, and the order of the
#: opponent's hidden zones is not public either. A forced shuffle fixes only
#: the cards the stream showed where it showed them; the rest keep the local
#: random order, which particles assign again at the root.
_ALL_TURNS_MARKERS = frozenset(("opaque_phase_pass_history_unvalidated", "opaque_declined_prompt_unvalidated",
                                "public_action_menu_rebind_unvalidated", "public_link_menu_rebind_unvalidated",
                                "sset_proxy_not_printed_identity", "source_entity_unproven",
                                "force_shuffle_unvalidated"))


class ParticleRefused(RootError):
    def __init__(self, reason, detail=""):
        self.reason, self.detail = reason, detail
        super().__init__(reason + (": " + detail if detail else ""))


class ParticleInvariantError(AssertionError):
    """A bank draw broke what the sampler and the root guarantee (a legal identity for every hidden card, written in
    place at the seat's decision point): a bug to fix, never a guess to skip."""

    def __init__(self, reason, detail=""):
        self.reason, self.detail = reason, detail
        super().__init__(reason + (": " + detail if detail else ""))


@dataclass(frozen=True)
class ParticleAttempt:
    index: int
    probability: float
    status: str
    reason: str = ""


@dataclass(frozen=True)
class ParticleAction:
    session: object
    node: int
    index: int
    actor: int


def _script(core, name, expected=None, *, loaded=False):
    path = core._resolve_script("./script/" + name)
    if path is None:
        raise ParticleRefused("RULE_SOURCE_MISSING", name)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    cached = core._script_cache.get("./script/" + name)
    if (expected is not None and sha != expected) or cached is not None and bytes(cached) != raw:
        raise ParticleRefused("RULE_SOURCE_DRIFT", name)
    if loaded and cached is None:
        raise ParticleRefused("OPENING_SCRIPT_NOT_LOADED", name)
    return sha


#: The levels a Normal Set from the hand allows after this many Tributes (the rulebook's; a monster whose own rule sets
#: it with fewer is outside the particles' hypotheses)
_TRIBUTE_LEVELS = {0: (1, 4), 1: (5, 6), 2: (7, 12)}
_TRIBUTE = 0x2 | 0x10  # REASON_RELEASE | REASON_SUMMON
_RITUAL = 0x80


def monster_sset_codes(core, codes) -> frozenset:
    """The monsters among ``codes`` whose scripts register EFFECT_MONSTER_SSET: a player may Set them in the Spell &
    Trap Zone like a Spell (the Artifacts, Toy Magician/Soldier/Tank, the Snipers, Mudragon of the Swamp)."""
    cards = core.card_pool().cards
    out = set()
    for code in codes:
        data = cards.get(code)
        if data is None or not data.type & C.TYPE_MONSTER:
            continue
        path = core._resolve_script("./script/c%d.lua" % code)
        if path is not None and b"EFFECT_MONSTER_SSET" in path.read_bytes():
            out.add(code)
    return frozenset(out)


def facedown_field_claims(packets, viewer, recipe, cards, public, settable=frozenset()) -> tuple:
    """The opponent's face-down field cards whose identity the packets never showed, each as a category claim on its
    slot: the cards its arrival allows. A particle is a guess, but a legal one: what it puts there must be a card that
    can lie there as the stream shows it. In a monster zone: after a Normal Set (``MSG_SET``) from the hand with n
    Tributes, a main-deck monster, not a Ritual one, of the levels n Tributes allow; after any other face-down arrival,
    a monster of the kind of the zone it came from (the Extra Deck's there, the main deck's otherwise). In the field
    zone, a Field Spell. In the other spell and trap zones, no Field Spell and: set by its player outside a chain
    link's resolution, a Spell or Trap or one of the ``settable`` monsters (``monster_sset_codes``); placed there
    while a chain link resolves (between ``MSG_CHAIN_SOLVING`` and ``MSG_CHAIN_SOLVED``), any card of the kind of the
    zone it came from, as an effect may place it. Particles draw each slot from its claim (``merge_field_claims``),
    and the adapter holds them to it."""
    opponent = 1 - viewer
    arrivals, tributes, pending, solving = {}, 0, None, False
    fields = (C.LOCATION_MZONE, C.LOCATION_SZONE)
    for packet in packets:
        msg = packet[0]
        if msg == C.MSG_CHAIN_SOLVING:
            solving = True
        elif msg in (C.MSG_CHAIN_SOLVED, C.MSG_CHAIN_END):
            solving = False
        elif msg == C.MSG_MOVE and len(packet) >= 17:
            code = struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff
            (fc, fl, fs), (tc, tl, ts, tp) = packet[5:8], packet[9:13]
            if fc == opponent and fl in fields:
                arrivals.pop((fl, fs), None)
                if fl == C.LOCATION_MZONE and struct.unpack_from("<I", packet, 13)[0] & _TRIBUTE == _TRIBUTE:
                    tributes += 1
            if tc == opponent and tl in fields:
                arrivals.pop((tl, ts), None)
                if tp & C.POS_FACEDOWN and not code:
                    arrivals[(tl, ts)] = ("effect" if solving else "own", fl) if tl == C.LOCATION_SZONE else fl
                    pending = (tl, ts) if tl == C.LOCATION_MZONE and fl == C.LOCATION_HAND else None
        elif msg == C.MSG_SET and len(packet) >= 9:
            at = struct.unpack_from("<I", packet, 5)[0]
            if (at & 0xff, (at >> 8 & 0xff, at >> 16 & 0xff)) == (opponent, pending):
                arrivals[pending] = ("set", min(tributes, 2))
            tributes, pending = 0, None
        elif msg in (C.MSG_SUMMONING, C.MSG_SPSUMMONING, C.MSG_FLIPSUMMONING) and len(packet) >= 9:
            at = struct.unpack_from("<I", packet, 5)[0]
            if at & 0xff == opponent:
                arrivals.pop((at >> 8 & 0xff, at >> 16 & 0xff), None)
            tributes, pending = 0, None
        elif msg in (C.MSG_POS_CHANGE, C.MSG_CHAINING) and len(packet) >= 8 and packet[5] == opponent:
            arrivals.pop((packet[6], packet[7]), None)  # face up it shows; turned face down it showed before
        elif msg == C.MSG_SWAP and len(packet) >= 17:
            for offset in (5, 13):
                at = struct.unpack_from("<I", packet, offset)[0]
                if at & 0xff == opponent:
                    arrivals.pop((at >> 8 & 0xff, at >> 16 & 0xff), None)
    known = {(card.location, card.sequence) for card in public.cards
             if card.controller == opponent and card.location in fields and card.code}
    claims = []
    for location, sequence in sorted(set(arrivals) - known):
        arrival = arrivals[(location, sequence)]
        codes = []
        for code, extra in zip(recipe.codes, recipe.extra):
            data = cards.get(code)
            if data is None:
                continue
            if location == C.LOCATION_SZONE:
                way, source = arrival
                if sequence == 5:
                    allowed = data.type & C.TYPE_FIELD
                elif data.type & C.TYPE_FIELD:
                    allowed = False
                elif way == "effect":
                    allowed = extra == (source == C.LOCATION_EXTRA)
                else:
                    allowed = data.type & (C.TYPE_SPELL | C.TYPE_TRAP) or code in settable
                if allowed:
                    codes.append(code)
            elif not data.type & C.TYPE_MONSTER:
                continue
            elif isinstance(arrival, tuple):
                low, high = _TRIBUTE_LEVELS[arrival[1]]
                if not extra and not data.type & (_EXTRA | _RITUAL) and low <= data.level & 0xff <= high:
                    codes.append(code)
            elif extra == (arrival == C.LOCATION_EXTRA):
                codes.append(code)
        claims.append(CategoryConstraint(location, sequence, frozenset(codes), 0))
    return tuple(claims)


def merge_field_claims(ledger, field) -> tuple:
    """The ledger's category claims with each face-down field slot's legal cards (``facedown_field_claims``) in them:
    one claim per slot, the intersection where the ledger claims that slot too. The sampler counts every claim as a
    card of its own, so a slot must not carry two. The ledger's other facts (``FreshSlots``) pass through."""
    others = tuple(claim for claim in ledger if not isinstance(claim, CategoryConstraint))
    ledger = tuple(claim for claim in ledger if isinstance(claim, CategoryConstraint))
    legal = {(claim.location, claim.sequence): claim.codes for claim in field}
    merged = tuple(dataclasses.replace(claim, codes=claim.codes & legal[(claim.location, claim.sequence)])
                   if (claim.location, claim.sequence) in legal else claim for claim in ledger)
    claimed = {(claim.location, claim.sequence) for claim in ledger}
    return merged + tuple(claim for claim in field if (claim.location, claim.sequence) not in claimed) + others


def _kind(core, code):
    row = core.card_pool().cards.get(int(code))
    if row is None or not row.type or code in BLANK_CODES:
        raise ParticleRefused("UNKNOWN_CARD_DATA", str(code))
    return int(row.type)


def _proposal_identity(layout):
    # The fitted energy reads hand counts, not the arbitrary known-first /
    # sampled row order. Preserve bank INDEX multiplicity, but quotient this
    # irrelevant hand ordering. Deck and extra orders remain explicit samples.
    return (tuple(sorted(Counter(layout.hand).items())), tuple(layout.deck),
            tuple(sorted(layout.facedown)), tuple(layout.extra))


@dataclass(frozen=True)
class OwnDeckPlan:
    """Our own deck at the root; UIDs are bottom first."""

    root_deck: tuple[int, ...]
    #: the deck's top cards a confirmation made public (and nothing has moved since), which stay on top
    pinned_top: tuple[int, ...] = ()


def _own_deck_plan(sync, viewer, root_rows, codes):
    """Constraints on our remaining deck order, from received packets only.

    A shuffle clears every position fact. After the last one, draws only take
    cards away. The top cards a confirmation showed (MSG_CONFIRM_DECKTOP, the
    Sky Striker field spell's look at three before it shuffles) stay on top:
    draws take them from the top, a card that leaves the deck leaves their
    count, and whatever remains is pinned. Anything else that fixes a position
    of a card still in the deck (a return to the deck, an announced top card, a
    grave/deck swap) is refused rather than ignored. Every particle orders the
    rest again at the root.
    """
    packets = sync["packets"][:sync["cursor"]]
    constraint, confirmed = None, []
    for packet in packets:
        msg, body = packet[0], packet[1:]
        if msg == C.MSG_SHUFFLE_DECK and body and body[0] == viewer:
            constraint, confirmed = None, []
        elif msg == C.MSG_CONFIRM_DECKTOP and len(body) >= 2 and body[0] == viewer:
            if len(body) != 2 + 7 * body[1]:
                raise ParticleRefused("OWN_DECK_CONFIRM_PACKET_MALFORMED")
            # Top first, as the core writes them.
            confirmed = [struct.unpack_from("<I", body, 2 + 7 * i)[0] & 0x7FFFFFFF for i in range(body[1])]
        elif msg in (C.MSG_DECK_TOP, C.MSG_SWAP_GRAVE_DECK) and body and body[0] == viewer:
            constraint = msg
        elif msg == C.MSG_MOVE and len(body) >= 12 and body[8] == viewer and body[9] == C.LOCATION_DECK:
            constraint = msg
        elif msg == C.MSG_DRAW and len(body) >= 2 and body[0] == viewer:
            confirmed = confirmed[body[1]:]
        elif msg == C.MSG_MOVE and len(body) >= 12 and body[4] == viewer and body[5] == C.LOCATION_DECK and confirmed:
            code = struct.unpack_from("<I", body, 0)[0] & 0x7FFFFFFF
            if code in confirmed:
                confirmed.remove(code)
    if constraint is not None:
        raise ParticleRefused("OWN_DECK_POSITION_CONSTRAINT_UNSUPPORTED", str(constraint))
    root_deck = tuple(row.uid for row in sorted((row for row in root_rows if row.controller == viewer
                                                 and row.location == C.LOCATION_DECK), key=lambda row: row.sequence))
    # The confirmed cards still in the deck are its top objects at the root, top first as confirmed.
    pinned = root_deck[len(root_deck) - len(confirmed):] if confirmed else ()
    if len(pinned) != len(confirmed) or [codes.get(uid) for uid in reversed(pinned)] != confirmed:
        raise ParticleRefused("OWN_DECK_CONFIRMED_TOP_MISMATCH")
    return OwnDeckPlan(root_deck, tuple(pinned))


def _admit_public(public, bank, draws, count, *, extra_origin_slots, categories):
    """``draws`` are the seeded public proposals of the bank's public view, ``count`` of them."""
    layouts, keys, sizes = _public_draws(public, bank.recipe, viewer=bank.viewer, count=count, seed=bank.seed,
                                         extra_origin_slots=extra_origin_slots, categories=categories)
    if (tuple(_proposal_identity(layout) for layout in layouts) != tuple(_proposal_identity(draw) for draw in draws)
            or keys != bank.field_keys or sizes != bank.zone_sizes):
        raise ParticleRefused("BANK_PUBLIC_Q_MISMATCH")


def _admit_direct_ar_public(public, bank, proof, *, extra_origin_slots, categories,
                            public_hand_scope=None, deadline=None):
    """Check an AR generated bank against the exact leak-audited current public support."""
    import json
    from ..agent.search.belief_current_law import CurrentPublicLayoutLaw, SCOPED_SCHEMA as CURRENT_PUBLIC_LAW
    from ..agent.search.belief_current_evidence import public_specification
    from .stage_a_joint_proposals import evidence_for_public_snapshot
    from .stage_a_joint_belief_runtime import ZoneClaim
    if type(proof) is not dict or set(proof) != {
            "schema", "bank_sha256", "feature_sha256", "head", "obs_sha256", "seed", "count",
            "specification_sha256", "search_admission", "teacher_channels", "hand_scope", "hand_physical_law",
            "max_nodes", "proposal_nodes"} \
            or proof.get("schema") != AR_NATIVE_PROOF \
            or proof.get("search_admission") is not False or proof.get("teacher_channels") is not False \
            or type(proof.get("head")) is not dict or type(proof.get("count")) is not int \
            or proof["count"] != len(bank.draws) or proof.get("seed") != bank.seed \
            or bank.proposal_law != AR_BANK_LAW \
            or bank.power != 0 or bank.head_sha256 is not None \
            or any(not math.isclose(weight, 1 / len(bank.draws), rel_tol=1e-12, abs_tol=1e-14)
                   for weight in bank.probabilities):
        raise ParticleRefused("DIRECT_AR_BANK_BINDING_MISMATCH")
    if public_hand_scope is None or proof["hand_scope"] != public_hand_scope \
            or proof["hand_physical_law"] != AR_HAND_ASSIGNMENT \
            or proof["max_nodes"] != MAX_NODES or type(proof["proposal_nodes"]) is not int \
            or not 0 <= proof["proposal_nodes"] < MAX_NODES \
            or type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ParticleRefused("DIRECT_AR_PHYSICAL_HAND_BINDING_MISMATCH")
    scope = PublicHandScope(public_hand_scope)
    if scope.to_dict()["obs_sha256"] != proof["obs_sha256"]:
        raise ParticleRefused("DIRECT_AR_PHYSICAL_HAND_PENDING_MISMATCH")
    for key in ("bank_sha256", "feature_sha256", "obs_sha256", "specification_sha256"):
        value = proof.get(key)
        if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ParticleRefused("DIRECT_AR_PROOF_SHA_INVALID", key)
    evidence = evidence_for_public_snapshot(public, bank.viewer, bank.recipe,
        extra_origin_slots=extra_origin_slots,
        categories=tuple(item for item in categories if not isinstance(item, ZoneClaim)))
    keys = tuple(sorted((location, sequence) for location, sequence in evidence.facedown_slot_keys
                        if location in (C.LOCATION_MZONE, C.LOCATION_SZONE)))
    sizes = (evidence.hand_size, evidence.deck_size, len(keys), evidence.extra_facedown_size)
    if keys != bank.field_keys or sizes != bank.zone_sizes:
        raise ParticleRefused("DIRECT_AR_PUBLIC_GEOMETRY_MISMATCH")
    specification = public_specification(public, bank.recipe, viewer=bank.viewer,
        extra_origin_slots=extra_origin_slots, categories=categories, hand_scope=scope.to_dict())
    spec_sha = hashlib.sha256(json.dumps(specification, sort_keys=True, separators=(",", ":"),
                                allow_nan=False).encode()).hexdigest()
    if specification.get("schema") != CURRENT_PUBLIC_LAW or spec_sha != proof["specification_sha256"]:
        raise ParticleRefused("DIRECT_AR_PUBLIC_EVIDENCE_MISMATCH")
    law = CurrentPublicLayoutLaw(specification, deadline=deadline, require_feasible=False)
    for index, draw in enumerate(bank.draws):
        layout = {"hand": list(draw.hand), "deck": list(draw.deck), "extra": list(draw.extra),
                  "facedown": [list(row) for row in draw.facedown]}
        positions = {(2, i): code for i, code in enumerate(layout["hand"])}
        positions.update({(1, i): code for i, code in enumerate(layout["deck"])})
        positions.update({(64, i): code for i, code in enumerate(layout["extra"])})
        positions.update({(location, sequence): code for location, sequence, code in layout["facedown"]})
        try:
            values = tuple(positions[(row[0], row[1])] for row in specification["slots"])
            sequence = tuple(values[i] for i in (*specification["field_targets"], *specification["hand_targets"]))
            law.check_complete_layout(layout, sequence)
        except (KeyError, IndexError, ValueError) as exc:
            raise ParticleRefused("DIRECT_AR_LAYOUT_INVALID", str(index)) from exc


def _class_bank_again(public, bank, record, count, *, extra_origin_slots, categories):
    """A response class bank derived again from the same public view, seed and recorded belief table; refused
    unless it records ``record``."""
    recorded = record.get("belief")
    belief = None if recorded is None else (
        tuple(sorted(code for code, extra in zip(bank.recipe.codes, bank.recipe.extra) if not extra)),
        recorded["adjustments"], recorded["power"], recorded["head_sha256"])
    again, again_record = response_class_bank(public, bank.recipe, viewer=bank.viewer, count=count, seed=bank.seed,
                                              extra_origin_slots=extra_origin_slots, categories=categories,
                                              belief=belief, allocation=record.get("allocation", "largest_remainder"))
    if again_record != record:
        raise ParticleRefused("BANK_PUBLIC_Q_MISMATCH")
    return again


def _admit_continued(public, bank, record, *, extra_origin_slots, categories):
    """A continued bank (design 10.13): first the worlds an earlier root kept, then a fresh bank of its own law.
    The fresh draws are derived again as that law's are; each kept world must honor every public claim here (its
    path matched the real one, which shows everything the claims come from); the weights are the kept worlds'
    shares of their part and the fresh bank's of the rest, the parts in proportion to their draw counts."""
    from .stage_a_joint_belief_runtime import ZoneClaim
    from .stage_a_joint_proposals import evidence_for_public_snapshot
    worlds, weights, law, count = (record.get(name) for name in ("worlds", "weights", "fresh_law", "fresh_count"))
    if record.get("law") != CONTINUED_LAW or type(worlds) is not int or type(count) is not int or worlds < 1 \
            or count < 1 or len(bank.draws) < worlds + 1 or not isinstance(weights, list) or len(weights) != worlds \
            or any(type(value) is not float or not math.isfinite(value) or value <= 0 for value in weights) \
            or law not in (PROPOSAL_LAW, RESPONSE_CLASS_LAW) or (law == RESPONSE_CLASS_LAW) != (record.get("classes") is not None):
        raise ParticleRefused("UNKNOWN_BANK_PROFILE")
    fresh = bank.draws[worlds:]
    if law == RESPONSE_CLASS_LAW:
        again = _class_bank_again(public, bank, record["classes"], count, extra_origin_slots=extra_origin_slots,
                                  categories=categories)
        if again.draws != fresh or again.field_keys != bank.field_keys or again.zone_sizes != bank.zone_sizes:
            raise ParticleRefused("BANK_PUBLIC_Q_MISMATCH")
        fresh_mass = again.probabilities
    else:
        if len(fresh) != count:
            raise ParticleRefused("BANK_PUBLIC_Q_MISMATCH")
        _admit_public(public, bank, fresh, count, extra_origin_slots=extra_origin_slots, categories=categories)
        fresh_mass = None
    share = worlds / (worlds + count)
    total = math.fsum(weights)
    if any(not math.isclose(p, share * value / total, rel_tol=1e-9, abs_tol=1e-12)
           for p, value in zip(bank.probabilities[:worlds], weights)) \
            or not math.isclose(math.fsum(bank.probabilities[worlds:]), 1 - share, rel_tol=1e-9, abs_tol=1e-12) \
            or fresh_mass is not None and any(not math.isclose(p, (1 - share) * q, rel_tol=1e-9, abs_tol=1e-12)
                                              for p, q in zip(bank.probabilities[worlds:], fresh_mass)):
        raise ParticleRefused("INVALID_BANK_WEIGHTS")
    zone_claims = tuple(item for item in categories if isinstance(item, ZoneClaim))
    evidence = evidence_for_public_snapshot(public, bank.viewer, bank.recipe, extra_origin_slots=extra_origin_slots,
                                           categories=tuple(item for item in categories
                                                            if not isinstance(item, ZoneClaim)))
    for draw in bank.draws[:worlds]:
        breach = kept_world_breach(evidence, zone_claims, draw)
        if breach is not None:
            raise ParticleInvariantError("CONTINUED_WORLD_BREAKS_CLAIM", breach)


def _support(root, bank, history_domain="opening"):
    """Concrete root-history domains, not a promise about every later root.

    ``all_turns`` admits every turn of the fixed Sky mirror: none of its 32
    card scripts registers a global watcher, so a blank hydrated at the root
    only gains the effects of its own initial_effect, while flags, relations,
    counters and slot history stay on the physical object and per-player
    count-limit codes live outside the card.
    """
    from pathlib import Path
    own = root.host["sync"]["decks"][bank.viewer]
    main, extra, _side = load_ydk(Path(__file__).resolve().parents[2] / "decks/stage-a/SkyStriker.ydk")
    recipe_main = Counter({c: n for c, n, e in zip(bank.recipe.codes, bank.recipe.copies, bank.recipe.extra) if not e})
    recipe_extra = Counter({c: n for c, n, e in zip(bank.recipe.codes, bank.recipe.copies, bank.recipe.extra) if e})
    core = root.owner.core
    for name, sha in _HELPERS.items():
        _script(core, name, sha, loaded=True)
    sky = (Counter(own.main) == recipe_main == Counter(main)
           and Counter(own.extra) == recipe_extra == Counter(extra))
    if sky:
        h = hashlib.sha256()
        names = list(_HELPERS) + [f"c{code}.lua" for code in sorted(set(main + extra))]
        for name in names:
            h.update(name.encode() + b"\0" + bytes.fromhex(_script(core, name, loaded=True)))
        if h.hexdigest() != _SKY_BUNDLE:
            raise ParticleRefused("RULE_SOURCE_DRIFT", "fixed Sky 40+15 script/helper bundle")
        # Full real recipe, but only its opening / one ordinary own action
        # domain. All its card scripts already initialized on our real cards.
        packets = root.host["sync"]["packets"][:root.host["sync"]["cursor"]]
        if history_domain == "all_turns":
            return "fixed-sky-all-turns/v1"
        summons = sum(packet[0] == C.MSG_SUMMONED for packet in packets)
        sets = sum(packet[0] == C.MSG_SET for packet in packets)
        if root.host["driver"]["turn"] != 1 or summons + sets > 1 or any(
                packet[0] in (C.MSG_CHAINING, C.MSG_SPSUMMONING, C.MSG_FLIPSUMMONING) for packet in packets):
            raise ParticleRefused("SKY_HISTORY_OUTSIDE_OPENING_DOMAIN",
                                  "later card-instance flags, chains and continuous history need separate witnesses")
        if any(packet[0] in (C.MSG_SUMMONING, C.MSG_SET) and packet[5] != bank.viewer for packet in packets):
            raise ParticleRefused("SKY_HISTORY_OUTSIDE_OPENING_DOMAIN", "opponent actions need their own instance-history witness")
        return "fixed-sky-shared-opening/v1"
    for code in set(own.main + own.extra) | set(bank.recipe.codes):
        kind = _kind(core, code)
        if kind & C.TYPE_MONSTER and kind & 0x10 and not kind & ~_NORMAL:
            continue
        if code not in _BASIC_SCRIPTS:
            raise ParticleRefused("SCRIPT_HISTORY_NOT_YET_WITNESSED", str(code))
        _script(core, f"c{code}.lua", _BASIC_SCRIPTS[code])
    return "basic-native-mechanisms/v1"


def _check_owned_hand_anchors(root, public_hand_scope, viewer):
    # Use the owned root's original ledger, NOT the canonical public hand
    # display (which can place unpositioned known cards at arbitrary slots).
    scope = PublicHandScope(public_hand_scope)
    anchors = root.host["driver"]["disclosure"].known_slots(viewer, 1-viewer, C.LOCATION_HAND)
    if scope.to_dict()["reference_anchors"] != [list(row) for row in sorted(anchors.items())]:
        raise ParticleRefused("PUBLIC_HAND_OWNED_LEDGER_ANCHORS_MISMATCH")


class ClientParticleAdapter:
    """One exact frozen root and its original, unrenormalized public bank."""

    search_ready = False
    own_deck_order_realized = False
    _FROZEN = frozenset(("root", "bank", "public", "context", "profile", "receipt", "root_entities",
                          "binding_digest", "_input_digest", "_rows", "_codes", "_base_integrity",
                          "own_deck_order", "history_domain", "own_deck_plan", "_ledger_extra",
                          "direct_public_proof", "public_hand_scope", "hand_deadline", "_hand_scope",
                          "_physical_hands", "_hand_uid_orders", "_hand_receipts", "_hand_local_prefixes",
                          "_hand_budget_nodes"))

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False) and name in self._FROZEN:
            raise ParticleRefused("FROZEN_ADAPTER_BINDING")
        object.__setattr__(self, name, value)

    def __init__(self, root: RootEnvelope, public, bank: JointParticleBank, original_context: SelectContext,
                 *, extra_origin_slots=(), categories=(), own_deck_order=False, history_domain="opening",
                 response_classes=None, continued=None, direct_public_proof=None,
                 public_hand_scope=None, hand_deadline=None):
        if type(root) is not RootEnvelope:
            raise ParticleRefused("UNOWNED_ROOT")
        if type(own_deck_order) is not bool or history_domain not in HISTORY_DOMAINS:
            raise ParticleRefused("UNKNOWN_ADAPTER_OPTION")
        root._check()
        if type(bank) is not JointParticleBank or bank.schema != BANK_SCHEMA \
                or bank.proposal_law not in (PROPOSAL_LAW, RESPONSE_CLASS_LAW, CONTINUED_LAW,
                                             AR_BANK_LAW, COUNT_BANK_LAW) \
                or (bank.proposal_law == RESPONSE_CLASS_LAW) != (response_classes is not None) \
                or (bank.proposal_law == CONTINUED_LAW) != (continued is not None):
            raise ParticleRefused("UNKNOWN_BANK_PROFILE")
        if type(bank.viewer) is not int or bank.viewer != root.owner.follower.viewer or public.view != bank.viewer or leaks(public):
            raise ParticleRefused("WRONG_PUBLIC_OBSERVER")
        if not isinstance(original_context, SelectContext) or original_context.our_player != bank.viewer:
            raise ParticleRefused("ORIGINAL_SELECTOR_CONTEXT_REQUIRED")
        if type(bank.seed) is not int or type(bank.power) not in (int, float) or not 0 <= bank.power <= 1 or any(
                not math.isclose(p, math.exp(logp), rel_tol=1e-10, abs_tol=1e-12)
                for p, logp in zip(bank.probabilities, bank.log_probabilities)):
            raise ParticleRefused("INVALID_BANK_WEIGHTS")
        if continued is not None:
            _admit_continued(public, bank, continued, extra_origin_slots=extra_origin_slots, categories=categories)
        elif bank.proposal_law == AR_BANK_LAW:
            _check_owned_hand_anchors(root, public_hand_scope, bank.viewer)
            _admit_direct_ar_public(public, bank, direct_public_proof,
                                    extra_origin_slots=extra_origin_slots, categories=categories,
                                    public_hand_scope=public_hand_scope, deadline=hand_deadline)
        elif bank.proposal_law == COUNT_BANK_LAW:
            from ..netduel.count_head_bank import admit as admit_count_head
            try:
                admit_count_head(public, bank, direct_public_proof, extra_origin_slots=extra_origin_slots,
                                 categories=categories)
            except ValueError:
                raise ParticleRefused("BANK_PUBLIC_Q_MISMATCH") from None
        elif response_classes is None:
            if direct_public_proof is not None:
                raise ParticleRefused("UNEXPECTED_DIRECT_AR_PROOF")
            _admit_public(public, bank, bank.draws, len(bank.draws), extra_origin_slots=extra_origin_slots,
                          categories=categories)
        else:
            again = _class_bank_again(public, bank, response_classes, len(bank.draws),
                                      extra_origin_slots=extra_origin_slots, categories=categories)
            if again != bank:
                raise ParticleRefused("BANK_PUBLIC_Q_MISMATCH")
        self.root, self.bank, self.response_classes, self.continued = root, bank, response_classes, continued
        self.direct_public_proof = copy.deepcopy(direct_public_proof)
        if bank.proposal_law != AR_BANK_LAW and (public_hand_scope is not None or hand_deadline is not None):
            raise ParticleRefused("UNEXPECTED_PHYSICAL_HAND_SCOPE")
        self.public_hand_scope = copy.deepcopy(public_hand_scope)
        self.hand_deadline = hand_deadline
        self._hand_scope = None if public_hand_scope is None else PublicHandScope(public_hand_scope)
        self._physical_hands = self._hand_uid_orders = self._hand_receipts = self._hand_local_prefixes = ()
        self._hand_budget_nodes = 0
        self._hand_readbacks = set()
        self.public = copy.deepcopy(public)
        self.context = RootMenuBinding._clone_context(original_context)
        binding_values = (SCHEMA, root.epoch, root.server_prompt, root.selector_path, self.public, bank)
        if self._hand_scope is not None:
            binding_values += (AR_HAND_ASSIGNMENT, self.public_hand_scope)
        self.binding_digest = _digest(binding_values)
        self._input_digest = _digest((self.public, self.bank, self.context, self.direct_public_proof,
                                      self.public_hand_scope, self.hand_deadline), self.context.card_pool)
        self._attempts, self._used = [], set()
        self.own_deck_order, self.history_domain = own_deck_order, history_domain
        self.profile = _support(root, bank, history_domain)
        state = root.host["sync"].get("receipt_state")
        origin = root.host["sync"]["origin"]
        saved = root.host["sync"].get("receipt_saved", {}).get(origin[0]) if origin else None
        if not state or not state.accepted or state.pending or saved is None:
            raise ParticleRefused("ORIGIN_RECEIPT_REQUIRED")
        self.receipt = state.accepted[-1]
        if (self.receipt.origin_native_sha256 != saved.native_sha256
                or self.receipt.matched_range != (origin[2], root.host["sync"]["cursor"])):
            raise ParticleRefused("ORIGIN_RECEIPT_BINDING_MISMATCH")
        # The follower's category facts still true here must be among the claims the bank was drawn under; then a
        # category proxy in the history is honored by every particle.
        claims = root_claims(root.host["sync"], bank.viewer)
        if not set(claims) <= set(categories):
            raise ParticleRefused("CATEGORY_FACTS_NOT_SAMPLED")
        sync = root.host["sync"]
        # What each face-down field card may legally be: every such slot was drawn under a claim (this one, or the
        # ledger's own), and each particle's card there is held to it.
        legal = facedown_field_claims(sync["packets"][:sync["cursor"]], bank.viewer, bank.recipe,
                                      root.owner.core.card_pool().cards, public,
                                      monster_sset_codes(root.owner.core, bank.recipe.codes))
        slots = {(claim.location, claim.sequence) for claim in categories if isinstance(claim, CategoryConstraint)}
        if any((claim.location, claim.sequence) not in slots for claim in legal):
            raise ParticleRefused("FACEDOWN_FIELD_CLAIMS_NOT_SAMPLED")
        self._field_claims = {(claim.location, claim.sequence): claim.codes for claim in legal}
        allowed = ({"public_evidence_after_batch", "public_token_birth_or_death", "category_proxy_unvalidated"}
                   | (_OWN_DECK_MARKERS if own_deck_order else set())
                   | (_ALL_TURNS_MARKERS if history_domain == "all_turns" else set()))
        for receipt in state.accepted:
            bad = set(receipt.markers) - allowed
            if "hidden_shuffle_mapping_unvalidated" in bad and history_domain == "all_turns" and all(
                    self._uid_keyed_shuffle(shuffle) for shuffle in receipt.shuffles):
                bad.discard("hidden_shuffle_mapping_unvalidated")
            if bad:
                raise ParticleRefused("UNVERIFIED_HISTORY", ",".join(sorted(bad)))
            for fact in receipt.facts:
                self._check_fact(fact)
        self.root_entities = capture_entities(root).entities
        if self.root_entities != self.receipt.root_entities:
            raise ParticleRefused("ROOT_ENTITY_RECEIPT_MISMATCH")
        self._rows = {row.uid: row for row in self.root_entities}
        self._codes = {row.uid: card_code(root.owner.core, root.duel, row.controller, row.location, row.sequence)
                       for row in self.root_entities if row.controller in (0, 1) and row.location in (1, 2, 4, 8, 16, 32, 64)}
        # The seat's ledger at this prompt, as the follower aligned the opponent's hidden identities with it.
        sync = root.host["sync"]
        readings = sync.get("public_identities") or ()
        self._ledger_extra = Counter(code for location, _sequence, code in readings[sync["answered"]].anchored
                                     if location == C.LOCATION_EXTRA) if sync["answered"] < len(readings) else Counter()
        self._validate_public()
        if self._hand_scope is not None:
            self._prepare_public_hand_plans()
        self.own_deck_plan = (_own_deck_plan(root.host["sync"], bank.viewer, self.root_entities, self._codes)
                              if own_deck_order else None)
        self._base_integrity = _digest(self._integrity_values())
        root.snapshot.verify()  # all constructor inspections must remain read-only
        self._sealed = True

    def _integrity_values(self):
        return (self.root.epoch, self.root.server_prompt, self.receipt, self.root_entities, self._rows, self._codes,
                self._physical_hands, self._hand_uid_orders, self._hand_receipts,
                self._hand_local_prefixes, self._hand_budget_nodes)

    def _prepare_public_hand_plans(self):
        """Sample physical hands once, under the proposal's ORIGINAL remaining budget.

        An unpositioned named object may move only within the public old group;
        fresh UIDs and genuine public anchors never move. Named old-group
        objects must already be justified by that group's public lower bounds,
        or this root is unrepresentable rather than a secretly narrower law.
        """
        scope, viewer = self._hand_scope, self.bank.viewer
        hand = sorted((row for row in self.root_entities if row.controller == 1 - viewer
                       and row.location == C.LOCATION_HAND), key=lambda row: row.sequence)
        if [row.sequence for row in hand] != list(range(scope.size)):
            raise ParticleRefused("PUBLIC_HAND_UID_GEOMETRY_MISMATCH")
        named_group = Counter()
        for row in hand:
            if row.sequence in scope.anchors:
                if not row.placeholder and self._codes[row.uid] != scope.anchors[row.sequence]:
                    raise ParticleRefused("PUBLIC_HAND_UID_ANCHOR_MISMATCH")
            elif not row.placeholder:
                if row.sequence not in scope.group:
                    raise ParticleRefused("UNPOSITIONED_HAND_UID_OUTSIDE_PUBLIC_OLD_GROUP")
                named_group[self._codes[row.uid]] += 1
        if any(count > scope.group_needed[code] for code, count in named_group.items()):
            raise ParticleRefused("HAND_UID_GROUP_IDENTITY_LACKS_PUBLIC_SCOPE_PROOF")
        budget = WorkBudget(deadline=self.hand_deadline, max_nodes=self.direct_public_proof["max_nodes"],
                            used=self.direct_public_proof["proposal_nodes"])
        # Certify the WHOLE UID-permutation orbit before any sampling. Checking
        # only the drawn order would accept unsupported menus on identity draws
        # and bias the allegedly uniform physical law. Transpositions from one
        # free old-group slot generate its entire symmetric group.
        original_order = tuple(row.uid for row in hand)
        generators = []
        for position in scope.free_group[1:]:
            budget.tick()
            trial = list(original_order)
            first = scope.free_group[0]
            trial[first], trial[position] = trial[position], trial[first]
            projected = hand_prompt_projection(self.root.message, self.root_entities, tuple(trial),
                viewer=viewer, scope=scope, invariant_error=ParticleRefused)
            try:
                checked = RootMenuBinding(self.root.server_prompt, bytes([projected.msg])+projected.payload,
                                          self.context, prefix=self.root.selector_path)
            except MenuBindingError as exc:
                raise ParticleRefused("PUBLIC_HAND_PENDING_MENU_ORBIT_UNSUPPORTED", str(exc)) from exc
            generators.append(checked.binding_sha256)
        orbit_sha = hand_digest({"server_prompt": self.root.server_prompt.hex(),
                                 "native_prompt": (bytes([self.root.message.msg])+self.root.message.payload).hex(),
                                 "scope_sha256": scope.sha256, "transposition_bindings": generators})
        physical, orders, receipts, prefixes = [], [], [], []
        for index, draw in enumerate(self.bank.draws):
            rng = random.Random((AR_HAND_ASSIGNMENT, self.bank.seed, index, scope.sha256).__repr__())
            sampled, proof = scope.sample(tuple(draw.hand), rng=rng, budget=budget)
            order = [row.uid for row in hand]
            remaining = set(scope.free_group)
            uid_matchings = 1
            for code in sorted(named_group):
                known = sorted(row.uid for row in hand if row.sequence in scope.free_group
                               and not row.placeholder and self._codes[row.uid] == code)
                positions = sorted(i for i in remaining if sampled[i] == code)
                if len(positions) < len(known):
                    raise ParticleInvariantError("PUBLIC_HAND_GROUP_MATCHING_CONTRADICTION")
                uid_matchings *= math.factorial(len(positions)) // math.factorial(len(positions)-len(known))
                budget.tick()
                rng.shuffle(positions)
                for uid, position in zip(known, positions):
                    budget.tick()
                    order[position] = uid
                    remaining.remove(position)
            blanks = [row.uid for row in hand if row.sequence in scope.free_group and row.placeholder]
            if len(blanks) != len(remaining):
                raise ParticleInvariantError("PUBLIC_HAND_GROUP_PLACEHOLDER_COUNT_MISMATCH")
            uid_matchings *= math.factorial(len(blanks))
            budget.tick()
            rng.shuffle(blanks)
            for position, uid in zip(sorted(remaining), blanks):
                order[position] = uid
            projected = hand_prompt_projection(self.root.message, self.root_entities, tuple(order),
                viewer=viewer, scope=scope, invariant_error=ParticleRefused)
            binding = RootMenuBinding(self.root.server_prompt, bytes([projected.msg]) + projected.payload,
                                      self.context, prefix=self.root.selector_path)
            budget.check()
            proof.update(uid_law=UID_LAW, uid_matching_count=str(uid_matchings),
                         joint_hand_probability_denominator=str(int(proof["support_count"])*uid_matchings),
                         uid_order_sha256=hand_digest(order),
                         menu_orbit_sha256=orbit_sha, menu_orbit_generators=len(generators),
                         root_response_law="current-hand-coordinates-to-unchanged-pending-vector/v1",
                         root_menu_binding_sha256=binding.binding_sha256)
            physical.append(sampled); orders.append(tuple(order)); receipts.append(proof)
            prefixes.append(binding.local_prefix)
        self._physical_hands, self._hand_uid_orders = tuple(physical), tuple(orders)
        self._hand_local_prefixes = tuple(prefixes)
        self._hand_budget_nodes = budget.nodes
        self._hand_receipts = tuple(json.dumps({**proof, "node_budget": {
            "max_nodes": budget.max_nodes, "proposal_nodes": self.direct_public_proof["proposal_nodes"],
            "total_nodes": budget.nodes}}, sort_keys=True, separators=(",", ":")) for proof in receipts)

    def hand_assignment_proof(self, index):
        self._check()
        if self._hand_scope is None or type(index) is not int or not 0 <= index < len(self._hand_receipts) \
                or index not in self._hand_readbacks:
            raise ParticleRefused("PUBLIC_HAND_ASSIGNMENT_NOT_READ_BACK")
        return {**json.loads(self._hand_receipts[index]), "readback_verified": True}

    def _uid_keyed_shuffle(self, shuffle):
        """A shuffle whose hidden result every particle re-assigns by UID at the root.

        The opponent's hand, deck and extra deck are hydrated per UID from the
        root layout, and our own deck is sampled again at the root when its
        order is realized. Our own hand is seen slot by slot after its shuffle
        and the local shuffle is forced to the same codes, so only copies of one
        card can have traded objects, and the seat's same-name cards are not
        told apart. A face-down field shuffle counts when it ends a group set
        (:meth:`_group_set_shuffle`); other face-down field shuffles and our
        extra deck keep their follower mapping and stay unvalidated.
        """
        if shuffle.source.packet[0] == C.MSG_SHUFFLE_SET_CARD:
            return self._group_set_shuffle(shuffle)
        viewer = self.bank.viewer
        for player, location in shuffle.zones:
            if location not in (C.LOCATION_HAND, C.LOCATION_DECK, C.LOCATION_EXTRA):
                return False
            if player == viewer and location != C.LOCATION_HAND \
                    and not (location == C.LOCATION_DECK and self.own_deck_order):
                return False
        return shuffle.source.packet[0] != C.MSG_SWAP_GRAVE_DECK

    def _group_set_shuffle(self, shuffle) -> bool:
        """Whether a face-down field shuffle is the last step of one group set (``sset_g``): every shuffled slot was
        just filled by that set, its move and set packets running up to the shuffle (a confirmation between).

        Such a shuffle draws no random number: each card goes to the zone its setter answered. The seat's own answers
        are the server's, so our cards stand where the server put them. The opponent's answers are not public, but
        its cards were set by one operation in one batch and differ only in identity, which the particles assign
        again at the root (the ledger holds them as exactly that multiset among these slots). A script shuffle
        (``Duel.ShuffleSetCard``, random) has no such run and stays unvalidated."""
        packets, index = self.root.host["sync"]["packets"], shuffle.source.matched_index
        position = index - 1
        while position >= 0 and packets[position][0] == C.MSG_CONFIRM_CARDS:
            position -= 1
        filled = set()
        while position >= 1 and packets[position][0] == C.MSG_SET and packets[position - 1][0] == C.MSG_MOVE:
            placed, moved = packets[position], packets[position - 1]
            if len(placed) < 9 or len(moved) < 17 or tuple(placed[5:8]) != tuple(moved[9:12]):
                return False
            filled.add(tuple(placed[5:8]))
            position -= 2
        return bool(filled) and set(shuffle.positions_before) <= filled

    def _check_fact(self, fact):
        sync = self.root.host["sync"]
        if fact.available_after > sync["cursor"] or not fact.sources:
            raise ParticleRefused("FACT_NOT_YET_PUBLIC")
        for source in fact.sources:
            if (source.attribution != "request" or not 0 <= source.matched_index < sync["cursor"]
                    or not 0 <= source.raw_index < len(sync["receipt_wire_packets"])
                    or sync["packets"][source.matched_index] != source.packet
                    or sync["receipt_wire_packets"][source.raw_index] != source.packet
                    or sync["receipt_packet_raw_indices"][source.matched_index] != source.raw_index):
                raise ParticleRefused("PUBLIC_FACT_SOURCE_MISMATCH")

    def _validate_public(self):
        public, opponent = self.public, 1 - self.bank.viewer
        driver = self.root.host["driver"]
        info = SinglePuzzle.field_info(SimpleNamespace(core=self.root.owner.core, pduel=self.root.duel,
                                                       _querybuf=ctypes.create_string_buffer(65536)))
        if (public.turn, public.turn_player, public.phase, tuple(public.lp)) != (
                driver["turn"], driver["turn_player"], driver["phase"], tuple(info.lp)):
            raise ParticleRefused("PUBLIC_TIMELINE_OR_LP_MISMATCH")
        for player in (0, 1):
            for location in (1, 2, 4, 8, 16, 32, 64):
                count = public.counts.get((player, location), len(public.zone(player, location)))
                actual = sum(row.controller == player and row.location == location for row in self.root_entities)
                if actual != count:
                    raise ParticleRefused("PUBLIC_ZONE_COUNT_MISMATCH", str((player, location)))
        for row in self.root_entities:
            if row.location == 128 or row.location and row.owner != row.controller \
                    and not self._public_control_change(row, public):
                raise ParticleRefused("OVERLAY_OR_CONTROL_TRANSFER_UNWITNESSED")
        for card in public.cards:
            if card.location not in (1, 2, 4, 8, 16, 32, 64):
                raise ParticleRefused("PUBLIC_ZONE_UNSUPPORTED")
            if card.location in (1, 2, 64):
                continue  # canonical rows are NOT physical coordinates
            matches = [row for row in self.root_entities if
                       (row.controller, row.location, row.sequence) == (card.controller, card.location, card.sequence)]
            if len(matches) != 1 or card.code and self._codes[matches[0].uid] != card.code:
                raise ParticleRefused("PUBLIC_FIELD_FACT_MISMATCH")
            if not card.code and card.controller != self.bank.viewer and not matches[0].placeholder:
                # A face-down card the stream never named: the follower may not hold an identity for it.
                raise ParticleRefused("PUBLIC_HIDDEN_KNOWLEDGE_MISMATCH", str(card.location))
        for location in (1, 2, 64):
            ours = Counter(self._codes[row.uid] for row in self.root_entities
                           if row.controller == self.bank.viewer and row.location == location)
            declared = Counter(card.code for card in public.zone(self.bank.viewer, location) if card.code)
            if ours != declared:
                raise ParticleRefused("OWN_PUBLIC_INVENTORY_MISMATCH", str(location))
            known = Counter(self._codes[row.uid] for row in self.root_entities
                            if row.controller == opponent and row.location == location and not row.placeholder)
            visible = Counter(card.code for card in public.cards
                              if card.controller == opponent and card.location == location and card.code)
            # The hand and the face-down extra deck are multisets to the rules and the particles: a card known to be
            # there keeps its object wherever the local duel put it. The deck's order is sampled and draws read it,
            # so there only proved slots may hold identities.
            unpositioned = Counter(code for player, loc, code in public.revealed_unpositioned
                                   if player == opponent and loc == location) \
                if location in (C.LOCATION_HAND, C.LOCATION_EXTRA) else Counter()
            expected = visible + unpositioned
            if location == C.LOCATION_EXTRA:
                # The published view keeps no identity of a face-down extra-deck card, while the seat's ledger (the
                # one the follower aligned with at this prompt) knows a card put back there in public: either is
                # public knowledge. The particle must hold them too (``_assignment``).
                expected |= self._ledger_extra
            if known != expected:
                raise ParticleRefused("PUBLIC_HIDDEN_KNOWLEDGE_MISMATCH", str(location))
            if known and location == C.LOCATION_DECK:
                raise ParticleRefused("KNOWN_HIDDEN_ORDER_REQUIRES_UID_REORDER", str(location))

    def _public_control_change(self, row, public) -> bool:
        """A card under another player's control that the particles need not fill: face up on the field with the
        code, controller and owner the public view shows (the sampler counts it against its owner's recipe)."""
        if row.location not in (C.LOCATION_MZONE, C.LOCATION_SZONE) or row.placeholder:
            return False
        shown = [card for card in public.cards
                 if (card.controller, card.location, card.sequence) == (row.controller, row.location, row.sequence)]
        return len(shown) == 1 and bool(shown[0].position & C.POS_FACEUP) and shown[0].code == self._codes[row.uid] \
            and shown[0].owner == row.owner

    def _check(self):
        self.root._check()
        if _digest((self.public, self.bank, self.context, self.direct_public_proof,
                    self.public_hand_scope, self.hand_deadline), self.context.card_pool) != self._input_digest:
            raise ParticleRefused("FROZEN_BANK_OR_CONTEXT_CHANGED")
        if _digest(self._integrity_values()) != self._base_integrity:
            raise ParticleRefused("FROZEN_ENTITY_BINDING_CHANGED")

    def _assignment(self, index):
        opponent, draw = 1 - self.bank.viewer, self.bank.draws[index]
        target = {}
        def zone(location):
            return sorted((row for row in self.root_entities if row.controller == opponent and row.location == location),
                          key=lambda row: row.sequence)
        hand = zone(C.LOCATION_HAND)
        if self._hand_scope is not None:
            target.update(zip(self._hand_uid_orders[index], self._physical_hands[index]))
        else:
            remaining = Counter(draw.hand)
            for row in hand:
                if not row.placeholder:
                    code = self._codes[row.uid]
                    remaining[code] -= 1
                    target[row.uid] = code
            if any(n < 0 for n in remaining.values()):
                raise ParticleInvariantError("HAND_KNOWN_MULTISET_CONTRADICTION")
            pool = sorted(remaining.elements())
            random.Random(("client-particle-hand/v1", self.bank.seed, index).__repr__()).shuffle(pool)
            blanks = [row for row in hand if row.placeholder]
            if len(pool) != len(blanks):
                raise ParticleInvariantError("HAND_CAPACITY_MISMATCH")
            target.update((row.uid, code) for row, code in zip(blanks, pool))
        for location, codes in ((C.LOCATION_DECK, draw.deck), (C.LOCATION_EXTRA, draw.extra)):
            rows = zone(location)
            if len(rows) != len(codes):
                raise ParticleInvariantError("ORDERED_ZONE_CAPACITY_MISMATCH")
            codes = list(codes)
            if location == C.LOCATION_EXTRA:
                # The particle's extra deck holds the known cards too (their order is not observable): each keeps
                # its object, the rest go to the placeholders in the particle's order.
                for row in rows:
                    if not row.placeholder:
                        if self._codes[row.uid] not in codes:
                            raise ParticleInvariantError("KNOWN_EXTRA_CARD_NOT_IN_PARTICLE")
                        codes.remove(self._codes[row.uid])
                rows = [row for row in rows if row.placeholder]
            target.update((row.uid, code) for row, code in zip(rows, codes))
        fields = {(location, seq): code for location, seq, code in draw.facedown}
        if len(fields) != len(draw.facedown):
            raise ParticleInvariantError("DUPLICATE_FIELD_ASSIGNMENT")
        for key, code in fields.items():
            location, sequence = key
            rows = [row for row in zone(location) if row.sequence == sequence]
            if len(rows) != 1:
                raise ParticleInvariantError("FIELD_SLOT_MISMATCH")
            target[rows[0].uid] = code
        for row in self.root_entities:
            if row.controller != opponent or not row.location:
                continue
            if row.placeholder and row.uid not in target:
                raise ParticleInvariantError("INCOMPLETE_OPPONENT_PARTICLE")
            if not row.placeholder:
                if row.uid in target and target[row.uid] != self._codes[row.uid]:
                    raise ParticleInvariantError("KNOWN_ENTITY_REWRITE_FORBIDDEN")
                target[row.uid] = self._codes[row.uid]
        for uid, code in target.items():
            row, kind = self._rows[uid], _kind(self.root.owner.core, code)
            if row.placeholder and bool(kind & _EXTRA) != (row.placeholder == 2):
                raise ParticleInvariantError("EXTRA_POOL_ASSIGNMENT_MISMATCH")
            if row.placeholder and row.location == C.LOCATION_MZONE and not kind & C.TYPE_MONSTER \
                    or row.placeholder and code not in self._field_claims.get((row.location, row.sequence), (code,)):
                raise ParticleInvariantError("PAST_FIELD_ARRIVAL_CONTRADICTION")
        return target

    def _claim(self, index):
        self._check()
        if type(index) is not int or not 0 <= index < len(self.bank.draws) or index in self._used:
            raise ParticleRefused("INVALID_OR_REUSED_BANK_INDEX")
        self._used.add(index)

    def _writer(self, index):
        """Bank draw ``index`` as the root's placeholders get it: ``(target, own deck order, materialize)``. The
        guesses go into the root's placeholders in place, and our own deck gets an order of its own (plan 5.13)."""
        target = self._assignment(index)
        viewer, plan, own_order = self.bank.viewer, self.own_deck_plan, None
        if plan is not None:
            # Bottom first: the rest in a uniform order, then the confirmed top cards still there.
            rest = [uid for uid in plan.root_deck if uid not in plan.pinned_top]
            random.Random(repr((OWN_DECK_LAW, self.bank.seed, index))).shuffle(rest)
            own_order = tuple(rest) + plan.pinned_top

        materialize = current_root_materializer(self.root_entities, target, viewer=viewer,
            own_deck_order=own_order, invariant_error=ParticleInvariantError,
            hand_order=None if self._hand_scope is None else self._hand_uid_orders[index],
            hand_scope=self._hand_scope,
            hand_prefix=None if self._hand_scope is None else self._hand_local_prefixes[index])
        return target, own_order, materialize

    def _written(self, branch, index, target, own_order):
        """The checks every written particle passes, and its future random words set: the menu binding."""
        viewer, plan = self.bank.viewer, self.own_deck_plan
        rows = capture_entities(branch).entities

        def layout(entities):
            # A realized own deck keeps its UIDs but not their slots.
            return tuple((row.uid, row.controller, row.location,
                          -1 if (plan is not None and row.controller == viewer and row.location == C.LOCATION_DECK)
                          or (self._hand_scope is not None and row.controller == 1 - viewer
                              and row.location == C.LOCATION_HAND) else row.sequence,
                          row.overlay_parent, row.overlay_ordinal) for row in entities)
        if layout(rows) != layout(self.root_entities):
            raise ParticleInvariantError("PARTICLE_ENTITY_LAYOUT_CHANGED")
        if plan is not None and tuple(row.uid for row in sorted(
                (row for row in rows if row.controller == viewer and row.location == C.LOCATION_DECK),
                key=lambda row: row.sequence)) != own_order:
            raise ParticleInvariantError("OWN_DECK_ORDER_READBACK_MISMATCH")
        actual = {row.uid: card_code(branch.driver.core, branch.driver.pduel, row.controller, row.location, row.sequence)
                  for row in rows if row.uid in target}
        if actual != target:
            raise ParticleInvariantError("PARTICLE_ASSIGNMENT_READBACK_MISMATCH")
        if any(row.controller == 1 - viewer and row.placeholder for row in rows):
            raise ParticleInvariantError("UNASSIGNED_OPPONENT_ENTITY")
        binding = RootMenuBinding(self.root.server_prompt, bytes([branch.message.msg]) + branch.message.payload,
                                  self.context, prefix=self.root.selector_path)
        if self._hand_scope is not None:
            hand = sorted((row for row in rows if row.controller == 1 - viewer and row.location == C.LOCATION_HAND),
                          key=lambda row: row.sequence)
            if tuple(row.uid for row in hand) != self._hand_uid_orders[index] \
                    or tuple(actual[row.uid] for row in hand) != self._physical_hands[index]:
                raise ParticleInvariantError("PUBLIC_HAND_UID_ORDER_READBACK_MISMATCH")
            self._hand_scope.check_physical(list(self._physical_hands[index]))
            if binding.binding_sha256 != json.loads(self._hand_receipts[index])["root_menu_binding_sha256"] \
                    or branch.path != self._hand_local_prefixes[index]:
                raise ParticleInvariantError("PUBLIC_HAND_PENDING_RESPONSE_BINDING_CHANGED")
        seeds = sample_future_core_seeds(self.bank.seed, f"client-particle-{index}")
        words = (ctypes.c_uint32 * 8)(*seeds)
        if branch.driver.core._lib.duel_set_future_seed(branch.driver.pduel, words) != 0:
            raise ParticleInvariantError("FUTURE_RNG_REALIZATION_FAILED")
        if self._hand_scope is not None:
            self._hand_readbacks.add(index)
        return binding

    @contextmanager
    def particle(self, index, *, max_seconds=None):
        """Realize bank draw ``index`` at the seat's decision point, as the follower left it: its guesses go into the
        root's placeholders in place, and our own deck gets an order of its own (plan 5.13). Nothing is run again, so
        the prompt the core asked and its menu stay the server's, whatever the guesses are; a guess only changes what
        happens after the seat's answer. Every check left here holds by the bank's draw and the root: one that fails
        is a ParticleInvariantError, never a particle to skip."""
        self._claim(index)
        status, reason = "error", ""
        session = None
        try:
            target, own_order, materialize = self._writer(index)
            with self.root.branch(materialize, max_seconds=max_seconds) as branch:
                binding = self._written(branch, index, target, own_order)
                session = ParticleSession(self, branch, binding, target, own_order)
                yield session
                status = "completed" if not session.barriers else "incomplete"
                reason = session.barriers[-1] if session.barriers else ""
        except ParticleRefused as exc:
            status, reason = "refused", exc.reason
            raise
        except ParticleInvariantError as exc:
            reason = exc.reason
            raise
        except BaseException as exc:
            reason = type(exc).__name__
            raise
        finally:
            self._attempts.append(ParticleAttempt(index, self.bank.probabilities[index], status, reason))

    @contextmanager
    def particles(self, indices, *, max_seconds=None):
        """Realize the bank draws ``indices`` in one branch, each written into the same unanswered root as
        ``particle`` writes it and passing the same checks, and keep each as a start: ``ParticleStarts``, whose
        lines are advanced together. A realization that fails is an error of the whole search, as for one particle.
        Each draw's attempt is recorded as completed only when the caller reports it complete (``complete``)."""
        indices = list(indices)
        if not indices or len(set(indices)) != len(indices):
            raise ParticleRefused("INVALID_OR_REUSED_BANK_INDEX")
        for index in indices:
            self._claim(index)
        starts = None
        try:
            writers = [self._writer(index) for index in indices]
            with self.root.branch(writers[0][2], max_seconds=max_seconds) as branch:
                starts = ParticleStarts(self, branch)
                for number, (index, (target, own_order, materialize)) in enumerate(zip(indices, writers)):
                    if number:
                        branch.rewrite(materialize)
                    starts.keep(index, self._written(branch, index, target, own_order))
                yield starts
        finally:
            done = set() if starts is None else starts.completed
            reasons = {} if starts is None else starts.reasons
            for index in indices:
                status = "completed" if index in done else reasons.get(index, ("error", ""))[0]
                reason = "" if index in done else reasons.get(index, ("error", "not completed"))[1]
                self._attempts.append(ParticleAttempt(index, self.bank.probabilities[index], status, reason))

    @property
    def attempts(self):
        return tuple(self._attempts)

    @property
    def completed_mass(self):
        return self.bank.covered_mass(row.index for row in self.attempts if row.status == "completed")

    @property
    def missing_mass(self):
        return 1. - self.completed_mass


@dataclass
class ParticleStart:
    """One realized particle at the unanswered root: its native snapshot and driver state, and its menu binding."""

    index: int
    snap: object
    pystate: dict
    binding: RootMenuBinding
    #: server menu row -> the branch-local action of the root prompt
    rows: dict
    message: object = None
    path: tuple = ()


class ParticleStarts:
    """The particles of one branch, each kept at the root as a start that lines are played from. A line restores its
    start (``restore``), then runs the branch's driver with the native calls the branch allows (``api``), parking at
    its own snapshots; everything stays inside the one branch and its budget."""

    def __init__(self, adapter, branch):
        self.adapter, self.branch, self.starts = adapter, branch, []
        self.completed, self.reasons = set(), {}
        lib = branch.driver.core._lib
        self.api = SimpleNamespace(duel_snapshot=lib.duel_snapshot, duel_rollback=lib.duel_rollback,
                                   duel_snapshot_free=lib.duel_snapshot_free)

    @property
    def driver(self):
        return self.branch.driver

    def keep(self, index, binding):
        branch = self.branch
        snap = self.api.duel_snapshot(branch.driver.pduel)
        if not snap:
            raise ParticleInvariantError("PARTICLE_START_NOT_KEPT")
        _result, selector = branch._selector()
        rows = {binding.choice(local).index: local for local in range(len(selector.options()))}
        self.starts.append(ParticleStart(index, snap, branch.driver.save_pystate(), binding, rows,
                                         branch.message, branch.path))

    def restore(self, start):
        """Back at ``start``: its particle written at the unanswered root, nothing answered."""
        branch = self.branch
        if self.api.duel_rollback(branch.driver.pduel, start.snap) != 0:
            raise ParticleInvariantError("PARTICLE_START_NOT_RESTORED")
        branch.driver.restore_pystate(start.pystate)
        branch.message, branch.path = (branch.root.message, branch.root.selector_path) if start.message is None \
            else (start.message, start.path)
        branch.node += 1
        branch.driver._answering_msg, branch.driver._answering_payload = branch.message.msg, branch.message.payload

    def complete(self, index):
        self.completed.add(index)

    def ended(self, index, status, reason=""):
        self.reasons[index] = (status, reason)


class ParticleSession:
    """Rule-only LIFO session. No evaluator or real policy/history callback."""

    search_ready = False

    def __init__(self, adapter, branch, binding, assignments, own_deck=None):
        self.adapter, self._branch, self.binding = adapter, branch, binding
        self._assignments = dict(assignments)  # internal diagnostic particle, not sampler input
        # Root order of our deck, bottom first (diagnostic, not sampler input).
        self._own_deck = None if own_deck is None else tuple(
            card_code(branch.driver.core, branch.driver.pduel, adapter.bank.viewer, C.LOCATION_DECK, sequence)
            for sequence in range(len(own_deck)))
        self.own_deck_order_realized = own_deck is not None
        self._root_responses = len(branch.driver.responses)
        self._failed, self._barriers = False, []
        self._action_key, self._actions = None, ()

    @property
    def assignments(self):
        return dict(self._assignments)

    @property
    def own_deck_codes(self):
        return self._own_deck

    @property
    def failed(self):
        return self._failed

    @property
    def barriers(self):
        return tuple(self._barriers)

    @property
    def actor(self):
        self._branch._check()
        msg = self._branch.message
        return None if msg is None else msg.payload[1] if msg.msg == C.MSG_SELECT_SUM else msg.payload[0]

    @property
    def finished(self):
        self._branch._check()
        return self._branch.driver.finished

    def _check(self):
        self._branch._check()
        if self.failed:
            raise ParticleRefused("BRANCH_NEEDS_LIFO_RESTORE")

    def menu(self):
        self._check()
        if self.finished:
            return (), None
        actions, automatic = self._branch.menu()
        key = (self._branch.node, self.actor, len(actions))
        if key != self._action_key:
            self._actions = tuple(ParticleAction(self, self._branch.node, row.index, self.actor) for row in actions)
            self._action_key = key
        return self._actions, automatic

    def _require_action(self, action):
        actions, _automatic = self.menu()
        if type(action) is not ParticleAction or not any(action is row for row in actions):
            raise ParticleRefused("STALE_OR_FOREIGN_PARTICLE_ACTION")

    def server_choice(self, action):
        self._check()
        self._require_action(action)
        if len(self._branch.driver.responses) != self._root_responses:
            raise ParticleRefused("NOT_CURRENT_SERVER_ROOT_ACTION")
        binding = self.binding
        if self._branch.path != binding.local_prefix:
            binding = binding.with_local_prefix(self._branch.path)
        return binding.choice(action.index)

    def _future_guard(self, messages):
        if self.own_deck_order_realized:
            return  # future draws, shuffles and deck selections run on the sampled order
        viewer = self.adapter.bank.viewer
        for message in messages:
            msg, body = message.msg, message.payload
            if (msg in (C.MSG_DRAW, C.MSG_SHUFFLE_DECK, C.MSG_CONFIRM_DECKTOP, C.MSG_DECK_TOP)
                    and body and body[0] == viewer):
                raise ParticleRefused("OWN_DECK_ORDER_UNREALIZED")
            if msg == C.MSG_MOVE and len(body) >= 12 and (
                    tuple(body[4:6]) == (viewer, C.LOCATION_DECK) or tuple(body[8:10]) == (viewer, C.LOCATION_DECK)):
                raise ParticleRefused("OWN_DECK_ORDER_UNREALIZED")
        if self._branch.message is not None:
            result, selector = self._branch._selector()
            for action in selector.options() if selector is not None else ():
                other, location, _seq, _overlay = spec_to_ls(action.spec)
                player = 1 - result.player if other else result.player
                if location == C.LOCATION_DECK and player == viewer:
                    raise ParticleRefused("OWN_DECK_ORDER_UNREALIZED")

    def step(self, action):
        self._check()
        self._require_action(action)
        start = len(self._branch.driver.messages)
        try:
            self._branch.choose(self._branch.menu()[0][action.index])
            self._future_guard(self._branch.driver.messages[start:])
        except BaseException as exc:
            self._failed = True
            self._barriers.append(exc.reason if isinstance(exc, ParticleRefused) else type(exc).__name__)
            raise
        return self.menu() if not self.finished else ((), None)

    def respond_automatic(self):
        self._check()
        _actions, automatic = self.menu()
        if automatic is None:
            raise ParticleRefused("NOT_AN_AUTOMATIC_PROMPT")
        start = len(self._branch.driver.messages)
        try:
            self._branch.respond(automatic)
            self._future_guard(self._branch.driver.messages[start:])
        except BaseException as exc:
            self._failed = True
            self._barriers.append(exc.reason if isinstance(exc, ParticleRefused) else type(exc).__name__)
            raise

    @contextmanager
    def checkpoint(self):
        self._check()
        failed = self.failed
        with self._branch.checkpoint():
            try:
                yield self
            finally:
                self._failed = failed
