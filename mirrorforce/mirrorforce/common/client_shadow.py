"""Experimental blank-card hydration for a continuously executed client core.

No opponent identities are accepted here before a public reveal. Hydration
keeps the physical card, including existing targets and external effects.
This primitive does not certify a search root: unhandled historical listeners,
hidden choices and unassigned blanks still require explicit rejection.
"""

from __future__ import annotations

import ctypes
from collections import Counter
from dataclasses import dataclass, replace
import struct

from ..immutable import ImmutableRecord
from ..netduel import constants as C
from ..netduel.board import ShadowBoard
from ..netduel.cards import CardData as MenuCard
from ..puzzle.core import CardData
from ..puzzle.messages import Message
from ..puzzle.single import RESPONSE_REQUIRED
from ..search.snap_engine import snapshot_api
from ..worldmodel.engine import _MSG_NAMES, DeckList, DuelError
from .client_sync import (ClientSync, GroupAsked, GroupHeld, RevealConflict, SyncError, _STILL_PACKETS, _boundary,
                          _prompt_player, _reveals, _shuffle_orders)
from .client_history_registry import (PROFILE as HISTORY_PROFILE, HistoryRegistryError, install_history_registry,
                                      uncovered_global_cards)

INFORMATION_SET_SEARCH = True
BLANK_MAIN, BLANK_EXTRA, BLANK_SSET, BLANK_FIELD = 999000001, 999000002, 999000003, 999000004
BLANK_CODES = frozenset((BLANK_MAIN, BLANK_EXTRA, BLANK_SSET, BLANK_FIELD))
#: The set proxies: a hidden hand card set face down shows only the zone it went to.
BLANK_SETS = frozenset((BLANK_SSET, BLANK_FIELD))
TYPE_NORMAL, TYPE_FUSION = 0x10, 0x40
_REASON_DRAW = 0x2000000


#: The opponent zones whose hidden cards the public ledger may name.
_ALIGNED_ZONES = (C.LOCATION_HAND, C.LOCATION_DECK, C.LOCATION_EXTRA, C.LOCATION_MZONE, C.LOCATION_SZONE,
                  C.LOCATION_REMOVED)
_EXTRA_TYPES = 0x40 | 0x2000 | 0x800000 | 0x4000000
#: Single pass: questions whose meaning only the card script (or a core rule) defines; when no packet rule reads
#: them, the engine computes each answer's own consequence (``_consequence_answers``).
_SCRIPT_QUESTIONS = frozenset((C.MSG_SELECT_YESNO, C.MSG_SELECT_EFFECTYN, C.MSG_SELECT_OPTION))


@dataclass(frozen=True)
class PublicIdentities(ImmutableRecord):
    """What the seat's public ledger holds about the opponent's hidden cards at one of the seat's own prompts.

    ``anchored``: ``(location, sequence, code)`` of every slot proved to hold that card. ``hand``: ``(code,
    count)`` of cards known to be in the hand whose slot is not known. Nothing else is known: every other hidden
    identity of the opponent is open to the particles.
    """

    anchored: tuple
    hand: tuple

    @classmethod
    def of(cls, ledger, viewer: int) -> "PublicIdentities":
        opponent = 1 - viewer
        anchored = tuple(sorted((location, sequence, code) for location in _ALIGNED_ZONES
                                for sequence, code in ledger.known_slots(viewer, opponent, location).items() if code))
        hand = tuple(sorted((code, count) for (controller, location, code), count
                            in ledger.unanchored_identities(viewer).items()
                            if controller == opponent and location == C.LOCATION_HAND and count > 0))
        return cls(anchored, hand)


@dataclass(frozen=True)
class OpponentHandPlan(ImmutableRecord):
    """What one batch's packets pin in the opponent's hand where the batch starts, solved as one arrangement (plan
    5.12): cards placed one by one each take an object another of them may be the only one to fit."""

    #: ``(slot, code)``: the card the packets show there
    identities: tuple
    #: ``(slot, (location, sequence))``: the zone the card there is set to face down, its identity not shown
    sets: tuple
    #: ``(slot, proxy)``: the set proxy a placeholder there stands as
    proxies: tuple
    #: the slots whose objects the opponent's last hidden hand shuffle left in an order the seat never saw
    free: frozenset


def arrange_hand(codes, plan: OpponentHandPlan, settable) -> list:
    """The arrangement of the opponent's hand ``plan`` asks for: ``source[slot]``, the slot whose object goes to
    ``slot`` (only objects of free slots move). ``codes``: the code at each slot now; ``settable(code, zone)``:
    whether a known card may be the one set there. A slot the packets name a card for takes an object with that
    code first -- any arrangement that exists still exists then, and no second copy of a known card appears --
    else a placeholder; a slot a card is set from takes a placeholder or a known card of its kind; objects stay
    where they are when they may. RevealConflict when no arrangement meets the plan."""
    count = len(codes)
    wanted = {}
    for slot, code in plan.identities:
        if slot >= count or wanted.get(slot, code) != code:
            raise RevealConflict("the opponent's hand has no slot %d for %d" % (slot, code))
        wanted[slot] = code
    zones = dict(plan.sets)
    if any(slot >= count for slot in zones):
        raise RevealConflict("the opponent's hand has no slot a card is set from")

    def fits(source, slot):
        code = codes[source]
        if slot in wanted:
            return code == wanted[slot] or code in BLANK_CODES
        return slot not in zones or code in BLANK_CODES or settable(code, zones[slot])

    free = sorted(slot for slot in plan.free if slot < count)
    for slot in range(count):
        if slot not in plan.free and not fits(slot, slot):
            raise RevealConflict("the opponent's hand slot %d is public and holds %d, not what the packets show"
                                 % (slot, codes[slot]))
    source = {slot: slot for slot in range(count) if slot not in plan.free}
    taken = set()
    for slot in free:
        if slot in wanted:
            known = [other for other in [slot] + free if other not in taken and codes[other] == wanted[slot]]
            if known:
                source[slot] = known[0]
                taken.add(known[0])
    owner = {}  # object slot -> the slot it goes to

    def assign(slot, seen):
        for other in sorted((other for other in free if other not in taken), key=lambda other: (other != slot, other)):
            if other in seen or not fits(other, slot):
                continue
            seen.add(other)
            if other not in owner or assign(owner[other], seen):
                owner[other] = slot
                return True
        return False

    for slot in free:
        if slot not in source and not assign(slot, set()):
            raise RevealConflict("no arrangement of the opponent's hand holds what the packets show at slot %d" % slot)
    source.update((slot, other) for other, slot in owner.items())
    return [source[slot] for slot in range(count)]


class HydrationError(SyncError):
    """The shadow cannot apply this reveal; callers must reject the root."""


class HydrationConflict(HydrationError, RevealConflict):
    """The revealed place holds another known card: the local duel took a different hidden path to it."""


class InitializationFailure(RuntimeError):
    """Fatal initialization failure, never swallowed as an alternative choice."""


def install_blanks(core) -> None:
    """Install reserved, scriptless card records only in this core's reader cache."""
    for code, kind in ((BLANK_MAIN, C.TYPE_MONSTER | TYPE_NORMAL),
                       (BLANK_EXTRA, C.TYPE_MONSTER | TYPE_NORMAL | TYPE_FUSION),
                       (BLANK_SSET, C.TYPE_SPELL | TYPE_NORMAL),
                       (BLANK_FIELD, C.TYPE_SPELL | C.TYPE_FIELD)):
        if core._query_card_data(code) is not None:
            raise HydrationError("reserved placeholder code exists in the database")
        data = CardData()
        data.code, data.type, data.level = code, kind, 4
        core._card_cache[code] = data
        core._known_missing.discard(code)
        core.missing_codes.discard(code)
        core.card_pool().cards[code] = MenuCard(code=code, type=kind, level=4, name="Unknown")


def card_code(core, pduel, player: int, location: int, sequence: int) -> int:
    """Read an identity in our own hypothetical engine, never on the server."""
    buf = ctypes.create_string_buffer(32)
    size = core._lib.query_card(pduel, player, location, sequence, 1, buf, 0)
    if size != 12:
        raise HydrationError("reveal does not refer to an existing shadow card")
    length, flags, code = struct.unpack_from("<III", buf)
    if length != 12 or flags != 1:
        raise HydrationError("unexpected shadow query format")
    return code


def card_reason(core, pduel, player: int, location: int, sequence: int) -> int:
    """Why a card of our own hypothetical engine last moved (``REASON_*`` bits).

    A card played from the hand keeps the reason it entered the hand with, and
    the server's MOVE shows it: drawn by rule, drawn by an effect, added by an
    effect. Blanks carry the same reasons, so this names a hidden hand object.
    """
    buf = ctypes.create_string_buffer(32)
    size = core._lib.query_card(pduel, player, location, sequence, C.QUERY_REASON, buf, 0)
    if size != 12:
        raise HydrationError("reason query does not refer to an existing shadow card")
    length, flags, reason = struct.unpack_from("<III", buf)
    if length != 12 or flags != C.QUERY_REASON:
        raise HydrationError("unexpected shadow query format")
    return reason


def card_position(core, pduel, player: int, location: int, sequence: int) -> int:
    """A card's position bits (face-up/down, attack/defense) in our own hypothetical engine."""
    buf = ctypes.create_string_buffer(32)
    size = core._lib.query_card(pduel, player, location, sequence, C.QUERY_POSITION, buf, 0)
    if size != 12:
        raise HydrationError("position query does not refer to an existing shadow card")
    length, flags, value = struct.unpack_from("<III", buf)
    if length != 12 or flags != C.QUERY_POSITION:
        raise HydrationError("unexpected shadow query format")
    return value >> 24 & 0xFF


def blank(core, pduel, player: int, location: int, sequence: int, code: int, placeholder: int) -> bool:
    """Make a hidden card a placeholder again (the inverse of :func:`hydrate`); False when the core refuses.

    The object, its status and every external effect or relation stay; only
    the card's own initial effects go. The core refuses a public card, one
    with counters or an active unique rule, and one whose effect a pending
    choice, chain or trigger still holds.
    """
    fn = getattr(core._lib, "duel_blank_card", None)
    if fn is None:
        raise HydrationError("core lacks the inverse of blank hydration")
    fn.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint32, ctypes.c_uint32]
    fn.restype = ctypes.c_int32
    before = len(core.log)
    result = fn(pduel, player, location, sequence, code, placeholder)
    if result in (-4, -5, -6) or len(core.log) != before:
        raise InitializationFailure("native blanking failed (%d): %s" % (result, " | ".join(core.log[before:before + 2])))
    return result == 0 and card_code(core, pduel, player, location, sequence) == placeholder


def hydrate(core, pduel, player: int, location: int, sequence: int, code: int) -> bool:
    """Reveal one blank in place; a known identity can only confirm itself.

    BLANK_SSET (BLANK_FIELD for the field zone) is only a temporary sset-capable execution proxy. Its printed
    type is not a known fact: real spells, traps and some monsters can all be
    set into SZONE. Queries against an unrevealed proxy remain unvalidated.

    Native initialization is transactional. External immutable script/card
    reader caches and diagnostic log entries are not duel state; they remain
    available after failure, while the arena and future execution are restored.
    """
    if player not in (0, 1) or location not in (C.LOCATION_DECK, C.LOCATION_HAND,
            C.LOCATION_EXTRA, C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_REMOVED):
        raise HydrationError("unsupported reveal coordinate")
    if not 0 <= sequence <= 255 or not 0 < code < 2**32 or code in (BLANK_MAIN, BLANK_EXTRA):
        raise HydrationError("invalid public reveal")
    old = card_code(core, pduel, player, location, sequence)
    if old == code:
        return False
    if old not in BLANK_CODES:
        raise HydrationConflict("public reveal conflicts with a previously bound physical card")
    fn = getattr(core._lib, "duel_hydrate_card", None)
    if fn is None:
        raise HydrationError("core lacks transactional blank hydration")
    fn.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8,
                   ctypes.c_uint32, ctypes.c_uint32]
    fn.restype = ctypes.c_int32
    before = len(core.log)
    api = snapshot_api(core)
    saved = api.duel_snapshot(pduel)
    if not saved:
        raise InitializationFailure("hydration snapshot failed")
    try:
        result = fn(pduel, player, location, sequence, old, code)
        # A nested script may log an error and let initial_effect return
        # normally. The Python callback is the source of this extra failure
        # signal; roll back the whole transaction before raising it.
        bad_log = len(core.log) != before
        if result != 0 or bad_log:
            if api.duel_rollback(pduel, saved) != 0:
                raise InitializationFailure("hydration rollback failed")
            error = InitializationFailure if result in (-4, -5, -6) or bad_log else HydrationError
            raise error("native hydration rejected (%d): %s" %
                        (result, " | ".join(core.log[before:before + 2])))
        if card_code(core, pduel, player, location, sequence) != code:
            api.duel_rollback(pduel, saved)
            raise InitializationFailure("native hydration readback differs")
    finally:
        api.duel_snapshot_free(saved)
    return True


class DeckOrderError(SyncError):
    pass


def reorder_deck(core, pduel, player: int, uids) -> None:
    """Put one player's main deck into an exact local-object order, bottom first.

    ``uids`` are local entity-map UIDs naming every deck card once. Objects keep
    their identity and history; only their deck slots change (no Lua, event or
    random word). The native call checks the whole order before writing.
    """
    _reorder(core, pduel, player, None, uids)


def order_cards(core, pduel, uids) -> None:
    """Make the local core process these objects in this order when they meet in one group (``duel_order_cards``):
    they take the sort positions they already hold among themselves."""
    uids = tuple(uids)
    if len(uids) < 2 or any(type(uid) is not int or uid <= 0 for uid in uids) or len(set(uids)) != len(uids):
        raise SyncError("invalid group order request")
    fn = getattr(core._lib, "duel_order_cards", None)
    if fn is None:
        raise SyncError("core lacks duel_order_cards")
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int32]
    fn.restype = ctypes.c_int32
    result = fn(pduel, (ctypes.c_uint64 * len(uids))(*uids), len(uids))
    if result == -6:
        raise GroupAsked("the group operation already asked about its first card")
    if result == -4:
        raise GroupHeld("native group order rejected (-4): an operation in progress keeps a card set of its own")
    if result != 0:
        raise SyncError("native group order rejected (%d)" % result)


def _chain(facts) -> tuple[int, ...]:
    """One order of every object the ``(first, second)`` facts name that honors them all (among objects no fact
    orders, the lower UID first)."""
    after, before = {}, {}
    for first, second in facts:
        after.setdefault(first, set()).add(second)
        before.setdefault(second, set()).add(first)
        after.setdefault(second, set())
        before.setdefault(first, set())
    ready = sorted(uid for uid, earlier in before.items() if not earlier)
    chain = []
    while ready:
        uid = ready.pop(0)
        chain.append(uid)
        for later in sorted(after[uid]):
            before[later].discard(uid)
            if not before[later]:
                ready.append(later)
        ready.sort()
    if len(chain) != len(after):
        raise SyncError("the server's group orders contradict each other")
    return tuple(chain)


def operation_groups(core, pduel) -> list[list[tuple[int, int]]]:
    """The card groups the local core's pending operations act on, innermost first: ``(uid, code)`` per card, in
    each group's order (``query_operation_groups``)."""
    fn = getattr(core._lib, "query_operation_groups", None)
    if fn is None:
        raise SyncError("core lacks query_operation_groups")
    fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]
    fn.restype = ctypes.c_int32
    size = fn(pduel, None, 0)
    if size < 4:
        raise SyncError("operation group query failed (%d)" % size)
    buf = ctypes.create_string_buffer(size)
    if fn(pduel, buf, size) != size:
        raise SyncError("operation group query changed size")
    raw, offset, groups = buf.raw, 4, []
    for _ in range(struct.unpack_from("<I", raw, 0)[0]):
        count = struct.unpack_from("<I", raw, offset)[0]
        groups.append([struct.unpack_from("<QI", raw, offset + 4 + 12 * index) for index in range(count)])
        offset += 4 + 12 * count
    return groups


def reorder_hand(core, pduel, player: int, uids) -> None:
    """The same exact local-object order for one player's hand, slot 0 first."""
    _reorder(core, pduel, player, C.LOCATION_HAND, uids)


def _reorder(core, pduel, player, location, uids):
    uids = tuple(uids)
    if player not in (0, 1) or any(type(uid) is not int or uid < 0 for uid in uids) or len(set(uids)) != len(uids):
        raise DeckOrderError("invalid zone order request")
    name = "duel_reorder_deck_uids" if location is None else "duel_reorder_zone_uids"
    fn = getattr(core._lib, name, None)
    if fn is None:
        raise DeckOrderError("core lacks exact zone reordering: " + name)
    words = (ctypes.c_uint64 * max(len(uids), 1))(*uids)
    if location is None:
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int32]
        fn.restype = ctypes.c_int32
        result = fn(pduel, player, words, len(uids))
    else:
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int32]
        fn.restype = ctypes.c_int32
        result = fn(pduel, player, location, words, len(uids))
    if result != 0:
        raise DeckOrderError("native zone reorder rejected (%d)" % result)


def prompt_permutation(real: bytes, local: bytes):
    """Certify ONLY record ordering, preserving all wire fields and envelopes.

    Returns a server-response -> local-response mapping or None. Ambiguous
    duplicate records and all other prompt schemas retain exact-byte matching.
    """
    if not real or not local or real[0] != local[0] or real[:2] != local[:2] or len(real) != len(local):
        return None
    msg = real[0]

    def parts(raw):
        if msg == C.MSG_SELECT_IDLECMD:
            offset, groups = 2, []
            for group in range(6):
                count, width = raw[offset], 11 if group == 5 else 7
                offset += 1
                groups.append(tuple(raw[offset + i * width:offset + (i + 1) * width] for i in range(count)))
                offset += count * width
            if offset + 3 != len(raw):
                raise ValueError("malformed idle prompt")
            return (raw[:2], raw[offset:]), groups
        if msg == C.MSG_SELECT_BATTLECMD:
            offset, groups = 2, []
            for width in (11, 8):  # activations, then attacks
                count = raw[offset]
                offset += 1
                groups.append(tuple(raw[offset + i * width:offset + (i + 1) * width] for i in range(count)))
                offset += count * width
            if offset + 2 != len(raw):
                raise ValueError("malformed battle prompt")
            return (raw[:2], raw[offset:]), groups
        if msg == C.MSG_SELECT_CHAIN:
            count, offset, width = raw[2], 12, 14
            if offset + count * width != len(raw):
                raise ValueError("malformed chain prompt")
            return raw[:offset], [tuple(raw[offset + i * width:offset + (i + 1) * width] for i in range(count))]
        raise ValueError("unsupported permutation schema")

    try:
        envelope, groups = parts(real)
        other_envelope, other_groups = parts(local)
    except (IndexError, ValueError):
        return None
    if envelope != other_envelope:
        return None
    mapping = {}
    for group, (source, target) in enumerate(zip(groups, other_groups)):
        if len(set(source)) != len(source) or sorted(source) != sorted(target):
            return None
        for old_index, record in enumerate(source):
            new_index = target.index(record)
            menu = msg in (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD)
            before = old_index << 16 | group if menu else old_index
            after = new_index << 16 | group if menu else new_index
            mapping[struct.pack("<i", before)] = struct.pack("<i", after)
    # Phase changes/cancel do not use effect record indices.
    special = [6 + index for index, enabled in enumerate(real[-3:]) if enabled] if msg == C.MSG_SELECT_IDLECMD \
        else [2 + index for index, enabled in enumerate(real[-2:]) if enabled] if msg == C.MSG_SELECT_BATTLECMD \
        else ([] if any(record[1] for record in groups[0]) else [-1])
    for value in special:
        mapping[struct.pack("<i", value)] = struct.pack("<i", value)
    return mapping


def chain_subset_mapping(real: bytes, local: bytes):
    """A server chain prompt that offers a subset of the local one's optional activations.

    The 2026-10-06 league server (EDOPro) withholds activations its own scripts do not allow where the pinned
    scripts do (Ash Blossom offered locally, an empty chain prompt received). When neither prompt forces a chain,
    every received record is a local record, the local extras are optional and every other field is equal, the
    seat's answer means the same activation (or the same pass) on both sides. Returns the server-response ->
    local-response mapping, or None.
    """
    width, offset = 14, 12
    if len(real) < offset or len(local) < offset or real[0] != C.MSG_SELECT_CHAIN or local[0] != C.MSG_SELECT_CHAIN \
            or real[1] != local[1] or real[4:offset] != local[4:offset] \
            or len(real) != offset + real[2] * width or len(local) != offset + local[2] * width \
            or real[2] >= local[2]:
        return None
    source = [real[offset + i * width:offset + (i + 1) * width] for i in range(real[2])]
    target = [local[offset + i * width:offset + (i + 1) * width] for i in range(local[2])]
    if any(record[1] for record in source + target):
        return None  # a forced activation is never optional on either side
    if len(set(source)) != len(source) or len(set(target)) != len(target) or any(r not in target for r in source):
        return None
    mapping = {struct.pack("<i", i): struct.pack("<i", target.index(record)) for i, record in enumerate(source)}
    mapping[struct.pack("<i", -1)] = struct.pack("<i", -1)
    return mapping


_TRACED_LISTS = (C.LOCATION_HAND, C.LOCATION_DECK, C.LOCATION_EXTRA)
#: the zones a group operation's cards are followed back through as ordered lists
_ORDER_LISTS = (*_TRACED_LISTS, C.LOCATION_GRAVE, C.LOCATION_REMOVED)
#: the seat's latest answers whose following state stays kept (a group operation asks at most a few zones)
_ANSWER_STATES = 4
_TRACED_FIELD = (C.LOCATION_MZONE, C.LOCATION_SZONE)


@dataclass(frozen=True)
class Shuffled:
    """A card's place in a hand, deck or extra deck as the shuffle at ``packet`` left it: ``slot`` of ``size``."""

    packet: int
    slot: int
    size: int


def batch_start_places(packets, start, wanted, player, sizes, *, shuffles=False):
    """Where each wanted card of ``player`` stood when the batch began, by the public packets from ``start`` on.

    ``wanted`` lists (packet index, place a card holds at that packet) pairs, ``sizes`` gives
    ``player``'s hand, deck and extra deck sizes at the batch start (and those of any other listed zone to follow,
    such as the graveyard). These zones are followed as ordered lists and the field by slot, each place carrying the batch-start place of its card: a card that
    entered a zone in the batch carries the place it came from (a searched card, its deck slot; a drawn card,
    the deck top), a shuffle keeps every slot but loses which card is in it, and a card from anywhere else
    carries nothing. The result lists, in order, each card's batch-start place, or None when the packets do not
    tell it; with ``shuffles``, a card a shuffle in the batch placed carries its :class:`Shuffled` place instead.
    A negative sequence of a hand, deck or extra deck counts from its end (-1: the deck top).
    """
    lists = {location: [(player, location, slot) for slot in range(sizes.get(location, 0))]
             for location in (*_TRACED_LISTS, *(location for location in sizes if location not in _TRACED_LISTS))}
    field, out = {}, {}

    def take(place):
        controller, location, sequence = place
        if controller != player:
            return None
        if location in lists:
            zone = lists[location]
            return zone.pop(sequence) if zone is not None and sequence < len(zone) else None
        if location in _TRACED_FIELD:
            return field.pop((location, sequence), (player, location, sequence))
        return None

    def token(place):
        controller, location, sequence = place
        if controller != player:
            return None
        if location in lists:
            zone = lists[location]
            if zone is not None and sequence < 0:
                sequence += len(zone)
            return zone[sequence] if zone is not None and 0 <= sequence < len(zone) else None
        if location in _TRACED_FIELD:
            return field.get((location, sequence), (player, location, sequence))
        return None

    asked = {}
    for number, (index, place) in enumerate(wanted):
        asked.setdefault(index, []).append((number, place))
    for index in range(start, max(asked, default=start - 1) + 1):
        for number, place in asked.get(index, ()):
            out[number] = token(tuple(place))
        packet = packets[index]
        msg = packet[0]
        if msg == C.MSG_MOVE and len(packet) >= 13:
            source, destination = tuple(packet[5:8]), tuple(packet[9:12])
            if source[1] & C.LOCATION_OVERLAY or destination[1] & C.LOCATION_OVERLAY:
                continue
            carried = take(source)
            controller, location, sequence = destination
            if controller == player and location in lists and lists[location] is not None:
                lists[location].insert(min(sequence, len(lists[location])), carried)
            elif controller == player and location in _TRACED_FIELD:
                field[(location, sequence)] = carried
        elif msg == C.MSG_DRAW and len(packet) >= 3 and packet[1] == player:
            deck, hand = lists[C.LOCATION_DECK], lists[C.LOCATION_HAND]
            for _ in range(packet[2]):
                drawn = deck.pop() if deck else None
                if hand is not None:
                    hand.append(drawn)
        elif msg in (C.MSG_SHUFFLE_HAND, C.MSG_SHUFFLE_DECK, C.MSG_SHUFFLE_EXTRA) and len(packet) >= 2 \
                and packet[1] == player:
            location = {C.MSG_SHUFFLE_HAND: C.LOCATION_HAND, C.MSG_SHUFFLE_DECK: C.LOCATION_DECK,
                        C.MSG_SHUFFLE_EXTRA: C.LOCATION_EXTRA}[msg]
            if lists[location] is not None:
                size = len(lists[location])
                lists[location] = [Shuffled(index, slot, size) if shuffles else None for slot in range(size)]
        elif msg == C.MSG_REVERSE_DECK and lists[C.LOCATION_DECK] is not None:
            lists[C.LOCATION_DECK].reverse()
        elif msg == C.MSG_SWAP_GRAVE_DECK and len(packet) >= 2 and packet[1] == player:
            lists[C.LOCATION_DECK] = None  # its size and order are no longer followed
        elif msg == C.MSG_SHUFFLE_SET_CARD and len(packet) >= 3 and len(packet) == 3 + 8 * packet[2]:
            for offset in range(3, 3 + 4 * packet[2], 4):
                controller, location, sequence = packet[offset:offset + 3]
                if controller == player and location in _TRACED_FIELD:
                    field[(location, sequence)] = None
    return [out.get(number) for number in range(len(wanted))]


def card_permutation(real: bytes, local: bytes):
    """For the seat's own MSG_SELECT_CARD listing the same cards as the received one in another order: the local
    index of each received record, else None.

    A deck card whose place the core hides is listed with a running number, and
    such a list follows the order of the core's own card objects, which the
    local duel created in another order than the host. Records match by code
    and place with the deck number and face-down position masked (see
    ``_comparable``); copies of one card match in order (the seat's same-name
    cards are not told apart).
    """
    if not real or not local or real[0] != C.MSG_SELECT_CARD or real[:6] != local[:6] or len(real) != len(local):
        return None
    count = real[5]
    if len(real) != 6 + 8 * count:
        return None
    return _record_order(_records(real, 6, count), _records(local, 6, count))


def unselect_permutation(real: bytes, local: bytes):
    """For the seat's own MSG_SELECT_UNSELECT_CARD listing the same cards as the received one in another order: the
    local index of each received record, the selectable list first and the chosen one after it (as the response
    counts them), else None. The chosen cards come from a group, in the order of the core's card objects."""
    if not real or not local or real[0] != C.MSG_SELECT_UNSELECT_CARD or real[:7] != local[:7] \
            or len(real) != len(local):
        return None
    lists = []
    for raw in (real, local):
        first = raw[6]
        offset = 7 + 8 * first
        if len(raw) < offset + 1 or len(raw) != offset + 1 + 8 * raw[offset]:
            return None
        lists.append((_records(raw, 7, first), _records(raw, offset + 1, raw[offset])))
    (real_select, real_chosen), (local_select, local_chosen) = lists
    select, chosen = _record_order(real_select, local_select), _record_order(real_chosen, local_chosen)
    if select is None or chosen is None:
        return None
    return select + tuple(len(local_select) + index for index in chosen)


def _records(raw: bytes, offset: int, count: int) -> list[bytes]:
    """A prompt's card records compared by code and place, a deck card's running number and position masked."""
    records = [raw[offset + 8 * index:offset + 8 * index + 8] for index in range(count)]
    return [record[:6] + b"\x00\x00" if record[5] == C.LOCATION_DECK else record for record in records]


def _record_order(real: list[bytes], local: list[bytes]):
    """The local index of each received record (copies of one card match in order), or None when the lists differ."""
    if len(real) != len(local):
        return None
    used, order = [False] * len(local), []
    for record in real:
        match = next((j for j in range(len(local)) if not used[j] and local[j] == record), None)
        if match is None:
            return None
        used[match] = True
        order.append(match)
    return tuple(order)


class ConsequenceBranches(SyncError):
    """A consequence check met more prompts the packets do not tell than it may play through."""


class BlankClientSync(ClientSync):
    """Opt-in ClientSync using only our deck and the opponent's public counts.

    One core runs from the opening draw. Reveals hydrate its existing objects;
    snapshots rewind execution before the revealing action, never reconstruct
    a board at each root. Unsupported hidden actions or conflicts permanently
    invalidate this follower. Search remains disabled until a separately
    validated history/particle contract exists (packet equality is insufficient).
    """

    def _new_disclosure(self):
        from .client_disclosure import AnonymousDisclosureLedger
        return AnonymousDisclosureLedger(BLANK_CODES)

    def __init__(self, *, viewer: int, own_deck: DeckList, opponent_main: int,
                 opponent_extra: int, core, history_registry: str | None = None,
                 record_origins: bool = False, record_replay: bool = False, phase_pass_audit: bool = False,
                 public_action_rebind: bool = False, public_action_target_witness: bool = False,
                 public_action_witness_main=None, hidden_hand_rebind: bool = False,
                 hidden_target_deferral: bool = False, hidden_target_pool=None, declined_prompt_audit: bool = False,
                 public_identities_only: bool = False, read_answers: bool = False, single_pass: bool = False,
                 public_opponent_recipe=None,
                 **kwargs):
        if viewer not in (0, 1) or not 1 <= opponent_main <= 255 or not 0 <= opponent_extra <= 255:
            raise ValueError("invalid public deck counts")
        if not hasattr(core._lib, "duel_hydrate_card"):
            raise HydrationError("core lacks transactional blank hydration")
        if history_registry not in (None, HISTORY_PROFILE):
            raise ValueError("unknown public history registry profile")
        if type(record_origins) is not bool:
            raise ValueError("record_origins must be an explicit boolean")
        if type(record_replay) is not bool or record_replay and not record_origins:
            raise ValueError("accepted replay journal requires explicit origin receipts")
        if type(phase_pass_audit) is not bool or phase_pass_audit and not record_origins:
            raise ValueError("phase-pass audit requires explicit origin receipts")
        if type(public_action_rebind) is not bool or public_action_rebind and not record_origins:
            raise ValueError("public-action rebind requires explicit origin receipts")
        if (type(public_action_target_witness) is not bool
                or public_action_target_witness and (not record_origins or not public_action_rebind)):
            raise ValueError("public-action target witness requires receipts and public-action rebind")
        witness_main = None if public_action_witness_main is None else tuple(public_action_witness_main)
        if public_action_target_witness and (len(witness_main or ()) != opponent_main
                or any(type(code) is not int or code <= 0 or code in BLANK_CODES for code in witness_main)):
            raise ValueError("public-action target witness needs the complete public/hypothetical main recipe")
        if not public_action_target_witness and witness_main is not None:
            raise ValueError("target witness recipe requires its independent opt-in")
        if type(hidden_hand_rebind) is not bool or hidden_hand_rebind and not hasattr(core._lib, "duel_reorder_zone_uids"):
            raise ValueError("hidden hand rebind requires the exact zone reorder core")
        if type(hidden_target_deferral) is not bool or hidden_target_deferral and not record_origins:
            raise ValueError("hidden target deferral requires explicit origin receipts")
        pool = None if hidden_target_pool is None else tuple(hidden_target_pool)
        if pool is not None and (not hidden_target_deferral or not pool or len(set(pool)) != len(pool)
                                 or any(type(code) is not int or code <= 0 or code in BLANK_CODES for code in pool)):
            raise ValueError("a hidden target pool is distinct real codes for the deferral")
        if pool is not None and history_registry is not None:
            # A pool card that keeps duel-long bookkeeping must have it from the opening, or a hidden copy given its
            # identity later reads it wrong.
            uncovered = uncovered_global_cards(core, pool)
            if uncovered:
                raise ValueError("pool cards keep duel-long state the history registry does not cover: %s"
                                 % (uncovered,))
        if type(declined_prompt_audit) is not bool or declined_prompt_audit and not record_origins:
            raise ValueError("declined-prompt audit requires explicit origin receipts")
        if single_pass and (phase_pass_audit or declined_prompt_audit):
            # The single pass lets the order in which the players were asked differ (plan 5.10): an omitted phase
            # pass or a declined prompt the local duel never asked is a WAITING it skips, with nothing to align.
            raise ValueError("a single pass skips the WAITING an omitted pass or a declined prompt takes; "
                             "the phase-pass and declined-prompt audits do not apply")
        if type(public_identities_only) is not bool or public_identities_only and (
                not record_origins or not hasattr(core._lib, "duel_blank_card")):
            raise ValueError("public identities only requires origin receipts and the core's blanking")
        if type(read_answers) is not bool or read_answers and not record_origins:
            raise ValueError("reading opponent answers requires explicit origin receipts")
        if type(single_pass) is not bool or single_pass and not (read_answers and public_identities_only):
            raise ValueError("a single pass reads opponent answers and holds only public identities")
        if hidden_target_deferral and public_action_target_witness:
            # The deferral replaces the recipe-candidate witness: no opponent recipe is assumed.
            raise ValueError("hidden target deferral and the recipe target witness are alternatives")
        recipe = None
        if public_opponent_recipe is not None:
            from .client_public_recipe import PublicRecipe
            recipe = PublicRecipe.checked(public_opponent_recipe)
            if not (record_origins and record_replay and public_identities_only and single_pass
                    and hidden_target_deferral) or (len(recipe.main), len(recipe.extra)) != (
                    opponent_main, opponent_extra) or pool is None or set(pool) != set(recipe.main + recipe.extra):
                raise ValueError("public recipe constraints require the exact blank single-pass/public-only registration")
        if phase_pass_audit:
            from .client_phase_pass import _functions
            _functions(core)  # old/default Core has no new-symbol dependency
        install_blanks(core)
        other = DeckList("unknown", (BLANK_MAIN,) * opponent_main, (BLANK_EXTRA,) * opponent_extra)
        decks = (own_deck, other) if viewer == 0 else (other, own_deck)
        super().__init__(viewer=viewer, decks=decks, core=core, **kwargs)
        self.failure: str | None = None
        self.prompt_maps = {}
        self.semantic_remaps = 0
        self.history_registry = history_registry
        self.history_registry_receipt = None
        self.record_origins = record_origins
        self.record_replay = record_replay
        self.phase_pass_audit = phase_pass_audit
        self.public_action_rebind = public_action_rebind
        self.public_action_target_witness = public_action_target_witness
        self.public_action_witness_main = witness_main
        self.hidden_hand_rebind = hidden_hand_rebind
        self.hidden_target_deferral = hidden_target_deferral
        self.hidden_target_pool = pool
        self.public_opponent_recipe = recipe
        self.hidden_target_origin = None
        # A single pass's forced activation being proved at its batch starts (``client_hidden_target.Construction``),
        # and the (activation packet, oldest batch start) pairs constructed so far.
        self.construction = None
        self.constructions_tried = frozenset()
        # In a single pass: the state before the batch that wrote the seat's latest prompt, until the next prompt.
        self.own_origin = None
        # The states right after the seat's latest answers, oldest first (single pass): a group operation that asked
        # the seat before the order it took showed is run again from before it asked.
        self.answer_states = []
        # Which of two objects the server's core takes first in a group, as the packets showed: facts of the duel.
        self.order_facts = []
        self.order_rewinds = set()
        # How many of those facts the local core holds now, and held when each kept state was saved: a run from a
        # state saved before a fact was learned (a backtrack, a construction point) puts them all back first.
        self.order_live = 0
        self.order_saved = {}
        self.hidden_target_skip = None
        self.declined_prompt_audit = declined_prompt_audit
        self.public_identities_only = public_identities_only
        self.read_answers = read_answers
        # Every batch runs once with all the evidence written first; a divergence is never searched past.
        self.single_pass = single_pass
        # Public facts about the opponent's hidden cards no single identity shows: (controller, location, codes,
        # at least, packet, zones) -- one of the zones ((location, codes), ...) held at least that many cards among
        # its codes (an activation proved it); the proxy stands in location with codes.
        self.category_constraints = []
        # While a menu is written again with evidenced identities, its answers are only read, never searched.
        self.reading_only = False
        # The seat's own public ledger, fed with every received packet the way its client feeds its board; one
        # frozen reading per own prompt, in order. The received stream is fixed, so the reading at the k-th own
        # prompt does not depend on which branch the follower took to reach it.
        self.public_board = ShadowBoard() if public_identities_only else None
        #: whether the public board read the server's MSG_START, which names the seat it keeps the books for
        self.public_started = False
        self.public_board_problems = []
        self.public_identities = []
        # Received WAITING packets an omitted opponent pass must take (proved by a failed attempt).
        self.phase_pass_consume = set()
        self.phase_pass_ambiguous = []
        # The receipts omitted passes would have written for the WAITING they kept, since the last load.
        self.phase_pass_kept = {}
        self.last_response_own = False
        self.deferred_action = None
        self.receipt_wire_packets, self.receipt_packet_raw_indices = [], []
        self.receipt_saved = {}
        self.receipt_state = None
        if record_origins:
            from .client_origin_receipt import ReceiptLedger
            self.receipt_state = ReceiptLedger()
        # These are explicit pending proof obligations, never derived facts.
        self.unresolved_history = frozenset(("hidden_type_predicates", "initialization_order",
                                             "unregistered_history_listeners", "hidden_choice_windows"))

    def _start(self):
        if self.record_origins:
            from .client_origin_receipt import boundary
            boundary(self)
        if self.history_registry is None:
            result = super()._start()
            if self.phase_pass_audit:
                from .client_phase_pass import configure
                configure(self)
            if self.record_origins:
                from .client_origin_receipt import started
                started(self)
            return result
        original = self.core.create_duel

        def create(seeds):
            duel = original(seeds)
            try:
                self.history_registry_receipt = install_history_registry(self.core, duel)
            except HistoryRegistryError as exc:
                self.core.end_duel(duel)
                raise InitializationFailure(str(exc)) from exc
            return duel

        # Core's Python reader/creation surface is single-threaded already.
        # This hook runs before any new_card initial_effect or first process.
        self.core.create_duel = create
        try:
            result = super()._start()
        finally:
            self.core.create_duel = original
        if self.phase_pass_audit:
            from .client_phase_pass import configure
            configure(self)
        if self.record_origins:
            from .client_origin_receipt import started
            started(self)
        return result

    def receive(self, packet):
        if self.public_board is not None:
            self._read_public(bytes(packet))
        if self.record_origins:
            packet = bytes(packet)
            index, count = len(self.receipt_wire_packets), len(self.packets)
            self.receipt_wire_packets.append(packet)
            super().receive(packet)
            if len(self.packets) != count:
                self.receipt_packet_raw_indices.append(index)
            return
        return super().receive(packet)

    def _read_public(self, packet):
        """Feed one received packet to the seat's public board exactly as the client does (``NetDuelClient``)."""
        if not packet:
            return
        msg, body = packet[0], packet[1:]
        if msg != C.MSG_START and not self.public_started:
            # Before MSG_START the board keeps seat 0's books: another seat's ledger would anchor what its own seat
            # never saw, and every identity aligned from it would be wrong.
            raise SyncError("the seat's public board reads the server's MSG_START before any other packet")
        try:
            if msg == C.MSG_START:
                own = self.decks[self.viewer]
                self.public_board.start(self.viewer, list(own.main), list(own.extra))
                self.public_started = True
            self.public_board.apply(msg, body)
        except Exception as exc:
            self.public_board_problems.append("msg %d: %s" % (msg, exc))
        if msg in RESPONSE_REQUIRED and _prompt_player(Message(msg, body)) == self.viewer:
            self.public_identities.append(PublicIdentities.of(self.public_board.disclosure, self.viewer))

    def _save(self):
        saved = super()._save()
        if self.record_origins:
            from .client_origin_receipt import remember_snapshot
            try:
                remember_snapshot(self, saved)
            except BaseException:
                super()._free(saved)
                raise
        self.order_saved[saved[0]] = self.order_live
        return saved

    def _load(self, saved):
        if self.record_origins and saved[0] not in self.receipt_saved:
            raise SyncError("snapshot has no owned receipt state")
        # Kept pass receipts belong to the batches run since the last load.
        self.phase_pass_kept = {}
        if saved[0] not in self.order_saved:
            raise SyncError("snapshot has no record of the group orders it holds")
        result = super()._load(saved)
        self.order_live = self.order_saved[saved[0]]
        if self.record_origins:
            from .client_origin_receipt import load_snapshot
            load_snapshot(self, saved)
        return result

    def _free(self, saved):
        if saved is not None and self.construction is not None and saved[0] in self.construction.held():
            return  # every replay of the construction starts from its bookkeeping, which it frees when it ends
        if saved is not None:
            self.receipt_saved.pop(saved[0], None)
            self.order_saved.pop(saved[0], None)
        return super()._free(saved)

    def _release(self, choice):
        if choice.pass_taken is not None:
            self.phase_pass_consume.discard(choice.pass_taken)
        return super()._release(choice)

    def _drop_origin(self):
        super()._drop_origin()
        self._free(self.own_origin)
        self.own_origin = None

    def _own_answered(self):
        # A single pass keeps the state before the batch that wrote the seat's prompt until the next prompt: the
        # core checks both players' optional triggers when it builds the turn player's choice, so an opponent's
        # trigger activated after the seat answered was allowed or dropped before the seat's prompt.
        if not (self.single_pass and self.hidden_target_deferral):
            return super()._own_answered()
        self._free(self.own_origin)
        self.own_origin, self.origin = self.origin, None

    def close(self):
        try:
            if self.hidden_target_origin is not None:
                self._free(self.hidden_target_origin)
                self.hidden_target_origin = None
            if self.construction is not None:
                from .client_hidden_target import drop_construction
                drop_construction(self)
            while self.answer_states:
                self._free(self.answer_states.pop())
            return super().close()
        finally:
            from .client_action_target_witness import clear
            clear(self)
            self.receipt_saved.clear()  # metadata only: never free handles twice

    def _batch(self):
        if self.record_origins:
            from .client_origin_receipt import run_batch
            return run_batch(self, lambda: super(BlankClientSync, self)._batch())
        return super()._batch()

    def _observed_messages(self, messages):
        if self.record_replay:
            from .client_replay_journal import batch_messages
            messages = tuple(messages)
            batch_messages(self, messages)
        if self.phase_pass_audit:
            from .client_phase_pass import observed_messages
            return observed_messages(self, messages)
        return super()._observed_messages(messages)

    def _fixed_batch(self, saved, *, receipt_origin=None):
        """The inherited batch; a divergence after an undecided omitted pass retries it taking that WAITING."""
        from .client_sync import _Divergence
        while True:
            self.phase_pass_ambiguous = []
            try:
                if self.record_origins:
                    from .client_origin_receipt import fixed_batch
                    return fixed_batch(self, saved, lambda: self._settled(super(BlankClientSync, self)._fixed_batch(
                        saved)), receipt_origin)
                return super()._fixed_batch(saved, receipt_origin=receipt_origin)
            except _Divergence as divergence:
                undecided = [index for index in self.phase_pass_ambiguous if index not in self.phase_pass_consume]
                if not undecided:
                    raise
                if self.single_pass:
                    raise SyncError("the divergence at packet %d leaves open whether an omitted pass took the WAITING "
                                    "at %d" % (divergence.index, undecided[-1])) from divergence
                self.phase_pass_consume.add(undecided[-1])
                self._load(saved)
                self.stats.replays += 1

    def _settled(self, prompt):
        """The batch's prompt; at the seat's own prompt, first the opponent's hidden identities are aligned to the
        seat's public ledger (no opponent choice is open then)."""
        if self.public_identities_only and prompt is not None and _prompt_player(prompt) == self.viewer:
            self._align_public_identities()
            if self.public_opponent_recipe is not None:
                from .client_public_recipe import native_rows
                self.public_opponent_recipe.remaining(native_rows(self)[0], opponent=1 - self.viewer)
        return prompt

    def _align_public_identities(self):
        """Hold exactly the opponent identities the seat's public ledger holds at this own prompt, no more, no less.

        The follower runs one concrete duel, so a hidden move must take some
        object: after the opponent's hand was shuffled and a face-down card
        left it, the local duel still names one object where the stream no
        longer does ("phantom" knowledge). Evidence fixes and hypotheses also
        name objects only a batch needs. Here every such identity goes back to a
        placeholder, and a card the ledger knows in the hand without a slot is
        kept on (or put on) one hand object: the hand is a multiset to the
        particles, the deck and extra deck an order they sample.
        """
        if self.public_board_problems:
            raise SyncError("the seat's public ledger is unavailable: " + self.public_board_problems[0])
        if self.answered >= len(self.public_identities):
            raise SyncError("no public ledger reading for this own prompt")
        target = self.public_identities[self.answered]
        anchored = {(location, sequence): code for location, sequence, code in target.anchored}
        opponent, pduel = 1 - self.viewer, self.local.pduel
        for location in _ALIGNED_ZONES:
            count = 7 if location == C.LOCATION_MZONE else 8 if location == C.LOCATION_SZONE \
                else self.core.query_field_count(pduel, opponent, location)
            hidden = {}
            for sequence in range(count):
                try:
                    code = card_code(self.core, pduel, opponent, location, sequence)
                except HydrationError:
                    continue  # an empty field zone
                if location not in (C.LOCATION_HAND, C.LOCATION_DECK) \
                        and self._recorded_query("position", opponent, location, sequence) & C.POS_FACEUP:
                    continue  # a public card
                hidden[sequence] = code
            for sequence in sorted(hidden):
                wanted = anchored.get((location, sequence))
                if not wanted or hidden[sequence] == wanted:
                    continue
                donor = next((slot for slot, code in sorted(hidden.items()) if code == wanted
                              and (location, slot) not in anchored), None)
                if donor is not None and location in (C.LOCATION_HAND, C.LOCATION_DECK):
                    # The object that carries this card moves to its proved slot; its history goes with it.
                    self._rebind(opponent, location, sequence, donor, wanted)
                    hidden[sequence], hidden[donor] = wanted, hidden[sequence]
                else:
                    hidden[sequence] = self._align_identity(location, sequence, hidden[sequence], wanted)
            free = {slot: code for slot, code in hidden.items() if (location, slot) not in anchored}
            if location in (C.LOCATION_HAND, C.LOCATION_DECK):
                # A set proxy stood for a card set face down; back in the hand or the deck (returned, shuffled in)
                # nothing says what kind of card it is.
                for slot in sorted(free):
                    if free[slot] in BLANK_SETS:
                        free[slot] = self._align_identity(location, slot, free[slot], None)
            wanted = Counter(dict(target.hand)) if location == C.LOCATION_HAND else Counter()
            held = Counter(code for code in free.values() if code not in BLANK_CODES)
            excess, deficit = held - wanted, wanted - held
            for slot in sorted(free, reverse=True):
                if excess[free[slot]] > 0:
                    excess[free[slot]] -= 1
                    free[slot] = self._align_identity(location, slot, free[slot], None)
            blanks = [slot for slot in sorted(free) if free[slot] in BLANK_CODES]
            for code in sorted(deficit.elements()):
                if not blanks:
                    raise SyncError("the public hand identities outnumber the local hand objects")
                slot = blanks.pop(0)
                free[slot] = self._align_identity(location, slot, free[slot], code)

    def _recorded_query(self, kind, controller, location, sequence):
        # QUERY_POSITION and QUERY_REASON update ocgcore's q_cache even with
        # use_cache=0. They are not physical-card mutations, but a complete
        # accepted native journal must preserve their ordering too.
        from .client_replay_journal import query
        function = {"position": card_position, "reason": card_reason}[kind]
        args = controller, location, sequence
        return query(self, kind, args, lambda: function(self.core, self.local.pduel, *args))

    def _align_identity(self, location, sequence, old, new):
        """Put ``new`` (None: a placeholder) on one hidden opponent card holding ``old``; the code it holds after.

        A card the core may not blank now (an open chain or choice still holds its
        effect) keeps its identity and is counted: the search root then fails the
        public identity check instead of hiding the difference.
        """
        from .client_origin_receipt import mutation
        opponent, pduel = 1 - self.viewer, self.local.pduel
        if new is not None:
            self._require_recipe(opponent, location, sequence, int(new))
        if old not in BLANK_CODES or old in BLANK_SETS and location in (C.LOCATION_HAND, C.LOCATION_DECK):
            kind = self.core.card_pool().cards.get(old)
            placeholder = BLANK_EXTRA if kind is not None and kind.type & _EXTRA_TYPES \
                else (BLANK_FIELD if sequence == 5 else BLANK_SSET) if location == C.LOCATION_SZONE else BLANK_MAIN
            if not mutation(self, "public_align", lambda: blank(self.core, pduel, opponent, location, sequence, old,
                                                                  placeholder),
                            target=(opponent, location, sequence), code=placeholder,
                            request=(opponent, location, sequence, old, placeholder)):
                self.stats.alignment_refused += 1
                return old
            self.stats.alignment_blanked += 1
            old = placeholder
        if new is None:
            return old
        mutation(self, "public_align", lambda: hydrate(self.core, pduel, opponent, location, sequence, int(new)),
                 target=(opponent, location, sequence), code=int(new), request=(opponent, location, sequence, old, new))
        self.stats.alignment_hydrated += 1
        return new

    def _force_shuffle(self, order):
        if self.record_origins:
            from .client_origin_receipt import mutation
            return mutation(self, "force_shuffle", lambda: super(BlankClientSync, self)._force_shuffle(order),
                            request=("clear",) if order is None else (*order[:-1], tuple(order[-1])))
        return super()._force_shuffle(order)

    def _force_random(self, outcome):
        # The outcome is what the received packets show; forcing it takes nothing the seat does not see.
        if self.record_origins:
            from .client_origin_receipt import mutation
            return mutation(self, "force_random", lambda: super(BlankClientSync, self)._force_random(outcome),
                            request=("clear",) if outcome is None else (outcome[0], outcome[1]))
        return super()._force_random(outcome)

    def _match(self, message):
        from .client_sync import _Divergence
        while True:
            start = self.cursor
            try:
                return super()._match(message)
            except _Divergence as divergence:
                if not self.declined_prompt_audit or not self._declined_unasked(divergence, start):
                    raise

    #: the seat's prompts whose records the core may list in another order than the server's
    _PERMUTED_PROMPTS = frozenset((C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD, C.MSG_SELECT_CHAIN, C.MSG_SELECT_CARD,
                                   C.MSG_SELECT_UNSELECT_CARD))

    def _equivalent(self, real, local, message):
        """The seat's prompt listing the same options in another order (the core lists same-named cards by object),
        or a hidden card leaving a hand from another slot that holds the same card (:meth:`_shuffled_hand_slot`,
        :meth:`_copy_in_hand`)."""
        from .client_sync import _Divergence
        if message.msg in self._PERMUTED_PROMPTS and _prompt_player(message) == self.viewer and local[:1] == real[:1]:
            mapping = card_permutation(real, local) if message.msg == C.MSG_SELECT_CARD \
                else unselect_permutation(real, local) if message.msg == C.MSG_SELECT_UNSELECT_CARD \
                else prompt_permutation(real, local)
            if mapping is None and message.msg == C.MSG_SELECT_CHAIN:
                mapping = chain_subset_mapping(real, local)
            if mapping is None:
                return False
            # ``_respond`` finds it at the cursor after the prompt, by the local prompt's own bytes.
            self.prompt_maps[(self.cursor + 1, local)] = mapping
            self.semantic_remaps += 1
            return True
        divergence = _Divergence(self.cursor, real, local)
        if self._shuffled_hand_slot(divergence, self.cursor) or self._copy_in_hand(divergence, self.cursor):
            self.semantic_remaps += 1
            return True
        return False

    def _copy_in_hand(self, divergence, start):
        """A known card left a hand from another slot than the local one, and the local hand held the same card there
        too: two copies of one card in a hand are the same card to the rules, and either way the hand keeps the same
        cards in the same slots. Which object the core took (after a shuffle, say) is not a divergence."""
        from .client_sync import _comparable
        real, local = divergence.real, divergence.local
        if divergence.index != start or len(real) < 17 or len(local) != len(real) or real[0] != C.MSG_MOVE \
                or local[0] != C.MSG_MOVE or real[6] != C.LOCATION_HAND or real[5] != local[5] or real[7] == local[7]:
            return False
        code = struct.unpack_from("<I", real, 1)[0] & 0x7FFFFFFF
        if not code or _comparable(real[:7] + local[7:8] + real[8:], self.viewer) != _comparable(local, self.viewer):
            return False
        # The local duel already moved its copy out of slot ``local[7]``: the real one's slot is one lower after it.
        controller, taken, shown = real[5], local[7], real[7]
        slot = shown - (shown > taken)
        try:
            return card_code(self.core, self.local.pduel, controller, C.LOCATION_HAND, slot) == code
        except HydrationError:
            return False

    def _shuffled_hand_slot(self, divergence, start):
        """A hidden card left the opponent's shuffled hand from another slot than the local one.

        After a shuffle of that hand its slots name no card until the stream
        shows one again (a confirmation, a card moved in face up). With no such
        packet since the shuffle, which slot the hidden card left is not a
        divergence: the local shuffle put its objects elsewhere.
        """
        real, local = divergence.real, divergence.local
        opponent = 1 - self.viewer
        if divergence.index != start or len(real) < 17 or len(local) != len(real) or real[0] != C.MSG_MOVE \
                or local[0] != C.MSG_MOVE or real[1:5] != bytes(4) or local[1:5] != bytes(4) \
                or tuple(real[5:7]) != (opponent, C.LOCATION_HAND) or real[:7] != local[:7] or real[8:] != local[8:]:
            return False
        for packet in reversed(self.packets[:divergence.index]):
            msg = packet[0]
            if msg == C.MSG_SHUFFLE_HAND and len(packet) >= 2 and packet[1] == opponent:
                return True
            if msg == C.MSG_MOVE and len(packet) >= 13 and tuple(packet[9:11]) == (opponent, C.LOCATION_HAND) \
                    and struct.unpack_from("<I", packet, 1)[0] & 0x7FFFFFFF:
                return False
            if msg == C.MSG_CONFIRM_CARDS and len(packet) >= 4 and any(
                    tuple(packet[offset + 4:offset + 6]) == (opponent, C.LOCATION_HAND)
                    for offset in range(4, len(packet) - 6, 7)):
                return False
        return False

    def _declined_unasked(self, divergence, start):
        """Take a received WAITING the local duel has no prompt for, when the real stream then goes on with the local
        packet: the opponent was asked something only its hidden cards allowed (an optional trigger whose target is
        in its deck, a chain window a hand trap opens) and declined, which leaves no public trace and no state the
        local duel lacks. Recorded as an unvalidated history receipt."""
        from .client_sync import _Frontier, _comparable
        index = divergence.index
        if index != start or divergence.real != bytes([C.MSG_WAITING]) or divergence.local == divergence.real:
            return False
        if index + 1 >= len(self.packets):
            raise _Frontier()
        if _comparable(self.packets[index + 1], self.viewer) != _comparable(divergence.local, self.viewer):
            return False
        if self.record_origins:
            from .client_origin_receipt import PublicMutation, PublicSource
            source = PublicSource(index, self.receipt_packet_raw_indices[index], self.packets[index],
                                  "opaque_declined_prompt")
            operation = PublicMutation("opaque_declined_prompt", (index,), None, 0, 0, False, (source,),
                                       ("opaque_declined_prompt_unvalidated",))
            self.receipt_state = replace(self.receipt_state, pending=self.receipt_state.pending + (operation,))
        self.cursor += 1
        self.stats.declined_prompts += 1
        return True

    def _run(self):
        try:
            return self._run_deferring()
        finally:
            # The batch start of the seat's prompt serves only the batches up to the next prompt.
            self._free(self.own_origin)
            self.own_origin = None

    def _run_deferring(self):
        """The inherited batch loop; a batch after our own response may defer on a hidden-target activation. In a
        single pass any batch may, where the real packet is an opponent's chain link the local duel did not write
        (it had no window for it: its legality needs a hidden card the packets show later)."""
        if not self.hidden_target_deferral or self.deferred_action is not None \
                or not (self.last_response_own or self.single_pass):
            return super()._run()
        from .client_hidden_target import HiddenTargetDeferred, begin
        from .client_sync import _Divergence, _GroupBeforeBatch
        while not self.local.finished:
            self._constructing(self.cursor)
            if self.single_pass and (not self.answer_states or self.answer_states[-1][3] != self.answered):
                self._keep_answer_state()
            saved = self._save()
            try:
                prompt = self._fixed_batch(saved)
            except _GroupBeforeBatch as early:
                self._free(saved)
                if not self._order_before(early.uids):
                    raise
                self.stats.replays += 1
                continue
            except _Divergence as divergence:
                if self.single_pass:
                    # An opponent's chain link the local duel did not write: its legality needs a hidden card, and it
                    # is forced from a construction point (plan 5.11). A divergence before the activation a
                    # construction is being proved for goes back to it instead.
                    if not (self._opponent_link(divergence) and not self._constructing(divergence.index)
                            and self._force_construction(saved, divergence.index)):
                        self._free(saved)
                    raise
                opponent_link = self._opponent_link(divergence)
                if opponent_link and self.own_origin is not None:
                    # The first batch after the seat's answer: the link's legality was settled before its prompt.
                    state = begin(self, self.own_origin, divergence.index)
                    if state is not None:
                        self._free(saved)
                        self.hidden_target_origin, self.own_origin = self.own_origin, None
                        raise HiddenTargetDeferred(state) from divergence
                state = begin(self, saved, divergence.index) \
                    if self.last_response_own and not self.single_pass or opponent_link else None
                if state is None:
                    self._free(saved)
                    raise
                # The batch start stays alive until the hidden target is public.
                self.hidden_target_origin = saved
                raise HiddenTargetDeferred(state) from divergence
            except BaseException:
                self._free(saved)
                raise
            if prompt is not None:
                self._constructing(self.cursor)
                return prompt, saved
            self._free(saved)
        return None, None

    def _constructing(self, index) -> bool:
        """Whether the construction being proved still covers ``index`` (the activation is not written yet); once the
        follower is past the activation, the construction is proved and dropped."""
        if self.construction is None:
            return False
        if index <= self.construction.state.chaining_index:
            return True
        from .client_hidden_target import drop_construction
        drop_construction(self)
        return False

    def _force_construction(self, first, index):
        """Single pass: force the opponent activation the packets show from ``index`` from a construction point
        (plan 5.11) -- the batch starts still held, oldest first, then ``first``, the newest, owned by the
        construction from now -- and raise ``Reconstructed``: the batch loop starts again from the state loaded.

        False (``first`` still the caller's) when the packets show no such activation there, or when the newest batch
        start already offers it unforced: its legality waits for no hidden card, and the divergence is the ordinary
        follower's (an open decision, else the failure). The same activation met again from the same oldest batch
        start means no window the stream allows took it: that is the failure, raised at once."""
        from .client_hidden_target import Construction, Reconstructed, _category, _offered, construct, forced_activation
        forced = forced_activation(self, index)
        if forced is None or forced.chaining_index == self.hidden_target_skip:
            return False
        if _offered(self, first, (), forced.code, forced.description):
            self.hidden_target_skip = forced.chaining_index
            return False
        # The candidates the activation's legality allows are a public category fact: particles must hold one. They
        # are read at the newest batch start, whose first opponent window is the activation's (from an older one,
        # the first window reached is an earlier one), with nothing forced; only claimed, never written.
        fact = _category(self, first, [], forced.code, forced.chaining_index) if self.hidden_target_pool else None
        older = self._older_batch_starts(first)
        oldest = older[-1] if older else first
        key = (forced.chaining_index, oldest[2], oldest[3])
        if key in self.constructions_tried:
            for state in older:
                self._free(state)
            raise SyncError("the opponent's activation at packet %d was forced again from the batch start at packet %d: "
                            "no window the stream allows takes it" % (forced.chaining_index, oldest[2]))
        self.constructions_tried = self.constructions_tried | {key}
        self.construction = Construction(forced, (*reversed(older), first), tuple(self.choices),
                                         tuple(self.answer_states), tuple(self.category_constraints),
                                         fact and (*fact, first[2]))
        construct(self)
        raise Reconstructed()

    def _force_activation(self, forced):
        """Have the local core take the opponent activation ``forced`` as legal: the stream shows it was made, and
        its legality reads hidden cards the local duel holds as blanks (``duel_force_activation``)."""
        lib = self.core._lib
        operation = lambda: lib.duel_force_activation(
            ctypes.c_void_p(self.local.pduel), forced.controller, ctypes.c_uint32(forced.code),
            ctypes.c_uint32(forced.description))
        if self.record_origins:
            from .client_origin_receipt import mutation
            result = mutation(self, "force_activation", operation,
                              request=(forced.controller, forced.code, forced.description, forced.chaining_index))
        else:
            result = operation()
        if result != 0:
            raise SyncError("the local core refused (%d) to force the opponent's activation at packet %d"
                            % (result, forced.chaining_index))
        self.stats.activations_forced += 1

    def _older_batch_starts(self, first):
        """Copies of the batch starts before ``first`` the follower still holds back to the oldest kept answer state,
        newest first: the states right after the seat's answers, the batches that wrote the opponent prompts, and the
        one that wrote the seat's prompt.

        They are ordered by when they were taken, not by stream position: the opponent's WAITINGs take no packet, so
        the batches of several opponent prompts in a row start at one cursor with one answer count."""
        # Taken in this order for each count of the seat's answers: the state right after the answer, the batch
        # starts of the opponent prompts in order, the batch that wrote the seat's next prompt; ``first`` when it is
        # none of them is the batch start taken last.
        taken = [((state[3], 0, index), state) for index, state in enumerate(self.answer_states)]
        taken += [((choice.origin[3], 1, index), choice.origin) for index, choice in enumerate(self.choices)
                  if choice.origin is not None]
        if self.own_origin is not None:
            taken.append(((self.own_origin[3], 2, 0), self.own_origin))
        newest = next((key for key, state in taken if state is first), (first[3], 3, 0))
        horizon = taken[0][0] if self.answer_states else newest
        copies = []
        for key, state in sorted(taken, key=lambda item: item[0], reverse=True):
            if horizon <= key < newest:
                self._load(state)
                copies.append(self._save())
        if copies:
            self._load(first)
        return tuple(copies)

    def _keep_answer_state(self):
        """The state right after the seat's latest answer (or at the start), kept for the next few answers."""
        self.answer_states.append(self._save())
        while len(self.answer_states) > _ANSWER_STATES:
            self._free(self.answer_states.pop(0))

    def _order_before(self, uids) -> bool:
        """Put the group order ``uids`` in place right after the latest answer of the seat before the operation
        asked it (the latest kept state where the core takes the order), and run on from there: the opponent
        choices and category facts read since belong to the run given up, the seat's answers are given again."""
        if tuple(uids) in self.order_rewinds:
            return False  # the run from before the operation met the same order again: it did not take
        self.order_rewinds.add(tuple(uids))
        for position in range(len(self.answer_states) - 1, -1, -1):
            state = self.answer_states[position]
            self._load(state)
            try:
                self._order_cards(uids)
            except GroupAsked:
                continue
            for later in self.answer_states[position:]:
                self._free(later)
            del self.answer_states[position:]
            self._free(self.own_origin)
            self.own_origin = None
            while self.choices and self.choices[-1].begin >= state[2]:
                self._release(self.choices.pop())
            self.category_constraints = [fact for fact in self.category_constraints if fact[4] < state[2]]
            return True
        return False

    def _opponent_link(self, divergence) -> bool:
        """Whether the real packet is an opponent's chain link, or the prompt that led straight to it: an activation
        the local duel did not offer, whose legality may need a hidden card the packets show later."""
        return self._link_at(divergence.index)

    def _link_at(self, index) -> bool:
        """Whether the received packets from ``index`` go straight (prompts and hints aside) to an opponent's chain
        link."""
        link = next((packet for packet in self.packets[index:] if packet[0] not in (C.MSG_WAITING, C.MSG_HINT)), b"")
        return len(link) >= 8 and link[0] == C.MSG_CHAINING and link[5] != self.viewer

    @property
    def _late_reveals(self):
        return not self.single_pass

    @property
    def _unseen_calls_join(self):
        # A single pass writes its evidence where the batch begins, so the batch begins before every core call that
        # could read it; the searching follower keeps its old batches, which its late reveals go back to.
        return self.single_pass

    @property
    def _order_tolerant(self):
        # A single pass needs the board the server has at the seat's prompts, not the order in which the players
        # were asked on the way (plan 5.10): the hidden cards the local duel lacks change that order.
        return self.single_pass

    def _backtrack(self, divergence):
        """Single pass: the choices ever gone back to are those with a proved alternative, when a later packet
        refutes the one taken: an open decision (every answer proved through the packets to the seat's next
        prompt) and an open phase exit (at the packet that tells its answers apart). Anything else is a failure,
        raised at once."""
        if not self.single_pass:
            return super()._backtrack(divergence)
        from .client_answer_evidence import settles
        from .client_sync import _Divergence, _EvidenceConflict, _Unread
        self._drop_origin()
        self.parked = None
        plain = isinstance(divergence, _Divergence) and not isinstance(divergence, (_Unread, _EvidenceConflict))

        def alternative(choice):
            return choice.next < len(choice.answers) and (choice.open or plain and settles(
                choice.prompt.msg, self.packets, choice.begin, divergence.index, divergence.real, divergence.local))

        index = next((number for number in range(len(self.choices) - 1, -1, -1)
                      if alternative(self.choices[number])), None) if isinstance(divergence, _Divergence) else None
        pending = self.construction
        if isinstance(divergence, _Divergence) and self._constructing(divergence.index) \
                and (index is None or self.choices[index].begin < pending.cursor):
            # The batch start replayed from does not make the activation legal: the next older one.
            from .client_hidden_target import construct
            self.stats.counts[(divergence.index, divergence.real.hex()[:40], divergence.local.hex()[:40])] += 1
            construct(self)
            return None
        if index is not None:
            self.stats.counts[(divergence.index, divergence.real.hex()[:40], divergence.local.hex()[:40])] += 1
            while len(self.choices) > index + 1:
                self._release(self.choices.pop())
            choice = self.choices[index]
            # Facts read and states kept after the choice belong to the refuted branch; the replay reads and keeps
            # them again.
            self.category_constraints = [fact for fact in self.category_constraints if fact[4] <= choice.begin]
            self._drop_answer_states(choice.saved[3])
            choice.refuted = str(self._unfollowable(divergence))
            self._try(choice)
            return None
        raise self._unfollowable(divergence)

    def _drop_answer_states(self, answered):
        """Forget the states kept after the seat's answers beyond the first ``answered``: they belong to a run given
        up, and the replay keeps its own when it answers again."""
        while self.answer_states and self.answer_states[-1][3] > answered:
            self._free(self.answer_states.pop())

    def _try(self, choice):
        # Another answer opens another branch: an activation that needed no hidden card on one may on the next. A
        # single pass has no other branch: an activation already waited for is not waited for again.
        if not self.single_pass:
            self.hidden_target_skip = None
        super()._try(choice)

    def _rewrite(self, choice):
        """The omitted pass's alignment first; then the inherited placements; then an opponent prompt that
        no native answer can turn into the public activation the stream shows next may defer from the
        start of its batch."""
        if self._phase_pass_shift(choice):
            return True
        if self.single_pass:
            # Nothing is rewritten on a divergence: a deferral starts only where the menu lacks the card the stream
            # activates (``_prepare_choice``).
            return False
        if super()._rewrite(choice):
            return True
        if not self.hidden_target_deferral or self.deferred_action is not None or choice.origin is None:
            return False
        from .client_hidden_target import HiddenTargetDeferred, begin
        state = begin(self, choice.origin, choice.begin)
        if state is None:
            return False
        self.hidden_target_origin, choice.origin = choice.origin, None
        raise HiddenTargetDeferred(state)

    def _respond(self, message, data):
        self.last_response_own = _prompt_player(message) == self.viewer
        if _prompt_player(message) == self.viewer:
            raw = bytes([message.msg]) + message.payload
            mapping = self.prompt_maps.get((self.cursor, raw))
            if isinstance(mapping, tuple) and data == struct.pack("<i", -1):
                pass  # finishing or cancelling the selection names no card
            elif isinstance(mapping, tuple):
                # A card selection: a count, then that many indices into the received list.
                if not data or len(data) != 1 + data[0] or any(index >= len(mapping) for index in data[1:]):
                    raise SyncError("response is not a selection of the permuted card list")
                data = bytes([data[0]] + [mapping[index] for index in data[1:]])
            elif mapping is not None:
                if data not in mapping:
                    raise SyncError("response is not in the certified prompt permutation")
                data = mapping[data]
        if self.record_replay:
            from .client_replay_journal import append, capture
            before = capture(self)
            super()._respond(message, data)
            append(self, "response", before, (bytes([message.msg]) + bytes(message.payload), bytes(data)))
        else:
            super()._respond(message, data)

    def _public_idle_reveal(self, choice):
        """An already-received hand activation with an unchanged source slot.

        Opaque waits/hints alone cannot select an identity. Require the first
        public state transition to be MOVE, followed by the SAME card's
        CHAINING at its exact destination. Draws, shuffles, other moves and
        our private prompts are hard boundaries. The local UID join is only
        between the already-owned origin and prompt snapshots.
        """
        if (not self.public_action_rebind or not self.record_origins or choice.prompt.msg != C.MSG_SELECT_IDLECMD
                or _prompt_player(choice.prompt) == self.viewer or choice.origin is None
                or choice.fixes is not None):
            return None
        origin = self.receipt_saved.get(choice.origin[0])
        current = self.receipt_saved.get(choice.saved[0])
        if origin is None or current is None or current.cursor != choice.begin + 1:
            return None
        # Even the origin batch must not have moved/reordered the hand before
        # writing its prompt. No fabricated cross-batch coordinate lease.
        if any(packet[0] not in _STILL_PACKETS
               for packet in self.packets[origin.cursor:choice.begin + 1]):
            return None
        move = None
        for packet in self.packets[choice.begin + 1:]:
            if packet[0] in (C.MSG_WAITING, C.MSG_HINT):
                continue
            if move is None and len(packet) == 17 and packet[0] == C.MSG_CHAINING:
                # A monster effect activated in the hand: the card stays where it is, its CHAINING names it.
                code = struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff
                player, location, sequence = packet[5:8]
                if not code or player != 1 - self.viewer or location != C.LOCATION_HAND:
                    return None
                move, request = packet, (player, location, sequence, code)
            elif move is None:
                if len(packet) != 17 or packet[0] != C.MSG_MOVE:
                    return None
                code = struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff
                player, location, sequence = packet[5:8]
                if (not code or player != 1 - self.viewer or location != C.LOCATION_HAND
                        or packet[9] != player or packet[10] != C.LOCATION_SZONE
                        or not packet[12] & C.POS_FACEUP):
                    return None
                move = packet
                request = (player, location, sequence, code)
                continue
            if move is not packet and (len(packet) != 17 or packet[0] != C.MSG_CHAINING
                    or struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff != request[3]
                    or packet[5:9] != move[9:13] or packet[9:12] != move[9:12]):
                return None
            coordinate = request[:3]
            before = [r for r in origin.entities if (r.controller, r.location, r.sequence) == coordinate]
            after = [r for r in current.entities if (r.controller, r.location, r.sequence) == coordinate]
            if len(before) != 1 or len(after) != 1 or before[0].uid != after[0].uid:
                return None
            # Another object holds the slot (a known card, or a blank while the known card waits in the deck):
            # with hidden rebinding the binding's placement exchanges objects the public stream cannot tell apart.
            if (before[0].placeholder != 1 or after[0].placeholder != 1) and not getattr(self, "hidden_hand_rebind",
                                                                                           False):
                return None
            return request, struct.unpack_from("<I", packet, 12)[0]
        return None

    def _public_idle_set(self, choice):
        """An already-received face-down set of a hidden hand card: ``(player, HAND, slot, proxy)``, or None.

        The first public transition after the opponent's IDLE prompt (waits and
        hints aside) is the MOVE of a face-down card from its hand to a spell/trap
        zone. Only a card that can be set there left that slot, so the slot takes
        the set proxy (the field zone, sequence 5, the field proxy) before the menu
        is enumerated; its printed identity stays unknown.
        """
        if (not self.record_origins or choice.prompt.msg != C.MSG_SELECT_IDLECMD
                or _prompt_player(choice.prompt) == self.viewer or choice.origin is None
                or choice.fixes is not None):
            return None
        # The batch that wrote the prompt must not have moved the hand.
        if any(packet[0] not in _STILL_PACKETS for packet in self.packets[choice.origin[2]:choice.begin + 1]):
            return None
        replaced = False
        for packet in self.packets[choice.begin + 1:]:
            if packet[0] in (C.MSG_WAITING, C.MSG_HINT):
                continue
            if (not replaced and len(packet) == 17 and packet[0] == C.MSG_MOVE and packet[5] == 1 - self.viewer
                    and tuple(packet[6:8]) == (C.LOCATION_SZONE, 5) and packet[10] == C.LOCATION_GRAVE):
                # A field spell set into an occupied field zone sends the old one to the graveyard first.
                replaced = True
                continue
            if len(packet) != 17 or packet[0] != C.MSG_MOVE or struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff:
                return None
            player, location, sequence = packet[5:8]
            if (player != 1 - self.viewer or location != C.LOCATION_HAND or packet[9] != player
                    or packet[10] != C.LOCATION_SZONE or packet[11] > 5 or packet[12] & C.POS_FACEUP
                    or replaced and packet[11] != 5):
                return None
            if card_code(self.core, self.local.pduel, player, location, sequence) != BLANK_MAIN:
                return None
            return player, location, sequence, BLANK_FIELD if packet[11] == 5 else BLANK_SSET
        return None

    def _bind_public_set(self, choice, reveal) -> bool:
        """Write the IDLE prompt again with the set proxy in place and keep only the answer that sets it; False
        (the choice untouched) when the native menu cannot reproduce it."""
        from .client_origin_receipt import history_attribution
        from .client_root import _clone, _driver_state
        driver_before = _clone(_driver_state(self.local), self.core.card_pool())
        backup = self._save()
        previous = (choice.saved, choice.prompt, choice.answers, choice.next, choice.fixes)
        try:
            choice.fixes = [(reveal,)]
            with history_attribution(self, "public_action_menu_rebind_unvalidated"):
                if not self._rewrite(choice):
                    raise SyncError("public set could not regenerate its original IDLE menu")
                player, location, sequence, code = reveal
                record = struct.pack("<IBBB", code, player, location, sequence)
                matches = [i for i, row in enumerate(self._idle_records(choice.prompt)[4]) if row == record]
                response = struct.pack("<i", matches[0] << 16 | 4) if len(matches) == 1 else None
                if response is None or response not in choice.answers:
                    raise SyncError("public set is absent from the native encoded answers")
                # The set has ALREADY happened in the received public prefix.
                choice.answers = [response]
            return True
        except SyncError:
            self._load(backup)
            self.local.restore_pystate(driver_before, consume=True)
            if choice.saved is not previous[0]:
                self._free(choice.saved)
                choice.saved, backup = backup, None  # transfer one live handle
            choice.prompt, choice.answers, choice.next, choice.fixes = previous[1:]
            return False
        finally:
            self._free(backup)

    @staticmethod
    def _idle_records(prompt):
        body, offset = prompt.payload, 1
        groups = []
        try:
            if prompt.msg != C.MSG_SELECT_IDLECMD or body[0] not in (0, 1):
                raise ValueError("not an IDLE prompt")
            for group in range(6):
                count = body[offset]
                offset += 1
                width = 11 if group == 5 else 7
                groups.append(tuple(body[offset + width*i:offset + width*(i+1)] for i in range(count)))
                offset += width * count
            if offset + 3 != len(body):
                raise ValueError("invalid IDLE envelope")
        except (IndexError, ValueError, struct.error) as exc:
            raise SyncError("invalid native IDLE menu") from exc
        return groups

    @staticmethod
    def _public_activation_answer(prompt, reveal, description):
        """Exact native IDLE activation record, never text/display matching."""
        player, location, sequence, code = reveal
        record = struct.pack("<IBBBI", code, player, location, sequence, description)
        matches = [i for i, row in enumerate(BlankClientSync._idle_records(prompt)[5]) if row == record]
        if len(matches) != 1:
            raise SyncError("public activation has no unique exact native IDLE record")
        return struct.pack("<i", matches[0] << 16 | 5)

    @staticmethod
    def _public_special_answer(prompt, reveal):
        player, location, sequence, code = reveal
        record = struct.pack("<IBBB", code, player, location, sequence)
        matches = [i for i, row in enumerate(BlankClientSync._idle_records(prompt)[1]) if row == record]
        if len(matches) != 1:
            raise SyncError("public special summon has no unique exact native IDLE record")
        return struct.pack("<i", matches[0] << 16 | 1)

    def _phase_pass_shift(self, choice):
        """Every answer of an opponent prompt diverged, and an omitted opponent pass kept the WAITING the
        prompt took: the pass takes it and the prompt the next WAITING.

        The local duel is the same either way; only the stream alignment and the
        pass's receipt differ. A later rewrite of the prompt from its batch's
        start keeps the pass taking that packet until the choice is released.
        """
        if choice.fixes is not None:  # an explicit public patch, or placements already under way
            return False
        receipt, choice.pass_receipt = choice.pass_receipt, None
        after = choice.begin + 1
        if receipt is None or after >= len(self.packets) or self.packets[after] != bytes([C.MSG_WAITING]):
            return False
        self._load(choice.saved)
        self.cursor += 1
        self.receipt_state = replace(self.receipt_state, pending=self.receipt_state.pending + (receipt,))
        replacement = self._save()
        self._free(choice.saved)
        self.phase_pass_consume.add(choice.begin)
        choice.pass_taken, choice.saved, choice.begin, choice.fixes = choice.begin, replacement, after, None
        choice.answers, choice.next = self._choice_answers(choice), 0
        self.stats.replays += 1
        return bool(choice.answers)

    def _choice_answers(self, choice):
        """The answer the received packets show, when they show one; else the ordinary answer search."""
        name = _MSG_NAMES.get(choice.prompt.msg, str(choice.prompt.msg))
        if self.read_answers:
            from .client_answer_evidence import read
            reading = read(choice.prompt.msg, choice.prompt.payload, self.packets, choice.begin, self.viewer)
            if reading.answers and reading.by_consequence and self.single_pass:
                # Hidden picks: placeholders are one class, each known card another. The packets to the seat's
                # next prompt must allow the pick; when several classes stay allowed, they differ only in which
                # hidden object went where, which the alignment there makes the public ledger's again: the first
                # (a placeholder) stands for them. Different decisions (a link chained in this window or a later
                # one) all stay: an open decision.
                fits = self._consequence_answers(choice, reading.answers)
                if fits:
                    self.stats.answers_read[name] += 1
                    if reading.open:
                        choice.open = len(fits) > 1
                        self.stats.answers_open[name] += choice.open
                        return fits
                    self.stats.picks_equivalent += len(fits) > 1
                    return fits[:1]
                reading = replace(reading, answers=(),
                                  reason="%d of the picks the packets allow have the received consequences" % len(fits))
            if reading.answers:
                self.stats.answers_read[name] += 1
                return list(reading.answers)
            if self.reading_only:
                return []
            if self.single_pass and choice.prompt.msg in _SCRIPT_QUESTIONS:
                fits = self._consequence_answers(choice)
                if fits:
                    # Several answers proved to the seat's next prompt: the packets tell them apart only later
                    # (the opponent may have passed a window its hidden cards opened). The first is taken, and a
                    # later packet refuting it takes the next: an open decision, followed in parallel lazily.
                    self.stats.answers_read[name] += 1
                    choice.open = len(fits) > 1
                    self.stats.answers_open[name] += choice.open
                    return fits
                reading = replace(reading, reason="%d answers' own consequences are the received packets" % len(fits))
            if self.single_pass:
                self.stats.answers_unread[name] += 1
                choice.unread = reading.reason or "identities %s were not written" % (reading.placements,)
                return []
        self.stats.answers_searched[name] += 1
        return super()._choice_answers(choice)

    #: prompts on the way whose answers a consequence check may play one by one (the packets do not tell them)
    _CONSEQUENCE_BRANCHES = 64

    def _consequence_answers(self, choice, answers=None):
        """The answers (by default every one) whose own consequences are the received packets: an answer fits when the
        local duel, from the prompt on, reaches the seat's next prompt through exactly the received packets. Each
        opponent prompt on the way takes the answer the packets show; where they do not tell it, some answer of it
        must lead on, and each is played until one does. The engine proves every step; nothing passes unproved, and
        more branches than the bound are an error."""
        fits = []
        deferral, self.hidden_target_deferral = self.hidden_target_deferral, False
        budget = [self._CONSEQUENCE_BRANCHES]
        try:
            for data in (answers if answers is not None else self._answers(choice.prompt)):
                self._load(choice.saved)
                if self._leads_on(choice.prompt, data, budget):
                    fits.append(data)
        finally:
            self.hidden_target_deferral = deferral
            self._load(choice.saved)
        return fits

    def _leads_on(self, prompt, data, budget):
        """Whether ``data`` answering ``prompt`` (from the current local state, which it consumes) leads the local duel
        to the seat's next prompt through the received packets."""
        from .client_answer_evidence import read
        from .client_sync import _Divergence, _Frontier
        try:
            self._respond(prompt, data)
            while True:
                prompt, origin = self._run()
                self._free(origin)
                if prompt is None or _prompt_player(prompt) == self.viewer:
                    return True
                reading = read(prompt.msg, prompt.payload, self.packets, self._prompt_begin(), self.viewer)
                if len(reading.answers) == 1:
                    self._respond(prompt, reading.answers[0])
                    continue
                if not reading.answers and self._link_at(self._prompt_begin()):
                    # The packets go straight on to an opponent activation this prompt does not offer: consistent up
                    # to it, as a divergence there is. Its answers are not played: one that leaves the duel where it
                    # was (an attack the core cancels) would come back to this prompt without end.
                    return True
                saved = self._save()
                try:
                    for option in reading.answers or self._answers(prompt):
                        if budget[0] <= 0:
                            raise ConsequenceBranches("the consequences of an answer branch over more than %d prompts "
                                                      "the packets do not tell" % self._CONSEQUENCE_BRANCHES)
                        budget[0] -= 1
                        self._load(saved)
                        if self._leads_on(prompt, option, budget):
                            return True
                    return False
                finally:
                    self._free(saved)
        except _Frontier:
            return True  # consistent up to the last packet received
        except ConsequenceBranches:
            raise
        except _Divergence as divergence:
            # Consistent up to an opponent activation the local duel did not offer: its legality may need a hidden
            # card (the follower waits for its evidence there), which a proof run cannot see.
            return self._opponent_link(divergence)
        except (SyncError, DuelError):
            return False

    def _read_with_identities(self, choice) -> bool:
        """Write the prompt again from its batch start with the identities the received packets name for the
        card the answer acted on, and take the answer they show; False (the choice untouched) when that does not
        give one."""
        from .client_answer_evidence import read
        from .client_origin_receipt import history_attribution
        from .client_root import _clone, _driver_state
        reading = read(choice.prompt.msg, choice.prompt.payload, self.packets, choice.begin, self.viewer)
        if not reading.placements or choice.origin is None or choice.fixes is not None:
            return False
        driver_before = _clone(_driver_state(self.local), self.core.card_pool())
        backup = self._save()
        previous = (choice.saved, choice.prompt, choice.answers, choice.next, choice.fixes)
        try:
            choice.fixes = [reading.placements]
            self.reading_only = True
            with history_attribution(self, "public_action_menu_rebind_unvalidated"):
                if not self._rewrite(choice):
                    raise SyncError("the evidenced identities did not give a readable answer")
            return True
        except SyncError:
            self._load(backup)
            self.local.restore_pystate(driver_before, consume=True)
            if choice.saved is not previous[0]:
                self._free(choice.saved)
                choice.saved, backup = backup, None  # transfer one live handle
            choice.prompt, choice.answers, choice.next, choice.fixes = previous[1:]
            return False
        finally:
            self.reading_only = False
            self._free(backup)

    def _prepare_choice(self, choice):
        choice.pass_receipt = self.phase_pass_kept.get(choice.begin)
        if self.single_pass:
            # The batch that wrote this prompt began with every identity the packets show: the answer is read, or
            # (a few answers told apart only by their consequences) kept open; else the card the stream activates
            # next may need a hidden card whose identity comes later.
            from .client_answer_evidence import read
            reading = read(choice.prompt.msg, choice.prompt.payload, self.packets, choice.begin, self.viewer)
            choice.answers = self._choice_answers(choice)
            if not choice.answers and self.hidden_target_deferral and choice.origin is not None and (
                    reading.placements or self._link_at(choice.begin)):
                # The card the stream activates next has no option here although its identity was written, or the
                # local duel asked something else where the real opponent went on to activate a card (its trigger
                # was not there, and the phase moved on without it): its legality needs another hidden card (a
                # cost, a search or summon target), which the packets show only after the seat's next prompt. The
                # follower waits for them at the batch start.
                if not self._constructing(choice.begin):
                    # The construction owns the batch start from now (the replay releases this choice).
                    origin, choice.origin = choice.origin, None
                    if not self._force_construction(origin, choice.begin):
                        choice.origin = origin
            return None
        if self.read_answers:
            from .client_answer_evidence import read
            if read(choice.prompt.msg, choice.prompt.payload, self.packets, choice.begin, self.viewer).answers:
                choice.answers = self._choice_answers(choice)
                return None
            if self._read_with_identities(choice):
                return None
        evidence = self._public_idle_reveal(choice)
        program = None
        if evidence is None:
            from .client_link_rebind import certify_link
            program = certify_link(self, choice)
            if program is None:
                hidden_set = self._public_idle_set(choice)
                if hidden_set is not None and self._bind_public_set(choice, hidden_set):
                    return None
                if self.hidden_target_deferral:
                    from .client_hidden_target import HiddenTargetDeferred, unoffered
                    state = unoffered(self, choice)
                    if state is not None:
                        self.hidden_target_origin, choice.origin = choice.origin, None
                        raise HiddenTargetDeferred(state)
                return super()._prepare_choice(choice)
            reveal = program.reveal
        else:
            reveal, description = evidence
        current = self.receipt_saved.get(choice.saved[0]) if evidence is not None else None
        # A binding that needs objects exchanged falls back to the ordinary answers when the exchange is refused.
        exchanged = current is not None and not any(
            (r.controller, r.location, r.sequence) == reveal[:3] and r.placeholder == 1 for r in current.entities)
        from .client_origin_receipt import history_attribution
        from .client_root import _clone, _driver_state
        # One exact public-source patch; do not enumerate donor hypotheses or
        # unrelated future reveals. Existing _rewrite retains/frees snapshots.
        driver_before = _clone(_driver_state(self.local), self.core.card_pool())
        backup = self._save()
        previous = (choice.saved, choice.prompt, choice.answers, choice.next, choice.fixes)
        try:
            choice.fixes = [(reveal,)]
            marker = "public_link_menu_rebind_unvalidated" if program else "public_action_menu_rebind_unvalidated"
            with history_attribution(self, marker, proofs=(program.receipt(),) if program else ()):
                if not self._rewrite(choice):
                    raise SyncError("public hand activation could not regenerate its original IDLE menu")
                response = (self._public_special_answer(choice.prompt, reveal) if program
                            else self._public_activation_answer(choice.prompt, reveal, description))
                if response not in choice.answers:
                    raise SyncError("public activation is absent from the native encoded answers")
                # The action has ALREADY happened in the received public
                # prefix. Other IDLE commands contradict that observation;
                # this is constraint binding, not hypothetical action pruning.
                choice.answers = [response]
        except BaseException as exc:
            self._load(backup)
            # Legacy save_pystate copies fields separately. Preserve the exact
            # pre-transaction graph, including the two Python RNG aliases.
            self.local.restore_pystate(driver_before, consume=True)
            if choice.saved is not previous[0]:
                self._free(choice.saved)
                choice.saved, backup = backup, None  # transfer one live handle
            choice.prompt, choice.answers, choice.next, choice.fixes = previous[1:]
            if program is None and exchanged and isinstance(exc, SyncError):
                return super()._prepare_choice(choice)
            unoffered = program is None and isinstance(exc, SyncError) \
                and str(exc) == "public activation has no unique exact native IDLE record"
            if unoffered and self.public_action_target_witness:
                from .client_action_target_witness import begin, TargetWitnessDeferred
                deferred = begin(self, choice, reveal, description)
                if deferred is not None:
                    raise TargetWitnessDeferred(deferred) from exc
            if self.hidden_target_deferral and program is None and isinstance(exc, SyncError) \
                    and choice.origin is not None:
                # The exact patch could not reproduce the public activation natively: its legality
                # needs a hidden card of the blank shadow beyond the activated one.
                from .client_hidden_target import HiddenTargetDeferred, begin as defer
                state = defer(self, choice.origin, choice.begin)
                if state is not None:
                    self.hidden_target_origin, choice.origin = choice.origin, None
                    raise HiddenTargetDeferred(state) from exc
            raise
        finally:
            self._free(backup)

    def advance(self):
        if self.failure is not None:
            raise SyncError("shadow permanently invalid: " + self.failure)
        from .client_action_target_witness import TargetWitnessDeferred
        from .client_hidden_target import DeferredHiddenTarget, HiddenTargetDeferred, Reconstructed
        prior_receipts = self.receipt_state
        try:
            if self.record_origins:
                from .client_origin_receipt import boundary
                boundary(self)
            while True:
                try:
                    if type(self.deferred_action) is DeferredHiddenTarget:
                        from .client_hidden_target import step
                        pending = step(self)
                        if pending is not None:
                            return pending
                    elif self.deferred_action is not None:
                        from .client_action_target_witness import resume
                        if len(self.own) <= self.deferred_action.own_index:
                            return self.deferred_action.message()
                        resume(self)
                    return super().advance()
                except Reconstructed:
                    # Back at a construction point with the opponent's activation forced: follow on from there.
                    self.parked = None
                except (TargetWitnessDeferred, HiddenTargetDeferred) as exc:
                    self.deferred_action = exc.state
                    self.parked = None
                    if type(exc.state) is not DeferredHiddenTarget or len(self.own) <= exc.state.own_index:
                        return exc.state.message()
                    # A replay met the deferral again past prompts already answered: resolve it now.
        except (SyncError, InitializationFailure, DuelError) as exc:
            self.receipt_state = prior_receipts
            self.failure = str(exc)
            refuted = next((choice for choice in reversed(self.choices) if choice.refuted), None)
            if refuted is not None:
                # The failure met on the branch taken last; the one that sent the pass there is the real cause.
                self.failure += "; the answer taken before at packet %d was refuted: %s" % (refuted.begin,
                                                                                         refuted.refuted)
            raise SyncError("shadow permanently invalid: " + self.failure) from exc
        except BaseException as exc:
            if self.record_origins:
                self.receipt_state = prior_receipts
                self.failure = "origin receipt advance aborted: " + str(exc)
            raise

    def require_search_root(self):
        if self.failure is not None:
            raise SyncError("shadow permanently invalid: " + self.failure)
        raise SyncError("blank shadow search is unvalidated: history coverage and complete particles required")

    def _enabling_places(self, message):
        """None. A blank shadow's hidden hand holds only blanks, which enable no prompt of their own: a local
        opponent prompt the real duel did not ask comes from an earlier answer (the Battle Phase left for the End
        Phase skips the Main Phase 2 menu), so the divergence goes back to the search."""
        return []

    def _replace(self, controller, places, attempt):
        raise SyncError("unexpected hidden enabling effect in a blank shadow")

    def _place(self, controller, location, sequence, code, keep=frozenset()):
        if self.record_origins:
            from .client_origin_receipt import boundary, place_request
            # The inherited own-deck placement can recurse. Its outer call
            # owns one unvalidated donor record; no mid-operation C queries.
            if boundary(self).phase == "mutation" and controller == self.viewer:
                return super()._place(controller, location, sequence, code)
            with place_request(self, (controller, location, sequence, code)):
                return self._place_impl(controller, location, sequence, code, keep)
        return self._place_impl(controller, location, sequence, code, keep)

    def _place_impl(self, controller, location, sequence, code, keep=frozenset()):
        """``keep``: the places the other cards of the same evidence go to. They are distinct cards, so no object is
        taken from there (two copies of one card revealed from a hand are two objects, not one moved twice)."""
        if controller == self.viewer and location == C.LOCATION_DECK and sequence is None:
            return self._arrange_own_deck(code)
        if controller != self.viewer and location == C.LOCATION_HAND and sequence is None:
            return self._arrange_opponent_hand(code)
        if self.single_pass and controller != self.viewer and location == C.LOCATION_HAND and code in BLANK_SETS:
            # A set proxy stands only for a placeholder: a known card at that slot is the card set.
            current = card_code(self.core, self.local.pduel, controller, location, sequence)
            if current not in BLANK_CODES or current == code:
                return False
            return self._hydrate(controller, location, sequence, code)
        if controller == self.viewer:
            if self.record_origins:
                from .client_origin_receipt import mutation
                return mutation(self, "own_place", lambda: super(BlankClientSync, self)._place(
                    controller, location, sequence, code), code=code,
                    request=(controller, location, sequence, code))
            return super()._place(controller, location, sequence, code)
        if location == C.LOCATION_DECK and sequence == -1:
            count = self.core.query_field_count(self.local.pduel, controller, location)
            return any([self._place(controller, location, count - 1 - depth, value, keep)
                        for depth, value in enumerate(code)])
        if location == C.LOCATION_DECK:
            source = self._deck_rebind_source(controller, sequence, int(code), keep)
            if source is not None:
                self._rebind(controller, location, sequence, source, int(code))
                if card_code(self.core, self.local.pduel, controller, location, sequence) == int(code):
                    return True
        if location == C.LOCATION_HAND:
            count = self.core.query_field_count(self.local.pduel, controller, location)
            if sequence >= count:
                # Only the newly drawn final hand slot is attributable without
                # a mapping; multiple intervening private moves are refused.
                if sequence != count:
                    raise SyncError("unresolved hidden draw/move attribution")
                deck_count = self.core.query_field_count(self.local.pduel, controller, C.LOCATION_DECK)
                return self._place(controller, C.LOCATION_DECK, deck_count - 1, code, keep)
            source = self._hand_rebind_source(controller, sequence, int(code), count, keep) \
                if self.hidden_hand_rebind else None
            if source is not None:
                self._rebind(controller, location, sequence, source, int(code))
                if card_code(self.core, self.local.pduel, controller, location, sequence) == int(code):
                    return True
            # With public identities only, no drawn object keeps an identity the stream did not pin to it.
            if self.hidden_hand_rebind and not self.public_identities_only \
                    and self._hand_deck_exchange(controller, sequence, int(code)) \
                    and card_code(self.core, self.local.pduel, controller, location, sequence) == int(code):
                return True
        try:
            card_code(self.core, self.local.pduel, controller, location, sequence)
        except HydrationError:
            # No card stands there yet: the reveal names the place a card reaches later in the batch, and a card
            # arriving from a hidden place is placed through the move that shows its source.
            return False
        return self._hydrate(controller, location, sequence, code)

    def _hand_rebind_source(self, controller, sequence, code, count, keep=frozenset()):
        """After a hidden shuffle of this hand, the slot whose object belongs in ``sequence``.

        The real hand was shuffled after its known cards last had public slots,
        so any arrangement of its objects is consistent with the stream. A known
        object with the revealed code is preferred to hydrating a blank (that
        would make a second copy of a card already in hand); a slot bound to a
        different known card takes a blank instead. Hand places another card of
        the same evidence goes to (``keep``) are never taken from. None keeps
        plain hydration.
        """
        codes = [card_code(self.core, self.local.pduel, controller, C.LOCATION_HAND, slot) for slot in range(count)]
        current = codes[sequence]
        if current == code:
            return None
        donors = [slot for slot in range(count) if slot != sequence and (controller, C.LOCATION_HAND, slot) not in keep]
        known = [slot for slot in donors if codes[slot] == code]
        if known:
            source, involved = known[0], {code} | ({current} - BLANK_CODES)
        elif current in BLANK_CODES:
            return None
        else:
            blanks = [slot for slot in donors if codes[slot] == BLANK_MAIN]
            if not blanks:
                return None
            source, involved = blanks[0], {current}
        return source if self._hand_hidden_since(controller, involved) else None

    def _deck_rebind_source(self, controller, sequence, code, keep=frozenset()):
        """After a shuffle of this deck, the slot whose object belongs in ``sequence``.

        The real deck was shuffled after its known cards last had public places,
        so any arrangement of its objects is consistent with the stream. As in
        the hand, a known object with the revealed code is preferred to
        hydrating a blank; a slot bound to a different known card takes a blank
        instead. Deck places the coming packets show (the other excavated cards)
        or another card of the same evidence goes to (``keep``) are never taken
        from.
        """
        count = self.core.query_field_count(self.local.pduel, controller, C.LOCATION_DECK)
        if not 0 <= sequence < count:
            return None
        codes = [card_code(self.core, self.local.pduel, controller, C.LOCATION_DECK, slot) for slot in range(count)]
        current = codes[sequence]
        if current == code:
            return None
        shown = {reveal[2] for reveal in self._reveals_ahead(self.cursor) if reveal[:2] == (controller, C.LOCATION_DECK)}
        donors = [slot for slot in range(count) if slot != sequence and slot not in shown
                  and (controller, C.LOCATION_DECK, slot) not in keep]
        known = [slot for slot in donors if codes[slot] == code]
        if known:
            source, involved = known[0], {code} | ({current} - BLANK_CODES)
        elif current in BLANK_CODES:
            return None
        else:
            blanks = [slot for slot in donors if codes[slot] == BLANK_MAIN]
            if not blanks:
                return None
            source, involved = blanks[0], {current}
        return source if self._deck_hidden_since(controller, involved) else None

    def _hand_deck_exchange(self, controller, sequence, code):
        """Exchange a drawn hand object with a deck object; True when done.

        A card drawn from a shuffled deck is hidden: which deck object the draw
        took is not public, and the local draw took its own. When the revealed
        slot holds a drawn object of another known card, or a known object with
        the revealed code waits in the deck (hydrating the blank would make a
        second copy), a deck object not shown by the coming packets takes the
        slot: that known object, else a blank. Recorded as unvalidated history.
        """
        hand_code = card_code(self.core, self.local.pduel, controller, C.LOCATION_HAND, sequence)
        if hand_code == code or not self._recorded_query("reason", controller, C.LOCATION_HAND,
                                                       sequence) & _REASON_DRAW:
            return False
        count = self.core.query_field_count(self.local.pduel, controller, C.LOCATION_DECK)
        deck = [card_code(self.core, self.local.pduel, controller, C.LOCATION_DECK, slot) for slot in range(count)]
        shown = {reveal[2] for reveal in self._reveals_ahead(self.cursor) if reveal[:2] == (controller, C.LOCATION_DECK)}
        donors = [slot for slot in range(count) if slot not in shown]
        known = [slot for slot in donors if deck[slot] == code]
        if hand_code in BLANK_CODES and not known:
            return False  # hydrating the blank is the plain case
        blanks = [slot for slot in donors if deck[slot] == BLANK_MAIN]
        source = (known or blanks or [None])[0]
        if source is None or not self._deck_hidden_since(controller, {code} | ({hand_code} - BLANK_CODES)):
            return False
        from ..search import particles as P
        layout = P.read_hidden_layout(self.local, controller)
        hand, pile = list(layout.hand), list(layout.deck)
        hand[sequence], pile[source] = pile[source], hand[sequence]
        operation = lambda: P.apply_particle(self.local, controller, P.HiddenLayout(hand=tuple(hand), deck=tuple(pile),
                                                                                 facedown=()))
        if self.record_origins:
            from .client_origin_receipt import mutation
            mutation(self, "hidden_draw_exchange", operation, target=(controller, C.LOCATION_HAND, sequence),
                     code=code, request=(controller, C.LOCATION_HAND, sequence, C.LOCATION_DECK, source))
        else:
            operation()
        return True

    def _deck_hidden_since(self, controller, codes):
        """A shuffle of this deck after the last public deck place of any of these cards."""
        placed = shuffled = -1
        for index, packet in enumerate(self.packets[:self.cursor]):
            if packet[0] == C.MSG_SHUFFLE_DECK and len(packet) >= 2 and packet[1] == controller:
                shuffled = index
            elif packet[0] == C.MSG_MOVE and len(packet) >= 13 and tuple(packet[9:11]) == (controller, C.LOCATION_DECK) \
                    and struct.unpack_from("<I", packet, 1)[0] & 0x7FFFFFFF in codes:
                placed = index
            elif any(reveal[:2] == (controller, C.LOCATION_DECK) and reveal[3] in codes
                     for reveal in _reveals(packet, self.viewer)):
                placed = index
        return shuffled > placed

    def _category_fact(self, controller, location, candidates, chaining, zones, since):
        """An activation at packet ``chaining`` was legal only with one of ``candidates`` in this hidden zone of the
        opponent, or one of each other zone's candidates in ``zones`` ((location, candidates), ..., this zone first),
        read from the batch start at packet ``since``: the public fact joins the category constraints, which the
        particles at the seat's roots must honor."""
        zones = tuple((int(zone), tuple(int(code) for code in codes)) for zone, codes in zones)
        if zones[0] != (location, tuple(int(code) for code in candidates)):
            raise SyncError("a category fact names its candidates outside its first zone")
        if len(zones) > 1:
            self.stats.categories_open += 1
        self.category_constraints.append((controller, location, zones[0][1], 1, chaining, zones, since))

    def _hypothesize(self, controller, location, sequence, code):
        """Assume one pool card in a blank hidden slot (deferral fallback), recorded as unvalidated."""
        self._require_recipe(controller, location, sequence, int(code))
        operation = lambda: hydrate(self.core, self.local.pduel, controller, location, sequence, int(code))
        if self.record_origins:
            from .client_origin_receipt import mutation
            return mutation(self, "hidden_target_hypothesis", operation, target=(controller, location, sequence),
                            code=int(code), request=(controller, location, sequence, int(code)))
        return operation()

    def _hand_hidden_since(self, controller, codes):
        """A hidden shuffle of this hand after the last public slot of any of these cards."""
        placed = shuffled = -1
        for index, packet in enumerate(self.packets[:self.cursor]):
            msg = packet[0]
            if msg == C.MSG_SHUFFLE_HAND and len(packet) >= 2 and packet[1] == controller:
                shuffled = index
            elif msg == C.MSG_MOVE and len(packet) >= 13 and tuple(packet[9:11]) == (controller, C.LOCATION_HAND) \
                    and struct.unpack_from("<I", packet, 1)[0] & 0x7FFFFFFF in codes:
                placed = index
            elif msg == C.MSG_CONFIRM_CARDS and len(packet) >= 4:
                for offset in range(4, len(packet) - 6, 7):
                    if (struct.unpack_from("<I", packet, offset)[0] & 0x7FFFFFFF in codes
                            and tuple(packet[offset + 4:offset + 6]) == (controller, C.LOCATION_HAND)):
                        placed = index
        return shuffled > placed

    def _rebind(self, controller, location, sequence, source, code):
        """Exchange two hand or deck objects by UID; identities and histories stay with them."""
        if self.record_origins:
            from .client_origin_receipt import entities, mutation
            rows = entities(self)
        else:
            from .client_entity_map import _read
            rows = _read(self.core._lib, self.local.pduel)
        order = [row.uid for row in sorted((row for row in rows if row.controller == controller
                                            and row.location == location), key=lambda row: row.sequence)]
        order[sequence], order[source] = order[source], order[sequence]
        reorder = reorder_deck if location == C.LOCATION_DECK else reorder_hand
        operation = lambda: reorder(self.core, self.local.pduel, controller, order)
        if self.record_origins:
            kind = "deck_rebind" if location == C.LOCATION_DECK else "hand_rebind"
            return mutation(self, kind, operation, target=(controller, location, sequence),
                            code=code, request=(controller, location, tuple(order)))
        return operation()

    def _reveal_entries(self, index, limit=64, bounded=True):
        """``(packet index, reveal)`` of the cards the real packets show from ``index``: the inherited reveals, a
        card leaving a hidden zone or the field with its code, a face-down field card turned face up. ``bounded``
        stops at the second decision boundary, as the inherited window does."""
        out, boundaries = [], 0
        for position in range(index, min(len(self.packets), index + limit)):
            packet = self.packets[position]
            last = False
            if bounded and _boundary(packet, self.viewer):
                boundaries += 1
                last = boundaries > 1 and packet[0] != C.MSG_WAITING or boundaries > 2
            out.extend((position, reveal) for reveal in _reveals(packet, self.viewer))
            if packet[0] == C.MSG_MOVE and len(packet) >= 13:
                code = struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff
                player, location, sequence = packet[5:8]
                # A set card can become public after a target relation already exists: its identity goes to the
                # object that holds it. A card leaving the opponent's blank deck (a milled top, a searched card):
                # its real deck place is as good as any other, the shadow's deck being interchangeable blanks.
                if code and player != self.viewer and location in (C.LOCATION_MZONE, C.LOCATION_SZONE,
                                                                    C.LOCATION_DECK):
                    out.append((position, (player, location, sequence, code)))
            # A face-down card turned face up on the field names itself where it stands.
            if packet[0] in (C.MSG_POS_CHANGE, C.MSG_CHAINING, C.MSG_FLIPSUMMONING) and len(packet) >= 8:
                code = struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff
                player, location, sequence = packet[5:8]
                if code and player != self.viewer and location in (C.LOCATION_MZONE, C.LOCATION_SZONE):
                    out.append((position, (player, location, sequence, code)))
            if last:
                break
        return out

    def _traced(self, entries, origin):
        """Each opponent hand or field reveal moved to the place its card held at the batch start ``origin``
        (``(cursor, sizes)``); one whose batch-start place the packets do not tell is left out: placing it where
        another object stood would give that object an identity nothing showed."""
        if origin is None:
            return [reveal for _position, reveal in entries]
        cursor, sizes = origin
        opponent = 1 - self.viewer
        followed = [number for number, (_position, reveal) in enumerate(entries)
                    if reveal[0] == opponent and reveal[1] in (C.LOCATION_HAND, *_TRACED_FIELD) and reveal[2] >= 0
                    and entries[number][0] >= cursor]
        places = batch_start_places(self.packets, cursor, [(entries[number][0], entries[number][1][:3])
                                                           for number in followed], opponent, sizes)
        moved = dict(zip(followed, places))
        out = []
        for number, (_position, reveal) in enumerate(entries):
            if number in moved:
                if moved[number] is None:
                    self.stats.reveals_untraced += 1
                    continue
                reveal = (*moved[number], reveal[3])
            if reveal not in out:
                out.append(reveal)
        return out

    def _batch_evidence(self, origin):
        """Single pass: every opponent hidden card the received packets show from the batch start on, at the
        place it held there, and the set proxy of each hidden hand card set face down whose identity they do not
        show, the opponent's hand as one arrangement (``_opponent_hand_plan``); the seat's own deck cards and
        shuffle orders they show (``_own_evidence``). A card whose batch start place the packets do not tell (a
        shuffle came between) is left out."""
        if not self.single_pass:
            return [], []
        cursor, sizes = origin
        opponent = 1 - self.viewer
        entries = [(position, reveal) for position, reveal in self._reveal_entries(cursor, len(self.packets) - cursor,
                                                                                  bounded=False)
                   if reveal[0] == opponent and reveal[2] >= 0]
        proxies = []
        for position in range(cursor, len(self.packets)):
            packet = self.packets[position]
            if (packet[0] == C.MSG_MOVE and len(packet) >= 13 and not struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff
                    and tuple(packet[5:7]) == (opponent, C.LOCATION_HAND)
                    and tuple(packet[9:11]) == (opponent, C.LOCATION_SZONE) and packet[11] <= 5):
                # Only a field spell sets into the field zone (sequence 5).
                proxies.append((position, (opponent, C.LOCATION_HAND, packet[7],
                                           BLANK_FIELD if packet[11] == 5 else BLANK_SSET)))
        shown, _orders = self._traced_places(entries, cursor, opponent, sizes)
        places = {reveal[:3] for reveal in shown}
        proxies, _orders = self._traced_places(proxies, cursor, opponent, sizes)
        proxies = [proxy for proxy in proxies if proxy[:3] not in places]
        own, orders = self._own_evidence(cursor)
        hand = (opponent, C.LOCATION_HAND)
        return (self._opponent_hand_plan(cursor, sizes, [reveal for reveal in shown if reveal[:2] == hand],
                                         [proxy for proxy in proxies if proxy[:2] == hand])
                + [reveal for reveal in shown + proxies if reveal[:2] != hand] + own), orders

    def _opponent_hand_plan(self, cursor, sizes, shown, proxies):
        """The opponent-hand part of a batch's evidence, as one placement ``(opponent, HAND, None, plan)`` (plan
        5.12): the cards ``shown`` at the hand slots of the batch start, the kind of each card set face down from the
        hand, the set ``proxies`` of the placeholders set, and the slots whose objects the opponent's last hidden
        hand shuffle left in an order the seat never saw. How a card entered the hand (drawn, added by an effect) is
        a trace of the core's objects, not a rule fact, and is not followed."""
        opponent = 1 - self.viewer
        count = sizes.get(C.LOCATION_HAND, 0)
        shuffles = [index for index in range(cursor) if self.packets[index][0] == C.MSG_SHUFFLE_HAND
                    and len(self.packets[index]) >= 3 and self.packets[index][1] == opponent]
        free = frozenset()
        if shuffles:
            after = self.packets[shuffles[-1]][2]
            # The batch start's hand slots whose objects were in the hand at that shuffle and have not moved into a
            # public slot since: their order is not public.
            tokens = batch_start_places(self.packets, shuffles[-1] + 1,
                                        [(cursor, (opponent, C.LOCATION_HAND, slot)) for slot in range(count)],
                                        opponent, {C.LOCATION_HAND: after, C.LOCATION_DECK: 0, C.LOCATION_EXTRA: 0})
            free = frozenset(slot for slot, token in enumerate(tokens)
                             if token is not None and token[1] == C.LOCATION_HAND)
        sets = [(position, (opponent, C.LOCATION_HAND, packet[7]), tuple(packet[10:12]))
                for position, packet in enumerate(self.packets[cursor:], cursor)
                if packet[0] == C.MSG_MOVE and len(packet) >= 17 and tuple(packet[5:7]) == (opponent, C.LOCATION_HAND)
                and packet[12] & C.POS_FACEDOWN and not struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff
                and packet[10] in (C.LOCATION_MZONE, C.LOCATION_SZONE)]
        starts = batch_start_places(self.packets, cursor, [(position, place) for position, place, _to in sets],
                                    opponent, sizes)
        kinds = tuple((start[2], zone) for (_p, _place, zone), start in zip(sets, starts)
                      if start is not None and start[1] == C.LOCATION_HAND)
        identities = tuple((slot, code) for _player, _location, slot, code in shown)
        proxies = tuple((slot, proxy) for _player, _location, slot, proxy in proxies)
        if not (identities or kinds or proxies):
            return []
        return [(opponent, C.LOCATION_HAND, None, OpponentHandPlan(identities, kinds, proxies, free))]

    def _arrange_opponent_hand(self, plan) -> bool:
        """Arrange the opponent's hand as ``plan`` asks (:func:`arrange_hand`) in one reorder of whole objects
        (identities and histories stay with them), then write each card the packets show on the placeholder that
        stands for it, and the set proxy on each placeholder set; True when anything changed."""
        opponent = 1 - self.viewer
        count = self.core.query_field_count(self.local.pduel, opponent, C.LOCATION_HAND)
        codes = [card_code(self.core, self.local.pduel, opponent, C.LOCATION_HAND, slot) for slot in range(count)]
        source = arrange_hand(codes, plan, lambda code, zone: self._can_be_set(code, *zone))
        changed = source != list(range(count))
        if changed:
            if self.record_origins:
                from .client_origin_receipt import entities, mutation
                rows = entities(self)
            else:
                from .client_entity_map import _read
                rows = _read(self.core._lib, self.local.pduel)
            uids = [row.uid for row in sorted((row for row in rows if row.controller == opponent
                                               and row.location == C.LOCATION_HAND), key=lambda row: row.sequence)]
            uids = [uids[slot] for slot in source]
            operation = lambda: reorder_hand(self.core, self.local.pduel, opponent, uids)
            if self.record_origins:
                mutation(self, "hand_rebind", operation, request=(opponent, C.LOCATION_HAND, tuple(uids)))
            else:
                operation()
        for slot, code in plan.identities:
            if card_code(self.core, self.local.pduel, opponent, C.LOCATION_HAND, slot) != code:
                changed = self._hydrate(opponent, C.LOCATION_HAND, slot, code) or changed
        for slot, proxy in plan.proxies:
            # A set proxy stands only for a placeholder: a known card at that slot is the card set.
            current = card_code(self.core, self.local.pduel, opponent, C.LOCATION_HAND, slot)
            if current in BLANK_CODES and current != proxy:
                changed = self._hydrate(opponent, C.LOCATION_HAND, slot, proxy) or changed
        return changed

    def _hydrate(self, controller, location, sequence, code):
        """Write ``code`` on the placeholder at this hidden place (:func:`hydrate`), with its receipt."""
        self._require_recipe(controller, location, sequence, int(code))
        operation = lambda: hydrate(self.core, self.local.pduel, controller, location, sequence, int(code))
        if self.record_origins:
            from .client_origin_receipt import mutation
            return mutation(self, "hydrate", operation, target=(controller, location, sequence), code=int(code))
        return operation()

    def _require_recipe(self, controller, location, sequence, code):
        if self.public_opponent_recipe is not None:
            from .client_public_recipe import require_native
            require_native(self, controller, location, sequence, code)

    def _own_evidence(self, cursor):
        """The seat's own hidden order the packets show from ``cursor`` on: the deck places at the batch start of the
        cards drawn, milled, excavated or taken from its deck, as one arrangement of its deck objects, and the
        order the first shuffle of the seat's hand and of its deck leave (the hand's shuffle names every card, the
        deck's top cards are the ones drawn after it)."""
        own = self.viewer
        entries = []
        for position, reveal in self._reveal_entries(cursor, len(self.packets) - cursor, bounded=False):
            if reveal[0] != own or reveal[1] != C.LOCATION_DECK:
                continue
            if reveal[2] == -1:
                entries.extend((position, (own, C.LOCATION_DECK, -1 - depth, code))
                               for depth, code in enumerate(reveal[3]))
            else:
                entries.append((position, reveal))
        sizes = {location: self.core.query_field_count(self.local.pduel, own, location) for location in _TRACED_LISTS}
        placed, orders = self._traced_places(entries, cursor, own, sizes)
        hand = next((_shuffle_orders(packet, own) for packet in self.packets[cursor:]
                     if packet[0] == C.MSG_SHUFFLE_HAND and len(packet) >= 2 and packet[1] == own), None)
        if hand is not None:
            orders.append((own, *hand))
        # The seat's deck cards are known objects with their own histories: they are arranged, never re-identified.
        arrangement = tuple((sequence, code) for _player, _location, sequence, code in placed)
        return ([(own, C.LOCATION_DECK, None, arrangement)] if arrangement else []), orders

    def _arrange_own_deck(self, arrangement) -> bool:
        """Put the seat's deck objects with these codes at these deck places (``((sequence, code), ...)``), moving
        whole objects: identities and histories stay with them. False when every card already stands there."""
        own = self.viewer
        count = self.core.query_field_count(self.local.pduel, own, C.LOCATION_DECK)
        codes = [card_code(self.core, self.local.pduel, own, C.LOCATION_DECK, slot) for slot in range(count)]
        if all(0 <= sequence < count and codes[sequence] == code for sequence, code in arrangement):
            return False
        if self.record_origins:
            from .client_origin_receipt import entities, mutation
            rows = entities(self)
        else:
            from .client_entity_map import _read
            rows = _read(self.core._lib, self.local.pduel)
        uids = [row.uid for row in sorted((row for row in rows if row.controller == own
                                           and row.location == C.LOCATION_DECK), key=lambda row: row.sequence)]
        pinned = set()
        for sequence, code in arrangement:
            if not 0 <= sequence < count:
                raise SyncError("the seat's deck has no place %d" % sequence)
            if codes[sequence] != code:
                source = next((slot for slot in range(count - 1, -1, -1)
                               if slot not in pinned and slot != sequence and codes[slot] == code), None)
                if source is None:
                    raise RevealConflict("the seat's deck holds no other %d for place %d" % (code, sequence))
                codes[sequence], codes[source] = codes[source], codes[sequence]
                uids[sequence], uids[source] = uids[source], uids[sequence]
            pinned.add(sequence)
        operation = lambda: reorder_deck(self.core, self.local.pduel, own, uids)
        if self.record_origins:
            mutation(self, "deck_rebind", operation, request=(own, C.LOCATION_DECK, tuple(uids)))
        else:
            operation()
        return True

    def _traced_places(self, entries, cursor, player, sizes):
        """Each reveal of ``entries`` (``player``'s cards) at the place its card held at the batch start, the
        hand, deck and extra deck followed as lists and the field by slot, and the deck order the first deck
        shuffle among them must leave for the cards it placed; untraced reveals are left out."""
        places = batch_start_places(self.packets, cursor, [(position, reveal[:3]) for position, reveal in entries],
                                    player, sizes, shuffles=True)
        out, shuffled = [], {}
        for (_position, reveal), place in zip(entries, places):
            if isinstance(place, Shuffled) and reveal[1] == C.LOCATION_DECK:
                shuffled.setdefault(place.packet, {})[place.size - 1 - place.slot] = reveal[3]
                continue
            if place is None or isinstance(place, Shuffled):
                self.stats.reveals_untraced += 1
                continue
            placed = (*place, reveal[3])
            if placed not in out:
                out.append(placed)
        orders = []
        if shuffled:
            # The deck's forced order names its top cards: those the packets show from the top down, while they
            # follow on from it. The core keeps one order per deck, used by its next shuffle.
            tops = shuffled[min(shuffled)]
            codes = []
            while len(codes) in tops:
                codes.append(tops[len(codes)])
            if codes:
                orders.append((player, C.LOCATION_DECK, codes))
        return out, orders

    def _group_order_shown(self, divergence, begin):
        """Single pass: two cards of one group operation the server and the local core took in another order, the
        server's first card first: ``("places", ((packet, place), (packet, place)))``, each at the place it held at
        that packet, or ``("objects", (uid, uid))``; None for any other divergence.

        The core takes a group's cards one by one in the order of its card objects. That order is the core's own
        and no rule of the game, so the server's may be any; what must agree is which card each step acted on,
        since that decides the board (which card went to which zone, the graveyard's order). Two moves out of
        different places of one zone show the step, or two zone-prompt hints naming different cards of a group
        the local core is operating on (the hint names a card only by its code; the moves come after the seat
        answers)."""
        if not self.single_pass:
            return None
        real, local, index = divergence.real, divergence.local, divergence.index
        if len(real) >= 13 and len(local) >= 13 and real[0] == local[0] == C.MSG_MOVE:
            first, second = tuple(real[5:8]), tuple(local[5:8])
            if first[:2] != second[:2] or first == second or first[1] & C.LOCATION_OVERLAY:
                return None
            return "places", ((index, first), (index, second))
        if len(real) == len(local) == 7 and real[0] == local[0] == C.MSG_HINT and real[1:3] == local[1:3] \
                and real[1] == C.HINT_SELECTMSG:
            codes = struct.unpack_from("<I", real, 3)[0], struct.unpack_from("<I", local, 3)[0]
            if codes[0] == codes[1]:
                return None
            for group in operation_groups(self.core, self.local.pduel):
                named = [[uid for uid, code in group if code == wanted] for wanted in codes]
                if all(len(uids) == 1 for uids in named):
                    return "objects", (named[0][0], named[1][0])
        return None

    def _start_uids(self, begin, shown):
        """The local objects ``shown`` names; ``(packet, place)`` pairs are traced back to where the batch starting at
        ``begin`` began (the local duel stands there now). None when the packets do not trace one back."""
        from .client_origin_receipt import entities
        kind, shown = shown
        if kind == "objects":
            return tuple(shown)
        places = {}
        for player in (0, 1):
            wanted = [(number, item) for number, item in enumerate(shown) if item[1][0] == player]
            if not wanted:
                continue
            sizes = {location: self.core.query_field_count(self.local.pduel, player, location)
                     for location in _ORDER_LISTS}
            traced = batch_start_places(self.packets, begin, [item for _number, item in wanted], player, sizes)
            places.update((number, place) for (number, _item), place in zip(wanted, traced))
        objects = {(row.controller, row.location, row.sequence): row.uid for row in entities(self)
                   if not row.location & C.LOCATION_OVERLAY}
        uids = tuple(objects.get(places.get(number) if isinstance(places.get(number), tuple) else None)
                     for number in range(len(shown)))
        return None if None in uids or len(set(uids)) != len(uids) else uids

    def _order_cards(self, uids):
        """The server took ``uids`` in this order: a fact of the whole duel (its core orders a group's cards by
        object, and its objects do not change), kept with every such fact before it. The local objects take one order
        that honors them all, so an order learned later never undoes an earlier one."""
        for first, second in zip(uids, uids[1:]):
            if (first, second) not in self.order_facts:
                self.order_facts.append((first, second))
        chain = _chain(self.order_facts)
        operation = lambda: order_cards(self.core, self.local.pduel, chain)
        if self.record_origins:
            from .client_origin_receipt import mutation
            mutation(self, "group_order", operation, request=chain)
        else:
            operation()
        self.order_live = len(self.order_facts)

    def _before_batch(self):
        """A batch run from a state saved before the latest group orders were learned puts them all back first. Where
        an operation already holds the cards (it asked about its first one, or keeps a set of its own), the core
        refuses the whole order and writes nothing: the batch's divergence at that group's first move takes the order
        back before the operation (``_order_before``), as when it was first learned."""
        if self.order_live < len(self.order_facts):
            try:
                self._order_cards(())
            except (GroupAsked, GroupHeld):
                return

    def _traced_sizes(self):
        opponent = 1 - self.viewer
        return {location: self.core.query_field_count(self.local.pduel, opponent, location)
                for location in _TRACED_LISTS}

    def _reveals_ahead(self, index, limit=64, origin=None):
        """The cards the real packets show from ``index`` through the next decision's consequences; with the batch
        start ``origin``, each at the place its card held there."""
        return self._traced(self._reveal_entries(index, limit), origin)

    def _fixes(self, begin):
        # Exact source coordinates in MOVE/CHAINING/CONFIRM precede guessed
        # source slots for summons, whose public packet only names destination.
        # The follower stands at the prompt's batch start here.
        origin = (self.cursor, self._traced_sizes())
        reveals = self._traced(self._reveal_entries(begin, 64, bounded=False), origin)
        summons = []
        hidden_sets = []
        for position in range(begin, min(len(self.packets), begin + 64)):
            packet = self.packets[position]
            if packet[0] == C.MSG_MOVE and len(packet) >= 13:
                code = struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff
                if not code and packet[5] != self.viewer and packet[6] == C.LOCATION_HAND:
                    if packet[10] == C.LOCATION_SZONE and packet[11] <= 5:
                        # Only a field spell sets into the field zone (sequence 5).
                        proxy = BLANK_FIELD if packet[11] == 5 else BLANK_SSET
                        hidden_sets.extend(self._traced([(position, (packet[5], C.LOCATION_HAND, packet[7], proxy))],
                                                        origin))
            if packet[0] in (C.MSG_SUMMONING, C.MSG_SPSUMMONING) and len(packet) >= 8:
                code = struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff
                if packet[5] != self.viewer and code:
                    summons.append(code)
                    break
        if reveals:
            yield tuple(reveals)
        for hidden in hidden_sets:
            yield tuple(reveals) + (hidden,)
        for code in summons:
            data = self.core.card_pool().cards.get(code)
            extra = data is not None and data.type & (0x40 | 0x2000 | 0x800000 | 0x4000000)
            location = C.LOCATION_EXTRA if extra else C.LOCATION_HAND
            player = 1 - self.viewer
            count = self.core.query_field_count(self.local.pduel, player, location)
            for sequence in range(count):
                if card_code(self.core, self.local.pduel, player, location, sequence) in BLANK_CODES:
                    yield tuple(reveals) + ((player, location, sequence, code),)
