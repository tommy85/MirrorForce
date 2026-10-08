"""Public mutation lineage for one accepted continuous-client native batch.

This records facts and unresolved attribution, not hidden particle guesses or
search admission. Immutable ledgers may be shared by saved snapshots; they
NEVER contain the snapshot side table, a native handle, or a follower/owner.
All native observations are at complete C boundaries, never in _observe.
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import struct

from ..immutable import ImmutableRecord
from ..netduel import constants as C
from .client_entity_map import EntityMapError, LocalEntity, _capture_advance_entities
from .client_root import SnapshotOwnershipError, _advance_boundary, _bind_started_follower
from .client_sync import SyncError, _reveals

#: A token's card type: a public object created on the field.
_TYPE_TOKEN = 0x4000

INFORMATION_SET_SEARCH = True
SCHEMA = "client-origin-public-receipt/v1"


class ReceiptError(SyncError):
    pass


@dataclass(frozen=True)
class PublicSource(ImmutableRecord):
    matched_index: int
    raw_index: int
    packet: bytes
    attribution: str


@dataclass(frozen=True)
class PublicMutation(ImmutableRecord):
    kind: str
    request: tuple
    target: tuple[int, int, int] | None
    uid: int
    code: int | tuple[int, ...] | None
    changed: bool
    sources: tuple[PublicSource, ...]
    markers: tuple[str, ...]


@dataclass(frozen=True)
class RootFactPatch(ImmutableRecord):
    uid: int
    origin: LocalEntity
    root: LocalEntity
    code: int
    sources: tuple[PublicSource, ...]
    available_after: int
    # A known packet and a local UID join are NOT general rule equivalence.
    attribution: str = "public_request_local_uid"


@dataclass(frozen=True)
class PublicShuffleScope(ImmutableRecord):
    """Wire-visible participants only; never an identity/UID permutation."""

    source: PublicSource
    zones: tuple[tuple[int, int], ...]
    positions_before: tuple[tuple[int, int, int], ...]
    whole_zone: bool
    reported_count: int | None


@dataclass(frozen=True)
class OriginReceipt(ImmutableRecord):
    origin_native_sha256: str
    matched_range: tuple[int, int]
    raw_range: tuple[int, int]
    own_range: tuple[int, int]
    own_responses: tuple[bytes, ...]
    origin_entities: tuple[LocalEntity, ...]
    attempt_entities: tuple[LocalEntity, ...]
    root_entities: tuple[LocalEntity, ...]
    mutations: tuple[PublicMutation, ...]
    facts: tuple[RootFactPatch, ...]
    shuffles: tuple[PublicShuffleScope, ...]
    markers: tuple[str, ...]
    attempts: int
    schema: str = SCHEMA
    search_ready: bool = False


@dataclass(frozen=True)
class ReceiptLedger(ImmutableRecord):
    accepted: tuple[OriginReceipt, ...] = ()
    pending: tuple[PublicMutation, ...] = ()
    own_cursor: int = 0
    replay_events: tuple = ()


@dataclass(frozen=True)
class SavedReceipt(ImmutableRecord):
    ledger: ReceiptLedger
    entities: tuple[LocalEntity, ...]
    cursor: int
    answered: int
    native_sha256: str


def boundary(follower):
    try:
        return _advance_boundary(follower)
    except SnapshotOwnershipError as exc:
        raise ReceiptError("origin receipts require the current controller-owned advance") from exc


@contextmanager
def history_attribution(follower, marker, *, proofs=()):
    """Scoped proof obligation, not an identity mutation or a native lease."""
    scope = boundary(follower)
    previous = scope.history_markers, scope.history_proofs
    scope.history_markers = previous[0] + (marker,)
    scope.history_proofs = previous[1] + tuple(proofs)
    try:
        yield
    finally:
        scope.history_markers, scope.history_proofs = previous


def started(follower):
    boundary(follower)
    _bind_started_follower(follower)
    entities(follower)  # bind/capability check before any receipt is recorded
    from .client_replay_journal import opening
    opening(follower)


def entities(follower):
    token = boundary(follower)
    try:
        return _capture_advance_entities(follower, token)
    except (EntityMapError, SnapshotOwnershipError) as exc:
        raise ReceiptError("cannot capture origin entities at this safe boundary: " + str(exc)) from exc


@contextmanager
def native_operation(follower, phase):
    scope = boundary(follower)
    if scope.phase != "idle":
        raise ReceiptError("nested receipt native operation")
    scope.phase = phase
    try:
        yield
    finally:
        scope.phase = "idle"


def remember_snapshot(follower, saved):
    handle, _state, cursor, answered = saved
    if handle in follower.receipt_saved:
        raise ReceiptError("receipt snapshot handle reused without release")
    rows = entities(follower)
    length = ctypes.c_size_t.from_address(handle).value
    if not 0 < length == int(follower.api.duel_arena_extent(follower.local.pduel)) <= 256 * 1024 * 1024:
        raise ReceiptError("unknown origin snapshot layout")
    digest = hashlib.sha256(ctypes.string_at(handle + ctypes.sizeof(ctypes.c_size_t), length)).hexdigest()
    follower.receipt_saved[handle] = SavedReceipt(follower.receipt_state, rows, cursor, answered, digest)


def load_snapshot(follower, saved):
    try:
        row = follower.receipt_saved[saved[0]]
    except KeyError as exc:
        raise ReceiptError("snapshot has no owned receipt state") from exc
    follower.receipt_state = row.ledger


def _packet_reveals(packet, viewer):
    out = list(_reveals(packet, viewer))
    if len(packet) >= 8 and packet[0] in (C.MSG_MOVE, C.MSG_CHAINING, C.MSG_POS_CHANGE):
        code = struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff
        player, location, sequence = packet[5:8]
        if code and location in (C.LOCATION_MZONE, C.LOCATION_SZONE):
            out.append((player, location, sequence, code))
    return out


def _sources(follower, request, code):
    exact, identity_only = [], []
    for index in range(follower.cursor, len(follower.packets)):
        packet = follower.packets[index]
        raw_index = follower.receipt_packet_raw_indices[index]
        revealed = _packet_reveals(packet, follower.viewer)
        if request in revealed:
            exact.append(PublicSource(index, raw_index, packet, "request"))
        elif code not in (None, 999000001, 999000002, 999000003, 999000004) and any(
                (code in item[3] if isinstance(item[3], tuple) else code == item[3]) for item in revealed):
            identity_only.append(PublicSource(index, raw_index, packet, "identity_only"))
    return tuple(exact or identity_only)


@contextmanager
def place_request(follower, request):
    scope = boundary(follower)
    scope.requests.append(request)
    try:
        yield
    finally:
        scope.requests.pop()


def mutation(follower, kind, callback, *, target=None, code=None, request=()):
    from . import client_replay_journal as journal
    scope = boundary(follower)
    replay_before = journal.capture(follower) if journal.enabled(follower) else None
    # Public attribution can name the enclosing placement request; execution
    # must retain the actual operation's arguments (e.g. the reordered UIDs).
    native_request = tuple(request)
    request = scope.requests[0] if scope.requests else request
    before = entities(follower)
    row = None
    if target is not None:
        matches = [item for item in before if (item.controller, item.location, item.sequence) == target]
        if len(matches) != 1:
            raise ReceiptError("public mutation target has no unique local entity")
        row = matches[0]
    try:
        with native_operation(follower, "mutation"):
            result = callback()
    except BaseException as exc:
        scope.owner.audit.append({"kind": "origin-mutation-failure", "operation": kind,
                                  "request": request, "error": str(exc)})
        raise
    after = entities(follower)
    changed = bool(result) if kind in ("hydrate", "own_place") else True
    sources = _sources(follower, request, code) if kind in ("hydrate", "own_place") else ()
    markers = []
    if kind == "own_place" and changed:
        markers.append("own_donor_unvalidated")
    if kind == "force_shuffle":
        markers.append(kind + "_unvalidated")
    if kind == "category_proxy":
        # A card standing for a public category claim (the deck held one of several cards); particles must honor
        # the claim before a root after it is admitted.
        markers.append("category_proxy_unvalidated")
    if kind in ("hidden_target_hypothesis", "hidden_draw_exchange"):
        # A pool card assumed in the blank deck so a public activation is legal, or a drawn hand object
        # exchanged with a deck object after a shuffle; neither is a public fact.
        markers.append(kind + "_unvalidated")
    if kind == "hydrate":
        if not any(item.uid == row.uid for item in after):
            raise ReceiptError("hydration lost the original local entity")
        if code in (999000003, 999000004):  # the set proxies of client_shadow
            markers.append("sset_proxy_not_printed_identity")
        elif not sources or any(item.attribution != "request" for item in sources):
            markers.append("source_entity_unproven")
    operation = PublicMutation(kind, tuple(request), target, row.uid if row else 0, code, changed,
                               sources, tuple(markers))
    scope.owner.audit.append({"kind": "origin-mutation-attempt", "operation": operation})
    follower.receipt_state = replace(follower.receipt_state,
                                     pending=follower.receipt_state.pending + (operation,))
    if replay_before is not None:
        journal.append(follower, "mutation", replay_before, (operation, native_request))
    return result


def run_batch(follower, callback):
    from . import client_replay_journal as journal
    scope = boundary(follower)
    if not scope.frames:
        raise ReceiptError("native batch has no owned origin receipt frame")
    frame = scope.frames[-1]
    before = entities(follower)
    replay_before = journal.capture(follower) if journal.enabled(follower) else None
    frame["replay_messages"] = ()
    frame["attempts"] += 1
    try:
        with native_operation(follower, "batch-and-observe"):
            result = callback()
    except BaseException as exc:
        # The entire C batch is already over even if _match stopped early.
        # This is a failed audit record, NEVER an accepted history element.
        if replay_before is not None:
            # Keep transient continuity until the owner rolls this attempt back.
            # The replay exporter rejects it if it somehow survives to a root.
            journal.append(follower, "process", replay_before, frame["replay_messages"], validated=False)
        scope.owner.audit.append({"kind": "origin-attempt-failure", "attempt": frame["attempts"],
                                  "cursor": follower.cursor, "error": str(exc),
                                  "mutations": follower.receipt_state.pending,
                                  "before_entities": before})
        raise
    frame["before"] = before
    frame["after"] = entities(follower)
    if replay_before is not None:
        journal.append(follower, "process", replay_before, frame["replay_messages"])
    return result


def _raw_cut(follower, cursor):
    return follower.receipt_packet_raw_indices[cursor - 1] + 1 if cursor else 0


def _shuffle_scope(source):
    raw = source.packet
    msg = raw[0]
    if msg == C.MSG_SHUFFLE_SET_CARD:
        if len(raw) < 3 or raw[1] not in (C.LOCATION_MZONE, C.LOCATION_SZONE) or len(raw) != 3 + 8 * raw[2]:
            raise ReceiptError("malformed public set-shuffle scope")
        positions = tuple(tuple(raw[3 + 4*i:6 + 4*i]) for i in range(raw[2]))
        if len(set(positions)) != len(positions) or any(p not in (0, 1) or loc != raw[1] or seq > 7
                                                     for p, loc, seq in positions):
            raise ReceiptError("invalid public set-shuffle participants")
        # The following words are NOT a public identity permutation. Keep the
        # original bytes, but do not join them to the private local UID map.
        return PublicShuffleScope(source, tuple(sorted({(p, loc) for p, loc, _ in positions})), positions, False, raw[2])
    if msg not in (C.MSG_SHUFFLE_HAND, C.MSG_SHUFFLE_DECK, C.MSG_SHUFFLE_EXTRA, C.MSG_SWAP_GRAVE_DECK):
        return None
    if len(raw) < 2 or raw[1] not in (0, 1):
        raise ReceiptError("malformed public zone-shuffle scope")
    player = raw[1]
    if msg in (C.MSG_SHUFFLE_HAND, C.MSG_SHUFFLE_EXTRA):
        if len(raw) < 3 or len(raw) != 3 + 4 * raw[2]:
            raise ReceiptError("malformed public hidden-zone shuffle count")
        location = C.LOCATION_HAND if msg == C.MSG_SHUFFLE_HAND else C.LOCATION_EXTRA
        return PublicShuffleScope(source, ((player, location),),
                                  tuple((player, location, seq) for seq in range(raw[2])),
                                  msg == C.MSG_SHUFFLE_HAND, raw[2])
    if len(raw) != 2:
        raise ReceiptError("malformed public deck shuffle/exchange")
    zones = ((player, C.LOCATION_DECK),)
    if msg == C.MSG_SWAP_GRAVE_DECK:
        zones += ((player, C.LOCATION_GRAVE),)
    return PublicShuffleScope(source, zones, (), True, None)


def _public_token_turnover(follower, start, old, new):
    """Every object born in the batch is a token, and every object gone was one: tokens are public cards with no
    hidden identity to certify.

    The core creates card objects only for tokens (the follower adds none while
    it runs), and a new token has no place until its summon puts it on the
    field: a born object is either still unplaced or a token in a monster
    zone. A gone object either never got a place or left a monster zone as a
    token the received packets show.
    """
    from .client_shadow import card_code
    core, pduel = follower.core, follower.local.pduel

    def token(code):
        data = core._query_card_data(code)
        return data is not None and bool(data.type & _TYPE_TOKEN)

    for uid in new.keys() - old.keys():
        row = new[uid]
        if row.location == 0:
            continue
        if row.location != C.LOCATION_MZONE or not token(card_code(core, pduel, row.controller, row.location,
                                                                     row.sequence)):
            return False
    gone = [old[uid] for uid in old.keys() - new.keys()]
    if any(row.location not in (0, C.LOCATION_MZONE) for row in gone):
        return False
    shown = sum(1 for packet in follower.packets[start:follower.cursor]
                if packet[0] == C.MSG_MOVE and len(packet) >= 13 and packet[6] == C.LOCATION_MZONE
                and token(struct.unpack_from("<I", packet, 1)[0] & 0x7FFFFFFF))
    return shown >= sum(1 for row in gone if row.location == C.LOCATION_MZONE)


def fixed_batch(follower, saved, callback, receipt_origin=None):
    scope = boundary(follower)
    origin = receipt_origin if receipt_origin is not None else saved
    try:
        start = follower.receipt_saved[origin[0]]
    except KeyError as exc:
        raise ReceiptError("batch origin lost its receipt snapshot") from exc
    frame = {"attempts": 0}
    scope.frames.append(frame)
    try:
        result = callback()
        ledger = follower.receipt_state
        finish = entities(follower)
        old, new = ({row.uid: row for row in rows} for rows in (start.entities, finish))
        markers = {marker for operation in ledger.pending for marker in operation.markers}
        markers.update(scope.history_markers)
        shuffle_rows = []
        for index in range(start.cursor, follower.cursor):
            source = PublicSource(index, follower.receipt_packet_raw_indices[index],
                                  follower.packets[index], "participants_only")
            shuffle = _shuffle_scope(source)
            if shuffle is not None:
                shuffle_rows.append(shuffle)
        shuffles = tuple(shuffle_rows)
        if shuffles:
            markers.add("hidden_shuffle_mapping_unvalidated")
        if old.keys() != new.keys():
            markers.add("public_token_birth_or_death" if _public_token_turnover(follower, start.cursor, old, new)
                        else "entity_birth_or_death_unvalidated")
        facts = []
        operations = ledger.pending + scope.history_proofs
        for operation in operations:
            sources = operation.sources
            if any(item.matched_index >= follower.cursor for item in operation.sources):
                markers.add("public_evidence_after_batch")
            if (operation.kind == "hydrate" and operation.changed and not operation.markers and sources
                    and operation.uid in old and operation.uid in new):
                facts.append(RootFactPatch(operation.uid, old[operation.uid], new[operation.uid],
                                           int(operation.code), sources,
                                           max(item.matched_index for item in sources) + 1))
        receipt = OriginReceipt(start.native_sha256, (start.cursor, follower.cursor),
            (_raw_cut(follower, start.cursor), _raw_cut(follower, follower.cursor)),
            (ledger.own_cursor, follower.answered), tuple(follower.own[ledger.own_cursor:follower.answered]),
            start.entities, frame["before"], finish, operations, tuple(facts), shuffles,
            tuple(sorted(markers)), frame["attempts"])
        follower.receipt_state = ReceiptLedger(ledger.accepted + (receipt,), (), follower.answered,
                                              ledger.replay_events)
        return result
    except BaseException as exc:
        scope.owner.audit.append({"kind": "origin-transaction-rejected", "origin": start.native_sha256,
                                  "mutations": follower.receipt_state.pending, "error": str(exc)})
        raise
    finally:
        scope.frames.pop()
