"""Rollback-owned execution order of a blank follower, never a server transcript.

This is the input evidence for Option-A replay, NOT an executable replay or a
search admission. It records local native prompts before response rebinding,
local UID maps and public mutations in their actual order. An alternative
particle must rebind each response semantically; replaying these raw menu
indices on a hydrated engine is expressly unsupported.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace

from ..immutable import ImmutableRecord

INFORMATION_SET_SEARCH = True
SCHEMA = "client-accepted-replay-journal/v1"


class JournalError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeBoundary(ImmutableRecord):
    arena_digest: int
    entities: tuple


@dataclass(frozen=True)
class ReplayEvent(ImmutableRecord):
    ordinal: int
    kind: str
    before: NativeBoundary
    after: NativeBoundary
    payload: tuple
    matched_cursor: int
    own_answered: int
    validated: bool = True


def enabled(follower):
    return bool(getattr(follower, "record_replay", False))


def capture(follower):
    """At a complete C boundary only; the digest detects omitted native writes.

    The arena digest is a local continuity check, not a cross-engine identity,
    cryptographic attestation or model feature. UID maps have no card identities.
    """
    from .client_origin_receipt import boundary, entities
    boundary(follower)
    lib = follower.core._lib
    if not hasattr(lib, "duel_arena_digest"):
        raise JournalError("replay journal requires the arena continuity ABI")
    fn = lib.duel_arena_digest
    fn.argtypes, fn.restype = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)), ctypes.c_int32
    out = ctypes.c_uint64()
    if fn(follower.local.pduel, ctypes.byref(out)) != 0:
        raise JournalError("cannot digest the local follower arena")
    return NativeBoundary(int(out.value), entities(follower))


def append(follower, kind, before, payload=(), *, validated=True):
    from .client_origin_receipt import boundary
    boundary(follower)
    state = follower.receipt_state
    events = state.replay_events
    if (not events and kind != "opening") or (events and kind == "opening"):
        raise JournalError("journal must have exactly one opening")
    if events and events[-1].after != before:
        raise JournalError("an unjournaled native operation breaks the accepted replay history "
                           f"after {events[-1].kind}#{events[-1].ordinal}, before {kind}; "
                           f"arena {events[-1].after.arena_digest:016x}->{before.arena_digest:016x}, "
                           f"entities_equal={events[-1].after.entities == before.entities}")
    row = ReplayEvent(len(events), kind, before, capture(follower), tuple(payload),
                      follower.cursor, follower.answered, validated)
    follower.receipt_state = replace(state, replay_events=events + (row,))


def opening(follower):
    if enabled(follower):
        # This is the local blank deck, not the server's concealed deck/order.
        payload = (follower.viewer, follower.seed, tuple(sorted(follower.rules.items())),
                   tuple((tuple(d.main), tuple(d.extra)) for d in follower.decks),
                   tuple(tuple(order) for order in follower.local.config.forced_deck_orders),
                   follower.history_registry)
        append(follower, "opening", capture(follower), payload)


def batch_messages(follower, messages):
    """Capture bytes without native queries inside the batch's observer window."""
    from .client_origin_receipt import boundary
    frame = boundary(follower).frames[-1]
    frame["replay_messages"] = tuple(bytes([m.msg]) + bytes(m.payload) for m in messages)


def query(follower, kind, args, callback):
    """Record native reads whose ocgcore query cache is itself mutable arena state."""
    if not enabled(follower):
        return callback()
    from .client_origin_receipt import native_operation
    before = capture(follower)
    try:
        with native_operation(follower, "query"):
            result = callback()
    except BaseException:
        append(follower, "query", before, (kind, tuple(args), None), validated=False)
        raise
    append(follower, "query", before, (kind, tuple(args), result))
    return result


def checked(events):
    """Validate immutable accepted order before handing it to a replay producer.

    Merely passing this check does not establish semantic response rebinding,
    mutation support, public-stream parity, or complete Option-A reconstruction.
    """
    if not events or not isinstance(events, tuple):
        raise JournalError("an immutable nonempty accepted journal is required")
    for index, row in enumerate(events):
        if not isinstance(row, ReplayEvent) or row.ordinal != index:
            raise JournalError("journal event order is incomplete")
        if (index == 0) != (row.kind == "opening"):
            raise JournalError("journal opening is absent or repeated")
        if row.kind not in ("opening", "mutation", "response", "process", "query"):
            raise JournalError("unknown native journal event")
        if not row.validated:
            raise JournalError("a rejected native attempt is not an accepted replay step")
        if index and events[index - 1].after != row.before:
            raise JournalError("journal native boundaries are discontinuous")
    return events


def at_root(root):
    """Export only a complete journal ending at this exact owned native root."""
    from .client_root import RootEnvelope, _digest
    from .client_entity_map import capture_entities
    if type(root) is not RootEnvelope:
        raise JournalError("journal export requires an owned root")
    root._check()
    if _digest(root.host, root.owner.core.card_pool()) != root.host_digest:
        raise JournalError("saved root metadata changed before journal export")
    events = checked(root.host["sync"]["receipt_state"].replay_events)
    value = ctypes.c_uint64()
    if root.owner.core._lib.duel_arena_digest(root.duel, ctypes.byref(value)) != 0 \
            or value.value != events[-1].after.arena_digest \
            or capture_entities(root).entities != events[-1].after.entities:
        raise JournalError("accepted journal does not end at the current native root")
    return events
