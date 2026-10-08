"""Opt-in LOCAL empty-phase-window witnesses; never server-menu truth.

One omitted native optional phase pass may explain one opaque WAITING packet.
The next ordinary packet still matches exactly. Every consumption is an
unvalidated history receipt and therefore remains outside search admission.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
import struct

from ..netduel import constants as C
from .client_sync import SyncError, _Frontier

INFORMATION_SET_SEARCH = True
SCHEMA = "local-phase-pass-audit/v1"
HEADER = struct.Struct("<4sHHIIQBB6s")
RECORD = struct.Struct("<IIHBB")


@dataclass(frozen=True)
class PhasePass:
    offset: int
    turn: int
    phase: int
    player: int
    priority_passed: int


def _functions(core):
    enable = getattr(core._lib, "duel_set_phase_pass_audit", None)
    query = getattr(core._lib, "query_phase_pass_audit", None)
    if enable is None or query is None:
        raise SyncError("core does not support opt-in local phase-pass witnesses")
    enable.argtypes, enable.restype = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint8], ctypes.c_int32
    query.argtypes, query.restype = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int32], ctypes.c_int32
    return enable, query


def configure(follower):
    from .client_root import _advance_boundary
    scope = _advance_boundary(follower)
    if scope.phase != "idle" or not follower.record_origins:
        raise SyncError("phase audit requires an owned initial idle receipt boundary")
    enable, _query = _functions(follower.core)
    if enable(follower.local.pduel, 1, 1) != 0:
        raise SyncError("native core rejected opening phase-pass audit enable")


def parse(raw):
    if len(raw) < HEADER.size:
        raise SyncError("short phase-pass witness header")
    magic, version, width, size, count, batch, enabled, overflow, reserved = HEADER.unpack_from(raw)
    if magic != b"PPW1" or version != 1 or width != RECORD.size or size != len(raw) \
            or count > 64 or size != HEADER.size + count * width or reserved != bytes(6) \
            or enabled != 1 or overflow not in (0, 1) or not batch:
        raise SyncError("invalid phase-pass witness ABI")
    if overflow:
        raise SyncError("phase-pass witness overflow; no opaque packet may be consumed")
    rows = tuple(PhasePass(*RECORD.unpack_from(raw, i)) for i in range(HEADER.size, size, width))
    if any(r.player not in (0, 1) or r.priority_passed not in (0, 1)
           or r.phase not in (C.PHASE_DRAW, C.PHASE_STANDBY, C.PHASE_BATTLE_START, C.PHASE_BATTLE, C.PHASE_END)
           or not r.turn for r in rows) or list(r.offset for r in rows) != sorted(r.offset for r in rows):
        raise SyncError("invalid phase-pass witness order/context")
    return batch, rows


def read(follower):
    from .client_root import _advance_boundary
    scope = _advance_boundary(follower)
    if scope.phase != "batch-and-observe":
        raise SyncError("phase witness query is not at its complete process boundary")
    _enable, query = _functions(follower.core)
    size = query(follower.local.pduel, 1, None, 0)
    if not HEADER.size <= size <= HEADER.size + 64 * RECORD.size:
        raise SyncError("native phase-pass witness query refused: " + str(size))
    buffer = ctypes.create_string_buffer(size)
    if query(follower.local.pduel, 1, buffer, size) != size:
        raise SyncError("phase-pass witness changed during read")
    return parse(buffer.raw)


def observed_messages(follower, messages):
    """Read once BEFORE Python observation; offsets refer to the full C batch.

    When the native message right after an omitted pass is itself an opponent
    prompt (the hints only the opponent sees before a prompt aside, and the
    next batch when only such hints are left in this one), the received
    WAITING may be that prompt's own: the pass then keeps
    the packet unless an earlier attempt of the batch proved it must be taken
    (``phase_pass_consume``); the undecided packet is listed for that retry,
    and the receipt the pass would have written is kept by packet
    (``phase_pass_kept``, cleared whenever the follower loads a snapshot) for
    the prompt that took it.
    """
    from .client_origin_receipt import PublicMutation, PublicSource
    from .client_sync import _WAITING_PROMPTS, _prompt_player
    batch, witnesses = read(follower)
    messages = list(messages)
    offset, next_witness = 0, 0
    for position, message in enumerate(messages):
        while next_witness < len(witnesses) and witnesses[next_witness].offset <= offset:
            witness = witnesses[next_witness]
            next_witness += 1
            if witness.offset != offset:
                raise SyncError("phase-pass witness is not at a native message boundary")
            if witness.player == follower.viewer or follower.cursor >= len(follower.packets) \
                    or follower.packets[follower.cursor] != bytes([C.MSG_WAITING]):
                continue
            if follower.cursor + 1 >= len(follower.packets):
                raise _Frontier()
            index = follower.cursor
            source = PublicSource(index, follower.receipt_packet_raw_indices[index],
                                  follower.packets[index], "opaque_phase_pass")
            operation = PublicMutation("opaque_phase_pass", (batch, next_witness - 1, witness.offset,
                witness.turn, witness.phase, witness.player, witness.priority_passed), None, 0, 0, False,
                (source,), ("opaque_phase_pass_history_unvalidated",))
            ahead = next((row for row in messages[position:]
                          if row.msg != C.MSG_HINT or row.payload[1] == follower.viewer), None)
            if (ahead is None or ahead.msg in _WAITING_PROMPTS and _prompt_player(ahead) != follower.viewer) \
                    and index not in follower.phase_pass_consume:
                follower.phase_pass_ambiguous.append(index)
                follower.phase_pass_kept[index] = operation
                continue
            ledger = follower.receipt_state
            follower.receipt_state = replace(ledger, pending=ledger.pending + (operation,))
            follower.cursor += 1
        yield message
        offset += 1 + len(message.payload)
    if any(r.offset != offset for r in witnesses[next_witness:]):
        raise SyncError("phase-pass witness outside native output buffer")
    # No trailing witness can consume a transport packet without a concrete
    # subsequent native message; that frontier stays unconsumed/replayable.
