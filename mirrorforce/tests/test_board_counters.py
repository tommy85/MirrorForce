"""Counters on the shadow board follow the reference client, including the silent face-down clear."""

import struct

from mirrorforce.netduel import constants as C
from mirrorforce.netduel.board import ShadowBoard


def _board():
    board = ShadowBoard()
    board.start(0, [89631139] * 40, [])
    return board


def _counter(add, ctype, player, location, sequence, count):
    msg = C.MSG_ADD_COUNTER if add else C.MSG_REMOVE_COUNTER
    return msg, struct.pack("<H", ctype) + bytes([player, location, sequence]) + struct.pack("<H", count)


def _pos_change(player, location, sequence, previous, current, code=89631139):
    return C.MSG_POS_CHANGE, struct.pack("<I", code) + bytes([player, location, sequence, previous, current])


def test_turning_face_down_clears_the_counters_the_core_clears_without_a_message():
    board = _board()
    board.apply(*_counter(True, 0x1, 1, C.LOCATION_MZONE, 2, 3))
    assert board.counters_of(1, C.LOCATION_MZONE, 2) == {0x1: 3}
    board.apply(*_pos_change(1, C.LOCATION_MZONE, 2, C.POS_FACEUP_ATTACK, C.POS_FACEDOWN_DEFENSE))
    assert board.counters_of(1, C.LOCATION_MZONE, 2) == {}


def test_face_up_position_changes_and_flips_up_keep_the_counters():
    board = _board()
    board.apply(*_counter(True, 0x1, 0, C.LOCATION_MZONE, 0, 2))
    board.apply(*_counter(True, 0x100e, 0, C.LOCATION_MZONE, 0, 1))
    board.apply(*_pos_change(0, C.LOCATION_MZONE, 0, C.POS_FACEUP_ATTACK, C.POS_FACEUP_DEFENSE))
    assert board.counters_of(0, C.LOCATION_MZONE, 0) == {0x1: 2, 0x100e: 1}
    # A face-down card flipped face-up has no counters to keep, and the flip must not invent a clear elsewhere.
    board.apply(*_counter(True, 0x1, 0, C.LOCATION_SZONE, 1, 1))
    board.apply(*_pos_change(0, C.LOCATION_MZONE, 3, C.POS_FACEDOWN_DEFENSE, C.POS_FACEUP_ATTACK))
    assert board.counters_of(0, C.LOCATION_SZONE, 1) == {0x1: 1}
    assert board.counters_of(0, C.LOCATION_MZONE, 0) == {0x1: 2, 0x100e: 1}


def test_only_the_flipped_slot_loses_its_counters():
    board = _board()
    board.apply(*_counter(True, 0x1, 0, C.LOCATION_MZONE, 0, 2))
    board.apply(*_counter(True, 0x1, 0, C.LOCATION_MZONE, 1, 4))
    board.apply(*_pos_change(0, C.LOCATION_MZONE, 1, C.POS_FACEUP_DEFENSE, C.POS_FACEDOWN_DEFENSE))
    assert board.counters_of(0, C.LOCATION_MZONE, 0) == {0x1: 2}
    assert board.counters_of(0, C.LOCATION_MZONE, 1) == {}
