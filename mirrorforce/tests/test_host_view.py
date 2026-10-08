"""Checks the ported host routing against ``single_duel.cpp``, rule by rule.

Each case names the ``ygopro-client/gframe/single_duel.cpp`` lines it pins. A wrong mask changes what a
local session hands a client, so the two upstream quirks (``MSG_CARD_SELECTED`` reaches nobody,
``MSG_RANDOM_SELECTED`` is re-sent to seat 1) are pinned too.
"""

from __future__ import annotations

import struct

import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.netduel import host_view as hv


def info(controler: int, location: int, sequence: int, position: int) -> bytes:
    return bytes([controler, location, sequence, position])


def test_hint_type_routing_matches_single_duel_649_680():
    for hint_type in (1, 2, 3, 5):
        d = hv.deliver(C.MSG_HINT, bytes([hint_type, 1]) + struct.pack("<i", 7))
        assert set(d.payloads) == {1}
    for hint_type in (4, 6, 7, 8, 9, 11):
        d = hv.deliver(C.MSG_HINT, bytes([hint_type, 1]) + struct.pack("<i", 7))
        assert set(d.payloads) == {0}
    d = hv.deliver(C.MSG_HINT, bytes([10, 1]) + struct.pack("<i", 7))
    assert set(d.payloads) == {0, 1}
    # A hint type outside the table reaches nobody (the switch has no default).
    assert hv.deliver(C.MSG_HINT, bytes([12, 1]) + struct.pack("<i", 7)).payloads == {}


def test_select_prompts_go_only_to_the_asked_seat():
    payload = bytes([1, 0, 0, 0, 0, 0]) + b"\x00" * 8
    for msg in (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD,
                C.MSG_SELECT_YESNO, C.MSG_SELECT_CHAIN):
        d = hv.deliver(msg, payload)
        assert set(d.payloads) == {1}, msg
    # SELECT_SUM carries the player in its second byte (single_duel.cpp:843 advances pbuf first).
    d = hv.deliver(C.MSG_SELECT_SUM, bytes([0, 1]) + b"\x00" * 16)
    assert set(d.payloads) == {1}


def test_idlecmd_refreshes_every_zone_before_sending():
    before, after = hv.refreshes(C.MSG_SELECT_IDLECMD, bytes([0]) + b"\x00" * 12)
    assert before == hv.REFRESH_ALL_ZONES
    assert after == ()
    # A new phase is the opposite: send first, then refresh (single_duel.cpp:997-1010).
    before, after = hv.refreshes(C.MSG_NEW_PHASE, b"\x01\x00")
    assert before == ()
    assert after == hv.REFRESH_ALL_ZONES


def test_select_card_hides_codes_of_cards_the_asked_seat_does_not_control():
    # single_duel.cpp:764-782: a code is zeroed exactly when c != player.
    player = 0
    payload = bytearray([player, 0, 1, 2, 2])          # player, cancelable, min, max, count
    payload += struct.pack("<i", 1111) + info(0, C.LOCATION_MZONE, 0, C.POS_FACEUP_ATTACK)
    payload += struct.pack("<i", 2222) + info(1, C.LOCATION_MZONE, 1, C.POS_FACEUP_ATTACK)
    d = hv.deliver(C.MSG_SELECT_CARD, bytes(payload))
    seen = d.payloads[player]
    assert struct.unpack_from("<i", seen, 1 + 5)[0] == 1111
    assert struct.unpack_from("<i", seen, 1 + 5 + 8)[0] == 0


def test_select_unselect_card_masks_both_blocks():
    player = 1
    payload = bytearray([player, 0, 0, 1, 1, 1])
    payload += struct.pack("<i", 1111) + info(0, C.LOCATION_MZONE, 0, 1)
    payload += bytes([1])
    payload += struct.pack("<i", 2222) + info(1, C.LOCATION_MZONE, 1, 1)
    d = hv.deliver(C.MSG_SELECT_UNSELECT_CARD, bytes(payload))
    seen = d.payloads[player]
    assert struct.unpack_from("<i", seen, 1 + 6)[0] == 0  # controlled by the opponent: zeroed
    assert struct.unpack_from("<i", seen, 1 + 6 + 8 + 1)[0] == 2222


def test_confirm_cards_from_deck_reaches_only_the_owner():
    # single_duel.cpp:883-900: pbuf[5] is the location of the first card.
    deck = bytes([1, 0, 1]) + struct.pack("<i", 4242) + info(1, C.LOCATION_DECK, 3, 0)
    assert set(hv.deliver(C.MSG_CONFIRM_CARDS, deck).payloads) == {1}
    grave = bytes([1, 0, 1]) + struct.pack("<i", 4242) + info(1, C.LOCATION_GRAVE, 3, 0)
    assert set(hv.deliver(C.MSG_CONFIRM_CARDS, grave).payloads) == {0, 1}


def test_shuffle_hand_shows_codes_only_to_the_owner():
    # single_duel.cpp:907-918
    payload = bytes([1, 2]) + struct.pack("<ii", 1111, 2222)
    d = hv.deliver(C.MSG_SHUFFLE_HAND, payload)
    assert struct.unpack_from("<i", d.payloads[1], 1 + 2)[0] == 1111
    assert struct.unpack_from("<ii", d.payloads[0], 1 + 2) == (0, 0)
    assert d.after == (hv.Refresh("hand", 1, 0x781fff, 0),)


def test_draw_hides_the_code_unless_the_public_bit_is_set():
    # single_duel.cpp:1249-1258: a draw with the 0x80 high bit is public.
    payload = bytes([0, 2]) + struct.pack("<Ii", 1111, 0)
    payload = bytes([0, 2]) + struct.pack("<II", 1111, 2222 | 0x80000000)
    d = hv.deliver(C.MSG_DRAW, payload)
    mine = d.payloads[0]
    theirs = d.payloads[1]
    assert struct.unpack_from("<I", mine, 1 + 2)[0] == 1111
    assert struct.unpack_from("<I", theirs, 1 + 2)[0] == 0
    assert struct.unpack_from("<I", theirs, 1 + 2 + 4)[0] == 2222 | 0x80000000


def test_move_hides_the_code_going_into_deck_or_hand_and_for_facedown():
    # single_duel.cpp:1011-1034
    def move(cc, cl, cp, pl=C.LOCATION_MZONE, pc=None):
        pc = cc if pc is None else pc
        return (struct.pack("<i", 5555)
                + info(pc, pl, 0, C.POS_FACEUP_ATTACK)
                + info(cc, cl, 1, cp)
                + struct.pack("<i", 0))

    to_grave = hv.deliver(C.MSG_MOVE, move(0, C.LOCATION_GRAVE, C.POS_FACEUP_ATTACK))
    assert struct.unpack_from("<i", to_grave.payloads[1], 1)[0] == 5555

    to_hand = hv.deliver(C.MSG_MOVE, move(0, C.LOCATION_HAND, C.POS_FACEDOWN_ATTACK))
    assert struct.unpack_from("<i", to_hand.payloads[0], 1)[0] == 5555
    assert struct.unpack_from("<i", to_hand.payloads[1], 1)[0] == 0

    facedown = hv.deliver(C.MSG_MOVE, move(0, C.LOCATION_SZONE,
                                           C.POS_FACEDOWN_DEFENSE))
    assert struct.unpack_from("<i", facedown.payloads[1], 1)[0] == 0

    # A face-down banish also hides the code: cl is REMOVED and the position is face-down.
    banished = hv.deliver(C.MSG_MOVE, move(0, C.LOCATION_REMOVED,
                                           C.POS_FACEDOWN_DEFENSE))
    assert struct.unpack_from("<i", banished.payloads[1], 1)[0] == 0


def test_move_strips_the_reveal_flag_for_both_seats():
    payload = (struct.pack("<i", 5555)
               + info(0, C.LOCATION_DECK, 0, C.POS_FACEDOWN_DEFENSE)
               + info(0, C.LOCATION_SZONE, 1,
                      C.POS_FACEDOWN_DEFENSE | C.POS_REVEAL)
               + struct.pack("<i", 0))
    d = hv.deliver(C.MSG_MOVE, payload)
    for seat in (0, 1):
        assert d.payloads[seat][1 + 8 + 3] == C.POS_FACEDOWN_DEFENSE
    # POS_REVEAL keeps the code visible to both players (hide_code is False).
    assert struct.unpack_from("<i", d.payloads[1], 1)[0] == 5555


def test_move_refreshes_the_destination_slot_only_on_a_real_move():
    same_slot = (struct.pack("<i", 1) + info(0, C.LOCATION_MZONE, 2, 1)
                 + info(0, C.LOCATION_MZONE, 2, 1) + struct.pack("<i", 0))
    assert hv.deliver(C.MSG_MOVE, same_slot).after == ()
    moved = (struct.pack("<i", 1) + info(0, C.LOCATION_HAND, 2, 1)
             + info(0, C.LOCATION_MZONE, 3, 1) + struct.pack("<i", 0))
    after = hv.deliver(C.MSG_MOVE, moved).after
    assert after == (hv.Refresh("single", 0, 0xf81fff, 0, C.LOCATION_MZONE, 3),)


def test_pos_change_refreshes_the_real_slot_after_the_four_byte_code():
    payload = struct.pack("<I", 0x08040102) + bytes([
        1, C.LOCATION_MZONE, 3, C.POS_FACEDOWN_DEFENSE, C.POS_FACEUP_ATTACK])
    delivery = hv.deliver(C.MSG_POS_CHANGE, payload)
    assert delivery.after == (
        hv.Refresh("single", 1, 0xf81fff, 0, C.LOCATION_MZONE, 3),)
    assert delivery.payloads == {0: bytes([C.MSG_POS_CHANGE]) + payload,
                                 1: bytes([C.MSG_POS_CHANGE]) + payload}
    not_flipped = payload[:-2] + bytes([C.POS_FACEUP_ATTACK, C.POS_FACEUP_DEFENSE])
    assert hv.deliver(C.MSG_POS_CHANGE, not_flipped).after == ()


def test_refresh_defaults_match_each_single_duel_h_declaration():
    assert [(r.kind, r.flag, r.use_cache) for r in hv.REFRESH_ALL_ZONES] == [
        ("mzone", 0x881fff, 1), ("mzone", 0x881fff, 1),
        ("szone", 0x681fff, 1), ("szone", 0x681fff, 1),
        ("hand", 0x681fff, 1), ("hand", 0x681fff, 1)]
    assert hv.deliver(C.MSG_SWAP_GRAVE_DECK, b"\x01").after == (
        hv.Refresh("grave", 1, 0x81fff, 1),)
    assert hv.deliver(C.MSG_SHUFFLE_EXTRA, b"\x01\x00").after == (
        hv.Refresh("extra", 1, 0xe81fff, 1),)


def test_set_hides_the_code_from_everyone():
    # single_duel.cpp:1051-1059: MSG_SET zeroes the code for both players.
    payload = struct.pack("<i", 9999) + info(0, C.LOCATION_SZONE, 0, 8)
    d = hv.deliver(C.MSG_SET, payload)
    assert struct.unpack_from("<i", d.payloads[0], 1)[0] == 0
    assert struct.unpack_from("<i", d.payloads[1], 1)[0] == 0


def test_spsummoning_hides_the_code_for_a_facedown_special_summon():
    payload = struct.pack("<i", 7777) + info(1, C.LOCATION_MZONE, 0,
                                             C.POS_FACEDOWN_DEFENSE)
    d = hv.deliver(C.MSG_SPSUMMONING, payload)
    assert struct.unpack_from("<i", d.payloads[1], 1)[0] == 7777
    assert struct.unpack_from("<i", d.payloads[0], 1)[0] == 0


def test_card_selected_reaches_nobody_and_random_selected_double_sends_to_seat_one():
    # The two upstream quirks: single_duel.cpp:1224 and 1230.
    assert hv.deliver(C.MSG_CARD_SELECTED,
                      bytes([0, 1]) + info(0, C.LOCATION_MZONE, 0, 1)).payloads == {}
    d = hv.deliver(C.MSG_RANDOM_SELECTED,
                   bytes([1, 1]) + info(1, C.LOCATION_HAND, 0, 0))
    assert d.counts == {1: 2}
    d0 = hv.deliver(C.MSG_RANDOM_SELECTED,
                    bytes([0, 1]) + info(0, C.LOCATION_HAND, 0, 0))
    assert d0.counts == {0: 1, 1: 1}


def test_chaining_is_public_and_carries_no_refresh():
    payload = (struct.pack("<i", 3333) + info(1, C.LOCATION_HAND, 0, 0)
               + bytes([1, 0, 0, 0]) + struct.pack("<i", 1))
    d = hv.deliver(C.MSG_CHAINING, payload)
    assert set(d.payloads) == {0, 1}
    assert struct.unpack_from("<i", d.payloads[0], 1)[0] == 3333
    assert d.before == () and d.after == ()


def test_client_stream_projects_a_whole_stream():
    stream = [
        (C.MSG_NEW_TURN, bytes([0])),
        (C.MSG_SELECT_IDLECMD, bytes([1]) + b"\x00" * 8),
        (C.MSG_CARD_SELECTED, bytes([0, 0])),
    ]
    assert len(hv.client_stream(stream, 0)) == 1
    assert len(hv.client_stream(stream, 1)) == 2


# Refresh packet masks.


def _segment(flags: int, code: int, position: int, extra: bytes = b"") -> bytes:
    body = struct.pack("<I", flags) + struct.pack("<i", code)
    body += struct.pack("<I", position) + extra
    return struct.pack("<i", len(body) + 4) + body


def test_update_data_for_a_facedown_field_slot_is_zeroed_for_the_opponent():
    query = _segment(C.QUERY_CODE | C.QUERY_POSITION, 4242,
                     (C.POS_FACEDOWN_DEFENSE << 24) | 0x0004)
    out = hv.mask_update_data("szone", 0, C.LOCATION_SZONE, query)
    assert struct.unpack_from("<i", out[0], 3 + 8)[0] == 4242
    # The length field stays; everything after it is zeroed.
    assert out[1][3:3 + 4] == out[0][3:3 + 4]
    assert out[1][3 + 4:] == b"\x00" * (len(query) - 4)
    assert len(out[0]) == len(out[1]) == 3 + len(query)


def test_update_data_strips_reveal_for_both_and_then_shows_the_code():
    position = ((C.POS_FACEDOWN_DEFENSE | C.POS_REVEAL) << 24) | 0x0004
    query = _segment(C.QUERY_CODE | C.QUERY_POSITION, 4242, position)
    out = hv.mask_update_data("mzone", 1, C.LOCATION_MZONE, query)
    for seat in (0, 1):
        assert out[seat][3 + 12 + 3] == C.POS_FACEDOWN_DEFENSE
    assert struct.unpack_from("<i", out[0], 3 + 8)[0] == 4242


def test_update_data_hand_hides_every_facedown_card_from_the_opponent():
    query = (_segment(C.QUERY_CODE | C.QUERY_POSITION, 11,
                      (C.POS_FACEDOWN_ATTACK << 24) | 0x0002)
             + _segment(C.QUERY_CODE | C.QUERY_POSITION, 22,
                        (C.POS_FACEUP_ATTACK << 24) | 0x0002))
    out = hv.mask_update_data("hand", 0, C.LOCATION_HAND, query)
    assert struct.unpack_from("<i", out[0], 3 + 8)[0] == 11
    assert struct.unpack_from("<i", out[1], 3 + 8)[0] == 0
    # The face-up card is visible to both players.
    second = 3 + 16
    assert struct.unpack_from("<i", out[1], second + 8)[0] == 22


def test_update_data_extra_reaches_only_the_owner_and_grave_reaches_both():
    query = _segment(C.QUERY_CODE | C.QUERY_POSITION, 33,
                     (C.POS_FACEDOWN_DEFENSE << 24) | 0x0040)
    assert set(hv.mask_update_data("extra", 1, C.LOCATION_EXTRA, query)) == {1}
    assert set(hv.mask_update_data("grave", 1, C.LOCATION_GRAVE, query)) == {0, 1}


def test_update_card_shrinks_to_a_16_byte_stub_when_hidden():
    query = _segment(C.QUERY_CODE | C.QUERY_POSITION | C.QUERY_ATTACK, 4242,
                     (C.POS_FACEDOWN_DEFENSE << 24) | 0x0008,
                     struct.pack("<i", 1800))
    out = hv.mask_update_card(0, C.LOCATION_SZONE, 2, query)
    assert struct.unpack_from("<i", out[0], 4 + 8)[0] == 4242
    stub = out[1]
    assert len(stub) == 4 + 16
    assert struct.unpack_from("<i", stub, 4)[0] == 16
    assert struct.unpack_from("<I", stub, 8)[0] == C.QUERY_CODE | C.QUERY_POSITION
    assert struct.unpack_from("<i", stub, 12)[0] == 0
    assert struct.unpack_from("<I", stub, 16)[0] == ((C.POS_FACEDOWN_DEFENSE << 24)
                                                     | 0x0008)


def test_update_card_faceup_is_identical_for_both_seats():
    query = _segment(C.QUERY_CODE | C.QUERY_POSITION, 4242,
                     (C.POS_FACEUP_ATTACK << 24) | 0x0004)
    out = hv.mask_update_card(1, C.LOCATION_MZONE, 0, query)
    assert out[0] == out[1]


def test_unknown_message_reaches_nobody_rather_than_being_guessed():
    assert hv.deliver(200, b"\x00\x01").payloads == {}


@pytest.mark.parametrize("msg", [C.MSG_SUMMONED, C.MSG_SPSUMMONED,
                                 C.MSG_FLIPSUMMONED])
def test_summon_completions_refresh_the_field_but_not_the_hands(msg):
    before, after = hv.refreshes(msg, b"")
    assert before == ()
    assert [r.kind for r in after] == ["mzone", "mzone", "szone", "szone"]
