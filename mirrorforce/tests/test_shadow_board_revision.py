"""The shadow board's revision changes whenever a snapshot could read different state."""

import struct

from mirrorforce.netduel import constants as C
from mirrorforce.netduel.board import _PROMPT_MESSAGES, _REFRESH_MESSAGES, _STATE_MESSAGES, ShadowBoard


def _board():
    board = ShadowBoard()
    board.start(0, [89631139] * 40, [])
    return board


def _mzone_refresh(player, attack):
    body = struct.pack("<I", 89631139) + bytes([player, C.LOCATION_MZONE, 0, C.POS_FACEUP_ATTACK]) + struct.pack("<i", attack)
    segment = struct.pack("<II", len(body) + 8, C.QUERY_CODE | C.QUERY_POSITION | C.QUERY_ATTACK) + body
    return bytes([player, C.LOCATION_MZONE]) + segment + struct.pack("<i", 4) * 6


def test_state_messages_reach_a_handler_prompts_are_read_and_every_other_message_is_only_counted():
    messages = sorted({value for name, value in vars(C).items() if name.startswith("MSG_") and isinstance(value, int)})
    for msg in messages:
        if msg in _REFRESH_MESSAGES:
            continue
        board = _board()
        before = board.revision
        try:
            board.apply(msg, bytes(64))
        except Exception:
            pass
        if msg in _STATE_MESSAGES:
            assert board.revision == before + 1 and board.unknown_messages[msg] == 0, msg
        elif msg in _PROMPT_MESSAGES:  # read for the own-deck group elimination; no departure to resolve here
            assert board.revision == before and board.unknown_messages[msg] == 0, msg
        else:
            assert board.revision == before and board.unknown_messages[msg] == 1, msg


def test_a_refresh_moves_the_revision_only_when_it_changes_the_board():
    board = _board()
    start = board.revision
    board.apply(C.MSG_UPDATE_DATA, _mzone_refresh(1, 3000))
    assert board.revision == start + 1 and board.zone(1, C.LOCATION_MZONE)[0].attack == 3000
    board.apply(C.MSG_UPDATE_DATA, _mzone_refresh(1, 3000))
    assert board.revision == start + 1
    board.apply(C.MSG_UPDATE_DATA, _mzone_refresh(1, 1500))
    assert board.revision == start + 2 and board.zone(1, C.LOCATION_MZONE)[0].attack == 1500
    board.apply(C.MSG_UPDATE_DATA, b"")
    assert board.revision == start + 2


def test_an_identity_learned_from_a_prompt_moves_the_revision():
    board = _board()
    board.apply(C.MSG_UPDATE_DATA, _mzone_refresh(0, 3000))
    card = board.zone(0, C.LOCATION_MZONE)[0]
    card.code, card.hidden = 0, True
    before = board.revision
    assert board.cross_check([(89631139, 0, C.LOCATION_MZONE, 0)]) == []
    assert board.revision == before + 1 and card.code == 89631139
    assert board.cross_check([(89631139, 0, C.LOCATION_MZONE, 0)]) == [] and board.revision == before + 1
