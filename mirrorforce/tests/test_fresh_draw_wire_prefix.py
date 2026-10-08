"""The exact original public packets that lost a pre-draw hand identity."""
import json
from pathlib import Path

from mirrorforce.netduel import constants as C
from mirrorforce.netduel.board import ShadowBoard
from mirrorforce.netduel.cards import load_ydk
from mirrorforce.netduel.agent_public_recipe import declare


def fixture():
    return json.loads((Path(__file__).parent / 'fixtures/search-fresh-draw-prefix-20261006.json').read_bytes())


def test_original_public_draw_and_set_keep_old_shuffled_hand_knowledge():
    data = fixture()
    main, extra, _ = load_ydk(Path(__file__).parents[1] / 'decks/stage-a/SkyStriker.ydk')
    assert declare(main, extra) == data['recipe']
    board = ShadowBoard()
    board.start(0, main, extra)
    for index, (msg, body) in enumerate(data['public_messages']):
        if index == 2781:
            assert msg == C.MSG_MOVE and body == '000000000102020a0108020a00040002'
            assert 2 in board.disclosure._fresh[0][1, C.LOCATION_HAND]
        board.apply(msg, bytes.fromhex(body))
        if index in (2062, 2781, 2805):
            assert board.disclosure.known_counts(0)[1, C.LOCATION_HAND, 51227866] == 1
    assert len(board.zone(1, C.LOCATION_HAND)) == 1
    assert board.disclosure.known_slots(0, 1, C.LOCATION_HAND) == {}
    assert board.disclosure.unanchored_identities(0)[1, C.LOCATION_HAND, 51227866] == 1
