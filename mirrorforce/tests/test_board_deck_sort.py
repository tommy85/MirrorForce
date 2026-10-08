"""A secret deck sort erases what the viewer knew about deck positions, never which cards are there.

``PROCESSOR_SORT_DECK`` reorders the top cards in private and then reports each as an anonymous
``MSG_MOVE`` onto its own deck slot. The shadow board must hand that anonymous marker to the disclosure
ledger instead of a remembered code, as the native ledger (fed the core's code 0) does.
"""

import struct

from mirrorforce.netduel import constants as C
from mirrorforce.netduel.board import ShadowBoard

TOP, SECOND = 14558127, 24203749


def _board():
    board = ShadowBoard()
    board.start(0, [89631139] * 40, [])
    # MSG_START: player type, rule, LPs, then deck/extra counts of both players.
    board.apply(C.MSG_START, bytes([0, 5]) + struct.pack("<ii", 8000, 8000) + struct.pack("<HHHH", 40, 0, 40, 0))
    # The opponent's top two deck cards are shown to both players at slots 39 and 38.
    board.apply(C.MSG_CONFIRM_DECKTOP, bytes([1, 2]) + struct.pack("<I", TOP) + bytes([1, C.LOCATION_DECK, 39])
                + struct.pack("<I", SECOND) + bytes([1, C.LOCATION_DECK, 38]))
    return board


def _move(code, previous, current):
    return C.MSG_MOVE, struct.pack("<IIII", code, previous, current, 0x40)


def _deck(sequence, controller=1):
    return controller | C.LOCATION_DECK << 8 | sequence << 16 | C.POS_FACEDOWN_DEFENSE << 24


def test_confirmed_deck_top_is_known_by_position():
    board = _board()
    assert board.disclosure.known_slots(0, 1, C.LOCATION_DECK) == {39: TOP, 38: SECOND}


def test_a_secret_sort_keeps_the_multiset_and_forgets_the_positions():
    board = _board()
    for sequence in (39, 38):
        board.apply(*_move(0, _deck(sequence), _deck(sequence)))
    assert board.disclosure.known_slots(0, 1, C.LOCATION_DECK) == {}
    counts = board.disclosure.known_counts(0)
    assert counts[(1, C.LOCATION_DECK, TOP)] == 1 and counts[(1, C.LOCATION_DECK, SECOND)] == 1
    # A later anonymous draw from the sorted top cannot be named either.
    board.apply(*_move(0, _deck(39), 1 | C.LOCATION_HAND << 8 | C.POS_FACEDOWN_DEFENSE << 24))
    assert board.disclosure.known_slots(0, 1, C.LOCATION_HAND) == {}


def test_a_known_card_leaving_its_deck_slot_is_still_recovered():
    board = _board()
    # Not a sort marker: the card leaves the deck, so the viewer still knows which one moved.
    board.apply(*_move(0, _deck(39), 1 | C.LOCATION_HAND << 8 | C.POS_FACEDOWN_DEFENSE << 24))
    assert board.disclosure.known_slots(0, 1, C.LOCATION_HAND) == {0: TOP}
