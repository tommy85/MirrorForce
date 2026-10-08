"""Anonymous departures from our deck named by the group elimination (Jack-in-the-Hand) or by their new field
slot, and the strict own-deck count check for every departure the stream leaves unnamed."""

import struct

import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.netduel.board import ShadowBoard
from mirrorforce.netduel.wmplay import snapshot_from_shadow

A, B, X, FILLER = 68810435, 10966439, 4215180, 89631139   # Cooky, Marshmao, Lollipo (Yummy), filler


def _board():
    board = ShadowBoard()
    board.start(0, [A] * 2 + [B] + [X] * 3 + [FILLER] * 34, [])
    board.deck_count[1] = 40
    return board


def _card(code, controller=0, location=C.LOCATION_DECK, sequence=0, position=C.POS_FACEDOWN_DEFENSE):
    return struct.pack("<I", code) + bytes([controller, location, sequence, position])


def _unselect(select, chosen, player=0):
    body = bytes([player, 0, 0, 3, 3, len(select)]) + b"".join(_card(c, sequence=i) for i, c in enumerate(select))
    return C.MSG_SELECT_UNSELECT_CARD, body + bytes([len(chosen)]) + b"".join(
        _card(c, sequence=len(select) + i) for i, c in enumerate(chosen))


def _select(cards, player=0):
    return C.MSG_SELECT_CARD, bytes([player, 0, 1, 1, len(cards)]) + b"".join(
        _card(code, controller, location, i) for i, (code, controller, location) in enumerate(cards))


def _move(code, source, target):
    place = lambda c, loc, seq, pos: c | loc << 8 | seq << 16 | pos << 24
    return C.MSG_MOVE, struct.pack("<IIII", code, place(*source), place(*target), 0x40)


def _reveal_three(board):
    """We select B, A and one X from our deck; the core re-lists what is chosen after each pick."""
    board.apply(C.MSG_CHAIN_SOLVING, bytes([1]))
    board.apply(*_unselect([A, A, B, X, X, X], []))
    board.observe_response(bytes([1, 2]))
    board.apply(*_unselect([A, A, X, X, X], [B]))
    board.observe_response(bytes([1, 0]))
    board.apply(*_unselect([X, X, X], [B, A]))
    board.observe_response(bytes([1, 1]))


def _deck(board):
    snapshot = snapshot_from_shadow(board, 0, (8000, 8000), 1)
    return sorted(card.code for card in snapshot.cards if card.controller == 0 and card.location == C.LOCATION_DECK)


def test_the_opponent_takes_one_selected_card_unseen_and_the_offer_of_the_other_two_names_it():
    board = _board()
    _reveal_three(board)
    board.apply(*_move(0, (0, C.LOCATION_DECK, 9, 0x0A), (1, C.LOCATION_HAND, 4, 0x0A)))
    assert board.unresolved_own_deck_departures() == 1
    revision = board.revision
    board.apply(*_select([(B, 0, C.LOCATION_DECK), (X, 0, C.LOCATION_DECK)]))
    assert board.unresolved_own_deck_departures() == 0 and board.resolved_departures == 1
    assert board.revision == revision + 1
    assert board.our_remaining_deck()[A] == 1 and board.deck_count[0] == 39
    board.observe_response(bytes([1, 0]))           # we take B; it reaches our hand with its code
    board.apply(*_move(B, (0, C.LOCATION_DECK, 3, 0x0A), (0, C.LOCATION_HAND, 5, 0x0A)))
    board.apply(C.MSG_CHAIN_SOLVED, bytes([1]))
    deck = _deck(board)
    assert len(deck) == board.deck_count[0] == 38 and deck.count(A) == 1 and deck.count(B) == 0 and deck.count(X) == 3


def _unresolved(board):
    with pytest.raises(ValueError, match="own-deck multiset/count drift.*unresolved_anonymous_departures=1"):
        snapshot_from_shadow(board, 0, (8000, 8000), 1)


def test_a_departure_outside_a_chain_resolution_stays_unresolved_and_the_view_is_refused():
    board = _board()
    board.apply(*_move(0, (0, C.LOCATION_DECK, 9, 0x0A), (1, C.LOCATION_HAND, 4, 0x0A)))
    board.apply(*_select([(B, 0, C.LOCATION_DECK)]))
    assert board.unresolved_own_deck_departures() == 1 and board.resolved_departures == 0
    _unresolved(board)


@pytest.mark.parametrize("offer", [
    [(B, 0, C.LOCATION_DECK), (FILLER, 0, C.LOCATION_DECK)],   # a card outside the selected group
    [(B, 0, C.LOCATION_DECK)],                                  # two short for one departure
    [(B, 0, C.LOCATION_DECK), (X, 0, C.LOCATION_HAND)],         # a card outside our deck
    [(B, 0, C.LOCATION_DECK), (0, 0, C.LOCATION_DECK)],         # a card without its code
])
def test_an_offer_that_does_not_complete_the_group_resolves_nothing(offer):
    board = _board()
    _reveal_three(board)
    board.apply(*_move(0, (0, C.LOCATION_DECK, 9, 0x0A), (1, C.LOCATION_HAND, 4, 0x0A)))
    board.apply(*_select(offer))
    board.apply(C.MSG_CHAIN_SOLVED, bytes([1]))
    assert board.unresolved_own_deck_departures() == 1 and board.resolved_departures == 0
    _unresolved(board)


def test_the_group_and_the_departure_belong_to_one_chain_link_resolution():
    board = _board()
    _reveal_three(board)
    board.apply(C.MSG_CHAIN_SOLVED, bytes([1]))
    board.apply(C.MSG_CHAIN_SOLVING, bytes([2]))
    board.apply(*_move(0, (0, C.LOCATION_DECK, 9, 0x0A), (1, C.LOCATION_HAND, 4, 0x0A)))
    board.apply(*_select([(B, 0, C.LOCATION_DECK), (X, 0, C.LOCATION_DECK)]))
    assert board.resolved_departures == 0
    board.apply(C.MSG_CHAIN_END, b"")
    assert board.unresolved_own_deck_departures() == 1
    _unresolved(board)


def test_a_response_that_names_no_listed_card_is_refused_and_a_response_without_a_prompt_is_ignored():
    board = _board()
    board.apply(C.MSG_CHAIN_SOLVING, bytes([1]))
    board.apply(*_select([(B, 0, C.LOCATION_DECK)]))
    with pytest.raises(ValueError, match="names listed cards"):
        board.observe_response(bytes([1, 3]))
    board.apply(*_unselect([A], []))
    with pytest.raises(ValueError, match="names one listed card"):
        board.observe_response(bytes([2, 0, 0]))
    board.observe_response(bytes([1, 0]))           # the prompt was consumed by the refused response
    board.apply(*_select([(B, 0, C.LOCATION_DECK)]))
    board.observe_response((-1).to_bytes(4, "little", signed=True))   # cancel
    with pytest.raises(ValueError, match="truncated"):
        board.apply(C.MSG_SELECT_CARD, bytes([0, 0, 1, 1, 2]) + _card(B))


def _confirm(code, controller, location, sequence, player=0):
    return C.MSG_CONFIRM_CARDS, bytes([player, 0, 1]) + struct.pack("<I", code) + bytes([controller, location, sequence])


def test_a_card_sent_face_down_to_the_opponent_field_is_named_when_confirmed_there():
    """Deal 152 of the internal Elo: one deck copy summoned to our field, the other face down to the opponent's,
    then shown to us at its new slot."""
    board = _board()
    board.apply(C.MSG_CHAIN_SOLVING, bytes([2]))
    board.apply(*_move(A, (0, C.LOCATION_DECK, 23, 0x08), (0, C.LOCATION_MZONE, 1, C.POS_FACEUP_ATTACK)))
    board.apply(*_move(0, (0, C.LOCATION_DECK, 26, 0x08), (1, C.LOCATION_MZONE, 3, C.POS_FACEDOWN_DEFENSE)))
    assert board.unresolved_own_deck_departures() == 1
    board.apply(*_confirm(A, 1, C.LOCATION_MZONE, 3))
    board.apply(C.MSG_CHAIN_SOLVED, bytes([2]))
    assert board.unresolved_own_deck_departures() == 0 and board.resolved_departures == 1
    deck = _deck(board)
    assert len(deck) == board.deck_count[0] == 38 and deck.count(A) == 0


def test_a_tracked_departure_follows_its_card_and_is_named_by_a_public_move_or_not_at_all():
    board = _board()
    board.apply(*_move(0, (0, C.LOCATION_DECK, 9, 0x08), (1, C.LOCATION_MZONE, 2, C.POS_FACEDOWN_DEFENSE)))
    board.apply(C.MSG_SWAP, struct.pack("<I", 0) + bytes([1, C.LOCATION_MZONE, 2, C.POS_FACEDOWN_DEFENSE])
                + struct.pack("<I", 0) + bytes([1, C.LOCATION_MZONE, 4, C.POS_FACEDOWN_DEFENSE]))
    board.apply(*_confirm(B, 1, C.LOCATION_MZONE, 2))    # the other card, now at the old slot
    assert board.unresolved_own_deck_departures() == 1
    board.apply(*_move(X, (1, C.LOCATION_MZONE, 4, C.POS_FACEDOWN_DEFENSE), (0, C.LOCATION_GRAVE, 0, C.POS_FACEUP)))
    assert board.unresolved_own_deck_departures() == 0 and board.our_remaining_deck()[X] == 2
    hand = _board()
    hand.apply(*_move(0, (0, C.LOCATION_DECK, 9, 0x08), (1, C.LOCATION_HAND, 4, 0x0A)))   # a list zone: not followed
    hand.apply(*_confirm(A, 1, C.LOCATION_HAND, 4))
    assert hand.unresolved_own_deck_departures() == 1
    _unresolved(hand)
