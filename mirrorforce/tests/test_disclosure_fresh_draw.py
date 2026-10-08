"""Fresh public hand slots cannot consume identities from an older shuffled group."""
import struct

import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.netduel.disclosure import DisclosureLedger

CODE = 51227866


def place(player, location, sequence, position=C.POS_FACEDOWN):
    return player | location << 8 | sequence << 16 | position << 24


def drawn(player, codes):
    return bytes([player, len(codes)]) + b''.join(struct.pack('<I', code) for code in codes)


def known_old_hand(viewer):
    other = 1 - viewer
    ledger = DisclosureLedger()
    ledger.disclose(other, C.LOCATION_HAND, CODE, sequence=0, audience=1 << viewer)
    ledger.observe_shuffle(C.MSG_SHUFFLE_HAND, drawn(other, [0, 0]))
    assert ledger.known_counts(viewer)[other, C.LOCATION_HAND, CODE] == 1
    return ledger, other


@pytest.mark.parametrize('viewer', [0, 1])
@pytest.mark.parametrize('visible_departure', [False, True])
def test_fresh_draw_leaves_without_consuming_old_identity(viewer, visible_departure):
    ledger, other = known_old_hand(viewer)
    ledger.observe_draw(drawn(other, [0]), viewer=viewer, hand_start=2)
    ledger.observe_move(place(other, C.LOCATION_HAND, 2),
        place(other, C.LOCATION_GRAVE if visible_departure else C.LOCATION_SZONE, 1,
              C.POS_FACEUP if visible_departure else C.POS_FACEDOWN),
        CODE if visible_departure else 0, viewer=viewer)
    assert ledger.known_counts(viewer)[other, C.LOCATION_HAND, CODE] == 1
    assert ledger.known_slots(viewer, other, C.LOCATION_HAND) == {}
    assert not ledger._fresh[viewer].get((other, C.LOCATION_HAND))


@pytest.mark.parametrize('viewer', [0, 1])
def test_shuffle_or_missing_public_count_does_not_invent_a_fresh_slot(viewer):
    for supplied, shuffled in ((False, False), (True, True)):
        ledger, other = known_old_hand(viewer)
        ledger.observe_draw(drawn(other, [0]), viewer=viewer, hand_start=2 if supplied else None)
        if shuffled:
            ledger.observe_shuffle(C.MSG_SHUFFLE_HAND, drawn(other, [0, 0, 0]))
        ledger.observe_move(place(other, C.LOCATION_HAND, 2), place(other, C.LOCATION_SZONE, 1), 0, viewer=viewer)
        assert ledger.known_counts(viewer).get((other, C.LOCATION_HAND, CODE), 0) == 0


def test_old_member_departure_and_multiple_fresh_draws_follow_public_slot_shifts():
    ledger, other = known_old_hand(0)
    ledger.observe_draw(drawn(other, [0, 0]), viewer=0, hand_start=2)
    for sequence in (2, 2):
        ledger.observe_move(place(other, C.LOCATION_HAND, sequence), place(other, C.LOCATION_SZONE, sequence), 0, viewer=0)
        assert ledger.known_counts(0)[other, C.LOCATION_HAND, CODE] == 1
    ledger, other = known_old_hand(0)
    ledger.observe_draw(drawn(other, [0]), viewer=0, hand_start=2)
    ledger.observe_move(place(other, C.LOCATION_HAND, 0), place(other, C.LOCATION_SZONE, 1), 0, viewer=0)
    assert ledger.known_counts(0).get((other, C.LOCATION_HAND, CODE), 0) == 0
    assert ledger._fresh[0][other, C.LOCATION_HAND] == {1}


def test_revealed_fresh_draw_is_a_real_anchor_and_an_additional_copy():
    ledger, other = known_old_hand(0)
    ledger.observe_draw(drawn(other, [CODE]), viewer=0, hand_start=2)
    assert ledger.known_slots(0, other, C.LOCATION_HAND) == {2: CODE}
    assert ledger.known_counts(0)[other, C.LOCATION_HAND, CODE] == 2
    ledger.observe_move(place(other, C.LOCATION_HAND, 2), place(other, C.LOCATION_SZONE, 1), 0, viewer=0)
    assert ledger.known_counts(0)[other, C.LOCATION_HAND, CODE] == 1


@pytest.mark.parametrize('start,body', [(True, drawn(1, [0])), (-1, drawn(1, [0])),
    (255, drawn(1, [0])), (2, b'\x01\x01'), (2, drawn(1, [0]) + b'\x00')])
def test_draw_coordinates_require_complete_bounded_public_evidence(start, body):
    ledger, _ = known_old_hand(0)
    before = ledger.known_counts(0)
    with pytest.raises(ValueError, match='public pre-draw hand count'):
        ledger.observe_draw(body, viewer=0, hand_start=start)
    assert ledger.known_counts(0) == before
