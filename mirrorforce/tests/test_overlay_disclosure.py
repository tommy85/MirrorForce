"""Material indices must not alias the host slot and invent later identities."""
import struct

import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.netduel.disclosure import DisclosureLedger


A, B, X = 45710945, 72705654, 7477101
OVERLAY = C.LOCATION_MZONE | C.LOCATION_OVERLAY


def word(location, sequence, position=C.POS_FACEUP_ATTACK):
    return (location << 8) | (sequence << 16) | (position << 24)


def material_scenario(observe):
    """Attach distinct cards, detach index zero twice, then return the first."""
    for code, host, index in ((A, 2, 0), (A, 2, 1), (B, 2, 2), (X, 3, 0)):
        observe(C.MSG_MOVE, struct.pack("<IIII", code, word(C.LOCATION_GRAVE, 0),
                                       word(OVERLAY, host, index), 0))
    for index in range(2):
        observe(C.MSG_MOVE, struct.pack("<IIII", A, word(OVERLAY, 2, 0),
                                       word(C.LOCATION_GRAVE, index), 0))
    observe(C.MSG_MOVE, struct.pack("<IIII", A, word(C.LOCATION_GRAVE, 0),
                                   word(C.LOCATION_HAND, 0, C.POS_FACEDOWN_DEFENSE), 0))


def test_detached_material_does_not_become_its_hosts_last_material():
    ledger = DisclosureLedger()
    material_scenario(lambda _msg, body: ledger.observe_move_message(body))
    known = ledger.known_counts(1)
    assert known[(0, C.LOCATION_HAND, A)] == 1
    assert known[(0, C.LOCATION_HAND, B)] == 0
    assert known[(0, C.LOCATION_GRAVE, A)] == 1
    assert known[(0, C.LOCATION_GRAVE, B)] == 0
    assert ledger.known_slots(1, 0, OVERLAY) == {2 << 8: B, 3 << 8: X}
    # The public material anchor still identifies a move whose destination
    # makes the server redact its code. This must not look up host 3's card.
    ledger.observe_move(word(OVERLAY, 2, 0), word(C.LOCATION_HAND, 1, 8), 0)
    assert ledger.known_counts(1)[(0, C.LOCATION_HAND, B)] == 1
    assert ledger.known_slots(1, 0, OVERLAY) == {3 << 8: X}


def test_overlay_insertion_shifts_only_materials_of_that_host():
    ledger = DisclosureLedger()
    for code, host in ((A, 2), (B, 3), (X, 2)):
        ledger.observe_move(word(C.LOCATION_GRAVE, 0), word(OVERLAY, host, 0), code)
    assert ledger.known_slots(1, 0, OVERLAY) == {2 << 8: X, (2 << 8) | 1: A, 3 << 8: B}


def test_a_host_move_or_swap_drops_stale_material_coordinates():
    for swap in (False, True):
        ledger = DisclosureLedger()
        ledger.observe_move(word(C.LOCATION_GRAVE, 0), word(OVERLAY, 2, 0), A)
        if swap:
            ledger.observe_swap(struct.pack("<IIII", X, word(C.LOCATION_MZONE, 2),
                                             B, word(C.LOCATION_MZONE, 3)))
        else:
            ledger.observe_move(word(C.LOCATION_MZONE, 2), word(C.LOCATION_GRAVE, 0), X)
        assert ledger.known_slots(1, 0, OVERLAY) == {}
        ledger.observe_move(word(OVERLAY, 2, 0), word(C.LOCATION_GRAVE, 1), B)
        assert ledger.known_counts(1)[(0, C.LOCATION_GRAVE, B)] == 1
        assert ledger.known_counts(1)[(0, C.LOCATION_GRAVE, A)] == 0


@pytest.mark.parametrize('change', ['host_leaves', 'list_insertion', 'shuffle'])
def test_extra_host_reuse_cannot_turn_a_material_into_the_old_hosts_material(change):
    ledger = DisclosureLedger()
    extra_overlay = C.LOCATION_EXTRA | C.LOCATION_OVERLAY
    ledger.observe_move(word(C.LOCATION_GRAVE, 0), word(extra_overlay, 4, 0), A)
    if change == 'host_leaves':
        ledger.observe_move(word(C.LOCATION_EXTRA, 4, 8), word(C.LOCATION_MZONE, 2), X)
    elif change == 'list_insertion':
        ledger.observe_move(word(C.LOCATION_MZONE, 2), word(C.LOCATION_EXTRA, 0, 8), X)
    else:
        ledger.observe_shuffle(C.MSG_SHUFFLE_EXTRA, bytes([0]))
    assert ledger.known_slots(1, 0, extra_overlay) == {}
    # B is publicly sent to GY. This used to be recorded as A, overflowing
    # public identity counts even though every graveyard card was face-up.
    ledger.observe_move(word(extra_overlay, 4, 0), word(C.LOCATION_GRAVE, 0), B)
    assert ledger.known_counts(1)[(0, C.LOCATION_GRAVE, B)] == 1
    assert ledger.known_counts(1)[(0, C.LOCATION_GRAVE, A)] == 0
