"""Synthetic follower placeholders cannot become an observer's public card facts."""
import copy
import pickle
import struct

import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.netduel.disclosure import DisclosureLedger
from mirrorforce.common.client_disclosure import AnonymousDisclosureLedger
from mirrorforce.common.client_shadow import BLANK_CODES, BlankClientSync
from mirrorforce.common.client_sync import ClientSync

RAYE = 26077387


def place(player, location, sequence=0, position=C.POS_FACEDOWN_DEFENSE):
    return player | location << 8 | sequence << 16 | position << 24


def facts(ledger):
    return [(ledger.known_counts(p), copy.deepcopy(ledger._slots[p]), copy.deepcopy(ledger._fresh[p]),
             ledger.faceup_extra_counts(p), ledger.category_constraints(p)) for p in (0, 1)]


@pytest.mark.parametrize('blank', sorted(BLANK_CODES))
@pytest.mark.parametrize('flag', [0, 0x80000000])
def test_hydrated_move_reveal_is_not_overridden_by_a_placeholder_anchor(blank, flag):
    ledger = AnonymousDisclosureLedger(BLANK_CODES)
    ledger.observe_move(place(1, C.LOCATION_DECK), place(1, C.LOCATION_HAND), blank | flag)
    assert ledger.known_slots(1, 1, C.LOCATION_HAND) == {}
    # Native hydration has now installed the publicly revealed real card.
    ledger.observe_move(place(1, C.LOCATION_HAND), place(1, C.LOCATION_GRAVE, 10, 5), RAYE)
    ledger.observe_move(place(1, C.LOCATION_GRAVE, 10, 5), place(0, C.LOCATION_MZONE, 0, 1), RAYE)
    for viewer in (0, 1):
        assert ledger.known_slots(viewer, 0, C.LOCATION_MZONE) == {0: RAYE}
        assert ledger.known_counts(viewer)[0, C.LOCATION_MZONE, RAYE] == 1
        assert not any(k[2] in BLANK_CODES for k in ledger.known_counts(viewer))


@pytest.mark.parametrize('viewer', [None, 0, 1])
def test_reserved_draw_is_unknown_but_preserves_public_slot_and_count_transitions(viewer):
    blank = min(BLANK_CODES)
    ledger = AnonymousDisclosureLedger(BLANK_CODES)
    ordinary = DisclosureLedger()
    raw = bytes([1, 2]) + struct.pack('<II', blank | 0x80000000, RAYE | 0x80000000)
    expected = bytes([1, 2]) + struct.pack('<II', 0, RAYE | 0x80000000)
    ledger.observe_draw(raw, viewer=viewer, hand_start=3)
    ordinary.observe_draw(expected, viewer=viewer, hand_start=3)
    assert facts(ledger) == facts(ordinary)
    assert struct.unpack_from('<I', raw, 2)[0] == blank | 0x80000000


def test_all_public_reveal_families_cannot_publish_reserved_codes():
    ledger = AnonymousDisclosureLedger(BLANK_CODES)
    blank = min(BLANK_CODES)
    ledger.disclose(1, C.LOCATION_HAND, blank, sequence=0)
    ledger.disclose_faceup_extra(1, blank)
    ledger.observe_confirm(bytes([1, 0, 1]) + struct.pack('<I', blank) + bytes([1, 2, 0]), C.MSG_CONFIRM_CARDS)
    ledger.observe_deck_top(bytes([1, 0]) + struct.pack('<I', blank | 0x80000000))
    ledger.observe_chaining(struct.pack('<IIII', blank, place(1, 2), 0, 0))
    ledger.observe_move(place(1, 2), place(1, 4, 0, 1), blank)
    ledger.observe_swap(struct.pack('<IIII', blank, place(1, 4, 0, 1), RAYE, place(0, 4, 0, 1)))
    assert ledger._chain_stack == [0] and ledger._last_chaining == 0
    for viewer in (0, 1):
        assert not any(k[2] in BLANK_CODES for k in ledger.known_counts(viewer))
        assert not any(k[1] in BLANK_CODES for k in ledger.faceup_extra_counts(viewer))


def test_real_public_facts_and_private_audiences_keep_the_ordinary_law_through_snapshots():
    ledger = AnonymousDisclosureLedger(BLANK_CODES)
    ordinary = DisclosureLedger()
    for target in (ledger, ordinary):
        target.disclose(0, C.LOCATION_HAND, RAYE, sequence=0, audience=1)
        target.observe_shuffle(C.MSG_SHUFFLE_HAND, bytes([0, 2]) + b'\0' * 8)
        target.observe_draw(bytes([0, 1]) + struct.pack('<I', RAYE), viewer=0, hand_start=2)
        target.observe_move(place(0, 2, 2), place(0, 8, 1), 0, viewer=0)
    assert facts(ledger) == facts(ordinary)
    assert not ledger.known_counts(1)
    for restored in (copy.deepcopy(ledger), pickle.loads(pickle.dumps(ledger, protocol=4))):
        assert facts(restored) == facts(ledger)
        assert restored.anonymous_codes == BLANK_CODES
        restored.disclose(1, C.LOCATION_HAND, min(BLANK_CODES), sequence=0)
        assert facts(restored) == facts(ledger)


def test_only_blank_follower_selects_the_reserved_identity_boundary():
    assert type(ClientSync._new_disclosure(None)) is DisclosureLedger
    assert type(BlankClientSync._new_disclosure(None)) is AnonymousDisclosureLedger


@pytest.mark.parametrize('codes', [(), (0,), (True,), (-1,), (2**31,)])
def test_reserved_identity_set_must_be_explicit_and_valid(codes):
    with pytest.raises(ValueError, match='reserved positive codes'):
        AnonymousDisclosureLedger(codes)
