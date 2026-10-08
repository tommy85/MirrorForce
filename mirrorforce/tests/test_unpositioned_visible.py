"""A copy the viewer already sees at a slot is positioned, not "unpositioned".

The disclosure ledger's multiset lower bound includes cards the viewer can see
directly -- a face-up continuous spell in the opponent's back row -- and
``resolve`` never needs to place those.  Before 2026-09-03 they landed in
``revealed_unpositioned`` on top of their own slot token, a double count that
broke the tensorizer's invariant ``k <= hidden slots`` on 0.30% of zone calls
(the design notes) and refused every export.
``mask_for`` now subtracts the visible copies the ledger did not place, and the
leak audit refuses more identities than a zone has hidden slots.
"""
from __future__ import annotations

import dataclasses

from mirrorforce.netduel import constants as C
from mirrorforce.worldmodel.state import CardState, StateSnapshot, leaks, mask_for

MULTIROLE = 24010609   # a continuous spell that sits face-up in the back row
VIEWER = 0
OPP = 1


def _back_row(*, faceup_code: int, set_codes=(0, 0)) -> StateSnapshot:
    """Opponent S/T zone: face-down at 0 and 4, a face-up card at 2."""
    cards = [
        CardState(controller=OPP, location=C.LOCATION_SZONE, sequence=0,
                  code=set_codes[0], position=C.POS_FACEDOWN),
        CardState(controller=OPP, location=C.LOCATION_SZONE, sequence=2,
                  code=faceup_code, position=C.POS_FACEUP,
                  type=C.TYPE_SPELL),
        CardState(controller=OPP, location=C.LOCATION_SZONE, sequence=4,
                  code=set_codes[1], position=C.POS_FACEDOWN),
    ]
    return StateSnapshot(turn=3, turn_player=OPP, phase=C.PHASE_MAIN1,
                         lp=(8000, 8000), cards=cards)


def test_a_visible_copy_the_ledger_did_not_place_leaves_the_column():
    masked = mask_for(_back_row(faceup_code=MULTIROLE), VIEWER, frozenset(),
                      unpositioned={(OPP, C.LOCATION_SZONE, MULTIROLE): 1})
    assert masked.revealed_unpositioned == ()
    faceup = [c for c in masked.cards if c.sequence == 2]
    assert faceup and faceup[0].code == MULTIROLE, "the slot token keeps the code"
    assert not leaks(masked)


def test_only_the_visible_copies_are_subtracted():
    masked = mask_for(_back_row(faceup_code=MULTIROLE), VIEWER, frozenset(),
                      unpositioned={(OPP, C.LOCATION_SZONE, MULTIROLE): 2})
    assert masked.revealed_unpositioned == ((OPP, C.LOCATION_SZONE, MULTIROLE),)
    assert not leaks(masked)


def test_a_copy_the_ledger_placed_is_not_subtracted_again():
    """Placed copies were already taken out by ``unanchored_identities``."""
    raw = _back_row(faceup_code=0, set_codes=(MULTIROLE, 0))
    placed = frozenset({(OPP, C.LOCATION_SZONE, 0, MULTIROLE)})
    masked = mask_for(raw, VIEWER, placed,
                      unpositioned={(OPP, C.LOCATION_SZONE, MULTIROLE): 1})
    assert masked.revealed_unpositioned == ((OPP, C.LOCATION_SZONE, MULTIROLE),)
    slot0 = [c for c in masked.cards if c.sequence == 0][0]
    assert slot0.code == MULTIROLE and slot0.public
    assert not leaks(masked)


def test_the_audit_refuses_more_identities_than_hidden_slots():
    masked = mask_for(_back_row(faceup_code=MULTIROLE), VIEWER, frozenset())
    crowded = dataclasses.replace(masked, revealed_unpositioned=(
        (OPP, C.LOCATION_SZONE, MULTIROLE),
        (OPP, C.LOCATION_SZONE, 63166095),
        (OPP, C.LOCATION_SZONE, 98338152),
    ))
    problems = leaks(crowded)
    assert any("with only 2 hidden slots" in p for p in problems), problems
    fits = dataclasses.replace(crowded, revealed_unpositioned=crowded.revealed_unpositioned[1:])
    assert not leaks(fits)
