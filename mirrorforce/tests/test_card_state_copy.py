"""Card-copy optimization must preserve every masking field and row boundary."""

from dataclasses import asdict, dataclass, field, fields, replace
import random

import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.worldmodel import state


def card(**changes):
    values = {item.name: index + 1 for index, item in enumerate(fields(state.CardState))}
    values.update(overlay=(10, 20), counters=((1, 2),), targets=(100,), setcode=(300,), hidden=False, public=True)
    values.update(changes)
    return state.CardState(**values)


@pytest.mark.parametrize('changes', [{}, {'public': False}, {'sequence': 7}, {'public': True, 'sequence': 0}, {'code': 0, 'hidden': True}])
def test_plain_copy_matches_every_standard_dataclass_field(changes):
    original = card()
    expected = replace(original, **changes)
    actual = state._replace_card(original, **changes)
    assert actual == expected and vars(actual) == vars(expected)
    assert actual is not original
    actual.attack = -9999
    assert original.attack != actual.attack


def test_extra_attributes_and_missing_default_fields_use_standard_semantics():
    original = card()
    original.diagnostic = object()
    actual = state._replace_card(original, sequence=3)
    assert vars(actual) == vars(replace(original, sequence=3))
    assert not hasattr(actual, 'diagnostic')
    del original.diagnostic
    del original.public
    assert vars(state._replace_card(original, sequence=3)) == vars(replace(original, sequence=3))


def test_subclass_constructor_and_noninit_field_semantics_are_preserved():
    @dataclass
    class Derived(state.CardState):
        checked: int = field(init=False)

        def __post_init__(self):
            self.checked = 2 * self.sequence

    original = Derived(0, C.LOCATION_HAND, 4)
    actual = state._replace_card(original, sequence=8)
    assert type(actual) is Derived and actual == replace(original, sequence=8)
    assert actual.checked == 16
    with pytest.raises(ValueError):
        state._replace_card(original, checked=123)


def test_unknown_update_is_not_silently_added():
    with pytest.raises(TypeError):
        state._replace_card(card(), unexpected=123)


@pytest.mark.parametrize('player', [0, 1])
def test_full_mask_matches_standard_copy_for_varied_disclosure_states(monkeypatch, player):
    rng = random.Random(202609755 + player)
    optimized = state._replace_card
    for _ in range(40):
        cards = []
        for controller in (0, 1):
            for location in state.ZONES:
                for sequence in range(rng.randrange(1, 5)):
                    cards.append(card(controller=controller, location=location, sequence=sequence,
                        code=rng.choice([111, 111, 222, 333]), position=rng.choice([1, 2, 4, 8]),
                        owner=controller, public=bool(rng.randrange(2)), hidden=False))
        revealed = frozenset((c.controller, c.location, c.sequence, c.code) for c in cards if rng.randrange(5) == 0)
        snapshot = state.StateSnapshot(turn=3, turn_player=1, phase=C.PHASE_MAIN1, lp=(8000, 7100), cards=cards,
            counts={(controller, location): sum(c.controller == controller and c.location == location for c in cards)
                for controller in (0, 1) for location in state.ZONES})
        original = asdict(snapshot)
        monkeypatch.setattr(state, '_replace_card', replace)
        expected = state.mask_for(snapshot, player, revealed)
        monkeypatch.setattr(state, '_replace_card', optimized)
        actual = state.mask_for(snapshot, player, revealed)
        assert asdict(actual) == asdict(expected)
        assert asdict(snapshot) == original
        assert not ({id(c) for c in actual.cards} & {id(c) for c in snapshot.cards})
        assert not ({id(c) for c in actual.cards} & {id(c) for c in expected.cards})


def test_masks_do_not_share_mutable_rows_between_observers():
    snapshot = state.StateSnapshot(turn=1, turn_player=0, phase=C.PHASE_MAIN1, lp=(8000, 8000),
        cards=[card(controller=0, location=C.LOCATION_MZONE, sequence=0, code=111, position=C.POS_FACEUP_ATTACK)])
    left, right = state.mask_for(snapshot, 0), state.mask_for(snapshot, 1)
    before = asdict(right)
    left.cards[0].code = 999
    left.cards[0].attack = 999
    assert asdict(right) == before and snapshot.cards[0].code == 111
