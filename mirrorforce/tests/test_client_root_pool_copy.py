"""Pure static-database isolation; no native core or search admission."""
import copy
from types import SimpleNamespace

import pytest

from mirrorforce.netduel.cards import CardData, CardPool
from mirrorforce.common.client_root import _copy_card_pool, _digest, _pool_digest


def pool_fixture():
    pool = object.__new__(CardPool)
    pool.db_path = '/not-opened/static-database.cdb'
    shared_codes = [0x115, 0x1015]
    first = CardData(code=10001, setcodes=shared_codes, name='same name', type=1, attack=1500)
    second = CardData(code=10002, setcodes=shared_codes, name='same name', type=2)
    pool.cards = {10001:first, 10002:second, 10003:first}
    pool.card_ids = {10001:1, 10002:2}
    return pool


def test_pool_copy_preserves_values_types_aliases_and_ordinary_methods():
    pool = pool_fixture()
    legacy, actual = copy.deepcopy(pool), _copy_card_pool(pool)
    assert type(actual) is CardPool
    assert _digest(actual) == _digest(legacy) == _digest(pool)
    assert _pool_digest(actual) == _pool_digest(legacy) == _pool_digest(pool)
    assert actual.cards[10001] is actual.cards[10003]
    assert actual.cards[10001].setcodes is actual.cards[10002].setcodes
    assert actual.name(10001) == pool.name(10001)
    assert actual.card_id(10002) == 2
    assert actual.cards is not pool.cards and actual.card_ids is not pool.card_ids
    assert actual.cards[10001] is not pool.cards[10001]
    assert actual.cards[10001].setcodes is not pool.cards[10001].setcodes


@pytest.mark.parametrize('change', [
    lambda p: p.cards[10001].setcodes.append(44),
    lambda p: setattr(p.cards[10002], 'attack', 999),
    lambda p: p.cards.pop(10002),
    lambda p: p.card_ids.update({10001:99}),
    lambda p: setattr(p, 'db_path', '/different'),
])
def test_mutating_one_branch_copy_cannot_mutate_parent_or_another_branch(change):
    parent = pool_fixture()
    first, second = _copy_card_pool(parent), _copy_card_pool(parent)
    before = _pool_digest(parent)
    change(first)
    assert _pool_digest(first) != before
    assert _pool_digest(parent) == _pool_digest(second) == before


def test_parent_mutation_cannot_reach_a_retained_branch():
    parent = pool_fixture()
    branch = _copy_card_pool(parent)
    before = _pool_digest(branch)
    parent.cards[10001].setcodes.clear()
    parent.cards[10001].attack = 0
    assert _pool_digest(branch) == before != _pool_digest(parent)


def test_pool_graph_extra_fields_and_cycles_remain_detached():
    parent = pool_fixture()
    parent.alias = parent.cards
    parent.self_reference = parent
    branch = _copy_card_pool(parent)
    assert branch.alias is branch.cards and branch.self_reference is branch
    assert _digest(branch) == _digest(copy.deepcopy(parent))
    assert branch.cards is not parent.cards


def test_nonstandard_pool_preserves_its_existing_deepcopy_hook():
    class CustomPool(SimpleNamespace):
        def __deepcopy__(self, memo):
            return SimpleNamespace(copied=True)
    assert _copy_card_pool(CustomPool()).copied is True
