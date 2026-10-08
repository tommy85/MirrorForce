from collections import Counter
import itertools
import random

import numpy as np
import pytest

from mirrorforce.agent.search import belief_current_law as L


def base():
    return {'schema': L.SCHEMA, 'pools': [[[10, 2], [20, 2], [30, 2]], []],
            'slots': [[8, 0, 0, [10, 20, 30]], [8, 1, 0, [10, 20, 30]],
                      [2, 0, 0, [10, 20, 30]], [2, 1, 0, [10, 20, 30]],
                      [1, 0, 0, [10, 20, 30]], [1, 1, 0, [10, 20, 30]]],
            'lower': [], 'categories': [], 'any': [], 'field_targets': [0, 1], 'hand_targets': [2, 3]}


def law(spec=None, **kwargs):
    return L.CurrentPublicLayoutLaw(spec or base(), deadline=1., clock=lambda: 0., max_nodes=1000000, **kwargs)


def honored(spec, values):
    if any(code not in row[3] for code, row in zip(values, spec['slots'])): return False
    if any(sum(values[i] == code for i in scope) < count for scope, code, count in spec['lower']): return False
    if any(not any(values[i] in codes for i in scope) for scope, codes in spec['any']): return False
    # Independent exhaustive distinct-witness test (not the production matcher).
    def categories(index, used):
        if index == len(spec['categories']): return True
        scope, codes = spec['categories'][index]
        return any(categories(index + 1, used | {i}) for i in scope if i not in used and values[i] in codes)
    return categories(0, set())


def support(spec):
    cards = [code for pool in spec['pools'] for code, count in pool for _ in range(count)]
    indices = spec['field_targets'] + spec['hand_targets']
    out = set()
    for values in set(itertools.permutations(cards)):
        hand = tuple(values[i] for i in spec['hand_targets'])
        if hand == tuple(sorted(hand)) and honored(spec, values):
            out.add(tuple(values[i] for i in indices))
    return out


def assert_support(spec):
    expected = support(spec)
    if not expected:
        with pytest.raises(ValueError, match='no complete'):
            law(spec)
        return
    instance = law(spec)
    pending, actual = [()], set()
    while pending:
        prefix = pending.pop()
        if len(prefix) == instance.length:
            instance.layout(prefix)
            actual.add(prefix)
        else:
            for code, allowed in zip(instance.codes, instance.mask(prefix)):
                if allowed: pending.append((*prefix, code))
    assert actual == expected


def test_overlapping_identity_and_categories_can_share_a_card_but_two_categories_cannot():
    spec = base()
    spec['lower'] = [[[0, 1], 10, 1]]
    spec['categories'] = [[[0, 1], [10]], [[0, 1], [10, 20]]]
    assert_support(spec)
    instance = law(spec)
    assert instance.feasible((10, 20))
    assert not instance.feasible((10, 30))
    assert instance.feasible((10, 10))


def test_fresh_slot_cannot_discharge_old_position_free_identity():
    spec = base()
    spec['lower'] = [[[0], 10, 1]]  # slot 1 is fresh, not in the witness scope
    instance = law(spec)
    assert not instance.feasible((20, 10)) and instance.feasible((10, 20))
    assert_support(spec)


def test_shuffled_group_has_exact_multiset_not_only_allowed_code_sets():
    spec = base()
    spec['slots'][0][3] = spec['slots'][1][3] = [10, 20]
    spec['lower'] = [[[0, 1], 10, 1], [[0, 1], 20, 1]]
    instance = law(spec)
    assert not instance.feasible((10, 10)) and instance.feasible((20, 10))
    assert_support(spec)


def test_zone_or_claims_share_witnesses_and_include_unpredicted_deck():
    spec = base()
    spec['any'] = [[[2, 3, 4, 5], [10]], [[4, 5], [10, 20]], [[4, 5], [10]]]
    instance = law(spec)
    assert not instance.feasible((10, 10))
    assert instance.feasible((10, 20, 20, 30))
    assert_support(spec)


def test_alternative_category_matchings_are_not_greedily_discarded():
    spec = base()
    spec['categories'] = [[[0, 1], [10, 20]], [[0, 1], [10]]]
    spec['slots'][0][3], spec['slots'][1][3] = [10], [20]
    assert_support(spec)
    instance = law(spec)
    prefix = (10, 20, 10, 20)
    completed = instance.complete(prefix, np.random.default_rng(0))
    assert instance.check_complete_layout(completed, prefix)
    spec['categories'].reverse()
    assert law(spec).check_complete_layout(completed, prefix)


@pytest.mark.parametrize('categories', [
    [[[0, 1], [10, 20]], [[0, 1], [10]]],
    [[[0, 1, 4, 5], [10]], [[0, 1], [10, 20]], [[4, 5], [20, 30]]],
    [[[0, 1], [10, 20, 30]]] * 3,
])
def test_complete_checker_matches_independent_exhaustive_category_assignments(categories):
    spec = base()
    spec['categories'] = categories
    instance = law(spec, require_feasible=False)
    for values in set(itertools.permutations([10, 10, 20, 20, 30, 30])):
        if values[2] > values[3]:
            continue
        completed = {'hand': list(values[2:4]), 'deck': list(values[4:6]), 'extra': [],
                     'facedown': [[8, 0, values[0]], [8, 1, values[1]]]}
        if honored(spec, values):
            assert instance.check_complete_layout(completed, values[:4])
        else:
            with pytest.raises(ValueError, match='distinct public category witnesses'):
                instance.check_complete_layout(completed, values[:4])


def test_seeded_random_constraints_match_exhaustive_physical_assignments():
    rng = random.Random(501099)
    def random_scope():
        picked = set(rng.sample(range(6), rng.randint(1, 6)))
        if picked & {2, 3}: picked |= {2, 3}
        return sorted(picked)
    for _ in range(60):
        spec = base()
        for row in spec['slots']:
            row[3] = sorted(rng.sample([10, 20, 30], rng.randint(1, 3)))
        spec['slots'][3][3] = spec['slots'][2][3][:]
        for _ in range(rng.randrange(3)):
            spec['lower'].append([random_scope(),
                                  rng.choice([10, 20, 30]), rng.randint(1, 2)])
        for name in ('categories', 'any'):
            for _ in range(rng.randrange(4)):
                spec[name].append([random_scope(),
                                   sorted(rng.sample([10, 20, 30], rng.randint(1, 3)))])
        assert_support(spec)


def test_complete_samples_every_residual_slot_preserves_pools_and_all_constraints():
    spec = base()
    spec['pools'][1] = [[100, 1], [200, 1]]
    spec['slots'] += [[32, 0, 1, [100, 200]], [64, 0, 1, [100, 200]]]
    spec['any'] = [[[4, 5, 7], [10, 100]]]
    spec['lower'] = [[[0, 1], 10, 1]]
    instance = law(spec)
    prefix = (10, 20, 20, 30)
    results = []
    for seed in range(8):
        result = instance.complete(prefix, np.random.Generator(np.random.PCG64(seed)))
        assert instance.check_complete_layout(result, prefix)
        positions = {(2, i): c for i, c in enumerate(result['hand'])}
        positions.update({(1, i): c for i, c in enumerate(result['deck'])})
        positions.update({(64, i): c for i, c in enumerate(result['extra'])})
        positions.update({(loc, seq): c for loc, seq, c in result['facedown']})
        values = tuple(positions[(row[0], row[1])] for row in spec['slots'])
        assert honored(spec, values) and Counter(values) == Counter({10: 2, 20: 2, 30: 2, 100: 1, 200: 1})
        results.append(result)
    assert len({str(row) for row in results}) > 1
    assert instance.complete(prefix, np.random.default_rng(0)) == results[0]


def test_complete_layout_checker_rejects_residual_tampering_and_missing_slots():
    spec = base()
    spec['pools'][1] = [[100, 1], [200, 1]]
    spec['slots'] += [[32, 0, 1, [100, 200]], [64, 0, 1, [100, 200]]]
    spec['lower'] = [[[0, 1], 10, 1]]
    spec['categories'] = [[[2, 3], [10]], [[0, 1], [20, 30]]]
    instance = law(spec)
    prefix = (10, 20, 10, 20)
    layout = instance.complete(prefix, np.random.default_rng(0))
    assert instance.check_complete_layout(layout, prefix)
    broken = {**layout, 'deck': layout['deck'][:-1]}
    with pytest.raises(ValueError, match='fill'):
        instance.check_complete_layout(broken, prefix)
    broken = {**layout, 'hand': [30, 30]}
    with pytest.raises(ValueError, match='multiset|lower bound'):
        instance.check_complete_layout(broken, prefix)
    broken = {**layout, 'hidden_hand': [10]}
    with pytest.raises(ValueError, match='fields'):
        instance.check_complete_layout(broken, prefix)


def test_law_owns_specification_and_returned_masks():
    spec = base()
    instance = law(spec)
    expected = instance.mask(())
    spec['slots'][0][3] = [10]
    instance.to_dict()['slots'][0][3] = [10]
    expected[:] = False
    assert instance.mask(()).all()


def test_deadline_and_node_budget_are_unknown_not_false_feasibility():
    with pytest.raises(L.CurrentLawBudgetExceeded):
        L.CurrentPublicLayoutLaw(base(), deadline=0., clock=lambda: 0.)
    instance = law()
    instance.deadline = 0.
    with pytest.raises(L.CurrentLawBudgetExceeded): instance.mask(())
    instance = law()
    instance.max_nodes = instance.nodes
    with pytest.raises(L.CurrentLawBudgetExceeded): instance.feasible(())


@pytest.mark.parametrize('fault', ['truth', 'pool-short', 'duplicate', 'bool-count', 'wrong-pool',
                                 'bad-domain', 'bad-scope', 'unsorted-field', 'overlap-targets'])
def test_closed_public_input_validation(fault):
    spec = base()
    if fault == 'truth': spec['truth'] = [10]
    elif fault == 'pool-short': spec['pools'][0][0][1] = 1
    elif fault == 'duplicate': spec['slots'][1][:2] = spec['slots'][0][:2]
    elif fault == 'bool-count': spec['pools'][0][0][1] = True
    elif fault == 'wrong-pool': spec['slots'][0][2] = 1
    elif fault == 'bad-domain': spec['slots'][0][3] = [999]
    elif fault == 'bad-scope': spec['categories'] = [[[9], [10]]]
    elif fault == 'unsorted-field': spec['field_targets'] = [1, 0]
    else: spec['hand_targets'] = [0, 2]
    with pytest.raises(ValueError): law(spec)


def test_sorted_hand_boundary_cannot_change_target_geometry_or_prefix_semantics():
    instance = law()
    with pytest.raises(ValueError, match='sorted'): instance.feasible((10, 20, 30, 20))
    with pytest.raises(ValueError, match='vocabulary'): instance.feasible((999,))
    with pytest.raises(ValueError, match='no next'): instance.mask((10, 20, 20, 30))


@pytest.mark.parametrize('kind', ['domain', 'witness'])
def test_nonexchangeable_hand_cannot_be_silently_sorted(kind):
    spec = base()
    if kind == 'domain': spec['slots'][2][3] = [10]
    else: spec['categories'] = [[[2], [10]]]
    with pytest.raises(ValueError, match='exchangeable'): law(spec)
