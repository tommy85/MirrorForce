from collections import Counter
import copy
from fractions import Fraction
import itertools
import math
import random
import json
from pathlib import Path

import pytest

from mirrorforce.agent.search.belief_hand_scope import (
    SCHEMA, PublicHandScope, WorkBudget, HandScopeBudgetExceeded)


def record(size=3, group=(0, 1), lower=((10, 1),), anchors=()):
    return {"schema": SCHEMA, "obs_sha256": "a" * 64, "world_sha256": "b" * 64,
            "hand": [0] * size, "group": list(group), "lower": [list(x) for x in lower],
            "reference_anchors": [list(x) for x in anchors]}


def budget(**kwargs):
    return WorkBudget(deadline=1., clock=lambda: 0., **kwargs)


def oracle(scope, hand):
    return {row for row in set(itertools.permutations(hand))
            if all(row[i] == c for i, c in scope.anchors.items())
            and all(sum(row[i] == c for i in scope.group) >= n for c, n in scope.lower.items())}


def test_actual_failure_geometry_old_group_and_fresh_same_multiset_are_not_equivalent():
    scope = PublicHandScope(record())
    assert scope.feasible_multiset([20, 30, 10])
    for good in ([10, 20, 30], [20, 10, 30]):
        assert scope.check_physical(good)
    with pytest.raises(ValueError, match="outside"):
        scope.check_physical([20, 30, 10])
    assert not scope.feasible_multiset([20, 30, 30])


def test_exact_uniform_physical_permutations_not_first_matching():
    scope = PublicHandScope(record())
    expected = oracle(scope, [10, 20, 30])
    counts = Counter()
    rng = random.Random(2026100404)
    for _ in range(4000):
        draw, proof = scope.sample([10, 20, 30], rng=rng, budget=budget())
        assert proof["support_count"] == "4" and draw in expected
        counts[draw] += 1
    assert set(counts) == expected and all(abs(n - 1000) < 130 for n in counts.values())
    assert scope.sample([10, 20, 30], rng=random.Random(13), budget=budget()) == \
           scope.sample([10, 20, 30], rng=random.Random(13), budget=budget())


def test_native_anchors_and_additional_anchors_are_exact_not_canonical_display_slots():
    value = record(size=4, anchors=((0, 10),))
    value["hand"][3] = 30
    scope = PublicHandScope(value)
    assert dict(scope.minimum) == {10: 1, 30: 1}
    draw, proof = scope.sample([10, 10, 20, 30], rng=random.Random(7), budget=budget())
    assert draw[0] == 10 and draw[3] == 30 and proof["support_count"] == "2"
    value["reference_anchors"].append([3, 20])
    with pytest.raises(ValueError, match="contradict"):
        PublicHandScope(value)


def test_seeded_small_cases_exact_dp_mass_equals_exhaustive_physical_support():
    rng = random.Random(2026100405)
    for _ in range(100):
        size = rng.randint(1, 6)
        hand = [rng.choice([10, 20, 30]) for _ in range(size)]
        group = sorted(rng.sample(range(size), rng.randint(0, size)))
        anchors = sorted((i, hand[i]) for i in rng.sample(range(size), rng.randint(0, size)))
        group_counts = Counter(hand[i] for i in group)
        lower = sorted((code, rng.randint(1, count)) for code, count in group_counts.items())
        scope = PublicHandScope(record(size, group, lower, anchors))
        support = oracle(scope, hand)
        codes, available, minima, table, count = scope.distribution(hand, budget=budget())
        assert count == len(support)
        # Independently derive each arrangement's mass: hypergeometric group
        # allocation, then uniform DISTINCT code permutations in each group.
        for physical in support:
            grouped = Counter(physical[i] for i in scope.free_group)
            allocated = tuple(grouped[c] for c in codes)
            allocation_weight = math.prod(math.comb(n, k) for n, k in zip(available, allocated))
            group_ways = math.factorial(len(scope.free_group)) // math.prod(math.factorial(k) for k in allocated)
            other_ways = math.factorial(len(scope.free_other)) // math.prod(
                math.factorial(n - k) for n, k in zip(available, allocated))
            actual = Fraction(allocation_weight, table[0][len(scope.free_group)] * group_ways * other_ways)
            assert actual == Fraction(1, len(support))
        for seed in range(3):
            assert scope.sample(hand, rng=random.Random(seed), budget=budget())[0] in support


def test_budget_unknown_never_returns_partial_or_draws_rng_before_admission():
    scope, rng = PublicHandScope(record()), random.Random(17)
    before = rng.getstate()
    with pytest.raises(HandScopeBudgetExceeded):
        scope.sample([10, 20, 30], rng=rng, budget=WorkBudget(deadline=0., clock=lambda: 0.))
    assert rng.getstate() == before
    with pytest.raises(HandScopeBudgetExceeded):
        scope.sample([10, 20, 30], rng=rng, budget=budget(max_nodes=1))
    assert rng.getstate() == before


def test_empty_hand_has_one_completion_no_rng_and_immutable_owned_scope():
    value = record(0, (), ())
    scope = PublicHandScope(value)
    rng = random.Random(31)
    before = rng.getstate()
    hand, proof = scope.sample([], rng=rng, budget=budget())
    assert hand == () and proof["support_count"] == "1" and rng.getstate() == before
    value["hand"].append(10)
    scope.to_dict()["hand"].append(20)
    assert scope.size == 0 and scope.to_dict()["hand"] == []
    with pytest.raises(AttributeError): scope.size = 9
    with pytest.raises(TypeError): scope.anchors[0] = 10


@pytest.mark.parametrize("fault", ["truth", "bad-group", "known-group", "short-group", "bool", "extra-anchor"])
def test_public_scope_closed_validation(fault):
    value = record()
    if fault == "truth": value["labels"] = [10]
    elif fault == "bad-group": value["group"] = [9]
    elif fault == "known-group": value["hand"][0] = 10
    elif fault == "short-group": value["lower"] = [[10, 3]]
    elif fault == "bool": value["lower"] = [[10, True]]
    else: value["reference_anchors"] = [[3, 10]]
    with pytest.raises(ValueError): PublicHandScope(value)


def test_scoped_AR_is_a_multiset_quotient_not_a_partial_physical_hand_order():
    from mirrorforce.agent.search.belief_current_law import CurrentPublicLayoutLaw, SCOPED_SCHEMA
    spec = {"schema": SCOPED_SCHEMA, "pools": [[[10, 1], [20, 1], [30, 1]], []],
            "slots": [[2, 0, 0, [10, 20, 30]], [2, 1, 0, [10, 20, 30]], [2, 2, 0, [10, 20, 30]]],
            "lower": [], "categories": [], "any": [], "field_targets": [], "hand_targets": [0, 1, 2],
            "hand_scope": record()}
    law = CurrentPublicLayoutLaw(spec, deadline=1., clock=lambda: 0.)
    # The third AR token is 30, but a physical hand may be [30,10,20].
    assert law.feasible((10, 20, 30))
    assert law.hand_scope.check_physical([30, 10, 20])
    assert law.to_dict() == spec  # derived total-hand constraints do not rewrite the sealed input
    with pytest.raises(ValueError, match="outside"):
        law.hand_scope.check_physical([20, 30, 10])
    broken = copy.deepcopy(spec)
    broken["lower"].append([[0, 1], 10, 1])
    with pytest.raises(ValueError, match="exchangeable"):
        CurrentPublicLayoutLaw(broken, deadline=1., clock=lambda: 0.)


def test_scoped_field_prefix_cannot_spend_a_required_hand_identity_elsewhere():
    from mirrorforce.agent.search.belief_current_law import CurrentPublicLayoutLaw, SCOPED_SCHEMA
    spec = {"schema": SCOPED_SCHEMA, "pools": [[[10, 1], [20, 2]], []],
            "slots": [[8, 0, 0, [10, 20]], [2, 0, 0, [10, 20]], [1, 0, 0, [10, 20]]],
            "lower": [], "categories": [], "any": [], "field_targets": [0], "hand_targets": [1],
            "hand_scope": record(size=1, group=(0,))}
    law = CurrentPublicLayoutLaw(spec, deadline=1., clock=lambda: 0.)
    assert law.mask(()).tolist() == [False, True]
    assert law.mask((20,)).tolist() == [True, False]


def test_canonical_known_hand_slot_is_not_mistaken_for_a_physical_anchor():
    from mirrorforce.agent.search.belief_current_law import CurrentPublicLayoutLaw, SCOPED_SCHEMA
    scope = record(size=2, group=(), lower=())
    scope["hand"] = [0, 10]  # true public physical anchor is slot1
    spec = {"schema": SCOPED_SCHEMA, "pools": [[[10, 1], [20, 1]], []],
            "slots": [[2, 0, 0, [10]], [2, 1, 0, [10, 20]]],
            "lower": [], "categories": [], "any": [], "field_targets": [], "hand_targets": [1],
            "hand_scope": scope}
    law = CurrentPublicLayoutLaw(spec, deadline=1., clock=lambda: 0.)
    layout = {"hand": [10, 20], "deck": [], "extra": [], "facedown": []}
    assert law.check_complete_layout(layout, (20,))
    physical, proof = law.hand_scope.sample(layout["hand"], rng=random.Random(1), budget=budget())
    assert physical == (20, 10) and proof["support_count"] == "1"


def test_original_real_roots_255_and_270_intersect_native_hand_without_replacing_field_evidence():
    import time
    from mirrorforce.agent.search.belief_hand_scope import from_world, digest
    from mirrorforce.agent.public_world_codec import world_sha256
    from mirrorforce.agent.search.belief_current_law import CurrentPublicLayoutLaw, SCOPED_SCHEMA
    fixture = json.loads((Path(__file__).parent/'fixtures/ar-current-public-hand-group-20261004.json').read_text())
    assert fixture['private_truth_in_fixture'] is False
    X, G, E = 99550630, 67441435, 63166095
    for case in fixture['cases']:
        original = case['reference_specification']
        if case['prompt'] == 255:
            assert digest(original) == case['original_specification_sha256']
            assert case['original_specification_hash_matched']
        scope = from_world(case['native_public_world'], obs_sha256=case['obs_sha256'],
                           world_sha256=world_sha256(case['native_public_world']))
        specification = {**copy.deepcopy(original), 'schema': SCOPED_SCHEMA, 'hand_scope': scope}
        law = CurrentPublicLayoutLaw(specification, deadline=time.monotonic()+30)
        assert law.to_dict() == specification
        assert {k:v for k,v in specification.items() if k not in ('schema','hand_scope')} == \
               {k:v for k,v in original.items() if k != 'schema'}
        # All original field categories, shuffled-group facts, existence facts
        # and deck/extra inventories remain; World is not their replacement.
        target_hand = sorted([G,E] if case['prompt'] == 255 else [G,X])
        domains = [original['slots'][i][3] for i in original['field_targets']]
        sequence = next(tuple(fields)+tuple(target_hand) for fields in itertools.product(*domains)
                        if law.feasible(tuple(fields)+tuple(target_hand)))
        assert law.feasible(sequence)
        physical = PublicHandScope(scope)
        if case['prompt'] == 255:
            assert physical.check_physical([X,G,E]) and physical.check_physical([G,X,E])
            with pytest.raises(ValueError, match='outside'):
                physical.check_physical([G,E,X])
            # Changing a hypothetical private arrangement cannot change any
            # sealed public-law or candidate input field.
            before = digest(specification)
            for truth in ([X,G,E], [G,X,E]): physical.check_physical(truth)
            assert digest(specification) == before
        else:
            old = CurrentPublicLayoutLaw(original, deadline=time.monotonic()+30)
            bad = next(tuple(fields)+tuple(sorted([G,E])) for fields in itertools.product(*domains)
                       if old.feasible(tuple(fields)+tuple(sorted([G,E]))))
            assert old.feasible(bad) and not law.feasible(bad)
            assert not physical.feasible_multiset([G,E])
