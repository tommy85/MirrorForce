"""Position-free field identities, joint constraints and exact probabilities."""
from collections import Counter
from dataclasses import replace
from itertools import permutations
import random

import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.netduel.disclosure import CategoryConstraint
from mirrorforce.search.belief import BeliefError, BeliefSampler, Evidence, evidence_from_snapshot
from mirrorforce.search.field_assignment import FieldAssignmentError, FieldAssignmentSampler
from mirrorforce.search.particles import HiddenLayout, particles_from_evidence
from mirrorforce.worldmodel.state import CardState, StateSnapshot


A, B, X = 52340444, 63166095, 100
S, M, H = C.LOCATION_SZONE, C.LOCATION_MZONE, C.LOCATION_HAND


def test_the_partial_shuffle_shape_from_production_is_sampled_without_inventing_a_position():
    pool = Counter({A: 2, B: 1, X: 4})
    cards = [CardState(1, S, seq, code=0, hidden=True, position=C.POS_FACEDOWN_DEFENSE)
             for seq in (1, 3, 4)]
    cards += [CardState(1, H, seq, code=0, hidden=True) for seq in range(2)]
    cards += [CardState(1, C.LOCATION_DECK, seq, code=0, hidden=True) for seq in range(2)]
    public = StateSnapshot(turn=4, turn_player=0, phase=C.PHASE_MAIN1, lp=(8000, 8000),
                           cards=cards, counts={}, view=0, revealed_unpositioned=((1, S, A),))
    claim = CategoryConstraint(S, 4, frozenset({A, B}), 0)
    evidence = evidence_from_snapshot(public, player=1, deck_list=pool, categories=(claim,))
    assert evidence.unexpressed_field_knowledge == 0
    assert evidence.shuffled_facedown == ()
    shape = HiddenLayout(hand=(0, 0), deck=(0, 0), facedown=tuple((S, seq, 0) for seq in (1, 3, 4)))
    particles = particles_from_evidence(evidence, 200, seed=47, truth=shape, shape_only=True)
    positions = set()
    for _, layout, _ in particles:
        field = {seq: code for _, seq, code in layout.facedown}
        assert A in field.values()
        assert field[4] in {A, B}
        assert layout.multiset() == pool
        positions.update(seq for seq, code in field.items() if code == A)
    assert positions == {1, 3, 4}
    assert particles_from_evidence(evidence, 200, seed=47, truth=shape, shape_only=True) == particles


def test_hand_and_field_draws_compete_for_the_same_card():
    evidence = Evidence(deck_list=Counter({A: 1, B: 1}), hand_size=1, facedown_slots=1,
                        facedown_slot_keys=((S, 4),), facedown_sampling_keys=((S, 4),),
                        hand_categories=(frozenset({A}),), unpositioned_facedown=((S, A),))
    with pytest.raises(BeliefError, match="no joint assignment"):
        BeliefSampler(evidence)
    sampler = BeliefSampler(replace(evidence, hand_categories=(frozenset({B}),)), random.Random(17))
    for _ in range(5):
        particle = sampler.sample()
        assert particle.hand == (B,) and particle.facedown == (A,)
    sampler.clear()


def test_zone_identity_lower_bounds_do_not_migrate_between_monsters_and_backrow():
    evidence = Evidence(deck_list=Counter({A: 2, B: 1, X: 2}), hand_size=1, deck_size=1,
                        facedown_slots=3, facedown_slot_keys=((S, 1), (M, 0), (S, 4)),
                        facedown_sampling_keys=((S, 1), (M, 0), (S, 4)),
                        unpositioned_facedown=((M, A), (S, A), (S, B)))
    shape = HiddenLayout(hand=(0,), deck=(0,), facedown=((S, 1, 0), (M, 0, 0), (S, 4, 0)))
    for _, layout, _ in particles_from_evidence(evidence, 20, seed=5, truth=shape, shape_only=True):
        assert layout.facedown[1] == (M, 0, A)
        assert Counter(code for loc, _, code in layout.facedown if loc == S) == Counter({A: 1, B: 1})
        assert layout.hand == (X,) and layout.deck == (X,)


def test_anonymous_category_claims_need_distinct_witnesses_but_may_share_identity_facts():
    with pytest.raises(FieldAssignmentError):
        FieldAssignmentSampler(Counter({A: 1, X: 1}), ((S, 0, None),),
                               categories=((S, {A, X}), (S, {A})))
    sampler = FieldAssignmentSampler(Counter({A: 1, X: 1}), ((S, 0, None),),
                                     identities=((S, A),), categories=((S, {A, B}),))
    assert sampler.sample(random.Random(5)) == ((A,), Counter({X: 1}))
    # An already public field identity can witness a category, but cannot
    # stand in for an additional position-free known copy.
    sampler = FieldAssignmentSampler(Counter({X: 1}), ((S, 0, None),),
                                     categories=((S, {A}),), fixed=((S, A),))
    assert sampler.sample(random.Random(5))[0] == (X,)
    with pytest.raises(FieldAssignmentError):
        FieldAssignmentSampler(Counter({X: 1}), ((S, 0, None),),
                               identities=((S, A),), fixed=((S, A),))


def test_completion_counts_and_sampling_match_an_exhaustive_physical_card_oracle():
    cards = (A, A, B, X, X)
    slots = ((S, 0, None), (S, 4, frozenset({A, B})), (H, 0, frozenset({A, X})))
    sampler = FieldAssignmentSampler(Counter(cards), slots, identities=((S, A),),
                                     categories=((S, {A, B}), (S, {A, X})))
    expected = Counter()
    for indices in permutations(range(len(cards)), len(slots)):
        picked = tuple(cards[i] for i in indices)
        if picked[1] not in {A, B} or picked[2] not in {A, X} or A not in picked[:2]:
            continue
        if not ((picked[0] in {A, B} and picked[1] in {A, X})
                or (picked[1] in {A, B} and picked[0] in {A, X})):
            continue
        expected[picked] += 1
    total = sampler._ways(0, sampler.initial_counts, sampler.initial_required, sampler.initial_matches)
    assert total == sum(expected.values())
    rng = random.Random(29)
    observed = Counter(sampler.sample(rng)[0] for _ in range(12000))
    assert set(observed) == set(expected)
    for assignment, count in expected.items():
        assert abs(observed[assignment] / 12000 - count / total) < 0.015
    sampler.clear()


def test_independent_hidden_truths_do_not_change_public_field_particles():
    evidence = Evidence(deck_list=Counter({A: 1, B: 1, X: 1}), hand_size=1, deck_size=1,
                        facedown_slots=1, facedown_slot_keys=((S, 4),), facedown_sampling_keys=((S, 4),),
                        unpositioned_facedown=((S, A),))
    first = HiddenLayout(hand=(X,), deck=(B,), facedown=((S, 4, A),))
    second = HiddenLayout(hand=(B,), deck=(X,), facedown=((S, 4, A),))
    assert particles_from_evidence(evidence, 20, seed=13, truth=first) == particles_from_evidence(
        evidence, 20, seed=13, truth=second)
