"""Exhaustive tiny-world proofs for the independent AR public mask, not a production strength evaluation."""
from itertools import permutations
import math

import numpy as np
import pytest

from mirrorforce.agent.search.belief_ar_law import PublicLayoutLaw, masked_log_probabilities, sample, transport_feasible


def world():
    return {"decklist_public": True, "hand": [0, 0], "hand_group": [0], "deck": [0], "extra": [],
            "facedown": [(4, 0, 0, False)], "pool_main": {11: 2, 22: 1, 33: 1}, "pool_extra": {},
            "unpositioned": {(2, 11): 1}, "own_deck": [], "own_deck_fixed": {},
            "types": {11: 1, 22: 1, 33: 4}}


def enumerate_sequences(law, prefix=()):
    if len(prefix) == law.length:
        return [prefix]
    return [sequence for code, valid in zip(law.codes, law.mask(prefix)) if valid
            for sequence in enumerate_sequences(law, (*prefix, code))]


def test_every_mask_matches_all_complete_public_assignments():
    law = PublicLayoutLaw(world())
    # Slots: field monster, shuffled hand group with known 11, later hand, deck.
    assignments = [p for p in set(permutations([11, 11, 22, 33])) if p[0] in (11, 22) and p[1] == 11]
    canonical = {(p[0], *sorted(p[1:3])) for p in assignments}
    assert set(enumerate_sequences(law)) == canonical
    for sequence in canonical:
        assert law.layout(sequence)["hidden_hand"] == list(sequence[1:])
        for n in range(law.length):
            prefix = sequence[:n]
            expected = {s[n] for s in canonical if s[:n] == prefix}
            assert {code for code, ok in zip(law.codes, law.mask(prefix)) if ok} == expected


def test_future_category_constraint_blocks_a_locally_available_card():
    w = dict(world(), hand=[0], hand_group=[], deck=[], unpositioned={}, pool_main={11: 1, 22: 1, 33: 1},
             facedown=[(4, 0, 0, False), (4, 1, 0, False)])
    law = PublicLayoutLaw(w, field_allowed={(4, 1): [11]})
    # Taking 11 in the first monster slot would strand the later public 11-only slot.
    np.testing.assert_array_equal(law.mask(()), [False, True, False])
    np.testing.assert_array_equal(law.mask((22,)), [True, False, False])
    assert enumerate_sequences(law) == [(22, 11, 33)]


def test_sorted_hand_looks_ahead_to_known_lower_identity():
    w = dict(world(), hand=[0, 0], hand_group=[0, 1], facedown=[], pool_main={11: 1, 22: 1, 33: 1})
    law = PublicLayoutLaw(w)
    np.testing.assert_array_equal(law.mask(()), [True, False, False])
    assert set(enumerate_sequences(law)) == {(11, 22), (11, 33)}
    with pytest.raises(ValueError, match="nondecreasing"):
        law.mask((22, 11))


def test_residual_deck_lower_bound_is_not_consumed_by_hand():
    w = dict(world(), hand=[0], hand_group=[], facedown=[], deck=[0], pool_main={11: 1, 22: 1},
             unpositioned={(1, 11): 1})
    law = PublicLayoutLaw(w)
    np.testing.assert_array_equal(law.mask(()), [False, True])


def test_extra_facedown_and_residual_extra_are_separate_pool():
    w = dict(world(), hand=[], hand_group=[], deck=[0], pool_main={11: 1}, pool_extra={44: 1, 55: 1}, extra=[0],
             facedown=[(4, 2, 0, True)], unpositioned={(64, 55): 1}, types={11: 1, 44: 65, 55: 65})
    law = PublicLayoutLaw(w)
    assert enumerate_sequences(law) == [(44,)]
    assert law.layout((44,))["facedown"] == [[4, 2, 44]]


def test_banished_cards_are_unpredicted_but_still_consume_the_pool():
    w = dict(world(), hand=[0], hand_group=[], deck=[], facedown=[(32, 0, 0, False)],
             pool_main={11: 1, 22: 1}, unpositioned={})
    law = PublicLayoutLaw(w)
    assert law.length == 1 and set(enumerate_sequences(law)) == {(11,), (22,)}


def test_visible_hand_slots_are_fixed_not_regenerated_and_shuffled_group_capacity_checked():
    w = dict(world(), hand=[99, 0], hand_group=[1], deck=[0], facedown=[], pool_main={11: 1, 22: 1})
    assert enumerate_sequences(PublicLayoutLaw(w)) == [(11,)]
    with pytest.raises(ValueError, match="subgroup capacity"):
        PublicLayoutLaw(dict(w, hand_group=[], unpositioned={(2, 11): 1}))


def test_two_hidden_truths_have_identical_masks_and_samples():
    # Truth is deliberately not a constructor argument. Two complete legal targets behind this public evidence
    # change neither the public object nor a seeded generation (targets are used only to check support here).
    w, law = world(), PublicLayoutLaw(world())
    truths = enumerate_sequences(law)[:2]
    assert len(truths) == 2 and truths[0] != truths[1]
    for truth in truths:
        assert law.feasible(truth)
        other = PublicLayoutLaw(w)
        for prefix in [(), truths[0][:1]]:
            np.testing.assert_array_equal(law.mask(prefix), other.mask(prefix))
        assert sample(law, lambda p: [0., 0.5, -0.3], seed=7) == sample(other, lambda p: [0., 0.5, -0.3], seed=7)
    with pytest.raises(ValueError, match="unknown fields"):
        PublicLayoutLaw(dict(w, truth=truths[0]))


def test_joint_probabilities_normalize_over_unique_sequences():
    law = PublicLayoutLaw(world())
    total = 0.
    for sequence in enumerate_sequences(law):
        logp = 0.
        for step, code in enumerate(sequence):
            logits = np.asarray([0.3, -0.2, 0.7]) + step * np.asarray([0., 0.2, -0.2])
            logp += masked_log_probabilities(logits, law.mask(sequence[:step]))[law.index[code]]
        total += math.exp(logp)
    assert total == pytest.approx(1.)
    result = sample(law, lambda p: np.zeros(3), seed=10)
    expected = sum(masked_log_probabilities(np.zeros(3), law.mask(result["sequence"][:i]))[law.index[code]]
                   for i, code in enumerate(result["sequence"]))
    assert result["log_probability"] == expected


def test_empty_layout_and_impossible_public_world():
    w = dict(world(), hand=[], hand_group=[], deck=[0, 0, 0, 0], facedown=[], unpositioned={})
    law = PublicLayoutLaw(w)
    assert sample(law, lambda p: pytest.fail("empty layout must not query the network"), seed=2)["log_probability"] == 0
    with pytest.raises(ValueError, match="no complete"):
        PublicLayoutLaw(dict(w, pool_main={11: 5}))
    with pytest.raises(ValueError, match="outside the known pools"):
        PublicLayoutLaw(dict(w, unpositioned={(1, 88): 1}))
    entirely_visible = PublicLayoutLaw(dict(w, deck=[], pool_main={}, types={}))
    assert entirely_visible.layout(())["hidden_hand"] == []


def test_flow_hall_violation_and_lower_bound_are_real_checks():
    # Both required groups accept only the same one available card, although every group has a local candidate.
    assert not transport_feasible(np.array([1, 1]), np.array([1, 1]), np.zeros((2, 2), int),
                                  np.array([[1, 1], [0, 0]]))
    assert transport_feasible(np.array([1, 1]), np.array([1, 1]), np.array([[0, 1], [0, 0]]),
                              np.ones((2, 2), int))


@pytest.mark.parametrize("logits,mask", [([0, float("nan")], [True, True]), ([0, 0], [False, False]),
                                        ([0], [True, True])])
def test_bad_logits_or_empty_mask_is_not_a_uniform_fallback(logits, mask):
    with pytest.raises(ValueError):
        masked_log_probabilities(logits, np.asarray(mask))


@pytest.mark.parametrize("change", ["float_lower", "bool_lower", "negative_lower", "lower_location", "bool_location",
                                    "float_group", "duplicate_group", "outside_group", "float_slot", "bool_slot",
                                    "float_fixed", "float_category"])
def test_public_integer_evidence_is_rejected_not_coerced(change):
    w = world()
    extra = {}
    if change in ("float_lower", "bool_lower", "negative_lower"):
        w["unpositioned"] = {(2, 11): {"float_lower": 1.5, "bool_lower": True, "negative_lower": -1}[change]}
    elif change in ("lower_location", "bool_location"):
        w["unpositioned"] = {(4 if change == "lower_location" else True, 11): 1}
    elif change in ("float_group", "duplicate_group", "outside_group"):
        w["hand_group"] = {"float_group": [0.5], "duplicate_group": [0, 0], "outside_group": [2]}[change]
    elif change == "float_slot":
        w["facedown"] = [(4, 0.5, 0, False)]
    elif change == "bool_slot":
        w["hand"] = [False, 0]
    elif change == "float_fixed":
        w["own_deck"], w["own_deck_fixed"] = [11], [(0.5, 11)]
    else:
        extra["field_allowed"] = {(4.0, 0): [11, 22]}
    with pytest.raises(ValueError):
        PublicLayoutLaw(w, **extra)


def test_transport_does_not_truncate_fractional_bounds():
    with pytest.raises(ValueError, match="integer"):
        transport_feasible(np.array([1]), np.array([1]), np.array([[0.5]]), np.array([[1]]))
