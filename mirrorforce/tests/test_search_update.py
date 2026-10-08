"""The root step and leaf values of update-equivalence search (mirrorforce/agent/search/update.py), against Ataraxos'
formulas (pyengine/core/search.py: estimate_q_values' TD(lambda) leaf weights, compute_search_policy)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from mirrorforce.agent.search import update as U


@pytest.mark.parametrize("td_lambda", [1.0, 0.9, 0.5, 0.0])
@pytest.mark.parametrize("depth", [1, 3, 5])
def test_the_weights_of_every_path_sum_to_one(depth, td_lambda):
    # cut at the leaf: values at own decisions 0 .. depth-1
    assert math.isclose(sum(U.value_weight(i, depth, td_lambda) for i in range(depth)), 1.0)
    # a game over after t own decisions (t < depth): values at 0 .. t-1, then the outcome
    for t in range(depth):
        total = sum(U.value_weight(i, depth, td_lambda) for i in range(t)) + U.outcome_weight(t, td_lambda)
        assert math.isclose(total, 1.0), (t, total)


def test_the_weights_are_ataraxos_td_lambda():
    """estimate_q_values: normalizer (1 - l) l^d before the last pair, l^d at it; terminal reward l^d; l = 1 only the
    last value or the reward."""
    assert U.value_weight(0, 5, 0.8) == pytest.approx(0.2)
    assert U.value_weight(2, 5, 0.8) == pytest.approx(0.2 * 0.64)
    assert U.value_weight(4, 5, 0.8) == pytest.approx(0.8 ** 4)
    assert U.outcome_weight(3, 0.8) == pytest.approx(0.8 ** 3)
    assert [U.value_weight(i, 5, 1.0) for i in range(5)] == [0, 0, 0, 0, 1]
    assert U.outcome_weight(3, 1.0) == 1.0
    with pytest.raises(ValueError):
        U.value_weight(5, 5, 0.8)


def test_leaf_values_and_q_from_continuations():
    leaves = U.LeafValues(depth=2, td_lambda=0.5)
    # continuation (0, 0): values 0.2 at own 0, 0.6 at own 1 (the leaf, cut)
    leaves.add_value(0, 0, 0, 0.2)
    leaves.add_value(0, 0, 1, 0.6)
    a = leaves.add_result({"root": 0, "continuation": 0, "error": "", "truncated": True, "own_decisions": 2,
                           "winner": -1, "root_player": 1})
    assert a == pytest.approx(0.5 * 0.2 + 0.5 * 0.6)
    # continuation (0, 1): value -0.4 at own 0, then the root player (1) wins after 1 own decision
    leaves.add_value(0, 1, 0, -0.4)
    b = leaves.add_result({"root": 0, "continuation": 1, "error": "", "truncated": False, "own_decisions": 1,
                           "winner": 1, "root_player": 1})
    assert b == pytest.approx(0.5 * -0.4 + 0.5 * 1.0)
    # an env error ends a continuation without a leaf
    leaves.add_value(0, 2, 0, 0.9)
    assert leaves.add_result({"root": 0, "continuation": 2, "error": "core script error", "truncated": False,
                              "own_decisions": 1, "winner": -1, "root_player": 1}) is None
    assert leaves.errors == 1
    q, counts = U.q_values([(0, a), (1, b), (1, -b)], legal=3)
    assert q.tolist() == pytest.approx([a, 0.0, 0.0]) and counts.tolist() == [1, 2, 0]
    assert U.outcome(2, 0) == 0.0 and U.outcome(0, 1) == -1.0


def test_the_search_policy_is_the_ataraxos_mirror_descent_step():
    logits = np.log(np.array([0.5, 0.3, 0.2]))
    q = np.array([0.1, -0.2, 0.4])
    eta, tau = 10.0, 0.006
    magnet = np.log(np.array([0.25, 0.25, 0.5]))
    z = (logits + eta * q + eta * tau * magnet) / (1 + eta * tau)
    expected = np.exp(z) / np.exp(z).sum()
    assert U.search_policy(q, logits, eta, tau, magnet) == pytest.approx(expected)
    uniform = (logits + eta * q + eta * tau * np.log(1 / 3)) / (1 + eta * tau)
    assert U.search_policy(q, logits, eta, tau) == pytest.approx(np.exp(uniform) / np.exp(uniform).sum())
    variant = (logits + eta * q) / (1 + tau * eta)
    assert U.search_policy(q, logits, eta, tau, uniform_magnet=True) == pytest.approx(
        np.exp(variant) / np.exp(variant).sum())
    # stepsize 0: the policy itself (magnet term vanishes with it)
    assert U.search_policy(q, logits, 0.0, tau) == pytest.approx(np.exp(logits) / np.exp(logits).sum())
    assert U.balanced_actions(3, 2) == [0, 0, 1, 1, 2, 2]
