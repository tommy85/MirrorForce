"""Arm A's recipe (mirrorforce/agent/train/ataraxos.py) against hand-computed values and the reference formulas.

Reference: ataraxos/stratego @ 92db29e (rl.py power_schedule and train(), buffer.py add_post_act/process_data).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mirrorforce.agent.train import ataraxos as A  # noqa: E402

CFG = A.AtaraxosConfig()
UNIFORM = A.AtaraxosConfig(magnet="uniform_legal")  # the default: the released code's uniform_magnet=True
GROUPED = A.AtaraxosConfig(magnet="magnet_card_grouped/v1")  # the paper's eq. 6 analogue (registered alternative)


def test_power_schedules_match_the_reference_formula():
    for step in (0, 10, 2300, 35000, 42400):
        ref = min(max(0.5 / (step + 1) ** 1.1, 5e-6), 1e-4)
        assert float(A.power_schedule(0.5, step, 1.1, 1e-4, 5e-6)) == pytest.approx(ref, rel=1e-6)
    # their lr leaves the 1e-4 ceiling near iteration 2.3k and reaches the floor near 35k
    assert float(A.power_schedule(0.5, 2000, 1.1, 1e-4, 5e-6)) == pytest.approx(1e-4)
    assert float(A.power_schedule(0.5, 3000, 1.1, 1e-4, 5e-6)) < 1e-4
    assert float(A.power_schedule(0.5, 40000, 1.1, 1e-4, 5e-6)) == pytest.approx(5e-6)
    # temperature: about 0.002 at their final iteration
    assert float(A.power_schedule(0.05, 42400, 0.3, 0.1, 0.001)) == pytest.approx(0.05 / 42401 ** 0.3, rel=1e-6)
    # our steps are samples seen / 4.9M: one update of 8,064 decisions is 1/607 of an iteration
    assert float(A.schedule_step(8064, CFG)) == pytest.approx(8064 / 4.9e6)
    day = 130e6  # about one day at the measured trainer-0 rate
    assert float(A.learning_rate(day, CFG)) == pytest.approx(1e-4)
    assert float(A.temperature(day, CFG)) == pytest.approx(0.05 / (day / 4.9e6 + 1) ** 0.3, rel=1e-6)


def test_targets_follow_each_seat_and_reset_at_the_game_end():
    """One environment: main acts at t=0,1, other at t=2, main at t=3 and ends the game (win for main); t=4 is the
    reset transition; a new game: other at t=5, then the segment ends with main to move."""
    T = 6
    d = np.array([[0.5, 0.2, 0.3], [0.6, 0.1, 0.3], [0.2, 0.3, 0.5], [0.7, 0.1, 0.2], [0.3, 0.4, 0.3],
                  [0.4, 0.4, 0.2]], np.float32)[:, None, :]
    rewards = np.zeros((T, 1), np.float32)
    rewards[3] = 1.0
    dones = np.zeros((T, 1), bool)
    dones[4] = True
    next_dones = np.zeros((T, 1), bool)
    next_dones[3] = True
    mains = np.array([1, 1, 0, 1, 1, 0], bool)[:, None]
    next_dist = np.array([[0.1, 0.8, 0.1]], np.float32)  # main to move after the segment
    returns, adv = A.targets_by_seat(jnp.asarray(d), jnp.asarray(rewards), jnp.asarray(dones),
                                     jnp.asarray(next_dones), jnp.asarray(mains), jnp.asarray(next_dist), CFG)
    returns, adv = np.asarray(returns)[:, 0], np.asarray(adv)[:, 0]
    lv, la = CFG.td_lambda, CFG.gae_lambda
    s = lambda x: x[0] - x[2]
    win, loss = np.array([1, 0, 0], np.float32), np.array([0, 0, 1], np.float32)
    # main's chain t=0 -> t=1 -> t=3 -> end (win); other's chain t=2 -> end (mirrored: loss)
    av3 = win - d[3, 0]
    av1 = d[3, 0] - d[1, 0] + lv * av3
    av0 = d[1, 0] - d[0, 0] + lv * av1
    av2 = loss - d[2, 0]
    np.testing.assert_allclose(returns[3], d[3, 0] + av3, atol=1e-6)
    np.testing.assert_allclose(returns[1], d[1, 0] + av1, atol=1e-6)
    np.testing.assert_allclose(returns[0], d[0, 0] + av0, atol=1e-6)
    np.testing.assert_allclose(returns[2], d[2, 0] + av2, atol=1e-6)
    aa3 = s(win) - s(d[3, 0])
    aa1 = s(d[3, 0]) - s(d[1, 0]) + la * aa3
    assert adv[3] == pytest.approx(aa3, abs=1e-6) and adv[1] == pytest.approx(aa1, abs=1e-6)
    assert adv[0] == pytest.approx(s(d[1, 0]) - s(d[0, 0]) + la * aa1, abs=1e-6)
    assert adv[2] == pytest.approx(s(loss) - s(d[2, 0]), abs=1e-6)
    # new game: other at t=5 bootstraps from the mirror of main's next value (truncated at the segment end)
    np.testing.assert_allclose(returns[5], next_dist[0, ::-1], atol=1e-6)
    assert adv[5] == pytest.approx(s(next_dist[0, ::-1]) - s(d[5, 0]), abs=1e-6)
    assert adv[4] == 0 and not returns[4].any()  # the reset transition is not a decision


def test_the_advantage_filter_keeps_the_top_quarter_with_a_floor():
    adv = jnp.asarray(np.linspace(-1, 1, 101, dtype=np.float32))
    valid = jnp.ones(101, bool)
    kept, threshold = A.advantage_mask(adv, valid, CFG)
    magnitude = np.abs(np.linspace(-1, 1, 101))
    assert float(threshold) == pytest.approx(np.quantile(magnitude, 0.75), rel=1e-5)
    assert int(kept.sum()) == int((magnitude >= np.quantile(magnitude, 0.75) - 1e-7).sum())
    small = jnp.asarray(np.full(20, 0.001, np.float32))
    kept, threshold = A.advantage_mask(small, jnp.ones(20, bool), CFG)
    assert float(threshold) == pytest.approx(0.01) and not bool(kept.any())  # the 0.01 floor
    kept, _ = A.advantage_mask(adv, valid.at[:50].set(False), CFG)
    assert not bool(kept[:50].any())  # invalid rows never kept, and not counted in the quantile


def test_loss_terms_match_a_hand_computed_batch():
    behaviour = np.array([[0.0, 1.0, -1e9], [2.0, 0.0, 0.5]], np.float32)
    new = np.array([[0.5, 0.5, 3.0], [1.0, 1.0, 1.0]], np.float32)
    actions = np.array([1, 2])
    advantages = np.array([0.4, -0.2], np.float32)
    returns = np.array([[1, 0, 0], [0.2, 0.3, 0.5]], np.float32)
    value_logits = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, -1.0]], np.float32)
    kept = np.array([True, True])
    temp = 0.05
    total, terms = A.loss_terms(jnp.asarray(new), jnp.asarray(value_logits), jnp.asarray(behaviour),
                                jnp.asarray(actions), jnp.asarray(advantages), jnp.asarray(returns),
                                jnp.asarray(kept), temp, UNIFORM)

    def logsoftmax(x, legal):
        z = np.where(legal, x, -np.inf)
        z = z - np.log(np.exp(z - z.max()).sum()) - z.max()
        return np.where(legal, z, 0.0)

    rows = []
    for i in range(2):
        legal = behaviour[i] > -1e8
        lp, old = logsoftmax(new[i], legal), logsoftmax(behaviour[i], legal)
        p = np.where(legal, np.exp(lp), 0.0)
        ratio = math.exp(lp[actions[i]] - old[actions[i]])
        policy = -min(advantages[i] * ratio, advantages[i] * min(max(ratio, 0.8), 1.2))
        kl = (p * (lp - old)).sum()
        entropy = -(p * lp).sum()
        magnet = (p * (lp - np.log(1.0 / legal.sum()))).sum()  # KL to uniform over legal (UNIFORM)
        lv = value_logits[i] - np.log(np.exp(value_logits[i]).sum())
        value = -(returns[i] * lv).sum()
        rows.append((policy, magnet, value, kl, entropy))
    policy, magnet, value, kl, entropy = np.mean(rows, axis=0)
    assert float(terms["policy"]) == pytest.approx(policy, rel=1e-5)
    assert float(terms["magnet_kl"]) == pytest.approx(magnet, rel=1e-5, abs=1e-7)
    assert float(terms["value"]) == pytest.approx(value, rel=1e-5)
    assert float(terms["kl"]) == pytest.approx(kl, rel=1e-5, abs=1e-7)
    assert float(terms["entropy"]) == pytest.approx(entropy, rel=1e-5)
    assert float(total) == pytest.approx(policy + temp * magnet + value + 0.1 * kl, rel=1e-5)
    # a dropped row leaves the loss
    _, dropped = A.loss_terms(jnp.asarray(new), jnp.asarray(value_logits), jnp.asarray(behaviour),
                              jnp.asarray(actions), jnp.asarray(advantages), jnp.asarray(returns),
                              jnp.asarray([True, False]), temp, UNIFORM)
    assert float(dropped["value"]) == pytest.approx(rows[0][2], rel=1e-5)


def test_ema_moves_a_thousandth_per_update():
    ema = {"w": jnp.zeros(3)}
    params = {"w": jnp.ones(3)}
    once = A.ema_update(ema, params, CFG.ema_decay)
    np.testing.assert_allclose(np.asarray(once["w"]), 0.001, rtol=1e-6)
    twice = A.ema_update(once, params, CFG.ema_decay)
    np.testing.assert_allclose(np.asarray(twice["w"]), 1 - 0.999 ** 2, rtol=1e-6)


def test_a_withdrawn_decision_is_skipped_by_both_chains():
    """A withdrawn decision (illegal_activation_withdrawal/v1) passed as a non-decision: the other decisions' targets
    equal those of the same data without that step, and it gets no return or advantage."""
    rng = np.random.default_rng(3)
    T = 7
    d = rng.dirichlet(np.ones(3), size=(T, 1)).astype(np.float32)
    rewards = np.zeros((T, 1), np.float32)
    rewards[5] = -1.0
    next_dones = np.zeros((T, 1), bool)
    next_dones[5] = True
    mains = np.array([1, 0, 1, 1, 0, 1, 0], bool)[:, None]
    next_dist = np.array([[0.2, 0.5, 0.3]], np.float32)
    withdrawn = np.zeros((T, 1), bool)
    withdrawn[2] = True  # main's decision at step 2 was withdrawn; step 3 re-presents it
    run = lambda *x: [np.asarray(a) for a in A.targets_by_seat(*(jnp.asarray(v) for v in x), CFG)]
    returns, adv = run(d, rewards, withdrawn, next_dones, mains, next_dist)
    keep = np.array([0, 1, 3, 4, 5, 6])
    ref_returns, ref_adv = run(d[keep], rewards[keep], np.zeros((6, 1), bool), next_dones[keep], mains[keep], next_dist)
    np.testing.assert_allclose(returns[keep], ref_returns, atol=1e-6)
    np.testing.assert_allclose(adv[keep], ref_adv, atol=1e-6)
    assert adv[2, 0] == 0 and not returns[2].any()


def test_the_default_magnet_is_uniform_over_the_prompts_legal_options():
    assert CFG.magnet == "uniform_legal"


def test_the_card_grouped_magnet_on_a_hand_built_menu():
    """The paper's magnet (eq. 6, S3): uniform over source cards, then over the card's options; options without a
    card are groups of their own. Menu: card 3 (2 options), card 5 (3 options), a phase move and a pass (no card),
    one illegal row. Four groups: rho = 1/8, 1/8, 1/12 x 3, 1/4, 1/4."""
    legal = jnp.asarray([[True, True, True, True, True, True, True, False]])
    groups = jnp.asarray([[3, 3, 5, 5, 5, 0, 0, 3]])
    rho = np.exp(np.asarray(A.magnet_log_probs(legal, groups)))[0]
    np.testing.assert_allclose(rho[:7], [1 / 8, 1 / 8, 1 / 12, 1 / 12, 1 / 12, 1 / 4, 1 / 4], rtol=1e-6)
    assert rho[:7].sum() == pytest.approx(1.0)
    # a selection prompt whose rows share one source card: uniform over its options
    rho = np.exp(np.asarray(A.magnet_log_probs(jnp.ones((1, 4), bool), jnp.full((1, 4), 9))))[0]
    np.testing.assert_allclose(rho, 0.25, rtol=1e-6)
    # the loss's magnet KL is KL(pi || rho)
    logits = jnp.asarray([[0.3, -0.2, 1.0, 0.1, 0.0, -1.0, 0.5, -1e9]])
    total, terms = A.loss_terms(logits, jnp.zeros((1, 3)), logits, jnp.asarray([0]), jnp.asarray([0.0]),
                                jnp.asarray([[1.0, 0.0, 0.0]]), jnp.asarray([True]), 0.05, GROUPED, groups=groups)
    p = np.exp(np.asarray(logits[0, :7]) - np.log(np.exp(np.asarray(logits[0, :7])).sum()))
    rho = np.asarray([1 / 8, 1 / 8, 1 / 12, 1 / 12, 1 / 12, 1 / 4, 1 / 4])
    assert float(terms["magnet_kl"]) == pytest.approx(float((p * np.log(p / rho)).sum()), rel=1e-5)
