"""A0's central critic: it reads both seats, and its targets follow each environment's segments across an iteration.

Runs in the venv on CPU.
"""
from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mirrorforce.agent.model.critic import CriticNet, other_seat
from mirrorforce.agent.train import ataraxos
from mirrorforce.agent.train.a0_buffer import Segment
from test_policy_net import SMALL, SMALL_SEMANTICS, make, random_obs, shapes

MENU = ("action_ir_", "action_single_refs_", "action_group_refs_", "action_group_mask_", "candidates_")


def both_seats(rng, batch, layout):
    acting = random_obs(rng, (batch,), layout, SMALL_SEMANTICS[0])
    other = random_obs(rng, (batch,), layout, SMALL_SEMANTICS[0])
    return {**acting, **{f"priv:{k}": v for k, v in other.items() if k not in MENU}}


def test_the_critic_reads_both_seats():
    rng = np.random.default_rng(3)
    layout = shapes()
    _, policy_variables = make(SMALL, SMALL_SEMANTICS, layout, 2, rng)  # the semantic constants
    model = CriticNet(SMALL, SMALL_SEMANTICS, mix_layers=1)
    obs = jax.tree_util.tree_map(jnp.asarray, both_seats(rng, 2, layout))
    params = model.init(jax.random.PRNGKey(1), obs)["params"]
    variables = {"params": params, "constants": {"trunk": policy_variables["constants"]}}
    run = jax.jit(model.apply)
    base = np.asarray(run(variables, obs))
    assert base.shape == (2, 3) and np.isfinite(base).all()
    # the other seat has no menu: its menu keys are the acting seat's zeroed
    other = other_seat(obs, {k: v for k, v in obs.items() if not k.startswith("priv:")})
    assert all(not np.asarray(other[k]).any() for k in MENU)
    # the other seat's view reaches the value, and so does the acting seat's
    changed = dict(obs, **{"priv:cards_": jnp.asarray(random_obs(rng, (2,), layout, SMALL_SEMANTICS[0])["cards_"])})
    assert np.abs(np.asarray(run(variables, changed)) - base).max() > 1e-4
    changed = dict(obs, cards_=jnp.asarray(random_obs(rng, (2,), layout, SMALL_SEMANTICS[0])["cards_"]))
    assert np.abs(np.asarray(run(variables, changed)) - base).max() > 1e-4


def series(column, steps):
    """One environment's decisions over the iteration: a game ending after step ``2 + column`` (won by the seat acting
    there), then a game still running; seats alternate."""
    end = 2 + column
    mains = np.arange(steps) % 2 == 0
    next_dones = np.zeros(steps, bool)
    next_dones[end] = True
    rewards = np.zeros(steps, np.float32)
    rewards[end] = 1.0
    truth = np.zeros((steps, 3), np.float32)  # the outcome from each decision's seat (known up to the end)
    for t in range(end + 1):
        truth[t] = [1, 0, 0] if mains[t] == mains[end] else [0, 0, 1]
    return mains, next_dones, rewards, truth, end


def test_critic_targets_follow_each_environment_across_segments():
    from mirrorforce.agent.train.cleanba import critic_targets
    threads, envs, T, updates = 2, 2, 4, 2
    steps = updates * T
    columns = threads * envs
    rng = np.random.default_rng(5)
    cfg = ataraxos.AtaraxosConfig()
    data = [series(c, steps) for c in range(columns)]
    critic = [np.where(np.arange(steps)[:, None] <= d[4], d[3], rng.dirichlet(np.ones(3), steps)).astype(np.float32)
              for d in data]  # exact on the finished game
    public = [rng.dirichlet(np.ones(3), steps).astype(np.float32) for _ in data]
    bootstrap = [rng.dirichlet(np.ones(3)).astype(np.float32) for _ in data]
    turns_of = [np.minimum(np.arange(steps) // 2 + 1, 12) for _ in data]  # two decisions per turn
    segments, dists, turns = [], [], []
    for u in range(updates):
        for q in range(threads):
            for e in range(envs):
                c, part = q * envs + e, slice(u * T, (u + 1) * T)
                mains, next_dones, rewards, _, _ = data[c]
                segments.append(Segment(b"", np.zeros(T, bool), mains[part], np.zeros(T, np.int32),
                                        np.zeros((T, 3), np.float32), np.full(T, 7.0, np.float32), np.zeros(T, bool),
                                        ((), ()), rewards=rewards[part], next_dones=next_dones[part],
                                        value_dists=public[c][part], bootstrap=bootstrap[c]))
                dists.append(critic[c][part])
                turns.append(turns_of[c][part])
    report = critic_targets(segments, dists, turns, threads, envs, T, True, cfg)
    stack = lambda i: jnp.asarray(np.stack([d[i] for d in data], axis=1))
    returns, advantages = ataraxos.targets_by_seat(
        jnp.asarray(np.stack(critic, axis=1)), stack(2), jnp.zeros((steps, columns), bool), stack(1), stack(0),
        jnp.asarray(np.stack(bootstrap)), cfg)
    returns, advantages = np.asarray(returns), np.asarray(advantages)
    for u in range(updates):
        for q in range(threads):
            for e in range(envs):
                seg, c = segments[(u * threads + q) * envs + e], q * envs + e
                np.testing.assert_allclose(seg.critic_returns, returns[u * T:(u + 1) * T, c], atol=1e-6)
                np.testing.assert_allclose(seg.advantages, advantages[u * T:(u + 1) * T, c], atol=1e-6)
    known = sum(d[4] + 1 for d in data)
    assert report["critic"]["n"] == report["public"]["n"] == known
    assert report["critic"]["brier"] == pytest.approx(0.0, abs=1e-9)
    assert report["critic"]["logloss"] == pytest.approx(0.0, abs=1e-6)
    rows = [(c, t) for c in range(columns) for t in range(data[c][4] + 1)]
    expected = np.mean([((public[c][t] - data[c][3][t]) ** 2).sum() for c, t in rows])
    assert report["public"]["brier"] == pytest.approx(expected, abs=1e-5)
    expected = np.mean([-np.log((public[c][t] * data[c][3][t]).sum()) for c, t in rows])
    assert report["public"]["logloss"] == pytest.approx(expected, abs=1e-5)
    by_turn = report["public"]["by_turn"]
    assert sum(b["n"] for b in by_turn.values()) == known
    assert by_turn["1-2"]["n"] == sum(1 for c, t in rows if turns_of[c][t] <= 2)
    # without --critic-advantages the policy keeps the public head's advantages
    for seg in segments:
        seg.advantages[:] = 7.0
    critic_targets(segments, dists, turns, threads, envs, T, False, cfg)
    assert all((seg.advantages == 7.0).all() for seg in segments)


def test_a_zero_initialized_output_starts_uniform():
    rng = np.random.default_rng(9)
    layout = shapes()
    _, policy_variables = make(SMALL, SMALL_SEMANTICS, layout, 2, rng)
    model = CriticNet(SMALL, SMALL_SEMANTICS, mix_layers=1, zero_out=True)
    obs = jax.tree_util.tree_map(jnp.asarray, both_seats(rng, 4, layout))
    params = model.init(jax.random.PRNGKey(2), obs)["params"]
    logits = jax.jit(model.apply)({"params": params, "constants": {"trunk": policy_variables["constants"]}}, obs)
    np.testing.assert_allclose(np.asarray(jax.nn.softmax(logits, -1)), np.full((4, 3), 1 / 3), atol=1e-7)


def test_monte_carlo_critic_targets():
    """td_lambda 1: a game that ends within the iteration gives every one of its decisions its outcome exactly (from
    that decision's seat); an unfinished game bootstraps from the value after the last segment (mirrored for the
    other seat), whatever the critic predicted on the way."""
    from mirrorforce.agent.train.cleanba import critic_targets
    threads, envs, T, updates = 1, 2, 4, 2
    steps = updates * T
    rng = np.random.default_rng(6)
    data = [series(0, steps), None]  # env 0: a game ends after step 2, then a new one runs on
    mains = np.arange(steps) % 2 == 0
    data[1] = (mains, np.zeros(steps, bool), np.zeros(steps, np.float32), None, None)  # env 1: never ends
    bootstrap = [np.array([0.2, 0.1, 0.7], np.float32), np.array([0.6, 0.3, 0.1], np.float32)]
    segments, dists, turns = [], [], []
    for u in range(updates):
        for e in range(envs):
            part = slice(u * T, (u + 1) * T)
            m, nd, r = data[e][0], data[e][1], data[e][2]
            segments.append(Segment(b"", np.zeros(T, bool), m[part], np.zeros(T, np.int32), np.zeros((T, 3), np.float32),
                                    np.zeros(T, np.float32), np.zeros(T, bool), ((), ()), rewards=r[part],
                                    next_dones=nd[part], value_dists=rng.dirichlet(np.ones(3), T).astype(np.float32),
                                    bootstrap=bootstrap[e]))
            dists.append(rng.dirichlet(np.ones(3), T).astype(np.float32))
            turns.append(np.ones(T, np.int64))
    critic_targets(segments, dists, turns, threads, envs, T, False, ataraxos.AtaraxosConfig(), td_lambda=1.0)
    returns = [np.concatenate([segments[u * envs + e].critic_returns for u in range(updates)]) for e in range(envs)]
    truth, end = data[0][3], data[0][4]
    np.testing.assert_allclose(returns[0][:end + 1], truth[:end + 1], atol=1e-6)  # the finished game: outcomes
    mirror = lambda d: d[::-1]
    for t in range(end + 1, steps):  # the game after it never ends: the bootstrap from each decision's seat
        np.testing.assert_allclose(returns[0][t], bootstrap[0] if mains[t] else mirror(bootstrap[0]), atol=1e-6)
    for t in range(steps):
        np.testing.assert_allclose(returns[1][t], bootstrap[1] if mains[t] else mirror(bootstrap[1]), atol=1e-6)


def test_the_critic_needs_the_both_seat_export():
    from mirrorforce.agent.train.cleanba import make_critic
    args = types.SimpleNamespace(iteration_decisions=4096, export_both_seats=False)
    with pytest.raises(ValueError, match="both-seat export"):
        make_critic(args, None, None, None, [], [], None)
