"""The kept-rows learner step equals the full segment forward: loss, every term and every parameter gradient, on
segments with completed turns, chunks, episode starts and carried memory; more kept rows than the capacity is
reported. Runs in the venv on CPU.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mirrorforce.agent.model.policy_net import BELIEF_COUNTS, BELIEF_LOCATIONS
from mirrorforce.agent.train import a0_kept, ataraxos
from test_policy_net import SMALL, SMALL_SEMANTICS, carried, episode, make, shapes

STEPS, SEGMENTS = 12, 4


def setup(seed):
    rng = np.random.default_rng(seed)
    layout = shapes()
    model, variables = make(SMALL, SMALL_SEMANTICS, layout, SEGMENTS, rng)
    obs, seat, first = episode(rng, STEPS, SEGMENTS, layout, SMALL_SEMANTICS[0], first_turn=2, resets=0.08)
    memory = carried(np.random.default_rng(seed + 1), 3, SMALL)
    memory = jax.tree.map(lambda x: jnp.concatenate([x, x[:1]], axis=0), memory)  # 4 segments
    rstate = (tuple(x[:, 0] for x in memory), tuple(x[:, 1] for x in memory))
    flat = lambda x: jnp.asarray(x).reshape((STEPS * SEGMENTS,) + x.shape[2:])
    obs = jax.tree.map(flat, obs)
    mains, dones = flat(seat == 0), flat(first)
    n, options = STEPS * SEGMENTS, obs["action_ir_"].shape[1]
    legal = np.asarray(obs["action_ir_"][..., 0] > 0)
    small = {"behaviour": jnp.asarray(np.where(legal, rng.normal(size=(n, options)), ataraxos.NEG * 10), jnp.float32),
             "actions": jnp.asarray(np.argmax(legal * rng.random((n, options)), -1), jnp.int32),
             "advantages": jnp.asarray(rng.normal(size=n), jnp.float32),
             "returns": jnp.asarray(rng.dirichlet(np.ones(3), n), jnp.float32),
             "targets": jnp.asarray(rng.integers(0, BELIEF_COUNTS, (n, obs["candidates_"].shape[1],
                                                                    len(BELIEF_LOCATIONS))), jnp.int32)}
    return model, variables, obs, small, mains, dones, rstate, rng


def make_loss(den, belief_den, cfg):
    def loss_rows(outputs, small, weight):
        logits, value, wdl, (belief_logits, belief_valid) = outputs
        total, terms = ataraxos.loss_terms(logits, wdl, small["behaviour"], small["actions"], small["advantages"],
                                           small["returns"], weight, 0.05, cfg, den)
        nll = -jnp.take_along_axis(jax.nn.log_softmax(belief_logits.astype(jnp.float32), axis=-1),
                                   small["targets"][..., None], axis=-1)[..., 0]
        rows = (belief_valid & (weight > 0)[:, None])[..., None].astype(jnp.float32)  # belief_rows: kept
        terms["belief"] = (nll * rows).sum() / belief_den
        return total + 0.1 * terms["belief"], terms
    return loss_rows


def test_kept_rows_equal_the_full_forward():
    model, variables, obs, small, mains, dones, rstate, rng = setup(3)
    params, consts = variables["params"], {k: v for k, v in variables.items() if k != "params"}
    kept = jnp.asarray((rng.random(STEPS * SEGMENTS) < 0.3) & ~np.asarray(dones))
    loss_rows = make_loss(kept.sum().astype(jnp.float32), 50.0, ataraxos.AtaraxosConfig())
    full = jax.jit(jax.value_and_grad(
        lambda p: a0_kept.full_loss(model, p, consts, obs, small, mains, dones, rstate, kept, loss_rows, steps=STEPS,
                                    belief=True), has_aux=True))
    (loss_ref, terms_ref), grads_ref = full(params)
    cap = a0_kept.capacity(STEPS * SEGMENTS, 0.3, chunk=8, slack=0.5)
    assert int(kept.sum()) <= cap < STEPS * SEGMENTS
    run = jax.jit(lambda p: a0_kept.kept_value_and_grad(model, p, consts, obs, small, mains, dones, rstate, kept,
                                                        loss_rows, steps=STEPS, cap=cap, chunk=8, belief=True))
    loss, terms, grads, bad, overflow = run(params)
    assert not bool(bad) and not bool(overflow)
    np.testing.assert_allclose(float(loss), float(loss_ref), rtol=1e-5, atol=1e-6)
    for name in terms_ref:
        np.testing.assert_allclose(float(terms[name]), float(terms_ref[name]), rtol=1e-4, atol=1e-6, err_msg=name)
    flat_ref = jax.tree_util.tree_leaves_with_path(grads_ref)
    flat = dict(jax.tree_util.tree_leaves_with_path(grads))
    scale = max(float(jnp.abs(g).max()) for _, g in flat_ref)
    touched = 0
    for path, g in flat_ref:
        np.testing.assert_allclose(np.asarray(flat[path]), np.asarray(g), atol=1e-5 * scale, rtol=1e-4,
                                   err_msg=jax.tree_util.keystr(path))
        touched += bool(jnp.abs(g).max() > 0)
    assert touched > 0.8 * len(flat_ref)  # the summaries' parameters get gradients through the prefixes too


def test_more_kept_rows_than_the_capacity_is_reported():
    model, variables, obs, small, mains, dones, rstate, rng = setup(4)
    params, consts = variables["params"], {k: v for k, v in variables.items() if k != "params"}
    kept = jnp.asarray(~np.asarray(dones))
    loss_rows = make_loss(kept.sum().astype(jnp.float32), 50.0, ataraxos.AtaraxosConfig())
    run = jax.jit(lambda p: a0_kept.kept_value_and_grad(model, p, consts, obs, small, mains, dones, rstate, kept,
                                                        loss_rows, steps=STEPS, cap=16, chunk=8, belief=True))
    assert bool(run(params)[4]) and not bool(run(params)[3])
