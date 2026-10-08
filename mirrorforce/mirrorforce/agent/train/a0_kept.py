"""A0's kept-rows learner step: the per-decision passes run only on the decisions the |A| filter keeps.

Every A0 loss term (policy, magnet, value, behaviour KL, and the belief loss: ``belief_rows: kept``, coordinator
2026-10-02) reads only the kept decisions (about a quarter), so the per-decision passes (state encoder, the turn pass
over memory, chunks and window, readout and heads) are needed on those rows only. The memory recurrence is not
per-decision: completed turns and chunks are summarized for every segment as before, so each kept decision sees
exactly the prefix the full segment forward gives it.

    phase one   ``policy_net.segment_prefix`` on the device's segments: summaries and every step's prefix (recomputed in
                the backward pass, so only its inputs are kept);
    gather      the kept rows (static capacity; more kept rows than the capacity is reported, never truncated);
    phase two   ``policy_net.rows_forward`` and the loss on chunks of kept rows, gradients summed over chunks, with the
                cotangent of each row's prefix;
    backward    phase one's vector-Jacobian product with the prefix cotangents scattered back to their rows.

The loss is a sum over rows divided by the caller's global denominators, so summing chunk losses gives the full loss;
``full_loss`` is the same loss on the full segment forward (the exactness reference of the tests).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from mirrorforce.agent.model import policy_net
from mirrorforce.agent.model.policy_net import Memory, Prefix

FLOAT = ("memory", "chunks")  # the prefix fields that carry gradients


def memory_of(rstate):
    """The agent's paired seat states (main, other) as policy_net's Memory [S, 2, ...]."""
    first_seat, second_seat = rstate
    return Memory(*(jnp.stack([a, b], axis=1) for a, b in zip(first_seat, second_seat)))


def time_major(x, steps):
    """[T * S, ...] time-major rows -> [T, S, ...]."""
    return x.reshape((steps, x.shape[0] // steps) + x.shape[1:])


def capacity(rows: int, rate: float, chunk: int, slack: float = 0.15) -> int:
    """Static capacity for the kept rows of ``rows`` decisions (kept rate ``rate`` plus ``slack``), in whole chunks."""
    need = int(rows * rate * (1 + slack)) + 1
    return min(rows, -(-need // chunk) * chunk)


def kept_value_and_grad(model, params, consts, obs, rows_small, mains, dones, rstate, kept, loss_rows, *, steps: int,
                        cap: int, chunk: int, belief: bool):
    """Loss, terms, gradients, the delivery check and the overflow flag of one device's minibatch.

    ``obs``: the policy's observation keys [T * S, ...] time-major; ``rows_small``: per-row arrays the loss reads
    (actions, behaviour logits, returns, advantages, labels, ...) [T * S, ...]; ``mains``/``dones`` [T * S];
    ``rstate``: the paired seat states at the segment starts [S, ...]; ``kept`` [T * S] the filter's rows;
    ``loss_rows(outputs, rows, weight) -> (loss, terms)`` the loss of rows given ``rows_forward``'s outputs, summed
    over rows with the caller's global denominators (``weight`` zero on padding)."""
    seat = time_major(jnp.where(mains, 0, 1).astype(jnp.int32), steps)
    first = time_major(dones, steps)
    obs_tm = jax.tree.map(lambda x: time_major(x, steps), obs)
    memory = memory_of(rstate)

    def phase_one(p):
        pre, plan = policy_net.segment_prefix(model, {"params": p, **consts}, obs_tm, seat, memory, first)
        floats = {k: getattr(pre, k) for k in FLOAT}
        rest = {k: getattr(pre, k) for k in Prefix._fields if k not in FLOAT}
        return floats, (rest, plan.bad.any())

    floats, vjp, (rest, bad) = jax.vjp(jax.checkpoint(phase_one), params, has_aux=True)
    count = kept.sum()
    index = jnp.nonzero(kept, size=cap, fill_value=0)[0]
    weight = (jnp.arange(cap) < count).astype(jnp.float32)
    blocks = cap // chunk
    gather = lambda x: x[index].reshape((blocks, chunk) + x.shape[1:])
    parts = (jax.tree.map(gather, obs), jax.tree.map(gather, floats), jax.tree.map(gather, rest),
             jax.tree.map(gather, rows_small), weight.reshape(blocks, chunk))

    def block(total, part):
        o, fl, rs, small, w = part

        def loss(p, fl):
            pre = Prefix(**fl, **rs)
            return loss_rows(policy_net.rows_forward(model, {"params": p, **consts}, o, pre, belief=belief), small, w)
        (value, terms), (g_params, g_floats) = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(params, fl)
        loss_total, terms_total, grads_total = total
        return (loss_total + value, jax.tree.map(jnp.add, terms_total, terms),
                jax.tree.map(jnp.add, grads_total, g_params)), g_floats

    first_part = jax.tree.map(lambda y: y[0], parts)
    probe = jax.eval_shape(lambda: loss_rows(
        policy_net.rows_forward(model, {"params": params, **consts}, first_part[0],
                           Prefix(**first_part[1], **first_part[2]), belief=belief), first_part[3], first_part[4]))
    zero = (jnp.zeros((), jnp.float32), jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), probe[1]),
            jax.tree.map(jnp.zeros_like, params))
    (value, terms, grads), g_floats = jax.lax.scan(block, zero, parts)
    scatter = lambda full, g: jnp.zeros_like(full).at[index].add(g.reshape((cap,) + g.shape[2:]).astype(full.dtype))
    (g_phase_one,) = vjp(jax.tree.map(scatter, floats, g_floats))
    grads = jax.tree.map(jnp.add, grads, g_phase_one)
    return value, terms, grads, bad, count > cap


def full_loss(model, params, consts, obs, rows_small, mains, dones, rstate, kept, loss_rows, *, steps: int,
              belief: bool):
    """The same loss on the full segment forward (every row's passes; weight = kept): the exactness reference."""
    seat = time_major(jnp.where(mains, 0, 1).astype(jnp.int32), steps)
    out = policy_net.segment(model, {"params": params, **consts}, jax.tree.map(lambda x: time_major(x, steps), obs), seat,
                        memory_of(rstate), time_major(dones, steps), belief=belief)
    flat = lambda x: x.reshape((-1,) + x.shape[2:])
    outputs = (flat(out[0]), flat(out[1]), flat(out[2]))
    if belief:
        outputs = outputs + ((flat(out[5][0]), flat(out[5][1])),)
    return loss_rows(outputs, rows_small, kept.astype(jnp.float32))


__all__ = ["kept_value_and_grad", "full_loss", "capacity", "memory_of"]
