"""The structure network (policy_net) behind the trainer's agent interface (``init_rnn_state`` / ``init`` / ``apply``).

The trainer keeps two recurrent states per environment, ``(rstate1, rstate2)``: the main seat's and the other seat's,
selected by ``main``. Here each is one seat's recurrent state ``(buffer [B, slots, d], count [B], last_turn [B],
chunks [B, chunk_slots, d], chunk_count [B], chunk_turn [B])``: its cross-turn memory and the current turn's chunk
summaries; seat 0 of ``policy_net.Memory`` is the main seat. ``done`` marks the first decision of a new episode (both
seats are cleared before it). ``apply`` runs ``policy_net.act`` when the observation batch equals the state batch, and
``policy_net.segment`` over [T, B] (time-major, as the trainer flattens it) when it is T times larger. An inconsistent
chunk or closed-turn delivery stops the run (a host callback raises).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mirrorforce.agent.model.policy_net import Memory, Prefix, PolicyNet, PolicyNetConfig, _check, act, segment


class ReferenceAgent:
    def __init__(self, config: PolicyNetConfig, semantic_shape):
        self.config = config
        self.model = PolicyNet(config, tuple(semantic_shape))

    def init_rnn_state(self, batch_size: int):
        c = self.config
        return (np.zeros((batch_size, c.memory_slots, c.d), np.float32), np.zeros((batch_size,), np.int32),
                np.full((batch_size,), -1, np.int32), np.zeros((batch_size, c.chunk_slots, c.d), np.float32),
                np.zeros((batch_size,), np.int32), np.full((batch_size,), -1, np.int32))

    def init(self, key, obs, rstate=None):
        c = self.config
        batch = jax.tree_util.tree_leaves(obs)[0].shape[0]
        positions = lambda n: jnp.broadcast_to(jnp.arange(n), (batch, n))
        prefix = Prefix(jnp.zeros((batch, c.memory_slots, c.d)), jnp.ones((batch, c.memory_slots), bool),
                        positions(c.memory_slots), jnp.zeros((batch, c.chunk_slots, c.d)),
                        jnp.ones((batch, c.chunk_slots), bool), positions(c.chunk_slots))
        return self.model.init(key, obs, prefix, method=PolicyNet.init_all)

    def apply(self, variables, obs, rstate, done=None, main=None, train=False, mutable=False, return_aux=False,
              return_bad=False):
        """``return_aux`` (segment mode): also the belief head's logits and candidate validity. ``return_bad``
        (decision mode): the delivery check [B] is returned (5th output) for the caller to check on the host,
        instead of a host callback inside the program."""
        refuse_private(obs)
        paired = isinstance(rstate, (tuple, list)) and len(rstate) == 2 and isinstance(rstate[0], (tuple, list))
        first_seat, second_seat = rstate if paired else (rstate, rstate)
        memory = Memory(*(jnp.stack([a, b], axis=1) for a, b in zip(first_seat, second_seat)))
        rows = memory.count.shape[0]
        batch = jax.tree_util.tree_leaves(obs)[0].shape[0]
        seat = jnp.zeros((batch,), jnp.int32) if main is None else jnp.where(main, 0, 1).astype(jnp.int32)
        first = None if done is None else jnp.asarray(done, bool)
        if batch == rows:
            logits, value, wdl, memory, bad = act(self.model, variables, obs, seat, memory, first)
            if not return_bad:
                jax.debug.callback(_check, bad)
        else:
            steps = batch // rows
            if steps * rows != batch:
                raise ValueError(f"observation batch {batch} is not a multiple of the state batch {rows}")
            shape = lambda x: x.reshape((steps, rows) + x.shape[1:])
            out = segment(self.model, variables, jax.tree_util.tree_map(shape, obs), shape(seat), memory,
                          None if first is None else shape(first), belief=return_aux)
            logits, value, wdl, memory, plan = out[:5]
            jax.debug.callback(_check, plan.bad)
            flat = lambda x: x.reshape((steps * rows,) + x.shape[2:])
            logits, value, wdl = flat(logits), flat(value), flat(wdl)
            if return_aux:
                aux = tuple(flat(x) for x in out[5])
        main_seat = tuple(x[:, 0] for x in memory)
        other_seat = tuple(x[:, 1] for x in memory)
        outputs = ((main_seat, other_seat) if paired else main_seat, logits, value[:, None], wdl)
        if return_bad:
            if batch != rows:
                raise ValueError("return_bad is the decision path's check")
            outputs = outputs + (bad,)
        if return_aux:
            if batch == rows:
                raise ValueError("belief outputs come from the segment (learner) path")
            outputs = outputs + (aux,)
        return (outputs, {"batch_stats": {}}) if mutable else outputs  # no batch statistics


def refuse_private(obs):
    """The policy's input boundary: an observation may carry no key of the other seat's private export (priv:),
    which only a training-time critic reads."""
    private = [k for k in obs if str(k).startswith("priv")]
    if private:
        raise ValueError(f"the policy refuses private (other seat's) inputs: {private[:4]}")


def select_rows(mask, first, second):
    """Per-row choice between two pytrees of [B, ...] leaves (any rank) by ``mask`` [B]."""
    return jax.tree_util.tree_map(
        lambda a, b: jnp.where(jnp.reshape(mask, (-1,) + (1,) * (jnp.ndim(a) - 1)), a, b), first, second)
