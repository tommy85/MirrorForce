"""A0's central critic (user decision 2026-10-02): a training-only value network over both seats' observations.

The acting seat's observation (``obs:`` keys) and the other seat's (the env's both-seat export, ``priv:`` keys, the
other player's own view) are encoded by the same small policy_net trunk (card semantics, input encoder and state block),
each with a learned seat embedding; a few self-attention layers mix the two token sets, and a pooled readout gives
the acting seat's (win, draw, loss). It reads hidden truth (the other seat's hand and private state), so it never
reaches the policy or deployment: only the learner's advantage estimates read it.

Each seat's view is encoded at the decision only (no cross-turn memory): the union of both views fixes the game
state apart from the unknown future (deck order), which is the state a central critic conditions on.
"""
from __future__ import annotations

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from mirrorforce.agent.model.policy_net import MenuTokens, SelfBlock, PolicyNet, PolicyNetConfig, ValueMenuAttention, _key_mask

PRIVATE = "priv:"


def other_seat(obs, acting):
    """The other seat's observation from the ``priv:`` keys, with the acting seat's menu keys zeroed (the other seat
    has no prompt; only the acting seat's legal menu is read)."""
    out = {k[len(PRIVATE):]: v for k, v in obs.items() if str(k).startswith(PRIVATE)}
    for key, value in acting.items():
        if key not in out:
            out[key] = jnp.zeros_like(value)
    return out


class CriticNet(nn.Module):
    config: PolicyNetConfig
    semantic_shape: Tuple[int, int, int, int]
    mix_layers: int = 2
    zero_out: bool = False  # the output layer starts at zero: the first predictions are uniform over win/draw/loss

    def setup(self):
        c = self.config
        self.trunk = PolicyNet(c, self.semantic_shape, name="trunk")
        self.seat = self.param("seat_embedding", nn.initializers.normal(0.02), (2, c.d))
        self.mix = [SelfBlock(c.d, c.heads, c.ff, c.compute_dtype, name=f"mix_{i}") for i in range(self.mix_layers)]
        self.norm = nn.LayerNorm(dtype=c.compute_dtype, name="mix_norm")
        self.hidden = nn.Dense(c.d, dtype=c.compute_dtype, name="value_hidden")
        zeros = nn.initializers.zeros
        self.out = (nn.Dense(3, kernel_init=zeros, bias_init=zeros, name="value_out") if self.zero_out else
                    nn.Dense(3, name="value_out"))
        if c.value_menu:
            self.menu_tokens = MenuTokens(c, name="menu_tokens")
            self.value_menu = ValueMenuAttention(c.d, c.heads, c.compute_dtype, name="value_menu")

    def __call__(self, obs):
        """``obs``: the acting seat's keys and the other seat's ``priv:`` keys, batch-major. Returns (win, draw, loss)
        logits [B, 3] of the acting seat."""
        acting = {k: v for k, v in obs.items() if not str(k).startswith(PRIVATE)}
        table = self.trunk.semantic_table()
        mine = self.trunk.encode(acting, table)
        theirs = self.trunk.encode(other_seat(obs, acting), table)
        dtype = mine.state.dtype
        x = jnp.concatenate([mine.state + self.seat[0].astype(dtype), theirs.state + self.seat[1].astype(dtype)],
                            axis=1)
        valid = jnp.concatenate([mine.state_valid, theirs.state_valid], axis=1)
        mask = _key_mask(x.shape[1], valid)
        for blk in self.mix:
            x = blk(x, mask)
        x = jnp.where(valid[..., None], self.norm(x), 0)
        pooled = x.sum(1) / jnp.maximum(valid.sum(1, keepdims=True), 1).astype(x.dtype)
        if self.config.value_menu:
            cards = mine.state[:, 1:1 + acting["cards_"].shape[1]]
            action = self.menu_tokens(cards, mine.action_base, acting)
            pooled = pooled + self.value_menu(pooled[:, None], action, mine.action_valid)[:, 0]
        return self.out(nn.gelu(self.hidden(pooled))).astype(jnp.float32)


def critic_loss(logits, returns, valid):
    """Cross-entropy of the critic's (win, draw, loss) against categorical returns over the valid decisions; the
    caller's denominator (``valid.sum()``) is global under accumulation."""
    ce = -(returns * jax.nn.log_softmax(logits, axis=-1)).sum(-1)
    return (ce * valid).sum(), valid.sum()


__all__ = ["CriticNet", "critic_loss", "other_seat", "PRIVATE"]
