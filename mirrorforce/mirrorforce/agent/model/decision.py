"""Role-bound decisions and next-decision public-outcome supervision."""
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from mirrorforce.agent.model.transformer import EncoderLayer


OUTCOME_DIM = 10
OUTCOME_NAMES = tuple(
    f"{side}_{name}"
    for side in ("acting", "opponent")
    for name in ("lp", "monsters", "spells", "grave", "banished")
)


def role_bound_features(card_tokens, refs, confidence):
    """Keep role slots ordered; absent/out-of-range references contribute zero."""
    valid = (refs > 0) & (refs <= card_tokens.shape[1])
    indices = jnp.clip(refs.astype(jnp.int32) - 1, 0, card_tokens.shape[1] - 1)
    batch = jnp.arange(card_tokens.shape[0])[:, None, None]
    cards = jnp.where(valid[..., None], card_tokens[batch, indices], 0)
    return jnp.concatenate(
        [cards.reshape(*refs.shape[:2], -1),
         valid.astype(cards.dtype), confidence.astype(cards.dtype)], axis=-1
    )


class RoleBinding(nn.Module):
    channels: int
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, cards, refs, confidence):
        features = role_bound_features(cards, refs, confidence)
        hidden = nn.gelu(nn.Dense(
            self.channels * 2, dtype=self.dtype, param_dtype=self.param_dtype,
            name="hidden",
        )(features))
        return nn.Dense(
            self.channels, dtype=self.dtype, param_dtype=self.param_dtype,
            kernel_init=nn.initializers.zeros, name="output",
        )(hidden)


class DecisionActor(nn.Module):
    channels: int
    num_heads: int = 4
    layers: int = 2
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, recurrent_state, context):
        # Local import keeps StructuredEncoder's optional branch dependency acyclic.
        from mirrorforce.agent.model.structured_agent import CrossAttentionBlock

        if self.layers < 1 or self.channels % self.num_heads:
            raise ValueError("decision layers must be positive and heads divide channels")
        scene, padding = context["scene"], context["scene_padding"]
        action_padding = context["action_padding"]
        safe_padding = action_padding.at[:, 0].set(
            action_padding[:, 0] & (~action_padding).any(axis=-1)
        )
        dense = lambda size, name: nn.Dense(
            size, dtype=self.dtype, param_dtype=self.param_dtype, name=name
        )
        query = nn.gelu(dense(self.channels, "query")(
            jnp.concatenate([recurrent_state, context["query"]], axis=-1)
        ))[:, None, :]
        candidates = dense(self.channels, "candidates")(
            jnp.concatenate([context["actions"], context["roles"]], axis=-1)
        )
        for layer in range(self.layers):
            query = CrossAttentionBlock(
                self.channels, self.num_heads, dtype=self.dtype,
                param_dtype=self.param_dtype, name=f"query_scene_{layer}",
            )(query, scene, padding)
            broadcast_query = jnp.broadcast_to(query, candidates.shape)
            update = nn.gelu(dense(self.channels * 2, f"condition_up_{layer}")(
                jnp.concatenate([candidates, broadcast_query], axis=-1)
            ))
            candidates = candidates + dense(
                self.channels, f"condition_down_{layer}"
            )(update)
            candidates = CrossAttentionBlock(
                self.channels, self.num_heads, dtype=self.dtype,
                param_dtype=self.param_dtype, name=f"candidate_scene_{layer}",
            )(candidates, scene, padding)
            candidates = jnp.where(action_padding[..., None], 0, candidates)
        candidates = EncoderLayer(
            self.num_heads, intermediate_size=self.channels * 4,
            activation="gelu", dtype=self.dtype, param_dtype=self.param_dtype,
            name="candidate_comparison",
        )(candidates, src_key_padding_mask=safe_padding)
        candidates = nn.LayerNorm(dtype=self.dtype, name="candidate_norm")(candidates)
        candidates = jnp.where(action_padding[..., None], 0, candidates)
        query = jnp.broadcast_to(query, candidates.shape)
        decisions = nn.gelu(dense(self.channels, "decision")(
            jnp.concatenate([query, candidates, query * candidates], axis=-1)
        ))
        delta = nn.Dense(
            1, dtype=jnp.float32, param_dtype=self.param_dtype,
            kernel_init=nn.initializers.zeros, name="logit_delta",
        )(decisions)[..., 0]
        outcome = nn.Dense(
            OUTCOME_DIM * 4, dtype=jnp.float32, param_dtype=self.param_dtype,
            name="outcome",
        )(decisions)
        return jnp.where(action_padding, 0, delta), {
            "delta": outcome[..., :OUTCOME_DIM],
            "change_logits": outcome[..., OUTCOME_DIM:].reshape(
                *outcome.shape[:2], OUTCOME_DIM, 3
            ),
        }


def public_outcome_targets(before, after, done, before_done=None):
    """Next engine decision, not chain resolution or per-card causal attribution.

    Global counts are public even when card identities are concealed. Align the
    next observer to the acting seat; never use terminal/reset observations.
    """
    a = np.asarray(before["global_"], dtype=np.int32)
    b = np.asarray(after["global_"], dtype=np.int32)

    def summary(g):
        lp = np.stack([g[:, 0] * 256 + g[:, 1],
                       g[:, 2] * 256 + g[:, 3]], axis=1) / 8000.0
        counts = g[:, 8:22].reshape(-1, 2, 7)
        return np.concatenate([lp[..., None], counts[:, :, 2:6] / 8.0], axis=-1)

    previous, following = summary(a), summary(b)
    swapped = a[:, 6] != b[:, 6]
    following = np.where(swapped[:, None, None], following[:, ::-1], following)
    # column 22 of global_ flags an observation without a decision (terminal); the room format and era follow it
    valid = (~np.asarray(done, dtype=np.bool_)) & (a[:, 22] == 0) & (b[:, 22] == 0)
    if before_done is not None:
        valid &= ~np.asarray(before_done, dtype=np.bool_)
    delta = (following - previous).reshape(-1, OUTCOME_DIM).astype(np.float32)
    return np.where(valid[:, None], delta, 0), valid


def outcome_loss(prediction, actions, targets, valid):
    """Only the executed action is supervised; no labels for unchosen actions."""
    row = jnp.arange(actions.shape[0])
    values = prediction["delta"][row, actions]
    logits = prediction["change_logits"][row, actions]
    targets = jax.lax.stop_gradient(targets)
    labels = jnp.sign(targets).astype(jnp.int32) + 1
    ce = -jnp.take_along_axis(
        jax.nn.log_softmax(logits, axis=-1), labels[..., None], axis=-1
    )[..., 0]
    error = values - targets
    abs_error = jnp.abs(error)
    huber = jnp.where(abs_error <= 1, 0.5 * error ** 2, abs_error - 0.5)
    per_sample = (ce + huber).mean(axis=-1)
    return jnp.where(valid, per_sample, 0).sum() / jnp.maximum(valid.sum(), 1)
