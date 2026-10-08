"""Independent public-conditioned AR decoder (T5 prototype), not wired to A0 or the policy service.

The caller supplies observer-legal state, current-turn features, chunk summaries and cross-turn memory. Every
conditioning tensor, including the shared candidate semantics, is stopped here. This does not itself certify the
features' provenance: integration must prove the right observer/history at extraction. ``previous_tokens`` holds
generated vocabulary indices (-1 for BOS/padding); deployment never passes teacher targets to this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


MEMORY_BLOCKS = ("state", "turn", "chunks", "memory")
PUBLIC_KEYS = frozenset({key for name in MEMORY_BLOCKS for key in (name, name + "_valid")}
                        | {"candidates", "candidate_valid", "slots", "slot_valid"})


@dataclass(frozen=True)
class ARConfig:
    width: int = 256
    heads: int = 4
    layers: int = 2
    dropout: float = 0.2
    max_targets: int = 80

    def __post_init__(self):
        if self.width <= 0 or self.heads <= 0 or self.width % self.heads or self.layers < 1 \
                or not 0 <= self.dropout < 1 or self.max_targets < 1:
            raise ValueError("invalid independent AR decoder dimensions or dropout")


def check_public_shapes(public, previous_tokens, config: ARConfig) -> None:
    """Static schema/shape checks, usable inside JIT. Public provenance and dynamic masks are checked by callers."""
    if set(public) != PUBLIC_KEYS:
        raise ValueError("AR conditioning accepts only its explicit public feature blocks (no truth/labels)")
    if previous_tokens.ndim != 2 or previous_tokens.dtype.kind != "i":
        raise ValueError("previous AR token indices must be [batch, slots] integers")
    batch, slots = previous_tokens.shape
    if not 0 < slots <= config.max_targets:
        raise ValueError("AR decoder slot count outside its registered maximum; pad an empty layout only for batching")
    for name in MEMORY_BLOCKS:
        features, valid = public[name], public[name + "_valid"]
        if features.ndim != 3 or features.shape[0] != batch or valid.shape != features.shape[:2] \
                or valid.dtype != jnp.bool_:
            raise ValueError(f"AR public {name} features and validity mask differ")
    candidates, valid = public["candidates"], public["candidate_valid"]
    if candidates.ndim != 3 or candidates.shape[0] != batch or candidates.shape[1] < 1 \
            or valid.shape != candidates.shape[:2] or valid.dtype != jnp.bool_:
        raise ValueError("AR public candidate embeddings and mask differ")
    if public["slots"].ndim != 3 or public["slots"].shape[:2] != (batch, slots) \
            or public["slot_valid"].shape != (batch, slots) or public["slot_valid"].dtype != jnp.bool_:
        raise ValueError("AR public slot features and validity mask differ")


def admit_previous_tokens(previous_tokens, candidate_valid, slot_valid) -> None:
    """Host-side input admission; never call on JIT tracers. Only -1 or a valid unpadded candidate index is legal.

    -1 denotes BOS or an as-yet ungenerated prefix slot. Unsigned arrays cannot represent this sentinel and are
    refused, even if they happen not to contain it. Safe gathers inside JIT are not an alternative to admission.
    """
    previous, candidates, slots = map(np.asarray, (previous_tokens, candidate_valid, slot_valid))
    if previous.ndim != 2 or previous.dtype.kind != "i" or candidates.ndim != 2 \
            or slots.shape != previous.shape or candidates.shape[0] != previous.shape[0] \
            or candidates.shape[1] < 1 or candidates.dtype != bool or slots.dtype != bool:
        raise ValueError("AR host admission needs signed token indices and boolean candidate/slot masks")
    if ((previous < -1) | (previous >= candidates.shape[1])).any():
        raise ValueError("AR prefix token is not BOS -1 or an in-range candidate index")
    safe = np.maximum(previous, 0)
    if ((previous >= 0) & ~np.take_along_axis(candidates, safe, axis=1)).any():
        raise ValueError("AR prefix references a padding candidate")
    if (slots.any(axis=1) & ~candidates.any(axis=1)).any():
        raise ValueError("an active AR sequence requires a nonempty public candidate set")


def admit_public(public, previous_tokens, config: ARConfig) -> None:
    """Validate host public blocks and prefix before compiled decoder execution; does not verify provenance."""
    check_public_shapes(public, previous_tokens, config)
    admit_previous_tokens(previous_tokens, public["candidate_valid"], public["slot_valid"])
    for name in (*MEMORY_BLOCKS, "candidates", "slots"):
        values = np.asarray(public[name])
        if (values.dtype.kind not in "fiu" and str(values.dtype) != "bfloat16") or not np.isfinite(values).all():
            raise ValueError(f"AR public {name} must contain finite numeric features")


def sample_decoder(law, model, variables, public, *, seed: int) -> dict:
    """Host-admitted single-world reference sampling; no teacher labels and dropout is explicitly disabled.

    Candidate rows must already be in ``law.codes`` order (the future feature producer proves that mapping).
    This simple independent reference is not the final batched/cached production sampling implementation.
    """
    from mirrorforce.agent.search.belief_ar_law import sample
    if law.length == 0:
        return sample(law, lambda _: (), seed=seed)
    if public["candidates"].shape[:2] != (1, len(law.codes)) or public["slots"].shape[:2] != (1, law.length):
        raise ValueError("reference AR sampling requires one complete unpadded public world")
    if not np.asarray(public["candidate_valid"]).all() or not np.asarray(public["slot_valid"]).all():
        raise ValueError("reference AR sampling has no candidate or target padding")
    empty_prefix = np.full((1, law.length), -1, np.int32)
    admit_public(public, empty_prefix, model.config)
    candidate_valid, slot_valid = np.asarray(public["candidate_valid"]), np.asarray(public["slot_valid"])

    def score(prefix):
        previous = np.full((1, law.length), -1, np.int32)
        for step, code in enumerate(prefix):
            previous[0, step + 1] = law.index[code]
        admit_previous_tokens(previous, candidate_valid, slot_valid)
        logits = model.apply(variables, public, jnp.asarray(previous), deterministic=True)
        return np.asarray(logits)[0, len(prefix)]

    return sample(law, score, seed=seed)


class _Attention(nn.Module):
    width: int
    heads: int

    @nn.compact
    def __call__(self, query, memory, mask):
        dense = lambda name: nn.DenseGeneral((self.heads, self.width // self.heads), name=name)
        q, k, v = dense("q")(query), dense("k")(memory), dense("v")(memory)
        attended = jax.nn.dot_product_attention(q, k, v, mask=mask[:, None])
        return nn.DenseGeneral(self.width, axis=(-2, -1), name="out")(attended)


class _Block(nn.Module):
    config: ARConfig

    @nn.compact
    def __call__(self, x, memory, causal, cross, *, deterministic):
        c = self.config
        normed = nn.LayerNorm(name="self_norm")(x)
        value = _Attention(c.width, c.heads, name="self_attention")(normed, normed, causal)
        x = x + nn.Dropout(c.dropout, name="self_dropout")(value, deterministic=deterministic)
        value = _Attention(c.width, c.heads, name="cross_attention")(
            nn.LayerNorm(name="cross_norm")(x), memory, cross)
        x = x + nn.Dropout(c.dropout, name="cross_dropout")(value, deterministic=deterministic)
        value = nn.Dense(c.width * 4, name="ff_up")(nn.LayerNorm(name="ff_norm")(x))
        value = nn.Dense(c.width, name="ff_down")(nn.gelu(value))
        return x + nn.Dropout(c.dropout, name="ff_dropout")(value, deterministic=deterministic)


def deterministic_candidate_lookup(candidate, indices):
    """Dense one-hot lookup avoids the repeated-index scatter-add transpose of a GPU gather.

    This does not add parameters or change public vocabulary/prefix semantics. HIGHEST keeps the one-hot
    contraction at full operand precision instead of allowing reduced-precision matrix-multiply inputs.
    End-to-end GPU replay is a separate gate; this alone is not a promise that every decoder operation is stable.
    """
    selector = jax.nn.one_hot(indices, candidate.shape[1], dtype=candidate.dtype)
    return jnp.einsum("bsv,bvd->bsd", selector, candidate, precision=jax.lax.Precision.HIGHEST)


class ARDecoder(nn.Module):
    config: ARConfig = ARConfig()

    @nn.compact
    def __call__(self, public, previous_tokens, *, deterministic: bool):
        check_public_shapes(public, previous_tokens, self.config)
        c = self.config
        batch, slots = previous_tokens.shape
        blocks, masks = [], []
        for name in MEMORY_BLOCKS:
            # Explicit type projections preserve all four inputs, even when feature widths happen to coincide.
            block = nn.Dense(c.width, name=name + "_projection")(jax.lax.stop_gradient(public[name]))
            blocks.append(block)
            masks.append(public[name + "_valid"])
        # A learned public null token makes cross-attention defined even for empty diagnostic feature blocks.
        null = self.param("null_memory", nn.initializers.normal(0.02), (1, 1, c.width))
        memory = jnp.concatenate([jnp.broadcast_to(null, (batch, 1, c.width)), *blocks], axis=1)
        memory_valid = jnp.concatenate([jnp.ones((batch, 1), bool), *masks], axis=1)
        candidate = nn.Dense(c.width, name="candidate_projection")(jax.lax.stop_gradient(public["candidates"]))
        safe = jnp.clip(previous_tokens, 0, candidate.shape[1] - 1)
        previous = deterministic_candidate_lookup(candidate, safe)
        bos = self.param("bos", nn.initializers.normal(0.02), (c.width,))
        previous = jnp.where((previous_tokens < 0)[..., None], bos, previous)
        x = previous + nn.Dense(c.width, name="slot_projection")(jax.lax.stop_gradient(public["slots"]))
        x += nn.Embed(c.max_targets, c.width, name="position")(jnp.arange(slots))[None]
        # Inputs are already right-shifted; query t may see prior-token input t, never t+1.
        causal = jnp.broadcast_to(jnp.tril(jnp.ones((slots, slots), bool)), (batch, slots, slots))
        causal &= public["slot_valid"][:, None, :] | jnp.eye(slots, dtype=bool)[None]
        cross = jnp.broadcast_to(memory_valid[:, None, :], (batch, slots, memory.shape[1]))
        for i in range(c.layers):
            x = _Block(c, name=f"decoder_{i}")(x, memory, causal, cross, deterministic=deterministic)
        query = nn.Dense(c.width, name="output_query")(nn.LayerNorm(name="output_norm")(x))
        keys = nn.Dense(c.width, name="output_key")(candidate)
        logits = jnp.einsum("bsd,bvd->bsv", query, keys) / jnp.sqrt(float(c.width))
        return jnp.where(public["candidate_valid"][:, None, :], logits, -1e30)
