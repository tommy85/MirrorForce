"""Frozen public actor feature capture for offline AR belief. No label/native object/optimizer enters this module.

The current ordinary event window is represented by the actor's actual QUERY and history-conditioned state
tokens, not by a claimed dump of all event tokens. Chunk/memory features include the same public position
embeddings the actor reads. The caller retains public candidate/slot provenance and pins the frozen checkpoint.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mirrorforce.agent.model import policy_net as V
from mirrorforce.agent.train.policy_io import Policy

SCHEMA = "mirrorforce_frozen_belief_features/v1"
BLOCKS = ("state", "turn", "chunks", "memory")


def _positioned_prefix(model, prefix):
    dtype = model.config.compute_dtype or jnp.float32
    memory = prefix.memory.astype(dtype) + model.recency(prefix.recency).astype(dtype)
    chunks = prefix.chunks.astype(dtype) + model.chunk_position(jnp.clip(
        prefix.chunk_index, 0, model.config.chunk_slots - 1)).astype(dtype)
    return (jnp.where(prefix.memory_valid[..., None], memory, 0),
            jnp.where(prefix.chunk_valid[..., None], chunks, 0))


def forward(model, variables, obs, seat, memory, first=None, *, full_shapes=False):
    """Original act outputs plus stop-gradient four-block features, public candidates and frozen old-head logits.

    ``obs`` is the unprefixed public observation only. The first five outputs have the exact policy_net.act contract.
    No new parameters or module paths are created. Feature extraction is opt-in, separate from A0 learners.
    """
    if any(str(key).startswith(("priv", "label")) for key in obs):
        raise ValueError("belief feature capture refuses private/label inputs")
    acted = V.act(model, variables, obs, seat, memory, first, return_context=True, full_shapes=full_shapes)
    context, after = acted[5], acted[3]
    config, batch = model.config, seat.shape[0]
    rows = jnp.arange(batch)
    count, chunk_count = after.count[rows, seat], after.chunk_count[rows, seat]
    prefix = V._prefix(after.buffer, after.chunks, rows, seat, jnp.full_like(count, config.memory_slots),
                       config.memory_slots - jnp.minimum(count, config.memory_slots),
                       jnp.full_like(count, config.chunk_slots), config.chunk_slots - chunk_count, config)
    memory_features, chunk_features = model.apply(variables, prefix, method=_positioned_prefix)
    table = model.apply(variables, method=V.PolicyNet.semantic_table)
    encoded = model.apply(variables, obs, table, method=V.PolicyNet.encode)
    old_logits, old_valid = model.apply(variables, encoded, obs, table, method=V.PolicyNet.belief)
    candidates = obs["candidates_"].astype(jnp.int32)
    candidate_valid = candidates[..., 2] > 0
    semantics = V._lookup(table, ids=V._u16(candidates[..., :2]))
    features = {"state": context.state, "state_valid": context.state_valid,
                "turn": context.query[:, None], "turn_valid": jnp.ones((batch, 1), bool),
                "chunks": chunk_features, "chunks_valid": prefix.chunk_valid,
                "memory": memory_features, "memory_valid": prefix.memory_valid,
                "candidates": jnp.where(candidate_valid[..., None], semantics, 0), "candidate_valid": candidate_valid}
    return (*acted[:5], jax.tree_util.tree_map(jax.lax.stop_gradient, features),
            jax.lax.stop_gradient(old_logits), old_valid)


def public_for_layout(features, observation_candidates, code_by_id, law):
    """Join public candidate semantics to law.codes and construct public target-slot queries; no target access.

    This host adapter operates on one decision's unbatched features and returns a batch of one. Empty layouts
    have one invalid pad slot, allowing feature admission and zero-token loss without inventing a prediction.
    """
    from mirrorforce.agent.model.belief_ar import PUBLIC_KEYS
    candidates = np.asarray(observation_candidates)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or candidates.dtype.kind not in "iu":
        raise ValueError("public candidate rows must be integer [id hi, id lo, tier]")
    active = candidates[:, 2] > 0
    ids = candidates[active, 0].astype(np.int64) * 256 + candidates[active, 1]
    if len(set(ids.tolist())) != len(ids) or any(int(cid) not in code_by_id for cid in ids):
        raise ValueError("public candidate IDs repeat or are absent from the pinned code list")
    codes = [int(code_by_id[int(cid)]) for cid in ids]
    active_rows = np.flatnonzero(active)
    lookup = {code: int(index) for code, index in zip(codes, active_rows)}
    if set(law.codes) - set(lookup):
        raise ValueError("public native candidates do not cover the AR public pool vocabulary")
    if set(features) != PUBLIC_KEYS - {"slots", "slot_valid"}:
        raise ValueError("feature blocks differ from the registered public capture schema")
    if not np.array_equal(np.asarray(features["candidate_valid"]), active):
        raise ValueError("feature/public candidate validity masks differ")
    out = {key: np.asarray(features[key])[None] for name in BLOCKS for key in (name, name + "_valid")}
    selected = [lookup[code] for code in law.codes]
    # Even an empty hidden pool has one padded candidate for compiled gathers.
    out["candidates"] = np.asarray(features["candidates"])[selected][None] if selected else np.zeros(
        (1, 1, np.asarray(features["candidates"]).shape[-1]), np.asarray(features["candidates"]).dtype)
    out["candidate_valid"] = np.ones((1, len(selected)), bool) if selected else np.zeros((1, 1), bool)
    slots = np.zeros((max(law.length, 1), 8), np.float32)
    for index, slot in enumerate(law.fields):
        slots[index] = [1, 0, slot.location == 4, slot.location == 8, slot.sequence / 7,
                        index / 80, law.hand_size / 80, len(law.fields) / 15]
    for index in range(len(law.fields), law.length):
        slots[index] = [0, 1, 0, 0, 0, index / 80, law.hand_size / 80, len(law.fields) / 15]
    out["slots"], out["slot_valid"] = slots[None], (np.arange(len(slots)) < law.length)[None]
    return out


class FrozenBeliefPolicy(Policy):
    """A fixed-shape offline capture runner. Normal policy service and learner paths are not modified."""

    def __init__(self, agent, variables, receipt, *, batch_size=64):
        super().__init__(agent, variables, receipt)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("belief capture batch size must be a positive registered integer")
        self.batch_size = batch_size

        def apply(variables, obs, rstate, first):
            memory = V.Memory(*(jnp.stack([a, b], axis=1) for a, b in zip(*rstate)))
            logits, value, wdl, after, bad, features, old_logits, old_valid = forward(
                agent.model, variables, obs, jnp.zeros(first.shape, jnp.int32), memory, first)
            rstate = (tuple(value[:, 0] for value in after), tuple(value[:, 1] for value in after))
            return rstate, logits, value, wdl, bad, features, old_logits, old_valid
        self._feature_apply = jax.jit(apply)

    def act_batch_features(self, observations, rstates, firsts):
        if len(observations) != self.batch_size or len(rstates) != self.batch_size or len(firsts) != self.batch_size:
            raise ValueError("belief capture refuses a change to its registered fixed batch")
        if any(not str(key).startswith("obs:") for obs in observations for key in obs):
            raise ValueError("belief capture accepts only obs: public arrays, never labels or control fields")
        keys = set(observations[0])
        if any(set(obs) != keys for obs in observations):
            raise ValueError("belief capture observation keys differ across the batch")
        obs = {key[4:]: np.stack([np.asarray(row[key]) for row in observations]) for key in sorted(keys)}
        if obs["action_ir_"].shape[1] != self.max_options:
            raise ValueError("belief capture requires its full registered action menu tensor")
        options = (obs["action_ir_"][..., 0] > 0).sum(-1)
        stacked = jax.tree_util.tree_map(lambda *values: np.concatenate([np.asarray(value) for value in values]), *rstates)
        result = self._feature_apply(self.variables, obs, stacked, jnp.asarray(firsts, bool))
        rstate, logits, value, wdl, bad, features, old_logits, old_valid = jax.tree_util.tree_map(np.asarray, result)
        if bad.any():
            raise RuntimeError("belief capture saw inconsistent actor history deliveries")
        return [(jax.tree_util.tree_map(lambda value: value[index:index + 1], rstate), logits[index, :options[index]],
                 float(value[index]), wdl[index], {key: value[index] for key, value in features.items()},
                 old_logits[index], old_valid[index]) for index in range(self.batch_size)]


__all__ = ["SCHEMA", "forward", "public_for_layout", "FrozenBeliefPolicy"]
