"""The model structure (``mirrorforce_torch_summary_model/v1``) in JAX on the observation (design section 8).

Structure, at the production size (d = 768, 12 heads, feed-forward 3072):

- state block: a set transformer (6 layers) over the state token, the card rows, the chain table, the
  unpositioned identities and two pooled tokens (the public opponent recipe, the turn's activation table);
- turn sequence: [BOS, cross-turn memory, the current turn's chunk summaries, the current turn's condensed events
  (a window of 512 rows), QUERY], 2 transformer layers;
- cross-turn memory: one summary per completed turn, the output of the same turn layers at a SUMMARY token over
  [BOS, memory, that turn's chunk summaries, its final window]; a fixed buffer of ``memory_slots`` summaries per
  seat, the most recent kept;
- within-turn compression: the rows of a long turn that leave the window arrive in chunks of 64; each becomes a
  chunk summary by the same SUMMARY pass over [BOS, memory, the turn's earlier chunk summaries, the chunk] (at most
  ``chunk_slots`` per turn, the environment's cap; nothing is dropped);
- stage-two readout: the state tokens cross-attend the turn sequence (2 layers);
- readout as the WorldReadout: each action queries the state (policy logit), four learned value queries give
  V as win/draw/loss logits.

There is no LSTM and no text channel. Card semantics are the frozen CDB/Lua tables (``constants``) plus a trainable
card identity residual.

Two ways to run the same network:

- ``act``: one decision per environment; the seat's memory and chunk carry are the recurrent state (``Memory``).
  Delivered chunks and new completed turns are summarized first (order in the runner section's header).
- ``segment``: the learner's path over a rollout segment [T, B]. Per-step work (encoding, the turn pass, readouts)
  runs batched over all steps; the summaries created inside the segment run in a scan over each environment's
  operations in order (bounded by the delivery slots per step, so it never runs out; empty entries are skipped), and
  each step gathers its seat's memory window and current chunks.

Both give the same outputs (``tests/test_policy_net.py``). Only ``obs:`` keys are read.
"""
from __future__ import annotations

import dataclasses
import functools
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from mirrorforce.agent.model.structured_agent import EffectSetEncoder, ExactSemanticEncoder

NEG = -1e9


@dataclasses.dataclass(frozen=True)
class PolicyNetConfig:
    d: int = 768
    heads: int = 12
    ff: int = 3072
    state_layers: int = 6
    turn_layers: int = 2
    readout_layers: int = 2
    semantic_dim: int = 256
    event_hidden: int = 2048
    action_hidden: int = 2048
    readout_hidden: int = 1536
    value_queries: int = 4
    memory_slots: int = 64
    chunk_slots: int = 48
    """the current turn's chunk summaries the model holds (64 event rows each, the rows of a long turn beyond its
    window); equals the environment's per-turn chunk cap (``history_chunk_cap``: 512 decisions x 7 rows)"""
    public_status: bool = False
    """read the public status inputs (public_effects/v1, reveal_returns/v1): obs:card_status_ [C, 6] on the card
    tokens (equip host and status source as card references, equip count, the zone it came from, status bits, turns
    since), obs:public_effects_ [16, 8] as state tokens (lingering player effects), and event rows 27 wide (reason
    bits 8-31 in columns 24-26)"""
    hand_limit: bool = False
    """read obs:hand_limit_ [2, 4] (each seat's valid, limit, hand size, excess; into the state token) and
    obs:action_discard_ [A] (the cards an end-phase move would discard; into the action rows), zero-initialized"""
    value_menu: bool = False
    """zero-output residual attention from the value queries to legal action tokens. False decodes legacy
    checkpoints; new training enables it, and legacy resumes must declare the additive migration."""
    row_blocks: Tuple[int, ...] = (128, 256)
    """event-row blocks below the full window: a turn pass (decision or summary) runs on the smallest block holding
    every active window; with chunk summaries present it runs on the full window and all chunk slots"""
    belief_width: int = 256
    belief_heads: int = 4
    belief_layers: int = 2
    remat: bool = False
    dtype: Optional[str] = None  # compute dtype ("bfloat16"); parameters stay float32

    @property
    def compute_dtype(self):
        return None if self.dtype is None else jnp.dtype(self.dtype)


class Memory(NamedTuple):
    """Each environment's two seats' recurrent state:

    - the cross-turn memory: the last ``memory_slots`` turn summaries, oldest first and right-aligned (slot -1 is
      the newest), how many summaries the seat has written, and the last summarized turn;
    - the current turn's chunk summaries (rows that left the 512-row window, 64 per chunk), in order from slot 0,
      their count and the turn they belong to (-1: none)."""
    buffer: jax.Array  # [B, 2, slots, d]
    count: jax.Array  # [B, 2] int32
    last_turn: jax.Array  # [B, 2] int32
    chunks: jax.Array  # [B, 2, chunk_slots, d]
    chunk_count: jax.Array  # [B, 2] int32
    chunk_turn: jax.Array  # [B, 2] int32

    @staticmethod
    def zeros(batch: int, config: PolicyNetConfig) -> "Memory":
        return Memory(jnp.zeros((batch, 2, config.memory_slots, config.d), jnp.float32),
                      jnp.zeros((batch, 2), jnp.int32), jnp.full((batch, 2), -1, jnp.int32),
                      jnp.zeros((batch, 2, config.chunk_slots, config.d), jnp.float32),
                      jnp.zeros((batch, 2), jnp.int32), jnp.full((batch, 2), -1, jnp.int32))


class Prefix(NamedTuple):
    """What precedes the event rows in a turn sequence: visible memory summaries (with recency) and the current
    turn's visible chunk summaries (with their index in the turn)."""
    memory: jax.Array  # [B, Lm, d]
    memory_valid: jax.Array  # [B, Lm]
    recency: jax.Array  # [B, Lm]
    chunks: jax.Array  # [B, Lc, d]
    chunk_valid: jax.Array  # [B, Lc]
    chunk_index: jax.Array  # [B, Lc]


def _gather_rows(tokens: jax.Array, refs: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Rows of ``tokens`` [B, C, d] named by ``refs`` (row + 1, 0 = none) of any trailing shape."""
    refs = refs.astype(jnp.int32)
    valid = refs > 0
    safe = jnp.clip(refs - 1, 0, tokens.shape[1] - 1)
    flat = safe.reshape(safe.shape[0], -1)
    # gathered in float32: the gradient is a scatter-add, which is slow and contended in bfloat16
    gathered = jnp.take_along_axis(tokens.astype(jnp.float32), flat[..., None], axis=1)
    gathered = gathered.reshape(safe.shape + (tokens.shape[-1],)).astype(tokens.dtype)
    return jnp.where(valid[..., None], gathered, 0), valid


def _u16(x: jax.Array) -> jax.Array:
    x = x.astype(jnp.int32)
    return x[..., 0] * 256 + x[..., 1]


class Attention(nn.Module):
    d: int
    heads: int
    dtype: Optional[jnp.dtype] = None

    @nn.compact
    def __call__(self, query: jax.Array, kv: jax.Array, mask: jax.Array) -> jax.Array:
        """``mask`` [B, Tq, Tk], True where a query may read a key; every query row must have a key."""
        head = self.d // self.heads
        dense = lambda name: nn.DenseGeneral((self.heads, head), dtype=self.dtype, name=name)
        q, k, v = dense("query")(query), dense("key")(kv), dense("value")(kv)
        out = jax.nn.dot_product_attention(q, k, v, mask=mask[:, None, :, :])
        return nn.DenseGeneral(self.d, axis=(-2, -1), dtype=self.dtype, name="out")(out)


class ValueMenuAttention(nn.Module):
    """An initially zero, learnable residual; padded rows and an empty menu contribute exactly zero."""
    d: int
    heads: int
    dtype: Optional[jnp.dtype] = None

    @nn.compact
    def __call__(self, query, action, valid):
        valid = valid.astype(bool)
        action = jnp.where(valid[..., None], action, 0)
        # A zero sentinel keeps softmax defined for rows with no legal actions.
        action = jnp.concatenate([action, jnp.zeros_like(query[:, :1])], axis=1)
        any_valid = valid.any(axis=1)
        keys = jnp.concatenate([valid, ~any_valid[:, None]], axis=1)
        dense = lambda name: nn.DenseGeneral((self.heads, self.d // self.heads), dtype=self.dtype, name=name)
        q, k, v = dense("query")(query), dense("key")(action), dense("value")(action)
        context = jax.nn.dot_product_attention(q, k, v, mask=_key_mask(query.shape[1], keys)[:, None])
        out = nn.DenseGeneral(self.d, axis=(-2, -1), dtype=self.dtype, use_bias=False,
                              kernel_init=nn.initializers.zeros, name="out")(context)
        return jnp.where(any_valid[:, None, None], out, 0)


class FeedForward(nn.Module):
    d: int
    ff: int
    dtype: Optional[jnp.dtype] = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.gelu(nn.Dense(self.ff, dtype=self.dtype, name="up")(x))
        return nn.Dense(self.d, dtype=self.dtype, name="down")(x)


class SelfBlock(nn.Module):
    d: int
    heads: int
    ff: int
    dtype: Optional[jnp.dtype] = None

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        h = nn.LayerNorm(dtype=self.dtype, name="attention_norm")(x)
        x = x + Attention(self.d, self.heads, self.dtype, name="attention")(h, h, mask)
        return x + FeedForward(self.d, self.ff, self.dtype, name="ffn")(nn.LayerNorm(dtype=self.dtype, name="ffn_norm")(x))


class CrossBlock(nn.Module):
    d: int
    heads: int
    ff: int
    dtype: Optional[jnp.dtype] = None

    @nn.compact
    def __call__(self, x: jax.Array, memory: jax.Array, mask: jax.Array) -> jax.Array:
        h = nn.LayerNorm(dtype=self.dtype, name="query_norm")(x)
        m = nn.LayerNorm(dtype=self.dtype, name="memory_norm")(memory)
        x = x + Attention(self.d, self.heads, self.dtype, name="attention")(h, m, mask)
        return x + FeedForward(self.d, self.ff, self.dtype, name="ffn")(nn.LayerNorm(dtype=self.dtype, name="ffn_norm")(x))


def _key_mask(query_len: int, key_valid: jax.Array) -> jax.Array:
    return jnp.broadcast_to(key_valid[:, None, :], (key_valid.shape[0], query_len, key_valid.shape[1]))


class Encoded(NamedTuple):
    card_inputs: jax.Array  # [B, C, d] card tokens before the state block (events gather these)
    state: jax.Array  # [B, S, d] state block output
    state_valid: jax.Array  # [B, S]
    action_base: jax.Array  # [B, A, d] action tokens before card references
    action_valid: jax.Array  # [B, A]


class DecisionContext(NamedTuple):
    """Public actor features after its real current-turn/memory readout, before the policy/value heads.

    This adds no parameters. Auxiliary heads must stop gradients through ALL these fields; no private critic
    view is ever accepted by the actor runner that produces them.
    """
    state: jax.Array  # [B, S, D], public state tokens after attending the actor's turn/history context
    state_valid: jax.Array  # [B, S]
    query: jax.Array  # [B, D], the actor's current-turn QUERY (including public memory/chunks)
    action_base: jax.Array  # [B, A, D]
    action_valid: jax.Array  # [B, A]


class _Fields(nn.Module):
    """Helpers shared by the compact input modules."""
    config: PolicyNetConfig

    def embed(self, name: str, size: int, values: jax.Array) -> jax.Array:
        """An embedding gathered from its float32 table (the gradient's scatter-add stays in float32, where atomics
        are native; a bfloat16 scatter-add is the slow, contended path) and cast to the compute dtype."""
        table = nn.Embed(size, self.config.d, name=name).embedding
        rows = table[jnp.clip(values.astype(jnp.int32), 0, size - 1)]
        return rows.astype(self.config.compute_dtype or jnp.float32)

    def dense(self, name: str, x: jax.Array) -> jax.Array:
        return nn.Dense(self.config.d, dtype=self.config.compute_dtype, name=name)(x.astype(jnp.float32))

    def norm(self, name: str, x: jax.Array) -> jax.Array:
        return nn.LayerNorm(dtype=self.config.compute_dtype, name=name)(x)

    def bound_units(self, name: str, tables, ids: jax.Array, ordinals: jax.Array) -> jax.Array:
        """The exact effect-unit binding of rows naming (card, description ordinal), through a zero-initialized
        projection (a row with no proven binding adds nothing)."""
        dtype = self.config.compute_dtype
        return nn.Dense(self.config.d, kernel_init=nn.initializers.zeros, use_bias=False, dtype=dtype, name=name)(
            _units(tables, ids, ordinals, dtype))


class CardSemantics(nn.Module):
    """The card semantic tables, computed once per forward for every card id (each lookup is then a gather; id 0 is
    the unknown card):

    - ``table`` [num_cards, d]: frozen CDB and Lua tables (``constants``) through small encoders, the card tables
      (32-dim setcode bag, 8 link-arrow bits, library genericity), and a trainable identity residual;
    - ``units`` [num_cards, 15, semantic_dim]: for each description ordinal 0..13 the pooled Lua units its effect
      binds (``effect_unit_bits``; no proven binding pools every unit), and at 14 every unit pooled."""
    config: PolicyNetConfig
    semantic_shape: Tuple[int, int, int, int]

    @nn.compact
    def __call__(self):
        c, dt = self.config, self.config.compute_dtype
        num_cards, cdb_dim, max_effects, lua_dim = self.semantic_shape
        const = lambda name, shape, dtype: self.variable("constants", name, lambda: jnp.zeros(shape, dtype)).value
        cdb = jax.lax.stop_gradient(const("cdb_exact", (num_cards, cdb_dim), jnp.float32))
        lua = jax.lax.stop_gradient(const("lua_effects", (num_cards, max_effects, lua_dim), jnp.float16))
        lua_mask = jax.lax.stop_gradient(const("lua_mask", (num_cards, max_effects), jnp.uint8)) > 0
        setcodes = jax.lax.stop_gradient(const("setcode_bag", (num_cards, 32), jnp.float32))
        arrows = jax.lax.stop_gradient(const("link_arrows", (num_cards, 8), jnp.float32))
        genericity = jax.lax.stop_gradient(const("genericity", (num_cards, 1), jnp.float32))
        bits = const("effect_unit_bits", (num_cards, ORDINALS), jnp.uint16).astype(jnp.int32)
        exact = ExactSemanticEncoder(c.semantic_dim, dtype=dt, name="exact")(cdb)
        # the effect units: the structured encoder's set pooling, here once per description ordinal
        unit_tokens = nn.gelu(nn.Dense(c.semantic_dim, dtype=dt, name="unit_projection")(lua.astype(dt or jnp.float32)))
        scores = nn.Dense(1, dtype=dt, name="unit_score")(unit_tokens)[..., 0].astype(jnp.float32)  # [N, U]
        bound = (bits[:, :, None] >> jnp.arange(max_effects)[None, None, :]) & 1  # [N, ORDINALS, U]
        bound = bound.astype(bool) & lua_mask[:, None, :]
        masks = jnp.where(bound.any(-1, keepdims=True), bound, lua_mask[:, None, :])  # unbound: every unit
        masks = jnp.concatenate([masks, lua_mask[:, None, :]], axis=1)  # [N, ORDINALS + 1, U]
        empty = ~masks.any(-1)
        weights = jax.nn.softmax(jnp.where(masks, scores[:, None, :], jnp.finfo(jnp.float32).min), axis=-1)
        weights = jnp.where(masks & ~empty[..., None], weights, 0)
        units = jnp.einsum("nou,nuc->noc", weights.astype(unit_tokens.dtype), unit_tokens)  # [N, ORDINALS + 1, C]
        effects = units[:, -1]
        gate = jax.nn.sigmoid(nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.constant(-2.0),
                                       dtype=dt, name="lua_gate")(jnp.concatenate([exact, effects], axis=-1)))
        identity = nn.Embed(num_cards, c.d, embedding_init=nn.initializers.zeros, name="identity").embedding
        projected = nn.Dense(c.d, dtype=dt, name="projection")(exact + gate * effects)
        tables = nn.Dense(c.d, kernel_init=nn.initializers.zeros, dtype=dt, name="card_tables")(
            jnp.concatenate([setcodes, arrows, genericity], axis=-1))
        table = projected.astype(jnp.float32) + tables.astype(jnp.float32) + identity  # float32: lookups gather
        return table, units.astype(jnp.float32)


ORDINALS = 14  # description ordinals the effect-unit evidence covers (card_tables.py)


def _lookup(tables, ids: jax.Array, dtype=None) -> jax.Array:
    """Rows of the float32 semantic table, cast to the compute dtype after the gather."""
    table = tables[0]
    rows = table[jnp.clip(ids.astype(jnp.int32), 0, table.shape[0] - 1)]
    return rows if dtype is None else rows.astype(dtype)


def _units(tables, ids: jax.Array, ordinals: jax.Array, dtype=None) -> jax.Array:
    """The pooled Lua units a row's description binds, minus the card's all-unit pool (zero for an unbound or
    out-of-range ordinal, such as a system string or another card's description)."""
    units = tables[1]
    ids = jnp.clip(ids.astype(jnp.int32), 0, units.shape[0] - 1)
    ordinals = ordinals.astype(jnp.int32)
    ordinals = jnp.where((ordinals >= 0) & (ordinals < ORDINALS), ordinals, ORDINALS)
    rows = units[ids, ordinals] - units[ids, ORDINALS]
    return rows if dtype is None else rows.astype(dtype)


class GroupRoles(nn.Module):
    """Each action's role groups of card references pooled by attention (the structured encoder's role-set
    attention): a member's score depends on its card and the role only, so scores are computed per card, and the
    weighted sum is a matrix product, never materializing [actions, roles, members, d]."""
    channels: int
    num_roles: int = 5
    dtype: Optional[jnp.dtype] = None

    @nn.compact
    def __call__(self, cards: jax.Array, refs: jax.Array, mask: jax.Array) -> jax.Array:
        b, num_cards = cards.shape[:2]
        role = nn.Embed(self.num_roles, self.channels, name="role_embedding").embedding.astype(cards.dtype)
        scores = nn.Dense(1, dtype=self.dtype, name="member_score")(
            jnp.tanh(cards[:, :, None, :] + role[None, None]))[..., 0]  # [B, C, R]
        refs = refs.astype(jnp.int32)
        valid = mask.astype(bool) & (refs > 0)
        safe = jnp.clip(refs - 1, 0, num_cards - 1)
        member = scores.astype(jnp.float32)[
            jnp.arange(b)[:, None, None, None], safe, jnp.arange(self.num_roles)[None, None, :, None]]
        empty = ~valid.any(axis=-1)
        member = jnp.where(valid, member, jnp.finfo(member.dtype).min)
        member = jnp.where(empty[..., None], 0, member)
        weights = jnp.where(valid, jax.nn.softmax(member, axis=-1), 0)
        spread = (jax.nn.one_hot(safe, num_cards, dtype=cards.dtype) * weights[..., None].astype(cards.dtype)).sum(3)
        pooled = jnp.einsum("barc,bcd->bard", spread, cards)
        pooled = jnp.where(empty[..., None], 0, pooled).reshape(pooled.shape[:2] + (-1,))
        return nn.Dense(self.channels, dtype=self.dtype, name="role_projection")(pooled)


class EventEncoder(_Fields):
    """Tokens of condensed event rows [B, K, 24] with their card references [B, K, 4] (history_obs.h layout)."""

    @nn.compact
    def __call__(self, rows: jax.Array, refs: jax.Array, card_inputs: jax.Array, table: jax.Array):
        c = self.config
        r = rows.astype(jnp.int32)
        e = self.embed
        x = (e("kind", 32, r[..., 1]) + e("subtype", 64, r[..., 2]) + e("msg", 256, r[..., 3])
             + e("player", 4, r[..., 4]) + e("phase", 16, r[..., 5])
             + e("from_controller", 4, r[..., 8]) + e("from_location", 16, r[..., 9])
             + e("from_sequence", 32, r[..., 10]) + e("to_controller", 4, r[..., 11])
             + e("to_location", 16, r[..., 12]) + e("to_sequence", 32, r[..., 13])
             + e("from_position", 16, r[..., 14]) + e("to_position", 16, r[..., 15])
             + e("reason", 256, r[..., 16]) + e("state", 256, r[..., 17])
             + e("count", 32, r[..., 20]) + e("link", 16, r[..., 21])
             + e("desc", 32, r[..., 22]) + e("act", 16, r[..., 23])
             + self.dense("amount", (r[..., 18:19] * 256 + r[..., 19:20]) / 8000.0)
             + _lookup(table, dtype=self.config.compute_dtype, ids=_u16(r[..., 6:8]))
             + self.bound_units("units", table, _u16(r[..., 6:8]), r[..., 22]))
        if c.public_status:  # the full reason: bits 8-15, 16-23, 24-31 (discard, redirect, link, ...)
            x = x + e("reason_b1", 256, r[..., 24]) + e("reason_b2", 256, r[..., 25]) + e("reason_b3", 256, r[..., 26])
        first, _ = _gather_rows(card_inputs, refs[..., 0])
        second, _ = _gather_rows(card_inputs, refs[..., 2])
        x = x + first + second + e("ref_first", 8, refs[..., 1]) + e("ref_second", 8, refs[..., 3])
        x = self.norm("input_norm", x)
        h = nn.Dense(c.event_hidden, dtype=c.compute_dtype, name="up")(x)
        x = self.norm("output_norm", x + nn.Dense(c.d, dtype=c.compute_dtype, name="down")(nn.gelu(h)))
        valid = r[..., 0] > 0
        return jnp.where(valid[..., None], x, 0), valid


class InputEncoder(_Fields):
    """The state block's input tokens and the actions' base tokens (event rows are encoded in the turn passes)."""

    @nn.compact
    def __call__(self, obs, table):
        c = self.config
        e, dense = self.embed, self.dense
        cards = obs["cards_"].astype(jnp.int32)
        card_ids = _u16(cards[..., :2])
        card_valid = cards[..., 2] > 0
        signed = lambda hi, lo: jnp.where(hi >= 128, hi * 256 + lo - 65536, hi * 256 + lo) / 10000.0
        card_numeric = jnp.concatenate([signed(cards[..., 12], cards[..., 13])[..., None],
                                        signed(cards[..., 14], cards[..., 15])[..., None], cards[..., 16:]], axis=-1)
        card_x = (e("card_location", 16, cards[..., 2]) + e("card_sequence", 128, cards[..., 3])
                  + e("card_controller", 3, cards[..., 4]) + e("card_position", 16, cards[..., 5])
                  + e("card_overlay", 3, cards[..., 6]) + e("card_attribute", 16, cards[..., 7])
                  + e("card_race", 32, cards[..., 8]) + e("card_level", 32, cards[..., 9])
                  + e("card_counter", 17, cards[..., 10]) + e("card_negated", 3, cards[..., 11])
                  + e("card_known", 2, (card_ids > 0).astype(jnp.int32))
                  + dense("card_numeric", card_numeric) + dense("card_turn", obs["card_turn_"])
                  + _lookup(table, dtype=self.config.compute_dtype, ids=card_ids))
        if c.public_status:  # each card's public status (public_status.h): equip host, source, bits, arrival zone
            st = obs["card_status_"].astype(jnp.int32)
            referenced = lambda ref: jnp.where(ref > 0, jnp.take_along_axis(
                card_ids, jnp.clip(ref - 1, 0, card_ids.shape[1] - 1), axis=1), 0)
            bits = ((st[..., 3:4] >> jnp.arange(8)) & 1).astype(jnp.float32)
            card_x = (card_x + e("status_equips", 8, jnp.clip(st[..., 1], 0, 7))
                      + e("status_arrived", 16, jnp.clip(st[..., 2], 0, 15)) + dense("status_bits", bits)
                      + e("status_turns", 8, jnp.clip(st[..., 5], 0, 7))
                      + dense("status_host", _lookup(table, ids=referenced(st[..., 0])))
                      + dense("status_source", _lookup(table, ids=referenced(st[..., 4]))))
        card_inputs = jnp.where(card_valid[..., None], self.norm("card_norm", card_x), 0)

        g = obs["global_"].astype(jnp.int32)
        b = g.shape[0]
        own_lp, opp_lp = (g[:, 0] * 256 + g[:, 1]) / 8000.0, (g[:, 2] * 256 + g[:, 3]) / 8000.0
        global_numeric = jnp.concatenate(
            [own_lp[:, None], opp_lp[:, None], (own_lp - opp_lp)[:, None], g[:, 8:22] / 60.0], axis=-1)
        s = obs["selection_"].astype(jnp.int32)
        state_token = (e("global_turn", 20, g[:, 4]) + e("global_phase", 16, g[:, 5]) + e("global_first", 2, g[:, 6])
                       + e("global_turn_owner", 2, g[:, 7]) + dense("global_numeric", global_numeric)
                       + e("selection_prompt", 32, s[:, 0]) + e("selection_role", 8, s[:, 1])
                       + dense("selection_numeric", s[:, 2:] / 8.0)
                       + dense("turn_ledger", obs["turn_ledger_"].reshape(b, -1) / 16.0)
                       + dense("player_hints", obs["player_hints_"].reshape(b, -1) / 16.0)
                       + e("room_format", 64, g[:, 23]) + e("room_era", 64, g[:, 24]))
        if c.hand_limit:  # the end-phase hand limit (public): each seat's limit, hand size and excess
            hand = obs["hand_limit_"].astype(jnp.float32).reshape(b, -1) / 16.0
            state_token = state_token + nn.Dense(c.d, kernel_init=nn.initializers.zeros, use_bias=False,
                                                 dtype=c.compute_dtype, name="hand_limit")(hand)
        state_token = self.norm("state_token_norm", state_token)

        ch = obs["chain_"].astype(jnp.int32)
        chain_valid = ch[..., 0] > 0
        chain = (e("chain_side", 4, ch[..., 1]) + e("chain_desc", 32, ch[..., 4]) + e("chain_location", 16, ch[..., 5])
                 + e("chain_sequence", 32, ch[..., 6]) + e("chain_state", 16, ch[..., 7])
                 + _lookup(table, dtype=self.config.compute_dtype, ids=_u16(ch[..., 2:4]))
                 + self.bound_units("chain_units", table, _u16(ch[..., 2:4]), ch[..., 4]))
        chain = jnp.where(chain_valid[..., None], self.norm("chain_norm", chain), 0)

        up = obs["unpositioned_"].astype(jnp.int32)
        unpos_valid = up[..., 3] > 0
        unpos = (e("unpositioned_location", 16, up[..., 2]) + e("unpositioned_count", 16, up[..., 3])
                 + _lookup(table, dtype=self.config.compute_dtype, ids=_u16(up[..., :2])))
        unpos = jnp.where(unpos_valid[..., None], self.norm("unpositioned_norm", unpos), 0)

        rc = obs["opponent_recipe_"].astype(jnp.int32)
        recipe_rows = (e("recipe_location", 16, rc[..., 2]) + e("recipe_count", 8, rc[..., 3])
                       + _lookup(table, dtype=self.config.compute_dtype, ids=_u16(rc[..., :2])))
        recipe = EffectSetEncoder(c.d, dtype=c.compute_dtype, name="recipe_pool")(recipe_rows, rc[..., 3] > 0)

        # the own recipe (DeckSet): every recipe card with its count, remaining copies and archetype share, pooled
        # permutation-invariantly; library genericity is in the card table
        own = obs["own_recipe_"].astype(jnp.int32)
        own_rows = (e("own_location", 16, own[..., 2]) + e("own_count", 8, own[..., 3])
                    + e("own_remaining", 8, own[..., 4]) + dense("own_share", own[..., 5:6] / 255.0)
                    + _lookup(table, dtype=self.config.compute_dtype, ids=_u16(own[..., :2])))
        own_recipe = EffectSetEncoder(c.d, dtype=c.compute_dtype, name="own_recipe_pool")(own_rows, own[..., 3] > 0)

        act = obs["turn_activations_"].astype(jnp.int32)
        act_rows = (_lookup(table, dtype=self.config.compute_dtype, ids=_u16(act[..., 1:3])) + e("activation_desc", 32, act[..., 3])
                    + self.bound_units("activation_units", table, _u16(act[..., 1:3]), act[..., 3])
                    + dense("activation_counts", act[..., 4:] / 4.0))
        activations = EffectSetEncoder(c.d, dtype=c.compute_dtype, name="activation_pool")(act_rows, act[..., 0] > 0)

        tokens = [state_token[:, None], card_inputs, chain, unpos, recipe[:, None], own_recipe[:, None],
                  activations[:, None]]
        valid = [jnp.ones((b, 1), bool), card_valid, chain_valid, unpos_valid, jnp.ones((b, 3), bool)]
        if c.public_status:  # lingering player effects (Maxx "C" draws, cannot-respond, attack locks): one token each
            pe = obs["public_effects_"].astype(jnp.int32)
            effect_valid = pe[..., 0] > 0
            effects = (_lookup(table, dtype=c.compute_dtype, ids=_u16(pe[..., 1:3])) + e("effect_desc", 32, pe[..., 3])
                       + e("effect_owner", 4, pe[..., 4]) + e("effect_kind", 16, pe[..., 5])
                       + e("effect_expiry", 8, pe[..., 6]) + e("effect_turns", 8, jnp.clip(pe[..., 7], 0, 7)))
            tokens.append(jnp.where(effect_valid[..., None], self.norm("effect_norm", effects), 0))
            valid.append(effect_valid)
        tokens = jnp.concatenate(tokens, axis=1)
        valid = jnp.concatenate(valid, axis=1)

        a = obs["action_ir_"].astype(jnp.int32)
        action_valid = a[..., 0] > 0
        # column 22 (the row's menu position) is not read: candidates are scored by content, never by position
        action_numeric = jnp.concatenate([a[..., 5:14], a[..., 17:18] / 2.0], axis=-1)
        action_base = (e("action_prompt", 32, a[..., 0]) + e("action_act", 16, a[..., 1])
                       + e("action_phase", 8, a[..., 2]) + e("action_role", 8, a[..., 3])
                       + e("action_stage", 16, a[..., 4]) + e("action_effect", 32, a[..., 14])
                       + e("action_position", 16, a[..., 18]) + e("action_place", 32, a[..., 19])
                       + e("action_number", 32, a[..., 20]) + e("action_attribute", 16, a[..., 21])
                       + e("action_mode", 8, a[..., 23]) + dense("action_numeric", action_numeric)
                       + (a[..., 17:18] / 2.0) * (_lookup(table, dtype=self.config.compute_dtype, ids=_u16(a[..., 15:17]))
                                                  + self.bound_units("action_units", table, _u16(a[..., 15:17]),
                                                                     a[..., 14] - 2)))
        if c.hand_limit:  # cards an end-phase move would discard to the hand limit, per menu row
            discard = nn.Embed(16, c.d, embedding_init=nn.initializers.zeros, name="action_discard").embedding
            action_base = action_base + discard[jnp.clip(obs["action_discard_"].astype(jnp.int32), 0, 15)].astype(
                action_base.dtype)
        return card_inputs, tokens, valid, action_base, action_valid


BELIEF_LOCATIONS = (1, 2, 3, 4, 6, 7)  # deck, hand, face-down monster, face-down spell/trap, face-down banished, extra
BELIEF_COUNTS = 4  # 0, 1, 2, 3 or more copies


class BeliefHead(_Fields):
    """The opponent's hidden cards (design S1): each public candidate card (obs:candidates_) reads the state block
    -- through stop-gradient, so it costs the policy nothing -- and predicts, per hidden location, how many copies
    of it are hidden there."""

    @nn.compact
    def __call__(self, state, state_valid, candidates, table):
        c = self.config
        width = c.belief_width
        cand = candidates.astype(jnp.int32)
        valid = cand[..., 2] > 0
        # the shared semantic table is read through stop-gradient too: the belief loss trains only this head
        x = (nn.Dense(width, name="candidate_card")(jax.lax.stop_gradient(_lookup(table, ids=_u16(cand[..., :2]))))
             + nn.Embed(8, width, name="candidate_tier")(jnp.clip(cand[..., 2], 0, 7)))
        memory = nn.Dense(width, name="state_projection")(jax.lax.stop_gradient(state).astype(jnp.float32))
        mask = _key_mask(x.shape[1], state_valid)
        for i in range(c.belief_layers):
            x = CrossBlock(width, c.belief_heads, width * 4, None, name=f"belief_{i}")(x, memory, mask)
        x = nn.LayerNorm(name="belief_norm")(x)
        logits = nn.Dense(len(BELIEF_LOCATIONS) * BELIEF_COUNTS, name="belief_out")(x)
        logits = logits.reshape(x.shape[:2] + (len(BELIEF_LOCATIONS), BELIEF_COUNTS))
        return logits, valid


def belief_targets(candidates, labels):
    """Hidden copies per (candidate, location) from the engine-truth rows (label:hidden_: id hi, id lo, location,
    count), clipped to 3; and how many hidden copies no candidate names (coverage)."""
    cand = _u16(candidates[..., :2].astype(jnp.int32))
    lab = labels.astype(jnp.int32)
    ids, location, count = _u16(lab[..., :2]), lab[..., 2], lab[..., 3]
    same = (cand[:, :, None] == ids[:, None, :]) & (candidates[..., 2][:, :, None] > 0) & (count[:, None, :] > 0)
    where = jnp.stack([location == l for l in BELIEF_LOCATIONS], axis=-1).astype(jnp.int32)  # [B, R, L]
    targets = jnp.einsum("bar,brl->bal", same.astype(jnp.int32) * count[:, None, :], where)
    covered = same.any(axis=1)
    uncovered = jnp.where(covered, 0, count).sum(-1)
    return jnp.clip(targets, 0, BELIEF_COUNTS - 1), uncovered, count.sum(-1)


def action_tokens(owner, cards, action_base, obs):
    """Shared token recipe, keeping the public readout's existing parameter paths unchanged."""
    c, dt = owner.config, owner.config.compute_dtype
    single, single_valid = _gather_rows(cards, obs["action_single_refs_"])
    role = nn.Embed(single.shape[2], c.d, name="single_role").embedding.astype(single.dtype)
    weights = single_valid[..., None].astype(single.dtype)
    single = ((single + role) * weights).sum(axis=2) / jnp.maximum(weights.sum(axis=2), 1)
    group = GroupRoles(c.d, dtype=dt, name="group_roles")(
        cards, obs["action_group_refs_"], obs["action_group_mask_"])
    action = owner.norm("action_input_norm", action_base + owner.dense("action_single", single) + group)
    hidden = nn.gelu(nn.Dense(c.action_hidden, dtype=dt, name="action_up")(action))
    return owner.norm("action_output_norm", action + nn.Dense(c.d, dtype=dt, name="action_down")(hidden))


class MenuTokens(_Fields):
    @nn.compact
    def __call__(self, cards, action_base, obs):
        return action_tokens(self, cards, action_base, obs)


class Readout(_Fields):
    """Action features from the read-out card tokens, then the WorldReadout: each action queries the latent
    (policy logit); learned value queries give win/draw/loss logits."""

    @nn.compact
    def __call__(self, state, state_valid, query, card_count, action_base, action_valid, obs):
        c, dt = self.config, self.config.compute_dtype
        cards = state[:, 1:1 + card_count]
        action = action_tokens(self, cards, action_base, obs)

        latent = jnp.concatenate([state, query[:, None]], axis=1)
        latent_valid = jnp.concatenate([state_valid, jnp.ones((state.shape[0], 1), bool)], axis=1)
        latent = jnp.where(latent_valid[..., None], self.norm("latent_norm", latent), 0)
        action_query = self.norm("action_query_norm", nn.Dense(c.d, dtype=dt, name="action_query")(action))
        context = Attention(c.d, c.heads, dt, name="action_attention")(
            action_query, latent, _key_mask(action.shape[1], latent_valid))
        fused = self.norm("action_fusion_norm", nn.gelu(nn.Dense(c.readout_hidden, dtype=dt, name="action_fusion")(
            jnp.concatenate([action_query, context], axis=-1))))
        logits = nn.Dense(1, dtype=dt, name="policy")(fused)[..., 0].astype(jnp.float32)
        logits = jnp.where(action_valid, logits, NEG)
        queries = self.param("value_queries", nn.initializers.normal(0.02), (c.value_queries, c.d))
        values = Attention(c.d, c.heads, dt, name="value_attention")(
            jnp.broadcast_to(queries.astype(latent.dtype), (latent.shape[0],) + queries.shape), latent,
            _key_mask(c.value_queries, latent_valid))
        if c.value_menu:
            values = values + ValueMenuAttention(c.d, c.heads, dt, name="value_menu")(
                jnp.broadcast_to(queries.astype(latent.dtype), (latent.shape[0],) + queries.shape), action,
                action_valid)
        values = values.reshape(values.shape[0], -1)
        hidden = self.norm("value_norm", nn.gelu(nn.Dense(c.readout_hidden, dtype=dt, name="value_up")(values)))
        wdl = nn.Dense(3, dtype=dt, name="value_out")(hidden).astype(jnp.float32)
        probabilities = jax.nn.softmax(wdl, axis=-1)
        return logits, probabilities[:, 0] - probabilities[:, 2], wdl


class PolicyNet(nn.Module):
    config: PolicyNetConfig
    semantic_shape: Tuple[int, int, int, int]

    def setup(self):
        c, dt = self.config, self.config.compute_dtype
        self.semantics = CardSemantics(c, self.semantic_shape, name="semantics")
        self.event_encoder = EventEncoder(c, name="events")
        self.inputs = InputEncoder(c, name="inputs")
        block = nn.remat(SelfBlock) if c.remat else SelfBlock
        cross = nn.remat(CrossBlock) if c.remat else CrossBlock
        self.state_blocks = [block(c.d, c.heads, c.ff, dt, name=f"state_{i}") for i in range(c.state_layers)]
        self.state_norm = nn.LayerNorm(dtype=dt, name="state_output_norm")
        self.turn_blocks = [block(c.d, c.heads, c.ff, dt, name=f"turn_{i}") for i in range(c.turn_layers)]
        self.turn_norm = nn.LayerNorm(dtype=dt, name="turn_output_norm")
        self.readout_blocks = [cross(c.d, c.heads, c.ff, dt, name=f"readout_{i}") for i in range(c.readout_layers)]
        self.special = self.param("special_tokens", nn.initializers.normal(0.02), (3, c.d))  # BOS, QUERY, SUMMARY
        self.recency = nn.Embed(c.memory_slots, c.d, embedding_init=nn.initializers.normal(0.02), name="recency")
        self.chunk_position = nn.Embed(c.chunk_slots, c.d, embedding_init=nn.initializers.normal(0.02),
                                       name="chunk_position")
        self.summary_projection = nn.Dense(c.d, dtype=dt, name="summary_projection")
        self.summary_norm = nn.LayerNorm(dtype=dt, name="summary_norm")
        self.readout = Readout(c, name="readout")
        self.belief_head = BeliefHead(c, name="belief")

    def semantic_table(self) -> jax.Array:
        return self.semantics()

    def encode(self, obs, table) -> Encoded:
        card_inputs, tokens, valid, action_base, action_valid = self.inputs(obs, table)
        mask = _key_mask(tokens.shape[1], valid)
        for blk in self.state_blocks:
            tokens = blk(tokens, mask)
        state = jnp.where(valid[..., None], self.state_norm(tokens), 0)
        return Encoded(card_inputs, state, valid, action_base, action_valid)

    def turn_pass(self, prefix: Prefix, events, event_valid, kind: int):
        """[BOS, memory, chunks, events, QUERY (kind 1) or SUMMARY (kind 2)] through the turn layers; returns all
        outputs, their validity and the last token's output."""
        b = events.shape[0]
        dtype = events.dtype
        special = self.special.astype(dtype)
        mem = prefix.memory.astype(dtype) + self.recency(
            jnp.clip(prefix.recency, 0, self.config.memory_slots - 1)).astype(dtype)
        mem = jnp.where(prefix.memory_valid[..., None], mem, 0)
        chunk = prefix.chunks.astype(dtype) + self.chunk_position(
            jnp.clip(prefix.chunk_index, 0, self.config.chunk_slots - 1)).astype(dtype)
        chunk = jnp.where(prefix.chunk_valid[..., None], chunk, 0)
        x = jnp.concatenate([jnp.broadcast_to(special[0], (b, 1, special.shape[1])), mem, chunk, events,
                             jnp.broadcast_to(special[kind], (b, 1, special.shape[1]))], axis=1)
        valid = jnp.concatenate([jnp.ones((b, 1), bool), prefix.memory_valid, prefix.chunk_valid, event_valid,
                                 jnp.ones((b, 1), bool)], axis=1)
        mask = _key_mask(x.shape[1], valid)
        for blk in self.turn_blocks:
            x = blk(x, mask)
        x = self.turn_norm(x)
        return x, valid, x[:, -1]

    def summarize(self, prefix: Prefix, rows, refs, card_inputs, table):
        """A summary token of event rows (a completed turn's final window, or a chunk) after ``prefix``."""
        window, window_valid = self.event_encoder(rows, refs, card_inputs, table)
        _, _, out = self.turn_pass(prefix, window, window_valid, kind=2)
        return self.summary_norm(self.summary_projection(out)).astype(jnp.float32)

    def decision_context(self, enc: Encoded, prefix: Prefix, rows, refs, table):
        """The original decision feature computation; factored without moving any module/parameter paths."""
        events, event_valid = self.event_encoder(rows, refs, enc.card_inputs, table)
        turn, turn_valid, query = self.turn_pass(prefix, events, event_valid, kind=1)
        state = enc.state
        mask = _key_mask(state.shape[1], turn_valid)
        for blk in self.readout_blocks:
            state = blk(state, turn, mask)
        state = jnp.where(enc.state_valid[..., None], state, 0)
        return DecisionContext(state, enc.state_valid, query, enc.action_base, enc.action_valid)

    def decide(self, enc: Encoded, prefix: Prefix, rows, refs, obs, table):
        """The decision: QUERY over [BOS, memory, chunks, the current window ``rows``/``refs``], then the readout."""
        ctx = self.decision_context(enc, prefix, rows, refs, table)
        return self.readout(ctx.state, ctx.state_valid, ctx.query, enc.card_inputs.shape[1], ctx.action_base,
                            ctx.action_valid, obs)

    def decide_with_context(self, enc: Encoded, prefix: Prefix, rows, refs, obs, table):
        """Opt-in auxiliary-head path; existing policy callers keep the unchanged three-output decide path."""
        if any(str(key).startswith(("priv", "label")) for key in obs):
            raise ValueError("public decision context refuses private/label inputs")
        ctx = self.decision_context(enc, prefix, rows, refs, table)
        outputs = self.readout(ctx.state, ctx.state_valid, ctx.query, enc.card_inputs.shape[1], ctx.action_base,
                               ctx.action_valid, obs)
        return (*outputs, ctx)

    def belief(self, enc: Encoded, obs, table):
        return self.belief_head(enc.state, enc.state_valid, obs["candidates_"], table)

    def init_all(self, obs, prefix: Prefix):
        """Touches every submodule once (for ``init``)."""
        table = self.semantic_table()
        self.belief(self.encode(obs, table), obs, table)
        enc = self.encode(obs, table)
        summary = self.summarize(prefix, obs["closed_turns_"][:, 0], obs["closed_turn_refs_"][:, 0],
                                 enc.card_inputs, table)
        return self.decide(enc, prefix, obs["turn_events_"], obs["turn_event_refs_"], obs, table), summary


# ----- the two runners --------------------------------------------------------------------------------------------
#
# Per decision of a seat, in this order (both runners):
#   1. the chunk slots of obs:turn_chunks_ (rows of a long turn that left the window, 64 per chunk, delivered once,
#      oldest first) and the new completed turns of obs:closed_turns_ (exported once, after all their chunks), in
#      chronological order: chunks of a turn before that turn's closed window, the older completed turn first;
#   2. a chunk summary = SUMMARY over [BOS, memory, the turn's earlier chunk summaries, the chunk's rows]; it joins
#      the seat's chunk carry (a new turn's first chunk clears the carry);
#   3. a turn summary = SUMMARY over [BOS, memory, that turn's chunk summaries, its final window]; it joins the
#      memory, and the turn's chunk carry is cleared;
#   4. the decision: QUERY over [BOS, memory, the current turn's chunk summaries, the current window].
# Inconsistent deliveries (a chunk index or a closed turn's chunk count that does not match the carry, more chunks
# than chunk_slots) raise on the host. Deliveries are idempotent: a completed turn already summarized and a chunk
# already held (or of a summarized turn) are skipped, so the observation of a withdrawn decision, which the
# environment re-presents exactly (illegal_activation_withdrawal/v1), leaves the recurrent state as if it had been
# dropped.

def _prefix(arrays, chunk_arrays, env, seat, length, floor, chunk_length, chunk_start, config: "PolicyNetConfig"):
    """The turn-sequence prefix of rows (``env``, ``seat``) [N] from chronological arrays [B, 2, size, d]: the memory
    window of ``memory_slots`` items ending at ``length`` (items before ``floor`` not visible; recency 0 = newest) and
    the current turn's chunk summaries, items [chunk_start, chunk_length) in a window of ``chunk_slots`` (with their
    index within the turn). Lengths are at least the window sizes (the carried slots come first)."""
    slots, cs = config.memory_slots, config.chunk_slots
    index = length[:, None] - slots + jnp.arange(slots)[None, :]
    mem = arrays[env[:, None], seat[:, None], index]
    recency = jnp.broadcast_to(slots - 1 - jnp.arange(slots), index.shape)
    cindex = chunk_length[:, None] - cs + jnp.arange(cs)[None, :]
    chunks = chunk_arrays[env[:, None], seat[:, None], cindex]
    return Prefix(mem, index >= floor[:, None], recency, chunks, cindex >= chunk_start[:, None],
                  cindex - chunk_start[:, None])


def _shapes(config: "PolicyNetConfig", rows: int):
    """The turn-pass shapes (event rows, chunk slots): each row block without chunks, and the full window with all
    chunk slots (chunks exist only past a full window, so no smaller shape needs them)."""
    shapes = [(b, 0) for b in config.row_blocks if b < rows] + [(rows, 0)]
    return shapes + [(rows, config.chunk_slots)] if config.chunk_slots else shapes


def _shaped(fn, config: "PolicyNetConfig", prefix: Prefix, rows, active: jax.Array, operands, remat=False,
            *, full_shapes=False):
    """``fn(prefix, rows_block, *operands)`` on the smallest shape holding every active row: its valid window rows
    (left-aligned) and visible chunk summaries. A device-side switch: one program, the chosen branch runs."""
    shapes = _shapes(config, rows.shape[1])
    held = jnp.where(active, (rows[..., 0] > 0).sum(-1), 0).max()
    chunked = (active[:, None] & prefix.chunk_valid).any()  # never true without chunk slots
    unchunked = jnp.asarray([b for b, c in shapes if c == 0])
    branch = jnp.where(chunked, len(shapes) - 1, jnp.sum(unchunked[:-1] < held))

    def run(shape):
        block, chunks = shape

        def body(op):
            pre, window, rest = op
            cut = pre._replace(chunks=pre.chunks[:, pre.chunks.shape[1] - chunks:],
                               chunk_valid=pre.chunk_valid[:, pre.chunk_valid.shape[1] - chunks:],
                               chunk_index=pre.chunk_index[:, pre.chunk_index.shape[1] - chunks:])
            return fn(cut, window[:, :block], *rest)
        return jax.checkpoint(body) if remat else body
    if full_shapes:
        # Explicit inference-only geometry. Retain every validity mask; only
        # computation padding becomes independent of other batch members.
        return run((rows.shape[1], config.chunk_slots))((prefix, rows, operands))
    if len(shapes) == 1:
        return run(shapes[0])((prefix, rows, operands))
    return jax.lax.switch(branch.astype(jnp.int32), [run(shape) for shape in shapes], (prefix, rows, operands))


def _summary(model, variables, prefix: Prefix, rows, refs, card_inputs, table, active: jax.Array, remat=False,
             *, full_shapes=False):
    """Summaries of event windows ``rows`` [B, K, 24] (rows left-aligned) after ``prefix``; ``active`` [B] marks the
    windows that are used (see ``_shaped``). ``remat``: the pass is recomputed in the backward pass, so a scan of
    summaries keeps only each pass's inputs."""
    fn = lambda pre, window, refs, card_inputs: model.apply(variables, pre, window, refs[:, :window.shape[1]],
                                                             card_inputs, table, method=PolicyNet.summarize)
    return _shaped(fn, model.config, prefix, rows, active, (refs, card_inputs), remat, full_shapes=full_shapes)


def _decision(model, variables, enc: Encoded, prefix: Prefix, obs, table, return_context: bool = False,
              *, full_shapes=False):
    """Logits, value and win/draw/loss of every row (see ``_shaped``; all rows are active)."""
    rows = obs["turn_events_"]
    method = PolicyNet.decide_with_context if return_context else PolicyNet.decide
    fn = lambda pre, window, refs, enc, obs: model.apply(variables, enc, pre, window, refs[:, :window.shape[1]],
                                                         obs, table, method=method)
    return _shaped(fn, model.config, prefix, rows, jnp.ones(rows.shape[:1], bool),
                   (obs["turn_event_refs_"], enc, obs), full_shapes=full_shapes)


def _check(flag):
    if bool(np.asarray(flag).any()):
        raise RuntimeError("policy_net: inconsistent chunk or closed-turn delivery (chunk index, a closed turn's chunk "
                           "count, the chunk cap) or a segment beyond its entry bound")


def reset(memory: Memory, first: jax.Array) -> Memory:
    """Both seats' memory and chunk carry cleared where a new episode starts (``first`` [B])."""
    keep = ~first
    k2, k4 = keep[:, None], keep[:, None, None, None]
    return Memory(jnp.where(k4, memory.buffer, 0), jnp.where(k2, memory.count, 0), jnp.where(k2, memory.last_turn, -1),
                  jnp.where(k4, memory.chunks, 0), jnp.where(k2, memory.chunk_count, 0),
                  jnp.where(k2, memory.chunk_turn, -1))


def _deliveries(obs):
    """Chunk slots [B, S, 4] (valid, turn, index, rows), oldest first, and closed slots [B, 2, 5] (valid, turn, rows,
    chunks, turn player), oldest first."""
    return obs["turn_chunk_meta_"].astype(jnp.int32), obs["closed_turn_meta_"].astype(jnp.int32)


def _op_order(chunk_meta, closed_meta, last):
    """The static order of a step's operations (see the header) with their masks: a list of ("chunk", j, mask) and
    ("closed", slot, mask); ``last`` [B] is the seat's last summarized turn before the step."""
    sd = chunk_meta.shape[1]
    done = [jnp.zeros(last.shape, bool) for _ in range(sd)]
    order = []
    for slot in (0, 1):  # the environment's order: the older completed turn in slot 0
        turn = closed_meta[:, slot, 1]
        new = (closed_meta[:, slot, 0] > 0) & (turn > last)
        for j in range(sd):
            take = (chunk_meta[:, j, 0] > 0) & ~done[j] & new & (chunk_meta[:, j, 1] <= turn)
            done[j] = done[j] | take
            order.append(("chunk", j, take))
        order.append(("closed", slot, new))
        last = jnp.where(new, turn, last)
    for j in range(sd):
        order.append(("chunk", j, (chunk_meta[:, j, 0] > 0) & ~done[j]))
    return order


def act(model: PolicyNet, variables, obs, seat: jax.Array, memory: Memory, first: Optional[jax.Array] = None,
        *, return_context: bool = False, full_shapes: bool = False, rematerialize_history: bool = False):
    """One decision per environment. ``seat`` [B] is the player to move (info:to_play) and ``first`` [B] marks the
    first decision of a new episode (both seats' memory is cleared first). Returns logits, value, win/draw/loss logits,
    the updated memory and an inconsistency flag [B] (the caller checks it on the host). With the static opt-in
    ``return_context``, append the public DecisionContext for auxiliary heads; default outputs are unchanged."""
    if type(rematerialize_history) is not bool:
        raise ValueError('history-operation recomputation is an explicit static execution option')
    c = model.config
    slots, cs = c.memory_slots, c.chunk_slots
    if first is not None:
        memory = reset(memory, first)
    rows = jnp.arange(seat.shape[0])
    buffer, count, last = memory.buffer[rows, seat], memory.count[rows, seat], memory.last_turn[rows, seat]
    chunks, ccount, cturn = memory.chunks[rows, seat], memory.chunk_count[rows, seat], memory.chunk_turn[rows, seat]
    table = model.apply(variables, method=PolicyNet.semantic_table)
    enc = model.apply(variables, obs, table, method=PolicyNet.encode)
    chunk_meta, closed_meta = _deliveries(obs)
    full, cfull, zero = jnp.full_like(count, slots), jnp.full_like(count, cs), jnp.zeros_like(count)
    bad = jnp.zeros(seat.shape, bool)

    def prefix(buffer, count, chunks, start):
        return _prefix(buffer[:, None], chunks[:, None], rows, zero, full, slots - jnp.minimum(count, slots), cfull,
                       start, c)

    order = _op_order(chunk_meta, closed_meta, last)
    kinds = jnp.asarray([1 if kind == "chunk" else 0 for kind, _, _ in order], jnp.int32)
    slots_of = jnp.asarray([k for _, k, _ in order], jnp.int32)
    masks = jnp.stack([mask for _, _, mask in order])
    take = lambda x, k: jnp.take(x, k, axis=1)

    def chunk_op(state, k, mask):
        buffer, count, last, chunks, ccount, cturn, bad = state
        turn, index = take(chunk_meta, k)[:, 1], take(chunk_meta, k)[:, 2]
        # a chunk delivered again (the environment re-presents a withdrawn decision's observation exactly) is
        # already held, or belongs to a turn already summarized: skipped, so delivery is idempotent
        mask = mask & ~((turn <= last) | ((turn == cturn) & (index < ccount)))
        restart = mask & (turn != cturn)
        held = jnp.where(restart, 0, ccount)
        bad = bad | (mask & ((index != held) | (held >= cs)))
        summary = jax.lax.cond(
            mask.any(),
            lambda _: _summary(model, variables, prefix(buffer, count, chunks, cs - held), take(obs["turn_chunks_"], k),
                               take(obs["turn_chunk_refs_"], k), enc.card_inputs, table, mask, full_shapes=full_shapes),
            lambda _: jnp.zeros_like(chunks[:, 0]), None)
        base = jnp.where(restart[:, None, None], 0, chunks)
        shifted = jnp.concatenate([base[:, 1:], summary[:, None]], axis=1)
        chunks = jnp.where(mask[:, None, None], shifted, chunks)
        ccount = jnp.where(mask, held + 1, ccount)
        cturn = jnp.where(mask, turn, cturn)
        return buffer, count, last, chunks, ccount, cturn, bad

    def closed_op(state, k, mask):
        buffer, count, last, chunks, ccount, cturn, bad = state
        turn, expected = take(closed_meta, k)[:, 1], take(closed_meta, k)[:, 3]
        same = cturn == turn
        held = jnp.where(same, ccount, 0)
        bad = bad | (mask & (held != expected))
        summary = jax.lax.cond(
            mask.any(),
            lambda _: _summary(model, variables, prefix(buffer, count, chunks, cs - held),
                               take(obs["closed_turns_"], k), take(obs["closed_turn_refs_"], k), enc.card_inputs,
                               table, mask, full_shapes=full_shapes),
            lambda _: jnp.zeros_like(buffer[:, 0]), None)
        shifted = jnp.concatenate([buffer[:, 1:], summary[:, None]], axis=1)
        buffer = jnp.where(mask[:, None, None], shifted, buffer)
        count = count + mask.astype(jnp.int32)
        last = jnp.where(mask, turn, last)
        clear = mask & same
        chunks = jnp.where(clear[:, None, None], 0, chunks)
        ccount = jnp.where(clear, 0, ccount)
        cturn = jnp.where(clear, -1, cturn)
        return buffer, count, last, chunks, ccount, cturn, bad

    def operation(state, x):
        # one copy of each summary pass in the program, whatever the number of delivery slots
        kind, k, mask = x
        return jax.lax.switch(kind, [closed_op, chunk_op], state, k, mask), None

    (buffer, count, last, chunks, ccount, cturn, bad), _ = jax.lax.scan(
        jax.checkpoint(operation) if rematerialize_history else operation,
        (buffer, count, last, chunks, ccount, cturn, bad), (kinds, slots_of, masks))
    decision = _decision(model, variables, enc, prefix(buffer, count, chunks, cs - ccount), obs, table, return_context,
                         full_shapes=full_shapes)
    logits, value, wdl = decision[:3]
    memory = Memory(memory.buffer.at[rows, seat].set(buffer), memory.count.at[rows, seat].set(count),
                    memory.last_turn.at[rows, seat].set(last), memory.chunks.at[rows, seat].set(chunks),
                    memory.chunk_count.at[rows, seat].set(ccount), memory.chunk_turn.at[rows, seat].set(cturn))
    out = (logits, value, wdl, memory, bad)
    return (*out, decision[3]) if return_context else out


class SegmentPlan(NamedTuple):
    """Integer bookkeeping of a segment. Entries are each environment's summary operations in chronological order;
    positions are in each seat's chronological arrays: memory [carried slots, new summaries] and chunks [carried
    chunk slots, new chunk summaries]."""
    step: jax.Array  # [B, M] the step of each operation
    kind: jax.Array  # [B, M] 0: a completed turn, 1: a chunk
    slot: jax.Array  # [B, M] its closed_turns_ or turn_chunks_ index at that step
    seat: jax.Array  # [B, M]
    length: jax.Array  # [B, M] the seat's memory length (the new summary's position for a completed turn)
    floor: jax.Array  # [B, M] the seat's episode floor
    chunk_length: jax.Array  # [B, M] the seat's chunk length (the new summary's position for a chunk)
    chunk_start: jax.Array  # [B, M] the first chunk the operation sees
    valid: jax.Array  # [B, M]
    step_length: jax.Array  # [T, B] the acting seat's memory length after step t's operations
    step_floor: jax.Array  # [T, B]
    step_chunk_length: jax.Array  # [T, B]
    step_chunk_start: jax.Array  # [T, B]
    bad: jax.Array  # [B] an inconsistent delivery or more operations than M (raises on the host)
    last_turn: jax.Array  # [B, 2] after the segment
    length_end: jax.Array  # [B, 2]
    floor_end: jax.Array  # [B, 2]
    chunk_length_end: jax.Array  # [B, 2]
    chunk_start_end: jax.Array  # [B, 2]
    chunk_turn_end: jax.Array  # [B, 2]
    reset: jax.Array  # [B] an episode started inside the segment


def plan_segment(chunk_meta: jax.Array, closed_meta: jax.Array, seat: jax.Array, first: jax.Array, memory: Memory,
                 config: "PolicyNetConfig", entries: int) -> SegmentPlan:
    """Which summary operations each environment runs inside a segment, in order, and what each step sees."""
    slots, cs = config.memory_slots, config.chunk_slots
    batch = seat.shape[1]
    rows = jnp.arange(batch)
    chunk_meta, closed_meta = chunk_meta.astype(jnp.int32), closed_meta.astype(jnp.int32)
    names = ("step", "kind", "slot", "seat", "length", "floor", "chunk_length", "chunk_start")

    def body(carry, x):
        state, n, bad, reset_seen, record = carry
        last, length, floor, clen, cstart, cturn = state
        chunk_t, closed_t, seat_t, first_t, t = x
        f = first_t[:, None]
        last = jnp.where(f, -1, last)
        floor = jnp.where(f, length, floor)
        cstart = jnp.where(f, clen, cstart)
        cturn = jnp.where(f, -1, cturn)
        reset_seen = reset_seen | first_t
        get = lambda a: a[rows, seat_t]
        put = lambda a, mask, value: a.at[rows, seat_t].set(jnp.where(mask, value, get(a)))

        def append(record, n, mask, values):
            at = jnp.clip(n, 0, entries - 1)
            take = mask & (n < entries)
            record = {name: record[name].at[rows, at].set(jnp.where(take, values[name], record[name][rows, at]))
                      for name in names} | {"valid": record["valid"].at[rows, at].set(record["valid"][rows, at] | take)}
            return record, n + mask.astype(jnp.int32)

        for kind, k, mask in _op_order(chunk_t, closed_t, get(last)):
            if kind == "chunk":
                turn, index = chunk_t[:, k, 1], chunk_t[:, k, 2]
                mask = mask & ~((turn <= get(last)) | ((turn == get(cturn)) & (index < get(clen) - get(cstart))))
                restart = mask & (turn != get(cturn))
                cstart = put(cstart, restart, get(clen))
                held = get(clen) - get(cstart)
                bad = bad | (mask & ((index != held) | (held >= cs)))
                values = {"step": t, "kind": 1, "slot": k, "seat": seat_t, "length": get(length), "floor": get(floor),
                          "chunk_length": get(clen), "chunk_start": get(cstart)}
                record, n = append(record, n, mask, values)
                clen = put(clen, mask, get(clen) + 1)
                cturn = put(cturn, mask, turn)
            else:
                turn, expected = closed_t[:, k, 1], closed_t[:, k, 3]
                same = get(cturn) == turn
                start = jnp.where(same, get(cstart), get(clen))
                bad = bad | (mask & (get(clen) - start != expected))
                values = {"step": t, "kind": 0, "slot": k, "seat": seat_t, "length": get(length), "floor": get(floor),
                          "chunk_length": get(clen), "chunk_start": start}
                record, n = append(record, n, mask, values)
                length = put(length, mask, get(length) + 1)
                last = put(last, mask, turn)
                clear = mask & same
                cstart = put(cstart, clear, get(clen))
                cturn = put(cturn, clear, -1)
        out = (get(length), get(floor), get(clen), get(cstart))
        return ((last, length, floor, clen, cstart, cturn), n, bad, reset_seen, record), out

    zeros = jnp.zeros((batch, entries), jnp.int32)
    record = {name: zeros for name in names} | {"valid": jnp.zeros((batch, entries), bool)}
    state = (memory.last_turn, jnp.full((batch, 2), slots, jnp.int32), slots - jnp.minimum(memory.count, slots),
             jnp.full((batch, 2), cs, jnp.int32), cs - memory.chunk_count, memory.chunk_turn)
    init = (state, jnp.zeros((batch,), jnp.int32), jnp.zeros((batch,), bool), jnp.zeros((batch,), bool), record)
    carry, (step_length, step_floor, step_clen, step_cstart) = jax.lax.scan(
        body, init, (chunk_meta, closed_meta, seat, first, jnp.arange(seat.shape[0])))
    (last, length, floor, clen, cstart, cturn), n, bad, reset_seen, record = carry
    return SegmentPlan(record["step"], record["kind"], record["slot"], record["seat"], record["length"],
                       record["floor"], record["chunk_length"], record["chunk_start"], record["valid"], step_length,
                       step_floor, step_clen, step_cstart, bad | (n > entries), last, length, floor, clen, cstart,
                       cturn, reset_seen)


def segment_entries(obs) -> int:
    """The most summary operations a segment can hold per environment: at most every chunk slot and both closed
    slots at each step (one seat acts per step), so the plan can never run out of entries."""
    steps = obs["closed_turn_meta_"].shape[0]
    return steps * (obs["turn_chunk_meta_"].shape[2] + obs["closed_turn_meta_"].shape[2])


def _entry_summaries(model, params, consts, table, card_inputs, arrays, chunk_arrays, plan_arrays, windows, m):
    """Entry m's two possible summaries (a completed turn, a chunk) for every environment, read from the arrays,
    with their masks and write positions."""
    c = model.config
    step, kind, slot, seat_m, length, floor, chunk_length, chunk_start, valid = (x[:, m] for x in plan_arrays)
    closed, closed_refs, chunks, chunk_refs = windows
    rows = jnp.arange(step.shape[0])
    variables = {"params": params, **consts}
    pre = _prefix(arrays, chunk_arrays, rows, seat_m, length, floor, chunk_length, chunk_start, c)
    is_turn, is_chunk = valid & (kind == 0), valid & (kind == 1)

    def summarize(window, refs, active):
        return jax.lax.cond(
            active.any(),
            lambda _: _summary(model, variables, pre, window[step, rows, slot], refs[step, rows, slot],
                               card_inputs[step, rows], table, active),
            lambda _: jnp.zeros((rows.shape[0], c.d), jnp.float32), None)
    return (summarize(closed, closed_refs, is_turn), summarize(chunks, chunk_refs, is_chunk), is_turn, is_chunk,
            seat_m, length, chunk_length)


def _segment_summaries_forward(model, params, consts, table, card_inputs, arrays, chunk_arrays, plan_arrays, windows):
    entries = plan_arrays[0].shape[1]
    rows = jnp.arange(plan_arrays[0].shape[0])
    valid = plan_arrays[-1]

    def operation(carry, m):
        arrays, chunk_arrays = carry
        turn, chunk, is_turn, is_chunk, seat_m, length, chunk_length = _entry_summaries(
            model, params, consts, table, card_inputs, arrays, chunk_arrays, plan_arrays, windows, m)
        at = jnp.clip(length, 0, arrays.shape[2] - 1)
        arrays = arrays.at[rows, seat_m, at].set(jnp.where(is_turn[:, None], turn, arrays[rows, seat_m, at]))
        cat = jnp.clip(chunk_length, 0, chunk_arrays.shape[2] - 1)
        chunk_arrays = chunk_arrays.at[rows, seat_m, cat].set(
            jnp.where(is_chunk[:, None], chunk, chunk_arrays[rows, seat_m, cat]))
        return arrays, chunk_arrays

    def operation_or_skip(carry, m):
        # entries are filled in order, so once no environment of this batch has entry m none has a later one: the
        # operation is skipped (a real branch: the predicate is one scalar for the batch)
        return jax.lax.cond(valid[:, m].any(), operation, lambda c_, _: c_, carry, m), None

    (arrays, chunk_arrays), _ = jax.lax.scan(operation_or_skip, (arrays, chunk_arrays), jnp.arange(entries))
    return arrays, chunk_arrays


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _segment_summaries(model, params, consts, table, card_inputs, arrays, chunk_arrays, plan_arrays, windows):
    """The summaries created inside a segment, written into each seat's chronological memory and chunk arrays.

    A custom VJP: the forward pass skips empty entries and saves nothing per entry; the backward pass walks the
    entries in reverse, recomputes each used entry's summaries from the final arrays (an entry reads only positions
    written before it, which the final arrays hold unchanged) and backpropagates through them, skipping empty
    entries too. Differentiating the scan directly would stack every branch's residuals (and loop-invariant inputs
    passed through the skip branch) for every entry, used or not."""
    return _segment_summaries_forward(model, params, consts, table, card_inputs, arrays, chunk_arrays, plan_arrays,
                                      windows)


def _segment_summaries_fwd(model, params, consts, table, card_inputs, arrays, chunk_arrays, plan_arrays, windows):
    out = _segment_summaries_forward(model, params, consts, table, card_inputs, arrays, chunk_arrays, plan_arrays,
                                     windows)
    return out, (params, consts, table, card_inputs, out, plan_arrays, windows)


def _float0(x):
    return np.zeros(x.shape, dtype=jax.dtypes.float0)


def _segment_summaries_bwd(model, residuals, cotangents):
    params, consts, table, card_inputs, (final, final_chunks), plan_arrays, windows = residuals
    d_arrays, d_chunks = cotangents
    entries = plan_arrays[0].shape[1]
    rows = jnp.arange(plan_arrays[0].shape[0])
    valid = plan_arrays[-1]
    zeros = lambda tree: jax.tree_util.tree_map(jnp.zeros_like, tree)

    def back(carry, m):
        def step_back(carry):
            d_arrays, d_chunks, d_params, d_table, d_cards = carry
            seat_m, length, chunk_length = plan_arrays[3][:, m], plan_arrays[4][:, m], plan_arrays[6][:, m]
            at = jnp.clip(length, 0, final.shape[2] - 1)
            cat = jnp.clip(chunk_length, 0, final_chunks.shape[2] - 1)

            def summaries(params, table, card_inputs, final, final_chunks):
                turn, chunk, is_turn, is_chunk = _entry_summaries(
                    model, params, consts, table, card_inputs, final, final_chunks, plan_arrays, windows, m)[:4]
                return jnp.where(is_turn[:, None], turn, 0), jnp.where(is_chunk[:, None], chunk, 0)
            _, vjp = jax.vjp(summaries, params, table, card_inputs, final, final_chunks)
            g_turn, g_chunk = d_arrays[rows, seat_m, at], d_chunks[rows, seat_m, cat]
            gp, gt, gc, ga, gch = vjp((g_turn, g_chunk))
            # the written slots' cotangents are consumed here (their summaries are recomputed, not stored inputs)
            is_turn = plan_arrays[8][:, m] & (plan_arrays[1][:, m] == 0)
            is_chunk = plan_arrays[8][:, m] & (plan_arrays[1][:, m] == 1)
            d_arrays = d_arrays.at[rows, seat_m, at].set(jnp.where(is_turn[:, None], 0, d_arrays[rows, seat_m, at]))
            d_chunks = d_chunks.at[rows, seat_m, cat].set(jnp.where(is_chunk[:, None], 0, d_chunks[rows, seat_m, cat]))
            add = lambda a, b: jax.tree_util.tree_map(jnp.add, a, b)
            return add(d_arrays, ga), add(d_chunks, gch), add(d_params, gp), add(d_table, gt), add(d_cards, gc)
        return jax.lax.cond(valid[:, m].any(), step_back, lambda c_: c_, carry), None

    init = (d_arrays, d_chunks, zeros(params), zeros(table), zeros(card_inputs))
    (d_arrays, d_chunks, d_params, d_table, d_cards), _ = jax.lax.scan(back, init, jnp.arange(entries), reverse=True)
    return (d_params, zeros(consts), d_table, d_cards, d_arrays, d_chunks,
            jax.tree_util.tree_map(_float0, plan_arrays), jax.tree_util.tree_map(_float0, windows))


_segment_summaries.defvjp(_segment_summaries_fwd, _segment_summaries_bwd)


def _segment_memory(model, variables, obs, seat, memory: Memory, first, table, card_inputs):
    """The summaries created inside a segment (chronological memory and chunk arrays), the plan and each step's
    prefix [T * B]: the acting seat's memory window and current chunks after the step's operations."""
    c = model.config
    entries = segment_entries(obs)
    steps, batch = seat.shape
    if first is None:
        first = jnp.zeros((steps, batch), bool)
    card_inputs = card_inputs.reshape((steps, batch) + card_inputs.shape[1:])
    plan = plan_segment(obs["turn_chunk_meta_"], obs["closed_turn_meta_"], seat, first, memory, c, entries)
    pad = lambda carried, n: jnp.concatenate([carried, jnp.zeros(carried.shape[:2] + (n, c.d), carried.dtype)], 2)
    arrays, chunk_arrays = pad(memory.buffer, entries), pad(memory.chunks, entries)
    params = variables["params"]
    consts = {k: v for k, v in variables.items() if k != "params"}
    plan_arrays = (plan.step, plan.kind, plan.slot, plan.seat, plan.length, plan.floor, plan.chunk_length,
                   plan.chunk_start, plan.valid)
    windows = (obs["closed_turns_"], obs["closed_turn_refs_"], obs["turn_chunks_"], obs["turn_chunk_refs_"])
    arrays, chunk_arrays = _segment_summaries(model, params, consts, table, card_inputs, arrays, chunk_arrays,
                                              plan_arrays, windows)
    env = jnp.broadcast_to(jnp.arange(batch)[None, :], (steps, batch)).reshape(-1)
    pre = _prefix(arrays, chunk_arrays, env, seat.reshape(-1), plan.step_length.reshape(-1),
                  plan.step_floor.reshape(-1), plan.step_chunk_length.reshape(-1),
                  plan.step_chunk_start.reshape(-1), c)
    return arrays, chunk_arrays, plan, pre


def segment_prefix(model: PolicyNet, variables, obs, seat: jax.Array, memory: Memory, first: Optional[jax.Array] = None):
    """Phase one of the kept-rows learner (``mirrorforce/agent/train/a0_kept.py``): ``segment``'s summaries and each
    step's prefix [T * B], without the per-decision passes; only the input encoder runs on every row (the event rows
    refer to its card inputs). Returns (prefix, plan)."""
    steps, batch = seat.shape
    flat = jax.tree_util.tree_map(lambda x: x.reshape((steps * batch,) + x.shape[2:]), obs)
    table = model.apply(variables, method=PolicyNet.semantic_table)
    card_inputs = model.apply(variables, flat, table, method=lambda m, o, t: m.inputs(o, t)[0])
    _, _, plan, pre = _segment_memory(model, variables, obs, seat, memory, first, table, card_inputs)
    return pre, plan


def rows_forward(model: PolicyNet, variables, obs, prefix: Prefix, belief: bool = False, *, return_context: bool = False):
    """Phase two of the kept-rows learner: the per-decision passes of rows ``obs`` [N, ...] after their prefixes
    (from ``segment_prefix``): logits, value, win/draw/loss and, with ``belief``, the belief head's outputs.
    Static ``return_context`` appends public actor features; the default path/parameter tree is unchanged."""
    table = model.apply(variables, method=PolicyNet.semantic_table)
    enc = model.apply(variables, obs, table, method=PolicyNet.encode)
    decision = _decision(model, variables, enc, prefix, obs, table, return_context)
    out = decision[:3]
    if belief:
        out = (*out, model.apply(variables, enc, obs, table, method=PolicyNet.belief))
    return (*out, decision[3]) if return_context else out


def segment(model: PolicyNet, variables, obs, seat: jax.Array, memory: Memory, first: Optional[jax.Array] = None,
            belief: bool = False):
    """The learner's forward over a segment: ``obs`` leaves [T, B, ...], ``seat`` [T, B], ``first`` [T, B] (a new
    episode starts at that step), ``memory`` the carry at the segment start. Returns logits [T, B, A], value [T, B],
    wdl [T, B, 3], the memory after the segment and the plan (its ``bad`` must be false; the caller checks it on the
    host); with ``belief`` also the belief head's (logits [T, B, A, L, C], candidate validity [T, B, A])."""
    c = model.config
    slots, cs = c.memory_slots, c.chunk_slots
    steps, batch = seat.shape
    flat = jax.tree_util.tree_map(lambda x: x.reshape((steps * batch,) + x.shape[2:]), obs)
    table = model.apply(variables, method=PolicyNet.semantic_table)
    enc = model.apply(variables, flat, table, method=PolicyNet.encode)
    arrays, chunk_arrays, plan, pre = _segment_memory(model, variables, obs, seat, memory, first, table,
                                                      enc.card_inputs)
    logits, value, wdl = _decision(model, variables, enc, pre, flat, table)

    # the carry after the segment: each seat's last ``slots`` summaries, right-aligned, and its current chunks
    index = plan.length_end[..., None] - slots + jnp.arange(slots)[None, None, :]
    buffer = jnp.take_along_axis(arrays, index[..., None], axis=2)
    stale = plan.reset[:, None, None] & (index < plan.floor_end[..., None])  # as act: cleared at an episode start
    buffer = jnp.where(stale[..., None], 0, buffer)
    count = jnp.where(plan.reset[:, None], plan.length_end - plan.floor_end, memory.count + plan.length_end - slots)
    cindex = plan.chunk_length_end[..., None] - cs + jnp.arange(cs)[None, None, :]
    chunks = jnp.take_along_axis(chunk_arrays, cindex[..., None], axis=2)
    chunks = jnp.where((cindex >= plan.chunk_start_end[..., None])[..., None], chunks, 0)
    after = Memory(buffer, count, plan.last_turn, chunks, plan.chunk_length_end - plan.chunk_start_end,
                   plan.chunk_turn_end)
    shape = lambda x: x.reshape((steps, batch) + x.shape[1:])
    if belief:
        belief_logits, belief_valid = model.apply(variables, enc, flat, table, method=PolicyNet.belief)
        return shape(logits), shape(value), shape(wdl), after, plan, (shape(belief_logits), shape(belief_valid))
    return shape(logits), shape(value), shape(wdl), after, plan


def parameter_count(variables) -> int:
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(variables["params"]))
