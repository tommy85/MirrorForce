from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from mirrorforce.agent.model.decision import RoleBinding, role_bound_features
from mirrorforce.agent.model.modules import decode_id
from mirrorforce.agent.model.transformer import (
    EncoderLayer,
    MlpBlock,
    MultiheadAttention,
    PositionalEncoding,
)


def _signed_u16(high: jax.Array, low: jax.Array) -> jax.Array:
    value = (high.astype(jnp.int32) << 8) + low.astype(jnp.int32)
    return jnp.where(value >= 32768, value - 65536, value)


def _gather_refs(
    card_tokens: jax.Array, refs: jax.Array
) -> tuple[jax.Array, jax.Array]:
    refs = refs.astype(jnp.int32)
    valid = refs > 0
    safe = jnp.clip(refs - 1, 0, card_tokens.shape[1] - 1)
    batch = jnp.arange(card_tokens.shape[0]).reshape(
        (card_tokens.shape[0],) + (1,) * (refs.ndim - 1)
    )
    gathered = card_tokens[batch, safe]
    return gathered, valid


class ExactSemanticEncoder(nn.Module):
    channels: int
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, values: jax.Array) -> jax.Array:
        values = values.astype(self.dtype or jnp.float32)
        values = nn.Dense(
            self.channels * 2,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="up",
        )(values)
        values = nn.gelu(values)
        return nn.Dense(
            self.channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="down",
        )(values)


class EffectSetEncoder(nn.Module):
    channels: int
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, units: jax.Array, mask: jax.Array) -> jax.Array:
        units = units.astype(self.dtype or jnp.float32)
        tokens = nn.Dense(
            self.channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="projection",
        )(units)
        tokens = nn.gelu(tokens)
        scores = nn.Dense(
            1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="score",
        )(tokens)[..., 0]
        mask = mask.astype(jnp.bool_)
        empty = ~mask.any(axis=-1)
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
        scores = jnp.where(empty[..., None], 0, scores)
        weights = jax.nn.softmax(scores, axis=-1)
        weights = jnp.where(mask, weights, 0)
        pooled = (tokens * weights[..., None]).sum(axis=-2)
        return jnp.where(empty[..., None], 0, pooled)


class CrossAttentionBlock(nn.Module):
    channels: int
    num_heads: int
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        query: jax.Array,
        memory: jax.Array,
        memory_padding_mask: jax.Array,
    ) -> jax.Array:
        normalized = nn.LayerNorm(dtype=self.dtype, name="query_norm")(query)
        context = MultiheadAttention(
            features=self.channels,
            num_heads=self.num_heads,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="cross_attention",
        )(
            normalized,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
        )
        query = query + context
        residual = MlpBlock(
            intermediate_size=self.channels * 4,
            activation="gelu",
            dtype=self.dtype or jnp.float32,
            param_dtype=self.param_dtype,
            name="ffn",
        )(nn.LayerNorm(dtype=self.dtype, name="ffn_norm")(query))
        return query + residual


class RoleSetAttention(nn.Module):
    channels: int
    num_roles: int = 5
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        card_tokens: jax.Array,
        refs: jax.Array,
        mask: jax.Array,
    ) -> jax.Array:
        gathered, ref_valid = _gather_refs(card_tokens, refs)
        valid = mask.astype(jnp.bool_) & ref_valid
        role_ids = jnp.arange(self.num_roles, dtype=jnp.int32)
        role_tokens = nn.Embed(
            self.num_roles,
            self.channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="role_embedding",
        )(role_ids)
        shaped_roles = role_tokens.reshape(
            (1, 1, self.num_roles, 1, self.channels)
        )
        scores = nn.Dense(
            1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="member_score",
        )(jnp.tanh(gathered + shaped_roles))[..., 0]
        empty = ~valid.any(axis=-1)
        scores = jnp.where(valid, scores, jnp.finfo(scores.dtype).min)
        scores = jnp.where(empty[..., None], 0, scores)
        weights = jax.nn.softmax(scores, axis=-1)
        weights = jnp.where(valid, weights, 0)
        pooled = (gathered * weights[..., None]).sum(axis=-2)
        pooled = jnp.where(empty[..., None], 0, pooled)
        pooled = pooled.reshape(
            pooled.shape[0], pooled.shape[1], self.num_roles * self.channels
        )
        return nn.Dense(
            self.channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="role_projection",
        )(pooled)


class StructuredEncoder(nn.Module):
    channels: int = 128
    semantic_shape: Optional[Tuple[int, int, int, int]] = None
    scene_layers: int = 2
    action_cross_layers: int = 1
    action_set_layers: int = 1
    num_heads: int = 4
    decision_enabled: bool = False
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, observation: dict[str, jax.Array]):
        if self.semantic_shape is None:
            raise ValueError("structured encoder requires semantic_shape")
        num_cards, cdb_dim, max_effects, lua_dim = self.semantic_shape
        if min(num_cards, cdb_dim, max_effects, lua_dim) <= 0:
            raise ValueError(f"invalid semantic shape: {self.semantic_shape}")

        cdb_table = self.variable(
            "constants",
            "cdb_exact",
            lambda: jnp.zeros((num_cards, cdb_dim), dtype=jnp.float32),
        ).value
        lua_table = self.variable(
            "constants",
            "lua_effects",
            lambda: jnp.zeros(
                (num_cards, max_effects, lua_dim), dtype=jnp.float16
            ),
        ).value
        lua_mask_table = self.variable(
            "constants",
            "lua_mask",
            lambda: jnp.zeros((num_cards, max_effects), dtype=jnp.uint8),
        ).value

        exact_encoder = ExactSemanticEncoder(
            self.channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="exact_semantics",
        )
        effect_encoder = EffectSetEncoder(
            self.channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="effect_semantics",
        )

        cards = observation["cards_"].astype(jnp.int32)
        card_ids = jnp.clip(decode_id(cards[..., :2]), 0, num_cards - 1)
        card_locations = cards[..., 2]
        card_padding = card_locations == 0
        exact_values = jax.lax.stop_gradient(cdb_table[card_ids])
        effect_values = jax.lax.stop_gradient(lua_table[card_ids])
        effect_mask = jax.lax.stop_gradient(lua_mask_table[card_ids]) > 0
        exact_tokens = exact_encoder(exact_values)
        effect_tokens = effect_encoder(effect_values, effect_mask)

        embed = lambda size, name: nn.Embed(
            size,
            self.channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name=name,
        )
        runtime_tokens = (
            embed(16, "card_location")(jnp.clip(cards[..., 2], 0, 15))
            + embed(128, "card_sequence")(jnp.clip(cards[..., 3], 0, 127))
            + embed(3, "card_controller")(jnp.clip(cards[..., 4], 0, 2))
            + embed(16, "card_position")(jnp.clip(cards[..., 5], 0, 15))
            + embed(3, "card_overlay")(jnp.clip(cards[..., 6], 0, 2))
            + embed(16, "card_attribute")(jnp.clip(cards[..., 7], 0, 15))
            + embed(32, "card_race")(jnp.clip(cards[..., 8], 0, 31))
            + embed(32, "card_level")(jnp.clip(cards[..., 9], 0, 31))
            + embed(17, "card_counter")(jnp.clip(cards[..., 10], 0, 16))
            + embed(3, "card_negated")(jnp.clip(cards[..., 11], 0, 2))
            + embed(2, "card_known")((card_ids > 0).astype(jnp.int32))
        )
        attack = _signed_u16(cards[..., 12], cards[..., 13]) / 10000.0
        defense = _signed_u16(cards[..., 14], cards[..., 15]) / 10000.0
        dynamic_numeric = jnp.concatenate(
            [
                attack[..., None],
                defense[..., None],
                cards[..., 16:].astype(jnp.float32),
            ],
            axis=-1,
        )
        runtime_tokens = runtime_tokens + nn.Dense(
            self.channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="card_dynamic_numeric",
        )(dynamic_numeric)
        gate_bias = self.param(
            "lua_gate_bias",
            nn.initializers.constant(-2.0),
            (1,),
            self.param_dtype,
        )
        gate = jax.nn.sigmoid(
            nn.Dense(
                1,
                use_bias=False,
                kernel_init=nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="lua_gate",
            )(jnp.concatenate([exact_tokens, effect_tokens], axis=-1))
            + gate_bias
        )
        card_tokens = nn.LayerNorm(dtype=self.dtype, name="card_norm")(
            runtime_tokens + exact_tokens + gate * effect_tokens
        )
        card_tokens = jnp.where(card_padding[..., None], 0, card_tokens)

        global_features = observation["global_"].astype(jnp.int32)
        own_lp = (
            (global_features[:, 0] << 8) + global_features[:, 1]
        ).astype(jnp.float32) / 8000.0
        opponent_lp = (
            (global_features[:, 2] << 8) + global_features[:, 3]
        ).astype(jnp.float32) / 8000.0
        global_numeric = jnp.concatenate(
            [
                own_lp[:, None],
                opponent_lp[:, None],
                (own_lp - opponent_lp)[:, None],
                global_features[:, 8:22].astype(jnp.float32) / 60.0,
            ],
            axis=-1,
        )
        state_token = (
            embed(20, "global_turn")(jnp.clip(global_features[:, 4], 0, 19))
            + embed(16, "global_phase")(jnp.clip(global_features[:, 5], 0, 15))
            + embed(2, "global_first")(jnp.clip(global_features[:, 6], 0, 1))
            + embed(2, "global_turn_owner")(
                jnp.clip(global_features[:, 7], 0, 1)
            )
            + nn.Dense(
                self.channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="global_numeric",
            )(global_numeric)
        )

        selection = observation["selection_"].astype(jnp.int32)
        selection_numeric = selection[:, 2:].astype(jnp.float32)
        selection_numeric = selection_numeric.at[:, :5].divide(8.0)
        state_token = state_token + (
            embed(32, "selection_prompt")(jnp.clip(selection[:, 0], 0, 31))
            + embed(8, "selection_role")(jnp.clip(selection[:, 1], 0, 7))
            + nn.Dense(
                self.channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="selection_numeric",
            )(selection_numeric)
        )
        state_token = nn.LayerNorm(dtype=self.dtype, name="state_input_norm")(
            state_token
        )

        public_events = observation["public_events_"].astype(jnp.int32)
        public_event_refs = observation["public_event_refs_"].astype(jnp.int32)
        event_padding = public_events[..., 0] == 0
        gathered_event_refs, event_ref_valid = _gather_refs(
            card_tokens, public_event_refs
        )
        event_ref_weights = event_ref_valid.astype(card_tokens.dtype)[..., None]
        event_ref_summary = (
            gathered_event_refs * event_ref_weights
        ).sum(axis=-2) / jnp.maximum(
            event_ref_weights.sum(axis=-2), 1
        )
        event_numeric = public_events[..., 6:].astype(jnp.float32)
        event_numeric = event_numeric.at[..., 0].divide(16.0)
        event_tokens = (
            embed(3, "event_actor")(jnp.clip(public_events[..., 1], 0, 2))
            + embed(32, "event_prompt")(jnp.clip(public_events[..., 2], 0, 31))
            + embed(16, "event_act")(jnp.clip(public_events[..., 3], 0, 15))
            + embed(8, "event_action_phase")(
                jnp.clip(public_events[..., 4], 0, 7)
            )
            + embed(8, "event_role")(jnp.clip(public_events[..., 5], 0, 7))
            + nn.Dense(
                self.channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="event_numeric",
            )(event_numeric)
            + nn.Dense(
                self.channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="event_refs",
            )(event_ref_summary)
        )
        event_tokens = PositionalEncoding(max_len=64)(event_tokens)
        safe_event_padding = event_padding.at[:, 0].set(False)
        event_tokens = EncoderLayer(
            self.num_heads,
            intermediate_size=self.channels * 4,
            activation="gelu",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="event_attention",
        )(event_tokens, src_key_padding_mask=safe_event_padding)
        event_tokens = jnp.where(
            event_padding[..., None], 0, event_tokens
        )

        scene = jnp.concatenate(
            [state_token[:, None, :], card_tokens, event_tokens], axis=1
        )
        scene_padding = jnp.concatenate(
            [
                jnp.zeros(
                    (scene.shape[0], 1), dtype=jnp.bool_
                ),
                card_padding,
                event_padding,
            ],
            axis=1,
        )
        for layer_index in range(self.scene_layers):
            scene = EncoderLayer(
                self.num_heads,
                intermediate_size=self.channels * 4,
                activation="gelu",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"scene_attention_{layer_index}",
            )(scene, src_key_padding_mask=scene_padding)
        scene = nn.LayerNorm(dtype=self.dtype, name="scene_norm")(scene)
        scene_state = scene[:, 0]
        card_tokens = scene[:, 1:1 + cards.shape[1]]

        action_ir = observation["action_ir_"].astype(jnp.int32)
        action_padding = action_ir[..., 0] == 0
        safe_action_padding = action_padding.at[:, 0].set(False)
        action_numeric = jnp.concatenate(
            [
                action_ir[..., 5:14].astype(jnp.float32),
                action_ir[..., 17:18].astype(jnp.float32) / 2.0,
                action_ir[..., 22:23].astype(jnp.float32) / 64.0,
            ],
            axis=-1,
        )
        action_base = (
            embed(32, "action_prompt")(jnp.clip(action_ir[..., 0], 0, 31))
            + embed(16, "action_act")(jnp.clip(action_ir[..., 1], 0, 15))
            + embed(8, "action_phase")(jnp.clip(action_ir[..., 2], 0, 7))
            + embed(8, "action_role")(jnp.clip(action_ir[..., 3], 0, 7))
            + embed(16, "action_stage")(jnp.clip(action_ir[..., 4], 0, 15))
            + embed(16, "action_position")(jnp.clip(action_ir[..., 18], 0, 15))
            + embed(32, "action_place")(jnp.clip(action_ir[..., 19], 0, 31))
            + embed(32, "action_number")(jnp.clip(action_ir[..., 20], 0, 31))
            + embed(16, "action_attribute")(
                jnp.clip(action_ir[..., 21], 0, 15)
            )
            + embed(8, "action_mode")(jnp.clip(action_ir[..., 23], 0, 7))
        )
        numeric_encoder = nn.Dense(
            self.channels, dtype=self.dtype, param_dtype=self.param_dtype,
            name="action_numeric",
        )
        action_tokens = action_base + numeric_encoder(action_numeric)
        if self.decision_enabled:
            decision_actions = action_base + numeric_encoder(
                action_numeric.at[..., -1].set(0)
            )

        source_ids = jnp.clip(
            decode_id(action_ir[..., 15:17]), 0, num_cards - 1
        )
        source_exact = exact_encoder(
            jax.lax.stop_gradient(cdb_table[source_ids])
        )
        source_units = jax.lax.stop_gradient(lua_table[source_ids])
        source_unit_mask = jax.lax.stop_gradient(
            lua_mask_table[source_ids]
        ) > 0
        effect_id = action_ir[..., 14]
        selected_effect = jnp.clip(effect_id - 2, 0, max_effects - 1)
        one_hot_effect = jax.nn.one_hot(
            selected_effect, max_effects, dtype=jnp.bool_
        )
        selected_mask = jnp.where(
            (effect_id >= 2)[..., None],
            source_unit_mask & one_hot_effect,
            source_unit_mask,
        )
        source_lua = effect_encoder(source_units, selected_mask)
        source_confidence = action_ir[..., 17:18].astype(jnp.float32) / 2.0
        action_tokens = action_tokens + source_confidence * (
            source_exact + source_lua
        )
        if self.decision_enabled:
            decision_actions = decision_actions + source_confidence * (
                source_exact + source_lua
            )

        single_refs = observation["action_single_refs_"].astype(jnp.int32)
        gathered_single, single_valid = _gather_refs(card_tokens, single_refs)
        if self.decision_enabled:
            role_features = role_bound_features(
                card_tokens, single_refs, source_confidence
            )
            role_delta = RoleBinding(
                self.channels, dtype=self.dtype, param_dtype=self.param_dtype,
                name="decision_roles",
            )(card_tokens, single_refs, source_confidence)
        single_role = nn.Embed(
            single_refs.shape[-1],
            self.channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="single_role",
        )(jnp.arange(single_refs.shape[-1]))
        gathered_single = gathered_single + single_role.reshape(
            (1, 1, single_refs.shape[-1], self.channels)
        )
        single_weights = single_valid.astype(gathered_single.dtype)[..., None]
        single_summary = (
            gathered_single * single_weights
        ).sum(axis=-2) / jnp.maximum(single_weights.sum(axis=-2), 1)
        action_tokens = action_tokens + nn.Dense(
            self.channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="single_ref_projection",
        )(single_summary)

        group_summary = RoleSetAttention(
            self.channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="group_role_attention",
        )(
            card_tokens,
            observation["action_group_refs_"],
            observation["action_group_mask_"],
        )
        action_tokens = action_tokens + group_summary
        if self.decision_enabled:
            action_tokens = action_tokens + role_delta
            decision_actions = decision_actions + group_summary + role_delta
            action_valid = ~action_padding
            weights = action_valid[..., None].astype(card_tokens.dtype)
            count = jnp.maximum(weights.sum(axis=1), 1)
            # Bind a global source only when every candidate has the same source
            # and effect. Never borrow the first card in a mixed-source prompt.
            source_identity = jnp.concatenate(
                [single_refs[..., :1], action_ir[..., 14:18]], axis=-1
            )
            minimum = jnp.where(action_valid[..., None], source_identity, 65536).min(axis=1)
            maximum = jnp.where(action_valid[..., None], source_identity, -1).max(axis=1)
            common_source = (
                (minimum == maximum).all(axis=-1)
                & (minimum[:, 0] > 0) & (minimum[:, -1] > 0)
                & action_valid.any(axis=-1)
            )
            source_context = (
                (gathered_single[..., 0, :] + source_exact + source_lua) * weights
            ).sum(axis=1) / count
            source_context = jnp.where(common_source[:, None], source_context, 0)
            decision_context = {
                "scene": scene,
                "scene_padding": scene_padding,
                "action_padding": action_padding,
                "actions": jnp.where(action_padding[..., None], 0, decision_actions),
                "roles": jnp.where(action_padding[..., None], 0, role_features),
                "query": jnp.concatenate([
                    state_token, scene_state, source_context,
                    common_source[:, None].astype(card_tokens.dtype),
                    (group_summary * weights).sum(axis=1) / count,
                ], axis=-1),
            }
        action_tokens = nn.LayerNorm(
            dtype=self.dtype, name="action_input_norm"
        )(action_tokens)

        for layer_index in range(self.action_cross_layers):
            action_tokens = CrossAttentionBlock(
                self.channels,
                self.num_heads,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"action_scene_cross_{layer_index}",
            )(action_tokens, scene, scene_padding)
        for layer_index in range(self.action_set_layers):
            action_tokens = EncoderLayer(
                self.num_heads,
                intermediate_size=self.channels * 4,
                activation="gelu",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"action_set_attention_{layer_index}",
            )(
                action_tokens,
                src_key_padding_mask=safe_action_padding,
            )
        action_tokens = nn.LayerNorm(
            dtype=self.dtype, name="action_output_norm"
        )(action_tokens)
        action_tokens = jnp.where(
            action_padding[..., None], 0, action_tokens
        )

        action_valid = (~action_padding).astype(action_tokens.dtype)[..., None]
        pooled_actions = (action_tokens * action_valid).sum(axis=1)
        pooled_actions = pooled_actions / jnp.maximum(
            action_valid.sum(axis=1), 1
        )
        state = nn.Dense(
            self.channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="state_projection",
        )(jnp.concatenate([scene_state, state_token, pooled_actions], axis=-1))
        state = nn.LayerNorm(dtype=self.dtype, name="state_output_norm")(state)
        valid = global_features[:, 22] == 0  # column 22: the decision-less (terminal) flag; 23 and 24 follow it
        outputs = (
            action_tokens,
            state,
            scene_state,
            safe_action_padding,
            valid,
        )
        return (*outputs, decision_context) if self.decision_enabled else outputs
