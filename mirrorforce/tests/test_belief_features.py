"""Real policy_net extraction, observer memory, stop-gradient and public candidate joins (small CPU models)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mirrorforce.agent import belief_features as F
from mirrorforce.agent.model import policy_net as V
from mirrorforce.agent.model.belief_ar import ARConfig, admit_public
from mirrorforce.agent.model.reference_agent import ReferenceAgent
from mirrorforce.agent.train.policy_io import Policy
from mirrorforce.agent.search.belief_ar_law import PublicLayoutLaw
from test_policy_net import SMALL, SMALL_SEMANTICS, carried, episode, make, shapes
from test_belief_ar_law import world


@pytest.fixture(scope="module")
def example():
    layout = shapes()
    model, variables = make(SMALL, SMALL_SEMANTICS, layout, 2, np.random.default_rng(8830))
    observations, seats = episode(np.random.default_rng(8831), 12, 2, layout, SMALL_SEMANTICS[0], first_turn=2)
    observations, seats = jax.tree_util.tree_map(jnp.asarray, observations), jnp.asarray(seats)
    memory = carried(np.random.default_rng(8832), 2, SMALL)
    return model, variables, observations, seats, memory


def test_extract_preserves_original_policy_and_nonzero_history_deliveries(example):
    model, variables, observations, seats, memory = example
    ordinary = jax.jit(lambda obs, seat, memory: V.act(model, variables, obs, seat, memory))
    extract = jax.jit(lambda obs, seat, memory: F.forward(model, variables, obs, seat, memory))
    # Both production paths are compiled inference. Keep this head oracle
    # independent of F/Policy, but compile table -> encode -> belief together:
    # eager per-operation CPU kernels have an import-order-dependent rounding
    # difference under cleanba's existing Eigen flag. Tolerance is unchanged.
    @jax.jit
    def old_head(obs):
        table = model.apply(variables, method=V.PolicyNet.semantic_table)
        enc = model.apply(variables, obs, table, method=V.PolicyNet.encode)
        return model.apply(variables, enc, obs, table, method=V.PolicyNet.belief)
    old_memory, new_memory = memory, memory
    for step in range(12):
        obs = jax.tree_util.tree_map(lambda value: value[step], observations)
        old, new = ordinary(obs, seats[step], old_memory), extract(obs, seats[step], new_memory)
        for left, right in zip(jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(new[:5])):
            np.testing.assert_array_equal(left, right)
        assert not np.asarray(new[4]).any()
        features = new[5]
        assert features["turn"].shape == (2, 1, SMALL.d)
        assert set(features) == {key for name in F.BLOCKS for key in (name, name + "_valid")} | {
            "candidates", "candidate_valid"}
        assert all(np.isfinite(np.asarray(value)).all() for value in features.values())
        for name in F.BLOCKS:
            values, valid = np.asarray(features[name]), np.asarray(features[name + "_valid"])
            assert valid.dtype == bool and valid.shape == values.shape[:2]
            assert not values[~valid].any()
        baseline = old_head(obs)
        for left, right in zip(baseline, new[6:]):
            np.testing.assert_allclose(left, right, atol=2e-6, rtol=2e-6)
        old_memory, new_memory = old[3], new[3]
    assert np.asarray(new_memory.count).sum() > np.asarray(memory.count).sum()


def test_feature_and_old_head_outputs_stop_all_actor_parameter_gradients(example):
    model, variables, observations, seats, memory = example
    obs = jax.tree_util.tree_map(lambda value: value[0], observations)
    def objective(params, buffer, chunks):
        outputs = F.forward(model, {**variables, "params": params}, obs, seats[0],
                            memory._replace(buffer=buffer, chunks=chunks))
        return sum(jnp.sum(value.astype(jnp.float32)) for value in outputs[5].values()) + jnp.sum(outputs[6])
    grads = jax.jit(jax.grad(objective, argnums=(0, 1, 2)))(variables["params"], memory.buffer, memory.chunks)
    assert all(np.count_nonzero(value) == 0 for value in jax.tree_util.tree_leaves(grads))


def test_full_shapes_optin_preserves_real_model_policy_outputs_and_memory(example):
    model, variables, observations, seats, memory = example
    ordinary = jax.jit(lambda obs, seat, state: V.act(model, variables, obs, seat, state, full_shapes=True))
    extract = jax.jit(lambda obs, seat, state: F.forward(model, variables, obs, seat, state, full_shapes=True))
    for step in (0, 5):
        obs = jax.tree_util.tree_map(lambda value: value[step], observations)
        expected, actual = ordinary(obs, seats[step], memory), extract(obs, seats[step], memory)
        for left, right in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual[:5])):
            np.testing.assert_array_equal(left, right)


def test_frozen_policy_adapter_keeps_original_pair_of_observer_states(example):
    _, variables, observations, _, memory = example
    agent = ReferenceAgent(SMALL, SMALL_SEMANTICS)
    receipt = {"config": {"max_options": observations["action_ir_"].shape[2]}}
    capture = F.FrozenBeliefPolicy(agent, variables, receipt, batch_size=2)
    original = Policy(agent, variables, receipt)
    public = [{"obs:" + key: np.asarray(value[0, index]) for key, value in observations.items()} for index in (0, 1)]
    states = [(tuple(np.asarray(value[index:index + 1, 0]) for value in memory),
               tuple(np.asarray(value[index:index + 1, 1]) for value in memory)) for index in (0, 1)]
    expected = original.act_batch(public, states, [False, False], full_menu=True)
    actual = capture.act_batch_features(public, states, [False, False])
    for old, new in zip(expected, actual):
        for left, right in zip(jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(new[:4])):
            np.testing.assert_array_equal(left, right)
    with pytest.raises(ValueError, match="fixed batch"):
        capture.act_batch_features(public[:1], states[:1], [False])
    with pytest.raises(ValueError, match="only obs"):
        capture.act_batch_features([{**public[0], "label:hidden_": np.zeros(2)}, public[1]], states, [False, False])


@pytest.mark.parametrize("key", ["priv:cards_", "priv_cards_", "label:hidden_"])
def test_private_and_target_fields_rejected_before_forward(example, key):
    model, variables, observations, seats, memory = example
    obs = jax.tree_util.tree_map(lambda value: value[0], observations)
    with pytest.raises(ValueError, match="refuses"):
        F.forward(model, variables, {**obs, key: jnp.zeros((2, 1))}, seats[0], memory)


def test_public_layout_joins_sorted_raw_codes_without_reading_targets():
    # Model vocabulary IDs and raw code order deliberately differ; padded candidates never become targets.
    candidates = np.array([[0, 7, 1], [0, 2, 2], [0, 9, 3], [0, 0, 0]], np.uint8)
    features = {key: value for name in F.BLOCKS for key, value in (
        (name, np.arange(12, dtype=np.float32).reshape(3, 4)), (name + "_valid", np.ones(3, bool)))}
    features.update(candidates=np.arange(20, dtype=np.float32).reshape(4, 5), candidate_valid=np.array([1, 1, 1, 0], bool))
    law = PublicLayoutLaw(world())
    public = F.public_for_layout(features, candidates, {7: 33, 2: 11, 9: 22}, law)
    np.testing.assert_array_equal(public["candidates"][0], features["candidates"][[1, 2, 0]])
    assert public["slots"].shape == (1, 3, 8)
    assert public["slots"][0, 0, 0] == 1 and np.all(public["slots"][0, 1:, 1] == 1)
    admit_public(public, np.full((1, 3), -1, np.int32), ARConfig())
    for key in (*F.BLOCKS, "candidates"):
        public[key] = np.asarray(jnp.asarray(public[key], jnp.bfloat16))
    admit_public(public, np.full((1, 3), -1, np.int32), ARConfig())
    with pytest.raises(ValueError, match="cover"):
        F.public_for_layout(features, candidates, {7: 33, 2: 11, 9: 44}, law)
    with pytest.raises(ValueError, match="schema"):
        F.public_for_layout({**features, "labels": np.ones(3)}, candidates, {7: 33, 2: 11, 9: 22}, law)


def test_empty_public_layout_has_only_invalid_padding():
    features = {key: value for name in F.BLOCKS for key, value in (
        (name, np.zeros((1, 4), np.float32)), (name + "_valid", np.zeros(1, bool)))}
    features.update(candidates=np.zeros((1, 5), np.float32), candidate_valid=np.zeros(1, bool))
    law = PublicLayoutLaw(dict(world(), hand=[], hand_group=[], deck=[], facedown=[], pool_main={}, unpositioned={}))
    public = F.public_for_layout(features, np.zeros((1, 3), np.uint8), {}, law)
    assert public["slots"].shape == (1, 1, 8) and not public["slot_valid"].any()
    assert not public["candidate_valid"].any()
    admit_public(public, np.full((1, 1), -1, np.int32), ARConfig())
