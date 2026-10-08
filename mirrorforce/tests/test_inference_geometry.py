import json
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mirrorforce.agent.inference_geometry import FIXED_PUBLIC_B64, FIXED_PUBLIC_B8, validate_geometry
from mirrorforce.agent.train.policy_io import Policy
from mirrorforce.agent import belief_features as F
from mirrorforce.agent.model import policy_net as V


def test_profile_is_exact_independent_json_data_and_metadata_imports_no_runtime():
    value = validate_geometry(FIXED_PUBLIC_B64)
    assert type(value) is dict and json.loads(json.dumps(value)) == value
    value['batch_size'] = 1
    assert FIXED_PUBLIC_B64['batch_size'] == 64
    code = ('from mirrorforce.agent.inference_geometry import FIXED_PUBLIC_B64, validate_geometry; '
            'import sys; assert not any(k in sys.modules for k in ("jax","numpy","mirrorforce.agent.train.policy_io")); '
            'assert validate_geometry(FIXED_PUBLIC_B64)["chunk_event_rows"] == 64')
    subprocess.run([sys.executable, '-c', code], check=True, timeout=10)


@pytest.mark.parametrize('change', [None, {}, {'batch_size': 1}, {'event_rows': 128}, {'chunk_slots': 0},
                                  {'chunk_event_rows': 512}, {'compute_dtype': 'float32'}, {'extra': True}])
def test_incomplete_or_different_geometry_is_not_implicitly_admitted(change):
    value = change if change is None or not change else {**FIXED_PUBLIC_B64, **change}
    with pytest.raises(ValueError, match='exact fixed'):
        validate_geometry(value)


def fake_policy(monkeypatch, geometry=FIXED_PUBLIC_B64):
    traces = []
    n = geometry['batch_size']

    def forward(model, variables, obs, seat, memory, first, *, full_shapes=False):
        traces.append(full_shapes)
        assert full_shapes is True
        logits = obs['action_ir_'][..., 0].astype(jnp.float32)
        value = logits.sum(-1)
        wdl = jnp.stack([value, value + 1, value + 2], -1)
        after = memory._replace(count=memory.count + 1)
        features = {'state': logits[:, :2, None], 'state_valid': jnp.ones((n, 2), bool)}
        return logits, value, wdl, after, jnp.zeros((n,), bool), features, jnp.zeros((n, 192, 6, 4)), \
            jnp.ones((n, 192), bool)

    monkeypatch.setattr(F, 'forward', forward)
    agent = SimpleNamespace(config=SimpleNamespace(dtype='bfloat16', memory_slots=64, chunk_slots=48, d=2), model=object())
    policy = Policy(agent, {}, {'config': {'max_options': 192}}, inference_geometry=dict(geometry))
    state = (np.zeros((1, 64, 2), np.float32), np.zeros((1,), np.int32), np.zeros((1,), np.int32),
             np.zeros((1, 48, 2), np.float32), np.zeros((1,), np.int32), np.zeros((1,), np.int32))
    obs = {'obs:action_ir_': np.zeros((192, 1), np.int32), 'obs:turn_events_': np.zeros((512, 1), np.int32),
           'obs:closed_turns_': np.zeros((2, 512, 1), np.int32),
           'obs:turn_chunks_': np.zeros((4, 64, 1), np.int32)}
    obs['obs:action_ir_'][:3] = 1
    return policy, obs, (state, state), traces


def test_serving_and_features_use_one_jit_with_identical_outputs_memory_and_probabilities(monkeypatch):
    policy, obs, state, traces = fake_policy(monkeypatch)
    batch, states, first = [obs] * 64, [state] * 64, [False] * 64
    baseline = policy.act_batch(batch, states, first, full_menu=True)
    features = policy.act_batch_features(batch, states, first)
    assert traces == [True] and policy._public_apply._cache_size() == 1
    assert policy.inference_geometry == dict(FIXED_PUBLIC_B64)
    for old, new in zip(baseline, features):
        for a, b in zip(jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(new[:4])):
            assert np.asarray(a).dtype == np.asarray(b).dtype
            assert np.asarray(a).shape == np.asarray(b).shape
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
        probabilities = []
        for logits in (old[1], new[1]):
            values = np.asarray(logits, np.float64)
            p = np.exp(values - values.max())
            probabilities.append(p / p.sum())
        assert probabilities[0].tobytes() == probabilities[1].tobytes()
        assert np.random.default_rng(25).choice(len(old[1]), p=probabilities[0]) == \
               np.random.default_rng(25).choice(len(new[1]), p=probabilities[1])
    solo = policy.act(obs, state, False)
    for a, b in zip(jax.tree_util.tree_leaves(solo), jax.tree_util.tree_leaves(baseline[0])):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    assert traces == [True]


def test_explicit_b8_policy_keeps_full_context_and_one_same_forward_program(monkeypatch):
    policy, obs, state, traces = fake_policy(monkeypatch,FIXED_PUBLIC_B8)
    ordinary = policy.act_batch([obs]*8,[state]*8,[False]*8,full_menu=True)
    features = policy.act_batch_features([obs]*8,[state]*8,[False]*8)
    assert len(features) == 8 and traces == [True] and policy._public_apply._cache_size() == 1
    for expected, actual in zip(ordinary,features):
        for left,right in zip(jax.tree_util.tree_leaves(expected),jax.tree_util.tree_leaves(actual[:4])):
            np.testing.assert_array_equal(left,right)
    with pytest.raises(ValueError,match='exact registered'):
        policy.act_batch([obs]*64,[state]*64,[False]*64,full_menu=True)
    assert traces == [True]


@pytest.mark.parametrize('change', ['batch', 'menu_mode', 'menu_rows', 'event_rows', 'chunk_rows',
                                  'memory_input', 'chunk_input', 'private', 'labels'])
def test_profile_refuses_other_shapes_and_private_targets_before_forward(monkeypatch, change):
    policy, obs, state, traces = fake_policy(monkeypatch)
    n, full = (1 if change == 'batch' else 64), change != 'menu_mode'
    if change == 'menu_rows':
        obs['obs:action_ir_'] = obs['obs:action_ir_'][:32]
    elif change == 'event_rows':
        obs['obs:turn_events_'] = obs['obs:turn_events_'][:128]
    elif change == 'chunk_rows':
        obs['obs:turn_chunks_'] = obs['obs:turn_chunks_'][:, :32]
    elif change in ('memory_input', 'chunk_input'):
        values = list(state[0])
        index = 0 if change == 'memory_input' else 3
        values[index] = values[index][:, :-1]
        state = (tuple(values), tuple(values))
    elif change in ('private', 'labels'):
        obs['priv:cards_' if change == 'private' else 'label:hidden_'] = np.zeros(1)
    with pytest.raises(ValueError):
        policy.act_batch([obs] * n, [state] * n, [False] * n, full_menu=full)
    assert traces == []


def test_direct_warmup_cannot_bypass_profile_geometry(monkeypatch):
    policy, obs, state, traces = fake_policy(monkeypatch)
    with pytest.raises(ValueError, match='direct public apply'):
        policy._apply({}, {k[4:]: v[None] for k, v in obs.items()}, state, jnp.array([False]))
    assert traces == []


def test_fixed_full_shaped_keeps_masks_and_ignores_batch_peer_shape_selection():
    config = V.PolicyNetConfig(d=16, memory_slots=4, chunk_slots=3, row_blocks=(2, 4))
    prefix = V.Prefix(jnp.zeros((2, 4, 16)), jnp.zeros((2, 4), bool), jnp.zeros((2, 4), jnp.int32),
                      jnp.zeros((2, 3, 16)), jnp.zeros((2, 3), bool), jnp.zeros((2, 3), jnp.int32))
    rows = jnp.zeros((2, 8, 1), jnp.int32)
    fn = lambda p, window: jnp.array([window.shape[1], p.chunks.shape[1], p.chunk_valid.sum()])
    for full, expected in ((False, [2, 0, 0]), (True, [8, 3, 0])):
        actual = jax.jit(lambda p, r: V._shaped(fn, config, p, r, jnp.ones((2,), bool), (),
                                               full_shapes=full))(prefix, rows)
        np.testing.assert_array_equal(actual, expected)
    active_peer = prefix._replace(chunk_valid=prefix.chunk_valid.at[1, 0].set(True))
    actual = V._shaped(fn, config, active_peer, rows, jnp.ones((2,), bool), (), full_shapes=True)
    np.testing.assert_array_equal(actual, [8, 3, 1])


def test_profile_rejects_parameter_geometry_or_precision_fallback():
    config = SimpleNamespace(dtype='bfloat16', memory_slots=64, chunk_slots=48)
    for changed, precision in ((dict(dtype='float32'), None), (dict(chunk_slots=4), None),
                               (dict(memory_slots=32), None), ({}, 'highest')):
        agent = SimpleNamespace(config=SimpleNamespace(**{**vars(config), **changed}))
        with pytest.raises(ValueError, match='dimensions/dtype/precision'):
            Policy(agent, {}, {'config': {'max_options': 192}}, matmul_precision=precision,
                   inference_geometry=FIXED_PUBLIC_B64)
    with pytest.raises(ValueError, match='dimensions/dtype/precision'):
        Policy(SimpleNamespace(config=config), {}, {'config': {'max_options': 32}},
               inference_geometry=FIXED_PUBLIC_B64)


def test_default_policy_keeps_legacy_dynamic_menu_and_empty_batch_behavior():
    traced = []

    def apply(v, obs, state, first, main, *, return_bad):
        traced.append(obs['action_ir_'].shape[:2])
        n = first.shape[0]
        return state, obs['action_ir_'][..., 0].astype(jnp.float32), jnp.zeros((n, 1)), \
            jnp.zeros((n, 3)), jnp.zeros((n,), bool)

    policy = Policy(SimpleNamespace(apply=apply), {}, {'config': {'max_options': 192}})
    assert policy.inference_geometry is None and not hasattr(policy, '_public_apply')
    assert policy.act_batch([], [], []) == []
    for states, first, full in (([0], [], False), ([], [False], False), ([], [], 'yes')):
        with pytest.raises(ValueError):
            policy.act_batch([], states, first, full_menu=full)
    obs = {'obs:action_ir_': np.zeros((192, 1), np.int32)}
    obs['obs:action_ir_'][:3] = 1
    state = (np.zeros((1, 2), np.float32),)
    assert len(policy.act_batch([obs] * 2, [state] * 2, [False] * 2)) == 2
    assert len(policy.act(obs, state, False)[1]) == 3
    assert traced == [(2, 32), (1, 32)]
