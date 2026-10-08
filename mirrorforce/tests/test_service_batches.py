"""Bounded search forward shapes preserve item order, public inputs and per-session state on a test backend.

This tests the wrapper; GPU bf16 kernel equivalence and performance require separate deployed measurements.
"""
import argparse

import numpy as np
import pytest

from tools import mf_runtime_policy_service as S


@pytest.mark.parametrize("value", ["", "0,1", "2,4", "1,1", "1,8,4", "1,257", "1,no", "1,-4"])
def test_bad_shape_lists_refused(value):
    with pytest.raises(argparse.ArgumentTypeError):
        S.search_batch_sizes(value)


def test_shapes_start_at_one_and_are_explicit():
    assert S.search_batch_sizes("1,4,16,64,256") == (1, 4, 16, 64, 256)
    assert list(S.padded_batches([], (1, 4))) == []
    assert list(S.padded_batches([1, 2, 3], ())) == [(3, [1, 2, 3])]
    assert list(S.padded_batches([1, 2, 3, 4, 5, 6, 7], (1, 4))) == [(4, [1, 2, 3, 4]), (3, [5, 6, 7, 7])]


class Policy:
    def __init__(self):
        self.sizes, self.obs = [], []

    def act_batch(self, observations, states, firsts):
        self.sizes.append(len(observations))
        self.obs.extend(observations)
        return [(state + obs["x"] + first, np.arange(obs["rows"], dtype=np.float32), obs["x"],
                 np.asarray([obs["x"], first, 0], np.float32))
                for obs, state, first in zip(observations, states, firsts)]


def backend(sizes):
    instance = object.__new__(S.CheckpointBackend)
    instance.policy, instance.batch_sizes = Policy(), sizes
    return instance


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5, 15, 16, 17, 50, 200, 255, 256, 257, 529])
def test_variable_requests_use_only_fixed_shapes_without_extra_session_updates(count):
    items = [({"x": i, "rows": 3 + i % 5}, np.array([i, i + 1]), i % 3 == 0, 3 + i % 5)
             for i in range(count)]
    before = [S.memory_sha256(item[1]) for item in items]
    plain, fixed = backend(()), backend((1, 4, 16, 64, 256))
    expected, result = plain.act_batch(items), fixed.act_batch(items)
    assert len(result) == count
    assert set(fixed.policy.sizes) <= {1, 4, 16, 64, 256}
    for old, new in zip(expected, result):
        np.testing.assert_array_equal(old[0], new[0])
        np.testing.assert_array_equal(old[1], new[1])
        assert old[2:] == new[2:]
    assert [S.memory_sha256(item[1]) for item in items] == before
    # Extra calls contain actual valid observation rows, never an all-zero invented observation.
    assert all(obs in [item[0] for item in items] for obs in fixed.policy.obs)


def test_backend_cannot_silently_drop_padded_results():
    fixed = backend((1, 4))
    original = fixed.policy.act_batch
    fixed.policy.act_batch = lambda *args: original(*args)[:-1]
    with pytest.raises(RuntimeError, match="different batch size"):
        fixed.act_batch([({"x": 2, "rows": 3}, np.array([1]), False, 3)])


def test_fixed_search_shapes_cannot_start_without_checkpoint_and_warmup():
    with pytest.raises(ValueError, match="checkpoint backend and a warmup deck"):
        S.main(["--backend", "uniform", "--selection", "sample", "--config", "{}",
                "--cards-db", "/unused", "--code-list", "/unused", "--script-root", "/unused",
                "--announce-tables", "/unused", "--socket", "/unused", "--out", "/unused",
                "--search-batch-sizes", "1,4"])


def test_warmup_covers_every_registered_batch_and_both_menu_blocks(monkeypatch, tmp_path):
    pytest.importorskip("jax")
    from types import SimpleNamespace
    from mirrorforce.worldmodel import engine
    from tools import mf_runtime_client_parity as parity
    monkeypatch.setattr(engine, "load_ydk", lambda path: None)
    monkeypatch.setattr(parity, "mirror_deal", lambda *args: {})
    fixed, shapes = backend((1, 4, 16)), []
    fixed.policy.max_options, fixed.policy.variables = 192, {}
    fixed.policy.initial_state = lambda: (np.zeros((1, 3), np.float32),)

    def apply(variables, obs, state, first):
        shapes.append((obs["action_ir_"].shape[0], obs["action_ir_"].shape[1]))
        assert state[0].shape == (len(first), 3)
        assert np.asarray(first).all()
        return state, np.zeros((len(first), obs["action_ir_"].shape[1]))

    fixed.policy._apply = apply
    duel = SimpleNamespace(start=lambda: None, prompt=lambda: (0, 11, [{}]),
                           observation=lambda: {"obs:action_ir_": np.ones((192, 24), np.uint8)})
    native = SimpleNamespace(ScriptedDuel=lambda *args: duel)
    assert fixed.warm_up(native, {}, tmp_path / "unused") >= 0
    assert shapes == [(size, menu) for size in (1, 4, 16) for menu in (32, 192)]


@pytest.mark.parametrize("value", ["", "0", "257", "-1", "1.5", "bad"])
def test_constant_shape_size_refuses_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        S.constant_batch_size(value)


class ConstantPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.full_menus = []

    def act(self, *args):
        raise AssertionError("constant mode must not call the batch-1 path")

    def act_batch(self, observations, states, firsts, *, full_menu):
        self.full_menus.append(full_menu)
        assert full_menu is True
        return super().act_batch(observations, states, firsts)


@pytest.mark.parametrize("count", [0, 1, 3, 4, 5, 16, 17, 65])
def test_constant_shape_includes_live_decisions_and_any_request_partition(count):
    instance = backend(())
    instance.constant_size, instance.policy = 4, ConstantPolicy()
    items = [({"x": i, "rows": 3 + i % 5}, np.array([i, i + 1]), i % 3 == 0, 3 + i % 5)
             for i in range(count)]
    before = [S.memory_sha256(item[1]) for item in items]
    single = [instance.act(*item) for item in items]
    batched = instance.act_batch(items)
    reverse = instance.act_batch(list(reversed(items)))[::-1]
    # Unrelated co-batched worlds and padding content may not affect a row.
    for expected, observed, permuted in zip(single, batched, reverse):
        for other in (observed, permuted):
            np.testing.assert_array_equal(expected[0], other[0])
            np.testing.assert_array_equal(expected[1], other[1])
            assert expected[2:] == other[2:]
    assert set(instance.policy.sizes) <= {4}
    assert all(instance.policy.full_menus)
    assert [S.memory_sha256(item[1]) for item in items] == before


def test_constant_mode_requires_batch_api_and_rejects_dual_modes():
    instance = backend(())
    instance.constant_size = 4
    instance.policy.act_batch = None
    with pytest.raises(RuntimeError, match="require Policy.act_batch"):
        instance.act({}, None, True, 1)
    for size, buckets in ((True, ()), (0, ()), (257, ()), (4, (1, 4))):
        with pytest.raises(ValueError, match="mutually exclusive"):
            S.CheckpointBackend(None, None, None, semantic_file=None, code_list=None,
                                card_tables=None, announce_tables=None, dormant_table=None,
                                batch_sizes=buckets, constant_size=size)


def test_constant_warmup_compiles_only_the_declared_shape(monkeypatch, tmp_path):
    pytest.importorskip("jax")
    from types import SimpleNamespace
    from mirrorforce.worldmodel import engine
    from tools import mf_runtime_client_parity as parity
    monkeypatch.setattr(engine, "load_ydk", lambda path: None)
    monkeypatch.setattr(parity, "mirror_deal", lambda *args: {})
    fixed, shapes = backend(()), []
    fixed.constant_size = 4
    fixed.policy.max_options, fixed.policy.variables = 192, {}
    fixed.policy.initial_state = lambda: (np.zeros((1, 3), np.float32),)

    def apply(variables, obs, state, first):
        shapes.append((obs["action_ir_"].shape[0], obs["action_ir_"].shape[1]))
        assert state[0].shape == (4, 3) and np.asarray(first).all()
        return state, np.zeros((4, 192))

    fixed.policy._apply = apply
    native = SimpleNamespace(ScriptedDuel=lambda *args: SimpleNamespace(start=lambda: None,
        prompt=lambda: (0, 11, [{}]), observation=lambda: {"obs:action_ir_": np.ones((192, 24), np.uint8)}))
    fixed.warm_up(native, {}, tmp_path / "unused")
    assert shapes == [(4, 192)]


def test_constant_cli_requires_checkpoint_and_warmup():
    with pytest.raises(ValueError, match="checkpoint backend and a warmup deck"):
        S.main(["--backend", "uniform", "--selection", "sample", "--config", "{}",
                "--cards-db", "/unused", "--code-list", "/unused", "--script-root", "/unused",
                "--announce-tables", "/unused", "--socket", "/unused", "--out", "/unused",
                "--constant-batch-size", "4"])


def test_shared_geometry_reaches_only_the_shared_policy_and_identity(monkeypatch):
    from types import SimpleNamespace
    from mirrorforce.agent.inference_geometry import FIXED_PUBLIC_B64
    from mirrorforce.agent.train import policy_io
    calls, variables = [], object()
    receipt = {"recipe": {"ataraxos": {"magnet": "uniform_legal"}},
               "model": {"args": {"dtype": "bfloat16"}}}
    agent = SimpleNamespace(config=SimpleNamespace(dtype="bfloat16"))
    monkeypatch.setattr(policy_io, "load_policy", lambda *a, **kw: (agent, variables, receipt))
    monkeypatch.setattr(policy_io, "native_setup", lambda *a, **kw: None)
    monkeypatch.setattr(S, "file_sha256", lambda path: "a" * 64)

    def make_policy(actual_agent, actual_variables, actual_receipt, **kwargs):
        assert actual_agent is agent and actual_variables is variables and actual_receipt is receipt
        calls.append(kwargs)
        return SimpleNamespace(max_options=192, inference_geometry=kwargs.get("inference_geometry"))

    monkeypatch.setattr(policy_io, "Policy", make_policy)
    kwargs = dict(semantic_file="s", code_list="c", card_tables="t", announce_tables="a", dormant_table=None,
                  constant_size=64)
    old = S.CheckpointBackend(None, "checkpoint", "iterate", **kwargs)
    fixed = S.CheckpointBackend(None, "checkpoint", "iterate", **kwargs, inference_geometry=FIXED_PUBLIC_B64)
    assert calls[0] == {"matmul_precision": None} and "inference_geometry" not in old.identity
    assert calls[1] == {"matmul_precision": None, "inference_geometry": FIXED_PUBLIC_B64}
    assert fixed.identity["inference_geometry"] == FIXED_PUBLIC_B64
    assert fixed.identity["inference_geometry"] is not FIXED_PUBLIC_B64
    for patch in ({"constant_size": None}, {"constant_size": 16}, {"compute_dtype": "float32"}):
        with pytest.raises(ValueError, match="constant B64"):
            S.CheckpointBackend(None, "checkpoint", "iterate", **{**kwargs, **patch},
                                inference_geometry=FIXED_PUBLIC_B64)
    receipt["model"]["args"]["dtype"] = agent.config.dtype = "float32"
    with pytest.raises(ValueError, match="BF16/full192"):
        S.CheckpointBackend(None, "checkpoint", "iterate", **kwargs, inference_geometry=FIXED_PUBLIC_B64)


@pytest.mark.parametrize("extra", [[], ["--constant-batch-size", "64"],
                                 ["--constant-batch-size", "64", "--compute-dtype", "float32"]])
def test_shared_geometry_cli_refuses_uniform_or_unregistered_computation(extra):
    with pytest.raises(ValueError, match="requires"):
        S.main(["--backend", "uniform", "--selection", "greedy", "--config", "{}",
                "--cards-db", "/unused", "--code-list", "/unused", "--script-root", "/unused",
                "--announce-tables", "/unused", "--socket", "/unused", "--out", "/unused",
                "--inference-geometry", "fixed-public-b64", *extra])
