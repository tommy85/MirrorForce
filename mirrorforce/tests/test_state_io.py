"""training checkpoints round-trip parameters, Adam state, learner keys and counters, and refuse mismatches.

Needs the JAX environment (Python 3.11 venv); skipped elsewhere.
"""
import json

import pytest

jax = pytest.importorskip("jax")
optax = pytest.importorskip("optax")
import numpy as np  # noqa: E402

from mirrorforce.agent.model.utils import TrainState  # noqa: E402
from mirrorforce.agent.train import checkpoint_store, state_io  # noqa: E402

IDENTITY = {"model": {"architecture": "decision-v1", "args": {"num_channels": 8}},
            "recipe": {"batch_size": 8192}, "tables": {"semantic_file_sha256": "a" * 64}}


def state(seed):
    params = {"dense": {"kernel": jax.random.normal(jax.random.PRNGKey(seed), (3, 4)), "bias": np.zeros(4, np.float32)}}
    tx = optax.adam(1e-3)
    built = TrainState.create(apply_fn=None, params=params, tx=tx, batch_stats={}, constants={"table": np.ones(5)})
    grads = jax.tree.map(np.ones_like, params)
    return built.apply_gradients(grads=grads)


def test_round_trip_restores_state_keys_and_counters(tmp_path):
    saved = state(0)
    keys = np.asarray(jax.random.split(jax.random.PRNGKey(3), 2))
    sha = state_io.save(tmp_path, saved, keys, {"global_step": 8192, "learner_update": 1}, {**IDENTITY, "parent": None})
    receipt = json.loads((tmp_path / f"{sha}.receipt.json").read_text())
    assert receipt["schema"] == state_io.SCHEMA and receipt["env_state_restored_on_resume"] is False
    restored, restored_keys, counters, _, actors, ema = state_io.restore(tmp_path / f"{sha}.ckpt", state(1), 2, IDENTITY)
    assert actors is None and ema is None
    assert counters == {"global_step": 8192, "learner_update": 1}
    assert np.array_equal(restored_keys, keys)
    for left, right in zip(jax.tree.leaves(saved.params), jax.tree.leaves(restored.params)):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    for left, right in zip(jax.tree.leaves(saved.opt_state), jax.tree.leaves(restored.opt_state)):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    assert int(restored.step) == 1
    assert np.array_equal(np.asarray(restored.constants["table"]), np.ones(5))  # constants come from the template


def test_restore_refuses_identity_tamper_and_missing_arrival(tmp_path):
    sha = state_io.save(tmp_path, state(0), np.zeros((1, 2), np.uint32),
                        {"global_step": 1, "learner_update": 1}, {**IDENTITY, "parent": None})
    path = tmp_path / f"{sha}.ckpt"
    with pytest.raises(ValueError, match="recipe"):
        state_io.restore(path, state(1), 1, {**IDENTITY, "recipe": {"batch_size": 2048}})
    with pytest.raises(ValueError, match="learner keys"):
        state_io.restore(path, state(1), 2, IDENTITY)
    for arrival in tmp_path.glob("*.arrival-*.json"):
        arrival.unlink()
    with pytest.raises(ValueError, match="arrival"):
        state_io.restore(path, state(1), 1, IDENTITY)
    checkpoint_store.arrive(tmp_path, sha, source_host="test", route="local")
    state_io.restore(path, state(1), 1, IDENTITY)
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="payload"):
        state_io.restore(path, state(1), 1, IDENTITY)


def test_actor_snapshots_round_trip(tmp_path):
    actors = [{"update": 4, "envs": ["state a", "state b"], "next_obs": {"cards_": np.arange(6, dtype=np.uint8)},
               "next_done": np.array([True, False]), "rstate": [np.ones((2, 3), np.float32)], "key": np.array([1, 2],
               np.uint32), "global_step": 512, "actor_policy_version": 3}]
    sha = state_io.save(tmp_path, state(0), np.zeros((1, 2), np.uint32), {"global_step": 1, "learner_update": 1},
                        {**IDENTITY, "parent": None}, actors=actors)
    receipt = json.loads((tmp_path / f"{sha}.receipt.json").read_text())
    assert receipt["env_state_restored_on_resume"] is True
    *_, restored, _ = state_io.restore(tmp_path / f"{sha}.ckpt", state(1), 1, IDENTITY)
    assert len(restored) == 1 and restored[0]["update"] == 4 and list(restored[0]["envs"]) == ["state a", "state b"]
    assert np.array_equal(restored[0]["next_obs"]["cards_"], np.arange(6, dtype=np.uint8))
    assert np.array_equal(restored[0]["next_done"], [True, False]) and restored[0]["global_step"] == 512
    assert np.array_equal(restored[0]["rstate"][0], np.ones((2, 3), np.float32))


def test_prune_keeps_the_last_checkpoints_and_every_kth(tmp_path):
    shas = {}
    for update in range(1, 9):
        shas[update] = state_io.save(tmp_path, state(0).replace(step=update), np.zeros((1, 2), np.uint32),
                                     {"global_step": update, "learner_update": update}, {**IDENTITY, "parent": None})
    deleted = checkpoint_store.prune(tmp_path, keep_last=2, keep_every=4)
    kept = {u for u, sha in shas.items() if (tmp_path / f"{sha}.ckpt").exists()}
    assert kept == {4, 7, 8} and len(deleted) == 5
    assert all((tmp_path / f"{sha}.receipt.json").exists() for sha in shas.values())
    assert len((tmp_path / "pruned.jsonl").read_text().splitlines()) == 5
