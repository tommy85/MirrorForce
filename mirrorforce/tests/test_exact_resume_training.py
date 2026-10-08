"""Exact resume of training: an interrupted run resumed from its checkpoint equals the uninterrupted run.

Deterministic mode: no actor/learner concurrency (each rollout uses the parameters of the update before it), CPU
XLA with single-threaded Eigen (deterministic reductions). A tiny policy_net trains on MD80 with two host devices (actor
and learner). Run A makes 6 updates with checkpoints at 2, 4 and 6; run B resumes from A's update-2 checkpoint and
makes 4 more. At updates 4 and 6 both runs' checkpoints hold identical parameters, Adam state, learner keys and actor
states (every environment's exported state, observations, recurrent memory, actor key and counters).

Slow (three trainer processes, a few minutes); runs in the venv with ``MF_DUEL_NATIVE`` and the MD80 assets.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ASSETS = Path(os.environ.get("MF_TEST_ASSETS", "/path/to/mirrorforce/infra/m1-20261001/assets-md80/project"))
TABLES = next((Path(__file__).resolve().parent / "fixtures" / "announce").glob("announce-tables-*.json"))
TINY = ["--architecture", "policy_net", "--net.d", "32", "--net.heads", "4", "--net.ff", "64", "--net.state-layers", "1",
        "--net.turn-layers", "1", "--net.readout-layers", "1", "--net.semantic-dim", "16", "--net.event-hidden", "48",
        "--net.action-hidden", "48", "--net.readout-hidden", "40", "--net.value-queries", "2", "--net.memory-slots", "8",
        "--actor-blocks"]  # menu blocks: the first step after a resume picks its block from the restored observation


def train(run: Path, updates: int, resume: Path | None = None, extra=()):
    run.mkdir(parents=True)
    # one CPU: XLA's CPU thread pool then has one thread, and reductions keep their order on a loaded machine
    argv = ["taskset", "-c", str(min(os.sched_getaffinity(0))), sys.executable, "-m", "mirrorforce.agent.train.cleanba",
            *TINY, "--no-concurrency",
            "--actor-device-ids", "0", "--learner-device-ids", "1", "--deck", str(ASSETS / "decks"), "--allow-unreviewed-public-effects",
            "--deck-schedule", "cluster_uniform", "--semantic-file", str(ASSETS / "frozen_semantics.npz"),
            "--code-list-file", str(ASSETS / "code_list.txt"), "--cards-db", str(ASSETS / "cards.cdb"),
            "--local-num-envs", "4", "--local-env-threads", "2", "--num-actor-threads", "1", "--num-steps", "32",
            "--num-minibatches", "2", "--update-epochs", "1", "--total-timesteps", str(128 * updates),
            "--learning-rate", "0.0003", "--gamma", "1.0", "--value", "gae", "--gae-lambda", "0.95",
            "--max-options", "192", "--max-steps", "1000", "--timeout", "120", "--log-frequency", "1",
            "--save-interval", "2", "--eval-interval", "0", "--local-eval-episodes", "4",
            "--ckpt-dir", str(run / "ckpt"), "--tb-dir", str(run / "tb"), "--run-name", "exact__20261001",
            "--seed", "11", "--announce-tables", str(TABLES)]
    argv += list(extra)
    if resume is not None:
        argv += ["--resume", str(resume)]
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_COMPILATION_CACHE_DIR=str(run / "jax-cache"),
               XLA_FLAGS="--xla_force_host_platform_device_count=2", PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run(argv, cwd=ASSETS, env=env, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-3000:]
    checkpoints = {}
    for receipt in (run / "ckpt").glob("*.receipt.json"):
        record = json.loads(receipt.read_text())
        checkpoints[record["counters"]["learner_update"]] = receipt.with_name(receipt.name.replace(".receipt.json",
                                                                                               ".ckpt"))
    return checkpoints


def payload(path: Path):
    import flax
    return flax.serialization.msgpack_restore(path.read_bytes())


def equal(a, b, where="payload"):
    if isinstance(a, dict):
        assert sorted(a) == sorted(b), where
        for key in a:
            equal(a[key], b[key], f"{where}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, f"{where}[{i}]")
    else:
        assert np.array_equal(np.asarray(a), np.asarray(b)), where


@pytest.mark.parametrize("extra", [(), ("--recipe", "ataraxos", "--lp-shaping-coef", "0")], ids=["fork", "ataraxos"])
def test_a_resumed_run_equals_the_uninterrupted_run(tmp_path, extra):
    if not os.environ.get("MF_DUEL_NATIVE") or not (ASSETS / "cards.cdb").is_file():
        pytest.skip("set MF_DUEL_NATIVE and MF_TEST_ASSETS")
    a = train(tmp_path / "a", 6, extra=extra)
    assert sorted(a) == [2, 4, 6]
    receipt = json.loads(a[2].with_name(a[2].name.replace(".ckpt", ".receipt.json")).read_text())
    assert receipt["env_state_restored_on_resume"] is True and receipt["resume_contract"]["exact"] is True
    assert receipt["window_law"] == {"window_rows": 512, "chunk_rows": 64, "chunk_cap": 48, "chunk_slots": 4}
    b = train(tmp_path / "b", 4, resume=a[2], extra=extra)
    assert sorted(b) == [4, 6]
    for update in (4, 6):
        left, right = payload(a[update]), payload(b[update])
        equal(left, right, f"update {update}")
        if extra:
            assert "ema" in left  # arm A's EMA is in the checkpoint and resumes exactly
