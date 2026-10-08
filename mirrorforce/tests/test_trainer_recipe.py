"""The trainer's optimization recipe: exact gradient accumulation and a belief head that never moves the policy.

Deterministic mode (no actor/learner concurrency, CPU XLA with single-threaded Eigen), a tiny policy_net on MD80 with two
host devices (actor and learner):

- accumulation: 2 optimizer steps per update over the same minibatches, one gradient per minibatch vs the sum of 2
  micro-batch gradients (every loss term normalized by the whole minibatch's counts): parameters after 2 updates
  agree within fp32 tolerance;
- belief isolation: --belief-coef 0 vs 0.1 for 3 updates: every parameter outside the belief head is byte-identical
  (stop-gradient on the state block and on the shared card table, and a separate clip and Adam state for the head).

Slow (four trainer processes); runs in the venv with ``MF_DUEL_NATIVE`` and the MD80 assets.
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
        "--net.belief-width", "32", "--net.belief-heads", "4", "--net.belief-layers", "1", "--actor-blocks"]


def train(run: Path, updates: int, extra):
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
            "--save-interval", str(updates), "--eval-interval", "0", "--local-eval-episodes", "4",
            "--ckpt-dir", str(run / "ckpt"), "--tb-dir", str(run / "tb"), "--run-name", "recipe__20261001",
            "--seed", "13", "--announce-tables", str(TABLES), *extra]
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_COMPILATION_CACHE_DIR=str(run / "jax-cache"),
               XLA_FLAGS="--xla_force_host_platform_device_count=2", PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run(argv, cwd=ASSETS, env=env, capture_output=True, text=True, timeout=1800)
    (run / "log.txt").write_text(done.stdout + done.stderr)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-3000:]
    receipts = sorted((run / "ckpt").glob("*.receipt.json"))
    final = max(receipts, key=lambda r: json.loads(r.read_text())["counters"]["learner_update"])
    import flax
    return flax.serialization.msgpack_restore(final.with_name(final.name.replace(".receipt.json", ".ckpt")).read_bytes())


def leaves(tree, prefix=""):
    if isinstance(tree, dict):
        for key in sorted(tree):
            yield from leaves(tree[key], f"{prefix}/{key}")
    else:
        yield prefix, np.asarray(tree)


def skip_without_assets():
    if not os.environ.get("MF_DUEL_NATIVE") or not (ASSETS / "cards.cdb").is_file():
        pytest.skip("set MF_DUEL_NATIVE and MF_TEST_ASSETS")


def test_accumulated_gradients_equal_one_batch_gradients(tmp_path):
    skip_without_assets()
    one = train(tmp_path / "one", 2, [])
    two = train(tmp_path / "two", 2, ["--grad-accum", "2"])
    worst = 0.0
    for (name, a), (other, b) in zip(leaves(one["state"]["params"]), leaves(two["state"]["params"])):
        assert name == other
        worst = max(worst, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0)
    assert worst < 1e-5, worst


def test_the_belief_loss_leaves_every_other_parameter_byte_identical(tmp_path):
    skip_without_assets()
    without = train(tmp_path / "without", 3, [])
    with_belief = train(tmp_path / "with", 3, ["--belief-coef", "0.1"])
    changed = []
    for (name, a), (other, b) in zip(leaves(without["state"]["params"]), leaves(with_belief["state"]["params"])):
        assert name == other
        if not name.startswith("/belief/") and not np.array_equal(a, b):
            changed.append(name)
    assert not changed, changed[:10]
    belief = [b for name, b in leaves(with_belief["state"]["params"]) if name.startswith("/belief/")]
    untrained = [a for name, a in leaves(without["state"]["params"]) if name.startswith("/belief/")]
    assert any(not np.array_equal(a, b) for a, b in zip(untrained, belief))  # the head itself did train


def test_arm_a_accumulated_gradients_equal_one_batch_gradients(tmp_path):
    """The first optimizer step of an update starts from equal parameters in both runs, so its gradient norm (logged
    before clipping) compares the accumulated gradient with the one-batch gradient directly; later steps differ by
    Adam's amplification of near-zero gradients at eps 1e-8, not by the accumulation."""
    skip_without_assets()
    arm = ["--recipe", "ataraxos", "--lp-shaping-coef", "0", "--belief-coef", "0.1"]
    one = train(tmp_path / "one", 1, arm)
    two = train(tmp_path / "two", 1, arm + ["--grad-accum", "2"])
    first = lambda run: json.loads(next(l for l in (tmp_path / run / "log.txt").read_text().splitlines()
                                        if l.startswith("ATARAXOS"))[len("ATARAXOS "):])["g_norm_first"]
    a, b = first("one"), first("two")
    assert a > 0 and abs(a - b) <= 1e-5 * a, (a, b)
    assert "ema" in one and "ema" in two


def test_withdrawn_decisions_are_marked_on_the_stored_steps():
    """mark_withdrawn: info:illegal_activation [0] + [1] most recent decisions of an environment are flagged (this
    step's included); more than the segment holds are counted as shipped; shortfalls and cards are counted; a recipe
    other than arm A refuses withdrawals."""
    from mirrorforce.agent.train.cleanba import GuardMonitor, Transition, mark_withdrawn
    step = lambda: Transition(*([None] * 8), withdrawn=np.zeros(3, np.bool_))
    storage = [step(), step(), step()]
    ill = np.zeros((3, 6), np.int32)
    ill[1, :3] = (1, 1, 38511382)  # two decisions of env 1 (one per player)
    ill[2, 0], ill[2, 2] = 4, 76725398  # four decisions of env 2: one is in the shipped segment
    ill[0, 4:] = (2, 8487449)
    monitor = GuardMonitor()
    mark_withdrawn(storage, {"illegal_activation": ill}, monitor, "ataraxos")
    flags = np.stack([t.withdrawn for t in storage])
    np.testing.assert_array_equal(flags, [[False, False, True], [False, True, True], [False, True, True]])
    report = monitor.interval()
    assert report["withdrawn"] == 6 and report["withdrawn_in_shipped_segment"] == 1
    assert report["withdrawn_cards"] == {76725398: 4, 38511382: 2} and report["shortfalls"] == 2
    assert report["shortfall_cards"] == {8487449: 1} and monitor.interval()["withdrawn"] == 0
    with pytest.raises(RuntimeError):
        mark_withdrawn([step()], {"illegal_activation": ill}, GuardMonitor(), "fork")
