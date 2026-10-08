"""A0 over several processes (M2): two CPU processes of two host devices each (actor device 0, learners 0 and 1,
gloo collectives) train two iterations as one run; process 0 writes the checkpoints (without environment states,
declared); a single process then resumes from them across the layout change (--resume-topology-change: parameters,
Adam state, EMA and the iteration counter carry over; environments restart) and trains one more iteration.

Slow (three trainer processes); runs in the venv with ``MF_DUEL_NATIVE`` and the MD80 assets.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_a0_iterations import ASSETS, CARD_TABLES, ROOMS, TABLES, TINY  # noqa: E402

PER_UPDATE = 4 * 32  # one process: 4 environments x 32 steps


def argv(run: Path, world: int, rank: int, iterations: int, port: int, extra=()):
    decisions = 2 * PER_UPDATE * 2  # an iteration: two updates of 8 environments in all
    return [sys.executable, "-m", "mirrorforce.agent.train.cleanba", *TINY, "--recipe", "ataraxos",
            "--lp-shaping-coef", "0", "--belief-coef", "0.1", "--iteration-decisions", str(decisions),
            "--iteration-steps", "2", "--iteration-micro-segments", "2", "--iteration-threads", "2",
            "--actor-device-ids", "0", "--learner-device-ids", "0", "1", "--deck", str(ASSETS / "decks"), "--allow-unreviewed-public-effects",
            "--deck-schedule", "cluster_uniform", "--semantic-file", str(ASSETS / "frozen_semantics.npz"),
            "--code-list-file", str(ASSETS / "code_list.txt"), "--cards-db", str(ASSETS / "cards.cdb"),
            "--local-num-envs", str(4 * 2 // world), "--local-env-threads", "2", "--num-actor-threads", "1",
            "--num-steps", "32", "--num-minibatches", "1", "--update-epochs", "1",
            "--total-timesteps", str(decisions * iterations), "--gamma", "1.0",
            "--max-options", "192", "--max-steps", "1000", "--timeout", "120", "--log-frequency", "1",
            "--save-interval", "1", "--eval-interval", "0", "--local-eval-episodes", "4",
            "--ckpt-dir", str(run / "ckpt"), "--tb-dir", str(run / f"tb{rank}"), "--run-name", "a0dist__20261002",
            "--seed", "19", "--announce-tables", str(TABLES), "--card-tables", str(CARD_TABLES),
            "--room-format", "md-2026H2", "--room-format-table", str(ROOMS),
            *(["--distributed", "--coordinator-address", f"127.0.0.1:{port}", "--num-processes", str(world),
               "--process-id", str(rank)] if world > 1 else []), *extra]


def env(run: Path):
    return dict(os.environ, JAX_PLATFORMS="cpu", JAX_COMPILATION_CACHE_DIR=str(run / "jax-cache"),
                XLA_FLAGS="--xla_force_host_platform_device_count=2", PYTHONDONTWRITEBYTECODE="1")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def lines(text, prefix):
    return [json.loads(l[len(prefix):]) for l in text.splitlines() if l.startswith(prefix)]


def test_two_processes_train_as_one_run_and_resume_across_the_layout(tmp_path):
    if not os.environ.get("MF_DUEL_NATIVE") or not (ASSETS / "cards.cdb").is_file() or not CARD_TABLES.is_file():
        pytest.skip("set MF_DUEL_NATIVE, MF_TEST_ASSETS and MF_TEST_CARD_TABLES")
    run = tmp_path / "two"
    run.mkdir()
    port = free_port()
    procs = [subprocess.Popen(argv(run, 2, rank, 2, port), cwd=ASSETS, env=env(run), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) for rank in range(2)]
    outputs = [p.communicate(timeout=3000)[0] for p in procs]
    for rank, (p, out) in enumerate(zip(procs, outputs)):
        (run / f"log{rank}.txt").write_text(out)
        assert p.returncode == 0, out[-3000:]
    a0 = [lines(out, "A0 ") for out in outputs]
    assert [l["iteration"] for l in a0[0]] == [1, 2] and [l["iteration"] for l in a0[1]] == [1, 2]
    for left, right in zip(a0[0], a0[1]):  # one run: the pmean'd terms agree in every process
        assert left["loss"] == pytest.approx(right["loss"], rel=1e-6)
        assert left["decisions"] == right["decisions"] == 2 * PER_UPDATE  # each process's half of the iteration
    receipts = sorted((run / "ckpt").glob("*.receipt.json"))
    by_iteration = {json.loads(r.read_text())["counters"]["learner_update"]: r for r in receipts}
    assert sorted(by_iteration) == [1, 2]
    receipt = json.loads(by_iteration[2].read_text())
    assert receipt["env_state_restored_on_resume"] is False and receipt["resume_contract"]["exact"] is False
    assert receipt["counters"]["global_step"] == 8 * PER_UPDATE  # two global iterations

    one = tmp_path / "one"
    one.mkdir()
    ckpt = by_iteration[2].with_name(by_iteration[2].name.replace(".receipt.json", ".ckpt"))
    done = subprocess.run(argv(one, 1, 0, 1, 0, ["--resume", str(ckpt), "--resume-topology-change"]),
                          cwd=ASSETS, env=env(one), capture_output=True, text=True, timeout=3000)
    out = done.stdout + done.stderr
    (one / "log.txt").write_text(out)
    assert done.returncode == 0, out[-3000:]
    change = lines(out, "TOPOLOGY-CHANGE ")
    assert change and change[0]["iteration"] == 2
    assert [l["iteration"] for l in lines(out, "A0 ")] == [3]

    # a resume of several processes on another environment build, the checkpoint on process 0's host only: process
    # 0 reads it and the other receives it (--resume-env-change declares the build change; environments restart)
    from mirrorforce.agent.train.checkpoint_store import write_checkpoint
    receipts = {json.loads(r.read_text())["counters"]["learner_update"]: r for r in (one / "ckpt").glob("*.receipt.json")}
    fields = json.loads(receipts[3].read_text())
    assert fields["counters"]["global_step"] == 12 * PER_UPDATE
    assert fields["a0_counter_basis"]["parent_checkpoint"] == ckpt.stem
    assert fields["a0_counter_basis"]["parent_global_step"] == 8 * PER_UPDATE
    assert fields["a0_counter_basis"]["actors_restored"] is False
    payload = receipts[3].with_name(receipts[3].name.replace(".receipt.json", ".ckpt")).read_bytes()
    built = {**{k: v for k, v in fields.items() if k not in ("payload_sha256", "payload_bytes")},
             "native_sha256": "0" * 64}
    foreign = tmp_path / "foreign"
    sha = write_checkpoint(foreign, payload, built)
    refused = subprocess.run(argv(tmp_path / "undeclared", 1, 0, 1, 0, ["--resume", str(foreign / f"{sha}.ckpt"),
                                                                        "--resume-topology-change"]),
                             cwd=ASSETS, env=env(one), capture_output=True, text=True, timeout=3000)
    assert refused.returncode != 0 and "native_sha256 differs" in refused.stdout + refused.stderr
    again = tmp_path / "again"
    again.mkdir()
    port = free_port()
    paths = [foreign / f"{sha}.ckpt", tmp_path / "absent" / f"{sha}.ckpt"]  # process 1 never reads its path
    procs = [subprocess.Popen(argv(again, 2, rank, 1, port, ["--resume", str(paths[rank]), "--resume-topology-change",
                                                             "--resume-env-change"]),
                              cwd=ASSETS, env=env(again), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for rank in range(2)]
    outputs = [p.communicate(timeout=3000)[0] for p in procs]
    for rank, (p, out) in enumerate(zip(procs, outputs)):
        (again / f"log{rank}.txt").write_text(out)
        assert p.returncode == 0, out[-3000:]
        change = lines(out, "ENV-CHANGE ")
        assert change and change[0]["iteration"] == 3 and sorted(change[0]["changed"]) == ["native_sha256"]
    a0 = [lines(out, "A0 ") for out in outputs]
    assert [l["iteration"] for l in a0[0]] == [4] == [l["iteration"] for l in a0[1]]
    assert a0[0][0]["loss"] == pytest.approx(a0[1][0]["loss"], rel=1e-6)
    receipt = json.loads(next((again / "ckpt").glob("*.receipt.json")).read_text())
    assert receipt["counters"]["learner_update"] == 4
    assert receipt["counters"]["global_step"] == 16 * PER_UPDATE
    assert receipt["env_changes"] == [{"iteration": 3, "changed": ["native_sha256"]}]
