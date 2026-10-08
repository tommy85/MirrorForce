"""A0's iteration trainer (cleanba --iteration-decisions): strict Ataraxos iterations with a compressed host buffer.

Deterministic mode, a tiny policy_net on MD80 with two host devices (actor and learner), the trainer pinned to one CPU.
An iteration is 2 updates of 4 environments x 32 steps (8 segments), trained in 2 optimizer steps of 4 segments, each
accumulated over 2 micro-batches of 2 segments. Run A makes 3 iterations with a checkpoint after each; run B resumes
from A's first and makes 2 more: at iterations 2 and 3 both hold identical parameters, Adam state, EMA, keys and
actor states (exact resume at iteration boundaries). The log carries one A0 line per iteration with the schedules
over the iteration count.

Slow (two trainer processes); runs in the venv with ``MF_DUEL_NATIVE`` and the MD80 assets.
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
DECISIONS = 2 * 4 * 32
CARD_TABLES = Path(os.environ.get(
    "MF_TEST_CARD_TABLES", "/path/to/mirrorforce/infra/card-tables/"
    "card-tables-b3ccc747e6950081c334c54bbdb458c3ec071dc6f51ec2af089ee56ec65f3968.npz"))
ROOMS = next((Path(__file__).resolve().parent / "fixtures" / "room_formats").glob("room-formats-*.json"))


def train(run: Path, iterations: int, resume: Path | None = None, extra=(), refused: str | None = None):
    run.mkdir(parents=True)
    argv = ["taskset", "-c", str(min(os.sched_getaffinity(0))), sys.executable, "-m", "mirrorforce.agent.train.cleanba",
            *TINY, "--no-concurrency", "--recipe", "ataraxos", "--lp-shaping-coef", "0", "--belief-coef", "0.1",
            "--iteration-decisions", str(DECISIONS), "--iteration-steps", "2", "--iteration-micro-segments", "2",
            "--iteration-threads", "2",
            "--actor-device-ids", "0", "--learner-device-ids", "1", "--deck", str(ASSETS / "decks"), "--allow-unreviewed-public-effects",
            "--deck-schedule", "cluster_uniform", "--semantic-file", str(ASSETS / "frozen_semantics.npz"),
            "--code-list-file", str(ASSETS / "code_list.txt"), "--cards-db", str(ASSETS / "cards.cdb"),
            "--local-num-envs", "4", "--local-env-threads", "2", "--num-actor-threads", "1", "--num-steps", "32",
            "--num-minibatches", "1", "--update-epochs", "1", "--total-timesteps", str(DECISIONS * iterations),
            "--gamma", "1.0", "--max-options", "192", "--max-steps", "1000", "--timeout", "120", "--log-frequency", "1",
            "--save-interval", "1", "--eval-interval", "0", "--local-eval-episodes", "4",
            "--ckpt-dir", str(run / "ckpt"), "--tb-dir", str(run / "tb"), "--run-name", "a0__20261002",
            "--seed", "17", "--announce-tables", str(TABLES), "--card-tables", str(CARD_TABLES),
            "--room-format", "md-2026H2", "--room-format-table", str(ROOMS)]
    if resume is not None:
        argv += ["--resume", str(resume)]
    argv += list(extra)
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_COMPILATION_CACHE_DIR=str(run / "jax-cache"),
               XLA_FLAGS="--xla_force_host_platform_device_count=2", PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run(argv, cwd=ASSETS, env=env, capture_output=True, text=True, timeout=3000)
    (run / "log.txt").write_text(done.stdout + done.stderr)
    if refused is not None:  # the run must stop before training with this error
        assert done.returncode != 0 and refused in done.stdout + done.stderr, done.stderr[-3000:]
        return None, None
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    checkpoints = {}
    for receipt in (run / "ckpt").glob("*.receipt.json"):
        record = json.loads(receipt.read_text())
        checkpoints[record["counters"]["learner_update"]] = receipt.with_name(receipt.name.replace(".receipt.json",
                                                                                               ".ckpt"))
    lines = [json.loads(l[3:]) for l in (run / "log.txt").read_text().splitlines() if l.startswith("A0 ")]
    return checkpoints, lines


def equal(a, b, where):
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


def test_a0_iterations_resume_exactly_and_log_their_schedules(tmp_path):
    if not os.environ.get("MF_DUEL_NATIVE") or not (ASSETS / "cards.cdb").is_file() or not CARD_TABLES.is_file():
        pytest.skip("set MF_DUEL_NATIVE, MF_TEST_ASSETS and MF_TEST_CARD_TABLES")
    import flax
    import jax
    a, lines = train(tmp_path / "a", 3)
    assert sorted(a) == [1, 2, 3]
    assert [line["iteration"] for line in lines] == [1, 2, 3] and all(line["decisions"] == DECISIONS for line in lines)
    assert [line["global_step"] for line in lines] == [DECISIONS * i for i in (1, 2, 3)]
    assert lines[0]["lr"] == pytest.approx(1e-4) and lines[0]["temperature"] == pytest.approx(0.05)
    assert lines[1]["temperature"] == pytest.approx(0.05 / 2 ** 0.3, rel=1e-6)  # the power schedule of iteration 1
    assert 0 < lines[0]["kept_fraction"] <= 0.5
    receipt = json.loads(a[1].with_name(a[1].name.replace(".ckpt", ".receipt.json")).read_text())
    assert receipt["a0"] == {"iteration_updates": 2, "iteration_steps": 2, "decisions_per_iteration": DECISIONS,
                             "advantages": "public", "advantages_since": 0, "advantages_switch": None}
    assert receipt["counters"]["global_step"] == DECISIONS
    assert receipt["a0_counter_basis"]["parent_checkpoint"] is None
    from mirrorforce.agent.model.policy_net import PolicyNet
    from mirrorforce.agent.train.policy_io import load_policy
    loaded = {which: load_policy(a[3], which, semantic_file=ASSETS / "frozen_semantics.npz",
                                 code_list_file=ASSETS / "code_list.txt", card_tables_file=CARD_TABLES)
              for which in ("ema", "iterate")}  # the bridge's loader on this run's checkpoint
    for agent, variables, _ in loaded.values():
        table, units = agent.model.apply(variables, method=PolicyNet.semantic_table)
        assert np.isfinite(np.asarray(table)).all()
    ema, iterate = (jax.tree_util.tree_leaves(loaded[w][1]["params"]) for w in ("ema", "iterate"))
    assert any(not np.array_equal(x, y) for x, y in zip(ema, iterate))
    # the bridge's per-decision call: obs:-prefixed arrays of one decision, logits over the real menu rows
    from mirrorforce.agent.train.policy_io import Policy
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_policy_net as shapes_source
    layout = shapes_source.shapes(cards=160, events=512, options=192, recipe=80, activations=64, hints=16,
                                  chunk_rows=64, chunk_delivery=4)
    observation = {f"obs:{k}": v for k, v in shapes_source.random_obs(np.random.default_rng(3), (), layout, 100).items()}
    observation["obs:action_ir_"][:, 0] = 0
    observation["obs:action_ir_"][:7, 0] = 1
    policy = Policy(*loaded["ema"])
    rstate, logits, value, wdl = policy.act(observation, policy.initial_state(), True)
    assert logits.shape == (7,) and np.isfinite(logits).all() and np.isfinite(value) and wdl.shape == (3,)
    rstate, logits, _, _ = policy.act(observation, rstate, False)
    assert logits.shape == (7,)
    # several decisions in one call equal one call each (rows with different menus)
    other = {k: v.copy() for k, v in observation.items()}
    other["obs:action_ir_"][:, 0] = 0
    other["obs:action_ir_"][:3, 0] = 1
    batch = policy.act_batch([observation, other], [policy.initial_state(), policy.initial_state()], [True, True])
    single = [policy.act(o, policy.initial_state(), True) for o in (observation, other)]
    for (r, l, v, w), (r1, l1, v1, w1) in zip(batch, single):
        assert l.shape == l1.shape
        np.testing.assert_allclose(l, l1, atol=1e-4)
        assert v == pytest.approx(v1, abs=1e-4)
    b, _ = train(tmp_path / "b", 2, resume=a[1])
    assert sorted(b) == [2, 3]
    for iteration in (2, 3):
        left = flax.serialization.msgpack_restore(a[iteration].read_bytes())
        right = flax.serialization.msgpack_restore(b[iteration].read_bytes())
        equal(left, right, f"iteration {iteration}")


CRITIC = ["--export-both-seats", "--critic", "--critic-advantages", "--critic-model.d", "32", "--critic-model.heads", "4",
          "--critic-model.ff", "64", "--critic-model.state-layers", "1", "--critic-model.semantic-dim", "16",
          "--critic-model.event-hidden", "48", "--critic-model.action-hidden", "48",
          "--critic-model.readout-hidden", "40", "--critic-model.memory-slots", "8", "--critic-model.belief-width", "32",
          "--critic-mix-layers", "1"]


def test_a0_with_a_central_critic(tmp_path):
    """--critic: the central critic reads both seats (the batch holds the other seat's view: the trainer refuses an
    empty export), trains on its TD(0.8) returns along each environment's iteration, gives the advantages
    (--critic-advantages), reports held-out Brier scores of itself and the public head, and is checkpointed."""
    if not os.environ.get("MF_DUEL_NATIVE") or not (ASSETS / "cards.cdb").is_file() or not CARD_TABLES.is_file():
        pytest.skip("set MF_DUEL_NATIVE, MF_TEST_ASSETS and MF_TEST_CARD_TABLES")
    import flax
    checkpoints, lines = train(tmp_path / "critic", 2, extra=CRITIC)
    log = (tmp_path / "critic" / "log.txt").read_text()
    critic = [json.loads(l[len("CRITIC "):]) for l in log.splitlines() if l.startswith("CRITIC ")]
    assert [c["iteration"] for c in critic] == [1, 2] and all(c["advantages"] == "critic" for c in critic)
    assert all(np.isfinite(c["loss"]) for c in critic)
    receipt = json.loads(checkpoints[2].with_name(checkpoints[2].name.replace(".ckpt", ".receipt.json")).read_text())
    assert receipt["critic_state"] is True and receipt["critic"]["mix_layers"] == 1 and receipt["a0"]["advantages"] == "critic" and receipt["export_both_seats"]
    payload = flax.serialization.msgpack_restore(checkpoints[2].read_bytes())
    assert "critic" in payload and "params" in payload["critic"]
    # exact resume: the critic's parameters and Adam state come back from the checkpoint
    resumed, _ = train(tmp_path / "resumed", 2, resume=checkpoints[1], extra=CRITIC)
    assert "CRITIC-RESUMED" in (tmp_path / "resumed" / "log.txt").read_text()
    again = flax.serialization.msgpack_restore(resumed[2].read_bytes())
    equal(payload["critic"], again["critic"], "critic")
    equal(payload["state"]["params"], again["state"]["params"], "params")
    # a fresh critic replaces the checkpoint's, under another critic configuration (Monte-Carlo targets, zero init)
    reset, _ = train(tmp_path / "reset", 1, resume=checkpoints[2],
                     extra=CRITIC + ["--resume-reset-critic", "--critic-td-lambda", "1.0", "--critic-zero-init"])
    assert "CRITIC-RESET" in (tmp_path / "reset" / "log.txt").read_text()
    receipt = json.loads(reset[3].with_name(reset[3].name.replace(".ckpt", ".receipt.json")).read_text())
    assert receipt["a0"]["critic_started"] == 2
    assert receipt["critic"]["td_lambda"] == 1.0 and receipt["critic"]["zero_init"] is True


def test_a_critic_is_added_at_an_iteration_boundary(tmp_path):
    """A run without the critic gets one at a declared resume (--resume-add-critic): CRITIC-ADDED, a fresh critic
    and a0.critic_started in the receipts; the same resume undeclared is refused, and so is a later resume of the
    critic run as if it had none."""
    if not os.environ.get("MF_DUEL_NATIVE") or not (ASSETS / "cards.cdb").is_file() or not CARD_TABLES.is_file():
        pytest.skip("set MF_DUEL_NATIVE, MF_TEST_ASSETS and MF_TEST_CARD_TABLES")
    plain, _ = train(tmp_path / "plain", 1)
    train(tmp_path / "undeclared", 2, resume=plain[1], extra=CRITIC, refused="the checkpoint receipt")
    train(tmp_path / "unjustified", 2, resume=plain[1], extra=CRITIC + ["--resume-add-critic"],
          refused="--advantages-switch-evidence")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"critic": {"brier": 0.3}, "public": {"brier": 0.4}}))
    added, _ = train(tmp_path / "added", 2, resume=plain[1],
                     extra=CRITIC + ["--resume-add-critic", "--advantages-switch-evidence", str(evidence)])
    log = (tmp_path / "added" / "log.txt").read_text()
    assert "CRITIC-ADDED" in log and "ADVANTAGES-SWITCH" in log and "CRITIC-RESUMED" not in log
    receipt = json.loads(added[2].with_name(added[2].name.replace(".ckpt", ".receipt.json")).read_text())
    assert receipt["a0"]["critic_started"] == 1 and receipt["a0"]["advantages_since"] == 1
    assert receipt["a0"]["advantages_switch"] == {"iteration": 1, "evidence": json.loads(evidence.read_text())}
    train(tmp_path / "again", 3, resume=added[2], extra=CRITIC + ["--resume-add-critic"],
          refused="export_both_seats differs")


def test_the_kept_rows_step_trains_as_the_full_step(tmp_path):
    """--iteration-kept-rows (belief coefficient 0, so both steps have the same loss): the same A0 lines and
    parameters as the micro-batched full step; belief_rows kept is in the identity, so switching it on at a resume
    is refused unless declared (--resume-declare recipe.belief_rows, DECLARED-CHANGE)."""
    if not os.environ.get("MF_DUEL_NATIVE") or not (ASSETS / "cards.cdb").is_file() or not CARD_TABLES.is_file():
        pytest.skip("set MF_DUEL_NATIVE, MF_TEST_ASSETS and MF_TEST_CARD_TABLES")
    import flax
    kept = ["--iteration-kept-rows", "--iteration-kept-chunk", "8"]
    no_belief = ["--belief-coef", "0"]
    full, full_lines = train(tmp_path / "full", 1, extra=no_belief)
    rows, rows_lines = train(tmp_path / "rows", 1, extra=no_belief + kept)
    for name in ("loss", "policy", "value", "kl", "kept_fraction", "g_norm_first"):
        if name in full_lines[0]:
            assert rows_lines[0][name] == pytest.approx(full_lines[0][name], rel=2e-4, abs=1e-6), name
    assert rows_lines[0]["kept_overflow"] == 0 and rows_lines[0]["delivery_bad"] == 0
    left = flax.traverse_util.flatten_dict(flax.serialization.msgpack_restore(full[1].read_bytes())["state"]["params"])
    right = flax.traverse_util.flatten_dict(flax.serialization.msgpack_restore(rows[1].read_bytes())["state"]["params"])
    for key in left:
        np.testing.assert_allclose(right[key], left[key], atol=2e-5, rtol=2e-4, err_msg="/".join(key))
    receipt = json.loads(rows[1].with_name(rows[1].name.replace(".ckpt", ".receipt.json")).read_text())
    assert receipt["recipe"]["belief_rows"] == "kept"
    train(tmp_path / "undeclared", 2, resume=full[1], extra=no_belief + kept, refused="the checkpoint receipt")
    declared, _ = train(tmp_path / "declared", 2, resume=full[1],
                        extra=no_belief + kept + ["--resume-declare", "recipe.belief_rows"])
    assert "DECLARED-CHANGE" in (tmp_path / "declared" / "log.txt").read_text()
    receipt = json.loads(declared[2].with_name(declared[2].name.replace(".ckpt", ".receipt.json")).read_text())
    assert receipt["declared_changes"] == [{"iteration": 1, "changed": {"recipe.belief_rows": {"from": None,
                                                                                              "to": "kept"}}}]
