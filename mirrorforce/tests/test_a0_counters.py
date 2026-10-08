"""CPU-only A0 counter accounting; no model, optimizer, environment or historical file is changed."""
import ast
import copy
import json
from pathlib import Path

import pytest

from mirrorforce.agent.train import a0_counters as C

PARENT = "a" * 64
DECISIONS = 4_915_200


def actors(step, *, threads=2, envs=3, update=845):
    return [{"global_step": step, "update": update, "envs": [b"unchanged engine"] * envs,
             "key": [10, 11], "rstate": {"bytes": b"unchanged memory"}} for _ in range(threads)]


def resume(saved_step, restored=None, *, iteration=422, **shape):
    return C.resume_basis(PARENT, {"global_step": saved_step, "learner_update": iteration}, restored,
                          **dict({"actor_threads": 2, "envs_per_actor": 3, "expected_update": 845}, **shape))


def test_topology_reset_adds_the_recorded_parent_once():
    basis = resume(1_808_793_600)
    assert C.iteration_counters(basis, DECISIONS, 423) == {
        "global_step": 1_813_708_800, "learner_update": 423}
    assert basis["parent_checkpoint"] == PARENT and basis["parent_global_step"] == 1_808_793_600
    assert basis["actor_start_global_step"] == 0 and not basis["actors_restored"]
    assert basis["history_reconstructed"] is False


def test_absolute_actor_restoration_neither_sums_threads_nor_double_counts_parent():
    restored = actors(1_808_793_600)
    before = copy.deepcopy(restored)
    basis = resume(1_808_793_600, restored)
    assert basis["offset"] == 0 and basis["actors_restored"]
    assert C.iteration_counters(basis, 1_808_793_600 + DECISIONS, 423)["global_step"] == 1_813_708_800
    assert restored == before  # actor RNG, native states and recurrent state are not rebased


def test_raw_actor_restore_after_a_reset_keeps_the_existing_base_across_resumes():
    parent_base = 1_808_793_600
    for completed in (1, 2, 3):
        local = completed * DECISIONS
        restored = actors(local, update=(422 + completed) * 2 + 1)
        basis = resume(parent_base + local, restored, expected_update=(422 + completed) * 2 + 1)
        assert basis["offset"] == parent_base
        assert C.iteration_counters(basis, local + DECISIONS, 423 + completed)["global_step"] == \
            parent_base + (completed + 1) * DECISIONS


def test_a_relative_inherited_checkpoint_is_not_silently_repaired_from_its_iteration():
    basis = resume(DECISIONS, iteration=423)
    result = C.iteration_counters(basis, DECISIONS, 424)
    assert result == {"global_step": 2 * DECISIONS, "learner_update": 424}
    assert result["global_step"] != 424 * DECISIONS
    assert basis["parent_global_step"] == DECISIONS and not basis["history_reconstructed"]
    assert json.loads(json.dumps(basis)) == basis


@pytest.mark.parametrize("offset", [0, 987_654_321, 2 ** 45])
def test_fresh_configured_offsets_and_large_integer_counts(offset):
    basis = C.initial_basis(offset)
    assert basis["parent_checkpoint"] is basis["parent_global_step"] is None
    assert basis["origin"] == "configured_offset"
    assert C.iteration_counters(basis, DECISIONS, 1) == {"global_step": offset + DECISIONS, "learner_update": 1}


@pytest.mark.parametrize("fault", ["divergent", "above_parent", "thread_count", "env_batch", "update", "empty"])
def test_inconsistent_restored_actor_boundaries_are_rejected(fault):
    restored = actors(100)
    if fault == "divergent": restored[1]["global_step"] = 99
    if fault == "above_parent": restored = actors(101)
    if fault == "thread_count": restored.pop()
    if fault == "env_batch": restored[1]["envs"].append(b"different batch size")
    if fault == "update": restored[1]["update"] = 843
    if fault == "empty": restored = []
    with pytest.raises(ValueError):
        resume(100, restored)


def test_changed_actor_batch_requires_discarding_actor_states_first():
    old = actors(100)
    with pytest.raises(ValueError, match="environment batch"):
        resume(100, old, envs_per_actor=4)
    with pytest.raises(ValueError, match="thread count"):
        resume(100, old, actor_threads=4)
    with pytest.raises(ValueError, match="iteration boundary"):
        resume(100, old, expected_update=423 * 4 + 1)
    # A declared topology/env reset clears actors before resume_basis is called.
    fresh = resume(100, None, actor_threads=4, envs_per_actor=4, expected_update=423 * 4 + 1)
    assert fresh["offset"] == 100 and C.iteration_counters(fresh, 128, 424)["global_step"] == 228


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_counter_metadata_must_be_nonnegative_integral(bad):
    with pytest.raises(ValueError): C.initial_basis(bad)
    with pytest.raises(ValueError): resume(bad)
    with pytest.raises(ValueError): C.iteration_counters(C.initial_basis(), bad, 1)


def test_actor_counter_cannot_move_backwards_from_a_restored_boundary():
    with pytest.raises(ValueError, match="regressed"):
        C.iteration_counters(resume(100, actors(100)), 99, 423)


def test_checkpoint_and_log_are_wired_to_the_same_counter_record():
    # Inspect the actual thin wiring without importing native/JAX or running a
    # trainer. The accounting function above is exercised directly, not copied.
    path = Path(C.__file__).with_name("cleanba.py")
    tree = ast.parse(path.read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_iterations")
    assignment = next(n for n in ast.walk(run) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "counters" for t in n.targets))
    assert ast.unparse(assignment.value) == "a0_counters.iteration_counters(args.a0_counter_basis, global_step, iteration)"
    saves = [n for n in ast.walk(run) if isinstance(n, ast.Call) and ast.unparse(n.func) == "state_io.save"]
    assert len(saves) == 1 and ast.unparse(saves[0].args[3]) == "counters"
    receipt = dict((k.value, v) for k, v in zip(saves[0].args[4].keys, saves[0].args[4].values) if k is not None)
    assert ast.unparse(receipt["a0_counter_basis"]) == "args.a0_counter_basis"
    assert any(isinstance(n, ast.Dict) and any(k is None and isinstance(v, ast.Name) and v.id == "counters"
               for k, v in zip(n.keys, n.values)) for n in ast.walk(run))  # CHECKPOINT JSON unpacks that same record
    logs = [n.value for n in ast.walk(run) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "line" for t in n.targets)]
    assert len(logs) == 1
    logged = {k.value: v for k, v in zip(logs[0].keys, logs[0].values) if k is not None}
    assert ast.unparse(logged["global_step"]) == "counters['global_step']"
