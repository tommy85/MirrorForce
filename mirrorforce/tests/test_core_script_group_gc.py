"""Weak script temporaries (ygopro-core branch ``mirrorforce-lane132-script-gc``), through the Python ctypes core.

The core used to keep every group a script call created until the outermost call returned, so one Lua call could
fill the duel's arena (deck-61 bursts, see ``test_cxx_core_script_group_bursts.py``). Now a library function's
result group turns weak once it is on the Lua stack, Lua's collector finds the unreachable ones (``Group.__gc``
marks them), and a full collection at a point fixed by the calls deletes them. The requirements this checks:

* one call churning through 300,000 temporary groups keeps the arena small;
* a full collection inside every callback of each library function that fills a result group, or keeps its
  group argument, while it calls back into Lua leaves the result intact (the group is strong until it is on the
  stack), with collections really happening (diagnostic core: ``Debug.CollectGarbage``);
* label objects, KeepAlive, a script calling ``Group.__gc`` itself: holds and the audit; and the audit catching
  a C++ structure that keeps a weak group without holding it (the negative case);
* a script using a released group refuses the duel (``ygopro-core refusal [released_group]``);
* group lifetimes repeat exactly across processes (other arena addresses, another Lua string-hash seed) and under
  GC pressure, and a rollback to a snapshot taken at a selection in the middle of a burst -- the call suspended,
  thousands of temporaries alive -- repeats the rest byte for byte, Lua heap included.

Only the Lua heap size and the count of groups found unreachable but not yet deleted follow Lua's own incremental
cycles, which its per-process string-hash seed shapes; they repeat after a rollback, not across processes.

Each scenario runs in a process of its own (``core_group_gc_scenarios.py``): a refusal is a C++ exception, which
cannot cross ctypes. ``MF_SCRIPT_GC_PYCORE_LIB`` / ``MF_SCRIPT_GC_AUDIT_PYCORE_LIB`` point at other builds.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from mirrorforce.puzzle.core import DEFAULT_DB, DEFAULT_SCRIPTS

HERE = Path(__file__).parent
CORE = Path(os.environ.get("MF_SCRIPT_GC_PYCORE_LIB", "/path/to/workspace/build-deps/priv-mfenv/core-pycore-ae46dd9.so"))
AUDIT_CORE = Path(os.environ.get("MF_SCRIPT_GC_AUDIT_PYCORE_LIB",
                                 "/path/to/workspace/build-deps/priv-mfenv/core-pycore-ae46dd9-audit.so"))
GC_PRESSURE = "100,1000"
URABY = 1784619
PER_PROCESS = ("lua_bytes", "pending")   # Lua's own incremental cycles: repeat after a rollback, not across processes

INTERACTIVE_CHECKS = {"duel_select_target", "duel_get_target_count", "duel_select_matching_cards",
                      "group_filter_select", "group_select_with_sum_equal", "group_select_with_sum_greater",
                      "duel_select_synchro_material", "duel_select_tuner_material", "duel_discard_hand", "duel_sets"}
CALLBACK_CHECKS = {"group_filter", "group_get_min_group", "group_get_max_group", "duel_get_matching_group",
                   "duel_get_matching_count", "card_is_synchro_summonable", "card_is_xyz_summonable",
                   "card_is_link_summonable", "duel_select_target", "duel_get_target_count",
                   "duel_select_matching_cards", "group_filter_select", "group_select_with_sum_equal",
                   "group_select_with_sum_greater", "duel_discard_hand"}


def scenario(name, core, *args, env=None, expect_ok=True):
    if not (core.is_file() and Path(DEFAULT_DB).is_file() and Path(DEFAULT_SCRIPTS).is_dir()):
        pytest.skip(f"needs the fixed core {core}, the card database and scripts")
    environment = dict(os.environ, **(env or {}))
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HERE.parent), environment.get("PYTHONPATH")]))
    proc = subprocess.run([sys.executable, str(HERE / "core_group_gc_scenarios.py"), name, str(core), *map(str, args)],
                          capture_output=True, text=True, env=environment, timeout=900)
    if not expect_ok:
        return proc
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_one_call_churning_through_temporary_groups_keeps_the_arena_small():
    result = scenario("burst", CORE)
    assert result["log"] == [] and result["audit"] == [0, ""]
    # 300,000 groups: kept to the outermost return they took well over 100 MiB of arena
    assert result["arena_growth"] < 16 << 20, result["arena_growth"]
    assert result["after"]["collected"] - result["before"]["collected"] > 250_000
    assert result["after"]["groups"] <= result["before"]["groups"] + 1


def test_a_full_collection_inside_every_callback_leaves_the_results_intact():
    result = scenario("forced_gc", AUDIT_CORE)
    checks = result["checks"]
    assert set(checks) == CALLBACK_CHECKS | INTERACTIVE_CHECKS
    failed = {name: check["error"] for name, check in checks.items() if not check["ok"]}
    assert not failed, failed
    idle = {name for name in CALLBACK_CHECKS if checks[name]["collected"] == 0}
    assert not idle, f"no group was collected during {sorted(idle)}"
    assert result["other_log"] == [] and result["audit"] == [0, ""]


def test_holds_survive_collection_and_the_audit_catches_an_unheld_structure():
    result = scenario("holds", AUDIT_CORE)
    assert result["log"] == [], result["log"]
    assert result["audit"] == [0, ""]


def test_a_released_group_refuses_the_duel():
    proc = scenario("released_group", CORE, expect_ok=False)
    assert "before the stale use" in proc.stdout and "after the stale use" not in proc.stdout
    assert proc.returncode == -signal.SIGABRT, (proc.returncode, proc.stderr[-2000:])
    assert "ygopro-core refusal [released_group]: " in proc.stderr
    assert "is a group the core already released" in proc.stderr


def test_the_redirect_operation_of_a_sent_card_gets_the_sent_group():
    # From its fourth step the send's processor unit keeps its internal record where the group was; the Crystal
    # Beast redirect used to hand that record to the script as the event group.
    result = scenario("redirect_event", CORE)
    assert result["log"] == ["MFEG 1 true"], result["log"]
    assert URABY in result["szone"], "the redirect did not place the monster in the Spell & Trap Zone"
    assert result["audit"] == [0, ""]


def across_processes(result):
    strip = lambda rows: [{k: v for k, v in row.items() if k not in PER_PROCESS} for row in rows]  # noqa: E731
    return {key: strip(result[key]) for key in ("head", "straight", "replayed")}


def test_group_lifetimes_repeat_across_processes_and_after_a_mid_burst_rollback():
    runs = [scenario("gc_sequence", CORE, noise) for noise in (0, 37, 4001)]
    pressured = [scenario("gc_sequence", AUDIT_CORE, noise, env={"MF_CORE_GC_PRESSURE": GC_PRESSURE})
                 for noise in (0, 4001)]
    for result in runs + pressured:
        assert result["log"] == [] and result["audit"] == [0, ""]
        assert result["at_snapshot"]["call_groups"] > 1000, "the snapshot was not taken in the middle of a burst"
        assert result["after_rollback"] == result["at_snapshot"]
        assert result["replayed"] == result["straight"], "the rollback did not repeat the rest of the burst"
    assert all(across_processes(r) == across_processes(runs[0]) for r in runs), "group lifetimes differ by process"
    # deletion happens only at the fixed points, after a full collection: pressure changes when Lua's own cycles
    # run, not which groups live when
    assert all(across_processes(r) == across_processes(runs[0]) for r in pressured), "GC pressure changed lifetimes"
    messages = lambda r: [row["messages"] for row in r["head"] + r["straight"]]  # noqa: E731
    assert messages(pressured[0]) == messages(runs[0]), "GC pressure changed the game"
    collected = runs[0]["straight"][-1]["collected"]
    assert collected > 50_000, collected
