import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mirrorforce.netduel import agent_core_upgrade as U, search_rules as R
from test_search_rules import rules as rules_fixture, sha


@pytest.fixture
def rules(tmp_path):
    return rules_fixture.__wrapped__(tmp_path)


def evidence(report):
    raw = json.dumps(report, sort_keys=True)
    return {"json": raw, "sha256": hashlib.sha256(raw.encode()).hexdigest()}


def example(rules):
    files, args = rules
    original = R.verify_rules(files, **args)
    checkpoint = args["checkpoint"]
    checkpoint["payload_sha256"] = "a" * 64
    checkpoint["native_core"]["core_tree"] = "1" * 40
    reference = {"schema": "mirrorforce_duelpool_build/v1", "uncommitted_worktree_build": False,
        "core_commit": checkpoint["native_core"]["core_commit"], "core_tree": "1" * 40,
        "source_tree": "2" * 40, "shared_tree": "3" * 40, "shared_sources": ["encoding.cc"],
        "compiler": "fixture", "flags": ["-O2"], "deps": {"lua": "5" * 64},
        "outputs": {"libmfcore.so": sha(b"core"), "liblua5.3.so.0": "5" * 64}}
    target = copy.deepcopy(reference)
    target.update(core_commit="new-core", core_tree="4" * 40)
    target["outputs"]["libmfcore.so"] = sha(b"core2")
    for path in (files.core, files.service_core):
        path.write_bytes(b"core2")
    args["service"].update(checkpoint_sha256="a" * 64, receipt_sha256=U.receipt_identity_sha(checkpoint), native_build=target)
    trace = {"finished": True, "winner": 0, "turns": 4, "responses": 120, "packets": 400,
             "responses_sha256": "a" * 64, "packets_sha256": "b" * 64, "menus_sha256": "c" * 64,
             "script_errors": [], "menu_rounds": 80, "truncations": 0, "fallbacks": 0}
    report = {"schema": U.SCHEMA, "law": U.LAW, "passed": True, "training_eligible": False,
              "training_core": copy.deepcopy(checkpoint["native_core"]), "reference_build": reference,
              "target_build": copy.deepcopy(target), "checkpoint_sha256": "a" * 64,
              "checkpoint_receipt_canonical_sha256": U.receipt_identity_sha(checkpoint),
              "stage_sha256": args["stage_sha256"], "scripts": original["scripts"], "assets": original["assets"],
              "traces": [{"seed": seed, "policy_seed": seed + 1000, "target_shuffle_plan_mode": 0,
                           "reference": copy.deepcopy(trace), "target": copy.deepcopy(trace)} for seed in U.SEEDS]}
    return files, args, report


def test_explicit_upgrade_preserves_training_receipt_and_keeps_old_default_refusal(rules, monkeypatch):
    from mirrorforce.puzzle import core
    monkeypatch.setattr(core, "get_core", lambda **_: pytest.fail("pure verifier loaded core"))
    files, args, report = example(rules)
    before = copy.deepcopy(args["checkpoint"])
    with pytest.raises(R.RulesMismatch):
        R.verify_rules(files, **args)
    accepted = R.verify_rules(files, **args, core_upgrade=evidence(report))
    assert accepted["core_commit"] == "new-core" and accepted["core_sha256"] == sha(b"core2")
    assert accepted["core_upgrade"]["training_core"] == before["native_core"]
    assert accepted["core_upgrade"]["training_receipt_changed"] is False
    assert args["checkpoint"] == before


@pytest.mark.parametrize("change", [
    lambda r: r.update(passed=False), lambda r: r.update(training_eligible=True),
    lambda r: r.update(checkpoint_sha256="0" * 64), lambda r: r.update(stage_sha256="0" * 64),
    lambda r: r.update(checkpoint_receipt_canonical_sha256="0" * 64),
    lambda r: r["reference_build"].update(source_tree="wrong"),
    lambda r: r["reference_build"].update(flags=["different"]),
    lambda r: r["reference_build"]["outputs"].update({"libmfcore.so": "0" * 64}),
    lambda r: r.update(traces=r["traces"][:-1]),
    lambda r: r["traces"][0].update(seed=0),
    lambda r: r["traces"][0].update(target_shuffle_plan_mode=1),
    lambda r: r["traces"][0]["target"].update(packets_sha256="0" * 64),
    lambda r: [side.update(finished=False) for side in (r["traces"][0]["reference"], r["traces"][0]["target"])],
    lambda r: [side.update(fallbacks=1) for side in (r["traces"][0]["reference"], r["traces"][0]["target"])],
])
def test_changed_but_self_consistently_rehashed_upgrade_is_rejected(rules, change):
    files, args, report = example(rules)
    change(report)
    with pytest.raises(R.RulesMismatch):
        R.verify_rules(files, **args, core_upgrade=evidence(report))


def test_pure_upgrade_module_does_not_import_a_native_or_model_stack():
    code = "import sys; import mirrorforce.netduel.agent_core_upgrade; assert not any(k.startswith(('jax','torch','mirrorforce.puzzle','duel_native')) for k in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                   env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
