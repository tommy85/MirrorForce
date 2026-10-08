import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from mirrorforce.netduel.search_numeric import AUDIT_SCHEMA, NumericContractError, numeric_contract, load_numeric_evidence
from mirrorforce.netduel.agent_search_policy import SearchConfig, SearchPolicyError, create_search_identity, plan_root


def fixture():
    batch = {"law": "constant-batch-full-menu/v1", "size": 64, "menu_rows": 192, "applies_to": "all_forwards"}
    backend = {"backend": "checkpoint", "checkpoint_sha256": "a" * 64, "receipt_sha256": "b" * 64,
               "weights": "iterate", "compute_dtype": "bfloat16", "magnet": "uniform_legal",
               "training_native_sha256": "c" * 64, "semantic_file_sha256": "d" * 64,
               "card_tables_sha256": "e" * 64, "search_batching": batch}
    service = {**backend, "serving_native_sha256": "f" * 64, "client_config": {"max_options": 192},
               "native_build": {"core_commit": "9" * 40, "outputs": {"libmfcore.so": "1" * 64}},
               "device": {"platform": "gpu", "kind": "NVIDIA L20", "count": 1, "jax": "0.5.3"},
               "opponent_recipe_mode": "mirror", "selection": "sample", "temperature": 1.0}
    replays = [{"seat": p, "decisions": 4, "noninitial_memory_steps": 1, "noninitial_memory": True,
                "trace_equal": True, "memory_equal": True, "live_trace_sha256": "2" * 64,
                "replay_trace_sha256": "2" * 64, "live_memory_sha256": "3" * 64,
                "replay_memory_sha256": "3" * 64} for p in (0, 1)]
    invariance = [{"requested_rows": n, "exact": True, "close_1e5": True, "finite": True,
                   "valid_inputs_unchanged": True, "max_abs": 0, "max_abs_memory": 0, "max_abs_logits": 0,
                   "max_policy_kl": 0, "max_policy_tv": 0, "greedy_changes": 0, "seconds": [0.1] * 3,
                   "comparisons": n * 3} for n in (1, 3, 63, 64, 65, 129) for _ in range(3)]
    passed = {"mode": "bfloat16-constant64", "identity": backend, "passed_exact": True,
              "samples": 8, "replays": replays, "invariance": invariance}
    failed = {"mode": "bfloat16-constant16", "passed_exact": False}
    report = {"schema": AUDIT_SCHEMA, "training_eligible": False, "passed_exact": False,
              "native_sha256": service["serving_native_sha256"], "native_build": service["native_build"],
              "core_sha256": "1" * 64, "device": service["device"], "modes": [passed, failed]}
    return copy.deepcopy(service), copy.deepcopy(report)


def evidence(report):
    raw = json.dumps(report, sort_keys=True, indent=1)
    return {"json": raw, "sha256": hashlib.sha256(raw.encode()).hexdigest()}


def test_specific_passed_mode_is_admitted_without_relabeling_global_failure():
    service, report = fixture()
    result = numeric_contract(service, mode="constant_batch_full_menu", evidence=evidence(report))
    assert result["passed_exact"] and result["audit_mode"] == "bfloat16-constant64"
    assert not report["passed_exact"] and not result["full_search_budget_accepted"]
    assert not result["equal_to_old_batch1"]


@pytest.mark.parametrize("change", [
    lambda r: r.update(native_sha256="0" * 64),
    lambda r: r.update(core_sha256="0" * 64),
    lambda r: r["device"].update(kind="other GPU"),
    lambda r: r["modes"][0].update(passed_exact=False),
    lambda r: r["modes"][0]["identity"].update(checkpoint_sha256="0" * 64),
    lambda r: r["modes"][0]["identity"].update(receipt_sha256="0" * 64),
    lambda r: r["modes"][0]["identity"].update(weights="ema"),
    lambda r: r["modes"][0]["identity"].update(card_tables_sha256="0" * 64),
    lambda r: r["modes"][0]["replays"][0].update(noninitial_memory=False),
    lambda r: r["modes"][0]["replays"][0].update(live_memory_sha256=None, replay_memory_sha256=None),
    lambda r: r["modes"][0].update(samples=7),
    lambda r: r["modes"][0]["invariance"].pop(),
    lambda r: r["modes"][0]["invariance"][0].update(max_abs_memory=0.01),
    lambda r: r["modes"][0]["invariance"][0].update(exact=False),
    lambda r: r["modes"].append(copy.deepcopy(r["modes"][0])),
])
def test_incomplete_changed_or_foreign_gpu_evidence_is_not_admitted(change):
    service, report = fixture()
    change(report)
    with pytest.raises(NumericContractError):
        numeric_contract(service, mode="constant_batch_full_menu", evidence=evidence(report))


def test_failed16_and_missing_or_wrong_file_digest_stay_closed(tmp_path):
    service, report = fixture()
    service["search_batching"]["size"] = 16
    with pytest.raises(NumericContractError, match="did not pass"):
        numeric_contract(service, mode="constant_batch_full_menu", evidence=evidence(report))
    service, report = fixture()
    with pytest.raises(NumericContractError, match="raw GPU"):
        numeric_contract(service, mode="constant_batch_full_menu")
    blob = evidence(report)
    path = tmp_path / "audit.json"
    path.write_text(blob["json"])
    assert load_numeric_evidence(path, blob["sha256"]) == blob
    with pytest.raises(NumericContractError, match="SHA"):
        load_numeric_evidence(path, "0" * 64)
    blob["json"] += " "
    with pytest.raises(NumericContractError, match="SHA"):
        numeric_contract(service, mode="constant_batch_full_menu", evidence=blob)


def test_pure_identity_creation_never_loads_a_core_and_critic_is_not_a_fake_rollout(monkeypatch):
    from mirrorforce.puzzle import core
    monkeypatch.setattr(core, "get_core", lambda **kwargs: pytest.fail("parent process loaded a native core"))
    service, report = fixture()
    rules = {"schema": "mirrorforce_search_rules/v1", "training_eligible": False,
             "core_sha256": "1" * 64, "core_commit": "9" * 40}
    identity = create_search_identity(service, rules, {"law": "public-fixture/v1"}, SearchConfig(),
                                      numeric_mode="constant_batch_full_menu", numeric_evidence=evidence(report))
    assert identity["numeric_contract"]["passed_exact"] and identity["settings"]["estimator"] == "rollout"
    with pytest.raises(SearchPolicyError, match="verified T6 Q"):
        plan_root(None, None, None, None, None, call=None, config=SearchConfig(estimator="critic"), seed=1, rng=None)


def test_registered_real_audit_roundtrip_when_provided():
    path = os.environ.get("MF_NUMERIC_AUDIT")
    if not path:
        pytest.skip("set MF_NUMERIC_AUDIT for a content-addressed real GPU report")
    path = Path(path)
    blob = load_numeric_evidence(path, path.stem.rsplit("-", 1)[-1])
    report = json.loads(blob["json"])
    selected = next(m for m in report["modes"] if m["mode"] == "bfloat16-constant64")
    service = {**selected["identity"], "serving_native_sha256": report["native_sha256"],
               "native_build": report["native_build"], "device": report["device"],
               "client_config": {"max_options": 192}}
    result = numeric_contract(service, mode="constant_batch_full_menu", evidence=blob)
    assert result["passed_exact"] and result["samples"] >= 2


def test_fixed_public_geometry_is_bound_to_same_shape_gpu_evidence():
    from mirrorforce.agent.inference_geometry import FIXED_PUBLIC_B64
    service, report = fixture()
    service["inference_geometry"] = dict(FIXED_PUBLIC_B64)
    # Old dynamic-context evidence is not an admission for the new shared forward.
    with pytest.raises(NumericContractError, match="different weights, receipts, tables or computation"):
        numeric_contract(service, mode="constant_batch_full_menu", evidence=evidence(report))
    report["modes"][0]["identity"]["inference_geometry"] = dict(FIXED_PUBLIC_B64)
    result = numeric_contract(service, mode="constant_batch_full_menu", evidence=evidence(report))
    assert result["inference_geometry"] == FIXED_PUBLIC_B64 and result["passed_exact"]
    del service["inference_geometry"]
    with pytest.raises(NumericContractError, match="different weights, receipts, tables or computation"):
        numeric_contract(service, mode="constant_batch_full_menu", evidence=evidence(report))


@pytest.mark.parametrize("field,value", [("event_rows", 256), ("chunk_slots", 24), ("memory_slots", 32),
                                        ("menu_rows", 32), ("compute_dtype", "float32"),
                                        ("law", "unregistered")])
def test_shared_geometry_cannot_weaken_context_or_silently_change_precision(field, value):
    from mirrorforce.agent.inference_geometry import FIXED_PUBLIC_B64
    service, report = fixture()
    service["inference_geometry"] = {**FIXED_PUBLIC_B64, field: value}
    report["modes"][0]["identity"]["inference_geometry"] = dict(service["inference_geometry"])
    with pytest.raises(NumericContractError, match="inference geometry"):
        numeric_contract(service, mode="constant_batch_full_menu", evidence=evidence(report))


def test_fixed_geometry_rejects_singleton_claim_wrong_batch_and_foreign_profile():
    from mirrorforce.agent.inference_geometry import FIXED_PUBLIC_B64
    service, report = fixture()
    service["inference_geometry"] = dict(FIXED_PUBLIC_B64)
    with pytest.raises(NumericContractError, match="constant B64"):
        numeric_contract(service, mode="exact_batch1")
    service["search_batching"]["size"] = 16
    with pytest.raises(NumericContractError, match="constant B64"):
        numeric_contract(service, mode="constant_batch_full_menu", evidence=evidence(report))
    service["search_batching"]["size"] = 64
    report["modes"][0]["identity"]["inference_geometry"] = {**FIXED_PUBLIC_B64, "event_rows": 256}
    with pytest.raises(NumericContractError, match="different weights, receipts, tables or computation"):
        numeric_contract(service, mode="constant_batch_full_menu", evidence=evidence(report))
