import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mirrorforce.agent.train import checkpoint_store
from mirrorforce.agent.train import debiased_ema as loader
from mirrorforce.agent.train import ema_reconstruction as audit
from test_ema_reconstruction import trees


def test_arithmetic_is_explicit_binary64_and_does_not_mutate_inputs():
    initial, payload, _ = trees(200)
    payload["ema"]["another"]["kernel"] *= np.float32(.8)
    before = (audit.tree_identity(initial), audit.tree_identity(payload["ema"]))
    derived, power, mass = loader.debias_parameters(initial, payload["ema"], 200)
    assert power == math.pow(.999, 200) and mass == 1 - power
    for path, actual in audit.flatten(derived).items():
        expected = ((audit.flatten(payload["ema"])[path].astype(np.float64)
                     - power * audit.flatten(initial)[path].astype(np.float64)) / mass).astype(np.float32)
        assert audit.bit_mismatches(actual, expected) == 0
        assert not np.shares_memory(actual, audit.flatten(initial)[path])
        assert not np.shares_memory(actual, audit.flatten(payload["ema"])[path])
    assert before == (audit.tree_identity(initial), audit.tree_identity(payload["ema"]))
    assert audit.tree_identity(derived) != before[1]
    # Correctly retains, rather than silently erasing, historical FP32 roundoff.
    assert any(audit.bit_mismatches(audit.frozen_rows(derived)[p], v)
               for p, v in audit.frozen_rows(initial).items())


@pytest.mark.parametrize("count", [0, -1, True, 2.0, 1000001])
def test_invalid_counts_are_not_silently_coerced(count):
    with pytest.raises(ValueError, match="iteration count"):
        loader.debias_parameters({"p": np.ones(1, np.float32)}, {"p": np.ones(1, np.float32)}, count)


@pytest.mark.parametrize("mutation", ["extra", "missing", "shape", "dtype", "nan", "inf", "overflow"])
def test_bad_full_tree_or_nonfinite_derivation_fails(mutation):
    initial = {"p": np.ones(2, np.float32)}
    ema = {"p": np.ones(2, np.float32)}
    if mutation == "extra":
        ema["q"] = ema["p"]
    elif mutation == "missing":
        ema = {}
    elif mutation == "shape":
        ema["p"] = np.ones(1, np.float32)
    elif mutation == "dtype":
        ema["p"] = ema["p"].astype(np.float64)
    elif mutation in ("nan", "inf"):
        ema["p"][0] = float(mutation)
    else:
        ema["p"][:] = np.finfo(np.float32).max
    with pytest.raises(ValueError):
        loader.debias_parameters(initial, ema, 1)


@pytest.fixture
def bundle(tmp_path):
    serialization = pytest.importorskip("flax.serialization")
    initial, payload, receipt = trees(200)
    payload["ema"]["another"]["kernel"] *= np.float32(.8)
    source = {"source_commit": audit.SOURCE, "verified": True, "files": 6207,
              "localized_commit": "a" * 40, "manifest_sha256": "b" * 64}
    report = {
        "schema": audit.SCHEMA, "source": source, "source_after": copy.deepcopy(source),
        "original_job": {"path": "/original-job.json", "sha256": audit.ORIGINAL_JOB},
        "original_seed": audit.ORIGINAL_SEED, "init_seed": audit.INIT_SEED,
        "frozen_law": audit.FROZEN_LAW, "recurrence_law": audit.RECURRENCE_LAW,
        "predeclared_rows": {"paths": [list(p) for p in audit.FROZEN_PATHS], "start": 1,
                             "stop": 64, "total_scalars": 48384},
        "initial_params": audit.tree_identity(initial), "model": receipt["model"],
        "checks": [], "static_gate_passed": True,
    }
    for count in (2, 140):
        _, old_payload, old_receipt = trees(count)
        check = audit.certify(initial, old_payload, old_receipt, receipt["model"])
        check["original_jax_recurrence_bit_mismatches"] = {path: 0 for path in check["rows"]}
        report["checks"].append(check)

    def publish():
        variables_ref = audit.publish(tmp_path, "initial", ".msgpack",
                                      serialization.msgpack_serialize({"params": initial}))
        report["reconstructed_variables"] = variables_ref
        payload_raw = serialization.msgpack_serialize(payload)
        payload_ref = audit.publish(tmp_path, "payload", ".ckpt", payload_raw)
        receipt.update(payload_sha256=payload_ref["sha256"], payload_bytes=len(payload_raw))
        return {"weights": "debiased_ema", "purpose": "behavior_diagnostic",
                "checkpoint": {"payload": payload_ref, "receipt": audit.publish(
                    tmp_path, "receipt", ".json", checkpoint_store.canonical(receipt))},
                "reconstruction": {"variables": variables_ref, "audit": audit.publish(
                    tmp_path, "audit", ".json", audit.canonical(report))}}

    return initial, payload, receipt, report, publish


def test_loader_preserves_receipt_and_has_separate_explicit_identity(bundle, monkeypatch):
    initial, payload, receipt, report, publish = bundle
    request = publish()
    before = copy.deepcopy(receipt)
    immutable = {r["path"]: Path(r["path"]).read_bytes()
                 for group in ("checkpoint", "reconstruction") for r in request[group].values()}

    def no_reinitialization(*args, **kwargs):
        pytest.fail("the loader must never run a fresh initializer")

    monkeypatch.setattr(audit, "reconstruct", no_reinitialization)
    params, returned, identity = loader.load_debiased_parameters(**request)
    assert receipt == returned == before
    assert all(Path(path).read_bytes() == raw for path, raw in immutable.items())
    assert identity["schema"] == loader.SCHEMA
    assert identity["weights"] == "debiased_ema" and identity["purpose"] == "behavior_diagnostic"
    assert identity["ema_update_count_from_payload"] == 200
    assert identity["initial_params"]["sha256"] == report["initial_params"]["sha256"]
    assert identity["debiased_params"]["sha256"] == audit.tree_identity(params)["sha256"]
    assert identity["historical_ema_params"]["sha256"] == audit.tree_identity(payload["ema"])["sha256"]
    assert identity["checkpoint_sha256"] == request["checkpoint"]["payload"]["sha256"]
    assert identity["checkpoint_receipt_file_sha256"] == request["checkpoint"]["receipt"]["sha256"]
    assert identity["checkpoint_receipt_normalized_sha256"] == hashlib.sha256(
        json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    assert identity["reconstructed_variables_sha256"] == request["reconstruction"]["variables"]["sha256"]
    assert identity["reconstruction_audit_sha256"] == request["reconstruction"]["audit"]["sha256"]
    assert identity["static_gate_passed"] and identity["target_frozen_row_check"]["passed"]
    assert not any(identity[field] for field in ("historical_fp32_roundoff_reversed", "behavior_gate_passed",
                   "formal_ema_evaluation_eligible", "training_eligible"))
    assert not np.shares_memory(params["another"]["kernel"], initial["another"]["kernel"])


@pytest.mark.parametrize("field,value", [("weights", "ema"), ("weights", "iterate"),
    ("purpose", "train"), ("purpose", "formal_evaluation"), ("purpose", None)])
def test_request_must_be_explicit_behavior_only(field, value):
    request = dict(weights="debiased_ema", purpose="behavior_diagnostic", checkpoint={}, reconstruction={})
    request[field] = value
    with pytest.raises(ValueError, match="explicit"):
        loader.load_debiased_parameters(**request)


@pytest.mark.parametrize("mutation", ["failed", "source", "source_drift", "seed", "job", "rows", "partial",
    "checks", "one_bit", "jax", "jax_bool", "count_float", "numpy_bool", "leaf_digest", "row_digest",
    "model", "count", "counter_receipt", "decay"])
def test_no_summary_flag_or_partial_tree_bypasses_gate(bundle, mutation):
    initial, payload, receipt, report, publish = bundle
    if mutation == "failed":
        report["static_gate_passed"] = False
    elif mutation == "source":
        report["source"]["source_commit"] = "c" * 40
    elif mutation == "source_drift":
        report["source_after"]["manifest_sha256"] = "c" * 64
    elif mutation == "seed":
        report["init_seed"] += 1
    elif mutation == "job":
        report["original_job"]["sha256"] = "c" * 64
    elif mutation == "rows":
        report["predeclared_rows"]["stop"] = 63
    elif mutation == "partial":
        del initial["another"]
    elif mutation == "checks":
        report["checks"].pop()
    elif mutation == "one_bit":
        payload["ema"]["inputs"]["room_era"]["embedding"].view(np.uint32)[63, 383] ^= np.uint32(1)
    elif mutation == "jax":
        report["checks"][0]["original_jax_recurrence_bit_mismatches"]["inputs/room_era/embedding"] = 1
    elif mutation == "jax_bool":
        report["checks"][0]["original_jax_recurrence_bit_mismatches"]["inputs/room_era/embedding"] = False
    elif mutation == "count_float":
        report["checks"][0]["iteration_count_from_payload"] = 2.0
    elif mutation == "numpy_bool":
        report["checks"][0]["rows"]["inputs/room_era/embedding"]["replay_vs_ema_bit_mismatches"] = False
    elif mutation == "leaf_digest":
        initial["another"]["kernel"].view(np.uint32)[0, 0] ^= np.uint32(1)
    elif mutation == "row_digest":
        report["checks"][0]["rows"]["inputs/room_era/embedding"]["historical_ema"]["sha256"] = "f" * 64
    elif mutation == "model":
        receipt["model"] = {"a": 2}
    elif mutation == "count":
        payload["counters"]["learner_update"] = 201
        receipt["counters"]["learner_update"] = 201
    elif mutation == "counter_receipt":
        receipt["counters"]["learner_update"] = 201
    elif mutation == "decay":
        receipt["config"]["ataraxos"]["ema_decay"] = .998
    with pytest.raises(ValueError):
        loader.load_debiased_parameters(**publish())


@pytest.mark.parametrize("reference", ["payload", "receipt", "variables", "audit"])
def test_every_file_sha_is_verified(bundle, reference):
    request = bundle[-1]()
    key = "checkpoint" if reference in ("payload", "receipt") else "reconstruction"
    request[key][reference]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        loader.load_debiased_parameters(**request)


@pytest.mark.parametrize("mutation", ["payload_sha", "payload_bytes", "variables_sha"])
def test_cross_artifact_bindings_are_verified_even_with_new_file_hashes(bundle, tmp_path, mutation):
    request = bundle[-1]()
    if mutation == "variables_sha":
        report = json.loads(Path(request["reconstruction"]["audit"]["path"]).read_bytes())
        report["reconstructed_variables"]["sha256"] = "1" * 64
        request["reconstruction"]["audit"] = audit.publish(tmp_path, "changed-audit", ".json",
                                                           audit.canonical(report))
    else:
        receipt = copy.deepcopy(bundle[2])
        receipt["payload_sha256" if mutation == "payload_sha" else "payload_bytes"] = (
            "1" * 64 if mutation == "payload_sha" else receipt["payload_bytes"] - 1)
        request["checkpoint"]["receipt"] = audit.publish(tmp_path, "changed-receipt", ".json",
                                                         checkpoint_store.canonical(receipt))
    with pytest.raises(ValueError):
        loader.load_debiased_parameters(**request)


def test_no_relative_symlink_extra_or_noncanonical_inputs(bundle, tmp_path):
    request = bundle[-1]()
    with pytest.raises(ValueError, match="absolute"):
        loader._ref({"path": "relative.json", "sha256": "0" * 64})
    link = tmp_path / "receipt-link"
    link.symlink_to(request["checkpoint"]["receipt"]["path"])
    linked = copy.deepcopy(request)
    linked["checkpoint"]["receipt"]["path"] = str(link)
    with pytest.raises(ValueError, match="non-symlink"):
        loader.load_debiased_parameters(**linked)
    extra = copy.deepcopy(request)
    extra["checkpoint"]["manual_count"] = 200
    with pytest.raises(ValueError, match="exactly"):
        loader.load_debiased_parameters(**extra)
    raw = json.dumps(bundle[2], indent=2).encode()
    request["checkpoint"]["receipt"] = audit.publish(tmp_path, "pretty", ".json", raw)
    with pytest.raises(ValueError, match="canonical"):
        loader.load_debiased_parameters(**request)
