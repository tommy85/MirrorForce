"""Explicit, evidence-bound numeric admission for optional client search.

Pure validation: no core/model is loaded. The finite GPU prefix audit does not
certify every future input, a different checkpoint/native/device, or a 600 s
turn budget. Its failed modes and old batch1 comparisons are never relabeled.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import math

SCHEMA = "mirrorforce_search_numeric_contract/v1"
AUDIT_SCHEMA = "mirrorforce_inference_shape_precision_audit/v1"


class NumericContractError(ValueError):
    pass


def load_numeric_evidence(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise NumericContractError("numeric audit file differs from its registered SHA-256")
    return {"sha256": expected_sha256, "json": raw.decode("utf-8")}


def numeric_contract(identity, *, mode, evidence=None):
    if identity.get("compute_dtype") != "bfloat16":
        raise NumericContractError("only the approved BF16 inference precision is admitted")
    batch = identity.get("search_batching")
    geometry = None
    if "inference_geometry" in identity:
        from mirrorforce.agent.inference_geometry import validate_geometry
        try:
            geometry = validate_geometry(identity["inference_geometry"])
        except (TypeError, ValueError) as exc:
            raise NumericContractError("unknown or incomplete shared inference geometry") from exc
        if mode != "constant_batch_full_menu" or not isinstance(batch, dict) \
                or batch.get("size") != geometry["batch_size"] or batch.get("menu_rows") != geometry["menu_rows"]:
            raise NumericContractError("shared inference geometry requires its registered constant B64/full192 mode")
    if mode == "exact_batch1":
        if batch != {"law": "valid-row-padding/v1", "sizes": [1]} or evidence is not None:
            raise NumericContractError("exact_batch1 needs the explicit singleton search-batch mode")
        return {"schema": SCHEMA, "mode": mode, "batching": batch, "diagnostic_baseline": True,
                "full_search_budget_accepted": False}
    if mode != "constant_batch_full_menu" or not isinstance(batch, dict) \
            or set(batch) != {"law", "size", "menu_rows", "applies_to"} \
            or batch["law"] != "constant-batch-full-menu/v1" or batch["applies_to"] != "all_forwards" \
            or type(batch["size"]) is not int or batch["size"] not in (16, 64) \
            or batch["menu_rows"] != identity.get("client_config", {}).get("max_options"):
        raise NumericContractError("unknown or incomplete constant-batch/full-menu contract")
    if not isinstance(evidence, dict) or set(evidence) != {"sha256", "json"} \
            or not isinstance(evidence["json"], str):
        raise NumericContractError("constant mode requires the registered raw GPU audit evidence")
    if hashlib.sha256(evidence["json"].encode()).hexdigest() != evidence["sha256"]:
        raise NumericContractError("numeric evidence payload does not match its SHA-256")
    report = json.loads(evidence["json"])
    if report.get("schema") != AUDIT_SCHEMA or report.get("training_eligible") is not False \
            or report.get("native_sha256") != identity.get("serving_native_sha256") \
            or report.get("native_build") != identity.get("native_build") \
            or report.get("core_sha256") != identity.get("native_build", {}).get("outputs", {}).get("libmfcore.so") \
            or report.get("device") != identity.get("device"):
        raise NumericContractError("numeric evidence belongs to a different native/core/device")
    name = f"bfloat16-constant{batch['size']}"
    modes = [item for item in report.get("modes", []) if item.get("mode") == name]
    if len(modes) != 1 or modes[0].get("passed_exact") is not True:
        raise NumericContractError("the requested constant mode did not pass its exact GPU gate")
    selected = modes[0]
    keys = {"backend", "checkpoint_sha256", "receipt_sha256", "weights", "compute_dtype", "magnet",
            "training_native_sha256", "semantic_file_sha256", "card_tables_sha256", "search_batching"}
    if geometry is not None:
        keys.add("inference_geometry")
    if identity.get('backend')=='specialization':
        from .agent_specialization_search import registered_service
        try:
            registered_service(identity)
        except ValueError as exc:
            raise NumericContractError('specialization numeric binding refused: '+str(exc)) from exc
        keys|={'specialization','specialization_search','parent_checkpoint_sha256'}
    if set(selected.get("identity", {})) != keys or any(selected["identity"][k] != identity.get(k) for k in keys):
        raise NumericContractError("numeric evidence is for different weights, receipts, tables or computation")
    replays = selected.get("replays", [])
    def sha(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    if len(replays) != 2 or {r.get("seat") for r in replays} != {0, 1} or any(
            r.get("trace_equal") is not True or r.get("memory_equal") is not True
            or r.get("noninitial_memory") is not True or r.get("noninitial_memory_steps", 0) < 1
            or type(r.get("decisions")) is not int or r["decisions"] < 1
            or not sha(r.get("live_memory_sha256")) or not sha(r.get("live_trace_sha256"))
            or r.get("live_memory_sha256") != r.get("replay_memory_sha256")
            or r.get("live_trace_sha256") != r.get("replay_trace_sha256") for r in replays):
        raise NumericContractError("numeric evidence lacks both seats' exact noninitial replay memories")
    if selected.get("samples") != sum(r["decisions"] for r in replays):
        raise NumericContractError("numeric sample count differs from its actual replay decisions")
    tests = selected.get("invariance", [])
    size = batch["size"]
    expected = Counter({n: 3 for n in {1, 3, size - 1, size, size + 1, 2 * size + 1}})
    if Counter(row.get("requested_rows") for row in tests) != expected or any(
            row.get("exact") is not True or row.get("finite") is not True
            or row.get("valid_inputs_unchanged") is not True or row.get("max_abs") != 0
            or row.get("close_1e5") is not True or row.get("max_abs_memory") != 0 or row.get("max_abs_logits") != 0
            or row.get("max_policy_kl") != 0 or row.get("max_policy_tv") != 0 or row.get("greedy_changes") != 0
            or len(row.get("seconds", [])) != 3 or row.get("comparisons") != 3 * row["requested_rows"]
            or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in row["seconds"])
            for row in tests):
        raise NumericContractError("numeric evidence lacks exact position/companion/chunk invariance")
    if "replay_belief" in identity:
        gate = report.get("replay_belief_gate")
        extra_identity = keys | {"replay_belief"}
        expected_belief = {"law": "same-public-forward-session-pending-count-head/v1", "locations": [1, 2, 3, 4, 6, 7],
                           "counts": 4, "source": "same-public-forward"}
        if identity["replay_belief"] != expected_belief or not isinstance(gate, dict) \
                or gate.get("schema") != "mirrorforce_replay_belief_numeric_gate/v1" \
                or gate.get("passed") is not True or set(gate.get("identity", {})) != extra_identity \
                or any(gate["identity"][key] != identity.get(key) for key in extra_identity) \
                or gate.get("actual_gpu") != identity.get("device") \
                or gate.get("actual_gpu", {}).get("platform") != "gpu" \
                or gate.get("actual_gpu", {}).get("count") != 1 \
                or gate.get("jax_default_matmul_precision", "missing") is not None \
                or gate.get("parameter_objects_shared") is not True \
                or not sha(gate.get("parameters_before_sha256")) \
                or gate.get("parameters_before_sha256") != gate.get("parameters_after_sha256") \
                or gate.get("source_inputs") != "own-public-packets-and-known-responses/v1" \
                or gate.get("ordinary_outputs_exact") is not True \
                or gate.get("count_heads_same_forward_exact") is not True \
                or type(gate.get("initial_samples")) is not int or gate["initial_samples"] < 2 \
                or type(gate.get("noninitial_samples")) is not int or gate["noninitial_samples"] < 2 \
                or gate["initial_samples"] + gate["noninitial_samples"] > selected["samples"] \
                or gate.get("count_head_invariance_rows") != 3 * 3 * sum((1, 3, 63, 64, 65, 129)):
            raise NumericContractError("replay belief requires its own exact GPU head/cache/parameter evidence")
        cache = gate.get("owned_cache")
        if not isinstance(cache, dict) or cache.get("other_session_did_not_pollute") is not True \
                or cache.get("additional_cache_forwards") != 0:
            raise NumericContractError("replay belief cache did not preserve forward/session ownership")
        rows = cache.get("rows", [])
        if len(rows) != 2 or {row.get("seat") for row in rows} != {0, 1} \
                or {row.get("noninitial") for row in rows} != {False, True} \
                or any(row.get("cache_read_only") is not True or row.get("stale_rejected") is not True
                       or row.get("foreign_rejected") is not True
                       or not all(sha(row.get(k)) for k in ("preceding_memory_sha256", "pending_obs_sha256", "head_sha256"))
                       for row in rows):
            raise NumericContractError("replay belief cache lacks both real initial/noninitial owned pending roots")
    return {"schema": SCHEMA, "mode": mode, "batching": batch, "report_sha256": evidence["sha256"],
            **({"inference_geometry": geometry} if geometry is not None else {}),
            "audit_mode": name, "passed_exact": True, "samples": sum(r["decisions"] for r in replays),
            "scope": "registered finite public-client prefixes and same-shape invariance only",
            "full_search_budget_accepted": False, "equal_to_old_batch1": False}
