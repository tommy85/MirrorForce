"""Explicit A0 debiased-EMA parameters for a behavior diagnostic, not deployment.

The caller supplies immutable checkpoint and successful reconstruction-audit
references. No initializer is called, constants are not replaced, and neither
the original checkpoint nor its receipt is rewritten. This module does not
alter ``policy_io`` or any service's default weight selection.

We remove the conventional initial contribution ``decay**t * theta0`` using
binary64 arithmetic and round each result to float32 once. This does NOT undo
the historical FP32 EMA recurrence's accumulated rounding error.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import numpy as np

from mirrorforce.agent.train import checkpoint_store
from mirrorforce.agent.train import ema_reconstruction as audit

SCHEMA = "mirrorforce_debiased_ema_identity/v1"
LAW = "a0-ema-minus-initial-contribution-behavior-only/v1"
ARITHMETIC_LAW = "binary64-pow-multiply-subtract-divide-then-float32/v1"
WEIGHTS = "debiased_ema"
PURPOSE = "behavior_diagnostic"


def _ref(ref):
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or not isinstance(ref["path"], str) or not Path(ref["path"]).is_absolute()
            or not isinstance(ref["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])):
        raise ValueError("artifact reference requires an absolute path and lowercase SHA-256")
    return audit.read_ref(ref)


def _pair(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(name + " requires exactly " + ", ".join(keys))


def _static_proof(report, initial, variables_sha):
    expected_rows = {"paths": [list(p) for p in audit.FROZEN_PATHS], "start": 1,
                     "stop": 64, "total_scalars": 48384}
    source = report.get("source", {})
    if (report.get("schema") != audit.SCHEMA or report.get("static_gate_passed") is not True
            or source.get("source_commit") != audit.SOURCE or source.get("verified") is not True
            or source != report.get("source_after")
            or report.get("original_job", {}).get("sha256") != audit.ORIGINAL_JOB
            or report.get("original_seed") != audit.ORIGINAL_SEED
            or report.get("init_seed") != audit.INIT_SEED
            or report.get("frozen_law") != audit.FROZEN_LAW
            or report.get("recurrence_law") != audit.RECURRENCE_LAW
            or report.get("predeclared_rows") != expected_rows):
        raise ValueError("the original-source full-row static reconstruction gate is required")
    if report.get("reconstructed_variables", {}).get("sha256") != variables_sha:
        raise ValueError("reconstruction audit names different initial variables")
    if report.get("initial_params") != audit.tree_identity(initial):
        raise ValueError("the complete reconstructed initial parameter tree differs from the audit")
    checks = report.get("checks")
    if (not isinstance(checks, list) or len(checks) != 2
            or any(not isinstance(c, dict) or type(c.get("iteration_count_from_payload")) is not int
                   for c in checks)
            or [c.get("iteration_count_from_payload") for c in checks] != [2, 140]):
        raise ValueError("both registered reconstruction checks at iterations 2 and 140 are required")
    initial_rows = audit.frozen_rows(initial)
    for check in checks:
        rows = check.get("rows", {})
        jax_mismatches = check.get("original_jax_recurrence_bit_mismatches", {})
        if (check.get("passed") is not True or check.get("decay") != audit.DECAY
                or check.get("frozen_scalar_count") != 48384
                or check.get("full_tree_layout_checked") is not True
                or set(rows) != set(initial_rows)
                or not isinstance(jax_mismatches, dict) or set(jax_mismatches) != set(initial_rows)
                or any(type(v) is not int or v != 0 for v in jax_mismatches.values())):
            raise ValueError("every frozen scalar must pass both NumPy and original JAX recurrence checks")
        for path, initial_row in initial_rows.items():
            row = rows[path]
            expected_ema = audit.array_identity(audit.fp32_recurrence(
                initial_row, check["iteration_count_from_payload"]))
            if (row.get("row_start") != 1 or row.get("row_stop_exclusive") != 64
                    or row.get("scalar_count") != 24192
                    or row.get("initial") != audit.array_identity(initial_row)
                    or row.get("params") != row["initial"]
                    or row.get("historical_ema") != expected_ema
                    or row.get("replayed_ema") != expected_ema
                    or any(type(row.get(key)) is not int or row[key] != 0 for key in
                           ("initial_vs_params_bit_mismatches", "replay_vs_ema_bit_mismatches"))):
                raise ValueError("reconstruction row identities or complete bitwise checks differ")


def debias_parameters(initial, historical_ema, iteration_count):
    """Pure arithmetic, not a proof/eligibility gate; inputs are never mutated.

    ``iteration_count`` is the number of original EMA updates, not optimizer
    minibatches, environment decisions, or a manually inferred epoch count.
    Production callers must use ``load_debiased_parameters`` for provenance.
    """
    if type(iteration_count) is not int or not 0 < iteration_count <= 1000000:
        raise ValueError("a positive original EMA iteration count is required")
    audit.same_tree_layout(initial, historical_ema)
    power = math.pow(audit.DECAY, iteration_count)
    mass = 1.0 - power
    if not math.isfinite(mass) or not 0 < mass <= 1:
        raise ValueError("invalid debiasing mass")

    def transform(start, ema):
        if isinstance(start, dict):
            return {key: transform(start[key], ema[key]) for key in sorted(start)}
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                contribution = np.multiply(np.float64(power), np.asarray(start, np.float64))
                numerator = np.subtract(np.asarray(ema, np.float64), contribution)
                result = np.divide(numerator, np.float64(mass)).astype(np.float32)
            except FloatingPointError as exc:
                raise ValueError("nonfinite or overflowing debiased parameter") from exc
        if not np.isfinite(result).all():
            raise ValueError("nonfinite debiased parameter")
        return result

    return transform(initial, historical_ema), power, mass


def load_debiased_parameters(*, weights, purpose, checkpoint, reconstruction):
    """Return ``(params, unchanged_receipt, explicit_identity)`` on the CPU.

    ``checkpoint`` has SHA references ``payload`` and ``receipt``;
    ``reconstruction`` has ``variables`` and ``audit``. References may be
    relocated but their bytes must match. Frozen inference constants must still
    be loaded through the checkpoint's normal semantic/card-table checks.
    Only an explicitly requested behavior diagnostic is admitted here.
    """
    if weights != WEIGHTS or purpose != PURPOSE:
        raise ValueError("explicit debiased_ema weights and behavior_diagnostic purpose are required")
    _pair(checkpoint, ("payload", "receipt"), "checkpoint")
    _pair(reconstruction, ("variables", "audit"), "reconstruction")
    report = json.loads(_ref(reconstruction["audit"]))
    receipt_raw = _ref(checkpoint["receipt"])
    receipt = json.loads(receipt_raw)
    if receipt_raw != checkpoint_store.canonical(receipt):
        raise ValueError("the original checkpoint receipt must be canonical and unchanged")
    payload_raw = _ref(checkpoint["payload"])
    if (receipt.get("payload_sha256") != checkpoint["payload"]["sha256"]
            or type(receipt.get("payload_bytes")) is not int
            or receipt["payload_bytes"] != len(payload_raw)):
        raise ValueError("checkpoint payload and original receipt binding differ")

    # Serialization needs Flax but no model/native imports or GPU computation.
    from flax.serialization import msgpack_restore
    variables = msgpack_restore(_ref(reconstruction["variables"]))
    if not isinstance(variables, dict) or "params" not in variables:
        raise ValueError("complete reconstructed variables must contain params")
    initial = variables["params"]
    _static_proof(report, initial, reconstruction["variables"]["sha256"])
    payload = msgpack_restore(payload_raw)
    current_check = audit.certify(initial, payload, receipt, report.get("model"))
    if not current_check["passed"]:
        raise ValueError("target EMA count/reset or frozen-row recurrence differs")
    count = current_check["iteration_count_from_payload"]
    params, power, mass = debias_parameters(initial, payload["ema"], count)

    def compact(tree):
        identity = audit.tree_identity(tree)
        return {key: identity[key] for key in ("sha256", "scalar_count")}

    identity = {
        "schema": SCHEMA, "weights": WEIGHTS, "purpose": PURPOSE,
        "law": LAW, "arithmetic_law": ARITHMETIC_LAW,
        "checkpoint_sha256": checkpoint["payload"]["sha256"],
        "checkpoint_receipt_file_sha256": checkpoint["receipt"]["sha256"],
        "checkpoint_receipt_normalized_sha256": audit.digest(json.dumps(receipt, sort_keys=True).encode()),
        "reconstructed_variables_sha256": reconstruction["variables"]["sha256"],
        "reconstruction_audit_sha256": reconstruction["audit"]["sha256"],
        "model_sha256": audit.digest(audit.canonical(receipt["model"])),
        "initial_params": compact(initial), "iterate_params": compact(payload["state"]["params"]),
        "historical_ema_params": compact(payload["ema"]), "debiased_params": compact(params),
        "counters": dict(payload["counters"]), "ema_update_count_from_payload": count,
        "decay": audit.DECAY, "initial_power_binary64_hex": power.hex(),
        "debias_mass_binary64_hex": mass.hex(), "target_frozen_row_check": current_check,
        "static_gate_passed": True, "historical_fp32_roundoff_reversed": False,
        "behavior_gate_passed": False, "formal_ema_evaluation_eligible": False,
        "training_eligible": False,
    }
    return params, receipt, identity
