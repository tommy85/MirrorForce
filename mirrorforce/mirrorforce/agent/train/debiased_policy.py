"""Opt-in behavior-only policy adapter; the ordinary policy loader is unchanged.

The original iterate is loaded through all normal checkpoint, native and
constant-table checks. Only its parameters are replaced, after matching its
receipt and complete parameter tree to the independently checked debiasing
request. A different request is a different service identity.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from mirrorforce.agent.train import debiased_ema as D
from mirrorforce.agent.train import ema_reconstruction as audit

SESSION_LAW = "explicit-behavior-purpose-and-debiased-identity/v1"


def read_request(checkpoint, request_ref):
    """Check the small explicit request before importing any model/native code."""
    request = json.loads(D._ref(request_ref))
    if (not isinstance(request, dict)
            or set(request) != {"weights", "purpose", "checkpoint", "reconstruction"}
            or request["weights"] != D.WEIGHTS or request["purpose"] != D.PURPOSE):
        raise ValueError("a debiased service needs an explicit behavior_diagnostic loader request")
    for group, keys in (("checkpoint", {"payload", "receipt"}), ("reconstruction", {"variables", "audit"})):
        if not isinstance(request[group], dict) or set(request[group]) != keys:
            raise ValueError("debiased request artifact groups have unexpected fields")
        for ref in request[group].values():
            if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                    or not isinstance(ref["path"], str) or not Path(ref["path"]).is_absolute()
                    or not isinstance(ref["sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])):
                raise ValueError("debiased request artifacts require absolute paths and lowercase SHA-256")
    payload = request.get("checkpoint", {}).get("payload", {})
    filename = payload.get("path")
    if (not isinstance(filename, str) or not Path(filename).is_absolute()
            or Path(filename).resolve() != Path(checkpoint).resolve()):
        raise ValueError("the debiased request names a different checkpoint path")
    return request


def load_policy(checkpoint, request_ref, *, semantic_file, code_list_file,
                card_tables_file, native=None, compute_dtype=None):
    """Return agent, variables, unchanged receipt and the debiased identity."""
    request = read_request(checkpoint, request_ref)
    params, expected_receipt, identity = D.load_debiased_parameters(**request)
    from mirrorforce.agent.train import policy_io
    agent, variables, receipt = policy_io.load_policy(
        checkpoint, "iterate", semantic_file=semantic_file, code_list_file=code_list_file,
        card_tables_file=card_tables_file, native=native, compute_dtype=compute_dtype)
    if receipt != expected_receipt:
        raise ValueError("normal policy checks and debiasing checks loaded different receipts")
    original = audit.tree_identity(variables["params"])
    if {key: original[key] for key in ("sha256", "scalar_count")} != identity["iterate_params"]:
        raise ValueError("normal policy checks and debiasing checks loaded different iterate parameters")
    variables = {**variables, "params": params}
    return agent, variables, receipt, identity


def service_identity(identity, request_ref):
    """Additional fields ONLY for this opt-in mode, including session admission."""
    return {"purpose": D.PURPOSE, "debiased_ema": identity,
            "debiased_ema_request_sha256": request_ref["sha256"],
            "session_admission": {"law": SESSION_LAW, "purpose": D.PURPOSE,
                                  "debiased_ema_identity_sha256": audit.digest(audit.canonical(identity))}}


def check_session(identity, request):
    """Existing formal clients cannot accidentally open this diagnostic service."""
    if identity.get("weights") != D.WEIGHTS:
        return
    derived = identity.get("debiased_ema", {})
    expected = {"law": SESSION_LAW, "purpose": D.PURPOSE,
                "debiased_ema_identity_sha256": audit.digest(audit.canonical(derived))}
    if (identity.get("purpose") != D.PURPOSE or derived.get("schema") != D.SCHEMA
            or derived.get("purpose") != D.PURPOSE
            or derived.get("weights") != D.WEIGHTS or derived.get("static_gate_passed") is not True
            or any(derived.get(key) is not False for key in
                   ("behavior_gate_passed", "formal_ema_evaluation_eligible", "training_eligible"))
            or identity.get("session_admission") != expected
            or request.get("purpose") != D.PURPOSE
            or request.get("debiased_ema_identity_sha256") != expected["debiased_ema_identity_sha256"]):
        raise ValueError("debiased EMA sessions require explicit behavior_diagnostic purpose and derived identity")
