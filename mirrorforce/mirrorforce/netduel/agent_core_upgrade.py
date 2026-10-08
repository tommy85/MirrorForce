"""Explicit search-only core upgrade evidence; original training receipts stay immutable.

This is finite default-disabled compatibility evidence, not a proof about all
possible future scripts. The new hypothetical ABI has its own native tests.
The pure checker loads no core/model and cannot fabricate native execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "mirrorforce_disabled_core_upgrade/v1"
LAW = "search-only-explicit-core-upgrade; default-disabled-finite-replay/v1"
SEEDS = (2026102201, 2026102202, 2026102203, 2026102204)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def receipt_identity_sha(value):
    """The existing CheckpointBackend receipt identity uses default JSON separators."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_core_upgrade(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("core upgrade evidence differs from its pinned file SHA")
    return {"sha256": expected_sha256, "json": raw.decode("utf-8")}


def check_core_upgrade(evidence, *, checkpoint, service, stage_sha256, scripts, assets):
    if not isinstance(evidence, dict) or set(evidence) != {"sha256", "json"} \
            or not isinstance(evidence["json"], str) \
            or hashlib.sha256(evidence["json"].encode()).hexdigest() != evidence["sha256"]:
        raise ValueError("explicit search core upgrade requires its original content-addressed evidence")
    report = json.loads(evidence["json"])
    actor_sha=service.get("checkpoint_sha256")
    specialization=None
    if service.get("backend")=="specialization":
        from .agent_specialization_search import registered_service
        specialization=registered_service(service)
        actor_sha=specialization['parent_checkpoint_sha256']
    original = checkpoint.get("native_core")
    reference, target = report.get("reference_build", {}), report.get("target_build", {})
    if report.get("schema") != SCHEMA or report.get("law") != LAW or report.get("training_eligible") is not False \
            or report.get("passed") is not True or report.get("training_core") != original \
            or report.get("checkpoint_sha256") != checkpoint.get("payload_sha256") \
            or report.get("checkpoint_sha256") != actor_sha \
            or report.get("checkpoint_receipt_canonical_sha256") != receipt_identity_sha(checkpoint) \
            or report.get("checkpoint_receipt_canonical_sha256") != service.get("receipt_sha256") \
            or report.get("stage_sha256") != stage_sha256 or report.get("scripts") != scripts \
            or report.get("assets") != assets or target != service.get("native_build"):
        raise ValueError("core upgrade evidence belongs to different weights, assets or rule identities")
    if not isinstance(original, dict) or reference.get("core_commit") != original.get("core_commit") \
            or reference.get("core_tree") != original.get("core_tree") \
            or reference.get("outputs", {}).get("libmfcore.so") != original.get("libmfcore_sha256") \
            or not target.get("core_commit") or target["core_commit"] == original.get("core_commit") \
            or reference.get("outputs", {}).get("liblua5.3.so.0") != target.get("outputs", {}).get("liblua5.3.so.0") \
            or any(build.get("schema") != "mirrorforce_duelpool_build/v1" or build.get("uncommitted_worktree_build") is not False
                   for build in (reference, target)):
        raise ValueError("core upgrade source/target is not the declared immutable core-only build")
    # The entire native C++ source tree is identical; only the separately
    # committed core changes. No observation wrapper or compiler change hides
    # behind this narrow opt-in.
    for key in ("source_tree", "shared_tree", "shared_sources", "compiler", "flags", "deps"):
        if key not in reference or reference[key] != target.get(key):
            raise ValueError("core-only upgrade also changed native sources or the build recipe: " + key)
    traces = report.get("traces", [])
    if not isinstance(traces, list) or len(traces) != len(SEEDS):
        raise ValueError("core upgrade lacks the complete predeclared native trace cohort")
    for seed, row in zip(SEEDS, traces):
        left, right = row.get("reference", {}), row.get("target", {})
        if row.get("seed") != seed or row.get("policy_seed") != seed + 1000 \
                or row.get("target_shuffle_plan_mode") != 0 or left != right \
                or left.get("finished") is not True or left.get("winner") not in (0, 1, 2) \
                or type(left.get("responses")) is not int or left["responses"] < 20 \
                or type(left.get("packets")) is not int or left["packets"] < 20 \
                or type(left.get("turns")) is not int or left["turns"] < 1 \
                or type(left.get("menu_rounds")) is not int or left["menu_rounds"] < 1 \
                or left.get("truncations") != 0 or left.get("fallbacks") != 0 \
                or left.get("script_errors") != []:
            raise ValueError("core upgrade native trace is incomplete, changed, or used an active shuffle plan")
        for key in ("responses_sha256", "packets_sha256", "menus_sha256"):
            value = left.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("core upgrade lacks exact full-stream hashes")
    return {"law": LAW, "evidence_sha256": evidence["sha256"], "training_core": original,
            "serving_core_sha256": target["outputs"]["libmfcore.so"], "default_disabled_games": len(SEEDS),
            "scope": "finite-native-default-paths; not universal-rule-equivalence", "training_receipt_changed": False,
            **({'specialization_search':specialization} if specialization is not None else {})}
