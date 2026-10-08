"""Pure records for explicitly registered same-root bank-unknown fallback.

This module cannot issue the live TypedBankUnknown capability. Old exception
strings or an admission status named ``unknown`` do not grant that capability.
No incomplete bank, Q value, or hypothetical continuation is used as policy.
"""
from __future__ import annotations

from enum import Enum
import math

from .causal_proof import sha256

FATAL = "fatal/v1"
LAW = "same-root-raw-greedy/v1"
SCHEMA = "mirrorforce_owned_bank_unknown/v1"
ROOT_SCHEMA = "mirrorforce_bank_unknown_root/v1"
CLEANUP = {"bank_branch_closed": True, "owned_drivers_closed": True,
           "runtime_capabilities_revoked": True, "root_snapshot_verified": True,
           "registered_source_assets_verified": True}


class UnknownReason(Enum):
    PROPOSAL_BUDGET = "proposal_budget"
    HISTORY_BUDGET = "history_budget"
    RESPONSE_BUDGET = "response_budget"
    WITNESSES_EXHAUSTED = "witnesses_exhausted"
    ADMISSION_DEADLINE = "admission_deadline"


def verify_unknown_record(record, audit, *, root_snapshot_sha256, obs_sha256,
                          requested_count, proposal_seed, entropy_profile_sha256,
                          source_sha256):
    """Check a saved capability record, never turn it into a live exception."""
    from .causal_rejection import check_conditioned_admission
    from .causal_profile import verify_bundle
    if verify_bundle(audit.get("entropy_profile")) != entropy_profile_sha256:
        raise ValueError("bank fallback requires its externally registered runtime profile")
    if not isinstance(record, dict) or record.get("schema") != SCHEMA or record.get("law") != LAW \
            or record.get("reason") not in {reason.value for reason in UnknownReason} \
            or record.get("root_snapshot_sha256") != root_snapshot_sha256 \
            or record.get("admission_sha256") != sha256(audit) or record.get("cleanup") != CLEANUP \
            or record.get("hard_errors") != [] or record.get("obs_sha256") != obs_sha256 \
            or record.get("source_sha256") != source_sha256 \
            or record.get("entropy_profile_sha256") != entropy_profile_sha256:
        raise ValueError("bank fallback lacks its typed owned-root cleanup and registration record")
    for name in ("root_snapshot_sha256", "history_sha256", "world_sha256", "source_sha256"):
        value = record.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("bank fallback lacks an exact public/root/source binding")
    if record["history_sha256"] != audit.get("history_sha256") \
            or record["world_sha256"] != audit.get("world_sha256"):
        raise ValueError("bank fallback changed the unknown bank's public state")
    check_conditioned_admission(audit, requested_count=requested_count, proposal_seed=proposal_seed,
        world_sha256=record["world_sha256"], obs_sha256=obs_sha256,
        history_sha256=record["history_sha256"], source_sha256=source_sha256,
        entropy_profile_sha256=entropy_profile_sha256, require_complete=False)
    if any(row.get("status") in ("error", "running") for row in audit.get("proposals", [])):
        raise ValueError("a hard or unfinished candidate cannot become a bank fallback")
    if audit.get("hard_error", False) is not False or "postcheck_failure" in audit:
        raise ValueError("a proposal integrity failure cannot become a bank fallback")
    if record["reason"] == UnknownReason.PROPOSAL_BUDGET.value:
        if record.get("phase") != "proposal" or audit.get("status") != "proposal_failed" \
                or audit.get("failure_kind") != record["reason"]:
            raise ValueError("only typed counting budget exhaustion may fall back during proposal construction")
    elif record.get("phase") != "admission" or audit.get("status") != "unknown" \
            or audit.get("failure_kind") != record["reason"] \
            or len(audit.get("accepted_ordinals", [])) >= requested_count:
        raise ValueError("bank fallback is not an explicit incomplete feasibility search")
    return record["reason"]


def decision_summary(roots):
    """Keep both denominators, including failed roots and uncompleted attempts."""
    prompts = len(roots)
    attempts = [attempt for root in roots for attempt in root.get("non_singleton_attempts", [])]
    if any(not isinstance(attempt, dict) or type(attempt.get("rows")) is not int or attempt["rows"] < 2
           or type(attempt.get("subdecision")) is not int or attempt["subdecision"] < 0
           or not isinstance(attempt.get("obs_sha256"), str) or len(attempt["obs_sha256"]) != 64 for attempt in attempts):
        raise ValueError("fallback accounting requires every actual non-singleton attempt")
    searches = [search for root in roots for search in root.get("searches", []) if not search.get("singleton")]
    if len(searches) > len(attempts):
        raise ValueError("fallback accounting omitted attempted decisions")
    unknown = [search for search in searches if search.get("fallback") == "bank_unknown"]
    unknown_prompts = sum(any(search.get("fallback") == "bank_unknown" for search in root.get("searches", []))
                          for root in roots)
    for search in unknown:
        if search.get("search_changed") is not False or search.get("search_gain") is not None \
                or search.get("counterfactual_q") is not None:
            raise ValueError("bank-unknown fallback cannot claim a search change, gain, or counterfactual Q")
    return {"law": LAW, "scope": "all reported original prompts and non-singleton decisions, including failed games",
        "non_singleton_attempts": len(attempts), "completed_non_singleton_decisions": len(searches), "wire_prompts": prompts,
        "bank_unknown_decisions": len(unknown),
        "bank_unknown_decision_fraction": len(unknown) / len(attempts) if attempts else None,
        "bank_unknown_prompts": unknown_prompts,
        "bank_unknown_prompt_fraction": unknown_prompts / prompts if prompts else None,
        "failed_prompts": sum(root.get("complete") is not True for root in roots),
        "fallback_is_search_gain": False}


def check_root_record(record, decision, identity, *, root_snapshot_sha256, candidates, count):
    """Independent check of the no-Q alternative; not a relaxed search record."""
    settings = identity["settings"]
    particles = identity["particles"]
    import hashlib
    import json
    if settings.get("bank_unknown_law") != LAW or settings.get("selection") != "greedy" \
            or settings.get("budget_law") != "completed-balanced-stripes/v1" \
            or particles.get("provider") != "own-pending-natural-conditioned-capacity/v1" \
            or record.get("schema") != ROOT_SCHEMA or record.get("fallback") != "bank_unknown" \
            or record.get("fallback_law") != LAW or record.get("training_eligible") is not False \
            or record.get("settings") != settings \
            or record.get("settings_sha256") != hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest() \
            or record.get("complete") is not True or record.get("bank_complete") is not False \
            or record.get("search_changed") is not False or record.get("search_gain") is not None \
            or record.get("counterfactual_q") is not None or record.get("particle_proofs") != [] \
            or record.get("particle_weights") != [] or "anytime" in record or "causal_bank" in record \
            or record.get("candidate_rows") != list(candidates) or record.get("lines_per_row") != count \
            or record.get("particle_law") != particles.get("law") or type(record.get("particle_seed")) is not int:
        raise ValueError("bank fallback differs from its explicit no-search contract")
    values = decision["logits"]
    rows = decision["rows"]
    if type(rows) is not int or rows < 2 or len(values) != rows \
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in values) \
            or record.get("rows") != rows or record.get("obs_sha256") != decision["obs_sha256"] \
            or record.get("world_obs_sha256") != decision["obs_sha256"] \
            or record.get("q") != [None] * rows or record.get("wide_menu") is not (rows > settings["rollouts"]):
        raise ValueError("bank fallback changed its actual root menu or invented Q")
    maximum = max(values)
    prior = [math.exp(x - maximum) for x in values]
    total = sum(prior)
    prior = [x / total for x in prior]
    chosen = values.index(maximum)
    for name in ("prior", "policy"):
        saved = record.get(name)
        if not isinstance(saved, list) or len(saved) != rows or any(type(x) not in (int, float)
                or not math.isfinite(x) or not math.isclose(x, p, rel_tol=1e-12, abs_tol=1e-12)
                for x, p in zip(saved, prior)):
            raise ValueError("bank fallback did not reuse the same pending raw policy")
    if type(decision.get("chosen")) is not int or decision["chosen"] != chosen or record.get("chosen") != chosen \
            or not math.isclose(decision["p_chosen"], prior[chosen], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("bank fallback changed raw-greedy selection")
    audit = record.get("conditional_admission", {})
    proof = record.get("fallback_record", {})
    if record.get("world_sha256") != audit.get("world_sha256"):
        raise ValueError("bank fallback changed the sampled World")
    verify_unknown_record(proof, audit, root_snapshot_sha256=root_snapshot_sha256,
        obs_sha256=decision["obs_sha256"], requested_count=count, proposal_seed=record["particle_seed"],
        entropy_profile_sha256=particles["entropy_profile_sha256"],
        source_sha256=particles["entropy_profile"]["producer_source_sha256"])
