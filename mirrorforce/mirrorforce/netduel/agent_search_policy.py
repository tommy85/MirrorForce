"""Option-A root planning against a registered client-policy session.

Particle proposals are supplied by a public-only provider before this call.
Every selected legal row receives the SAME complete bank once, with unchanged
weights. Replay/line failures reject the root; they never select a different
action or silently condition a particle bank on a future action's outcome.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import ctypes
import time

import numpy as np

from .opponent_stream import ACTIVATION_LEGALITY, synthetic_roots
from .agent_rollout import (RestoredRoot, RootLines, q_values, BOUNDED_LANES_LAW, REPLAY_FILTERED_LANES)
from .agent_policy import RemotePolicy
from . import agent_wire as W
from . import agent_anytime as A
from . import agent_stripe_batching as MULTI
from . import bank_fallback as F
from . import current_root_protocol as CURRENT
from . import agent_search_gate as G
from . import agent_final_room_budget as FB
from . import agent_opponent_clear as OC
from . import agent_opponent_known as OK

INFORMATION_SET_SEARCH = True
SCHEMA = "mirrorforce_option_a_root/v1"
FOLLOWER_STATS_SCHEMA = "mirrorforce_follower_stats/v1"
REPLAY_FILTERED_PROVIDER = "own-pending-replay-filtered/v1"
ALL_CANDIDATES = "all-legal-common-bank/v1"
LEGACY_CANDIDATES = "budget-top-prior/v1"
REPLAY_FILTERED_RESOURCES = {"law": BOUNDED_LANES_LAW, "max_live_lines": REPLAY_FILTERED_LANES,
    "base_sessions": "one-per-active-stripe", "partial_candidate_returns": "discard-entire-stripe"}
CURRENT_PROVIDERS = (CURRENT.PROVIDER, CURRENT.AR_PROVIDER, CURRENT.COUNT_PROVIDER)


def is_current_provider(provider):
    return provider in CURRENT_PROVIDERS
ANYTIME_LAWS = (A.LAW, MULTI.LAW)


def current_identity_schema(config):
    if config.opponent_known_profile is not None:
        return "mirrorforce_current_root_identity/v7-opponent-known"
    if config.opponent_clear_profile is not None:
        return "mirrorforce_current_root_identity/opponent-clear"
    if config.final_room_profile is not None:
        return "mirrorforce_current_root_identity/final-room-turn450"
    if config.on_demand is not None:
        return "mirrorforce_current_root_identity/v3"
    return "mirrorforce_current_root_identity/v2" if config.budget_law == MULTI.LAW \
        else "mirrorforce_current_root_identity/v1"


def current_root_schema(config):
    if config.opponent_known_profile is not None:
        return "mirrorforce_current_root_search/v9-opponent-known"
    if config.opponent_clear_profile is not None:
        return "mirrorforce_current_root_search/v8-opponent-clear"
    if config.final_room_profile is not None:
        return "mirrorforce_current_root_search/v7-final-room-turn450"
    if config.on_demand is not None:
        return "mirrorforce_current_root_search/v5"
    return "mirrorforce_current_root_search/v4" if config.budget_law == MULTI.LAW else CURRENT.ROOT_SCHEMA


def current_client_schema(config):
    if config.opponent_known_profile is not None:
        return "mirrorforce_current_root_client/v7-opponent-known"
    if config.opponent_clear_profile is not None:
        return "mirrorforce_current_root_client/opponent-clear"
    if config.final_room_profile is not None:
        return "mirrorforce_current_root_client/final-room-turn450"
    if config.on_demand is not None:
        return "mirrorforce_current_root_client/v3"
    return "mirrorforce_current_root_client/v2" if config.budget_law == MULTI.LAW \
        else "mirrorforce_current_root_client/v1"


def current_resources(config):
    return dict(MULTI.RESOURCES if config.budget_law == MULTI.LAW else REPLAY_FILTERED_RESOURCES)


def resource_record(config):
    limits = {"law": MULTI.RESOURCE_LAW, **MULTI.DEFAULT_LIMITS} if config.budget_law == MULTI.LAW else {
        "law": BOUNDED_LANES_LAW, "max_live_lines": REPLAY_FILTERED_LANES}
    return {**limits, "peak_line_sessions": 0, "peak_base_sessions": 0, "chunks": 0}


class SearchPolicyError(RuntimeError):
    pass


def follower_stats_wire(stats):
    """Lossless JSON for SyncStats, including non-string Counter keys.

    Do not use dataclasses.asdict here: it reconstructs a Counter from an
    iterable of (key, count) pairs, turning those pairs into keys with count 1.
    Typed entries also keep e.g. 11, "11", and nested tuples distinct. This is
    diagnostic data only; it does not change the follower's counters or rules.
    """
    def key_wire(key):
        if type(key) in (int, str):
            return {"type": type(key).__name__, "value": key}
        if type(key) is tuple:
            return {"type": "tuple", "items": [key_wire(item) for item in key]}
        raise TypeError("unsupported follower Counter key type: " + type(key).__name__)

    result = {"schema": FOLLOWER_STATS_SCHEMA}
    for field in fields(stats):
        value = getattr(stats, field.name)
        if isinstance(value, Counter):
            entries = []
            for key, count in value.items():
                if type(count) is not int:
                    raise TypeError("follower Counter counts must be integers")
                entries.append({"key": key_wire(key), "count": count})
            result[field.name] = sorted(entries, key=lambda row: json.dumps(row["key"], sort_keys=True))
        elif type(value) is int:
            result[field.name] = value
        else:
            raise TypeError("unsupported follower stats field: " + field.name)
    return result


@dataclass(frozen=True)
class SearchConfig:
    rollouts: int = 50
    depth: int = 5
    td_lambda: float = 1.0
    alpha: float = 0.002
    beta: float = 0.02
    seconds: float = 60.0
    selection: str = "sample"
    estimator: str = "rollout"
    budget_law: str = "complete-bank/v1"
    clock_reserve: float = 30.0
    clock_share: float = 0.1
    response_margin: float = 3.0
    finalize_seconds: float = 3.0
    rpc_seconds: float = 20.0
    bank_unknown_law: str = F.FATAL
    total_seconds: float | None = None
    candidate_law: str = LEGACY_CANDIDATES
    particles: int | None = None
    on_demand: G.GateConfig | None = None
    follower_recipe_law: str | None = None
    final_room_profile: dict | None = None
    opponent_clear_profile: dict | None = None
    opponent_known_profile: dict | None = None
    untimed_profile: dict | None = None

    def __post_init__(self):
        gate = self.on_demand
        if isinstance(gate, dict):
            gate = G.GateConfig(**gate)
        if gate is not None and type(gate) is not G.GateConfig:
            raise ValueError("on-demand search needs its explicit gate configuration")
        if gate is not None and not gate.enabled:
            gate = None
        object.__setattr__(self, "on_demand", gate)
        if self.final_room_profile is not None:
            object.__setattr__(self, 'final_room_profile', FB.validate_profile(self.final_room_profile))
        if self.opponent_clear_profile is not None:
            object.__setattr__(self, 'opponent_clear_profile', OC.validate_profile(self.opponent_clear_profile))
        if self.opponent_known_profile is not None:
            object.__setattr__(self, 'opponent_known_profile', OK.validate_profile(self.opponent_known_profile))
            if self.opponent_clear_profile is not None:
                raise ValueError('opponent-known and historical opponent-clear are distinct identities')
        if self.untimed_profile is not None:
            from . import untimed_search as U
            if self.untimed_profile != U.profile() or any(value is not None for value in (
                    self.final_room_profile, self.opponent_clear_profile, self.opponent_known_profile)):
                raise ValueError('the untimed evaluation profile is exact and excludes every room-clock profile')
        if gate is not None and gate.turn_seconds is None and self.final_room_profile is None:
            raise ValueError('removing a turn-work cap requires the explicit final-room clock profile')
        if self.follower_recipe_law is not None:
            from ..common.client_public_recipe import LAW
            if self.follower_recipe_law != LAW or gate is None:
                raise ValueError("public follower recipe constraints require their explicit on-demand law")
        if type(self.rollouts) is not int or self.rollouts < 2 or type(self.depth) is not int or self.depth < 1 \
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in (self.td_lambda, self.alpha, self.beta, self.seconds)) \
                or not 0 <= self.td_lambda <= 1 or min(self.alpha, self.beta, self.seconds) <= 0 \
                or self.selection not in ("sample", "greedy") or self.estimator not in ("rollout", "critic"):
            raise ValueError("invalid explicit search budgets, update settings or root selection")
        if self.budget_law not in ("complete-bank/v1", *ANYTIME_LAWS) \
                or self.budget_law in ANYTIME_LAWS and self.selection != "greedy" \
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in
                       (self.clock_reserve, self.clock_share, self.response_margin, self.finalize_seconds, self.rpc_seconds)) \
                or not 0 < self.response_margin < self.clock_reserve < 600 or self.clock_share > 1:
            raise ValueError("invalid explicit anytime law, greedy fallback or public-clock margins")
        if self.bank_unknown_law not in (F.FATAL, F.LAW) or self.bank_unknown_law == F.LAW \
                and (self.budget_law != A.LAW or self.selection != "greedy"):
            raise ValueError("bank-unknown fallback needs its explicit same-root anytime raw-greedy law")
        if self.total_seconds is not None and (self.budget_law not in ANYTIME_LAWS or self.selection != "greedy" \
                or type(self.total_seconds) not in (int, float) or not math.isfinite(self.total_seconds) \
                or not self.finalize_seconds < self.total_seconds <= self._prompt_cap()):
            raise ValueError("an explicit whole-prompt cap requires anytime greedy search and <=40 seconds")
        if self.candidate_law not in (LEGACY_CANDIDATES, ALL_CANDIDATES) or self.candidate_law == ALL_CANDIDATES \
                and (self.budget_law not in ANYTIME_LAWS or self.selection != "greedy"):
            raise ValueError("all legal candidates require the explicit balanced common-bank law")
        if self.particles is not None and (type(self.particles) is not int or not 1 <= self.particles <= 128):
            raise ValueError("an explicit particle target cap must be an integer in 1..128")
        if self.particles is not None and self.candidate_law != ALL_CANDIDATES:
            raise ValueError("particle target caps belong to the explicit all-legal common-bank law")
        if self.budget_law == MULTI.LAW and (self.candidate_law != ALL_CANDIDATES or self.total_seconds is None
                or self.particles is None or self.bank_unknown_law != F.FATAL or self.estimator != "rollout"
                or self.td_lambda != 1.):
            raise ValueError("multiplexed search requires its explicit all-legal finite current-root rollout settings")
        if gate is not None and (self.budget_law != MULTI.LAW or self.total_seconds != gate.lethal_seconds
                or self.seconds != gate.lethal_seconds - gate.finalize_seconds
                or self.finalize_seconds != gate.finalize_seconds or gate.uncertain_seconds <= self.finalize_seconds):
            raise ValueError("on-demand caps must bind the unchanged public finalization reserve and explicit prompt limits")
        if self.opponent_clear_profile is not None and search_settings(self) != OC.settings('on'):
            raise ValueError('opponent-clear search must retain its exact600/10+3/150/turn450 profile')
        if self.opponent_known_profile is not None and search_settings(self) != OK.settings('on'):
            raise ValueError('opponent-known search must retain its exact600/10+3/150/turn450 profile')
        if self.opponent_known_profile is None and self.opponent_clear_profile is None and self.final_room_profile is not None \
                and search_settings(self) != FB.settings('on'):
            raise ValueError('final-room search must retain its exact600/10+3/150 profile and public triggers')
        if self.untimed_profile is not None:
            from . import untimed_search as U
            if search_settings(self) != U.settings():
                raise ValueError('untimed evaluation keeps every released search setting except its clocks')

    def _prompt_cap(self):
        if self.untimed_profile is None:
            return 40
        from . import untimed_search as U
        return U.PROMPT_SECONDS


def check_follower_recipe(config, particle_identity):
    """The public mirror-recipe follower law goes with the seeded uniform current-root proposals, whether used as they
    are or resampled under the count-belief head."""
    if config.follower_recipe_law is not None \
            and particle_identity.get("provider") not in (CURRENT.PROVIDER, CURRENT.COUNT_PROVIDER):
        raise SearchPolicyError("public follower recipe constraints require the current-root uniform proposals")


def search_settings(config):
    """Keep existing default identities/artifacts unchanged; caps are opt-in."""
    result = asdict(config)
    if result["total_seconds"] is None:
        del result["total_seconds"]
    if result["candidate_law"] == LEGACY_CANDIDATES:
        del result["candidate_law"]
    if result["particles"] is None:
        del result["particles"]
    if result["on_demand"] is None:
        del result["on_demand"]
    if result["follower_recipe_law"] is None:
        del result["follower_recipe_law"]
    if result['final_room_profile'] is None:
        del result['final_room_profile']
    if result['opponent_clear_profile'] is None:
        del result['opponent_clear_profile']
    if result['opponent_known_profile'] is None:
        del result['opponent_known_profile']
    if result['untimed_profile'] is None:
        del result['untimed_profile']
    return result


@dataclass(frozen=True)
class PublicParticles:
    """A frozen proposed bank from the client's public prefix, not a server state.

    ``law`` must name the actually used provider (uniform diagnostics must say
    so). The planner does not infer or silently replace a learned proposal law.
    """
    assignments: tuple
    weights: tuple
    law: str
    world_sha256: str | None = None
    obs_sha256: str | None = None

    def __post_init__(self):
        if not self.assignments or len(self.assignments) != len(self.weights) \
                or type(self.assignments) is not tuple or type(self.weights) is not tuple \
                or not isinstance(self.law, str) or not self.law.startswith("public-") \
                or any(type(w) not in (float, int) or not math.isfinite(w) or w <= 0 for w in self.weights):
            raise ValueError("public particle bank needs immutable assignments, positive weights and a named law")
        for assignment in self.assignments:
            if type(assignment) is not tuple or any(type(row) is not tuple or len(row) != 2 for row in assignment) \
                    or len(dict(assignment)) != len(assignment):
                raise ValueError("each assignment is an immutable distinct UID/code list")
        if (self.world_sha256 is None) != (self.obs_sha256 is None) or any(
                not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                for value in (self.world_sha256, self.obs_sha256) if value is not None):
            raise ValueError("a public-world proposal must bind both its World and pending observation SHAs")


def make_particle_provider(name, *, entropy_profile=None, service_identity=None):
    """Closed provider registry; construction has no file, native engine or model effects."""
    if name == "public_capacity_natural_conditioned":
        from .conditioned_particles import PublicConditionedCapacityParticles
        return PublicConditionedCapacityParticles(entropy_profile=entropy_profile)
    if name == "public_replay_filtered":
        from .replay_filtered_particles import PublicReplayFilteredParticles
        return PublicReplayFilteredParticles(entropy_profile=entropy_profile)
    if name == "public_current_root_uniform":
        if entropy_profile is not None:
            raise ValueError("current-root sampling does not accept an opening replay entropy profile")
        from .current_particles import CurrentRootProvider
        return CurrentRootProvider()
    if name == "public_current_root_count_head":
        if entropy_profile is not None:
            raise ValueError("count-head current-root sampling does not accept an opening replay entropy profile")
        from .current_particles import CurrentRootProvider
        return CurrentRootProvider(count_head=True)
    if name == "public_current_root_ar_diagnostic":
        if entropy_profile is not None:
            raise ValueError("current AR sampling does not accept an opening replay entropy profile")
        from .current_particles import CurrentRootProvider
        from .current_root_protocol import AR_PROVIDER
        if not isinstance(service_identity, dict) or type(service_identity.get("ar_head")) is not dict \
                or service_identity["ar_head"].get("diagnostic_only") is not True:
            raise ValueError("AR current-root provider requires the service's verified diagnostic head identity")
        provider = CurrentRootProvider(ar_head_identity=service_identity["ar_head"])
        if provider.identity.get("provider") != AR_PROVIDER:
            raise ValueError("AR current-root provider identity differs")
        return provider
    if entropy_profile is not None:
        raise ValueError("entropy certification belongs only to the explicit natural-conditioned provider")
    if name == "uniform_diagnostic":
        return UniformDiagnosticParticles()
    if name == "public_world_uniform_diagnostic":
        from .agent_public_particles import PublicWorldDiagnosticParticles
        return PublicWorldDiagnosticParticles()
    if name == "public_capacity_uniform":
        from .causal_particles import PublicCapacityParticles
        return PublicCapacityParticles()
    raise ValueError("unknown explicit public particle provider")


def allocation(logits, config):
    """Plan before any rollout: all rows or the top-prior subset for a wide menu."""
    logits = np.asarray(logits, np.float64)
    if logits.ndim != 1 or not len(logits) or not np.isfinite(logits).all():
        raise SearchPolicyError("root logits must match a finite nonempty legal menu")
    rows = len(logits)
    if config.candidate_law == ALL_CANDIDATES:
        candidates = tuple(range(rows))
        per_row = min(config.particles if config.particles is not None else 128, max(1, config.rollouts // rows), 128)
    elif rows <= config.rollouts:
        candidates = tuple(range(rows))
        per_row = config.rollouts // rows
    else:
        candidates = tuple(sorted(range(rows), key=lambda row: (-logits[row], row))[:config.rollouts])
        per_row = 1
    return candidates, per_row


def conditional_sampling_summary(roots):
    """Count every attempted bank, including unknown/cap/failed proposal generation.

    Complete game timing can be summarized separately, but conditional
    rejection/cap denominators must not silently omit a failed real prompt.
    Full immutable proposals and proofs remain in the original root reports.
    """
    rows, statuses = [], Counter()
    for root in roots:
        for index, audit in enumerate(root.get("conditional_admissions", [])):
            for key in ("attempted", "rejected", "unknown"):
                if type(audit.get(key)) is not int or audit[key] < 0:
                    raise ValueError("conditional admission lacks exact nonnegative attempt counts")
            for key in ("proposal_seconds", "admission_seconds", "total_seconds", "rejection_rate"):
                if type(audit.get(key)) not in (int, float) or not math.isfinite(audit[key]) or audit[key] < 0:
                    raise ValueError("conditional admission lacks finite sampling times and rejection rate")
            expected = audit["rejected"] / audit["attempted"] if audit["attempted"] else 0.
            if audit["rejected"] + audit["unknown"] > audit["attempted"] or audit["rejection_rate"] != expected \
                    or type(audit.get("cap_reached")) is not bool or not isinstance(audit.get("status"), str):
                raise ValueError("conditional admission rejection/cap accounting differs")
            statuses[audit["status"]] += 1
            rows.append({"prompt": root.get("prompt"), "bank": index,
                         **{key: audit[key] for key in ("status", "attempted", "rejected", "unknown", "rejection_rate",
                                                       "cap_reached", "proposal_seconds", "admission_seconds", "total_seconds")}})
    attempts = sum(row["attempted"] for row in rows)
    rejected = sum(row["rejected"] for row in rows)
    caps = sum(row["cap_reached"] for row in rows)
    return {"scope": "all attempted banks including failed and incomplete games",
            "banks": len(rows), "status_counts": dict(sorted(statuses.items())),
            "attempted_proposals": attempts, "certified_rejections": rejected,
            "rejection_rate": rejected / attempts if attempts else None,
            "cap_reached": caps, "cap_reached_fraction": caps / len(rows) if rows else None,
            "sampling_seconds": {"p50": float(np.percentile([row["total_seconds"] for row in rows], 50)),
                                 "p95": float(np.percentile([row["total_seconds"] for row in rows], 95)),
                                 "max": max(row["total_seconds"] for row in rows)} if rows else None,
            "per_bank": rows}


def replay_filtered_summary(roots):
    """Retain every model attempt, including failed roots and empty banks."""
    banks, discards, modes = [], Counter(), Counter()
    attempts = positive = 0
    for root in roots:
        attempts += len(root.get("non_singleton_attempts", []))
        positive += sum(search.get("anytime", {}).get("completed_stripes", 0) > 0
                        for search in root.get("searches", []))
        for audit in root.get("replay_filtered_admissions", []):
            modes[audit.get("proposal_mode", "unrecorded")] += 1
            for row in audit.get("attempts", []):
                if row.get("status") != "accepted":
                    discards[row.get("status", "unfinished")] += 1
            banks.append({"prompt": root.get("prompt"), "status": audit.get("status"),
                "attempted": len(audit.get("attempts", [])), "accepted": len(audit.get("accepted_ordinals", [])),
                "proposal_mode": audit.get("proposal_mode"), "uniform_reason": audit.get("uniform_reason"),
                **{key: audit.get(key) for key in ("proposal_seconds", "construction_seconds", "replay_seconds",
                                                  "admission_seconds", "discard_counts")}})
    return {"scope": "all non-singleton attempts including failed and incomplete games",
        "wire_prompt_attempts": len(roots), "failed_wire_prompts": sum(root.get("complete") is not True for root in roots),
        "model_attempts": attempts, "positive_searches": positive,
        "positive_search_fraction": positive / attempts if attempts else None,
        "banks": len(banks), "attempted_proposals": sum(row["attempted"] for row in banks),
        "accepted_proposals": sum(row["accepted"] for row in banks),
        "discard_counts": dict(sorted(discards.items())), "proposal_mode_counts": dict(sorted(modes.items())),
        "per_bank": banks}


def current_root_summary(roots):
    searches = [search for root in roots for search in root.get("searches", []) if not search.get("singleton")]
    attempts = sum(len(root.get("non_singleton_attempts", [])) for root in roots)
    positive = sum(search.get("anytime", {}).get("completed_stripes", 0) > 0 for search in searches)
    return {"scope": "all original prompts and non-singleton attempts, including failures",
            "wire_prompt_attempts": len(roots), "failed_wire_prompts": sum(root.get("complete") is not True for root in roots),
            "model_attempts": attempts, "positive_searches": positive,
            "positive_search_fraction": positive / attempts if attempts else None,
            "realized_particles": sum(len(search.get("particle_proofs", [])) for search in searches),
            "zero_proposal_budget_roots": sum(search.get("current_budget_empty") is True for search in searches)}


def require_exact_service(identity):
    """Correctness/debug baseline until a different numeric gate is explicitly approved."""
    if identity.get("backend") != "checkpoint" or identity.get("weights") != "iterate" \
            or identity.get("magnet") != "uniform_legal":
        raise SearchPolicyError("search requires raw iterate weights and the checkpoint's uniform_legal magnet")
    if identity.get("search_batching") != {"law": "valid-row-padding/v1", "sizes": [1]}:
        raise SearchPolicyError("search currently requires explicit exact_batch1: --search-batch-sizes 1")
    if identity.get("opponent_recipe_mode") != "mirror":
        raise SearchPolicyError("asymmetric known recipes need a dual-perspective binding")


def create_search_identity(service_identity, verified_rules_receipt, particle_identity, config, *,
                           numeric_mode="exact_batch1", numeric_evidence=None, stripe_evidence=None):
    """Pure parent-process registration; no native core, model, socket or file I/O.

    The Elo host can verify files and register this identity without mapping the
    follower core. Only its separately spawned client process loads that core.
    Critic is a declared future estimator, not permission to substitute a fake
    Q model or silently run rollouts under a critic label.
    """
    from .search_rules import SCHEMA as RULE_SCHEMA
    from .search_numeric import numeric_contract, NumericContractError
    config = SearchConfig(**config) if isinstance(config, dict) else config
    if type(config) is not SearchConfig:
        raise ValueError("registered search configuration is required")
    from .agent_specialization_search import registration as specialization_registration
    try:
        specialization=specialization_registration(service_identity,particle_identity,search_settings(config))
    except ValueError as exc:
        raise SearchPolicyError("search actor binding refused: "+str(exc)) from exc
    if service_identity.get("weights") != "iterate" \
            or service_identity.get("magnet") != "uniform_legal" or service_identity.get("opponent_recipe_mode") != "mirror":
        raise SearchPolicyError("search requires raw checkpoint/uniform_legal and explicit mirror mode")
    if service_identity.get("selection") != config.selection \
            or config.selection == "sample" and service_identity.get("temperature") != 1.0:
        raise SearchPolicyError("the root selection/temperature must match the registered comparison mode")
    if not isinstance(verified_rules_receipt, dict) or verified_rules_receipt.get("schema") != RULE_SCHEMA \
            or verified_rules_receipt.get("training_eligible") is not False \
            or verified_rules_receipt.get("core_sha256") != service_identity.get("native_build", {}).get("outputs", {}).get("libmfcore.so") \
            or verified_rules_receipt.get("core_commit") != service_identity.get("native_build", {}).get("core_commit"):
        raise SearchPolicyError("verified rule receipt does not bind this service's core")
    if not isinstance(particle_identity, dict) or not isinstance(particle_identity.get("law"), str) \
            or not particle_identity["law"].startswith("public-"):
        raise SearchPolicyError("particle registration must explicitly name its public-only law")
    if config.budget_law == MULTI.LAW and not is_current_provider(particle_identity.get("provider")):
        raise SearchPolicyError("multiplexed scheduling is admitted only for current-root particles")
    check_follower_recipe(config, particle_identity)
    if config.budget_law != MULTI.LAW and stripe_evidence is not None:
        raise SearchPolicyError("a legacy scheduler cannot silently consume multiplexed evidence")
    if config.bank_unknown_law == F.LAW:
        from .causal_profile import is_bundle, verify_bundle
        if particle_identity.get("provider") != "own-pending-natural-conditioned-capacity/v1" \
                or not is_bundle(particle_identity.get("entropy_profile")):
            raise SearchPolicyError("bank fallback requires the registered live-audited conditioned producer")
        verify_bundle(particle_identity["entropy_profile"])
    if "world_rpc" in particle_identity:
        from ..agent.public_world_codec import RPC_CAPABILITY
        if particle_identity["world_rpc"] != RPC_CAPABILITY or service_identity.get("client_public_world") != RPC_CAPABILITY:
            raise SearchPolicyError("the public-world provider requires this service's registered read-only capability")
    try:
        numeric = numeric_contract(service_identity, mode=numeric_mode, evidence=numeric_evidence)
    except NumericContractError as exc:
        raise SearchPolicyError(str(exc)) from exc
    registration = {"schema": "mirrorforce_option_a_identity/v1", "law": "option-a-from-opening/v1",
                                  "numeric_mode": numeric_mode, "numeric_contract": numeric,
                                  "settings": search_settings(config), "particles": particle_identity,
                                  "rules": verified_rules_receipt}
    if config.final_room_profile is not None:
        registration['final_room_profile'] = dict(FB.PROFILE)
    if config.opponent_clear_profile is not None:
        registration['opponent_clear_profile'] = OC.validate_profile(OC.PROFILE)
    if config.opponent_known_profile is not None:
        registration['opponent_known_profile'] = OK.validate_profile(OK.PROFILE)
    if specialization is not None:
        if verified_rules_receipt.get('specialization_search')!=specialization:
            raise SearchPolicyError('specialization actor differs from the registered parent-rule proof')
        registration['specialization_search']=specialization
    provider = particle_identity.get("provider")
    if is_current_provider(provider):
        expected_particles = CURRENT.PARTICLE_IDENTITY if provider == CURRENT.PROVIDER else \
            CURRENT.COUNT_PARTICLE_IDENTITY if provider == CURRENT.COUNT_PROVIDER else {
            **CURRENT.AR_PARTICLE_IDENTITY, "ar_head": service_identity.get("ar_head")}
        if particle_identity != expected_particles or service_identity.get("current_root") != CURRENT.CAPABILITY \
                or config.estimator != "rollout" or config.budget_law not in ANYTIME_LAWS or config.total_seconds is None \
                or config.candidate_law != ALL_CANDIDATES or config.particles is None \
                or config.bank_unknown_law != F.FATAL:
            raise SearchPolicyError("current-root search requires its explicit observer capability and all-legal finite law")
        if provider == CURRENT.AR_PROVIDER and (service_identity.get("ar_features", {}).get("source") != "same-public-forward"
                or service_identity.get("ar_head", {}).get("parent_actor_sha256") != service_identity.get("checkpoint_sha256")
                or service_identity.get("ar_head", {}).get("search_admission") is not False):
            raise SearchPolicyError("diagnostic AR must bind same-forward public features to the exact serving parent")
        if provider == CURRENT.COUNT_PROVIDER and service_identity.get("replay_belief", {}).get("source") \
                != "same-public-forward":
            raise SearchPolicyError("count-head particles need the service's same-forward count belief")
        registration.update(schema=current_identity_schema(config), law="current-root-in-place/v1",
                            memory_law=CURRENT.MEMORY_LAW, rollout_resources=current_resources(config))
        if config.budget_law == MULTI.LAW:
            from .agent_stripe_contract import stripe_contract
            try:
                registration["stripe_contract"] = stripe_contract(service_identity, config, stripe_evidence)
            except ValueError as exc:
                raise SearchPolicyError(str(exc)) from exc
    if particle_identity.get("provider") in ("own-pending-public-capacity/v1", "own-pending-natural-conditioned-capacity/v1"):
        from .causal_proof import producer_contract
        registration["producer"] = producer_contract()
    if particle_identity.get("provider") == REPLAY_FILTERED_PROVIDER:
        from .replay_filtered_particles import PublicReplayFilteredParticles
        expected = PublicReplayFilteredParticles(particle_identity.get("entropy_profile")).identity
        if particle_identity != expected or config.total_seconds is None or config.estimator != "rollout" \
                or config.candidate_law != ALL_CANDIDATES or config.particles is None \
                or config.bank_unknown_law != F.FATAL:
            raise SearchPolicyError("replay-filtered search needs its explicit producer and whole-prompt cap")
        registration["rollout_resources"] = dict(REPLAY_FILTERED_RESOURCES)
    if particle_identity.get("provider") == "own-pending-natural-conditioned-capacity/v1":
        from .causal_proof import sha256
        from .causal_rejection import verify_entropy_profile
        profile = particle_identity.get("entropy_profile")
        if particle_identity.get("entropy_profile_sha256") != (sha256(profile) if profile is not None else None):
            raise SearchPolicyError("conditioned entropy evidence differs from its declared identity")
        if profile is not None:
            verify_entropy_profile(profile, core_sha256=verified_rules_receipt["core_sha256"],
                scripts_sha256=verified_rules_receipt["scripts"]["file_map_sha256"],
                core_commit=verified_rules_receipt["core_commit"])
    return json.loads(json.dumps(registration, allow_nan=False))


def bank_unknown_root(root, pending, failure, *, config, seed, particle_identity):
    """Use the already pending prior only; no service call or memory update."""
    from .conditioned_particles import TypedBankUnknown
    started = time.monotonic()
    if config.bank_unknown_law != F.LAW or type(failure) is not TypedBankUnknown:
        raise SearchPolicyError("no explicit live bank-unknown fallback capability")
    root._check()
    root.snapshot.verify()
    audit, proof = failure.admission, failure.fallback_record
    candidates, count = allocation(pending["logits"], config)
    if pending["rows"] != len(pending["logits"]) or len(pending["groups"]) != pending["rows"]:
        raise SearchPolicyError("fallback root menu/logits/groups differ")
    profile = particle_identity["entropy_profile"]
    F.verify_unknown_record(proof, audit, root_snapshot_sha256=root.snapshot.digest,
        obs_sha256=pending["obs_sha256"], requested_count=count, proposal_seed=seed,
        entropy_profile_sha256=particle_identity["entropy_profile_sha256"],
        source_sha256=profile["producer_source_sha256"])
    chosen, policy = A.greedy_without_search(pending["logits"])
    settings = search_settings(config)
    return chosen, {"schema": F.ROOT_SCHEMA, "fallback": "bank_unknown", "fallback_law": F.LAW,
        "fallback_record": proof, "conditional_admission": audit, "training_eligible": False,
        "settings": settings, "settings_sha256": hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
        "obs_sha256": pending["obs_sha256"], "rows": pending["rows"], "candidate_rows": list(candidates),
        "lines_per_row": count, "particle_seed": seed, "particle_law": particle_identity["law"],
        "particle_weights": [], "particle_proofs": [], "world_sha256": audit["world_sha256"],
        "world_obs_sha256": pending["obs_sha256"], "q": [None] * pending["rows"], **policy,
        "chosen": chosen, "wide_menu": pending["rows"] > config.rollouts, "complete": True,
        "bank_complete": False, "search_changed": False, "search_gain": None, "counterfactual_q": None,
        "phase_seconds": {"admission": 0., "rollout": 0., "update": time.monotonic() - started}}


def plan_root(root, pending, own_session, bank, public_recipe, *, call, config, seed, rng, deadline=None, on_admission=None):
    """Rebuild hypotheses, continue each legal root row, and take one registered MMD step.

    The caller owns the real session and commits the returned index once. This
    function only clones it; it never sends a game response or changes its RNG.
    It owns all synthetic base/line sessions and snapshots under one root lease.
    """
    if config.estimator != "rollout":
        raise SearchPolicyError("critic estimator awaits a verified T6 Q checkpoint and hypothetical-input binding")
    if deadline is not None and (type(deadline) not in (float, int) or not math.isfinite(deadline)):
        raise ValueError("absolute root deadline must be finite")
    causal = conditioned = filtered = current = False
    if type(bank) is not PublicParticles:
        from .causal_particles import PublicCausalParticles
        from .conditioned_particles import ConditionedCausalParticles
        from .replay_filtered_particles import ReplayFilteredBank
        from .current_particles import CurrentRootParticles
        conditioned = type(bank) is ConditionedCausalParticles
        filtered = type(bank) is ReplayFilteredBank
        current = type(bank) is CurrentRootParticles
        if type(bank) is not PublicCausalParticles and not conditioned and not filtered and not current:
            raise SearchPolicyError("an explicit immutable public particle bank is required")
        causal = not current
    if bank.obs_sha256 is not None and bank.obs_sha256 != pending["obs_sha256"]:
        raise SearchPolicyError("the public-world bank belongs to another pending observation")
    root._check()
    rows = pending["rows"]
    if rows != len(pending["logits"]) or len(pending["groups"]) != rows:
        raise SearchPolicyError("root menu, logits and groups have different lengths")
    candidates, per_row = allocation(pending["logits"], config)
    bank_count = bank.requested_count if conditioned or filtered else len(bank.assignments)
    if bank_count != per_row:
        raise SearchPolicyError("every root row must use the same full bank once; bank size differs from allocation")
    if rows < 2:
        raise SearchPolicyError("singleton prompts need a root but no rollout decision")
    started = time.monotonic()
    deadline = min(root.deadline, started + config.seconds, deadline if deadline is not None else float("inf"))
    values = [None] * rows
    base_sessions, snapshots, cleanup_errors = [], [], []
    runner = None
    proofs = []
    anytime = None
    conditional_admission = None
    filtered_admission = None
    actual_count = per_row
    weights = None if filtered else bank.weights
    bounded = filtered or current
    multiplexed = config.budget_law == MULTI.LAW
    if multiplexed and not current:
        raise SearchPolicyError("multiplexed scheduling is admitted only for current-root particles")
    if current and (config.budget_law not in ANYTIME_LAWS or config.candidate_law != ALL_CANDIDATES
                    or config.total_seconds is None or bank.own_session != own_session):
        raise SearchPolicyError("current-root search requires bounded all-legal stripes and its exact live own session")
    resource_audit = resource_record(config) if bounded else None
    live_line_sessions = set()

    def tracked_call(request):
        result = call(request)
        if request["op"] == "clone":
            name = result["session"]
            if name in live_line_sessions or name == own_session or name in base_sessions:
                raise SearchPolicyError("a multiplexed clone did not return an independent owned session")
            live_line_sessions.add(name)
            resource_audit["peak_line_sessions"] = max(resource_audit["peak_line_sessions"], len(live_line_sessions))
        elif request["op"] == "close":
            live_line_sessions.discard(request["session"])
        return result
    admission_started = time.monotonic()
    opened = bank.open_roots(root, public_recipe, seed=seed, deadline=deadline) if causal or current else \
        synthetic_roots(root, [dict(x) for x in bank.assignments], public_recipe, seed=seed)
    with opened as particles:
        try:
            particle_list = tuple(particles)
            actual_count = len(particles)
            if filtered:
                if actual_count > per_row or config.budget_law != A.LAW or config.total_seconds is None:
                    raise SearchPolicyError("replay-filtered particles require their variable common-bank anytime law")
                filtered_admission = particles.admission
                weights = (1.0,) * actual_count
                if on_admission is not None:
                    on_admission(json.loads(json.dumps(filtered_admission, allow_nan=False)))
            elif actual_count != per_row:
                raise SearchPolicyError("natural producer did not admit the entire fixed requested bank")
            if conditioned:
                conditional_admission = particles.admission
                if on_admission is not None:
                    on_admission(json.loads(json.dumps(conditional_admission, allow_nan=False)))
            # ALL hypotheses were admitted above, before allocating any action
            # return or opening synthetic memory. Rebuild each history once,
            # then clone that synthetic session for its independent lines.
            for particle in particle_list:
                if current:
                    # retains each start on one shared branch: observing the
                    # last written driver for every particle would collapse the bank.
                    particle.activate()
                state = particle.driver.save_pystate()
                snap = root.api.duel_snapshot(particle.driver.pduel)
                if not snap:
                    raise SearchPolicyError("could not retain an admitted synthetic root")
                snapshots.append((snap, state))
                if not bounded:
                    base_sessions.append(call(particle.stream)["session"])
                proofs.append(particle.proof)

            def restore(index):
                particle, (snap, state) = particle_list[index], snapshots[index]
                if root.api.duel_rollback(particle.driver.pduel, snap) != 0:
                    raise SearchPolicyError("failed to restore the admitted particle's own engine")
                particle.driver.restore_pystate(state)
                return RestoredRoot(particle.driver, particle.prompt,
                                    particle.map_root_response if current else None)

            def opponent(index, line):
                return call({"op": "clone", "session": base_sessions[index],
                             "seed": (int(seed) * 1_000_003 + line + 71) & 0x7fffffff})["session"]

            admission_seconds = time.monotonic() - admission_started
            rollout_started = time.monotonic()

            def instrument(owner):
                def dispatch(request):
                    began = time.monotonic()
                    try:
                        return call(request)
                    finally:
                        owner.stats["service_seconds"] += time.monotonic() - began
                owner._call = dispatch
                return owner

            if config.budget_law in ANYTIME_LAWS:
                def make_runner(index, stripe_seed, dispatch=None):
                    rpc = call if dispatch is None else dispatch
                    if bounded:
                        base = rpc(particle_list[index].stream)["session"]
                        base_sessions.append(base)  # register before any later clock or constructor check
                        resource_audit["peak_base_sessions"] = max(resource_audit["peak_base_sessions"], len(base_sessions))
                    else:
                        base = base_sessions[index]
                    def common_opponent(start, line):
                        return rpc({"op": "clone", "session": base,
                                     "seed": (stripe_seed * 1_000_003 + 71) & 0x7fffffff})["session"]
                    owner = RootLines(sock=None, driver=None, api=root.api, restore=restore, starts=[index],
                        root_message=None, own_session=own_session, opponent_session=common_opponent,
                        seat=root.owner.follower.viewer, rows=len(candidates), lines_per_row=1,
                        depth=config.depth, td_lambda=config.td_lambda, seed=stripe_seed,
                        max_seconds=config.seconds, candidate_rows=candidates, seed_law="common-stripe/v1",
                        **({"max_live_lines": min(len(candidates), MULTI.DEFAULT_LIMITS["lines_per_stripe"])
                           if multiplexed else REPLAY_FILTERED_LANES} if bounded else {}))
                    if dispatch is None:
                        instrument(owner)
                    else:
                        owner._call = dispatch
                    if bounded:
                        original_close = owner.close
                        def close_stripe():
                            if owner.closed:
                                return
                            try:
                                original_close()
                            finally:
                                if not multiplexed:
                                    resource_audit["peak_line_sessions"] = max(resource_audit["peak_line_sessions"],
                                        owner.stats["peak_owned_sessions"])
                                resource_audit["chunks"] += owner.stats["chunks"]
                                if base in base_sessions:
                                    rpc({"op": "close", "session": base, "messages": []})
                                    base_sessions.remove(base)
                        owner.close = close_stripe
                    return owner
                if actual_count:
                    if multiplexed:
                        estimates, anytime = MULTI.multiplexed_stripes(rows=len(candidates), weights=weights, seed=seed,
                            deadline=deadline, make_runner=make_runner, call=tracked_call)
                        if live_line_sessions:
                            raise SearchPolicyError("multiplexed rollout left owned live line sessions")
                    else:
                        estimates, anytime = A.balanced_stripes(rows=len(candidates), weights=weights, seed=seed,
                                                               deadline=deadline, make_runner=make_runner)
                else:
                    estimates = [None] * len(candidates)
                    anytime = {"law": A.LAW, "planned_stripes": 0, "completed_stripes": 0,
                        "budget_exhausted": False, "budget_zero_search": True, "completed_indices": [], "stripes": [],
                        "stats": {"lines": 0, "attempted_lines": 0, "rounds": 0, "items": 0,
                            "service_seconds": 0., "engine_seconds": 0., "void": 0, "void_logs": []}}
                stats = anytime["stats"]
                if current:
                    for index in anytime["completed_indices"]:
                        particles.complete(index)
            else:
                runner = instrument(RootLines(sock=None, driver=None, api=root.api, restore=restore, starts=range(per_row),
                                   root_message=None, own_session=own_session, opponent_session=opponent,
                                   seat=root.owner.follower.viewer, rows=len(candidates), lines_per_row=per_row,
                                   depth=config.depth, td_lambda=config.td_lambda, seed=seed,
                                   max_seconds=config.seconds, candidate_rows=candidates))
                returns = runner.run(deadline=deadline)
                estimates = q_values(returns, len(candidates), dict(enumerate(weights)))
                stats = runner.stats
            rollout_seconds = time.monotonic() - rollout_started
            zero = anytime is not None and anytime["budget_zero_search"]
            if not zero and any(v is None or not math.isfinite(v) for v in estimates):
                raise SearchPolicyError("an action lost its required particle returns")
            for row, value in zip(candidates, estimates):
                values[row] = value
        finally:
            if runner is not None:
                try:
                    runner.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            for snap, _ in snapshots:
                try:
                    root.api.duel_snapshot_free(snap)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            for name in base_sessions:
                try:
                    call({"op": "close", "session": name, "messages": []})
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors:
                raise SearchPolicyError("failed to clean synthetic root resources") from cleanup_errors[0]
    root._check()
    update_started = time.monotonic()
    if anytime is not None and anytime["budget_zero_search"]:
        _, update = A.greedy_without_search(pending["logits"])
    else:
        update = call({"op": "root_step", "logits": pending["logits"], "groups": pending["groups"], "q": values,
                       "alpha": config.alpha, "beta": config.beta, "magnet": "uniform_legal"})
    probability = np.asarray(update["policy"], np.float64)
    if probability.shape != (rows,) or not np.isfinite(probability).all() or np.any(probability < 0) \
            or not np.isclose(probability.sum(), 1):
        raise SearchPolicyError("root update returned an invalid legal policy")
    chosen = int(np.argmax(probability)) if config.selection == "greedy" else int(rng.choice(rows, p=probability))
    settings = search_settings(config)
    record = {"schema": current_root_schema(config) if current else SCHEMA,
              "training_eligible": False, "settings": settings,
              "settings_sha256": hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
              "obs_sha256": pending["obs_sha256"], "rows": rows, "candidate_rows": list(candidates),
              "lines_per_row": per_row, "particle_law": bank.law, "particle_weights": list(weights),
              "particle_seed": int(seed),
              "particle_proofs": proofs, "q": values, "prior": update["prior"], "policy": update["policy"],
              "world_sha256": bank.world_sha256, "world_obs_sha256": bank.obs_sha256,
              "chosen": chosen, "wide_menu": rows > config.rollouts, "complete": True,
              "stats": stats, "seconds": time.monotonic() - started,
              "phase_seconds": {"admission": admission_seconds, "rollout": rollout_seconds,
                                "update": time.monotonic() - update_started}}
    if anytime is not None:
        record["anytime"] = anytime
    if current:
        record.update(memory_law=CURRENT.MEMORY_LAW, opening_replayed=False,
                      own_deck_order_realized=True, rollout_resources=resource_audit,
                      root_snapshot_sha256=bank.root_hash, root_id=bank.root_id)
    elif filtered:
        record.update(replay_filtered_admission=filtered_admission, actual_bank_count=actual_count,
                      requested_bank_count=per_row, particle_target_cap=config.particles,
                      rollout_resources=resource_audit,
                      replay_filtered_empty=actual_count == 0)
    elif conditioned:
        record["conditional_admission"] = conditional_admission
    elif causal:
        from .causal_proof import sha256
        record["causal_bank"] = {"history_sha256": bank.history_sha256, "source_sha256": bank.source_sha256,
                                 "layout_sha256": [sha256(layout) for layout in bank.assignments],
                                 "sampler": json.loads(bank.sampler_proof_json)}
    return chosen, record


def current_budget_result(root, pending, *, config, seed, deadline, phase, particle_law=CURRENT.LAW):
    """An actually exhausted local search budget, not an unknown/rejected bank.

    Requires the real root to remain valid after producer cleanup. There are
    zero realized particles, no Q values and no MMD update or extra forward.
    """
    began = time.monotonic()
    if config.budget_law not in ANYTIME_LAWS or config.candidate_law != ALL_CANDIDATES or config.selection != "greedy" \
            or config.total_seconds is None or type(deadline) not in (int, float) or not math.isfinite(deadline) \
            or time.monotonic() < deadline or phase not in ("before_proposals", "before_materialization", "materialization_or_view"):
        raise SearchPolicyError("current-root prior fallback needs an actually exhausted registered search clock")
    if type(pending.get("rows")) is not int or pending["rows"] < 2 \
            or pending["rows"] != len(pending.get("logits", [])) or pending["rows"] != len(pending.get("groups", [])):
        raise SearchPolicyError("current-root prior fallback requires the exact non-singleton pending menu")
    root._check()
    root.snapshot.verify()
    candidates, count = allocation(pending["logits"], config)
    chosen, update = A.greedy_without_search(pending["logits"])
    stats = {"lines": 0, "attempted_lines": 0, "rounds": 0, "items": 0, "service_seconds": 0.,
             "engine_seconds": 0., "void": 0, "void_logs": []}
    settings = search_settings(config)
    anytime = MULTI.empty_stripes(len(candidates)) if config.budget_law == MULTI.LAW else {
        "law": A.LAW, "planned_stripes": 0, "completed_stripes": 0, "budget_exhausted": False,
        "budget_zero_search": True, "completed_indices": [], "stripes": [], "stats": stats}
    return chosen, {"schema": current_root_schema(config), "training_eligible": False, "settings": settings,
        "settings_sha256": hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
        "obs_sha256": pending["obs_sha256"], "rows": pending["rows"], "candidate_rows": list(candidates),
        "lines_per_row": count, "particle_law": particle_law, "particle_weights": [], "particle_seed": seed,
        "particle_proofs": [], "q": [None] * pending["rows"], **update, "world_sha256": None,
        "world_obs_sha256": pending["obs_sha256"], "chosen": chosen,
        "wide_menu": pending["rows"] > config.rollouts, "complete": True, "stats": stats,
        "current_budget_empty": True, "proposal_budget_exhausted": True, "budget_phase": phase,
        "actual_bank_count": 0, "requested_bank_count": count,
        "memory_law": CURRENT.MEMORY_LAW, "opening_replayed": False, "own_deck_order_realized": False,
        "root_snapshot_sha256": root.snapshot.digest, "root_id": "current-root-" + str(root.epoch) + "-" + root.snapshot.digest,
        "rollout_resources": resource_record(config),
        # No stripes were planned. Their existing accounting law is unchanged;
        # the producer's exhausted budget is recorded separately above.
        "anytime": anytime,
        "phase_seconds": {"admission": 0., "rollout": 0., "update": time.monotonic() - began}}


def gate_prior_root(root, pending, *, config, seed, gate):
    """Intentional no-search decision, distinct from an exhausted producer."""
    if config.on_demand is None or gate.search or pending['rows'] <= 1:
        raise SearchPolicyError("gate prior fallback needs an explicit no-search admission")
    root._check()
    root.snapshot.verify()
    chosen, policy = A.greedy_without_search(pending['logits'])
    candidates, count = allocation(pending['logits'], config)
    settings = search_settings(config)
    anytime = MULTI.empty_stripes(pending['rows'])
    return chosen, {"schema": G.ROOT_SCHEMA, "training_eligible": False, "gate_skipped": True,
        "settings": settings, "settings_sha256": hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
        "obs_sha256": pending['obs_sha256'], "rows": pending['rows'], "candidate_rows": list(candidates),
        "lines_per_row": count, "chosen": chosen, "complete": True, "root_snapshot_sha256": root.snapshot.digest,
        **policy, "q": [None] * pending['rows'], "particle_seed": seed,
        "particle_weights": [], "particle_proofs": [], "actual_bank_count": 0, "requested_bank_count": count,
        "phase_seconds": {"admission": 0., "rollout": 0., "update": 0.}, "anytime": anytime,
        "stats": anytime['stats'], "gate_decision": G.decision_wire(gate)}


class UniformDiagnosticParticles:
    """Explicit uniform diagnostic, never advertised as the checkpoint's learned belief.

    Reuses the existing public ledger/category constrained sampler and its rule
    checks; no rule fingerprint is weakened to make a new asset tree pass.
    """
    identity = {"law": "public-opponent-uniform-diagnostic/v1", "provider": "client-admitted-joint-bank/v1",
                "own_deck": "accepted-follower-replay-order; not independently resampled"}

    def propose(self, root, client, *, count, seed, context, extra_origins, call=None, session=None, pending=None):
        from ..common.client_midgame import admit_root
        from ..common.stage_a_joint_belief import JointRecipe
        prior = JointRecipe.from_decks(client.main, client.extra)
        adapter = admit_root(root, client, prior, proposals=count, seed=seed, context=context,
                             extra_origins=extra_origins,
                             adapter_options={"own_deck_order": True, "history_domain": "all_turns"})
        origin = root.host["sync"]["receipt_state"].replay_events[0].after.entities
        opponent = 1 - client.result.our_player
        wanted = {item.uid for item in origin if item.owner == opponent}
        assignments = []
        for index in range(count):
            current = {**adapter._codes, **adapter._assignment(index)}
            if not wanted <= current.keys():
                raise SearchPolicyError("an initial opponent object lost its root identity binding")
            assignments.append(tuple(sorted((uid, current[uid]) for uid in wanted)))
        return PublicParticles(tuple(assignments), tuple(adapter.bank.probabilities), self.identity["law"])


class SearchPolicy(RemotePolicy):
    """A network client with a required owned follower root at every prompt.

    Only this client's received packets and actual submitted responses reach
    the follower. The root is held through the network send, then the sent
    response is committed before the next received packet (the existing response-tee lifecycle). Only a separately registered typed bank unknown
    may reuse the already pending prior; missing roots and hard errors fail.
    """
    name = "option-a-search"
    report_schema = "mirrorforce_option_a_client/v1"

    def __init__(self, address, expected_identity, *, rule_files, stage_sha256, checkpoint,
                 particle_provider, config=SearchConfig(), seed=0, timeout=120.0,
                 numeric_mode="exact_batch1", numeric_evidence=None, core_upgrade=None, stripe_evidence=None):
        from .search_rules import verify_rules
        if config.estimator != "rollout":
            raise SearchPolicyError("critic estimator awaits a verified T6 Q checkpoint and hypothetical-input binding")
        if not callable(getattr(particle_provider, "propose", None)) \
                or not isinstance(getattr(particle_provider, "identity", None), dict):
            raise ValueError("an explicitly registered public particle provider is required")
        # Verify actual files BEFORE constructing/loading the follower core.
        receipt = verify_rules(rule_files, stage_sha256=stage_sha256, checkpoint=checkpoint,
                               service=expected_identity, core_upgrade=core_upgrade)
        registration = create_search_identity(expected_identity, receipt, particle_provider.identity, config,
            numeric_mode=numeric_mode, numeric_evidence=numeric_evidence, stripe_evidence=stripe_evidence)
        from ..puzzle.core import get_core
        core = get_core(lib_path=rule_files.core, db_path=rule_files.cards_db, script_dirs=[rule_files.scripts],
                        mode=ctypes.RTLD_LOCAL)
        if Path(core.lib_path).resolve() != Path(rule_files.core).resolve():
            raise SearchPolicyError("another core is already mapped in the client process")
        if Path(core.db_path).resolve() != Path(rule_files.cards_db).resolve() \
                or tuple(Path(p).resolve() for p in core.script_dirs) != (Path(rule_files.scripts).resolve(),):
            raise SearchPolicyError("the mapped follower core uses different database/script paths")
        super().__init__(address, expected_identity, seed=seed, timeout=timeout)
        self.follower_core, self.rules_receipt = core, receipt
        self.config, self.particles = config, particle_provider
        self.provider_identity = json.loads(json.dumps(particle_provider.identity))
        self.search_identity = registration
        if is_current_provider(self.provider_identity.get("provider")):
            self.name = "current-root-search"
            self.report_schema = current_client_schema(config)
        self.owner = self.capture_root = self.root = None
        self.sent, self.committed = [], 0
        self.root_rows, self.current_row = [], None
        self.final_response = self.prompt_packet = self.context = None
        self.rng = np.random.default_rng(seed)
        from ..search.banish_origin import BanishOriginTracker
        self.banish = BanishOriginTracker()
        self.final_follower_stats = None
        self.allow_server_surrender = False
        self.allow_server_network_terminal = False
        self.external_terminal = None
        self._search_deadline = self._response_deadline = self._prompt_started = None
        self._clock_record = None
        self.search_gate = G.TurnSearchGate(config.on_demand) if config.on_demand is not None else None
        self.final_room_clock_enabled = config.final_room_profile is not None
        self._room_clock_receipts, self._latest_room_clock = [], {}
        self._clock_confirmations = {}
        self._clock_bound_at = self._last_successful_send_at = self._final_clock_record = None

    def _call(self, request):
        if self.config.budget_law in ANYTIME_LAWS:
            deadline = time.monotonic() + self.config.rpc_seconds
            if self._response_deadline is not None:
                deadline = min(deadline, self._response_deadline)
            # In-flight requests may finish in the explicitly reserved cleanup
            # window. A transport timeout is fatal, not a budget-only fallback:
            # the response stream and returned resource ownership are uncertain.
            return W.call(self.sock, request, deadline=deadline)
        return W.call(self.sock, request)

    def bind(self, client):
        if getattr(client, "response_tee", None) is not None:
            raise SearchPolicyError("another consumer already owns the client's submitted-response stream")
        if self.owner is not None or not client.track_board or client.card_pool is None:
            raise SearchPolicyError("search needs a fresh client and its pinned public shadow board")
        if self.final_room_clock_enabled:
            from .room_clock import verified_room_seconds
            if verified_room_seconds(client) != FB.ROOM_SECONDS or self._clock_bound_at is not None:
                raise SearchPolicyError('final-room clock needs a fresh session in its actual600-second room')
            self._clock_delivery = getattr(client, 'clock_delivery', 'before_prompt')
            if self._clock_delivery not in FB.DELIVERIES:
                raise SearchPolicyError('final-room clock delivery was not explicitly supported')
            if getattr(client, 'response_ready_guard', None) is not None:
                raise SearchPolicyError('another consumer owns the pre-send response guard')
        super().bind(client)
        if self.final_room_clock_enabled:
            self._clock_bound_at = time.monotonic()
            client.response_ready_guard = self._guard_response
        client.response_tee = self.sent
        if self.search_gate is not None:
            if getattr(client, 'response_sent_callback', None) is not None:
                raise SearchPolicyError("another consumer owns the successful-send callback")
            client.response_sent_callback = self._response_sent

    def observe_room_clock(self, payload, *, received_at, client):
        """Called only by the verified TIME_LIMIT handler, with its one receipt timestamp."""
        if not self.final_room_clock_enabled or not self.active:
            return
        if client is not self._client or self.failures or self._clock_bound_at is None or self.committed != len(self.sent) \
                or self.committed != len(self.responses):
            raise SearchPolicyError('a room clock cannot certify an uncommitted or failed response stream')
        sample = FB.receive_time_limit(payload, game_id=self.session, prompt_index=self.committed,
            received_at=received_at, room_seconds=self._client.limits.room_seconds)
        sequence = len(self._room_clock_receipts)
        self._room_clock_receipts.append({'sequence': sequence, 'sample': asdict(sample)})
        self._latest_room_clock[sample.player] = (sequence, sample)

    def _final_public_deadlines(self, prompt_started, now):
        sequence, clock = self._latest_room_clock.get(self.seat, (None, None))
        previous = self._clock_bound_at if self._last_successful_send_at is None else self._last_successful_send_at
        allocation = FB.allocate(game_id=self.session, prompt_index=self.committed, player=self.seat,
            prompt_started=prompt_started, now=now, clock=clock, clock_delivery=self._clock_delivery,
            previous_response_at=previous, room_verified=self._client.room_verified,
            room_seconds=self._client.limits.room_seconds, enabled=True)
        self._final_clock_record = FB.trace(allocation, clock_sequence=sequence,
            clock_events_seen=len(self._room_clock_receipts), clock=clock,
            clock_delivery=self._clock_delivery, previous_response_at=previous)
        # The original policy forward may finish below the optional-search
        # floor. Only an actual search admission tightens the response deadline.
        self._search_deadline = allocation.search_deadline
        self._response_deadline = allocation.original_response_deadline
        self._clock_record = None
        return allocation

    def confirm_room_clock(self, payload, *, received_at, confirmed_at_ns, client):
        """Record only a TIME_CONFIRM whose original socket send succeeded."""
        if not self.final_room_clock_enabled or not getattr(client, 'room_started', False):
            return
        if getattr(client, 'policy', None) is not self or self.failures \
                or self.active and (client is not self._client or self.committed != len(self.sent)
                                   or self.committed != len(self.responses)) \
                or type(confirmed_at_ns) is not int or confirmed_at_ns <= 0 \
                or confirmed_at_ns/1e9 < received_at:
            raise SearchPolicyError('TIME_CONFIRM cannot certify a foreign or uncommitted client')
        self._clock_confirmations[payload[0]] = {'client': client,
            'schema': FB.CONFIRM_SCHEMA, 'session': self.session if self.active else None,
            'response_index': self.committed if self.active else 0, 'player': payload[0],
            'payload_hex': bytes(payload).hex(), 'received_at': received_at, 'confirmed_at_ns': confirmed_at_ns}

    def _guard_response(self, data):
        """No final-profile RESPONSE, tee or board mutation before its own TIME_CONFIRM."""
        from .client import DuelError
        confirmation = self._clock_confirmations.get(self.seat)
        if confirmation is None or confirmation['client'] is not self._client \
                or confirmation['response_index'] != self.committed or confirmation['player'] != self.seat \
                or confirmation['session'] not in (None, self.session) \
                or confirmation['session'] is None and (self.committed != 0 or
                    confirmation['confirmed_at_ns']/1e9 > self._clock_bound_at) \
                or self.committed != len(self.sent) or len(self.responses) != self.committed+1 \
                or self.root is None or data != self.final_response or self.failures:
            raise DuelError('final response has no current same-client TIME_CONFIRM permission')
        if self._clock_delivery == 'before_prompt' and confirmation['received_at'] > self._prompt_started \
                or self._clock_delivery == 'prompt_then_clock' and confirmation['received_at'] < self._prompt_started:
            raise DuelError('final response TIME_CONFIRM differs from its registered prompt ordering')
        now = time.monotonic()
        if confirmation['confirmed_at_ns']/1e9 > now or self._response_deadline is None or now >= self._response_deadline:
            raise DuelError('final response permission or original deadline is no longer valid')
        self.current_row['final_time_confirmation'] = {k:v for k,v in confirmation.items() if k != 'client'}

    def _response_sent(self, data):
        """Opt-in only: commit/cleanup AFTER one successful network send."""
        from .client import DuelError
        try:
            if self.search_gate is None or self.search_gate.prompt is None or data != self.final_response:
                raise SearchPolicyError("successful-send hook lost its owned prompt")
            stamp = self._client.response_sent_timestamp_ns
            if type(stamp) is not int or stamp <= 0:
                raise SearchPolicyError("successful-send hook lost its actual network completion time")
            self.current_row['on_demand']['successful_send_ns'] = stamp
            self._commit_sent()
            if self.final_room_clock_enabled:
                self._last_successful_send_at = stamp / 1e9
            row = self.current_row
            finished = self.search_gate.finish_prompt(now=time.monotonic(), delivered=True, cleanup_complete=True)
            row['on_demand']['finish'] = finished
            row['on_demand']['non_search_seconds'] = finished['prompt_seconds'] - row['on_demand']['optional_seconds']
            if self.final_room_clock_enabled and (finished['over_response_deadline'] or finished['over_admitted_prompt_cap']):
                raise SearchPolicyError('final-room successful send/cleanup exceeded its original clock deadline')
        except Exception as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
            raise DuelError("on-demand post-send cleanup failed: " + str(exc)) from exc

    def _start_follower(self, body):
        from ..worldmodel.engine import DeckList
        from ..common.client_root import ClientRootController
        from ..common.client_shadow import BlankClientSync
        from ..common.client_history_registry import PROFILE
        room, client, viewer = self._client.host_info, self._client, self.seat
        counts = struct.unpack_from("<HHHH", body, 10)
        declared = self.public_opponent_recipe
        if (counts[2 * (1 - viewer)], counts[2 * (1 - viewer) + 1]) != (len(declared["main"]), len(declared["extra"])):
            raise SearchPolicyError("the public opponent counts differ from the declared mirror recipe")
        sync = BlankClientSync(viewer=viewer, own_deck=DeckList("client-own", tuple(client.main), tuple(client.extra)),
                               opponent_main=len(declared["main"]), opponent_extra=len(declared["extra"]),
                               core=self.follower_core, seed=19, record_origins=True, record_replay=True,
                               history_registry=PROFILE, public_action_rebind=True, hidden_hand_rebind=True,
                               hidden_target_deferral=True, public_identities_only=True, read_answers=True,
                               single_pass=True, max_tries=256, max_seed_tries=16,
                               hidden_target_pool=tuple(sorted(set(declared["main"] + declared["extra"]))),
                               **({"public_opponent_recipe": declared} if self.config.follower_recipe_law else {}),
                               start_lp=room.start_lp, start_hand=room.start_hand, draw_count=room.draw_count,
                               duel_options=room.duel_rule << 16)
        self.owner = ClientRootController(sync, quiesce=lambda: None, native_idle=lambda: True)

    def observe_game_message(self, msg, body):
        from . import constants as C
        from .client import SELECT_MESSAGES, DuelError
        from ..common.client_root_menu import RootMenuBinding
        if self.external_terminal is not None:
            self.failures.append('game message after external terminal')
            raise DuelError('game message after external terminal')
        prompt_started = time.monotonic() if msg in SELECT_MESSAGES else None
        if prompt_started is not None:
            from .room_clock import received_prompt_start
            prompt_started = received_prompt_start(self._client, msg, body, prompt_started)
        super().observe_game_message(msg, body)
        if not self.active:
            return
        try:
            if msg == C.MSG_START:
                self._start_follower(bytes(body))
            if self.owner is None:
                raise SearchPolicyError("the follower did not receive the duel start")
            self._commit_sent()
            from . import agent_external_terminal as external
            law = external.NETWORK_LAW if self.allow_server_network_terminal else external.LAW
            if (self.allow_server_surrender or self.allow_server_network_terminal) and msg == C.MSG_WIN \
                    and len(body) == 2 and body[1] in external.reasons(law):
                if self.failures or self.root is not None:
                    raise SearchPolicyError('server terminal cannot erase an earlier failure or unanswered root')
                self.external_terminal = external.boundary(self.capture, self.owner, self.committed, law=law)
                return
            self.banish.observe(msg, bytes(body))
            packet = bytes([msg]) + bytes(body)
            self.owner.receive(packet)
            if msg in SELECT_MESSAGES:
                if self.root is not None:
                    raise SearchPolicyError("a new prompt arrived before the preceding response was submitted")
                self._prompt_started = prompt_started
                tracked = self.provider_identity.get("provider") in (REPLAY_FILTERED_PROVIDER, *CURRENT_PROVIDERS)
                if tracked:
                    # A failed root acquisition is still an original prompt
                    # attempt and must survive in timing/coverage denominators.
                    self.current_row = {"prompt": len(self.root_rows), "msg": int(msg), "root": False,
                        "native_snapshot_sha256": None, "searches": [], "complete": False, "sent_verified": False,
                        "turn": self._client.result.turns, "viewer": self.seat,
                        "timing": {"root_seconds": 0., "sample_seconds": 0., "rollout_seconds": 0.,
                                   "update_seconds": 0., "total_seconds": 0.}}
                    self.root_rows.append(self.current_row)
                if self.config.budget_law in ANYTIME_LAWS:
                    client = self._client
                    from .room_clock import verified_room_seconds
                    room_seconds = verified_room_seconds(client)
                    deferred_clock = getattr(client, '_clock_prompt_origin', None) is not None
                    budget_now = time.monotonic() if deferred_clock or self.final_room_clock_enabled else prompt_started
                    seat = client.result.our_player
                    final_allocation = None
                    if self.final_room_clock_enabled:
                        final_allocation = self._final_public_deadlines(prompt_started, budget_now)
                    else:
                        self._search_deadline, self._response_deadline, self._clock_record = A.public_deadlines(
                            now=budget_now, clock_received=getattr(client, "clock_received", {}).get(seat),
                            clock_left=client.time_left.get(seat), room_seconds=room_seconds, search_seconds=self.config.seconds,
                            clock_reserve=self.config.clock_reserve, clock_share=self.config.clock_share,
                            response_margin=self.config.response_margin, finalize_seconds=self.config.finalize_seconds,
                            prompt_cap=self.config._prompt_cap(),
                            **({"total_seconds": self.config.total_seconds} if self.config.total_seconds is not None else {}),
                            **({"prompt_elapsed": budget_now - prompt_started} if deferred_clock else {}))
                    if self.search_gate is not None:
                        self.search_gate.begin_prompt(turn=self._client.result.turns, started=prompt_started,
                            search_deadline=self._search_deadline, response_deadline=self._response_deadline,
                            **({'optional_response_deadline': final_allocation.response_deadline}
                               if final_allocation is not None and final_allocation.allow_optional else {}))
                        self.current_row['on_demand'] = {
                            'started': prompt_started, 'public_search_deadline': self._search_deadline,
                            'public_response_deadline': self._response_deadline,
                            'decisions': [], 'optional_seconds': 0.}
                self.context = RootMenuBinding._clone_context(self._client.ctx)
                # the particle certificate validates the whole engine prompt,
                # including END alongside BATTLE/MAIN2. The ordinary client's
                # legacy Python parse may omit that phase row. This private
                # certificate context is NOT the native model menu: neither
                # the live context nor the service's trained action law changes.
                self.context.full_phase_menu = True
                message = self.owner.advance()
                if message is None:
                    raise SearchPolicyError("the client's prompt has no native follower root")
                self.prompt_packet = packet
                lease_seconds = self.config.seconds if self._response_deadline is None \
                    else self._response_deadline - time.monotonic()
                if lease_seconds <= 0:
                    raise SearchPolicyError("root acquisition exhausted the public response clock")
                self.capture_root = self.owner.capture_root(max_seconds=lease_seconds)
                self.root = self.capture_root.__enter__()
                row = {"prompt": len(self.root_rows) - int(tracked), "msg": int(msg), "root": True,
                                    "native_snapshot_sha256": self.root.snapshot.digest,
                                    "searches": [], "complete": False, "sent_verified": False,
                                    "turn": self._client.result.turns, "viewer": self.seat,
                                    "timing": {"root_seconds": time.monotonic() - prompt_started,
                                               "sample_seconds": 0.0, "rollout_seconds": 0.0,
                                               "update_seconds": 0.0, "total_seconds": 0.0}}
                if self._clock_record is not None:
                    row["public_budget"] = dict(self._clock_record)
                if self._final_clock_record is not None:
                    row['final_room_budget'] = self._final_clock_record
                if tracked:
                    self.current_row.update(row)
                else:
                    self.current_row = row
                    self.root_rows.append(self.current_row)
        except Exception as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
            if msg in SELECT_MESSAGES and self.provider_identity.get("provider") in (REPLAY_FILTERED_PROVIDER, *CURRENT_PROVIDERS) \
                    and self.current_row is not None and self.current_row.get("msg") == int(msg):
                self.current_row["error"] = f"{type(exc).__name__}: {exc}"
                elapsed = time.monotonic() - prompt_started
                self.current_row["timing"].update(root_seconds=elapsed, total_seconds=elapsed)
            raise DuelError("search follower failed: " + str(exc)) from exc

    def _parse(self, msg, body, ctx):
        from .actions import SelectResult, parse_select
        from .client import DuelError
        from . import constants as C
        standard = parse_select(msg, body, ctx)
        try:
            if self.root is None or self.prompt_packet != bytes([msg]) + bytes(body) or not self.pending \
                    or self.pending[-1] != (int(msg), bytes(body)):
                raise SearchPolicyError("decision arrived without its complete owned client prompt")
            reply = self._call({"op": "prompt", "session": self.session,
                                "messages": [[m, b.hex()] for m, b in self.pending]})
            self.pending = []
            if self.search_gate is not None:
                clear = None
                if self.config.opponent_clear_profile is not None:
                    clear = OC.extract(self._client, self.capture)
                    self.current_row['opponent_clear'] = clear
                if self.config.opponent_known_profile is not None:
                    clear = OK.extract(self._client, self.capture)
                    self.current_row['opponent_known'] = clear
                self.current_row['on_demand']['public'] = (
                    None if clear is not None and not clear['allow_search'] else
                    G.public_combat_witness(self._client, standard))
            response_index = len(self.responses)
            if reply.get("forced"):
                self.forced += 1
                self.decisions.append({"forced": True, "msg": int(msg), "response_index": response_index,
                                       "turn": self._client.result.turns, "viewer": self.seat})
            decision_started = self._prompt_started
            for subdecision in range(4096):
                if "pending" not in reply:
                    break
                pending = reply["pending"]
                sample_seconds = 0.
                gate = None
                if self.search_gate is not None:
                    gate_now = time.monotonic()
                    public = self.current_row['on_demand']['public']
                    gate = self.search_gate.decide(logits=pending['logits'], rows=pending['rows'],
                        hint=None if public is None else G.hint_from_witness(public), now=gate_now,
                        **({'opponent_clear': self.current_row['opponent_clear']}
                           if self.config.opponent_clear_profile is not None else {}),
                        **({'opponent_known': self.current_row['opponent_known']}
                           if self.config.opponent_known_profile is not None else {}))
                    self.current_row['on_demand']['decisions'].append({
                        'at': gate_now, 'obs_sha256': pending['obs_sha256'],
                        'gate': G.decision_wire(gate), 'optional_seconds': 0.})
                    if gate.search:
                        self._search_deadline = min(self._search_deadline, gate.search_deadline)
                        self._response_deadline = min(self._response_deadline, gate.response_deadline)
                if pending["rows"] == 1:
                    chosen, probability = 0, 1.0
                    self.current_row["searches"].append({"singleton": True, "complete": True,
                                                        "obs_sha256": pending["obs_sha256"], "chosen": 0})
                else:
                    replay_filtered = self.provider_identity.get("provider") == REPLAY_FILTERED_PROVIDER
                    current_provider = is_current_provider(self.provider_identity.get("provider"))
                    if self.config.bank_unknown_law == F.LAW or replay_filtered or current_provider:
                        self.current_row.setdefault("non_singleton_attempts", []).append({
                            "subdecision": subdecision, "rows": pending["rows"], "obs_sha256": pending["obs_sha256"]})
                    if self.particles.identity != self.provider_identity:
                        raise SearchPolicyError("the particle provider changed after registration")
                    _, count = allocation(pending["logits"], self.config)
                    seed = self.seed * 1_000_003 + len(self.decisions)
                    sample_started = time.monotonic()
                    arguments = {}
                    if self.provider_identity.get("provider") in ("own-pending-public-capacity/v1", "own-pending-natural-conditioned-capacity/v1", REPLAY_FILTERED_PROVIDER, *CURRENT_PROVIDERS):
                        arguments["deadline"] = self._search_deadline
                    from .conditioned_particles import TypedBankUnknown
                    from .current_particles import CurrentRootBudgetExpired
                    admission_started = None
                    if gate is not None and not gate.search:
                        chosen, record = gate_prior_root(self.root, pending, config=self.config, seed=seed, gate=gate)
                    else:
                        try:
                            bank = self.particles.propose(self.root, self._client, count=count, seed=seed, context=self.context,
                                                          extra_origins=self.banish.extra_origin_slots(1 - self.seat),
                                                          call=self._call, session=self.session, pending=pending, **arguments)
                            sample_seconds = time.monotonic() - sample_started
                            if bank.law != self.provider_identity.get("law"):
                                raise SearchPolicyError("the actual particle law differs from its registration")
                            admission_started = time.monotonic()
                            chosen, record = plan_root(self.root, pending, self.session, bank, self.public_opponent_recipe,
                                call=self._call, config=self.config, seed=seed, rng=self.rng,
                                deadline=self._search_deadline,
                                on_admission=lambda audit: self.current_row.setdefault(
                                    "replay_filtered_admissions" if replay_filtered else "conditional_admissions", []).append(audit))
                        except CurrentRootBudgetExpired as exc:
                            if not is_current_provider(self.provider_identity.get("provider")) or type(exc) is not CurrentRootBudgetExpired:
                                raise
                            elapsed = time.monotonic() - sample_started
                            chosen, record = current_budget_result(self.root, pending, config=self.config, seed=seed,
                                deadline=self._search_deadline, phase=exc.phase,
                                particle_law=self.provider_identity["law"])
                            sample_seconds = 0.
                            record["phase_seconds"]["admission"] = elapsed
                        except TypedBankUnknown as exc:
                            if self.config.bank_unknown_law != F.LAW or type(exc) is not TypedBankUnknown:
                                raise
                            # Its producer has already closed every detached
                            # driver/branch and verified the original root. Keep
                            # the failed attempt even if a later live check fails.
                            self.current_row.setdefault("conditional_admissions", []).append(exc.admission)
                            if exc.fallback_record["phase"] != ("proposal" if admission_started is None else "admission"):
                                raise SearchPolicyError("bank-unknown capability belongs to another construction phase")
                            if admission_started is None:
                                sample_seconds = time.monotonic() - sample_started
                                admission_seconds = 0.
                            else:
                                admission_seconds = time.monotonic() - admission_started
                            chosen, record = bank_unknown_root(self.root, pending, exc, config=self.config,
                                                               seed=seed, particle_identity=self.provider_identity)
                            record["phase_seconds"]["admission"] = admission_seconds
                        finally:
                            if gate is not None:
                                optional_seconds = time.monotonic() - sample_started
                                self.current_row['on_demand']['optional_seconds'] += optional_seconds
                                self.current_row['on_demand']['decisions'][-1]['optional_seconds'] = optional_seconds
                    self.current_row["timing"]["sample_seconds"] += sample_seconds
                    phases = record.get("phase_seconds", {})
                    self.current_row["timing"]["sample_seconds"] += phases.get("admission", 0.)
                    self.current_row["timing"]["rollout_seconds"] += phases.get("rollout", 0.)
                    self.current_row["timing"]["update_seconds"] += phases.get("update", 0.)
                    self.current_row["searches"].append(record)
                    probability = record["policy"][chosen]
                if gate is not None:
                    self.current_row['searches'][-1]['gate_decision'] = G.decision_wire(gate)
                self.decisions.append({"forced": False, "msg": pending["msg"], "rows": pending["rows"],
                                       "chosen": chosen, "p_chosen": probability, "value": pending["value"],
                                       "wdl": pending["wdl"], "logits": pending["logits"],
                                       "obs_sha256": pending["obs_sha256"], "response_index": response_index,
                                       "subdecision": subdecision, "turn": self._client.result.turns, "viewer": self.seat})
                reply = self._call({"op": "commit", "session": self.session, "index": chosen})
                if self.config.budget_law in ANYTIME_LAWS and pending["rows"] > 1:
                    phases = record["phase_seconds"]
                    bank_unknown = record.get("fallback") == "bank_unknown"
                    stripe = record.get("anytime")
                    record["timing"] = {"game": self.session, "seat": self.seat,
                        "turn": self._client.result.turns, "response_index": response_index, "subdecision": subdecision,
                        "root_seconds": self.current_row["timing"]["root_seconds"] if subdecision == 0 else 0.,
                        "sample_seconds": sample_seconds + phases["admission"],
                        "rollout_seconds": phases["rollout"], "update_seconds": phases["update"],
                        "total_seconds": time.monotonic() - decision_started,
                        "completed_stripes": 0 if bank_unknown else stripe["completed_stripes"],
                        "budget_zero_search": False if bank_unknown else stripe["budget_zero_search"]}
                    if self.config.bank_unknown_law == F.LAW:
                        record["timing"]["bank_unknown"] = bank_unknown
                decision_started = time.monotonic()
            else:
                raise SearchPolicyError("the root never completed its wire response")
            response = bytes.fromhex(reply["response"])
            self.responses.append(response.hex())
            self.final_response = response
            self.current_row["complete"] = True
            self.current_row["timing"]["total_seconds"] = time.monotonic() - self._prompt_started
            if self._response_deadline is not None and time.monotonic() >= self._response_deadline:
                raise SearchPolicyError("real response exceeded its reserved public-clock deadline")
        except Exception as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
            if self.current_row is not None:
                self.current_row["error"] = f"{type(exc).__name__}: {exc}"
                self.current_row["timing"]["total_seconds"] = time.monotonic() - self._prompt_started
                from .conditioned_particles import ConditionalAdmissionError
                if isinstance(exc, ConditionalAdmissionError):
                    self.current_row.setdefault("conditional_admissions", []).append(exc.admission)
                if hasattr(exc, "replay_filter_audit"):
                    self.current_row.setdefault("replay_filtered_admissions", []).append(exc.replay_filter_audit)
            raise DuelError("option-A search failed: " + str(exc)) from exc
        player = body[1] if msg == C.MSG_SELECT_SUM else body[0]
        return SelectResult(msg, player, auto_response=response, note="option-A search", cards=standard.cards)

    def _commit_sent(self):
        pending = self.sent[self.committed:]
        if not pending:
            return
        if len(pending) != 1 or self.root is None or self.capture_root is None \
                or bytes(pending[0]) != self.final_response:
            raise SearchPolicyError("the client sent a response not certified by its live search root")
        root, capture, data, prompt, row = self.root, self.capture_root, bytes(pending[0]), self.prompt_packet, self.current_row
        self.root = self.capture_root = None
        capture.__exit__(None, None, None)
        self.owner.commit_real_response(root, data, send=lambda _: None,
                                        validate_original=lambda raw, path, answer:
                                        raw == prompt and not path and answer == data)
        self.committed += 1
        row["sent_verified"] = True
        self._search_deadline = self._response_deadline = self._prompt_started = self._clock_record = None
        self._final_clock_record = None

    def on_duel_end(self, result):
        try:
            self._commit_sent()
            if self.external_terminal is not None:
                from .agent_external_terminal import check
                check(self.report(), result.as_dict(), allowed=self.allow_server_surrender,
                      allow_network=self.allow_server_network_terminal)
            if self.owner is not None and not self.failures and self.external_terminal is None:
                self.owner.advance()
        except Exception as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
        finally:
            try:
                if self.capture_root is not None:
                    self.capture_root.__exit__(None, None, None)
            finally:
                self.root = self.capture_root = None
                try:
                    if self.owner is not None:
                        try:
                            self.final_follower_stats = follower_stats_wire(self.owner.follower.stats)
                        except Exception as exc:
                            self.failures.append(f"{type(exc).__name__}: {exc}")
                        try:
                            self.owner.close()
                        except Exception as exc:
                            self.failures.append(f"{type(exc).__name__}: {exc}")
                finally:
                    self.owner = None
                    super().on_duel_end(result)
                    if self.search_gate is not None and self.search_gate.prompt is not None:
                        finished = self.search_gate.finish_prompt(
                            now=time.monotonic(), delivered=False, cleanup_complete=False)
                        self.current_row['on_demand']['finish'] = finished
                        self.current_row['on_demand']['non_search_seconds'] = \
                            finished['prompt_seconds'] - self.current_row['on_demand']['optional_seconds']

    def report(self):
        report = {**super().report(), "schema": self.report_schema, "search_identity": self.search_identity,
                "roots": list(self.root_rows), "sent_committed": self.committed,
                "follower_stats": self.final_follower_stats}
        if self.external_terminal is not None:
            report['external_terminal'] = json.loads(json.dumps(self.external_terminal))
        if self.provider_identity.get("provider") == "own-pending-natural-conditioned-capacity/v1":
            report["conditional_sampling"] = conditional_sampling_summary(self.root_rows)
        if self.provider_identity.get("provider") == REPLAY_FILTERED_PROVIDER:
            report["replay_filtered_sampling"] = replay_filtered_summary(self.root_rows)
        if is_current_provider(self.provider_identity.get("provider")):
            report["current_root_sampling"] = current_root_summary(self.root_rows)
        if self.search_gate is not None:
            report['on_demand'] = G.summary(self.root_rows, config=self.config.on_demand)
        if self.config.final_room_profile is not None:
            report['final_room_clocks'] = {'schema': FB.CLOCK_SCHEMA, 'profile': dict(FB.PROFILE),
                'game_id': self.session, 'player': getattr(self, 'seat', None), 'bound_at': self._clock_bound_at,
                'clock_delivery': getattr(self, '_clock_delivery', None), 'receipts': list(self._room_clock_receipts)}
        if self.config.opponent_clear_profile is not None:
            report['opponent_clear_profile'] = OC.validate_profile(OC.PROFILE)
        if self.config.opponent_known_profile is not None:
            report['opponent_known_profile'] = OK.validate_profile(OK.PROFILE)
        if self.config.bank_unknown_law == F.LAW:
            report["bank_fallbacks"] = F.decision_summary(self.root_rows)
        if self.config.budget_law in ANYTIME_LAWS:
            rows = [search["timing"] for root in self.root_rows for search in root["searches"] if "timing" in search]
            totals = {}
            for root in self.root_rows:
                key = root["turn"], root["viewer"]
                totals[key] = totals.get(key, 0.) + root["timing"]["total_seconds"]
            report["timing"] = {"model_decisions": A.timing_summary(rows, law=self.config.budget_law),
                                "all_wire_prompts": len(self.root_rows),
                                "all_prompt_worst_turn_seconds": max(totals.values(), default=None)}
        return report


def check_root_record(record, decision, search_identity, *, native_snapshot_sha256=None):
    """Independently check allocation and MMD, including unsearched prior mass."""
    if record.get("singleton") is True:
        if decision["rows"] != 1 or decision["chosen"] != 0 or record.get("chosen") != 0 \
                or record.get("complete") is not True or record.get("obs_sha256") != decision["obs_sha256"]:
            raise ValueError("invalid singleton search root")
        return
    config = SearchConfig(**search_identity["settings"])
    candidates, count = allocation(decision["logits"], config)
    if record.get('schema') == G.ROOT_SCHEMA:
        chosen, policy = A.greedy_without_search(decision['logits'])
        settings = search_settings(config)
        gate = record.get('gate_decision', {})
        skip_reasons = ('confident', 'no_search_budget', 'previous_overrun')
        if config.opponent_clear_profile is not None:
            skip_reasons += ('opponent_hand_nonempty', 'opponent_facedown_field',
                             'opponent_field_unknown', 'opponent_info_unknown')
        if config.opponent_known_profile is not None:
            skip_reasons += tuple(OK.BLOCKED_REASONS)
        if config.on_demand is None or record.get('gate_skipped') is not True or gate.get('search') is not False \
                or gate.get('reason') not in skip_reasons \
                or gate.get('confidence') != asdict(G.policy_confidence(decision['logits'], rows=decision['rows'])) \
                or record.get('training_eligible') is not False or record.get('complete') is not True \
                or record.get('settings') != settings \
                or record.get('settings_sha256') != hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest() \
                or record.get('obs_sha256') != decision['obs_sha256'] or record.get('rows') != decision['rows'] \
                or record.get('candidate_rows') != list(candidates) or record.get('lines_per_row') != count \
                or record.get('root_snapshot_sha256') != native_snapshot_sha256 \
                or record.get('chosen') != chosen or decision['chosen'] != chosen \
                or record.get('q') != [None] * decision['rows'] or record.get('actual_bank_count') != 0 \
                or record.get('requested_bank_count') != count or record.get('particle_proofs') != [] \
                or record.get('particle_weights') != [] or record.get('anytime') != MULTI.empty_stripes(decision['rows']) \
                or record.get('stats') != record['anytime']['stats'] \
                or record.get('phase_seconds') != {'admission': 0., 'rollout': 0., 'update': 0.} \
                or any(record.get(k) != v for k, v in policy.items()) \
                or decision.get('p_chosen') != policy['policy'][chosen] \
                or any(k in record for k in ('fallback', 'current_budget_empty', 'world_sha256', 'particle_law')):
            raise ValueError("on-demand prior fallback changed its raw policy, source root, allocation or reason")
        return
    if record.get("schema") == F.ROOT_SCHEMA:
        settings = record.get("settings")
        if not isinstance(settings, dict) or search_settings(SearchConfig(**settings)) != search_settings(config):
            raise ValueError("bank fallback differs from registered settings")
        return F.check_root_record(record, decision, {**search_identity, "settings": settings},
                                  root_snapshot_sha256=native_snapshot_sha256, candidates=candidates, count=count)
    if "fallback" in record or "fallback_record" in record:
        raise ValueError("a normal search record cannot hide a bank fallback")
    completed = count
    zero = False
    filtered = search_identity["particles"].get("provider") == REPLAY_FILTERED_PROVIDER
    current = is_current_provider(search_identity["particles"].get("provider"))
    if config.budget_law == MULTI.LAW and not current:
        raise ValueError("multiplexed root record requires its registered current-root provider")
    empty_current = current and record.get("current_budget_empty") is True
    actual_count = record.get("actual_bank_count") if filtered or empty_current else count
    if empty_current and (actual_count != 0 or record.get("requested_bank_count") != count
            or record.get("proposal_budget_exhausted") is not True
                or record.get("budget_phase") not in ("before_proposals", "before_materialization", "materialization_or_view",
                                                     "public_hand_assignment")):
        raise ValueError("current-root zero-proposal budget accounting differs")
    if filtered and (type(actual_count) is not int or not 0 <= actual_count <= count \
            or record.get("requested_bank_count") != count or record.get("particle_target_cap") != config.particles \
            or record.get("replay_filtered_empty") is not (actual_count == 0)):
        raise ValueError("replay-filtered root has inconsistent actual bank accounting")
    if config.budget_law in ANYTIME_LAWS:
        if type(record.get("particle_seed")) is not int:
            raise ValueError("balanced search lacks its fixed seed")
        check = MULTI.check_stripes if config.budget_law == MULTI.LAW else A.check_stripes
        completed = check(record.get("anytime"), rows=len(candidates), planned=actual_count, seed=record["particle_seed"])
        zero = completed == 0
        if record.get("stats") != record["anytime"]["stats"]:
            raise ValueError("root and stripe resource counts differ")
    elif "anytime" in record:
        raise ValueError("strict complete-bank search cannot silently use a partial-budget law")
    registered_settings = search_settings(config)
    serialized_settings = record.get("settings")
    if not isinstance(serialized_settings, dict) or search_settings(SearchConfig(**serialized_settings)) != registered_settings:
        raise ValueError("search root configuration differs from its registered settings")
    settings_sha = hashlib.sha256(json.dumps(serialized_settings, sort_keys=True).encode()).hexdigest()
    if record.get("schema") != (current_root_schema(config) if current else SCHEMA) or record.get("training_eligible") is not False \
            or record.get("settings_sha256") != settings_sha \
            or record.get("obs_sha256") != decision["obs_sha256"] or record.get("rows") != decision["rows"] \
            or record.get("candidate_rows") != list(candidates) or record.get("lines_per_row") != count \
            or record.get("particle_law") != search_identity["particles"]["law"] \
            or record.get("wide_menu") != (decision["rows"] > config.rollouts) or record.get("complete") is not True \
            or record.get("chosen") != decision["chosen"] or record.get("stats", {}).get("void") != 0 \
            or record.get("stats", {}).get("lines") != len(candidates) * completed:
        raise ValueError("search root differs from its registered rule, menu or complete equal budget")
    if "world_rpc" in search_identity["particles"] and not empty_current:
        world_sha = record.get("world_sha256")
        if record.get("world_obs_sha256") != decision["obs_sha256"] or not isinstance(world_sha, str) \
                or len(world_sha) != 64 or any(c not in "0123456789abcdef" for c in world_sha):
            raise ValueError("search root lacks its pending-bound public World evidence")
    weights, proofs = record.get("particle_weights", []), record.get("particle_proofs", [])
    if len(weights) != actual_count or len(proofs) != actual_count or any(
            type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("search root lacks its complete fixed positive-weight bank")
    if current:
        from .current_particles import SCHEMA as CURRENT_PROOF
        provider = search_identity["particles"].get("provider")
        expected_particles = CURRENT.PARTICLE_IDENTITY if provider == CURRENT.PROVIDER else \
            CURRENT.COUNT_PARTICLE_IDENTITY if provider == CURRENT.COUNT_PROVIDER else {
            **CURRENT.AR_PARTICLE_IDENTITY, "ar_head": search_identity["particles"].get("ar_head")}
        if search_identity["particles"] != expected_particles \
                or search_identity.get("schema") != current_identity_schema(config) \
                or search_identity.get("law") != "current-root-in-place/v1" \
                or search_identity.get("memory_law") != CURRENT.MEMORY_LAW \
                or record.get("memory_law") != CURRENT.MEMORY_LAW or record.get("opening_replayed") is not False \
                or record.get("own_deck_order_realized") is not (not empty_current) \
                or type(decision.get("viewer")) is not int or decision["viewer"] not in (0, 1) \
                or not isinstance(record.get("root_id"), str) or not record["root_id"] \
                or not isinstance(native_snapshot_sha256, str) or record.get("root_snapshot_sha256") != native_snapshot_sha256 \
                or any(not math.isclose(weight, 1 / count, rel_tol=1e-12, abs_tol=1e-14) for weight in weights) \
                or any(key in record for key in ("causal_bank", "conditional_admission", "replay_filtered_admission")):
            raise ValueError("current-root record changed its original root, registered bank or memory approximation")
        direct_bank = None
        for index, proof in enumerate(proofs):
            if proof.get("schema") != CURRENT_PROOF or proof.get("index") != index \
                    or proof.get("root_id") != record["root_id"] or proof.get("root_hash") != native_snapshot_sha256 \
                    or proof.get("proposal_seed") != record["particle_seed"] \
                    or proof.get("proposal_law") != search_identity["particles"]["law"] \
                    or proof.get("memory_law") != CURRENT.MEMORY_LAW or proof.get("opening_replayed") is not False \
                    or proof.get("own_deck_order_realized") is not True or proof.get("training_eligible") is not False \
                    or proof.get("viewer") != 1 - decision["viewer"] \
                    or proof.get("native_adapter") != "fixed-sky-all-turns/v1" \
                    or proof.get("root_response_law") != CURRENT.ROOT_RESPONSE_LAW \
                    or any(not isinstance(proof.get(k), str) or len(proof[k]) != 64 \
                           or any(c not in "0123456789abcdef" for c in proof[k])
                           for k in ("root_hash", "hypothesis_hash", "view_sha256", "adapter_binding",
                                     "root_menu_binding_sha256")):
                raise ValueError("current-root proof differs from its original bound particle")
            if provider == CURRENT.AR_PROVIDER:
                direct = proof.get("direct_ar")
                if type(direct) is not dict or set(direct) != {"schema", "bank_sha256", "feature_sha256", "head",
                        "obs_sha256", "seed", "count", "specification_sha256", "search_admission", "teacher_channels",
                        "hand_scope", "hand_physical_law", "max_nodes", "proposal_nodes"} \
                        or direct.get("schema") != CURRENT.AR_NATIVE_PROOF \
                        or direct.get("head") != search_identity["particles"].get("ar_head") \
                        or direct.get("obs_sha256") != record.get("obs_sha256") \
                        or direct.get("seed") != record.get("particle_seed") \
                        or direct.get("count") != actual_count \
                        or direct.get("search_admission") is not False or direct.get("teacher_channels") is not False \
                        or any(type(direct.get(key)) is not str or len(direct[key]) != 64 \
                               or any(c not in "0123456789abcdef" for c in direct[key])
                               for key in ("bank_sha256", "feature_sha256", "specification_sha256")):
                    raise ValueError("direct AR particle proof differs from its original pending observation and bank")
                from ..agent.search.belief_hand_scope import PublicHandScope, MAX_NODES, RNG_LAW
                scope = PublicHandScope(direct["hand_scope"])
                if scope.to_dict()["obs_sha256"] != record.get("obs_sha256") \
                        or scope.to_dict()["world_sha256"] != record.get("world_sha256") \
                        or direct["hand_physical_law"] != CURRENT.AR_HAND_ASSIGNMENT \
                        or direct["max_nodes"] != MAX_NODES or type(direct["proposal_nodes"]) is not int \
                        or not 0 <= direct["proposal_nodes"] < MAX_NODES:
                    raise ValueError("AR physical hand scope changed its original pending or node budget")
                hand = proof.get("hand_assignment")
                fields = {"law", "rng_law", "scope_sha256", "support_count", "hand_sha256", "uid_law",
                          "uid_order_sha256", "root_response_law", "root_menu_binding_sha256", "node_budget",
                          "readback_verified", "menu_orbit_sha256", "menu_orbit_generators",
                          "uid_matching_count", "joint_hand_probability_denominator"}
                if type(hand) is not dict or set(hand) != fields \
                        or hand["law"] != CURRENT.AR_HAND_ASSIGNMENT or hand["rng_law"] != RNG_LAW \
                        or hand["scope_sha256"] != scope.sha256 or hand["readback_verified"] is not True \
                        or hand["uid_law"] != CURRENT.AR_HAND_UID_LAW \
                        or hand["root_response_law"] != "current-hand-coordinates-to-unchanged-pending-vector/v1" \
                        or hand["root_menu_binding_sha256"] != proof["root_menu_binding_sha256"] \
                        or type(hand["menu_orbit_generators"]) is not int \
                        or hand["menu_orbit_generators"] != max(0, len(scope.free_group)-1) \
                        or any(type(hand[key]) is not str or len(hand[key]) != 64
                               or any(c not in "0123456789abcdef" for c in hand[key])
                               for key in ("hand_sha256", "uid_order_sha256", "menu_orbit_sha256")):
                    raise ValueError("AR hand UID realization lacks its complete conditional-uniform readback")
                support = hand["support_count"]
                matchings = hand["uid_matching_count"]
                denominator = hand["joint_hand_probability_denominator"]
                nodes = hand["node_budget"]
                if type(support) is not str or not 1 <= len(support) <= 200 or not support.isascii() \
                        or not support.isdigit() or support.startswith('0') \
                        or type(nodes) is not dict or set(nodes) != {"max_nodes", "proposal_nodes", "total_nodes"} \
                        or nodes["max_nodes"] != MAX_NODES or nodes["proposal_nodes"] != direct["proposal_nodes"] \
                        or type(nodes["total_nodes"]) is not int \
                        or not nodes["proposal_nodes"] <= nodes["total_nodes"] < MAX_NODES:
                    raise ValueError("AR physical hand probability support or remaining work budget differs")
                if any(type(value) is not str or not 1 <= len(value) <= limit or not value.isascii()
                       or not value.isdigit() or value.startswith('0')
                       for value, limit in ((matchings, 200), (denominator, 400))) \
                        or int(denominator) != int(support)*int(matchings):
                    raise ValueError("AR hand code/UID conditional probabilities were conflated or changed")
                if direct_bank is not None and direct != direct_bank:
                    raise ValueError("current-root AR particles were drawn from more than one original pending bank")
                direct_bank = direct
            elif provider == CURRENT.COUNT_PROVIDER:
                from .count_head_bank import FACTOR, RESAMPLE_LAW, WEIGHED, SMOOTHING, _sha as count_sha
                bank = proof.get("count_head")
                if type(bank) is not dict or set(bank) != {"schema", "law", "obs_sha256", "seed", "count", "proposals",
                        "factor", "smoothing", "weighed_locations", "resample_law", "selected", "effective_sample_size",
                        "distinct", "belief", "belief_sha256", "search_admission"} \
                        or bank["schema"] != CURRENT.COUNT_NATIVE_PROOF or bank["law"] != CURRENT.COUNT_BANK_LAW \
                        or bank["obs_sha256"] != record.get("obs_sha256") or bank["seed"] != record.get("particle_seed") \
                        or bank["count"] != actual_count or bank["factor"] != FACTOR or bank["smoothing"] != SMOOTHING \
                        or bank["proposals"] != actual_count * FACTOR \
                        or bank["weighed_locations"] != list(WEIGHED) or bank["resample_law"] != RESAMPLE_LAW \
                        or bank["search_admission"] is not False or bank["belief_sha256"] != count_sha(bank["belief"]) \
                        or type(bank["selected"]) is not list or len(bank["selected"]) != actual_count \
                        or any(type(i) is not int or not 0 <= i < bank["proposals"] for i in bank["selected"]) \
                        or bank["selected"] != sorted(bank["selected"]) or bank["distinct"] != len(set(bank["selected"])) \
                        or "direct_ar" in proof or "hand_assignment" in proof:
                    raise ValueError("count-head particle proof differs from its original pending bank")
                if direct_bank is not None and bank != direct_bank:
                    raise ValueError("count-head particles were drawn from more than one original pending bank")
                direct_bank = bank
            elif "direct_ar" in proof or "hand_assignment" in proof or "count_head" in proof:
                raise ValueError("uniform current-root proof cannot carry a direct AR or count-head bank")
        resources = record.get("rollout_resources", {})
        expected_resources = resource_record(config)
        if search_identity.get("rollout_resources") != current_resources(config) \
                or any(resources.get(k) != v for k,v in expected_resources.items() if not k.startswith("peak_") and k != "chunks") \
                or any(type(resources.get(key)) is not int or resources[key] < 0
                       for key in ("peak_line_sessions", "peak_base_sessions", "chunks")) \
                or resources["peak_line_sessions"] > 2 * expected_resources["max_live_lines"] \
                or resources["peak_base_sessions"] > expected_resources.get("max_active_stripes", 1):
            raise ValueError("current-root rollout exceeded its registered live resource bounds")
        if config.budget_law == MULTI.LAW:
            from .agent_stripe_contract import check_registration
            check_registration(search_identity.get("stripe_contract"), config)
            audit_resources = record["anytime"]["resources"]
            if any(audit_resources.get(k) != v for k,v in MULTI.DEFAULT_LIMITS.items()) \
                    or resources["peak_line_sessions"] > 2 * audit_resources["reserved_live_lines_peak"] \
                    or resources["peak_base_sessions"] != audit_resources["peak_active_stripes"]:
                raise ValueError("current-root live resource counts differ from their multiplexed reservations")
        elif "stripe_contract" in search_identity:
            raise ValueError("legacy current-root identity cannot claim a multiplexed contract")
    elif filtered:
        from .replay_filtered_particles import check_filtered_admission
        from .causal_proof import sha256
        audit = record.get("replay_filtered_admission", {})
        resources = record.get("rollout_resources", {})
        if search_identity.get("rollout_resources") != REPLAY_FILTERED_RESOURCES \
                or resources.get("law") != BOUNDED_LANES_LAW or resources.get("max_live_lines") != REPLAY_FILTERED_LANES \
                or any(type(resources.get(key)) is not int or resources[key] < 0
                       for key in ("peak_line_sessions", "peak_base_sessions", "chunks")) \
                or resources["peak_line_sessions"] > 2 * REPLAY_FILTERED_LANES or resources["peak_base_sessions"] > 1:
            raise ValueError("replay-filtered rollout exceeded its registered live resource bounds")
        if "conditional_admission" in record or "causal_bank" in record or any(weight != 1.0 for weight in weights):
            raise ValueError("replay-filtered bank must retain its actual samples with equal MC weights")
        check_filtered_admission(audit, requested_count=count, proposal_seed=record.get("particle_seed"),
            world_sha256=record.get("world_sha256"), obs_sha256=decision["obs_sha256"],
            source_sha256=search_identity["particles"].get("producer_source_sha256",
                                                        search_identity["particles"].get("proposal_source_sha256")),
            expected_source_bundle=search_identity["particles"].get("source_bundle"))
        selected = [audit["attempts"][index] for index in audit["accepted_ordinals"]]
        if len(selected) != actual_count or [sha256(row["proof"]) for row in selected] != [sha256(proof) for proof in proofs]:
            raise ValueError("rollouts differ from the successfully replayed common bank")
    elif search_identity["particles"].get("provider") == "own-pending-natural-conditioned-capacity/v1":
        from .causal_proof import producer_contract, verify_natural_root_proof, sha256
        from .causal_rejection import check_conditioned_admission
        audit = record.get("conditional_admission", {})
        if search_identity.get("producer") != producer_contract() or "causal_bank" in record \
                or any(weight != 1.0 for weight in weights):
            raise ValueError("conditioned search lacks its registered producer or fixed uniform bank")
        check_conditioned_admission(audit, requested_count=count, proposal_seed=record.get("particle_seed"),
            world_sha256=record.get("world_sha256"), obs_sha256=decision["obs_sha256"],
            history_sha256=audit.get("history_sha256"), source_sha256=audit.get("source_sha256"),
            entropy_profile_sha256=search_identity["particles"].get("entropy_profile_sha256"), require_complete=True)
        selected = [audit["proposals"][index] for index in audit["accepted_ordinals"]]
        if len(selected) != count or [sha256(row["proof"]) for row in selected] != [sha256(proof) for proof in proofs]:
            raise ValueError("rollouts did not use exactly the admitted proposal-prefix bank")
        for proof, row in zip(proofs, selected):
            verify_natural_root_proof(proof, hypothesis_sha256=row["layout_sha256"],
                history_sha256=audit["history_sha256"], source_sha256=audit["source_sha256"])
    elif search_identity["particles"].get("provider") == "own-pending-public-capacity/v1":
        from .causal_proof import producer_contract, verify_natural_root_proof
        bank = record.get("causal_bank", {})
        if "conditional_admission" in record or search_identity.get("producer") != producer_contract() \
                or len(bank.get("layout_sha256", [])) != count:
            raise ValueError("natural root record lacks its explicitly registered producer/bank")
        sampler = bank.get("sampler", {})
        if sampler.get("schema") != "mirrorforce_causal_root_sampler/v1" \
                or sampler.get("law") != record["particle_law"] or sampler.get("history_sha256") != bank.get("history_sha256") \
                or sampler.get("world_sha256") != record.get("world_sha256") \
                or type(sampler.get("distinct_root_layouts")) is not int or sampler["distinct_root_layouts"] < 1 \
                or sampler.get("historical_witness_multiplicity_counted") is not False \
                or sampler.get("native_replay_admitted") is not False \
                or sampler.get("proposal_retries") != 0 or any(weight != 1.0 for weight in weights):
            raise ValueError("natural causal bank differs from its declared distinct-root uniform counting law")
        for proof, layout_sha in zip(proofs, bank["layout_sha256"]):
            verify_natural_root_proof(proof, hypothesis_sha256=layout_sha, history_sha256=bank["history_sha256"],
                                      source_sha256=bank["source_sha256"])
    elif "causal_bank" in record or "conditional_admission" in record or any(p.get("schema") != "mirrorforce_synthetic_root/v1" or p.get("public_stream_equal") is not True
                   or p.get("root_menu_equal") is not True or p.get("training_eligible") is not False
                   or p.get("activation_legality") != ACTIVATION_LEGALITY for p in proofs):
        raise ValueError("search root lacks a complete admitted particle bank")
    logits = np.asarray(decision["logits"], np.float64)
    q = record.get("q", [])
    valued_rows = () if zero else candidates
    if len(q) != len(logits) or any((q[i] is None) != (i not in valued_rows) for i in range(len(logits))) \
            or any(type(q[i]) not in (int, float) or not math.isfinite(q[i]) for i in valued_rows):
        raise ValueError("search root values do not match exactly its rolled-out rows")

    def softmax(values):
        p = np.exp(values - np.max(values))
        return p / p.sum()

    prior = softmax(logits)
    selected = np.asarray(candidates, np.int64)
    p = prior.copy()
    # Uniform magnet is constant across these rows; its normalization cancels.
    if not zero:
        p[selected] = prior[selected].sum() * softmax(
            (np.asarray([q[i] for i in candidates]) + config.beta * logits[selected]) / (config.alpha + config.beta))
    if not np.allclose(record.get("prior"), prior, atol=1e-12, rtol=1e-12) \
            or not np.allclose(record.get("policy"), p, atol=1e-12, rtol=1e-12) \
            or not 0 <= decision["chosen"] < len(logits) \
            or not math.isclose(decision["p_chosen"], float(p[decision["chosen"]]), abs_tol=1e-12, rel_tol=1e-12) \
            or config.selection == "greedy" and decision["chosen"] != int(np.argmax(p)):
        raise ValueError("root policy does not implement the registered update or preserve unsearched prior mass")


def recorded_allocation(root, config, room_seconds):
    """The public allocation a legacy anytime root must record, in the runtime's own arithmetic.

    A gated prompt allocated at its own monotonic start, which its on-demand trace keeps; an absolute deadline minus
    that start rounds differently from the same allocation at a small origin. A deferred clock serializes its
    allocation in relative coordinates, which any origin repeats exactly.
    """
    budget = root.get("public_budget", {})
    origin = root["on_demand"]["started"] if config.on_demand is not None and "prompt_elapsed" not in budget \
        else budget.get("clock_elapsed")
    return A.allocation(now=origin, elapsed=budget.get("clock_elapsed"),
        clock_left=budget.get("clock_left"), room_seconds=room_seconds, search_seconds=config.seconds,
        clock_reserve=config.clock_reserve, clock_share=config.clock_share,
        response_margin=config.response_margin, finalize_seconds=config.finalize_seconds,
        prompt_cap=config._prompt_cap(),
        **({"total_seconds": config.total_seconds} if config.total_seconds is not None else {}),
        **({"prompt_elapsed": budget["prompt_elapsed"]} if "prompt_elapsed" in budget else {}))[2]


def check_search_report(report, service_identity, search_identity, record, *, room_seconds=600,
                        allow_server_surrender=False, allow_server_network_terminal=False):
    """The evaluator's search-player validator, separate from pure-policy reports."""
    from .room_clock import registered_room_seconds
    config = SearchConfig(**search_identity["settings"])
    if config.untimed_profile is not None:  # its room is part of the registered profile
        from .untimed_search import ROOM_SECONDS
        room_seconds = ROOM_SECONDS
    room_seconds = registered_room_seconds(room_seconds)
    from .agent_policy import REPORT_SCHEMA, check_report
    current = is_current_provider(search_identity["particles"].get("provider"))
    expected_schema = current_client_schema(config) if current else SearchPolicy.report_schema
    if report.get("schema") != expected_schema or report.get("search_identity") != search_identity:
        raise ValueError("the search client differs from its registered strategy")
    room_clocks = None
    if config.final_room_profile is not None:
        if room_seconds != FB.ROOM_SECONDS or search_identity.get('final_room_profile') != FB.PROFILE:
            raise ValueError('final-room search cannot reuse a different room or legacy clock identity')
        room_clocks = FB.check_clocks(report.get('final_room_clocks'), player=record['our_player'])
    elif 'final_room_clocks' in report or 'final_room_profile' in search_identity:
        raise ValueError('legacy search cannot carry an unregistered final-room clock profile')
    if config.opponent_clear_profile is not None:
        if search_identity.get('opponent_clear_profile') != OC.PROFILE or report.get('opponent_clear_profile') != OC.PROFILE:
            raise ValueError('opponent-clear search lost its explicit public-only identity')
    elif 'opponent_clear_profile' in report or 'opponent_clear_profile' in search_identity:
        raise ValueError('legacy search cannot be relabeled as opponent-clear search')
    if config.opponent_known_profile is not None:
        if search_identity.get('opponent_known_profile') != OK.PROFILE or report.get('opponent_known_profile') != OK.PROFILE:
            raise ValueError('opponent-known search lost its explicit viewer-scoped identity')
    elif 'opponent_known_profile' in report or 'opponent_known_profile' in search_identity:
        raise ValueError('legacy search cannot be relabeled as opponent-known search')
    if config.on_demand is None and 'on_demand' in report:
        raise ValueError("a legacy search client cannot silently add a gate")
    if config.budget_law == MULTI.LAW and report.get("timing", {}).get("model_decisions", {}).get("law") != MULTI.LAW:
        raise ValueError("multiplexed client timing cannot claim the legacy serial scheduler")
    from .agent_external_terminal import check as check_external_terminal
    external = check_external_terminal(report, record, allowed=allow_server_surrender,
                                       allow_network=allow_server_network_terminal)
    check_report({**report, "schema": REPORT_SCHEMA}, service_identity, record,
                 allow_forced_only_terminal=external)
    roots, decisions = report.get("roots", []), report.get("reports", [])
    if config.opponent_clear_profile is not None:
        from .client import SELECT_MESSAGES
        prompts = [i for i,(msg,body) in enumerate(report['public_messages']) if msg in SELECT_MESSAGES]
        if len(prompts) != len(roots):
            raise ValueError('opponent-clear ledger omitted an original prompt')
        for root, packet_index in zip(roots,prompts):
            OC.check_binding(root.get('opponent_clear'),report['public_messages'],
                             packet_index=packet_index,viewer=record['our_player'])
    if config.opponent_known_profile is not None:
        from .client import SELECT_MESSAGES
        prompts = [i for i,(msg,body) in enumerate(report['public_messages']) if msg in SELECT_MESSAGES]
        if len(prompts) != len(roots):
            raise ValueError('opponent-known ledger omitted an original prompt')
        for root, packet_index in zip(roots,prompts):
            OK.check_binding(root.get('opponent_known'),report['public_messages'],
                             packet_index=packet_index,viewer=record['our_player'])
    if config.on_demand is not None:
        G.check_traces(roots, decisions, config.on_demand, room_clocks=room_clocks,
                       opponent_clear_profile=config.opponent_clear_profile,
                       opponent_known_profile=config.opponent_known_profile)
        if report.get('on_demand') != G.summary(roots, config=config.on_demand):
            raise ValueError("on-demand report omitted skips, empty searches or wall-clock work")
    if len(roots) != len(report["responses"]) or report.get("sent_committed") != len(roots):
        raise ValueError("search root/response submission coverage is incomplete")
    conditioned = search_identity["particles"].get("provider") == "own-pending-natural-conditioned-capacity/v1"
    filtered = search_identity["particles"].get("provider") == REPLAY_FILTERED_PROVIDER
    if current and report.get("current_root_sampling") != current_root_summary(roots):
        raise ValueError("current-root search changed its full attempt/positive-search denominator")
    if conditioned and report.get("conditional_sampling") != conditional_sampling_summary(roots):
        raise ValueError("conditioned search changed its full admission/rejection/cap denominator")
    if filtered and report.get("replay_filtered_sampling") != replay_filtered_summary(roots):
        raise ValueError("replay-filtered search changed its full attempt/positive-search denominator")
    if search_identity["settings"].get("bank_unknown_law") == F.LAW \
            and report.get("bank_fallbacks") != F.decision_summary(roots):
        raise ValueError("bank fallback report changed its full prompt/attempt denominators")
    for index, root in enumerate(roots):
        if root.get("prompt") != index or root.get("root") is not True or root.get("complete") is not True \
                or root.get("sent_verified") is not True or "error" in root:
            raise ValueError("a client prompt lacks its successful native root and verified response")
        if room_clocks is not None:
            FB.check_confirmation(root, room_clocks)
        elif 'final_time_confirmation' in root:
            raise ValueError('legacy response cannot carry an unregistered confirmation guard')
        config = SearchConfig(**search_identity["settings"])
        if config.budget_law in ANYTIME_LAWS:
            budget, timing = root.get("public_budget", {}), root.get("timing", {})
            phases = ("root_seconds", "sample_seconds", "rollout_seconds", "update_seconds", "total_seconds")
            if set(timing) != set(phases) or any(type(timing[key]) not in (float, int)
                    or not math.isfinite(timing[key]) or timing[key] < 0 for key in phases):
                raise ValueError("each anytime root needs complete finite phase timings")
            if room_clocks is not None:
                if 'public_budget' in root:
                    raise ValueError('final-room search cannot relabel a legacy public-clock allocation')
                trace = root['on_demand']
                previous = room_clocks['bound_at'] if index == 0 else roots[index-1]['on_demand']['successful_send_ns']/1e9
                allocation = FB.check_trace(root['final_room_budget'], room_clocks, prompt_index=index,
                                           started=trace['started'], previous_response_at=previous)
                searched = any(event['gate']['search'] for event in trace['decisions'])
                hard = allocation.response_deadline if searched else allocation.original_response_deadline
                if trace['finish']['finished'] > hard or searched and not allocation.allow_optional:
                    raise ValueError('actual search or cleanup crossed the reserved150-second floor')
                response_seconds = hard - trace['started']
            else:
                if 'final_room_budget' in root:
                    raise ValueError('legacy prompt cannot silently add a final-room clock trace')
                if budget != recorded_allocation(root, config, room_seconds):
                    raise ValueError('anytime root changed its registered public allocation')
                response_seconds = budget['response_seconds'] + budget.get('prompt_elapsed', 0.)
            if timing["total_seconds"] >= response_seconds \
                    or sum(timing[key] for key in phases[:-1]) > timing["total_seconds"] + 1e-6:
                raise ValueError("anytime root time exceeds its public allocation or duplicates a phase")
        choices = [d for d in decisions if d["response_index"] == index and not d["forced"]]
        if (config.bank_unknown_law == F.LAW or filtered or current) and root.get("non_singleton_attempts", []) != [
                {"subdecision": d["subdecision"], "rows": d["rows"], "obs_sha256": d["obs_sha256"]}
                for d in choices if d["rows"] > 1]:
            raise ValueError("complete fallback-enabled root omitted a non-singleton attempt")
        searches = root.get("searches", [])
        if conditioned and root.get("conditional_admissions", []) != [search["conditional_admission"] for search in searches
                                                                      if not search.get("singleton")]:
            raise ValueError("search prompt differs from its complete pre-Q admission escrow")
        if filtered and root.get("replay_filtered_admissions", []) != [search["replay_filtered_admission"] for search in searches
                                                                   if not search.get("singleton")]:
            raise ValueError("search prompt differs from its complete replay-filtered admission escrow")
        if len(searches) != len(choices):
            raise ValueError("a subdecision lost its root-search report")
        for search, decision in zip(searches, choices):
            if room_clocks is not None and 'timing' in search and search['timing'].get('game') != room_clocks['game_id']:
                raise ValueError('final-room clock evidence belongs to another actual policy session')
            check_root_record(search, decision, search_identity, native_snapshot_sha256=root.get("native_snapshot_sha256"))
