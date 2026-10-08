"""Pre-registered capacity proposals conditioned on certified natural feasibility.

Only a whole-candidate impossibility certificate may advance to a replacement
proposal. Unknown, exhausted or unsupported checks invalidate the whole bank.
The complete proposal prefix and every attempted verdict survive both success
and failure. No candidate action, Q estimate or game outcome enters this module.
"""
from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import copy
import json
import math
import time

from .causal_particles import PublicCausalParticles, PublicCapacityParticles
from .causal_sampler import LAW as BASE_LAW
from .agent_causal_plan import _sha
from .causal_replay import public_history
from .bank_fallback import UnknownReason
from ..common.client_replay_journal import at_root

INFORMATION_SET_SEARCH = True
LAW = "public-capacity-uniform-certified-natural-prefix/v1"
ADMISSION_SCHEMA = "mirrorforce_conditioned_causal_admission/v1"
SEQUENCE_LAW = "independent-sha256-counter-capacity-proposal-prefix/v1"
PREDICATE = "complete-natural-prefix-witness-or-certified-whole-root-impossibility/v1"
PROPOSAL_LIMIT = 128


class CertifiedNaturalInfeasible(RuntimeError):
    def __init__(self, message, certificate):
        super().__init__(message)
        self._certificate = json.dumps(certificate, sort_keys=True, allow_nan=False)

    @property
    def certificate(self):
        return json.loads(self._certificate)


class NaturalFeasibilityUnknown(RuntimeError):
    def __init__(self, message, *, evidence=None, reason=None):
        super().__init__(message)
        if reason is not None and type(reason) is not UnknownReason:
            raise ValueError("natural unknown reason must be a typed producer cause")
        self.reason = reason  # None preserves old strict-only unknown records.
        self._evidence = json.dumps(evidence or {}, sort_keys=True, allow_nan=False)

    @property
    def evidence(self):
        return json.loads(self._evidence)


class ConditionalAdmissionError(RuntimeError):
    """A hard failed bank; ``admission`` is pure immutable-by-copy report data."""
    def __init__(self, message, admission):
        super().__init__(message)
        self._admission = json.dumps(admission, sort_keys=True, allow_nan=False)

    @property
    def admission(self):
        return json.loads(self._admission)


_UNKNOWN_ISSUER = object()


class TypedBankUnknown(ConditionalAdmissionError):
    """Issued only after producer cleanup; serialized records cannot issue it."""
    def __init__(self, message, admission, record, *, _issuer=None):
        if _issuer is not _UNKNOWN_ISSUER:
            raise ValueError("bank fallback needs a live producer cleanup capability")
        super().__init__(message, admission)
        self._record = json.dumps(record, sort_keys=True, allow_nan=False)

    @property
    def fallback_record(self):
        return json.loads(self._record)


def _issue_unknown(root, audit, reason, *, phase):
    """Validate all owned state after the bank's with/finally blocks exited."""
    from . import bank_fallback as F
    from .causal_must_emit_runtime import verify_restored_bank
    profile = audit["entropy_profile"]
    cleanup = verify_restored_bank(root, profile)
    record = {"schema": F.SCHEMA, "law": F.LAW, "reason": reason.value, "phase": phase,
        "admission_sha256": _sha(audit), "root_snapshot_sha256": cleanup["root_snapshot_sha256"],
        "history_sha256": audit.get("history_sha256"), "world_sha256": audit.get("world_sha256"),
        "obs_sha256": audit.get("obs_sha256"), "source_sha256": profile["producer_source_sha256"],
        "entropy_profile_sha256": _sha(profile), "cleanup": dict(F.CLEANUP), "hard_errors": []}
    F.verify_unknown_record(record, audit, root_snapshot_sha256=root.snapshot.digest,
        obs_sha256=audit["obs_sha256"], requested_count=audit["requested_count"],
        proposal_seed=audit["proposal_seed"], entropy_profile_sha256=_sha(profile),
        source_sha256=profile["producer_source_sha256"])
    return TypedBankUnknown("typed bank feasibility unknown after verified cleanup: " + reason.value,
                            audit, record, _issuer=_UNKNOWN_ISSUER)


@dataclass(frozen=True)
class ConditionalRoots(Sequence):
    roots: tuple
    admission_json: str

    def __len__(self):
        return len(self.roots)

    def __getitem__(self, index):
        return self.roots[index]

    @property
    def admission(self):
        return json.loads(self.admission_json)


@dataclass(frozen=True)
class ConditionedCausalParticles:
    proposals: tuple
    requested_count: int
    proposal_seed: int
    weights: tuple
    law: str
    world_sha256: str
    obs_sha256: str
    history_sha256: str
    source_sha256: str
    sampler_proof_json: str
    proposal_seconds: float
    entropy_profile_json: str = "null"  # no certificate can use an absent/unreviewed entropy profile
    proposal_limit: int = PROPOSAL_LIMIT

    def __post_init__(self):
        if self.law != LAW or type(self.requested_count) is not int or not 1 <= self.requested_count <= self.proposal_limit \
                or type(self.proposal_seed) is not int or type(self.proposal_limit) is not int \
                or self.proposal_limit != PROPOSAL_LIMIT or type(self.proposals) is not tuple \
                or len(self.proposals) != self.proposal_limit or type(self.weights) is not tuple \
                or self.weights != (1.0,) * self.requested_count \
                or type(self.proposal_seconds) not in (int, float) or not math.isfinite(self.proposal_seconds) \
                or self.proposal_seconds < 0:
            raise ValueError("conditional bank must bind its entire fixed proposal prefix and requested uniform bank")
        # Reuse the strict physical layout validator, without granting the old
        # unconditional law to this new conditional bank.
        PublicCausalParticles(self.proposals, (1.0,) * len(self.proposals), BASE_LAW, self.world_sha256,
            self.obs_sha256, self.history_sha256, self.source_sha256, self.sampler_proof_json)
        json.loads(self.entropy_profile_json)

    def audit(self):
        return {"schema": ADMISSION_SCHEMA, "law": LAW, "base_law": BASE_LAW, "sequence_law": SEQUENCE_LAW,
                "acceptance_predicate": PREDICATE, "proposal_seed": self.proposal_seed,
                "proposal_limit": self.proposal_limit, "requested_count": self.requested_count,
                "world_sha256": self.world_sha256, "obs_sha256": self.obs_sha256,
                "history_sha256": self.history_sha256, "source_sha256": self.source_sha256,
                "proposal_sequence_sha256": _sha(self.proposals),
                "sampler": json.loads(self.sampler_proof_json), "entropy_profile": json.loads(self.entropy_profile_json),
                "proposal_seconds": self.proposal_seconds, "admission_seconds": 0.0, "total_seconds": self.proposal_seconds,
                "status": "not_started", "attempted": 0, "accepted_ordinals": [], "rejected": 0, "unknown": 0,
                "rejection_rate": 0.0, "cap_reached": False,
                "proposals": [{"ordinal": i, "layout": [list(row) for row in layout], "layout_sha256": _sha(layout),
                               "status": "not_attempted", "seconds": 0.0, "proof": None}
                              for i, layout in enumerate(self.proposals)]}

    def open_roots(self, root, public_recipe, *, seed, deadline):
        if seed != self.proposal_seed:
            raise ConditionalAdmissionError("conditional proposal seed changed after registration", self.audit())
        return conditioned_roots(root, self, public_recipe, deadline=deadline)


def _bank_runtime_scope(root, bank):
    from . import causal_profile as profile_bundle
    profile = json.loads(bank.entropy_profile_json)
    if not profile_bundle.is_bundle(profile):
        return nullcontext()
    from .causal_must_emit_runtime import bank_scope
    return bank_scope(root, profile, source_sha256=bank.source_sha256, history_sha256=bank.history_sha256)


def _unknown_failure(exc, *, bundled):
    # A runtime rule/source closure failure is hard even if an old report
    # described that failure as unknown. Only an actual typed search may stop.
    return isinstance(exc, NaturalFeasibilityUnknown)


@contextmanager
def conditioned_roots(root, bank, public_recipe, *, deadline, trial=None):
    """All original proposals are fixed before this lease; no Q is available here."""
    from .causal_replay import trial_natural_candidate
    trial = trial_natural_candidate if trial is None else trial  # injection is a unit-test seam, never a config option
    audit, admitted = bank.audit(), []
    from . import causal_profile as profile_bundle
    bundled = profile_bundle.is_bundle(audit["entropy_profile"])
    started = time.monotonic()

    def update():
        audit["admission_seconds"] = time.monotonic() - started
        audit["total_seconds"] = bank.proposal_seconds + audit["admission_seconds"]
        audit["rejection_rate"] = audit["rejected"] / audit["attempted"] if audit["attempted"] else 0.0
        audit["cap_reached"] = audit["attempted"] == bank.proposal_limit

    try:
        history = public_history(root, public_recipe)
        if _sha(history.record()) != bank.history_sha256:
            raise ValueError("conditioned bank belongs to another received public history")
        events = at_root(root)
        with root.branch(), _bank_runtime_scope(root, bank):
            try:
                for index, layout in enumerate(bank.proposals):
                    root._check()
                    if time.monotonic() >= deadline:
                        raise NaturalFeasibilityUnknown("conditional admission exhausted its absolute deadline",
                                                        reason=UnknownReason.ADMISSION_DEADLINE)
                    row = audit["proposals"][index]
                    row["status"] = "running"
                    audit["attempted"] += 1
                    began = time.monotonic()
                    try:
                        particle = trial(root, history, layout, public_recipe, events=events,
                            seed=bank.proposal_seed + 1009 * index, deadline=deadline,
                            source_sha256=bank.source_sha256, entropy_profile=json.loads(bank.entropy_profile_json))
                        row["status"], row["proof"] = "accepted", copy.deepcopy(particle.proof)
                        admitted.append(particle)
                        audit["accepted_ordinals"].append(index)
                    except CertifiedNaturalInfeasible as exc:
                        # The producer's certificate is checked independently
                        # before it can justify skipping this original proposal.
                        from .causal_rejection import verify_negative_certificate
                        try:
                            if bundled:
                                from .causal_must_emit_runtime import verify_issued_certificate
                                verify_issued_certificate(root, exc.certificate, profile=audit["entropy_profile"],
                                    hypothesis_sha256=row["layout_sha256"], history_sha256=bank.history_sha256,
                                    source_sha256=bank.source_sha256)
                            verify_negative_certificate(exc.certificate, hypothesis_sha256=row["layout_sha256"],
                                history_sha256=bank.history_sha256, source_sha256=bank.source_sha256,
                                entropy_profile=json.loads(bank.entropy_profile_json))
                        except BaseException as failure:
                            row["status"], row["proof"] = "error", exc.certificate
                            row["error"] = f"{type(failure).__name__}: {failure}"
                            raise
                        row["status"], row["proof"] = "certified_infeasible", exc.certificate
                        audit["rejected"] += 1
                    except BaseException as exc:
                        row["status"] = "unknown" if _unknown_failure(exc, bundled=bundled) else "error"
                        row["proof"] = getattr(exc, "evidence", None)
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        if row["status"] == "unknown":
                            audit["unknown"] += 1
                        raise
                    finally:
                        row["seconds"] = time.monotonic() - began
                        update()
                    if len(admitted) == bank.requested_count:
                        if bundled:
                            from .causal_must_emit_runtime import verify_bank_assets
                            verify_bank_assets(root)
                        audit["status"] = "complete"
                        yield ConditionalRoots(tuple(admitted), json.dumps(audit, sort_keys=True, allow_nan=False))
                        return
                audit["cap_reached"] = True
                audit["status"] = "proposal_cap_exhausted"
                raise ConditionalAdmissionError("fixed proposal prefix did not fill the requested complete bank", audit)
            finally:
                cleanup_errors = []
                for particle in reversed(admitted):
                    try:
                        particle.driver.close()
                        if getattr(particle.driver, "pduel", None) is not None:
                            raise RuntimeError("an admitted native driver survived bank cleanup")
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                if cleanup_errors:
                    raise RuntimeError("failed to clean every owned bank driver") from cleanup_errors[0]
    except ConditionalAdmissionError as exc:
        if hasattr(exc, "bank_postcheck_error"):
            audit["status"] = "error"
            audit["postcheck_failure"] = exc.bank_postcheck_error
            audit["failure"] = f"{type(exc).__name__}: {exc}"
            raise ConditionalAdmissionError(audit["failure"], audit) from exc
        raise
    except BaseException as exc:
        if audit["status"] == "complete":
            raise  # a consumer failure is not an admission verdict or a resampling opportunity
        update()
        audit["status"] = "unknown" if _unknown_failure(exc, bundled=bundled) else "error"
        audit["failure"] = f"{type(exc).__name__}: {exc}"
        if hasattr(exc, "bank_postcheck_error"):
            audit["postcheck_failure"] = exc.bank_postcheck_error
        if bundled and type(exc) is NaturalFeasibilityUnknown and exc.reason is not None:
            audit["failure_kind"] = exc.reason.value
            try:
                unknown = _issue_unknown(root, audit, exc.reason, phase="admission")
            except BaseException as hard:
                audit["status"] = "error"
                audit["failure"] = f"bank cleanup/registration verification failed: {type(hard).__name__}: {hard}"
                raise ConditionalAdmissionError(audit["failure"], audit) from hard
            raise unknown from exc
        raise ConditionalAdmissionError("conditional bank failed without replacement after unknown/error: " + str(exc), audit) from exc


class PublicConditionedCapacityParticles:
    identity = {**PublicCapacityParticles.identity, "law": LAW, "base_law": BASE_LAW,
                "provider": "own-pending-natural-conditioned-capacity/v1", "sequence_law": SEQUENCE_LAW,
                "acceptance_predicate": PREDICATE, "proposal_limit": PROPOSAL_LIMIT,
                "negative_certificate": "unique-opening-no-response-deterministic-prefix/v1",
                "unknown": "entire-bank-fails; never rejection/v1"}

    def __init__(self, entropy_profile=None):
        self.entropy_profile_json = json.dumps(entropy_profile, sort_keys=True, allow_nan=False)
        self.identity = {**self.identity, "entropy_profile": json.loads(self.entropy_profile_json),
                         "entropy_profile_sha256": _sha(entropy_profile) if entropy_profile else None}
        from . import causal_pass as PASS
        from . import causal_profile as profile_bundle
        if profile_bundle.is_bundle(entropy_profile):
            profile_bundle.verify_bundle(entropy_profile)
            from .causal_must_emit import LAW as MUST_EMIT_LAW
            self.identity["negative_certificate"] = [self.identity["negative_certificate"], PASS.EARLY_LAW, MUST_EMIT_LAW]
            self.identity["source_binding_law"] = profile_bundle.SOURCE_LAW
            self.identity["producer_source_sha256"] = entropy_profile["producer_source_sha256"]
        if isinstance(entropy_profile, dict) and entropy_profile.get("scope") in PASS.SCOPES:
            PASS.verify_profile(entropy_profile)
            self.identity["negative_certificate"] = [self.identity["negative_certificate"],
                                                       entropy_profile["own_chain_pass_lemma"]["law"]]

    def propose(self, root, client, *, count, seed, context, extra_origins, call, session, pending, deadline=None):
        began = time.monotonic()
        from . import causal_profile as profile_bundle
        profile = json.loads(self.entropy_profile_json)
        bundled = profile_bundle.is_bundle(profile)
        preflight = {"schema": ADMISSION_SCHEMA, "law": LAW, "base_law": BASE_LAW, "sequence_law": SEQUENCE_LAW,
                     "acceptance_predicate": PREDICATE, "proposal_seed": seed, "proposal_limit": PROPOSAL_LIMIT,
                     "requested_count": count, "obs_sha256": pending.get("obs_sha256"), "status": "proposal_pending",
                     "attempted": 0, "accepted_ordinals": [], "rejected": 0, "unknown": 0,
                     "cap_reached": False, "rejection_rate": 0.0, "proposals": [], "proposal_seconds": 0.0,
                     "admission_seconds": 0.0, "total_seconds": 0.0,
                     "entropy_profile": json.loads(self.entropy_profile_json),
                     "generated_proposals": 0, "requested_proposals": PROPOSAL_LIMIT}
        try:
            if bundled:
                from .causal_must_emit_runtime import verify_registered_source
                verify_registered_source(profile)
            if type(count) is not int or not 1 <= count <= PROPOSAL_LIMIT:
                raise ValueError("requested bank exceeds its pre-registered fixed proposal prefix")
            base = PublicCapacityParticles().propose(root, client, count=PROPOSAL_LIMIT, seed=seed,
                context=context, extra_origins=extra_origins, call=call, session=session, pending=pending, deadline=deadline)
            return ConditionedCausalParticles(base.assignments, count, seed, (1.0,) * count, LAW,
                base.world_sha256, base.obs_sha256, base.history_sha256,
                profile["producer_source_sha256"] if bundled else base.source_sha256,
                base.sampler_proof_json, time.monotonic() - began, self.entropy_profile_json)
        except BaseException as exc:
            preflight["status"] = "proposal_failed"
            preflight["proposal_seconds"] = preflight["total_seconds"] = time.monotonic() - began
            preflight["failure"] = f"{type(exc).__name__}: {exc}"
            preflight.update(getattr(exc, "proposal_evidence", {}))
            if bundled:
                preflight["source_sha256"] = profile["producer_source_sha256"]
            partial = preflight.get("partial_layouts", [])
            preflight["proposals"] = [{"ordinal": i, "layout": layout, "layout_sha256": _sha(layout),
                "status": "not_attempted", "seconds": 0.0, "proof": None} for i, layout in enumerate(partial)]
            preflight["proposal_sequence_sha256"] = _sha(partial)
            from .causal_sampler import SamplingBudgetExceeded
            if bundled and type(exc) is SamplingBudgetExceeded:
                preflight["failure_kind"] = UnknownReason.PROPOSAL_BUDGET.value
                try:
                    unknown = _issue_unknown(root, preflight, UnknownReason.PROPOSAL_BUDGET, phase="proposal")
                except BaseException as hard:
                    preflight["failure"] = f"proposal cleanup/registration verification failed: {type(hard).__name__}: {hard}"
                    preflight["hard_error"] = True
                    raise ConditionalAdmissionError(preflight["failure"], preflight) from hard
                raise unknown from exc
            raise ConditionalAdmissionError("conditional proposal generation failed: " + str(exc), preflight) from exc
