"""Public sequential count-factor proposals, filtered only by bounded natural replay.

This explicitly accepts proposal/order/timeout bias. It is neither an exact
joint posterior nor the old capacity-uniform certified distribution. No labels,
host layouts, complete capacity counting or separate hand-feasibility oracle.
"""
from __future__ import annotations

from collections import Counter
import ast
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .agent_causal_plan import _sha, root_layout, PlanIncompatible
from .causal_replay import public_history, trial_replay_filtered_candidate, ReplayCandidateDropped
from ..common.client_replay_journal import at_root
from ..agent.public_world_codec import RPC_SCHEMA, RPC_CAPABILITY, decode_world, encode_world, world_sha256
from ..agent.search.particles import _groups, LOCATION_ID

INFORMATION_SET_SEARCH = True
LAW = "public-sequential-count-factors-two-second-natural-replay-filter/v1"
PROVIDER = "own-pending-replay-filtered/v1"
AUDIT_SCHEMA = "mirrorforce_replay_filtered_bank/v1"
HEAD_SCHEMA = "mirrorforce_pending_count_belief/v1"
HEAD_LAW = "same-public-forward-session-pending-count-head/v1"
LOCATIONS = (1, 2, 3, 4, 6, 7)
SINGLE_SECONDS = 2.
ADMISSION_SHARE = .35
PROPOSAL_LIMIT = 128
SOURCE_SCHEMA = "mirrorforce_replay_filter_python_source_bundle/v1"
SOURCE_LAW = "package-local-transitive-import-closure; external-native-rules-separately-bound/v1"
SOURCE_ROOTS = ("mirrorforce.netduel.replay_filtered_particles", "mirrorforce.netduel.causal_replay",
                "mirrorforce.netduel.agent_causal_plan", "mirrorforce.netduel.causal_linear")
SOURCE_REQUIRED = frozenset({"netduel/replay_filtered_particles.py", "netduel/causal_replay.py",
    "netduel/agent_causal_plan.py", "netduel/causal_linear.py", "netduel/causal_history.py",
    "netduel/causal_proof.py", "netduel/causal_profile.py", "netduel/causal_must_emit_runtime.py",
    "netduel/agent_public_recipe.py", "netduel/agent_shuffle_plan.py", "netduel/wire_projection.py",
    "netduel/board.py", "netduel/constants.py", "netduel/actions.py", "netduel/host_view.py",
    "common/client_replay_journal.py", "common/client_root.py", "common/client_shadow.py", "common/client_sync.py",
    "common/sidecar_io.py", "agent/public_world_codec.py", "agent/search/particles.py", "worldmodel/engine.py",
    "puzzle/core.py", "puzzle/messages.py", "puzzle/single.py"})
_SOURCE_PATHS = None


def _source_paths():
    """Bind the imported helpers too, including optional legacy module branches.

    This is a Python-source closure, not a claim about external NumPy/SciPy or
    native binaries. Those retain the separate deployment/rules receipts.
    """
    global _SOURCE_PATHS
    if _SOURCE_PATHS is None:
        root = Path(__file__).resolve().parents[1]
        pending, seen = list(SOURCE_ROOTS), {}
        while pending:
            name = pending.pop()
            if name in seen or name != "mirrorforce" and not name.startswith("mirrorforce."):
                continue
            local = root.joinpath(*name.split(".")[1:])
            path = local.with_suffix(".py")
            if not path.is_file():
                path = local / "__init__.py"
            if not path.is_file():
                continue  # an imported class/function is not a child module
            raw = path.read_bytes()
            seen[name] = path
            package = name if path.name == "__init__.py" else name.rpartition(".")[0]
            parts = name.split(".")
            pending.extend(".".join(parts[:i]) for i in range(1, len(parts)))
            for node in ast.walk(ast.parse(raw)):
                if isinstance(node, ast.Import):
                    pending.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    parent = ".".join(package.split(".")[:len(package.split(".")) - node.level + 1]) if node.level else ""
                    target = ((parent + "." if parent else "") + (node.module or "")).rstrip(".")
                    pending.append(target)
                    pending.extend(target + "." + alias.name for alias in node.names if alias.name != "*")
        required = {root.joinpath(*name.split(".")[1:]).with_suffix(".py") for name in SOURCE_ROOTS}
        if not required <= set(seen.values()) or not SOURCE_REQUIRED <= {str(path.relative_to(root)) for path in seen.values()}:
            raise ValueError("replay-filter source closure lost a required constructor/producer module")
        _SOURCE_PATHS = tuple(sorted(set(seen.values())))
    return _SOURCE_PATHS


def source_bundle():
    root = Path(__file__).resolve().parents[1]
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in _source_paths()}
    body = {"schema": SOURCE_SCHEMA, "law": SOURCE_LAW, "roots": list(SOURCE_ROOTS), "files": files}
    return {**body, "sha256": _sha(body)}


def check_source_bundle(bundle, *, live=False):
    from ..common.sidecar_io import require_sha256
    root = Path(__file__).resolve().parents[1]
    expected = {str(path.relative_to(root)) for path in _source_paths()}
    if not isinstance(bundle, dict) or set(bundle) != {"schema", "law", "roots", "files", "sha256"} \
            or bundle["schema"] != SOURCE_SCHEMA or bundle["law"] != SOURCE_LAW \
            or bundle["roots"] != list(SOURCE_ROOTS) or not isinstance(bundle["files"], dict) \
            or set(bundle["files"]) != expected:
        raise ValueError("replay-filter source map is not the complete registered Python helper closure")
    for value in bundle["files"].values():
        require_sha256(value, "replay source dependency", error=ValueError)
    body = {key: value for key, value in bundle.items() if key != "sha256"}
    if bundle["sha256"] != _sha(body):
        raise ValueError("replay-filter source map digest differs")
    if live and source_bundle() != bundle:
        raise ValueError("replay-filter constructor/producer/helper source changed during admission")
    return bundle


def check_filtered_admission(audit, *, requested_count, proposal_seed, world_sha256, obs_sha256,
                             history_sha256=None, source_sha256=None, expected_source_bundle=None):
    """Independent accounting and positive replay-proof verification; never a posterior certificate."""
    from .causal_proof import verify_natural_root_proof
    from ..common.sidecar_io import require_sha256
    if not isinstance(audit, dict) or audit.get("schema") != AUDIT_SCHEMA or audit.get("law") != LAW \
            or audit.get("requested_count") != requested_count or audit.get("proposal_seed") != proposal_seed \
            or audit.get("world_sha256") != world_sha256 or audit.get("obs_sha256") != obs_sha256 \
            or audit.get("proposal_limit") != PROPOSAL_LIMIT or audit.get("single_seconds") != SINGLE_SECONDS \
            or audit.get("admission_share") != ADMISSION_SHARE \
            or audit.get("timeout_filter_bias_accepted") is not True or audit.get("exact_conditional_posterior") is not False:
        raise ValueError("replay-filtered admission differs from its explicit proposal/timeout law")
    for name, expected in (("history_sha256", history_sha256), ("source_sha256", source_sha256),
                           ("world_sha256", world_sha256), ("obs_sha256", obs_sha256)):
        require_sha256(audit.get(name), "replay filtered " + name, error=ValueError)
        if expected is not None and audit[name] != expected:
            raise ValueError("replay admission belongs to another " + name)
    rows = audit.get("attempts")
    bundle = check_source_bundle(audit.get("source_bundle"))
    if expected_source_bundle is not None and bundle != expected_source_bundle:
        raise ValueError("replay admission source bundle differs from the registered identity")
    if audit.get("source_checks") != {"before": bundle["sha256"], "after": bundle["sha256"]}:
        raise ValueError("replay admission lacks its before/after source closure verification")
    if type(requested_count) is not int or not 1 <= requested_count <= PROPOSAL_LIMIT \
            or type(proposal_seed) is not int or type(audit.get("proposal_seed")) is not int \
            or type(audit.get("requested_count")) is not int or not isinstance(rows, list) or len(rows) > PROPOSAL_LIMIT:
        raise ValueError("replay-filtered bank has invalid attempt/target limits")
    accepted = []
    remaining, allocated = audit.get("search_remaining_seconds"), audit.get("allocated_admission_seconds")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in (remaining, allocated)) \
            or remaining > 40 \
            or not math.isclose(allocated, remaining * ADMISSION_SHARE, abs_tol=1e-6, rel_tol=0):
        raise ValueError("replay admission allocation differs from its nonrenewing search remainder")
    previous_finish = 0.
    statuses = {"accepted", "proposal_dead_end", "construction_mismatch", "construction_timeout", "replay_timeout", "packet_mismatch"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or type(row.get("ordinal")) is not int or row["ordinal"] != index \
                or row.get("status") not in statuses:
            raise ValueError("replay admission lost or reordered an original attempt")
        for phase in ("seconds", "proposal_seconds", "construction_seconds", "replay_seconds", "cleanup_seconds",
                      "overhead_seconds", "started_seconds", "deadline_seconds", "finished_seconds"):
            if type(row.get(phase)) not in (float, int) or not math.isfinite(row[phase]) or row[phase] < 0:
                raise ValueError("replay attempt needs all finite nonnegative phase times")
        start, finish, limit = row["started_seconds"], row["finished_seconds"], row["deadline_seconds"]
        left = row.get("deadline_remaining_seconds")
        if type(left) not in (int, float) or not math.isfinite(left) or start + 1e-6 < previous_finish \
                or start >= allocated or not math.isclose(limit, min(allocated, start + SINGLE_SECONDS), abs_tol=1e-6) \
                or finish < start or not math.isclose(row["seconds"], finish - start, abs_tol=1e-6) \
                or not math.isclose(left, limit - finish, abs_tol=1e-6) \
                or row.get("source_checks") != {"before": bundle["sha256"], "after": bundle["sha256"]}:
            raise ValueError("replay attempt deadline, ordering or source checks differ")
        previous_finish = finish
        if not math.isclose(sum(row[p] for p in ("proposal_seconds", "construction_seconds", "replay_seconds",
                                                "cleanup_seconds", "overhead_seconds")), row["seconds"], abs_tol=1e-6):
            raise ValueError("replay attempt duplicated phase time")
        if row["status"] == "accepted":
            if finish >= limit or row["seconds"] > SINGLE_SECONDS or row["cleanup_seconds"] != 0:
                raise ValueError("an expired two-second replay candidate cannot be admitted")
            accepted.append(index)
            require_sha256(row.get("layout_sha256"), "accepted replay layout", error=ValueError)
            verify_natural_root_proof(row.get("proof"), hypothesis_sha256=row["layout_sha256"],
                history_sha256=audit["history_sha256"], source_sha256=audit["source_sha256"])
            filtered = row["proof"].get("replay_filter", {})
            if filtered.get("law") != "first-mismatch-or-two-second-discard/v1" \
                    or filtered.get("historical_existence_node_budget", "absent") is not None \
                    or filtered.get("negative_certificate") is not False:
                raise ValueError("accepted replay proof lacks the declared positive-only filter")
            for key in ("proposal_seconds", "construction_seconds", "replay_seconds", "attempt_seconds",
                        "started_seconds", "deadline_seconds", "finished_seconds", "deadline_remaining_seconds"):
                expected = row["seconds"] if key == "attempt_seconds" else row[key]
                if type(filtered.get(key)) not in (int, float) or not math.isfinite(filtered[key]) \
                        or not math.isclose(filtered[key], expected, abs_tol=1e-6, rel_tol=0):
                    raise ValueError("accepted replay proof phase/deadline differs from its original attempt")
            if filtered.get("source_bundle_sha256") != bundle["sha256"]:
                raise ValueError("accepted replay proof belongs to another source closure")
    status = "complete" if len(accepted) == requested_count else "partial" if accepted else "empty"
    if len(accepted) > requested_count or audit.get("accepted_ordinals") != accepted \
            or type(audit.get("accepted")) is not int or type(audit.get("attempted")) is not int \
            or audit["accepted"] != len(accepted) or audit["attempted"] != len(rows) \
            or audit.get("status") != status \
            or audit.get("discard_counts") != dict(Counter(row["status"] for row in rows if row["status"] != "accepted")) \
            or audit.get("discard_rate") != (1 - len(accepted) / len(rows) if rows else 0.):
        raise ValueError("replay acceptance/drop denominators or retained prefix differ")
    if any(type(value) is not int or value < 1 for value in audit["discard_counts"].values()):
        raise ValueError("replay discard counters must be exact positive integers")
    for phase in ("proposal_seconds", "construction_seconds", "replay_seconds", "admission_seconds",
                  "initial_proposal_seconds", "preparation_seconds", "between_attempt_seconds", "finalization_seconds"):
        if type(audit.get(phase)) not in (float, int) or not math.isfinite(audit[phase]) or audit[phase] < 0:
            raise ValueError("replay bank needs finite total phase times")
    for phase in ("construction_seconds", "replay_seconds"):
        if not math.isclose(audit[phase], sum(row[phase] for row in rows), abs_tol=1e-6):
            raise ValueError("replay bank totals differ from original attempts")
    if not math.isclose(audit["proposal_seconds"], audit["initial_proposal_seconds"] + sum(row["proposal_seconds"] for row in rows), abs_tol=1e-6) \
            or not math.isclose(audit["admission_seconds"], sum(row["seconds"] for row in rows)
                + sum(audit[phase] for phase in ("preparation_seconds", "between_attempt_seconds", "finalization_seconds")), abs_tol=1e-6) \
            or not math.isclose(audit["preparation_seconds"], rows[0]["started_seconds"] if rows else audit["admission_seconds"], abs_tol=1e-6) \
            or not math.isclose(audit["finalization_seconds"], audit["admission_seconds"] - previous_finish if rows else 0., abs_tol=1e-6) \
            or not math.isclose(audit["between_attempt_seconds"], sum(right["started_seconds"] - left["finished_seconds"]
                for left, right in zip(rows, rows[1:])), abs_tol=1e-6) \
            or audit["admission_seconds"] > remaining + sum(row["cleanup_seconds"] for row in rows) + 1e-6:
        raise ValueError("replay bank allocation, cumulative phase time or cleanup overrun differs")
    if audit.get("proposal_mode") not in ("uniform", "count_head") \
            or audit["proposal_mode"] == "uniform" and audit.get("uniform_reason") not in (
                "belief_capability_unavailable", "no_public_candidates") \
            or audit["proposal_mode"] == "count_head" and audit.get("uniform_reason") is not None:
        raise ValueError("replay proposal head/fallback evidence differs")
    return audit


class ProposalDeadEnd(Exception):
    pass


class SequentialProposal:
    def __init__(self, world, belief=None):
        self.world = decode_world(encode_world(world, complete=True), complete=True)
        self.groups = _groups(self.world)
        self.logp = {}
        self.mode, self.uniform_reason = "uniform", "belief_capability_unavailable"
        if belief is not None:
            if not isinstance(belief, dict) or set(belief) != {"codes", "locations", "logits"} \
                    or belief["locations"] != list(LOCATIONS) or not isinstance(belief["codes"], list) \
                    or any(type(c) is not int or c <= 0 for c in belief["codes"]) \
                    or len(set(belief["codes"])) != len(belief["codes"]):
                raise ValueError("belief head needs exact distinct public raw codes and six declared locations")
            logits = np.asarray(belief["logits"], np.float64)
            if not belief["codes"] and logits.size == 0:
                logits = logits.reshape(0, 6, 4)
            if logits.shape != (len(belief["codes"]), 6, 4) or not np.isfinite(logits).all():
                raise ValueError("belief count logits must be finite [public candidates,6,4]")
            shifted = logits - logits.max(-1, keepdims=True)
            logp = shifted - np.logaddexp.reduce(shifted, axis=-1)[..., None]
            if not np.isfinite(logp).all():
                raise ValueError("belief count normalization is not finite")
            self.logp = dict(zip(belief["codes"], logp))
            self.mode, self.uniform_reason = ("count_head", None) if self.logp else ("uniform", "no_public_candidates")
        for pool, groups in zip((self.world["pool_main"], self.world["pool_extra"]), self.groups):
            if sum(pool.values()) != sum(group.capacity for group in groups):
                raise ValueError("public recipe pool and hidden slot count differ")
            if any(sum(group.lower.values()) > group.capacity for group in groups):
                raise ValueError("public lower counts exceed their declared group capacity")

    def sample(self, seed, ordinal):
        rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal]))
        world = self.world
        particle = {name: list(world[name]) for name in ("hand", "deck", "extra")}
        particle["facedown"] = [[loc, seq, code] for loc, seq, code, _ in world["facedown"]]
        facedown = {(loc, seq): i for i, (loc, seq, _) in enumerate(particle["facedown"])}
        counts, missing = Counter(), set()
        for original, groups in zip((world["pool_main"], world["pool_extra"]), self.groups):
            remaining, assigned = Counter(original), [[] for _ in groups]
            for index, group in enumerate(groups):
                for code, copies in sorted(group.lower.items()):
                    if copies < 0 or copies > remaining[code] or not group.eligible(code):
                        raise ProposalDeadEnd("public lower-count reservation has no remaining copy")
                    assigned[index].extend([code] * copies)
                    remaining[code] -= copies
                    counts[code, LOCATION_ID[group.name]] += copies
            # Group predicates are the original public type categories. The
            # order is declared, deterministic and never based on a true card.
            order = sorted(range(len(groups)), key=lambda i: (
                sum(n for code, n in original.items() if groups[i].eligible(code)), i))
            for index in order:
                group, values = groups[index], assigned[index]
                location = LOCATION_ID[group.name]
                while len(values) < group.capacity:
                    codes = sorted(code for code, n in remaining.items() if n > 0 and group.eligible(code))
                    if not codes:
                        raise ProposalDeadEnd("sequential public-category allocation exhausted eligible copies")
                    logs = np.asarray([np.log(remaining[code]) + (
                        self.logp[code][LOCATIONS.index(location), min(counts[code, location] + 1, 3)]
                        - self.logp[code][LOCATIONS.index(location), min(counts[code, location], 3)]
                        if code in self.logp else 0.) for code in codes])
                    if self.mode == "count_head":
                        missing.update(code for code in codes if code not in self.logp)
                    mass = np.exp(logs - logs.max())
                    code = codes[int(rng.choice(len(codes), p=mass / mass.sum()))]
                    values.append(code)
                    remaining[code] -= 1
                    counts[code, location] += 1
                rng.shuffle(values)
                for (area, place), code in zip(group.places, values):
                    if area == "facedown":
                        particle[area][facedown[place]][2] = code
                    else:
                        particle[area][place] = code
            if any(remaining.values()):
                raise RuntimeError("sequential proposal failed to consume exactly its public pool")
        free = list(world["own_deck"])
        fixed = world["own_deck_fixed"]
        for code in fixed.values():
            if code not in free:
                raise ValueError("public own fixed identities exceed the declared own inventory")
            free.remove(code)
        rng.shuffle(free)
        values = iter(free)
        particle["own_deck"] = [fixed[i] if i in fixed else next(values) for i in range(len(world["own_deck"]))]
        return particle, sorted(missing)


@dataclass
class FilteredRoots:
    particles: tuple
    admission: dict

    def __iter__(self):
        return iter(self.particles)

    def __len__(self):
        return len(self.particles)


@dataclass(frozen=True)
class ReplayFilteredBank:
    proposer: SequentialProposal
    history: object
    requested_count: int
    proposal_seed: int
    world_sha256: str
    obs_sha256: str
    history_sha256: str
    source_sha256: str
    proposal_seconds: float
    entropy_profile: dict | None = None
    law: str = LAW
    source_bundle_json: str | None = None

    def __post_init__(self):
        if self.source_bundle_json is None:
            object.__setattr__(self, "source_bundle_json", json.dumps(source_bundle(), sort_keys=True))
        check_source_bundle(json.loads(self.source_bundle_json))

    @property
    def source_bundle(self):
        return json.loads(self.source_bundle_json)

    @contextmanager
    def open_roots(self, root, public_recipe, *, seed, deadline):
        admitted, owned, audit = [], [], {"schema": AUDIT_SCHEMA, "law": LAW, "status": "running",
            "world_sha256": self.world_sha256, "obs_sha256": self.obs_sha256,
            "history_sha256": self.history_sha256, "source_sha256": self.source_sha256,
            "proposal_seed": self.proposal_seed, "requested_count": self.requested_count,
            "proposal_limit": PROPOSAL_LIMIT, "single_seconds": SINGLE_SECONDS, "admission_share": ADMISSION_SHARE,
            "proposal_mode": self.proposer.mode, "uniform_reason": self.proposer.uniform_reason,
            "accepted_ordinals": [], "attempts": [], "proposal_seconds": self.proposal_seconds,
            "initial_proposal_seconds": self.proposal_seconds,
            "construction_seconds": 0., "replay_seconds": 0., "timeout_filter_bias_accepted": True,
            "exact_conditional_posterior": False}
        began = time.monotonic()
        if type(deadline) not in (float, int) or not math.isfinite(deadline) or deadline - began > 40:
            raise ValueError("replay-filtered admission requires its finite <=40s remaining prompt deadline")
        bundle = self.source_bundle
        check_source_bundle(bundle, live=True)
        audit["source_bundle"] = bundle
        audit["source_checks"] = {"before": bundle["sha256"]}
        audit["search_remaining_seconds"] = max(0., deadline - began)
        admission_deadline = min(deadline, began + max(0., deadline - began) * ADMISSION_SHARE)
        audit["allocated_admission_seconds"] = max(0., admission_deadline - began)
        from contextlib import nullcontext
        scope = nullcontext()
        if self.entropy_profile is not None:
            from .causal_must_emit_runtime import bank_scope
            scope = bank_scope(root, self.entropy_profile, source_sha256=self.source_sha256,
                               history_sha256=self.history_sha256)
        events = at_root(root)  # entity/journal attestation precedes lending the root arena
        try:
            with root.branch(), scope:
                for ordinal in range(PROPOSAL_LIMIT):
                    root._check()
                    start = time.monotonic()
                    if start >= admission_deadline or len(admitted) == self.requested_count:
                        break
                    candidate_deadline = min(admission_deadline, start + SINGLE_SECONDS)
                    result, proof = None, None
                    row = {"ordinal": ordinal, "status": "running", "proposal_seconds": 0.,
                           "construction_seconds": 0., "replay_seconds": 0., "cleanup_seconds": 0.,
                           "started_seconds": start - began, "deadline_seconds": candidate_deadline - began}
                    audit["attempts"].append(row)
                    try:
                        check_source_bundle(bundle, live=True)
                        row["source_checks"] = {"before": bundle["sha256"]}
                        particle, missing = self.proposer.sample(self.proposal_seed, ordinal)
                        row["uniform_missing_codes"] = missing
                        row["proposal_seconds"] = time.monotonic() - start
                        layout = root_layout(self.history, particle)
                        row["layout_sha256"] = _sha(layout)
                        result = trial_replay_filtered_candidate(root, self.history, layout, public_recipe,
                            events=events, seed=seed + 1009 * ordinal,
                            deadline=candidate_deadline, source_sha256=self.source_sha256,
                            entropy_profile=self.entropy_profile)
                        owned.append(result)  # take ownership before copying or checking its report
                        proof = copy.deepcopy(result.proof)
                        row.update(status="candidate",
                            construction_seconds=result.proof["replay_filter"]["construction_seconds"],
                            replay_seconds=result.proof["replay_filter"]["replay_seconds"])
                    except ProposalDeadEnd as exc:
                        row.update(status="proposal_dead_end", error=str(exc), proposal_seconds=time.monotonic() - start)
                    except PlanIncompatible as exc:
                        row.update(status="construction_mismatch", error=str(exc),
                                   construction_seconds=max(0., time.monotonic() - start - row["proposal_seconds"]))
                    except ReplayCandidateDropped as exc:
                        row.update(status=exc.reason, error=str(exc), evidence=exc.evidence,
                            construction_seconds=getattr(exc, "construction_seconds", 0.),
                            replay_seconds=getattr(exc, "replay_seconds", 0.))
                    except BaseException as exc:
                        row.update(status="hard_error", error=f"{type(exc).__name__}: {exc}")
                        exc.replay_filter_audit = audit
                        raise  # diagnostic preservation never converts source/native failures to drops
                    finally:
                        try:
                            check_source_bundle(bundle, live=True)
                            row.setdefault("source_checks", {})["after"] = bundle["sha256"]
                        except BaseException as exc:
                            row.update(status="hard_error", error=f"{type(exc).__name__}: {exc}")
                            exc.replay_filter_audit = audit
                            raise
                        ended = time.monotonic()
                        if row["status"] == "candidate" and ended >= candidate_deadline:
                            row.update(status="replay_timeout", error="candidate completed beyond its registered cap",
                                       late_proof=proof)
                            cleanup_started = time.monotonic()
                            result.driver.close()
                            if getattr(result.driver, "pduel", None) is not None:
                                raise RuntimeError("late replay result survived cleanup")
                            row["cleanup_seconds"] = time.monotonic() - cleanup_started
                            ended = time.monotonic()
                        elif row["status"] == "candidate":
                            row["status"] = "accepted"
                        row.update(seconds=ended - start, finished_seconds=ended - began,
                                   deadline_remaining_seconds=candidate_deadline - ended)
                        stages = sum(row[p] for p in ("proposal_seconds", "construction_seconds", "replay_seconds", "cleanup_seconds"))
                        row["overhead_seconds"] = row["seconds"] - stages
                        if row["overhead_seconds"] < -1e-6:
                            raise ValueError("replay attempt reported overlapping stage times")
                        row["overhead_seconds"] = max(0., row["overhead_seconds"])
                        if row["status"] == "accepted":
                            proof["replay_filter"].update(source_bundle_sha256=bundle["sha256"],
                                proposal_seconds=row["proposal_seconds"], attempt_seconds=row["seconds"],
                                **{key: row[key] for key in ("started_seconds", "deadline_seconds", "finished_seconds", "deadline_remaining_seconds")})
                            row["proof"] = proof
                            result.proof["replay_filter"] = copy.deepcopy(proof["replay_filter"])
                    if row["status"] == "accepted":
                        admitted.append(result)
                        audit["accepted_ordinals"].append(ordinal)
                check_source_bundle(bundle, live=True)
                audit["source_checks"]["after"] = bundle["sha256"]
                audit.update(status="complete" if len(admitted) == self.requested_count else "partial" if admitted else "empty",
                    attempted=len(audit["attempts"]), accepted=len(admitted),
                    admission_seconds=time.monotonic() - began,
                    discard_counts=dict(Counter(row["status"] for row in audit["attempts"] if row["status"] != "accepted")))
                rows = audit["attempts"]
                audit["preparation_seconds"] = rows[0]["started_seconds"] if rows else audit["admission_seconds"]
                audit["between_attempt_seconds"] = sum(next_row["started_seconds"] - row["finished_seconds"] for row, next_row in zip(rows, rows[1:]))
                audit["finalization_seconds"] = audit["admission_seconds"] - rows[-1]["finished_seconds"] if rows else 0.
                for phase in ("proposal_seconds", "construction_seconds", "replay_seconds"):
                    audit[phase] += sum(row[phase] for row in audit["attempts"])
                audit["discard_rate"] = 1 - len(admitted) / len(audit["attempts"]) if audit["attempts"] else 0.
                yield FilteredRoots(tuple(admitted), audit)
        finally:
            failures = []
            for particle in reversed(owned):
                try:
                    particle.driver.close()
                    if getattr(particle.driver, "pduel", None) is not None:
                        raise RuntimeError("accepted replay driver survived bank cleanup")
                except BaseException as exc:
                    failures.append(exc)
            if failures:
                raise RuntimeError("failed to clean all replay-filtered bank drivers") from failures[0]
            root.snapshot.verify()
            check_source_bundle(bundle, live=True)


class PublicReplayFilteredParticles:
    identity = {"law": LAW, "provider": PROVIDER, "world_rpc": RPC_CAPABILITY,
        "proposal": "sequential-lower-reservation-public-category-count-increments/v1",
        "belief": HEAD_LAW, "fallback": "missing-capability-or-candidate-uniform/v1",
        "proposal_limit": PROPOSAL_LIMIT, "single_seconds": SINGLE_SECONDS, "admission_share": ADMISSION_SHARE,
        "acceptance": "fresh-opening-native-replay-full-observer-prefix-root-layout/v1",
        "first_mismatch": "discard-one-proposal", "historical_existence_node_budget": None,
        "negative_certificate": False, "timeout_filter_bias_accepted": True, "exact_conditional_posterior": False}

    def __init__(self, entropy_profile=None):
        self.entropy_profile = copy.deepcopy(entropy_profile)
        self.identity = {**self.identity, "entropy_profile": self.entropy_profile,
                         "proposal_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "construction": "positive-exact-integer-witness-only; no-negative-or-node-budget/v1",
                         "own_deck": "independent-public-inventory-shuffle; fixed-positions-preserved/v1"}
        self.identity["source_bundle"] = source_bundle()
        if entropy_profile is not None:
            from .causal_profile import verify_bundle
            verify_bundle(entropy_profile)
            self.identity["producer_source_sha256"] = entropy_profile["producer_source_sha256"]

    def propose(self, root, client, *, count, seed, context, extra_origins, call, session, pending, deadline):
        started = time.monotonic()
        if type(count) is not int or not 1 <= count <= PROPOSAL_LIMIT:
            raise ValueError("replay bank needs a finite declared positive target count")
        root._check()
        check_source_bundle(self.identity["source_bundle"], live=True)
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self.identity["proposal_source_sha256"]:
            raise ValueError("registered public proposal source changed before this root")
        reply = call({"op": "public_world", "session": session, "expected_obs_sha256": pending["obs_sha256"]})
        if not isinstance(reply, dict) or set(reply) != {"schema", "session", "obs_sha256", "world_sha256", "world"} \
                or reply["schema"] != RPC_SCHEMA or reply["session"] != session \
                or reply["obs_sha256"] != pending["obs_sha256"] or world_sha256(reply["world"]) != reply["world_sha256"]:
            raise ValueError("replay proposals require this owned pending public World")
        belief = None
        from .agent_public_recipe import declare
        if pending.get("count_belief_available") is True:
            head = call({"op": "public_belief", "session": session, "expected_obs_sha256": pending["obs_sha256"]})
            if not isinstance(head, dict) or set(head) != {"schema", "law", "session", "obs_sha256", "belief"} \
                    or head["schema"] != HEAD_SCHEMA or head["law"] != HEAD_LAW or head["session"] != session \
                    or head["obs_sha256"] != pending["obs_sha256"]:
                raise ValueError("replay belief belongs to another pending public input")
            belief = head["belief"]
        history = public_history(root, declare(client.main, client.extra))
        source = self.entropy_profile["producer_source_sha256"] if self.entropy_profile is not None else \
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if self.entropy_profile is not None:
            from .causal_must_emit_runtime import verify_registered_source
            verify_registered_source(self.entropy_profile)
        check_source_bundle(self.identity["source_bundle"], live=True)
        return ReplayFilteredBank(SequentialProposal(decode_world(reply["world"], complete=True), belief), history,
            count, seed, reply["world_sha256"], pending["obs_sha256"], _sha(history.record()), source,
            time.monotonic() - started, self.entropy_profile,
            source_bundle_json=json.dumps(self.identity["source_bundle"], sort_keys=True))
