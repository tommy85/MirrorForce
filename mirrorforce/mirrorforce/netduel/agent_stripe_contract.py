"""Evidence-bound admission of one explicit current-root batching scheduler.

Pure JSON/file validation: no model, native core, training labels or diagnostic
tool imports. A successful finite full-bank audit is not a room-time/strength
gate. Its original report, completion and configuration are all SHA-bound.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

from mirrorforce.agent.inference_geometry import FIXED_PUBLIC_B64
from . import agent_anytime as A, agent_stripe_batching as B
from .current_root_protocol import CAPABILITY

SCHEMA = "mirrorforce_stripe_numeric_contract/v1"
MIRROR_SCHEMA = "mirrorforce_stripe_numeric_contract/v2"
AUDIT_SCHEMA = "mirrorforce_current_stripe_comparison/v1"
EVIDENCE_KEYS = {"report", "worker", "config"}
IDENTITY_KEYS = {"backend", "weights", "checkpoint_sha256", "receipt_sha256", "compute_dtype", "magnet",
                 "training_native_sha256", "semantic_file_sha256", "card_tables_sha256", "search_batching",
                 "inference_geometry", "replay_belief", "opponent_recipe_mode", "public_opponent_recipe", "current_root"}


def identity_keys(identity):
    if identity.get('backend')!='specialization':
        return IDENTITY_KEYS
    from .agent_specialization_search import registered_service
    registered_service(identity)
    return (IDENTITY_KEYS-{'replay_belief'})|{'specialization','specialization_search','parent_checkpoint_sha256'}


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def checked_refs(refs):
    if not isinstance(refs, dict) or set(refs) != EVIDENCE_KEYS or any(
            not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or not isinstance(ref["path"], str) or not Path(ref["path"]).is_absolute() or not _sha(ref["sha256"])
            for ref in refs.values()):
        raise ValueError("stripe evidence needs the exact report, worker completion and configuration references")
    return refs


def load_stripe_evidence(refs):
    from .search_numeric import load_numeric_evidence
    return {key: load_numeric_evidence(ref["path"], ref["sha256"]) for key, ref in checked_refs(refs).items()}


def _decode(evidence):
    if not isinstance(evidence, dict) or set(evidence) != EVIDENCE_KEYS:
        raise ValueError("multiplexed search needs its own registered raw GPU evidence bundle")
    out = {}
    for key, envelope in evidence.items():
        if not isinstance(envelope, dict) or set(envelope) != {"sha256", "json"} \
                or not _sha(envelope["sha256"]) or not isinstance(envelope["json"], str) \
                or hashlib.sha256(envelope["json"].encode()).hexdigest() != envelope["sha256"]:
            raise ValueError("stripe evidence payload differs from its registered SHA-256")
        out[key] = json.loads(envelope["json"])
        if not isinstance(out[key], dict):
            raise ValueError("stripe evidence is not a JSON object")
    return out


def _configuration(config, report, worker, requested, evidence):
    files = {"checkpoint", "receipt", "core", "cards_db", "code_list", "announce_tables", "semantic_file",
             "card_tables", "deck", "attempt", "public_seed", "public_board"}
    fields={"inputs","script_root","out","socket","geometry","seed","follower_seed","particles",
            "depth","repeats","diagnostic_seconds"}
    special_keys={'specialization_request','specialization_search_request'}
    mirror_key={'follower_recipe_law'}
    from ..common.client_public_recipe import LAW as MIRROR_LAW
    if config.get('follower_recipe_law') != requested.follower_recipe_law or (
            'follower_recipe_law' in config and config['follower_recipe_law'] != MIRROR_LAW):
        raise ValueError('stripe evidence differs from the explicit follower recipe law')
    if set(config) not in (fields,fields|special_keys,fields|mirror_key,fields|special_keys|mirror_key) \
            or config.get("geometry") != FIXED_PUBLIC_B64 or type(config.get("repeats")) is not int or config["repeats"] != 3 \
            or any(type(config.get(key)) is not int or config[key] != getattr(requested, key) for key in ("particles", "depth")) \
            or any(type(config.get(key)) is not int or config[key] < 0 for key in ("seed", "follower_seed")) \
            or type(config.get("diagnostic_seconds")) not in (int, float) \
            or not math.isfinite(config["diagnostic_seconds"]) or not 30 <= config["diagnostic_seconds"] <= 300 \
            or any(not isinstance(config.get(k), str) or not Path(config[k]).is_absolute() for k in ("script_root", "out", "socket")) \
            or not isinstance(config.get("inputs"), dict) or set(config["inputs"]) != files \
            or report.get("inputs") != config["inputs"] \
            or any(not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                   or not isinstance(ref["path"], str) or not Path(ref["path"]).is_absolute() or not _sha(ref["sha256"])
                   for ref in config["inputs"].values()) \
            or report.get("config_sha256") != evidence["config"]["sha256"] \
            or worker.get("config_sha256") != evidence["config"]["sha256"]:
        raise ValueError("stripe audit configuration differs from the original full-bank/geometry registration")
    identity=worker.get('identity',{})
    if identity.get('backend')=='specialization':
        ready=identity.get('specialization_search',{})
        if not special_keys<=set(config) or config['specialization_request']!=ready.get('behavior_request') \
                or type(config['specialization_search_request']) is not dict \
                or set(config['specialization_search_request'])!={'path','sha256'} \
                or config['specialization_search_request']['sha256']!=ready.get('request_sha256'):
            raise ValueError('stripe specialization config differs from its actual loaded actor requests')
    elif special_keys&set(config):
        raise ValueError('ordinary stripe identity cannot silently contain specialization requests')


def _worker_binding(report, worker, identity):
    expected_extra = {"parameters_after", "sessions_remaining", "dropped_sessions", "clean_stop"}
    ready = report.get("worker")
    keys=identity_keys(identity)
    if not isinstance(ready, dict) or set(worker) != set(ready) | expected_extra \
            or any(worker.get(k) != value for k, value in ready.items()) \
            or worker.get("schema") != AUDIT_SCHEMA or worker.get("role") != "worker" \
            or worker.get("clean_stop") is not True \
            or type(worker.get("sessions_remaining")) is not int or worker["sessions_remaining"] != 0 \
            or type(worker.get("dropped_sessions")) is not int or worker["dropped_sessions"] != 0 \
            or not _sha(worker.get("parameters_before")) or worker.get("parameters_after") != worker["parameters_before"] \
            or not isinstance(worker.get("identity"), dict) or set(worker["identity"]) != keys \
            or any(worker["identity"][key] != identity.get(key) for key in keys) \
            or worker.get("client_config") != identity.get("client_config") \
            or worker.get("native_build") != identity.get("native_build") \
            or worker.get("native_build", {}).get("outputs", {}).get("duel_native.cpython-311-x86_64-linux-gnu.so") \
                != identity.get("serving_native_sha256") \
            or worker.get("device") != identity.get("device") \
            or worker.get("device", {}).get("platform") != "gpu" or worker["device"].get("count") != 1:
        raise ValueError("stripe audit is incomplete or belongs to different model/native/device/session inputs")


def _runs(report, config):
    runs = report.get("runs", [])
    if not isinstance(runs, list) or len(runs) != 8 or any(not isinstance(r, dict) for r in runs) \
            or {(r.get("mode"), r.get("repeat")) for r in runs} != {
                (mode, repeat) for mode in ("serial", "multiplexed") for repeat in range(4)}:
        raise ValueError("stripe gate requires the traced pair and all three timing pairs")
    reference = next(r for r in runs if r["mode"] == "serial" and r["repeat"] == 0)
    values = reference.get("values")
    if not isinstance(values, list) or len(values) < 2 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("stripe reference lacks finite values for every candidate")
    rows = len(values)
    paths = reference.get("native_paths")
    if not isinstance(paths, dict) or set(paths) != {f"{i}:{row}" for i in range(config["particles"]) for row in range(rows)} \
            or any(not isinstance(path, list) or not path for path in paths.values()) \
            or not _sha(reference.get("assignment_sha256")):
        raise ValueError("stripe audit omitted original candidate native paths or assignments")
    for run in runs:
        audit = run.get("audit", {})
        check = A.check_stripes if run["mode"] == "serial" else B.check_stripes
        if check(audit, rows=rows, planned=config["particles"], seed=config["seed"]) != config["particles"] \
                or type(run.get("repeat")) is not int or type(run.get("traced")) is not bool \
                or run["traced"] != (run["repeat"] == 0) or run.get("native_root_restored") is not True \
                or run.get("own_session_unchanged") is not True \
                or type(run.get("seconds")) not in (int, float) or not math.isfinite(run["seconds"]) or run["seconds"] <= 0 \
                or any(_canonical_sha(run.get(key)) != _canonical_sha(reference.get(key))
                       for key in ("values", "native_paths", "assignment_sha256")):
            raise ValueError("stripe audit changed a full path/value/assignment or mutated the original root")
        if run["mode"] == "multiplexed" and any(audit["resources"].get(k) != v for k,v in B.DEFAULT_LIMITS.items()):
            raise ValueError("stripe audit used different live resource limits")
    traced = next(r for r in runs if r["mode"] == "multiplexed" and r["repeat"] == 0)
    trace = reference.get("trace", {})
    if not isinstance(trace, dict) or not trace or _canonical_sha(trace) != _canonical_sha(traced.get("trace")):
        raise ValueError("stripe gate has different per-session public inputs, memory, RNG or network outputs")
    seats, firsts, forwards = set(), set(), 0
    for key, sequence in trace.items():
        try:
            index, row, seat = (int(part) for part in key.split(":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("stripe trace lost its particle/row/seat binding") from exc
        if not 0 <= index < config["particles"] or not 0 <= row < rows or seat not in (0,1) or not sequence:
            raise ValueError("stripe trace is outside the declared full bank")
        seats.add(seat)
        previous = None
        for item in sequence:
            before, pending = item.get("before", {}), item.get("pending", {})
            if any(not _sha(value) for value in (before.get("observation"), before.get("memory"), item.get("memory_after"))) \
                    or pending.get("obs_sha256") != before["observation"] or type(before.get("first")) is not bool \
                    or not isinstance(before.get("rng"), dict) or previous is not None and before["memory"] != previous:
                raise ValueError("stripe trace lacks exact actual public memory/observation continuity")
            previous = item["memory_after"]
            firsts.add(before["first"])
            forwards += 1
    if seats != {0,1} or firsts != {False,True}:
        raise ValueError("stripe audit lacks both seats and actual noninitial memories")
    for run in runs:
        batches = run.get("score_batches")
        if not isinstance(batches, list) or not batches or any(not isinstance(batch, dict)
                or type(batch.get("items")) is not int or batch["items"] < 1
                or type(batch.get("seconds")) not in (int, float) or not math.isfinite(batch["seconds"])
                or batch["seconds"] < 0 for batch in batches) or sum(b["items"] for b in batches) != forwards:
            raise ValueError("stripe audit did not perform the same full model work in each run")
    times = {mode:[r["seconds"] for r in runs if r["mode"] == mode and r["repeat"] > 0]
             for mode in ("serial","multiplexed")}
    if report.get("timed_seconds") != times or report.get("median_seconds") != {
            mode:statistics.median(values) for mode,values in times.items()}:
        raise ValueError("stripe timing summary differs from its full unfiltered runs")
    return rows, forwards, _canonical_sha(trace)


def stripe_contract(identity, config, evidence):
    from .agent_specialization_search import registered_service
    specialization=registered_service(identity)
    if config.budget_law != B.LAW or config.selection != "greedy" or config.td_lambda != 1. \
            or identity.get("current_root") != CAPABILITY or identity.get("inference_geometry") != FIXED_PUBLIC_B64 \
            or identity.get("weights") != "iterate" \
            or identity.get("compute_dtype") != "bfloat16":
        raise ValueError("multiplexed scheduling requires the explicit current-root raw BF16/B64 rollout law")
    decoded = _decode(evidence)
    report, worker, original = (decoded[key] for key in ("report", "worker", "config"))
    if report.get("schema") != AUDIT_SCHEMA or any(report.get(k) is not True for k in (
            "all_exact", "same_full_paths_values_and_assignments", "same_public_inputs_memory_rng_and_outputs")) \
            or any(report.get(k) is not False for k in ("training_eligible", "search_admission", "turn_budget_accepted")):
        raise ValueError("stripe audit did not pass its exact finite diagnostic gate")
    _configuration(original, report, worker, config, evidence)
    _worker_binding(report, worker, identity)
    source_checkpoint=identity["checkpoint_sha256"] if specialization is None else specialization['parent_checkpoint_sha256']
    if report["inputs"]["checkpoint"]["sha256"] != source_checkpoint \
            or report["inputs"]["core"]["sha256"] != identity["native_build"]["outputs"]["libmfcore.so"]:
        raise ValueError("stripe source input checksums differ from the registered model/core")
    rows, forwards, trace_sha = _runs(report, original)
    return {"schema": MIRROR_SCHEMA if config.follower_recipe_law else SCHEMA,
            **({'follower_recipe_law': config.follower_recipe_law} if config.follower_recipe_law else {}),
            "law": B.LAW, "resources": dict(B.RESOURCES), "passed_exact": True,
            "report_sha256": evidence["report"]["sha256"], "worker_sha256": evidence["worker"]["sha256"],
            "config_sha256": evidence["config"]["sha256"], "parameters_sha256": worker["parameters_after"],
            "trace_sha256": trace_sha, "candidate_rows": rows, "trace_forwards": forwards,
            "depth": config.depth, "particles": config.particles, "td_lambda": 1.,
            "inference_geometry": dict(FIXED_PUBLIC_B64),
            "scope": "registered finite original public current-root full-bank continuations only",
            "full_search_budget_accepted": False}


def check_registration(contract, config):
    """Validate the compact, parent-issued registration carried with each game."""
    digests = {"report_sha256", "worker_sha256", "config_sha256", "parameters_sha256", "trace_sha256"}
    keys = digests | {"schema", "law", "resources", "passed_exact", "candidate_rows", "trace_forwards", "depth",
                      "particles", "td_lambda", "inference_geometry", "scope", "full_search_budget_accepted"}
    if config.follower_recipe_law:
        keys |= {'follower_recipe_law'}
    if not isinstance(contract, dict) or set(contract) != keys \
            or contract.get('follower_recipe_law') != config.follower_recipe_law \
            or contract.get("schema") != (MIRROR_SCHEMA if config.follower_recipe_law else SCHEMA) \
            or contract.get("law") != B.LAW or contract.get("resources") != B.RESOURCES \
            or contract.get("inference_geometry") != FIXED_PUBLIC_B64 or contract.get("passed_exact") is not True \
            or contract.get("full_search_budget_accepted") is not False or any(not _sha(contract[k]) for k in digests) \
            or any(type(contract.get(k)) is not int or contract[k] < 2 for k in ("candidate_rows", "trace_forwards")) \
            or any(type(contract.get(k)) is not int or contract[k] != getattr(config,k) for k in ("particles", "depth")) \
            or type(contract.get("td_lambda")) not in (int,float) \
            or contract["td_lambda"] != config.td_lambda or contract["td_lambda"] != 1. \
            or contract.get("scope") != "registered finite original public current-root full-bank continuations only":
        raise ValueError("current-root identity lacks its exact finite multiplexed-stripe registration")
    return contract
