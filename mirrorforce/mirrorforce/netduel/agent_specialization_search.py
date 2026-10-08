"""Explicit diagnostic current-root search binding for a behavior-only branch.

The parent owns training rules; the specialization owns actual actor parameters.
This opt-in never admits A0 resume, an old AR head, or competition deployment.
Native/numeric/stripe/complete-game evidence must be checked independently.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from ..agent.inference_geometry import FIXED_PUBLIC_B64, validate_geometry
from .current_root_protocol import PARTICLE_IDENTITY

INFORMATION_SET_SEARCH = True
SCHEMA="mirrorforce_specialization_search_request/v1"
LAW="parent-training-rules;actual-specialization-actor;public-uniform-only/v1"
BUDGET={"triggers":["uncertain","public_lethal_candidate"],"uncertain_seconds":8.,"lethal_seconds":12.,
        "reserve_seconds":3.,"soft_cap_seconds":9.,"turn_admission_seconds":30.,
        "hard_turn_seconds":600,"whole_duel_seconds":1800}
FIELDS={"schema","law","behavior_request","parent_checkpoint_sha256","branch_checkpoint_sha256",
        "parameter_sha256","particles","inference_geometry","budget","diagnostic_only",
        "production_admitted","training_resume_admitted"}


def sha(value):
    return type(value) is str and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def checked(value):
    """Closed shape; preserving an explicit prototype is not search admission."""
    if type(value) is not dict or set(value)!=FIELDS or value["schema"]!=SCHEMA or value["law"]!=LAW \
            or any(not sha(value[k]) for k in ("parent_checkpoint_sha256","branch_checkpoint_sha256","parameter_sha256")) \
            or value["branch_checkpoint_sha256"]==value["parent_checkpoint_sha256"] \
            or value["particles"]!=PARTICLE_IDENTITY or validate_geometry(value["inference_geometry"])!=FIXED_PUBLIC_B64 \
            or value["budget"]!=BUDGET or value["diagnostic_only"] is not True \
            or value["production_admitted"] is not False or value["training_resume_admitted"] is not False:
        raise ValueError("explicit diagnostic specialization/uniform/B64/original-budget contract required")
    ref=value["behavior_request"]
    if type(ref) is not dict or set(ref)!={"path","sha256"} or type(ref["path"]) is not str \
            or not Path(ref["path"]).is_absolute() or not sha(ref["sha256"]):
        raise ValueError("specialization search must pin its exact behavior request")
    return copy.deepcopy(value)


def read_request(parent,reference,*,behavior_reference):
    """Retain the full behavior loader's original scene/result/parent checks."""
    from ..agent.train.specialization_policy import read_ref,read_request as behavior_request
    request=checked(json.loads(read_ref(reference).read_bytes()))
    if request["behavior_request"]!=behavior_reference:
        raise ValueError("specialization search names another behavior request")
    behavior,result,_=behavior_request(parent,behavior_reference)
    if request["parent_checkpoint_sha256"]!=behavior["parent_checkpoint_sha256"] \
            or request["branch_checkpoint_sha256"]!=behavior["branch"]["sha256"] \
            or request["parameter_sha256"]!=result["candidate_parameter_sha256"]:
        raise ValueError("specialization search differs from the completed branch/parent/parameters")
    return request


def bind_ready(request,identity,*,request_sha256):
    """Bind actual loaded backend identity; no checkpoint backend relabeling."""
    value=checked(request)
    specialization=identity.get("specialization",{})
    if not sha(request_sha256) or identity.get("backend")!="specialization" or identity.get("weights")!="iterate" \
            or identity.get("checkpoint_sha256")!=value["branch_checkpoint_sha256"] \
            or identity.get("parent_checkpoint_sha256")!=value["parent_checkpoint_sha256"] \
            or specialization.get("checkpoint_sha256")!=value["branch_checkpoint_sha256"] \
            or specialization.get("parent_checkpoint_sha256")!=value["parent_checkpoint_sha256"] \
            or specialization.get("parameter_sha256")!=value["parameter_sha256"] \
            or specialization.get("request_sha256")!=value["behavior_request"]["sha256"] \
            or specialization.get("training_resume_admitted") is not False \
            or identity.get("inference_geometry")!=value["inference_geometry"] \
            or any(key in identity for key in ("ar_head","ar_features","replay_belief")):
        raise ValueError("specialization search READY differs from the actual branch or contains an old belief head")
    return {"schema":SCHEMA+"#identity","law":LAW,"request_sha256":request_sha256,
            "parent_checkpoint_sha256":value["parent_checkpoint_sha256"],
            "branch_checkpoint_sha256":value["branch_checkpoint_sha256"],
            "parameter_sha256":value["parameter_sha256"],"behavior_request_sha256":value["behavior_request"]["sha256"],
            "behavior_request":value["behavior_request"],
            "particles":value["particles"],"budget":value["budget"],"inference_geometry":value["inference_geometry"],
            "diagnostic_only":True,"production_admitted":False,"training_resume_admitted":False}


def registered_service(identity):
    """Default checkpoint path stays closed to fabricated specialization metadata."""
    if identity.get("backend")=="checkpoint":
        if "specialization" in identity or "specialization_search" in identity:
            raise ValueError("a specialization cannot be relabeled as its parent checkpoint")
        return None
    ready=identity.get("specialization_search")
    if identity.get("backend")!="specialization" or type(ready) is not dict:
        raise ValueError("specialization search has no explicit registered diagnostic request")
    try:
        request={key:ready[key] for key in FIELDS-{"schema"}}
        request["schema"]=SCHEMA
        expected=bind_ready(request,identity,request_sha256=ready["request_sha256"])
    except (KeyError,TypeError) as exc:
        raise ValueError("specialization search READY is incomplete") from exc
    if ready!=expected:
        raise ValueError("specialization search READY changed or contains undeclared fields")
    return ready


def registration(identity,particles,settings):
    """Bind the actor to the legacy or exact final-human client configuration.

    READY remains the original immutable actor/provenance receipt. Its budget
    is not relabeled: actual execution limits live in the independently pinned
    search identity's settings. Neither this check nor numeric/stripe evidence
    admits the new profile to a room; it still needs its own complete-game gate.
    """
    ready=registered_service(identity)
    if ready is None:
        return None
    from .final_search import settings as final_human_settings
    if particles == PARTICLE_IDENTITY and settings == final_human_settings('on'):
        return ready
    from .agent_final_room_budget import settings as final_room_settings
    if particles == PARTICLE_IDENTITY and settings == final_room_settings('on'):
        return ready
    from .agent_opponent_clear import settings as opponent_clear_settings
    from .agent_opponent_known import settings as opponent_known_settings
    if particles == PARTICLE_IDENTITY and settings == opponent_known_settings('on'):
        return ready
    if particles == PARTICLE_IDENTITY and settings == opponent_clear_settings('on'):
        return ready
    gate=settings.get("on_demand")
    if particles!=PARTICLE_IDENTITY or type(gate) is not dict or gate.get("enabled") is not True \
            or any(gate.get(k)!=v for k,v in {"uncertain_seconds":8.,"lethal_seconds":12.,
                "finalize_seconds":3.,"turn_seconds":30.,"entropy_threshold":.6,"margin_threshold":.2}.items()) \
            or any(settings.get(k)!=v for k,v in {"seconds":9.,"total_seconds":12.,"response_margin":3.,
                "finalize_seconds":3.,"particles":8,"rollouts":50,"depth":5,"selection":"greedy",
                "candidate_law":"all-legal-common-bank/v1","budget_law":"multiplexed-balanced-prefix/v1",
                "bank_unknown_law":"fatal/v1","estimator":"rollout"}.items()) \
            or settings.get("tactical") is not None:
        raise ValueError("specialization search is limited to its unchanged uniform uncertain/lethal recipe")
    return ready


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
