"""A0 reporting/checkpoint counts in the recorded parent's coordinate system.

Actor counters remain untouched. A restarted actor begins at zero; restored
actors can carry either a cumulative count or a run-relative one. Their common
stored count determines the offset, never the iteration number or an inferred
historical lifetime. No old checkpoint/receipt is changed by this module.
"""
from __future__ import annotations

import operator

SCHEMA = "mirrorforce_a0_counter_basis/v1"


def _count(value, name):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer count") from exc
    if isinstance(value, bool) or result < 0:
        raise ValueError(f"{name} must be a nonnegative integer count")
    return int(result)


def initial_basis(offset=0):
    return {"schema": SCHEMA, "law": "offset-plus-actor-count/v1", "offset": _count(offset, "offset"),
            "parent_checkpoint": None, "parent_global_step": None, "actor_start_global_step": 0,
            "actors_restored": False, "origin": "configured_offset", "history_reconstructed": False}


def resume_basis(parent_checkpoint, counters, actors, *, actor_threads, envs_per_actor, expected_update):
    """Bind a new run to the recorded parent without rebasing the actor snapshots.

    Call after declared environment/topology resets have discarded ``actors``.
    Otherwise every restored actor must still describe the current batch shape
    and the same iteration boundary. Reject disagreement instead of choosing a
    max/sum or silently treating incompatible actor states as fresh ones.
    """
    recorded = _count(counters["global_step"], "parent global_step")
    start = 0
    if actors is not None:
        if not actors or len(actors) != _count(actor_threads, "actor_threads"):
            raise ValueError("restored actor thread count differs; declare an environment/topology reset")
        envs_per_actor = _count(envs_per_actor, "envs_per_actor")
        expected_update = _count(expected_update, "expected_update")
        if envs_per_actor == 0 or expected_update == 0:
            raise ValueError("restored actor batch and update must be positive")
        steps = []
        for actor in actors:
            if len(actor["envs"]) != envs_per_actor:
                raise ValueError("restored actor environment batch differs; declare an environment/topology reset")
            if _count(actor["update"], "actor update") != expected_update:
                raise ValueError("restored actor update differs from the A0 iteration boundary")
            steps.append(_count(actor["global_step"], "actor global_step"))
        if len(set(steps)) != 1:
            raise ValueError("restored actors have divergent global_step counters")
        start = steps[0]
        if start > recorded:
            raise ValueError("actor global_step exceeds the recorded parent; no historical correction is inferred")
    basis = initial_basis(recorded - start)
    basis.update(parent_checkpoint=str(parent_checkpoint), parent_global_step=recorded,
                 actor_start_global_step=start, actors_restored=actors is not None, origin="recorded_parent")
    return basis


def iteration_counters(basis, actor_global_step, iteration):
    """The single counter record shared by A0's log and checkpoint writes."""
    if basis.get("schema") != SCHEMA or basis.get("law") != "offset-plus-actor-count/v1":
        raise ValueError("unknown A0 counter basis")
    current = _count(actor_global_step, "actor global_step")
    if current < _count(basis["actor_start_global_step"], "actor start"):
        raise ValueError("actor global_step regressed behind its restored boundary")
    return {"global_step": _count(basis["offset"], "offset") + current,
            "learner_update": _count(iteration, "iteration")}
