"""Bounded B64-friendly multiplexing of existing common-stripe continuations.

This is an explicit scheduler law, not a new model geometry or rollout policy.
The complete immutable bank is still admitted by the caller. A fixed wave of
owners yields their real waiting items, which share one service RPC. Only the
ordered, fully certified prefix contributes to Q; faster later particles may
never skip an unfinished earlier particle. No native/transport/cleanup failure
is downgraded to a time budget. Serving registration remains separate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

from .agent_rollout import MAX_ITEMS, RootLines, RolloutBudgetExceeded, RolloutError, q_values

LAW = "multiplexed-balanced-prefix/v1"
RESOURCE_LAW = "bounded-multiplexed-stripe-lanes/v1"
INFORMATION_SET_SEARCH = True
DEFAULT_LIMITS = {"max_live_lines": 32, "lines_per_stripe": 8, "max_active_stripes": 8}
RESOURCES = {"law": RESOURCE_LAW, **DEFAULT_LIMITS, "base_sessions": "one-per-active-stripe",
             "partial_candidate_returns": "discard-entire-stripe", "estimates": "ordered-whole-prefix"}


def empty_stripes(rows):
    """A producer-expired current root, before any particle was admitted."""
    _, width = _limits(rows, DEFAULT_LIMITS)
    return {"law": LAW, "planned_stripes": 0, "completed_stripes": 0, "completed_indices": [],
            "budget_exhausted": False, "budget_zero_search": True, "stripes": [],
            "stats": {"lines": 0, "attempted_lines": 0, "rounds": 0, "items": 0,
                      "service_seconds": 0., "engine_seconds": 0., "void": 0, "void_logs": []},
            "resources": {"law": RESOURCE_LAW, **DEFAULT_LIMITS, "wave_width": width, "waves": 0,
                          "peak_active_stripes": 0, "reserved_live_lines_peak": 0,
                          "rollout_calls": 0, "peak_rpc_items": 0}}


def _limits(rows, limits):
    if type(rows) is not int or rows < 2 or set(limits) != set(DEFAULT_LIMITS) \
            or any(type(v) is not int or v < 1 for v in limits.values()) \
            or limits["max_live_lines"] > MAX_ITEMS \
            or limits["lines_per_stripe"] > limits["max_live_lines"] \
            or limits["max_active_stripes"] > MAX_ITEMS:
        raise ValueError("multiplexed stripes require explicit bounded lane and owner limits")
    lanes = min(rows, limits["lines_per_stripe"])
    return lanes, min(limits["max_active_stripes"], limits["max_live_lines"] // lanes)


def _check_owner(owner, *, index, seed, rows, lanes):
    if not isinstance(owner, RootLines) or owner.closed or owner.seed_law != "common-stripe/v1" \
            or owner.rows != rows or owner.seed != seed or owner.starts != [index] \
            or owner.max_live_lines != lanes or len(owner.lines) != rows \
            or any(line.start != index or line.row != row or line.sessions or line.snap or line.end is not None
                   for row, line in enumerate(owner.lines)):
        raise RolloutError("owner does not implement the fresh predeclared bounded common stripe")


def _check_values(values, *, index, rows):
    if not isinstance(values, dict) or set(values) != set(range(rows)) \
            or any(len(values[row]) != 1 or values[row][0][0] != index
                   or type(values[row][0][1]) not in (int, float)
                   or not math.isfinite(values[row][0][1]) for row in range(rows)):
        raise RolloutError("completed stripe needs exactly one finite value for every candidate")


@dataclass
class _Stripe:
    index: int
    seed: int
    owner: RootLines
    began: float
    batches: object = None
    pending: dict | None = None
    values: dict | None = None
    certified: bool = False
    seconds: float = 0.

    def advance(self, reply, *, deadline, clock):
        try:
            self.pending = next(self.batches) if reply is None else self.batches.send(reply)
        except StopIteration as completed:
            _check_values(completed.value, index=self.index, rows=self.owner.rows)
            if self.owner.stats["void"]:
                raise RolloutError("a completed stripe contains an invalid native line")
            self.values = completed.value
            self.certified = clock() < deadline
            self.pending = None
            self.seconds = clock() - self.began
        if self.pending is not None and (self.pending.get("op") != "rollout_step"
                or not isinstance(self.pending.get("items"), list)
                or not 1 <= len(self.pending["items"]) <= self.owner.max_live_lines):
            raise RolloutError("stripe yielded a request outside its lane reservation")


class _Dispatch:
    """One clock for ALL service calls, including factory-owned bases/cleanup."""
    def __init__(self, call, clock):
        self.call, self.clock = call, clock
        self.seconds, self.rollout_calls, self.peak_items = 0., 0, 0

    def __call__(self, request):
        if request.get("op") == "rollout_step":
            self.rollout_calls += 1
            self.peak_items = max(self.peak_items, len(request["items"]))
        began = self.clock()
        try:
            return self.call(request)
        finally:
            self.seconds += self.clock() - began


def _close_wave(wave):
    errors = []
    for stripe in wave:
        for close in ((stripe.batches.close,) if stripe.batches is not None else ()) + (stripe.owner.close,):
            try:
                close()
            except BaseException as exc:
                errors.append(exc)
    if errors:
        raise RolloutError("multiplexed cleanup failed after attempting every owner") from errors[0]


def _drive_wave(wave, *, dispatch, deadline, clock, lanes, make_runner, begin, stop, seed, rows):
    """Keep native advancement inside each existing owner; only concatenate RPCs."""
    for index in range(begin, stop):
        if clock() >= deadline:
            raise RolloutBudgetExceeded("multiplexed budget expired before opening a stripe")
        stripe_seed = (seed * 1_000_003 + index) & 0x7fffffff
        began = clock()
        owner = make_runner(index, stripe_seed, dispatch)
        stripe = _Stripe(index, stripe_seed, owner, began)
        wave.append(stripe)  # register before validation, clocks or generator initialization
        _check_owner(owner, index=index, seed=stripe_seed, rows=rows, lanes=lanes)
        owner._call = dispatch
        stripe.batches = owner.batches(deadline=deadline)
        stripe.advance(None, deadline=deadline, clock=clock)
    while any(stripe.pending is not None for stripe in wave):
        if clock() >= deadline:
            raise RolloutBudgetExceeded("multiplexed budget expired before shared RPC")
        waiting = [stripe for stripe in wave if stripe.pending is not None]
        items = [item for stripe in waiting for item in stripe.pending["items"]]
        # Sessions/RNGs belong to one line only, including within a common seed.
        if len({item["session"] for item in items}) != len(items):
            raise RolloutError("multiplexed batch contains duplicate live sessions")
        reply = dispatch({"op": "rollout_step", "items": items})
        if not isinstance(reply, dict) or not isinstance(reply.get("items"), list) \
                or len(reply["items"]) != len(items):
            raise RolloutError("the shared service dropped or added rollout items")
        if clock() >= deadline:
            raise RolloutBudgetExceeded("multiplexed budget expired during shared RPC")
        offset = 0
        for stripe in waiting:
            count = len(stripe.pending["items"])
            stripe.advance({"items": reply["items"][offset:offset + count]}, deadline=deadline, clock=clock)
            offset += count


def multiplexed_stripes(*, rows, weights, seed, deadline, make_runner, call, limits=None, clock=None):
    """Play fixed waves of bounded owners and return Q plus a new-law audit.

    ``make_runner(index, stripe_seed, dispatch)`` must return a fresh RootLines
    with ``max_live_lines=min(rows, limits['lines_per_stripe'])``. Use dispatch
    for ALL service calls, including opponent clones and factory-owned bases.
    The factory owns and must clean allocations if it fails before returning;
    after return the scheduler always closes the owner (and its wrapped bases).
    The service's hard deadline must leave cleanup/real-response time beyond
    the supplied soft deadline. No production registration is implied here.
    """
    clock = time.monotonic if clock is None else clock
    limits = dict(DEFAULT_LIMITS) if limits is None else dict(limits)
    lanes, width = _limits(rows, limits)
    if type(weights) is not tuple or not weights \
            or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights) \
            or type(seed) is not int or type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("multiplexed stripes require a fixed positive bank, seed and finite deadline")
    dispatch = _Dispatch(call, clock)
    returns = {row: [] for row in range(rows)}
    stripes = []
    stats = {"lines": 0, "attempted_lines": 0, "rounds": 0, "items": 0,
             "service_seconds": 0., "engine_seconds": 0., "void": 0, "void_logs": []}
    resources = {"law": RESOURCE_LAW, **limits, "wave_width": width, "waves": 0,
                 "peak_active_stripes": 0, "reserved_live_lines_peak": 0,
                 "rollout_calls": 0, "peak_rpc_items": 0}
    for begin in range(0, len(weights), width):
        if clock() >= deadline:
            break
        wave, expired = [], False
        try:
            try:
                _drive_wave(wave, dispatch=dispatch, deadline=deadline, clock=clock, lanes=lanes,
                    make_runner=make_runner, begin=begin, stop=min(begin + width, len(weights)), seed=seed, rows=rows)
            except RolloutBudgetExceeded:
                if any(stripe.owner.stats.get("void") for stripe in wave):
                    raise RolloutError("a timed multiplexed wave contains an invalid native line")
                expired = True
        finally:
            _close_wave(wave)
        resources["waves"] += bool(wave)
        resources["peak_active_stripes"] = max(resources["peak_active_stripes"], len(wave))
        resources["reserved_live_lines_peak"] = resources["peak_active_stripes"] * lanes
        for stripe in wave:
            complete = stripe.certified and stripe.index == len(returns[0])
            if complete:
                for row in range(rows):
                    returns[row].extend(stripe.values[row])
                stats["lines"] += rows
            stripes.append({"index": stripe.index, "seed": stripe.seed, "complete": complete,
                "certified": stripe.certified,
                "seconds": stripe.seconds if stripe.certified else clock() - stripe.began,
                "finished_lines": sum(line.value is not None for line in stripe.owner.lines),
                "discarded_lines": 0 if complete else rows})
            stats["attempted_lines"] += rows
            for key in ("rounds", "items", "engine_seconds"):
                stats[key] += stripe.owner.stats[key]
        if expired or len(returns[0]) < begin + len(wave):
            break
    count = len(returns[0])
    stats["service_seconds"] = dispatch.seconds
    resources.update(rollout_calls=dispatch.rollout_calls, peak_rpc_items=dispatch.peak_items)
    estimates = q_values(returns, rows, dict(enumerate(weights))) if count else [None] * rows
    return estimates, {"law": LAW, "planned_stripes": len(weights), "completed_stripes": count,
        "budget_exhausted": count < len(weights), "budget_zero_search": count == 0,
        "completed_indices": list(range(count)), "stripes": stripes, "stats": stats, "resources": resources}


def check_stripes(record, *, rows, planned, seed):
    """Independently validate the multiplexed ordered-prefix and bound certificate."""
    if type(seed) is not int or not isinstance(record, dict) or record.get("law") != LAW \
            or type(record.get("planned_stripes")) is not int or record["planned_stripes"] != planned:
        raise ValueError("different multiplexed-stripe registration")
    resources = record.get("resources", {})
    if not isinstance(resources, dict):
        raise ValueError("multiplexed resource certificate is not an object")
    lanes, width = _limits(rows, {key: resources.get(key) for key in DEFAULT_LIMITS})
    count, stripes = record.get("completed_stripes"), record.get("stripes")
    if type(planned) is not int or planned < 0 or type(count) is not int or not 0 <= count <= planned \
            or not isinstance(stripes, list) or not count <= len(stripes) <= min(count + width, planned) \
            or record.get("completed_indices") != list(range(count)) \
            or type(record.get("budget_zero_search")) is not bool or record["budget_zero_search"] != (count == 0) \
            or type(record.get("budget_exhausted")) is not bool or record["budget_exhausted"] != (count < planned):
        raise ValueError("multiplexed completion differs from the ordered whole prefix")
    for index, stripe in enumerate(stripes):
        complete = index < count
        if not isinstance(stripe, dict) or type(stripe.get("index")) is not int or stripe["index"] != index \
                or type(stripe.get("seed")) is not int or stripe["seed"] != (seed * 1_000_003 + index) & 0x7fffffff \
                or type(stripe.get("complete")) is not bool or stripe["complete"] != complete \
                or type(stripe.get("certified")) is not bool or complete and not stripe["certified"] \
                or index == count and stripe["certified"] \
                or type(stripe.get("finished_lines")) is not int or not 0 <= stripe["finished_lines"] <= rows \
                or stripe["certified"] and stripe["finished_lines"] != rows \
                or type(stripe.get("discarded_lines")) is not int or stripe["discarded_lines"] != (0 if complete else rows) \
                or type(stripe.get("seconds")) not in (int, float) or not math.isfinite(stripe["seconds"]) \
                or stripe["seconds"] < 0:
            raise ValueError("multiplexed stripe seed/coverage/discard certificate differs")
    stats = record.get("stats", {})
    if not isinstance(stats, dict) or type(stats.get("void")) is not int or stats["void"] != 0 \
            or stats.get("void_logs") != [] or stats.get("lines") != rows * count \
            or stats.get("attempted_lines") != rows * len(stripes) \
            or any(type(stats.get(key)) is not int or stats[key] < 0 for key in ("rounds", "items", "lines", "attempted_lines")) \
            or any(type(stats.get(key)) not in (int, float) or not math.isfinite(stats[key]) or stats[key] < 0
                   for key in ("service_seconds", "engine_seconds")):
        raise ValueError("multiplexed resource/value counts differ")
    if resources.get("law") != RESOURCE_LAW or resources.get("wave_width") != width \
            or any(type(resources.get(key)) is not int or resources[key] < 0
                   for key in ("wave_width", "waves", "peak_active_stripes", "reserved_live_lines_peak", "rollout_calls", "peak_rpc_items")) \
            or resources["waves"] != (len(stripes) + width - 1) // width \
            or resources["peak_active_stripes"] != min(len(stripes), width) \
            or resources["reserved_live_lines_peak"] != resources["peak_active_stripes"] * lanes \
            or resources["reserved_live_lines_peak"] > resources["max_live_lines"] \
            or resources["peak_rpc_items"] > resources["reserved_live_lines_peak"] \
            or resources["rollout_calls"] > stats["rounds"]:
        raise ValueError("multiplexed wave reservations exceed their declared bounds")
    return count
