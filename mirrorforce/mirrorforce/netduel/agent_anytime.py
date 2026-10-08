"""Explicit time-limited search law, independent of the particle producer.

The complete immutable bank must be admitted before entering this module.
Only a whole prefix of balanced stripes enters Q. A late or interrupted stripe
is discarded for EVERY candidate. Native, information and transport errors
are not clock exhaustion and propagate. No games are filtered by this law.
"""
from __future__ import annotations

from collections import Counter
import math
import time

import numpy as np

from .agent_rollout import RolloutBudgetExceeded, RolloutError, q_values

LAW = "completed-balanced-stripes/v1"
INFORMATION_SET_SEARCH = True


def balanced_stripes(*, rows, weights, seed, deadline, make_runner, clock=None):
    """Run one fixed particle/seed across all candidates, then the next.

    `make_runner(index, seed)` must return a RootLines owner with exactly this
    stripe's rows, one line per row, one start equal to index and common seeds.
    It must not create sessions until run(). close() always runs, even when the
    clock expires, and a cleanup error remains fatal. Its service RPC deadline
    must leave time for cleanup and the real response beyond `deadline`.
    """
    clock = time.monotonic if clock is None else clock
    if type(rows) is not int or rows < 2 or type(weights) is not tuple or not weights \
            or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights) \
            or type(seed) is not int or type(deadline) not in (float, int) or not math.isfinite(deadline):
        raise ValueError("balanced stripes require a fixed positive bank, seed and absolute deadline")
    returns = {row: [] for row in range(rows)}
    stripes, stats = [], {"lines": 0, "attempted_lines": 0, "rounds": 0, "items": 0,
                          "service_seconds": 0.0, "engine_seconds": 0.0, "void": 0, "void_logs": []}
    exhausted = False
    for index in range(len(weights)):
        if clock() >= deadline:
            exhausted = True
            break
        stripe_seed = (seed * 1_000_003 + index) & 0x7fffffff
        runner = make_runner(index, stripe_seed)
        completed, values = False, None
        began = clock()
        try:
            if runner.seed_law != "common-stripe/v1" or runner.rows != rows \
                    or runner.seed != stripe_seed \
                    or runner.starts != [index] or len(runner.lines) != rows \
                    or any(line.start != index or line.row != row for row, line in enumerate(runner.lines)):
                raise RolloutError("runner does not implement the predeclared balanced stripe")
            try:
                values = runner.run(deadline=deadline)
            except RolloutBudgetExceeded:
                # Hard failures found before the clock expired cannot be hidden
                # by a later clock check in the same runner.
                if runner.stats.get("void"):
                    raise RolloutError("a timed stripe also contains an invalid native line")
                exhausted = True
            else:
                if set(values) != set(range(rows)) or any(len(values[row]) != 1
                        or values[row][0][0] != index or type(values[row][0][1]) not in (int, float)
                        or not math.isfinite(values[row][0][1]) for row in range(rows)):
                    raise RolloutError("completed stripe did not return exactly one finite value for every row")
                if runner.stats.get("void"):
                    raise RolloutError("a completed stripe contains an invalid native line")
                completed = clock() < deadline
                exhausted = not completed
        finally:
            runner.close()
        stripes.append({"index": index, "seed": stripe_seed, "complete": completed,
                        "seconds": clock() - began, "finished_lines": sum(line.value is not None for line in runner.lines),
                        "discarded_lines": 0 if completed else rows})
        stats["attempted_lines"] += rows
        for key in ("rounds", "items", "service_seconds", "engine_seconds"):
            stats[key] += runner.stats[key]
        if not completed:
            break
        for row in range(rows):
            returns[row].extend(values[row])
        stats["lines"] += rows
    completed_count = len(returns[0])
    estimates = q_values(returns, rows, dict(enumerate(weights))) if completed_count else [None] * rows
    return estimates, {"law": LAW, "planned_stripes": len(weights), "completed_stripes": completed_count,
                       "budget_exhausted": exhausted, "budget_zero_search": completed_count == 0,
                       "completed_indices": list(range(completed_count)), "stripes": stripes,
                       "stats": stats}


def public_deadlines(*, now, clock_received, clock_left, room_seconds, search_seconds,
                     clock_reserve, clock_share, response_margin, finalize_seconds, total_seconds=None,
                     prompt_elapsed=None, prompt_cap=40.):
    """Public-clock allocation at prompt receipt; never reset by subdecisions.

    Reserve most of a turn for later decisions, and a separate final window
    for session cleanup/root update/real commit. No guessed clock is accepted.
    ``prompt_cap`` is the registered whole-prompt ceiling (40 seconds, or the untimed evaluation's).
    """
    if any(type(value) not in (float, int) or not math.isfinite(value) or value < 0 for value in (now, clock_received)) \
            or clock_received > now:
        raise ValueError("invalid or missing received public clock and search margins")
    return allocation(now=now, elapsed=now - clock_received, clock_left=clock_left, room_seconds=room_seconds,
                      search_seconds=search_seconds, clock_reserve=clock_reserve, clock_share=clock_share,
                      response_margin=response_margin, finalize_seconds=finalize_seconds,
                      total_seconds=total_seconds, prompt_elapsed=prompt_elapsed, prompt_cap=prompt_cap)


def allocation(*, now, elapsed, clock_left, room_seconds, search_seconds, clock_reserve, clock_share,
               response_margin, finalize_seconds, total_seconds=None, prompt_elapsed=None, prompt_cap=40.):
    """``public_deadlines`` from the public clock already elapsed at monotonic ``now``.

    A verifier repeats the runtime arithmetic exactly with the recorded ``clock_elapsed`` and the prompt's recorded
    monotonic start: an absolute deadline minus ``now`` rounds differently at different clock magnitudes.
    """
    values = (now, elapsed, clock_left, room_seconds, search_seconds, clock_reserve,
              clock_share, response_margin, finalize_seconds)
    if any(type(value) not in (float, int) or not math.isfinite(value) or value < 0 for value in values) \
            or not 0 <= clock_left <= room_seconds \
            or not 0 < clock_share <= 1 or not 0 < response_margin < clock_reserve < room_seconds \
            or search_seconds <= 0 or finalize_seconds <= 0:
        raise ValueError("invalid or missing received public clock and search margins")
    waited = 0. if prompt_elapsed is None else prompt_elapsed
    if type(waited) not in (int, float) or not math.isfinite(waited) or waited < 0 \
            or prompt_elapsed is not None and total_seconds is None:
        raise ValueError('a deferred prompt needs its finite elapsed wait and original total cap')
    left = clock_left - elapsed
    hard = now + left - response_margin
    if total_seconds is not None:
        if type(total_seconds) not in (float, int) or not math.isfinite(total_seconds) \
                or not finalize_seconds < total_seconds <= prompt_cap:
            raise ValueError("explicit replay-filtered prompt cap must exceed finalization and be <=40s")
        hard = min(hard, now + total_seconds - waited)
    if hard <= now:
        raise RolloutError("public room clock has no safe response margin")
    # A deferred clock's elapsed wait is fractional. Keep the serialized
    # allowance in relative coordinates so verification is exact even when
    # the process monotonic origin is hundreds of thousands of seconds.
    response_seconds = hard - now if prompt_elapsed is None else min(left - response_margin, total_seconds - waited)
    remaining_search = search_seconds if prompt_elapsed is None else search_seconds - waited
    allowance = max(0., min(remaining_search, (left - clock_reserve) * clock_share,
                           response_seconds - finalize_seconds))
    return now + allowance, hard, {"clock_left": clock_left, "clock_elapsed": elapsed,
                                   "allocated_seconds": allowance, "response_seconds": response_seconds,
                                   "clock_reserve": clock_reserve, "clock_share": clock_share,
                                   "response_margin": response_margin, "finalize_seconds": finalize_seconds,
                                   "room_seconds": room_seconds,
                                   **({"total_cap_seconds": total_seconds} if total_seconds is not None else {}),
                                   **({"prompt_elapsed": waited} if prompt_elapsed is not None else {})}


def check_stripes(record, *, rows, planned, seed):
    """Independent serialized-law check; a partially finished row never counts."""
    if not isinstance(record, dict) or record.get("law") != LAW or record.get("planned_stripes") != planned:
        raise ValueError("different balanced-stripe registration")
    count, stripes = record.get("completed_stripes"), record.get("stripes")
    if type(count) is not int or not 0 <= count <= planned or not isinstance(stripes, list) \
            or len(stripes) not in (count, min(count + 1, planned)) \
            or record.get("completed_indices") != list(range(count)) \
            or type(record.get("budget_zero_search")) is not bool or record["budget_zero_search"] != (count == 0) \
            or type(record.get("budget_exhausted")) is not bool or record["budget_exhausted"] != (count < planned):
        raise ValueError("stripe completion/zero-search accounting differs from its whole prefix")
    for index, stripe in enumerate(stripes):
        complete = index < count
        if stripe.get("index") != index or stripe.get("seed") != (seed * 1_000_003 + index) & 0x7fffffff \
                or type(stripe.get("complete")) is not bool or stripe["complete"] != complete \
                or type(stripe.get("finished_lines")) is not int or not 0 <= stripe["finished_lines"] <= rows \
                or complete and stripe["finished_lines"] != rows \
                or stripe.get("discarded_lines") != (0 if complete else rows) \
                or type(stripe.get("seconds")) not in (float, int) or not math.isfinite(stripe["seconds"]) \
                or stripe["seconds"] < 0:
            raise ValueError("stripe row coverage/seed/discard certificate is inconsistent")
    stats = record.get("stats", {})
    if stats.get("void") != 0 or stats.get("void_logs") != [] or stats.get("lines") != rows * count \
            or stats.get("attempted_lines") != rows * len(stripes) \
            or any(type(stats.get(key)) is not int or stats[key] < 0 for key in ("rounds", "items")) \
            or any(type(stats.get(key)) not in (int, float) or not math.isfinite(stats[key]) or stats[key] < 0
                   for key in ("service_seconds", "engine_seconds")):
        raise ValueError("balanced-stripe resource/value counts differ")
    return count


def greedy_without_search(logits):
    values = np.asarray(logits, np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("raw-greedy fallback needs this same finite root menu")
    prior = np.exp(values - values.max())
    prior /= prior.sum()
    return int(np.argmax(values)), {"prior": prior.tolist(), "policy": prior.tolist()}


def timing_summary(decisions, *, law=LAW):
    """All decisions, including zero-search, with separate per-game/seat turns."""
    from .agent_stripe_batching import LAW as MULTIPLEXED
    if law not in (LAW, MULTIPLEXED):
        raise ValueError("timing summary requires its explicit registered scheduler law")
    phases = ("root_seconds", "sample_seconds", "rollout_seconds", "update_seconds", "total_seconds")
    turns, stripe_counts = Counter(), Counter()
    samples = {phase: [] for phase in phases}
    zero = unknown = 0
    fallback_accounting = any("bank_unknown" in row for row in decisions)
    for row in decisions:
        bank_unknown = row.get("bank_unknown", False)
        if any(type(row.get(phase)) not in (int, float) or not math.isfinite(row[phase])
               or row[phase] < 0 for phase in phases) or type(row.get("completed_stripes")) is not int \
                or row["completed_stripes"] < 0 or type(row.get("budget_zero_search")) is not bool \
                or type(bank_unknown) is not bool or bank_unknown and row["completed_stripes"] != 0 \
                or row["budget_zero_search"] != (row["completed_stripes"] == 0 and not bank_unknown):
            raise ValueError("timing records need every phase and explicit stripe/zero-search accounting")
        for phase in phases:
            samples[phase].append(row[phase])
        turns[(row["game"], row["seat"], row["turn"])] += row["total_seconds"]
        stripe_counts[row["completed_stripes"]] += 1
        zero += row["budget_zero_search"]
        unknown += bank_unknown
    result = {"law": law, "decisions": len(decisions), "budget_zero_search": zero,
            "budget_zero_search_fraction": zero / len(decisions) if decisions else None,
            "completed_stripes_histogram": dict(sorted(stripe_counts.items())),
            "phases": {phase: ({"p50": float(np.percentile(values, 50)),
                                "p95": float(np.percentile(values, 95)), "max": max(values)} if values else None)
                       for phase, values in samples.items()},
            "worst_turn_seconds": max(turns.values(), default=None), "turns": len(turns)}
    if fallback_accounting:
        result.update(bank_unknown=unknown, bank_unknown_fraction=unknown / len(decisions) if decisions else None,
                      bank_unknown_law="same-root-raw-greedy/v1")
    return result
