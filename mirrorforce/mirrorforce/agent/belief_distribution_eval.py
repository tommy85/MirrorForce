"""Bounded public-only AR draws and genuine marginal diagnostics.

Generation never receives teacher targets. Scoring is a separate operation on
the sealed distribution. Zero Monte Carlo hits are NOT a model probability of
zero; no pseudocount or probability floor is used. This reference is not a
uniform-complete-root baseline or a production/heldout admission certificate.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from statistics import NormalDist
import time

import numpy as np

from mirrorforce.agent.search.belief_ar_law import masked_log_probabilities

SCHEMA = "mirrorforce_public_ar_distribution_diagnostic/v1"
LOCATIONS = (2, 4, 8)  # hidden hand, monster, spell/trap; core location bits


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class DistributionBudgetExceeded(TimeoutError):
    pass


def _geometry(codes, fields, hand):
    if not isinstance(codes, (list, tuple)) or any(type(c) is not int or c <= 0 for c in codes) \
            or list(codes) != sorted(set(codes)) or len(codes) > 128 \
            or not isinstance(fields, (list, tuple)) \
            or any(not isinstance(s, (list, tuple)) or len(s) != 2 or type(s[0]) is not int
                   or s[0] not in (4, 8) or type(s[1]) is not int
                   or not 0 <= s[1] < (7 if s[0] == 4 else 8) for s in fields) \
            or type(hand) is not int or hand < 0 or len(fields) + hand > 80:
        raise ValueError("AR diagnostics need canonical public vocabulary/field/sorted-hand geometry")
    fields = [list(s) for s in fields]
    if fields != sorted(fields) or len({tuple(s) for s in fields}) != len(fields) or (len(fields) + hand and not codes):
        raise ValueError("AR diagnostics need unique ordered fields and a nonempty active vocabulary")
    return {"codes": list(codes), "fields": fields, "hand_size": hand}


class _Evaluator:
    def __init__(self, law, score, *, deadline, max_queries, clock):
        if type(deadline) not in (int, float) or not math.isfinite(deadline) or not callable(clock) \
                or not callable(score) or type(max_queries) is not int or not 1 <= max_queries <= 1000000:
            raise ValueError("AR diagnostics require finite deadline and bounded public-prefix queries")
        self.geometry = _geometry(law.codes, [[s.location, s.sequence] for s in law.fields], law.hand_size)
        if type(law.length) is not int or law.length != len(law.fields) + law.hand_size:
            raise ValueError("AR law length differs from public geometry")
        bind = getattr(score, "bind_distribution", None)
        self.decoder_identity = None if bind is None else bind(law.codes, law.length)
        if self.decoder_identity is not None:
            # Own the binding before any prefix/mask/model computation. A raw
            # callable remains useful for toy diagnostics, but is explicitly unbound.
            self.decoder_identity = json.loads(json.dumps(self.decoder_identity, allow_nan=False))
            _decoder_geometry(self.decoder_identity, self.geometry)
        self.law, self.score, self.clock = law, score, clock
        self.deadline, self.max_queries = deadline, max_queries
        self.cache = {}
        self.check()

    def check(self):
        if self.clock() >= self.deadline:
            raise DistributionBudgetExceeded("AR distribution deadline exhausted; no partial distribution accepted")

    def probabilities(self, prefix):
        self.check()
        if prefix not in self.cache:
            if len(self.cache) >= self.max_queries:
                raise DistributionBudgetExceeded("AR distribution public-prefix query budget exhausted")
            # Masks are recomputed from generated prefixes, never copied from
            # a teacher-forced row. Unknown feasibility must propagate.
            mask = np.asarray(self.law.mask(prefix))
            self.check()  # A slow feasibility query must not trigger an already-expired GPU call.
            if mask.dtype != bool or mask.shape != (len(self.law.codes),):
                raise ValueError("AR public support returned a non-boolean or different-vocabulary mask")
            logits = self.score(prefix)
            self.check()
            logp = masked_log_probabilities(logits, mask)
            probabilities = np.exp(logp)
            if not np.isfinite(probabilities).all() or not math.isclose(float(probabilities.sum()), 1., abs_tol=1e-12) \
                    or np.any(probabilities[mask] == 0):
                raise FloatingPointError("AR probability underflow/normalization error; support cannot be silently removed")
            self.cache[prefix] = probabilities
        return self.cache[prefix]

    def finish(self, rows, *, method, samples=None, seed=None):
        self.check()
        result = {"schema": SCHEMA, "method": method, "geometry": self.geometry, "rows": rows,
            "samples": samples, "seed": seed, "public_prefix_queries": len(self.cache),
            "decoder_identity": self.decoder_identity,
            "decoder_geometry_verified": self.decoder_identity is not None,
            "deterministic_score_contract": "dropout disabled; fixed public features/parameters; generated prefixes only",
            "support": "caller-supplied public law; full-history/native coverage must be bound by caller",
            "uniform_complete_root_baseline": False, "search_admission": False}
        result["sha256"] = _digest(result)
        return result


def _decoder_geometry(identity, geometry):
    if not isinstance(identity, dict) or identity.get("schema") != "mirrorforce_public_ar_distribution_decoder/v2" \
            or identity.get("codes") != geometry["codes"] \
            or type(identity.get("target_length")) is not int \
            or identity["target_length"] != len(geometry["fields"]) + geometry["hand_size"] \
            or identity.get("deterministic") is not True or identity.get("teacher_channels") is not False \
            or identity.get("search_admission") is not False or identity.get("checkpoint_admitted") is not False \
            or not isinstance(identity.get("actual_parameter_tree_sha256"), str) \
            or len(identity["actual_parameter_tree_sha256"]) != 64 \
            or any(c not in "0123456789abcdef" for c in identity["actual_parameter_tree_sha256"]):
        raise ValueError("AR distribution decoder identity differs from its public geometry/parameter snapshot")


def draw_ar(law, score, *, samples, seed, deadline, max_queries, clock=time.monotonic):
    """IID free-generated complete sequences; memoization does not change their law.

    ``score(prefix)`` must be deterministic and take no truth/teacher input.
    One PCG64 stream is registered before any target is inspected.
    """
    if type(samples) is not int or not 1 <= samples <= 1000000 or type(seed) is not int or not 0 <= seed < 2 ** 64:
        raise ValueError("AR Monte Carlo needs a registered uint64 seed and 1..1000000 draws")
    evaluator = _Evaluator(law, score, deadline=deadline, max_queries=max_queries, clock=clock)
    rng, counts = np.random.Generator(np.random.PCG64(seed)), Counter()
    for _ in range(samples):
        evaluator.check()
        prefix = ()
        while len(prefix) < law.length:
            probabilities = evaluator.probabilities(prefix)
            prefix += (law.codes[int(rng.choice(len(law.codes), p=probabilities))],)
        law.layout(prefix)  # complete residual feasibility, including unpredicted zones
        evaluator.check()
        counts[prefix] += 1
    return evaluator.finish([[list(sequence), count] for sequence, count in sorted(counts.items())],
                            method="iid-free-generated-AR-PCG64/v1", samples=samples, seed=seed)


def enumerate_ar(law, score, *, deadline, max_queries, clock=time.monotonic):
    """Exact support enumeration with floating-point AR masses, for small reference cases only.

    This sums AR probabilities, NOT equal mass per canonical sequence and NOT
    uniform mass over complete assignments. Exceeding a bound returns no table.
    """
    evaluator = _Evaluator(law, score, deadline=deadline, max_queries=max_queries, clock=clock)
    pending, rows = [((), 1.)], []
    while pending:
        evaluator.check()
        prefix, mass = pending.pop()
        if len(prefix) == law.length:
            if len(rows) >= max_queries:
                raise DistributionBudgetExceeded("AR exact-distribution leaf budget exhausted")
            law.layout(prefix)
            rows.append([list(prefix), mass])
            continue
        for code, probability in reversed(list(zip(law.codes, evaluator.probabilities(prefix)))):
            if probability > 0:
                child_mass = mass * float(probability)
                if not child_mass > 0:
                    raise FloatingPointError("AR joint mass underflow; no partial-support enumeration")
                pending.append((prefix + (code,), child_mass))
    if not math.isclose(math.fsum(row[1] for row in rows), 1., rel_tol=1e-10, abs_tol=1e-12):
        raise FloatingPointError("complete AR enumeration failed normalization")
    return evaluator.finish(sorted(rows), method="complete-AR-support-enumeration-float64/v1")


def _read(distribution):
    if not isinstance(distribution, dict) or distribution.get("schema") != SCHEMA \
            or distribution.get("sha256") != _digest({k: v for k, v in distribution.items() if k != "sha256"}):
        raise ValueError("AR distribution changed after generation/sealing")
    method = distribution["method"]
    exact = method == "complete-AR-support-enumeration-float64/v1"
    if not exact and method != "iid-free-generated-AR-PCG64/v1":
        raise ValueError("unknown AR diagnostic sampling law")
    geometry, rows = distribution["geometry"], distribution["rows"]
    if not isinstance(geometry, dict) or set(geometry) != {"codes", "fields", "hand_size"}:
        raise ValueError("sealed AR geometry has missing or additional channels")
    geometry = _geometry(geometry["codes"], geometry["fields"], geometry["hand_size"])
    bound = distribution.get("decoder_identity")
    if distribution.get("decoder_geometry_verified", False) is not (bound is not None):
        raise ValueError("AR diagnostic decoder binding flag differs")
    if bound is not None:
        _decoder_geometry(bound, geometry)
    codes, fields, hand = geometry["codes"], geometry["fields"], geometry["hand_size"]
    length = len(fields) + hand
    sequences = []
    for sequence, weight in rows:
        if not isinstance(sequence, list) or len(sequence) != length \
                or any(type(c) is not int or c not in codes for c in sequence) \
                or sequence[len(fields):] != sorted(sequence[len(fields):]) \
                or (type(weight) not in (float, int) if exact else type(weight) is not int) \
                or not math.isfinite(weight) or weight <= 0:
            raise ValueError("AR diagnostic table contains invalid sequences or weights")
        sequences.append(sequence)
    if not rows or sequences != sorted(sequences) or len({tuple(s) for s in sequences}) != len(rows):
        raise ValueError("AR diagnostic table must aggregate each complete sequence exactly once")
    total = math.fsum(row[1] for row in rows)
    if exact and not math.isclose(total, 1., rel_tol=1e-10, abs_tol=1e-12) \
            or not exact and (type(distribution["samples"]) is not int or total != distribution["samples"]):
        raise ValueError("AR diagnostic weights fail their exact/Monte Carlo total")
    return exact, geometry, rows, total


def _estimate(mass, total, *, exact, confidence):
    p = mass / total
    result = {"probability": p, "nll": -math.log(p) if p else None,
        "zero_sample_hits": not exact and mass == 0, "true_model_zero": exact and mass == 0,
        "sample_hits": None if exact else int(mass), "probability_interval": None, "nll_interval": None}
    if not exact:
        # Pointwise Wilson score intervals for Bernoulli hit indicators;
        # deliberately not called a whole-dataset or simultaneous CI.
        z = NormalDist().inv_cdf((1. + confidence) / 2.)
        denominator = 1. + z * z / total
        center = (p + z * z / (2. * total)) / denominator
        half = z * math.sqrt(p * (1. - p) / total + z * z / (4. * total * total)) / denominator
        low, high = max(0., center - half), min(1., center + half)
        if mass == 0:
            low = 0.
        if mass == total:
            high = 1.
        result.update(probability_interval=[low, high],
                      nll_interval=[-math.log(high), -math.log(low) if low else None])
    return result


def _summary(cells):
    zeros = sum(cell["probability"] == 0 for cell in cells)
    return {"denominator": len(cells), "zero_probability_estimates": zeros,
        "nll_sum": math.fsum(cell["nll"] for cell in cells) if not zeros else None,
        "mean_nll": math.fsum(cell["nll"] for cell in cells) / len(cells) if cells and not zeros else None,
        "finite_point_estimate": not zeros, "probability_floor": None}


def score_target(distribution, target, *, confidence=.95):
    """PRIVILEGED_TARGET scoring only, after public-only generation is complete.

    A missing sampled target need not be impossible. Exact target support is
    independently checked by the heldout data loader; MC never invents it.
    Sorted-hand positional marginals are ranks in the canonical multiset, not
    the opponent's physical hand slots. Count marginals are clipped at 3+.
    """
    if type(confidence) not in (int, float) or not 0 < confidence < 1 or (1. + confidence) / 2. == 1.:
        raise ValueError("AR diagnostic confidence must be strictly between zero and one")
    exact, geometry, rows, total = _read(distribution)
    codes, fields = geometry["codes"], geometry["fields"]
    target = tuple(target)
    if len(target) != len(fields) + geometry["hand_size"] or any(type(c) is not int or c not in codes for c in target) \
            or list(target[len(fields):]) != sorted(target[len(fields):]):
        raise ValueError("target differs from canonical public field/sorted-hand geometry")
    if exact and not any(tuple(sequence) == target for sequence, _ in rows):
        raise ValueError("target has no support under the complete enumerated AR law")
    estimate = lambda mass: _estimate(mass, total, exact=exact, confidence=confidence)
    positional = [{"position": i, "location": fields[i][0] if i < len(fields) else 2,
                   "coordinate": fields[i][1] if i < len(fields) else i - len(fields),
                   **estimate(math.fsum(weight for sequence, weight in rows if sequence[i] == code))}
                  for i, code in enumerate(target)]
    areas = [field[0] for field in fields] + [2] * geometry["hand_size"]

    def counts(sequence):
        return Counter(zip(sequence, areas))

    target_counts = counts(target)
    table = [(counts(sequence), weight) for sequence, weight in rows]
    count_cells = [{"code": code, "location": location, "true_count_clipped": min(target_counts[code, location], 3),
                    **estimate(math.fsum(weight for count, weight in table
                        if min(count[code, location], 3) == min(target_counts[code, location], 3)))}
                   for code in codes for location in LOCATIONS]
    return {"schema": SCHEMA + "#target-score", "distribution_sha256": distribution["sha256"],
        "method": distribution["method"], "sample_count": distribution["samples"],
        "joint": estimate(math.fsum(weight for sequence, weight in rows if tuple(sequence) == target)),
        "positional": {"summary": _summary(positional), "cells": positional},
        "counts": {"summary": _summary(count_cells), "cells": count_cells},
        "uncertainty": {"confidence": confidence, "kind": "none-exact" if exact else "pointwise-Wilson-score",
            "simultaneous": False, "whole_game_bootstrap": False,
            "nll_interval_null_upper": "unbounded; zero MC hits do not establish a true model zero"},
        "training_eligible": False, "first_version_acceptance_complete": False, "search_admission": False}


__all__ = ["draw_ar", "enumerate_ar", "score_target", "DistributionBudgetExceeded"]
