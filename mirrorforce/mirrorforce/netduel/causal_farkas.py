"""Exact negative certificates for public capacity existence, not replay.

For the original distinguished-token one-hot variables x >= 0 and equations
A x = b, an integer y with A.T y >= 0 and b.T y < 0 proves infeasibility.
An optional floating LP merely proposes y. Every accepted certificate is
checked using Python integers against ALL original domains and equalities.
Backend status, floating residuals and timeouts never prove a negative.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import warnings

LAW = "original-token-capacity-integer-farkas/v1"
INFORMATION_SET_SEARCH = True


def problem_sha256(domains, constraints):
    record = {"domains": [sorted(d) for d in domains], "constraints": [
        [list(left), list(right), None if fixed is None else sorted(fixed.items())]
        for left, right, fixed in constraints]}
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def exact_certificate(certificate, domains, constraints):
    """Independent ungrouped integer verification; no scipy or float arithmetic."""
    if not isinstance(certificate, dict) or certificate.get("law") != LAW \
            or certificate.get("problem_sha256") != problem_sha256(domains, constraints):
        return False
    token_weights = certificate.get("token_weights")
    rows = certificate.get("constraint_weights")
    if not isinstance(token_weights, list) or len(token_weights) != len(domains) \
            or any(type(value) is not int for value in token_weights) or not isinstance(rows, list):
        return False
    coefficients = [Counter() for _ in domains]
    weighted_rhs = sum(token_weights)
    seen = set()
    for row in rows:
        if not isinstance(row, list) or len(row) != 3 or any(type(v) is not int for v in row):
            return False
        ci, code, weight = row
        if not 0 <= ci < len(constraints) or (ci, code) in seen:
            return False
        seen.add((ci, code))
        left, right, fixed = constraints[ci]
        for token in left:
            coefficients[token][code] += weight
        if fixed is None:
            for token in right:
                coefficients[token][code] -= weight
        else:
            weighted_rhs += weight * fixed[code]
    if weighted_rhs >= 0:
        return False
    # A column exists for EVERY allowed (original token, code), even if it
    # disappeared or was incorrectly combined in the candidate LP encoding.
    return all(token_weights[token] + coefficients[token][code] >= 0
               for token, domain in enumerate(domains) for code in domain)


def _candidate(matrix, rhs, seconds):
    import numpy as np
    from scipy.optimize import linprog
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*threads.*")
        return linprog(rhs, A_ub=-matrix.T, b_ub=np.zeros(matrix.shape[1]),
                       bounds=(-1., 1.), method="highs",
                       options={"time_limit": seconds, "threads": 1})


def try_certificate(domains, constraints, *, deadline, clock, max_seconds=1., max_variables=100000):
    if type(deadline) not in (float, int) or not math.isfinite(deadline) or not callable(clock) \
            or type(max_seconds) not in (float, int) or not math.isfinite(max_seconds) or max_seconds <= 0 \
            or type(max_variables) is not int or max_variables < 1:
        raise ValueError("integer negative certificates require explicit finite budgets")
    started = clock()
    report = {"law": LAW, "attempted": False, "negative_certified": False, "seconds": 0., "nodes": 0}

    def finish(certificate=None):
        report["seconds"] = max(0., clock() - started)
        report["negative_certified"] = certificate is not None
        return certificate, report

    if started >= deadline or not domains or sum(map(len, domains)) > max_variables:
        return finish()
    try:
        import numpy as np
        import scipy
        from scipy.sparse import coo_matrix
    except ImportError:
        report["status"] = "optional_backend_unavailable"
        return finish()
    from .causal_linear import _exchange_classes
    classes = _exchange_classes(domains, constraints)
    columns = {(index, code): column for column, (index, code) in enumerate(
        (i, code) for i, (_, codes, _) in enumerate(classes) for code in codes)}
    row_ids, col_ids, values, rhs, row_keys = [], [], [], [], []

    def add(entries, value, key):
        row = len(rhs)
        rhs.append(value)
        row_keys.append(key)
        for column, coefficient in entries:
            if coefficient:
                row_ids.append(row)
                col_ids.append(column)
                values.append(coefficient)

    for index, (tokens, codes, _) in enumerate(classes):
        add(((columns[index, code], 1) for code in codes), len(tokens), None)
    incidence = [dict(signature) for _, _, signature in classes]
    for ci, (left, right, fixed) in enumerate(constraints):
        if clock() >= deadline:
            return finish()
        codes = set().union(*(domains[t] for t in (*left, *(right if fixed is None else ()))))
        if fixed is not None:
            codes.update(fixed)
        for code in sorted(codes):
            add(((columns[index, code], signature[ci]) for index, signature in enumerate(incidence)
                 if ci in signature and (index, code) in columns),
                0 if fixed is None else fixed[code], (ci, code))
    seconds = min(float(max_seconds), (deadline - clock()) / 2)
    if seconds <= 0 or not columns:
        return finish()
    matrix = coo_matrix((values, (row_ids, col_ids)), shape=(len(rhs), len(columns))).tocsc()
    report.update(attempted=True, nodes=1, scipy_version=scipy.__version__,
                  variables=len(columns), equalities=len(rhs))
    result = _candidate(matrix, np.asarray(rhs), seconds)
    report["status"] = int(result.status)
    candidate = getattr(result, "x", None)
    if clock() >= deadline or candidate is None:
        return finish()
    candidate = np.asarray(candidate)
    if candidate.shape != (len(rhs),) or not np.isfinite(candidate).all() or np.abs(candidate).max() > 2:
        return finish()
    def integer_candidates():
        for scale in (1, 10, 100, 1000, 10000, 100000, 1000000):
            yield [int(round(float(value) * scale)) for value in candidate]
        # Basic dual solutions can have e.g. thirds; powers of ten would
        # perpetually round one zero reduced cost in the wrong direction.
        for denominator in (10, 100, 1000, 10000, 1000000):
            fractions = [Fraction(float(value)).limit_denominator(denominator) for value in candidate]
            scale = math.lcm(*(value.denominator for value in fractions))
            if scale <= 10 ** 12:
                yield [value.numerator * (scale // value.denominator) for value in fractions]

    for weights in integer_candidates():
        if clock() >= deadline:
            return finish()
        token_weights = [0] * len(domains)
        for weight, (tokens, _, _) in zip(weights, classes):
            for token in tokens:
                token_weights[token] = weight
        certificate = {"law": LAW, "problem_sha256": problem_sha256(domains, constraints),
            "token_weights": token_weights, "constraint_weights": [
                [*key, weight] for key, weight in zip(row_keys[len(classes):], weights[len(classes):]) if weight]}
        if exact_certificate(certificate, domains, constraints) and clock() < deadline:
            return finish(certificate)
    return finish()
