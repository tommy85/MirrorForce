"""Optional positive-witness accelerator for public integer capacity existence.

HiGHS is only a candidate generator. Neither its infeasibility status nor a
floating residual ever proves a negative mask. A candidate is rounded and
then checked as exact Python integer colors against EVERY original domain and
Counter equality. No verified witness means the caller uses its exact solver.
This path does not choose the seeded natural replay witness or count roots.
"""
from __future__ import annotations

from collections import Counter
import math
import warnings

LAW = "positive-integer-witness-only; exact-counter-verification/v1"
INFORMATION_SET_SEARCH = True


def exact_witness(colors, domains, constraints):
    if type(colors) is not tuple or len(colors) != len(domains) \
            or any(type(code) is not int or code not in domain for code, domain in zip(colors, domains)):
        return False
    return all(Counter(colors[t] for t in left) ==
               (Counter(colors[t] for t in right) if fixed is None else fixed)
               for left, right, fixed in constraints)


def _optimize(matrix, rhs, size, seconds, nodes, upper=None, presolve=True):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    # scipy forwards this explicitly documented HiGHS option. Pin one worker
    # so a CPU-only mask check cannot create a machine-wide solver pool.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*threads.*", category=RuntimeWarning)
        limits = {"time_limit": seconds, "mip_rel_gap": 0., "threads": 1, "presolve": presolve}
        if nodes is not None:
            limits["node_limit"] = nodes
        return milp(np.zeros(size), integrality=np.ones(size), bounds=Bounds(0, 1 if upper is None else upper),
                    constraints=LinearConstraint(matrix, rhs, rhs),
                    options=limits)


def _exchange_classes(domains, constraints):
    """Identical domains AND every signed incidence; no graph/zone name is approximated.

    A class variable counts copies of one code, not a binary choice. Any
    integral count vector expands to distinguished original tokens, whose
    complete assignment is still independently checked by exact_witness().
    This is only a positive-existence accelerator, never a root sampler.
    """
    incidence = [Counter() for _ in domains]
    for index, (left, right, fixed) in enumerate(constraints):
        for token in left:
            incidence[token][index] += 1
        if fixed is None:
            for token in right:
                incidence[token][index] -= 1
    groups = {}
    for token, domain in enumerate(domains):
        signature = (tuple(sorted(domain)), tuple((ci, n) for ci, n in sorted(incidence[token].items()) if n))
        groups.setdefault(signature, []).append(token)
    return tuple((tuple(tokens), codes, signature) for (codes, signature), tokens in groups.items())


def try_witness(domains, constraints, *, deadline, clock, max_nodes, max_seconds=1.0, max_variables=100000):
    if type(deadline) not in (float, int) or not math.isfinite(deadline) or not callable(clock) \
            or max_nodes is not None and (type(max_nodes) is not int or max_nodes < 1) or type(max_seconds) not in (float, int) \
            or not math.isfinite(max_seconds) or max_seconds <= 0 or type(max_variables) is not int or max_variables < 1:
        raise ValueError("positive-witness accelerator requires finite explicit budgets")
    started = clock()
    report = {"law": LAW, "attempted": False, "accepted": False, "negative_certified": False,
              "nodes": 0, "seconds": 0., "status": "not_attempted"}

    def finish(colors=None):
        report["seconds"] = max(0., clock() - started)
        report["accepted"] = colors is not None
        return colors, report

    if started >= deadline or max_nodes is not None and max_nodes < 2:
        return finish()
    size = sum(len(domain) for domain in domains)
    if not domains or not size or size > max_variables or any(not domain for domain in domains):
        return finish()
    try:
        import numpy as np
        import scipy
        from scipy.sparse import coo_matrix
    except ImportError:
        report["status"] = "optional_backend_unavailable"
        return finish()
    report["scipy_version"] = scipy.__version__
    grouped = len(domains) >= 64
    classes = _exchange_classes(domains, constraints) if grouped else tuple(
        ((token,), tuple(sorted(domain)), ()) for token, domain in enumerate(domains))
    indices, columns_by_class, upper = {}, [], []
    for index, (tokens, codes, _) in enumerate(classes):
        if clock() >= deadline:
            return finish()
        columns = []
        for code in codes:
            indices[index, code] = len(indices)
            columns.append(indices[index, code])
            upper.append(len(tokens))
        columns_by_class.append(columns)
    original_size, size = size, len(indices)
    token_class = {token: index for index, (tokens, _, _) in enumerate(classes) for token in tokens}
    row_ids, col_ids, coefficients, rhs = [], [], [], []

    def add(values, value):
        row = len(rhs)
        rhs.append(value)
        for column, coefficient in values.items():
            if coefficient:
                row_ids.append(row)
                col_ids.append(column)
                coefficients.append(coefficient)

    for columns, (tokens, _, _) in zip(columns_by_class, classes):
        add(dict.fromkeys(columns, 1), len(tokens))
    for ci, (left, right, fixed) in enumerate(constraints):
        if clock() >= deadline:
            return finish()
        codes = set().union(*(domains[t] for t in (*left, *right)))
        if fixed is not None:
            codes.update(fixed)
        for code in sorted(codes):
            values = Counter()
            if grouped:
                for index, (_, allowed, incidence) in enumerate(classes):
                    coefficient = dict(incidence).get(ci, 0)
                    if coefficient and code in allowed:
                        values[indices[index, code]] = coefficient
            else:
                for token in left:
                    if code in domains[token]:
                        values[indices[token_class[token], code]] += 1
                for token in right:
                    if code in domains[token]:
                        values[indices[token_class[token], code]] -= 1
            add(values, 0 if fixed is None else fixed[code])
    seconds = min(float(max_seconds), (deadline - clock()) / 2)
    if seconds <= 0:
        return finish()
    matrix = coo_matrix((coefficients, (row_ids, col_ids)), shape=(len(rhs), size)).tocsc()
    report.update(attempted=True, variables=size, equalities=len(rhs),
                  uncompressed_variables=original_size, token_classes=len(classes),
                  encoding="exact-exchange-class-counts/v1" if grouped else "distinguished-token-binary/v1")
    rhs, upper_array = np.asarray(rhs), np.asarray(upper)
    # A backend may return an invalid incumbent even with status=optimal,
    # or claim infeasibility without a candidate for a feasible problem.
    # Retry those cases with presolve disabled within the SAME backend
    # time/node allowance. Neither attempt's status can prove infeasibility.
    backend_deadline = min(deadline, clock() + seconds)
    report.update(presolve_retry=False, attempt_statuses=[])

    def decode(candidate):
        candidate = np.asarray(candidate)
        if candidate.shape != (size,) or not np.isfinite(candidate).all():
            return None
        counts = np.rint(candidate)
        if np.any(counts < 0) or np.any(counts > upper_array) or any(
                counts[columns].sum() != len(tokens)
                for columns, (tokens, _, _) in zip(columns_by_class, classes)):
            return None
        colors = [None] * len(domains)
        for index, (tokens, codes, _) in enumerate(classes):
            expanded = [code for code in codes for _ in range(int(counts[indices[index, code]]))]
            for token, code in zip(tokens, expanded):
                colors[token] = code
        colors = tuple(colors)
        return colors if exact_witness(colors, domains, constraints) else None

    for attempt in range(2):
        remaining_nodes = None if max_nodes is None else max_nodes - report["nodes"]
        remaining_seconds = backend_deadline - clock()
        if remaining_seconds <= 0 or remaining_nodes is not None and remaining_nodes < 2:
            return finish()
        arguments = (matrix, rhs, size, remaining_seconds, None if remaining_nodes is None else remaining_nodes - 1)
        if attempt:
            report["presolve_retry"] = True
            result = _optimize(*arguments, upper_array, False)
        else:
            result = _optimize(*arguments, upper_array) if grouped else _optimize(*arguments)
        nodes = getattr(result, "mip_node_count", None)
        used = 1 + (max(0, int(nodes)) if nodes is not None and math.isfinite(nodes) else 0)
        report["nodes"] += used if remaining_nodes is None else min(remaining_nodes, used)
        report["status"] = int(result.status)
        report["attempt_statuses"].append(int(result.status))
        # Partial MIP incumbents are allowed only after the full integer check.
        candidate = getattr(result, "x", None)
        if clock() >= deadline:
            return finish()
        if result.status == 2 and candidate is None:
            continue  # optional second candidate search, NOT a negative proof
        if candidate is None or result.status not in (0, 1):
            return finish()
        colors = decode(candidate)
        if clock() >= deadline:
            return finish()
        if colors is not None:
            return finish(colors)
    return finish()
