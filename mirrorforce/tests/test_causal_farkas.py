from collections import Counter
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from mirrorforce.netduel import causal_farkas as F
from mirrorforce.netduel.causal_linear import exact_witness
from mirrorforce.netduel.agent_causal_plan import _Solver, PlanIncompatible, PlanBudgetExceeded


def attempt(domains, constraints):
    return F.try_certificate(domains, constraints, deadline=time.monotonic() + 5, clock=time.monotonic)


def impossible():
    return [{1, 2}, {1, 2}], [((0, 1), (), Counter({1: 2})), ((0, 1), (), Counter({2: 2}))]


def test_real_dual_candidate_proves_negative_using_original_integer_columns():
    domains, constraints = impossible()
    original = deepcopy((domains, constraints))
    cert, report = attempt(domains, constraints)
    assert report['negative_certified'] and F.exact_certificate(cert, domains, constraints)
    assert (domains, constraints) == original
    assert all(type(x) is int for x in cert['token_weights'])


@pytest.mark.parametrize('corruption', ['domain', 'constraint', 'float', 'bool', 'duplicate', 'zero', 'wrong_law'])
def test_independent_checker_rejects_corrupt_or_rebound_certificates(corruption):
    domains, constraints = impossible()
    cert, _ = attempt(domains, constraints)
    if corruption == 'domain':
        domains[0] = {1}
    elif corruption == 'constraint':
        constraints[0][2][1] = 1
    elif corruption == 'float':
        cert['token_weights'][0] = float(cert['token_weights'][0])
    elif corruption == 'bool':
        cert['token_weights'][0] = False
    elif corruption == 'duplicate':
        cert['constraint_weights'].append(cert['constraint_weights'][0])
    elif corruption == 'zero':
        cert['token_weights'] = [0, 0]
        cert['constraint_weights'] = []
    else:
        cert['law'] = 'solver-said-infeasible'
    assert not F.exact_certificate(cert, domains, constraints)


@pytest.mark.parametrize('value', [None, [], [float('nan')] * 6, [0.] * 6, [3.] * 6])
def test_backend_status_alone_or_bad_candidate_cannot_prove_negative(monkeypatch, value):
    monkeypatch.setattr(F, '_candidate', lambda *args: SimpleNamespace(status=2, x=value))
    cert, report = attempt(*impossible())
    assert cert is None and not report['negative_certified']


def test_expired_absolute_budget_cannot_accept_even_an_exact_candidate(monkeypatch):
    now = [0.]
    original = F._candidate
    def delayed(*args):
        result = original(*args)
        now[0] = 10.
        return result
    monkeypatch.setattr(F, '_candidate', delayed)
    cert, report = F.try_certificate(*impossible(), deadline=10., clock=lambda: now[0])
    assert cert is None and not report['negative_certified']


def test_negative_rhs_is_not_enough_when_one_original_allowed_column_is_negative():
    domains = [{1, 2}]
    constraints = [((0,), (), Counter({1: 1}))]
    cert = {'law': F.LAW, 'problem_sha256': F.problem_sha256(domains, constraints),
            'token_weights': [0], 'constraint_weights': [[0, 1, -1]]}
    assert not F.exact_certificate(cert, domains, constraints)


def test_certificate_verifier_has_no_floating_solver_dependency():
    import subprocess
    import sys
    code = "from mirrorforce.netduel.causal_farkas import exact_certificate; import sys; assert 'scipy' not in sys.modules and 'numpy' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_fractionally_feasible_integer_impossibility_still_uses_exact_search():
    # Three binary colors, every pair must have one of each: x_i=.5 is an LP
    # solution, but no integer coloring exists. A dual certificate MUST NOT exist.
    domains = [{1, 2}] * 3 + [{3}] * 61
    constraints = [((a, b), (), Counter({1: 1, 2: 1})) for a, b in ((0, 1), (1, 2), (0, 2))]
    cert, report = attempt(domains, constraints)
    assert cert is None and not report['negative_certified']
    solver = _Solver(constraints, 64, 0, 10000, time.monotonic() + 5, time.monotonic)
    with pytest.raises(PlanIncompatible):
        solver.feasible_witness(domains)
    assert solver.linear_infeasibility_certificate is None and solver.operations > 0


def test_certificates_never_reject_an_exhaustively_feasible_overlapping_problem():
    rng = np.random.default_rng(9045)
    negatives = 0
    for _ in range(48):
        domains = [{1, 2}] * 5 + [{3}] * 59
        constraints = []
        for _ in range(3):
            left = tuple(sorted(map(int, rng.choice(5, 3, replace=False))))
            count = int(rng.integers(0, 4))
            constraints.append((left, (), Counter({1: count, 2: 3 - count})))
        feasible = any(exact_witness(values + (3,) * 59, domains, constraints)
                       for values in product((1, 2), repeat=5))
        cert, report = attempt(domains, constraints)
        if cert is not None:
            negatives += 1
            assert not feasible and report['negative_certified']
            assert F.exact_certificate(cert, domains, constraints)
        if feasible:
            assert cert is None
    assert negatives > 5


def test_original_signed_multiplicity_and_fixed_constraints_are_checked():
    # Deliberately retain a right side on a fixed equality: exact Counter
    # semantics ignore it, and signed repeated-token occurrences matter.
    domains = [{1, 2}, {1, 2}]
    constraints = [((0, 0, 1), (0,), Counter({1: 3})), ((0,), (1,), None),
                   ((1,), (), Counter({2: 1}))]
    cert, report = attempt(domains, constraints)
    assert report['negative_certified'] and F.exact_certificate(cert, domains, constraints)


def test_real_teacher_query_timeout_is_certified_without_masking_unknown():
    fixture = json.loads((Path(__file__).parent / 'fixtures/capacity-teacher-query-row189.json').read_text())
    assert fixture['authority'] == 'PRIVILEGED_TARGET_DIAGNOSTIC'
    assert fixture['inference_eligible'] is False and fixture['complete_privileged_targets_included'] is False
    domains = [set(fixture['domain_table'][i]) for i in fixture['domain_indices']]
    constraints = [(tuple(left), tuple(right), None if fixed is None else Counter(dict(fixed)))
                   for left, right, fixed in fixture['constraints']]
    assert len(domains) == 427 and len(constraints) == 26 and fixture['prior_nodes'] == 22352
    solver = _Solver(constraints, len(domains), 0, 1000000, time.monotonic() + 5, time.monotonic)
    with pytest.raises(PlanIncompatible, match='integer Farkas'):
        solver.feasible_witness(domains)
    assert solver.linear_infeasibility_report['negative_certified']
    assert F.exact_certificate(solver.linear_infeasibility_certificate, domains, constraints)
    assert solver.linear_witness_report['attempt_statuses'] == [2, 2]
    assert solver.nodes == 3 and solver.operations == 0  # two candidate searches, then exact dual proof
    expired = _Solver(constraints, len(domains), 0, 1000000, 1., lambda: 1.)
    with pytest.raises(PlanBudgetExceeded):
        expired.feasible_witness(domains)
