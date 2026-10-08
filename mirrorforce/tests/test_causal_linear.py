from collections import Counter
from itertools import product
from types import SimpleNamespace
from pathlib import Path
import json
import time

import numpy as np
import pytest


def test_explicit_unlimited_replay_backend_does_not_install_a_node_limit(monkeypatch):
    import numpy as np
    from scipy.sparse import eye
    from scipy import optimize
    from mirrorforce.netduel import causal_linear as L
    calls = []
    def milp(*args, **kw):
        calls.append(kw['options'])
        return None
    monkeypatch.setattr(optimize, 'milp', milp)
    L._optimize(eye(1), np.asarray([1]), 1, .5, None)
    assert calls == [{'time_limit': .5, 'mip_rel_gap': 0., 'threads': 1, 'presolve': True}]

from mirrorforce.netduel import causal_linear as L
from mirrorforce.netduel.agent_causal_plan import _Solver, PlanIncompatible, PlanBudgetExceeded


def small():
    domains = [{1, 2}, {1, 2}, {1}, {2}]
    constraints = [((0, 1), (), Counter({1: 1, 2: 1})), ((0, 1), (2, 3), None)]
    return domains, constraints


def test_exact_verifier_checks_every_domain_and_counter_not_float_residuals():
    domains, constraints = small()
    expected = {(1, 2, 1, 2), (2, 1, 1, 2)}
    assert {colors for colors in product((1, 2), repeat=4) if L.exact_witness(colors, domains, constraints)} == expected
    assert not L.exact_witness((1., 2, 1, 2), domains, constraints)
    assert not L.exact_witness((1, 1, 1, 2), domains, constraints)


def test_real_linear_candidate_is_verified_against_original_integer_constraints():
    domains, constraints = small()
    before = [set(domain) for domain in domains]
    colors, report = L.try_witness(domains, constraints, deadline=time.monotonic() + 10,
                                    clock=time.monotonic, max_nodes=100)
    assert report["attempted"] and report["accepted"] and not report["negative_certified"]
    assert L.exact_witness(colors, domains, constraints) and domains == before


@pytest.mark.parametrize("candidate,status", [([1, 0, 1, 0, 1, 1], 0), ([0.] * 6, 0),
    ([float("nan")] * 6, 0), ([1], 0), (None, 2), (None, 1), (None, 4)])
def test_status_or_invalid_rounded_candidate_never_certifies_a_false_mask(monkeypatch, candidate, status):
    domains, constraints = small()
    monkeypatch.setattr(L, "_optimize", lambda *args: SimpleNamespace(status=status, x=candidate, mip_node_count=0))
    colors, report = L.try_witness(domains, constraints, deadline=time.monotonic() + 10,
                                    clock=time.monotonic, max_nodes=100)
    assert colors is None and not report["accepted"] and not report["negative_certified"]


def test_valid_partial_incumbent_is_an_exact_positive_proof(monkeypatch):
    domains, constraints = small()
    monkeypatch.setattr(L, "_optimize", lambda *args: SimpleNamespace(status=1, x=[1, 0, 0, 1, 1, 1], mip_node_count=1))
    colors, report = L.try_witness(domains, constraints, deadline=time.monotonic() + 10,
                                    clock=time.monotonic, max_nodes=100)
    assert colors == (1, 2, 1, 2) and report["accepted"] and report["nodes"] == 2


def test_accelerator_cannot_claim_negative_even_when_backend_says_infeasible(monkeypatch):
    monkeypatch.setattr(L, "try_witness", lambda *args, **kwargs: (None, {"nodes": 1, "status": 2}))
    domains = [{1, 2} for _ in range(64)]
    constraints = [(tuple(range(64)), (), Counter({1: 32, 2: 32}))]
    solver = _Solver(constraints, 64, 0, 1000, time.monotonic() + 5, time.monotonic)
    assert L.exact_witness(solver.feasible_witness(domains), domains, constraints)
    assert solver.nodes > 1


def test_exact_solver_still_certifies_impossible_and_expired_budget_stays_unknown(monkeypatch):
    monkeypatch.setattr(L, "try_witness", lambda *args, **kwargs: (None, {"nodes": 1, "status": 2}))
    domains, constraints = [{1} for _ in range(64)], [(tuple(range(64)), (), Counter({2: 64}))]
    solver = _Solver(constraints, 64, 0, 1000, time.monotonic() + 5, time.monotonic)
    with pytest.raises(PlanIncompatible):
        solver.feasible_witness(domains)
    expired = _Solver(constraints, 64, 0, 1000, 1., lambda: 1.)
    with pytest.raises(PlanBudgetExceeded):
        expired.feasible_witness(domains)


def test_absolute_deadline_prevents_accepting_late_valid_candidate(monkeypatch):
    now = [0.]
    domains, constraints = small()
    def optimize(*args):
        now[0] = 10
        return SimpleNamespace(status=0, x=np.array([1, 0, 0, 1, 1, 1]), mip_node_count=0)
    monkeypatch.setattr(L, "_optimize", optimize)
    colors, report = L.try_witness(domains, constraints, deadline=10., clock=lambda: now[0], max_nodes=100)
    assert colors is None and not report["accepted"]


def test_real_272_token_public_failure_gets_an_exact_witness_without_exponential_dfs():
    fixture = json.loads((Path(__file__).parent / "fixtures/public-capacity-row102.json").read_text())
    domains = [set(fixture["domain_table"][index]) for index in fixture["domain_indices"]]
    constraints = [(tuple(left), tuple(right), None if fixed is None else Counter(dict(fixed)))
                   for left, right, fixed in fixture["constraints"]]
    assert len(domains) == 272 and len(constraints) == 17 and fixture["prior_nodes"] == 64805
    assert fixture["privileged_targets_included"] is False
    original = [set(domain) for domain in domains]
    solver = _Solver(constraints, len(domains), 0, 100000, time.monotonic() + 5, time.monotonic)
    colors = solver.feasible_witness(domains)
    assert L.exact_witness(colors, original, constraints) and domains == original
    assert solver.linear_witness_report["accepted"] and not solver.linear_witness_report["negative_certified"]
    assert solver.nodes < 100 and solver.operations == 0


def test_exchange_classes_require_full_signed_incidence_and_domains():
    domains = [{1, 2}, {1, 2}, {1, 2}, {1}, {1, 2}]
    constraints = [((0, 1, 2, 3, 4), (), Counter({1: 3, 2: 2})),
                   ((0, 1, 4), (2, 4), None)]
    groups = L._exchange_classes(domains, constraints)
    assert [tokens for tokens, _, _ in groups] == [(0, 1), (2,), (3,), (4,)]
    # Token 4 cancels in the second equality, not in the pool capacity.
    assert groups[-1][2] == ((0, 1),)
    assert groups[1][2] == ((0, 1), (1, -1))


def test_real_grouped_integer_counts_expand_all_distinguished_original_tokens():
    domains = [{1, 2} for _ in range(64)]
    constraints = [(tuple(range(32)), (), Counter({1: 11, 2: 21})),
                   (tuple(range(32)), tuple(range(32, 64)), None)]
    colors, report = L.try_witness(domains, constraints, deadline=time.monotonic() + 5,
                                  clock=time.monotonic, max_nodes=100)
    assert L.exact_witness(colors, domains, constraints)
    assert report["variables"] == 4 and report["uncompressed_variables"] == 128
    assert report["token_classes"] == 2 and report["encoding"] == "exact-exchange-class-counts/v1"
    assert Counter(colors[:32]) == Counter({1: 11, 2: 21}) == Counter(colors[32:])


@pytest.mark.parametrize("candidate", [[33, 31], [-1, 65], [31, 32], [32, 32], [float("nan"), 64], [1]])
def test_grouped_status_or_rounding_cannot_bypass_original_exact_constraints(monkeypatch, candidate):
    domains = [{1, 2} for _ in range(64)]
    constraints = [(tuple(range(64)), (), Counter({1: 16, 2: 48}))]
    monkeypatch.setattr(L, "_optimize", lambda *args: SimpleNamespace(status=0, x=candidate, mip_node_count=0))
    colors, report = L.try_witness(domains, constraints, deadline=time.monotonic() + 5,
                                  clock=time.monotonic, max_nodes=100)
    assert colors is None and not report["accepted"] and not report["negative_certified"]


def test_grouped_positive_incumbent_is_accepted_without_claiming_negative_proof(monkeypatch):
    domains = [{1, 2} for _ in range(64)]
    constraints = [(tuple(range(64)), (), Counter({1: 16, 2: 48}))]
    seen = []
    def optimize(*args):
        seen.append(args[-1].tolist())
        return SimpleNamespace(status=1, x=[16, 48], mip_node_count=1)
    monkeypatch.setattr(L, "_optimize", optimize)
    colors, report = L.try_witness(domains, constraints, deadline=time.monotonic() + 5,
                                  clock=time.monotonic, max_nodes=100)
    assert seen == [[64, 64]] and colors == (1,) * 16 + (2,) * 48
    assert L.exact_witness(colors, domains, constraints) and report["accepted"] and not report["negative_certified"]


def test_grouped_positive_existence_matches_exhaustive_overlapping_public_capacities():
    rng = np.random.default_rng(713)
    for _ in range(24):
        target = tuple(map(int, rng.integers(1, 3, size=7)))
        domains = [{1, 2} for _ in target] + [{3} for _ in range(57)]
        constraints = [(tuple(range(7)), (), Counter(target)), (tuple(range(7, 64)), (), Counter({3: 57}))]
        for _ in range(4):
            left = tuple(sorted(map(int, rng.choice(7, size=3, replace=False))))
            constraints.append((left, (), Counter(target[index] for index in left)))
        feasible = {colors for values in product((1, 2), repeat=7)
                    if L.exact_witness(colors := values + (3,) * 57, domains, constraints)}
        assert target + (3,) * 57 in feasible
        colors, report = L.try_witness(domains, constraints, deadline=time.monotonic() + 5,
                                      clock=time.monotonic, max_nodes=100)
        assert report["accepted"] and colors in feasible and not report["negative_certified"]


def test_real_four_game_gate_timeout_gets_exact_public_witness_with_grouped_counts():
    fixture = json.loads((Path(__file__).parent / "fixtures/public-capacity-smoke-row133.json").read_text())
    assert fixture["original_failure_sha256"] == "55391f677b3bf6c6e3c93952f2931318992461eb0f57e46b66fea3a8f4752334"
    assert fixture["privileged_targets_included"] is False and fixture["prior_nodes"] == 42644
    domains = [set(fixture["domain_table"][index]) for index in fixture["domain_indices"]]
    constraints = [(tuple(left), tuple(right), None if fixed is None else Counter(dict(fixed)))
                   for left, right, fixed in fixture["constraints"]]
    assert len(domains) == 410 and len(constraints) == 21
    solver = _Solver(constraints, len(domains), 0, 1000000, time.monotonic() + 5, time.monotonic)
    colors = solver.feasible_witness(domains)
    assert L.exact_witness(colors, domains, constraints)
    assert solver.linear_witness_report["accepted"] and solver.linear_witness_report["variables"] == 409
    assert solver.linear_witness_report["uncompressed_variables"] == 7057
    assert not solver.linear_witness_report["negative_certified"] and solver.nodes < 100


def test_invalid_optimal_incumbent_gets_one_exact_checked_retry_in_shared_budget(monkeypatch):
    domains, constraints = small()
    now, calls = [0.], []

    def optimize(*args):
        calls.append(args)
        now[0] += .25
        # First candidate has an illegal negative entry, despite status=optimal.
        x = [-1, 2, 2, -1, 1, 1] if len(calls) == 1 else [1, 0, 0, 1, 1, 1]
        return SimpleNamespace(status=0, x=x, mip_node_count=2)

    monkeypatch.setattr(L, '_optimize', optimize)
    colors, report = L.try_witness(domains, constraints, deadline=10., clock=lambda: now[0],
                                   max_nodes=10, max_seconds=1.)
    assert L.exact_witness(colors, domains, constraints)
    assert len(calls) == 2 and calls[1][-1] is False
    assert calls[0][3:5] == (1., 9) and calls[1][3:5] == (.75, 6)
    assert report['presolve_retry'] and report['attempt_statuses'] == [0, 0]
    assert report['nodes'] == 6 and not report['negative_certified']


@pytest.mark.parametrize('limit', ['backend_time', 'nodes', 'absolute_time'])
@pytest.mark.parametrize('status', [0, 2])
def test_unproved_backend_retry_cannot_renew_any_budget(monkeypatch, limit, status):
    now, calls = [0.], []

    def optimize(*args):
        calls.append(args)
        if limit != 'nodes':
            now[0] = 1. if limit == 'backend_time' else 10.
        return SimpleNamespace(status=status, x=[-1, 2, 2, -1, 1, 1] if status == 0 else None, mip_node_count=2)

    monkeypatch.setattr(L, '_optimize', optimize)
    colors, report = L.try_witness(*small(), deadline=10., clock=lambda: now[0],
                                   max_nodes=4 if limit == 'nodes' else 100, max_seconds=1.)
    assert colors is None and len(calls) == 1 and report['nodes'] == 3
    assert not report['presolve_retry'] and not report['negative_certified']


def test_retry_never_accepts_a_valid_witness_after_absolute_deadline(monkeypatch):
    now, calls = [0.], []

    def optimize(*args):
        calls.append(args)
        if len(calls) == 1:
            return SimpleNamespace(status=0, x=[-1, 2, 2, -1, 1, 1], mip_node_count=0)
        now[0] = 10.
        return SimpleNamespace(status=0, x=[1, 0, 0, 1, 1, 1], mip_node_count=0)

    monkeypatch.setattr(L, '_optimize', optimize)
    colors, report = L.try_witness(*small(), deadline=10., clock=lambda: now[0], max_nodes=100)
    assert colors is None and len(calls) == 2 and report['presolve_retry']
    assert report['nodes'] == 2 and not report['negative_certified']


def test_backend_negative_status_is_retried_once_but_never_proves_negative(monkeypatch):
    calls = []

    def optimize(*args):
        calls.append(args)
        return SimpleNamespace(status=2, x=None, mip_node_count=0)

    monkeypatch.setattr(L, '_optimize', optimize)
    colors, report = L.try_witness(*small(), deadline=time.monotonic() + 5,
                                   clock=time.monotonic, max_nodes=100)
    assert colors is None and len(calls) == 2 and report['presolve_retry']
    assert calls[1][-1] is False and report['attempt_statuses'] == [2, 2]
    assert not report['negative_certified']


def test_false_infeasible_without_candidate_gets_one_exact_checked_retry(monkeypatch):
    now, calls = [0.], []

    def optimize(*args):
        calls.append(args)
        now[0] += .25
        return SimpleNamespace(status=2 if len(calls) == 1 else 0,
                               x=None if len(calls) == 1 else [1, 0, 0, 1, 1, 1], mip_node_count=2)

    monkeypatch.setattr(L, '_optimize', optimize)
    colors, report = L.try_witness(*small(), deadline=10., clock=lambda: now[0],
                                   max_nodes=10, max_seconds=1.)
    assert L.exact_witness(colors, *small()) and len(calls) == 2 and calls[1][-1] is False
    assert calls[0][3:5] == (1., 9) and calls[1][3:5] == (.75, 6)
    assert report['nodes'] == 6 and report['attempt_statuses'] == [2, 0]
    assert report['presolve_retry'] and not report['negative_certified']


@pytest.mark.parametrize('status,candidate', [(1, None), (3, None), (4, None), (2, [0.] * 6)])
def test_other_status_or_timeout_does_not_expand_the_retry_trigger(monkeypatch, status, candidate):
    calls = []

    def optimize(*args):
        calls.append(args)
        return SimpleNamespace(status=status, x=candidate, mip_node_count=0)

    monkeypatch.setattr(L, '_optimize', optimize)
    colors, report = L.try_witness(*small(), deadline=time.monotonic() + 5,
                                   clock=time.monotonic, max_nodes=100)
    assert colors is None and len(calls) == 1 and not report['presolve_retry']
    assert not report['negative_certified']


@pytest.mark.parametrize('second_status,second_candidate', [(1, None), (2, None), (0, [0.] * 6)])
def test_unproved_no_presolve_result_continues_original_farkas_and_dfs(monkeypatch, second_status, second_candidate):
    from mirrorforce.netduel import causal_farkas as F
    events = []

    def optimize(*args):
        events.append('no-presolve' if len(args) == 7 and args[-1] is False else 'presolve')
        return SimpleNamespace(status=second_status if len(events) == 2 else 2,
                               x=second_candidate if len(events) == 2 else None, mip_node_count=0)

    def certificate(*args, **kwargs):
        events.append('farkas')
        return None, {'nodes': 1, 'negative_certified': False}

    monkeypatch.setattr(L, '_optimize', optimize)
    monkeypatch.setattr(F, 'try_certificate', certificate)
    domains, constraints = small()
    domains += [{3} for _ in range(60)]
    solver = _Solver(constraints, len(domains), 0, 1000000, time.monotonic() + 30, time.monotonic)
    colors = solver.feasible_witness(domains)
    assert events == ['presolve', 'no-presolve', 'farkas'] and solver.operations > 0
    assert L.exact_witness(colors, domains, constraints)
    assert solver.linear_infeasibility_certificate is None
    assert not solver.linear_witness_report['negative_certified']


@pytest.mark.parametrize('game,sequence,columns', [(27, 246, 8615), (30, 213, 8289)])
def test_real_false_infeasible_teacher_queries_get_exact_witness_without_dfs(game, sequence, columns):
    fixture = json.loads((Path(__file__).parent / 'fixtures' /
        f'capacity-false-infeasible-game{game}-row{sequence}.json').read_text())
    assert fixture['authority'] == 'PRIVILEGED_TARGET_DIAGNOSTIC' and fixture['inference_eligible'] is False
    assert fixture['complete_privileged_targets_included'] is False
    domains = [set(fixture['domain_table'][index]) for index in fixture['domain_indices']]
    constraints = [(tuple(left), tuple(right), None if fixed is None else Counter(dict(fixed)))
                   for left, right, fixed in fixture['constraints']]
    from mirrorforce.netduel.causal_farkas import problem_sha256
    assert problem_sha256(domains, constraints) == fixture['problem_sha256']
    assert sum(map(len, domains)) == columns
    original = [set(domain) for domain in domains]
    solver = _Solver(constraints, len(domains), 0, 1000000, time.monotonic() + 30, time.monotonic)
    colors = solver.feasible_witness(domains)
    assert L.exact_witness(colors, original, constraints) and domains == original
    assert solver.operations == 0 and solver.nodes < 100 and solver.linear_witness_report['accepted']
    assert solver.linear_infeasibility_report is None and not solver.linear_witness_report['negative_certified']


def test_real_original_teacher_problem_has_exact_witness_despite_invalid_presolve_incumbent():
    fixture = json.loads((Path(__file__).parent / 'fixtures/capacity-invalid-incumbent-row145.json').read_text())
    assert fixture['authority'] == 'PRIVILEGED_TARGET_DIAGNOSTIC'
    assert fixture['inference_eligible'] is False and fixture['complete_privileged_targets_included'] is False
    domains = [set(fixture['domain_table'][index]) for index in fixture['domain_indices']]
    constraints = [(tuple(left), tuple(right), None if fixed is None else Counter(dict(fixed)))
                   for left, right, fixed in fixture['constraints']]
    from mirrorforce.netduel.causal_farkas import problem_sha256
    assert problem_sha256(domains, constraints) == fixture['problem_sha256']
    assert len(domains) == 352 and sum(map(len, domains)) == 6274 and len(constraints) == 21
    before = [set(domain) for domain in domains]
    solver = _Solver(constraints, len(domains), 0, 1000000, time.monotonic() + 5, time.monotonic)
    colors = solver.feasible_witness(domains)
    assert L.exact_witness(colors, before, constraints) and domains == before
    assert solver.linear_witness_report['accepted'] and solver.operations == 0
    assert solver.linear_infeasibility_report is None and solver.nodes < 100
