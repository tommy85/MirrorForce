from types import SimpleNamespace

import pytest

from mirrorforce.netduel import agent_anytime as A
from mirrorforce.netduel import agent_rollout as R


def test_explicit_prompt_total_cap_reserves_finalization_without_changing_old_deadline():
    arguments = dict(now=100., clock_received=100., clock_left=600., room_seconds=600., search_seconds=60.,
                     clock_reserve=30., clock_share=.1, response_margin=3., finalize_seconds=3.)
    old_search, old_response, old_record = A.public_deadlines(**arguments)
    assert old_search == 157. and old_response == 697. and 'total_cap_seconds' not in old_record
    new_search, new_response, new_record = A.public_deadlines(**arguments, total_seconds=40.)
    assert new_search == 137. and new_response == 140. and new_record['total_cap_seconds'] == 40.
    assert new_record['allocated_seconds'] == 37. and new_record['response_seconds'] == 40.
    for cap in (41., 3., float('nan'), True):
        with pytest.raises(ValueError):
            A.public_deadlines(**arguments, total_seconds=cap)


def fixture_factory(now, effects, *, rows=2):
    runners, calls = [], []

    def factory(index, seed):
        calls.append((index, seed))
        runner = SimpleNamespace(seed_law="common-stripe/v1", seed=seed, rows=rows, starts=[index], closed=False,
                                 lines=[SimpleNamespace(start=index, row=row, value=None) for row in range(rows)],
                                 stats={"rounds": 1, "items": rows, "service_seconds": .1,
                                        "engine_seconds": .2, "void": 0})

        def run(*, deadline):
            assert deadline == 10
            result = effects[index](runner)
            if isinstance(result, BaseException):
                raise result
            for line in runner.lines:
                line.value = result[line.row][0][1]
            return result

        def close():
            runner.closed = True

        runner.run, runner.close = run, close
        runners.append(runner)
        return runner
    return factory, runners, calls


def test_only_complete_balanced_prefix_enters_weighted_q():
    now = [0]

    def partial(runner):
        runner.lines[0].value = 1000.0
        now[0] = 10
        return R.RolloutBudgetExceeded("clock only")

    factory, runners, calls = fixture_factory(now, [lambda _: {0: [(0, 1.)], 1: [(0, -1.)]},
                                                   lambda _: {0: [(1, -1.)], 1: [(1, 1.)]}, partial])
    q, report = A.balanced_stripes(rows=2, weights=(1., 3., 7.), seed=11, deadline=10,
                                   make_runner=factory, clock=lambda: now[0])
    assert q == [-.5, .5]
    assert report["completed_indices"] == [0, 1] and report["completed_stripes"] == 2
    assert report["budget_exhausted"] and not report["budget_zero_search"]
    assert report["stats"]["lines"] == 4 and report["stats"]["attempted_lines"] == 6
    assert report["stripes"][-1]["discarded_lines"] == 2
    assert report["stripes"][-1]["finished_lines"] == 1
    assert [seed for _, seed in calls] == [11000033, 11000034, 11000035]
    assert all(runner.closed for runner in runners)
    assert A.check_stripes(report, rows=2, planned=3, seed=11) == 2
    from copy import deepcopy
    for key, value in (("completed_stripes", 3), ("budget_zero_search", True), ("budget_exhausted", False),
                       ("completed_indices", [0, 2])):
        bad = deepcopy(report)
        bad[key] = value
        with pytest.raises(ValueError):
            A.check_stripes(bad, rows=2, planned=3, seed=11)
    for key, value in (("seed", 0), ("discarded_lines", 1), ("complete", True)):
        bad = deepcopy(report)
        bad["stripes"][-1][key] = value
        with pytest.raises(ValueError):
            A.check_stripes(bad, rows=2, planned=3, seed=11)


def test_zero_budget_allocates_nothing_and_raw_greedy_keeps_full_prior():
    q, report = A.balanced_stripes(rows=3, weights=(1.,), seed=2, deadline=10,
                                   make_runner=lambda *_: pytest.fail("expired"), clock=lambda: 10)
    assert q == [None] * 3 and report["budget_zero_search"] and report["completed_stripes"] == 0
    chosen, update = A.greedy_without_search([2., 2., -5.])
    assert chosen == 0 and update["policy"] == update["prior"]
    assert sum(update["policy"]) == pytest.approx(1)


@pytest.mark.parametrize("failure", [R.RolloutError("rule mismatch"), ValueError("wrong source"),
                                    TimeoutError("socket timeout, not computation clock"), OSError("transport")])
def test_hard_errors_propagate_instead_of_turning_into_budget_fallback(failure):
    now = [0]
    factory, runners, _ = fixture_factory(now, [lambda _: failure])
    with pytest.raises(type(failure), match=str(failure)):
        A.balanced_stripes(rows=2, weights=(1.,), seed=1, deadline=10, make_runner=factory, clock=lambda: now[0])
    assert runners[0].closed


def test_void_before_timeout_is_still_a_hard_failure():
    now = [0]

    def effect(runner):
        runner.stats["void"] = 1
        return R.RolloutBudgetExceeded("clock")

    factory, runners, _ = fixture_factory(now, [effect])
    with pytest.raises(R.RolloutError, match="invalid native"):
        A.balanced_stripes(rows=2, weights=(1.,), seed=1, deadline=10, make_runner=factory, clock=lambda: now[0])
    assert runners[0].closed


@pytest.mark.parametrize("values", [{0: [(0, 1.)]}, {0: [(1, 1.)], 1: [(0, 1.)]},
                                   {0: [(0, float("nan"))], 1: [(0, 1.)]}])
def test_malformed_complete_stripe_is_not_a_partial_success(values):
    factory, runners, _ = fixture_factory([0], [lambda _: values])
    # Do not let the fixture itself assume missing rows exist.
    def make(index, seed):
        runner = factory(index, seed)
        runner.run = lambda **_: values
        return runner
    with pytest.raises(R.RolloutError, match="exactly one finite"):
        A.balanced_stripes(rows=2, weights=(1.,), seed=1, deadline=10, make_runner=make, clock=lambda: 0)
    assert runners[0].closed


def test_cleanup_error_does_not_get_hidden_by_a_successful_stripe():
    factory, _, _ = fixture_factory([0], [lambda _: {0: [(0, 1.)], 1: [(0, 1.)]}])
    def make(index, seed):
        runner = factory(index, seed)
        def fail():
            raise R.RolloutError("cleanup")
        runner.close = fail
        return runner
    with pytest.raises(R.RolloutError, match="cleanup"):
        A.balanced_stripes(rows=2, weights=(1.,), seed=1, deadline=10, make_runner=make, clock=lambda: 0)


def test_full_bank_finishes_without_budget_exhaustion():
    factory, _, _ = fixture_factory([0], [lambda _: {0: [(0, -1.)], 1: [(0, 0.)]}])
    q, report = A.balanced_stripes(rows=2, weights=(1.,), seed=0, deadline=10, make_runner=factory, clock=lambda: 0)
    assert q == [-1., 0.] and report["completed_stripes"] == 1
    assert not report["budget_exhausted"] and not report["budget_zero_search"]


def test_common_stripe_has_same_clone_rng_for_all_candidate_rows(monkeypatch):
    monkeypatch.setattr(R.time, "monotonic", lambda: 0)
    root = R.RootLines(sock=None, driver=None, api=None, restore=lambda _: None, starts=[4],
                       root_message=None, own_session="own", opponent_session=lambda *_: "other", seat=0,
                       rows=3, lines_per_row=1, depth=1, td_lambda=1., seed=7, seed_law="common-stripe/v1")
    seeds = []
    def call(request):
        if request["op"] == "clone":
            seeds.append(request["seed"])
            return {"session": "new" + str(len(seeds))}
        if request["op"] == "rollout_step":
            return {"items": [{"decisions": [{"wdl": [.5, 0., .5]}], "cut": True} for _ in request["items"]]}
        return {}
    root._call = call
    assert root.run(deadline=10) == {0: [(4, 0.)], 1: [(4, 0.)], 2: [(4, 0.)]}
    root.close()
    assert seeds == [7000021] * 3


def test_timing_aggregation_keeps_zero_search_and_separates_games_seats():
    rows = [{"game": game, "seat": seat, "turn": 1, "root_seconds": .1, "sample_seconds": .2,
             "rollout_seconds": 0., "update_seconds": .1, "total_seconds": total,
             "completed_stripes": stripes, "budget_zero_search": stripes == 0}
            for game, seat, total, stripes in [("a", 0, 1., 0), ("a", 0, 3., 1), ("a", 1, 2., 2), ("b", 0, 8., 2)]]
    report = A.timing_summary(rows)
    assert report["decisions"] == 4 and report["turns"] == 3 and report["worst_turn_seconds"] == 8
    assert report["phases"]["total_seconds"] == {"p50": 2.5, "p95": pytest.approx(7.25), "max": 8.}
    assert report["budget_zero_search_fraction"] == .25
    assert report["completed_stripes_histogram"] == {0: 1, 1: 1, 2: 2}
    with pytest.raises(ValueError, match="every phase"):
        A.timing_summary([{**rows[0], "rollout_seconds": -1}])


def test_public_clock_does_not_renew_between_phases_and_separates_cleanup_margin():
    args = dict(clock_received=100., clock_left=600, room_seconds=600, search_seconds=60.,
                clock_reserve=30., clock_share=.1, response_margin=3., finalize_seconds=3.)
    soft, hard, record = A.public_deadlines(now=102., **args)
    assert soft == pytest.approx(158.8) and hard == 697
    assert record["clock_elapsed"] == 2 and record["allocated_seconds"] == pytest.approx(56.8)
    soft2, hard2, _ = A.public_deadlines(now=690., **args)
    assert soft2 == 690 and hard2 == hard  # no room-clock reset; only raw-greedy still possible
    with pytest.raises(R.RolloutError, match="safe response"):
        A.public_deadlines(now=697., **args)
    for change in ({"clock_received": 103.}, {"clock_left": None}, {"clock_left": 601}, {"clock_share": 2.}):
        with pytest.raises(ValueError, match="received public clock"):
            A.public_deadlines(now=102., **{**args, **change})


def test_allocation_from_an_elapsed_clock_is_the_public_deadline_allocation():
    args = dict(clock_left=437, room_seconds=450, search_seconds=9, clock_reserve=30, clock_share=.1,
                response_margin=3, finalize_seconds=3, total_seconds=12)
    now, received = 450000.3333333, 450000.3123456
    assert A.allocation(now=now, elapsed=now - received, **args) == \
        A.public_deadlines(now=now, clock_received=received, **args)
    with pytest.raises(ValueError):
        A.allocation(now=now, elapsed=-1., **args)
