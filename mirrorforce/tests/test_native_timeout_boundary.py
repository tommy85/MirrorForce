"""Only the current native advance's expired search cap is a budget fallback."""
from types import SimpleNamespace

import pytest

from mirrorforce.netduel import agent_anytime as A, agent_rollout as R
from mirrorforce.worldmodel import engine as E


class ControlledDriver:
    run = E.DuelDriver.run

    def __init__(self, now):
        self.pduel = 1
        self.steps, self.turn = 537, 3
        self.winner, self.finished = None, False
        self.answer_error = None
        self.process_calls = 0

        def process(_):
            self.process_calls += 1
            now[0] += .02
            return 0

        self.core = SimpleNamespace(log=[], process=process)

    def _answer(self, message, responder):
        if self.answer_error is not None:
            raise self.answer_error
        self._replay.pop(0)


def owner(now, monkeypatch, *, index=0, seed=1):
    monkeypatch.setattr(R.time, "monotonic", lambda: now[0])
    driver = ControlledDriver(now)
    root = R.RootLines(sock=None, driver=driver, api=None, restore=lambda _: None,
        starts=[index], root_message=None, own_session="own", opponent_session=lambda *_: "other",
        seat=0, rows=2, lines_per_row=1, depth=1, td_lambda=1., seed=seed,
        max_seconds=5., seed_law="common-stripe/v1")
    root._deadline = now[0] + .01
    return root, driver


def advance(root):
    root._run(root.lines[0], b"\x00", at_root=True)


def test_real_engine_positive_cap_rounded_to_zero_converts(monkeypatch):
    now = [0.]
    root, driver = owner(now, monkeypatch)
    with pytest.raises(R.RolloutBudgetExceeded) as caught:
        advance(root)
    cause = caught.value.__cause__
    assert type(cause) is E.DuelDeadlineExceeded
    assert cause.max_seconds == .01 and cause.seconds == .02
    assert "0.0s" in str(cause) and cause.steps == 538 and cause.turn == 3
    assert driver.process_calls == 1 and type(driver) is ControlledDriver
    assert root.lines[0].value is None


def test_already_expired_budget_never_enters_native_run(monkeypatch):
    now = [0.]
    root, driver = owner(now, monkeypatch)
    now[0] = root._deadline
    with pytest.raises(R.RolloutBudgetExceeded, match="before native advance"):
        advance(root)
    assert driver.process_calls == 0 and type(driver) is ControlledDriver


def test_expired_rollback_error_is_still_fatal(monkeypatch):
    now = [0.]
    root, driver = owner(now, monkeypatch)
    root.lines[0].driver = driver
    root.api = SimpleNamespace(duel_rollback=lambda *_: 1)
    now[0] = .02
    with pytest.raises(R.RolloutError, match="could not roll back") as caught:
        root._run(root.lines[0], b"\x00", at_root=False)
    assert not isinstance(caught.value, R.RolloutBudgetExceeded)
    assert driver.process_calls == 0 and type(driver) is ControlledDriver


class OtherDeadline(E.DuelDeadlineExceeded):
    pass


@pytest.mark.parametrize("case", ["early", "wrongcap", "missingcap", "subclass", "equal",
    "nan", "inf", "no_deadline", "changed_deadline", "ordinary"])
def test_nonmatching_native_exception_keeps_identity(monkeypatch, case):
    now = [0.]
    root, driver = owner(now, monkeypatch)
    if case == "no_deadline":
        root._deadline = None
    errors = []

    def run(self, responder, *, max_steps, max_seconds):
        now[0] = .005 if case == "early" else 10.
        if case == "changed_deadline":
            root._deadline += 1.
        cls = OtherDeadline if case == "subclass" else E.DuelDeadlineExceeded
        seconds = {"equal": max_seconds, "nan": float("nan"), "inf": float("inf")}.get(case, max_seconds + .02)
        cap = {"wrongcap": max_seconds / 2, "missingcap": None}.get(case, max_seconds)
        exc = RuntimeError("unrelated native failure") if case == "ordinary" else cls(
            "controlled", seconds=seconds, steps=538, turn=3, max_seconds=cap)
        errors.append(exc)
        raise exc

    monkeypatch.setattr(ControlledDriver, "run", run)
    with pytest.raises(Exception) as caught:
        advance(root)
    assert caught.value is errors[0]
    assert type(driver) is ControlledDriver


@pytest.mark.parametrize("where", ["answer", "restore"])
def test_matching_timeout_outside_run_is_fatal(monkeypatch, where):
    now = [0.]
    root, driver = owner(now, monkeypatch)
    exc = E.DuelDeadlineExceeded("outside native run", seconds=.02, steps=538, turn=3, max_seconds=.01)
    if where == "answer":
        driver.answer_error = exc
        now[0] = .02
    else:
        def restore(_):
            now[0] = .02
            raise exc
        root.restore = restore
    with pytest.raises(E.DuelDeadlineExceeded) as caught:
        advance(root)
    assert caught.value is exc and driver.process_calls == 0
    assert type(driver) is ControlledDriver


def test_script_error_dominates_real_native_timeout(monkeypatch):
    now = [0.]
    root, driver = owner(now, monkeypatch)
    process = driver.core.process

    def bad_process(handle):
        driver.core.log.append("script failure")
        return process(handle)

    driver.core.process = bad_process
    with pytest.raises(R.EnvLawViolation, match="script failure"):
        advance(root)
    assert type(driver) is ControlledDriver


def test_diagnostic_time_cannot_convert_early_timeout(monkeypatch):
    now = [0.]
    root, driver = owner(now, monkeypatch)
    exc = E.DuelDeadlineExceeded("early", seconds=.02, steps=538, turn=3, max_seconds=.01)

    def run(*args, **kwargs):
        raise exc

    def diagnostics(*args):
        now[0] = 1.

    monkeypatch.setattr(ControlledDriver, "run", run)
    root._script_errors = diagnostics
    with pytest.raises(E.DuelDeadlineExceeded) as caught:
        advance(root)
    assert caught.value is exc


@pytest.mark.parametrize("cleanup_error", [False, True])
def test_real_native_timeout_discards_partial_stripe_keeps_prefix(monkeypatch, cleanup_error):
    now, closed, drivers = [0.], [], []

    def make(index, seed):
        root, driver = owner(now, monkeypatch, index=index, seed=seed)
        drivers.append(driver)

        def run(*, deadline):
            root._deadline = deadline
            if index == 0:
                for line, value in zip(root.lines, (.25, -.5)):
                    line.value = value
                return {0: [(0, .25)], 1: [(0, -.5)]}
            root.lines[1].value = 999.
            advance(root)

        def close():
            closed.append(index)
            if cleanup_error and index == 1:
                raise RuntimeError("cleanup failure")

        root.run, root.close = run, close
        return root

    kwargs = dict(rows=2, weights=(1., 7., 9.), seed=11, deadline=.01,
                  make_runner=make, clock=lambda: now[0])
    if cleanup_error:
        with pytest.raises(RuntimeError, match="cleanup failure"):
            A.balanced_stripes(**kwargs)
    else:
        q, report = A.balanced_stripes(**kwargs)
        assert q == [.25, -.5]
        assert report["completed_indices"] == [0]
        assert report["stripes"][1]["finished_lines"] == 1
        assert report["stripes"][1]["discarded_lines"] == 2
        assert report["budget_exhausted"] and not report["budget_zero_search"]
        assert A.check_stripes(report, rows=2, planned=3, seed=11) == 1
    assert closed == [0, 1] and drivers[1].process_calls == 1
