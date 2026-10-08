"""An expired root rejects its whole bank, while retaining handles for cleanup."""
from types import SimpleNamespace

import pytest

from mirrorforce.netduel import agent_rollout as R
from mirrorforce.puzzle.messages import Message


def make_root(**kwargs):
    return R.RootLines(sock=None, driver=None, api=None, restore=lambda _: None, starts=[0],
        root_message=None, own_session="own", opponent_session=lambda start, line: "other" + str(line),
        seat=0, rows=2, lines_per_row=1, depth=1, td_lambda=1, seed=4, **kwargs)


def test_already_expired_root_does_not_allocate_a_single_session(monkeypatch):
    monkeypatch.setattr(R.time, "monotonic", lambda: 10)
    root = make_root()
    root._call = lambda _: pytest.fail("expired root must allocate nothing")
    with pytest.raises(R.RolloutError, match="session initialization"):
        root.run(deadline=10)
    assert all(not line.sessions for line in root.lines)


@pytest.mark.parametrize("overdue", ["own", "other"])
def test_slow_clone_is_owned_before_expiration_and_all_handles_can_be_closed(monkeypatch, overdue):
    now = [10]
    monkeypatch.setattr(R.time, "monotonic", lambda: now[0])
    root = make_root()
    closed = []

    def call(request):
        if request["op"] == "close":
            closed.append(request["session"])
            return {}
        assert request["op"] == "clone"
        if overdue == "own":
            now[0] = 12
        return {"session": "new-own"}

    def other(*_):
        now[0] = 12
        return "new-other"

    root._call, root.opponent_session = call, other
    with pytest.raises(R.RolloutError, match="time ran out"):
        root.run(deadline=12)
    assert root.lines[0].sessions == ({0: "new-own"} if overdue == "own"
                                      else {0: "new-own", 1: "new-other"})
    root.close()
    assert set(closed) == ({"new-own"} if overdue == "own" else {"new-own", "new-other"})
    assert "own" not in closed


def test_late_chunk_never_starts_a_native_line_or_certifies_partial_returns(monkeypatch):
    now, calls = [10], []
    monkeypatch.setattr(R.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(R, "MAX_ITEMS", 1)
    root = make_root()

    def call(request):
        calls.append(request["op"])
        if request["op"] == "clone":
            return {"session": "clone" + str(request["seed"])}
        now[0] = 12
        return {"items": [{"decisions": [], "response": "00"}]}

    root._call = call
    root._run = lambda *_a, **_k: pytest.fail("do not advance an overdue service response")
    with pytest.raises(R.RolloutError, match="result processing"):
        root.run(deadline=12)
    assert calls.count("rollout_step") == 1
    assert all(line.value is None for line in root.lines)


def test_native_run_receives_only_remaining_allocation(monkeypatch):
    now, caps = [10], []
    monkeypatch.setattr(R.time, "monotonic", lambda: now[0])

    class Driver:
        pduel = 1
        core = SimpleNamespace(log=[])
        finished, winner = False, None

        def _answer(self, *_):
            self._replay = []
            now[0] = 11.5

        def run(self, _, *, max_steps, max_seconds):
            caps.append(max_seconds)
            self.finished, self.winner = True, 0

    root = make_root(max_seconds=60)
    root.api = SimpleNamespace(duel_snapshot_free=lambda _: None)
    root.restore = lambda _: R.RestoredRoot(Driver(), Message(11, b"\0"))
    root._deadline = 12
    root._run(root.lines[0], b"\0", at_root=True)
    assert caps == [.5]  # The native loop is not granted a fresh 60 seconds.


@pytest.mark.parametrize("bad", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_relative_allowance_refused(bad):
    with pytest.raises(ValueError, match="time allowance"):
        make_root(max_seconds=bad)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf")])
def test_invalid_absolute_deadline_refused(bad):
    with pytest.raises(ValueError, match="absolute"):
        make_root().run(deadline=bad)
