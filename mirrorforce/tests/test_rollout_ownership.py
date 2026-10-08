"""Particle handles stay with their lines; failed lines never silently reweight actions."""
from types import SimpleNamespace

import pytest

from mirrorforce.netduel import agent_rollout as R
from mirrorforce.puzzle.messages import Message


def test_invalid_line_rejects_whole_root_instead_of_reweighting_surviving_actions():
    root = R.RootLines(sock=None, driver=None, api=None, restore=lambda _: None, starts=[0],
                       root_message=None, own_session="own", opponent_session=lambda *_: "other", seat=0,
                       rows=2, lines_per_row=1, depth=1, td_lambda=1, seed=4)
    counter = 0

    def call(request):
        nonlocal counter
        if request["op"] == "clone":
            counter += 1
            return {"session": str(counter)}
        return {"items": [{"decisions": [], "response": "00"} for _ in request["items"]]}

    def run(line, *_args, **_kwargs):
        if line.row == 0:
            raise R.EnvLawViolation("controlled invalid hypothetical activation")
        line.end, line.value = "terminal", 1

    root._call, root._run = call, run
    with pytest.raises(R.RolloutError, match="per-action particle budgets"):
        root.run()
    assert root.stats["void"] == 1 and root.lines[1].value == 1


def test_each_interleaved_line_uses_its_own_duel_for_snapshot_and_rollback():
    calls = []

    class Driver:
        def __init__(self, duel):
            self.pduel = duel
            self.core = SimpleNamespace(log=[])
            self.finished, self.winner = False, None

        def _answer(self, message, responder):
            self._replay = []

        def run(self, *_args, **_kwargs):
            self._answering_msg, self._answering_payload = 11, b"\0"
            raise R._Park()

        def save_pystate(self):
            return {"duel": self.pduel}

        def restore_pystate(self, state, consume=False):
            assert state["duel"] == self.pduel

    class API:
        def duel_snapshot(self, duel):
            calls.append(("snapshot", duel))
            return duel * 100

        def duel_rollback(self, duel, snap):
            calls.append(("rollback", duel, snap))
            assert snap == duel * 100
            return 0

        def duel_snapshot_free(self, snap):
            calls.append(("free", snap))

    drivers = [Driver(1), Driver(2)]
    message = Message(11, b"\0")
    root = R.RootLines(sock=None, driver=None, api=API(),
                       restore=lambda start: R.RestoredRoot(drivers[start], message), starts=[0, 1],
                       root_message=None, own_session="own", opponent_session=lambda *_: "other", seat=0,
                       rows=2, lines_per_row=1, depth=1, td_lambda=1, seed=4)
    for line in root.lines:
        root._run(line, b"\0", at_root=True)
    for line in reversed(root.lines):
        root._run(line, b"\0", at_root=False)
    assert ("rollback", 1, 100) in calls and ("rollback", 2, 200) in calls
    assert drivers[0].pduel == 1 and drivers[1].pduel == 2
    for line in root.lines:
        root._release(line)
    assert calls.count(("free", 100)) == calls.count(("free", 200)) == 2


def test_original_response_is_remapped_once_before_the_root_engine_answer_only():
    received, mapped = [], []
    class Driver:
        pduel, finished, winner = 7, False, None
        core = SimpleNamespace(log=[])
        def _answer(self, message, responder):
            received.append(self._replay.pop(0))
        def run(self, *_args, **_kwargs):
            self._answering_msg, self._answering_payload = 11, b"\0"
            raise R._Park()
        def save_pystate(self):
            return {}
        def restore_pystate(self, state, consume=False):
            assert state == {}
    def translate(raw):
        mapped.append(raw)
        assert raw == b"server-root"
        return b"local-root"
    driver = Driver()
    api = SimpleNamespace(duel_snapshot=lambda _: 1, duel_rollback=lambda *_: 0,
                          duel_snapshot_free=lambda _: None)
    root = R.RootLines(sock=None, driver=None, api=api,
        restore=lambda _: R.RestoredRoot(driver, Message(11, b"\0"), translate), starts=[0],
        root_message=None, own_session="own", opponent_session=lambda *_: "other", seat=0,
        rows=2, lines_per_row=1, depth=1, td_lambda=1, seed=4)
    line = root.lines[0]
    root._run(line, b"server-root", at_root=True)
    root._run(line, b"already-local-future", at_root=False)
    root._release(line)
    assert mapped == [b"server-root"]
    assert received == [b"local-root", b"already-local-future"]


def test_cleanup_attempts_every_session_on_error_and_never_closes_producer_duels():
    root = R.RootLines(sock=None, driver=SimpleNamespace(pduel=37), api=None, restore=lambda _: None, starts=[0],
                       root_message=None, own_session="real-own", opponent_session=lambda *_: "other", seat=0,
                       rows=2, lines_per_row=1, depth=1, td_lambda=1, seed=4)
    root.lines[0].sessions = {0: "clone-a", 1: "synthetic-a"}
    root.lines[1].sessions = {0: "clone-b", 1: "synthetic-b"}
    called = []

    def failed(request):
        called.append(request["session"])
        raise RuntimeError("controlled service close failure")

    root._call = failed
    with pytest.raises(R.RolloutError, match="every owned"):
        root.close()
    assert set(called) == {"clone-a", "clone-b", "synthetic-a", "synthetic-b"}
    assert root.driver.pduel == 37 and "real-own" not in called
    root.close()
    assert len(called) == 4
    with pytest.raises(R.RolloutError, match="closed"):
        root.run()


def test_wide_menu_candidates_keep_their_original_service_row_numbers():
    root = R.RootLines(sock=None, driver=None, api=None, restore=lambda _: None, starts=[0],
                       root_message=None, own_session="own", opponent_session=lambda *_: "other", seat=0,
                       rows=2, lines_per_row=1, depth=1, td_lambda=1, seed=4, candidate_rows=[4, 1])
    seen = []

    def call(request):
        if request["op"] == "clone":
            return {"session": "clone" + str(request["seed"])}
        seen.extend(item["index"] for item in request["items"])
        return {"items": [{"decisions": [], "response": "00"} for _ in request["items"]]}

    def run(line, *_args, **_kwargs):
        line.end, line.value = "terminal", 0

    root._call, root._run = call, run
    assert root.run() == {0: [(0, 0)], 1: [(0, 0)]}
    assert seen == [4, 1]
