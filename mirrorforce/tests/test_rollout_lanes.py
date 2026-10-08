"""Bounded live clones preserve all candidate rows and the noninitial root memory."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mirrorforce.netduel import agent_anytime as A, agent_rollout as R
from tools import mf_runtime_policy_service as S


class Builder:
    def __init__(self, rows):
        self.rows, self.selected, self.remaining = rows, -1, 2
    def clone(self): return copy.deepcopy(self)
    def prompt(self): return (0, 15, [{}] * (self.rows if self.selected < 0 else 2)) if self.remaining else None
    def observation(self): return {"obs:selection": np.asarray([self.selected], np.int32)}
    def step(self, row):
        self.remaining -= 1
        if self.selected < 0: self.selected = row
        if not self.remaining: return int(self.selected).to_bytes(4, "little")
    def forced_count(self): return 0


def noninitial_memory():
    return tuple((np.full((1, 3, 4), .5, np.float32), np.asarray([2], np.int32), np.asarray([5], np.int32),
                  np.full((1, 2, 4), .75, np.float32), np.asarray([1], np.int32), np.asarray([6], np.int32))
                 for _ in range(2))


class Backend(S.UniformBackend):
    def initial_state(self): return noninitial_memory()
    def act(self, observation, state, first, rows):
        assert first is False and state[0][1].item() == 2
        after = copy.deepcopy(state)
        after[0][1][:] += 1
        selected = int(observation["obs:selection"][0])
        win = .5 + .001 * (selected % 100)
        return after, np.zeros(rows), 0., [win, 0., 1. - win]


def service(logits, memory=None):
    svc = S.Service(None, Backend(), {}, "greedy", 1., {}, client_factory=lambda *a, **k: Builder(len(logits)))
    name = svc.open({"seat": 0, "main": [11], "extra": [], "seed": 7})["session"]
    session = svc.sessions[name]
    if memory is not None: session.state = memory
    session.first = False
    session.pending = {"msg": 15, "rows": len(logits), "logits": np.asarray(logits), "value": 0.,
        "wdl": [.5, 0., .5], "groups": [0] * len(logits), "obs_sha256": S.observation_sha256(session.client.observation())}
    other = svc.clone({"session": name, "seed": 9})["session"]
    return svc, name, other


def run(logits, *, lanes, deadline, clock, memory=None, fault=None):
    svc, own, base = service(logits, memory)
    before = S.memory_sha256(svc.sessions[own].state)
    peaks, cloned, closed = [len(svc.sessions)], [], []
    def call(request):
        if request["op"] in ("clone", "close", "rollout_step", "commit"):
            clock[0] += .04
        if fault == request["op"]:
            raise RuntimeError("controlled " + fault + " RPC failure")
        result = getattr(svc, request["op"])(request)
        peaks.append(len(svc.sessions))
        if request["op"] == "clone": cloned.append(result["session"])
        if request["op"] == "close": closed.append(request["session"])
        return result
    runners = []
    def make(index, seed):
        owner = R.RootLines(sock=None, driver=None, api=None, restore=lambda _: None, starts=[index],
            root_message=None, own_session=own, opponent_session=lambda *_: call(
                {"op": "clone", "session": base, "seed": seed})["session"], seat=0, rows=len(logits),
            lines_per_row=1, depth=1, td_lambda=1., seed=seed, max_seconds=60.,
            seed_law="common-stripe/v1", max_live_lines=lanes)
        owner._call = call
        runners.append(owner)
        return owner
    q, audit = A.balanced_stripes(rows=len(logits), weights=(1.,), seed=17, deadline=deadline,
                                   make_runner=make, clock=lambda: clock[0])
    assert S.memory_sha256(svc.sessions[own].state) == before
    assert set(svc.sessions) == {own, base}
    assert set(closed) == set(cloned) and len(closed) == len(cloned)
    call({"op": "close", "session": base})
    chosen = int(np.argmax(logits))
    call({"op": "commit", "session": own, "index": chosen})
    call({"op": "commit", "session": own, "index": 0})
    return q, audit, runners[0], max(peaks), clock[0]


def test_thousand_row_timeout_cleans_bounded_clones_and_commits_before_40s(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(R.time, "monotonic", lambda: clock[0])
    q, audit, owner, peak, wall = run(np.arange(1000), lanes=8, deadline=37., clock=clock)
    assert len(owner.lines) == len(q) == 1000 and q == [None] * 1000
    assert audit["completed_stripes"] == 0 and audit["budget_zero_search"]
    assert any(line.value is not None for line in owner.lines)  # prior chunks were still discarded for every row
    assert owner.stats["peak_owned_sessions"] <= 16 and peak <= 18
    assert wall <= 40. and all(not line.sessions for line in owner.lines)
    assert all(line.pystate is line.message is line.driver is None and line.outbox == {0: [], 1: []}
               for line in owner.lines)
    A.check_stripes(audit, rows=1000, planned=1, seed=17)


def test_bounded_and_old_unbounded_paths_have_identical_complete_row_values(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(R.time, "monotonic", lambda: clock[0])
    old = run(np.arange(25), lanes=None, deadline=37., clock=clock)[0]
    clock[0] = 0.
    new, audit, owner, peak, _ = run(np.arange(25), lanes=8, deadline=37., clock=clock)
    assert new == old and len(new) == 25 and all(value is not None for value in new)
    assert audit["completed_stripes"] == 1 and owner.stats["chunks"] == 4 and peak <= 18


def test_real_gui_overwide_menu_and_real_noninitial_memory(monkeypatch):
    pytest.importorskip("flax", reason="real ReferenceAgent recurrent initializer needs the environment")
    from mirrorforce.agent.model.reference_agent import ReferenceAgent
    path = Path(__file__).resolve().parents[1] / "infra/search-20261003/gui8-3b178149-jobs/gui8-results/output/paired-r1/game-000/search-attempt-5b4205c5445e9b679ebed9275783f8f6b17dfd1ba5e6db8d491d15f4e0cfccaf.json"
    if not path.is_file(): pytest.skip("the original GUI8 public search report is not present")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == path.stem.removeprefix("search-attempt-")
    reports = json.loads(raw)["policy_report"]["reports"]
    menu = max((row for row in reports if not row["forced"]), key=lambda row: row["rows"])
    assert menu["rows"] == len(menu["logits"]) == 25  # original complete legal SELECT_CARD menu, no top-prior subset
    agent = object.__new__(ReferenceAgent)
    agent.config = SimpleNamespace(memory_slots=3, chunk_slots=2, d=4)
    memory = tuple(agent.init_rnn_state(1) for _ in range(2))
    for state, expected in zip(memory, noninitial_memory()):
        for leaf, value in zip(state, expected): leaf[...] = value
    clock = [0.]
    monkeypatch.setattr(R.time, "monotonic", lambda: clock[0])
    q, audit, owner, peak, wall = run(menu["logits"], lanes=8, deadline=37., clock=clock, memory=memory)
    assert len(q) == 25 and all(value is not None for value in q)
    assert audit["completed_stripes"] == 1 and len(owner.lines) == 25 and peak <= 18 and wall <= 40.


@pytest.mark.parametrize("fault", ["clone", "close", "rollout_step"])
def test_rpc_faults_are_hard_not_zero_search(monkeypatch, fault):
    clock = [0.]
    monkeypatch.setattr(R.time, "monotonic", lambda: clock[0])
    with pytest.raises((RuntimeError, R.RolloutError), match="failure|cleanup"):
        run(np.arange(25), lanes=8, deadline=37., clock=clock, fault=fault)
