"""Scheduler tests; fake-policy numbers are NOT a GPU numerical admission."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from mirrorforce.netduel import agent_anytime as A, agent_rollout as R, agent_stripe_batching as B


class Harness:
    """Independent per-seat memory/RNG, variable lengths, and owned snapshots."""
    def __init__(self, monkeypatch, *, rows=3, particles=8, stalled=(), fault=None):
        self.rows, self.particles, self.stalled, self.fault = rows, particles, stalled, fault
        self.now, self.serial, self.snap_serial = 0., 0, 0
        self.sessions, self.snapshots, self.owners, self.trace = {}, set(), [], {}
        self.peaks, self.rpc_sizes, self.seeds = [], [], []
        self.closed, self.calls, self.native = [], [], []
        self.close_attempts = []
        self.deadline_after_rpc = None
        self.api = SimpleNamespace(duel_snapshot_free=self.free)
        monkeypatch.setattr(R.time, "monotonic", lambda: self.now)

    def free(self, snapshot):
        assert snapshot in self.snapshots
        self.snapshots.remove(snapshot)

    def call(self, request):
        op = request["op"]
        self.calls.append(op)
        self.now += .01 if op != "rollout_step" else .2
        if op == "close": self.close_attempts.append(request["session"])
        if op == self.fault:
            raise RuntimeError("controlled " + op + " error")
        if op == "close":
            assert request["session"] not in self.closed
            del self.sessions[request["session"]]
            self.closed.append(request["session"])
            return {}
        if op == "clone":
            self.serial += 1
            name = str(self.serial)
            self.sessions[name] = [np.random.default_rng(request["seed"]), 17, None]
            self.seeds.append(request["seed"])
            self.peaks.append(len(self.sessions))
            return {"session": name}
        assert op == "rollout_step"
        self.rpc_sizes.append(len(request["items"]))
        out = []
        for item in request["items"]:
            rng, memory, key = self.sessions[item["session"]]
            response = int(rng.integers(256)).to_bytes(1, "little").hex()
            win = .1 + .001 * (memory % 113)
            result = {"decisions": [{"wdl": [win, 0., 1 - win]}], "response": response,
                      "cut": item.get("stop_after") == 1}
            self.trace.setdefault(key, []).append((memory,
                copy.deepcopy({k: v for k, v in item.items() if k != "session"}), result))
            self.sessions[item["session"]][1] += 1
            out.append(result)
        if self.deadline_after_rpc == len(self.rpc_sizes):
            self.now = 30.
        if self.fault == "dropped": out.pop()
        return {"items": out}

    def make(self, index, seed, dispatch):
        owner = R.RootLines(sock=None, driver=None, api=self.api, restore=None, starts=[index],
            root_message=None, own_session="real-own", opponent_session=None, seat=0,
            rows=self.rows, lines_per_row=1, depth=5, td_lambda=.9, seed=seed,
            max_seconds=1000., seed_law="common-stripe/v1", max_live_lines=min(self.rows, 8))
        owner._call = dispatch
        def opponent(start, row):
            assert start == index
            own = owner.lines[row].sessions[0]
            self.sessions[own][2] = (index, row, 0)
            other = dispatch({"op": "clone", "session": "base", "seed": (seed * 1_000_003 + 71) & 0x7fffffff})["session"]
            self.sessions[other][2] = (index, row, 1)
            return other
        owner.opponent_session = opponent
        def advance(line, response, *, at_root):
            self.native.append((index, line.row, response, at_root))
            if self.fault == "native": raise RuntimeError("controlled native error")
            if self.fault == "void": raise R.EnvLawViolation("controlled script error")
            if line.snap: self.free(line.snap)
            self.snap_serial += 1
            line.snap = self.snap_serial
            self.snapshots.add(line.snap)
            line.seat = 1 if at_root or index in self.stalled else 1 - line.seat
            line.outbox[line.seat].append(b"\x01" + response)
        owner._run = advance
        self.owners.append(owner)
        return owner

    def run(self, *, multiplex=True, deadline=30., limits=None):
        args = dict(rows=self.rows, weights=tuple(float(i + 1) for i in range(self.particles)),
                    seed=17, deadline=deadline, clock=lambda: self.now)
        if multiplex:
            return B.multiplexed_stripes(**args, call=self.call, make_runner=self.make, limits=limits)
        return A.balanced_stripes(**args, make_runner=lambda i, s: self.make(i, s, self.call))


@pytest.mark.parametrize("rows", [2, 3, 8, 25])
def test_complete_serial_and_multiplexed_values_inputs_rng_memory_and_native_paths_match(monkeypatch, rows):
    serial = Harness(monkeypatch, rows=rows)
    expected, old = serial.run(multiplex=False, deadline=1000.)
    batched = Harness(monkeypatch, rows=rows)
    actual, new = batched.run(deadline=1000.)
    assert actual == expected
    assert new["completed_stripes"] == old["completed_stripes"] == 8
    assert batched.trace == serial.trace  # every request/response, not just Q
    assert sorted(batched.native) == sorted(serial.native)
    assert sorted(batched.seeds) == sorted(serial.seeds)
    assert len(batched.rpc_sizes) < len(serial.rpc_sizes)
    assert max(batched.rpc_sizes) <= 32 and max(batched.peaks) <= 64
    assert not batched.sessions and not batched.snapshots
    assert len(batched.closed) == batched.serial
    assert new["stats"]["service_seconds"] == pytest.approx(batched.now)
    assert new["resources"]["rollout_calls"] == len(batched.rpc_sizes)
    B.check_stripes(new, rows=rows, planned=8, seed=17)
    with pytest.raises(ValueError, match="different balanced"):
        A.check_stripes(new, rows=rows, planned=8, seed=17)
    with pytest.raises(ValueError, match="different multiplexed"):
        B.check_stripes(old, rows=rows, planned=8, seed=17)


@pytest.mark.parametrize("stalled,expected", [((0,), 0), ((1,), 1), ((2,), 2)])
def test_later_certified_particles_never_skip_an_unfinished_prefix(monkeypatch, stalled, expected):
    h = Harness(monkeypatch, stalled=stalled)
    h.deadline_after_rpc = 12
    q, audit = h.run()
    assert audit["completed_stripes"] == expected
    assert audit["completed_indices"] == list(range(expected))
    assert audit["budget_zero_search"] == (expected == 0) and audit["budget_exhausted"]
    assert len(audit["stripes"]) == 8 and audit["stripes"][-1]["certified"]
    assert audit["stripes"][-1]["discarded_lines"] == h.rows
    if expected == 0: assert q == [None] * h.rows
    else:
        reference = Harness(monkeypatch, particles=expected)
        assert q == reference.run(multiplex=False)[0]
    assert not h.sessions and not h.snapshots and all(owner.closed for owner in h.owners)
    B.check_stripes(audit, rows=h.rows, planned=h.particles, seed=17)


def test_expired_shared_reply_advances_no_engine_and_closes_every_clone(monkeypatch):
    h = Harness(monkeypatch)
    h.deadline_after_rpc = 1
    q, audit = h.run()
    assert not h.native and q == [None] * h.rows
    assert all(not s["finished_lines"] for s in audit["stripes"])
    assert not h.sessions and not h.snapshots
    B.check_stripes(audit, rows=h.rows, planned=8, seed=17)


@pytest.mark.parametrize("fault", ["clone", "rollout_step", "native", "void", "dropped"])
def test_hard_errors_do_not_become_budget_fallback_and_cleanup_is_exhaustive(monkeypatch, fault):
    h = Harness(monkeypatch, fault=fault)
    with pytest.raises((RuntimeError, R.RolloutError), match="error|incomplete|dropped"):
        h.run()
    assert not h.sessions and not h.snapshots and all(owner.closed for owner in h.owners)


def test_a_seen_void_is_fatal_even_when_the_next_round_expires(monkeypatch):
    h = Harness(monkeypatch, fault="void")
    h.deadline_after_rpc = 2
    with pytest.raises(R.RolloutError, match="incomplete|invalid native"):
        h.run()
    assert not h.sessions and not h.snapshots


def test_cleanup_failure_attempts_every_owner_and_session(monkeypatch):
    h = Harness(monkeypatch, fault="close")
    with pytest.raises(R.RolloutError, match="cleanup failed"):
        h.run()
    assert len(h.owners) == 8 and all(owner.closed for owner in h.owners)
    assert set(h.close_attempts) == set(h.sessions)
    # A failed chunk cleanup can be retried by owner.close(); a successful
    # close is never repeated. All failures remain fatal.
    assert h.calls.count("close") == 2 * h.serial
    assert not h.snapshots


@pytest.mark.parametrize("deadline", [0., -1.])
def test_already_expired_does_not_construct_an_owner(monkeypatch, deadline):
    h = Harness(monkeypatch)
    q, audit = h.run(deadline=deadline)
    assert not h.owners and not h.calls and q == [None] * h.rows
    B.check_stripes(audit, rows=h.rows, planned=8, seed=17)


@pytest.mark.parametrize("field,value", [("seed", 99), ("starts", [8]), ("max_live_lines", None),
                                        ("seed_law", "independent-lines/v1"), ("closed", True)])
def test_malformed_owner_rejected_before_its_first_clone(monkeypatch, field, value):
    h = Harness(monkeypatch)
    old = h.make
    def changed(*args):
        owner = old(*args)
        setattr(owner, field, value)
        return owner
    h.make = changed
    with pytest.raises(R.RolloutError, match="predeclared"):
        h.run()
    assert not h.calls


def test_duplicate_sessions_in_merged_batch_are_a_hard_error(monkeypatch):
    h = Harness(monkeypatch)
    old = h.make
    def changed(*args):
        owner = old(*args)
        original = owner.batches
        def batches(**kwargs):
            generator = original(**kwargs)
            try:
                request = next(generator)
                request["items"][1]["session"] = request["items"][0]["session"]
                yield request
            finally:
                generator.close()
        owner.batches = batches
        return owner
    h.make = changed
    with pytest.raises(R.RolloutError, match="duplicate"):
        h.run()
    assert not h.sessions and not h.snapshots


@pytest.mark.parametrize("patch", [{"max_live_lines":257}, {"max_live_lines":True},
    {"lines_per_stripe":33}, {"max_active_stripes":0}, {"unknown":8}])
def test_invalid_lane_reservations_rejected(monkeypatch, patch):
    h = Harness(monkeypatch)
    with pytest.raises(ValueError, match="limits"):
        h.run(limits={**B.DEFAULT_LIMITS, **patch})
    assert not h.calls


def test_new_audit_rejects_prefix_gaps_forged_bounds_and_legacy_law(monkeypatch):
    h = Harness(monkeypatch, stalled=(1,))
    h.deadline_after_rpc = 12
    _, good = h.run()
    changes = [lambda r: r.update(completed_indices=[0, 2]),
        lambda r: r["stripes"][1].update(index=True),
        lambda r: r.update(planned_stripes=True),
        lambda r: r["stripes"][2].update(complete=True),
        lambda r: r["stripes"][1].update(certified=True, finished_lines=h.rows),
        lambda r: r["stripes"][2].update(discarded_lines=0),
        lambda r: r["stripes"][2].update(seed=99),
        lambda r: r["stats"].update(lines=99),
        lambda r: r["stats"].update(void=False),
        lambda r: r["resources"].update(reserved_live_lines_peak=999),
        lambda r: r["resources"].update(peak_rpc_items=999),
        lambda r: r["resources"].update(rollout_calls=999999),
        lambda r: r.update(law=A.LAW)]
    for change in changes:
        bad = copy.deepcopy(good)
        change(bad)
        with pytest.raises(ValueError):
            B.check_stripes(bad, rows=h.rows, planned=8, seed=17)


def test_rpc_stop_iteration_is_not_an_owner_success(monkeypatch):
    h = Harness(monkeypatch)
    owner = h.make(0, 17, h.call)
    original = owner._call
    def call(request):
        if request["op"] == "rollout_step": raise StopIteration("invalid RPC result")
        return original(request)
    owner._call = call
    try:
        with pytest.raises(StopIteration, match="invalid RPC"):
            owner.run()
    finally:
        owner.close()
    assert not h.sessions and not h.snapshots


def test_real_service_preserves_original_pending_memory_and_rng(monkeypatch):
    from test_rollout_lanes import service
    from tools import mf_runtime_policy_service as S
    def run(multiplex):
        now = [0.]
        monkeypatch.setattr(R.time, "monotonic", lambda: now[0])
        svc, own, base = service(np.arange(25))
        before = S.memory_sha256(svc.sessions[own].state)
        rng = copy.deepcopy(svc.sessions[own].rng.bit_generator.state)
        pending = copy.deepcopy(svc.sessions[own].pending)
        peaks = []
        def call(request):
            now[0] += .01
            reply = getattr(svc, request["op"])(request)
            peaks.append(len(svc.sessions))
            return reply
        def make(index, seed, dispatch):
            root = R.RootLines(sock=None, driver=None, api=None, restore=None, starts=[index],
                root_message=None, own_session=own, opponent_session=lambda *_: dispatch(
                    {"op":"clone", "session":base, "seed":(seed*1_000_003+71)&0x7fffffff})["session"],
                seat=0, rows=25, lines_per_row=1, depth=1, td_lambda=1., seed=seed,
                max_live_lines=8, seed_law="common-stripe/v1")
            root._call = dispatch
            return root
        args = dict(rows=25, weights=(1.,)*8, seed=17, deadline=30., clock=lambda:now[0])
        if multiplex: q, audit = B.multiplexed_stripes(**args, make_runner=make, call=call)
        else: q, audit = A.balanced_stripes(**args, make_runner=lambda i,s:make(i,s,call))
        assert before == S.memory_sha256(svc.sessions[own].state)
        assert rng == svc.sessions[own].rng.bit_generator.state
        assert pending.keys() == svc.sessions[own].pending.keys()
        for key, value in pending.items(): np.testing.assert_array_equal(value,svc.sessions[own].pending[key])
        assert set(svc.sessions) == {own,base}
        svc.close({"session":base})
        svc.commit({"session":own,"index":24})
        svc.commit({"session":own,"index":0})
        return q, audit, max(peaks)
    old, _, _ = run(False)
    new, audit, peak = run(True)
    assert new == old and peak <= 66
    assert audit["completed_stripes"] == 8
