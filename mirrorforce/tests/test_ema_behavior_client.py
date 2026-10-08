"""CPU-only diagnostic handshake, finite-memory transaction and default regression."""
import copy

import numpy as np
import pytest

from mirrorforce.netduel import ema_behavior as A
from mirrorforce.netduel import agent_wire as W
from mirrorforce.netduel.agent_policy import RemotePolicy
from mirrorforce.agent.train import debiased_policy as D
from tools import mf_runtime_policy_service as S


def identity(candidate=False, audit=True):
    result = {"protocol": W.PROTOCOL, "weights": "debiased_ema" if candidate else "iterate",
              "selection": "greedy"}
    if audit:
        result["behavior_finite_audit"] = copy.deepcopy(A.FINITE_IDENTITY)
    if candidate:
        derived = {"schema": "mirrorforce_debiased_ema_identity/v1", "weights": "debiased_ema",
                   "purpose": A.PURPOSE, "static_gate_passed": True, "behavior_gate_passed": False,
                   "formal_ema_evaluation_eligible": False, "training_eligible": False}
        result.update(D.service_identity(derived, {"sha256": "1" * 64}))
    return result


class Backend:
    def __init__(self, fault=None):
        self.fault, self.calls = fault, 0

    def initial_state(self):
        return {"h": np.zeros((1, 2), np.float32), "nested": [np.zeros((1, 3), np.float32)]}

    def act(self, obs, state, first, rows):
        self.calls += 1
        state = copy.deepcopy(state)
        state["h"] += 1
        state["nested"][0] += 2
        logits, value, wdl = np.array([1., 2.]), .25, [.5, .25, .25]
        if self.fault == "logits":
            logits[0] = np.nan
        elif self.fault == "value":
            value = np.inf
        elif self.fault == "wdl":
            wdl[2] = np.nan
        elif self.fault == "memory-h":
            state["h"][0, 1] = np.nan
        elif self.fault == "memory-nested":
            state["nested"][0][0, 2] = -np.inf
        elif self.fault == "shape":
            logits = np.ones(3)
        return state, logits, value, wdl

    def act_batch(self, items):
        return [self.act(*item) for item in items]


class Builder:
    def __init__(self):
        self.pending, self.commits = False, []

    def feed(self, msg, body):
        if body:
            self.pending = True

    def prompt(self):
        return (0, 13, [{}, {}]) if self.pending else None

    def observation(self):
        return {"x": np.ones((1, 1), np.float32)}

    def step(self, index):
        self.commits.append(index)
        self.pending = False
        return bytes([index])

    def forced_count(self):
        return 0


def service(candidate=False, fault=None, audit=True):
    backend, builder, ident = Backend(fault), Builder(), identity(candidate, audit)
    svc = S.Service(None, backend, {}, "greedy", 1, ident, client_factory=lambda *a, **k: builder)
    request = {"op": "open", "seat": 0, "main": [1], "extra": [], "seed": 5}
    if candidate:
        request.update(purpose=A.PURPOSE,
                       debiased_ema_identity_sha256=ident["session_admission"]["debiased_ema_identity_sha256"])
    name = svc.dispatch(request)["session"]
    return svc, name, backend, builder


@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("fault", ["logits", "value", "wdl", "memory-h", "memory-nested", "shape"])
def test_any_bad_output_fails_before_memory_commit_or_action(candidate, fault):
    svc, name, backend, builder = service(candidate, fault)
    state = svc.sessions[name].state
    before = A.finite_memory(state)
    with pytest.raises((ValueError, RuntimeError)):
        svc.dispatch({"op": "decide", "session": name, "messages": [[13, "00"]]})
    session = svc.sessions[name]
    assert session.state is state and A.finite_memory(state) == before and session.first
    assert session.behavior_forward_checks == 0 and session.decisions == 0
    assert builder.commits == [] and backend.calls == 1  # No fallback/retry/alternate weight.


@pytest.mark.parametrize("candidate", [False, True])
def test_all_successful_forwards_and_initial_state_have_exact_session_counts(candidate):
    svc, name, backend, builder = service(candidate)
    initial = A.finite_memory(svc.sessions[name].state)
    for number in (1, 2):
        reply = svc.dispatch({"op": "decide", "session": name, "messages": [[13, "00"]]})
        row = reply["decisions"][0]
        assert row["behavior_finite_audit"] == {"law": A.FINITE_LAW, "passed": True,
            "forward_number": number, "memory_sha256": A.finite_memory(svc.sessions[name].state)}
        assert row["chosen"] == 1 and reply["response"] == "01"
    final = A.finite_memory(svc.sessions[name].state)
    closed = svc.dispatch({"op": "close", "session": name, "messages": []})
    assert closed["behavior_finite_audit"] == {"law": A.FINITE_LAW, "passed": True,
        "initial_state_checks": 1, "initial_memory_sha256": initial, "forward_count": 2, "memory_sha256": final}
    assert backend.calls == 2 and builder.commits == [1, 1]


@pytest.mark.parametrize("leaf", ["h", "nested"])
def test_invalid_initial_memory_cannot_register_a_session(leaf):
    backend = Backend()
    state = backend.initial_state()
    (state["h"] if leaf == "h" else state["nested"][0])[0, 0] = np.nan
    backend.initial_state = lambda: state
    svc = S.Service(None, backend, {}, "greedy", 1, identity(), client_factory=lambda *a, **k: Builder())
    with pytest.raises(ValueError, match="recurrent memory"):
        svc.dispatch({"op": "open", "seat": 0, "main": [1], "extra": [], "seed": 1})
    assert not svc.sessions and svc.counter == 0 and backend.calls == 0


def test_wrapper_checks_each_batched_return_and_retains_exact_objects():
    backend = Backend()
    guard = A.FiniteAuditBackend(backend)
    items = [({}, backend.initial_state(), True, 2), ({}, backend.initial_state(), False, 2)]
    result = guard.act_batch(items)
    assert len(result) == 2 and backend.calls == 2
    backend.fault = "memory-nested"
    with pytest.raises(ValueError, match="recurrent memory"):
        guard.act_batch(items)


@pytest.mark.parametrize("fault", [None, "value", "wdl", "memory-nested"])
def test_replay_belief_and_finite_audit_combo_checks_all_outputs_before_memory_commit(fault):
    class BeliefBackend(Backend):
        def act_batch_belief(self, items):
            self.last_result = [(*self.act(*item), {"codes": [11], "locations": [1, 2, 3, 4, 6, 7],
                                  "logits": np.zeros((1, 6, 4)).tolist()}) for item in items]
            return self.last_result
    backend, builder = BeliefBackend(fault), Builder()
    ident = {**identity(), "replay_belief": {"law": "same-public-forward-session-pending-count-head/v1"}}
    svc = S.Service(None, backend, {}, "greedy", 1., ident, client_factory=lambda *a, **k: builder)
    name = svc.open({"seat": 0, "main": [11], "extra": [], "seed": 7})["session"]
    session = svc.sessions[name]
    before, state = A.finite_memory(session.state), session.state
    builder.feed(13, b"\0")
    if fault:
        with pytest.raises(ValueError, match="nonfinite"):
            svc._score([session])
        assert session.state is state and A.finite_memory(state) == before
        assert session.behavior_forward_checks == 0 and session.pending is None and session.first
    else:
        result = svc.backend.act_batch_belief([({}, session.state, session.first, 2)])
        assert result is backend.last_result and result[0][4] is backend.last_result[0][4]
        svc._score([session])
        assert session.behavior_forward_checks == 1 and session.pending["value"] == .25
        assert session.pending["count_belief"] == backend.last_result[0][4]


@pytest.mark.parametrize("state", [None, {}, [], np.array([], np.float32), np.array(["text"]), {2: np.ones(1)}])
def test_non_memory_cannot_certify_finiteness(state):
    with pytest.raises(ValueError):
        A.finite_memory(state)


def test_bfloat16_memory_is_checked_without_conversion_or_mutation():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    state = np.array([1, 2], dtype=ml_dtypes.bfloat16)
    before = state.tobytes()
    assert len(A.finite_memory(state)) == 64 and state.tobytes() == before
    state[1] = np.nan
    with pytest.raises(ValueError):
        A.finite_memory(state)


@pytest.mark.parametrize("op", ["open", "open_stream"])
def test_candidate_only_adds_two_explicit_public_handshake_fields(monkeypatch, op):
    seen = []
    monkeypatch.setattr(RemotePolicy, "_call", lambda self, req: seen.append(copy.deepcopy(req)) or {})
    candidate = A.BehaviorRemotePolicy("unix:/unused", identity(True), candidate=True, seed=17)
    reference = A.BehaviorRemotePolicy("unix:/unused", identity(), candidate=False, seed=19)
    request = {"op": op, "seat": 1, "main": [11], "extra": [], "seed": 17}
    original = copy.deepcopy(request)
    candidate._call(request)
    reference._call(request)
    assert request == original and seen[1] == original
    assert set(seen[0]) - set(original) == {"purpose", "debiased_ema_identity_sha256"}
    assert seen[0]["purpose"] == A.PURPOSE
    assert candidate.report()["behavior_diagnostic"]["policy_seed"] == 17
    candidate._call({"op": "identity"})
    assert seen[-1] == {"op": "identity"}
    with pytest.raises(ValueError, match="override"):
        candidate._call({**request, "purpose": "other"})


def test_opt_in_does_not_mutate_callers_identity():
    ident = identity(True)
    policy = A.BehaviorRemotePolicy("unix:/unused", ident, candidate=True, seed=1)
    ident["debiased_ema"]["static_gate_passed"] = False
    assert policy.expected_identity["debiased_ema"]["static_gate_passed"] is True
    with pytest.raises(ValueError):
        A.BehaviorRemotePolicy("unix:/unused", ident, candidate=True, seed=1)


def test_default_service_output_remains_exact_pre_audit_shape(monkeypatch):
    monkeypatch.setattr(S.time, "perf_counter", lambda: 10.0)
    svc, name, backend, builder = service(audit=False)
    assert svc.backend is backend and svc.identity == identity(audit=False)
    reply = svc.dispatch({"op": "decide", "session": name, "messages": [[13, "00"]]})
    assert reply == {"response": "01", "forced": False, "decisions": [{"forced": False, "msg": 13,
        "rows": 2, "chosen": 1, "p_chosen": None, "value": .25, "wdl": [.5, .25, .25],
        "logits": [1., 2.], "obs_sha256": S.observation_sha256({"x": np.ones((1, 1), np.float32)}), "act_ms": 0.0}]}
    assert svc.dispatch({"op": "close", "session": name, "messages": []}) == {
        "decisions": 1, "forced": 0, "prompts": 1, "builder_forced": 0,
        "act_ms": {"p50": 0.0, "p90": 0.0, "max": 0.0, "first": 0.0}}


def test_audited_service_does_not_admit_search_ops():
    svc, name, _, _ = service()
    for op in ("clone", "prompt", "commit", "root_step", "rollout_step"):
        with pytest.raises(ValueError, match="not search"):
            svc.dispatch({"op": op, "session": name})


def test_cli_switch_defaults_off_before_native_load(monkeypatch, tmp_path):
    class Seen(Exception):
        pass

    def native(args):
        assert args.behavior_finite_audit is False
        raise Seen

    monkeypatch.setattr(S, "load_native", native)
    argv = ["--selection", "greedy", "--checkpoint", "/unused", "--semantic-file", "/unused", "--cards-db", "/unused",
            "--code-list", "/unused", "--script-root", "/unused", "--announce-tables", "/unused",
            "--socket", str(tmp_path / "socket"), "--out", str(tmp_path / "out")]
    with pytest.raises(Seen):
        S.main(argv)
    with pytest.raises(ValueError, match="greedy iterate/debiased"):
        S.main(argv + ["--behavior-finite-audit"])
