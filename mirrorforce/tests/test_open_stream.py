"""Fresh synthetic-stream sessions: response-path replay advances memory at EVERY sub-choice.

These protocol tests do not claim to implement Option A's particle-to-packet producer.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from tools import mf_runtime_policy_service as S


class MemoryBackend(S.UniformBackend):
    """Each forward depends on the exact observation, preceding memory, and first flag."""

    def initial_state(self):
        return (np.zeros((1, 2), np.uint64),)

    def act(self, obs, state, first, rows):
        digest = int(S.observation_sha256(obs)[:12], 16)
        old = state[0]
        next_state = (np.asarray([[int(old[0, 0]) + 1, (int(old[0, 1]) * 17 + digest + first) % (1 << 63)]],
                                 np.uint64),)
        return next_state, np.arange(rows, dtype=np.float64), float(next_state[0][0, 0]), [0.2, 0.3, 0.5]


class Builder:
    """10: two sub-choices, 11: one sub-choice, 12: forced response, everything else: history."""

    def __init__(self, *args, **kwargs):
        self.history, self.remaining, self.forced, self.chosen = [], 0, 0, []

    def feed(self, msg, payload):
        assert not self.remaining
        self.history.append((msg, list(payload)))
        if msg == 12:
            self.forced += 1
            return b"f"
        self.remaining = {10: 2, 11: 1}.get(msg, 0)
        self.chosen = []

    def prompt(self):
        return (0, 10, [{}, {}, {}]) if self.remaining else None

    def observation(self):
        return {"obs:x": np.asarray([len(self.history), self.remaining, sum(self.chosen)], np.uint8)}

    def step(self, row):
        self.remaining -= 1
        self.chosen.append(row)
        if not self.remaining:
            return bytes(self.chosen)

    def response_path(self, response):
        return list(response)

    def forced_count(self):
        return self.forced


def service(builder=Builder, *, config=None, identity=None):
    return S.Service(None, MemoryBackend(), config or {}, "sample", 1, identity or {}, client_factory=builder)


def request():
    return {"op": "open_stream", "seat": 1, "main": [1, 2], "extra": [], "seed": 7,
            "frames": [{"messages": [[1, "ab"], [10, ""]], "response": "0102"},
                       {"messages": [[12, ""]], "response": "66"},
                       {"messages": [[11, ""]], "response": "00"}], "messages": [[2, "cd"]]}


def test_replay_equals_live_forwards_including_subchoices_and_forced_prompts():
    svc, data = service(), request()
    live = svc.open({k: data[k] for k in ("seat", "main", "extra", "seed")})["session"]
    expected_trace = []
    for frame in data["frames"]:
        before = S.memory_sha256(svc.sessions[live].state)
        result = svc.prompt({"session": live, "messages": frame["messages"]})
        rows = list(bytes.fromhex(frame["response"])) if "pending" in result else []
        for row in rows:
            after = S.memory_sha256(svc.sessions[live].state)
            expected_trace.append((result["pending"]["obs_sha256"], before, after))
            before = after
            result = svc.commit({"session": live, "index": row})
        assert result["response"] == frame["response"]
    replay = svc.dispatch(data)
    assert replay["schema"] == S.STREAM_SCHEMA
    assert replay["stream_sha256"] == hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert replay["decisions"] == 3 and replay["prompts"] == 3 and replay["forced"] == 1
    assert [(x["obs_sha256"], x["memory_before_sha256"], x["memory_after_sha256"])
            for x in replay["trace"]] == expected_trace
    assert [x["subdecision"] for x in replay["trace"]] == [0, 1, 0]
    assert replay["memory_sha256"] == S.memory_sha256(svc.sessions[live].state)
    assert svc.sessions[replay["session"]].pending is None
    assert svc.sessions[replay["session"]].first is False
    # Historical actions are replayed, not sampled; the new stream's RNG is untouched.
    assert svc.sessions[replay["session"]].rng.bit_generator.state == np.random.default_rng(7).bit_generator.state


@pytest.mark.parametrize("change", [
    lambda d: d.update(session="real-opponent"),
    lambda d: d.update(hidden_truth={"hand": [5]}),
    lambda d: d.update(seat=True),
    lambda d: d.update(seed=False),
    lambda d: d.update(main=[True]),
    lambda d: d.update(extra=[0]),
    lambda d: d.update(frames={}),
    lambda d: d["frames"][0].update(hidden=42),
    lambda d: d["frames"][0].update(messages=[[256, ""]]),
    lambda d: d["frames"][0].update(messages=[[1, "AB"]]),
    lambda d: d["frames"][0].update(messages=[[1, "not-hex"]]),
    lambda d: d["frames"][0].update(messages=[[True, ""]]),
    lambda d: d["frames"][0].update(response=""),
    lambda d: d["frames"][0].update(response="01"),
    lambda d: d["frames"][0].update(response="010200"),
    lambda d: d["frames"][0].update(response="0902"),
    lambda d: d["frames"][0].update(messages=[[10, ""], [2, ""]]),
    lambda d: d["frames"][0].update(messages=[[12, ""], [2, ""]]),
    lambda d: d["frames"][0].update(messages=[]),
    lambda d: d["frames"][1].update(response="67"),
    lambda d: d.update(messages=[[10, ""]]),
    lambda d: d.update(messages=[[12, ""]]),
])
def test_bad_streams_fail_without_registering_or_changing_existing_sessions(change):
    svc, data = service(), request()
    existing = svc.open({"seat": 0, "main": [1], "extra": [], "seed": 3})["session"]
    old = svc.sessions[existing]
    before, rng = S.memory_sha256(old.state), copy.deepcopy(old.rng.bit_generator.state)
    change(data)
    with pytest.raises((ValueError, RuntimeError)):
        svc.dispatch(data)
    assert svc.sessions == {existing: old} and svc.counter == 1
    assert S.memory_sha256(old.state) == before and old.first and old.pending is None
    assert old.rng.bit_generator.state == rng and old.client.history == []


def test_bad_native_response_path_is_checked_against_produced_bytes():
    class BadBuilder(Builder):
        def response_path(self, response):
            return [0, 0]
    svc = service(BadBuilder)
    with pytest.raises(ValueError, match="recorded bytes"):
        svc.dispatch(request())
    assert svc.sessions == {}


def test_identical_synthetic_inputs_rebuild_identical_memory_not_real_opponent_sessions():
    svc = service()
    one = svc.dispatch(request())
    # An unrelated, differently progressed live session cannot influence a fresh replay.
    real = svc.open({"seat": 1, "main": [9], "extra": [], "seed": 99})["session"]
    svc.decide({"session": real, "messages": [[11, ""]]})
    two = svc.dispatch(request())
    assert one["session"] != two["session"]
    assert {k: v for k, v in one.items() if k != "session"} == {k: v for k, v in two.items() if k != "session"}


def test_stream_keeps_explicit_public_recipe_contract():
    from mirrorforce.netduel.agent_public_recipe import declare
    svc = service(config={"public_opponent_recipe": True}, identity={"opponent_recipe_mode": "mirror"})
    data = request()
    with pytest.raises(ValueError, match="mode"):
        svc.dispatch(data)
    data.update(opponent_recipe_mode="mirror", public_opponent_recipe=declare(data["main"], data["extra"]))
    assert svc.dispatch(data)["decisions"] == 3
    data["public_opponent_recipe"] = declare([99], [])
    with pytest.raises(ValueError, match="declaration"):
        svc.dispatch(data)
    assert len(svc.sessions) == 1


def test_empty_stream_has_initial_memory_and_memory_digest_preserves_structure():
    svc, data = service(), request()
    data.update(frames=[], messages=[])
    result = svc.dispatch(data)
    assert result["trace"] == [] and result["decisions"] == 0
    assert svc.sessions[result["session"]].first
    assert result["memory_sha256"] == S.memory_sha256(svc.backend.initial_state())
    assert S.memory_sha256(np.array(1)) != S.memory_sha256(np.array([1]))
    assert S.memory_sha256((np.array([1]),)) != S.memory_sha256([np.array([1])])
    assert S.memory_sha256(np.array([1], np.int8)) != S.memory_sha256(np.array([1], np.int64))
    assert S.memory_sha256(np.array([1])) != S.memory_sha256(np.array([2]))
