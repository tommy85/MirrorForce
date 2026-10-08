"""Current-root RPC lifecycle with a fake native builder, not a GPU/native parity claim."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.netduel.current_root_view import SCHEMA, ZONES
from tools import mf_runtime_policy_service as S


class NativeClient:
    def __init__(self, *args, **kwargs):
        self.seeded = None
        self.messages = []

    def initialize_current_root(self, view):
        assert self.seeded is None
        self.seeded = copy.deepcopy(view)
        return {10: 1, 20: 1, 30: 1, 80: 1}

    def prompt(self):
        return None

    def clone(self):
        return copy.deepcopy(self)


class LiveClient:
    def __init__(self, native, seat, main, extra, config, **kwargs):
        self.seat, self.main, self.extra = seat, main, extra
        self.steps = []

    def clone(self):
        return copy.deepcopy(self)

    def step(self, index):
        self.steps.append(index)
        return bytes([index])

    def forced_count(self):
        return 0


class Backend:
    def __init__(self):
        self.initial_calls = 0

    def initial_state(self):
        self.initial_calls += 1
        return np.zeros(3, np.float32)

    def act_batch(self, *args):
        pytest.fail("opening/cloning a current-root view must not perform a forward")


def card(code, player, loc, seq):
    return [code, player, loc, seq, C.POS_FACEDOWN_DEFENSE, 0, 0, 0, 0, 0, [], [], player, 0, 0, 0, 0, 0, False]


def view():
    cards = [[[] for _ in ZONES] for _ in range(2)]
    cards[1][0] = [card(10, 1, C.LOCATION_DECK, 0), card(20, 1, C.LOCATION_DECK, 1)]
    cards[1][1] = [card(30, 1, C.LOCATION_HAND, 0)]
    cards[1][6] = [card(80, 1, C.LOCATION_EXTRA, 0)]
    cards[0][0] = [card(0, 0, C.LOCATION_DECK, 0)]
    cards[0][1] = [card(0, 0, C.LOCATION_HAND, i) for i in range(2)]
    cards[0][6] = [card(0, 0, C.LOCATION_EXTRA, 0)]
    return dict(schema=SCHEMA, root_id="root-7-15", root_hash="b" * 64, hypothesis_hash="c" * 64,
                viewer=1, turn=7, turn_player=0, phase=C.PHASE_MAIN1, lp=[4000, 7200], cards=cards,
                main=[10, 20, 30], extra=[80], opponent_main=[10, 20, 30], opponent_extra=[80], material_owners=[])


@pytest.fixture
def live():
    identity = {"weights": "iterate", "opponent_recipe_mode": "mirror", "public_opponent_recipe": None,
                "current_root": dict(S.CURRENT_ROOT_CAPABILITY)}
    service = S.Service(SimpleNamespace(ClientDuel=NativeClient), Backend(), {"public_opponent_recipe": True},
                        "greedy", 1., identity, client_factory=LiveClient)
    owner = S.ConnectionOwner()
    parent = service.dispatch({"op": "open", "seat": 0, "main": [10, 20, 30], "extra": [80], "seed": 19,
                               "opponent_recipe_mode": "mirror", "public_opponent_recipe": S.PR.declare([10, 20, 30], [80])},
                              owner=owner)["session"]
    session = service.sessions[parent]
    session.state[:] = [9, 8, 7]
    session.first = False
    session.pending = {"obs_sha256": "a" * 64, "rows": 2, "logits": np.array([1., 2.])}
    request = {"op": "open_root_view", "session": parent, "expected_obs_sha256": "a" * 64,
               "seed": 41, "view": view(), "memory_law": S.CURRENT_ROOT_MEMORY_LAW}
    return service, owner, parent, request


def test_opponent_current_view_is_cold_and_own_pending_memory_rng_are_untouched(live):
    service, owner, parent, request = live
    own = service.sessions[parent]
    pending, memory, rng = own.pending, own.state.copy(), json.dumps(own.rng.bit_generator.state, sort_keys=True)
    result = service.dispatch(request, owner=owner)
    opponent = service.sessions[result["session"]]
    assert opponent.owner is owner and opponent.client.seat == 1 and opponent.first
    assert opponent.pending is None and opponent.client.prompt() is None
    assert not opponent.client.duel.messages and opponent.client.duel.seeded["turn"] == 7
    assert opponent.decisions == opponent.prompts == opponent.forced == 0
    np.testing.assert_array_equal(opponent.state, np.zeros(3, np.float32))
    assert result["memory_sha256"] == S.memory_sha256(opponent.state)
    assert result["root_hash"] == "b" * 64 and result["hypothesis_hash"] == "c" * 64
    assert result["obs_sha256"] == "a" * 64 and result["first"] is True
    assert own.pending is pending and own.first is False and own.client.steps == []
    np.testing.assert_array_equal(own.state, memory)
    assert json.dumps(own.rng.bit_generator.state, sort_keys=True) == rng


@pytest.mark.parametrize("change", ["capability", "law", "extra_field", "stale", "no_pending", "bool_seed", "negative_seed",
                                    "wrong_seat", "wrong_recipe", "private_stats", "deck_order"])
def test_invalid_root_requests_register_nothing_or_change_no_live_session(live, change):
    service, owner, parent, request = live
    own, before = service.sessions[parent], copy.deepcopy(request)
    if change == "capability":
        service.identity.pop("current_root")
    elif change == "law":
        request["memory_law"] = "copy_our_private_memory"
    elif change == "extra_field":
        request["server_truth"] = {}
    elif change == "stale":
        request["expected_obs_sha256"] = "d" * 64
    elif change == "no_pending":
        own.pending = None
    elif change == "bool_seed":
        request["seed"] = True
    elif change == "negative_seed":
        request["seed"] = -1
    elif change == "wrong_seat":
        own.client.seat = 1
    elif change == "wrong_recipe":
        request["view"]["opponent_main"] = [10]
    elif change == "private_stats":
        request["view"]["cards"][0][1][0][7] = 2800
    elif change == "deck_order":
        request["view"]["cards"][1][0][0][0] = 20
        request["view"]["cards"][1][0][1][0] = 10
    with pytest.raises(ValueError):
        service.dispatch(request, owner=owner)
    assert list(service.sessions) == [parent] and own.client.steps == []
    assert service.backend.initial_calls == 1
    assert before["view"]["root_hash"] == "b" * 64


def test_foreign_connection_cannot_seed_clone_or_close_current_root(live):
    service, owner, parent, request = live
    stranger = S.ConnectionOwner()
    with pytest.raises(ValueError, match="connection"):
        service.dispatch(request, owner=stranger)
    name = service.dispatch(request, owner=owner)["session"]
    for op in ({"op": "clone", "session": name, "seed": 1}, {"op": "close", "session": name, "messages": []}):
        with pytest.raises(ValueError, match="connection"):
            service.dispatch(op, owner=stranger)
    assert set(service.sessions) == {parent, name}


@pytest.mark.parametrize("end", ["commit", "same_sha_new_pending", "close", "replace"])
def test_parent_pending_end_invalidates_all_descendants_but_permits_cleanup(live, end):
    service, owner, parent, request = live
    child = service.dispatch(request, owner=owner)["session"]
    clone = service.dispatch({"op": "clone", "session": child, "seed": 5}, owner=owner)["session"]
    original = service.sessions[parent]
    if end == "commit":
        service.dispatch({"op": "commit", "session": parent, "index": 1}, owner=owner)
    elif end == "same_sha_new_pending":
        original.pending = dict(original.pending)
    elif end == "close":
        service.dispatch({"op": "close", "session": parent, "messages": []}, owner=owner)
    else:
        replacement = S.Session(original.client, service.backend, 7)
        replacement.owner, replacement.pending = owner, original.pending
        service.sessions[parent] = replacement
    for name in (child, clone):
        with pytest.raises(ValueError, match="current-root"):
            service.dispatch({"op": "clone", "session": name, "seed": 9}, owner=owner)
        service.dispatch({"op": "close", "session": name, "messages": []}, owner=owner)
    assert child not in service.sessions and clone not in service.sessions


def test_nested_root_and_closed_connection_cannot_make_new_sessions(live):
    service, owner, parent, request = live
    name = service.dispatch(request, owner=owner)["session"]
    service.sessions[name].pending = service.sessions[parent].pending
    with pytest.raises(ValueError, match="live real-session"):
        service.dispatch({**request, "session": name}, owner=owner)
    assert service.close_owned(owner) == 2 and not service.sessions
    with pytest.raises(ValueError):
        service.dispatch(request, owner=owner)


def test_same_pending_common_public_seed_is_read_without_forward_or_state_updates(live):
    from test_current_root_export import reply as fixture_reply
    service, owner, parent, _ = live
    session = service.sessions[parent]
    dto = fixture_reply()["seed"]
    calls = []
    def get():
        calls.append(1)
        return copy.deepcopy(dto)
    session.client.current_public_root_seed = get
    before, pending = session.state.copy(), session.pending
    request = {"op": "current_public_root_seed", "session": parent, "expected_obs_sha256": "a"*64}
    result = service.dispatch(request, owner=owner)
    assert result["seed"] == dto and result["session"] == parent and calls == [1]
    assert session.pending is pending and session.client.steps == []
    np.testing.assert_array_equal(session.state, before)
    for altered in ({**request, "expected_obs_sha256": "b"*64}, {**request, "viewer": 1}):
        with pytest.raises(ValueError):
            service.dispatch(altered, owner=owner)
    with pytest.raises(ValueError, match="connection"):
        service.dispatch(request, owner=S.ConnectionOwner())
    assert calls == [1]


def test_expired_opponent_in_batch_fails_before_any_forward_or_feed(live):
    service, owner, parent, request = live
    first = service.dispatch(request, owner=owner)["session"]
    second = service.dispatch({**request, "seed": 42}, owner=owner)["session"]
    service.sessions[parent].pending = None
    with pytest.raises(ValueError, match="pending decision has ended"):
        service.dispatch({"op": "rollout_step", "items": [
            {"session": first, "messages": []}, {"session": second, "messages": []}]}, owner=owner)
    assert not service.sessions[first].client.duel.messages
    assert not service.sessions[second].client.duel.messages


def test_seed_and_parent_commit_are_serialized_without_registry_lock_deadlock(live, monkeypatch):
    import threading
    service, owner, parent, request = live
    entered, release, committed = threading.Event(), threading.Event(), threading.Event()
    errors, results = [], []
    original = NativeClient.initialize_current_root

    def seed(client, payload):
        entered.set()
        assert release.wait(2)
        return original(client, payload)

    def open_root():
        try:
            results.append(service.dispatch(request, owner=owner))
        except BaseException as exc:
            errors.append(exc)

    def commit():
        try:
            service.dispatch({"op": "commit", "session": parent, "index": 0}, owner=owner)
            committed.set()
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(NativeClient, "initialize_current_root", seed)
    creating = threading.Thread(target=open_root)
    advancing = threading.Thread(target=commit)
    creating.start()
    try:
        assert entered.wait(2)
        advancing.start()
        assert not committed.wait(.05)
    finally:
        release.set()
        creating.join(2)
        if advancing.ident is not None:
            advancing.join(2)
    assert not creating.is_alive() and not advancing.is_alive() and not errors and committed.is_set()
    assert len(results) == 1
    with pytest.raises(ValueError, match="pending decision has ended"):
        service.dispatch({"op": "clone", "session": results[0]["session"], "seed": 5}, owner=owner)


@pytest.mark.parametrize("args", [[], ["--weights", "ema"], ["--opponent-mode", "known"],
                                 ["--behavior-finite-audit"]])
def test_cli_rejects_unregistered_current_root_modes_before_model_load(monkeypatch, args):
    monkeypatch.setattr(S, "load_native", lambda *_: pytest.fail("invalid mode reached native/model load"))
    defaults = ["--backend", "checkpoint", "--weights", "iterate", "--opponent-mode", "mirror"] if args else []
    with pytest.raises(ValueError, match="current-root search requires"):
        S.main(["--current-root-search", "--selection", "greedy", "--cards-db", "/unused",
                "--code-list", "/unused", "--script-root", "/unused", "--announce-tables", "/unused",
                "--socket", "/unused", "--out", "/unused", *defaults, *args])
