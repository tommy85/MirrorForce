"""Continuous wire games reset all viewer state; test identity and secrets stay scoped."""
from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from mirrorforce.netduel import constants as C, protocol as P
from mirrorforce.netduel.client import DuelError, NetDuelClient
from mirrorforce.netduel.league import LeagueConnection, read_credentials, round_robin_pairs


class Policy:
    def __init__(self, index):
        self.index = index
        self.active = False
        self.pending = []
        self.own_hand = []
        self.memory = np.zeros(3)
        self.session = None
        self.starts = []
        self.closed = 0

    def bind(self, client):
        assert client.result.our_player in (0, 1)
        assert not self.memory.any() and not self.pending and not self.own_hand
        self.active = True
        self.session = "synthetic-session-" + str(self.index)
        self.starts.append(client.result.our_player)

    def observe_game_message(self, msg, body):
        if self.active:
            self.pending.append((msg, body))
            self.memory += 1
            if msg == C.MSG_DRAW:
                self.own_hand.extend(struct.unpack_from("<I", body, 2))

    def on_duel_end(self, result):
        self.closed += 1
        self.active = False
        self.pending = []

    def report(self):
        return {"active": self.active, "pending_messages": len(self.pending), "failures": [],
                "private_view_fixture": list(self.own_hand)}


class Stream:
    def __init__(self):
        self.sent = []

    def send(self, op, payload=b""):
        self.sent.append((op, payload))


def setup(*, games=20, label="m-01", player="player1"):
    created, rows = [], []
    def factory(index):
        policy = Policy(index)
        client = NetDuelClient("unused", 6009, player, [1] * 40, [100], policy,
                               allow_match_mode=True, seed=10 + index)
        created.append(client)
        return client
    connection = LeagueConnection(factory, room=label, player=player, games=games, on_game=rows.append)
    connection.stream = connection.client.stream = Stream()
    connection.client.host_info = SimpleNamespace(mode=1)
    return connection, created, rows


def start(connection, seat):
    payload = bytes([C.MSG_START, seat, 4]) + struct.pack("<iiHHHH", 8000, 8000, 40, 15, 40, 15)
    connection.receive(P.STOC.GAME_MSG, payload)
    connection.receive(P.STOC.GAME_MSG, bytes([C.MSG_NEW_TURN, seat]))


def win(connection, winner, reason=1):
    if reason == 1 and connection.in_game:
        connection.receive(P.STOC.GAME_MSG, bytes([C.MSG_LPUPDATE, 1 - winner]) + struct.pack("<I", 0))
    connection.receive(P.STOC.GAME_MSG, bytes([C.MSG_WIN, winner, reason]))


def test_twenty_natural_games_are_ten_bo2_not_twenty_reconnections_and_swap_each_leg():
    connection, created, rows = setup()
    stream = connection.stream
    for index in range(20):
        start(connection, index % 2)
        win(connection, index % 2)
        connection.receive(P.STOC.DUEL_END, b"")
        if index != 19:
            connection.receive(P.STOC.CHANGE_SIDE, b"")
    assert len(rows) == len(created) == 20
    assert [row["bo2"] for row in rows] == [index // 2 + 1 for index in range(20)]
    assert [row["leg"] for row in rows] == [1, 2] * 10
    assert [row["result"]["our_player"] for row in rows] == [0, 1] * 10
    assert all(row["natural_terminal"] and row["training_eligible"] is False for row in rows)
    assert all(client.stream is stream and client.policy.closed == 1 for client in created)
    assert len({id(client.policy.memory) for client in created}) == 20
    assert len({id(client.board) for client in created}) == 20
    assert len({row["policy_session"] for row in rows}) == 20
    assert stream.sent == [(P.CTOS.UPDATE_DECK, P.update_deck([1] * 40 + [100], []))] * 19


def test_two_simultaneous_rooms_with_same_player_never_exchange_hands_memory_or_pending():
    left, left_clients, _ = setup(games=2)
    right, right_clients, _ = setup(games=2, label="m-02")
    start(left, 0); start(right, 1)
    left.receive(P.STOC.GAME_MSG, bytes([C.MSG_DRAW, 0, 1]) + struct.pack("<I", 23434538))
    assert left.client.policy.own_hand == [23434538]
    assert right.client.policy.own_hand == []
    before = right.client.policy.memory.copy()
    win(left, 0)
    np.testing.assert_array_equal(right.client.policy.memory, before)
    assert right.client.policy.active and right.client.policy.pending
    assert not left.client.policy.memory.any() and not left.client.policy.pending
    assert left.client.policy.own_hand == []
    start(left, 1)
    assert left_clients[0].policy.starts == [0] and left_clients[1].policy.starts == [1]
    assert right_clients[0].policy.starts == [1]


def test_duplicate_terminal_is_not_another_game_and_invalid_terminal_is_not_counted():
    connection, _, rows = setup(games=2)
    with pytest.raises(DuelError, match="without a new"):
        win(connection, 0)
    start(connection, 0); win(connection, 0)
    win(connection, 0)
    assert len(rows) == 1 and connection.duplicate_terminals == 1
    start(connection, 1)
    with pytest.raises(DuelError, match="special/administrative"):
        win(connection, 1, reason=4)
    assert len(rows) == 1


def test_side_before_terminal_and_duel_end_without_win_are_refused():
    connection, _, _ = setup(games=2)
    with pytest.raises(DuelError, match="side prompt"):
        connection.receive(P.STOC.CHANGE_SIDE, b"")
    start(connection, 0)
    with pytest.raises(DuelError, match="without the natural"):
        connection.receive(P.STOC.DUEL_END, b"")


def test_fresh_policy_rng_is_installed_before_next_rps_not_only_next_start():
    connection, created, _ = setup(games=2)
    start(connection, 0); win(connection, 0)
    assert connection.client is created[1]
    connection.receive(P.STOC.SELECT_HAND, b"")
    assert created[0].rps_throws == 0 and created[1].rps_throws == 1


def test_each_new_client_installs_policy_validator_even_without_reconnecting():
    clients = []
    class CheckedPolicy(Policy):
        def validate_client(self, client):
            client.native_handler_owner = self
        def bind(self, client):
            assert client.native_handler_owner is self
            return super().bind(client)
    def factory(index):
        client = NetDuelClient("unused", 6009, "player1", [1] * 40, [100], CheckedPolicy(index),
                               allow_match_mode=True)
        clients.append(client)
        return client
    connection = LeagueConnection(factory, room="m-01", player="player1", games=2, on_game=lambda row: None)
    connection.stream = connection.client.stream = Stream()
    connection.client.host_info = SimpleNamespace(mode=1)
    start(connection, 0); win(connection, 0); start(connection, 1); win(connection, 1)
    assert len(clients) == 2 and all(client.native_handler_owner is client.policy for client in clients)


def credential_file(tmp_path, *, count=3, test=True):
    tmp_path.chmod(0o700)
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"host": "127.0.0.1", "port": 6009, "test_only_multiple_identities": test,
        "players": [{"name": "player" + str(i), "room_passwords": ["TestRoomToken" + str(i)]} for i in range(count)]}))
    path.chmod(0o600)
    return path


def test_credentials_remain_private_and_three_test_identities_cannot_enter_competition(tmp_path):
    path = credential_file(tmp_path)
    value = read_credentials(path, mode="test")
    assert "TestRoomToken" not in repr(value) and "password" not in json.dumps(value.public())
    assert value.scrub("bad TestRoomToken1") == "bad <redacted>"
    with pytest.raises(ValueError, match="exactly one"):
        read_credentials(path, mode="competition")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        read_credentials(path, mode="test")


def test_single_real_identity_and_five_player_round_robin_are_explicit(tmp_path):
    value = read_credentials(credential_file(tmp_path, count=1, test=False), mode="competition")
    assert len(value.players) == 1
    assert len(round_robin_pairs(["a", "b", "c", "d", "e"])) == 10


def test_a_symlink_cannot_substitute_credential_bytes(tmp_path):
    original = credential_file(tmp_path)
    link = tmp_path / "linked.json"
    link.symlink_to(original)
    with pytest.raises(OSError):
        read_credentials(link, mode="test")
