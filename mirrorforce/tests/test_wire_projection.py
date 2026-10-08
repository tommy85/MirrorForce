"""Live-query projection: cache sharing, ordering and observable state changes."""

from __future__ import annotations

import ctypes
import struct

import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.netduel.board import ShadowBoard, parse_query_segments
from mirrorforce.netduel.host_view import Refresh
from mirrorforce.netduel.wire_projection import (
    InitialRefreshCore, project_initial_refreshes, project_message, project_refresh,
)


def _segment(player=0, location=C.LOCATION_MZONE, sequence=0,
             position=C.POS_FACEUP_ATTACK, *, code=89631139, attack=3000,
             status=0):
    flags = C.QUERY_CODE | C.QUERY_POSITION
    body = struct.pack("<I", code) + bytes([player, location, sequence, position])
    if attack is not None:
        flags |= C.QUERY_ATTACK
        body += struct.pack("<i", attack)
    if status is not None:
        flags |= C.QUERY_STATUS
        body += struct.pack("<I", status)
    return struct.pack("<II", len(body) + 8, flags) + body


class _QueryCore:
    """Native ABI-shaped test double; second cached read omits state fields."""

    def __init__(self, *, hidden=False):
        self._lib = self
        self.calls = []
        self.cached = {}
        self.hidden = hidden
        self.attack = 3000
        self.status = 0

    def _read(self, player, location, sequence, buf, use_cache):
        key = (player, location, sequence)
        state = self.attack, self.status
        suppress = use_cache and self.cached.get(key) == state
        self.cached[key] = state
        query = _segment(
            player, location, sequence,
            C.POS_FACEDOWN_DEFENSE if self.hidden else C.POS_FACEUP_ATTACK,
            attack=None if suppress else self.attack,
            status=None if suppress else self.status)
        ctypes.memmove(buf, query, len(query))
        return len(query)

    def query_field_card(self, pduel, player, location, flag, buf, cache):
        self.calls.append(("field", pduel, player, location, flag, cache))
        return self._read(player, location, 0, buf, cache)

    def query_card(self, pduel, player, location, sequence, flag, buf, cache):
        self.calls.append(("single", pduel, player, location, sequence, flag, cache))
        return self._read(player, location, sequence, buf, cache)


def _fields(raw):
    header = 3 if raw[0] == C.MSG_UPDATE_DATA else 4
    return parse_query_segments(raw[header:])[0]


def test_each_refresh_is_queried_once_and_shared_before_the_idle_prompt():
    core = _QueryCore()
    packets = project_message(core, 17, C.MSG_SELECT_IDLECMD, b"\x00")
    assert len(core.calls) == 6
    assert packets[0][:-1] == packets[1][:-1]
    assert packets[0][-1] == bytes([C.MSG_SELECT_IDLECMD, 0])
    assert packets[1][-1] == bytes([C.MSG_WAITING])
    for raw in packets[0][:-1]:
        assert _fields(raw)["attack"] == 3000
        assert _fields(raw)["status"] == 0
    again = project_message(core, 17, C.MSG_SELECT_IDLECMD, b"\x00")
    assert len(core.calls) == 12
    assert all("attack" not in _fields(raw) for raw in again[0][:-1])


def test_before_and_after_refreshes_preserve_the_host_wire_order():
    core = _QueryCore()
    turn = project_message(core, 17, C.MSG_NEW_TURN, b"\x01")[0]
    phase = project_message(core, 17, C.MSG_NEW_PHASE, b"\x04\x00")[0]
    assert [raw[0] for raw in turn] == [C.MSG_UPDATE_DATA] * 6 + [C.MSG_NEW_TURN]
    assert [raw[0] for raw in phase] == [C.MSG_NEW_PHASE] + [C.MSG_UPDATE_DATA] * 6
    flip = (struct.pack("<I", 89631139)
            + bytes([1, C.LOCATION_MZONE, 3, C.POS_FACEUP_ATTACK]))
    packets = project_message(core, 17, C.MSG_FLIPSUMMONING, flip)
    assert [raw[0] for raw in packets[0]] == [C.MSG_UPDATE_CARD, C.MSG_FLIPSUMMONING]
    assert core.calls[-1] == ("single", 17, 1, C.LOCATION_MZONE, 3, 0xf81fff, 0)


def test_initial_extra_refreshes_are_owner_only_and_use_the_host_flags():
    core = _QueryCore(hidden=True)
    packets = project_initial_refreshes(core, 17)
    assert core.calls == [
        ("field", 17, 0, C.LOCATION_EXTRA, 0xe81fff, 1),
        ("field", 17, 1, C.LOCATION_EXTRA, 0xe81fff, 1)]
    for player in (0, 1):
        assert len(packets[player]) == 1
        assert packets[player][0][:3] == bytes([C.MSG_UPDATE_DATA, player, C.LOCATION_EXTRA])
        assert _fields(packets[player][0])["code"] == 89631139


def test_initial_queries_happen_before_start_duel_sets_the_rules():
    core = _QueryCore()
    core.start_duel = lambda pduel, options: core.calls.append(("start", pduel, options))
    proxy = InitialRefreshCore(core)
    assert proxy._lib is core._lib
    proxy.start_duel(17, 5 << 16)
    assert [call[0] for call in core.calls] == ["field", "field", "start"]
    assert len(proxy.initial_packets[0]) == len(proxy.initial_packets[1]) == 1


@pytest.mark.parametrize("kind,location", [
    ("mzone", C.LOCATION_MZONE), ("szone", C.LOCATION_SZONE),
    ("hand", C.LOCATION_HAND),
])
def test_hidden_refreshes_erase_status_and_attack_as_well_as_identity(kind, location):
    core = _QueryCore(hidden=True)
    packets = project_refresh(core, 17, Refresh(kind, 1, 0x681fff))
    assert _fields(packets[1])["code"] == 89631139
    assert _fields(packets[0]) == {"_flags": 0, "_hidden": True}
    assert packets[0][:3] == bytes([C.MSG_UPDATE_DATA, 1, location])


def test_single_hidden_refresh_is_only_the_host_stub_and_uses_cache_zero():
    core = _QueryCore(hidden=True)
    packets = project_refresh(core, 17, Refresh("single", 1, 0xf81fff, 1,
                                               C.LOCATION_SZONE, 2))
    assert core.calls[-1][-1] == 0  # RefreshSingle has no use_cache parameter.
    assert _fields(packets[0])["code"] == 0
    assert "attack" not in _fields(packets[0])
    assert "status" not in _fields(packets[0])
    assert len(packets[0]) == 20


def test_public_state_changes_reach_the_existing_shadow_board():
    core = _QueryCore()
    board = ShadowBoard()
    board.start(0, [], [])
    refresh = Refresh("mzone", 1, 0x881fff)
    for attack, status in ((3000, 0), (1500, 1), (3000, 0)):
        core.attack, core.status = attack, status
        packet = project_refresh(core, 17, refresh)[0]
        board.apply(packet[0], packet[1:])
        card = board.zone(1, C.LOCATION_MZONE)[0]
        assert (card.attack, card.status) == (attack, status)


@pytest.mark.parametrize("query", [b"\x01", struct.pack("<I", 0),
                                     struct.pack("<I", 999), struct.pack("<III", 12, 3, 1)])
def test_malformed_native_query_is_not_published(query):
    core = _QueryCore()

    def read(*args):
        ctypes.memmove(args[-2], query, len(query))
        return len(query)

    core.query_field_card = read
    with pytest.raises(ValueError):
        project_refresh(core, 17, Refresh("mzone", 0, 0x881fff))
