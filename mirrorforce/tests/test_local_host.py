"""An in-process host plays a whole duel between two ordinary network clients.

The real core runs locally; each client only receives the packets the host
projection routes to its seat. The duel must finish with the clients' shadow
boards agreeing with every selection prompt, hide the opponent's private
draws, repeat byte for byte, and leave responses a plain replay reproduces.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from mirrorforce.effectinfo import DEFAULT_EFFECTINFO_LIB, get_effectinfo_core
from mirrorforce.netduel import constants as C
from mirrorforce.netduel import protocol as P
from mirrorforce.netduel.cards import load_ydk
from mirrorforce.netduel.client import NetDuelClient
from mirrorforce.netduel.local_host import LocalHostDuel, MemoryStream, start_packet
from mirrorforce.netduel.policy import RandomPolicy
from mirrorforce.puzzle.core import DEFAULT_DB, DEFAULT_SCRIPTS
from mirrorforce.worldmodel.engine import DeckList, DuelConfig, DuelDriver, DuelError

DECK = Path(__file__).resolve().parents[1] / "decks/stage-a/SkyStriker.ydk"
ENGINE_READY = Path(DEFAULT_EFFECTINFO_LIB).is_file() and Path(DEFAULT_DB).is_file() and Path(DEFAULT_SCRIPTS).is_dir()


def _config(seed):
    main, extra, _side = load_ydk(DECK)
    deck = DeckList("local", tuple(main), tuple(extra))
    return DuelConfig(decks=(deck, deck), seed=seed, full_phase_menu=True, auto_end_phase_discard=False), main, extra


def _room(config):
    return P.HostInfo(0, 0, 0, config.duel_options >> 16, 0, 0, config.start_lp, config.start_hand,
                      config.draw_count, 0)


def _play(core, seed):
    config, main, extra = _config(seed)
    clients = [NetDuelClient("local", 0, name, main, extra, RandomPolicy(seed + player), capture=[])
               for player, name in enumerate(("First", "Second"))]
    duel = LocalHostDuel(config, core, clients, _room(config))
    duel.play(max_seconds=120.0)
    return config, duel, clients


def test_memory_stream_takes_one_packet_type_in_order():
    stream = MemoryStream()
    stream.send(P.CTOS.RESPONSE, b"\x01")
    stream.send(P.CTOS.TIME_CONFIRM)
    stream.send(P.CTOS.RESPONSE, b"\x02")
    assert stream.take(P.CTOS.RESPONSE) == [b"\x01", b"\x02"]
    assert stream.sent == [(int(P.CTOS.TIME_CONFIRM), b"")]


def test_start_packet_matches_the_host_layout():
    room = P.HostInfo(0, 0, 0, 5, 0, 0, 8000, 5, 1, 0)
    packet = start_packet(1, room, (40, 15, 38, 14))
    assert len(packet) == 19 and packet[:3] == bytes([C.MSG_START, 1, 5])
    assert struct.unpack_from("<iiHHHH", packet, 3) == (8000, 8000, 40, 15, 38, 14)


def test_room_information_must_match_the_duel():
    config, main, extra = _config(1)
    clients = [NetDuelClient("local", 0, name, main, extra, RandomPolicy(0)) for name in ("a", "b")]
    room = _room(config)
    room.start_lp = 4000
    with pytest.raises(ValueError, match="room information"):
        LocalHostDuel(config, object(), clients, room)
    with pytest.raises(ValueError, match="two distinct clients"):
        LocalHostDuel(config, object(), [clients[0], clients[0]], _room(config))


@pytest.mark.skipif(not ENGINE_READY, reason="needs the effectinfo core, card database and scripts")
def test_two_clients_finish_a_duel_through_the_local_host_repeatably():
    core = get_effectinfo_core()  # the process-global core, script overrides included
    config, duel, clients = _play(core, 20260917)
    assert duel.finished and duel.winner in (0, 1)
    for player, client in enumerate(clients):
        assert client.result.our_player == player and client._duel_over and not client.result.error
        assert client.result.winner == duel.winner and client.result.won == (duel.winner == player)
        assert client.board.checks > 0 and client.board.mismatches == 0 and client.board_problems == []
        assert client.capture[0][0] == C.MSG_START and client.capture[1][0] == C.MSG_UPDATE_DATA
        # The opponent's private draws reach this seat with zeroed codes.
        for msg, body in client.capture:
            if msg == C.MSG_DRAW and body[0] != player:
                codes = struct.unpack_from("<%dI" % body[1], body, 2)
                assert all(code == 0 or code & 0x80000000 for code in codes)
    assert sum(client.result.decisions + client.result.forced_actions + client.result.auto_responses
               for client in clients) >= len(duel.responses)

    _, again, repeated = _play(core, 20260917)
    assert again.responses == duel.responses and again.winner == duel.winner
    assert [client.capture for client in repeated] == [client.capture for client in clients]

    core.reset_session()
    with DuelDriver(config, core) as replay:
        replay._replay = [bytes(response) for response in duel.responses]
        replay.run(lambda prompt, driver: (_ for _ in ()).throw(DuelError("replay asked a new choice")),
                   max_steps=400000, max_seconds=120.0)
        assert replay.finished and replay.winner == duel.winner and replay.replay_remaining() == 0
