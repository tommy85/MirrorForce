from types import SimpleNamespace

import pytest

from mirrorforce.netduel import protocol as P
from mirrorforce.netduel.client import DuelError, NetDuelClient


class RecordingStream:
    def __init__(self):
        self.sent = []

    def send(self, opcode, payload=b""):
        self.sent.append((opcode, payload))


def make_client(*, allow_match_mode=False):
    client = NetDuelClient("unused", 1, "league", [1] * 40, [100], policy=None,
                           side=[], allow_match_mode=allow_match_mode)
    client.stream = RecordingStream()
    return client


def test_match_mode_is_rejected_by_default(monkeypatch):
    monkeypatch.setattr(P.HostInfo, "unpack", lambda payload: SimpleNamespace(mode=1))
    client = make_client()
    with pytest.raises(DuelError, match="not a single duel"):
        client._handle(P.STOC.JOIN_GAME, b"")
    assert client.stream.sent == []


def test_opted_in_fixed_deck_match_mode_reuses_the_join_deck_on_side_prompt(monkeypatch):
    monkeypatch.setattr(P.HostInfo, "unpack", lambda payload: SimpleNamespace(mode=1))
    client = make_client(allow_match_mode=True)

    client._handle(P.STOC.JOIN_GAME, b"")
    assert client.stream.sent == [(P.CTOS.UPDATE_DECK, P.update_deck([1] * 40 + [100], []))]

    # The fixed deck was sent on JOIN_GAME. A side prompt does not mutate or
    # resubmit it; the following individual duel gets a new client instance.
    client._handle(P.STOC.CHANGE_SIDE, b"")
    client._handle(P.STOC.WAITING_SIDE, b"")
    assert len(client.stream.sent) == 1


def test_opt_in_still_rejects_tag_mode(monkeypatch):
    monkeypatch.setattr(P.HostInfo, "unpack", lambda payload: SimpleNamespace(mode=2))
    client = make_client(allow_match_mode=True)
    with pytest.raises(DuelError, match="not a single duel"):
        client._handle(P.STOC.JOIN_GAME, b"")
