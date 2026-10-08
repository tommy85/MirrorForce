"""Each seat's received message stream for a recorded game, through the in-process host.

The audience acceptance (design §6, item 9a) compares, per observer, the facts the env derives from the core
stream with the facts the same tracker derives from what that observer's client actually receives. This module
produces the second input: a game given as an explicit deal (mfenv's ABI: seed words, deck orders, extra decks,
options) and its raw recorded responses is replayed through ``netduel.local_host.LocalHostDuel``, which routes and
masks every core message as a network host does; each seat is a recording stand-in that answers its own prompts with
the next recorded response. Every recorded response must be asked for, in order, and the game must end.

Not part of the tree (it imports the Python host and core); tests and audit tools use it.
"""
from __future__ import annotations

from types import SimpleNamespace

from mirrorforce.netduel import protocol as P
from mirrorforce.netduel.local_host import LocalHostDuel
from mirrorforce.worldmodel.engine import DeckList, DuelConfig, DuelError

#: Prompt messages a seat answers (ygopro-core common.h: MSG_SELECT_*, sorts, RPS, announces).
PROMPTS = frozenset({10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 132, 140, 141, 142, 143})


class _Seat:
    """A host-side stand-in for one client: records every game packet and answers its prompts from the shared queue."""

    def __init__(self, responses):
        self.stream = None
        self.host_info = None
        self.packets = []
        self.responses = responses
        self.policy = SimpleNamespace(on_duel_end=lambda result: None)
        self.result = SimpleNamespace(seconds=0.0, lp=None)
        self._lp = [0, 0]

    def _handle(self, op, packet):
        if op != P.STOC.GAME_MSG:
            return
        self.packets.append(bytes(packet))
        if packet and packet[0] in PROMPTS:
            if not self.responses:
                raise DuelError("the host asked for more responses than the game recorded")
            self.stream.send(P.CTOS.RESPONSE, self.responses.pop(0))


def host_config(deal) -> DuelConfig:
    """The host configuration that rebuilds an mfenv deal exactly (forced deck orders and core seeds)."""
    decks = tuple(DeckList(f"p{seat}", tuple(sorted(deal["deck_orders"][seat])), tuple(deal["extra"][seat]))
                  for seat in (0, 1))
    return DuelConfig(decks=decks, seed=0, start_lp=deal["start_lp"], start_hand=deal["start_hand"],
                      draw_count=deal["draw_count"], duel_options=deal["duel_options"],
                      max_options=deal.get("max_options", 128), max_sub_rounds=deal.get("max_sub_rounds", 512),
                      full_phase_menu=deal.get("full_phase_menu", True),
                      auto_end_phase_discard=deal.get("auto_end_phase_discard", False),
                      full_card_sort_menu=deal.get("full_card_sort_menu", True),
                      commit_command_selections=deal.get("commit_command_selections", True),
                      forced_deck_orders=tuple(tuple(order) for order in deal["deck_orders"]),
                      forced_core_seeds=tuple(deal["seed_words"]))


def client_streams(core, deal, responses, *, max_seconds=600.0):
    """``{"winner", "streams": [seat 0's [(msg, payload)], seat 1's ...]}`` of one recorded game."""
    queue = [bytes(r) for r in responses]
    seats = [_Seat(queue), _Seat(queue)]
    room = P.HostInfo(0, 0, 0, deal["duel_options"] >> 16, 0, 0, deal["start_lp"], deal["start_hand"],
                      deal["draw_count"], 0)
    host = LocalHostDuel(host_config(deal), core, seats, room)
    host.play(max_seconds=max_seconds)
    if queue:
        raise DuelError(f"{len(queue)} recorded responses were never asked for")
    return {"winner": host.winner, "streams": [[(p[0], p[1:]) for p in seat.packets] for seat in seats]}
