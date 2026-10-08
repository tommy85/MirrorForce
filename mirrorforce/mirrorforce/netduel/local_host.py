"""An in-process duel host whose clients receive the bytes a network host would send.

The core runs in this process. Every core message is routed and masked by
:mod:`.host_view`, host refresh queries run on the same core through
:mod:`.wire_projection`, and each seat's packets reach an ordinary
:class:`~.client.NetDuelClient` through an in-memory stream. The client
answers exactly as it would over TCP, and its ``CTOS_RESPONSE`` bytes go back
into the core. Lobby handshakes, rock-paper-scissors and turn timers are outside
this host: the caller fixes the seats, the first player and both deck orders.

A client fed this way builds the same model inputs as a network client only if
the projection is byte-exact; ``common/stage_a_local_host_parity.py`` measures that
on recorded network games. Nothing here grants training eligibility.
"""

from __future__ import annotations

import struct
import time

from ..worldmodel.engine import DuelConfig, DuelDriver, DuelError
from . import constants as C
from . import protocol as P
from .wire_projection import InitialRefreshCore, project_message

__all__ = ["LocalHostDuel", "MemoryStream", "start_packet"]


class MemoryStream:
    """The client end of an in-process connection; it records what the client sends."""

    def __init__(self):
        self.sent: list[tuple[int, bytes]] = []

    def send(self, op, payload: bytes = b"") -> None:
        self.sent.append((int(op), bytes(payload)))

    def close(self) -> None:
        pass

    def take(self, op) -> list[bytes]:
        """Remove and return the payloads of one packet type, in sending order."""
        taken = [payload for kind, payload in self.sent if kind == int(op)]
        self.sent = [(kind, payload) for kind, payload in self.sent if kind != int(op)]
        return taken


def start_packet(player: int, host_info: P.HostInfo, deck_counts: tuple[int, int, int, int]) -> bytes:
    """``MSG_START`` as ``SingleDuel::TPResult`` writes it: 19 bytes, before the Extra Deck refreshes."""
    if player not in (0, 1) or len(deck_counts) != 4:
        raise ValueError("MSG_START needs a duel player and four zone counts")
    return (bytes([C.MSG_START, player, host_info.duel_rule])
            + struct.pack("<ii", host_info.start_lp, host_info.start_lp)
            + struct.pack("<HHHH", *deck_counts))


class LocalHostDuel(DuelDriver):
    """One duel; ``clients[p]`` plays duel player ``p``, and player 0 moves first.

    Each client must be a fresh :class:`~.client.NetDuelClient` that has not
    connected. The host replaces its stream, sets the room information, and
    delivers its packets in the order ``SingleDuel`` sends them: ``MSG_START``,
    the owner-only Extra Deck refresh, then every projected message. Packets
    for the player who is not asked are delivered before the asked player's
    prompt; each client's own order is what the network preserves.
    """

    def __init__(self, config: DuelConfig, core, clients, host_info: P.HostInfo):
        if len(clients) != 2 or clients[0] is clients[1] \
                or any(getattr(client, "stream", None) is not None for client in clients):
            raise ValueError("a local host needs two distinct clients that have not connected")
        if (host_info.start_lp, host_info.start_hand, host_info.draw_count) \
                != (config.start_lp, config.start_hand, config.draw_count) \
                or host_info.duel_rule != config.duel_options >> 16:
            raise ValueError("room information differs from the duel configuration")
        super().__init__(config, InitialRefreshCore(core))
        self.clients = tuple(clients)
        self.host_info = host_info
        self._outboxes: list[list[bytes]] = [[], []]
        for client in self.clients:
            validator = getattr(client.policy, "validate_client", None)
            if validator is not None:
                validator(client)
            client.stream = MemoryStream()
            client.host_info = host_info

    def build(self) -> "LocalHostDuel":
        super().build()
        orders = self.config.deck_orders()
        counts = (len(orders[0]), len(self.config.decks[0].extra), len(orders[1]), len(self.config.decks[1].extra))
        initial = self.core.initial_packets
        if initial is None:
            raise DuelError("the host's Extra Deck refreshes were not captured before start_duel")
        for player in (0, 1):
            self._outboxes[player] = [start_packet(player, self.host_info, counts), *initial[player],
                                      *self._outboxes[player]]
        return self

    def _observe(self, message) -> None:
        ended = self.winner is not None
        super()._observe(message)
        if ended:
            # SingleDuel::Analyze returns at MSG_WIN; later messages in the buffer reach nobody.
            return
        packets = project_message(self.core, self.pduel, message.msg, message.payload)
        for player in (0, 1):
            self._outboxes[player].extend(packets[player])

    def deliver(self, player: int) -> None:
        """Hand every pending packet of one seat to its client."""
        packets, self._outboxes[player] = self._outboxes[player], []
        client = self.clients[player]
        for packet in packets:
            client._handle(P.STOC.GAME_MSG, packet)

    def _answer(self, message, responder) -> None:
        msg, body = message.msg, message.payload
        self._answering_msg, self._answering_payload = msg, bytes(body)
        self._pending_response_context = "local client"
        player = body[1] if msg == C.MSG_SELECT_SUM else body[0]
        if player not in (0, 1) or not self._outboxes[player] or self._outboxes[player][-1][0] != msg:
            raise DuelError(f"message {msg} needs a response but the host sends it to no client")
        self.deliver(1 - player)
        self.deliver(player)
        responses = self.clients[player].stream.take(P.CTOS.RESPONSE)
        if len(responses) != 1 or self.clients[1 - player].stream.take(P.CTOS.RESPONSE):
            raise DuelError(f"message {msg} received {len(responses)} responses from its client")
        self.prompt_count += 1
        self._respond(responses[0])

    def play(self, *, max_steps: int = 400000, max_seconds: float | None = None) -> "LocalHostDuel":
        """Build, run to the core's end, deliver the tail and ``STOC_DUEL_END``, then close."""
        started = time.monotonic()
        try:
            self.build()
            self.run(None, max_steps=max_steps, max_seconds=max_seconds)
            if not self.finished:
                raise DuelError("local duel did not end within its step budget")
            for player in (0, 1):
                self.deliver(player)
            for client in self.clients:
                client._handle(P.STOC.DUEL_END, b"")
        finally:
            for client in self.clients:
                client.result.seconds = time.monotonic() - started
                client.result.lp = (client._lp[0], client._lp[1])
            self.close()
        for client in self.clients:
            client.policy.on_duel_end(client.result)
        return self
