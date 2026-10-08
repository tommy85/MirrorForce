"""Standard-library client for one MirrorForce practice-room duel."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import random
import socket
import time

from mirrorforce.netduel import protocol as P
from mirrorforce.netduel.cards import CardPool, load_ydk
from mirrorforce.netduel.client import DuelError, NetDuelClient


class AgentPolicy:
    """Adapt a participant's choose(state) callback to the maintained client."""

    name = "participant"

    def __init__(self, choose, seed=0):
        self.callback = choose
        self.rng = random.Random(seed)
        self.capture = []

    def choose(self, state):
        choice = self.callback(state)
        if type(choice) is not int or not 0 <= choice < state.n:
            raise DuelError("choose(state) must return an integer in [0, state.n)")
        return choice

    def on_duel_end(self, result):
        pass


class PracticeClient(NetDuelClient):
    """Expose full phase/sort/discard choices and an explicit smoke concession."""

    def __init__(self, *args, smoke_decisions=0, no_idle_timeout=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctx.full_phase_menu = True
        self.ctx.full_card_sort_menu = True
        self.ctx.auto_end_phase_discard = False
        self.smoke_decisions = smoke_decisions
        self.no_idle_timeout = no_idle_timeout

    def connect(self):
        if self.no_idle_timeout and (self.timeout is None or not math.isfinite(self.timeout) or self.timeout <= 0):
            raise DuelError("an unlimited idle read still needs a finite positive connection timeout")
        super().connect()
        if self.no_idle_timeout:
            self.timeout = None

    def _handle(self, op, payload):
        if self.no_idle_timeout and op == P.STOC.JOIN_GAME and P.HostInfo.unpack(payload).time_limit != 0:
            raise DuelError("--no-idle-timeout requires a room with time_limit=0")
        if self.no_idle_timeout and op == P.STOC.TIME_LIMIT:
            raise DuelError("a clock-disabled room unexpectedly sent a turn timer")
        return super()._handle(op, payload)

    def _send_response(self, data):
        super()._send_response(data)
        if self.smoke_decisions and self.result.decisions >= self.smoke_decisions:
            self.surrender()
            self.smoke_decisions = 0


def probe(host, port, name, password="", timeout=15.):
    """Join the lobby without sending a deck or READY, then immediately leave."""
    stream = P.PacketStream(socket.create_connection((host, port), timeout=timeout))
    try:
        stream.send(P.CTOS.PLAYER_INFO, P.player_info(name))
        stream.send(P.CTOS.JOIN_GAME, P.join_game(password=password))
        info = seat = None
        deadline = time.monotonic() + timeout
        while info is None or seat is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("room handshake timed out")
            op, payload = stream.recv(remaining)
            if op == P.STOC.ERROR_MSG:
                raise DuelError("room rejected join: " + payload.hex())
            if op == P.STOC.JOIN_GAME:
                info = P.HostInfo.unpack(payload).as_dict()
            elif op == P.STOC.TYPE_CHANGE:
                seat = payload[0] & 0xF
        if seat not in (0, 1):
            raise DuelError("both player seats are occupied; retry after this duel")
        return {"schema": "mirrorforce_room_sdk_probe/v1", "host": host, "port": port,
                "protocol_version": P.PRO_VERSION, "seat": seat, "room": info,
                "ready_sent": False}
    finally:
        try:
            stream.send(P.CTOS.LEAVE_GAME)
        except OSError:
            pass
        stream.close()


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--name", default="Participant")
    parser.add_argument("--password", default="")
    parser.add_argument("--agent", default="example_agent", help="module exposing choose(state) -> int")
    parser.add_argument("--deck", type=Path, default=root / "assets/SkyStriker.ydk")
    parser.add_argument("--cards", type=Path, default=root / "assets/cards.cdb")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--timeout", type=float, default=1800.)
    parser.add_argument("--no-idle-timeout", action="store_true",
                        help="clock-disabled rooms only: --timeout bounds connect, then idle reads block without a deadline")
    parser.add_argument("--probe", action="store_true", help="check lobby only; does not start a game")
    parser.add_argument("--smoke-decisions", type=int, default=0,
                        help="explicitly concede after this many decisions; 0 plays the full duel")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not math.isfinite(args.timeout) or args.timeout <= 0 or args.smoke_decisions < 0:
        parser.error("invalid port, timeout, or smoke decision count")
    if args.probe:
        print(json.dumps(probe(args.host, args.port, args.name, args.password, min(args.timeout, 15.)),
                         ensure_ascii=False), flush=True)
        return 0
    main_deck, extra, side = load_ydk(args.deck)
    if len(main_deck) != 40 or len(extra) != 15 or side:
        parser.error("these practice rooms use the supplied 40 main / 15 extra / 0 side deck")
    pool = CardPool(args.cards)
    if any(code not in pool.cards for code in main_deck + extra):
        parser.error("deck contains a card absent from cards.cdb")
    callback = getattr(importlib.import_module(args.agent), "choose")
    if not callable(callback):
        parser.error("agent module must expose callable choose(state)")
    policy = AgentPolicy(callback, args.seed)
    client = PracticeClient(args.host, args.port, args.name, main_deck, extra, policy,
                            seed=args.seed, timeout=args.timeout, password=args.password, card_pool=pool,
                            max_options=len(pool.cards), smoke_decisions=args.smoke_decisions,
                            no_idle_timeout=args.no_idle_timeout,
                            log=lambda text: print("ROOM " + str(text), flush=True))
    print("SDK-READY " + json.dumps({"host": args.host, "port": args.port, "agent": args.agent,
                                      "training_eligible": False}), flush=True)
    result = client.run()
    print("SDK-RESULT " + json.dumps(result.as_dict(), ensure_ascii=False), flush=True)
    return 1 if result.error or result.our_player not in (0, 1) else 0


if __name__ == "__main__":
    raise SystemExit(main())
