"""One persistent league connection, with a fresh viewer/client/policy for every duel.

No game engine, truth exporter or opponent process is available here. The only
policy inputs are this connection's received public game messages. Read-only
weights may be shared by the policy service; client boards, RPC sessions,
Memory, hands, pending messages and RNGs may never be shared between viewers.
The ordinary single-duel client and human-duel entry are deliberately unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import json
import math
import os
from pathlib import Path
import stat
import time

from . import constants as C, protocol as P
from .client import DuelError, NetDuelClient

SCHEMA = "mirrorforce_league_viewer_game/v1"
NATURAL_REASONS = {1: "life_points", 2: "deck_out"}


@dataclass(frozen=True)
class Enrollment:
    player: str
    passwords: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True)
class Credentials:
    host: str
    port: int
    test_only_multiple_identities: bool
    players: tuple[Enrollment, ...]

    def public(self):
        return {"host": self.host, "port": self.port,
                "test_only_multiple_identities": self.test_only_multiple_identities,
                "players": [{"name": p.player, "room_count": len(p.passwords)} for p in self.players]}

    def scrub(self, value):
        """Only for diagnostic text; credentials never enter numeric or replay records."""
        if isinstance(value, str):
            for player in self.players:
                for token in player.passwords:
                    value = value.replace(token, "<redacted>")
            return value
        if isinstance(value, dict):
            return {self.scrub(k): self.scrub(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [self.scrub(v) for v in value]
        return value


def read_credentials(path, *, mode):
    """Explicit test enrollment; a real competition represents exactly ONE identity.

    Open the approved local 0600 file without following symlinks. Passwords are
    used only for JOIN_GAME and are never included in argv, logs or reports.
    """
    if mode not in ("test", "competition"):
        raise ValueError("explicit test or single-identity competition mode is required")
    path = Path(path)
    directory = path.parent.lstat()
    if not stat.S_ISDIR(directory.st_mode) or stat.S_IMODE(directory.st_mode) != 0o700 \
            or directory.st_uid != os.geteuid():
        raise ValueError("credential parent must be an owned 0700 real directory")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 \
                or info.st_uid != os.geteuid() or info.st_size > 16384:
            raise ValueError("credentials must be an owned 0600 regular small file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(16385)
        try:
            value = json.loads(raw)
        except (UnicodeError, ValueError) as exc:
            raise ValueError("credential file is malformed") from exc
    finally:
        os.close(fd)
    if type(value) is not dict or set(value) != {"host", "port", "test_only_multiple_identities", "players"} \
            or type(value["test_only_multiple_identities"]) is not bool \
            or not isinstance(value["host"], str) or not value["host"] \
            or type(value["port"]) is not int or not 1 <= value["port"] <= 65535 \
            or not isinstance(value["players"], list) or not 1 <= len(value["players"]) <= 5:
        raise ValueError("closed league credential enrollment fields are required")
    if mode == "competition" and (len(value["players"]) != 1 or value["test_only_multiple_identities"]):
        raise ValueError("a real competition represents exactly one identity, never three test players")
    if mode == "test" and value["test_only_multiple_identities"] is not True:
        raise ValueError("multiple-identity testing must be explicitly authorized")
    players, names, tokens = [], set(), set()
    for player in value["players"]:
        if type(player) is not dict or set(player) != {"name", "room_passwords"} \
                or not isinstance(player["name"], str) or not player["name"] \
                or len(player["name"]) > 19 or player["name"] in names \
                or not isinstance(player["room_passwords"], list) or not 1 <= len(player["room_passwords"]) <= 4:
            raise ValueError("each unique player needs 1-4 dynamic room credentials")
        for token in player["room_passwords"]:
            if not isinstance(token, str) or not 8 <= len(token) <= 19 or not token.isascii() \
                    or any(c.isspace() or c in "#," for c in token) or token in tokens:
                raise ValueError("room credentials must be unique supported wire tokens")
            tokens.add(token)
        names.add(player["name"])
        players.append(Enrollment(player["name"], tuple(player["room_passwords"])))
    return Credentials(value["host"], value["port"], value["test_only_multiple_identities"], tuple(players))


def round_robin_pairs(players):
    """Enumeration only; never creates rooms or adds games to an admitted plan."""
    if not 2 <= len(players) <= 5 or len(set(players)) != len(players):
        raise ValueError("a round robin needs 2-5 unique players")
    return list(itertools.combinations(players, 2))


class LeagueConnection:
    """One token/room/viewer on one TCP connection; no reconnect masquerades as a game."""
    def __init__(self, client_factory, *, room, player, games, on_game, timeout=600., deadline=None, start_game=1,
                 admit_special=False, lobby_wait_until_deadline=False):
        if type(games) is not int or games < 1 or type(start_game) is not int or start_game < 1 \
                or (start_game + games - 1) % 2 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("league quota needs an even positive game count and finite timeout")
        if deadline is not None and (not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
            raise ValueError("league wall deadline must be finite")
        self.factory, self.room, self.player, self.games, self.on_game = client_factory, room, player, games, on_game
        self.timeout, self.deadline = timeout, deadline
        self.start_game = start_game
        self.admit_special = bool(admit_special)
        self.lobby_wait_until_deadline = bool(lobby_wait_until_deadline)
        self.client = self._fresh(start_game - 1)
        self.stream = None
        self.records = []
        self.in_game = False
        self.awaiting_start = False
        self.started_at = None
        self.last_terminal = None
        self.duplicate_terminals = 0

    def _fresh(self, index):
        client = self.factory(index)
        if not isinstance(client, NetDuelClient) or not client.allow_match_mode:
            raise ValueError("a league factory must produce a fresh opted-in NetDuelClient")
        # Later games keep the transport and do not call connect(), whose
        # validator normally installs the native prompt handler. Every
        # fresh client still needs that handler before its MSG_START bind.
        validator = getattr(client.policy, "validate_client", None)
        if validator is not None:
            validator(client)
        return client

    def _finish(self):
        client = self.client
        if not self.in_game or client.result.our_player not in (0, 1) \
                or client.result.winner not in (0, 1, 2) or client.result.error:
            raise DuelError("league terminal lacks a started clean viewer/game/seat")
        client.result.seconds = time.monotonic() - self.started_at
        client.result.lp = tuple(client._lp)
        client.policy.on_duel_end(client.result)
        report = client.policy.report()
        if report.get("active") is not False or report.get("pending_messages") != 0 or report.get("failures") != []:
            raise DuelError("league policy did not close its own completed game cleanly")
        index = self.start_game - 1 + len(self.records)
        record = {"schema": SCHEMA, "room": self.room, "player": self.player, "game": index + 1,
                  "bo2": index // 2 + 1, "leg": index % 2 + 1, "result": client.result.as_dict(),
                  "natural_terminal": client.result.win_reason in NATURAL_REASONS,
                  "terminal_kind": NATURAL_REASONS.get(client.result.win_reason, "special_or_administrative"),
                  "terminal_evidence": "received_MSG_WIN", "policy_session": getattr(client.policy, "session", None),
                  "policy_report": report, "training_eligible": False}
        if not self._terminal_admitted(record):
            raise DuelError("league game has a special/administrative ending, not an admitted natural terminal")
        if client.result.win_reason == 1:
            loser = 1 - client.result.winner if client.result.winner in (0, 1) else None
            if (loser is not None and client.result.lp[loser] != 0) \
                    or (loser is None and client.result.lp != (0, 0)):
                raise DuelError("life-point terminal disagrees with the received public LP")
        self.on_game(record)
        self.records.append(record)
        self.in_game, self.awaiting_start = False, True

    def _terminal_admitted(self, record):
        # The resume profile is deliberately narrow: reason zero is the
        # gateway's explicitly approved surrender or time-up MSG_WIN class.  A
        # disconnect has no MSG_WIN and never reaches this method; unknown
        # future reason codes remain refused.
        return record['natural_terminal'] or (getattr(self, 'admit_special', False)
                                              and record['result']['win_reason'] in (0, 3))

    def _next(self):
        previous = self.client
        client = self._fresh(self.start_game - 1 + len(self.records))
        if client is previous or client.policy is previous.policy or client.board is previous.board:
            raise DuelError("league factory reused another game's mutable policy/client/view")
        client.stream = self.stream
        client.host_info = previous.host_info
        client.lobby_pos, client.is_host = previous.lobby_pos, previous.is_host
        # An engine-side continuous series starts itself after the fixed-deck
        # side handshake. Do not duplicate the initial lobby HS_START.
        client._start_sent = True
        self.client = client

    def receive(self, op, payload):
        """Individually framed real packets; useful for exact protocol CPU tests too."""
        msg = payload[0] if op == P.STOC.GAME_MSG and payload else None
        if msg == C.MSG_START:
            if self.in_game or len(self.records) >= self.games:
                raise DuelError("unexpected league START before terminal or beyond the admitted quota")
            self.in_game, self.awaiting_start = True, False
            self.started_at = time.monotonic()
        elif msg == C.MSG_WIN and not self.in_game:
            if self.last_terminal == payload and self.records:
                self.duplicate_terminals += 1
                return
            raise DuelError("league WIN without a new started duel")
        if op == P.STOC.DUEL_END:
            if self.in_game:
                raise DuelError("league DUEL_END without the natural MSG_WIN result")
            if not self.records:
                raise DuelError("league connection ended without playing a duel")
            return
        if op == P.STOC.CHANGE_SIDE:
            if self.in_game or not self.records or self.client.host_info is None or self.client.host_info.mode != 1:
                raise DuelError("unexpected league fixed-deck side prompt")
            # This is a fixed-deck resubmission, not a side-deck change. The
            # 6009 gateway checks/replaces it with its public 40+15 deck.
            self.stream.send(P.CTOS.UPDATE_DECK, P.update_deck(self.client.main + self.client.extra, []))
            return
        if op == P.STOC.WAITING_SIDE:
            if self.in_game or not self.records:
                raise DuelError("league WAITING_SIDE arrived inside an active duel")
            return
        self.client._handle(op, payload)
        if msg == C.MSG_WIN:
            self.last_terminal = bytes(payload)
            self._finish()
            if len(self.records) < self.games:
                self._next()  # even pre-START RPS/TP packets use the NEW game's RNG

    def receive_timeout(self):
        remaining = None if self.deadline is None else self.deadline - time.monotonic()
        if getattr(self, 'lobby_wait_until_deadline', False) and not self.in_game and remaining is not None:
            return remaining
        return self.timeout if remaining is None else min(self.timeout, remaining)

    def run(self):
        try:
            self.client.connect()
            self.stream = self.client.stream
            while len(self.records) < self.games:
                remaining = self.receive_timeout()
                if remaining <= 0:
                    raise TimeoutError("league finite wall budget expired; unfinished games are not counted")
                op, payload = self.stream.recv(remaining)
                self.receive(op, payload)
            return self.records
        finally:
            if self.in_game:
                # Transport failures are diagnostic, not a natural terminal.
                try:
                    self.client.policy.on_duel_end(self.client.result)
                except Exception:
                    pass
            stream = self.stream if self.stream is not None else self.client.stream
            if stream is not None:
                stream.close()
                self.stream = None


__all__ = ["Credentials", "read_credentials", "round_robin_pairs", "LeagueConnection", "SCHEMA"]
