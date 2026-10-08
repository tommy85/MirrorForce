"""A checkpoint as a network client: every received game message goes to the policy service, which answers
each prompt with the exact response bytes (``netduel.agent_wire``, ``tools/mf_runtime_policy_service.py``).

The observation is the env's own, rebuilt from this client's stream by ``duel_native.ClientDuel`` inside the
service; the menu, its sub-choices and the response encoding are the env's too, so this client never parses a menu
for the policy. The standard parser still reads each prompt for the shadow board's cross-check. The captured stream
is kept in the report, so an audit can feed it to a fresh builder and compare. Evaluation and deployment only.
"""
from __future__ import annotations

import hashlib
import json

from . import constants as C
from .actions import SelectResult, parse_select
from .client import DuelError
from . import agent_wire as W
from . import agent_public_recipe as PR

REPORT_SCHEMA = "mirrorforce_client_report/v1"


def identity_sha256(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class RemotePolicy:
    """One game's client side. ``expected_identity`` is the service identity this game must be played by."""

    name = "policy"

    def __init__(self, address, expected_identity, *, seed=0, timeout=120.0):
        if not isinstance(expected_identity, dict) or expected_identity.get("protocol") != W.PROTOCOL:
            raise ValueError("a client needs the registered service identity")
        self.address, self.expected_identity, self.seed, self.timeout = address, expected_identity, int(seed), timeout
        self.opponent_recipe_mode = PR.mode(expected_identity)
        self.public_opponent_recipe = None
        self.sock = self.session = self._client = None
        self.pending, self.capture, self.decisions, self.failures = [], [], [], []
        self.responses, self.forced = [], 0
        self.active = self.identity_verified = False
        self.closed = None

    # -- client hooks ------------------------------------------------------

    def _call(self, request):
        """Default evaluation transport; deployment may opt in to a separately audited deadline/recovery law."""
        return W.call(self.sock, request)

    def validate_client(self, client):
        if client.version != 0x1362:
            raise DuelError("the client speaks the registered 0x1362 protocol")
        client.selection_parser = self._parse

    def bind(self, client):
        try:
            self._client = client
            if client.selection_parser != self._parse:
                raise W.WireError("the policy's prompt handler is not installed")
            if self.sock is None:
                self.sock = W.connect(self.address, self.timeout)
                identity = self._call({"op": "identity"})
                if identity != self.expected_identity:
                    raise W.WireError("the policy service is not the registered checkpoint and rule")
                self.identity_verified = True
            seat = client.result.our_player
            if seat not in (0, 1):
                raise W.WireError("the client has no seat at duel start")
            request = {"op": "open", "seat": seat, "main": list(client.main),
                       "extra": list(client.extra), "seed": self.seed}
            self.public_opponent_recipe = PR.for_game(self.expected_identity, list(client.main), list(client.extra))
            if self.public_opponent_recipe is not None:
                request["public_opponent_recipe"] = self.public_opponent_recipe
                request["opponent_recipe_mode"] = self.opponent_recipe_mode
            reply = self._call(request)
            self.session, self.active, self.seat = reply["session"], True, seat
        except Exception as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
            raise DuelError("policy session failed: " + str(exc)) from exc

    def observe_game_message(self, msg, body):
        if not self.active:
            return
        entry = (int(msg), bytes(body))
        self.pending.append(entry)
        self.capture.append(entry)

    def _parse(self, msg, body, ctx):
        """Every prompt: the standard parse for the board cross-check, then the service's response bytes."""
        standard = parse_select(msg, body, ctx)
        try:
            if not self.active or not self.pending or self.pending[-1] != (int(msg), bytes(body)):
                raise W.WireError("the prompt did not reach the policy's message stream")
            messages = [[m, b.hex()] for m, b in self.pending]
            reply = self._call({"op": "decide", "session": self.session, "messages": messages})
            self.pending = []
            response = bytes.fromhex(reply["response"])
            index = len(self.responses)
            self.responses.append(response.hex())
            self.forced += bool(reply["forced"])
            for row in reply["decisions"]:
                self.decisions.append({**row, "response_index": index, "turn": self._client.result.turns,
                                       "viewer": self.seat})
        except Exception as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
            raise DuelError("policy decision failed: " + str(exc)) from exc
        player = body[1] if msg == C.MSG_SELECT_SUM else (body[0] if body else 0)
        return SelectResult(msg, player, auto_response=response, note="policy service", cards=standard.cards)

    def on_duel_end(self, result):
        try:
            if self.active and self.sock is not None:
                messages = [[m, b.hex()] for m, b in self.pending]
                self.closed = self._call({"op": "close", "session": self.session, "messages": messages})
                self.pending = []
        except Exception as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
        finally:
            self.active = False
            if self.sock is not None:
                self.sock.close()
                self.sock = None

    def report(self):
        return {"schema": REPORT_SCHEMA, "training_eligible": False, "input_boundary": "client_public_observation",
                "opponent_recipe_mode": self.opponent_recipe_mode,
                "public_opponent_recipe": self.public_opponent_recipe,
                "server_identity": self.expected_identity, "server_identity_sha256": identity_sha256(self.expected_identity),
                "server_identity_verified": self.identity_verified, "active": self.active,
                "failures": list(self.failures), "pending_messages": len(self.pending),
                "responses": list(self.responses), "forced_responses": self.forced,
                "decisions": len(self.decisions), "reports": list(self.decisions), "closed": self.closed,
                "public_messages": [[m, b.hex()] for m, b in self.capture]}


def check_report(report, service_identity, record, *, allow_forced_only_terminal=False):
    """A client ended cleanly as the registered service in its seat: the identity was verified, every response it
    sent came from the service (``record`` is the client's game record), no message was left unsent, no failure."""
    if report.get("schema") != REPORT_SCHEMA or report.get("server_identity") != service_identity \
            or report.get("server_identity_verified") is not True or report.get("training_eligible") is not False \
            or report.get("active") is not False or report.get("failures") != [] or report.get("pending_messages") != 0 \
            or report.get("input_boundary") != "client_public_observation":
        raise ValueError("a client report is not the registered service's clean game")
    mode = PR.mode(service_identity)
    if report.get("opponent_recipe_mode") != mode:
        raise ValueError("the report's opponent recipe mode differs from the service")
    if mode is not None:
        declaration = PR.checked(report.get("public_opponent_recipe"))
        if mode == "known" and declaration != service_identity["public_opponent_recipe"]:
            raise ValueError("the report's public opponent recipe differs from the service")
    elif report.get("public_opponent_recipe") is not None:
        raise ValueError("a closed-decklist report cannot include an opponent recipe")
    rows, closed = report.get("reports") or [], report.get("closed") or {}
    if len(report.get("responses") or ()) != record.get("auto_responses") \
            or closed.get("prompts") != len(report["responses"]) or report.get("decisions") != len(rows) \
            or sum(1 for row in rows if not row["forced"]) != closed.get("decisions") \
            or sum(1 for row in rows if row["forced"]) != report.get("forced_responses") \
            or any(row.get("viewer") != record.get("our_player") for row in rows) \
            or (all(row["forced"] for row in rows) and not allow_forced_only_terminal):
        raise ValueError("a client report lost its responses, decisions or seat")
