"""Explicit finite multi-room league client; real competition represents ONE identity.

Tokens live only in the approved local 0600 file and JOIN_GAME packets. This
tool never starts a policy service, changes a league server, enters finals,
trains a model, or infers a larger BO2 quota from the number of players.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time

from mirrorforce.netduel.cards import CardPool, load_ydk
from mirrorforce.netduel.client import NetDuelClient
from mirrorforce.netduel.league import LeagueConnection, read_credentials
from mirrorforce.netduel import agent_public_recipe as PR
from mirrorforce.netduel.agent_policy import RemotePolicy, check_report

SCHEMA = "mirrorforce_league_run/v1"
RESUME_SCHEMA = "mirrorforce_league_run/v2-resume"
SUBSET_SCHEMA = "mirrorforce_league_run/v3-resume-subset"
LOBBY_SUBSET_SCHEMA = "mirrorforce_league_run/resume-subset-lobby"
PLAN_FIELDS = {"schema", "mode", "approved_total_bo2", "finals_allowed", "rooms", "service", "deck", "cards_db",
               "seed", "max_seconds", "timeout_seconds"}
RESUME_ROOM_FIELDS = {"id", "session_id", "bo2", "members", "server_bo2_limit", "resume"}
RESUME_FIELDS = {"profile", "start_game", "remaining_games", "consumed"}
SUBSET_FIELDS = {"parent_plan", "selected_room_ids"}
CONSUMED_FIELDS = {"game", "session_id", "side", "winner", "win_reason", "natural_terminal", "evidence"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def pinned(reference):
    if type(reference) is not dict or set(reference) != {"path", "sha256"}:
        raise ValueError("a content-pinned public input is required")
    path = Path(reference["path"])
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
        raise ValueError("public league input content differs")
    return path


def validate_consumed_evidence(room, consumed):
    try:
        evidence = json.loads(pinned(consumed["evidence"]).read_bytes())
    except (UnicodeError, ValueError) as exc:
        raise ValueError("consumed evidence is not a valid official JSON record") from exc
    if type(evidence) is not dict:
        raise ValueError("official consumed evidence must be a JSON object")
    result = evidence.get("official_result")
    if evidence.get("schema") != "mirrorforce_league_official_consumed_evidence/v1" \
            or evidence.get("room") != room["id"] or evidence.get("session_id") != consumed["session_id"] \
            or type(evidence.get("absolute_game")) is not int or evidence.get("absolute_game") != consumed["game"] \
            or evidence.get("registered_side") != consumed["side"] or type(result) is not dict \
            or type(result.get("game_number")) is not int or result.get("game_number") != consumed["game"] \
            or type(result.get("winner")) is not str or result.get("winner") != consumed["winner"] \
            or type(result.get("reason")) is not int or result.get("reason") != consumed["win_reason"]:
        raise ValueError("official consumed evidence content differs from its resume contract")
    expected_class = {0: "official_special_or_administrative_not_natural", 1: "official_natural",
                      2: "official_natural", 3: "official_timeup_not_natural"}[consumed["win_reason"]]
    if evidence.get("classification") != expected_class:
        raise ValueError("official consumed evidence terminal classification differs")


def validate_plan(plan, credentials):
    subset_schema = type(plan) is dict and plan.get("schema") in (SUBSET_SCHEMA, LOBBY_SUBSET_SCHEMA)
    expected_plan_fields = PLAN_FIELDS | ({"resume_subset"} if subset_schema else set()) \
        | ({"lobby_wait_until_deadline"} if type(plan) is dict and plan.get("schema") == LOBBY_SUBSET_SCHEMA else set())
    if type(plan) is not dict or set(plan) != expected_plan_fields \
            or plan["schema"] not in (SCHEMA, RESUME_SCHEMA, SUBSET_SCHEMA, LOBBY_SUBSET_SCHEMA) \
            or plan["mode"] not in ("test", "competition") or plan["finals_allowed"] is not False:
        raise ValueError("closed explicit league plan is required; finals are excluded")
    if plan["schema"] in (RESUME_SCHEMA, SUBSET_SCHEMA, LOBBY_SUBSET_SCHEMA) and plan["mode"] != "competition":
        raise ValueError("the special-preserving resume contract is only for the approved real competition")
    if plan["schema"] == LOBBY_SUBSET_SCHEMA and plan["lobby_wait_until_deadline"] is not True:
        raise ValueError("extended lobby wait must be an explicit opt-in to the unchanged whole-run deadline")
    parent = None
    if plan["schema"] in (SUBSET_SCHEMA, LOBBY_SUBSET_SCHEMA):
        subset = plan["resume_subset"]
        if type(subset) is not dict or set(subset) != SUBSET_FIELDS or not isinstance(subset["selected_room_ids"], list) \
                or not subset["selected_room_ids"] or len(set(subset["selected_room_ids"])) != len(subset["selected_room_ids"]):
            raise ValueError("a subset resume must name its exact unique rooms and pinned parent plan")
        parent = json.loads(pinned(subset["parent_plan"]).read_bytes())
        if parent.get("schema") != RESUME_SCHEMA or parent.get("approved_total_bo2") != plan["approved_total_bo2"]:
            raise ValueError("subset resume parent is not the approved full resume plan")
        validate_plan(parent, credentials)
    if type(plan["approved_total_bo2"]) is not int or plan["approved_total_bo2"] < 1 \
            or type(plan["seed"]) is not int or plan["seed"] < 0 \
            or type(plan["max_seconds"]) not in (int, float) or not 0 < plan["max_seconds"] <= 10800 \
            or type(plan["timeout_seconds"]) not in (int, float) or not 0 < plan["timeout_seconds"] <= plan["max_seconds"]:
        raise ValueError("league needs an explicit quota and finite positive budgets")
    if not isinstance(plan["rooms"], list) or not plan["rooms"]:
        raise ValueError("enroll explicit dynamic rooms; never invent them from identities")
    names = {player.player: player for player in credentials.players}
    rooms, used = set(), set()
    total = 0
    for room in plan["rooms"]:
        expected_room_fields = RESUME_ROOM_FIELDS if plan["schema"] in (RESUME_SCHEMA, SUBSET_SCHEMA, LOBBY_SUBSET_SCHEMA) else \
            {"id", "session_id", "bo2", "members", "server_bo2_limit"}
        if type(room) is not dict or set(room) != expected_room_fields \
                or not isinstance(room["id"], str) or not room["id"] or room["id"] in rooms \
                or not isinstance(room["session_id"], str) or not room["session_id"] \
                or type(room["bo2"]) is not int or room["bo2"] < 1 or room["bo2"] != room["server_bo2_limit"]:
            raise ValueError("each room needs its verified server quota; do not leave a larger series half played")
        if plan["schema"] in (RESUME_SCHEMA, SUBSET_SCHEMA, LOBBY_SUBSET_SCHEMA):
            resume = room["resume"]
            if type(resume) is not dict or set(resume) != RESUME_FIELDS \
                    or resume["profile"] not in ("official-special-preserving/v1", "official-special-preserving/v2") \
                    or type(resume["start_game"]) is not int or type(resume["remaining_games"]) is not int \
                    or not 1 <= resume["start_game"] <= 2 * room["bo2"] \
                    or resume["remaining_games"] != 2 * room["bo2"] - resume["start_game"] + 1 \
                    or not isinstance(resume["consumed"], list) \
                    or len(resume["consumed"]) != resume["start_game"] - 1:
                raise ValueError("resume must bind the exact consumed prefix and remaining absolute games")
            for index, consumed in enumerate(resume["consumed"], 1):
                if type(consumed) is not dict or set(consumed) != CONSUMED_FIELDS \
                        or consumed["game"] != index or consumed["session_id"] != room["session_id"] \
                        or consumed["side"] not in ("a", "b") or consumed["winner"] not in ("a", "b", "draw") \
                        or type(consumed["win_reason"]) is not int or consumed["win_reason"] not in (0, 1, 2, 3) \
                        or type(consumed["natural_terminal"]) is not bool \
                        or consumed["natural_terminal"] != (consumed["win_reason"] in (1, 2)):
                    raise ValueError("consumed resume prefix identity/result differs")
                validate_consumed_evidence(room, consumed)
        rooms.add(room["id"])
        if not isinstance(room["members"], list) or len(room["members"]) != (2 if plan["mode"] == "test" else 1):
            raise ValueError("tests enroll both room viewers; real competition enrolls one identity")
        sides, members = set(), set()
        for member in room["members"]:
            if type(member) is not dict or set(member) != {"player", "credential_index", "side"} \
                    or member["player"] not in names or member["player"] in members or member["side"] not in ("a", "b") \
                    or member["side"] in sides or type(member["credential_index"]) is not int \
                    or not 0 <= member["credential_index"] < len(names[member["player"]].passwords):
                raise ValueError("dynamic enrollment mapping differs from the approved identity/tokens")
            key = member["player"], member["credential_index"]
            if key in used:
                raise ValueError("one dynamic room credential was enrolled twice")
            used.add(key); members.add(member["player"]); sides.add(member["side"])
        if plan["schema"] in (RESUME_SCHEMA, SUBSET_SCHEMA, LOBBY_SUBSET_SCHEMA):
            side = room["members"][0]["side"]
            if any(row["side"] != side for row in room["resume"]["consumed"]):
                raise ValueError("consumed resume side differs from the registered room identity")
        total += room["bo2"]
    if plan["schema"] in (SUBSET_SCHEMA, LOBBY_SUBSET_SCHEMA):
        parent_rooms = {room["id"]: room for room in parent["rooms"]}
        if subset["selected_room_ids"] != [room["id"] for room in plan["rooms"]] or any(
                room["id"] not in parent_rooms or any(room[key] != parent_rooms[room["id"]][key]
                for key in ("id", "session_id", "bo2", "members", "server_bo2_limit")) for room in plan["rooms"]):
            raise ValueError("subset rooms differ from the exact parent enrollment")
    elif total != plan["approved_total_bo2"]:
        raise ValueError("server rooms would exceed/change the approved total BO2; ask for direction before joining")
    if type(plan["service"]) is not dict or set(plan["service"]) != {"socket", "identity", "actor_sha256"}:
        raise ValueError("league requires a registered read-only policy service")
    return used


def pair_results(rows, rooms):
    """Only paired, natural terminal viewer evidence becomes a completed duel/BO2."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row["room"], row["game"]), []).append(row)
    duels = []
    for room in rooms:
        names = {member["player"] for member in room["members"]}
        for game in range(1, room["bo2"] * 2 + 1):
            views = grouped.get((room["id"], game), [])
            if len(views) != 2 or {view["player"] for view in views} != names:
                continue
            if not all(view["natural_terminal"] and not view["result"]["error"] for view in views):
                raise ValueError("a failed/administrative view cannot become a natural completed duel")
            a, b = views
            if {a["result"]["our_player"], b["result"]["our_player"]} != {0, 1} \
                    or any(a["result"][key] != b["result"][key] for key in ("winner", "win_reason", "lp")):
                raise ValueError("two room viewers disagree on terminal or actual seat mapping")
            sides = {member["player"]: member["side"] for member in room["members"]}
            wanted = {name: ((0 if side == "a" else 1) if game % 2 else (1 if side == "a" else 0))
                      for name, side in sides.items()}
            actual = {view["player"]: view["result"]["our_player"] for view in views}
            if actual != wanted:
                raise ValueError("the server did not perform the registered automatic BO2 seat swap")
            duels.append({"room": room["id"], "session_id": room["session_id"], "game": game,
                          "bo2": (game - 1) // 2 + 1, "leg": (game - 1) % 2 + 1,
                          "seat_mapping": actual, "winner": a["result"]["winner"],
                          "win_reason": a["result"]["win_reason"], "natural_terminal": True})
    complete = sum(all(any(d["room"] == room["id"] and d["game"] == game for d in duels)
                       for game in (2 * bo2 - 1, 2 * bo2))
                   for room in rooms for bo2 in range(1, room["bo2"] + 1))
    return {"completed_duels": len(duels), "completed_bo2": complete, "duels": duels}


class LeagueRun:
    def __init__(self, plan, credentials, output):
        validate_plan(plan, credentials)
        self.plan, self.credentials, self.output = plan, credentials, Path(output)
        raw = pinned(plan["service"]["identity"]).read_bytes()
        self.identity = json.loads(raw)["identity"]
        if self.identity["checkpoint_sha256"] != plan["service"]["actor_sha256"] \
                or self.identity.get("selection") != "greedy" or self.identity.get("weights") != "iterate" \
                or PR.mode(self.identity) != "mirror" or self.identity.get("ar_search") is not None:
            raise ValueError("league currently admits only its exact pure raw greedy mirror actor, no search")
        self.deck = load_ydk(pinned(plan["deck"]))
        if self.deck[2]:
            raise ValueError("league tests use the fixed public deck, never side cards")
        self.pool = CardPool(pinned(plan["cards_db"]))
        self.rows, self.references = [], []
        self.lock = threading.Lock()
        self.connections = []

    def publish(self, category, value):
        raw = canonical(self.credentials.scrub(value))
        digest = hashlib.sha256(raw).hexdigest()
        path = self.output / category / (digest + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        return {"path": str(path), "sha256": digest}

    def accept(self, row):
        # Own legal-view reports remain separate files. Their bytes are not
        # an input to any other player or room, only terminal protocol audit.
        room = next((room for room in self.plan["rooms"] if room["id"] == row["room"]), None)
        member = None if room is None else next((m for m in room["members"] if m["player"] == row["player"]), None)
        if member is None or type(row["game"]) is not int or not 1 <= row["game"] <= 2 * room["bo2"]:
            raise ValueError("viewer terminal is outside its exact enrolled room/quota")
        first = 0 if member["side"] == "a" else 1
        if row["result"]["our_player"] != (first if row["game"] % 2 else 1 - first):
            raise ValueError("viewer MSG_START seat differs from the server's automatic swap plan")
        # An official surrender/time-up can occur before this viewer receives
        # any actionable prompt. Preserve only a clean registered report that
        # also carries the actual MSG_WIN evidence and approved reason; natural
        # terminals retain the stricter response audit.
        promptless_special = row["natural_terminal"] is False \
            and row.get("terminal_evidence") == "received_MSG_WIN" \
            and row["result"].get("win_reason") in (0, 3)
        check_report(row["policy_report"], self.identity, row["result"],
                     allow_forced_only_terminal=promptless_special)
        with self.lock:
            key = row["room"], row["player"], row["game"]
            if any((x["room"], x["player"], x["game"]) == key for x in self.rows):
                raise ValueError("duplicate viewer terminal is not another completed duel")
            reference = self.publish("viewer-games", row)
            self.rows.append(row)
            self.references.append({"room": key[0], "player": key[1], "game": key[2], **reference})

    def run(self):
        if self.output.exists():
            raise ValueError("preserve old league attempts; choose a new output root")
        self.output.mkdir(parents=True, mode=0o700)
        self.publish("enrollment", {"schema": SCHEMA + "#enrollment", "credentials": self.credentials.public(),
                                  "plan": self.plan, "training_eligible": False})
        deadline = time.monotonic() + self.plan["max_seconds"]
        by_name = {player.player: player for player in self.credentials.players}
        for room in self.plan["rooms"]:
            for member in room["members"]:
                player = by_name[member["player"]]
                seed = int(hashlib.sha256(canonical({"seed": self.plan["seed"], "room": room["id"],
                                                    "player": player.player})).hexdigest()[:8], 16)
                def factory(game, room=room, player=player, member=member, seed=seed):
                    policy = RemotePolicy("unix:" + self.plan["service"]["socket"], self.identity,
                                            seed=(seed + game) % (1 << 32), timeout=self.plan["timeout_seconds"])
                    return NetDuelClient(self.credentials.host, self.credentials.port, player.player,
                        self.deck[0], self.deck[1], policy, seed=(seed + game) % (1 << 32),
                        card_pool=self.pool, password=player.passwords[member["credential_index"]],
                        allow_match_mode=True, start_when_ready=True, timeout=self.plan["timeout_seconds"],
                        max_options=self.identity["client_config"]["max_options"], capture=[])
                self.connections.append(LeagueConnection(factory, room=room["id"], player=player.player,
                    games=(room.get("resume") or {}).get("remaining_games", 2 * room["bo2"]),
                    start_game=(room.get("resume") or {}).get("start_game", 1), on_game=self.accept,
                    timeout=self.plan["timeout_seconds"], deadline=deadline,
                    admit_special=self.plan["schema"] in (RESUME_SCHEMA, SUBSET_SCHEMA, LOBBY_SUBSET_SCHEMA),
                    lobby_wait_until_deadline=self.plan.get("lobby_wait_until_deadline", False)))
        failures = []
        with ThreadPoolExecutor(max_workers=len(self.connections)) as pool:
            futures = {pool.submit(connection.run): connection for connection in self.connections}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    failures.append({"room": futures[future].room, "player": futures[future].player,
                                     "error": self.credentials.scrub(type(exc).__name__ + ": " + str(exc))})
                    # A room failure must not manufacture disconnects in the other
                    # independently admitted rooms.  Each connection is finite and
                    # retains its own failure/unknown outcome.
        paired = None
        if self.plan["mode"] == "test":
            try:
                paired = pair_results(self.rows, self.plan["rooms"])
            except ValueError as exc:
                failures.append({"room": "pairing-audit", "player": None,
                                 "error": self.credentials.scrub(type(exc).__name__ + ": " + str(exc))})
        expected_views = sum((room.get("resume") or {}).get("remaining_games", 2 * room["bo2"])
                             * len(room["members"]) for room in self.plan["rooms"])
        complete = not failures and len(self.rows) == expected_views \
            and (paired is None or paired["completed_bo2"] == self.plan["approved_total_bo2"])
        consumed = [dict(room=room["id"], **row) for room in self.plan["rooms"]
                    for row in (room.get("resume") or {}).get("consumed", [])]
        result = {"schema": self.plan["schema"] + "#result", "complete": complete,
                  "approved_total_bo2": self.plan["approved_total_bo2"],
                  "viewer_records": len(self.rows), "viewer_references": self.references, "paired": paired,
                  "official_consumed": consumed,
                  "natural_terminals": sum(row["natural_terminal"] for row in self.rows),
                  "special_or_administrative_terminals": sum(not row["natural_terminal"] for row in self.rows),
                  "failures": failures, "training_eligible": False, "finals_entered": False,
                  "server_identity_sha256": self.plan["service"]["identity"]["sha256"]}
        return self.publish("results", result), result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--credentials-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prepare-only", action="store_true", help="validate public quota/mapping; do not connect")
    args = parser.parse_args(argv)
    plan = json.loads(pinned({"path": args.plan, "sha256": args.plan_sha256}).read_bytes())
    credentials = read_credentials(args.credentials_file, mode=plan["mode"])
    used = validate_plan(plan, credentials)
    if args.prepare_only:
        print(json.dumps({"configured": True, "network_joined": False, "gpu_started": False,
            "approved_total_bo2": plan["approved_total_bo2"], "viewer_connections": len(used),
            "room_count": len(plan["rooms"]), "finals_allowed": False}))
        return 0
    reference, result = LeagueRun(plan, credentials, args.out).run()
    print("LEAGUE-RESULT " + json.dumps({"reference": reference, "complete": result["complete"],
        "paired": None if result["paired"] is None else {key: result["paired"][key] for key in
            ("completed_duels", "completed_bo2")}, "viewer_records": result["viewer_records"]}), flush=True)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
