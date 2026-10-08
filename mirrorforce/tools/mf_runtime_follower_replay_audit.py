"""Read-only replay of recorded live games through the search follower, to find server/core message differences.

Each input is one recorded game of a client (a league-client viewer record or a search-series game/last-client
record) holding its seat's received ``public_messages`` and sent wire ``responses``. The follower is built exactly as
the search client builds it (``SearchPolicy._start_follower``) and is driven through every recorded prompt with the
recorded answer. The first divergence of each game is reported with the differing bytes; nothing is sent, no model
runs, and no hidden information is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

SCHEMA = "mirrorforce_follower_replay_audit/v1"


def prompt_player(packet: bytes) -> int:
    from mirrorforce.netduel import constants as C
    return packet[2] if packet[0] == C.MSG_SELECT_SUM else packet[1]


def load_game(path: Path) -> dict:
    """The seat's public stream, its answers and the prompt each answer belongs to."""
    from mirrorforce.netduel.client import SELECT_MESSAGES
    record = json.loads(path.read_bytes())
    report = record.get("policy_report") or {}
    messages, responses = report.get("public_messages"), report.get("responses")
    viewer = (record.get("result") or {}).get("our_player")
    if not isinstance(messages, list) or not isinstance(responses, list) or viewer not in (0, 1):
        raise ValueError("record has no seat stream, answers and seat")
    ends = [i for i, (msg, payload) in enumerate(messages)
            if msg in SELECT_MESSAGES and prompt_player(bytes([msg]) + bytes.fromhex(payload)) == viewer]
    return {"viewer": viewer, "messages": messages, "responses": responses, "prompt_ends": ends,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def follower(core, game, deck, room):
    from mirrorforce.common.client_history_registry import PROFILE
    from mirrorforce.common.client_root import ClientRootController
    from mirrorforce.common.client_shadow import BlankClientSync
    from mirrorforce.netduel.agent_public_recipe import declare
    declared = declare(deck.main, deck.extra)
    sync = BlankClientSync(viewer=game["viewer"], own_deck=deck, opponent_main=len(declared["main"]),
                           opponent_extra=len(declared["extra"]), core=core, seed=19, record_origins=True,
                           record_replay=True, history_registry=PROFILE, public_action_rebind=True,
                           hidden_hand_rebind=True, hidden_target_deferral=True, public_identities_only=True,
                           read_answers=True, single_pass=True, max_tries=256, max_seed_tries=16,
                           hidden_target_pool=tuple(sorted(set(declared["main"] + declared["extra"]))),
                           public_opponent_recipe=declared, start_lp=room["start_lp"], start_hand=room["start_hand"],
                           draw_count=room["draw_count"], duel_options=room["duel_rule"] << 16)
    return sync, ClientRootController(sync, quiesce=lambda: None, native_idle=lambda: True)


def external_end(messages) -> int:
    """Index of a server-adjudicated MSG_WIN (surrender 0 / time-up 3), which no engine reproduces, else len."""
    from mirrorforce.netduel import constants as C
    for index, (msg, payload) in enumerate(messages):
        body = bytes.fromhex(payload)
        if msg == C.MSG_WIN and len(body) == 2 and body[1] in (C.WIN_REASON_SURRENDER, C.WIN_REASON_TIMEUP):
            return index
    return len(messages)


def replay(core, game, deck, room, seconds):
    """Drive the follower through every recorded answer; the first failure, or None when the game is followed.

    A game the server ended by surrender or time-up is followed up to that MSG_WIN, as the search client does.
    """
    stop = external_end(game["messages"])
    prompts = sum(1 for e in game["prompt_ends"] if e < stop)
    if len(game["responses"]) < prompts:
        return {"kind": "incomplete", "prompts": prompts, "answers": len(game["responses"])}
    sync, owner = follower(core, game, deck, room)
    messages, responses, cursor = game["messages"][:stop], game["responses"], 0
    deadline = time.monotonic() + seconds
    wire, answered = -1, False
    try:
        for wire, end in enumerate(e for e in game["prompt_ends"][:len(responses)] if e < len(messages)):
            if time.monotonic() >= deadline:
                return {"kind": "deadline", "wire": wire, "packet": cursor}
            while cursor <= end:
                msg, payload = messages[cursor]
                owner.receive(bytes([msg]) + bytes.fromhex(payload))
                cursor += 1
            if owner.advance() is None:
                return {"kind": "no_root", "wire": wire, "packet": cursor - 1, "msg": messages[end][0]}
            with owner.capture_root(max_seconds=min(60., max(1., deadline - time.monotonic()))) as root:
                pass
            owner.commit_real_response(root, bytes.fromhex(responses[wire]), validate_original=lambda *_a: True,
                                       send=lambda _data: None)
        answered = True
        while cursor < len(messages):
            msg, payload = messages[cursor]
            owner.receive(bytes([msg]) + bytes.fromhex(payload))
            cursor += 1
        owner.advance()
        return None
    except Exception as exc:
        if answered and stop < len(game["messages"]):
            # Every prompt before the server's verdict was answered and followed; the server then ended the game
            # by surrender or time-up, and the stream stops where the local duel goes on. The live client stops its
            # follower at that MSG_WIN.
            return None
        text = f"{type(exc).__name__}: {exc}"
        stats = getattr(sync, "stats", None)
        tried = [[int(i), real, local, n] for (i, real, local), n in stats.counts.most_common(8)] \
            if stats is not None else []
        last = getattr(sync, "last_local", None)
        return {"kind": "error", "wire": wire, "packet": cursor, "error": text[:600],
                "tried_divergences": tried,
                "last_local": None if last is None else [int(last.msg), bytes(last.payload).hex()[:96]],
                "received_window": [[i, m, p[:64]] for i, (m, p) in enumerate(messages)
                                    if cursor - 12 <= i <= cursor + 6],
                "trace": traceback.format_exc()[-1500:]}
    finally:
        owner.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, nargs="+", required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=600.)
    args = parser.parse_args(argv)
    from mirrorforce.effectinfo import get_effectinfo_core
    from mirrorforce.netduel.cards import load_ydk
    from mirrorforce.worldmodel.engine import DeckList
    main_deck, extra, _ = load_ydk(args.deck)
    deck = DeckList("recorded-own", tuple(main_deck), tuple(extra))
    room = {"start_lp": 8000, "start_hand": 5, "draw_count": 1, "duel_rule": 5}
    core = get_effectinfo_core()
    rows = []
    for path in args.games:
        started = time.monotonic()
        try:
            game = load_game(path)
            failure = replay(core, game, deck, room, args.seconds)
            row = {"game": str(path), "sha256": game["sha256"], "viewer": game["viewer"],
                   "external_end": external_end(game["messages"]) < len(game["messages"]),
                   "packets": len(game["messages"]), "answers": len(game["responses"]),
                   "prompts": len(game["prompt_ends"]), "failure": failure}
        except Exception as exc:
            row = {"game": str(path), "failure": {"kind": "load", "error": f"{type(exc).__name__}: {exc}"[:600]}}
        row["seconds"] = round(time.monotonic() - started, 2)
        rows.append(row)
        print(json.dumps({"game": Path(path).name, "ok": row["failure"] is None,
                          "failure": (row["failure"] or {}).get("error", (row["failure"] or {}).get("kind"))}),
              flush=True)
    summary = {"schema": SCHEMA, "games": len(rows), "followed": sum(r["failure"] is None for r in rows),
               "rows": rows, "training_eligible": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(summary, handle, sort_keys=True, indent=1)
    print(json.dumps({"out": str(args.out), "games": summary["games"], "followed": summary["followed"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
