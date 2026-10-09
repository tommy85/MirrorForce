"""Play two policy services against each other without a room server.

Each game runs in the in-process host (``netduel.local_host.LocalHostDuel``). Both players are ordinary network
clients (``netduel.client.NetDuelClient`` with ``netduel.agent_policy.RemotePolicy``): each receives only its own
seat's bytes and asks its policy service for every answer, exactly as it would in a room. Both players use the same
deck. Games come in pairs: games ``2k`` and ``2k + 1`` shuffle both decks from seed ``--seed + k``, with player ``a``
moving first in the first game of the pair and second in the other, so luck of the deal cancels out between the
players.

    python -m tools.mf_runtime_local_match --a <socket> --b <socket> --deck <ydk> --cards-db <cards.cdb> \\
        --script-root <directory holding script/> --native <duel_native...so> --games 20 --out <new directory>

``--a`` and ``--b`` may name the same service. The output directory receives ``games.jsonl`` (one row per game) and
``summary.json`` (player ``a``'s wins, losses and draws when moving first, moving second and in total). A game that
fails operationally (an error, a surrender or disconnect, or clients that disagree about the ending) is listed as a
failure and never counted as a result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROLES = ("a", "b")


def schedule(games, seed):
    """``[(game, deal_seed, a_first)]``: each deal seed ``seed + k`` is played twice, ``a`` first and then second."""
    if games < 2 or games % 2:
        raise ValueError("play a positive even number of games, two per deal")
    return [(game, seed + game // 2, game % 2 == 0) for game in range(games)]


def service_identity(socket_path, timeout):
    from mirrorforce.netduel import agent_wire as W
    sock = W.connect("unix:" + str(socket_path), timeout)
    try:
        return W.call(sock, {"op": "identity"})
    finally:
        sock.close()


def play_game(core, cards, recipe, deal_seed, a_first, sockets, identities, *, max_seconds):
    """One game; returns its row. Errors are recorded in the row, never raised."""
    from mirrorforce.netduel import constants as C
    from mirrorforce.netduel import protocol as P
    from mirrorforce.netduel.client import NetDuelClient
    from mirrorforce.netduel.local_host import LocalHostDuel
    from mirrorforce.netduel.agent_policy import RemotePolicy
    from mirrorforce.probes.client_streams import host_config
    from tools.mf_runtime_client_parity import mirror_deal

    deal = dict(mirror_deal(recipe, deal_seed), max_options=192)
    seats = {"a": 0 if a_first else 1, "b": 1 if a_first else 0}
    policies = {role: RemotePolicy("unix:" + str(sockets[role]), identities[role], seed=seats[role]) for role in ROLES}
    clients = [None, None]
    for role, seat in seats.items():
        clients[seat] = NetDuelClient(host="local", port=0, name=role, main=list(recipe.main), extra=list(recipe.extra),
                                      side=[], policy=policies[role], seed=seat, card_pool=cards)
    room = P.HostInfo(0, 0, 0, deal["duel_options"] >> 16, 0, 0, deal["start_lp"], deal["start_hand"],
                      deal["draw_count"], 0)
    started, error, winner = time.monotonic(), "", None
    try:
        host = LocalHostDuel(host_config(deal), core, clients, room)
        host.play(max_seconds=max_seconds)
        winner = host.winner
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]
    results = {role: clients[seat].result for role, seat in seats.items()}
    reports = {role: policies[role].report() for role in ROLES}
    if not error:
        ends = {(r.winner, r.win_reason, r.turns) for r in results.values()}
        if len(ends) != 1 or results["a"].winner != winner:
            error = "the clients disagree about the ending"
        elif results["a"].win_reason in (C.WIN_REASON_SURRENDER, C.WIN_REASON_DISCONNECT):
            error = "the game ended by " + C.WIN_REASONS[results["a"].win_reason]
        elif any(clients[seat].board.mismatches for seat in (0, 1)):
            error = "a client's public board disagreed with the host"
        elif any(report["failures"] for report in reports.values()):
            error = "a policy session failed: " + "; ".join(f for r in reports.values() for f in r["failures"])[:400]
    row = {"deal_seed": deal_seed, "a_first": a_first, "seconds": round(time.monotonic() - started, 1),
           "error": error}
    if not error:
        a = results["a"]
        row.update(winner="draw" if winner == 2 else ("a" if winner == seats["a"] else "b"),
                   win_reason=C.WIN_REASONS.get(a.win_reason, str(a.win_reason)), turns=a.turns, lp=list(a.lp),
                   decisions={role: reports[role]["decisions"] for role in ROLES})
    return row


def summarize(rows):
    """Player ``a``'s record moving first, moving second and in total; failures are counted apart."""
    def record(selected):
        done = [row for row in selected if not row["error"]]
        wins = sum(row["winner"] == "a" for row in done)
        draws = sum(row["winner"] == "draw" for row in done)
        out = {"games": len(done), "wins": wins, "losses": len(done) - wins - draws, "draws": draws}
        out["win_rate"] = round((wins + 0.5 * draws) / len(done), 4) if done else None
        return out
    return {"a_first": record([row for row in rows if row["a_first"]]),
            "a_second": record([row for row in rows if not row["a_first"]]),
            "total": record(rows), "failures": sum(bool(row["error"]) for row in rows)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--a", type=Path, required=True, help="Unix socket of player a's policy service")
    parser.add_argument("--b", type=Path, required=True, help="Unix socket of player b's policy service")
    parser.add_argument("--deck", type=Path, required=True, help="the .ydk both players use")
    parser.add_argument("--cards-db", type=Path, required=True)
    parser.add_argument("--script-root", type=Path, required=True, help="the directory that contains script/")
    parser.add_argument("--native", type=Path, required=True,
                        help="the environment extension; the host loads a copy of the core library next to it")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--game-seconds", type=float, default=1800.0, help="give up a game after this long")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for a service reply")
    parser.add_argument("--out", type=Path, required=True, help="a new output directory")
    args = parser.parse_args(argv)
    from mirrorforce.netduel.cards import CardPool
    from mirrorforce.worldmodel.engine import load_ydk
    from tools.mf_runtime_client_parity import host_core

    args.out.mkdir(parents=True, exist_ok=False)
    sockets = {"a": args.a, "b": args.b}
    identities = {role: service_identity(sockets[role], args.timeout) for role in ROLES}
    recipe, cards = load_ydk(args.deck), CardPool(args.cards_db)
    plan = schedule(args.games, args.seed)
    rows = []
    with tempfile.TemporaryDirectory(dir=args.out) as scratch:
        core = host_core(args.native, args.cards_db, args.script_root, scratch)
        with open(args.out / "games.jsonl", "w") as log:
            for game, deal_seed, a_first in plan:
                row = {"game": game, **play_game(core, cards, recipe, deal_seed, a_first, sockets, identities,
                                                 max_seconds=args.game_seconds)}
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print("GAME " + json.dumps(row), flush=True)
    summary = {"games": args.games, "seed": args.seed, "deck_sha256": sha256(args.deck),
               "cards_db_sha256": sha256(args.cards_db), "native_sha256": sha256(args.native),
               "services": {role: identities[role].get("checkpoint_sha256") for role in ROLES},
               "result": summarize(rows)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("SUMMARY " + json.dumps(summary["result"]), flush=True)
    return 0 if summary["result"]["failures"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
