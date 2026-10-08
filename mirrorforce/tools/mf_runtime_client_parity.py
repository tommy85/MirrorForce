"""Client-mode parity of the observation over many games (``mirrorforce.probes.client_parity``).

Plays mirror games of each ``--deck`` on ``ScriptedDuel`` under seeded choice rules, replays each game's responses
through the in-process network host on a distinct copy of the module's core, and checks, for both seats, that
``ClientDuel`` fed that seat's received stream shows the same menus and the same ``obs:``/``info:`` arrays, byte for
byte, at every decision, and sends the same responses. With ``--host-only`` it checks only the host replay (the
game ends the same way and every response has its seat), for a module without ``ClientDuel``.

Writes one content-addressed report: per game the deck, seed, rule, decisions and the first mismatches per seat; in
total the games checked, the games equal on both seats and, per array key, the decisions where it differed.

    MF_DUEL_NATIVE=<module> python mf_runtime_client_parity.py --cards-db <cdb> --code-list <list> \
        --script-root <dir with script/> --announce-tables <json> --deck <ydk> --games 200 --out-dir <dir>

CPU only, no model. Run it through the job runner on a shared host, in its own process (two core copies).
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

SCHEMA = "mirrorforce_client_parity/v1"
RULES = ("random", "active")


def load_native(cards_db, code_list, script_root, tables, cap):
    import importlib.util
    path = os.environ["MF_DUEL_NATIVE"]
    spec = importlib.util.spec_from_file_location("duel_native", path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    here = os.getcwd()
    os.chdir(script_root)  # init_module reads ./script
    try:
        native.init_module(str(cards_db), str(code_list), {})
    finally:
        os.chdir(here)
    from mirrorforce.agent.env.announce_law import register
    register(native, tables, cap)
    return native, Path(path)


def host_core(native_path, cards_db, script_root, scratch):
    from mirrorforce.puzzle.core import get_core
    copy = Path(scratch) / "libmfcore-host.so"
    shutil.copyfile(native_path.resolve().parent / "libmfcore.so", copy)  # a distinct file: its globals stay apart
    core = get_core(lib_path=copy, db_path=cards_db, script_dirs=[script_root / "script"], mode=ctypes.RTLD_LOCAL)
    if Path(core.lib_path).resolve() != copy.resolve():
        raise RuntimeError("another core is already loaded in this process")
    return core


def mirror_deal(recipe, seed):
    rng = random.Random(seed)
    orders = [list(recipe.main), list(recipe.main)]
    for order in orders:
        rng.shuffle(order)
    return {"seed_words": [rng.getrandbits(32) for _ in range(8)], "deck_orders": orders,
            "extra": [list(recipe.extra)] * 2, "start_lp": 8000, "start_hand": 5, "draw_count": 1,
            "duel_options": 5 << 16}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    from mirrorforce.probes import client_parity as V
    from mirrorforce.netduel import agent_public_recipe as PR
    from mirrorforce.worldmodel.engine import load_ydk

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cards-db", type=Path, required=True)
    parser.add_argument("--code-list", type=Path, required=True)
    parser.add_argument("--script-root", type=Path, required=True)
    parser.add_argument("--announce-tables", type=Path, required=True)
    parser.add_argument("--announce-cap", type=int, default=192)
    parser.add_argument("--config", default='{"max_options": 192}', help="the env configuration, as JSON")
    parser.add_argument("--client-config", help="the client builder's configuration, as JSON (default: --config); "
                        "env-only switches such as export_both_seats go in --config only")
    parser.add_argument("--opponent-mode", choices=("mirror",),
                        help="explicitly declare the input deck as both players' public recipe (this tool plays mirrors)")
    parser.add_argument("--deck", type=Path, action="append", default=[])
    parser.add_argument("--games", type=int, default=0, help="games per deck")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rules", default=",".join(RULES))
    parser.add_argument("--stop-at", type=int, default=3, help="mismatches kept per seat")
    parser.add_argument("--host-only", action="store_true")
    parser.add_argument("--save-digests", type=Path, help="also write every game's env record as digests (JSON lines)")
    parser.add_argument("--against-digests", type=Path, help="replay these digested games (another build's) on this "
                        "module and compare every decision's arrays, menus and the responses; no client check")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config)
    client_config = json.loads(args.client_config) if args.client_config else config
    if args.against_digests is None and not args.host_only:
        public = args.opponent_mode == "mirror"
        if config.get("public_opponent_recipe", False) != public \
                or client_config.get("public_opponent_recipe", False) != public:
            raise ValueError("env/client public_opponent_recipe must agree with the explicit --opponent-mode")
    rules = args.rules.split(",")
    if not rules or set(rules) - set(RULES) or (args.against_digests is None) == (args.games <= 0 or not args.deck):
        raise ValueError("decks and a game count, or --against-digests (not both)")
    native, native_path = load_native(args.cards_db.resolve(), args.code_list.resolve(), args.script_root.resolve(),
                                      args.announce_tables.resolve(), args.announce_cap)
    if not args.host_only and args.against_digests is None and not hasattr(native, "ClientDuel"):  # under the wrapper
        raise SystemExit("this duel_native build has no ClientDuel; use --host-only")
    scratch = tempfile.mkdtemp(prefix="parity-")
    core = host_core(native_path, args.cards_db.resolve(), args.script_root.resolve(), scratch)
    print("CONFIG-READBACK " + json.dumps({"config": config, "rules": rules, "games_per_deck": args.games,
          "decks": [str(d) for d in args.deck], "host_only": args.host_only, "seed": args.seed,
          "opponent_recipe_mode": args.opponent_mode}), flush=True)
    games, keys, info_keys, started = [], Counter(), Counter(), time.monotonic()
    card_view_msgs, card_view_fields, law = Counter(), Counter(), V.card_view_law(native)
    if args.against_digests is not None:
        for line in args.against_digests.read_text().splitlines():
            record = json.loads(line)
            row = {"seed": record.get("seed"), "deck": record.get("deck")}
            try:
                game = V.play_game(native, record["deal"], config, V.replay_rule(record))
                diff = V.compare_digests(record, V.digest_game(game))
                row.update(decisions=len(record["decisions"]), equal=not diff, differences=diff)
                for item in diff:
                    for key in item.get("keys", [item["kind"]]):
                        keys[key] += 1
            except Exception as exc:  # noqa: BLE001 - one row per game either way
                row["error"] = f"{type(exc).__name__}: {exc}"[:400]
            games.append(row)
        args.deck = []
    saved = args.save_digests.open("w") if args.save_digests is not None else None
    for deck in args.deck:
        recipe = load_ydk(deck)
        declared = PR.declare(recipe.main, recipe.extra) if args.opponent_mode == "mirror" else None
        for number in range(args.games):
            seed = args.seed * 1_000_003 + number
            rule = rules[number % len(rules)]
            row = {"deck": deck.name, "seed": seed, "rule": rule}
            try:
                choose = (V.random_rule if rule == "random" else V.active_rule)(seed)
                game = V.play_game(native, mirror_deal(recipe, seed), config, choose)
                streams = V.seat_streams(core, game)
                row.update(winner=game.winner, decisions=[len(d) for d in game.decisions], responses=len(game.responses))
                if saved is not None:
                    saved.write(json.dumps({"deck": deck.name, "seed": seed, **V.digest_game(game)}) + "\n")
                if not args.host_only:
                    seats = [V.check_seat(native, game, seat, streams[seat], client_config, stop_at=args.stop_at,
                                          law=law["law"], opponent_recipe=declared) for seat in (0, 1)]
                    row["public_opponent_recipe"] = declared
                    row["privileged_keys"] = sorted({k for d in game.decisions for x in d for k in x.arrays
                                                     if k.startswith(V.PRIVILEGED_PREFIX)})
                    row["card_view_decisions"] = sum(seat["card_view_decisions"] for seat in seats)
                    for seat in seats:
                        for item in seat["card_view"]:
                            card_view_msgs[item["msg"]] += 1
                            for entry in item["rows"]:
                                for field in entry["fields"]:
                                    card_view_fields[field] += 1
                    row["seats"] = seats
                    row["equal"] = all(seat["equal"] for seat in seats)
                    for seat in seats:
                        for mismatch in seat["mismatches"]:
                            for key in mismatch.get("keys", {}) or [mismatch["kind"]]:
                                keys[key] += 1
                        for key, value in seat["info_differences"].items():
                            info_keys[key] += value["decisions"]
            except Exception as exc:  # noqa: BLE001 - one row per game either way
                row["error"] = f"{type(exc).__name__}: {exc}"[:400]
            games.append(row)
            print("PARITY-GAME " + json.dumps({k: row.get(k) for k in ("deck", "seed", "rule", "decisions", "equal",
                                                                       "error")}), flush=True)
    if saved is not None:
        saved.close()
    report = {"schema": SCHEMA, "native_sha256": sha256(native_path), "client_config": client_config,
              "opponent_recipe_mode": args.opponent_mode,
              "privileged_keys": sorted({k for g in games for k in g.get("privileged_keys", [])}),
              "against_digests_sha256": sha256(args.against_digests) if args.against_digests else None,
              "core_sha256": sha256(native_path.resolve().parent / "libmfcore.so"),
              "cards_db_sha256": sha256(args.cards_db), "code_list_sha256": sha256(args.code_list),
              "announce_tables_sha256": sha256(args.announce_tables), "config": config, "rules": rules,
              "decks": {d.name: sha256(d) for d in args.deck}, "host_only": args.host_only,
              "games": len(games), "errors": sum("error" in g for g in games),
              "equal_games": None if args.host_only and args.against_digests is None
              else sum(bool(g.get("equal")) for g in games),
              "decisions": sum(sum(d) if isinstance(d, list) else d for d in (g.get("decisions", 0) for g in games)),
              "mismatch_keys": dict(keys.most_common()),
              "info_difference_decisions": dict(info_keys.most_common()),
              "card_view_law": law,
              "card_view": {"decisions": sum(g.get("card_view_decisions", 0) for g in games),
                            "client_decisions": sum(sum(g["decisions"]) for g in games
                                                    if isinstance(g.get("decisions"), list) and "seats" in g),
                            "by_msg": {str(m): n for m, n in card_view_msgs.most_common()},
                            "by_field": dict(card_view_fields.most_common())},
              "seconds": round(time.monotonic() - started, 1),
              "rows": games, "training_eligible": False}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(report, indent=1, sort_keys=True).encode()
    target = args.out_dir / f"client-parity-{hashlib.sha256(body).hexdigest()}.json"
    target.write_bytes(body)
    print("PARITY-REPORT " + json.dumps({k: report[k] for k in ("games", "errors", "equal_games", "decisions",
                                                              "mismatch_keys", "info_difference_decisions",
                                                              "card_view", "seconds")}
                                        | {"report": str(target)}),
          flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    return 0 if report["errors"] == 0 and (report["equal_games"] is None or report["equal_games"] == len(games)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
