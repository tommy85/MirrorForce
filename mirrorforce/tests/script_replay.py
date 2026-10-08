"""Replays of game records (tests/fixtures/repro/script-*.json) against a chosen card-script tree, each in a
subprocess (the card table and announce law are process-global; the scripts are the working directory's ./script).

``replay`` returns, per decision, the prompt (player, message, menu rows as (act, code)), the row taken, the
``info:illegal_activation`` of the observation after the step and the message count before it, the error that ended the replay if any, the repro
record's menu indices and withdrawals, and the final prompt and message count. It can stop the record's actions at
the first prompt offering a given row (taking it), then keep answering row 0 while the prompt is one of
``repeat_msgs``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIXTURES = sorted((HERE / "fixtures" / "repro").glob("script-*.json"))
PATCHES = sorted((ROOT / "patches").glob("ygopro-scripts-*.patch"))
MD80 = Path("/path/to/mirrorforce/infra/m1-20261001/assets-md80/project")
FRIEND = Path("/path/to/workspace/friend-ygo-agent-20260920/source/pretrain-project")
POOLS = {
    "md80": {"scripts": MD80 / "script", "cdb": MD80 / "cards.cdb", "codes": MD80 / "code_list.txt",
             "tables": HERE / "fixtures" / "announce" /
             "announce-tables-013c330623ae5332c0337bfde2b727cb099c58c5f3a0e7eb8bf3b576dd826009.json"},
    "friend": {"scripts": FRIEND / "script", "cdb": FRIEND / "assets" / "locale" / "en" / "cards.cdb",
               "codes": FRIEND / "scripts" / "code_list.txt",
               "tables": ROOT / "infra" / "announce-tables" /
               "announce-tables-2b4fa8a17647a315ea159c0056402f9376f3be4e2d6230e7f8a592d876c96cb6.json"},
}
ACTIVATE, PASS = 8, 9

RUNNER = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from mirrorforce.agent.env.announce_law import register
from mirrorforce.agent.env.duel import native
cdb, codes, tables = sys.argv[2:5]
spec = json.loads(sys.argv[5])
native.init_module(cdb, codes, {})
register(native, tables, 192)
rec = spec["record"]
deal = {"seed_words": [rec["seed"]], "deck_orders": rec["main_decks"], "extra": rec["extra_decks"], "start_lp": 8000,
        "start_hand": 5, "draw_count": 1, "duel_options": 5 << 16}
duel = native.ScriptedDuel(deal, {"max_options": 192, "max_steps": 1000})
duel.start()
decisions, error = [], None
def menu(prompt):
    return [prompt[0], prompt[1], [[row.get("act"), row.get("code")] for row in prompt[2]]]
def take(action):
    prompt = duel.prompt()
    duel.observation()
    entry = menu(prompt)
    if action >= len(prompt[2]):
        return False
    before = duel.message_count()
    duel.step(int(action))
    info = [int(v) for v in duel.observation()["info:illegal_activation"].reshape(-1)] if duel.prompt() else None
    decisions.append(entry + [int(action), info, before])
    return True
try:
    for action in spec["actions"]:
        prompt = duel.prompt()
        if prompt is None:
            break
        until = spec.get("until")
        if until and until in menu(prompt)[2]:
            take(menu(prompt)[2].index(until))
            break
        if not take(action):
            break
    for _ in range(spec.get("repeat", 0)):
        prompt = duel.prompt()
        if prompt is None or prompt[1] not in spec["repeat_msgs"]:
            break
        take(0)
except RuntimeError as e:
    error = str(e)
record = json.loads(duel.repro_record("replay"))
final = duel.prompt()
print(json.dumps({"decisions": decisions, "error": error, "actions": record["actions"],
                  "withdrawn": record["withdrawn"], "final": menu(final) if final else None,
                  "messages": duel.message_count()}))
"""


def fixture(path) -> dict:
    return json.loads(Path(path).read_text())


def replay(pool: str, scripts: Path, workdir: Path, actions, record: dict, repeat: int = 0, repeat_msgs=(),
           until=None) -> dict:
    """The record's deal stepped with ``actions`` -- or, with ``until`` (a menu row as [act, code]), up to the first
    prompt offering that row, where it is taken -- then row 0 while the prompt is in ``repeat_msgs`` (at most
    ``repeat`` times), under ``scripts`` as ./script."""
    workdir.mkdir(parents=True, exist_ok=True)
    if not (workdir / "script").exists():
        (workdir / "script").symlink_to(scripts)
    p = POOLS[pool]
    spec = {"record": record, "actions": list(actions), "repeat": repeat, "repeat_msgs": list(repeat_msgs),
            "until": list(until) if until else None}
    done = subprocess.run([sys.executable, "-c", RUNNER, str(ROOT), str(p["cdb"]), str(p["codes"]), str(p["tables"]),
                           json.dumps(spec)], cwd=workdir, capture_output=True, text=True, timeout=900,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])
