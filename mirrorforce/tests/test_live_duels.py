"""Many live duels in one process (core mfsnap registry; env DuelEnvImpl::checked_duel).

The core's arena registry once held 256 live duels and enforced nothing: duel 257 and later ran outside an arena, and
every arena query called them invalid (an A0 smoke of 1,536 envs died at its first step). The registry now holds
``native.live_duel_capacity`` (4096) live duels and refuses a creation beyond it, which the env turns into a fatal
error. Here more than 256 duels are alive at once and each one snapshots, steps, restores and is queried.
``MF_LIVE_DUELS`` sets the count (300 by default; about 3 MB of arena each).

Runs in the venv with ``MF_DUEL_NATIVE``; run it in its own pytest process.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import random
import sqlite3

import pytest

RUN = Path(os.environ.get("MF_SCRIPTED_RUN", "/path/to/workspace/ygopro-client-run"))
TABLES = next((Path(__file__).resolve().parent / "fixtures" / "announce").glob("announce-tables-*.json"))
DECK = Path(__file__).resolve().parents[1] / "decks/stage-a/SkyStriker.ydk"
COUNT = int(os.environ.get("MF_LIVE_DUELS", "300"))


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    path = os.environ.get("MF_DUEL_NATIVE")
    if not path or not (RUN / "cards.cdb").is_file():
        pytest.skip("set MF_DUEL_NATIVE to a built duel_native module (and have the card database and scripts)")
    spec = importlib.util.spec_from_file_location("duel_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with sqlite3.connect(f"file:{RUN / 'cards.cdb'}?mode=ro", uri=True) as conn:
        codes = sorted(row[0] for row in conn.execute("SELECT id FROM datas"))
    code_list = tmp_path_factory.mktemp("live-duels") / "code_list.txt"
    code_list.write_text("".join(f"{code} {int((RUN / 'script' / f'c{code}.lua').is_file())}\n" for code in codes))
    here = os.getcwd()
    os.chdir(RUN)
    try:
        module.init_module(str(RUN / "cards.cdb"), str(code_list), {})
    finally:
        os.chdir(here)
    from mirrorforce.agent.env.announce_law import register
    register(module, TABLES, 192)
    return module


def deal(seed):
    from mirrorforce.worldmodel.engine import load_ydk
    recipe = load_ydk(DECK)
    rng = random.Random(seed)
    orders = [list(recipe.main), list(recipe.main)]
    for order in orders:
        rng.shuffle(order)
    return {"seed_words": [rng.getrandbits(32) for _ in range(8)], "deck_orders": orders,
            "extra": [list(recipe.extra)] * 2, "start_lp": 8000, "start_hand": 5, "draw_count": 1,
            "duel_options": 5 << 16}


def test_more_than_256_duels_live_at_once_each_snapshot_restore_and_query(native):
    assert native.live_duel_capacity >= 4096
    assert COUNT > 256
    duels = []
    for seed in range(COUNT):
        duel = native.SearchDuel(deal(seed), {"max_options": 192, "max_steps": 1000}, keep_history=False)
        duel.start()
        duels.append(duel)
    rng = random.Random(0)
    for seed, duel in enumerate(duels):
        before = duel.prompt()
        obs = duel.observation()
        snapshot = duel.take()
        for _ in range(3):
            prompt = duel.prompt()
            if prompt is None:
                break
            duel.step(rng.randrange(len(prompt[2])))
        duel.restore(snapshot)
        assert duel.prompt() == before, seed
        again = duel.observation()
        for key in obs:
            assert (obs[key] == again[key]).all(), (seed, key)
        assert duel.effect_info(), seed  # an arena query answers for every live duel
