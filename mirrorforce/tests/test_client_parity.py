"""Client-mode observation parity (``probes.client_parity``): Sky Striker mirror games on ``ScriptedDuel`` against
``ClientDuel`` fed each seat's received stream from the in-process host.

Runs in the venv (Python 3.11) with ``MF_DUEL_NATIVE`` naming a built module; the ClientDuel checks skip until the
module has ``ClientDuel``. Loads a second core copy through ctypes: run this file in its own pytest process.
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import random
import shutil
import sqlite3

import pytest

from mirrorforce.probes import client_parity as V
from mirrorforce.probes.client_streams import PROMPTS

RUN = Path("/path/to/workspace/ygopro-client-run")
TABLES = next((Path(__file__).resolve().parent / "fixtures" / "announce").glob("announce-tables-*.json"))
DECKS = {"pinned": Path(__file__).resolve().parents[1] / "decks/stage-a/SkyStriker.ydk",
         "meta": Path(__file__).resolve().parents[1] / "decks/meta-2026-08/SkyStriker.ydk"}
CONFIG = {"max_options": 192}


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    path = os.environ.get("MF_DUEL_NATIVE")
    if not path or not (RUN / "cards.cdb").is_file() or not (RUN / "script").is_dir():
        pytest.skip("set MF_DUEL_NATIVE to a built duel_native module (and have the card database and scripts)")
    spec = importlib.util.spec_from_file_location("duel_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with sqlite3.connect(f"file:{RUN / 'cards.cdb'}?mode=ro", uri=True) as conn:
        codes = sorted(row[0] for row in conn.execute("SELECT id FROM datas"))
    code_list = tmp_path_factory.mktemp("parity") / "code_list.txt"
    code_list.write_text("".join(f"{code} {int((RUN / 'script' / f'c{code}.lua').is_file())}\n" for code in codes))
    here = os.getcwd()
    os.chdir(RUN)  # init_module reads ./script
    try:
        module.init_module(str(RUN / "cards.cdb"), str(code_list), {})
    finally:
        os.chdir(here)
    from mirrorforce.agent.env.announce_law import register
    register(module, TABLES, 192)
    return module


@pytest.fixture(scope="module")
def host_core(native, tmp_path_factory):
    from mirrorforce.puzzle.core import get_core
    module_core = Path(os.environ["MF_DUEL_NATIVE"]).resolve().parent / "libmfcore.so"
    path = tmp_path_factory.mktemp("parity-host") / "libmfcore-host.so"
    shutil.copyfile(module_core, path)  # a distinct file: its globals (script and card readers) stay apart
    core = get_core(lib_path=path, db_path=RUN / "cards.cdb", script_dirs=[RUN / "script"], mode=ctypes.RTLD_LOCAL)
    assert Path(core.lib_path).resolve() == path.resolve(), "run this test file in its own pytest process"
    return core


def mirror_deal(deck, seed):
    from mirrorforce.worldmodel.engine import load_ydk
    recipe = load_ydk(DECKS[deck])
    rng = random.Random(seed)
    orders = [list(recipe.main), list(recipe.main)]
    for order in orders:
        rng.shuffle(order)
    return {"seed_words": [rng.getrandbits(32) for _ in range(8)], "deck_orders": orders,
            "extra": [list(recipe.extra)] * 2, "start_lp": 8000, "start_hand": 5, "draw_count": 1,
            "duel_options": 5 << 16}


@pytest.mark.parametrize("deck,seed,rule", [("pinned", 1, "random"), ("pinned", 2, "active"), ("meta", 3, "active")])
def test_the_host_replay_gives_each_seat_its_prompts(native, host_core, deck, seed, rule):
    """Plumbing: the host replays the env's game to the same end, every response has its seat, and each seat's env
    decisions fit within the prompts it received (a decision is a prompt or a sub-choice of one; forced rows are not)."""
    choose = (V.random_rule if rule == "random" else V.active_rule)(seed)
    game = V.play_game(native, mirror_deal(deck, seed), CONFIG, choose)
    streams = V.seat_streams(host_core, game)
    assert len(game.owners) == len(game.responses) and set(game.owners) <= {0, 1}
    for seat in (0, 1):
        prompts = sum(1 for msg, _ in streams[seat] if msg in PROMPTS)
        assert prompts == game.owners.count(seat)
        assert game.decisions[seat], f"seat {seat} made no decision"


@pytest.mark.parametrize("deck,seed,rule", [("pinned", 11, "random"), ("pinned", 12, "active"), ("pinned", 13, "active"),
                                            ("meta", 14, "random")])
def test_the_client_builder_sees_what_the_env_shows(native, host_core, deck, seed, rule):
    if not hasattr(native, "ClientDuel") or importlib.util.find_spec("mirrorforce.netduel.agent_client") is None:
        pytest.skip("this build has no ClientDuel, or the client wrapper (netduel.agent_client) is not on main yet")
    choose = (V.random_rule if rule == "random" else V.active_rule)(seed)
    game = V.play_game(native, mirror_deal(deck, seed), CONFIG, choose)
    streams = V.seat_streams(host_core, game)
    # the meta list holds cards outside the public effect table (public_effects/v1): a client refuses them unless allowed
    client_config = {**CONFIG, "allow_unreviewed_public_effects": 1} if deck == "meta" else CONFIG
    for seat in (0, 1):
        report = V.check_seat(native, game, seat, streams[seat], client_config)
        assert report["equal"], report["mismatches"]
        assert report["checked_decisions"] == len(game.decisions[seat])
