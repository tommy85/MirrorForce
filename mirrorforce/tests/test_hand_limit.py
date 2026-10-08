"""The End Phase hand limit in the observation (cxx/duelpool/duel/duel_env.h; core query_hand_limit):

- ``obs:hand_limit_`` [2, 4], rows own and opponent: valid, the limit the End Phase applies (the last
  EFFECT_HAND_LIMIT affecting the player, else 6), the hand size, and the excess max(0, hand - limit);
- ``obs:action_discard_`` [max_options]: on the command menus' move to the End Phase, the cards the End Phase would
  discard if the turn ended now (clipped at 15); 0 on every other row. It follows the shown menu.

A seeded set of games (A0's Sky Striker mirror and the meta decks; no card of theirs changes the limit; in some games
one player ends every turn at once, so its hand passes the limit) checks every observation against the card rows; a
turn ended with an excess leads to the End Phase's discard prompt. Both keys are covered by the generic two-truth tests (test_search_api.py, test_scripted_duels.py), which
compare every observation key. Runs in the venv with ``MF_DUEL_NATIVE``, on the card database and scripts the
scripted fixtures use; run it in its own pytest process (the announce law is process-global).
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
DECKS = Path(__file__).resolve().parents[1] / "decks/meta-2026-08"
STAGE_A = Path(__file__).resolve().parents[1] / "decks/stage-a/SkyStriker.ydk"
NAMES = ["Branded", "Elfnote", "KewlTune", "RyzealMitsurugi", "SkyStriker", "Toon"]
SELECT_CARD, IDLE, BATTLE, NEW_TURN = 15, 11, 10, 40
PHASE_END = 3  # ActionPhase::End
HAND = 2  # obs:cards_ column 2: location id of a hand card


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
    code_list = tmp_path_factory.mktemp("hand-limit") / "code_list.txt"
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
    rng = random.Random(seed)
    if seed % 2:
        first = second = load_ydk(STAGE_A)
    else:
        first, second = (load_ydk(DECKS / f"{rng.choice(NAMES)}.ydk") for _ in range(2))
    orders = [list(first.main), list(second.main)]
    for order in orders:
        rng.shuffle(order)
    return {"seed_words": [rng.getrandbits(32) for _ in range(8)], "deck_orders": orders,
            "extra": [list(first.extra), list(second.extra)], "start_lp": 8000, "start_hand": 5, "draw_count": 1,
            "duel_options": 5 << 16}


def hand_rows(cards, side):
    """Hand cards of a side (0 own, 1 opponent) in obs:cards_ (column 2 location id, column 4 side)."""
    rows = cards[(cards[:, 2] == HAND) & (cards[:, 4] == side)]
    return len(rows)


def test_the_hand_limit_rows_and_the_end_phase_discard(native):
    observed = ended_with_excess = discarded = 0
    for seed in range(1, 61):
        duel = native.ScriptedDuel(deal(seed), {"max_options": 192, "max_steps": 1000})
        duel.start()
        rng = random.Random(seed)
        pending = None  # (player, excess) after a move to the End Phase with an excess, until its discard prompt
        while (prompt := duel.prompt()) is not None:
            player, msg, rows = prompt
            obs = duel.observation()
            limit = obs["obs:hand_limit_"]
            cards = obs["obs:cards_"]
            for side in (0, 1):
                hand = hand_rows(cards, side)
                assert list(limit[side]) == [1, 6, hand, max(hand - 6, 0)], (seed, side, list(limit[side]), hand)
            discard = obs["obs:action_discard_"]
            shown = int(obs["info:num_options"])
            assert not discard[shown:].any()
            excess = int(limit[0, 3])
            for i, row in enumerate(rows):
                ends = msg in (IDLE, BATTLE) and row["phase"] == PHASE_END
                assert discard[i] == (min(excess, 15) if ends else 0), (seed, msg, i, row)
            observed += 1
            if pending and player == pending[0] and msg == SELECT_CARD:
                discarded += 1
                pending = None
            # player 0 of every other game holds its cards: it ends each turn at once, so its hand grows past the limit
            choice = rng.randrange(len(rows))
            passive = seed % 4 < 2 and player == 0
            if msg in (IDLE, BATTLE) and any(r["phase"] == PHASE_END for r in rows) and (passive or rng.random() < 0.3):
                choice = next(i for i, r in enumerate(rows) if r["phase"] == PHASE_END)
                if excess:
                    ended_with_excess += 1
                    pending = (player, excess)
            duel.step(choice)
    assert observed > 10000 and ended_with_excess >= 2 and discarded == ended_with_excess, (observed, ended_with_excess, discarded)
