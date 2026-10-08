"""The both-seat export (config export_both_seats; duel_env.h write_other_seat): critic inputs under priv:.

- Off (the default): every priv: key has zero rows and the observation has none.
- On: the acting seat's obs: arrays are byte-identical to the same game's without the export, at every decision (so
  the policy's inputs keep every public-information property; test_search_api.py also runs its two-truth test
  with the export on), and the observation never carries a priv: key.
- The export is the other seat's own observation: where nothing reaches that seat between a decision and its own next
  decision (no fact of its stream, only hints and a prompt that refreshes nothing), every priv: array equals the
  obs: array it then gets -- history windows, chunks and closed turns included, which the export does not consume.

Runs in the venv with ``MF_DUEL_NATIVE``; run it in its own pytest process.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import random
import sqlite3

import numpy as np
import pytest

RUN = Path(os.environ.get("MF_SCRIPTED_RUN", "/path/to/workspace/ygopro-client-run"))
TABLES = next((Path(__file__).resolve().parent / "fixtures" / "announce").glob("announce-tables-*.json"))
A0 = Path(__file__).resolve().parents[1] / "decks/stage-a/SkyStriker.ydk"
CONFIG = {"max_options": 192}
ON = {**CONFIG, "export_both_seats": 1}
MSG_HINT, MSG_SELECT_BATTLECMD, MSG_SELECT_IDLECMD = 2, 10, 11


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
    code_list = tmp_path_factory.mktemp("both-seats") / "code_list.txt"
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
    recipe = load_ydk(A0)
    rng = random.Random(seed)
    orders = [list(recipe.main), list(recipe.main)]
    for order in orders:
        rng.shuffle(order)
    return {"seed_words": [rng.getrandbits(32) for _ in range(8)], "deck_orders": orders,
            "extra": [list(recipe.extra)] * 2, "start_lp": 8000, "start_hand": 5, "draw_count": 1,
            "duel_options": 5 << 16}


def test_off_the_priv_keys_have_zero_rows(native):
    duel = native.ScriptedDuel(deal(1), CONFIG)
    duel.start()
    labels = duel.labels()
    priv = [key for key in labels if key.startswith("priv:")]
    assert len(priv) == 21 and all(labels[key].shape[0] == 0 for key in priv)
    assert not any(key.startswith("priv:") for key in duel.observation())


def test_on_the_acting_seats_observation_is_unchanged(native):
    for seed in (2, 3):
        off, on = native.ScriptedDuel(deal(seed), CONFIG), native.ScriptedDuel(deal(seed), ON)
        off.start()
        on.start()
        rng, decisions = random.Random(seed), 0
        while (prompt := off.prompt()) is not None:
            assert on.prompt() == prompt
            a, b = off.observation(), on.observation()
            assert a.keys() == b.keys() and not any(key.startswith("priv:") for key in b)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key], err_msg=f"{key} seed {seed} decision {decisions}")
            assert all(value.shape[0] > 0 for key, value in on.labels().items() if key.startswith("priv:"))
            choice = rng.randrange(len(prompt[2]))
            off.step(choice)
            on.step(choice)
            decisions += 1
        assert on.prompt() is None and on.winner == off.winner and decisions > 100


def test_the_export_is_the_other_seats_next_observation(native):
    compared = 0
    for seed in range(4, 64):
        duel = native.ScriptedDuel(deal(seed), ON)
        duel.start()
        rng = random.Random(seed)
        pending = {}  # seat -> (its export at the other seat's decision, its fact count, the message count then)
        while (prompt := duel.prompt()) is not None:
            seat = prompt[0]
            obs, labels = duel.observation(), duel.labels()
            messages = duel.messages()
            if seat in pending:
                export, facts, at = pending.pop(seat)
                between = messages[at:]
                quiet = (len(duel.facts(seat)) == facts and all(m == MSG_HINT for m, _ in between[:-1])
                         and between and between[-1][0] not in (MSG_SELECT_IDLECMD, MSG_SELECT_BATTLECMD))
                if quiet:
                    for key, value in export.items():
                        np.testing.assert_array_equal(value, obs["obs:" + key[len("priv:"):]],
                                                      err_msg=f"{key} seed {seed} seat {seat}")
                    compared += 1
            other = 1 - seat
            pending[other] = ({k: v.copy() for k, v in labels.items() if k.startswith("priv:")},
                              len(duel.facts(other)), len(messages))
            duel.step(rng.randrange(len(prompt[2])))
    assert compared >= 40, compared


def test_a_client_refuses_the_export(native):
    from mirrorforce.worldmodel.engine import load_ydk
    recipe = load_ydk(A0)
    with pytest.raises(RuntimeError, match="export_both_seats"):
        native.ClientDuel(0, sorted(recipe.main), list(recipe.extra), dict(ON))
