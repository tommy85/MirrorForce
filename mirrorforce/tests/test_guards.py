"""The no-progress guards of the env (design item 4; cxx/duelpool/duel/guards.h), ported from the structured runtime.

Runs in the venv with ``MF_DUEL_NATIVE`` naming a built module. ``guards_simulate`` runs the guards over a given
sequence of prompts and choices, so the laws are checked on the loop shapes retained
(the design notes): a quick effect re-activated at the same open chain
window (Galaxy-Eyes), a command menu activation (D/D/D Zeus), a resolution's yes/cancel lap (Yummy Surprise) and a
command repeated with nothing public in between; each is withheld at its first repeat, windows reset, a menu is never
emptied and only ACTIVATE rows leave in the activation scope. In the env, random games are unchanged by the guards and
every prompt's shown menu is the full menu less the rows ``info:guard`` counts.
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
BATTLECMD, IDLECMD, YESNO, SELECT_CARD, SELECT_CHAIN = 10, 11, 13, 15, 16
BATTLE, MAIN1 = 0x08, 0x04


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    path = os.environ.get("MF_DUEL_NATIVE")
    if not path:
        pytest.skip("set MF_DUEL_NATIVE to a built duel_native module")
    spec = importlib.util.spec_from_file_location("duel_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def step(msg, key, activate, choice, *, player=0, turn=5, phase=BATTLE, links=0, resolution=0, facts=0):
    return {"player": player, "msg": msg, "turn": turn, "phase": phase, "links": links, "resolution": resolution,
            "facts": facts, "key": key, "activate": activate, "choice": choice}


def withheld(native, steps):
    return [sorted(set(r["no_progress"]) | set(r["cycle"])) for r in native.guards_simulate(steps)]


def test_the_laws_are_exported(native):
    assert tuple(native.guard_laws) == ("own_view_no_progress_exclusion/v1", "no_progress_command_cycle_exclusion/v2",
                                        "illegal_activation_withdrawal/v1")


def test_an_open_chain_window_activation_loop_is_withheld_at_its_first_repeat(native):
    # Galaxy-Eyes: at an open chain window (activate, do not activate) the player activates; the chain resolves and
    # the same own view comes back; the activation is withheld, "do not activate" stays
    lap = [step(SELECT_CHAIN, "K", [True, False], 0), step(SELECT_CHAIN, "K", [True, False], 1)]
    assert withheld(native, lap) == [[], [0]]
    # the opponent's prompts and the player's sub-selections between the two do not matter
    between = [step(SELECT_CHAIN, "K", [True, False], 0),
               step(SELECT_CARD, "T", [False, False], 1, links=1),
               step(SELECT_CHAIN, "O", [True, False], 1, player=1, links=1),
               step(SELECT_CHAIN, "K", [True, False], 1)]
    assert withheld(native, between)[-1] == [0]
    # a different own view (the effect changed something) is progress
    assert withheld(native, [step(SELECT_CHAIN, "K", [True, False], 0), step(SELECT_CHAIN, "K2", [True, False], 0),
                             step(SELECT_CHAIN, "K", [True, False], 0)]) == [[], [], []]
    # declining is never withheld; a pending chain is out of scope
    assert withheld(native, [step(SELECT_CHAIN, "K", [True, False], 1), step(SELECT_CHAIN, "K", [True, False], 1)]) \
        == [[], []]
    assert withheld(native, [step(SELECT_CHAIN, "K", [True, False], 0, links=1),
                             step(SELECT_CHAIN, "K", [True, False], 0, links=1)]) == [[], []]


def test_a_command_menu_activation_loop_is_withheld_and_the_window_resets(native):
    # D/D/D Sky King Zeus Ragnarok in the main phase: (activate, summon, to battle)
    menu = [True, False, False]
    seq = [step(IDLECMD, "M", menu, 0, phase=MAIN1, facts=1), step(IDLECMD, "M", menu, 0, phase=MAIN1, facts=2),
           step(IDLECMD, "M", menu, 2, phase=MAIN1, facts=3)]
    assert withheld(native, seq) == [[], [0], [0]]
    # a new phase or turn is a new window
    assert withheld(native, [step(IDLECMD, "M", menu, 0, phase=MAIN1, facts=1),
                             step(IDLECMD, "M", menu, 0, phase=0x200, facts=2)]) == [[], []]
    assert withheld(native, [step(IDLECMD, "M", menu, 0, phase=MAIN1, facts=1),
                             step(IDLECMD, "M", menu, 0, turn=7, phase=MAIN1, facts=2)]) == [[], []]


def test_a_resolution_lap_is_withheld_at_its_first_repeat(native):
    # Yummy Surprise: yes/no -> yes, cancelable card selection -> cancel, back to the same yes/no: "yes" is withheld
    # (any row, in this scope), so the player answers "no" and the resolution ends
    seq = [step(YESNO, "Q", [False, False], 0, resolution=7, phase=MAIN1),
           step(SELECT_CARD, "C", [False, False], 1, resolution=7, phase=MAIN1),
           step(YESNO, "Q", [False, False], 1, resolution=7, phase=MAIN1)]
    assert withheld(native, seq) == [[], [], [0]]
    # a key met again keeps every row last chosen at it withheld for the rest of the resolution
    seq += [step(SELECT_CARD, "C", [False, False], 0, resolution=7, phase=MAIN1),
            step(SELECT_CARD, "C", [False, False, False], 2, resolution=7, phase=MAIN1)]
    assert withheld(native, seq)[3:] == [[1], [0, 1]]
    # the next resolution is a new window
    assert withheld(native, [step(YESNO, "Q", [False, False], 0, resolution=7, phase=MAIN1),
                             step(YESNO, "Q", [False, False], 0, resolution=8, phase=MAIN1)]) == [[], []]


def test_a_command_repeated_with_nothing_public_between_is_withheld(native):
    # an attack whose target prompt was cancelled: back at the same battle menu, no public fact in between
    menu = [False, False, False]  # (attack, to main 2, to end phase)
    seq = [step(BATTLECMD, "B", menu, 0, facts=40), step(BATTLECMD, "B", menu, 0, facts=40)]
    assert [r["cycle"] for r in native.guards_simulate(seq)] == [[], [0]]
    # a public fact in between forgets the tried commands
    assert withheld(native, [step(BATTLECMD, "B", menu, 0, facts=40), step(BATTLECMD, "B", menu, 0, facts=41)]) \
        == [[], []]


def test_a_menu_is_never_emptied(native):
    seq = [step(SELECT_CHAIN, "K", [True, True], 0), step(SELECT_CHAIN, "K", [True, True], 1),
           step(SELECT_CHAIN, "K", [True, True], 0)]
    rows = native.guards_simulate(seq)
    assert [sorted(r["no_progress"]) for r in rows] == [[], [0], []] and rows[2]["kept_menu"]


@pytest.fixture(scope="module")
def env(native, tmp_path_factory):
    if not (RUN / "cards.cdb").is_file():
        pytest.skip("needs the card database and scripts")
    with sqlite3.connect(f"file:{RUN / 'cards.cdb'}?mode=ro", uri=True) as conn:
        codes = sorted(row[0] for row in conn.execute("SELECT id FROM datas"))
    code_list = tmp_path_factory.mktemp("guards") / "code_list.txt"
    code_list.write_text("".join(f"{code} {int((RUN / 'script' / f'c{code}.lua').is_file())}\n" for code in codes))
    here = os.getcwd()
    os.chdir(RUN)
    try:
        native.init_module(str(RUN / "cards.cdb"), str(code_list), {})
    finally:
        os.chdir(here)
    from mirrorforce.agent.env.announce_law import register
    register(native, TABLES, 192)
    return native


def test_the_shown_menu_is_the_full_menu_less_the_withheld_rows(env):
    from mirrorforce.worldmodel.engine import load_ydk
    guarded = 0
    for deck, seed in (("SkyStriker", 1), ("RyzealMitsurugi", 2), ("KewlTune", 3)):
        recipe = load_ydk(DECKS / f"{deck}.ydk")
        rng = random.Random(seed)
        orders = [list(recipe.main), list(recipe.main)]
        for order in orders:
            rng.shuffle(order)
        duel = env.ScriptedDuel({"seed_words": [rng.getrandbits(32) for _ in range(8)], "deck_orders": orders,
                                 "extra": [list(recipe.extra)] * 2, "start_lp": 8000, "start_hand": 5,
                                 "draw_count": 1, "duel_options": 5 << 16}, {"max_options": 192})
        duel.start()
        while (prompt := duel.prompt()) is not None:
            obs = duel.observation()
            shown, guard = int(obs["info:num_options"]), list(obs["info:guard"])
            assert shown == len(prompt[2]) >= 1
            guarded += guard[0] + guard[1]
            duel.step(rng.randrange(shown))
    assert guarded == 0  # random play on these decks never repeats an own view after an activation
