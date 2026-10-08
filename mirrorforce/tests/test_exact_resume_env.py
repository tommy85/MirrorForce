"""Exact resume of the env pool (``export_states`` / ``import_states``): a fresh pool that imports another pool's
export continues identically -- the same observations, rewards, dones and infos for the same actions -- including
games in progress (replayed from their records) and games that finish and restart after the import; a tampered
export is refused.

Runs in the venv with ``MF_DUEL_NATIVE`` naming a built module and the MD80 assets under ``MF_TEST_ASSETS``
(the project directory with decks/, code_list.txt, cards.cdb and script/).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ASSETS = Path(os.environ.get("MF_TEST_ASSETS", "/path/to/mirrorforce/infra/m1-20261001/assets-md80/project"))
TABLES = next((Path(__file__).resolve().parent / "fixtures" / "announce").glob("announce-tables-*.json"))


@pytest.fixture(scope="module")
def make():
    if not os.environ.get("MF_DUEL_NATIVE") or not (ASSETS / "cards.cdb").is_file():
        pytest.skip("set MF_DUEL_NATIVE and MF_TEST_ASSETS")
    os.chdir(ASSETS)  # the module reads ./script
    import tyro
    from mirrorforce.agent.train import cleanba
    from mirrorforce.agent.env.announce_law import register_from_args
    from mirrorforce.agent.utils import init_duel
    args = tyro.cli(cleanba.Args, args=[
        "--deck", str(ASSETS / "decks"), "--allow-unreviewed-public-effects", "--code-list-file", str(ASSETS / "code_list.txt"),
        "--cards-db", str(ASSETS / "cards.cdb"), "--announce-tables", str(TABLES), "--deck-schedule",
        "cluster_uniform", "--max-options", "192", "--max-steps", "1000", "--seed", "77", "--local-num-envs", "6",
        "--local-env-threads", "3"])
    deck = init_duel(args.env_id, "english", args.deck, args.code_list_file, db_path=args.cards_db)
    register_from_args(args)
    args.deck1 = args.deck2 = deck
    return lambda: cleanba.make_env(args, args.seed, args.local_num_envs, args.local_env_threads)


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            if key in ("step_time",):  # wall-clock timings
                continue
            same(a[key], b[key])
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_an_imported_pool_continues_identically(make):
    rng = np.random.default_rng(3)
    first = make()
    obs, info = first.reset()
    for _ in range(int(rng.integers(150, 250))):  # some games finish, others are in progress
        obs, _, _, _, info = first.step(rng.integers(np.asarray(info["num_options"])).astype(np.int32))
    states = first.export_states()
    assert any(s.split("\n")[6].split()[0] == "1" for s in states)  # games in progress are carried
    second = make()
    second.reset()
    second.import_states(states)
    actions_rng = np.random.default_rng(4)
    for _ in range(400):  # long enough that every game finishes and new ones start
        actions = actions_rng.integers(np.asarray(info["num_options"])).astype(np.int32)
        a = first.step(actions)
        b = second.step(actions)
        for x, y in zip(a, b):
            same(x, y)
        info = a[4]


def test_a_tampered_export_is_refused(make):
    rng = np.random.default_rng(5)
    pool = make()
    _, info = pool.reset()
    for _ in range(40):
        _, _, _, _, info = pool.step(rng.integers(np.asarray(info["num_options"])).astype(np.int32))
    states = pool.export_states()
    i = next(i for i, s in enumerate(states) if s.split("\n")[6].split()[0] == "1" and int(s.split("\n")[6].split()[2]) > 3)
    lines = states[i].split("\n")
    fields = lines[6].split()
    fields[3] = str((int(fields[3]) + 1) % 2)  # a different first menu index
    lines[6] = " ".join(fields)
    states[i] = "\n".join(lines)
    other = make()
    other.reset()
    with pytest.raises(Exception):
        other.import_states(states)
