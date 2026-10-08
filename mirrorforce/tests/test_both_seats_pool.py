"""The both-seat export on the env pool path the trainer runs (cleanba.make_env with --export-both-seats).

Two pools on the same seeds, one with the export: every step's observation (each obs: array), reward, done and info
other than the export are byte-identical; the export arrives in info["priv"], each key shaped as its observation key
(zero rows in the pool without it), carrying the other seat's view (its own rows differ from the acting seat's). The
two-truth property of the observation with the export on is tested on the same writer in test_search_api.py
(test_two_truths_with_the_both_seat_export_on); here the observation equals the export-off pool's byte for byte.

Runs in the venv with ``MF_DUEL_NATIVE`` and the MD80 assets under ``MF_TEST_ASSETS``; its own pytest process.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ASSETS = Path(os.environ.get("MF_TEST_ASSETS", "/path/to/mirrorforce/infra/m1-20261001/assets-md80/project"))
TABLES = next((Path(__file__).resolve().parent / "fixtures" / "announce").glob("announce-tables-*.json"))
NUM_ENVS = 6


@pytest.fixture(scope="module")
def make():
    if not os.environ.get("MF_DUEL_NATIVE") or not (ASSETS / "cards.cdb").is_file():
        pytest.skip("set MF_DUEL_NATIVE and MF_TEST_ASSETS")
    os.chdir(ASSETS)  # the module reads ./script
    import tyro
    from mirrorforce.agent.train import cleanba
    from mirrorforce.agent.env.announce_law import register_from_args
    from mirrorforce.agent.utils import init_duel

    def args_for(export):
        return tyro.cli(cleanba.Args, args=[
            "--deck", str(ASSETS / "decks"), "--allow-unreviewed-public-effects",
            "--code-list-file", str(ASSETS / "code_list.txt"), "--cards-db", str(ASSETS / "cards.cdb"),
            "--announce-tables", str(TABLES), "--deck-schedule", "cluster_uniform", "--max-options", "192",
            "--max-steps", "1000", "--seed", "91", "--local-num-envs", str(NUM_ENVS), "--local-env-threads", "3",
            *(["--export-both-seats"] if export else [])])

    first = args_for(False)
    deck = init_duel(first.env_id, "english", first.deck, first.code_list_file, db_path=first.cards_db)
    register_from_args(first)

    def make_pool(export):
        args = args_for(export)
        args.deck1 = args.deck2 = deck
        return cleanba.make_env(args, args.seed, args.local_num_envs, args.local_env_threads)

    return make_pool


def same(a, b, where):
    if isinstance(a, tuple):
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{where}[{i}]")
        return
    if isinstance(a, dict):
        assert a.keys() == b.keys(), where
        for key in a:
            if key != "step_time":  # wall-clock timings
                same(a[key], b[key], f"{where}.{key}")
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=where)


def test_the_export_leaves_the_pool_observation_unchanged_and_arrives_in_info(make):
    off, on = make(False), make(True)
    obs_off, info_off = off.reset()
    obs_on, info_on = on.reset()
    rng = np.random.default_rng(7)
    other_rows = 0
    for step in range(300):  # games finish and restart within it
        priv_off, priv_on = info_off.pop("priv"), info_on.pop("priv")
        same(obs_off, obs_on, f"obs at step {step}")
        same(info_off, info_on, f"info at step {step}")
        assert priv_off.keys() == priv_on.keys() and len(priv_on) == 21
        for key, value in priv_on.items():
            expected = np.asarray(obs_on[key]).shape
            assert np.asarray(value).shape == expected, (key, np.asarray(value).shape, expected)
            assert np.asarray(priv_off[key]).shape == (NUM_ENVS, 0, *expected[2:]), key
        cards, export = np.asarray(obs_on["cards_"]), np.asarray(priv_on["cards_"])
        # an env at a decision exports the other seat's view (a finished game's terminal state exports nothing):
        # written, and not the acting seat's rows
        for i in range(NUM_ENVS):
            if export[i].any():
                assert not np.array_equal(cards[i], export[i]), (step, i)
                other_rows += 1
        actions = rng.integers(np.asarray(info_on["num_options"])).astype(np.int32)
        obs_off, reward_off, term_off, trunc_off, info_off = off.step(actions)
        obs_on, reward_on, term_on, trunc_on, info_on = on.step(actions)
        same((reward_off, term_off, trunc_off), (reward_on, term_on, trunc_on), f"step {step}")
    assert other_rows > 0.9 * 300 * NUM_ENVS, other_rows
