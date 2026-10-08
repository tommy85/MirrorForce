"""A training run's room format and card tables, through the trainer's own entry points: ``make_env`` writes the
format's index and era into every observation's ``global_`` columns 23 and 24, and the checkpoint identity records
the format and the card tables, which must match the run's code list and semantics file.

Runs in the venv with ``MF_DUEL_NATIVE`` naming a built module and the MD80 assets under ``MF_TEST_ASSETS``; run
it in its own pytest process (the announce law is process-global).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ASSETS = Path(os.environ.get("MF_TEST_ASSETS", "/path/to/mirrorforce/infra/m1-20261001/assets-md80/project"))
HERE = Path(__file__).resolve().parent
TABLES = next((HERE / "fixtures" / "announce").glob("announce-tables-*.json"))
ROOM_FORMATS = next((HERE / "fixtures" / "room_formats").glob("room-formats-*.json"))
# the MD80 run's card tables (tools/mf_runtime_card_tables.py; npz files are not tracked)
CARD_TABLES = Path("/path/to/mirrorforce/infra/card-tables/"
                   "card-tables-b3ccc747e6950081c334c54bbdb458c3ec071dc6f51ec2af089ee56ec65f3968.npz")
FRIEND_SEMANTICS = Path("/path/to/workspace/friend-ygo-agent-20260920/source/pretrain-project/assets/structured/"
                        "frozen_semantics_v1.npz")


@pytest.fixture(scope="module")
def run():
    if not os.environ.get("MF_DUEL_NATIVE") or not (ASSETS / "cards.cdb").is_file() or not CARD_TABLES.is_file():
        pytest.skip("set MF_DUEL_NATIVE and MF_TEST_ASSETS (and build the MD80 card tables)")
    os.chdir(ASSETS)  # the module reads ./script
    import tyro
    from mirrorforce.agent.train import cleanba
    from mirrorforce.agent.utils import init_duel

    def args_for(*extra):
        args = tyro.cli(cleanba.Args, args=[
            "--deck", str(ASSETS / "decks"), "--allow-unreviewed-public-effects", "--code-list-file", str(ASSETS / "code_list.txt"),
            "--cards-db", str(ASSETS / "cards.cdb"), "--announce-tables", str(TABLES), "--max-options", "192",
            "--seed", "5", "--local-num-envs", "4", "--local-env-threads", "2", "--semantic-file",
            str(ASSETS / "frozen_semantics.npz"), *extra])
        if not deck:
            deck.append(init_duel(args.env_id, "english", args.deck, args.code_list_file, db_path=args.cards_db))
        args.deck1 = args.deck2 = deck[0]
        args.semantic_shape = None
        return args

    deck = []
    return cleanba, args_for


def test_the_run_format_fills_every_observation(run):
    cleanba, args_for = run
    args = args_for("--room-format", "md-2026H2", "--room-format-table", str(ROOM_FORMATS))
    cleanba.checkpoint_identity(args)  # registers the announce law
    envs = cleanba.make_env(args, args.seed, args.local_num_envs, args.local_env_threads)
    obs, info = envs.reset()
    rng = np.random.default_rng(5)
    for _ in range(60):
        assert obs["global_"].shape == (4, 25)
        assert (obs["global_"][:, 23] == 2).all() and (obs["global_"][:, 24] == 43).all()
        obs, _, _, _, info = envs.step(rng.integers(np.asarray(info["num_options"])).astype(np.int32))


def test_the_identity_records_the_format_and_the_card_tables(run):
    cleanba, args_for = run
    identity = cleanba.checkpoint_identity(args_for("--room-format", "md-2026H2", "--room-format-table",
                                                    str(ROOM_FORMATS), "--card-tables", str(CARD_TABLES)))
    assert identity["room_format"]["key"] == "md-2026H2" and identity["room_format"]["era"] == 43
    assert identity["card_tables"]["sha256"] == CARD_TABLES.name[len("card-tables-"):-4]
    bare = cleanba.checkpoint_identity(args_for())
    assert "room_format" not in bare and "card_tables" not in bare
    with pytest.raises(ValueError, match="go together"):
        cleanba.checkpoint_identity(args_for("--room-format", "md-2026H2"))
    if FRIEND_SEMANTICS.is_file():
        with pytest.raises(ValueError, match="another semantics file"):
            args = args_for("--card-tables", str(CARD_TABLES))
            args.semantic_file = str(FRIEND_SEMANTICS)
            cleanba.checkpoint_identity(args)


def test_card_tables_need_a_model_that_declares_them(run):
    from mirrorforce.agent.env import card_tables
    tables, _, _ = card_tables.load(CARD_TABLES, ASSETS / "code_list.txt")
    with pytest.raises(ValueError, match="found 0"):
        card_tables.inject({"constants": {"cdb_exact": np.zeros((2, 2))}}, tables)
    declared = {name: np.zeros(array.shape, array.dtype) for name, array in tables.items()}
    tree = {"constants": {"semantics": declared}}
    card_tables.inject(tree, tables)
    assert all(np.array_equal(np.asarray(tree["constants"]["semantics"][k]), tables[k]) for k in tables)
    declared["genericity"] = np.zeros((3, 1), np.float32)
    with pytest.raises(ValueError, match="the model declares"):
        card_tables.inject({"constants": {"semantics": declared}}, tables)
