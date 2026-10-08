"""the deck and card features in the observation (the MT input set, the design notes):

- ``obs:own_recipe_``: the viewer's own recipe, one row per distinct (location, card id) with its count, the copies
  remaining (exact against the engine's deck and face-down extra deck) and the archetype share;
- ``obs:global_`` columns 23 and 24: the run's room format index and era (``mirrorforce/agent/env/room_format.py``);
- history rows write description index 15 for a description that belongs to another card (the effect-unit evidence
  is per card row and ordinal);
- the card tables file (``mirrorforce/agent/env/card_tables.py``): deterministic, content-addressed, bound to its code
  list and semantics, with the setcode bag and link arrows, deck-type genericity and the effect-unit evidence.

Each observed feature also holds under a second truth behind the same public stream (another hidden order of the
opponent's cards, another order of the viewer's own deck). Runs in the venv with ``MF_DUEL_NATIVE``, on the card
database and scripts the scripted fixtures use; run it in its own pytest process (the announce law is process-global).
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sqlite3

import numpy as np
import pytest

RUN = Path(os.environ.get("MF_SCRIPTED_RUN", "/path/to/workspace/ygopro-client-run"))
HERE = Path(__file__).resolve().parent
TABLES = next((HERE / "fixtures" / "announce").glob("announce-tables-*.json"))
ROOM_FORMATS = next((HERE / "fixtures" / "room_formats").glob("room-formats-*.json"))
DECKS = Path(__file__).resolve().parents[1] / "decks/meta-2026-08"
MD80 = Path("/path/to/mirrorforce/infra/m1-20261001/assets-md80/project")
MD80_TABLES = Path("/path/to/mirrorforce/infra/announce-tables/"
                   "announce-tables-013c330623ae5332c0337bfde2b727cb099c58c5f3a0e7eb8bf3b576dd826009.json")
MSG_CHAINING = 70


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
    code_list = tmp_path_factory.mktemp("deck-features") / "code_list.txt"
    code_list.write_text("".join(f"{code} {int((RUN / 'script' / f'c{code}.lua').is_file())}\n" for code in codes))
    here = os.getcwd()
    os.chdir(RUN)
    try:
        module.init_module(str(RUN / "cards.cdb"), str(code_list), {})
    finally:
        os.chdir(here)
    from mirrorforce.agent.env.announce_law import register
    register(module, TABLES, 192)
    module.card_ids = {code: i for i, code in enumerate(codes, start=1)}
    return module


@pytest.fixture(scope="module")
def cards():
    with sqlite3.connect(f"file:{RUN / 'cards.cdb'}?mode=ro", uri=True) as conn:
        return {code: (alias, setcode) for code, alias, setcode in conn.execute("SELECT id, alias, setcode FROM datas")}


def deal(seed, deck="SkyStriker", other=None):
    from mirrorforce.worldmodel.engine import load_ydk
    first, second = load_ydk(DECKS / f"{deck}.ydk"), load_ydk(DECKS / f"{other or deck}.ydk")
    rng = random.Random(seed)
    orders = [list(first.main), list(second.main)]
    for order in orders:
        rng.shuffle(order)
    return {"seed_words": [rng.getrandbits(32) for _ in range(8)], "deck_orders": orders,
            "extra": [list(first.extra), list(second.extra)], "start_lp": 8000, "start_hand": 5, "draw_count": 1,
            "duel_options": 5 << 16}


def rows_of(table):
    return [tuple(int(v) for v in row) for row in table if row[2]]


def counted(rows, location):
    return Counter({(r[0] << 8) | r[1]: r[4] for r in rows if r[2] == location})


def bases(cards, code):
    alias, _ = cards[code]
    if alias and code != 5405695 and alias < code + 20 and code < alias + 20:
        code = alias
    setcode = cards[code][1]
    return {(setcode >> (16 * slot)) & 0xFFF for slot in range(4)} - {0}


@pytest.mark.parametrize("deck,other,seed", [("SkyStriker", None, 1), ("RyzealMitsurugi", "KewlTune", 2),
                                             ("Toon", "Elfnote", 3)])
def test_own_recipe_rows_hold_the_recipe_and_engine_exact_remaining_counts(native, cards, deck, other, seed):
    d = deal(seed, deck, other)
    duel = native.SearchDuel(d, {"max_options": 192}, keep_history=True)
    duel.start()
    rng, decisions, facedown = random.Random(seed), 0, None
    while (prompt := duel.prompt()) is not None and decisions < 600:
        viewer = prompt[0]
        obs, priv = duel.observation(), duel.privileged()["priv:cards_"]
        assert list(duel.unseen_own(viewer)) == [{}, {}]  # every own card outside the deck is seen: rows are the engine's
        rows = rows_of(obs["obs:own_recipe_"])
        recipe = Counter((1, native.card_ids[c]) for c in d["deck_orders"][viewer])
        recipe += Counter((7, native.card_ids[c]) for c in d["extra"][viewer])
        assert [(r[2], (r[0] << 8) | r[1]) for r in rows] == sorted(recipe)  # (location, id) order
        assert {(r[2], (r[0] << 8) | r[1]): r[3] for r in rows} == recipe
        ids = priv[:, 0].astype(int) * 256 + priv[:, 1]
        own = priv[:, 4] == 0
        deck_rows = own & (priv[:, 2] == 1)
        if facedown is None and deck_rows.any():
            facedown = int(priv[deck_rows, 5][0])
        engine_deck = Counter(ids[deck_rows].tolist())
        engine_extra = Counter(ids[own & (priv[:, 2] == 7) & (priv[:, 5] == facedown)].tolist())
        assert counted(rows, 1) == +Counter({i: engine_deck[i] for i in counted(rows, 1)})
        assert counted(rows, 7) == +Counter({i: engine_extra[i] for i in counted(rows, 7)})
        cards_ = obs["obs:cards_"]
        shown = Counter((cards_[(cards_[:, 2] == 1) & (cards_[:, 4] == 0), 0].astype(int) * 256
                         + cards_[(cards_[:, 2] == 1) & (cards_[:, 4] == 0), 1]).tolist())
        assert +counted(rows, 1) == +Counter({i: n for i, n in shown.items() if i in counted(rows, 1)})
        main = Counter(d["deck_orders"][viewer])
        total = sum(main.values())
        for code in set(d["deck_orders"][viewer]) | set(d["extra"][viewer]):
            shared = sum(n for other_code, n in main.items() if bases(cards, code) & bases(cards, other_code))
            share = math.floor(255.0 * shared / total + 0.5)
            row = next(r for r in rows if (r[0] << 8) | r[1] == native.card_ids[code])
            assert row[5] == share, code
        duel.step(rng.randrange(len(prompt[2])))
        decisions += 1
    assert decisions > 50


def test_room_format_and_era_fill_the_last_global_columns(native):
    for config, expected in (({}, (0, 0)), ({"room_format": 2, "room_era": 43}, (2, 43))):
        duel = native.SearchDuel(deal(4), {"max_options": 192, **config}, keep_history=True)
        duel.start()
        rng, seen = random.Random(4), 0
        while (prompt := duel.prompt()) is not None and seen < 100:
            g = duel.observation()["obs:global_"]
            assert g.shape == (25,) and tuple(int(v) for v in g[23:]) == expected
            duel.step(rng.randrange(len(prompt[2])))
            seen += 1
    for broken in ({"room_format": 256}, {"room_era": -1}):
        with pytest.raises(RuntimeError, match="outside 0..255"):
            native.SearchDuel(deal(4), {"max_options": 192, **broken}, keep_history=True)


def test_the_room_format_resolves_from_the_registered_table(tmp_path):
    from mirrorforce.agent.env import room_format
    md = room_format.resolve(ROOM_FORMATS, "md-2026H2")
    assert (md["index"], md["era"], md["era_law"]) == (2, 43, "quarters_since_2016q1/v1")
    assert room_format.resolve(ROOM_FORMATS, "md-2022H1")["era"] == 25
    assert room_format.resolve(ROOM_FORMATS, "unknown")["index"] == 0
    assert room_format.resolve(ROOM_FORMATS, "unrestricted")["era"] == 0
    assert room_format.env_config(None) == {"room_format": 0, "room_era": 0}
    with pytest.raises(ValueError, match="not in"):
        room_format.resolve(ROOM_FORMATS, "md-2099H1")
    renamed = tmp_path / ("room-formats-" + "0" * 64 + ".json")
    renamed.write_bytes(ROOM_FORMATS.read_bytes())
    with pytest.raises(ValueError, match="do not match"):
        room_format.read_table(renamed)


def test_deck_features_are_the_same_under_another_truth(native):
    d = deal(5)
    duel = native.SearchDuel(d, {"max_options": 192, "room_format": 2, "room_era": 43}, keep_history=True)
    duel.start()
    rng = random.Random(5)
    for _ in range(12):
        duel.step(rng.randrange(len(duel.prompt()[2])))
    viewer = duel.prompt()[0]
    opponent = 1 - viewer
    keys = ("obs:own_recipe_", "obs:global_")
    before = {k: duel.observation()[k].copy() for k in keys}
    layout = duel.hidden_layout(opponent)
    duel.permute_hidden(opponent, viewer=viewer, hand=list(reversed(layout["hand"])),
                        deck=list(reversed(layout["deck"])), facedown=layout["facedown"],
                        extra=list(reversed(layout["extra"])))
    assert duel.hidden_layout(opponent)["deck"] != layout["deck"]
    assert all(np.array_equal(duel.observation()[k], before[k]) for k in keys)
    own = duel.hidden_layout(viewer)["deck"]
    duel.reshuffle_future(99)  # another order of the viewer's own deck
    assert duel.hidden_layout(viewer)["deck"] != own
    assert all(np.array_equal(duel.observation()[k], before[k]) for k in keys)


def test_a_description_of_another_card_writes_index_15(native, cards):
    code = 63288573  # Sky Striker Ace - Kagari
    assert native.own_desc_index(code, code * 16 + 3) == 3
    assert native.own_desc_index(code, 50005218 * 16 + 3) == 15
    assert native.own_desc_index(code, 1160) == 15  # a system string
    variant, base = next((c, a) for c, (a, _) in sorted(cards.items())
                         if a and c != 5405695 and a < c + 20 and c < a + 20)
    assert native.own_desc_index(variant, base * 16 + 2) == 2
    assert native.own_desc_index(base, variant * 16 + 2) == 15


@pytest.mark.parametrize("deck,other,seed", [("SkyStriker", None, 6), ("RyzealMitsurugi", "KewlTune", 7)])
def test_chain_rows_write_each_description_by_the_card_rule(native, deck, other, seed):
    duel = native.SearchDuel(deal(seed, deck, other), {"max_options": 192}, keep_history=True)
    duel.start()
    rng, links, seen, checked = random.Random(seed), {}, 0, 0
    while (prompt := duel.prompt()) is not None:
        for msg, payload in duel.messages_since(seen):
            if msg == MSG_CHAINING:
                payload = bytes(payload)
                links[payload[15]] = (int.from_bytes(payload[0:4], "little"), int.from_bytes(payload[11:15], "little"))
            seen += 1
        chain = duel.observation()["obs:chain_"]
        for i, row in enumerate(chain):
            if row[0]:
                code, desc = links[i + 1]
                assert (int(row[2]) << 8 | int(row[3])) == native.card_ids[code]
                assert row[4] == native.own_desc_index(code, desc)
                checked += 1
        duel.step(rng.randrange(len(prompt[2])))
    assert checked > 0


def table_inputs(tmp_path, codes):
    """A small run: a code list of ``codes``, its frozen semantics (the builder over the MD80 assets' scripts) and a
    two-recipe announce library."""
    from mirrorforce.agent.semantics import build_structured_semantics as builder
    code_list = tmp_path / "code_list.txt"
    code_list.write_text("".join(f"{c} 1\n" for c in codes))
    cdb_exact, lua_effects, lua_mask, metadata = builder.build_tables(MD80 / "cards.cdb", MD80 / "script", codes, 16, 64)
    semantics = tmp_path / "semantics.npz"
    np.savez_compressed(semantics, cdb_exact=cdb_exact, lua_effects=lua_effects, lua_mask=lua_mask,
                        metadata=np.asarray(json.dumps(metadata)))
    library = [{"main": [codes[0], codes[0], codes[1]], "extra": [], "cluster": "a", "deck_type": "A", "format": "x"},
               {"main": [codes[1], codes[2]], "extra": [], "cluster": "b", "deck_type": "B", "format": "x"},
               {"main": [codes[1]], "extra": [], "cluster": "c", "deck_type": "B", "format": "x"}]
    raw = json.dumps({"schema": "mirrorforce_announce_tables/v1", "library": library, "staples": [],
                      "threshold": 0.1}).encode()
    tables = tmp_path / f"announce-tables-{hashlib.sha256(raw).hexdigest()}.json"
    tables.write_bytes(raw)
    return code_list, semantics, tables


def test_card_tables_are_deterministic_bound_and_carry_the_laws(tmp_path):
    if not (MD80 / "cards.cdb").is_file():
        pytest.skip("needs the MD80 pool assets (cards.cdb, scripts)")
    from mirrorforce.common import card_exact
    from mirrorforce.agent.env import card_tables
    from tools import mf_runtime_card_tables as T
    # an old-style script (one whole-script unit), a link monster, a modern script whose descriptions bind units
    kagari, goddess, modern = 63288573, 98127546, 20508881
    codes = sorted((kagari, goddess, modern))
    code_list, semantics, tables = table_inputs(tmp_path, codes)
    arrays, metadata = T.build(code_list, MD80 / "cards.cdb", MD80 / "script", semantics, tables)
    first, sha = card_tables.write(arrays, metadata, tmp_path / "out")
    again, _ = card_tables.encode(*T.build(code_list, MD80 / "cards.cdb", MD80 / "script", semantics, tables))
    assert first.read_bytes() == again and first.name == f"card-tables-{sha}.npz"
    loaded, meta, digest = card_tables.load(first, code_list)
    assert digest == sha and set(loaded) == set(card_tables.MODEL_ARRAYS)
    exact = card_exact.build_table(MD80 / "cards.cdb")
    for row, code in enumerate(codes, start=1):
        vector = exact.row(code)
        offset = card_exact.SEGMENT_OFFSETS["setcodes"]
        assert np.array_equal(loaded["setcode_bag"][row], vector[offset:offset + 32])
        offset = card_exact.SEGMENT_OFFSETS["link_arrows"]
        assert np.array_equal(loaded["link_arrows"][row], vector[offset:offset + 8])
    assert loaded["link_arrows"][codes.index(goddess) + 1].any()
    # deck types A and B: codes[0] in A only, codes[1] in both, codes[2] in B only
    assert loaded["genericity"][1:, 0].tolist() == [0.5, 1.0, 0.5] and loaded["genericity"][0, 0] == 0
    assert meta["library"]["deck_types"] == 2
    with np.load(semantics) as payload:
        lua_mask = payload["lua_mask"]
    bits = loaded["effect_unit_bits"]
    assert bits[0].sum() == 0 and not bits[codes.index(kagari) + 1].any()
    assert bits[codes.index(modern) + 1, :2].tolist() == [1, 2]  # its descriptions 0 and 1: units 0 and 1
    for row in range(1, len(codes) + 1):
        for ordinal in range(14):
            units = [u for u in range(16) if bits[row, ordinal] >> u & 1]
            assert all(lua_mask[row, u] for u in units)
    # refusals: another code list, a renamed or edited file, a semantics file of other scripts
    other = tmp_path / "other_list.txt"
    other.write_text("".join(f"{c} 1\n" for c in reversed(codes)))
    with pytest.raises(ValueError, match="another code list"):
        card_tables.load(first, other)
    renamed = first.with_name("card-tables-" + "0" * 64 + ".npz")
    renamed.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="do not match"):
        card_tables.load(renamed, code_list)
    with np.load(semantics) as payload:
        meta_sem = json.loads(str(payload["metadata"].item()))
        stale = {name: payload[name] for name in ("cdb_exact", "lua_effects", "lua_mask")}
    meta_sem["script_manifest_sha256"] = "0" * 64
    np.savez_compressed(semantics, metadata=np.asarray(json.dumps(meta_sem)), **stale)
    with pytest.raises(ValueError, match="scripts differ"):
        T.build(code_list, MD80 / "cards.cdb", MD80 / "script", semantics, tables)
    with pytest.raises(ValueError, match="not in the code list"):
        bad = json.loads(tables.read_bytes())
        bad["library"][0]["main"].append(46986414)
        raw = json.dumps(bad).encode()
        path = tmp_path / f"announce-tables-{hashlib.sha256(raw).hexdigest()}.json"
        path.write_bytes(raw)
        meta_sem["script_manifest_sha256"] = metadata["semantics"]["script_manifest_sha256"]
        np.savez_compressed(semantics, metadata=np.asarray(json.dumps(meta_sem)), **stale)
        T.build(code_list, MD80 / "cards.cdb", MD80 / "script", semantics, path)


def test_the_pool_card_tables_load_against_their_runs():
    """The two pools' registered tables (infra/card-tables, built by tools/mf_runtime_card_tables.py; npz files
    are not tracked) load against their code lists and name the semantics files the runs read."""
    friend = Path("/path/to/workspace/friend-ygo-agent-20260920/source/pretrain-project")
    pools = [(MD80 / "code_list.txt", MD80 / "frozen_semantics.npz"),
             (friend / "scripts" / "code_list.txt", friend / "assets" / "structured" / "frozen_semantics_v1.npz")]
    built = sorted(Path("/path/to/mirrorforce/infra/card-tables").glob("card-tables-*.npz"))
    if len(built) < 2 or not all(c.is_file() and s.is_file() for c, s in pools):
        pytest.skip("needs the pool assets and their built card tables")
    from mirrorforce.agent.env import card_tables
    for code_list, semantics in pools:
        sha = hashlib.sha256(semantics.read_bytes()).hexdigest()
        matches = []
        for path in built:
            try:
                _, meta, _ = card_tables.load(path, code_list)
            except ValueError:
                continue
            matches.append(meta["semantics"]["sha256"] == sha)
        assert matches == [True], code_list


def test_the_terminal_flag_stays_column_22_behind_the_new_columns():
    """global_ grew from 23 to 25 columns: readers of the decision-less flag name column 22, so a room era in the last
    column never makes a decision look terminal (or a terminal one look like a decision)."""
    from mirrorforce.agent.model.decision import public_outcome_targets
    g = np.zeros((2, 25), np.uint8)
    g[:, 23], g[:, 24] = 2, 43
    g[1, 22] = 1  # the second observation has no decision
    _, valid = public_outcome_targets({"global_": g}, {"global_": g.copy()}, np.zeros(2, bool))
    assert valid.tolist() == [True, False]
