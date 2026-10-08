"""The announce law (``duel_native.announce_candidates``, cxx/duelpool/duel/announce_law.h) equals its oracle.

Runs in the venv (Python 3.11) with ``MF_DUEL_NATIVE`` naming a built module. The module's card table is
initialized from a code list over the card database (no scripts are read: every line has no script), and the oracle
(``announce_law_oracle``) reads the same database and the same card ids. Every fuzz case, the synthetic edge cases
and the refusals must agree exactly: candidates in order, tier counts, truncation counts and branch.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sqlite3

import pytest

import announce_fuzz
import announce_law_oracle as L

CARDS_DB = Path(os.environ.get("MF_CARDS_DB", "/path/to/workspace/ygopro-client-run/cards.cdb"))
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "announce"


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    path = os.environ.get("MF_DUEL_NATIVE")
    if not path or not CARDS_DB.is_file():
        pytest.skip("set MF_DUEL_NATIVE to a built duel_native module (and have the card database)")
    spec = importlib.util.spec_from_file_location("duel_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with sqlite3.connect(f"file:{CARDS_DB}?mode=ro", uri=True) as conn:
        codes = sorted(row[0] for row in conn.execute("SELECT id FROM datas"))
    code_list = tmp_path_factory.mktemp("announce-law") / "code_list.txt"
    code_list.write_text("".join(f"{code} 0\n" for code in codes))
    module.init_module(str(CARDS_DB), str(code_list), {})
    return module, {code: i + 1 for i, code in enumerate(codes)}


@pytest.fixture(scope="module")
def tables():
    path = next(FIXTURES.glob("announce-tables-*.json"))
    return json.loads(path.read_text())


def both(native, cards, case, staples, library, cap):
    module, ids = native
    args = dict(seen_opponent=case["seen_opponent"], seen_own=case["seen_own"], own_main=case["own_main"],
                own_extra=case["own_extra"], room_format=case["room_format"], cap=cap)
    expected = L.candidates(case["opcodes"], staples=staples, library=library, cards=cards, card_ids=ids, **args)
    actual = module.announce_candidates(case["opcodes"], staples=sorted(staples.items()), library=library, **args)
    return expected, actual


def test_the_native_law_equals_the_oracle_on_fuzz_cases(native, tables):
    cards = L.load_cards(CARDS_DB)
    staples = {code: count for code, count in tables["staples"]}
    for case in announce_fuzz.cases(20261001, 400, tables["library"], cards):
        for cap in (128, 4096):
            expected, actual = both(native, cards, case, staples, tables["library"], cap)
            assert actual == expected, case["index"]


def test_the_native_law_equals_the_oracle_on_edge_cases(native, tables):
    cards = L.load_cards(CARDS_DB)
    staples = {code: count for code, count in tables["staples"]}
    empty = {"seen_opponent": [], "seen_own": [], "own_main": [], "own_extra": [], "room_format": None}
    # The empty union: no tier yields a card, so every declarable card in card id order, cut at the cap.
    expected, actual = both(native, cards, {**empty, "opcodes": [L.TYPE_MONSTER, L.OPCODE_ISTYPE]}, {}, [], 5)
    assert actual == expected and actual["branch"] == "empty_union" and actual["truncated"]["empty_union"] > 0
    # Nothing declarable at all.
    expected, actual = both(native, cards, {**empty, "opcodes": [1, L.OPCODE_ISCODE]}, {}, [], 5)
    assert actual == expected and actual["candidates"] == []
    # Tiers 1-4 over the cap refuse on both sides.
    crowded = {**empty, "opcodes": [L.TYPE_MONSTER, L.OPCODE_ISTYPE], "own_main": tables["library"][0]["main"]}
    with pytest.raises(ValueError, match="tiers 1-4"):
        L.candidates(crowded["opcodes"], staples=staples, library=[], cards=cards, card_ids=native[1], cap=3,
                     **{k: crowded[k] for k in ("seen_opponent", "seen_own", "own_main", "own_extra", "room_format")})
    with pytest.raises(RuntimeError, match="tiers 1-4"):
        native[0].announce_candidates(crowded["opcodes"], staples=sorted(staples.items()), library=[], cap=3,
                                      **{k: crowded[k] for k in ("seen_opponent", "seen_own", "own_main", "own_extra",
                                                                  "room_format")})
