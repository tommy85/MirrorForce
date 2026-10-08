"""Write a run's card tables (``mirrorforce/agent/env/card_tables.py``): setcode bag, link arrows, deck-type genericity
and effect-unit evidence, row-aligned with the run's code list and bound to its card database, frozen semantics,
scripts and announce tables. The output file is named by its bytes; rebuilding from the same inputs gives the same file.

The arrays come from the exact card channel (``common/card_exact.py``) and B1's evidence law
(``cardrules/static_extras.py``, through the unit split of the semantics builder); the tree itself imports neither.

Usage: ``--code-list <code_list.txt> --cards-db <cards.cdb> --scripts <script dir> --semantics <frozen_semantics.npz>
--announce-tables <announce-tables-<sha>.json> --out <dir>``; prints the file, its digest and the evidence counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

from mirrorforce.common import card_exact
from mirrorforce.cardrules import static_extras
from mirrorforce.agent.env import announce_law, card_tables as T
from mirrorforce.agent.semantics import build_structured_semantics as builder


def script_manifest(codes, scripts, aliases):
    """The semantics builder's script manifest (``build_tables``' ``script_manifest_sha256``) recomputed."""
    h = hashlib.sha256()
    for code in codes:
        path, script_code = builder.resolve_script_path(Path(scripts), code, aliases)
        if path is None or script_code is None:
            h.update(f"{code}:missing\n".encode())
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        h.update(f"{code}:{script_code}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}\n".encode())
    return h.hexdigest()


def genericity(library, codes):
    """[N, 1] deck-type document frequency over a library (the announce tables' recipes) and its description: per
    card, the share of deck types with at least one recipe holding it, main or extra (the announce staples' count)."""
    rows = {code: i for i, code in enumerate(codes, start=1)}
    holders, types = {}, set()
    for recipe in library:
        types.add(recipe["deck_type"])
        for code in set(recipe["main"]) | set(recipe["extra"]):
            if int(code) not in rows:
                raise ValueError(f"library card {code} is not in the code list")
            holders.setdefault(int(code), set()).add(recipe["deck_type"])
    if not types:
        raise ValueError("an empty library")
    out = np.zeros((len(codes) + 1, 1), np.float32)
    for code, kinds in holders.items():
        out[rows[code], 0] = len(kinds) / len(types)
    return out, {"law": T.LIBRARY_LAW, "recipes": len(library), "deck_types": len(types), "cards": len(holders)}


def build(code_list, cards_db, scripts, semantics, announce_tables):
    """The arrays (name -> array) and metadata of a run's card tables."""
    codes = T.read_code_list(code_list)
    with np.load(semantics, allow_pickle=False) as payload:
        meta = json.loads(str(payload["metadata"].item()))
        lua_mask = np.asarray(payload["lua_mask"])
    if meta.get("code_list_sha256") != T.code_list_digest(codes):
        raise ValueError("the semantics file was built for another code list")
    if lua_mask.shape != (len(codes) + 1, T.MAX_UNITS) or meta.get("max_effects") != T.MAX_UNITS:
        raise ValueError(f"the semantics file has effect slots {lua_mask.shape}, the tables need {T.MAX_UNITS}")
    connection = sqlite3.connect(f"file:{Path(cards_db).resolve()}?mode=ro&immutable=1", uri=True)
    try:
        aliases = {int(code): int(alias) for code, alias in connection.execute("SELECT id, alias FROM datas")}
    finally:
        connection.close()
    manifest = script_manifest(codes, scripts, aliases)
    if manifest != meta.get("script_manifest_sha256"):
        raise ValueError("the scripts differ from the ones the semantics file was built from")

    table = card_exact.build_table(cards_db)
    arrows, setcodes = card_exact.SEGMENT_OFFSETS["link_arrows"], card_exact.SEGMENT_OFFSETS["setcodes"]
    n = len(codes) + 1
    arrays = {name: np.zeros((n, width), dtype) for name, (width, dtype) in T.SHAPES.items()}
    arrays["effect_unit_reasons"][0] = static_extras.NO_SCRIPT
    for row, code in enumerate(codes, start=1):
        vector = table.row(code)
        arrays["link_arrows"][row] = vector[arrows:arrows + 8]
        arrays["setcode_bag"][row] = vector[setcodes:setcodes + card_exact.SETCODE_DIM]
        path, script_code = builder.resolve_script_path(Path(scripts), code, aliases)
        if path is None:
            arrays["effect_unit_reasons"][row] = static_extras.NO_SCRIPT
            continue
        if script_code != code:
            arrays["effect_unit_reasons"][row] = static_extras.ALIAS_SCRIPT
            continue
        text = builder.canonicalize_lua(path.read_text(encoding="utf-8", errors="replace"), script_code)
        masks, reasons = static_extras.card_evidence(builder, text)
        arrays["effect_unit_bits"][row] = masks
        arrays["effect_unit_reasons"][row] = reasons
    units = (arrays["effect_unit_bits"].astype(np.int64)[:, :, None] >> np.arange(T.MAX_UNITS)) & 1
    if np.any(units.astype(bool) & ~lua_mask.astype(bool)[:, None, :]):
        raise ValueError("an evidence mask names an effect unit the semantics file does not have")
    tables, tables_sha256 = announce_law.read_tables(announce_tables)
    arrays["genericity"], library = genericity(tables["library"], codes)

    counts = {name: int((arrays["effect_unit_reasons"][1:] == value).sum())
              for value, name in enumerate(static_extras.REASONS)}
    metadata = {
        "schema": T.SCHEMA,
        "rows": "unknown-zero-then-code-list-order",
        "code_list_sha256": T.code_list_digest(codes),
        "cards": len(codes),
        "cards_db_sha256": hashlib.sha256(Path(cards_db).read_bytes()).hexdigest(),
        "semantics": {"sha256": hashlib.sha256(Path(semantics).read_bytes()).hexdigest(), "schema": meta.get("schema"),
                      "script_manifest_sha256": manifest},
        "builder_sha256": hashlib.sha256(Path(builder.__file__).read_bytes()).hexdigest(),
        "card_exact": {"layout": table.metadata["layout"], "setcode_hash": table.metadata["setcode_hash"],
                       "artwork_variant_rule": table.metadata["artwork_variant_rule"]},
        "library": {**library, "announce_tables_sha256": tables_sha256},
        "evidence": {"law": T.EVIDENCE_LAW, "ordinals": T.ORDINALS, "max_units": T.MAX_UNITS,
                     "reasons": list(static_extras.REASONS), "reason_counts": counts,
                     "registration_order_used": False},
        "arrays": {name: [list(arrays[name].shape), str(arrays[name].dtype)] for name in T.SHAPES},
        "learned_weights": False,
    }
    return arrays, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("--code-list", "--cards-db", "--scripts", "--semantics", "--announce-tables", "--out"):
        parser.add_argument(name, required=True)
    args = parser.parse_args(argv)
    arrays, metadata = build(args.code_list, args.cards_db, args.scripts, args.semantics, args.announce_tables)
    path, sha256 = T.write(arrays, metadata, args.out)
    T.load(path, args.code_list)
    bound = arrays["effect_unit_bits"][arrays["effect_unit_bits"] != 0]
    popcount = {str(k): int(v) for k, v in zip(*np.unique([bin(int(m)).count("1") for m in bound],
                                                           return_counts=True))}
    print(json.dumps({"path": str(path), "sha256": sha256, "library": metadata["library"],
                      "evidence": metadata["evidence"]["reason_counts"], "bound_units": popcount,
                      "semantics": metadata["semantics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
