"""The public recipe library and generic staple table of the announce-card law, from a registered deck pool.

The announce law (design §6.1) takes two tables built offline from the training pool: the public library (each
recipe's main and extra deck, cluster and format, in the pool's registered order) and the generic staple table (deck-type
document frequency: for each card, the number of deck types with at least one recipe containing it, main or extra;
cards at or above ``threshold`` of all deck types). This tool writes both as one content-addressed JSON file from the
pool file, for one group and split (M1: ``md-2026H2`` ``train``, the 80 MD recipes).

Usage: ``--pool <pool json> --pool-sha256 <sha> --group md-2026H2 --split train --threshold 0.1 --out <dir>``.

A deck directory with its manifest (the friend pool: ``.ydk`` files and ``manifests/train.jsonl`` rows with
``deck_name`` and ``content_cluster``) is read as a pool of one group: ``--ydk-dir <dir> --manifest <jsonl> --format
<label> --threshold 0.1 --out <dir>``. Such a pool names no deck types, so each content cluster counts as one deck type;
the output records the deck files' tree digest in place of a pool digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mirrorforce.common.sidecar_io import canonical, digest, publish_immutable, read_regular

SCHEMA = "mirrorforce_announce_tables/v1"


def tables(pool, *, group, split, threshold):
    """``{"library": [...], "staples": [[code, deck_types], ...], "deck_types": n, ...}`` for one pool group."""
    recipes = [r for r in pool["recipes"] if r.get("group") == group and r.get("split") == split]
    if not recipes:
        raise ValueError(f"the pool has no {group}/{split} recipes")
    if not 0 < threshold < 1:
        raise ValueError("the staple threshold is a fraction of the deck types")
    library = []
    for recipe in recipes:
        if not all(isinstance(recipe.get(k), list) for k in ("main", "extra")) or not recipe.get("cluster") \
                or not recipe.get("deck_type") or not recipe.get("format"):
            raise ValueError("a pool recipe lacks its main, extra, cluster, deck type or format")
        library.append({"main": [int(c) for c in recipe["main"]], "extra": [int(c) for c in recipe["extra"]],
                        "cluster": recipe["cluster"], "deck_type": recipe["deck_type"], "format": recipe["format"]})
    types = sorted({r["deck_type"] for r in library})
    holders = {}
    for recipe in library:
        for code in set(recipe["main"]) | set(recipe["extra"]):
            holders.setdefault(code, set()).add(recipe["deck_type"])
    staples = sorted(([code, len(kinds)] for code, kinds in holders.items() if len(kinds) >= threshold * len(types)),
                     key=lambda row: (-row[1], row[0]))
    return {"schema": SCHEMA, "group": group, "split": split, "threshold": threshold, "recipes": len(library),
            "deck_types": len(types), "distinct_cards": len(holders), "library": library, "staples": staples}


def read_ydk(path):
    """``(main, extra)`` card codes of a ``.ydk`` deck file, in file order (the side deck is not read)."""
    main, extra, section = [], [], None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line in ("#main", "#extra", "!side"):
            section = line
        elif line and not line.startswith("#") and section in ("#main", "#extra"):
            (main if section == "#main" else extra).append(int(line))
    if not main:
        raise ValueError(f"{path} has no main deck")
    return main, extra


def ydk_pool(deck_dir, manifest, *, format_label):
    """A pool of the manifest's decks, in manifest order: group ``ydk``, split ``train``, deck type = content cluster;
    with the decks' tree digest (name and SHA-256 of every listed file)."""
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("the manifest lists no decks")
    recipes, tree = [], hashlib.sha256()
    for row in rows:
        path = Path(deck_dir) / f"{row['deck_name']}.ydk"
        main, extra = read_ydk(path)
        recipes.append({"group": "ydk", "split": "train", "main": main, "extra": extra,
                        "cluster": row["content_cluster"], "deck_type": row["content_cluster"],
                        "format": format_label})
        tree.update(path.name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).hexdigest().encode() + b"\n")
    return {"recipes": recipes}, tree.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pool", type=Path)
    source.add_argument("--ydk-dir", type=Path)
    parser.add_argument("--pool-sha256")
    parser.add_argument("--group")
    parser.add_argument("--split")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--format")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.pool is not None:
        if not (args.pool_sha256 and args.group and args.split):
            parser.error("--pool needs --pool-sha256, --group and --split")
        raw = read_regular(args.pool, "deck pool")
        if digest(raw) != args.pool_sha256:
            raise ValueError("deck pool checksum differs")
        result = tables(json.loads(raw), group=args.group, split=args.split, threshold=args.threshold)
        result["pool_sha256"] = args.pool_sha256
    else:
        if not (args.manifest and args.format):
            parser.error("--ydk-dir needs --manifest and --format")
        pool, tree_sha256 = ydk_pool(args.ydk_dir, args.manifest, format_label=args.format)
        result = tables(pool, group="ydk", split="train", threshold=args.threshold)
        result["deck_tree_sha256"] = tree_sha256
    args.out.mkdir(parents=True, exist_ok=True)
    raw = canonical(result)
    path = args.out / f"announce-tables-{digest(raw)}.json"
    publish_immutable(path, raw)
    print(json.dumps({"path": str(path), "sha256": digest(raw), "recipes": result["recipes"],
                      "deck_types": result["deck_types"], "staples": len(result["staples"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
