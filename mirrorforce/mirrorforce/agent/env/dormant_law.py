"""The dormant-identity table of a run (cxx/duelpool/duel/dormant_law.h): reading, writing and registering it.

A table lists the codes of a run's candidate universe whose card script makes duel-level registrations when it
loads (global watchers, activity counters, global flags), with what each load did. Registered in a process, every
duel created afterwards first gets one inert object per table code, so those registrations no longer depend on the
decks -- which belief-world search needs (a replaced hidden card must not leave the true deck's watchers behind).
Tables are content-addressed: ``dormant-table-<sha256>.json``, the digest over the canonical JSON; a file whose name
does not match its content is refused. ``tools/mf_runtime_dormant_table.py`` writes them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "mirrorforce_dormant_table/v2"
LAW = "dormant_identities/v1"
# duel_create_dormant's report bits (dormant_law.h)
WATCHERS, ACTIVITY_COUNTERS, GLOBAL_FLAGS, RULES, GRANTED, UNOWNED = 1, 2, 4, 8, 16, 32
ALLOWED = WATCHERS | ACTIVITY_COUNTERS | GLOBAL_FLAGS | UNOWNED


def canonical(table):
    return json.dumps(table, sort_keys=True, separators=(",", ":")).encode()


def refused_by(first, second):
    """Whether a code may not be dormant: its load registers rules or gives effects to other cards, or every instance
    registers at duel level something it does not own (a replacement could not remove it with the card)."""
    return bool(first & ~ALLOWED or second & (RULES | GRANTED | UNOWNED))


def build(universe, report):
    """The table of a universe from ``dormant_scan``'s report (same order, each code's first and second load): codes
    whose load only watches, the codes whose load does more (refused: they may not be in decks while the table is
    registered), and the universe digest."""
    universe = sorted(int(c) for c in universe)
    if len(universe) != len(report):
        raise ValueError("one report per universe code")
    kinds = {code: [int(first), int(second)] for code, (first, second) in zip(universe, report) if first or second}
    return {
        "schema": SCHEMA,
        "law": LAW,
        "universe_size": len(universe),
        "universe_sha256": hashlib.sha256(" ".join(map(str, universe)).encode()).hexdigest(),
        "codes": sorted(code for code, (first, second) in kinds.items() if first and not refused_by(first, second)),
        "kinds": {str(code): bits for code, bits in sorted(kinds.items())},
        "refused": sorted(code for code, (first, second) in kinds.items() if refused_by(first, second)),
    }


def write(table, out_dir):
    digest = hashlib.sha256(canonical(table)).hexdigest()
    path = Path(out_dir) / f"dormant-table-{digest}.json"
    path.write_bytes(canonical(table))
    return path, digest


def read_table(path):
    """The table and its digest; refuses a file whose name is not its content's digest, or another schema."""
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if path.name != f"dormant-table-{digest}.json":
        raise ValueError(f"{path.name} is not named by its content (sha256 {digest})")
    table = json.loads(data)
    if table.get("schema") != SCHEMA or table.get("law") != LAW:
        raise ValueError(f"{path.name} is not a {SCHEMA} table")
    if canonical(table) != data:
        raise ValueError(f"{path.name} is not in canonical form")
    if table["refused"]:
        raise ValueError(f"{path.name} lists codes whose load does more than watch: {table['refused'][:8]}")
    return table, digest


def register(native, path):
    """Registers a table in the process (one per process); returns its identity."""
    table, digest = read_table(path)
    native.register_dormant_table(codes=table["codes"], sha256=digest)
    return identity(native)


def identity(native):
    law = native.dormant_identity()
    if law is None:
        return None
    return {"law": law["law"], "sha256": law["sha256"], "codes": len(law["codes"])}


def register_from_args(args):
    """The run's table (``--dormant-table``), registered in the env module; None when the run has none."""
    if not getattr(args, "dormant_table", None):
        return None
    from mirrorforce.agent.env.duel import native
    return register(native, args.dormant_table)
