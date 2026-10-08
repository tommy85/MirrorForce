"""The run's announce-card candidate law (design 6.1): one registered law per process, from content-addressed tables.

The tables file (``announce-tables-<sha256>.json``, schema ``mirrorforce_announce_tables/v1``: the public recipe
library and the generic staple table of the training pool, with the staple threshold) is written by the pool asset
builder; its name carries the SHA-256 of its bytes. ``register`` checks both and registers the law with the built
module (``duel_native.register_announce_law``). The env refuses an announce prompt while no law is registered and
refuses a second, different law in the same process; the returned identity goes into checkpoint receipts, so a run
resumed, evaluated or deployed under another law is refused.
"""
import hashlib
import json
from pathlib import Path
import re

SCHEMA = "mirrorforce_announce_tables/v1"
NAME = re.compile(r"announce-tables-([0-9a-f]{64})\.json")


def read_tables(path):
    """``(tables, file sha256)`` of a content-addressed tables file; a renamed or edited file is refused."""
    path = Path(path)
    match = NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"{path.name} is not a content-addressed announce tables file (announce-tables-<sha256>.json)")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != match.group(1):
        raise ValueError(f"{path.name}: the file's bytes do not match its name")
    tables = json.loads(raw)
    if tables.get("schema") != SCHEMA:
        raise ValueError(f"{path.name}: not {SCHEMA}")
    for key in ("library", "staples", "threshold"):
        if key not in tables:
            raise ValueError(f"{path.name}: no {key}")
    return tables, match.group(1)


def register(native, path, cap, room_format=None):
    """Register the law from ``path`` with ``native``; returns its identity (with the tables file's digest)."""
    tables, file_sha256 = read_tables(path)
    identity = native.register_announce_law(
        library=tables["library"], staples=[(int(code), int(count)) for code, count in tables["staples"]],
        staple_threshold=float(tables["threshold"]), cap=int(cap), room_format=room_format)
    return {**identity, "tables_file_sha256": file_sha256}


def register_from_args(args):
    """The training and env-check entry: ``--announce-tables`` is required, with ``--announce-cap`` and the format;
    every announce menu must fit the env's menu, so a cap above ``--max-options`` is refused here, at startup."""
    if not args.announce_tables:
        raise ValueError("--announce-tables must name the run's announce law tables (announce-tables-<sha256>.json)")
    if args.announce_cap > args.max_options:
        raise ValueError(f"--announce-cap {args.announce_cap} exceeds --max-options {args.max_options}: "
                         "an announce menu must fit the env's menu")
    from mirrorforce.agent.env.duel import native
    return register(native, args.announce_tables, args.announce_cap, args.announce_room_format)
