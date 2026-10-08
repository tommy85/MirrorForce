"""The public room format a run declares: ``obs:global_`` columns 23 (format index) and 24 (era).

A room's card-limit list is public before the duel (``era_format.py``). The run names a format key of a registered
format table (``room-formats-<sha256>.json``, schema ``mirrorforce_room_format/v1``, entries only ever appended);
this module resolves its index and era once and the env writes both for every decision of every game (env config
``room_format``, ``room_era``). Era law ``quarters_since_2016q1/v1``: 0 for an undated format (unknown, unrestricted),
else 1 + the quarters from 2016 Q1 to the format's effective date (md-2022H1 25, md-2026H2 43). A run that names no
format gets index 0 (``unknown``) and era 0, and records none. The returned identity goes into checkpoint receipts.
"""
from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path
import re

NAME = re.compile(r"room-formats-([0-9a-f]{64})\.json")
ERA_LAW = "quarters_since_2016q1/v1"
CAPACITY = 256  # the obs columns are bytes


KINDS = ("unknown", "unrestricted", "md", "ocg")


class FormatTable:
    """A registered format table's entries (``era_format.FormatTable``'s rules, checked here so the tree stays
    self-contained): index 0 the undated unknown format, unique keys, one undated ``unrestricted`` list, exactly the MD
    and OCG lists dated."""

    def __init__(self, entries):
        entries = [dict(e) for e in entries]
        if not entries or entries[0] != {"key": "unknown", "kind": "unknown", "effective_date": None}:
            raise ValueError("format index 0 must be the undated unknown format")
        for entry in entries:
            if set(entry) != {"key", "kind", "effective_date"} or entry["kind"] not in KINDS \
                    or not isinstance(entry["key"], str) or not entry["key"].strip():
                raise ValueError("a format entry is a key, a kind and an effective date")
            dated = entry["kind"] in ("md", "ocg")
            if dated != isinstance(entry["effective_date"], str):
                raise ValueError("exactly MD and OCG formats carry an effective date")
            if dated:
                date.fromisoformat(entry["effective_date"])
        self.keys = [e["key"] for e in entries]
        if len(set(self.keys)) != len(self.keys) or self.keys.count("unrestricted") != 1 \
                or entries[self.keys.index("unrestricted")]["kind"] != "unrestricted":
            raise ValueError("format keys are unique and one of them is the unrestricted list")
        self.entries = entries

    def index(self, key):
        return self.keys.index(key)


def read_table(path):
    """``(FormatTable, sha256)`` of a content-addressed format table; a renamed, edited or non-canonical file is
    refused (the name is the SHA-256 of the canonical JSON bytes, as registers it)."""
    import json

    path = Path(path)
    match = NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"{path.name} is not a content-addressed room format table (room-formats-<sha256>.json)")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != match.group(1):
        raise ValueError(f"{path.name}: the file's bytes do not match its name")
    data = json.loads(raw)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    if canonical != raw or data.get("schema") != "mirrorforce_room_format/v1" or set(data) != {"schema", "formats"}:
        raise ValueError(f"{path.name}: not a canonical mirrorforce_room_format/v1 table")
    table = FormatTable(data["formats"])
    if len(table.entries) > CAPACITY:
        raise ValueError(f"{path.name}: {len(table.entries)} formats, the observation holds {CAPACITY}")
    return table, match.group(1)


def era(effective_date):
    """The era of an effective date (ISO string or None) under ``ERA_LAW``."""
    if effective_date is None:
        return 0
    day = date.fromisoformat(effective_date)
    value = 1 + (day.year - 2016) * 4 + (day.month - 1) // 3
    if not 1 <= value < CAPACITY:
        raise ValueError(f"effective date {effective_date} outside the era law's range")
    return value


def resolve(path, key):
    """The identity of format ``key`` in the table at ``path``: table digest, key, index, era and the era law."""
    table, sha256 = read_table(path)
    if key not in table.keys:
        raise ValueError(f"format {key!r} is not in {Path(path).name}")
    index = table.index(key)
    return {"table_sha256": sha256, "key": key, "index": index, "era": era(table.entries[index]["effective_date"]),
            "era_law": ERA_LAW}


def from_args(args):
    """The run's format (``--room-format`` in ``--room-format-table``), or None: index 0, era 0."""
    key, path = getattr(args, "room_format", None), getattr(args, "room_format_table", None)
    if key is None and path is None:
        return None
    if key is None or path is None:
        raise ValueError("--room-format and --room-format-table go together")
    return resolve(path, key)


def env_config(identity):
    """The env config entries of a resolved format (or of none)."""
    return {"room_format": identity["index"] if identity else 0, "room_era": identity["era"] if identity else 0}
