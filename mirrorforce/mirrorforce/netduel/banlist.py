"""Read YGOPro ``lflist.conf`` with the server's exact hash semantics.

The banlist is enforced by ``DeckManager::CheckDeck`` before the duel starts,
not by ocgcore while a chain is resolving.  A network room therefore exposes
only the selected list's 32-bit hash in ``HostInfo``.  Policy code must resolve
that hash against the same local file before it can encode the per-card limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["Banlist", "load_lflists", "load_lflist_map"]


@dataclass(frozen=True)
class Banlist:
    name: str
    hash: int
    limits: dict[int, int]


def _rotl32(value: int, shift: int) -> int:
    shift &= 31
    value &= 0xFFFFFFFF
    return ((value << shift) | (value >> (32 - shift))) & 0xFFFFFFFF


def load_lflists(path: str | Path) -> list[Banlist]:
    """Parse every ``!name`` section, preserving YGOPro's computed hash."""
    result: list[Banlist] = []
    name: str | None = None
    limits: dict[int, int] = {}
    # gframe/deck_manager.cpp seeds every named section before XORing rows.
    # Omitting this constant produces a stable-looking hash that can never
    # match HostInfo.lflist from the same client build.
    list_hash = 0x7DFCEE6A

    def finish() -> None:
        if name is not None:
            result.append(Banlist(name, list_hash & 0xFFFFFFFF, dict(limits)))

    with Path(path).open("r", encoding="utf-8-sig", errors="replace") as src:
        for raw in src:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("$"):
                continue
            if line.startswith("!"):
                finish()
                name = line[1:].strip()
                limits = {}
                list_hash = 0x7DFCEE6A
                continue
            if name is None:
                continue
            fields = line.split("--", 1)[0].split()
            if len(fields) < 2:
                continue
            try:
                code, count = int(fields[0], 10), int(fields[1], 10)
            except ValueError:
                continue
            if not (0 <= code <= 0xFFFFFFFF and 0 <= count <= 2):
                continue
            limits[code] = count
            # gframe/deck_manager.cpp: the two rotate expressions are XORed
            # into the section hash once per valid input row.
            list_hash ^= _rotl32(code, 18) ^ _rotl32(code, 27 + count)
    finish()
    return result


def load_lflist_map(paths) -> dict[int, Banlist]:
    """Load files in order; like the client, keep the first matching hash."""
    out: dict[int, Banlist] = {}
    for path in paths:
        for banlist in load_lflists(path):
            out.setdefault(banlist.hash, banlist)
    out.setdefault(0, Banlist("N/A", 0, {}))
    return out
