"""Card data for the network client.

The engine lives on the host, so the client needs its own copy of the card
database for two things:

* ``MSG_ANNOUNCE_CARD`` with a general filter -- the core sends an RPN program,
  not a list of codes, and the answer must be a code that satisfies it;
* the ``code -> card id`` mapping ygoenv's observations use (``code_list.txt``
  line numbers, 1-based), needed once a trained policy is plugged in.

The alias/rule_code split mirrors ``db_query_card_data`` in ``ygopro.h``: the
cdb has one ``alias`` column that historically meant two different things, and
the current core wants them separated.  ``card_is_declarable`` rejects both
artwork variants and "treated as" cards, so getting this wrong changes which
names are declarable.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .actions import card_is_declarable

__all__ = ["CardData", "CardPool", "load_ydk"]

_ARTWORK_VERSIONS_OFFSET = 20
#: ``TYPE_LINK``; the cdb stores a Link monster's markers in the def column.
_TYPE_LINK = 0x4000000
_BLACK_LUSTER_SOLDIER_2 = 5405695


def _write_setcode(value: int) -> list[int]:
    """``write_setcode``: a 64 bit column becomes up to 4 non-zero uint16."""
    out = []
    while value:
        low = value & 0xFFFF
        if low:
            out.append(low)
        value >>= 16
    return out


@dataclass
class CardData:
    code: int = 0
    alias: int = 0
    rule_code: int = 0
    setcodes: list = field(default_factory=list)
    type: int = 0
    level: int = 0
    lscale: int = 0
    rscale: int = 0
    attack: int = 0
    defense: int = 0
    link_marker: int = 0
    race: int = 0
    attribute: int = 0
    name: str = ""


class CardPool:
    """Every card in the database, keyed by code."""

    def __init__(self, db_path: str | Path, code_list: str | Path | None = None):
        self.db_path = str(db_path)
        self.cards: dict[int, CardData] = {}
        self.card_ids: dict[int, int] = {}
        self._load_db()
        if code_list:
            self._load_code_list(code_list)

    def _load_db(self) -> None:
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT d.id, d.alias, d.setcode, d.type, d.level, d.atk, d.def,"
                " d.race, d.attribute, t.name"
                " FROM datas d LEFT JOIN texts t ON t.id = d.id"
            ).fetchall()
        finally:
            con.close()
        for code, alias, setcode, type_, level, atk, def_, race, attribute, name in rows:
            artwork = (
                alias != 0
                and code != _BLACK_LUSTER_SOLDIER_2
                and alias < code + _ARTWORK_VERSIONS_OFFSET
                and code < alias + _ARTWORK_VERSIONS_OFFSET
            )
            self.cards[code] = CardData(
                code=code,
                alias=alias if artwork else 0,
                rule_code=0 if artwork else alias,
                setcodes=_write_setcode(setcode or 0),
                type=type_ or 0,
                # The cdb overloads two columns, and the client decodes both
                # before anything reads them (``data_manager.cpp:50-60``):
                # a Link monster's ``def`` holds its marker bitmask and its
                # defense is zero, and a Pendulum monster's ``level`` packs the
                # two scales into the high bytes.  Reading either raw feeds a
                # marker bitmask into a defense field -- 505 Link cards in this
                # database have a nonzero ``def`` column -- and every consumer
                # of the pool, live encoder included, then disagrees with what
                # the engine reports for the same card.
                level=(level or 0) & 0xFF,
                lscale=((level or 0) >> 24) & 0xFF,
                rscale=((level or 0) >> 16) & 0xFF,
                attack=atk or 0,
                defense=0 if (type_ or 0) & _TYPE_LINK else (def_ or 0),
                link_marker=(def_ or 0) if (type_ or 0) & _TYPE_LINK else 0,
                race=race or 0,
                attribute=attribute or 0,
                name=name or "",
            )

    def _load_code_list(self, path: str | Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                parts = line.split()
                if parts and parts[0].isdigit():
                    self.card_ids[int(parts[0])] = i

    def card_id(self, code: int) -> int:
        """ygoenv's ``c_get_card_id``: the 1-based line in ``code_list.txt``."""
        return self.card_ids.get(code, 0)

    def name(self, code: int) -> str:
        card = self.cards.get(code)
        return card.name if card else ""

    def declarable(
        self, opcodes: list[int], preferred: list[int] | None = None, limit: int = 24
    ) -> list[int]:
        """Codes satisfying an ``MSG_ANNOUNCE_CARD`` filter.

        ygoenv scans both players' decks first and only falls back to the whole
        pool.  A network client knows its own deck and whatever it has seen of
        the opponent's, never the opponent's full list -- so the candidate set
        can differ from the offline one even though every answer here is a
        legal one.  Order is deterministic so a run is reproducible.
        """
        out: list[int] = []
        seen: set[int] = set()

        def consider(code: int) -> bool:
            if len(out) >= limit or code in seen:
                return False
            seen.add(code)
            card = self.cards.get(code)
            if card is None:
                return False
            if card_is_declarable(card, opcodes):
                out.append(code)
            return True

        for code in preferred or []:
            consider(code)
            if len(out) >= limit:
                return out
        for code in sorted(self.cards):
            consider(code)
            if len(out) >= limit:
                break
        return out


def load_ydk(path: str | Path) -> tuple[list[int], list[int], list[int]]:
    """Parse a ``.ydk`` into ``(main, extra, side)``.

    The host decides what belongs in the extra deck by card type, so the split
    here only needs to be good enough to send: ``#main`` / ``#extra`` / ``!side``.
    """
    main: list[int] = []
    extra: list[int] = []
    side: list[int] = []
    target = main
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("#extra"):
            target = extra
            continue
        if low.startswith("!side"):
            target = side
            continue
        if low.startswith("#"):
            target = main if low.startswith("#main") else target
            continue
        if line.isdigit():
            target.append(int(line))
    return main, extra, side
