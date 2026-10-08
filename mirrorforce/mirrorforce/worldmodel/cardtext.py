"""Card text and the effect-slot axis the enumerator counts on.

The candidate enumerator is not allowed to know what a card *does*; it is only
allowed to read the card the way a player reads it.  So "how many effects does
this card have" has to come out of the printed text, not out of the engine.

OCG/CN card text numbers its effects ``①②③``, and splitting on those markers is
exact.  The English text this project's ``cards.cdb`` carries does not, so the
fallback is a sentence split.  Both are versioned (:data:`SEGMENTER_VERSION`)
and the version is written into every shard, because re-segmenting changes what
effect index ``i`` means and therefore invalidates every label that names one.

Alignment with the engine
-------------------------

``ygopro`` encodes an effect description as ``code * 16 + i``; the netduel
action layer unpacks that into ``effect = i + CARD_EFFECT_OFFSET``.  ``i``
indexes the card script's own string table.  It follows the printed order in
practice but is **not bound to it** -- a single ``①`` clause whose script
registers two effects consumes two slots, and some scripts keep prompt hints in
the same table.  So no text-derived axis can be *proved* complete, and the
question is only how small the residual is.

Two things keep it small.  The slot *count* is the widest of every
segmentation available -- numbered clauses, sentences, and the same card in
another localisation -- while the *segments* stay the semantic ones.  All of
those are pure functions of printed text, so no card knowledge leaks in, and
over-enumerating a slot that is never legal only produces a negative.
``slot_floor`` then buys the rest outright: raising it enumerates a fixed
minimum per card, at ``MAX_EFFECT_SLOTS`` the axis is complete by construction
and the candidate count is about four times larger.

Measured on 1,038,755 engine menu entries: the ``①`` count alone missed 0.06%,
the widest segmentation missed 0.0001%.

``desc == 0`` means "activate the card itself" rather than a numbered effect;
it gets its own action type (``ACTIVATE_CARD``) instead of a slot.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CIRCLED_MARKERS",
    "CardText",
    "CardTextIndex",
    "MAX_EFFECT_SLOTS",
    "SEGMENTER_VERSION",
    "segment_text",
    "sentence_split",
]

#: bump whenever :func:`segment_text` or the slot-count rule changes; shards
#: carry it, because re-segmenting changes what effect index ``i`` names
SEGMENTER_VERSION = "seg-v3-circled-or-sentence-max"

#: ``unpack_desc`` rejects an effect index of 14 or more, so the engine can
#: never name a slot beyond this and enumerating past it is pure waste
MAX_EFFECT_SLOTS = 14

CIRCLED_MARKERS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭"

_CIRCLED_SPLIT = re.compile(f"[{CIRCLED_MARKERS}]")
#: Latin sentences end with punctuation *and* a space; CJK sentences end with
#: their own full-width punctuation and no space at all.  Matching only the
#: Latin form made the fallback a no-op on the Chinese database, which is the
#: one that carries the numbered clauses -- the bug that let a card's effect
#: index run past the end of its slot axis.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[。！？；])")

DEFAULT_DB = Path(
    os.environ.get("MF_YGOPRO_DB", "/path/to/workspace/ygopro-client-run/cards.cdb")
)


def sentence_split(text: str) -> list[str]:
    return [p.strip() for p in _SENTENCE_SPLIT.split((text or "").strip()) if p.strip()]


def segment_text(text: str) -> tuple[list[str], str]:
    """Split card text into effect clauses; returns ``(segments, mode)``.

    ``mode`` is ``"circled"`` when the OCG ``①②③`` markers carried the split and
    ``"sentence"`` when it fell back to punctuation, so a shard can be filtered
    down to the exactly-segmented half.
    """
    text = (text or "").strip()
    if not text:
        return [], "empty"
    if any(marker in text for marker in CIRCLED_MARKERS):
        parts = _CIRCLED_SPLIT.split(text)
        # text before the first marker is flavour / summoning conditions, not
        # a numbered effect, so it is dropped rather than made into slot 0
        return [p.strip() for p in parts[1:] if p.strip()], "circled"
    return sentence_split(text), "sentence"


@dataclass(frozen=True)
class CardText:
    """One card as the enumerator is allowed to see it."""

    code: int
    name: str
    text: str
    segments: tuple[str, ...]
    mode: str
    #: slots to enumerate: the larger of this card's two segmentations, so a
    #: numbered clause that registers more than one effect still has an index
    n_slots: int = 1
    #: Text-derived slots before ``slot_floor`` padding. A malformed external
    #: Stringid may safely fall back only when this value is exactly one.
    semantic_slots: int = 1


class CardTextIndex:
    """``code -> CardText`` over one or more card-text databases.

    ``db_path`` supplies the text and the semantic segments.  Any further
    databases in ``cover_dbs`` are read for their segment counts only: the
    localisations disagree about how many clauses a card has, and the slot axis
    has to be wide enough for whichever one the engine's script indexed
    against.  Passing the English database alongside the Chinese one is the
    configuration the corpus is generated with.

    The whole ``texts`` table is 15k rows, so it is read once and kept; that
    removes a per-candidate sqlite round trip from the inner loop.
    """

    def __init__(
        self,
        db_path: str | os.PathLike[str] = DEFAULT_DB,
        cover_dbs: list[str | os.PathLike[str]] | None = None,
        slot_floor: int = 0,
    ):
        self.db_path = str(db_path)
        self.cover_dbs = [str(p) for p in (cover_dbs or [])]
        self.slot_floor = max(0, min(int(slot_floor), MAX_EFFECT_SLOTS))
        self._card_types = _read_card_types(self.db_path)
        cover_counts: dict[int, int] = {}
        for path in self.cover_dbs:
            for code, desc in _read_texts(path):
                segments, _ = segment_text(desc)
                n = max(len(segments), len(sentence_split(desc)))
                cover_counts[code] = max(cover_counts.get(code, 0), n)

        self._by_code: dict[int, CardText] = {}
        for code, name, desc in _read_texts(self.db_path, with_name=True):
            segments, mode = segment_text(desc)
            # Only the primary localisation defines semantic clauses. The
            # English cover database and slot floor widen the menu axis; an
            # English multi-sentence paragraph is still one numbered OCG
            # effect and must not make an unambiguous fallback look ambiguous.
            semantic_slots = min(max(len(segments), 1), MAX_EFFECT_SLOTS)
            n = max(
                semantic_slots,
                len(sentence_split(desc)),
                cover_counts.get(code, 0),
                self.slot_floor,
            )
            self._by_code[code] = CardText(
                code=code,
                name=name,
                text=desc,
                segments=tuple(segments),
                mode=mode,
                n_slots=min(n, MAX_EFFECT_SLOTS),
                semantic_slots=semantic_slots,
            )
        self._unknown_cache: dict[int, CardText] = {}

    def __len__(self) -> int:
        return len(self._by_code)

    def get(self, code: int) -> CardText:
        """The card, or a blank one-slot entry for a code the database lacks."""
        hit = self._by_code.get(int(code))
        if hit is not None:
            return hit
        blank = self._unknown_cache.get(int(code))
        if blank is None:
            blank = CardText(
                code=int(code), name="", text="", segments=(), mode="empty",
                n_slots=1, semantic_slots=0,
            )
            self._unknown_cache[int(code)] = blank
        return blank

    def n_slots(self, code: int) -> int:
        return self.get(code).n_slots

    def known(self, code: int) -> bool:
        """Whether ``code`` has a pinned CDB text row.

        Runtime/synthetic codes can still be public. Their text is unknown, so
        the candidate grammar must conservatively expose every effect slot.
        """
        return int(code) in self._by_code

    def effect_count(self, code: int) -> int:
        """Text-derived effect count, excluding candidate floor padding."""
        return self.get(code).semantic_slots

    def card_type(self, code: int) -> int | None:
        """Return the printed CDB type, or ``None`` for an unknown code."""
        return self._card_types.get(int(code))

    def mode_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for card in self._by_code.values():
            out[card.mode] = out.get(card.mode, 0) + 1
        return out

    def provenance(self) -> dict:
        return {
            "segmenter": SEGMENTER_VERSION,
            "text_db": self.db_path,
            "cover_dbs": list(self.cover_dbs),
            "slot_floor": self.slot_floor,
            "mean_slots": round(
                sum(c.n_slots for c in self._by_code.values())
                / max(1, len(self._by_code)),
                4,
            ),
            "modes": self.mode_counts(),
        }


def _read_texts(path: str, with_name: bool = False):
    db = sqlite3.connect(path)
    try:
        if with_name:
            rows = db.execute("SELECT id, name, desc FROM texts").fetchall()
            return [(int(c), n or "", d or "") for c, n, d in rows]
        rows = db.execute("SELECT id, desc FROM texts").fetchall()
        return [(int(c), d or "") for c, d in rows]
    finally:
        db.close()


def _read_card_types(path: str) -> dict[int, int]:
    db = sqlite3.connect(path)
    try:
        return {
            int(code): int(card_type)
            for code, card_type in db.execute("SELECT id, type FROM datas")
        }
    finally:
        db.close()
