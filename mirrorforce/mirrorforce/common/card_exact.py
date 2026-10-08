"""Exact static card channel (F1): printed CDB fields as numbers and bits.

Design: the design notes.  The
``card_structure`` and ``card_name`` channels read printed data only through a
frozen text embedding, where "ATK 2500" and "ATK 2400" are not linearly
related and "?" is indistinguishable from 0.  This channel gives the same
printed fields exactly, keyed by passcode like the other card channels and
built from the same ``cards.cdb``.  It follows a colleague's
``exact_cdb_features``, with link arrows as eight bits and a setcode bag added.

Layout ``mirrorforce_card_exact_layout/v1`` (129 float32 per card):

====================  =====  ==============================================
segment               width  content
====================  =====  ==============================================
``kind``                  4  monster, spell, trap, other
``type_bits``            32  bit ``i`` of the ``type`` column
``race_bits``            32  bit ``i`` of the ``race`` column
``attribute_bits``        7  bit ``i`` of the ``attribute`` column
``values``                7  ATK/10000, DEF/10000 (clipped to [-2, 2]),
                             level/13, rank/13, link rating/8,
                             left scale/13, right scale/13
``known``                 7  1 where the value above applies and is printed
``link_arrows``           8  bottom-left, bottom, bottom-right, left, right,
                             top-left, top, top-right
``setcodes``             32  L2-normalised signed-hash bag of the setcodes
====================  =====  ==============================================

ATK and DEF apply to monsters, DEF not to link monsters; a negative printed
value ("?") is unknown: value 0 and flag 0.  Level applies to monsters that
are neither xyz nor link, rank to xyz, link rating to link, the scales to
pendulum monsters.  Arrows are the link monster's DEF column.  Each nonzero
16-bit setcode ``s`` adds the tokens ``set:%04x`` (``s``) and
``setbase:%03x`` (``s & 0xfff``), hashed like the R7 Lua channel
(``blake2b64-le/index=mod-dim/sign=bit63/l2``) into 32 slots.

An artwork variant (the engine's rule: a nonzero alias within 20 of the
passcode, except Black Luster Soldier - Envoy of the Evening Twilight's
rule-code alias) takes its base card's row; any other alias is a rule code
and keeps the card's own printed data.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .effect_lua_hash import HASH_SPEC, hash_slot
from .frozen_npz import atomic_write, npz_bytes

__all__ = [
    "CARD_EXACT_DIM",
    "CARD_EXACT_LAYOUT",
    "CARD_EXACT_LAYOUT_SCHEMA",
    "CARD_EXACT_SCHEMA",
    "CardExactBuildError",
    "CardExactTable",
    "artifact_bytes",
    "build_table",
    "card_features",
    "setcode_bag",
    "setcode_tokens",
    "write_artifact",
]


CARD_EXACT_SCHEMA = "mirrorforce_card_exact/v1"
CARD_EXACT_LAYOUT_SCHEMA = "mirrorforce_card_exact_layout/v1"
SETCODE_DIM = 32
CARD_EXACT_LAYOUT: tuple[tuple[str, int], ...] = (
    ("kind", 4),
    ("type_bits", 32),
    ("race_bits", 32),
    ("attribute_bits", 7),
    ("values", 7),
    ("known", 7),
    ("link_arrows", 8),
    ("setcodes", SETCODE_DIM),
)
CARD_EXACT_DIM = sum(width for _, width in CARD_EXACT_LAYOUT)
VALUE_NAMES = ("atk", "def", "level", "rank", "link_rating", "left_scale", "right_scale")

TYPE_MONSTER = 0x1
TYPE_SPELL = 0x2
TYPE_TRAP = 0x4
TYPE_XYZ = 0x800000
TYPE_PENDULUM = 0x1000000
TYPE_LINK = 0x4000000
#: LINK_MARKER_* in the engine's order BL, B, BR, L, R, TL, T, TR (0x10 is the centre).
LINK_ARROW_MASKS = (0x1, 0x2, 0x4, 0x8, 0x20, 0x40, 0x80, 0x100)
#: The engine loader's artwork-variant rule (``puzzle/core.py`` ``_query_card_data``).
ARTWORK_OFFSET = 20
BLACK_LUSTER_SOLDIER_2 = 5405695


class CardExactBuildError(ValueError):
    """The card database cannot produce a well-defined exact table."""


def _offset(name: str) -> int:
    offset = 0
    for segment, width in CARD_EXACT_LAYOUT:
        if segment == name:
            return offset
        offset += width
    raise KeyError(name)


SEGMENT_OFFSETS = {name: _offset(name) for name, _ in CARD_EXACT_LAYOUT}


def setcode_tokens(setcode: int) -> list[str]:
    tokens: list[str] = []
    value = int(setcode) & 0xFFFFFFFFFFFFFFFF
    while value:
        part = value & 0xFFFF
        if part:
            tokens.extend((f"set:{part:04x}", f"setbase:{part & 0xFFF:03x}"))
        value >>= 16
    return tokens


def setcode_bag(setcode: int) -> np.ndarray:
    bag = np.zeros(SETCODE_DIM, np.float64)
    for token in setcode_tokens(setcode):
        index, sign = hash_slot(token, SETCODE_DIM)
        bag[index] += sign
    norm = float(np.linalg.norm(bag))
    return (bag / norm if norm > 0 else bag).astype(np.float32)


def _bits(value: int, width: int) -> list[float]:
    return [float((int(value) >> bit) & 1) for bit in range(width)]


def _stat(value: int, scale: float) -> tuple[float, float]:
    if value < 0:
        return 0.0, 0.0
    return max(-2.0, min(2.0, value / scale)), 1.0


def card_features(
    card_type: int, attack: int, defense: int, level_info: int, race: int, attribute: int, setcode: int,
) -> np.ndarray:
    """The 129 exact features of one printed card, in layout order."""
    monster = bool(card_type & TYPE_MONSTER)
    xyz = monster and bool(card_type & TYPE_XYZ)
    link = monster and bool(card_type & TYPE_LINK)
    pendulum = monster and bool(card_type & TYPE_PENDULUM)
    level = level_info & 0xFF
    left_scale = (level_info >> 24) & 0xFF
    right_scale = (level_info >> 16) & 0xFF
    kind = [
        float(monster),
        float(bool(card_type & TYPE_SPELL)),
        float(bool(card_type & TYPE_TRAP)),
        float(not card_type & (TYPE_MONSTER | TYPE_SPELL | TYPE_TRAP)),
    ]
    absent = (0.0, 0.0)
    pairs = [
        _stat(attack, 10000.0) if monster else absent,
        _stat(defense, 10000.0) if monster and not link else absent,
        (level / 13.0, 1.0) if monster and not xyz and not link else absent,
        (level / 13.0, 1.0) if xyz else absent,
        (level / 8.0, 1.0) if link else absent,
        (left_scale / 13.0, 1.0) if pendulum else absent,
        (right_scale / 13.0, 1.0) if pendulum else absent,
    ]
    arrows = [float(bool(defense & mask)) if link and defense >= 0 else 0.0 for mask in LINK_ARROW_MASKS]
    vector = np.asarray(
        kind + _bits(card_type, 32) + _bits(race, 32) + _bits(attribute, 7)
        + [value for value, _ in pairs] + [known for _, known in pairs] + arrows,
        dtype=np.float32,
    )
    return np.concatenate((vector, setcode_bag(setcode)))


def is_artwork_variant(code: int, alias: int) -> bool:
    return bool(alias) and code != BLACK_LUSTER_SOLDIER_2 \
        and alias < code + ARTWORK_OFFSET and code < alias + ARTWORK_OFFSET


@dataclass(frozen=True)
class CardExactTable:
    passcodes: np.ndarray
    vectors: np.ndarray
    metadata: Mapping[str, object]

    def row(self, passcode: int) -> np.ndarray:
        index = int(np.searchsorted(self.passcodes, passcode))
        if index >= len(self.passcodes) or int(self.passcodes[index]) != int(passcode):
            raise KeyError(passcode)
        return self.vectors[index]


def key_sha256(passcodes: Sequence[int]) -> str:
    """Same canonical key hash as ``effect_encoder._key_sha256`` over card passcodes."""
    return hashlib.sha256("".join(f"{int(code)}\n" for code in passcodes).encode("ascii")).hexdigest()


def build_table(cards_db: str | Path) -> CardExactTable:
    """Build the table over every card of one ``cards.cdb``.

    The file's SHA-256 is taken before and after reading, so the table is bound
    to exactly the bytes it was read from.
    """
    path = Path(cards_db)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)
    try:
        rows = connection.execute(
            "SELECT id, alias, setcode, type, atk, def, level, race, attribute FROM datas ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise CardExactBuildError(f"{path} changed while it was read")
    records = {int(row[0]): tuple(int(value) for value in row[1:]) for row in rows}
    if not records or any(code <= 0 for code in records):
        raise CardExactBuildError("cards.cdb has no cards or a non-positive passcode")
    passcodes = sorted(records)
    vectors = np.zeros((len(passcodes), CARD_EXACT_DIM), np.float32)
    variants = 0
    for index, code in enumerate(passcodes):
        alias = records[code][0]
        source = code
        if is_artwork_variant(code, alias) and alias in records:
            source = alias
            variants += 1
        _, setcode, card_type, attack, defense, level, race, attribute = records[source]
        vectors[index] = card_features(card_type, attack, defense, level, race, attribute, setcode)
    metadata = {
        "version": CARD_EXACT_SCHEMA,
        "layout": CARD_EXACT_LAYOUT_SCHEMA,
        "segments": [[name, width] for name, width in CARD_EXACT_LAYOUT],
        "values": list(VALUE_NAMES),
        "dim": CARD_EXACT_DIM,
        "setcode_hash": HASH_SPEC.replace("mod-dim", f"mod-{SETCODE_DIM}"),
        "setcode_tokens": ["set:%04x", "setbase:%03x"],
        "artwork_variant_rule": f"alias != 0 and |code - alias| < {ARTWORK_OFFSET} "
                                f"and code != {BLACK_LUSTER_SOLDIER_2}",
        "artwork_variants": variants,
        "card_count": len(passcodes),
        "cards_db_sha256": digest,
        "key_sha256": key_sha256(passcodes),
    }
    return CardExactTable(np.asarray(passcodes, np.int64), vectors, metadata)


def artifact_bytes(table: CardExactTable) -> bytes:
    return npz_bytes((
        ("passcodes", table.passcodes.astype(np.int64)),
        ("vectors", table.vectors.astype(np.float32)),
        ("meta", np.asarray(json.dumps(table.metadata, sort_keys=True))),
    ))


def write_artifact(path: str | Path, table: CardExactTable) -> str:
    return atomic_write(path, artifact_bytes(table))


def validate_vectors(vectors: np.ndarray) -> str | None:
    """Why a table's rows break the layout, or ``None``.

    Bits and known flags are 0 or 1; a value whose flag is 0 is 0; link arrows
    only on rows flagged as link; the setcode bag is empty or unit length.
    """
    if vectors.ndim != 2 or vectors.shape[1] != CARD_EXACT_DIM:
        return f"rows must have {CARD_EXACT_DIM} columns"
    binary = np.concatenate([
        vectors[:, SEGMENT_OFFSETS[name]:SEGMENT_OFFSETS[name] + width]
        for name, width in CARD_EXACT_LAYOUT if name not in ("values", "setcodes")
    ], axis=1)
    if not np.isin(binary, (0.0, 1.0)).all():
        return "bit and flag columns must be 0 or 1"
    values = vectors[:, SEGMENT_OFFSETS["values"]:SEGMENT_OFFSETS["values"] + 7]
    known = vectors[:, SEGMENT_OFFSETS["known"]:SEGMENT_OFFSETS["known"] + 7]
    if (values[known == 0] != 0).any() or (np.abs(values) > 2).any():
        return "a value without its known flag is nonzero, or a value is outside [-2, 2]"
    arrows = vectors[:, SEGMENT_OFFSETS["link_arrows"]:SEGMENT_OFFSETS["link_arrows"] + 8]
    if (arrows.any(axis=1) & (known[:, VALUE_NAMES.index("link_rating")] == 0)).any():
        return "link arrows on a row without a link rating"
    bag = vectors[:, SEGMENT_OFFSETS["setcodes"]:]
    norms = np.linalg.norm(bag, axis=1)
    if not (np.isclose(norms, 0.0) | np.isclose(norms, 1.0, atol=1e-5)).all():
        return "setcode bags must be empty or unit length"
    return None

