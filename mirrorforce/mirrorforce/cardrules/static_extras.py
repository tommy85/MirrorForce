"""Static per-card-row tables the runtime model adds beside CDB78/Lua: facts only, no learned weights.

* ``card extras`` (``mirrorforce_card_extras/v2``): the eight printed link arrows and the 32-slot setcode
  bag of the exact card channel (``common/card_exact.py``), plus each card's four raw 16-bit series codes (for
  exact same-series tests, which a hashed bag cannot decide), row-aligned with a registered rule table and
  read from the same card database the rule table pins.
* ``effect-unit evidence`` (``mirrorforce_effect_unit_evidence/v1``): for each card row and printed
  description ordinal 0..13, the set of the reviewed builder's Lua effect units that provably carry it,
  as a 16-bit mask; zero means unknown and the model keeps pooling every unit. A unit is bound only by a
  literal ``SetDescription(aux.Stringid(self, n))`` of its own variable (or inherited through ``Clone()``),
  and only when every use of that string in the whole script is such a line. Registration order is never
  used; any dynamic, reused or out-of-segment description makes the ordinal (or the card) unknown.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote

import numpy as np

from mirrorforce.common import card_exact
from mirrorforce.common.sidecar_io import canonical, digest, publish_immutable, read_regular, sha256_file, sha256_tree
from . import rules

EXTRAS_SCHEMA = "mirrorforce_card_extras/v2"
SETCODE_SLOTS = 4                  # a 64-bit CDB setcode holds up to four 16-bit series codes
EVIDENCE_SCHEMA = "mirrorforce_effect_unit_evidence/v1"
EXTRA_WIDTH = 40
ORDINALS = 14                      # actions.cc UnpackDesc rejects description ordinals from 14 on
MAX_UNITS = 16

# Why an ordinal is (not) bound; recorded per cell for audits.
BOUND, NOT_SET, NO_SCRIPT, ALIAS_SCRIPT, WHOLE_SCRIPT, DYNAMIC, REUSED, MULTI_SET, OUTSIDE_SEGMENT, OVERFLOW, \
    UNIT_MISMATCH = range(11)
REASONS = ("bound", "not_set", "no_script", "alias_script", "whole_script", "dynamic_description",
           "reused_string", "unit_sets_twice", "setter_outside_segment", "overflow_unit", "unit_count_mismatch")

_STRINGID = re.compile(r"aux\.Stringid\(\s*([^,()]+?)\s*,\s*([^()]*?)\s*\)")
_SET_DESCRIPTION = re.compile(r"^(\w+):SetDescription\((.*)\)$")
_ANY_SET_DESCRIPTION = re.compile(r"(\w+):SetDescription\(")
_GETID = re.compile(r"^local\s+\w+\s*,\s*id\b[^=\n]*=\s*GetID\(\)", re.M)
_SELF_ALIAS = re.compile(r"^local\s+(\w+)\s*=\s*<SELF>\s*$", re.M)


def _rule_inputs(rules_path: Path, rules_sha256: str):
    codes, _, _, _, metadata = rules.load(rules_path, rules_sha256)
    return codes, metadata


def build_extras(rules_path: Path, rules_sha256: str, cards_db: Path, out: Path):
    codes, metadata = _rule_inputs(rules_path, rules_sha256)
    if sha256_file(cards_db) != metadata["source"]["cdb_sha256"]:
        raise ValueError("card extras must read the card database the rule table pins")
    table = card_exact.build_table(cards_db)
    arrows = card_exact.SEGMENT_OFFSETS["link_arrows"]
    setcodes = card_exact.SEGMENT_OFFSETS["setcodes"]
    connection = sqlite3.connect("file:" + quote(str(cards_db.resolve())) + "?mode=ro", uri=True)
    try:
        records = {int(code): (int(alias), int(setcode))
                   for code, alias, setcode in connection.execute("SELECT id, alias, setcode FROM datas")}
    finally:
        connection.close()
    rows = np.zeros((len(codes), EXTRA_WIDTH), np.float32)
    series = np.zeros((len(codes), SETCODE_SLOTS), np.uint16)
    for index, code in enumerate(codes[1:].tolist(), start=1):
        vector = table.row(int(code))
        rows[index, :8] = vector[arrows:arrows + 8]
        rows[index, 8:] = vector[setcodes:setcodes + card_exact.SETCODE_DIM]
        alias, value = records[int(code)]
        if card_exact.is_artwork_variant(int(code), alias) and alias in records:
            value = records[alias][1]   # the same source row the exact channel uses
        series[index] = [(value >> (16 * slot)) & 0xFFFF for slot in range(SETCODE_SLOTS)]
    description = {"schema": EXTRAS_SCHEMA, "rules_sha256": rules_sha256,
                   "code_rows_sha256": metadata["code_rows_sha256"], "cdb_sha256": metadata["source"]["cdb_sha256"],
                   "columns": [["link_arrows", 8], ["setcodes", card_exact.SETCODE_DIM]],
                   "series": {"slots": SETCODE_SLOTS, "bits": 16, "archetype": "low 12 bits (IsSetCard base)"},
                   "source_layout": table.metadata["layout"], "setcode_hash": table.metadata["setcode_hash"],
                   "artwork_variant_rule": table.metadata["artwork_variant_rule"], "learned_weights": False}
    return _publish(out, "card-extras", description, extras=rows, series=series)


def load_extras(path: Path, checksum: str, *, rules_sha256: str, codes: np.ndarray):
    """(``N×40`` float extras, ``N×4`` uint16 series codes) of a registered rule table."""
    metadata, arrays = _read(path, checksum, {"extras", "series"}, EXTRAS_SCHEMA)
    rows, series = arrays["extras"], arrays["series"]
    if metadata.get("rules_sha256") != rules_sha256 or metadata.get("code_rows_sha256") != digest(codes.tobytes()) \
            or rows.shape != (len(codes), EXTRA_WIDTH) or rows.dtype != np.float32 or not np.isfinite(rows).all() \
            or rows[0].any() or not np.isin(rows[:, :8], (0.0, 1.0)).all() \
            or series.shape != (len(codes), SETCODE_SLOTS) or series.dtype != np.uint16 or series[0].any():
        raise ValueError("card extras differ from the registered rule rows or their layout")
    return rows, series


def _literal_desc(expression: str, self_names: set[str]):
    """``n`` when ``expression`` is exactly a literal self-owned Stringid, else None."""
    match = _STRINGID.fullmatch(expression.strip())
    if match and match.group(1) in self_names and match.group(2).isdigit():
        return int(match.group(2))
    return None


def card_evidence(builder, canonical_text: str):
    """(mask per ordinal, reason per ordinal) of one card's canonical script."""
    masks, reasons = [0] * ORDINALS, [NOT_SET] * ORDINALS
    units = builder.effect_units(canonical_text)
    if not units or units[0][0] == "whole_script_fallback":
        return masks, [WHOLE_SCRIPT] * ORDINALS
    bodies = builder.function_bodies(canonical_text)
    initial = bodies.get("initial_effect", "")
    creations = list(builder.EFFECT_CREATE.finditer(initial))
    if len(creations) != len(units):
        return masks, [UNIT_MISMATCH] * ORDINALS
    self_names = {"<SELF>"}
    if _GETID.search(canonical_text):
        self_names.add("id")
    aliases = _SELF_ALIAS.findall(canonical_text)
    self_names.update(name for name in set(aliases) if aliases.count(name) == 1)
    uses = {}
    for owner, ordinal in _STRINGID.findall(canonical_text):
        if owner.strip() not in self_names:
            continue
        if not ordinal.strip().isdigit():
            return masks, [DYNAMIC] * ORDINALS
        uses[int(ordinal)] = uses.get(int(ordinal), 0) + 1
    latest, effective, lines_of, multi = {}, [], {}, set()
    for index, creation in enumerate(creations):
        variable = creation.group(1)
        end = creations[index + 1].start() if index + 1 < len(creations) else len(initial)
        segment = initial[creation.start():end]
        source = re.fullmatch(r"(\w+):Clone\(\)", creation.group(2))
        inherited = effective[latest[source.group(1)]] if source and source.group(1) in latest else None
        own = []
        for line in segment.splitlines():
            for setter in _ANY_SET_DESCRIPTION.findall(line):
                if setter != variable:
                    return masks, [OUTSIDE_SEGMENT] * ORDINALS
            match = _SET_DESCRIPTION.match(line.strip())
            if match and match.group(1) == variable:
                own.append(_literal_desc(match.group(2), self_names))
                lines_of.setdefault(own[-1], 0)
                lines_of[own[-1]] = lines_of[own[-1]] + 1
            elif _ANY_SET_DESCRIPTION.search(line):
                return masks, [OUTSIDE_SEGMENT] * ORDINALS
        if len(own) > 1:
            multi.update(value for value in own if value is not None)
        effective.append(own[-1] if own else inherited)
        latest[variable] = index
    # Descriptions set outside the unit segments of initial_effect make every ordinal they could name unknown.
    for name, body in bodies.items():
        if name != "initial_effect" and _ANY_SET_DESCRIPTION.search(body):
            for owner, ordinal in _STRINGID.findall(body):
                if owner.strip() in self_names and ordinal.strip().isdigit():
                    uses[int(ordinal)] = uses.get(int(ordinal), 0) + 1
    overflow = len(units) > MAX_UNITS
    for ordinal in range(ORDINALS):
        members = [index for index, value in enumerate(effective) if value == ordinal]
        if not members:
            reasons[ordinal] = NOT_SET
        elif ordinal in multi:
            reasons[ordinal] = MULTI_SET
        elif uses.get(ordinal, 0) != lines_of.get(ordinal, 0):
            reasons[ordinal] = REUSED
        elif overflow and any(index >= MAX_UNITS - 1 for index in members):
            reasons[ordinal] = OVERFLOW
        else:
            reasons[ordinal] = BOUND
            masks[ordinal] = sum(1 << index for index in members)
    return masks, reasons


def build_evidence(rules_path: Path, rules_sha256: str, cards_db: Path, scripts: Path, builder_path: Path, out: Path):
    codes, metadata = _rule_inputs(rules_path, rules_sha256)
    builder = rules.reference_builder(builder_path)
    source = {"cdb_sha256": sha256_file(cards_db), "scripts_sha256": sha256_tree(scripts),
              "builder_sha256": rules.REFERENCE_BUILDER_SHA256}
    if source != metadata["source"]:
        raise ValueError("effect-unit evidence must read the database, scripts and builder the rule table pins")
    connection = sqlite3.connect("file:" + quote(str(cards_db.resolve())) + "?mode=ro", uri=True)
    try:
        aliases = {int(code): int(alias) for code, alias in connection.execute("SELECT id, alias FROM datas")}
    finally:
        connection.close()
    masks = np.zeros((len(codes), ORDINALS), np.uint16)
    reasons = np.full((len(codes), ORDINALS), NO_SCRIPT, np.uint8)  # row zero is the unknown card
    for index, code in enumerate(codes[1:].tolist(), start=1):
        path, script_code = builder.resolve_script_path(scripts, int(code), aliases)
        if path is None:
            reasons[index] = NO_SCRIPT
            continue
        if script_code != int(code):
            reasons[index] = ALIAS_SCRIPT
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        row_masks, row_reasons = card_evidence(builder, builder.canonicalize_lua(text, script_code))
        masks[index], reasons[index] = row_masks, row_reasons
    if source != {"cdb_sha256": sha256_file(cards_db), "scripts_sha256": sha256_tree(scripts),
                  "builder_sha256": digest(read_regular(builder_path, "reviewed rule-feature builder"))}:
        raise ValueError("rule inputs changed while the evidence was built")
    counts = {name: int((reasons[1:] == value).sum()) for value, name in enumerate(REASONS)}
    description = {"schema": EVIDENCE_SCHEMA, "rules_sha256": rules_sha256,
                   "code_rows_sha256": metadata["code_rows_sha256"], "source": source, "ordinals": ORDINALS,
                   "max_units": MAX_UNITS, "reasons": list(REASONS), "reason_counts": counts,
                   "law": "literal_own_setdescription_every_use_accounted/v1", "registration_order_used": False,
                   "learned_weights": False}
    return _publish(out, "effect-units", description, masks=masks, reasons=reasons)


def load_evidence(path: Path, checksum: str, *, rules_sha256: str, codes: np.ndarray, units: int) -> np.ndarray:
    metadata, arrays = _read(path, checksum, {"masks", "reasons"}, EVIDENCE_SCHEMA)
    masks, reasons = arrays["masks"], arrays["reasons"]
    if metadata.get("rules_sha256") != rules_sha256 or metadata.get("code_rows_sha256") != digest(codes.tobytes()) \
            or metadata.get("registration_order_used") is not False \
            or masks.shape != (len(codes), ORDINALS) or masks.dtype != np.uint16 \
            or reasons.shape != masks.shape or reasons.dtype != np.uint8 or (reasons >= len(REASONS)).any() \
            or masks[0].any() or ((masks != 0) != (reasons == BOUND)).any() \
            or (masks.astype(np.int64) >> units).any():
        raise ValueError("effect-unit evidence differs from the registered rule rows or its law")
    return masks


def _publish(out: Path, prefix: str, description: dict, **arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, metadata=np.asarray(canonical(description).decode("ascii")), **arrays)
    raw = stream.getvalue()
    out.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}-{digest(raw)}.npz"
    publish_immutable(out / name, raw)
    return out / name, {**description, "artifact": {"path": name, "sha256": digest(raw), "bytes": len(raw)}}


def _read(path: Path, checksum: str, names: set[str], schema: str):
    raw = read_regular(path, schema)
    if digest(raw) != checksum:
        raise ValueError(schema + " checksum differs")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != names | {"metadata"}:
            raise ValueError(schema + " contains unexpected arrays")
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {name: archive[name].copy() for name in names}
    if metadata.get("schema") != schema or metadata.get("learned_weights") is not False:
        raise ValueError("unsupported " + schema)
    return metadata, arrays
