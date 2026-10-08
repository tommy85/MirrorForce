"""Build deterministic CDB/Lua features, never import learned checkpoints."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sqlite3
from types import ModuleType
from urllib.parse import quote

import numpy as np

from mirrorforce.common.sidecar_io import canonical, digest, publish_immutable, read_regular, sha256_file, sha256_tree


SCHEMA = "mirrorforce_structured_rule_tables/v1"
REFERENCE_BUILDER_SHA256 = "d02e3572176c9bd5fa5186702e03d534ec77225596c90575119ed06224d65bab"


def reference_builder(path: Path):
    """Execute only the reviewed standalone builder's exact captured bytes."""
    raw = read_regular(path, "reviewed rule-feature builder")
    if digest(raw) != REFERENCE_BUILDER_SHA256:
        raise ValueError("rule-feature builder differs from the reviewed reference")
    module = ModuleType("_mf_structured_rules_d02e3572")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def card_codes(cdb: Path):
    connection = sqlite3.connect("file:" + quote(str(cdb.resolve())) + "?mode=ro", uri=True)
    try:
        codes = [int(row[0]) for row in connection.execute("SELECT id FROM datas ORDER BY id")]
    finally:
        connection.close()
    if not codes or len(codes) > 65535 or len(codes) != len(set(codes)) \
            or any(not 0 < code <= 0xFFFFFFFF for code in codes):
        raise ValueError("card vocabulary needs unique positive uint32 codes fitting packed uint16 row IDs")
    return codes


def build(cdb: Path, scripts: Path, builder_path: Path, out: Path):
    if not cdb.is_file() or not scripts.is_dir() or out == scripts or scripts in out.parents:
        raise ValueError("rule inputs must exist and outputs must be outside the script tree")
    builder = reference_builder(builder_path)
    source = {"cdb_sha256": sha256_file(cdb), "scripts_sha256": sha256_tree(scripts),
        "builder_sha256": REFERENCE_BUILDER_SHA256}
    codes = card_codes(cdb)
    cdb_exact, lua_effects, lua_mask, metadata = builder.build_tables(cdb, scripts, codes, 16, 64)
    if metadata["missing_cdb_count"] or cdb_exact.shape != (len(codes) + 1, 78) \
            or lua_effects.shape != (len(codes) + 1, 16, 64) or lua_mask.shape != (len(codes) + 1, 16) \
            or not np.isfinite(cdb_exact).all() or not np.isfinite(lua_effects).all() \
            or np.any(cdb_exact[0]) or np.any(lua_effects[0]) or np.any(lua_mask[0]):
        raise ValueError("deterministic rule tables have missing cards, wrong shapes or a nonempty unknown row")
    if source != {"cdb_sha256": sha256_file(cdb), "scripts_sha256": sha256_tree(scripts),
            "builder_sha256": digest(read_regular(builder_path, "reviewed rule-feature builder"))}:
        raise ValueError("rule inputs changed while features were built")
    code_rows = np.asarray([0, *codes], dtype=np.uint32)
    description = {"schema": SCHEMA, "source": source, "reference_metadata": metadata,
        "row_order": "unknown-zero-then-ascending-card-code", "code_rows_sha256": digest(code_rows.tobytes()),
        "learned_weights": False, "printed_effect_unit_alignment": "unknown-pool-all-units"}
    stream = io.BytesIO()
    np.savez_compressed(stream, codes=code_rows, cdb_exact=cdb_exact, lua_effects=lua_effects,
        lua_mask=lua_mask, metadata=np.asarray(canonical(description).decode("ascii")))
    raw = stream.getvalue()
    name = "tables-" + digest(raw) + ".npz"
    out.mkdir(parents=True, exist_ok=True)
    publish_immutable(out / name, raw)
    result = {**description, "artifact": {"path": name, "sha256": digest(raw), "bytes": len(raw)},
        "shapes": {"cdb": list(cdb_exact.shape), "lua": list(lua_effects.shape), "lua_mask": list(lua_mask.shape)}}
    path = out / ("rules-" + digest(result) + ".json")
    publish_immutable(path, canonical(result))
    return path, result


def load(path: Path, checksum: str):
    """Read facts only; unknown schemas, forged arrays and learned-weight files fail closed."""
    raw = read_regular(path, "structured rule tables")
    if digest(raw) != checksum:
        raise ValueError("rule-table checksum differs")
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        if set(arrays.files) != {"codes", "cdb_exact", "lua_effects", "lua_mask", "metadata"}:
            raise ValueError("rule tables contain unexpected arrays")
        metadata = json.loads(str(arrays["metadata"].item()))
        if metadata.get("schema") != SCHEMA or metadata.get("learned_weights") is not False \
                or metadata.get("printed_effect_unit_alignment") != "unknown-pool-all-units":
            raise ValueError("unsupported rule-table origin or effect alignment")
        codes, cdb, lua, mask = (arrays[name].copy() for name in ("codes", "cdb_exact", "lua_effects", "lua_mask"))
    if codes.ndim != 1 or codes.dtype != np.uint32 or codes.size < 2 or codes.size > 65536 \
            or codes[0] != 0 or not (codes[1:] > codes[:-1]).all() \
            or digest(codes.tobytes()) != metadata["code_rows_sha256"] \
            or cdb.shape != (codes.size, 78) or cdb.dtype != np.float32 \
            or lua.shape != (codes.size, 16, 64) or lua.dtype != np.float16 \
            or mask.shape != (codes.size, 16) or mask.dtype != np.uint8 or (mask > 1).any() \
            or not np.isfinite(cdb).all() or not np.isfinite(lua).all() \
            or np.any(cdb[0]) or np.any(lua[0]) or np.any(mask[0]):
        raise ValueError("rule-table rows, types or unknown-card masking differ")
    return codes, cdb, lua, mask, metadata
