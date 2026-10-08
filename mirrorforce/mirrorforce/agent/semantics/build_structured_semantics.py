from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path

import numpy as np


SCHEMA = "structured-semantics-v1"
TYPE_FLAGS = (
    0x1, 0x2, 0x4, 0x10, 0x20, 0x40, 0x80, 0x100, 0x200, 0x400,
    0x800, 0x1000, 0x2000, 0x4000, 0x10000, 0x20000, 0x40000,
    0x80000, 0x100000, 0x200000, 0x400000, 0x800000, 0x1000000,
    0x2000000, 0x4000000,
)
RACE_FLAGS = tuple(1 << index for index in range(26))
ATTRIBUTE_FLAGS = tuple(1 << index for index in range(7))
TYPE_MONSTER = 0x1
TYPE_SPELL = 0x2
TYPE_TRAP = 0x4
TYPE_XYZ = 0x800000
TYPE_PENDULUM = 0x1000000
TYPE_LINK = 0x4000000

FUNCTION_START = re.compile(
    r"(?m)^function\s+([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)"
    r"\(([^)]*)\)"
)
EFFECT_CREATE = re.compile(
    r"(?m)^(?:local\s+)?(e[0-9]+)\s*=\s*"
    r"(Effect\.CreateEffect\([^\n]+\)|e[0-9]+:Clone\(\))$"
)
SETTER = re.compile(
    r"(?m)^(e[0-9]+):Set([A-Za-z0-9_]+)\((.*)\)$"
)
API_CALL = re.compile(
    r"\b(Duel|Effect|Card|Group|aux)[.:]([A-Za-z_][A-Za-z0-9_]*)"
)
CONSTANT = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
NUMBER = re.compile(r"(?<![A-Za-z0-9_])(-?[0-9]+)(?![A-Za-z0-9_])")
HEX_NUMBER = re.compile(r"\b0x[0-9A-Fa-f]+\b")
CALLBACK_SETTERS = {
    "Condition": "condition",
    "Cost": "cost",
    "Target": "target",
    "Operation": "operation",
    "Value": "value",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen exact-CDB and effect-level Lua semantics."
    )
    parser.add_argument("--cdb", type=Path, required=True)
    parser.add_argument("--script-dir", type=Path, required=True)
    parser.add_argument("--code-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-effects", type=int, default=16)
    parser.add_argument("--lua-dim", type=int, default=64)
    return parser.parse_args()


def read_code_list(path: Path) -> list[int]:
    codes: list[int] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if not fields:
            continue
        try:
            code = int(fields[0])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid card code"
            ) from exc
        if code <= 0:
            raise ValueError(f"{path}:{line_number}: card code must be positive")
        codes.append(code)
    if len(codes) != len(set(codes)):
        raise ValueError("code list contains duplicate card IDs")
    return codes


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_script_path(
    script_dir: Path,
    code: int,
    aliases: dict[int, int],
) -> tuple[Path | None, int | None]:
    current = code
    visited: set[int] = set()
    while current > 0 and current not in visited:
        visited.add(current)
        script_path = script_dir / f"c{current}.lua"
        if script_path.is_file():
            return script_path, current
        current = aliases.get(current, 0)
    return None, None


def bit_features(value: int, flags: tuple[int, ...]) -> list[float]:
    return [float(bool(value & flag)) for flag in flags]


def known_stat(value: int, scale: float) -> tuple[float, float]:
    if value < 0:
        return 0.0, 0.0
    return max(-2.0, min(2.0, value / scale)), 1.0


def exact_cdb_features(row: tuple[int, int, int, int, int, int]) -> np.ndarray:
    card_type, attack, defense, level_info, race, attribute = row
    is_monster = bool(card_type & TYPE_MONSTER)
    is_xyz = bool(card_type & TYPE_XYZ)
    is_link = bool(card_type & TYPE_LINK)
    is_pendulum = bool(card_type & TYPE_PENDULUM)
    raw_level = level_info & 0xFF
    left_scale = (level_info >> 24) & 0xFF
    right_scale = (level_info >> 16) & 0xFF

    kinds = [
        float(is_monster),
        float(bool(card_type & TYPE_SPELL)),
        float(bool(card_type & TYPE_TRAP)),
        float(not bool(card_type & (TYPE_MONSTER | TYPE_SPELL | TYPE_TRAP))),
    ]
    numeric_pairs = [
        known_stat(attack, 10000.0) if is_monster else (0.0, 0.0),
        known_stat(defense, 10000.0)
        if is_monster and not is_link
        else (0.0, 0.0),
        (raw_level / 13.0, 1.0)
        if is_monster and not is_xyz and not is_link
        else (0.0, 0.0),
        (raw_level / 13.0, 1.0) if is_xyz else (0.0, 0.0),
        (raw_level / 8.0, 1.0) if is_link else (0.0, 0.0),
        (left_scale / 13.0, 1.0) if is_pendulum else (0.0, 0.0),
        (right_scale / 13.0, 1.0) if is_pendulum else (0.0, 0.0),
        (defense / 511.0, 1.0) if is_link and defense >= 0 else (0.0, 0.0),
    ]
    numeric = [pair[0] for pair in numeric_pairs]
    known = [pair[1] for pair in numeric_pairs]
    return np.asarray(
        kinds
        + bit_features(card_type, TYPE_FLAGS)
        + bit_features(race, RACE_FLAGS)
        + bit_features(attribute, ATTRIBUTE_FLAGS)
        + numeric
        + known,
        dtype=np.float32,
    )


def canonicalize_lua(text: str, card_code: int) -> str:
    normalized = text.replace("\ufeff", "").replace("\r\n", "\n").replace(
        "\r", "\n"
    )
    normalized = re.sub(r"--\[\[.*?\]\]", "", normalized, flags=re.DOTALL)
    lines: list[str] = []
    for raw_line in normalized.splitlines():
        line = strip_line_comment(raw_line).strip()
        if line:
            lines.append(re.sub(r"\s+", " ", line))
    normalized = "\n".join(lines)
    normalized = re.sub(rf"\bc{card_code}\b", "card<SELF>", normalized)
    normalized = re.sub(rf"\b{card_code}\b", "<SELF>", normalized)
    normalized = re.sub(
        r"\b(?:[1-9][0-9]{6,9})\b", "<CARD_REF>", normalized
    )

    def replace_setcode(match: re.Match[str]) -> str:
        digest = hashlib.blake2b(
            match.group(0).lower().encode(), digest_size=2
        ).hexdigest()
        return f"<ARCH_{digest}>"

    return HEX_NUMBER.sub(replace_setcode, normalized)


def strip_line_comment(line: str) -> str:
    quote = ""
    escaped = False
    index = 0
    while index < len(line):
        character = line[index]
        if escaped:
            escaped = False
        elif character == "\\" and quote:
            escaped = True
        elif character in ("'", '"'):
            quote = "" if quote == character else character if not quote else quote
        elif not quote and line[index:index + 2] == "--":
            return line[:index]
        index += 1
    return line


def function_bodies(canonical: str) -> dict[str, str]:
    matches = list(FUNCTION_START.finditer(canonical))
    bodies: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(
            canonical
        )
        body = canonical[start:end].strip()
        if body.endswith("end"):
            body = body[:-3].rstrip()
        bodies[match.group(2)] = body
    return bodies


def effect_units(canonical: str) -> list[tuple[str, str]]:
    bodies = function_bodies(canonical)
    initial = bodies.get("initial_effect", "")
    creations = list(EFFECT_CREATE.finditer(initial))
    units: list[tuple[str, str]] = []
    for index, creation in enumerate(creations):
        variable = creation.group(1)
        end = creations[index + 1].start() if index + 1 < len(creations) else len(
            initial
        )
        segment = initial[creation.start():end]
        parts = [f"role:registration\n{segment}"]
        for setter_variable, method, expression in SETTER.findall(segment):
            if setter_variable != variable:
                continue
            role = CALLBACK_SETTERS.get(method)
            if role is None:
                continue
            callback_name = expression.split(".")[-1].strip()
            callback_name = re.sub(r"[^A-Za-z0-9_].*$", "", callback_name)
            callback = bodies.get(callback_name)
            if callback:
                parts.append(f"role:{role}\n{callback}")
        units.append((f"effect_{index}", "\n".join(parts)))
    if not units and canonical:
        units.append(("whole_script_fallback", canonical))
    return units


def semantic_tokens(unit: str) -> list[str]:
    tokens: list[str] = []
    current_role = "effect"
    for line in unit.splitlines():
        if line.startswith("role:"):
            current_role = line.split(":", 1)[1]
            tokens.append(f"role:{current_role}")
            continue
        for owner, method in API_CALL.findall(line):
            tokens.append(f"{current_role}:api:{owner}.{method}")
        for constant in CONSTANT.findall(line):
            tokens.append(f"{current_role}:const:{constant}")
        for setter in re.findall(r":Set([A-Za-z0-9_]+)\(", line):
            tokens.append(f"setter:{setter}")
        for number_text in NUMBER.findall(line):
            value = int(number_text)
            if abs(value) >= 1_000_000:
                continue
            if abs(value) <= 64:
                bucket = str(value)
            elif abs(value) <= 1000:
                bucket = f"pow2:{int(math.log2(abs(value)))}"
            else:
                bucket = f"large:{int(math.log10(abs(value)))}"
            tokens.append(f"{current_role}:number:{bucket}")
        for keyword in ("and", "or", "not", "if", "then", "return"):
            if re.search(rf"\b{keyword}\b", line):
                tokens.append(f"{current_role}:keyword:{keyword}")
    return tokens


def hash_embedding(tokens: list[str], dimension: int) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    for token in tokens:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = value % dimension
        sign = 1.0 if (value >> 63) == 0 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def build_tables(
    cdb: Path,
    script_dir: Path,
    codes: list[int],
    max_effects: int,
    lua_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if max_effects < 2:
        raise ValueError("max_effects must be at least 2")
    if lua_dim < 8:
        raise ValueError("lua_dim must be at least 8")

    connection = sqlite3.connect(str(cdb))
    try:
        records = {
            int(row[0]): (
                int(row[1]),
                tuple(int(value) for value in row[2:]),
            )
            for row in connection.execute(
                "SELECT id, alias, type, atk, def, level, race, attribute "
                "FROM datas"
            )
        }
    finally:
        connection.close()
    aliases = {code: record[0] for code, record in records.items()}
    rows = {code: record[1] for code, record in records.items()}

    sample = exact_cdb_features((0, 0, 0, 0, 0, 0))
    cdb_exact = np.zeros((len(codes) + 1, sample.shape[0]), dtype=np.float32)
    lua_effects = np.zeros(
        (len(codes) + 1, max_effects, lua_dim), dtype=np.float16
    )
    lua_mask = np.zeros((len(codes) + 1, max_effects), dtype=np.uint8)

    missing_cdb: list[int] = []
    missing_lua: list[int] = []
    effect_counts: list[int] = []
    overflow_cards = 0
    alias_lua_fallbacks = 0
    script_digest = hashlib.sha256()

    for semantic_id, code in enumerate(codes, start=1):
        row = rows.get(code)
        if row is None:
            missing_cdb.append(code)
        else:
            cdb_exact[semantic_id] = exact_cdb_features(row)

        script_path, script_code = resolve_script_path(script_dir, code, aliases)
        if script_path is None or script_code is None:
            missing_lua.append(code)
            script_digest.update(f"{code}:missing\n".encode())
            continue
        if script_code != code:
            alias_lua_fallbacks += 1
        text = script_path.read_text(encoding="utf-8", errors="replace")
        script_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        script_digest.update(f"{code}:{script_code}:{script_hash}\n".encode())
        canonical = canonicalize_lua(text, script_code)
        units = effect_units(canonical)
        effect_counts.append(len(units))
        if len(units) > max_effects:
            overflow_cards += 1
            retained = units[: max_effects - 1]
            merged = "\n".join(unit for _, unit in units[max_effects - 1:])
            units = retained + [("overflow_aggregate", merged)]
        for effect_index, (_, unit) in enumerate(units):
            lua_effects[semantic_id, effect_index] = hash_embedding(
                semantic_tokens(unit), lua_dim
            ).astype(np.float16)
            lua_mask[semantic_id, effect_index] = 1

    metadata: dict[str, object] = {
        "schema": SCHEMA,
        "encoder": "api-aware-effect-ir-feature-hash-v1",
        "card_count": len(codes),
        "cdb_feature_dim": int(cdb_exact.shape[1]),
        "max_effects": max_effects,
        "lua_dim": lua_dim,
        "missing_cdb_count": len(missing_cdb),
        "missing_lua_count": len(missing_lua),
        "alias_lua_fallback_count": alias_lua_fallbacks,
        "effect_overflow_card_count": overflow_cards,
        "max_observed_effect_count": max(effect_counts, default=0),
        "mean_effect_count": float(np.mean(effect_counts)) if effect_counts else 0.0,
        "cdb_sha256": file_sha256(cdb),
        "script_manifest_sha256": script_digest.hexdigest(),
        "script_resolution": "direct-then-cdb-alias-chain-v1",
        "code_list_sha256": hashlib.sha256(
            "\n".join(str(code) for code in codes).encode()
        ).hexdigest(),
        "privacy": {
            "card_names_in_features": False,
            "raw_card_codes_in_features": False,
            "raw_setcodes_in_features": False,
            "self_code_replacement": "<SELF>",
            "external_card_code_replacement": "<CARD_REF>",
        },
    }
    return cdb_exact, lua_effects, lua_mask, metadata


def main() -> None:
    args = parse_args()
    codes = read_code_list(args.code_list)
    cdb_exact, lua_effects, lua_mask, metadata = build_tables(
        args.cdb,
        args.script_dir,
        codes,
        args.max_effects,
        args.lua_dim,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cdb_exact=cdb_exact,
            lua_effects=lua_effects,
            lua_mask=lua_mask,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(args.output)
    print(json.dumps(metadata, sort_keys=True))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cdb_exact_shape": list(cdb_exact.shape),
                "lua_effects_shape": list(lua_effects.shape),
                "lua_mask_shape": list(lua_mask.shape),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
