"""Hashed role-token representation of each registered effect's Lua logic (R7).

Design: the design notes.  The
frozen Qwen3-Embedding channel over normalised Lua text (``effect_lua``) is
replaced by a bag of role-prefixed tokens, sign-hashed into a fixed width and
L2-normalised.  The idea and the number buckets follow a colleague's
structured-semantics builder; the tokens here are read from this repository's
cardfeat artifacts instead of from a second Lua splitter:

``op_sequences`` (``mirrorforce_op_sequences/v4``)
    Per effect and callback role, the ordered calls with their classified
    arguments.  Local helpers and predicates are already inlined.
``atomic_ops`` (``mirrorforce_atomic_ops/v3``)
    Per effect registration info: activation kind, registration scope, spell
    speed and the declared type/category/property/event/code/zone/timing/reset
    flags, resolved through clones.
``effect_blocks`` (``mirrorforce_effect_blocks/v4``)
    Only the registration segment of each block (the lines before the first
    ``function``) is read, and only its ``vN:SetX(...)`` setter lines, for the
    setter names and the readable arguments the other two artifacts do not
    keep (count-limit codes, target ranges, literal values, callbacks bound to
    ``aux.*`` helpers).

Keys are ``(passcode, effect_index)`` in the shared cardfeat registration
order, identical to the ``effect_lua`` and ``effect_text`` channels.

Token scheme ``mirrorforce_effect_role_tokens/v1`` (one effect):

* registration: ``registration:kind:<activation_kind>``,
  ``registration:scope:<card|duel|unregistered>``, ``registration:speed:<n>``,
  ``registration:const:<FLAG>`` for every declared flag,
  ``registration:zone:<zone>`` and ``role:<callback>`` for every declared
  callback (condition, cost, target, operation, value);
* setters: ``setter:<Name>`` for every setter applied to the effect (a clone
  inherits its parent's).  Setters whose arguments the registration flags
  already carry (Type, Code, Category, Property, Range, Reset, HintTiming) or
  that carry no semantics (Description, Label, LabelObject) stop there.  The
  others add their argument pieces, prefixed ``<role>`` for a callback setter
  (``cost:api:aux.bfgcost``, ``value:number:pow2:8``) and ``setter:<Name>``
  otherwise (``setter:CountLimit:number:1``,
  ``setter:TargetRange:const:LOCATION_MZONE``).  A callback bound to an
  ``aux.*`` helper is visible only here, because op_sequences cannot follow
  it (it marks the role unresolved, ``<role>:unresolved``);
* calls: for each role in engine order and each call, ``<role>:api:<op>``
  where ``<op>`` is ``Duel.<Name>`` for a duel primitive, ``aux.<Name>`` for a
  helper-library call and ``method.<Name>`` for a card/group/effect method,
  followed by the call's argument tokens.

Argument tokens are ``<prefix>:const:<NAME>`` (each name of a ``+``/``|``
composition), ``<prefix>:number:<bucket>``, ``<prefix>:literal:TRUE|FALSE``,
``<prefix>:api:<op>`` for a function passed by reference,
``<prefix>:ref:<SELF>`` for the card's own passcode,
``<prefix>:ref:<CARD_REF>`` for any other passcode-sized integer,
``<prefix>:ref:<SELF_SETCODE>``, ``<prefix>:ref:<SETCODE>`` and
``<prefix>:ref:<HEX>``.  ``V`` (a computed value), ``FN`` (a local function,
already inlined), ``NIL`` and ``@`` (a nested call, listed as its own entry)
add nothing.  Number buckets: ``|x| <= 64`` keeps its value, ``65..1000`` is
``pow2:floor(log2|x|)``, below ``10**6`` it is ``large:floor(log10|x|)`` and
larger values are skipped.  No token names a card or an archetype.

The static ``aux.*`` helper bodies live in the engine's ``utility.lua``, which
op_sequences does not compile, so a helper enters as its ``aux.<Name>`` token
and is not expanded.

Hash ``blake2b64-le/index=mod-dim/sign=bit63/l2``: the 8-byte BLAKE2b digest
of the UTF-8 token read as a little-endian unsigned integer ``h``; the token
adds ``+1`` (bit 63 clear) or ``-1`` (bit 63 set) at ``h mod dim``; a token
repeated ``k`` times adds ``k`` times; the sum is L2-normalised to float32.
Python's salted ``hash()`` is never used.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..cardfeat import (
    ARTIFACT_SCHEMA as ATOMIC_OPS_SCHEMA,
    EFFECT_BLOCKS_SCHEMA,
    OP_SEQUENCES_SCHEMA,
    ROLES,
    AtomicOpsIndex,
    EffectBlock,
    EffectFeature,
    EffectBlockIndex,
    EffectSequence,
    OpCall,
    OpSequenceIndex,
)
from .frozen_npz import atomic_write, npz_bytes

__all__ = [
    "COLLISION_DIMS",
    "DEFAULT_DIM",
    "EFFECT_LUA_HASH_SCHEMA",
    "HASH_SPEC",
    "LuaHashTable",
    "TOKEN_SCHEME",
    "build_table",
    "collision_report",
    "effect_tokens",
    "hash_slot",
    "hash_vectors",
    "number_bucket",
    "write_artifact",
    "write_token_audit",
]


EFFECT_LUA_HASH_SCHEMA = "mirrorforce_effect_lua_hash/v1"
TOKEN_SCHEME = "mirrorforce_effect_role_tokens/v1"
HASH_SPEC = "blake2b64-le/index=mod-dim/sign=bit63/l2"
DEFAULT_DIM = 512
COLLISION_DIMS = (256, 512)

#: Integers at or above this are passcodes or opaque keys, not quantities; the
#: op-sequence compiler uses the same threshold for ``#REF`` / ``#CODE``.
_BIG_INTEGER = 10_000
_SKIPPED_NUMBER = 1_000_000

_CALLBACK_SETTERS = {
    "Condition": "condition",
    "Cost": "cost",
    "Target": "target",
    "Operation": "operation",
    "Value": "value",
}
#: Setters whose arguments the registration flags already carry, or that carry
#: no effect semantics.  They contribute ``setter:<Name>`` only.
_NO_ARGUMENT_SETTERS = frozenset({
    "Category", "Code", "Description", "HintTiming", "Label", "LabelObject",
    "Property", "Range", "Reset", "Type",
})
_REGISTRATION_FLAG_FIELDS = (
    "type_flags", "category_flags", "property_flags", "event_flags",
    "effect_code_flags", "hint_timing_flags", "reset_flags",
)

_REFERENCE_ARGUMENTS = {
    "#SELF": "<SELF>",
    "#REF": "<CARD_REF>",
    "#CODE": "<CARD_REF>",
    "#SELF_SETCODE": "<SELF_SETCODE>",
    "#SETCODE": "<SETCODE>",
    "#HEX": "<HEX>",
}
_SILENT_ARGUMENTS = frozenset({"V", "FN", "NIL"})
_LITERAL_ARGUMENTS = frozenset({"TRUE", "FALSE"})
_INTEGER_ARGUMENT = re.compile(r"^#(-?\d+)$")
_CONSTANT_EXPRESSION = re.compile(r"^[A-Z][A-Z0-9_]*(?:\s*[+|]\s*[A-Z][A-Z0-9_]*)*$")
_CONSTANT_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
_OP_NAME = re.compile(r"^(?:aux:|m:)?[A-Za-z_]\w*$")

_FUNCTION_LINE = re.compile(r"^function\s")
_CREATE_LINE = re.compile(r"^(?:local\s+)?(v\d+)\s*=\s*Effect\.CreateEffect\(")
_CLONE_LINE = re.compile(r"^(?:local\s+)?(v\d+)\s*=\s*(v\d+):Clone\(\)\s*$")
_SETTER_LINE = re.compile(r"^(v\d+):Set([A-Za-z]+)\((.*)\)\s*$")
_REGISTER_CALL = re.compile(r"\b\w+:RegisterEffect\((v\d+)|\bDuel\.RegisterEffect\((v\d+)")
#: Readable pieces of one setter argument expression, scanned left to right.
_EXPRESSION_PIECE = re.compile(
    r"(?P<api>\b(?:aux|Duel|Card|Group|Effect|math|bit)\.[A-Za-z_]\w*)"
    r"|(?P<method>:[A-Za-z_]\w*(?=\())"
    r"|(?P<fn>\bs\.[A-Za-z_]\w*)"
    r"|(?P<hex>\b0[xX][0-9A-Fa-f]+\b)"
    r"|(?P<number>(?<![A-Za-z0-9_.])-?\d+(?![A-Za-z0-9_.]))"
    r"|(?P<self>\bid\b)"
    r"|(?P<literal>\b(?:true|false)\b)"
    r"|(?P<const>\b[A-Z][A-Z0-9_]{2,}\b)"
)
_API_OWNERS = {"aux": "aux", "Duel": "Duel", "Card": "method", "Group": "method",
               "Effect": "method", "math": "math", "bit": "bit"}


class LuaHashBuildError(ValueError):
    """The cardfeat inputs disagree or carry a token the scheme cannot read."""


def number_bucket(value: int) -> str | None:
    """The bucket of one integer, or ``None`` for a skipped (huge) value."""
    magnitude = abs(int(value))
    if magnitude >= _SKIPPED_NUMBER:
        return None
    if magnitude <= 64:
        return str(int(value))
    if magnitude <= 1000:
        return f"pow2:{magnitude.bit_length() - 1}"
    return f"large:{len(str(magnitude)) - 1}"


def render_op(op: str) -> str:
    """``Draw`` -> ``Duel.Draw``, ``aux:X`` -> ``aux.X``, ``m:X`` -> ``method.X``."""
    if not _OP_NAME.fullmatch(op):
        raise LuaHashBuildError(f"unreadable operation name {op!r}")
    if op.startswith("aux:"):
        return "aux." + op[4:]
    if op.startswith("m:"):
        return "method." + op[2:]
    return "Duel." + op


def _integer_tokens(prefix: str, value: int, passcode: int | None) -> list[str]:
    if passcode is not None and int(value) == int(passcode):
        return [f"{prefix}:ref:<SELF>"]
    if abs(int(value)) >= _BIG_INTEGER:
        return [f"{prefix}:ref:<CARD_REF>"]
    bucket = number_bucket(value)
    return [] if bucket is None else [f"{prefix}:number:{bucket}"]


def argument_tokens(prefix: str, argument: str) -> list[str]:
    """Tokens of one classified op-sequence argument; unknown forms fail closed."""
    if argument in _SILENT_ARGUMENTS or argument.startswith("@"):
        return []
    if argument in _LITERAL_ARGUMENTS:
        return [f"{prefix}:literal:{argument}"]
    reference = _REFERENCE_ARGUMENTS.get(argument)
    if reference is not None:
        return [f"{prefix}:ref:{reference}"]
    if argument.startswith("&"):
        return [f"{prefix}:api:{render_op(argument[1:])}"]
    integer = _INTEGER_ARGUMENT.fullmatch(argument)
    if integer is not None:
        # The compiler already turned passcode-sized integers into #REF/#CODE.
        return _integer_tokens(prefix, int(integer.group(1)), None)
    if _CONSTANT_EXPRESSION.fullmatch(argument):
        return [f"{prefix}:const:{name}" for name in _CONSTANT_NAME.findall(argument)]
    raise LuaHashBuildError(f"unclassified op-sequence argument {argument!r}")


def call_tokens(role: str, call: OpCall) -> list[str]:
    tokens = [f"{role}:api:{render_op(call.op)}"]
    for argument in call.args:
        tokens.extend(argument_tokens(role, argument))
    return tokens


def registration_tokens(effect: EffectFeature) -> list[str]:
    tokens = [
        f"registration:kind:{effect.activation_kind}",
        f"registration:scope:{effect.registration}",
        f"registration:speed:{int(effect.spell_speed)}",
    ]
    for field in _REGISTRATION_FLAG_FIELDS:
        tokens.extend(f"registration:const:{flag}" for flag in getattr(effect, field))
    tokens.extend(f"registration:zone:{zone}" for zone in effect.zones)
    tokens.extend(f"role:{callback}" for callback in effect.callbacks)
    return tokens


def expression_tokens(expression: str, passcode: int) -> list[tuple[str, str]]:
    """``(kind, payload)`` pieces of one setter argument, left to right."""
    pieces: list[tuple[str, str]] = []
    for match in _EXPRESSION_PIECE.finditer(expression):
        kind = match.lastgroup
        text = match.group(kind)
        if kind == "api":
            owner, name = text.split(".", 1)
            pieces.append(("api", f"{_API_OWNERS[owner]}.{name}"))
        elif kind == "method":
            pieces.append(("api", "method." + text[1:]))
        elif kind == "fn":
            pieces.append(("fn", "local"))
        elif kind == "hex":
            pieces.append(("ref", "<HEX>"))
        elif kind == "self":
            pieces.append(("ref", "<SELF>"))
        elif kind == "literal":
            pieces.append(("literal", text.upper()))
        elif kind == "const":
            pieces.append(("const", text))
        else:
            value = int(text)
            if value == int(passcode):
                pieces.append(("ref", "<SELF>"))
            elif abs(value) >= _BIG_INTEGER:
                pieces.append(("ref", "<CARD_REF>"))
            else:
                bucket = number_bucket(value)
                if bucket is not None:
                    pieces.append(("number", bucket))
    return pieces


def effect_setters(code: str) -> tuple[tuple[str, str], ...]:
    """The ``(setter name, argument text)`` pairs applied to one block's effect.

    Only the registration segment is read.  A clone starts with a copy of its
    parent's setters; a repeated setter replaces the earlier value, as in Lua.
    The effect is the last variable the segment registers, else the last one
    it creates.
    """
    setters: dict[str, dict[str, str]] = {}
    registered: str | None = None
    created: str | None = None
    for raw in code.split("\n"):
        line = raw.strip()
        if _FUNCTION_LINE.match(raw):
            break
        clone = _CLONE_LINE.match(line)
        if clone is not None:
            setters[clone.group(1)] = dict(setters.get(clone.group(2), {}))
            created = clone.group(1)
            continue
        create = _CREATE_LINE.match(line)
        if create is not None:
            setters[create.group(1)] = {}
            created = create.group(1)
            continue
        setter = _SETTER_LINE.match(line)
        if setter is not None:
            values = setters.setdefault(setter.group(1), {})
            values.pop(setter.group(2), None)
            values[setter.group(2)] = setter.group(3)
            continue
        for register in _REGISTER_CALL.finditer(line):
            registered = register.group(1) or register.group(2)
    effect = registered if registered in setters else created
    return tuple(setters.get(effect, {}).items()) if effect is not None else ()


def setter_tokens(code: str, passcode: int) -> list[str]:
    """``setter:<Name>`` per applied setter, then its argument pieces.

    A callback setter's pieces take the callback role as prefix
    (``cost:api:aux.bfgcost``, ``value:number:pow2:8``); any other setter's
    take ``setter:<Name>`` (``setter:CountLimit:number:1``).  A local-function
    callback of the four op-sequence roles adds nothing: op_sequences already
    lists its calls.
    """
    tokens: list[str] = []
    for name, arguments in effect_setters(code):
        tokens.append(f"setter:{name}")
        if name in _NO_ARGUMENT_SETTERS:
            continue
        role = _CALLBACK_SETTERS.get(name)
        prefix = role if role is not None else f"setter:{name}"
        for kind, payload in expression_tokens(arguments, passcode):
            if kind == "fn" and role is not None and role != "value":
                continue
            tokens.append(f"{prefix}:{kind}:{payload}")
    return tokens


def effect_tokens(
    passcode: int,
    effect: EffectFeature,
    sequence: EffectSequence,
    block: EffectBlock,
) -> list[str]:
    """Every token of one effect, in scheme order (registration, setters, calls)."""
    tokens = registration_tokens(effect)
    tokens.extend(setter_tokens(block.code, passcode))
    unresolved = set(sequence.unresolved_roles)
    for role in ROLES:
        if role in unresolved:
            tokens.append(f"{role}:unresolved")
        for call in sequence.role(role):
            tokens.extend(call_tokens(role, call))
    return tokens


def hash_slot(token: str, dim: int) -> tuple[int, float]:
    """``(index, sign)`` of one token at width ``dim``."""
    value = int.from_bytes(
        hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little"
    )
    return value % int(dim), (1.0 if (value >> 63) == 0 else -1.0)


def hash_vectors(
    token_offsets: np.ndarray,
    token_ids: np.ndarray,
    vocabulary: Sequence[str],
    dim: int,
) -> np.ndarray:
    """``[rows, dim]`` float32 signed-hash sums, L2-normalised per row."""
    slots = [hash_slot(token, dim) for token in vocabulary]
    index = np.asarray([slot[0] for slot in slots], np.int64)
    sign = np.asarray([slot[1] for slot in slots], np.float64)
    rows = len(token_offsets) - 1
    owner = np.repeat(np.arange(rows, dtype=np.int64), np.diff(token_offsets))
    vectors = np.zeros((rows, int(dim)), np.float64)
    ids = np.asarray(token_ids, np.int64)
    np.add.at(vectors, (owner, index[ids]), sign[ids])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    return vectors.astype(np.float32)


@dataclass(frozen=True)
class LuaHashTable:
    """One built table: stable keys, audited tokens, vectors and metadata."""

    passcodes: np.ndarray
    effect_indices: np.ndarray
    token_offsets: np.ndarray
    token_ids: np.ndarray
    vocabulary: tuple[str, ...]
    vectors: np.ndarray
    metadata: Mapping[str, object]

    def tokens(self, row: int) -> tuple[str, ...]:
        start, stop = int(self.token_offsets[row]), int(self.token_offsets[row + 1])
        return tuple(self.vocabulary[int(i)] for i in self.token_ids[start:stop])


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def key_sha256(passcodes: Iterable[int], effect_indices: Iterable[int]) -> str:
    """Same canonical key hash as ``effect_encoder._key_sha256`` over effect identities."""
    canonical = "".join(
        f"{int(code)}:{int(index)}\n" for code, index in zip(passcodes, effect_indices)
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def collision_report(
    token_offsets: np.ndarray,
    token_ids: np.ndarray,
    vocabulary: Sequence[str],
    dims: Sequence[int] = COLLISION_DIMS,
) -> dict[str, dict[str, object]]:
    """Hash collision statistics at each width.

    ``vocabulary``: distinct tokens whose slot another distinct token also uses
    (``expected_shared_fraction`` is the uniform-hash expectation).
    ``within_effect``: the collisions that actually interfere -- two distinct
    tokens of the same effect on one slot -- and how many of those pairs have
    opposite signs and so cancel.  ``vectors``: effects whose token multisets
    differ but whose hashed vectors are identical.
    """
    rows = len(token_offsets) - 1
    vocabulary_size = len(vocabulary)
    multisets: dict[tuple[tuple[int, int], ...], int] = {}
    per_effect: list[tuple[np.ndarray, np.ndarray]] = []
    for row in range(rows):
        start, stop = int(token_offsets[row]), int(token_offsets[row + 1])
        ids, counts = np.unique(np.asarray(token_ids[start:stop], np.int64), return_counts=True)
        per_effect.append((ids, counts))
        key = tuple(zip(ids.tolist(), counts.tolist()))
        multisets.setdefault(key, row)
    report: dict[str, dict[str, object]] = {}
    for dim in dims:
        slots = [hash_slot(token, dim) for token in vocabulary]
        index = np.asarray([slot[0] for slot in slots], np.int64)
        sign = np.asarray([slot[1] for slot in slots], np.int64)
        occupancy = np.bincount(index, minlength=int(dim))
        shared = int((occupancy[index] > 1).sum())
        expected = (
            vocabulary_size * (1.0 - (1.0 - 1.0 / dim) ** (vocabulary_size - 1))
            if vocabulary_size else 0.0
        )
        effects_with_collision = 0
        colliding_tokens = 0
        distinct_tokens = 0
        pairs = 0
        cancelling = 0
        for ids, _counts in per_effect:
            if not len(ids):
                continue
            distinct_tokens += len(ids)
            slot_counts = Counter(index[ids].tolist())
            members = [count for count in slot_counts.values() if count > 1]
            if members:
                effects_with_collision += 1
                colliding_tokens += sum(members)
                by_slot: dict[int, list[int]] = {}
                for token in ids.tolist():
                    by_slot.setdefault(int(index[token]), []).append(int(sign[token]))
                for signs in by_slot.values():
                    positive = signs.count(1)
                    negative = len(signs) - positive
                    pairs += len(signs) * (len(signs) - 1) // 2
                    cancelling += positive * negative
        vectors = hash_vectors(token_offsets, token_ids, vocabulary, dim)
        distinct_vectors = len({vectors[row].tobytes() for row in multisets.values()})
        report[str(int(dim))] = {
            "vocabulary": {
                "tokens": vocabulary_size,
                "slots_used": int((occupancy > 0).sum()),
                "max_tokens_per_slot": int(occupancy.max()) if vocabulary_size else 0,
                "shared_tokens": shared,
                "shared_fraction": round(shared / vocabulary_size, 6) if vocabulary_size else 0.0,
                "expected_shared_fraction": (
                    round(expected / vocabulary_size, 6) if vocabulary_size else 0.0
                ),
            },
            "within_effect": {
                "effects": rows,
                "effects_with_collision": effects_with_collision,
                "effects_with_collision_fraction": round(effects_with_collision / rows, 6) if rows else 0.0,
                "colliding_token_fraction": (
                    round(colliding_tokens / distinct_tokens, 6) if distinct_tokens else 0.0
                ),
                "colliding_pairs": pairs,
                "cancelling_pairs": cancelling,
            },
            "vectors": {
                "distinct_token_multisets": len(multisets),
                "distinct_vectors": distinct_vectors,
                "merged_multisets": len(multisets) - distinct_vectors,
            },
        }
    return report


def _read_input(data: bytes, loader, name: str):
    try:
        payload = json.loads(gzip.decompress(data).decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise LuaHashBuildError(f"{name}: cannot read gzipped JSON: {exc}") from exc
    try:
        return payload, loader(payload)
    except ValueError as exc:
        raise LuaHashBuildError(f"{name}: {exc}") from exc


def build_table(
    *,
    op_sequences: bytes,
    atomic_ops: bytes,
    effect_blocks: bytes,
    dim: int = DEFAULT_DIM,
    collision_dims: Sequence[int] = COLLISION_DIMS,
) -> LuaHashTable:
    """Build the table from the raw bytes of the three cardfeat artifacts.

    The same input bytes always give the same table.  The three inputs must
    come from one script revision and number every card's effects alike.
    """
    if int(dim) < 8:
        raise LuaHashBuildError("hash width must be at least 8")
    sequence_payload, sequences = _read_input(op_sequences, OpSequenceIndex, "op_sequences")
    atomic_payload, atomic = _read_input(atomic_ops, AtomicOpsIndex, "atomic_ops")
    block_payload, blocks = _read_input(effect_blocks, EffectBlockIndex, "effect_blocks")
    revisions = {sequences.script_revision, atomic.script_revision, blocks.script_revision}
    if len(revisions) != 1 or not re.fullmatch(r"[0-9a-f]{40}", sequences.script_revision):
        raise LuaHashBuildError(f"cardfeat inputs disagree on the script revision: {sorted(revisions)}")
    codes = sorted(set(sequences) | set(atomic) | set(blocks))

    vocabulary_ids: dict[str, int] = {}
    effect_token_ids: list[list[int]] = []
    passcodes: list[int] = []
    indices: list[int] = []
    unresolved_roles = 0
    aux_tokens = 0
    for code in codes:
        sequence_card = sequences.get(code)
        atomic_card = atomic.get(code)
        card_blocks = blocks.blocks(code)
        counts = (
            len(sequence_card.effects) if sequence_card is not None else 0,
            len(atomic_card.effects) if atomic_card is not None else 0,
            len(card_blocks),
        )
        if len(set(counts)) != 1:
            raise LuaHashBuildError(
                f"passcode {code}: effect counts differ (op_sequences, atomic_ops, "
                f"effect_blocks) = {counts}"
            )
        for index in range(counts[0]):
            assert sequence_card is not None and atomic_card is not None
            sequence = sequence_card.effects[index]
            tokens = effect_tokens(code, atomic_card.effects[index], sequence, card_blocks[index])
            unresolved_roles += len(sequence.unresolved_roles)
            aux_tokens += sum(1 for token in tokens if ":api:aux." in token)
            effect_token_ids.append([
                vocabulary_ids.setdefault(token, len(vocabulary_ids)) for token in tokens
            ])
            passcodes.append(int(code))
            indices.append(index)
    if not passcodes:
        raise LuaHashBuildError("cardfeat inputs contain no effects")

    # Canonical vocabulary order (sorted), independent of discovery order.
    vocabulary = tuple(sorted(vocabulary_ids))
    remap = np.zeros(len(vocabulary_ids), np.int64)
    for new, token in enumerate(vocabulary):
        remap[vocabulary_ids[token]] = new
    lengths = np.asarray([len(ids) for ids in effect_token_ids], np.int64)
    token_offsets = np.zeros(len(lengths) + 1, np.int64)
    np.cumsum(lengths, out=token_offsets[1:])
    flat = np.fromiter(
        (token for ids in effect_token_ids for token in ids), np.int64, count=int(lengths.sum())
    )
    token_ids = remap[flat].astype(np.int32)
    vectors = hash_vectors(token_offsets, token_ids, vocabulary, dim)
    passcode_array = np.asarray(passcodes, np.int64)
    index_array = np.asarray(indices, np.int32)
    widths = tuple(sorted({int(width) for width in collision_dims} | {int(dim)}))
    collisions = collision_report(token_offsets, token_ids, vocabulary, widths)
    metadata = {
        "version": EFFECT_LUA_HASH_SCHEMA,
        "token_scheme": TOKEN_SCHEME,
        "hash": HASH_SPEC,
        "dim": int(dim),
        "block_count": len(passcodes),
        "card_count": len(set(passcodes)),
        "key_sha256": key_sha256(passcodes, indices),
        "script_revision": sequences.script_revision,
        "op_sequences_schema": sequence_payload["schema"],
        "op_sequences_sha256": hashlib.sha256(op_sequences).hexdigest(),
        "atomic_ops_schema": atomic_payload["schema"],
        "atomic_ops_sha256": hashlib.sha256(atomic_ops).hexdigest(),
        "effect_blocks_schema": block_payload["schema"],
        "effect_blocks_sha256": hashlib.sha256(effect_blocks).hexdigest(),
        "vocabulary_size": len(vocabulary),
        "token_count": int(lengths.sum()),
        "tokens_per_effect": {
            "mean": round(float(lengths.mean()), 6),
            "max": int(lengths.max()),
            "min": int(lengths.min()),
        },
        "empty_effects": int((lengths == 0).sum()),
        "unresolved_roles": unresolved_roles,
        "aux_helper_tokens": aux_tokens,
        "aux_helpers_expanded": False,
        "tokens_sha256": hashlib.sha256(
            _token_audit_bytes(passcode_array, index_array, token_offsets, token_ids, vocabulary)
        ).hexdigest(),
        "collisions": collisions,
        "privacy": {
            "card_names_in_tokens": False,
            "passcodes_in_tokens": False,
            "setcodes_in_tokens": False,
            "self_reference": "<SELF>",
            "other_card_reference": "<CARD_REF>",
        },
    }
    for name, expected in (
        ("op_sequences_schema", OP_SEQUENCES_SCHEMA),
        ("atomic_ops_schema", ATOMIC_OPS_SCHEMA),
        ("effect_blocks_schema", EFFECT_BLOCKS_SCHEMA),
    ):
        if metadata[name] != expected:
            raise LuaHashBuildError(f"{name} {metadata[name]!r} != {expected!r}")
    return LuaHashTable(
        passcodes=passcode_array,
        effect_indices=index_array,
        token_offsets=token_offsets,
        token_ids=token_ids,
        vocabulary=vocabulary,
        vectors=vectors,
        metadata=metadata,
    )


def _token_audit_bytes(
    passcodes: np.ndarray,
    effect_indices: np.ndarray,
    token_offsets: np.ndarray,
    token_ids: np.ndarray,
    vocabulary: Sequence[str],
) -> bytes:
    """The canonical JSON-lines token audit, one effect per line."""
    out = io.BytesIO()
    for row in range(len(passcodes)):
        start, stop = int(token_offsets[row]), int(token_offsets[row + 1])
        out.write(_json_bytes({
            "passcode": int(passcodes[row]),
            "effect_index": int(effect_indices[row]),
            "tokens": [vocabulary[int(i)] for i in token_ids[start:stop]],
        }))
        out.write(b"\n")
    return out.getvalue()


def artifact_bytes(table: LuaHashTable) -> bytes:
    """The NPZ bytes of one table; the same table always gives the same bytes."""
    vocabulary = np.asarray(table.vocabulary, dtype=f"<U{max(map(len, table.vocabulary))}")
    arrays = (
        ("passcodes", table.passcodes.astype(np.int64)),
        ("effect_indices", table.effect_indices.astype(np.int32)),
        ("vectors", table.vectors.astype(np.float32)),
        ("token_offsets", table.token_offsets.astype(np.int64)),
        ("token_ids", table.token_ids.astype(np.int32)),
        ("vocabulary", vocabulary),
        ("meta", np.asarray(json.dumps(table.metadata, sort_keys=True))),
    )
    return npz_bytes(arrays)


def write_artifact(path: str | Path, table: LuaHashTable) -> str:
    """Write the artifact atomically and return its SHA-256."""
    return atomic_write(path, artifact_bytes(table))


def write_token_audit(path: str | Path, table: LuaHashTable) -> str:
    """Write the per-effect token list as gzipped JSON lines; returns its SHA-256."""
    raw = _token_audit_bytes(
        table.passcodes, table.effect_indices, table.token_offsets, table.token_ids, table.vocabulary
    )
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=6) as handle:
        handle.write(raw)
    return atomic_write(path, buffer.getvalue())
