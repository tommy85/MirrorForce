"""Pure JSON codec for the client's public belief World, shared with fixed corpora.

The default encode/decode form is the original fixed-corpus wire law. ``complete``
adds the strict service boundary: every public field, exact scalar types and no
private namespaces. Neither mode reads a model, native module or engine.
"""
from __future__ import annotations

import copy
import hashlib
import json

SCHEMA = "mirrorforce_public_world_codec/v1"
RPC_SCHEMA = "mirrorforce_client_public_world/v1"
RPC_CAPABILITY = {"schema": RPC_SCHEMA, "codec": SCHEMA, "source": "cloned-own-pending-client/v1",
                  "binding": "expected_obs_sha256", "ownership": "client-connection/v1"}
WORLD_KEYS = frozenset({"decklist_public", "hand", "hand_group", "deck", "extra", "facedown", "pool_main",
                        "pool_extra", "unpositioned", "own_deck", "own_deck_fixed", "types", "public_owned"})


def _integer(value, noun, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{noun} must be an exact integer >= {minimum}")


def _sequence(value, noun):
    if not isinstance(value, (list, tuple)):
        raise ValueError(noun + " must be an explicit sequence")
    return value


def _complete(world):
    if not isinstance(world, dict) or set(world) != WORLD_KEYS or world["decklist_public"] is not True:
        raise ValueError("a complete public world needs exactly its declared fields and public decklist mode")
    for name in ("hand", "deck", "extra", "own_deck"):
        for code in _sequence(world[name], name):
            _integer(code, name + " card code")
    group = _sequence(world["hand_group"], "hand group")
    for index in group:
        _integer(index, "hand group index")
    if len(set(group)) != len(group) or any(index >= len(world["hand"]) for index in group):
        raise ValueError("hand group indices must be distinct existing client slots")
    places = []
    for row in _sequence(world["facedown"], "face-down slots"):
        if len(_sequence(row, "face-down row")) != 4:
            raise ValueError("a face-down row needs location, sequence, shown code and extra kind")
        location, sequence, code, extra = row
        for value, noun in ((location, "face-down location"), (sequence, "face-down slot"), (code, "shown code")):
            _integer(value, noun)
        if location not in (4, 8, 32) or location == 4 and sequence >= 7 or location == 8 and sequence >= 8 \
                or type(extra) not in (bool, int) or extra not in (False, True):
            raise ValueError("invalid public face-down coordinate or extra-kind flag")
        places.append((location, sequence))
    if len(set(places)) != len(places):
        raise ValueError("public face-down coordinates repeat")
    for name in ("pool_main", "pool_extra", "types"):
        for code, value in world[name].items():
            _integer(code, name + " card code", 1)
            _integer(value, name + " value")
    if (world["pool_main"].keys() | world["pool_extra"].keys()) - world["types"].keys():
        raise ValueError("a public pool card has no declared type")
    for (location, code), count in world["unpositioned"].items():
        if location not in (1, 2, 64):
            raise ValueError("a public lower-bound location is not deck, hand or extra")
        _integer(code, "public lower-bound card code", 1)
        _integer(count, "public lower-bound count")
    for position, code in world["own_deck_fixed"].items():
        _integer(position, "own fixed deck position")
        _integer(code, "own fixed deck identity", 1)
        if position >= len(world["own_deck"]):
            raise ValueError("own fixed deck position is outside the current deck")
    for row in _sequence(world["public_owned"], "public owned cards"):
        if len(_sequence(row, "public owned row")) != 2:
            raise ValueError("a public owned row needs identity and extra kind")
        _integer(row[0], "public owned identity", 1)
        if type(row[1]) not in (bool, int) or row[1] not in (False, True):
            raise ValueError("public owned extra kind is an explicit flag")


def encode_world(world, *, complete=False):
    """Original fixed-corpus JSON form; map keys become explicit sorted integer rows."""
    if complete and (not isinstance(world, dict) or set(world) != WORLD_KEYS):
        raise ValueError("a complete public world needs exactly its declared fields")
    out = dict(world)
    for key in ("pool_main", "pool_extra", "types", "own_deck_fixed"):
        source = world[key]
        pairs = list(source.items()) if isinstance(source, dict) else list(source)
        if len({k for k, _ in pairs}) != len(pairs):
            raise ValueError("public world repeats a map key")
        out[key] = [[k, v] for k, v in sorted(pairs)]
    out["unpositioned"] = [[location, code, count] for (location, code), count in sorted(world["unpositioned"].items())]
    if complete:
        _complete(decode_world(out))
        out = copy.deepcopy(out)
    return out


def decode_world(record, *, complete=False):
    """Decode the original wire law, optionally enforcing the complete public-only contract."""
    if not isinstance(record, dict) or set(record) - WORLD_KEYS:
        raise ValueError("only declared public world fields are allowed")
    if complete and set(record) != WORLD_KEYS:
        raise ValueError("a complete public world needs exactly its declared fields")
    out = dict(record)
    for key, width in (("pool_main", 2), ("pool_extra", 2), ("types", 2), ("own_deck_fixed", 2), ("unpositioned", 3)):
        rows = out[key]
        if not isinstance(rows, list) or any(not isinstance(row, list) or len(row) != width
                                           or any(type(v) is not int for v in row) for row in rows):
            raise ValueError("public world maps use exact integer rows")
        pairs = [(tuple(row[:-1]) if width == 3 else row[0], row[-1]) for row in rows]
        if len({key_ for key_, _ in pairs}) != len(pairs):
            raise ValueError("public world repeats a map key")
        out[key] = dict(pairs)
    if complete:
        _complete(out)
        out = copy.deepcopy(out)
    return out


def world_sha256(record):
    """Hash only a complete, validated wire World; map/slot ordering remains explicit."""
    decode_world(record, complete=True)
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
