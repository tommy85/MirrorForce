"""Deterministic announce-law fuzz cases: real filter shapes and random combinations crossed with pool recipes.

A case is everything public the law reads at one ``MSG_ANNOUNCE_CARD`` prompt (the filter program, the opponent's
and the declarer's publicly seen cards, the declarer's own recipe, the room format), plus the opponent's actual
recipe for the offline privileged audit only (the law never reads it). The same ``seed`` always gives the same
cases, so the training env, the evaluation path and the deployment client can be fed identical inputs.
"""
from __future__ import annotations

import random

from announce_law_oracle import (OPCODE_AND, OPCODE_ISATTRIBUTE, OPCODE_ISCODE, OPCODE_ISRACE, OPCODE_ISSETCARD,
                                 OPCODE_ISTYPE, OPCODE_NOT, OPCODE_OR)

TYPE_MONSTER, TYPE_SPELL, TYPE_TRAP, TYPE_NORMAL = 0x1, 0x2, 0x4, 0x10
EXTRA_TYPES = 0x40 | 0x2000 | 0x800000 | 0x4000000  # fusion, synchro, xyz, link
RACES = [1 << i for i in range(26)]
ATTRIBUTES = [1 << i for i in range(7)]


def _leaf(rng, cards, codes):
    kind = rng.randrange(6)
    if kind == 0:
        return [rng.choice((TYPE_MONSTER, TYPE_SPELL, TYPE_TRAP, TYPE_NORMAL, EXTRA_TYPES)), OPCODE_ISTYPE]
    if kind == 1:
        return [rng.choice(RACES), OPCODE_ISRACE]
    if kind == 2:
        return [rng.choice(ATTRIBUTES), OPCODE_ISATTRIBUTE]
    if kind == 3:
        setcodes = [s for c in codes for s in cards[c].setcodes]
        return [rng.choice(setcodes) if setcodes else 0x51, OPCODE_ISSETCARD]
    return [rng.choice(codes), OPCODE_ISCODE]


def program(rng, cards, codes):
    """One filter program: a real script shape or a random combination of leaves."""
    shape = rng.randrange(8)
    if shape == 0:
        return [TYPE_MONSTER, OPCODE_ISTYPE]                                     # "declare a monster card name"
    if shape == 1:
        return [EXTRA_TYPES, OPCODE_ISTYPE, OPCODE_NOT]                          # a main-deck card name
    if shape == 2:
        return [TYPE_MONSTER, OPCODE_ISTYPE, EXTRA_TYPES, OPCODE_ISTYPE, OPCODE_NOT, OPCODE_AND]
    if shape == 3:                                                               # a name from a list (Crossout)
        picked = rng.sample(codes, k=min(len(codes), rng.randint(1, 6)))
        out = [picked[0], OPCODE_ISCODE]
        for code in picked[1:]:
            out += [code, OPCODE_ISCODE, OPCODE_OR]
        return out
    if shape == 4:                                                               # any name but one
        return [rng.choice(codes), OPCODE_ISCODE, OPCODE_NOT]
    if shape == 5:                                                               # an archetype monster but itself
        setcodes = [s for c in codes for s in cards[c].setcodes] or [0x51]
        return [rng.choice(setcodes), OPCODE_ISSETCARD, TYPE_MONSTER, OPCODE_ISTYPE, OPCODE_AND,
                rng.choice(codes), OPCODE_ISCODE, OPCODE_NOT, OPCODE_AND]
    out = _leaf(rng, cards, codes)
    for _ in range(rng.randint(1, 3)):
        out += _leaf(rng, cards, codes) + [rng.choice((OPCODE_AND, OPCODE_OR))]
        if rng.random() < 0.3:
            out.append(OPCODE_NOT)
    return out


def cases(seed, count, library, cards):
    """``count`` cases over the library's recipes (codes outside ``cards`` are left out of every draw)."""
    rng = random.Random(seed)
    out = []
    for index in range(count):
        own, opponent = rng.sample(library, 2) if len(library) > 1 else (library[0], library[0])
        own_codes = sorted({c for c in [*own["main"], *own["extra"]] if c in cards})
        opp_codes = sorted({c for c in [*opponent["main"], *opponent["extra"]] if c in cards})
        pool = sorted(set(own_codes) | set(opp_codes))
        out.append({
            "index": index,
            "opcodes": program(rng, cards, pool),
            "seen_opponent": sorted(rng.sample(opp_codes, k=rng.randint(0, min(15, len(opp_codes))))),
            "seen_own": sorted(rng.sample(own_codes, k=rng.randint(0, min(10, len(own_codes))))),
            "own_main": list(own["main"]), "own_extra": list(own["extra"]),
            "room_format": rng.choice((own["format"], None)),
            "opponent_recipe_audit_only": {"main": list(opponent["main"]), "extra": list(opponent["extra"])},
        })
    return out
