"""Reference implementation of the announce-card candidate law (design §6.1), as a test oracle.

The law is a pure function of public state: the filter program of the ``MSG_ANNOUNCE_CARD`` prompt, the cards the
declaring player has publicly seen this duel (the opponent's and its own), its own recipe, the generic staple table,
the public recipe library and the room format. Its C++ implementation (exported from ``duel_native`` and called by
the training env, the evaluation path and the deployment client) must return exactly this result; the argument list
mirrors that function.

Order of the candidate list: five tiers, in order
  1. the ``ISCODE`` literals of the filter program;
  2. cards publicly seen this duel: the opponent's, then the declarer's own;
  3. the declarer's own recipe (main and extra);
  4. generic staples (the staple table: deck-type document frequency over the library, computed offline);
  5. cards of the opponent-deck belief: the deterministic public posterior over library recipes.
Within a tier: score descending (staples: deck-type count; belief: posterior; other tiers: no score), then card id
ascending (the env's card table order). Every candidate must pass the core's declarable filter (the env's
``evaluate_announce_card_filter``, ported below), and a card keeps only its first position. Tiers 1-4 are never
truncated: if they do not fit the cap the configuration is wrong and the law refuses. Tier 5 is cut at the cap and
the dropped count is reported. If no tier yields a candidate, the named branch ``empty_union`` takes every
declarable card in card id order up to the cap and reports how many it dropped.

Belief (``public_posterior_presence/v1``): the library recipes of the room's format (all recipes when the format is
unknown) that contain every card the opponent has publicly shown (by presence, main or extra deck); each cluster
with at least one such recipe weighs 1/(clusters), split evenly over its compatible recipes; a card's score is the
total weight of the compatible recipes containing it. Scores are IEEE doubles summed in library order, recipe by
recipe and card by card in recipe order, so a C++ implementation reproduces them bit for bit.
"""
from __future__ import annotations

from dataclasses import dataclass
import sqlite3

LAW = "announce_public_candidates/v2"
BELIEF_LAW = "public_posterior_presence/v1"
TIERS = ("literal", "seen", "own_recipe", "staple", "belief")
EMPTY_UNION = "empty_union"

OPCODE_ADD, OPCODE_SUB, OPCODE_MUL, OPCODE_DIV = 0x40000000, 0x40000001, 0x40000002, 0x40000003
OPCODE_AND, OPCODE_OR, OPCODE_NEG, OPCODE_NOT = 0x40000004, 0x40000005, 0x40000006, 0x40000007
OPCODE_ISCODE, OPCODE_ISSETCARD, OPCODE_ISTYPE = 0x40000100, 0x40000101, 0x40000102
OPCODE_ISRACE, OPCODE_ISATTRIBUTE = 0x40000103, 0x40000104
TYPE_MONSTER, TYPE_TOKEN = 0x1, 0x4000
#: Declarable despite their alias (the env's ``evaluate_announce_card_filter``).
ALIAS_EXCEPTIONS = (78734254, 13857930)


@dataclass(frozen=True)
class Card:
    code: int
    alias: int
    setcodes: tuple  # the 16-bit set codes, as the core's card_data holds them (zeros dropped)
    type: int
    race: int
    attribute: int


def _setcodes(packed: int) -> tuple:
    out = []
    while packed:
        if packed & 0xFFFF:
            out.append(packed & 0xFFFF)
        packed >>= 16
    return tuple(out)


def load_cards(cdb_path) -> dict:
    """Every card of a card database, as the env's card table reads it."""
    with sqlite3.connect(f"file:{cdb_path}?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT id, alias, setcode, type, race, attribute FROM datas").fetchall()
    return {code: Card(code, alias or 0, _setcodes(setcode or 0), type_ or 0, race or 0, attribute or 0)
            for code, alias, setcode, type_, race, attribute in rows}


def _check_setcode(setcode: int, value: int) -> bool:
    settype, setsubtype = value & 0x0FFF, value & 0xF000
    return bool(setcode) and (setcode & 0x0FFF) == settype and (setcode & setsubtype) == setsubtype


def declarable(card: Card, opcodes) -> bool:
    """Port of the env's ``evaluate_announce_card_filter`` (cxx/duelpool/duel/duel_env.h)."""
    stack = []
    for opcode in opcodes:
        if opcode in (OPCODE_ADD, OPCODE_SUB, OPCODE_MUL, OPCODE_DIV, OPCODE_AND, OPCODE_OR):
            if len(stack) >= 2:
                rhs, lhs = stack.pop(), stack.pop()
                if opcode == OPCODE_ADD:
                    stack.append(lhs + rhs)
                elif opcode == OPCODE_SUB:
                    stack.append(lhs - rhs)
                elif opcode == OPCODE_MUL:
                    stack.append(lhs * rhs)
                elif opcode == OPCODE_DIV:
                    if rhs == 0:
                        return False
                    quotient = abs(lhs) // abs(rhs)  # C++ int64 division truncates toward zero
                    stack.append(quotient if (lhs < 0) == (rhs < 0) else -quotient)
                elif opcode == OPCODE_AND:
                    stack.append(int(bool(lhs) and bool(rhs)))
                else:
                    stack.append(int(bool(lhs) or bool(rhs)))
        elif opcode == OPCODE_NEG:
            if stack:
                stack[-1] = -stack[-1]
        elif opcode == OPCODE_NOT:
            if stack:
                stack[-1] = int(not stack[-1])
        elif opcode == OPCODE_ISCODE:
            if stack:
                stack[-1] = int(card.code == (stack[-1] & 0xFFFFFFFF))
        elif opcode == OPCODE_ISSETCARD:
            if stack:
                value = stack[-1] & 0xFFFFFFFF
                stack[-1] = int(any(_check_setcode(s, value) for s in card.setcodes))
        elif opcode == OPCODE_ISTYPE:
            if stack:
                stack[-1] = card.type & (stack[-1] & 0xFFFFFFFF)
        elif opcode == OPCODE_ISRACE:
            if stack:
                stack[-1] = card.race & (stack[-1] & 0xFFFFFFFF)
        elif opcode == OPCODE_ISATTRIBUTE:
            if stack:
                stack[-1] = card.attribute & (stack[-1] & 0xFFFFFFFF)
        else:
            stack.append(_int32(opcode))
    if len(stack) != 1 or stack[-1] == 0:
        return False
    return card.code in ALIAS_EXCEPTIONS or (
        not card.alias and (card.type & (TYPE_MONSTER | TYPE_TOKEN)) != (TYPE_MONSTER | TYPE_TOKEN))


def _int32(value: int) -> int:
    """``static_cast<int32_t>(opcode)`` of an operand pushed onto the int64 stack."""
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


def literals(opcodes) -> list:
    """The ``ISCODE`` operands of a filter program, in program order."""
    return [opcodes[i - 1] & 0xFFFFFFFF for i in range(1, len(opcodes)) if opcodes[i] == OPCODE_ISCODE]


def belief_scores(library, seen_opponent, room_format) -> dict:
    """``public_posterior_presence/v1``: card code -> posterior weight of the compatible library recipes.

    ``library``: recipes as dicts ``{"main", "extra", "cluster", "format"}`` in their registered order."""
    shown = set(seen_opponent)
    compatible = [r for r in library if (room_format is None or r["format"] == room_format)
                  and shown <= (set(r["main"]) | set(r["extra"]))]
    clusters = {}
    for recipe in compatible:
        clusters.setdefault(recipe["cluster"], []).append(recipe)
    scores = {}
    for recipe in compatible:
        weight = (1.0 / len(clusters)) / len(clusters[recipe["cluster"]])
        seen = set()
        for code in [*recipe["main"], *recipe["extra"]]:
            if code not in seen:
                seen.add(code)
                scores[code] = scores.get(code, 0.0) + weight
    return scores


def candidates(opcodes, *, seen_opponent, seen_own, own_main, own_extra, staples, library, room_format, cap,
               cards, card_ids) -> dict:
    """The law's result: ``{"law", "candidates", "tiers", "branch", "truncated"}``.

    ``staples``: ``{code: deck-type count}``; ``cards``: code -> ``Card`` (the env's card table); ``card_ids``:
    code -> card id (the env's card table order). Codes outside the card table are never candidates."""
    if type(cap) is not int or cap < 1:
        raise ValueError("the announce cap is a positive integer")
    order = lambda code: card_ids[code]  # noqa: E731

    def eligible(codes):
        return [c for c in codes if c in cards and c in card_ids and declarable(cards[c], opcodes)]

    beliefs = belief_scores(library, seen_opponent, room_format)
    tiers = [
        sorted(set(eligible(literals(opcodes))), key=order),
        sorted(set(eligible(seen_opponent)), key=order) + sorted(set(eligible(seen_own)) - set(seen_opponent),
                                                                key=order),
        sorted(set(eligible([*own_main, *own_extra])), key=order),
        sorted(set(eligible(staples)), key=lambda c: (-staples[c], order(c))),
        sorted(set(eligible(beliefs)), key=lambda c: (-beliefs[c], order(c))),
    ]
    out, placed, per_tier = [], set(), []
    for tier in tiers:
        added = [c for c in tier if c not in placed]
        placed.update(added)
        per_tier.append(added)
    fixed = sum(len(t) for t in per_tier[:4])
    if fixed > cap:
        raise ValueError(f"announce tiers 1-4 hold {fixed} cards, more than the cap {cap}")
    for tier in per_tier[:4]:
        out.extend(tier)
    room = cap - len(out)
    out.extend(per_tier[4][:room])
    truncated = {"belief": max(0, len(per_tier[4]) - room), EMPTY_UNION: 0}
    branch = "union"
    if not out:
        branch = EMPTY_UNION
        every = sorted(eligible(card_ids), key=order)
        out = every[:cap]
        truncated[EMPTY_UNION] = len(every) - len(out)
    return {"law": LAW, "belief_law": BELIEF_LAW, "candidates": out, "branch": branch, "truncated": truncated,
            "tiers": {name: len(t) for name, t in zip(TIERS, per_tier)}}
