"""The category representatives of a belief sampler: drawn exactly from the law of a whole-pool shuffle conditioned on
each claimed slot's category, by grouping the claims with one option set and counting how many copies of each code go
to each group; no tuple is enumerated, there is no cap, and the counts are made once per sampler."""
from __future__ import annotations

import math
import random
import time
from collections import Counter
from itertools import product

import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.search import belief as B
from mirrorforce.search.belief import BeliefSampler, Evidence, representative_counts

SPELLS = frozenset(range(101, 109))
TRAPS = frozenset(range(105, 113))
MONSTERS = frozenset(range(110, 116))


def _evidence(facedown, hand=(MONSTERS,)):
    slots = tuple((C.LOCATION_SZONE, index) for index in range(len(facedown)))
    return Evidence(deck_list=Counter({code: 2 if code % 3 else 3 for code in range(101, 121)}), hand_size=5,
                    deck_size=sum(2 if code % 3 else 3 for code in range(101, 121)) - 5 - len(facedown),
                    facedown_slots=len(facedown), facedown_slot_keys=slots, hand_categories=tuple(hand),
                    facedown_categories=tuple(zip(slots, facedown)))


def _tuple_weights(options, pool):
    """Every representative tuple with its number of instance tuples: the exact law, by enumeration."""
    out = {}
    for candidate in product(*options):
        used, weight = Counter(), 1
        for code in candidate:
            weight *= pool[code] - used[code]
            used[code] += 1
        if weight > 0:
            out[candidate] = weight
    return out


@pytest.mark.parametrize("sets", [
    (SPELLS, SPELLS, SPELLS),
    (MONSTERS, SPELLS, SPELLS, TRAPS),
    (TRAPS, SPELLS, TRAPS, MONSTERS, SPELLS),
    (frozenset({110, 111}), frozenset({110, 111}), frozenset({111, 112})),
])
def test_the_counts_are_the_number_of_representative_systems(sets):
    pool = Counter({code: 2 if code % 3 else 3 for code in range(101, 121)})
    options = [tuple(sorted(codes)) for codes in sets]
    groups = sorted(set(options))
    sizes = tuple(options.count(group) for group in groups)
    _codes, _members, counts = representative_counts(groups, sizes, pool)
    # The counts leave each group's order out: k! orders of each group's k slots.
    assert counts[0][sizes] * math.prod(math.factorial(size) for size in sizes) \
        == sum(_tuple_weights(options, pool).values())


def test_draws_follow_the_exact_conditional_law():
    sets = (frozenset({110, 111}), frozenset({110, 111, 112}), frozenset({111, 112}))
    evidence = _evidence(sets[1:], hand=sets[:1])
    pool = evidence.unknown_pool()
    law = _tuple_weights([tuple(sorted(codes)) for codes in sets], pool)
    total = sum(law.values())
    sampler = BeliefSampler(evidence, random.Random(20260926))
    draws = 30000
    seen = Counter()
    for _ in range(draws):
        particle = sampler.sample()
        seen[(particle.hand[0],) + tuple(particle.facedown)] += 1
    assert set(seen) <= set(law)
    for representatives, weight in law.items():
        assert abs(seen[representatives] / draws - weight / total) < 0.01, representatives


def test_many_facedown_claims_draw_at_once_with_counts_made_once_per_sampler(monkeypatch):
    built = []
    counting = B.representative_counts

    def recorded(*args):
        built.append(args[1])
        return counting(*args)

    monkeypatch.setattr(B, "representative_counts", recorded)
    spells_and_traps = SPELLS | TRAPS
    evidence = _evidence((spells_and_traps,) * 6)  # 12^6 ≈ 3 million tuples, far past any enumeration
    sampler = BeliefSampler(evidence, random.Random(7))
    started = time.monotonic()
    for _ in range(64):
        particle = sampler.sample()
        assert set(particle.hand) & MONSTERS
        assert all(code in spells_and_traps for code in particle.facedown)
    assert time.monotonic() - started < 2.0
    assert built == [(6, 1)]  # the spell/trap group sorts before the monster group
