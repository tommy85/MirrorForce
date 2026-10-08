"""Exact public field assignments with position-free identity lower bounds.

Cards with identical constraint membership share one counting class. Dynamic
programming counts distinct physical-card assignments, without enumerating the
unconstrained deck. Anonymous category claims are existential matchings, not
extra draws: several witnesses for the same assignment must not overweight it.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import random


class FieldAssignmentError(ValueError):
    """The public field constraints have no simultaneous assignment."""


def _extend_matches(states: tuple[int, ...], eligible: int) -> tuple[int, ...]:
    """All claim subsets witnessed after adding one physical field card."""
    extended = set(states)
    for state in states:
        available = eligible & ~state
        while available:
            bit = available & -available
            extended.add(state | bit)
            available -= bit
    # Keep only maximal subsets: a subset cannot enable a completion that its
    # superset cannot. This also keeps repeated broad claims inexpensive.
    return tuple(sorted(state for state in extended
                        if not any(state != other and state & other == state for other in extended)))


class FieldAssignmentSampler:
    """Uniform physical-card draws conditional on public slot/zone evidence.

    ``slots`` are (location, sequence, allowed codes); None allows any code.
    Hand-category representatives may use synthetic sequence numbers. Every
    ``identities`` entry requires one additional copy somewhere in that zone.
    ``categories`` require distinct witnesses, but can overlap identity facts.
    Fixed public field cards can witness categories, never additional identities.
    ``fresh`` are the indices of slots whose cards arrived after the identities
    were proven (``DisclosureLedger.fresh_slots``): no identity sits there.
    """

    def __init__(self, pool: Counter, slots: tuple, *, identities=(), categories=(), fixed=(), fresh=()):
        self.pool = +Counter(pool)
        self.slots = tuple(slots)
        self.fresh = frozenset(int(index) for index in fresh)
        if any(not 0 <= index < len(self.slots) for index in self.fresh):
            raise FieldAssignmentError("a fresh slot is not one of the assigned slots")
        required = Counter((int(location), int(code)) for location, code in identities)
        self.required = tuple(sorted(required))
        self.initial_required = tuple(required[key] for key in self.required)
        self.categories = tuple((int(location), frozenset(codes)) for location, codes in categories)
        self.full_mask = (1 << len(self.categories)) - 1
        initial_matches = (0,)
        for location, code in fixed:
            eligible = sum(1 << index for index, (zone, codes) in enumerate(self.categories)
                           if zone == location and code in codes)
            initial_matches = _extend_matches(initial_matches, eligible)
        self.initial_matches = initial_matches

        classes: dict[tuple, list[int]] = {}
        for code in sorted(self.pool):
            signature = (
                tuple(allowed is None or code in allowed for _, _, allowed in self.slots),
                tuple(code == known for _, known in self.required),
                tuple(code in codes for _, codes in self.categories),
            )
            classes.setdefault(signature, []).append(code)
        self.signatures = tuple(classes)
        self.codes = tuple(tuple(codes) for codes in classes.values())
        self.initial_counts = tuple(sum(self.pool[code] for code in codes) for codes in self.codes)
        self._ways = lru_cache(maxsize=None)(self._count)
        if not self._ways(0, self.initial_counts, self.initial_required, self.initial_matches):
            raise FieldAssignmentError("public field identities and categories have no joint assignment")

    def _next(self, index, class_index, counts, required, matches):
        location = self.slots[index][0]
        signature = self.signatures[class_index]
        after = list(counts)
        after[class_index] -= 1
        # A fresh slot's card is none of the identities (it arrived after they were proven).
        known = index not in self.fresh
        remaining = tuple(max(0, count - int(known and zone == location and signature[1][j]))
                          for j, ((zone, _code), count) in enumerate(zip(self.required, required)))
        eligible = sum(1 << j for j, (zone, _codes) in enumerate(self.categories)
                       if zone == location and signature[2][j])
        return tuple(after), remaining, _extend_matches(matches, eligible)

    def _count(self, index, counts, required, matches) -> int:
        if index == len(self.slots):
            return int(not any(required) and self.full_mask in matches)
        # A lower bound names distinct cards in its zone; reject an impossible
        # branch before expanding any card classes.
        needed = Counter()
        for (location, _code), count in zip(self.required, required):
            needed[location] += count
        available = Counter(slot[0] for number, slot in enumerate(self.slots[index:], start=index)
                            if number not in self.fresh)
        if any(count > available[zone] for zone, count in needed.items()):
            return 0
        total = 0
        for class_index, count in enumerate(counts):
            if count and self.signatures[class_index][0][index]:
                state = self._next(index, class_index, counts, required, matches)
                total += count * self._ways(index + 1, *state)
        return total

    def sample(self, rng: random.Random) -> tuple[tuple[int, ...], Counter]:
        """Return cards in slot order and the remaining, unassigned pool."""
        pool = self.pool.copy()
        counts, required, matches = self.initial_counts, self.initial_required, self.initial_matches
        chosen = []
        for index in range(len(self.slots)):
            candidates, weights = [], []
            for class_index, count in enumerate(counts):
                if count and self.signatures[class_index][0][index]:
                    state = self._next(index, class_index, counts, required, matches)
                    weight = count * self._ways(index + 1, *state)
                    if weight:
                        candidates.append((class_index, state))
                        weights.append(weight)
            draw = rng.randrange(sum(weights))
            for (class_index, state), weight in zip(candidates, weights):
                if draw < weight:
                    break
                draw -= weight
            draw = rng.randrange(counts[class_index])
            for code in self.codes[class_index]:
                if draw < pool[code]:
                    break
                draw -= pool[code]
            chosen.append(code)
            pool[code] -= 1
            counts, required, matches = state
        return tuple(chosen), +pool

    def clear(self) -> None:
        """Release per-root completion counts when the caller is finished."""
        self._ways.cache_clear()
