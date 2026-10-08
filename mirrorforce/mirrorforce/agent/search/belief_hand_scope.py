"""Public physical hand scope, separate from the AR sorted-multiset quotient.

The native observer supplies only already-public anchors and the membership of
its old shuffled group. A hypothetical full hand must place that group's known
multiset *inside the group*. Complete code arrangements are sampled exactly
uniformly conditional on these public constraints, never by first matching.
No engine, model, hidden target or guessed card identity is an input here.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import time
from types import MappingProxyType

SCHEMA = "mirrorforce_same_pending_public_hand_scope/v1"
ASSIGNMENT_LAW = "uniform-physical-hand-codes-given-public-anchors-and-shuffle-group/v1"
UID_LAW = "conditional-uniform-old-group-UID-matching;fresh-and-public-anchors-fixed/v1"
RNG_LAW = "python-Random-exact-integer-DP-then-uniform-group-permutations/v1"
MAX_NODES = 50000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("public hand scope needs exact bounded integers")
    return value


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("public hand scope needs its pending/world SHA binding")
    return value


class HandScopeBudgetExceeded(TimeoutError):
    """Support/sampling remains unknown; never return a partial hand."""


class WorkBudget:
    def __init__(self, *, deadline, max_nodes=MAX_NODES, used=0, clock=time.monotonic):
        if type(deadline) not in (int, float) or not math.isfinite(deadline) \
                or type(max_nodes) is not int or not 1 <= max_nodes <= 1000000 \
                or type(used) is not int or not 0 <= used <= max_nodes or not callable(clock):
            raise ValueError("hand allocation needs the original finite deadline and remaining node budget")
        self.deadline, self.max_nodes, self.nodes, self.clock = deadline, max_nodes, used, clock

    def check(self):
        if self.clock() >= self.deadline or self.nodes >= self.max_nodes:
            raise HandScopeBudgetExceeded("public hand allocation budget exhausted; no partial assignment")

    def tick(self):
        self.check()
        self.nodes += 1


def from_world(world_record, *, obs_sha256, world_sha256, reference_anchors=()):
    """Project just the hand from the closed same-pending public World RPC.

    ``reference_anchors`` are actual ledger-known physical positions, not canonical
    displayed hand rows or arbitrary named objects in a hypothetical engine.
    """
    from ..public_world_codec import decode_world, world_sha256 as checksum
    if checksum(world_record) != _sha(world_sha256):
        raise ValueError("same-pending public World changed")
    world = decode_world(world_record, complete=True)
    record = {"schema": SCHEMA, "obs_sha256": _sha(obs_sha256), "world_sha256": world_sha256,
              "hand": list(world["hand"]), "group": list(world["hand_group"]),
              "lower": [[code, count] for (location, code), count in sorted(world["unpositioned"].items())
                        if location == 2], "reference_anchors": [list(row) for row in reference_anchors]}
    return PublicHandScope(record).to_dict()


class PublicHandScope:
    __slots__ = ("size", "group", "anchors", "lower", "group_needed", "free_group", "free_other",
                 "minimum", "_json", "_frozen")

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("public hand scope is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, record):
        if type(record) is not dict or set(record) != {
                "schema", "obs_sha256", "world_sha256", "hand", "group", "lower", "reference_anchors"} \
                or record["schema"] != SCHEMA:
            raise ValueError("public hand scope accepts only its closed public record")
        _sha(record["obs_sha256"]); _sha(record["world_sha256"])
        hand, group = record["hand"], record["group"]
        if type(hand) is not list or len(hand) > 80 or any(type(code) is not int or code < 0 for code in hand):
            raise ValueError("public hand rows must be bounded known codes or anonymous zeros")
        if type(group) is not list or group != sorted(set(group)) \
                or any(type(i) is not int or not 0 <= i < len(hand) or hand[i] != 0 for i in group):
            raise ValueError("old shuffled group must contain distinct currently unidentified hand slots")
        anchors = {i: code for i, code in enumerate(hand) if code}
        provided = record["reference_anchors"]
        if type(provided) is not list or any(type(row) is not list or len(row) != 2 for row in provided) \
                or provided != sorted(provided) or len({row[0] for row in provided}) != len(provided):
            raise ValueError("physical hand anchors must be canonical distinct position/code rows")
        for position, code in provided:
            _integer(position); _integer(code, 1)
            if position >= len(hand) or position in anchors and anchors[position] != code:
                raise ValueError("native/v4 public hand anchors contradict each other")
            anchors[position] = code
        lower = record["lower"]
        if type(lower) is not list or any(type(row) is not list or len(row) != 2 for row in lower) \
                or lower != sorted(lower) or len({row[0] for row in lower}) != len(lower):
            raise ValueError("public old-group identities must be canonical code/count rows")
        counts = Counter({_integer(code, 1): _integer(count, 1) for code, count in lower})
        self.size, self.group = len(hand), frozenset(group)
        self.anchors, self.lower = MappingProxyType(anchors), MappingProxyType(counts)
        fixed_group = Counter(code for i, code in anchors.items() if i in self.group)
        self.group_needed = MappingProxyType(counts - fixed_group)
        self.free_group = tuple(i for i in group if i not in anchors)
        self.free_other = tuple(i for i in range(self.size) if i not in anchors and i not in self.group)
        if sum(self.group_needed.values()) > len(self.free_group):
            raise ValueError("public old shuffled group cannot contain its required identities")
        self.minimum = MappingProxyType(Counter(anchors.values()) + Counter(self.group_needed))
        if sum(self.minimum.values()) > self.size:
            raise ValueError("public physical hand anchors/group exceed the hand size")
        self._json = json.dumps(deepcopy(record), sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._frozen = True

    def to_dict(self):
        return json.loads(self._json)

    @property
    def sha256(self):
        return hashlib.sha256(self._json.encode()).hexdigest()

    def feasible_multiset(self, hand):
        """Exact existence of a physical assignment for this complete multiset.

        There is one publicly identified old group, no per-slot unknown-card
        domains. After fixed anchors, its lower bounds and capacity are the
        necessary and sufficient constraints; leftover cards can fill either
        complement. This does not pretend sorted AR positions are physical.
        """
        if type(hand) not in (tuple, list) or any(type(code) is not int or code <= 0 for code in hand):
            raise ValueError("a hypothetical hand must contain explicit positive codes")
        counts = Counter(hand)
        return len(hand) == self.size and all(counts[code] >= count for code, count in self.minimum.items())

    def check_physical(self, hand):
        if not self.feasible_multiset(hand):
            raise ValueError("hypothetical hand multiset violates its public anchors/old group")
        if any(hand[i] != code for i, code in self.anchors.items()) \
                or any(sum(hand[i] == code for i in self.group) < count for code, count in self.lower.items()):
            raise ValueError("physical hand moved public identities outside their anchors/old shuffled group")
        return True

    def distribution(self, hand, *, budget):
        """Exact allocation DP; weights count distinct physical code arrangements."""
        budget.check()
        if not self.feasible_multiset(hand):
            raise ValueError("no physical hand can realize this public-constrained multiset")
        counts = Counter(hand) - Counter(self.anchors.values())
        codes = tuple(sorted(counts))
        available = tuple(counts[code] for code in codes)
        minima = tuple(self.group_needed[code] for code in codes)
        group_size = len(self.free_group)
        table = [dict() for _ in range(len(codes) + 1)]
        table[-1][0] = 1
        for index in range(len(codes) - 1, -1, -1):
            for need in range(group_size + 1):
                budget.tick()
                table[index][need] = sum(math.comb(available[index], take) * table[index + 1].get(need - take, 0)
                    for take in range(minima[index], min(available[index], need) + 1))
        labeled_ways = table[0].get(group_size, 0)
        if not labeled_ways:
            raise ValueError("public hand exact allocation has zero completions")
        numerator = labeled_ways * math.factorial(group_size) * math.factorial(len(self.free_other))
        denominator = math.prod(math.factorial(n) for n in available)
        if numerator % denominator:
            raise AssertionError("physical hand count is not an integer")
        return codes, available, minima, table, numerator // denominator

    def sample(self, hand, *, rng, budget):
        """Uniform among all distinct physical code arrangements, with a proof count.

        Integer binomial DP chooses group multiplicities; uniform permutations
        within each group then give each eligible full arrangement probability
        exactly 1/support_count. No rejection, floating normalization or first
        matching approximation is used. ``rng`` is an independently seeded
        random.Random instance owned by this registered physical-draw law.
        """
        codes, available, minima, table, support_count = self.distribution(hand, budget=budget)
        grouped, other, need = [], [], len(self.free_group)
        for index, code in enumerate(codes):
            budget.tick()
            options = [(take, math.comb(available[index], take) * table[index + 1].get(need - take, 0))
                       for take in range(minima[index], min(available[index], need) + 1)]
            total = sum(weight for _, weight in options)
            if total != table[index][need] or total <= 0:
                raise AssertionError("hand sampling DP mass changed")
            ticket = rng.randrange(total)
            for take, weight in options:
                if ticket < weight:
                    grouped.extend([code] * take)
                    other.extend([code] * (available[index] - take))
                    need -= take
                    break
                ticket -= weight
        budget.check()
        rng.shuffle(grouped); rng.shuffle(other)
        result = [None] * self.size
        for i, code in self.anchors.items(): result[i] = code
        for i, code in zip(self.free_group, grouped): result[i] = code
        for i, code in zip(self.free_other, other): result[i] = code
        self.check_physical(result)
        if Counter(result) != Counter(hand):
            raise AssertionError("physical hand sampler changed its original multiset")
        return tuple(result), {"law": ASSIGNMENT_LAW, "rng_law": RNG_LAW, "scope_sha256": self.sha256,
                               "support_count": str(support_count), "hand_sha256": digest(result)}
