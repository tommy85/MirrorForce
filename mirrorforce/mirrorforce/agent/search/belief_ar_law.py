"""Public-only constraints for AR field identities followed by a sorted hidden-hand multiset.

This independent prototype does not replace the production sampler. Each token mask checks that a *complete*
assignment, including the ungenerated deck/extra/banished areas, exists. It is not uniform sampling of those
assignments: network sampling uses the masked autoregressive conditionals. No truth or training labels are read.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import numpy as np

from mirrorforce.agent.search import particles as P
from mirrorforce.agent.public_world_codec import WORLD_KEYS

SCHEMA = "mirrorforce_belief_ar_public_law/v1"


def _integer(value, noun: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{noun} must be a non-boolean integer >= {minimum}")
    return int(value)


def _validate_world(world: Mapping) -> None:
    """Reject malformed public evidence before integer arrays or group construction can coerce it."""
    P._check_world(world)
    for name in ("hand", "deck", "extra", "own_deck"):
        for code in world[name]:
            _integer(code, f"{name} slot card code")
    indices = [_integer(i, "hand group index") for i in world["hand_group"]]
    if len(indices) != len(set(indices)) or any(i >= len(world["hand"]) for i in indices):
        raise ValueError("hand group indices must be unique valid hand slots")
    seen = set()
    for row in world["facedown"]:
        if len(row) != 4:
            raise ValueError("a public face-down slot has location, sequence, shown code and extra-kind")
        location, sequence, code, extra = row
        location, sequence = _integer(location, "face-down location"), _integer(sequence, "face-down sequence")
        _integer(code, "face-down shown card code")
        if location not in (P.LOCATION_MZONE, P.LOCATION_SZONE, P.LOCATION_REMOVED) \
                or location == P.LOCATION_MZONE and sequence >= 7 \
                or location == P.LOCATION_SZONE and sequence >= 8 or (location, sequence) in seen:
            raise ValueError("invalid or duplicate public face-down slot")
        if not isinstance(extra, (bool, np.bool_, int, np.integer)) or extra not in (0, 1):
            raise ValueError("face-down extra-kind is an explicit 0/1 flag")
        seen.add((location, sequence))
    if not isinstance(world["unpositioned"], Mapping):
        raise ValueError("public identity lower bounds must be a mapping")
    for key, count in world["unpositioned"].items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("public lower-bound keys are (location, card code)")
        location = _integer(key[0], "lower-bound location")
        _integer(key[1], "lower-bound card code", minimum=1)
        _integer(count, "public identity lower bound")
        if location not in (0x01, 0x02, 0x40):
            raise ValueError("public identity lower-bound location is not deck, hand or extra")
    for name in ("pool_main", "pool_extra", "types"):
        if not isinstance(world[name], Mapping):
            raise ValueError(f"{name} must be an integer mapping")
        for code, value in world[name].items():
            _integer(code, f"{name} card code", minimum=1)
            _integer(value, f"{name} value")
    if (set(world["pool_main"]) | set(world["pool_extra"])) - set(world["types"]):
        raise ValueError("a public pool card has no public card type")
    fixed = world["own_deck_fixed"]
    pairs = list(fixed.items()) if isinstance(fixed, Mapping) else list(fixed)
    positions = []
    for position, code in pairs:
        positions.append(_integer(position, "own fixed deck position"))
        _integer(code, "own fixed deck card code", minimum=1)
    if len(positions) != len(set(positions)) or any(p >= len(world["own_deck"]) for p in positions):
        raise ValueError("own fixed deck positions must be unique valid slots")


def _max_flow(size: int, edges: Sequence[tuple[int, int, int]], source: int, sink: int) -> int:
    """Integral Dinic flow; capacities and graph sizes here are public, small card/slot counts."""
    graph = [[] for _ in range(size)]
    for a, b, capacity in edges:
        forward = [b, int(capacity), len(graph[b])]
        backward = [a, 0, len(graph[a])]
        graph[a].append(forward)
        graph[b].append(backward)
    total = 0
    while True:
        level = [-1] * size
        level[source] = 0
        queue = deque([source])
        while queue:
            a = queue.popleft()
            for b, capacity, _ in graph[a]:
                if capacity and level[b] < 0:
                    level[b] = level[a] + 1
                    queue.append(b)
        if level[sink] < 0:
            return total
        cursor = [0] * size

        def push(a: int, amount: int) -> int:
            if a == sink:
                return amount
            while cursor[a] < len(graph[a]):
                edge = graph[a][cursor[a]]
                b, capacity, reverse = edge
                if capacity and level[b] == level[a] + 1:
                    sent = push(b, min(amount, capacity))
                    if sent:
                        edge[1] -= sent
                        graph[b][reverse][1] += sent
                        return sent
                cursor[a] += 1
            return 0

        while (sent := push(source, 1 << 30)):
            total += sent


def transport_feasible(counts: np.ndarray, capacities: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    """Whether code supplies can exactly fill groups with integral lower/upper bounds on code/group counts."""
    arrays = [np.asarray(a) for a in (counts, capacities, lower, upper)]
    if any(a.dtype.kind not in "iu" for a in arrays):
        raise ValueError("transport counts and bounds must be non-boolean integer arrays")
    counts, capacities, lower, upper = (a.astype(np.int64) for a in arrays)
    if lower.shape != (len(counts), len(capacities)) or upper.shape != lower.shape:
        raise ValueError("transport bounds differ from code/group dimensions")
    if (lower < 0).any() or (upper < lower).any() or counts.sum() != capacities.sum():
        return False
    supplies, demands = counts - lower.sum(1), capacities - lower.sum(0)
    if (supplies < 0).any() or (demands < 0).any():
        return False
    source, sink = len(counts) + len(capacities), len(counts) + len(capacities) + 1
    edges = [(source, i, int(n)) for i, n in enumerate(supplies) if n]
    edges += [(len(counts) + j, sink, int(n)) for j, n in enumerate(demands) if n]
    edges += [(i, len(counts) + j, int(upper[i, j] - lower[i, j]))
              for i in range(len(counts)) for j in range(len(capacities)) if upper[i, j] > lower[i, j]]
    return _max_flow(sink + 1, edges, source, sink) == int(supplies.sum())


@dataclass(frozen=True)
class _Pool:
    counts: tuple[int, ...]
    capacities: tuple[int, ...]
    lower: tuple[tuple[int, ...], ...]
    upper: tuple[tuple[int, ...], ...]
    hand: int | None


@dataclass(frozen=True)
class FieldSlot:
    location: int             # core location bit, not model location ID
    sequence: int
    pool: int
    group: int


class PublicLayoutLaw:
    """Field slots in public (location, sequence) order, then nondecreasing raw card codes in the hidden hand.

    ``field_allowed`` optionally tightens a slot with *publicly witnessed* category constraints. It maps a public
    field (core location, sequence) to allowed card codes; it never receives a true hidden identity. The caller
    must retain the public-evidence provenance. This module does not claim the current native world exports every
    possible category witness; missing required witnesses must be handled by the future integration gate.
    """

    def __init__(self, world: Mapping, *, field_allowed: Mapping[tuple[int, int], Sequence[int]] | None = None):
        if not isinstance(world, Mapping) or set(world) - WORLD_KEYS:
            raise ValueError("AR public world has unknown fields (truth/label channels are forbidden)")
        if world.get("decklist_public") is not True:
            raise ValueError("AR belief requires an explicitly declared public decklist")
        _validate_world(world)
        self.codes = tuple(sorted(set(world["pool_main"]) | set(world["pool_extra"])))
        if any(type(c) is not int or c <= 0 for c in self.codes):
            raise ValueError("public pool codes are positive integers")
        if any(type(n) is not int or n < 0 for key in ("pool_main", "pool_extra") for n in world[key].values()):
            raise ValueError("public pool counts are nonnegative integers")
        self.index = {code: i for i, code in enumerate(self.codes)}
        base_groups = P._groups(world)
        fields, pools = [], []
        hand_size = sum(code == 0 for code in world["hand"])
        field_allowed = dict(field_allowed or {})
        for key in field_allowed:
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError("public field category keys are (location, sequence)")
            _integer(key[0], "public category location")
            _integer(key[1], "public category sequence")
        known_fields = {(loc, seq) for loc, seq, shown, _ in world["facedown"]
                        if not shown and loc in (P.LOCATION_MZONE, P.LOCATION_SZONE)}
        if set(field_allowed) - known_fields:
            raise ValueError("public category masks name unknown or already shown field slots")
        for pool_index, (key, old_groups) in enumerate(zip(("pool_main", "pool_extra"), base_groups)):
            # The shuffled subgroup has only identity lower bounds, no extra eligibility rules; existence of an
            # allocation to it is equivalent to these bounds on the total hidden-hand multiset plus its capacity.
            hand_groups = [g for g in old_groups if g.name in ("hand", "hand_group")]
            hand_lower = Counter()
            for group in hand_groups:
                if sum(group.lower.values()) > group.capacity:
                    raise ValueError("known shuffled-hand identities exceed the public subgroup capacity")
                hand_lower.update(group.lower)
            groups, hand_index = [], None
            if hand_groups:
                hand_index = 0
                groups.append(P.Group("hand", list(range(hand_size)), lambda code: True, dict(hand_lower)))
            for group in old_groups:
                if group.name in ("hand", "hand_group"):
                    continue
                predicted = group.name in ("monster", "spell", "field", "extra_monster")
                if predicted:
                    for place in group.places:
                        location, sequence = place[1]
                        allowed = frozenset(field_allowed.get((location, sequence), self.codes))
                        if any(type(c) is not int or c not in self.index for c in allowed):
                            raise ValueError("public field category mask has a code outside the public pools")
                        eligibility = lambda code, base=group.eligible, allow=allowed: base(code) and code in allow
                        fields.append(FieldSlot(location, sequence, pool_index, len(groups)))
                        groups.append(P.Group(group.name, [place], eligibility))
                else:
                    groups.append(group)
            if any(code not in self.index and n for group in groups for code, n in group.lower.items()):
                raise ValueError("public lower bound names a card outside the known pools")
            counts = tuple(world[key].get(code, 0) for code in self.codes)
            lower = tuple(tuple(int(group.lower.get(code, 0)) for group in groups) for code in self.codes)
            upper = tuple(tuple(min(counts[ci], group.capacity) if group.eligible(code) else 0 for group in groups)
                          for ci, code in enumerate(self.codes))
            pools.append(_Pool(counts, tuple(g.capacity for g in groups), lower, upper, hand_index))
        self.pools = tuple(pools)
        self.fields = tuple(sorted(fields, key=lambda s: (s.location, s.sequence)))
        if len({(s.location, s.sequence) for s in self.fields}) != len(self.fields):
            raise ValueError("duplicate public field slot")
        self.hand_size = hand_size
        self.length = len(self.fields) + hand_size
        if not self.feasible(()):
            raise ValueError("no complete assignment satisfies the public constraints")

    def _prefix(self, prefix: Sequence[int]) -> tuple[int, ...]:
        prefix = tuple(prefix)
        if len(prefix) > self.length or any(type(code) is not int or code not in self.index for code in prefix):
            raise ValueError("AR prefix is too long or contains a code outside the public vocabulary")
        hand = prefix[len(self.fields):]
        if tuple(sorted(hand)) != hand:
            raise ValueError("the hidden hand prefix must be nondecreasing by raw card code")
        return prefix

    def feasible(self, prefix: Sequence[int]) -> bool:
        prefix = self._prefix(prefix)
        hand_prefix = prefix[len(self.fields):]
        generated = Counter(hand_prefix)
        for pi, pool in enumerate(self.pools):
            lower = np.asarray(pool.lower, np.int64).reshape(len(self.codes), len(pool.capacities)).copy()
            upper = np.asarray(pool.upper, np.int64).reshape(lower.shape).copy()
            for slot, code in zip(self.fields, prefix):
                if slot.pool != pi:
                    continue
                ci = self.index[code]
                if upper[ci, slot.group] < 1:
                    return False
                lower[:, slot.group] = upper[:, slot.group] = 0
                lower[ci, slot.group] = upper[ci, slot.group] = 1
            if pool.hand is not None and hand_prefix:
                for ci, code in enumerate(self.codes):
                    lower[ci, pool.hand] = max(lower[ci, pool.hand], generated[code])
                    if code < hand_prefix[-1] or len(hand_prefix) == self.hand_size:
                        upper[ci, pool.hand] = min(upper[ci, pool.hand], generated[code])
            if not transport_feasible(np.asarray(pool.counts, np.int64), np.asarray(pool.capacities, np.int64),
                                      lower, upper):
                return False
        return True

    def mask(self, prefix: Sequence[int]) -> np.ndarray:
        prefix = self._prefix(prefix)
        if len(prefix) == self.length:
            raise ValueError("a complete layout has no next token")
        if not self.feasible(prefix):
            raise ValueError("AR prefix has no complete public-consistent continuation")
        hand = prefix[len(self.fields):]
        return np.asarray([False if hand and code < hand[-1] else self.feasible((*prefix, code))
                           for code in self.codes], bool)

    def layout(self, sequence: Sequence[int]) -> dict:
        sequence = self._prefix(sequence)
        if len(sequence) != self.length or not self.feasible(sequence):
            raise ValueError("a layout must be a complete public-consistent canonical sequence")
        return {"schema": SCHEMA,
                "facedown": [[s.location, s.sequence, code] for s, code in zip(self.fields, sequence)],
                "hidden_hand": list(sequence[len(self.fields):]),
                "residual_assignment_exists": True}


def masked_log_probabilities(logits: Sequence[float], mask: np.ndarray) -> np.ndarray:
    values, mask = np.asarray(logits, np.float64), np.asarray(mask, bool)
    if values.ndim != 1 or values.shape != mask.shape or not np.isfinite(values).all() or not mask.any():
        raise ValueError("AR token probabilities need finite logits and a nonempty feasible mask")
    out = np.full(values.shape, -np.inf)
    active = values[mask] - values[mask].max()
    out[mask] = active - math.log(float(np.exp(active).sum()))
    return out


def sample(law: PublicLayoutLaw, score: Callable[[tuple[int, ...]], Sequence[float]], *, seed: int) -> dict:
    """Draw one canonical layout from public-only prefix scores, recording its normalized AR joint probability."""
    rng, prefix, log_probability = np.random.Generator(np.random.PCG64(seed)), (), 0.0
    while len(prefix) < law.length:
        logp = masked_log_probabilities(score(prefix), law.mask(prefix))
        ci = int(rng.choice(len(law.codes), p=np.exp(logp)))
        prefix += (law.codes[ci],)
        log_probability += float(logp[ci])
    return {**law.layout(prefix), "sequence": list(prefix), "log_probability": log_probability, "seed": seed}
