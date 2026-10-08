"""Exact, bounded counting of distinct public-capacity-compatible root layouts.

The sample space is roots, NOT historical color/UID witnesses. A Boolean
existence oracle eliminates the historical variables. Root variables are
grouped only when their full constraint incidence, fixed facts and public
World group membership prove exchange symmetry. Integer multinomial weights
count the distinct slot assignments inside a group, exactly once each.

This correctness baseline can be expensive. Exhausted counting/oracle budgets
fail the entire construction; unknown feasibility is never treated as false.
There is no proposal rejection loop and no claim of script-conditioned or
opponent-policy-conditioned posterior sampling.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import time

from . import constants as C
from .agent_causal_plan import _graph_problem, _sha, _Solver, PlanBudgetExceeded, PlanIncompatible
from ..agent.public_world_codec import encode_world, decode_world, world_sha256
from ..agent.search.particles import _groups

INFORMATION_SET_SEARCH = True
LAW = "public-world-and-position-capacity-uniform/v1"
SCHEMA = "mirrorforce_causal_root_sampler/v1"


class SamplingError(ValueError):
    pass


class SamplingBudgetExceeded(SamplingError):
    pass


@dataclass(frozen=True)
class _Class:
    tokens: tuple
    places: tuple
    pool: int
    group: str
    codes: tuple
    incidence: tuple


class _Random:
    """Versioned SHA-256 counter stream; exact unbiased arbitrary-size integers."""
    def __init__(self, seed):
        if type(seed) is not int:
            raise SamplingError("independent sampler seed must be an exact integer")
        self.seed, self.counter = seed, 0

    def below(self, bound):
        if bound <= 0:
            raise SamplingError("cannot sample from empty exact mass")
        bits = bound.bit_length()
        while True:
            raw = b""
            while len(raw) * 8 < bits:
                raw += hashlib.sha256(f"{LAW}:{self.seed}:{self.counter}".encode()).digest()
                self.counter += 1
            value = int.from_bytes(raw, "big") >> (8 * len(raw) - bits)
            if value < bound:
                return value

    def shuffle(self, values):
        for i in range(len(values) - 1, 0, -1):
            j = self.below(i + 1)
            values[i], values[j] = values[j], values[i]


class CapacitySampler:
    """One fully counted support, followed by draws with no repairs or retries.

    ``max_states`` bounds counting/composition states. ``max_oracle_nodes`` is
    a shared total across ALL historical feasibility calls, not a renewable
    per-branch allowance. The absolute deadline covers compile/count/sample.
    """
    def __init__(self, history, world, *, deadline, max_states=10000, max_oracle_nodes=100000,
                 clock=time.monotonic):
        if type(deadline) not in (int, float) or not math.isfinite(deadline) or not callable(clock) \
                or type(max_states) is not int or max_states < 1 \
                or type(max_oracle_nodes) is not int or max_oracle_nodes < 1:
            raise SamplingError("exact counting requires explicit finite, positive budgets")
        self.deadline, self.clock = deadline, clock
        self.max_states, self.max_oracle_nodes = max_states, max_oracle_nodes
        self.states = self.oracle_nodes = self.oracle_calls = 0
        self.cache = {}
        self._bounds = {}
        self._existence_path = []
        self.propagation_rejections = self.witness_reuses = 0
        self._check()
        encoded = encode_world(world, complete=True)
        self.world = decode_world(encoded, complete=True)
        self.world_sha256 = world_sha256(encoded)
        self.record, self.domains, self.constraints = _graph_problem(history)
        self.history_sha256, self.viewer = _sha(self.record), history.viewer
        self.slots = {(p, z, s): token - 1 for p, z, s, token in self.record["root_slots"]}
        self.fixed = {place: next(iter(self.domains[token])) for place, token in self.slots.items()
                      if len(self.domains[token]) == 1}
        self.classes, self.pools, self.lower = self._compile()
        self.root_count = self._mass(())
        if not self.root_count:
            raise SamplingError("no distinct root layout satisfies the World and complete public capacity graph")
        self._check()

    def _check(self, *, state=False):
        if self.clock() >= self.deadline:
            raise SamplingBudgetExceeded("history-conditioned exact sampling deadline exhausted")
        if state:
            self.states += 1
            if self.states > self.max_states:
                raise SamplingBudgetExceeded("history-conditioned counting-state budget exhausted")

    def _compile(self):
        world, opponent = self.world, 1 - self.viewer
        incidence = [[] for _ in self.domains]
        for ci, (left, right, _) in enumerate(self.constraints):
            for token in left:
                incidence[token].append((ci, 1))
            for token in right:
                incidence[token].append((ci, -1))
        groups = _groups(world)
        pools = [Counter(world["pool_main"]), Counter(world["pool_extra"]), Counter(world["own_deck"])]
        mutable, lower = {}, {}

        def token_at(place):
            if place not in self.slots:
                raise SamplingError("public World names an absent physical root position")
            return self.slots[place]

        def fixed(place, code):
            token = token_at(place)
            if type(code) is not int or code not in self.domains[token]:
                raise SamplingError("public World identity conflicts with the complete position history")
            self.domains[token] = {code}
            self.fixed[place] = code

        for name, location in (("hand", C.LOCATION_HAND), ("deck", C.LOCATION_DECK), ("extra", C.LOCATION_EXTRA)):
            expected = {place for place in self.slots if place[:2] == (opponent, location)}
            places = {(opponent, location, i) for i in range(len(world[name]))}
            if places != expected:
                raise SamplingError("public World hidden-zone counts differ from the root history")
            for i, code in enumerate(world[name]):
                if code:
                    fixed((opponent, location, i), code)
        for location, sequence, code, _ in world["facedown"]:
            token_at((opponent, location, sequence))
            if code:
                fixed((opponent, location, sequence), code)
        for pi, pool_groups in enumerate(groups):
            if sum(g.capacity for g in pool_groups) != sum(pools[pi].values()):
                raise SamplingError("public hidden pool does not fill its declared positions")
            for group in pool_groups:
                name = f"{pi}:{group.name}"
                lower[name] = Counter(group.lower)
                for area, index in group.places:
                    location, sequence = (index if area == "facedown" else (
                        {"hand": C.LOCATION_HAND, "deck": C.LOCATION_DECK, "extra": C.LOCATION_EXTRA}[area], index))
                    place = (opponent, location, sequence)
                    if place in mutable:
                        raise SamplingError("a public hidden place belongs to two pool groups")
                    domain = self.domains[token_at(place)] & {c for c, n in pools[pi].items() if n and group.eligible(c)}
                    mutable[place] = (pi, name, tuple(sorted(domain)))
        own_places = {(self.viewer, C.LOCATION_DECK, i) for i in range(len(world["own_deck"]))}
        if own_places != {place for place in self.slots if place[:2] == (self.viewer, C.LOCATION_DECK)}:
            raise SamplingError("own-deck World count differs from the public history")
        for sequence, code in world["own_deck_fixed"].items():
            fixed((self.viewer, C.LOCATION_DECK, sequence), code)
            pools[2][code] -= 1
            if pools[2][code] < 0:
                raise SamplingError("own fixed deck identities exceed the declared own inventory")
        lower["2:own_deck"] = Counter()
        for place in sorted(own_places):
            if place[2] not in world["own_deck_fixed"]:
                domain = self.domains[self.slots[place]] & {c for c, n in pools[2].items() if n}
                mutable[place] = (2, "2:own_deck", tuple(sorted(domain)))
        if set(mutable) | self.fixed.keys() != self.slots.keys():
            raise SamplingError("World plus received identities leaves an unmodeled root position")
        by_signature = defaultdict(list)
        for place, (pool, group, codes) in mutable.items():
            token = self.slots[place]
            if not codes:
                raise SamplingError("a hidden root position has no eligible public identity")
            # Full incidence in EVERY historical constraint, not merely a
            # shared opening pool or most recent shuffle. Facts are encoded
            # in the domain; World eligibility and ALL lower-bound membership
            # enter this signature too. Permuting a class preserves each term
            # of every constraint, proving exchange symmetry directly.
            # A group's name is not a constraint. With no positive lower
            # bound it can be forgotten once its complete eligibility domain
            # is included. This proves, for example, that opening hidden hand
            # and deck positions in one unconstrained pool are exchangeable;
            # their output coordinates remain distinct. Any real lower bound
            # keeps its group membership in the signature.
            membership = group if any(n > 0 for n in lower[group].values()) else f"{pool}:no-lower-bound"
            signature = (pool, membership, codes, tuple(incidence[token]))
            by_signature[signature].append(place)
            self.domains[token] = set(codes)
            self.fixed.pop(place, None)  # singleton inferred facts still consume a World-pool copy
        classes = []
        for (pool, group, codes, neighbors), places in by_signature.items():
            ordered = tuple(sorted(places))
            classes.append(_Class(tuple(self.slots[p] for p in ordered), ordered, pool, group, codes, neighbors))
        classes.sort(key=lambda group: (len(group.codes) != 1, len(group.places), group.pool, group.places))
        return tuple(classes), tuple(pools), lower

    def _remaining(self, prefix):
        pools = [p.copy() for p in self.pools]
        for group, counts in zip(self.classes, prefix):
            pools[group.pool].subtract(dict(counts))
        return pools

    def _lower_possible(self, prefix):
        assigned = defaultdict(Counter)
        for group, counts in zip(self.classes, prefix):
            assigned[group.group].update(dict(counts))
        for name, needs in self.lower.items():
            for code, copies in needs.items():
                possible = assigned[name][code] + sum(len(g.places) for g in self.classes[len(prefix):]
                                                      if g.group == name and code in g.codes)
                if possible < copies:
                    return False
        return True

    def _exists(self, prefix):
        self._check()
        remaining = self.max_oracle_nodes - self.oracle_nodes
        if remaining <= 0:
            raise SamplingBudgetExceeded("shared historical existence-node budget exhausted")
        # Retain only the current DFS ancestry, not a full copy of the token
        # graph for every counted branch. These are propagated DOMAINS, never
        # the chosen colors of one feasible historical witness.
        parents = [entry for entry in self._existence_path
                   if len(entry[0]) < len(prefix) and prefix[:len(entry[0])] == entry[0]]
        parent = parents[-1] if parents else None
        domains = (parent[1] if parent is not None else self.domains).copy()
        for group, counts in zip(self.classes, prefix):
            values = [code for code, n in counts for _ in range(n)]
            for token, code in zip(group.tokens, values):
                if code not in domains[token]:
                    self.propagation_rejections += 1
                    return False
                domains[token] = {code}
        oracle = _Solver(self.constraints, len(domains), 0, remaining, self.deadline, self.clock)
        self.oracle_calls += 1
        propagated = False
        try:
            oracle.propagate(domains)
            propagated = True
            colors = None
            if parent is not None:
                # A permutation of one full-incidence exchange class changes
                # no historical equality. Reuse is positive evidence only and
                # STILL checks every original token/domain/Counter exactly.
                candidate = list(parent[2])
                for group, counts in zip(self.classes, prefix):
                    if Counter(candidate[token] for token in group.tokens) != Counter(dict(counts)):
                        break
                    values = [code for code, n in counts for _ in range(n)]
                    for token, code in zip(group.tokens, values):
                        candidate[token] = code
                else:
                    from .causal_linear import exact_witness
                    if exact_witness(tuple(candidate), domains, self.constraints):
                        colors = tuple(candidate)
                        self.witness_reuses += 1
            if colors is None:
                colors = oracle.feasible_witness(domains)
            self._check()
            if len(prefix) < len(self.classes):
                self._bounds[prefix] = self._composition_bounds(prefix, domains)
            self._existence_path = [*parents, (prefix, domains, colors)]
            return True
        except PlanIncompatible:
            if not propagated:
                self.propagation_rejections += 1
            return False
        except PlanBudgetExceeded as exc:
            raise SamplingBudgetExceeded("historical feasibility is unknown: " + str(exc)) from exc
        finally:
            self.oracle_nodes += oracle.nodes

    def _composition_bounds(self, prefix, domains):
        """Necessary integer count intervals; never a sufficient feasibility test.

        For every original equality, coefficient*n + outside = rhs. Bound
        outside using the forced/possible contributions of original tokens.
        Narrow only the count range of the NEXT unchanged exchange class.
        This removes zero-mass compositions without reordering any live ones.
        """
        group = self.classes[len(prefix)]
        tokens = set(group.tokens)
        size = len(tokens)
        bounds = {code: [sum(domains[t] == {code} for t in tokens),
                         sum(code in domains[t] for t in tokens)] for code in group.codes}
        for ci in sorted({ci for ci, _ in group.incidence}):
            self._check(state=True)
            left, right, fixed = self.constraints[ci]
            coefficients = Counter(left)
            if fixed is None:
                coefficients.subtract(right)
            same = {coefficients[token] for token in tokens}
            if len(same) != 1:
                raise SamplingError("root exchange class lost its full-constraint symmetry")
            coefficient = same.pop()
            if not coefficient:
                continue
            for code, bound in bounds.items():
                self._check()
                outside_low = outside_high = 0
                for token, weight in coefficients.items():
                    if token in tokens or not weight or code not in domains[token]:
                        continue
                    if domains[token] == {code}:
                        outside_low += weight
                        outside_high += weight
                    elif weight > 0:
                        outside_high += weight
                    else:
                        outside_low += weight
                rhs = 0 if fixed is None else fixed[code]
                low, high = rhs - outside_high, rhs - outside_low
                divisor = coefficient
                if divisor < 0:
                    low, high, divisor = -high, -low, -divisor
                bound[0] = max(bound[0], -(-low // divisor))
                bound[1] = min(bound[1], high // divisor)
        return tuple((code, max(0, low), min(size, high)) for code, (low, high) in bounds.items())

    def _choices(self, group, available, bounds=None):
        codes = tuple(code for code in group.codes if available[code] > 0)
        size = len(group.places)
        limits = {} if bounds is None else {code: (low, high) for code, low, high in bounds}
        if any(low > available[code] or low > high for code, (low, high) in limits.items()):
            return
        lower = {code: limits.get(code, (0, size))[0] for code in codes}
        upper = {code: min(available[code], limits.get(code, (0, size))[1]) for code in codes}

        def visit(index, needed, selected):
            self._check(state=True)
            if sum(lower[code] for code in codes[index:]) > needed \
                    or sum(upper[code] for code in codes[index:]) < needed:
                return
            if index == len(codes):
                if needed == 0:
                    yield tuple((code, n) for code, n in selected if n)
                return
            code = codes[index]
            for n in range(lower[code], min(upper[code], needed) + 1):
                yield from visit(index + 1, needed - n, (*selected, (code, n)))
        yield from visit(0, size, ())

    @staticmethod
    def _multiplicity(group, counts):
        mass = math.factorial(len(group.places))
        for _, count in counts:
            mass //= math.factorial(count)
        return mass

    def _mass(self, prefix):
        self._check()
        if prefix in self.cache:
            return self.cache[prefix]
        self._check(state=True)
        remaining = self._remaining(prefix)
        if any(n < 0 for pool in remaining for n in pool.values()) or not self._lower_possible(prefix):
            self.cache[prefix] = 0
            return 0
        if not self._exists(prefix):
            self.cache[prefix] = 0
            return 0
        if len(prefix) == len(self.classes):
            mass = int(all(sum(pool.values()) == 0 for pool in remaining))
        else:
            group = self.classes[len(prefix)]
            mass = sum(self._multiplicity(group, counts) * self._mass((*prefix, counts))
                       for counts in self._choices(group, remaining[group.pool], self._bounds.get(prefix)))
        self.cache[prefix] = mass
        return mass

    def sample(self, seed, count):
        if type(count) is not int or count < 1:
            raise SamplingError("sample count must be positive")
        self._check()
        rng, result = _Random(seed), []
        try:
            for _ in range(count):
                result.append(self._sample_one(rng))
                self._check()
        except SamplingError as exc:
            # A timed-out construction is not a retry opportunity. Preserve
            # every already-generated original draw, including the draw whose
            # post-check crossed the deadline, for the failed-bank audit.
            exc.partial_particles = copy.deepcopy(result)
            exc.sampler_proof = self.proof()
            raise
        return result

    def _sample_one(self, rng):
        prefix, assignments = (), dict(self.fixed)
        for group in self.classes:
            available = self._remaining(prefix)[group.pool]
            draw = rng.below(self.cache[prefix])
            chosen = None
            for counts in self._choices(group, available, self._bounds.get(prefix)):
                child = (*prefix, counts)
                mass = self._multiplicity(group, counts) * self.cache[child]
                if draw < mass:
                    chosen = counts
                    break
                draw -= mass
            if chosen is None:
                raise SamplingError("exact counted branches do not cover their declared mass")
            values = [code for code, n in chosen for _ in range(n)]
            rng.shuffle(values)
            assignments.update(zip(group.places, values))
            prefix = (*prefix, chosen)
        opponent = 1 - self.viewer
        particle = {name: [assignments[opponent, location, i] for i in range(len(self.world[name]))]
                    for name, location in (("hand", C.LOCATION_HAND), ("deck", C.LOCATION_DECK), ("extra", C.LOCATION_EXTRA))}
        particle["facedown"] = [[location, sequence, assignments[opponent, location, sequence]]
                                for location, sequence, _, _ in self.world["facedown"]]
        particle["own_deck"] = [assignments[self.viewer, C.LOCATION_DECK, i]
                                for i in range(len(self.world["own_deck"]))]
        return particle

    def probability(self, particle):
        """Exact root mass for independent audits; never a historical-path mass."""
        self._check()
        if not isinstance(particle, dict) or set(particle) != {"hand", "deck", "extra", "facedown", "own_deck"}:
            raise SamplingError("a root mass query needs exactly the five particle fields")
        assignments = dict(self.fixed)

        def bind(place, code):
            if type(code) is not int or code <= 0 or place not in self.slots:
                raise SamplingError("invalid root mass query position or identity")
            if place in self.fixed and self.fixed[place] != code:
                return False
            assignments[place] = code
            return True

        opponent = 1 - self.viewer
        for name, player, location in (("hand", opponent, C.LOCATION_HAND), ("deck", opponent, C.LOCATION_DECK),
                                       ("extra", opponent, C.LOCATION_EXTRA), ("own_deck", self.viewer, C.LOCATION_DECK)):
            values = particle[name]
            if not isinstance(values, (tuple, list)) or len(values) != len(self.world[name]):
                raise SamplingError("root mass query count differs from the public World")
            for i, code in enumerate(values):
                if not bind((player, location, i), code):
                    return Fraction(0)
        expected = {(location, sequence) for location, sequence, _, _ in self.world["facedown"]}
        seen = set()
        if not isinstance(particle["facedown"], (list, tuple)):
            raise SamplingError("root mass query face-down rows must be explicit")
        for row in particle["facedown"]:
            if not isinstance(row, (list, tuple)) or len(row) != 3 or any(type(v) is not int for v in row):
                raise SamplingError("malformed face-down root mass query")
            location, sequence, code = row
            if (location, sequence) in seen or (location, sequence) not in expected:
                raise SamplingError("root mass query repeats or invents a face-down coordinate")
            seen.add((location, sequence))
            if not bind((opponent, location, sequence), code):
                return Fraction(0)
        if seen != expected:
            raise SamplingError("root mass query omits a face-down coordinate")
        prefix = tuple(tuple(sorted(Counter(assignments[place] for place in group.places).items()))
                       for group in self.classes)
        return Fraction(1, self.root_count) if self.cache.get(prefix, 0) else Fraction(0)

    def proof(self):
        return {"schema": SCHEMA, "law": LAW, "history_sha256": self.history_sha256,
                "world_sha256": self.world_sha256, "distinct_root_layouts": self.root_count,
                "root_classes": len(self.classes), "counting_states": self.states,
                "oracle_calls": self.oracle_calls, "oracle_nodes": self.oracle_nodes,
                "propagation_rejections": self.propagation_rejections, "verified_witness_reuses": self.witness_reuses,
                "sampling": "integer-multinomial; sha256-counter-unbiased/v1",
                "historical_witness_multiplicity_counted": False, "proposal_retries": 0,
                "native_replay_admitted": False}
