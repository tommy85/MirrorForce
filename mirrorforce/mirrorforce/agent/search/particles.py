"""Particles for belief-world search (search plan S1): abstract hidden worlds drawn from public information.

A particle is one assignment of card identities to every place the viewer cannot see: its opponent's hidden hand,
deck (in order), face-down field and banished cards and face-down extra deck, and the order of its own deck (hidden
even from its owner). It is drawn from ``SearchDuel.public_world(viewer)`` only -- what the viewer's observation shows
at each place, the identities it knows without their positions (obs:unpositioned_), its own deck's cards and the pool
the hidden places hold, which exists only when the opponent's decklist is declared public (public_opponent_recipe:
the decklist minus every card of the opponent the viewer has seen). So two hidden truths behind one public history
give the same world, and the same seed gives byte-identical particles.

Distribution: uniform over the assignments that respect every constraint -- the pool exactly, a face-down monster
zone holds a monster, a face-down spell/trap zone a spell or trap (the field zone a field spell), an extra-deck-kind
place an extra deck card, and the hand and deck (and extra deck) hold at least their known identities -- or, with a
weight hook ``weights(location_id, code)``, proportional to the product over places of the weight of the identity
placed there (the belief head's per-location scores, later). It is exact: a dynamic program over the codes of a pool
with the remaining capacity of every constrained group as its state counts the weighted assignments, and particles
are drawn from it code by code; within a group the identities take its places in uniform random order (the deck's
order is a uniform shuffle). Nothing is ever corrected or rejected: a world no assignment satisfies raises.

Realization (``realize``) is a separate step: it writes a particle into a SearchDuel through permute_hidden, which
refuses anything that breaks what the viewer sees and can only reorder the true pool (which a public decklist
guarantees), or through replace_hidden, which rewrites the hidden identities in place for any pool (with a
registered dormant table, so that the world does not keep the true deck's load-time registrations).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Callable, Optional

import numpy as np

SCHEMA = "mirrorforce_particles/v1"
TYPE_MONSTER, TYPE_SPELL, TYPE_TRAP, TYPE_FIELD = 0x1, 0x2, 0x4, 0x80000
LOCATION_MZONE, LOCATION_SZONE, LOCATION_REMOVED = 0x04, 0x08, 0x20
FIELD_ZONE = 5
# location ids of the weight hook and the metrics (as in cards_): deck 1, hand 2, monster zone 3, spell/trap zone 4,
# banished 6, extra deck 7
LOCATION_ID = {"hand": 2, "hand_group": 2, "deck": 1, "monster": 3, "spell": 4, "field": 4, "banished": 6,
               "extra": 7, "extra_monster": 3, "extra_banished": 6}

Weights = Callable[[int, int], float]


@dataclass
class Group:
    """Hidden places that take identities from one pool under one eligibility rule."""
    name: str
    places: list            # opaque place keys, in a fixed order
    eligible: Callable[[int], bool]
    lower: dict = field(default_factory=dict)  # code -> copies the group must hold

    @property
    def capacity(self):
        return len(self.places)


@dataclass
class Pool:
    """One pool (main or extra) and the groups its cards fill; the last group takes what the others leave."""
    counts: dict            # code -> copies
    groups: list
    codes: list = field(init=False)
    radix: list = field(init=False)
    tables: list = field(init=False)
    log_scale: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.codes = sorted(c for c, n in self.counts.items() if n > 0)
        self.radix = [g.capacity + 1 for g in self.groups[:-1]]


def _check_world(world):
    for key in ("hand", "hand_group", "deck", "extra", "facedown", "pool_main", "pool_extra", "unpositioned",
                "own_deck", "own_deck_fixed", "types"):
        if key not in world:
            raise ValueError(f"a public world names {key}")


def _groups(world):
    """The main and extra pools' groups of hidden places (place keys: (area, index or (location, sequence)))."""
    types = world["types"]
    lower = {"hand": {}, "deck": {}, "extra": {}}
    for (location, code), copies in world["unpositioned"].items():
        lower[{0x02: "hand", 0x01: "deck", 0x40: "extra"}[location]][code] = copies
    # the hand's known identities are among the cards of the viewer's shuffled group, not among the cards that came
    # to the hand since: two groups of hand places, the bound on the first
    group = set(world["hand_group"])
    hand_group = [("hand", i) for i, code in enumerate(world["hand"]) if code == 0 and i in group]
    hand = [("hand", i) for i, code in enumerate(world["hand"]) if code == 0 and i not in group]
    deck = [("deck", i) for i, code in enumerate(world["deck"]) if code == 0]
    extra = [("extra", i) for i, code in enumerate(world["extra"]) if code == 0]
    monster, spell, fieldzone, banished, extra_monster, extra_banished = [], [], [], [], [], []
    for location, sequence, shown, extra_kind in world["facedown"]:
        if shown:
            continue
        place = ("facedown", (location, sequence))
        if extra_kind:
            (extra_banished if location == LOCATION_REMOVED else extra_monster).append(place)
        elif location == LOCATION_MZONE:
            monster.append(place)
        elif location == LOCATION_SZONE:
            (fieldzone if sequence == FIELD_ZONE else spell).append(place)
        else:
            banished.append(place)
    any_card = lambda code: True  # noqa: E731
    main = [Group("hand_group", hand_group, any_card, lower["hand"]),
            Group("hand", hand, any_card),
            Group("monster", monster, lambda c: bool(types[c] & TYPE_MONSTER)),
            Group("spell", spell, lambda c: bool(types[c] & (TYPE_SPELL | TYPE_TRAP)) and not types[c] & TYPE_FIELD),
            Group("field", fieldzone, lambda c: bool(types[c] & TYPE_FIELD)),
            Group("banished", banished, any_card),
            Group("deck", deck, any_card, lower["deck"])]
    extra_groups = [Group("extra_monster", extra_monster, any_card),
                    Group("extra_banished", extra_banished, any_card),
                    Group("extra", extra, any_card, lower["extra"])]
    return main, extra_groups


def _splits(n, groups, code, state, radix_caps):
    """Every way to put ``n`` copies of ``code`` into the groups (the last takes the rest) within the remaining
    capacities ``state``, honoring eligibility and lower bounds: tuples of per-group counts."""
    out = []
    last = groups[-1]
    k = len(groups) - 1

    def rec(i, left, acc):
        if i == k:
            if left < last.lower.get(code, 0) or (left and not last.eligible(code)):
                return
            out.append(tuple(acc) + (left,))
            return
        g = groups[i]
        low = g.lower.get(code, 0)
        high = min(left, state[i]) if g.eligible(code) else 0
        for c in range(low, high + 1):
            acc.append(c)
            rec(i + 1, left - c, acc)
            acc.pop()

    rec(0, n, [])
    return out


def _weight(groups, code, split, weights):
    value = 1.0
    for g, c in zip(groups, split):
        if c:
            w = 1.0 if weights is None else float(weights(LOCATION_ID[g.name], code))
            if w < 0 or not math.isfinite(w):
                raise ValueError(f"weight {w} for {code} at {g.name}")
            value *= w ** c / math.factorial(c)
    return value


def _prepare(pool, weights):
    """Fills pool.tables: tables[i][state] is the weighted count of assignments of codes i.. into the remaining
    capacities ``state`` (mixed radix over every group but the last), each table rescaled to stay finite."""
    groups = pool.groups
    radix = pool.radix
    size = int(np.prod(radix)) if radix else 1
    states = [np.unravel_index(s, radix) if radix else () for s in range(size)]
    table = np.zeros(size)
    table[0] = 1.0  # every constrained group exactly filled
    tables = [table]
    log_scale = 0.0
    for code in reversed(pool.codes):
        n = pool.counts[code]
        nxt = np.zeros(size)
        for s in range(size):
            state = states[s]
            total = 0.0
            for split in _splits(n, groups, code, state, radix):
                rest = tuple(st - c for st, c in zip(state, split[:-1]))
                idx = int(np.ravel_multi_index(rest, radix)) if radix else 0
                if tables[-1][idx]:
                    total += _weight(groups, code, split, weights) * tables[-1][idx]
            nxt[s] = total
        peak = nxt.max()
        if peak > 0:
            nxt /= peak
            log_scale += math.log(peak)
        tables.append(nxt)
    pool.tables = tables[::-1]  # tables[i] for codes i.., tables[len(codes)] the base
    pool.log_scale = log_scale


def _full(pool):
    return tuple(g.capacity for g in pool.groups[:-1])


def _index(pool, state):
    return int(np.ravel_multi_index(state, pool.radix)) if pool.radix else 0


class Sampler:
    """Draws particles of one public world (``SearchDuel.public_world``), optionally weighted per location."""

    def __init__(self, world, weights: Optional[Weights] = None):
        if not world.get("decklist_public"):
            raise ValueError("the sampler draws from a declared decklist (public_opponent_recipe); this world has none")
        _check_world(world)
        self.world = world
        self.weights = weights
        main_groups, extra_groups = _groups(world)
        self.pools = [Pool(dict(world["pool_main"]), main_groups), Pool(dict(world["pool_extra"]), extra_groups)]
        for pool in self.pools:
            slots = sum(g.capacity for g in pool.groups)
            if sum(pool.counts.values()) != slots:
                raise ValueError(f"a pool of {sum(pool.counts.values())} cards for {slots} hidden places")
            _prepare(pool, weights)
            if pool.tables[0][_index(pool, _full(pool))] <= 0:
                raise ValueError("no assignment satisfies the public constraints of this world")
        fixed = dict(world["own_deck_fixed"])
        free = list(world["own_deck"])
        for code in fixed.values():
            free.remove(code)
        self.own_free = sorted(free)
        self.own_fixed = fixed
        self.own_size = len(world["own_deck"])

    def log_partition(self, pool):
        return math.log(pool.tables[0][_index(pool, _full(pool))]) + pool.log_scale

    def counts_of(self, pool, rng):
        """One count matrix of a pool: per code, its copies per group."""
        state = list(_full(pool))
        out = {}
        for i, code in enumerate(pool.codes):
            options, mass = [], []
            for split in _splits(pool.counts[code], pool.groups, code, state, pool.radix):
                rest = tuple(st - c for st, c in zip(state, split[:-1]))
                value = pool.tables[i + 1][_index(pool, rest)]
                if value:
                    options.append(split)
                    mass.append(_weight(pool.groups, code, split, self.weights) * value)
            mass = np.asarray(mass)
            choice = options[int(rng.choice(len(options), p=mass / mass.sum()))]
            out[code] = choice
            state = [st - c for st, c in zip(state, choice[:-1])]
        if any(state):
            raise RuntimeError("the sampled counts do not fill the hidden places")
        return out

    def particle(self, rng):
        world = self.world
        p = {"hand": list(world["hand"]), "deck": list(world["deck"]), "extra": list(world["extra"]),
             "facedown": [[loc, seq, shown] for loc, seq, shown, _ in world["facedown"]]}
        facedown_at = {(loc, seq): i for i, (loc, seq, _) in enumerate(p["facedown"])}
        for pool in self.pools:
            counts = self.counts_of(pool, rng)
            for gi, group in enumerate(pool.groups):
                codes = [code for code in pool.codes for _ in range(counts[code][gi])]
                order = rng.permutation(len(codes)) if codes else []
                for place, k in zip(group.places, order):
                    area, where = place
                    if area == "facedown":
                        p["facedown"][facedown_at[where]][2] = codes[k]
                    else:
                        p[area][where] = codes[k]
        own = [0] * self.own_size
        for seq, code in self.own_fixed.items():
            own[seq] = code
        order = rng.permutation(len(self.own_free))
        free_places = [i for i in range(self.own_size) if i not in self.own_fixed]
        for place, k in zip(free_places, order):
            own[place] = self.own_free[k]
        p["own_deck"] = own
        return p

    def sample(self, seed, n):
        """``n`` particles from ``seed`` (numpy PCG64): the same world and seed give the same particles."""
        rng = np.random.Generator(np.random.PCG64(seed))
        return [self.particle(rng) for _ in range(n)]

    def log_likelihood(self, counts):
        """The log-probability of per-group identity counts ({group name: {code: copies}}, both pools), e.g. the
        truth's (an audit metric; -inf when the constraints exclude them)."""
        total = 0.0
        for pool in self.pools:
            score = 0.0
            for code in pool.codes:
                split = tuple(counts.get(g.name, {}).get(code, 0) for g in pool.groups)
                if sum(split) != pool.counts[code]:
                    return -math.inf
                for g, c in zip(pool.groups, split):
                    if c < g.lower.get(code, 0) or (c and not g.eligible(code)):
                        return -math.inf
                score += math.log(_weight(pool.groups, code, split, self.weights) or 1e-300)
            if sum(sum(counts.get(g.name, {}).values()) for g in pool.groups) != sum(pool.counts.values()):
                return -math.inf
            total += score - self.log_partition(pool)
        return total

    def group_counts(self, particle):
        """Per-group identity counts of a particle (for the metrics)."""
        out = {}
        for pool in self.pools:
            for g in pool.groups:
                c = out.setdefault(g.name, {})
                for area, where in g.places:
                    code = particle["facedown"][[tuple(x[:2]) for x in particle["facedown"]].index(where)][2] \
                        if area == "facedown" else particle[area][where]
                    c[code] = c.get(code, 0) + 1
        return out


def encode(particles):
    """Canonical bytes of particles (the two-truth identity check)."""
    return json.dumps({"schema": SCHEMA, "particles": particles}, sort_keys=True, separators=(",", ":")).encode()


def particle_recipe(world, particle):
    """The opponent's recipe a particle implies: the cards of its the viewer sees (``public_owned``) plus the particle's
    identities at the places the viewer does not see -- hand and deck places hold main-deck cards, a face-down place
    its kind, extra deck places extra-deck cards. Returns ``(main, extra)`` sorted; replace_hidden installs it."""
    main = [code for code, extra in world["public_owned"] if not extra]
    extra = [code for code, kind in world["public_owned"] if kind]
    for area in ("hand", "deck"):
        main += [code for code, shown in zip(particle[area], world[area]) if shown == 0]
    for slot, shown in zip(particle["facedown"], world["facedown"]):
        if shown[2] == 0:
            (extra if shown[3] else main).append(slot[2])
    extra += [code for code, shown in zip(particle["extra"], world["extra"]) if shown == 0]
    return sorted(main), sorted(extra)


def realize(duel, viewer, particle, replace=False, world=None):
    """Writes a particle into a SearchDuel: the opponent's hidden places (extra deck included) and the viewer's own
    deck order. The opponent's places go through permute_hidden, which can only reorder the true hidden cards (a
    public decklist guarantees the particle holds them), or with ``replace`` through replace_hidden, which gives the
    places the particle's identities in place whatever the true cards are and installs the recipe the particle implies
    (``particle_recipe`` over ``world``, the viewer's public world; it needs a registered dormant table); both refuse
    anything that changes what the viewer sees. The own deck order always goes through permute_hidden."""
    opponent = 1 - viewer
    facedown = [tuple(x) for x in particle["facedown"]]
    if replace:
        if world is None:
            raise ValueError("replacing needs the viewer's public world (the particle's recipe)")
        main, extra = particle_recipe(world, particle)
        duel.replace_hidden(opponent, viewer=viewer, hand=particle["hand"], deck=particle["deck"], facedown=facedown,
                            extra=particle["extra"], recipe_main=main, recipe_extra=extra)
    else:
        duel.permute_hidden(opponent, viewer=viewer, hand=particle["hand"], deck=particle["deck"], facedown=facedown,
                            extra=particle["extra"])
    own = duel.hidden_layout(viewer)  # the viewer's own cards: it sees them all; only its deck order is redrawn
    duel.permute_hidden(viewer, viewer=viewer, hand=own["hand"], deck=particle["own_deck"], facedown=own["facedown"])
