"""Response classes of the opponent's hidden cards (play-time search design 10.11).

In the seat's own turn the opponent acts only through cards it can use in that turn: monster effects usable from
the hand at spell speed 2 (hand traps), and face-down Spell/Trap cards that activate at spell speed 2 (Traps,
Quick-Play Spells). A response class is which of those the opponent's hidden cards give it: each hand responder
held or not, and each set responder among its face-down Spell/Trap slots or not. Two hidden layouts of one class
give the opponent the same answers in the seat's turn (a second copy matters only for the few responders that are
not once per turn; the class's layouts carry the copies their own draws have).

The class probabilities are exact, not sampled. The public law of the opponent's hidden cards is the unknown pool
shuffled into its unknown slots, conditioned on the public claims (``search.belief``, ``search.field_assignment``):
a category per claimed hand representative and per claimed face-down slot, identities known to be in a face-down
zone without their positions, position-free categories that distinct cards of a zone witness, and zone claims (the
hand, the deck and the extra deck together hold one of some codes; ``ZoneClaim``); the extra pool is shuffled into
its own slots apart from the main deck's. The law, and the belief head's tilt of the opponent's hand
(``exp(power * adjustments[card, copies])`` for each card, ``stage_a_belief.belief_log_weights``), both factor by
card code, so a dynamic program over the codes counts every class: for each code, how many copies go to each group
of slots (the hand, each claim, the free face-down slots of each zone, the extra pool's slots and deck; the rest of
the main pool is the deck), weighted by the ways to choose those copies and by the tilt of the hand's count. Walked
back from a class's end states, the same program draws that class's layouts exactly.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import itertools
import math
import random

from ..cardfeat import load_atomic_ops
from ..netduel import constants as C
from ..search.belief import Particle
from ..search.field_assignment import _extend_matches
from .stage_a_belief import BeliefRecipe
from .stage_a_joint_belief_runtime import JointDraw, JointParticleBank, ZoneClaim
from .stage_a_joint_proposals import evidence_for_public_snapshot, proposal_bank

RESPONSE_CLASS_LAW = "mirrorforce_response_classes/v2"


class ResponseClassError(ValueError):
    """A root whose public evidence the class law cannot hold: claims that contradict its slots or its pool."""


@dataclass(frozen=True)
class Responders:
    """The opponent recipe's cards that can act in the seat's turn."""

    hand: tuple[int, ...]
    set: tuple[int, ...]

    def key(self, hand, facedown):
        """The class of a layout: ``hand`` its codes, ``facedown`` its ``(location, sequence, code)`` slots; the
        responders it holds in hand and the responders it has set, each in the recipe's order."""
        held = set(hand)
        placed = {code for location, _sequence, code in facedown if location == C.LOCATION_SZONE}
        return (tuple(code for code in self.hand if code in held), tuple(code for code in self.set if code in placed))


def responders(codes, index=None) -> Responders:
    """The recipe's hand responders (an effect of spell speed 2 or more usable from the hand) and set responders (an
    activation of spell speed 2 or more), read from the card data every script registers (``cardfeat``)."""
    index = load_atomic_ops() if index is None else index
    hand, placed = [], []
    for code in sorted(set(codes)):
        card = index[code]
        if any(effect.activation_kind == "quick" and effect.spell_speed >= 2 and "hand" in effect.zones
               for effect in card.effects):
            hand.append(code)
        if any(effect.activation_kind == "activate" and effect.spell_speed >= 2 for effect in card.effects):
            placed.append(code)
    return Responders(tuple(hand), tuple(placed))


@dataclass(frozen=True)
class _Group:
    """``size`` unknown slots filled alike, holding codes of ``allowed`` (None: any code of its pool)."""

    size: int
    allowed: frozenset | None = None
    hand: bool = False
    #: the face-down field zone of its slots
    location: int | None = None
    #: the one face-down slot a slot claim names
    slot: tuple | None = None
    #: the extra pool's: its face-down slots, or (``extra_deck``) the face-down extra deck
    extra: bool = False
    extra_deck: bool = False
    #: slots whose cards arrived after the zone's known identities were proven (none of them is there)
    fresh: bool = False


class _Law:
    """The public law of one root's hidden layouts, counted code by code (the module's dynamic program).

    A state is ``(slots each group still needs, the class so far, which zone claims a card has settled, the
    position-free category subsets distinct zone cards witness)``; ``layers[k]`` holds every state reachable before
    the ``k``-th code with the weight of the ways to reach it."""

    def __init__(self, evidence, responders_, *, tilt=None, claims=()):
        self.evidence, self.responders, self.tilt = evidence, responders_, tilt
        self.groups = self._groups(evidence)
        pool = evidence.unknown_pool()
        hand_n, facedown_n, other_n, deck_n = evidence.unknown_slots()
        if sum(pool.values()) != hand_n + facedown_n + other_n + deck_n:
            raise ResponseClassError("the evidence's unknown slots do not hold its unknown pool")
        extra_pool = Counter()
        if evidence.extra_slot_keys:
            extra_pool = evidence.unknown_extra_pool()
            if sum(extra_pool.values()) != sum(evidence.unknown_extra_slots()):
                raise ResponseClassError("the evidence's extra slots do not hold its extra pool")
        self.claims = self._open_claims(evidence, claims, extra_pool)
        special = set(responders_.hand) | set(responders_.set)
        # The extra pool first and the codes that change a class last, so the class part of the state grows only at
        # the end.
        self.codes = [(code, extra_pool[code], True) for code in sorted(extra_pool)]
        self.codes += [(code, pool[code], False) for code in sorted(pool, key=lambda code: (code in special, code))]
        self.members = [tuple(number for number, group in enumerate(self.groups)
                              if group.extra == extra and (group.allowed is None or code in group.allowed))
                        for code, _copies, extra in self.codes]
        self.required = Counter((int(location), int(code)) for location, code in evidence.unpositioned_facedown)
        self.categories = tuple((int(location), frozenset(codes))
                                for location, codes in evidence.unpositioned_facedown_categories)
        matches = (0,)
        for location, code in evidence.fixed_field_identities:
            matches = _extend_matches(matches, self._eligible(code, location))
        self.full = (1 << len(self.categories)) - 1
        self.initial = (tuple(group.size for group in self.groups), ((), ()), (False,) * len(self.claims), matches)
        self.layers = [{self.initial: 1.0}]
        for number in range(len(self.codes)):
            grown = {}
            for state, weight in self.layers[-1].items():
                for _take, ways, after in self._moves(number, state):
                    grown[after] = grown.get(after, 0.0) + weight * ways
            self.layers.append(grown)

    @staticmethod
    def _groups(evidence):
        hand_n, facedown_n, other_n, _deck_n = evidence.unknown_slots()
        keys = tuple(tuple(key) for key in evidence.facedown_sampling_keys)
        claimed = [tuple(key) for key, _codes in evidence.facedown_categories]
        if len(keys) != facedown_n or len(set(keys)) != len(keys) or len(set(claimed)) != len(claimed) \
                or not set(claimed) <= set(keys) or len(evidence.hand_categories) > hand_n:
            raise ResponseClassError("the evidence's face-down slots and claims do not partition its keys")
        groups = [_Group(1, frozenset(codes), hand=True) for codes in evidence.hand_categories]
        groups.append(_Group(hand_n - len(evidence.hand_categories), hand=True))
        fresh = {tuple(key) for key in evidence.fresh_facedown_keys}
        if not fresh <= set(keys):
            raise ResponseClassError("a fresh slot is not one of the evidence's face-down slots")
        groups += [_Group(1, frozenset(codes), location=int(key[0]), slot=tuple(key), fresh=tuple(key) in fresh)
                   for key, codes in evidence.facedown_categories]
        free = Counter((int(key[0]), key in fresh) for key in keys if key not in set(claimed))
        groups += [_Group(size, location=location, fresh=is_fresh)
                   for (location, is_fresh), size in sorted(free.items())]
        groups.append(_Group(other_n))
        if evidence.extra_slot_keys:
            slots_n, extra_n = evidence.unknown_extra_slots()
            groups += [_Group(slots_n, extra=True), _Group(extra_n, extra=True, extra_deck=True)]
        return [group for group in groups if group.size > 0]

    @staticmethod
    def _open_claims(evidence, claims, extra_pool):
        """The zone claims no card the public places settles: ``[(in hand, in deck, in the extra deck, codes)]``.
        Without extra slots the face-down extra deck is the whole extra pool, which settles a claim or not."""
        known_hand = set(evidence.disclosed_hand)
        known_deck = set(evidence.disclosed_deck) | set(evidence.deck_top)
        known_extra = set(evidence.disclosed_extra_deck)
        if not evidence.extra_slot_keys:
            known_extra |= set(evidence.unknown_extra_pool())
        out = []
        for claim in claims:
            zones, codes = set(claim.zones), frozenset(claim.codes)
            if not zones <= {C.LOCATION_HAND, C.LOCATION_DECK, C.LOCATION_EXTRA}:
                raise ResponseClassError("a zone claim names a zone the class law does not hold")
            if C.LOCATION_HAND in zones and known_hand & codes or C.LOCATION_DECK in zones and known_deck & codes \
                    or C.LOCATION_EXTRA in zones and known_extra & codes:
                continue
            out.append((C.LOCATION_HAND in zones, C.LOCATION_DECK in zones,
                        C.LOCATION_EXTRA in zones and bool(extra_pool), codes))
        return out

    def _eligible(self, code, location):
        return sum(1 << number for number, (zone, codes) in enumerate(self.categories)
                   if zone == location and code in codes)

    def _moves(self, number, state):
        """Every way the ``number``-th code's copies go to the groups from ``state``: ``(copies per group, ways,
        next state)``."""
        code, copies, extra = self.codes[number]
        need, key, settled, matches = state
        members = self.members[number]
        ranges = [range(min(need[index], copies) + 1) if index in members else range(1)
                  for index in range(len(self.groups))]
        for take in itertools.product(*ranges):
            used = sum(take)
            # The extra pool fills its slots exactly; the main pool's rest is the deck.
            if used > copies or extra and used != copies:
                continue
            zones, known_zones = Counter(), Counter()
            for count, group in zip(take, self.groups):
                if count and group.location is not None:
                    zones[group.location] += count
                    if not group.fresh:
                        known_zones[group.location] += count
            # The identities known in a zone sit in its slots that are not fresh.
            if any(known == code and known_zones[location] < least
                   for (location, known), least in self.required.items()):
                continue
            ways = math.factorial(copies) / (math.factorial(copies - used)
                                             * math.prod(math.factorial(count) for count in take))
            in_hand = sum(count for count, group in zip(take, self.groups) if group.hand)
            if self.tilt is not None and not extra:
                ways *= math.exp(self.tilt(code, in_hand))
            in_extra_deck = sum(count for count, group in zip(take, self.groups) if group.extra_deck)
            in_deck = 0 if extra else copies - used
            grown_settled = tuple(done or code in codes and (hand and in_hand > 0 or deck and in_deck > 0
                                                             or extra_zone and in_extra_deck > 0)
                                  for done, (hand, deck, extra_zone, codes) in zip(settled, self.claims))
            grown_matches = matches
            for location, count in zones.items():
                eligible = self._eligible(code, location)
                for _ in range(count if eligible else 0):
                    grown_matches = _extend_matches(grown_matches, eligible)
            hand_part, zone_part = key
            if code in self.responders.hand and in_hand:
                hand_part += (code,)
            if code in self.responders.set and zones[C.LOCATION_SZONE]:
                zone_part += (code,)
            left = tuple(wanted - taken for wanted, taken in zip(need, take))
            yield take, ways, (left, (hand_part, zone_part), grown_settled, grown_matches)

    def class_of(self, state):
        """The class of an end state in the recipe order ``Responders.key`` uses; None for a state the claims
        refuse."""
        need, (hand_part, zone_part), settled, matches = state
        if any(need) or not all(settled) or self.categories and self.full not in matches:
            return None
        return (tuple(sorted(hand_part, key=self.responders.hand.index)),
                tuple(sorted(zone_part, key=self.responders.set.index)))

    def classes(self) -> dict:
        classes = {}
        for state, weight in self.layers[-1].items():
            key = self.class_of(state)
            if key is not None and weight > 0:
                classes[key] = classes.get(key, 0.0) + weight
        total = math.fsum(classes.values())
        if not total > 0:
            raise ResponseClassError("no layout of the unknown pool honors the evidence's claims")
        return {key: weight / total for key, weight in sorted(classes.items(), key=lambda item: -item[1])}

    def particle(self, takes, rng):
        """The ``search.belief.Particle`` of each code's copies per group, the slots within a group and the deck
        shuffled as the belief sampler shuffles them; its face-down cards in ``facedown_sampling_keys`` order."""
        evidence = self.evidence
        placed = [[] for _ in self.groups]
        deck = []
        for (code, copies, extra), take in zip(self.codes, takes):
            for index, count in enumerate(take):
                placed[index] += [code] * count
            if not extra:
                deck += [code] * (copies - sum(take))
        for cards in placed:
            rng.shuffle(cards)
        groups = list(zip(self.groups, placed))
        # The claimed representatives first, as the belief sampler orders the hand.
        hand = [cards[0] for group, cards in groups if group.hand and group.allowed is not None]
        hand += [code for group, cards in groups if group.hand and group.allowed is None for code in cards]
        slots = {group.slot: cards[0] for group, cards in groups if group.slot is not None}
        free = {(group.location, group.fresh): iter(cards) for group, cards in groups
                if group.location is not None and group.slot is None}
        fresh = {tuple(key) for key in evidence.fresh_facedown_keys}
        facedown = tuple(slots[tuple(key)] if tuple(key) in slots else next(free[(int(key[0]), tuple(key) in fresh)])
                         for key in evidence.facedown_sampling_keys)
        other = tuple(code for group, cards in groups if not (group.hand or group.location is not None or group.extra)
                      for code in cards)
        deck += list(evidence.disclosed_deck.elements())
        rng.shuffle(deck)
        extra, extra_deck = (), None
        if evidence.extra_slot_keys:
            extra = tuple(code for group, cards in groups if group.extra and not group.extra_deck for code in cards)
            extra_deck = tuple(sorted([code for group, cards in groups if group.extra_deck for code in cards]
                                      + list(evidence.disclosed_extra_deck.elements())))
        return Particle(hand=tuple(hand), facedown=facedown, other=other, deck=tuple(deck) + tuple(evidence.deck_top),
                        extra=extra, extra_deck=extra_deck)


class _ClassSampler:
    """Exact draws of the law's layouts whose class is one of ``keys``, as ``search.belief.BeliefSampler`` gives
    them to ``search.particles.particles_from_evidence``; their face-down cards come in slot order."""

    keyed_facedown = True

    def __init__(self, law: _Law, keys, rng):
        self.law, self.rng = law, rng
        keys = set(keys)
        self.back = [{state: 1.0 for state in law.layers[-1] if law.class_of(state) in keys}]
        for number in reversed(range(len(law.codes))):
            table = {}
            for state in law.layers[number]:
                total = math.fsum(ways * self.back[0].get(after, 0.0)
                                  for _take, ways, after in law._moves(number, state))
                if total > 0:
                    table[state] = total
            self.back.insert(0, table)
        if law.initial not in self.back[0]:
            raise ResponseClassError("a class of the law has no layout")

    def sample(self):
        law, state, takes = self.law, self.law.initial, []
        for number in range(len(law.codes)):
            moves = [(take, ways * self.back[number + 1].get(after, 0.0), after)
                     for take, ways, after in law._moves(number, state)]
            moves = [move for move in moves if move[1] > 0]
            point = self.rng.random() * math.fsum(weight for _take, weight, _after in moves)
            for take, weight, after in moves:
                point -= weight
                if point < 0:
                    break
            takes.append(take)
            state = after
        return law.particle(takes, self.rng)

    def clear(self):
        pass


def class_probabilities(evidence, responders_, *, tilt=None, claims=()) -> dict:
    """``{class: probability}`` of the evidence's public law, tilted by ``tilt(code, copies drawn into the hand)`` (a
    log weight; None for the public law itself) and conditioned on the zone claims ``claims`` (``ZoneClaim``)."""
    return _Law(evidence, responders_, tilt=tilt, claims=claims).classes()


def _allocate(masses, count):
    """``count`` draws over strata of ``masses`` by largest remainder; the strata left without a draw form one
    residual stratum, which takes a draw of its own when there is any (its mass is theirs together)."""
    total = math.fsum(masses)
    shares = [count * mass / total for mass in masses]
    counts = [int(share) for share in shares]
    for number in sorted(range(len(masses)), key=lambda number: (-(shares[number] - counts[number]), number)):
        if sum(counts) >= count:
            break
        counts[number] += 1
    # A residual stratum needs a draw: the smallest allocated stratum gives one up when every draw is taken.
    if any(value == 0 for value in counts) and sum(counts) >= count:
        smallest = min((number for number, value in enumerate(counts) if value), key=lambda number: masses[number])
        counts[smallest] -= 1
    return counts


#: How a class bank spreads its draws over the strata: ``largest_remainder`` (strata left without a draw pooled into
#: one residual stratum), or ``every_stratum`` (each stratum of positive mass gets a draw first when there are at least
#: as many draws as strata, the rest by largest remainder; with fewer draws, as ``largest_remainder``).
ALLOCATIONS = ("largest_remainder", "every_stratum")


def _allocate_every(masses, count):
    """``count`` draws with a draw for every stratum of positive mass first, then the rest by largest remainder of
    the masses; with fewer draws than such strata, ``_allocate``."""
    live = [number for number, mass in enumerate(masses) if mass > 0]
    if count < len(live):
        return _allocate(masses, count)
    rest, total = count - len(live), math.fsum(masses)
    shares = [rest * mass / total if mass > 0 else 0.0 for mass in masses]
    extra = [int(share) for share in shares]
    for number in sorted(live, key=lambda number: (-(shares[number] - extra[number]), number)):
        if sum(extra) >= rest:
            break
        extra[number] += 1
    return [(1 if mass > 0 else 0) + more for mass, more in zip(masses, extra)]


def response_class_bank(public, recipe, *, viewer, count, seed, extra_origin_slots=(), categories=(), belief=None,
                        allocation="largest_remainder"):
    """A stratified bank of ``count`` draws at a root of the seat's turn (``JointParticleBank`` with ``proposal_law``
    ``RESPONSE_CLASS_LAW``): the strata are the hand-responder classes (which hand traps the opponent holds), each of
    exact probability; the draws go to the strata by largest remainder, those left without one pooled into a residual
    stratum; each stratum's draws are exact draws of the law conditioned on the stratum (every other hidden card, set
    cards included, drawn as the law draws it), weighted by the stratum's mass over its draws. Identical draws are
    merged into one with their weights summed. Nothing is left out: the weights are the strata's exact masses.

    ``belief`` is ``(codes, adjustments, power, head_sha256)``: the belief head's log adjustments at the root, one
    row per code of its recipe (``codes``, the main-deck codes of ``recipe`` in ascending order) for holding 0 to 3
    copies, and the power they are applied with; it tilts the law, masses and draws alike, by the opponent's whole
    hand (the disclosed cards with the drawn ones, as ``belief_log_weights`` reads a layout's hand). None is the
    public law itself. Deterministic in its inputs, so the particle adapter derives it again to admit it."""
    if type(count) is not int or count < 1 or type(seed) is not int or allocation not in ALLOCATIONS:
        raise ValueError("a response class bank needs a positive draw count, an integer seed and a known allocation")
    zone_claims = tuple(item for item in categories if isinstance(item, ZoneClaim))
    evidence = evidence_for_public_snapshot(public, viewer, recipe, extra_origin_slots=extra_origin_slots,
                                           categories=tuple(item for item in categories
                                                            if not isinstance(item, ZoneClaim)))
    main = sorted((code, copies) for code, copies, extra in zip(recipe.codes, recipe.copies, recipe.extra)
                  if not extra)
    found = responders([code for code, _copies in main])
    belief_recipe = BeliefRecipe(tuple(code for code, _ in main), tuple(copies for _, copies in main))
    tilt = None
    if belief is not None:
        codes, adjustments, power, _head = belief
        if tuple(codes) != belief_recipe.codes or len(adjustments) != len(codes):
            raise ValueError("the belief reads another opponent main deck than the bank's recipe")
        row, disclosed = {code: number for number, code in enumerate(belief_recipe.codes)}, evidence.disclosed_hand
        tilt = lambda code, copies: power * float(adjustments[row[code]][disclosed[code] + copies])
    law = _Law(evidence, Responders(found.hand, ()), tilt=tilt, claims=zone_claims)
    strata = list(law.classes().items())
    counts = (_allocate if allocation == "largest_remainder" else _allocate_every)([mass for _key, mass in strata], count)
    groups = [([key], mass, drawn) for (key, mass), drawn in zip(strata, counts) if drawn]
    residual = [key for (key, _mass), drawn in zip(strata, counts) if not drawn]
    if residual:
        groups.append((residual, math.fsum(mass for (_key, mass), drawn in zip(strata, counts) if not drawn), 1))
    draws, weights = {}, {}
    for rank, (keys, mass, drawn) in enumerate(groups):
        stratum_seed = (seed * 1_000_003 + rank * 101) & 0x7FFF_FFFF
        sampler = _ClassSampler(law, keys, random.Random(repr(("response-class", stratum_seed))))
        for layout in proposal_bank(public, evidence, recipe, player=1 - viewer, count=drawn, seed=stratum_seed,
                                    sampler=sampler):
            if not all(claim.honored_by(layout) for claim in zone_claims):
                raise ResponseClassError("a class draw breaks a zone claim its law conditions on")
            draw = JointDraw.from_layout(layout)
            draws.setdefault(draw, draw)
            weights[draw] = weights.get(draw, 0.0) + mass / drawn
    order = sorted(draws, key=lambda draw: -weights[draw])
    total = math.fsum(weights.values())
    probabilities = tuple(weights[draw] / total for draw in order)
    keys = tuple(sorted((location, sequence) for location, sequence in evidence.facedown_slot_keys
                        if location in (C.LOCATION_MZONE, C.LOCATION_SZONE)))
    sizes = (evidence.hand_size, evidence.deck_size, len(keys), evidence.extra_facedown_size)
    record = {"law": RESPONSE_CLASS_LAW, "responders": [list(found.hand), list(found.set)],
              "strata": [[list(key), mass, drawn] for (key, mass), drawn in zip(strata, counts)],
              "draws": len(order),
              **({} if allocation == "largest_remainder" else {"allocation": allocation}),
              "belief": None if belief is None else {"power": belief[2], "head_sha256": belief[3],
                                                     "adjustments": [list(map(float, row)) for row in belief[1]]}}
    return JointParticleBank(viewer, recipe, tuple(order), probabilities,
                             tuple(math.log(value) for value in probabilities), keys, sizes, seed,
                             0.0 if belief is None else float(belief[2]), None if belief is None else belief[3],
                             proposal_law=RESPONSE_CLASS_LAW), record


__all__ = ["ALLOCATIONS", "RESPONSE_CLASS_LAW", "ResponseClassError", "Responders", "class_probabilities",
           "responders", "response_class_bank"]
