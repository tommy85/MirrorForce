"""Belief particle sampler: sample the opponent's hidden cards from public evidence in a closed environment.

The deployment rule for search requires search particles to be **rebuilt from public information + sampled hidden
cards**, never forked from the real duel. This module is the "sampled hidden cards".

What a closed environment buys
--------------

The decklists of the fixed environment are fixed and known (a manifest with
``all_cards_known: true``). So the **multiset** of the opponent's main deck
is not something to guess: it is given. Only one thing is guessed: which cards of that multiset are now
in the hand, which are under face-down cards, and in what order the rest lie in the deck.

Remove the located ones (face-up on the field, graveyard, banished), then the disclosed identities the disclosure
ledger records; what is left is the **unknown pool**. The unknown slots (undisclosed hand cards + set cards / face-down
monsters + deck) are drawn from this pool without replacement: a multivariate hypergeometric distribution, with no learned prior.

This is not an "opponent model"
------------------

This module only answers "where are the cards". "How will the opponent play" is another matter: as in AlphaZero,
the self-play policy network plays that role itself at the search nodes. A two-stage scheme pretraining a response
head on a random corpus is out of this module's scope.

Known conservative choices
--------------

``DisclosureLedger.OWNER_ONLY_LOCATIONS`` treats confirmations **inside the opponent's deck** as visible to the
deck's owner only (``MSG_CONFIRM_CARDS`` in ``libduel.cpp`` carries a playerid the server distributes by).
That rule is right for "what did they search", but it also drops facts **both sides see**, such as a card publicly
returned to the top of the deck. So this module takes deck disclosures and deck-top positions as explicit
parameters (:attr:`Evidence.disclosed_deck` / :attr:`Evidence.deck_top`), empty by default:
using less information only widens the distribution and cannot leak; filling it in is the ledger's job.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from dataclasses import dataclass, field

from .field_assignment import FieldAssignmentError, FieldAssignmentSampler

__all__ = [
    "BeliefError",
    "BeliefOwnershipUnsupported",
    "BeliefSampler",
    "Evidence",
    "Particle",
    "hypergeom_pmf",
    "evidence_from_snapshot",
]


class BeliefError(ValueError):
    """The public evidence contradicts itself; no legal particle can be sampled."""


class BeliefOwnershipUnsupported(BeliefError):
    """A public change of control that the current per-player permutation pools cannot express."""


def hypergeom_pmf(k: int, copies: int, pool: int, draws: int) -> float:
    """Probability of drawing exactly ``k`` copies of a card when drawing ``draws`` cards from ``pool`` without replacement.

    ``copies`` is the number of copies of the card in the pool. Computed exactly with :func:`math.comb`, without
    scipy: checking the sampling distribution needs the analytic truth, not another approximation.
    """
    if k < 0 or k > min(copies, draws):
        return 0.0
    if draws < 0 or draws > pool:
        return 0.0
    return (math.comb(copies, k) * math.comb(pool - copies, draws - k)
            / math.comb(pool, draws))


# -- evidence ----------------------------------------------------------------


@dataclass(frozen=True)
class FreshSlots:
    """A public fact passed with the category constraints to :func:`evidence_from_snapshot`: the field slots ``keys``
    (``(zone, sequence)``) are "fresh" slots in the observer's ledger (:meth:`DisclosureLedger.fresh_slots`);
    the cards they hold are none of the zone's cards known by identity but not by position."""

    keys: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict:
        return {"fresh_slots": [list(key) for key in self.keys]}


def ledger_claims(ledger, viewer: int, player: int) -> tuple:
    """Every public constraint the observer's ledger gives the sampler: :meth:`DisclosureLedger.category_constraints`
    and, when there are fresh slots, :class:`FreshSlots`. Every caller taking evidence from the ledger goes through here, so sampler and ledger keep the same calibration."""

    claims = tuple(ledger.category_constraints(viewer, player))
    fresh = ledger.fresh_slots(viewer, player)
    return claims + ((FreshSlots(fresh),) if fresh else ())


@dataclass(frozen=True)
class Evidence:
    """One player's hidden zones as they look under what the observer may legally see.

    Every field must be derivable **from the observer's view**. Omniscient quantities (the opponent's real hand) never
    go in here; this is where the deployment rule lands at the type level.
    """

    #: multiset of the player's main-deck list (closed environment: given by the manifest)
    deck_list: Counter
    #: the player's located cards: face-up on the field, graveyard, banished, and known Extra Deck destinations.
    #: Only **cards that appear in the main deck**; the Extra Deck takes no part in assigning hand / deck.
    located: Counter = field(default_factory=Counter)
    #: hand cards whose identity is disclosed (disclosure ledger ``(controller, LOCATION_HAND, code)``)
    disclosed_hand: Counter = field(default_factory=Counter)
    #: deck cards whose identity is disclosed. Empty by default; see the module notes.
    disclosed_deck: Counter = field(default_factory=Counter)
    #: hand size (public)
    hand_size: int = 0
    #: cards left in the main deck (public)
    deck_size: int = 0
    #: cards of unknown identity whose **slot can be located**: set spells/traps, face-down monsters, face-down banished.
    #: They all come from the main deck and must take part in the assignment, or the unknown pool has that many extra cards.
    #: "Locatable" is the boundary with :attr:`other_hidden_slots`: these kinds
    #: ``Debug.PermuteHidden`` can move, that kind it cannot.
    facedown_slots: int = 0
    #: face-down field cards whose identity is disclosed (a set card that was revealed, say)
    disclosed_facedown: Counter = field(default_factory=Counter)
    #: other cards that "take a main-deck slot, have no visible identity and cannot be located either".
    #: Only pendulum monsters in the Extra Deck zone remain of that kind, and once ``public_extra`` is given
    #: they count as located too. Not counting them would leave the unknown pool larger than the unknown slots by that many,
    #: and :meth:`check` fails.
    other_hidden_slots: int = 0
    #: the disclosed part of the slots above
    disclosed_other: Counter = field(default_factory=Counter)
    #: cards known to be pinned in order on top of the deck; the last element is the top card. Empty by default.
    deck_top: tuple[int, ...] = ()
    #: which hidden slots can be permuted, ``[(zone, sequence)]``. Its length is
    #: ``facedown_slots + extra_in_slots``: the former counts slots holding main-deck cards,
    #: the latter slots holding Extra Deck monsters. Sampling and permutation must use the same list, or the slot
    #: counts match while the positions do not.
    facedown_slot_keys: tuple[tuple[int, int], ...] = ()
    #: how many of those slots hold **Extra Deck** monsters (face-down banished extra monsters, Xyz monsters flipped
    #: face-down). They take no main-deck place, so they are not in the unknown pool; they are sampled from
    #: :meth:`unknown_extra_pool`.
    extra_in_slots: int = 0
    #: the slots of :attr:`facedown_slot_keys` holding extra monsters (**extra-class slots**). The source is
    #: public: for a face-down banishment the source location word of ``MSG_MOVE`` says it came face-down from the Extra Deck
    #: (:mod:`mirrorforce.search.banish_origin`), or the slot's identity is disclosed and its code is in the
    #: Extra Deck list. When it is shorter than :attr:`extra_in_slots`, the difference is extra monsters that
    #: "conservation says exist but nobody can place"; such a decision point is unrealizable (the sampler does not guess).
    extra_slot_keys: tuple[tuple[int, int], ...] = ()
    #: multiset of the player's Extra Deck list (closed environment). Extra-class slots are sampled from it minus the
    #: located extra monsters; without it extra-class slots cannot be sampled.
    extra_list: Counter = field(default_factory=Counter)
    #: located extra monsters: face-up on the field, graveyard, face-up banished, Xyz materials, face-up in the Extra Deck zone.
    located_extra: Counter = field(default_factory=Counter)
    #: the disclosed part of the extra-class slots (an Xyz monster flipped face-down, say)
    disclosed_extra_slots: Counter = field(default_factory=Counter)
    #: the cards of the **face-down** pile of the Extra Deck zone that the disclosure ledger named. They still take
    #: places of :attr:`extra_facedown_size` (the engine still has them face-down) but need no sampling:
    #: they are taken out of the extra pool and merged back into the particle's ``extra_deck`` after sampling. The order
    #: is unobservable, so "named" goes down to the multiset only, not to a position.
    disclosed_extra_deck: Counter = field(default_factory=Counter)
    #: number of face-down cards in the Extra Deck zone (public: the opponent's Extra Deck reports only a count; face-up ones are counted separately)
    extra_facedown_size: int = 0
    #: **category constraints (hand)**: each item says "one of the undisclosed hand cards is in this set",
    #: and items take different cards (no card serves two). The source is the disclosure ledger
    #: :meth:`~mirrorforce.netduel.disclosure.DisclosureLedger.category_constraints`
    #: (a card searched from the deck to the hand by an effect, identity undisclosed, while the activating card is
    #: public and its text limits the admissible set). The hand is a multiset, so there are no slots here.
    hand_categories: tuple[frozenset[int], ...] = ()
    #: **category constraints (set-card slots)**: ``((zone, sequence), admissible set)``. A slot set from the deck by an
    #: effect must hold an identity in the set. Unlike the hand column, these carry slots, because positions of
    #: set cards are meaningful.
    facedown_categories: tuple[tuple[tuple[int, int], frozenset[int]], ...] = ()
    #: diagnostics: foreign cards this player controls, including opponent cards whose name is also in this player's list.
    #: Told apart by public ownership; tokens and other cards outside the list count too. They take no place of this player's list.
    foreign_controlled: int = 0
    #: **shuffled set-card groups**: ``(zone, (sequence, ...), (code, ...))``. After set cards are shuffled the observer
    #: knows these slots hold **exactly** this multiset, only not which is where. These cards are already in
    #: :attr:`disclosed_facedown` (not in the unknown pool), and particles arrange them randomly over these slots.
    shuffled_facedown: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...] = ()
    #: lower bounds of unpositioned identities outside complete shuffle groups, with codes repeated per zone. These
    #: identities stay in the unassigned pool; joint conditional sampling keeps them in their zones without inventing slots.
    unpositioned_facedown: tuple[tuple[int, int], ...] = ()
    #: field category constraints without slot anchors; each needs a different card as witness.
    unpositioned_facedown_categories: tuple[tuple[int, frozenset[int]], ...] = ()
    #: set-card slots the main-deck pool actually has to fill, excluding located identities, shuffle groups and extra-class slots.
    facedown_sampling_keys: tuple[tuple[int, int], ...] = ()
    #: field identities of located / complete shuffle groups; they can satisfy unpositioned category claims but do not offset the extra identity lower bounds above.
    fixed_field_identities: tuple[tuple[int, int], ...] = ()
    #: field knowledge that cannot yet be located in the main / Extra Deck permutation pools; when non-zero, skip.
    unexpressed_field_knowledge: int = 0
    #: the "fresh" sampled slots (:meth:`DisclosureLedger.fresh_slots`): a card of unknown identity arrived and was
    #: never disclosed since. The identities of :attr:`unpositioned_facedown` are not in these slots.
    fresh_facedown_keys: tuple[tuple[int, int], ...] = ()

    def unknown_pool(self) -> Counter:
        """Unknown pool: the decklist minus every card whose whereabouts are settled."""
        pool = Counter(self.deck_list)
        for other in (self.located, self.disclosed_hand, self.disclosed_deck,
                      self.disclosed_facedown, self.disclosed_other):
            pool.subtract(other)
        for code in self.deck_top:
            pool[code] -= 1
        negative = {code: count for code, count in pool.items() if count < 0}
        if negative:
            raise BeliefError(
                f"public evidence exceeds the decklist: {negative} (copies of a card booked twice, "
                "or the decklist does not match the duel)")
        return +pool  # drop zeros and negatives

    def unknown_slots(self) -> tuple[int, int, int, int]:
        """``(unknown hand slots, unknown set-card slots, other unknown slots, unknown deck slots)``."""
        hand = self.hand_size - sum(self.disclosed_hand.values())
        facedown = self.facedown_slots - sum(self.disclosed_facedown.values())
        other = self.other_hidden_slots - sum(self.disclosed_other.values())
        deck = (self.deck_size - sum(self.disclosed_deck.values())
                - len(self.deck_top))
        if hand < 0 or facedown < 0 or other < 0 or deck < 0:
            raise BeliefError(
                f"more disclosed cards than the zone holds: hand={hand} "
                f"facedown={facedown} other={other} deck={deck}")
        return hand, facedown, other, deck

    # -- extra pool ----------------------------------------------------------

    def unattributed_extra_slots(self) -> int:
        """Face-down banished slots that conservation says hold extra monsters but that nobody can name.

        Non-zero means unrealizable: the sampler does not guess slot ownership; see
        :func:`mirrorforce.search.particles.particles_from_evidence`.
        """

        return int(self.extra_in_slots) - len(self.extra_slot_keys)

    def unknown_extra_pool(self) -> Counter:
        """Extra pool: the Extra Deck list minus the located extra monsters and the disclosed extra-class slots."""

        pool = Counter(self.extra_list)
        for other in (self.located_extra, self.disclosed_extra_slots,
                      self.disclosed_extra_deck):
            pool.subtract(other)
        negative = {code: count for code, count in pool.items() if count < 0}
        if negative:
            raise BeliefError(
                f"public evidence exceeds the Extra Deck list: {negative} (copies of an extra monster booked twice, "
                "or the Extra Deck list does not match the duel)")
        return +pool

    def unknown_extra_slots(self) -> tuple[int, int]:
        """``(undisclosed cards in extra-class slots, face-down Extra Deck cards still to sample)``.

        Both already exclude the cards the ledger named; those are pinned back by
        :attr:`disclosed_extra_slots` and :attr:`disclosed_extra_deck`.
        """

        slots = len(self.extra_slot_keys) - sum(self.disclosed_extra_slots.values())
        facedown = (int(self.extra_facedown_size)
                    - sum(self.disclosed_extra_deck.values()))
        if slots < 0 or facedown < 0:
            raise BeliefError(
                f"more disclosed extra-class slots than slots: slots={slots} "
                f"extra_facedown={facedown}")
        return slots, facedown

    def all_categories(self) -> tuple[frozenset[int], ...]:
        """The admissible sets of the hand and set-card category constraints together."""

        return tuple(self.hand_categories) + tuple(
            codes for _key, codes in self.facedown_categories)

    def check(self) -> None:
        """Evidence consistency: the unknown pool must be exactly as large as the unknown slots.

        The extra pool is checked only when there are extra-class slots: a decision point without them does not use it,
        and the opponent's Extra Deck bookkeeping (an extra monster whose control was taken, say) should not make a
        decision point that does not use the extra pool inapplicable.
        """
        pool = self.unknown_pool()
        need = sum(self.unknown_slots())
        have = sum(pool.values())
        if need != have:
            raise BeliefError(
                f"{need} unknown slots but {have} cards in the unknown pool; they do not match. "
                "Most likely located missed a zone, or the decklist is not this one.")
        self._check_categories(pool)
        if self.unattributed_extra_slots() < 0:
            raise BeliefError(
                f"{len(self.extra_slot_keys)} extra-class slots, more than the "
                f"{self.extra_in_slots} extra monsters conservation gives")
        if self.extra_slot_keys:
            self._check_extra_pool()

    def _check_extra_pool(self) -> None:
        if not self.extra_list:
            raise BeliefError(
                "extra-class slots without an Extra Deck list (extra_list); cannot sample")
        if any(key not in self.facedown_slot_keys for key in self.extra_slot_keys):
            raise BeliefError("extra-class slots are not in the list of permutable slots")
        need = sum(self.unknown_extra_slots())
        have = sum(self.unknown_extra_pool().values())
        if need != have:
            raise BeliefError(
                f"{need} unknown extra-class slots but {have} cards in the extra pool; they do not match. "
                "Most likely located_extra missed a zone, or the Extra Deck list is not this one.")

    def _check_categories(self, pool: Counter) -> None:
        """The category constraints must be satisfiable together: each item takes a different card.

        The test is Hall's theorem (the necessary and sufficient condition for a system of distinct representatives):
        for any set of items, the cards of the union of their admissible sets in the unknown pool must be at least as many
        as the items. There are single-digit items, so subsets are enumerated. Which slot a constraint sits in does not affect satisfiability, so both columns are tested together.
        """

        if self.unpositioned_facedown or self.unpositioned_facedown_categories:
            # Slot and position-free facts can describe the same physical
            # card. Counting each as an extra draw would reject valid roots.
            sampler = _field_sampler(self, pool)
            sampler.clear()
            return
        constraints = self.all_categories()
        if not constraints:
            return
        hand_unknown, facedown_unknown, _other, _deck = self.unknown_slots()
        if len(self.hand_categories) > hand_unknown:
            raise BeliefError(
                f"{len(self.hand_categories)} hand category constraints but only "
                f"{hand_unknown} undisclosed hand cards; they do not fit")
        if len(self.facedown_categories) > facedown_unknown:
            raise BeliefError(
                f"{len(self.facedown_categories)} set-card category constraints but only "
                f"{facedown_unknown} undisclosed set-card slots; they do not fit")
        for index, codes in enumerate(constraints):
            if not codes:
                raise BeliefError(f"the admissible set of category constraint {index} is empty")
        for mask in range(1, 1 << len(constraints)):
            union: set[int] = set()
            size = 0
            for index, codes in enumerate(constraints):
                if mask >> index & 1:
                    size += 1
                    union |= codes
            available = sum(pool.get(code, 0) for code in union)
            if available < size:
                raise BeliefError(
                    f"{size} category constraints can take only {available} cards of the unknown pool; "
                    "no system of distinct representatives exists: the ledger's category limits do not match the decklist")


def _field_sampler(evidence: Evidence, pool: Counter) -> FieldAssignmentSampler:
    """Build the shared hand/field conditional sampler from public evidence."""
    from ..netduel import constants as C

    hand_n, field_n, _other_n, _deck_n = evidence.unknown_slots()
    if len(evidence.hand_categories) > hand_n:
        raise BeliefError("hand category claims exceed the unknown hand capacity")
    keys = evidence.facedown_sampling_keys
    if len(keys) != field_n or len(set(keys)) != len(keys):
        raise BeliefError("position-free field knowledge needs every open main-deck slot")
    allowed = {}
    for key, codes in evidence.facedown_categories:
        if key not in keys:
            raise BeliefError("field category claim has no open sampling slot")
        allowed[key] = frozenset(codes) & allowed.get(key, frozenset(codes))
    zones = {location for location, _code in evidence.unpositioned_facedown}
    zones.update(location for location, _codes in evidence.unpositioned_facedown_categories)
    # Unconstrained slots outside these zones stay in the ordinary residual
    # draw; face-down banished piles need not enlarge the counting problem.
    selected = tuple(key for key in keys if key[0] in zones or key in allowed)
    slots = tuple((C.LOCATION_HAND, index, codes)
                  for index, codes in enumerate(evidence.hand_categories))
    slots += tuple((location, sequence, allowed.get((location, sequence)))
                   for location, sequence in selected)
    fresh = set(evidence.fresh_facedown_keys)
    try:
        return FieldAssignmentSampler(
            pool, slots, identities=evidence.unpositioned_facedown,
            categories=evidence.unpositioned_facedown_categories,
            fixed=evidence.fixed_field_identities,
            fresh=tuple(len(evidence.hand_categories) + number for number, key in enumerate(selected)
                        if key in fresh),
        )
    except FieldAssignmentError as exc:
        raise BeliefError(str(exc)) from exc


# -- particles ----------------------------------------------------------------


@dataclass(frozen=True)
class Particle:
    """One sample of the hidden cards."""

    #: the undisclosed hand cards in sampling order; together with the disclosed ones they make the whole hand
    hand: tuple[int, ...] = ()
    #: the face-down field cards in slot order
    facedown: tuple[int, ...] = ()
    #: the face-down banished cards / pendulum monsters in the Extra Deck zone, in slot order
    other: tuple[int, ...] = ()
    #: the main deck, **index 0 is the bottom and the last element the top**: the same direction as ``field::list_main``
    #: (``field.cpp`` takes the top with ``list_main.back()``), so it can be fed in ascending index order
    #: to ``Debug.AddCard(..., LOCATION_DECK, SEQ_DECKTOP, ...)``.
    deck: tuple[int, ...] = ()
    #: extra monsters in the extra-class slots (the undisclosed slots of :attr:`Evidence.extra_slot_keys`),
    #: in slot order; sampled from the extra pool
    extra: tuple[int, ...] = ()
    #: the target order of the face-down Extra Deck cards (index 0 = ``list_extra[0]``): the extra pool minus
    #: :attr:`extra`. The order is unobservable and does not affect the rules, so it is given in ascending code order
    #: without spending random numbers; ``None`` means this particle does not touch the Extra Deck.
    extra_deck: tuple[int, ...] | None = None

    def full_hand(self, disclosed_hand: Counter) -> Counter:
        out = Counter(self.hand)
        out.update(disclosed_hand)
        return out


# -- sampler ------------------------------------------------------------------


def _allocations(copies: int, members: tuple[int, ...], need: tuple[int, ...]):
    """Every way ``(count per group, number of ways)`` to split a code with ``copies`` copies over the groups: only to
    the groups it belongs to (``members``), each at most what it still needs, in total at most the copies; the number of ways is ``n! / ((n − Σa)! ∏ a_t!)``."""
    ranges = [range(min(need[number], copies) + 1) if number in members else range(1)
              for number in range(len(need))]
    for take in itertools.product(*ranges):
        used = sum(take)
        if used <= copies:
            yield take, math.factorial(copies) // (
                math.factorial(copies - used) * math.prod(math.factorial(taken) for taken in take))


def representative_counts(groups, sizes, pool: Counter):
    """Table of the number of ways for category representatives: ``(codes, groups of each code, table)``, where ``table[i][needed]`` is
    the number of ways to fill the groups' remaining needs with the codes from the i-th on (unordered within a group). ``groups`` are the
    groups' admissible sets, ``sizes`` their item counts."""
    codes = sorted(set().union(*groups))
    members = [tuple(number for number, group in enumerate(groups) if code in group) for code in codes]
    needs = list(itertools.product(*(range(size + 1) for size in sizes)))
    counts: list[dict] = [{} for _ in range(len(codes) + 1)]
    counts[-1][(0,) * len(sizes)] = 1
    for index in range(len(codes) - 1, -1, -1):
        later = counts[index + 1]
        for need in needs:
            total = sum(ways * later.get(tuple(wanted - taken for wanted, taken in zip(need, take)), 0)
                        for take, ways in _allocations(pool[codes[index]], members[index], need))
            if total:
                counts[index][need] = total
    return codes, members, counts


class BeliefSampler:
    """Sample hidden-card configurations consistent with the public evidence from an :class:`Evidence`.

    Distribution: treat the unknown pool as a well-shuffled deck and fill, without replacement, the unknown hand slots,
    then the unknown set-card slots, then the unknown deck slots. That is exactly multivariate hypergeometric: the marginal
    probability of each card in each slot is ``copies in the pool / pool size``, and the joint distribution over any
    group of slots is analytic (:func:`hypergeom_pmf`), so the sampler can be checked directly, not on faith.

    The cards pinned on top of the deck (:attr:`Evidence.deck_top`) go on top of the sampled deck
    and are not shuffled.
    """

    def __init__(self, evidence: Evidence, rng: random.Random | None = None,
                 *, check: bool = True):
        self.evidence = evidence
        self.rng = rng if rng is not None else random.Random()
        if check:
            evidence.check()
        self._pool_list = self._expand(evidence.unknown_pool())
        # The extra pool is built only when there are extra-class slots: a decision point without them need not even give the Extra Deck list
        self._extra_pool_list = (
            self._expand(evidence.unknown_extra_pool())
            if evidence.extra_slot_keys else [])
        # The extra pool uses its own **independent** random stream. For the same root, with or without the table of
        # face-down banishment sources, the hands and decks sampled on the main-deck side must be byte-identical;
        # otherwise "adding the extra pool" would change every main-deck sample and audits before and after would
        # disagree. One shared stream cannot do that: once the first ``_sample_extra`` shuffles the extra pool, every later
        # main-pool shuffle starts elsewhere. The state is derived from the main rng (``getstate`` consumes no random numbers),
        # so it is still determined by the caller's seed only; without extra-class slots it is never built and the old calibration is byte-identical.
        self._extra_rng = (
            random.Random(("belief-extra", self.rng.getstate()).__repr__())
            if evidence.extra_slot_keys else None)
        self._field_sampler = (
            _field_sampler(evidence, evidence.unknown_pool())
            if evidence.unpositioned_facedown or evidence.unpositioned_facedown_categories else None)
        # whether the sampled set cards are already arranged slot by slot by ``facedown_sampling_keys`` (the joint conditional sampler does that);
        # otherwise they are a multiset that ``particles_from_evidence`` matches to slots by the category constraints.
        self.keyed_facedown = self._field_sampler is not None
        # Tables of the number of ways for category representatives, keyed by (admissible sets, item counts, copies of these codes in the pool):
        # every draw of one sampler uses the same table, so it is computed once.
        self._representative_tables: dict = {}

    @staticmethod
    def _expand(pool: Counter) -> list[int]:
        """Expand a multiset into an ordered list with copies adjacent. The order is fixed; sampling relies on the rng only."""
        out: list[int] = []
        for code in sorted(pool):
            out.extend([code] * pool[code])
        return out

    # -- sampling --------------------------------------------------------------

    def _draw_category_representatives(self, cards: list[int]
                                       ) -> tuple[list[int], list[int]]:
        """Draw one distinct representative for each category constraint first; return ``(representatives, remaining cards)``.

        Exact calibration: arrange the whole pool randomly over all slots, then condition on "the slot of item j holds a
        card in A_j". The conditional probability of a representative tuple ``(c_1..c_m)`` is proportional to the
        number of instance tuples that produce it: ``∏_j (copies of c_j in the pool − copies of c_j used before)``.

        Items with the same admissible set form a group (most set spells/traps share one set), and slots within a group
        are exchangeable. So it only remains to decide how many copies of each code go to each group: for a code c with n
        copies, giving a_t to the groups can be done in ``n! / ((n − Σa)! ∏ a_t!)`` ways; codes are drawn one by one by "ways of this split ×
        ways for the later codes to fill the remaining needs" (:func:`representative_counts`), and each group's cards are
        arranged randomly over its slots. Same distribution as enumerating weighted tuples, without enumerating tuples and without a limit.
        """

        constraints = self.evidence.all_categories()
        if not constraints:
            return [], cards
        pool = Counter(cards)
        options = [tuple(sorted(code for code in codes if pool.get(code, 0))) for codes in constraints]
        for index, choices in enumerate(options):
            if not choices:
                raise BeliefError(
                    f"category constraint {index} has no admissible card in the unknown pool")
        groups = sorted(set(options))
        group_of = [groups.index(choices) for choices in options]
        sizes = tuple(group_of.count(number) for number in range(len(groups)))
        key = (tuple(groups), sizes, tuple((code, pool[code]) for code in sorted(set().union(*groups))))
        table = self._representative_tables.get(key)
        if table is None:
            table = self._representative_tables[key] = representative_counts(groups, sizes, pool)
        codes, members, counts = table
        need = sizes
        if not counts[0].get(need):
            raise BeliefError(
                "the category constraints have no system of distinct representatives: the unknown pool cannot give each its own card")
        chosen: list[list[int]] = [[] for _ in groups]
        for index, code in enumerate(codes):
            if not any(need):
                break
            # Integer threshold: the first split whose cumulative number of ways exceeds it; splits with 0 ways are never drawn.
            threshold = self.rng.randrange(counts[index][need])
            running = 0
            for take, ways in _allocations(pool[code], members[index], need):
                left = tuple(wanted - taken for wanted, taken in zip(need, take))
                running += ways * counts[index + 1].get(left, 0)
                if running > threshold:
                    break
            for number, taken in enumerate(take):
                chosen[number].extend([code] * taken)
            need = left
        assert not any(need), "every group's representatives are drawn"
        for picked_group in chosen:
            self.rng.shuffle(picked_group)
        picked = [chosen[number].pop() for number in group_of]
        rest = list(cards)
        for code in picked:
            rest.remove(code)
        return picked, rest

    def sample(self) -> Particle:
        if self._field_sampler is not None:
            return self._sample_field()
        hand_n, facedown_n, other_n, deck_n = self.evidence.unknown_slots()
        cards = list(self._pool_list)
        self.rng.shuffle(cards)
        hand_categories = len(self.evidence.hand_categories)
        facedown_categories = len(self.evidence.facedown_categories)
        if hand_categories or facedown_categories:
            # The cards of the category constraints are fixed first, then the rest is shuffled into the other slots. Representatives are
            # ordered "hand first, then set cards", the same as :meth:`Evidence.all_categories`.
            representatives, rest = self._draw_category_representatives(cards)
            self.rng.shuffle(rest)
            hand_fixed = representatives[:hand_categories]
            facedown_fixed = representatives[hand_categories:]
            free_hand = hand_n - hand_categories
            free_facedown = facedown_n - facedown_categories
            cards = (hand_fixed + rest[:free_hand]
                     + facedown_fixed + rest[free_hand:free_hand + free_facedown]
                     + rest[free_hand + free_facedown:])
        hand = tuple(cards[:hand_n])
        facedown = tuple(cards[hand_n:hand_n + facedown_n])
        other = tuple(cards[hand_n + facedown_n:hand_n + facedown_n + other_n])
        rest = cards[hand_n + facedown_n + other_n:]
        if len(rest) != deck_n:
            raise BeliefError(
                f"{deck_n} unknown deck slots but {len(rest)} cards left in the pool")
        return self._finish_sample(hand, facedown, other, rest)

    def _finish_sample(self, hand, facedown, other, rest) -> Particle:
        """Restore deck disclosures and sample the independent extra pool."""
        # Disclosed cards still in the deck: position unknown, shuffled in with the unknown part
        for code, count in self.evidence.disclosed_deck.items():
            rest.extend([code] * count)
        self.rng.shuffle(rest)
        deck = tuple(rest) + tuple(self.evidence.deck_top)
        extra, extra_deck = self._sample_extra()
        return Particle(hand=hand, facedown=facedown, other=other, deck=deck,
                        extra=extra, extra_deck=extra_deck)

    def _sample_field(self) -> Particle:
        """Draw the constrained slots jointly, then uniformly fill the rest."""
        hand_n, _field_n, other_n, deck_n = self.evidence.unknown_slots()
        assigned, remaining = self._field_sampler.sample(self.rng)
        hand_claims = len(self.evidence.hand_categories)
        placed = {(location, sequence): code
                  for (location, sequence, _allowed), code in
                  zip(self._field_sampler.slots[hand_claims:], assigned[hand_claims:])}
        rest = self._expand(remaining)
        self.rng.shuffle(rest)
        free_hand = hand_n - hand_claims
        hand = tuple(assigned[:hand_claims]) + tuple(rest[:free_hand])
        offset = free_hand
        facedown = []
        for key in self.evidence.facedown_sampling_keys:
            if key in placed:
                facedown.append(placed[key])
            else:
                facedown.append(rest[offset])
                offset += 1
        other = tuple(rest[offset:offset + other_n])
        rest = rest[offset + other_n:]
        if len(rest) != deck_n:
            raise BeliefError("conditional field assignment changed the residual deck size")
        return self._finish_sample(hand, tuple(facedown), other, rest)

    def clear(self) -> None:
        """Release per-root field assignment counts after building particles."""
        if self._field_sampler is not None:
            self._field_sampler.clear()

    def _sample_extra(self) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        """Extra-class slots are drawn from the extra pool without replacement; the rest returns to the Extra Deck zone in ascending code order.

        The extra pool uses its own random stream (``_extra_rng``), independent of the main-deck pool's sampling sequence:
        adding extra-class slots never changes the sampled hands / decks, for every single draw.
        """

        if not self.evidence.extra_slot_keys:
            return (), None
        slots_n, facedown_n = self.evidence.unknown_extra_slots()
        cards = list(self._extra_pool_list)
        self._extra_rng.shuffle(cards)
        if len(cards) != slots_n + facedown_n:
            raise BeliefError(
                f"{len(cards)} cards in the extra pool, but {slots_n} unknown extra-class slots plus "
                f"{facedown_n} in the Extra Deck zone; they do not match")
        # The face-down Extra Deck cards the ledger named are not sampled but are still in the face-down pile:
        # merge them back, or the particle's Extra Deck is shorter than the engine's and the permutation is refused
        extra_deck = sorted(cards[slots_n:]
                            + list(self.evidence.disclosed_extra_deck.elements()))
        return tuple(cards[:slots_n]), tuple(extra_deck)

    def sample_many(self, n: int) -> list[Particle]:
        return [self.sample() for _ in range(n)]

    # -- analytic reference --------------------------------------------------------

    def pool_size(self) -> int:
        return len(self._pool_list)

    def copies(self, code: int) -> int:
        return self._pool_list.count(code)

    def hand_pmf(self, code: int) -> dict[int, float]:
        """Analytic: probability that the unknown hand slots hold exactly k copies of ``code``."""
        hand_n = self.evidence.unknown_slots()[0]
        copies = self.copies(code)
        return {k: hypergeom_pmf(k, copies, self.pool_size(), hand_n)
                for k in range(min(copies, hand_n) + 1)}

    def expected_in_hand(self, code: int) -> float:
        hand_n = self.evidence.unknown_slots()[0]
        pool = self.pool_size()
        return hand_n * self.copies(code) / pool if pool else 0.0

    def probability_holds(self, code: int, at_least: int = 1) -> float:
        """Probability that the opponent holds at least ``at_least`` copies of ``code`` (counting disclosed ones)."""
        already = self.evidence.disclosed_hand.get(code, 0)
        need = at_least - already
        if need <= 0:
            return 1.0
        pmf = self.hand_pmf(code)
        return sum(p for k, p in pmf.items() if k >= need)


# -- evidence from a board snapshot ------------------------------------------------


@dataclass
class _Tally:
    """Ledger of :func:`evidence_from_snapshot` while it scans the board; used only inside that function."""

    player: int
    deck_list: Counter
    extra_codes: frozenset
    located: Counter = field(default_factory=Counter)
    located_extra: Counter = field(default_factory=Counter)
    disclosed_hand: Counter = field(default_factory=Counter)
    disclosed_deck: Counter = field(default_factory=Counter)
    disclosed_facedown: Counter = field(default_factory=Counter)
    disclosed_extra_slots: Counter = field(default_factory=Counter)
    #: **face-down** Extra Deck cards whose identity the ledger named
    disclosed_extra_deck: Counter = field(default_factory=Counter)
    hand_size: int = 0
    deck_size: int = 0
    extra_rows: int = 0
    extra_faceup: int = 0
    facedown_slots: int = 0
    foreign: int = 0
    facedown_keys: list = field(default_factory=list)
    extra_keys: list = field(default_factory=list)
    #: face-down banished slots of invisible identity ``(zone, sequence, disclosed code or 0)``
    removed_candidates: list = field(default_factory=list)
    #: identities of set cards "known which, unknown where": ``zone -> code -> count``
    unpositioned_field: dict = field(default_factory=dict)

    def credit(self, code: int) -> None:
        """A card with a public identity and a fixed position: main-deck cards go to ``located``, extra monsters to
        ``located_extra``; one in neither list is a token or an opponent's card."""

        if not code:
            return
        if code in self.extra_codes:
            self.located_extra[code] += 1
        elif self.deck_list.get(code):
            self.located[code] += 1
        else:
            # a token or an opponent's card whose control was taken: takes no slot of their main deck
            self.foreign += 1

    def observe(self, card, revealed_keys: dict) -> None:
        from ..netduel import constants as C

        owner = getattr(card, "owner", -1)
        if owner in (0, 1) and owner != card.controller:
            self._observe_control_change(card, owner)
            return
        if card.controller != self.player:
            return
        location = card.location
        known = revealed_keys.get((card.controller, location, card.sequence),
                                  card.code)
        for material in getattr(card, "overlay", ()) or ():
            self.credit(material)  # Xyz material: the code under a face-up host is public
        if location == C.LOCATION_HAND:
            self.hand_size += 1
            if known:
                self.disclosed_hand[known] += 1
        elif location == C.LOCATION_DECK:
            self.deck_size += 1
            if known:
                self.disclosed_deck[known] += 1
        elif location == C.LOCATION_EXTRA:
            self._observe_extra(card, known)
        elif location in (C.LOCATION_MZONE, C.LOCATION_SZONE):
            self._observe_field(card, known)
        elif location in (C.LOCATION_GRAVE, C.LOCATION_REMOVED):
            if card.code:
                self.credit(card.code)
            elif location == C.LOCATION_REMOVED:
                # face-down banished: identity invisible, slot locatable; whether it is a main-deck card or an extra
                # monster is decided by its public source; see :func:`_classify_removed`
                self.removed_candidates.append((location, card.sequence, known))
            # anonymous cards in the graveyard (should not exist under normal rules) are left to conservation

    def _observe_control_change(self, card, owner: int) -> None:
        """Lists are reduced by ownership; permutable slots are still located by the current controller."""
        from ..netduel import constants as C

        if (card.location not in (C.LOCATION_MZONE, C.LOCATION_SZONE)
                or not (card.position & C.POS_FACEUP) or not card.code
                or getattr(card, "hidden", False)):
            # A foreign face-down card belongs to the controller's physical
            # permutation pool, not their fixed recipe. Do not invent that
            # cross-owner pool or consult the true hidden assignment.
            raise BeliefOwnershipUnsupported(
                "control-changed card is not a known face-up field card")
        if getattr(card, "overlay", ()):
            # CardState exposes material codes, not each material's owner.
            raise BeliefOwnershipUnsupported(
                "control-changed host lacks per-material public ownership")
        if owner == self.player:
            self.credit(card.code)
        elif card.controller == self.player:
            self.foreign += 1

    def _observe_extra(self, card, known: int) -> None:
        """Extra Deck zone: only a count is reported; the face-up cards have public identities.

        Face-up **main-deck** cards (pendulum monsters) are booked by the caller through ``public_extra``, not here;
        face-up extra monsters count as located directly; everything else is face-down.

        Face-up or not is decided by the **position**, not by whether the identity is known.
        :func:`~mirrorforce.worldmodel.state.mask_for` shows the identity of every slot the disclosure ledger names,
        whatever the position, so a face-down Extra Deck card whose identity the ledger knows
        also has a non-zero ``known``. Judging face-up by ``known`` would deduct it from both the face-down count and the
        extra pool, the sampled Extra Deck would be shorter than the engine's and the permutation always refused (measured
        on the multi-deck corpus: 5 roots, 20 ``zone_multiset_conservation`` failures). It is
        :attr:`Evidence.disclosed_extra_deck`: identity disclosed, but still in the face-down pile.

        The test is ``position & POS_FACEUP``, not ``& POS_FACEDOWN``: the mask zeroes the position of rows whose
        identity is invisible (``state._canonical_hidden_row``), so neither bit is set
        and only the face-up side can be judged reliably.
        """
        from ..netduel import constants as C

        self.extra_rows += 1
        if not (int(getattr(card, "position", 0)) & C.POS_FACEUP):
            if known and known in self.extra_codes:
                self.disclosed_extra_deck[known] += 1
            return
        self.extra_faceup += 1
        if known and known in self.extra_codes:
            self.located_extra[known] += 1

    def observe_unpositioned(self, snapshot) -> None:
        """``StateSnapshot.revealed_unpositioned``: "they have this card" is known, the slot is not.

        For sampling the position does not matter (the hand is a multiset anyway, and the positions of set cards are
        constrained separately by the ``facedown_categories`` column), so the counts are added to the disclosed counts
        of the matching zone and the unknown slots shrink accordingly.

        Field set cards are **not** added to ``disclosed_facedown``: after a shuffle these identities are expressed as
        per-slot category constraints, and that column already carries the identities. Booking both would count the same
        card twice: one unknown set-card slot deducted while the constraint still occupies it, so
        :meth:`Evidence.check` reports "does not fit".
        """
        from ..netduel import constants as C

        for controller, location, code in getattr(
                snapshot, "revealed_unpositioned", ()) or ():
            if int(controller) != self.player or not code:
                continue
            location = int(location)
            if location == C.LOCATION_HAND:
                self.disclosed_hand[int(code)] += 1
            elif location == C.LOCATION_DECK:
                self.disclosed_deck[int(code)] += 1
            elif location in (C.LOCATION_MZONE, C.LOCATION_SZONE):
                # paired into groups with this zone's shuffle constraints by :func:`_shuffled_groups`
                self.unpositioned_field.setdefault(location, Counter())[int(code)] += 1

    def _observe_field(self, card, known: int) -> None:
        from ..netduel import constants as C

        key = (card.location, card.sequence)
        if (card.position & C.POS_FACEUP) and card.code:
            self.credit(card.code)
            return
        self.facedown_keys.append(key)
        if known and known in self.extra_codes:
            # an extra monster flipped face-down: identity public, slot permutable, but it goes through the extra pool
            self.extra_keys.append(key)
            self.disclosed_extra_slots[known] += 1
            return
        self.facedown_slots += 1
        if known:
            self.disclosed_facedown[known] += 1


def _shuffled_groups(categories, *, tally, revealed_keys: dict) -> tuple[tuple, tuple, tuple]:
    """Pair the per-slot constraints of a set-card shuffle into groups "these slots hold exactly this multiset".

    The ledger records a shuffle as per-slot constraints on the participating slots (``source_code == 0``, the
    admissible set being the identities of those cards before the shuffle), and the unpositioned identities
    (``revealed_unpositioned``) give the exact count of each code. With per-slot constraints alone, when the unknown
    pool has other copies of a card, two slots could both draw that code, contradicting the multiset the observer
    knows. So when a zone's shuffle-constrained slots equal its unpositioned identities in number and every slot's
    admissible set covers that multiset, they form a group: the cards count as disclosed and the per-slot constraints are dropped (the group is the stronger constraint).

    Returns ``(groups, remaining constraints, remaining unpositioned identities)``. Identities after partial disclosure
    or departure are still real information, left to the joint field-constraint sampling; they must neither be dropped nor pinned to an unproven slot.
    """
    player = tally.player
    groups, covered, unpositioned = [], set(), []
    for location, known in sorted(tally.unpositioned_field.items()):
        known = +known
        claims = [claim for claim in categories or ()
                  if int(claim.location) == location and claim.sequence is not None
                  and int(claim.source_code) == 0
                  and (location, int(claim.sequence)) in tally.facedown_keys
                  and not revealed_keys.get((player, location, int(claim.sequence)))]
        codes = set(known)
        if len(claims) != sum(known.values()) or not claims \
                or any(set(int(code) for code in claim.codes) != codes for claim in claims):
            unpositioned.extend((location, code) for code in sorted(known.elements()))
            continue
        sequences = tuple(sorted(int(claim.sequence) for claim in claims))
        groups.append((location, sequences, tuple(sorted(known.elements()))))
        covered.update((location, sequence) for sequence in sequences)
    remaining = tuple(claim for claim in categories or ()
                      if claim.sequence is None or (int(claim.location), int(claim.sequence)) not in covered
                      or int(claim.source_code) != 0)
    return tuple(groups), remaining, tuple(unpositioned)


def _split_categories(categories, *, player: int, revealed_keys: dict,
                      facedown_keys) -> tuple[tuple, tuple]:
    """Split the ledger's ``(zone, sequence | None, admissible set)`` into the hand and set-card columns.

    A constraint on a slot whose identity is already pinned is redundant and dropped.
    """
    from ..netduel import constants as C

    hand_category_list: list = []
    facedown_category_list: list = []
    for constraint in categories or ():
        codes = frozenset(int(code) for code in constraint.codes)
        if not codes:
            continue
        location = int(constraint.location)
        sequence = constraint.sequence
        if location == C.LOCATION_HAND:
            if sequence is not None and revealed_keys.get(
                    (player, location, int(sequence))):
                continue
            hand_category_list.append(codes)
        elif location in (C.LOCATION_MZONE, C.LOCATION_SZONE):
            if sequence is None:
                # handed to the joint sampler through unpositioned_facedown_categories;
                # only category claims with a concrete position are output here.
                continue
            key = (location, int(sequence))
            if key not in facedown_keys or revealed_keys.get(
                    (player, location, int(sequence))):
                continue
            facedown_category_list.append((key, codes))
    return tuple(hand_category_list), tuple(facedown_category_list)


def _classify_removed(tally: _Tally, extra_origin_slots) -> list:
    """Classify each face-down banished slot: a slot of a main-deck card or of an extra monster.

    Three rules by priority: a slot with a disclosed identity is classified by its code; one the source table says was
    banished face-down from the Extra Deck is extra-class; otherwise main-deck class. ``extra_origin_slots`` ``None`` means
    the caller gave no source table: then slots of unknown identity are all left for conservation to infer **by count** (the old calibration),
    and the unclassified slots are returned.
    """

    origins = (None if extra_origin_slots is None
               else {(int(location), int(sequence))
                     for location, sequence in extra_origin_slots})
    unattributed: list = []
    for location, sequence, known in tally.removed_candidates:
        key = (location, sequence)
        tally.facedown_keys.append(key)
        if known and known in tally.extra_codes:
            tally.extra_keys.append(key)
            tally.disclosed_extra_slots[known] += 1
        elif origins is not None and key in origins:
            tally.extra_keys.append(key)
        elif known:
            tally.facedown_slots += 1
            tally.disclosed_facedown[known] += 1
        elif origins is not None:
            tally.facedown_slots += 1
        else:
            unattributed.append(key)
    return unattributed


def evidence_from_snapshot(snapshot, *, player: int, deck_list: Counter,
                           revealed=(), extra_codes=frozenset(),
                           public_extra=(), categories=(), extra_list=None,
                           extra_origin_slots=None) -> Evidence:
    """Build hidden-card evidence about ``player`` from a snapshot **already masked for the observer**.

    ``snapshot`` is the product of :func:`mirrorforce.worldmodel.state.mask_for`:
    the opponent's hand and deck are anonymous slots and the public zones are as usual. ``revealed`` is
    the set of instance coordinates given by :meth:`DisclosureLedger.resolve`,
    ``(controller, zone, sequence, code)``: the "disclosure is input" part.

    **This function reads only the masked snapshot.** Given an unmasked omniscient snapshot it would book the opponent's
    real hand as ``disclosed_hand``, which is a leak. Masking first is the caller's job.

    Face-down banished slots: classified by public source
    ----------------------------

    A face-down banished card does not show whether it is a main-deck card or an extra monster, but **its source is
    public**: the server does not erase the location words of ``MSG_MOVE``, and "this card was banished from a face-down
    Extra Deck position" is visible to both sides. ``extra_origin_slots`` is that source table (the ``(zone, sequence)``
    set given by :mod:`mirrorforce.search.banish_origin`): slots in it are **extra-class** and drawn from
    :meth:`Evidence.unknown_extra_pool`; slots not in it are main-deck class and drawn from the unknown pool.
    Slots with disclosed identities are classified by code, regardless of the source table.

    ``extra_origin_slots=None`` is the old calibration: without a source table, face-down banished slots of unknown
    identity can only have their **count** inferred by conservation:

        extra monsters = face-down banished slots − min(remainder, face-down banished slots)

    and nobody can say which slot. At such a decision point :attr:`Evidence.extra_slot_keys` is empty,
    :attr:`Evidence.extra_in_slots` is non-zero, and the sampler judges it unrealizable instead of guessing.

    Other hidden slots are **inferred** by conservation
    --------------------------

    Every main-deck card is right now either known by identity (face-up on the field / graveyard / banished, under an Xyz
    monster, or disclosed) or occupies a slot of unknown identity. The slot counts of the first kinds are public; the remainder

        other hidden slots = list size − located − disclosed − unknown hand − unknown set cards − unknown deck

    is the number of main-deck cards "somewhere invisible that cannot be permuted either" (now only cards whose control
    was taken). They **must** be deducted from the unknown pool, or hand and deck are sampled from a pool that is too large;
    when non-zero the decision point is unrealizable.

    ``public_extra`` are the **face-up** main-deck cards in the opponent's Extra Deck zone (pendulum monsters returned face-up
    to the Extra Deck). Their identity is public, but the snapshot's Extra Deck zone reports only a count, so they are an
    explicit parameter; given them, these cards count as located and the unknown pool shrinks. ``extra_list`` is the
    opponent's Extra Deck list; the extra pool is it minus the located extra monsters; without it extra-class slots cannot be sampled.

    ``categories`` may also contain :class:`FreshSlots` (fresh slots), which go into
    :attr:`Evidence.fresh_facedown_keys`.
    """
    from ..netduel import constants as C

    fresh_keys = {tuple(int(value) for value in key) for item in categories or ()
                  if isinstance(item, FreshSlots) for key in item.keys}
    categories = tuple(item for item in categories or () if not isinstance(item, FreshSlots))
    extra_list = Counter(extra_list or ())
    tally = _Tally(player=player, deck_list=deck_list,
                   extra_codes=frozenset(extra_codes) | frozenset(extra_list))
    for code in public_extra:
        tally.credit(int(code))
    revealed_keys = {(c, l, s): code for c, l, s, code in revealed}
    for card in snapshot.cards:
        tally.observe(card, revealed_keys)
    tally.observe_unpositioned(snapshot)
    shuffled, categories, unpositioned = _shuffled_groups(categories, tally=tally, revealed_keys=revealed_keys)
    for _location, _sequences, codes in shuffled:
        tally.disclosed_facedown.update(codes)
    hand_categories, facedown_categories = _split_categories(
        categories, player=player, revealed_keys=revealed_keys,
        facedown_keys=tally.facedown_keys)
    unattributed = _classify_removed(tally, extra_origin_slots)
    # Extra-deck identities without a publicly known permutation-pool slot
    # remain unsupported; never guess pool membership from the hidden truth.
    unexpressed = sum(code in tally.extra_codes for _location, code in unpositioned)
    unpositioned = tuple((location, code) for location, code in unpositioned if code not in tally.extra_codes)
    field_categories = tuple((int(claim.location), frozenset(int(code) for code in claim.codes))
                             for claim in categories if claim.sequence is None and claim.codes
                             and int(claim.location) in (C.LOCATION_MZONE, C.LOCATION_SZONE))
    fixed = {(int(card.location), int(card.sequence)): int(card.code)
             for card in snapshot.cards if card.controller == player and card.code
             and (card.location, card.sequence) in tally.facedown_keys}
    fixed.update(((location, sequence), code) for (controller, location, sequence), code in revealed_keys.items()
                 if controller == player and (location, sequence) in tally.facedown_keys)
    grouped = {(location, sequence) for location, sequences, _codes in shuffled for sequence in sequences}
    sampling_keys = tuple(key for key in tally.facedown_keys
                          if key not in fixed and key not in grouped and key not in tally.extra_keys)
    fixed_identities = tuple((location, code) for (location, _sequence), code in sorted(fixed.items()))
    fixed_identities += tuple((location, code) for location, _sequences, codes in shuffled for code in codes)

    total = sum(deck_list.values())
    accounted = (sum(tally.located.values()) + sum(tally.disclosed_hand.values())
                 + sum(tally.disclosed_deck.values())
                 + sum(tally.disclosed_facedown.values()))
    unknown_here = ((tally.hand_size - sum(tally.disclosed_hand.values()))
                    + (tally.facedown_slots - sum(tally.disclosed_facedown.values()))
                    + (tally.deck_size - sum(tally.disclosed_deck.values())))
    residual = total - accounted - unknown_here
    # the old calibration's inference from the remainder: the count can be told, the slots cannot
    main_in_removed = max(0, min(residual, len(unattributed)))
    tally.facedown_slots += main_in_removed
    residual -= main_in_removed
    inferred_extra = len(unattributed) - main_in_removed
    if residual < 0:
        raise BeliefError(
            f"the public evidence has {total - residual} main-deck cards, but the decklist only "
            f"{total}: {-residual} too many; the decklist does not match this duel, "
            "or a code was booked twice")
    extra_total = int(snapshot.counts.get((player, C.LOCATION_EXTRA),
                                          tally.extra_rows))
    return Evidence(
        deck_list=Counter(deck_list),
        located=tally.located,
        disclosed_hand=tally.disclosed_hand,
        disclosed_deck=tally.disclosed_deck,
        disclosed_facedown=tally.disclosed_facedown,
        hand_size=tally.hand_size,
        deck_size=tally.deck_size,
        facedown_slots=tally.facedown_slots,
        facedown_slot_keys=tuple(tally.facedown_keys),
        extra_in_slots=len(tally.extra_keys) + inferred_extra,
        extra_slot_keys=tuple(tally.extra_keys),
        extra_list=extra_list,
        located_extra=tally.located_extra,
        disclosed_extra_slots=tally.disclosed_extra_slots,
        disclosed_extra_deck=tally.disclosed_extra_deck,
        extra_facedown_size=max(0, extra_total - tally.extra_faceup),
        other_hidden_slots=residual,
        foreign_controlled=tally.foreign,
        hand_categories=hand_categories,
        facedown_categories=facedown_categories,
        shuffled_facedown=shuffled,
        unpositioned_facedown=unpositioned,
        unpositioned_facedown_categories=field_categories,
        facedown_sampling_keys=sampling_keys,
        fixed_field_identities=fixed_identities,
        unexpressed_field_knowledge=unexpressed,
        fresh_facedown_keys=tuple(key for key in sampling_keys if key in fresh_keys),
    )
