"""Replay-style particles: replay the real prefix once, fork K copies, and permute the hidden zones inside each fork.

This is the main particle channel at training time. Constructive rebuilding
(:mod:`mirrorforce.search.rebuild` + six setters) is demoted to the deployment form and fallback.

Why replay is more exact than construction
------------------

Constructive rebuilding cannot restore `effect` objects of the kind "the source card has left, its field effect remains":
they carry pointers to the handler card and Lua references to its script, not a few integers. Measured, this item differs at
94–99% of decision points, and it is the only reason constructive rebuilding is stuck at 91.7–94.7%.

Replay has no such problem: **the engine plays the whole duel itself**, effects of departed cards register and expire as usual,
and fidelity is exact by definition. Self-play collection logs both sides' answers, so the prefix is available.

Why permutation is safe
--------------

A particle is **not another deck**. In a closed environment the decklist is known, so "which cards remain" is fixed by
conservation; only **where each card is now** is unknown. Belief sampling gives another assignment of the same multiset,
so permutation "rearranges existing card objects", it does not "change card identities": no objects are created or destroyed,
no scripts rebound, no effects re-registered, no events raised. ``Debug.PermuteHidden``
only changes "which slot holds which object" and the moved objects' ``current.location/sequence/position``.

Hidden cards have never acted, so there is no history state to move with them. **Set cards on the field are the exception**: their identity
is unknown, but "in which turn it was set" is public, and that history belongs to the slot, not to the card object, so after permutation
``Debug.SetCardState`` writes the slot history back.

Two hard rules (protocol, not advice)
--------------------------

1. **No query or evaluation before the permutation completes.** Right after the fork, the child holds the opponent's
   **real** hidden cards. The permutation hangs on the ``prepare`` hook of :class:`~mirrorforce.search.forkbranch.SearchSession`,
   which runs after the fork and before the responder returns; the responder itself
   raises :class:`BranchNotPreparedError` when ``prepare`` has not finished.
2. **Permutation completeness is asserted per particle.** After permutation the hidden zones are read back and must equal the
   particle's assignment slot by slot; the overlap with the real assignment is reported too. Overlap is not a failure (sampling may
   hit the truth), but "read-back == real assignment" while "particle != real assignment" means the permutation did not take effect,
   and that particle fails.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from ..netduel import constants as C
from ..worldmodel.engine import DuelConfig, DuelDriver, StopDuel
from .belief import BeliefSampler, Evidence
from .forkbranch import BranchSpec, SearchSession

__all__ = [
    "HiddenLayout",
    "ParticleError",
    "ParticleUnrealizable",
    "ReplayParticleSource",
    "particles_from_evidence",
    "sample_future_core_seeds",
    "sample_observer_hidden_layout",
    "apply_particle",
    "permute_chunk",
    "read_hidden_layout",
]


class ParticleError(RuntimeError):
    """A particle did not land correctly. Fatal for that particle: never "run first and see"."""


class ParticleUnrealizable(ParticleError):
    """The belief at this decision point **cannot** be realized by permutation, regardless of whether sampling is right.

    Two cases, both "conservation says there are this many cards, but nobody can say which slot they are in":

    1. ``Evidence.other_hidden_slots``: main-deck cards in places permutation cannot reach (now
       only cards whose control was taken). Sampling would exchange them with cards in the hand and deck,
       which permutation cannot do, so the particle's multiset would not line up with the engine's.
    2. ``Evidence.unattributed_extra_slots()``: extra monsters in face-down banished slots, but the caller
       gave no source table (:mod:`mirrorforce.search.banish_origin`), so nobody can say which slot.
       ``Debug.PermuteHidden`` splits the main-deck and extra pools by each slot's current occupant, and one wrong
       slot rejects the whole particle, so it does not guess.

    Rather than letting the engine refuse in Lua, this is decided here in advance and counted separately: it is a gap in the
    sampling input, not a sampler error. ``reason`` says which kind, for separate counts in audits.
    """

    OTHER_HIDDEN_SLOTS = "other_hidden_slots"
    UNATTRIBUTED_EXTRA_SLOTS = "unattributed_extra_slots"

    def __init__(self, message: str, *, reason: str = OTHER_HIDDEN_SLOTS) -> None:
        super().__init__(message)
        self.reason = str(reason)


# -- reading and writing hidden zones ------------------------------------------------------------


@dataclass
class HiddenLayout:
    """What a player's hidden zones hold right now, by slot.

    Index 0 of ``deck`` is the bottom and the last the top, the same direction as ``field::list_main``.
    ``facedown`` is ``[(zone, sequence, code)]``, with three kinds of slots of invisible identity: set
    spells/traps on the field, face-down monsters, and **face-down banished** cards. Face-down banished cards make up
    93% of hidden cards outside hand and deck (measured 76/82), clustered in the decks that banish face-down; missing
    them is not just a small coverage gap, it is a coverage gap **by deck**.

    ``extra`` is the target order of the **face-down** Extra Deck cards (index 0 = ``list_extra[0]``;
    face-up pendulum monsters at the end are public and not included). Face-down banished slots may hold extra monsters
    (Pot of Extravagance banishes random face-down Extra Deck cards face-down); they can only be exchanged with face-down
    Extra Deck cards, so the particle gives the Extra Deck too. ``None`` means the Extra Deck takes no part in the permutation:
    the engine is unchanged, and extra monsters are exchanged only among face-down banished slots.
    """

    hand: tuple[int, ...] = ()
    deck: tuple[int, ...] = ()
    facedown: tuple[tuple[int, int, int], ...] = ()
    extra: tuple[int, ...] | None = None

    def multiset(self) -> Counter:
        out = Counter(self.hand)
        out.update(self.deck)
        out.update(code for _l, _s, code in self.facedown)
        if self.extra is not None:
            out.update(self.extra)
        return out

    def assignment(self) -> tuple:
        """The comparable complete assignment. Two layouts are equal exactly when every slot holds the same code.

        Without the Extra Deck the shape is as before (a triple); with it one more item; so the digest of particles that
        do not touch the Extra Deck (``snap_engine._layout_digest``) stays byte-identical.
        """
        out = (self.hand, self.deck, tuple(sorted(self.facedown)))
        if self.extra is not None:
            out = out + (self.extra,)
        return out

    def restrict(self, keys, *, keep_extra: bool = True) -> "HiddenLayout":
        """Keep only the set-card slots in ``keys``; hand and deck stay.

        Before comparing, both sides are trimmed to the same slots with this: set-card slots not in the particle should
        not move anyway, and comparing them would report "not moved" as "not permuted". ``keep_extra=False`` trims
        the Extra Deck too, for comparison with a particle that does not touch it.
        """
        keep = set(keys)
        return HiddenLayout(
            hand=self.hand, deck=self.deck,
            facedown=tuple(x for x in self.facedown if (x[0], x[1]) in keep),
            extra=self.extra if keep_extra else None)


def sample_observer_hidden_layout(snapshot, player: int, *, seed: int) -> HiddenLayout:
    """Sample the observer's own unknown deck order from a PUB snapshot.

    The observer knows every identity in their hand, main deck and own
    facedown slots, but not the shuffled main-deck order.  ``mask_for``
    canonicalizes that order, so this function uses an independent public seed
    to permute it rather than preserving the live core's true order.
    """

    if player not in (0, 1):
        raise ParticleError(f"observer player must be seat 0 or 1, got {player!r}")
    if getattr(snapshot, "view", None) != player:
        raise ParticleError(
            f"observer layout requires a PUB snapshot for seat {player}, got "
            f"view={getattr(snapshot, 'view', None)!r}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ParticleError("observer deck-order seed must be a nonnegative integer")

    def zone(location: int):
        return sorted(
            (
                card
                for card in snapshot.cards
                if int(card.controller) == player and int(card.location) == location
            ),
            key=lambda card: int(card.sequence),
        )

    hand_cards = zone(C.LOCATION_HAND)
    deck_cards = zone(C.LOCATION_DECK)
    if any(not int(card.code) for card in hand_cards + deck_cards):
        raise ParticleError(
            "observer PUB snapshot omitted an identity from its own hand/deck"
        )
    expected_hand = int(snapshot.counts.get((player, C.LOCATION_HAND), len(hand_cards)))
    expected_deck = int(snapshot.counts.get((player, C.LOCATION_DECK), len(deck_cards)))
    if len(hand_cards) != expected_hand or len(deck_cards) != expected_deck:
        raise ParticleError(
            "observer PUB snapshot does not contain every own hand/deck slot"
        )

    deck = [int(card.code) for card in deck_cards]
    rng = random.Random(
        (
            "observer-deck-order-v1",
            int(seed),
            int(player),
            tuple(sorted(deck)),
        ).__repr__()
    )
    rng.shuffle(deck)
    facedown = []
    for location in (C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_REMOVED):
        for card in zone(location):
            if not card.face_down:
                continue
            if not int(card.code):
                raise ParticleError(
                    "observer PUB snapshot omitted an identity from its own "
                    "facedown slot"
                )
            facedown.append((location, int(card.sequence), int(card.code)))
    return HiddenLayout(
        hand=tuple(int(card.code) for card in hand_cards),
        deck=tuple(deck),
        facedown=tuple(facedown),
    )


def sample_future_core_seeds(seed: int, particle_id: str) -> tuple[int, ...]:
    """Derive an independent eight-word future-core RNG particle.

    This sampler depends only on caller-owned public randomness and a particle
    id.  It never reads or advances the live duel RNG.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ParticleError("future-core seed must be a nonnegative integer")
    if not str(particle_id):
        raise ParticleError("future-core particle_id may not be blank")
    rng = random.Random(
        ("future-core-rng-v1", int(seed), str(particle_id)).__repr__()
    )
    return tuple(rng.getrandbits(32) for _ in range(8))


def read_hidden_layout(driver: DuelDriver, player: int) -> HiddenLayout:
    """Read a player's hidden zones from a live duel. **Only inside a fork or in offline diagnostics.**"""
    import ctypes

    from ..netduel.board import parse_query_segments
    from ..puzzle.core import SIZE_QUERY_BUFFER
    from .rebuild import FULL_QUERY_FLAGS

    buf = ctypes.create_string_buffer(SIZE_QUERY_BUFFER)

    def zone(location):
        length = driver.core.query_field_card(
            driver.pduel, player, location, FULL_QUERY_FLAGS, buf)
        if length <= 0:
            return []
        return parse_query_segments(buf.raw[:length])

    def faceup(fields) -> bool:
        return bool(((fields.get("info_location", 0) >> 24) & 0x0F) & C.POS_FACEUP)

    hand = tuple(f.get("code", 0) for f in zone(C.LOCATION_HAND) if f)
    deck = tuple(f.get("code", 0) for f in zone(C.LOCATION_DECK) if f)
    facedown = []
    for location in (C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_REMOVED):
        for sequence, fields in enumerate(zone(location)):
            if not fields or not fields.get("code"):
                continue
            if faceup(fields):
                continue
            facedown.append((location, sequence, fields["code"]))
    # only face-down cards of the Extra Deck zone: face-up pendulum monsters at the end are public and permutation leaves them alone
    extra = tuple(f["code"] for f in zone(C.LOCATION_EXTRA)
                  if f and f.get("code") and not faceup(f))
    return HiddenLayout(hand=hand, deck=deck, facedown=tuple(facedown),
                        extra=extra)


def _lua_array(codes) -> str:
    return "{" + ",".join(str(int(c)) for c in codes) + "}"


def permute_chunk(player: int, target: HiddenLayout, slot_history=()) -> str:
    """The Lua ``apply_particle`` sends to the core; a pure function.

    ``PermuteChunk`` in ``snapshot.cc`` on the C++ side generates the same text byte for byte, and the cross-check test
    compares this string directly; if either side changes the format, the cross-check fails first.
    """
    facedown_flat = [v for slot in target.facedown for v in slot]
    args = "%d,%s,%s,%s" % (player, _lua_array(target.hand),
                            _lua_array(target.deck), _lua_array(facedown_flat))
    if target.extra is not None:
        args += "," + _lua_array(target.extra)
    lines = ["mf_permute_ok = Debug.PermuteHidden(%s)" % args]
    for location, sequence, status, turnid, turn_counter in slot_history:
        lines.append(
            f"local s=Duel.GetFieldCard({player},{location},{sequence}) "
            f"if s then Debug.SetCardState(s,{int(status)},{int(turnid)},"
            f"{int(turn_counter)}) end")
    lines.append("if mf_permute_ok ~= true then "
                 "error('PermuteHidden refused the assignment') end")
    return "\n".join(lines) + "\n"


def apply_particle(driver: DuelDriver, player: int, target: HiddenLayout,
                   *, script_name: str = "./script/mirrorforce-r3-permute.lua",
                   slot_history=()) -> HiddenLayout:
    """Permute ``player``'s hidden zones into ``target``, then read back and check.

    ``slot_history`` is ``[(zone, sequence, status, turnid, turn_counter)]``: the history of a set-card slot
    belongs to the **slot** and is written back onto the card permuted in.

    Returns the layout read back after permutation. Any failing step raises :class:`ParticleError`:
    half a particle is more dangerous than none, since it still carries the opponent's real cards.
    """
    import ctypes

    core = driver.core
    source = permute_chunk(player, target, slot_history).encode("utf-8")
    core.log.clear()
    core._script_cache[script_name] = (
        ctypes.c_ubyte * len(source)).from_buffer_copy(source)
    if not core.preload_script(driver.pduel, script_name):
        raise ParticleError(
            "the permutation script did not run; core log: " + " | ".join(core.log[:3]))
    if core.log:
        raise ParticleError("permutation script error: " + core.log[0].replace("\n", " ")[:240])

    # Compare only the slots target itself declares: set-card slots not in target should not move anyway,
    # and comparing them would report "not moved" as "not permuted". Likewise the Extra Deck: not compared unless the particle gives it.
    got = read_hidden_layout(driver, player).restrict(
        ((location, sequence) for location, sequence, _code in target.facedown),
        keep_extra=target.extra is not None)
    if got.assignment() != target.assignment():
        raise ParticleError(
            f"read-back after permutation differs from the particle: hand {got.hand} vs {target.hand}, "
            f"deck size {len(got.deck)} vs {len(target.deck)}, "
            f"Extra Deck {got.extra} vs {target.extra}")
    return got


# -- particle source ------------------------------------------------------------------


class ReplayParticleSource:
    """Replay the real prefix once -> fork K copies at the decision point -> permute the hidden zones inside each fork.

    Usage::

        source = ReplayParticleSource(config, responses, upto, viewer=0, core=core)
        outcomes = source.expand(particles, on_ready=score_the_menu)

    ``on_ready(prompt, driver)`` is called **after the permutation** and before the branch continues;
    all evaluation goes here, so by timing it cannot come before the permutation. To bring results back to the host, use
    :meth:`SearchSession.report`: branches run in their own processes, and the host sees nothing written any other way.
    """

    def __init__(self, config: DuelConfig, responses, upto: int, *,
                 viewer: int, core=None, base_responder=None):
        self.config = config
        self.responses = [bytes(r) for r in responses[:upto]]
        self.upto = upto
        self.viewer = viewer
        self.core = core
        self.base_responder = base_responder
        self.driver: DuelDriver | None = None
        #: the real hidden assignment at the decision point, only for diagnostics and assertions, never fed to the policy
        self.truth: HiddenLayout | None = None
        self.prompt = None

    # -- prefix replay ----------------------------------------------------------

    def replay(self) -> DuelDriver:
        """Rebuild a duel and feed the real answers back, parking at the first prompt that needs a new decision.

        The fidelity of this step is exact by definition: the core is a pure function of (seed sequence, card order, answer stream),
        and feeding back the same bytes gives the same duel. Effects of departed cards, once-per-turn bookkeeping and each card's turnid
        all grow out of the engine itself and need no restoring.
        """
        from ..puzzle.core import get_core

        core = self.core if self.core is not None else get_core()
        self.core = core
        driver = DuelDriver(self.config, core).build()
        driver._replay = list(self.responses)
        self.driver = driver
        return driver

    # -- expansion --------------------------------------------------------------

    def expand(self, particles, *, script=None, on_ready=None,
               max_seconds: float = 30.0, max_decisions: int = 1,
               parallel: int = 0, policy_seed: int = 0):
        """Replay to the decision point and fork each particle into a branch there.

        ``particles`` is ``[(tag, HiddenLayout, slot_history)]``.
        Each branch: permute -> record the menu -> walk ``max_decisions`` steps by ``script`` and stop.

        ``script`` is ``[action description]``: branches follow by **description**, not index (indices shift
        when menu orders differ). With a script, follow it; without one, always choose item 0.
        Every step's menu is recorded in ``menus`` of the payload, for step-by-step reconciliation with the real continuation.

        The menu at the decision point itself was emitted by the core **before the permutation**, so for us it necessarily
        equals the real menu: the opponent's hidden cards should not affect our legal actions anyway. As a check that the
        permutation did not pollute the shared board it is meaningful; as evidence of "particle fidelity" it is not;
        the real evidence is the menus from step 1 on, which are computed after the permutation.
        """
        driver = self.driver if self.driver is not None else self.replay()
        results: dict = {}
        viewer = self.viewer
        opponent = 1 - viewer
        by_tag = {tag: (layout, history) for tag, layout, history in particles}

        def prepare(session, spec: BranchSpec, prompt):
            # In the child, after the fork, before the responder returns. Where the first hard rule is enforced.
            layout, history = by_tag[spec.tag]
            got = apply_particle(session.driver, opponent, layout,
                                 slot_history=history)
            # Second hard rule: permutation completeness. apply_particle already compared the read-back with the particle slot by slot;
            # one more copy is recorded here for the host to check afterwards "was it permuted, and into what".
            session.report("hidden_hand", list(got.hand))
            session.report("hidden_deck_head", list(got.deck[-8:]))
            session.report("hidden_facedown", [list(x) for x in got.facedown])
            menu = sorted(a.describe() for a in prompt.actions)
            session.report("menus", [menu])
            session.report("prompt_msgs", [int(prompt.msg)])
            if on_ready is not None:
                on_ready(prompt, session.driver, session)
            return _pick(prompt, script, 0)

        def on_point(prompt, session):
            self.prompt = prompt
            # The real assignment is read once on the host for reconciliation afterwards; it is no branch's input
            self.truth = read_hidden_layout(driver, opponent)
            specs = [BranchSpec(action=0, policy_seed=policy_seed,
                                max_decisions=max_decisions, tag=tag)
                     for tag, _layout, _history in particles]
            outcomes, _lat = session.expand(
                prompt, specs, max_seconds=max_seconds, measure=False,
                parallel=parallel, prepare=prepare,
                inner_policy=_script_policy(script, session))
            results["outcomes"] = outcomes
            raise StopDuel

        session = SearchSession(
            driver, self.base_responder or (lambda p, d: 0))
        session.arm(lambda prompt: True, on_point)
        try:
            session.run(max_steps=200000, max_seconds=180)
        except StopDuel:
            pass
        return results.get("outcomes", [])


def _pick(prompt, script, step: int) -> int:
    """Find the step the script wants in the menu by action **description**; fall back to item 0 if absent."""
    if not script or step >= len(script):
        return 0
    want = script[step]
    for index, action in enumerate(prompt.actions):
        if action.describe() == want:
            return index
    return 0


def _script_policy(script, session):
    """Every step after the permutation: record the menu and pick the action by script.

    ``prepare`` has already finished (otherwise the responder raises ``BranchNotPreparedError`` first),
    so by timing nothing here can run before the permutation.
    """
    state = {"step": 1}

    def choose(prompt, driver) -> int:
        step = state["step"]
        state["step"] = step + 1
        session._payload.setdefault("menus", []).append(
            sorted(a.describe() for a in prompt.actions))
        session._payload.setdefault("prompt_msgs", []).append(int(prompt.msg))
        return _pick(prompt, script, step)

    return choose


def _order_facedown_for_categories(drawn, slots, pinned, categories):
    """Reorder the sampled set cards so that slots with category constraints get cards in their sets.

    :class:`~mirrorforce.search.belief.BeliefSampler` only guarantees a system of distinct representatives in the
    sampled **multiset**; which card goes to which slot is decided here, because the slot order is only known
    in :func:`particles_from_evidence` (pinned slots are filtered out first).

    A bipartite matching (Kuhn's augmenting paths) runs between constrained slots and candidate cards. The sampler already
    guarantees a system of distinct representatives exists, so the matching must succeed; if it somehow does not, the input is
    returned unchanged so the later conservation check fails, rather than quietly relaxing the constraint here.
    """

    allowed = {tuple(key): codes for key, codes in categories}
    if not allowed:
        return list(drawn)
    open_slots = [key for key in slots if key not in pinned]
    cards = list(drawn)
    wanted = [(index, allowed[key]) for index, key in enumerate(open_slots)
              if key in allowed]
    if not wanted or len(open_slots) != len(cards):
        return cards
    match_for_card: dict[int, int] = {}

    def augment(slot_index, codes, seen):
        for card_index, code in enumerate(cards):
            if code not in codes or card_index in seen:
                continue
            seen.add(card_index)
            holder = match_for_card.get(card_index)
            if holder is None or augment(holder[0], holder[1], seen):
                match_for_card[card_index] = (slot_index, codes)
                return True
        return False

    for slot_index, codes in wanted:
        if not augment(slot_index, codes, set()):
            return cards
    placed: dict[int, int] = {}
    used = set()
    for card_index, (slot_index, _codes) in match_for_card.items():
        placed[slot_index] = cards[card_index]
        used.add(card_index)
    spare = [code for index, code in enumerate(cards) if index not in used]
    spare_iter = iter(spare)
    return [placed[index] if index in placed else next(spare_iter)
            for index in range(len(open_slots))]


def _permutable_slots(evidence: Evidence, truth: HiddenLayout
                      ) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """The set-card slots ``[(zone, sequence)]`` a particle fills, and the set of permutable slots.

    The slots follow the evidence, not the real board: the evidence already decided by public sources
    which slots are permutable. A hand-made Evidence (unit tests) without a slot list falls back to
    "every set-card slot on the real board".
    """
    permutable = set(evidence.facedown_slot_keys)
    if evidence.facedown_slot_keys:
        slots = [(location, sequence) for location, sequence, _code in truth.facedown
                 if (location, sequence) in permutable]
    elif evidence.facedown_slots == 0:
        slots = []
    else:
        slots = [(location, sequence) for location, sequence, _code in truth.facedown]
        permutable = set(slots)
    want_slots = evidence.facedown_slots + evidence.extra_in_slots
    if len(slots) != want_slots:
        raise ParticleError(
            f"the evidence says there are {want_slots} permutable hidden slots "
            f"(main deck {evidence.facedown_slots} + extra {evidence.extra_in_slots}), "
            f"but only {len(slots)} match on the real board")
    return slots, permutable


def _check_extra_kinds(evidence: Evidence, pinned: dict, extra_codes) -> None:
    """Identities pinned to extra-class slots must be extra monsters; checked only when ``extra_codes`` is given."""
    if not extra_codes:
        return
    for key in evidence.extra_slot_keys:
        code = pinned.get(tuple(key))
        if code is not None and code not in extra_codes:
            raise ParticleError(
                f"slot {key} is an extra-class slot, but its pinned identity {code} is not in the Extra Deck list")


def particles_from_evidence(evidence: Evidence, count: int, *, seed: int,
                            truth: HiddenLayout, revealed=(), player: int | None = None,
                            extra_codes=frozenset(), shape_only: bool = False,
                            slot_history=(), sampler=None) -> list[tuple[str, HiddenLayout, tuple]]:
    """Sample ``count`` particles and translate them into :class:`HiddenLayout`.

    The **positions** of set-card slots (which zone, which sequence) come from the real board, which is public; what they
    hold comes from sampling. **Slots already disclosed must be pinned**: ``BeliefSampler`` only gives the undisclosed cards
    (``Evidence.disclosed_*`` are already deducted from the unknown pool), so the particle must put the disclosed
    cards back in place, or the slot counts do not match. ``revealed`` is the
    ``(controller, zone, sequence, code)`` set given by :meth:`DisclosureLedger.resolve`.

    Without ``revealed`` it falls back to the old calibration: disclosed hand cards are pinned at the front in ascending sequence order
    (the same convention as the canonical assignment of ``netduel.disclosure.resolve_disclosure``),
    and set-card slots are all assumed undisclosed.

    **Extra-class slots are drawn from the extra pool.** The face-down banished slots holding extra monsters
    (``Evidence.extra_slot_keys``, publicly sourced) are drawn from the extra pool without replacement, and the rest becomes the
    particle's ``extra`` in ascending code order (the target order of the face-down Extra Deck cards); main-deck class slots are
    drawn from the unknown pool as before. ``Debug.PermuteHidden`` takes cards from two pools by each slot's current occupant, so
    the two kinds must not mix: mixing them rejects the whole particle. A decision point where conservation says there are extra monsters but
    the source table cannot say which slot (``Evidence.unattributed_extra_slots() > 0``) is unrealizable, not guessed.

    ``shape_only=True`` is the online (search server) mode: ``truth`` has only slot shapes and
    all identities are zero, since the rules forbid reading real hidden identities outside the fork. Then the final identity check
    "sampled multiset == engine multiset" has nothing to compare with and is left to the permuter inside the fork (which holds the engine's
    truth, and whose pre-write permutation refuses unrealizable layouts anyway); only structural checks remain here.

    ``sampler`` samples hidden cards instead of the evidence's own :class:`BeliefSampler`: it gives the same
    :class:`Particle`, and ``keyed_facedown`` says whether set cards are already arranged slot by slot by
    ``Evidence.facedown_sampling_keys``.
    """
    if evidence.other_hidden_slots:
        raise ParticleUnrealizable(
            f"{evidence.other_hidden_slots} cards of the unknown pool are where permutation cannot reach; "
            "particles for this decision point cannot be realized",
            reason=ParticleUnrealizable.OTHER_HIDDEN_SLOTS)
    unattributed = evidence.unattributed_extra_slots()
    if unattributed > 0:
        raise ParticleUnrealizable(
            f"{unattributed} extra monsters occupy face-down banished slots, but no source table can say which; "
            "particles for this decision point cannot be realized",
            reason=ParticleUnrealizable.UNATTRIBUTED_EXTRA_SLOTS)

    pinned = {}
    for controller, location, sequence, code in revealed:
        if player is None or controller == player:
            pinned[(location, sequence)] = code
    _check_extra_kinds(evidence, pinned, extra_codes)

    if sampler is None:
        sampler = BeliefSampler(evidence, random.Random(("particle", seed).__repr__()))
    # shuffled set-card groups: each particle arranges the group's multiset randomly over its slots (positions unknowable)
    shuffles = random.Random(("shuffled-facedown", seed).__repr__())
    # Disclosed hand cards **without a position** (``StateSnapshot.revealed_unpositioned``) must
    # go into the hand too: ``Evidence.disclosed_hand`` already deducted them from the unknown pool, and
    # without them the hand slots cannot be filled. Cards with pinned positions go through ``pinned``; the rest
    # are added from this list in any order, since the hand is a multiset.
    pinned_hand: Counter = Counter(
        code for (location, _sequence), code in pinned.items()
        if location == C.LOCATION_HAND)
    fallback_hand: list[int] = []
    for code, n in sorted((Counter(evidence.disclosed_hand)
                           - pinned_hand).items()):
        fallback_hand.extend([code] * n)
    slots, permutable = _permutable_slots(evidence, truth)
    extra_keys = set(evidence.extra_slot_keys)
    main_slots = [key for key in slots if key not in extra_keys]
    truth_multiset = (
        None if shape_only
        else truth.restrict(permutable, keep_extra=bool(extra_keys)).multiset())

    out = []
    for i in range(count):
        drawn = sampler.sample()
        hand_pool = iter(drawn.hand)
        spare_known = iter(fallback_hand)
        hand = []
        for sequence in range(len(truth.hand)):
            code = pinned.get((C.LOCATION_HAND, sequence))
            if code is None:
                code = next(spare_known, None)
            hand.append(code if code is not None else next(hand_pool, None))
        slot_pins = dict(pinned)
        for location, sequences, codes in evidence.shuffled_facedown:
            order = list(codes)
            shuffles.shuffle(order)
            slot_pins.update(((location, sequence), code) for sequence, code in zip(sequences, order))
        if sampler.keyed_facedown:
            # The joint sampler already chose the concrete open slots. A
            # subsequent category-only matching could move a known identity
            # to another zone and invalidate the position-free evidence.
            open_keys = tuple(key for key in main_slots if key not in slot_pins)
            if open_keys != evidence.facedown_sampling_keys:
                raise ParticleError("conditional field slots differ from the public layout")
            facedown_pool = iter(drawn.facedown)
        else:
            facedown_pool = iter(_order_facedown_for_categories(
                drawn.facedown, main_slots, slot_pins, evidence.facedown_categories))
        extra_pool = iter(drawn.extra)
        facedown = []
        for location, sequence in slots:
            code = slot_pins.get((location, sequence))
            if code is None:
                source = extra_pool if (location, sequence) in extra_keys else facedown_pool
                code = next(source, None)
            facedown.append((location, sequence, code))
        if any(c is None for c in hand) or any(c is None for _l, _s, c in facedown):
            raise ParticleError(
                f"sampling did not give enough cards to fill the slots: {len(truth.hand)} hand slots, "
                f"{len(slots)} set-card slots, but sampling gave only {len(drawn.hand)} + "
                f"{len(drawn.facedown)} + {len(drawn.extra)} cards")
        deck = tuple(drawn.deck)
        if len(deck) != len(truth.deck):
            raise ParticleError(
                f"the particle deck has {len(deck)} cards, the real deck {len(truth.deck)}")
        layout = HiddenLayout(hand=tuple(hand), deck=deck,
                              facedown=tuple(facedown), extra=drawn.extra_deck)
        # shape mode has no identities to compare (truth is all zero); the pre-write permutation inside the fork holds the engine's
        # truth and refuses unrealizable layouts there
        if truth_multiset is not None and layout.multiset() != truth_multiset:
            raise ParticleError(
                "the particle's multiset does not line up with the engine's: permutation cannot realize it")
        out.append((f"p{i}", layout, tuple(slot_history)))
    sampler.clear()
    return out
