"""Opponent public activations the blank shadow can replay only once their hidden card is public.

The real opponent activated a public card (the stream shows its CHAINING) that
the local core never offered: its cost or effect needs a card the shadow holds
only as a blank -- a hand card to discard, a deck card to search, an extra-deck
monster to summon. A seat is not told which hidden card made the activation
legal until the chain resolves, and our own prompts in between (usually chain
responses) must be answered first.

So the follower stays at the start of the batch that diverged, answers those
prompts opaquely (the real client still decides; no native state, no search),
and waits for the chain to end. Every card that left the opponent's hand, deck
or extra deck publicly since the batch start -- the activated card itself, a
MOVE carrying its code, or a MOVE to a hidden place followed by its CONFIRM --
is traced back to its slot at the batch start, those exact blanks are hydrated
there, and the whole received history is replayed natively.

When the activation's legality needed a deck card that never became public
(its effect was negated, say), an optional candidate pool -- the environment's
builds of that deck, most common cards first -- supplies one card assumed in a
blank deck slot: the first that makes the opponent's IDLE offer the
activation. It is recorded as an unvalidated hypothesis, never a public fact.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import struct

from ..netduel import constants as C
from ..puzzle.messages import Message
from ..puzzle.single import RESPONSE_REQUIRED
from .client_sync import RevealConflict, SyncError, _Divergence, _Frontier, _prompt_player

INFORMATION_SET_SEARCH = True
SCHEMA = "client-hidden-target-deferral/v1"
#: How far after a divergence the real stream may show the activation.
_LOOKAHEAD = 24
_HIDDEN_ZONES = (C.LOCATION_HAND, C.LOCATION_DECK, C.LOCATION_EXTRA)


class HiddenTargetError(SyncError):
    pass


class HiddenTargetDeferred(Exception):
    def __init__(self, state):
        super().__init__("public activation waits for its hidden target; native history is deferred")
        self.state = state


@dataclass(frozen=True)
class DeferredHiddenTarget:
    chaining_index: int
    source: tuple[int, int, int, int]
    origin_cursor: int
    origin_answered: int
    prompt_index: int
    server_prompt: bytes
    own_index: int
    # Opaque-root metadata; this deferral enumerates no candidate set.
    candidate_set_sha256: str = ""
    candidate_count: int = 0
    schema: str = SCHEMA
    search_ready: bool = False
    native_ready: bool = False

    def message(self):
        return Message(self.server_prompt[0], self.server_prompt[1:])


def _own_prompt(follower, packet) -> bool:
    return packet[0] in RESPONSE_REQUIRED and _prompt_player(Message(packet[0], packet[1:])) == follower.viewer


def own_prompt_index(follower, ordinal: int):
    """Packet index of the seat's ``ordinal``-th prompt (0-based) among the received packets, or None."""
    seen = 0
    for index, packet in enumerate(follower.packets):
        if _own_prompt(follower, packet):
            if seen == ordinal:
                return index
            seen += 1
    return None


def activation(follower, index: int):
    """The opponent's public CHAINING the real stream shows from ``index``, before the seat's next prompt."""
    for position in range(index, min(len(follower.packets), index + _LOOKAHEAD)):
        packet = follower.packets[position]
        if _own_prompt(follower, packet):
            return None
        if packet[0] == C.MSG_CHAINING:
            if len(packet) < 17:
                return None
            code = struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff
            controller, location, sequence = packet[5:8]
            if not code or controller == follower.viewer:
                return None
            return position, (controller, location, sequence, code)
    return None


def begin(follower, saved, index):
    """Defer from ``saved`` when the real stream shows, from ``index``, an opponent public activation.

    None leaves the failure to the ordinary follower. ``saved`` is the start of
    the batch the activation belongs to: after our own response, or before an
    opponent prompt none of whose native answers can open that activation.
    """
    found = activation(follower, index)
    if found is None or follower.deferred_action is not None:
        return None
    chaining, source = found
    if chaining == follower.hidden_target_skip:
        return None  # this activation already proved to need no hidden card
    ordinal = saved[3] + sum(1 for packet in follower.packets[saved[2]:chaining] if _own_prompt(follower, packet))
    if ordinal > len(follower.own):
        return None
    # A replay after the search went back to an earlier choice may meet the activation again with our prompts
    # after it already answered: the state then resolves at once from the packets already received.
    prompt = own_prompt_index(follower, ordinal)
    if prompt is None or prompt <= chaining:
        return None
    return DeferredHiddenTarget(chaining, source, saved[2], saved[3], prompt, bytes(follower.packets[prompt]), ordinal)


def unoffered(follower, choice):
    """Defer at an opponent IDLE whose native menu lacks the card the real stream activates next.

    The blank shadow's hand, deck and extra hold no identities, so an effect
    whose legality reads them (a search target in the deck, a cost in the hand)
    is never offered. None when the menu offers that card or nothing is activated.
    """
    if choice.origin is None or choice.prompt.msg != C.MSG_SELECT_IDLECMD:
        return None
    found = activation(follower, choice.begin)
    if found is None:
        return None
    code = found[1][3]
    if any(struct.unpack_from("<I", row, 0)[0] & 0x7fffffff == code for row in follower._idle_records(choice.prompt)[5]):
        return None
    return begin(follower, choice.origin, choice.begin)


def revealed(follower, state):
    """Every opponent hand/deck/extra card made public from the batch start until the activation's chain
    ended -- the activated card itself included -- as (packet, zone, slot, code).

    None while the chain is still open; an empty list once it ended without any. A duel won while the chain resolves
    ends it too: no later packet can show more.
    """
    opponent = 1 - follower.viewer
    reveals, hidden = [], {}
    for index in range(state.origin_cursor, len(follower.packets)):
        packet = follower.packets[index]
        msg = packet[0]
        if msg == C.MSG_MOVE and len(packet) >= 13:
            code = struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff
            controller, location, sequence = packet[5:8]
            if controller == opponent and location in _HIDDEN_ZONES:
                if code:
                    reveals.append((index, location, sequence, code))
                else:
                    hidden[tuple(packet[9:12])] = (index, location, sequence)
        elif msg == C.MSG_CONFIRM_CARDS and len(packet) >= 4:
            count = packet[3]
            if len(packet) != 4 + 7 * count:
                continue
            for offset in range(4, len(packet), 7):
                code = struct.unpack_from("<I", packet, offset)[0] & 0x7fffffff
                key = tuple(packet[offset + 4:offset + 7])
                if code and key in hidden:
                    reveals.append((*hidden.pop(key), code))
                elif code and key[:2] == (opponent, C.LOCATION_DECK):
                    # Deck cards shown where they stand (an excavation, cards revealed to pick from).
                    reveals.append((index, C.LOCATION_DECK, key[2], code))
        elif msg == C.MSG_CHAIN_END and index > state.chaining_index or msg == C.MSG_WIN:
            return sorted(reveals)
    return None


def origin_slots(follower, state, reveals, sizes):
    """Each revealed card's slot at the batch start, tracing the zones' public changes up to its move.

    ``sizes`` are the opponent's hand/deck/extra sizes at the batch start. One
    entry per reveal: None when a shuffle of its zone came first, or the card
    entered the zone during the batch; such a card played no part in what the
    batch start allowed, and the replay's own reveal fixes place it.
    """
    opponent = 1 - follower.viewer
    zones = {location: list(range(size)) for location, size in sizes.items()}
    scrambled, fresh, placed = set(), -1, []
    wanted = {index: (location, sequence, code) for index, location, sequence, code in reveals}
    for index in range(state.origin_cursor, max(wanted) + 1):
        packet = follower.packets[index]
        msg = packet[0]
        if index in wanted:
            location, sequence, code = wanted[index]
            slots = zones[location]
            if location in scrambled or not 0 <= sequence < len(slots) or slots[sequence] < 0:
                placed.append(None)
            else:
                placed.append((opponent, location, slots[sequence], code))
        if msg == C.MSG_MOVE and len(packet) >= 13:
            before, after = tuple(packet[5:8]), tuple(packet[9:12])
            if before[0] == opponent and before[1] in zones and before[1] not in scrambled:
                if before[2] >= len(zones[before[1]]):
                    raise HiddenTargetError("a hidden zone lost track of its public moves")
                del zones[before[1]][before[2]]
            if after[0] == opponent and after[1] in zones:
                zones[after[1]].insert(min(after[2], len(zones[after[1]])), fresh)
        elif msg == C.MSG_DRAW and len(packet) >= 3 and packet[1] == opponent:
            for _ in range(packet[2]):
                if zones[C.LOCATION_DECK]:
                    zones[C.LOCATION_DECK].pop()
                zones[C.LOCATION_HAND].append(fresh)
        elif msg in (C.MSG_SHUFFLE_HAND, C.MSG_SHUFFLE_DECK, C.MSG_SHUFFLE_EXTRA) and len(packet) >= 2 \
                and packet[1] == opponent:
            scrambled.add({C.MSG_SHUFFLE_HAND: C.LOCATION_HAND, C.MSG_SHUFFLE_DECK: C.LOCATION_DECK,
                           C.MSG_SHUFFLE_EXTRA: C.LOCATION_EXTRA}[msg])
    return placed


def step(follower):
    """While deferred (the searching follower): the pending prompt to answer opaquely, or None once native history
    resumed. A single pass never defers: it forces the activation at once (:func:`construct`)."""
    state = follower.deferred_action
    if len(follower.own) <= state.own_index:
        return state.message()
    target = revealed(follower, state)
    if target is None:
        ordinal = len(follower.own)
        prompt = own_prompt_index(follower, ordinal)
        if prompt is None:
            raise HiddenTargetError("no received own prompt while the hidden target is unresolved")
        follower.deferred_action = replace(state, prompt_index=prompt, server_prompt=bytes(follower.packets[prompt]),
                                           own_index=ordinal)
        return follower.deferred_action.message()
    origin = follower.hidden_target_origin
    follower.deferred_action = follower.hidden_target_origin = None
    follower.parked = None
    _commit_evidence(follower, state, origin)
    return None


@dataclass(frozen=True)
class ForcedActivation:
    """An opponent activation the stream shows and the local duel did not write, named as the core forces it
    (``duel_force_activation``): the controller, the card's code and the effect's description."""

    chaining_index: int
    controller: int
    code: int
    description: int


def forced_activation(follower, index):
    """The opponent's activation the received packets show from ``index`` on, before the seat's next prompt."""
    found = activation(follower, index)
    if found is None:
        return None
    chaining, (controller, _location, _sequence, code) = found
    return ForcedActivation(chaining, controller, code, struct.unpack_from("<I", follower.packets[chaining], 12)[0])


class Reconstructed(Exception):
    """The single pass went back to a construction point: its batch loop starts again from the state loaded."""


@dataclass(frozen=True)
class Construction:
    """A single pass's forced activation (plan 5.11), tried at the batch starts still held, oldest first: the
    follower replays from one with the activation forced; a divergence before the activation is written takes the
    next. The mark holds until the activation is chained, so from an older batch start every window from there on
    may take it; which one did the readings tell (a chain window keeps passing it on as an open decision).

    Every replay starts from the follower's bookkeeping when the construction began -- its opponent choices,
    answer states and category facts -- cut back to the batch start replayed from: a replay given up leaves
    nothing of its own for the next. Their snapshots are held here until the construction ends."""

    state: ForcedActivation
    #: the batch starts not tried yet, oldest first, each a follower state owned here
    candidates: tuple
    #: the follower's opponent choices, answer states and category facts when the construction began
    choices: tuple = ()
    answer_states: tuple = ()
    facts: tuple = ()
    #: the category fact the activation's legality gives, read at the newest batch start: ``(controller, zone,
    #: slot, candidates, zones, batch start packet)``, or None
    fact: tuple = None
    #: the cursor of the batch start replayed from now
    cursor: int = -1

    def held(self) -> dict:
        """The snapshots of the bookkeeping the construction began with, by handle."""
        states = [state for choice in self.choices for state in (choice.saved, choice.origin)]
        states += self.answer_states
        return {state[0]: state for state in states if state is not None}


def construct(follower):
    """Replay from the pending construction's next batch start with the activation forced. HiddenTargetError when no
    batch start is left: forced from each, the replay did not write the activation."""
    pending = follower.construction
    forced = pending.state
    if not pending.candidates:
        drop_construction(follower)
        raise HiddenTargetError("no batch start from packet %d on makes the local duel write the opponent's "
                                "activation at packet %d, forced" % (pending.cursor, forced.chaining_index))
    origin, rest = pending.candidates[0], pending.candidates[1:]
    follower.construction = replace(pending, candidates=rest, cursor=origin[2])
    _replay_from(follower, origin)
    follower._force_activation(forced)
    if pending.fact is not None:
        controller, location, _slot, codes, zones, since = pending.fact
        follower._category_fact(controller, location, codes, forced.chaining_index, zones, since)


def drop_construction(follower):
    """The construction ends (its activation was written, or no batch start is left): the batch starts not tried
    and the snapshots of its starting bookkeeping the follower no longer holds are freed."""
    from .client_root import _saved_handles
    pending, follower.construction = follower.construction, None
    for origin in pending.candidates:
        follower._free(origin)
    held = _saved_handles(follower)
    for handle, state in pending.held().items():
        if handle not in held:
            follower._free(state)


def _replay_from(follower, origin):
    """Follow again from the batch start ``origin`` (consumed) with the bookkeeping the construction began with, cut
    back to it: what a replay given up read -- its opponent choices, category facts and answer states -- goes."""
    pending = follower.construction
    follower._drop_origin()
    started = {id(choice) for choice in pending.choices}
    for choice in follower.choices:
        if id(choice) not in started:
            follower._release(choice)
    held = pending.held()
    for state in follower.answer_states:
        if state[0] not in held:
            follower._free(state)
    follower.choices = [choice for choice in pending.choices if choice.begin < origin[2]]
    follower.answer_states = [state for state in pending.answer_states if state[3] <= origin[3]]
    follower.category_constraints = [fact for fact in pending.facts if fact[4] < origin[2]]
    follower._load(origin)
    follower.parked = None
    follower._free(origin)


def _commit_evidence(follower, state, origin):
    """The searching follower: replay from ``origin`` (consumed) with the chain's public cards in their slots there,
    and a pool card assumed where those do not make the activation legal; an activation whose evidence changes
    nothing is followed again the ordinary way and not deferred a second time."""
    target = revealed(follower, state)
    follower._load(origin)
    opponent = 1 - follower.viewer
    sizes = {location: follower.core.query_field_count(follower.local.pduel, opponent, location)
             for location in _HIDDEN_ZONES}
    slots = origin_slots(follower, state, target, sizes) if target else []
    placed = [coordinate for coordinate in slots if coordinate is not None]
    hypothesis = _category(follower, origin, placed, state.source[3],
                           state.chaining_index) if follower.hidden_target_pool else None
    from .client_shadow import card_code
    unchanged = hypothesis is None and all(
        card_code(follower.core, follower.local.pduel, *coordinate[:3]) == coordinate[3] for coordinate in placed)
    follower.hidden_target_skip = state.chaining_index if unchanged else None
    # Opponent prompts met inside the deferred region belong to the discarded attempt.
    while follower.choices and follower.choices[-1].begin >= state.origin_cursor:
        follower._release(follower.choices.pop())
    follower._free(origin)
    # The hidden cards the chain made public, in their slots at the batch start, before it runs again; each a card
    # of its own.
    targets = frozenset(coordinate[:3] for coordinate in placed)
    for coordinate in placed:
        try:
            follower._place(*coordinate, keep=targets - {coordinate[:3]})
        except RevealConflict:
            # The slot holds another known card: the local duel reached it another way. The batch is replayed
            # without this card, and its divergence goes back to the search over earlier answers.
            continue
    if hypothesis is not None:
        controller, location, slot, candidates, _zones = hypothesis
        follower._hypothesize(controller, location, slot, candidates[0])


def _offers(prompt, code, description=None) -> bool:
    """Whether an opponent prompt offers an activation of ``code`` with this effect ``description`` (None: any): an
    idle or battle command's activations, a chain window's links, or the one optional effect a yes/no question asks
    about (by the card alone: the question carries the core's own description, not the effect's)."""
    body = prompt.payload
    fits = lambda raw, at: struct.unpack_from("<I", raw, at)[0] & 0x7fffffff == code \
        and (description is None or struct.unpack_from("<I", raw, at + 7)[0] == description)
    if prompt.msg == C.MSG_SELECT_IDLECMD:
        from .client_shadow import BlankClientSync
        return any(fits(row, 0) for row in BlankClientSync._idle_records(prompt)[5])
    if prompt.msg == C.MSG_SELECT_BATTLECMD:
        return any(fits(body, 2 + 11 * i) for i in range(body[1]))
    if prompt.msg == C.MSG_SELECT_CHAIN:
        # Only a link: an entry of another kind (a field-only operation, a reset such as lent control returning)
        # names the card without activating it.
        return any(body[11 + 14 * i] == 0 and struct.unpack_from("<I", body, 11 + 14 * i + 2)[0] & 0x7fffffff == code
                   and (description is None or struct.unpack_from("<I", body, 11 + 14 * i + 10)[0] == description)
                   for i in range(body[1]))
    if prompt.msg == C.MSG_SELECT_EFFECTYN:
        return struct.unpack_from("<I", body, 1)[0] & 0x7fffffff == code
    return False


def _offered(follower, origin, cards, code, description=None):
    """From ``origin`` with these (controller, location, slot, code) cards: whether the opponent's next prompt
    offers it an activation of ``code``, or None when no opponent prompt is reached. The seat's own prompts on
    the way take its recorded answers: the core checks both players' optional triggers when it builds the turn
    player's choice, so the opponent's window may come after the seat's answer. A proof run only; ``origin`` is
    loaded after."""
    from .client_origin_receipt import boundary
    from .client_shadow import HydrationError
    owner = boundary(follower).owner
    audit = len(owner.audit)
    follower._load(origin)
    try:
        for card in cards:
            # A proof trial is still a native mutation. Keep it in the
            # transient journal as an explicitly unvalidated hypothesis;
            # the final origin rollback removes it with all trial batches.
            # Raw hydrate here breaks continuity of later recorded batches.
            follower._hypothesize(*card)
        prompt, first = None, True
        while prompt is None and not follower.local.finished:
            # Batches run on to the next prompt, as the follower's own loop does (a trigger's window, say, takes
            # more than one core batch).
            fixed = follower._save()
            try:
                prompt = follower._fixed_batch(fixed, receipt_origin=origin if first else None)
            finally:
                follower._free(fixed)
            first = False
            if prompt is not None and _prompt_player(prompt) == follower.viewer:
                if follower.answered >= len(follower.own):
                    return None
                follower._respond(prompt, follower.own[follower.answered])
                follower.answered += 1
                prompt = None
    except _Divergence:
        return False  # the batch went elsewhere before any prompt: no window offered the activation
    except (SyncError, HydrationError, _Frontier):
        return None
    finally:
        follower._load(origin)
        del owner.audit[audit:]
    if prompt is None:
        return None
    return _offers(prompt, code, description)


def _category(follower, origin, placed, code, chaining):
    """When the public cards do not make the opponent's window offer ``code``: ``(opponent, zone, blank slot,
    candidates, zones)``, the pool cards any one of which, in that slot, makes it offer it, and every hidden zone
    with such candidates as ``((zone, candidates), ...)``.

    The activation was legal, so one of the opponent's hidden zones (its deck, its extra deck, its hand) held such
    a card; which one no packet shows, and it touched nothing public (the activation was negated, say). The
    candidates are that public fact: the zone held at least one of them. When several zones have candidates (a
    Tuner summoned "from the hand or Deck"), the fact is that one of those zones did, and no packet tells which.
    The first such zone holds the proxy: the proxy lives until the next own prompt's alignment blanks it, and the
    replay checks every packet up to there, so a zone the stream contradicts is a divergence, never a silent pick.
    Each trial holds its candidate as that fact would (the seat's own prompts on the way keep it).
    """
    link = follower.packets[chaining]
    description = struct.unpack_from("<I", link, 12)[0] if link[0] == C.MSG_CHAINING and len(link) >= 16 else None
    if _offered(follower, origin, placed, code, description) is not False:
        return None
    from .client_shadow import _EXTRA_TYPES, BLANK_CODES, card_code
    opponent, pduel = 1 - follower.viewer, follower.local.pduel
    cards = follower.core.card_pool().cards
    taken = {(location, slot) for _player, location, slot, _code in placed}
    found = []
    for location in (C.LOCATION_DECK, C.LOCATION_EXTRA, C.LOCATION_HAND):
        count = follower.core.query_field_count(pduel, opponent, location)
        blank = next((slot for slot in range(count) if (location, slot) not in taken
                      and card_code(follower.core, pduel, opponent, location, slot) in BLANK_CODES), None)
        if blank is None:
            continue
        extra = location == C.LOCATION_EXTRA
        from .client_public_recipe import permits_native
        candidates = tuple(candidate for candidate in follower.hidden_target_pool
                           if candidate in cards and bool(cards[candidate].type & _EXTRA_TYPES) == extra
                           and permits_native(follower, opponent, location, blank, candidate)
                           and _trial(follower, origin, placed, (opponent, location, blank, candidate), code, chaining,
                                      description))
        if candidates:
            found.append((opponent, location, blank, candidates))
    if not found:
        return None
    return (*found[0], tuple((location, candidates) for _player, location, _slot, candidates in found))


def root_claims(sync, viewer: int) -> tuple:
    """The follower's category facts about the opponent's hidden zones still true at the root ``sync["cursor"]``, as
    sampler claims (:class:`~mirrorforce.common.stage_a_joint_belief_runtime.ZoneClaim`).

    A fact holds from its activation's packet on, and speaks of the zones as they were from the batch start it was
    read at. Moves between the fact's own zones keep it (a draw keeps "the deck or the hand"); a card that leaves
    them unseen, or one of the candidates leaving them, from that batch start on, may have been the one it speaks
    of, and ends it (the activated card itself leaves the hand before its chain link shows). So does the deck
    changing places with the graveyard. A fact of an activation after the root (only a replay from an earlier batch
    start knows one) says nothing about the root."""
    from .stage_a_joint_belief_runtime import ZoneClaim
    opponent, packets, cursor = 1 - viewer, sync["packets"], sync["cursor"]
    claims = []
    for controller, _location, _codes, _count, chaining, zones, since in sync["category_constraints"]:
        if controller != opponent or chaining >= cursor:
            continue
        places = frozenset(location for location, _candidates in zones)
        codes = frozenset(code for _location, candidates in zones for code in candidates)
        alive = True
        for index in range(since, cursor):
            packet = packets[index]
            if packet[0] == C.MSG_MOVE and len(packet) >= 13 and packet[5] == opponent and packet[6] in places:
                code = struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff
                stays = packet[9] == opponent and packet[10] in places
                alive = stays or (code != 0 and code not in codes)
            elif packet[0] == C.MSG_DRAW and len(packet) >= 3 and packet[1] == opponent and C.LOCATION_DECK in places \
                    and C.LOCATION_HAND not in places:
                drawn = [struct.unpack_from("<I", packet, 3 + 4 * i)[0] & 0x7fffffff for i in range(packet[2])
                         if 3 + 4 * i + 4 <= len(packet)]
                alive = len(drawn) == packet[2] and all(code and code not in codes for code in drawn)
            elif packet[0] == C.MSG_SWAP_GRAVE_DECK and len(packet) >= 2 and packet[1] == opponent \
                    and C.LOCATION_DECK in places:
                alive = False
            if not alive:
                break
        if alive:
            claims.append(ZoneClaim(tuple(sorted(places)), codes))
    return tuple(claims)


def _trial(follower, origin, placed, card, code, chaining, description=None):
    """Whether ``card`` in its blank slot, held there as a category fact of the activation at ``chaining``, makes
    the opponent's window offer that effect of ``code``."""
    follower.category_constraints.append((card[0], card[1], (card[3],), 1, chaining, ((card[1], (card[3],)),),
                                          origin[2]))
    try:
        return _offered(follower, origin, (*placed, card), code, description)
    finally:
        follower.category_constraints.pop()
