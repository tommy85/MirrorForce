"""Shadow board: the duel state rebuilt from the message stream alone.

A network client has no duel engine.  Everything it knows comes from
``STOC_GAME_MSG``, which is what makes this the same problem the product's
shadow duel has to solve.

The load-bearing message is ``MSG_UPDATE_DATA``: after a board change the host
re-queries a zone and forwards the result -- but only for **three** zones.
``RefreshMzone``, ``RefreshSzone`` and ``RefreshHand`` run from the main
message block; ``RefreshGrave`` is reachable only from ``MSG_SWAP_GRAVE_DECK``,
``RefreshExtra`` only from ``MSG_SHUFFLE_EXTRA`` (``single_duel.cpp:882`` and
``:871``), and the banished pile has no refresh at all.  So the graveyard, the
banished pile and the extra deck are not cross-checked by a refresh: the
message stream is their only source, and a card leaving one of them renumbers
every card behind it whether we follow the move or not.

The payload of a refresh is

    uint8  MSG_UPDATE_DATA
    uint8  player
    uint8  location
    then, per card slot, one self-delimiting segment:
        int32  segment length        (4 == empty slot, nothing follows)
        uint32 present_flags         (QUERY_* bits actually included)
        ...    the fields, in the order card::get_infos writes them

Two things make this workable and are worth stating plainly:

* the segments are length-prefixed, so the *number of cards in the zone* falls
  out of parsing -- unlike the reference GUI client, which walks its own list
  and therefore needs separate move tracking just to know the zone size;
* the host queries with ``use_cache=1``, so a segment carries only the fields
  that changed since the core last queried that card.  The core clears a card's
  cache whenever its ``info_location`` changes (``card.cpp``, QUERY_POSITION
  branch), so a card that moved always arrives complete -- merging by slot with
  an ``info_location`` check is enough, and that is what :meth:`_merge` does.

Not reconstructible from the stream, and tracked separately here:

* the *contents* of either deck.  Ours is known from the list we submitted
  minus everything that has been seen elsewhere (the "remaining multiset,
  unknown order" state the project already settled on for observations); the
  opponent's is never known, only its size.
* zones the host does not refresh (``LOCATION_DECK``), whose size is tracked
  from ``MSG_START`` plus the moves that cross the deck boundary.
* the **graveyard, the banished pile and the extra deck**, maintained from
  ``MSG_MOVE`` with the reference client's own semantics (``client_field.cpp``
  ``AddCard`` / ``RemoveCard`` / ``ResetSequence``): list zones erase and
  renumber, field zones are fixed slots, and a card returning face-down to the
  extra deck goes in front of the face-up pendulum block at its end.  The
  opponent's extra deck is a *count* rather than a zone, because
  ``RefreshExtra`` is sent to its owner alone.
* **our deck cards that leave anonymously**.  A card of our deck that goes to
  the opponent's hand, or face down to the opponent's field, reaches us as
  ``MSG_MOVE`` with its code blanked (the host keeps the code for the new
  controller, ``single_duel.cpp``). Such a departure is named in one of two
  ways. By elimination: Jack-in-the-Hand (51697825) has us select three deck
  cards, the opponent take one unseen and then asks us to pick from the other
  two; the departed card is the group we selected minus the group we are
  offered (``_resolve_group``; group, departures and offer in one chain-link
  resolution). By its new place: a card that went to a field zone is tracked
  there, and once the ledger proves the code at that slot (a confirmation, a
  public move), that is the departed card (``_name_departures``). A departure
  named neither way stays pending (``unresolved_own_deck_departures``) and the
  strict own-deck count check of the observation refuses the decision.
* the **hand after a draw**.  ``MSG_DRAW`` is not followed by a hand refresh,
  so a hand rebuilt from refreshes alone is a card short until some unrelated
  message happens to trigger one.
* the **position of a face-down card**.  The host blanks the whole query
  segment of a face-down card -- position included -- so the face-down
  *flavour* survives only in ``MSG_MOVE`` and ``MSG_POS_CHANGE``.
* **xyz materials, counters, equip links and persistent card targets.**  The
  host's refresh flags are
  ``0x881fff`` for the monster zone and ``0x681fff`` for the rest
  (``single_duel.h:37-42``); neither contains ``QUERY_OVERLAY_CARD``
  (0x10000) or ``QUERY_COUNTERS`` (0x20000), so a query segment never carries
  either.  They come from the message stream instead -- ``MSG_MOVE`` in and
  out of ``LOCATION_OVERLAY`` for materials, ``MSG_ADD_COUNTER`` /
  ``MSG_REMOVE_COUNTER`` for counters, ``MSG_EQUIP`` for equip links, and
  ``MSG_CARD_TARGET`` / ``MSG_CANCEL_TARGET`` for effect-target links --
  exactly as the reference GUI client does it (``duelclient.cpp:2750-2845``
  and ``:3436-3480``).  The standard protocol has no active ``MSG_UNEQUIP``;
  movement retirement is exact, while a relation-only unequip cannot be
  reconstructed until the server exports it.

  Materials are load-bearing for observations, not decoration:
  ``get_cards_in_location`` emits one row per material *before* the row of the
  monster carrying them (``ygopro.h``, the ``n_xyz`` loop), so losing them
  shifts every later row and silently corrupts the card index every action
  refers to.

  Both are kept in board-level dicts keyed by ``(controller, location,
  sequence)`` rather than on :class:`ShadowCard`.  A refresh rebuilds the card
  objects of a zone whenever ``info_location`` changes, so anything held on
  the object itself would be dropped by a move that ``MSG_UPDATE_DATA``
  reports; a slot key survives that, and is reconciled against the refreshed
  zone afterwards (:meth:`_reconcile_slots`), which makes the state
  self-healing rather than merely hopeful.
"""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass, field

from . import constants as C
from .resolution import ChainResolution
from .disclosure import DisclosureLedger

__all__ = ["ShadowCard", "ShadowBoard", "parse_query_segments"]

# QUERY_* bits in the order card::get_infos writes them, with the number of
# uint32 words each contributes.  Variable-length ones are handled explicitly.
_SCALAR_FIELDS = (
    (C.QUERY_CODE, "code"),
    (C.QUERY_POSITION, "info_location"),
    (C.QUERY_ALIAS, "alias"),
    (C.QUERY_TYPE, "type"),
    (C.QUERY_LEVEL, "level"),
    (C.QUERY_RANK, "rank"),
    (C.QUERY_ATTRIBUTE, "attribute"),
    (C.QUERY_RACE, "race"),
    (C.QUERY_ATTACK, "attack"),
    (C.QUERY_DEFENSE, "defense"),
    (C.QUERY_BASE_ATTACK, "base_attack"),
    (C.QUERY_BASE_DEFENSE, "base_defense"),
    (C.QUERY_REASON, "reason"),
    (C.QUERY_REASON_CARD, "reason_card"),
    (C.QUERY_EQUIP_CARD, "equip_card"),
)

_SIGNED = {"attack", "defense", "base_attack", "base_defense"}


def parse_query_segments(data: bytes) -> list[dict | None]:
    """Split one query buffer into per-slot field dicts (``None`` = empty)."""
    out: list[dict | None] = []
    pos = 0
    n = len(data)
    while pos + 4 <= n:
        (seg_len,) = struct.unpack_from("<i", data, pos)
        if seg_len < 4 or pos + seg_len > n:
            break
        if seg_len <= C.LEN_EMPTY:
            out.append(None)
            pos += seg_len
            continue
        (flags,) = struct.unpack_from("<I", data, pos + 4)
        if seg_len == C.LEN_HEADER:
            # header only: every requested field was cache-suppressed
            out.append({"_flags": 0, "_nochange": True})
            pos += seg_len
            continue
        if flags == 0:
            # the host memsets the whole segment body for a card we are not
            # allowed to see (single_duel.cpp RefreshMzone / RefreshHand), so
            # this is "an unknown card occupies this slot", not "no change".
            out.append({"_flags": 0, "_hidden": True})
            pos += seg_len
            continue
        p = pos + 8
        fields: dict = {"_flags": flags}
        for bit, name in _SCALAR_FIELDS:
            if not (flags & bit):
                continue
            fmt = "<i" if name in _SIGNED else "<I"
            (value,) = struct.unpack_from(fmt, data, p)
            fields[name] = value
            p += 4
        if flags & C.QUERY_TARGET_CARD:
            (count,) = struct.unpack_from("<i", data, p)
            p += 4
            fields["targets"] = list(struct.unpack_from(f"<{count}I", data, p))
            p += 4 * count
        if flags & C.QUERY_OVERLAY_CARD:
            (count,) = struct.unpack_from("<i", data, p)
            p += 4
            fields["overlay"] = list(struct.unpack_from(f"<{count}I", data, p))
            p += 4 * count
        if flags & C.QUERY_COUNTERS:
            (count,) = struct.unpack_from("<i", data, p)
            p += 4
            raw = struct.unpack_from(f"<{count}I", data, p)
            fields["counters"] = {v & 0xFFFF: v >> 16 for v in raw}
            p += 4 * count
        if flags & C.QUERY_OWNER:
            (fields["owner"],) = struct.unpack_from("<i", data, p)
            p += 4
        if flags & C.QUERY_STATUS:
            (fields["status"],) = struct.unpack_from("<I", data, p)
            p += 4
        if flags & C.QUERY_LSCALE:
            (fields["lscale"],) = struct.unpack_from("<I", data, p)
            p += 4
        if flags & C.QUERY_RSCALE:
            (fields["rscale"],) = struct.unpack_from("<I", data, p)
            p += 4
        if flags & C.QUERY_LINK:
            fields["link"], fields["link_marker"] = struct.unpack_from("<II", data, p)
            p += 8
        out.append(fields)
        pos += seg_len
    return out


@dataclass
class ShadowCard:
    """What the stream lets us know about one card."""

    code: int = 0
    alias: int = 0
    type: int = 0
    level: int = 0
    rank: int = 0
    attribute: int = 0
    race: int = 0
    attack: int = 0
    defense: int = 0
    # The host sends QUERY_BASE_ATTACK / QUERY_BASE_DEFENSE and
    # ``parse_query_segments`` already decodes them; they are the engine's
    # runtime base stats, which are not the printed ones -- the cdb stores -2
    # for a "?" ATK and a Link monster's markers in its def column, and a card
    # off the field reports zero.  The corpus captures the engine's values, so
    # a bridge that substitutes the database disagrees with training on every
    # CARD token of those cards.
    base_attack: int = 0
    base_defense: int = 0
    lscale: int = 0
    rscale: int = 0
    link: int = 0
    link_marker: int = 0
    status: int = 0
    controller: int = 0
    location: int = 0
    sequence: int = 0
    position: int = 0
    overlay: list = field(default_factory=list)
    counters: dict = field(default_factory=dict)
    equip_card: int = 0
    targets: list = field(default_factory=list)
    # QUERY_OWNER is absent from the stock host refresh masks.  We still keep
    # it here because a client can infer ownership from the card's first public
    # zone and then preserve it across control changes.
    owner: int = -1
    info_location: int = 0
    hidden: bool = False  # the host blanked the whole segment for us
    #: True once a query segment has filled this card's stats in.  A card the
    #: stream only ever *named* -- one just drawn, or one followed through
    #: MSG_MOVE into a zone the host does not refresh -- has zeroes where the
    #: engine would report the card's printed values, so the encoder has to
    #: know to read the database instead of trusting the zeroes.
    queried: bool = False

    @property
    def known(self) -> bool:
        """False for an opponent card whose identity the host withheld."""
        return not self.hidden and self.code != 0

    @property
    def face_down(self) -> bool:
        return bool(self.position & C.POS_FACEDOWN)

    def apply(self, fields: dict) -> None:
        self.queried = True
        if "code" in fields:
            # QUERY_CODE with code 0 is the other "you may not see this" shape
            # (MSG_UPDATE_CARD's censored 16-byte segment).  A later visible
            # refresh must also clear the flag; otherwise a named card remains
            # spuriously hidden forever.
            self.hidden = int(fields.get("code") or 0) == 0
        for name in (
            "code",
            "alias",
            "type",
            "level",
            "rank",
            "attribute",
            "race",
            "attack",
            "defense",
            "base_attack",
            "base_defense",
            "lscale",
            "rscale",
            "link",
            "link_marker",
            "status",
            "equip_card",
            "owner",
        ):
            if name in fields:
                setattr(self, name, fields[name])
        if "overlay" in fields:
            self.overlay = fields["overlay"]
        if "counters" in fields:
            self.counters = fields["counters"]
        if "targets" in fields:
            self.targets = fields["targets"]
        if "info_location" in fields:
            loc = fields["info_location"]
            self.info_location = loc
            self.controller = loc & 0xFF
            self.location = (loc >> 8) & 0xFF
            self.sequence = (loc >> 16) & 0xFF
            # EFFECT_REVEAL_ONFIELD can OR POS_REVEAL into the position byte
            self.position = ((loc >> 24) & 0xFF) & 0x0F
            if self.owner not in (0, 1):
                self.owner = self.controller


#: fixed-slot zones: a move writes the slot, it does not shift the list
_SLOT_ZONES = frozenset({C.LOCATION_MZONE, C.LOCATION_SZONE})

#: list zones: a move appends or erases and everything after it renumbers
_LIST_ZONES = frozenset(
    {C.LOCATION_HAND, C.LOCATION_GRAVE, C.LOCATION_REMOVED, C.LOCATION_EXTRA}
)

#: the three messages that publish a card's identity by showing it
_CONFIRM_MSGS = (C.MSG_CONFIRM_CARDS, C.MSG_CONFIRM_DECKTOP, C.MSG_CONFIRM_EXTRATOP)
#: Messages ``ShadowBoard.apply`` hands to a state handler; every other message only counts as unknown.
#: Refreshes are compared before and after instead, because most leave the board unchanged.
_STATE_MESSAGES = frozenset((
    C.MSG_START, C.MSG_DRAW, C.MSG_DECK_TOP, C.MSG_REVERSE_DECK, C.MSG_SHUFFLE_SET_CARD, C.MSG_MOVE,
    C.MSG_SHUFFLE_EXTRA, C.MSG_SHUFFLE_DECK, C.MSG_SHUFFLE_HAND, C.MSG_SWAP_GRAVE_DECK, C.MSG_ADD_COUNTER,
    C.MSG_REMOVE_COUNTER, C.MSG_SWAP, C.MSG_POS_CHANGE, C.MSG_EQUIP, C.MSG_UNEQUIP, C.MSG_CARD_TARGET,
    C.MSG_CANCEL_TARGET, C.MSG_CHAINING, C.MSG_CHAIN_SOLVING, C.MSG_CHAIN_SOLVED, C.MSG_CHAIN_END,
    *_CONFIRM_MSGS, C.MSG_NEW_PHASE, C.MSG_NEW_TURN,
))
_REFRESH_MESSAGES = frozenset((C.MSG_UPDATE_DATA, C.MSG_UPDATE_CARD))
#: Our own card prompts, read for the own-deck group elimination (``ShadowBoard._resolve_group``). They change
#: no state unless they resolve an anonymous departure from our deck.
_PROMPT_MESSAGES = frozenset((C.MSG_SELECT_CARD, C.MSG_SELECT_UNSELECT_CARD))
_CANCEL_RESPONSE = (-1).to_bytes(4, "little", signed=True)


def _deck_sort_marker(previous: int, current: int) -> bool:
    """The anonymous per-card ``MSG_MOVE`` that ``PROCESSOR_SORT_DECK`` writes after a secret reorder.

    Source and destination are the same deck slot (``processor.cpp``, ``PROCESSOR_SORT_DECK``); the
    disclosure ledger's same test (``_move_for_viewer``) drops the deck's coordinates on it.
    """
    source, destination = int(previous), int(current)
    return bool(destination) and (source >> 8) & 0xFF == C.LOCATION_DECK \
        and source & 0xFFFFFF == destination & 0xFFFFFF

_REFRESHED = (
    C.LOCATION_MZONE,
    C.LOCATION_SZONE,
    C.LOCATION_HAND,
    C.LOCATION_GRAVE,
    C.LOCATION_REMOVED,
    C.LOCATION_EXTRA,
)


class ShadowBoard:
    """Duel state maintained from ``STOC_GAME_MSG`` only."""

    def __init__(self) -> None:
        self.our_player = 0
        self.turn_player = 0
        self.phase = 0
        self.zones: dict[tuple[int, int], list] = {
            (p, loc): [] for p in (0, 1) for loc in _REFRESHED
        }
        self.deck_count = [0, 0]
        self.extra_count = [0, 0]
        # our own deck: the submitted list minus everything seen elsewhere
        self.our_deck_start: Counter = Counter()
        self.our_extra_start: tuple[int, ...] = ()
        self.seen_out_of_deck: Counter = Counter()
        self._reset_group_elimination()
        self.updates = 0
        self.unknown_messages: Counter = Counter()
        # xyz materials, counters and face-down positions, keyed by
        # (controller, location, sequence); see the module docstring for why
        # they do not live on ShadowCard
        self.materials: dict[tuple[int, int, int], list] = {}
        self.counters: dict[tuple[int, int, int], dict] = {}
        self.positions: dict[tuple[int, int, int], int] = {}
        # Public relation state reconstructed from MSG_EQUIP and
        # MSG_CARD_TARGET/CANCEL_TARGET.  The stock host refresh masks omit
        # QUERY_EQUIP_CARD/TARGET_CARD, so the wire stream is the only online
        # source.  Forward edges are sufficient; core's reverse sets are
        # exact inverses.
        self.equip_targets: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        self.card_targets: dict[tuple[int, int, int], set[tuple[int, int, int]]] = {}
        #: Cards whose identity the duel has made public, by instance
        #: The public identity ledger. **It shares the same ``DisclosureLedger`` implementation with the corpus side
        #: (``worldmodel/engine.py``)**: the visibility predicate is shared, and two copies would
        #: drift apart silently. It books multiset counts by ``(controller, zone, code)``, not instance coordinates:
        #: ``field::remove_card`` silently reorders sequences without sending ``MSG_MOVE``, so coordinate keys
        #: stop matching after the opponent plays any hand card, anonymizing disclosed cards again.
        #: Nor by code alone: that would expose same-named copies in the deck too.
        self.disclosure = DisclosureLedger()
        #: face-up pendulum cards sit at the end of the extra deck and the
        #: insertion point of a face-down one depends on how many there are
        self.extra_faceup = [0, 0]
        # cross_check bookkeeping
        self.checks = 0
        self.mismatches = 0
        self.learned = 0
        self.slot_problems: list[str] = []
        #: Increases whenever state a snapshot can read may have changed:
        #: start, a state message, a refresh that changed something, or an
        #: identity learned from a prompt. Equal revisions mean equal state.
        self.revision = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self, our_player: int, main: list[int], extra: list[int]) -> None:
        self.revision += 1
        self.our_player = our_player
        self.our_deck_start = Counter(main)
        # single_duel.cpp loads both deck containers with ``rbegin()`` and the
        # core appends each card.  Shadow sequence must follow that exact ABI,
        # not the .ydk presentation order.
        self.our_extra_start = tuple(int(code) for code in reversed(extra))
        self.seen_out_of_deck.clear()
        self.deck_count = [0, 0]
        self.extra_count = [0, 0]
        self.deck_count[our_player] = len(main)
        self.extra_count[our_player] = len(extra)
        for key in self.zones:
            self.zones[key] = []
        # A player supplies and may inspect their own Extra Deck.  Seed it now
        # so MSG_START-before-initial-refresh and compatibility hosts with a
        # censored refresh cannot erase that entitled identity.  Sequence is a
        # protocol address derived only from the host's rbegin ABI here (and
        # from owner shuffle payloads later); it is never guessed.
        self.zones[(our_player, C.LOCATION_EXTRA)] = [
            ShadowCard(
                code=code,
                hidden=False,
                controller=our_player,
                owner=our_player,
                location=C.LOCATION_EXTRA,
                sequence=sequence,
                position=C.POS_FACEDOWN_DEFENSE,
            )
            for sequence, code in enumerate(self.our_extra_start)
        ]
        self.materials.clear()
        self.counters.clear()
        self.positions.clear()
        self.equip_targets.clear()
        self.card_targets.clear()
        self.disclosure.clear()
        self.extra_faceup = [0, 0]
        self._reset_group_elimination()

    def _reset_group_elimination(self) -> None:
        #: the chain-link resolution the stream is in (0 outside one): the group elimination's window
        self.resolution = ChainResolution()
        #: the last own card prompt, until its response: (msg, listed cards as (code, controller, location))
        self._prompt = None
        #: our deck cards this player selected in the window and not yet seen leaving the deck, by code
        self._own_group: Counter = Counter()
        #: our deck cards that left anonymously and are not named yet: each with its chain-link resolution
        #: (0 outside one) and its current field slot (None where it cannot be followed: a list zone)
        self._departures: list[dict] = []
        #: departures named, by elimination or by their new place
        self.resolved_departures = 0

    def zone(self, player: int, location: int) -> list:
        return self.zones.get((player, location), [])

    def our_remaining_deck(self) -> Counter:
        """Multiset of cards still in our deck; order is genuinely unknown."""
        return self.our_deck_start - self.seen_out_of_deck

    # -- message application ----------------------------------------------

    def _refresh_state(self, key: tuple[int, int]) -> tuple:
        """What a refresh of one zone may change: its cards and the slot-keyed tables, copied for comparison."""
        cards = self.zones.get(key)
        zone = None if cards is None else [
            None if card is None else (id(card), {name: list(value) if isinstance(value, list) else value
                                                  for name, value in vars(card).items()})
            for card in cards]
        return (zone, {slot: list(value) for slot, value in self.materials.items()},
                {slot: dict(value) for slot, value in self.counters.items()}, dict(self.positions),
                dict(self.equip_targets), {slot: set(value) for slot, value in self.card_targets.items()})

    def apply(self, msg: int, body: bytes) -> None:
        if msg in _PROMPT_MESSAGES:
            self._observe_prompt(msg, body)
            return
        # The response to an own card prompt is sent before any further message arrives.
        self._prompt = None
        if msg in _REFRESH_MESSAGES:
            key = (body[0], body[1] & 0x7F) if len(body) >= 2 else None
            before = self._refresh_state(key)
            try:
                if msg == C.MSG_UPDATE_DATA:
                    self._update_data(body)
                else:
                    self._update_card(body)
            finally:
                if self._refresh_state(key) != before:
                    self.revision += 1
            return
        if msg in _STATE_MESSAGES:
            self.revision += 1
        if msg == C.MSG_START:
            self._start_msg(body)
        elif msg == C.MSG_DRAW:
            # The wire's draw count appends to the public hand length. Keep
            # those new slots distinct from an earlier shuffled hand group.
            hand_start = len(self.zone(body[0], C.LOCATION_HAND)) if body and body[0] in (0, 1) else None
            self.disclosure.observe_draw(body, viewer=self.our_player, hand_start=hand_start)
            self._draw(body)
        elif msg == C.MSG_DECK_TOP:
            self.disclosure.observe_deck_top(body)
        elif msg == C.MSG_REVERSE_DECK:
            self.disclosure.observe_reverse_deck()
        elif msg == C.MSG_SHUFFLE_SET_CARD:
            self.disclosure.observe_shuffle_set_card(body)
            self._shuffle_set_card(body)
        elif msg == C.MSG_MOVE:
            # Book first, then change the board: the ledger consumes only messages and does not depend on rebuilding the board.
            effective_code = None
            if len(body) >= 16:
                code, previous, current, _reason = struct.unpack_from(
                    "<IIII", body
                )
                code &= 0x7FFFFFFF
                if not code and not _deck_sort_marker(previous, current):
                    # A private deck sort (``PROCESSOR_SORT_DECK``) reports each sorted card as an anonymous
                    # move onto its own deck slot. The permutation is secret, so a remembered identity at
                    # that slot is stale: the ledger must see the anonymous marker to forget coordinates.
                    code = self._known_move_source_code(previous)
                effective_code = code
                self.disclosure.observe_move(
                    previous,
                    current,
                    code,
                    viewer=self.our_player,
                )
            # The routed MSG_MOVE may redact ``code`` even though this viewer
            # already knew the exact source card.  The disclosure ledger and
            # the shadow board must consume the same recovered public identity;
            # otherwise the card arrives in a public destination as code zero.
            self._move(body, effective_code=effective_code)
        elif msg == C.MSG_SHUFFLE_EXTRA:
            self.disclosure.observe_shuffle(msg, body)
            self._shuffle_extra(body)
        elif msg in (C.MSG_SHUFFLE_DECK, C.MSG_SHUFFLE_HAND):
            self.disclosure.observe_shuffle(msg, body)
        elif msg == C.MSG_SWAP_GRAVE_DECK:
            self.disclosure.observe_shuffle(msg, body)
            self._swap_grave_deck(body)
        elif msg == C.MSG_ADD_COUNTER:
            self._counter(body, add=True)
        elif msg == C.MSG_REMOVE_COUNTER:
            self._counter(body, add=False)
        elif msg == C.MSG_SWAP:
            # Book first, then change the board, in the same order as ``MSG_MOVE``.
            self.disclosure.observe_swap(body)
            self._swap(body)
        elif msg == C.MSG_POS_CHANGE:
            self.disclosure.observe_pos_change(body)
            self._pos_change(body)
        elif msg == C.MSG_EQUIP:
            self._equip(body)
        elif msg == C.MSG_UNEQUIP:
            self._unequip(body)
        elif msg == C.MSG_CARD_TARGET:
            self._card_target(body, add=True)
        elif msg == C.MSG_CANCEL_TARGET:
            self._card_target(body, add=False)
        elif msg == C.MSG_CHAINING:
            self.disclosure.observe_chaining(body)
        elif msg == C.MSG_CHAIN_SOLVING:
            self.disclosure.observe_chain_solving(body)
            self._close_window()
            self.resolution.observe(msg)
        elif msg == C.MSG_CHAIN_SOLVED:
            self._close_window()
            self.resolution.observe(msg)
        elif msg == C.MSG_CHAIN_END:
            self.disclosure.observe_chain_end()
            self._close_window()
            self.resolution.observe(msg)
        elif msg in _CONFIRM_MSGS:
            self.disclosure.observe_confirm(body, msg)
        elif msg == C.MSG_NEW_PHASE:
            self.phase = struct.unpack_from("<H", body)[0]
        elif msg == C.MSG_NEW_TURN:
            self.disclosure.observe_new_turn(body)
            self.turn_player = body[0]
        else:
            self.unknown_messages[msg] += 1
        if self._departures:
            self._name_departures()

    def _start_msg(self, body: bytes) -> None:
        # playertype, duel_rule, lp0, lp1, deck0, extra0, deck1, extra1
        if len(body) < 18:
            return
        d0, e0, d1, e1 = struct.unpack_from("<HHHH", body, 10)
        self.deck_count = [d0, d1]
        self.extra_count = [e0, e1]

    def _update_data(self, body: bytes) -> None:
        if len(body) < 2:
            return
        player, location = body[0], body[1]
        segments = parse_query_segments(body[2:])
        key = (player, location & 0x7F)
        if key not in self.zones:
            return
        if key[1] == C.LOCATION_EXTRA and key[0] != self.our_player:
            # The opponent's Extra Deck is count-only for this viewer.  A
            # compatibility host may route a censored refresh here, but its raw
            # sequences are not public and must never manufacture zone rows.
            # ``extra_count`` remains sourced from START/MOVE/SHUFFLE_EXTRA.
            self.zones[key] = []
            self.updates += 1
            return
        self._merge(key, segments)
        self._reconcile_slots(key)
        self.updates += 1

    def _update_card(self, body: bytes) -> None:
        if len(body) < 3:
            return
        player, location, sequence = body[0], body[1], body[2]
        key = (player, location & 0x7F)
        if key[1] == C.LOCATION_EXTRA and key[0] != self.our_player:
            # MSG_MOVE into a face-down opponent Extra slot is followed by a
            # censored RefreshSingle.  Expanding a count-only zone up to that
            # raw sequence would create default ShadowCard placeholders whose
            # controller looks like player zero.  Keep the zone empty; public
            # identities that returned here live in DisclosureLedger instead.
            self.zones[key] = []
            return
        segments = parse_query_segments(body[3:])
        if not segments or segments[0] is None:
            return
        fields = segments[0]
        if fields.get("_nochange"):
            return
        cards = self.zones.get(key)
        if cards is None:
            return
        while len(cards) <= sequence:
            cards.append(ShadowCard())
        card = cards[sequence]
        if card is None:
            card = cards[sequence] = ShadowCard()
        if self._is_censored_refresh(fields):
            cards[sequence] = self._censored_refresh_card(
                previous=card,
                player=player,
                location=location & 0x7F,
                sequence=sequence,
                fields=fields,
            )
            return
        card.hidden = False
        card.apply(fields)

    def _merge(self, key: tuple[int, int], segments: list[dict | None]) -> None:
        old = self.zones[key]
        new: list = []
        for i, fields in enumerate(segments):
            if fields is None:
                new.append(None)
                continue
            prev = old[i] if i < len(old) else None
            if fields.get("_nochange"):
                new.append(prev if prev is not None else ShadowCard())
                continue
            if self._is_censored_refresh(fields):
                # keep the slot occupied but drop whatever we used to know:
                # the reference client calls ClearData() here, and carrying
                # stale attack/type across a face-down flip is exactly the bug
                # WindBot has (client_card.cpp:51-54).
                new.append(self._censored_refresh_card(
                    previous=prev,
                    player=key[0],
                    location=key[1],
                    sequence=i,
                    fields=fields,
                ))
                continue
            loc = fields.get("info_location")
            if prev is not None and (loc is None or prev.info_location == loc):
                # a cached delta for a card that did not move: merge in place.
                # QUERY_CODE and QUERY_POSITION are never cache-suppressed, so
                # a missing info_location means "nothing changed at all".
                prev.hidden = False
                prev.apply(fields)
                new.append(prev)
            else:
                card = ShadowCard()
                if prev is not None and prev.owner in (0, 1) and "owner" not in fields \
                        and int(prev.code or 0) == int(fields.get("code") or 0):
                    # The card the stream moved here, refreshed at its new place. The refresh does not name an
                    # owner, and the controller is not one: a control change moves a card to the other side.
                    card.owner = prev.owner
                card.apply(fields)
                new.append(card)
        self.zones[key] = new

    def _censored_refresh_card(
        self,
        *,
        previous: ShadowCard | None,
        player: int,
        location: int,
        sequence: int,
        fields: dict | None = None,
    ) -> ShadowCard:
        """Clear private runtime fields while retaining entitled Extra identity."""

        card = ShadowCard(
            hidden=True,
            controller=int(player),
            location=int(location),
            sequence=int(sequence),
        )
        if previous is not None:
            card.controller = previous.controller
            card.location = previous.location
            card.sequence = previous.sequence
            card.position = previous.position
            card.owner = previous.owner
            # A censored refresh must not make the owner's submitted Extra Deck
            # anonymous.  Preserve only its passcode/geometry; database fallback
            # will reconstruct printed fields, while stale runtime fields stay
            # cleared.  Other censored zones remain genuinely anonymous.
            if (
                int(player) == self.our_player
                and int(location) == C.LOCATION_EXTRA
                and int(previous.code or 0)
            ):
                card.code = int(previous.code) & 0x7FFFFFFF
                card.hidden = False
                if card.owner not in (0, 1):
                    card.owner = self.our_player
                if not card.position:
                    card.position = C.POS_FACEDOWN_DEFENSE
        info_location = None if fields is None else fields.get("info_location")
        if info_location is not None:
            packed = int(info_location)
            card.info_location = packed
            card.controller = packed & 0xFF
            card.location = (packed >> 8) & 0xFF
            card.sequence = (packed >> 16) & 0xFF
            card.position = ((packed >> 24) & 0xFF) & 0x0F
            if card.owner not in (0, 1) and card.code:
                card.owner = card.controller
        return card

    @staticmethod
    def _is_censored_refresh(fields: dict) -> bool:
        """Recognise both whole-zone and RefreshSingle censorship shapes."""

        return bool(fields.get("_hidden")) or (
            "code" in fields and int(fields.get("code") or 0) == 0
        )

    def _draw(self, body: bytes) -> None:
        """A draw moves cards into the hand, and nothing refreshes afterwards.

        ``MSG_DRAW`` is not followed by ``RefreshHand`` (the host refreshes the
        hand from the main message block and from ``MSG_SHUFFLE_HAND``, not
        from the draw), so a hand rebuilt only from refreshes is one card short
        for as long as it takes some other message to trigger one.  The
        reference client adds the cards here too (``duelclient.cpp``
        ``MSG_DRAW`` -> ``AddCard(..., LOCATION_HAND, ...)``).
        """
        if len(body) < 2:
            return
        player, count = body[0], body[1]
        self.deck_count[player] = max(0, self.deck_count[player] - count)
        hand = self.zones.setdefault((player, C.LOCATION_HAND), [])
        for i in range(count):
            off = 2 + 4 * i
            code = raw_code = 0
            if off + 4 <= len(body):
                (raw_code,) = struct.unpack_from("<I", body, off)
                code = raw_code & 0x7FFFFFFF
            card = ShadowCard()
            card.code = code
            card.hidden = code == 0
            card.controller = player
            card.owner = player
            card.location = C.LOCATION_HAND
            # operations.cpp::draw sets the hand's public/facedown mask, not
            # a monster battle position. MSG_DRAW carries the public bit and
            # may not be followed by a hand refresh before the next decision.
            card.position = C.POS_FACEUP if raw_code & 0x80000000 else C.POS_FACEDOWN
            hand.append(card)
            if code and player == self.our_player:
                self.seen_out_of_deck[code] += 1
        self._resequence(player, C.LOCATION_HAND)

    def _shuffle_extra(self, body: bytes) -> None:
        """Apply MSG_SHUFFLE_EXTRA's post-shuffle face-down code vector.

        The owner receives real codes in current vector order; the other seat
        receives zeros.  A following censored RefreshExtra must not overwrite
        the owner's new mapping with the pre-shuffle identities.
        """

        if len(body) < 2:
            return
        player, count = int(body[0]), int(body[1])
        if player not in (0, 1) or len(body) < 2 + count * 4:
            return
        codes = [
            int(struct.unpack_from("<I", body, 2 + 4 * index)[0]) & 0x7FFFFFFF
            for index in range(count)
        ]
        self.extra_count[player] = count
        if player != self.our_player:
            # The remote seat receives an all-zero payload, and this viewer has
            # never owned its face-down Extra Deck.  Keep that side count-only;
            # manufacturing anonymous rows would make later source sequences
            # look trackable when they are not.
            self.zones[(player, C.LOCATION_EXTRA)] = []
            return

        current = [
            card
            for card in self.zone(player, C.LOCATION_EXTRA)
            if card is not None
        ]
        faceup_n = min(max(0, int(self.extra_faceup[player])), count)
        facedown_n = count - faceup_n
        current_faceup = [
            card for card in current if card.position & C.POS_FACEUP
        ][-faceup_n:] if faceup_n else []
        facedown = []
        for sequence, code in enumerate(codes[:facedown_n]):
            known = code != 0
            facedown.append(ShadowCard(
                code=code if known else 0,
                hidden=not known,
                controller=player,
                owner=player if known else -1,
                location=C.LOCATION_EXTRA,
                sequence=sequence,
                position=C.POS_FACEDOWN_DEFENSE,
            ))

        faceup = []
        for index, code in enumerate(codes[facedown_n:]):
            if index < len(current_faceup):
                card = current_faceup[index]
            else:
                card = ShadowCard(
                    controller=player,
                    owner=player,
                    location=C.LOCATION_EXTRA,
                    position=C.POS_FACEUP_DEFENSE,
                )
            if code:
                card.code = code
                card.hidden = False
            card.controller = player
            card.location = C.LOCATION_EXTRA
            if not (card.position & C.POS_FACEUP):
                card.position = C.POS_FACEUP_DEFENSE
            if card.owner not in (0, 1):
                card.owner = player
            faceup.append(card)

        self.zones[(player, C.LOCATION_EXTRA)] = facedown + faceup
        self.extra_faceup[player] = faceup_n
        self._resequence(player, C.LOCATION_EXTRA)

    def _shuffle_set_card(self, body: bytes) -> None:
        if len(body) < 2:
            return
        count = int(body[1])
        coordinates = set()
        positions_by_zone = {}
        proven_after = {}
        for block in (0, 1):
            start = 2 + block * count * 4
            for index in range(count):
                offset = start + index * 4
                if offset + 4 > len(body):
                    break
                packed = struct.unpack_from("<I", body, offset)[0]
                if packed:
                    coordinate = (
                        packed & 0xFF,
                        (packed >> 8) & 0x7F,
                        (packed >> 16) & 0xFF,
                    )
                    coordinates.add(coordinate)
                    position = (packed >> 24) & 0x0F
                    if block:
                        proven_after[coordinate] = position
                    else:
                        positions_by_zone.setdefault(coordinate[:2], set()).add(position)
        for controller, location, sequence in coordinates:
            zone = self.zone(controller, location)
            if not 0 <= sequence < len(zone) or zone[sequence] is None:
                continue
            # ShuffleSetCard changes identity assignment, not the position
            # masks themselves. Uniform public masks survive the permutation;
            # a mixed group has no known slot assignment unless the second
            # (material-host) block explicitly supplies its post-shuffle word.
            known = positions_by_zone.get((controller, location), set())
            position = proven_after.get((controller, location, sequence),
                next(iter(known)) if len(known) == 1 else 0)
            zone[sequence] = ShadowCard(
                hidden=True,
                controller=controller,
                owner=-1,
                location=location,
                sequence=sequence,
                position=position,
            )
            key = (controller, location, sequence)
            self.materials.pop(key, None)
            self.counters.pop(key, None)
            self.positions[key] = position
            self._drop_relations_at(key)

    # -- own-deck group elimination ----------------------------------------

    def _close_window(self) -> None:
        self._own_group = Counter()

    def chain_resolution(self) -> int:
        """The chain-link resolution in progress (``netduel.resolution``): the native frame's ``chain_resolution``."""
        return self.resolution.current

    def unresolved_own_deck_departures(self) -> int:
        """Anonymous departures from our deck whose identity the stream has not (yet) resolved."""
        return len(self._departures)

    def _depart(self, code: int) -> None:
        self.seen_out_of_deck[code] += 1
        self.resolved_departures += 1
        self.revision += 1

    def _name_departures(self) -> None:
        """Name each pending departure whose field slot now holds a code the ledger proves."""
        for departure in list(self._departures):
            if departure["at"] is None:
                continue
            code = self.disclosure.known_code_at(self.our_player, *departure["at"])
            if code:
                self._departures.remove(departure)
                self._depart(int(code))

    def _follow_departures(self, src, dst, code: int, fixed_dst: bool) -> None:
        """A card left a field slot: a pending departure there is named by the move's code, or follows it."""
        for departure in self._departures:
            if departure["at"] == src:
                if code:
                    self._departures.remove(departure)
                    self._depart(code)
                else:
                    departure["at"] = dst if fixed_dst else None
                return

    def _observe_prompt(self, msg: int, body: bytes) -> None:
        """Read an own ``MSG_SELECT_CARD`` / ``MSG_SELECT_UNSELECT_CARD``: resolve pending departures by the
        cards it offers, then (select-unselect) take the core's list of cards chosen so far as the group."""
        # player, (finishable,) cancelable, min, max, then each block: count and 8 bytes per card
        offset = 5 if msg == C.MSG_SELECT_UNSELECT_CARD else 4
        blocks = []
        for _ in range(2 if msg == C.MSG_SELECT_UNSELECT_CARD else 1):
            if len(body) < offset + 1:
                raise ValueError(f"card prompt {msg} is truncated")
            count = body[offset]
            end = offset + 1 + 8 * count
            if len(body) < end:
                raise ValueError(f"card prompt {msg} is truncated")
            blocks.append([(struct.unpack_from("<I", body, at)[0] & 0x7FFFFFFF, body[at + 4], body[at + 5])
                           for at in range(offset + 1, end, 8)])
            offset = end
        self._resolve_group([card for block in blocks for card in block])
        if msg == C.MSG_SELECT_UNSELECT_CARD and self.resolution.current:
            self._own_group = Counter(code for code, controller, location in blocks[1]
                                      if controller == self.our_player and location == C.LOCATION_DECK and code)
        self._prompt = (msg, blocks[0])

    def _resolve_group(self, offered_cards) -> None:
        """Resolve the window's anonymous departures from our deck by elimination.

        Every card offered must be one of our deck cards with its code, all of them from the group we selected
        in this window, and the group minus the offer must be exactly as many cards as departed anonymously:
        those are the departed cards. Otherwise nothing is inferred."""
        pending = [d for d in self._departures if self.resolution.current and d["window"] == self.resolution.current]
        if not pending or not self._own_group:
            return
        if any(controller != self.our_player or location != C.LOCATION_DECK or not code
               for code, controller, location in offered_cards):
            return
        offered = Counter(code for code, _, _ in offered_cards)
        departed = self._own_group - offered
        if offered - self._own_group or sum(departed.values()) != len(pending):
            return
        self.seen_out_of_deck.update(departed)
        self.resolved_departures += len(pending)
        self._departures = [d for d in self._departures if all(d is not p for p in pending)]
        self._own_group = offered
        self.revision += 1

    def observe_response(self, data: bytes) -> None:
        """This player's response to its last prompt: the own deck cards it selected join the window's group
        (a select-unselect pick is added to the core's chosen list; a card selection replaces the group)."""
        prompt, self._prompt = self._prompt, None
        if prompt is None or not self.resolution.current or bytes(data) == _CANCEL_RESPONSE:
            return
        msg, listed = prompt
        if msg == C.MSG_SELECT_UNSELECT_CARD:
            if len(data) != 2 or data[0] != 1:
                raise ValueError("a select-unselect response names one listed card")
            picked = [listed[data[1]]] if data[1] < len(listed) else []  # an unselect is re-listed by the core
        else:
            if not data or len(data) != data[0] + 1 or any(index >= len(listed) for index in data[1:]):
                raise ValueError("a card selection response names listed cards")
            picked = [listed[index] for index in data[1:]]
            self._own_group = Counter()
        for code, controller, location in picked:
            if controller == self.our_player and location == C.LOCATION_DECK and code:
                self._own_group[code] += 1

    def _known_move_source_code(self, previous: int) -> int:
        """Recover only a source identity this network viewer already knew."""

        controller = int(previous) & 0xFF
        location = (int(previous) >> 8) & 0xFF
        sequence = (int(previous) >> 16) & 0xFF
        position = (int(previous) >> 24) & 0xFF
        if location & C.LOCATION_OVERLAY:
            materials = self.materials.get(
                (controller, location & 0x7F, sequence), ()
            )
            if 0 <= position < len(materials):
                return int(materials[position].code or 0) & 0x7FFFFFFF
            return 0
        exact = self.disclosure.known_code_at(
            self.our_player,
            controller,
            location,
            sequence,
        )
        if exact:
            return exact
        visible = (
            controller == self.our_player
            and location != C.LOCATION_DECK
        ) or (
            location not in (
                C.LOCATION_DECK,
                C.LOCATION_HAND,
                C.LOCATION_EXTRA,
            )
            and not bool(position & C.POS_FACEDOWN)
        )
        if not visible:
            return 0
        zone = self.zone(controller, location & 0x7F)
        if 0 <= sequence < len(zone) and zone[sequence] is not None:
            return int(zone[sequence].code or 0) & 0x7FFFFFFF
        return 0

    def _swap_grave_deck(self, body: bytes) -> None:
        """Apply the public bulk grave/deck swap before ``RefreshGrave``.

        The message carries only a player byte.  The old grave is public, so
        its non-Extra monsters determine the new deck count for either seat.
        For our own seat the submitted deck multiset also determines every card
        moved into the new grave; keeping both sides current prevents a prompt
        following the swap from observing stale counts.
        """

        if not body or body[0] not in (0, 1):
            return
        player = int(body[0])
        grave = [
            card for card in self.zone(player, C.LOCATION_GRAVE)
            if card is not None
        ]
        extra_mask = C.TYPE_FUSION | C.TYPE_SYNCHRO | C.TYPE_XYZ | C.TYPE_LINK
        new_deck_cards = [
            card for card in grave if not (int(getattr(card, "type", 0)) & extra_mask)
        ]
        extra_cards = [
            card for card in grave if int(getattr(card, "type", 0)) & extra_mask
        ]
        moved_to_extra = len(extra_cards)
        old_deck_count = max(0, int(self.deck_count[player]))
        self.deck_count[player] = len(new_deck_cards)
        self.extra_count[player] += moved_to_extra

        if player != self.our_player:
            # The old opponent deck is anonymous until the host's immediate
            # RefreshGrave packet names it.  Clear the obsolete old-grave rows;
            # the authoritative count is already correct above.
            self.zones[(player, C.LOCATION_GRAVE)] = []
            return

        old_deck = self.our_remaining_deck()
        if sum(old_deck.values()) != old_deck_count:
            # Preserve the discrepancy so snapshot_from_shadow fails closed;
            # inventing cards here would hide the original tracking fault.
            return
        new_deck = Counter(
            int(card.code) for card in new_deck_cards if int(getattr(card, "code", 0))
        )
        if sum(new_deck.values()) != len(new_deck_cards):
            # Our grave must be identity-complete.  Leave the old tracker in
            # place so the next strict snapshot detects the count mismatch.
            return
        self.seen_out_of_deck = self.our_deck_start.copy()
        self.seen_out_of_deck.subtract(new_deck)
        own_extra = self.zones.setdefault((player, C.LOCATION_EXTRA), [])
        insert_at = max(0, len(own_extra) - int(self.extra_faceup[player]))
        for card in extra_cards:
            card.controller = player
            card.location = C.LOCATION_EXTRA
            card.position = C.POS_FACEDOWN_DEFENSE
            own_extra.insert(insert_at, card)
            insert_at += 1
        self._resequence(player, C.LOCATION_EXTRA)
        rebuilt = []
        for code, count in sorted(old_deck.items()):
            for _ in range(max(0, int(count))):
                rebuilt.append(ShadowCard(
                    code=int(code),
                    controller=player,
                    owner=player,
                    location=C.LOCATION_GRAVE,
                    position=C.POS_FACEUP_DEFENSE,
                ))
        self.zones[(player, C.LOCATION_GRAVE)] = rebuilt
        self._resequence(player, C.LOCATION_GRAVE)

    def _move(self, body: bytes, *, effective_code: int | None = None) -> None:
        if len(body) < 16:
            return
        code, previous, current, _reason = struct.unpack_from("<IIII", body)
        code = int(code) & 0x7FFFFFFF
        if effective_code is not None:
            recovered = int(effective_code) & 0x7FFFFFFF
            if code and recovered != code:
                raise ValueError("recovered MSG_MOVE code differs from wire code")
            code = recovered
        prev_ctl, prev_loc, prev_seq, prev_pos = (
            previous & 0xFF,
            (previous >> 8) & 0xFF,
            (previous >> 16) & 0xFF,
            (previous >> 24) & 0xFF,
        )
        cur_ctl, cur_loc, cur_seq, cur_pos = (
            current & 0xFF,
            (current >> 8) & 0xFF,
            (current >> 16) & 0xFF,
            (current >> 24) & 0xFF,
        )
        base_prev = prev_loc & 0x7F
        base_cur = cur_loc & 0x7F
        # the four overlay cases of duelclient.cpp:2687-2845
        was_overlay = bool(prev_loc & C.LOCATION_OVERLAY)
        is_overlay = bool(cur_loc & C.LOCATION_OVERLAY)
        src = (prev_ctl, base_prev, prev_seq)
        dst = (cur_ctl, base_cur, cur_seq)

        # Zone counting has to look at the overlay bit, not at the location
        # underneath it.  An xyz summon attaches its materials while the xyz
        # monster is still in the *extra deck*, so the core reports the
        # material's destination as LOCATION_OVERLAY | LOCATION_EXTRA (0xc0):
        # masking the overlay bit off first reads that as "a card entered the
        # extra deck" and inflates the count by one per material.
        left = base_prev if not was_overlay else None
        entered = base_cur if not is_overlay else None
        if left in _SLOT_ZONES and self._departures:
            self._follow_departures(src, dst, code, entered in _SLOT_ZONES)
        if left == C.LOCATION_DECK and entered != C.LOCATION_DECK:
            self.deck_count[prev_ctl] = max(0, self.deck_count[prev_ctl] - 1)
            if prev_ctl == self.our_player and code:
                self.seen_out_of_deck[code] += 1
                if self._own_group[code] > 0:
                    self._own_group[code] -= 1
            elif prev_ctl == self.our_player:
                fixed = entered in _SLOT_ZONES
                self._departures.append({"window": self.resolution.current, "at": dst if fixed else None})
        elif entered == C.LOCATION_DECK and left != C.LOCATION_DECK:
            self.deck_count[cur_ctl] += 1
            if cur_ctl == self.our_player and code:
                self.seen_out_of_deck[code] -= 1
        # RefreshExtra goes to the owner only (single_duel.cpp:1580), so the
        # opponent's extra deck is a count we maintain, not a zone we see
        if left == C.LOCATION_EXTRA and entered != C.LOCATION_EXTRA:
            self.extra_count[prev_ctl] = max(0, self.extra_count[prev_ctl] - 1)
        elif entered == C.LOCATION_EXTRA and left != C.LOCATION_EXTRA:
            self.extra_count[cur_ctl] += 1

        if not was_overlay and not is_overlay:
            self._move_relations(src, dst, base_prev, base_cur)
            self._relocate(code, src, dst, cur_pos)
            # The host blanks a face-down card's whole query segment
            # (``single_duel.cpp`` RefreshMzone / RefreshSzone), position
            # included, but never blanks the position byte of MSG_MOVE, so the
            # stream is the only place the face-down *flavour* survives.
            self.positions.pop(src, None)
            if base_cur & C.LOCATION_ONFIELD:
                self.positions[dst] = cur_pos & 0x0F
            # A card that left the field drops its counters; one that only
            # changed zone inside the same location keeps them, which is why
            # the reference compares the raw location bytes and not the slot.
            if (prev_loc & C.LOCATION_ONFIELD) and cur_loc != prev_loc:
                self.counters.pop(src, None)
            else:
                self._rekey(self.counters, src, dst)
            # Materials follow their host wherever it goes, and the core never
            # says so: the reference client hangs them off the card object
            # (``duelclient.cpp``: ``olcard->overlayed``), which moves for free.
            # The load-bearing case is the xyz summon, where the materials are
            # attached in the extra deck and the monster then walks to the
            # monster zone under its own MSG_MOVE -- with no message per
            # material.  A stack left behind at the old slot would be dropped
            # by the next refresh and the monster would arrive bare.
            #
            # Renumbering after the move is not cosmetic: each material carries
            # the host's location and sequence in its own fields, and those are
            # what the observation encoder reads.  Moving the stack without
            # restating them leaves every material claiming to still sit on a
            # card in the extra deck.
            if src in self.materials:
                self._rekey(self.materials, src, dst)
                self._renumber(dst)
        elif not was_overlay and is_overlay:
            self._move_relations(src, None, base_prev, 0)
            self.counters.pop(src, None)
            self._rekey(self.materials, src, None)
            # take the card out of the zone it came from, so a material pulled
            # from the graveyard or the banished pile does not stay there too
            card = self._take_from(src)
            if card is None:
                card = ShadowCard()
            if code:
                card.code = code & 0x7FFFFFFF
                card.hidden = False
            card.controller = cur_ctl
            if card.owner not in (0, 1):
                card.owner = prev_ctl if base_prev else cur_ctl
            card.location = base_cur
            card.sequence = cur_seq
            self.materials.setdefault(dst, []).append(card)
            self._renumber(dst)
        elif was_overlay and not is_overlay:
            # A detached material lands somewhere real -- usually the
            # graveyard, which nothing refreshes.  Dropping it here instead of
            # filing it would leave that zone one card short and renumber every
            # card behind it for the rest of the duel.
            card = self._detach(src, prev_pos)
            if card is None:
                card = ShadowCard()
            if code:
                card.code = code & 0x7FFFFFFF
                card.hidden = False
            card.status = 0
            card.controller = cur_ctl
            if card.owner not in (0, 1):
                card.owner = cur_ctl
            card.location = base_cur
            card.position = cur_pos & 0x0F
            self._put_into(card, dst, cur_pos)
            if base_cur & C.LOCATION_ONFIELD:
                self.positions[dst] = cur_pos & 0x0F
        else:
            card = self._detach(src, prev_pos)
            if card is not None:
                card.controller = cur_ctl
                card.location = base_cur
                card.sequence = cur_seq
                self.materials.setdefault(dst, []).append(card)
                self._renumber(dst)

    def _relocate(self, code, src, dst, cur_pos) -> None:
        """Move one card between zones, the way the reference client does.

        The module docstring's "the host re-queries a zone after every board
        change" holds for the monster zone, the spell/trap zone and the hand,
        and for nothing else: ``RefreshGrave`` is reachable only from
        ``MSG_SWAP_GRAVE_DECK`` and ``RefreshExtra`` only from
        ``MSG_SHUFFLE_EXTRA`` (``single_duel.cpp:882`` and ``:871``), and the
        banished zone has no refresh at all.  So for the graveyard, the
        banished pile and the extra deck the message stream is not a
        cross-check on the refresh -- it is the *only* source, and without it
        those zones drift the moment a card leaves one of them and every later
        card's sequence is off by one.

        Membership and order follow ``ClientField::AddCard`` /
        ``RemoveCard`` (``client_field.cpp:157`` and ``:216``): list zones
        erase-and-renumber, field zones are fixed slots, and a card returning
        face-down to the extra deck goes in front of the face-up pendulum
        block that lives at its end.  The card object itself travels, so what
        we already knew about it is not thrown away on the way.
        """
        card = self._take_from(src)
        if card is None:
            card = ShadowCard()
        if code:
            card.code = code & 0x7FFFFFFF
            card.hidden = False
        elif dst[1] == C.LOCATION_EXTRA:
            # the reference resets the code here even when it is 0, because a
            # card going back to the extra deck face-down is not identifiable
            card.code = 0
            card.hidden = True
        if card.owner not in (0, 1):
            card.owner = src[0] if src[1] else dst[0]
        if dst[1] != src[1] or not (dst[1] & C.LOCATION_ONFIELD):
            # A card that left the field is no longer negated or forbidden.
            # Only the field zones are refreshed, so a status that went stale
            # in the graveyard would never be corrected and would keep
            # reporting a card as disabled for the rest of the duel.
            card.status = 0
        card.controller = dst[0]
        card.location = dst[1]
        card.position = cur_pos & 0x0F
        self._put_into(card, dst, cur_pos)

    def _take_from(self, src: tuple[int, int, int]):
        player, location, sequence = src
        if location in _SLOT_ZONES:
            zone = self.zones.get((player, location))
            if zone is None or sequence >= len(zone):
                return None
            card, zone[sequence] = zone[sequence], None
            return card
        if not self._tracks_list(player, location):
            return None
        zone = self.zones.get((player, location))
        if zone is None or sequence >= len(zone):
            return None
        card = zone.pop(sequence)
        if location == C.LOCATION_EXTRA and card is not None:
            if card.position & C.POS_FACEUP:
                self.extra_faceup[player] = max(0, self.extra_faceup[player] - 1)
        self._resequence(player, location)
        return card

    def _put_into(self, card, dst: tuple[int, int, int], cur_pos: int) -> None:
        player, location, sequence = dst
        if location in _SLOT_ZONES:
            zone = self.zones.setdefault((player, location), [])
            while len(zone) <= sequence:
                zone.append(None)
            zone[sequence] = card
            card.sequence = sequence
            return
        if not self._tracks_list(player, location):
            return
        zone = self.zones.setdefault((player, location), [])
        if location == C.LOCATION_EXTRA:
            faceup = self.extra_faceup[player]
            if faceup == 0 or (cur_pos & C.POS_FACEUP):
                zone.append(card)
            else:
                zone.insert(max(0, len(zone) - faceup), card)
            if cur_pos & C.POS_FACEUP:
                self.extra_faceup[player] += 1
        else:
            zone.append(card)
        self._resequence(player, location)

    def _tracks_list(self, player: int, location: int) -> bool:
        """Whether we hold the contents of this list zone at all.

        ``RefreshExtra`` is sent to the owner only, so the opponent's extra
        deck was never populated and must not be maintained: removing from an
        empty list would silently eat somebody else's card.  Its size is kept
        in ``extra_count`` instead.
        """
        if location not in _LIST_ZONES:
            return False
        if location == C.LOCATION_EXTRA and player != self.our_player:
            return False
        return True

    def _resequence(self, player: int, location: int) -> None:
        for i, card in enumerate(self.zones.get((player, location), ())):
            if card is not None:
                card.sequence = i


    def _pos_change(self, body: bytes) -> None:
        """``MSG_POS_CHANGE``: the only announcement of a flip we ever get."""
        if len(body) < 9:
            return
        player, location, sequence = body[4], body[5], body[6]
        new_pos = body[8] & 0x0F
        key = (player, location & 0x7F, sequence)
        # A card turned face-down loses its counters without any counter
        # message: the core clears them in card::reset(RESET_TURN_SET) and the
        # reference client clears them here (duelclient.cpp MSG_POS_CHANGE).
        if (body[7] & C.POS_FACEUP) and (new_pos & C.POS_FACEDOWN):
            self.counters.pop(key, None)
        if location & C.LOCATION_ONFIELD:
            self.positions[key] = new_pos
        zone = self.zones.get((player, location & 0x7F))
        if zone is not None and sequence < len(zone) and zone[sequence] is not None:
            zone[sequence].position = new_pos

    @staticmethod
    def _at_key(at: int) -> tuple[int, int, int]:
        return (at & 0xFF, ((at >> 8) & 0xFF) & 0x7F, (at >> 16) & 0xFF)

    def _equip(self, body: bytes) -> None:
        """``MSG_EQUIP`` publishes the directed equip-card -> host edge."""
        if len(body) < 8:
            return
        source, target = struct.unpack_from("<II", body, 0)
        self.equip_targets[self._at_key(source)] = self._at_key(target)

    def _unequip(self, body: bytes) -> None:
        # MSG_UNEQUIP is reserved/commented in this core, but accepting the
        # legacy four-byte shape costs nothing and keeps compatible servers exact.
        if len(body) >= 4:
            (source,) = struct.unpack_from("<I", body, 0)
            self.equip_targets.pop(self._at_key(source), None)

    def _card_target(self, body: bytes, add: bool) -> None:
        """Maintain the persistent SetCardTarget source -> target relation."""
        if len(body) < 8:
            return
        source_at, target_at = struct.unpack_from("<II", body, 0)
        source, target = self._at_key(source_at), self._at_key(target_at)
        if add:
            self.card_targets.setdefault(source, set()).add(target)
            return
        targets = self.card_targets.get(source)
        if targets is None:
            return
        targets.discard(target)
        if not targets:
            self.card_targets.pop(source, None)

    def _drop_relations_at(self, key: tuple[int, int, int]) -> None:
        self.equip_targets.pop(key, None)
        self.card_targets.pop(key, None)
        self.equip_targets = {
            source: target for source, target in self.equip_targets.items()
            if target != key
        }
        for source, targets in list(self.card_targets.items()):
            targets.discard(key)
            if not targets:
                self.card_targets.pop(source, None)

    def _move_relations(self, src, dst, prev_loc: int, cur_loc: int) -> None:
        """Move an on-field endpoint, or retire every edge when it leaves.

        Core clears equip/target relations on the reset paths that leave the
        field.  A control/zone move that stays on field can keep an endpoint,
        so re-key both forward sources and forward targets.
        """
        if dst is None or not (prev_loc & C.LOCATION_ONFIELD) \
                or not (cur_loc & C.LOCATION_ONFIELD):
            self._drop_relations_at(src)
            return
        if src == dst:
            return
        if src in self.equip_targets:
            self.equip_targets[dst] = self.equip_targets.pop(src)
        if src in self.card_targets:
            self.card_targets[dst] = self.card_targets.pop(src)
        self.equip_targets = {
            source: (dst if target == src else target)
            for source, target in self.equip_targets.items()
        }
        for targets in self.card_targets.values():
            if src in targets:
                targets.remove(src)
                targets.add(dst)

    def _swap_relation_endpoints(self, a, b) -> None:
        def moved(key):
            return b if key == a else (a if key == b else key)

        self.equip_targets = {
            moved(source): moved(target)
            for source, target in self.equip_targets.items()
        }
        self.card_targets = {
            moved(source): {moved(target) for target in targets}
            for source, targets in self.card_targets.items()
        }

    def _detach(self, src: tuple[int, int, int], index: int):
        """Take material ``index`` off the monster at ``src``.

        ``index`` is the *previous position* byte of ``MSG_MOVE``, which the
        core reuses as the material index for a card leaving an overlay
        (``duelclient.cpp:2795``).
        """
        stack = self.materials.get(src)
        if not stack or index >= len(stack):
            self.slot_problems.append(
                f"detach material {index} of {src} but the stack holds "
                f"{0 if not stack else len(stack)}"
            )
            return None
        card = stack.pop(index)
        if stack:
            self._renumber(src)
        else:
            self.materials.pop(src, None)
        return card

    def _renumber(self, key: tuple[int, int, int]) -> None:
        """Restate each material the way ``get_cards_in_location`` would.

        The engine hands a material over with the host's location OR-ed with
        ``LOCATION_OVERLAY``, the host's sequence, and the material's index in
        the stack as its position -- which is also what turns into the ``a`` /
        ``b`` / ``c`` suffix of its spec.
        """
        for i, card in enumerate(self.materials.get(key, ())):
            card.position = i
            card.controller = key[0]
            card.location = key[1] | C.LOCATION_OVERLAY
            card.sequence = key[2]

    @staticmethod
    def _rekey(table: dict, src, dst) -> None:
        value = table.pop(src, None)
        if value is None or dst is None:
            return
        table[dst] = value

    def _counter(self, body: bytes, add: bool) -> None:
        if len(body) < 7:
            return
        ctype = struct.unpack_from("<H", body)[0]
        player, location, sequence = body[2], body[3], body[4]
        count = struct.unpack_from("<H", body, 5)[0]
        key = (player, location & 0x7F, sequence)
        slot = self.counters.setdefault(key, {})
        if add:
            slot[ctype] = slot.get(ctype, 0) + count
        else:
            slot[ctype] = slot.get(ctype, 0) - count
            if slot[ctype] <= 0:
                slot.pop(ctype, None)
        if not slot:
            self.counters.pop(key, None)

    def _swap(self, body: bytes) -> None:
        """Two monsters trade zones; their materials and counters go with them."""
        if len(body) < 16:
            return
        # code(4) controller location sequence position, twice
        a = (body[4], body[5] & 0x7F, body[6])
        b = (body[12], body[13] & 0x7F, body[14])
        self._swap_relation_endpoints(a, b)
        for departure in self._departures:
            departure["at"] = b if departure["at"] == a else a if departure["at"] == b else departure["at"]
        for table in (self.materials, self.counters):
            va, vb = table.pop(a, None), table.pop(b, None)
            if va is not None:
                table[b] = va
            if vb is not None:
                table[a] = vb
        self._renumber(a)
        self._renumber(b)

    def _reconcile_slots(self, key: tuple[int, int]) -> None:
        """Materials and counters may only sit under an occupied slot.

        The refresh is the engine's own view of which slots hold a card, so
        anything left over after it is state we failed to retire -- dropping it
        here keeps a missed message from compounding across a whole duel.
        """
        player, location = key
        occupied = {
            i for i, card in enumerate(self.zones.get(key, ())) if card is not None
        }
        for table in (self.materials, self.counters, self.positions):
            stale = [
                k
                for k in table
                if k[0] == player and k[1] == location and k[2] not in occupied
            ]
            for k in stale:
                table.pop(k, None)
        stale_rel = {
            k for k in set(self.equip_targets) | set(self.card_targets)
            if k[0] == player and k[1] == location and k[2] not in occupied
        }
        stale_rel.update(
            target for target in self.equip_targets.values()
            if target[0] == player and target[1] == location
            and target[2] not in occupied
        )
        stale_rel.update(
            target for targets in self.card_targets.values() for target in targets
            if target[0] == player and target[1] == location
            and target[2] not in occupied
        )
        for key_ in stale_rel:
            self._drop_relations_at(key_)

    # -- observation support -----------------------------------------------

    def cards_in_location(self, player: int, location: int) -> list:
        """``ygopro.h``'s ``get_cards_in_location`` order, minus the engine.

        Each xyz material is emitted *before* the monster that carries it, with
        ``location`` OR-ed with ``LOCATION_OVERLAY``, the monster's sequence and
        its own index as the position -- the exact rows the observation encoder
        expects.  Empty slots contribute nothing, as in the engine, whose query
        buffer skips them.
        """
        out: list = []
        for seq, card in enumerate(self.zone(player, location)):
            if card is None:
                continue
            for material in self.materials.get((player, location, seq), ()):
                out.append(material)
            if card.hidden and not card.position:
                # the refresh blanked the segment; the position we followed
                # through MSG_MOVE / MSG_POS_CHANGE is the only copy left
                card.position = self.positions.get((player, location, seq), 0)
            out.append(card)
        return out

    def counters_of(self, player: int, location: int, sequence: int) -> dict:
        return self.counters.get((player, location & 0x7F, sequence), {})

    @staticmethod
    def _pack_key(key: tuple[int, int, int]) -> int:
        return key[0] | (key[1] << 8) | (key[2] << 16)

    def equip_target_of(self, player: int, location: int, sequence: int) -> int:
        target = self.equip_targets.get((player, location & 0x7F, sequence))
        return self._pack_key(target) if target is not None else 0

    def targets_of(self, player: int, location: int, sequence: int) -> tuple[int, ...]:
        targets = self.card_targets.get((player, location & 0x7F, sequence), ())
        return tuple(self._pack_key(key) for key in sorted(targets))

    # -- consistency -------------------------------------------------------

    def cross_check(self, cards: list[tuple[int, int, int, int]]) -> list[str]:
        """Compare a prompt's card list against the shadow board.

        Every ``MSG_SELECT_*`` prompt names the cards it offers as
        ``(code, controller, location, sequence)``, written by the engine from
        the real board.  Agreeing with those tuples is agreeing with the
        engine, and unlike a replay-based check it costs nothing and runs on
        every live duel.

        Only non-zero codes are checked: the host blanks the code of a card we
        are not allowed to identify.
        """
        problems: list[str] = []
        for code, controller, location, sequence in cards:
            if location & C.LOCATION_OVERLAY:
                continue  # xyz materials are not tracked, see the module doc
            loc = location & 0x7F
            if loc == C.LOCATION_DECK:
                continue  # deck contents are never sent to anyone
            zone = self.zones.get((controller, loc))
            if zone is None:
                continue
            self.checks += 1
            if sequence >= len(zone):
                problems.append(
                    f"code {code} at ({controller}, {loc}, {sequence}) but the "
                    f"shadow zone holds {len(zone)} slots"
                )
                continue
            card = zone[sequence]
            if card is None:
                problems.append(
                    f"code {code} at ({controller}, {loc}, {sequence}) but the "
                    "shadow slot is empty"
                )
            elif card.code and card.code != code:
                problems.append(
                    f"code {code} at ({controller}, {loc}, {sequence}) but the "
                    f"shadow holds {card.code}"
                )
            elif not card.code:
                # we were never told this card's identity; the prompt just did
                self.learned += 1
                self.revision += 1
                card.code = code
                card.hidden = False
        self.mismatches += len(problems)
        return problems

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict:
        def zone_codes(p, loc):
            return [c.code if c else 0 for c in self.zone(p, loc)]

        return {
            "our_player": self.our_player,
            "turn_player": self.turn_player,
            "phase": self.phase,
            "deck_count": list(self.deck_count),
            "hand": [zone_codes(0, C.LOCATION_HAND), zone_codes(1, C.LOCATION_HAND)],
            "mzone": [zone_codes(0, C.LOCATION_MZONE), zone_codes(1, C.LOCATION_MZONE)],
            "szone": [zone_codes(0, C.LOCATION_SZONE), zone_codes(1, C.LOCATION_SZONE)],
            "grave": [zone_codes(0, C.LOCATION_GRAVE), zone_codes(1, C.LOCATION_GRAVE)],
            "extra": [zone_codes(0, C.LOCATION_EXTRA), zone_codes(1, C.LOCATION_EXTRA)],
            "removed": [
                zone_codes(0, C.LOCATION_REMOVED),
                zone_codes(1, C.LOCATION_REMOVED),
            ],
            "updates": self.updates,
            "checks": self.checks,
            "mismatches": self.mismatches,
            "learned": self.learned,
        }
