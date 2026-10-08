"""Board snapshots, the information state each player is allowed to see, and
the difference between two snapshots.

Three jobs, in order of how easy they are to get subtly wrong:

**Capture.**  ``query_field_card`` over every zone of both players, decoded with
the same segment parser the network client uses, so a card looks the same here
as it does on the shadow board.

**Masking.**  A sample recorded at player *p*'s decision point must contain what
*p* knows and nothing else, or the menu head learns to read the opponent's hand.
The rule is ygoenv's (``_set_obs_cards`` in ``ygoenv/ygopro/ygopro.h``):

* the opponent's deck, hand and extra deck become counts -- the cards are
  present as anonymous slots, their codes are dropped;
* the opponent's field, graveyard and banished pile are public, except that a
  face-down card keeps its position and loses its code;
* our own deck is known as a *multiset* but not as an order, so it is sorted
  into a canonical order before it is written out.  Leaving the true order in
  would hand the model next turn's draw.

``leaks`` re-derives the violations from a masked snapshot and is what the leak
assertion in the test suite calls.

**Diff.**  What the settlement head predicts.  Snapshot subtraction gives the
*result*; the ``MSG_MOVE`` / ``MSG_POS_CHANGE`` stream over the same interval
gives the *provenance* -- which card went where and why.  Both are kept: the
move log is the compact target, and replaying it onto the before-snapshot has
to reproduce the after-snapshot's occupancy, which is the round-trip test.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace

from ..netduel import constants as C
from ..netduel.board import parse_query_segments
# The public identity ledger lives in netduel: this module already depends on netduel.board, so the reverse import would be circular.
from ..netduel.disclosure import DisclosureLedger, resolve_disclosure

__all__ = [
    "Attack",
    "CardState",
    "ChainEvent",
    "ChainTarget",
    "Draw",
    "LpEvent",
    "CANONICAL_HIDDEN_FOR_OPPONENT",
    "HIDDEN_FOR_OPPONENT",
    "LOCATION_NAMES",
    "Move",
    "PublicEvent",
    "PUBLIC_LOCATIONS",
    "STATUS_DISABLED",
    "StateDiff",
    "StateSnapshot",
    "ZONES",
    "capture",
    "diff_snapshots",
    "leaks",
    "supervision_leaks",
    "identity_visible",
    "observer_transition_code",
    "parse_public_events",
    "DisclosureLedger",
    "resolve_disclosure",
    "roundtrip_occupancy",
]

#: ``common.h``: the card's own effects are negated
STATUS_DISABLED = 0x0001

#: everything ``card::get_infos`` can write except QUERY_REASON_CARD, which is
#: a pointer and means nothing outside the core
FULL_QUERY_FLAGS = 0xFFDFFF

ZONES = (
    C.LOCATION_DECK,
    C.LOCATION_HAND,
    C.LOCATION_MZONE,
    C.LOCATION_SZONE,
    C.LOCATION_GRAVE,
    C.LOCATION_REMOVED,
    C.LOCATION_EXTRA,
)

#: zones of the opponent whose contents are simply not known
HIDDEN_FOR_OPPONENT = frozenset(
    {C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_EXTRA}
)

#: Zones whose cards the opponent's side reports only as a count. The hand is not one of them: the client keeps its order from messages.
#: The **sequence itself** of these two zones is invisible to the other side (``RefreshExtra`` is sent only to the deck's owner,
#: and the opponent's deck is never a zone the client holds), so the engine's real indices must not enter the input.
#: Our own deck was canonicalized for the same reason long ago (see the end of :func:`mask_for`).
COUNT_ONLY_FOR_OPPONENT = frozenset({C.LOCATION_DECK, C.LOCATION_EXTRA})

# Opponent hand order does not carry identity after a hand shuffle.  A viewer
# retains the multiset of card names that were actually disclosed, but cannot
# attach those names to the engine's private vector slots.  Canonicalizing all
# three zones makes that information state reproducible from the wire.
CANONICAL_HIDDEN_FOR_OPPONENT = COUNT_ONLY_FOR_OPPONENT | frozenset(
    {C.LOCATION_HAND}
)

#: zones of the opponent that are public, subject to face-down cards
PUBLIC_LOCATIONS = frozenset(
    {C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_GRAVE, C.LOCATION_REMOVED}
)

#: Zones whose face-up occupants are common knowledge to *both* players.
#: ``LOCATION_EXTRA`` is here only through the face-up pendulum rule below,
#: and overlay materials are public because their pile is.
PUBLIC_EFFECT_ORIGIN_LOCATIONS = PUBLIC_LOCATIONS | frozenset(
    {C.LOCATION_EXTRA, C.LOCATION_OVERLAY}
)

LOCATION_NAMES = {
    C.LOCATION_DECK: "deck",
    C.LOCATION_HAND: "hand",
    C.LOCATION_MZONE: "mzone",
    C.LOCATION_SZONE: "szone",
    C.LOCATION_GRAVE: "grave",
    C.LOCATION_REMOVED: "removed",
    C.LOCATION_EXTRA: "extra",
    C.LOCATION_OVERLAY: "overlay",
}


def location_name(location: int) -> str:
    return LOCATION_NAMES.get(location, hex(location))


@dataclass
class CardState:
    """One card instance, as the engine reports it."""

    controller: int
    location: int
    sequence: int
    code: int = 0
    position: int = 0
    type: int = 0
    level: int = 0
    rank: int = 0
    attribute: int = 0
    race: int = 0
    attack: int = 0
    defense: int = 0
    status: int = 0
    lscale: int = 0
    rscale: int = 0
    link: int = 0
    link_marker: int = 0
    overlay: tuple[int, ...] = ()
    counters: tuple[tuple[int, int], ...] = ()
    # -- completion: the fields below have **always been parsed** by `parse_query_segments`,
    # but used to be dropped at the CardState layer (the same family as `ov_code` being written and never read,
    # the fourth case of "read but not passed on"). No change on the engine side, only wiring.
    #: second card name (`QUERY_ALIAS`). Effects that name a card judge by both names.
    alias: int = 0
    #: **original** ATK/DEF (`QUERY_BASE_*`). The difference from `attack`/`defense` = the modifier;
    #: an "ATK reduction" needs this pair, since the current value alone cannot tell an original 1800 from a reduced 2800.
    base_attack: int = 0
    base_defense: int = 0
    #: owner (`QUERY_OWNER`), different from `controller`: after a control change a leaving card returns to its owner.
    owner: int = -1
    #: equip relation: whom this card is equipped to (the packed location of `QUERY_EQUIP_CARD`).
    equip_card: int = 0
    #: locations targeted by this card (its effect) (`QUERY_TARGET_CARD`).
    #: The persistent "targeted" state of relation checks lands here.
    targets: tuple[int, ...] = ()
    #: archetype codes. **Not from an engine query**: static card-table data
    #: injected from `cards.cdb`; modern support between cards works by archetype, and without it the model is blind to mainstream builds.
    setcode: tuple[int, ...] = ()
    #: masking stripped this card's identity
    hidden: bool = False
    #: Safe lower bound that both seats know this identity.  It is true only in
    #: the non-owner viewer's snapshot when that exact instance is in this
    #: viewer's audience-filtered ``revealed`` set.  Own cards always write
    #: false: an owner-only message sent to the opponent is not observable here.
    public: bool = False

    @property
    def face_down(self) -> bool:
        return bool(self.position & C.POS_FACEDOWN)

    @property
    def negated(self) -> bool:
        return bool(self.status & STATUS_DISABLED)

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.controller, self.location, self.sequence)

    def as_row(self) -> dict:
        return {
            "controller": self.controller,
            "location": self.location,
            "sequence": self.sequence,
            "code": self.code,
            "position": self.position,
            "type": self.type,
            "level": self.level,
            "rank": self.rank,
            "attribute": self.attribute,
            "race": self.race,
            "attack": self.attack,
            "defense": self.defense,
            "status": self.status,
            "lscale": self.lscale,
            "rscale": self.rscale,
            "link": self.link,
            "link_marker": self.link_marker,
            "overlay": list(self.overlay),
            "negated": self.negated,
            "hidden": self.hidden,
        }


_CARD_COPY_FIELDS = frozenset(item.name for item in fields(CardState) if item.init)


def _replace_card(card: CardState, **changes) -> CardState:
    """Copy plain card rows without rediscovering each dataclass field.

    Keep constructor semantics and a distinct row for every observer. Richer
    subclasses, extra instance attributes, and incomplete rows retain the
    standard dataclass path; no visibility decision or card value is cached.
    """
    if type(card) is CardState and vars(card).keys() == _CARD_COPY_FIELDS:
        return CardState(**(vars(card) | changes))
    return replace(card, **changes)


@dataclass
class StateSnapshot:
    """The board at one instant."""

    turn: int
    turn_player: int
    phase: int
    lp: tuple[int, int]
    cards: list[CardState] = field(default_factory=list)
    #: (player, location) -> number of cards, including hidden ones
    counts: dict[tuple[int, int], int] = field(default_factory=dict)
    #: the player this snapshot has been masked for; ``None`` = raw engine view
    view: int | None = None
    #: the set of "disclosed card instances" used when masking, carried with the snapshot,
    #: so ``leaks`` need not take it from outside again; otherwise a caller forgetting to pass it would
    #: report legally visible activated cards as leaks.
    revealed: frozenset[tuple] = frozenset()
    #: cards the observer knows **which** they are but not **in which slot**, one entry each
    #: ``(controller, zone, code)``. After a hand shuffle or a set-card shuffle positions mean nothing, but "they have
    #: this card" still holds; this part used to be pinned to the lowest slots in ascending sequence order, which was
    #: made-up position. "What" and "where" are now expressed separately: ``revealed`` holds those with positions,
    #: this list those without.
    revealed_unpositioned: tuple[tuple[int, int, int], ...] = ()
    #: optional exact ygo-core state dump; raw captures are filtered by
    #: ``mask_for`` before they can reach the training writer.
    core_info: dict | None = None

    def by_key(self) -> dict[tuple[int, int, int], CardState]:
        return {c.key: c for c in self.cards}

    def zone(self, player: int, location: int) -> list[CardState]:
        return [
            c for c in self.cards
            if c.controller == player and c.location == location
        ]


def capture(driver, core_info=None) -> StateSnapshot:
    """Read the whole board out of a live duel."""
    core, pduel = driver.core, driver.pduel
    buf = getattr(driver, "_querybuf", None)
    if buf is None:
        import ctypes

        from ..puzzle.core import SIZE_QUERY_BUFFER

        buf = ctypes.create_string_buffer(SIZE_QUERY_BUFFER * 4)
        driver._querybuf = buf

    lp = _read_lp(driver)
    cards: list[CardState] = []
    counts: dict[tuple[int, int], int] = {}
    for player in (0, 1):
        for location in ZONES:
            length = core.query_field_card(
                pduel, player, location, FULL_QUERY_FLAGS, buf
            )
            segments = parse_query_segments(buf.raw[:length])
            n = 0
            for sequence, fields in enumerate(segments):
                if not fields or not fields.get("code"):
                    continue
                n += 1
                cards.append(_card_from_fields(player, location, sequence, fields))
            counts[(player, location)] = n
    return StateSnapshot(
        turn=driver.turn,
        turn_player=driver.turn_player,
        phase=driver.phase,
        lp=lp,
        cards=cards,
        counts=counts,
        core_info=(core_info.as_dict() if hasattr(core_info, "as_dict")
                   else core_info),
    )


def _read_lp(driver) -> tuple[int, int]:
    import ctypes

    from ..puzzle.core import SIZE_QUERY_BUFFER

    buf = getattr(driver, "_infobuf", None)
    if buf is None:
        buf = ctypes.create_string_buffer(SIZE_QUERY_BUFFER)
        driver._infobuf = buf
    length = driver.core.query_field_info(driver.pduel, buf)
    raw = buf.raw[:length]
    # MSG_RELOAD_FIELD id, duel rule, then per player: lp, zones, counts
    pos = 2
    lp = []
    for _ in range(2):
        (value,) = struct.unpack_from("<i", raw, pos)
        lp.append(value)
        pos += 4
        for _ in range(7):
            pos += 3 if raw[pos] else 1
        for _ in range(8):
            pos += 2 if raw[pos] else 1
        pos += 6
    return (lp[0], lp[1])


def _card_from_fields(player: int, location: int, sequence: int, f: dict) -> CardState:
    info_location = f.get("info_location", 0)
    counters = tuple(sorted((f.get("counters") or {}).items()))
    return CardState(
        controller=player,
        location=location,
        sequence=sequence,
        code=f.get("code", 0),
        # get_public_info_location packs position into the top byte; POS_REVEAL
        # can be OR-ed in above it, hence the mask
        position=(info_location >> 24) & 0x0F,
        type=f.get("type", 0),
        level=f.get("level", 0),
        rank=f.get("rank", 0),
        attribute=f.get("attribute", 0),
        race=f.get("race", 0),
        attack=f.get("attack", 0),
        defense=f.get("defense", 0),
        status=f.get("status", 0),
        lscale=f.get("lscale", 0),
        rscale=f.get("rscale", 0),
        link=f.get("link", 0),
        link_marker=f.get("link_marker", 0),
        overlay=tuple(f.get("overlay") or ()),
        counters=counters,
        alias=f.get("alias", 0),
        base_attack=f.get("base_attack", 0),
        base_defense=f.get("base_defense", 0),
        owner=f.get("owner", -1),
        equip_card=f.get("equip_card", 0),
        targets=tuple(f.get("targets") or ()),
    )


# -- masking ---------------------------------------------------------------


def _blank(card: CardState, keep_position: bool) -> CardState:
    return CardState(
        controller=card.controller,
        location=card.location,
        sequence=card.sequence,
        code=0,
        position=card.position if keep_position else 0,
        # Ownership is public rule state and does not reveal the card's
        # identity.  It matters after a known control change, when leaving the
        # field sends the card back to its owner rather than its controller.
        owner=card.owner,
        hidden=True,
    )


def _canonical_hidden_row(card: CardState, visible: bool) -> CardState:
    """Erase every per-instance field not fixed by a public hidden multiset."""

    if not visible:
        return CardState(
            controller=card.controller,
            location=card.location,
            sequence=0,
            code=0,
            position=0,
            owner=-1,
            hidden=True,
        )
    faceup_extra = (
        card.location == C.LOCATION_EXTRA and not card.face_down
    )
    return CardState(
        controller=card.controller,
        location=card.location,
        sequence=0,
        code=card.code,
        position=(
            C.POS_FACEUP_DEFENSE if faceup_extra
            else C.POS_FACEDOWN_DEFENSE
        ),
        owner=-1,
        hidden=False,
        public=bool(card.public or faceup_extra),
    )


def identity_visible(
    card: CardState,
    player: int,
    revealed: frozenset[tuple] | set[tuple] | None = None,
) -> bool:
    """Whether the **identity (code)** of this card is visible to ``player``.

    **The observation side (``mask_for``) and the loss side (``supervision_leaks``) share this one predicate.**
    Two copies of the visibility rules are where this corpus drifts most easily and silently: change the observation and forget the loss,
    and the model is supervised to predict what it cannot see, **without any error**;
    once only one of the two branches of ``leaks()`` was changed, and the "activation is public" fix was undone by itself.
    To change visibility, **change only this**.
    """
    revealed = revealed or frozenset()
    if card.controller == player:
        return True
    # **This very card** was disclosed: match by instance, not by code; matching by code would expose same-named cards in the deck too.
    if card.code and (
        card.controller, card.location, card.sequence, card.code
    ) in revealed:
        return True
    if card.location in HIDDEN_FOR_OPPONENT:
        # A pendulum monster leaving the field for the graveyard goes **face-up** to the Extra Deck, and face-up means public.
        return bool(
            card.location == C.LOCATION_EXTRA
            and not card.face_down
            and card.type & C.TYPE_PENDULUM
        )
    return not card.face_down


#: The fields one ``query_effect_info`` effect row uses per card reference.
#: Zeroing a whole group is how a non-public reference leaves the block
#: without leaving a half-filled coordinate behind.
EFFECT_REFERENCE_FIELDS = {
    "owner": (
        "owner_info_location", "owner_controler", "owner_location",
        "owner_sequence", "owner_position", "owner_code",
    ),
    "handler": (
        "handler_info_location", "handler_controler", "handler_location",
        "handler_sequence", "handler_position", "handler_code",
    ),
    "label_object": (
        "label_object_info_location", "label_object_type",
        "label_object_controler", "label_object_location",
        "label_object_sequence", "label_object_position", "label_object_code",
    ),
}


def public_effect_origin(
    controller: int,
    location: int,
    sequence: int,
    position: int,
    code: int,
    revealed: frozenset[tuple] | set[tuple] | None = None,
) -> bool:
    """Is a registered effect's origin card public knowledge to *both* seats?

    This is deliberately stricter than :func:`identity_visible`, which answers
    "may this viewer know it".  The engine-derived active-effect block is a
    *public* state group, and the self-play god-view core registers effects a
    deployment client could never learn about -- a hand trap's condition
    effect, an unactivated set card's aura.  Admitting those on the strength
    of "the viewer happens to own that hand" would make self-play inputs
    richer than online inputs for one seat only, so the block asks a single
    seat-independent question instead: is the card itself face-up in a public
    zone, or has this exact instance been revealed by a resolved public chain?

    ``LOCATION_EXTRA`` counts only for a face-up pendulum card, matching
    :func:`identity_visible`; a face-down extra deck is not public.
    """

    if not code:
        return False
    revealed = revealed or frozenset()
    if (controller, location, sequence, code) in revealed:
        return True
    # ``mask_for`` canonicalizes slots inside count-only zones, so a revealed
    # card there is addressed by identity rather than by its engine slot.
    if location in CANONICAL_HIDDEN_FOR_OPPONENT and any(
        row[0] == controller and row[1] == location and row[3] == code
        for row in revealed
    ):
        return True
    if location not in PUBLIC_EFFECT_ORIGIN_LOCATIONS:
        return False
    return not bool(position & C.POS_FACEDOWN)


def _effect_reference_is_public(row, prefix: str, revealed) -> bool:
    """Is one named card reference of an effect row public knowledge?

    An absent reference (all zeros) is vacuously fine: the core simply did not
    name a card there.
    """

    packed = int(row.get(f"{prefix}_info_location", 0) or 0)
    code = int(row.get(f"{prefix}_code", 0) or 0)
    if not packed and not code:
        return True
    return public_effect_origin(
        controller=packed & 0xFF,
        location=(packed >> 8) & 0xFF,
        sequence=(packed >> 16) & 0xFF,
        position=(packed >> 24) & 0x0F,
        code=code,
        revealed=revealed,
    )


def _effect_origin_is_public(row: dict, revealed) -> bool:
    """Apply :func:`public_effect_origin` to one ``query_effect_info`` row.

    The handler is the card the effect is *applied from*; when the core did
    not name one -- a continuous effect still sitting on its source -- the
    owner card is the origin.  A row that names no card at all (a player-level
    flag effect) has no public evidence to point at and is excluded; the
    once-per-turn accounting it usually stands for lives in the dump's
    ``count_codes`` table, not in this block.
    """

    for prefix in ("handler", "owner"):
        packed = int(row.get(f"{prefix}_info_location", 0) or 0)
        code = int(row.get(f"{prefix}_code", 0) or 0)
        if not packed and not code:
            continue
        return public_effect_origin(
            controller=packed & 0xFF,
            location=(packed >> 8) & 0xFF,
            sequence=(packed >> 16) & 0xFF,
            position=(packed >> 24) & 0x0F,
            code=code,
            revealed=revealed,
        )
    return False


def observer_transition_code(
    kind: str,
    payload_code: int,
    viewer: int,
    to_at: tuple[int, int, int, int],
    *,
    already_routed: bool,
) -> int:
    """Sanitize a god-view transition code exactly as ``single_duel.cpp``.

    This helper reads no before/after snapshot.  Offline prompt intervals can
    contain earlier confirms and later UPDATE_DATA packets, so either snapshot
    would assign evidence to the wrong message time.  Online already receives
    this routing-sanitized payload; applying the same deterministic rule to the
    offline raw payload makes the two history streams identical.
    """

    if kind not in ("move", "position"):
        raise ValueError(f"unknown transition kind {kind!r}")
    if viewer not in (0, 1):
        raise ValueError("transition viewer must be seat zero or one")

    code = int(payload_code or 0) & 0x7FFFFFFF
    if not code:
        return 0
    if kind == "position":
        # MSG_POS_CHANGE is broadcast without rewriting its code field.
        return code
    if already_routed:
        # A network client receives the post-ShouldHideFacedownCode payload;
        # POS_REVEAL was already consumed and stripped by the host.
        return code
    controller, location, _sequence, position = to_at
    if controller == viewer:
        return code
    hidden_facedown = bool(position & C.POS_FACEDOWN) and not bool(
        position & C.POS_REVEAL
    )
    if not (location & (C.LOCATION_GRAVE | C.LOCATION_OVERLAY)) and (
        location & (C.LOCATION_DECK | C.LOCATION_HAND) or hidden_facedown
    ):
        return 0
    return code


def observer_event_code(
    kind: int,
    payload_code: int,
    viewer: int,
    at: int,
    *,
    already_routed: bool,
) -> int:
    """A public event's code as ``single_duel.cpp`` routes it to ``viewer``.

    Only ``MSG_SPSUMMONING`` is rewritten: the non-controller's copy loses a
    face-down summon's code unless the core marked it ``POS_REVEAL``
    (``netduel/host_view.py`` ``should_hide_facedown_code``). Normal and flip
    summons are broadcast unrewritten and are never face-down. A network
    stream already carries the routed code, with ``POS_REVEAL`` stripped.
    """

    if viewer not in (0, 1):
        raise ValueError("event viewer must be seat zero or one")
    code = int(payload_code or 0) & 0x7FFFFFFF
    if already_routed or not code or int(kind) != PUB_SPSUMMONING:
        return code
    controller, position = int(at) & 0xFF, (int(at) >> 24) & 0xFF
    if controller == viewer:
        return code
    if position & C.POS_FACEDOWN and not position & C.POS_REVEAL:
        return 0
    return code


def public_view(snapshot: StateSnapshot, viewer: int, ledger) -> StateSnapshot:
    """Ledger + raw snapshot -> this observer's public snapshot.

    Masking needs **both** outputs of the ledger, and missing one loses information: identities with positions go through ``revealed``,
    those without through ``revealed_unpositioned``. The two calls are bound together so a caller cannot take
    only one of them.
    """

    revealed = ledger.resolve(snapshot, viewer)
    return mask_for(
        snapshot, viewer, revealed,
        unpositioned=ledger.unanchored_identities(viewer, revealed),
    )


def _unpositioned_rows(
    counts, viewer: int, visible_unplaced=None,
) -> tuple[tuple[int, int, int], ...]:
    """Spread ``(controller, zone, code) -> count`` into one entry per card, sorted into a fixed order.

    The observer's **own** cards do not go into this list: they are visible card by card anyway, and repeating them would make
    downstream count the same card twice. This list only expresses "of the opponent's cards I know which, but not in which slot".

    **Cards the observer already sees directly in some slot do not go into this list either.** The ledger's multiset
    lower bound also contains cards the observer can see anyway: for a face-up continuous spell in the opponent's spell & trap zone,
    ``resolve`` need not give it a slot (it sits there itself), so the whole card would fall into "known what,
    unknown where". That is double counting: the masked card list already has a slot token with the real code,
    and an extra unpositioned identity token would add a card from nothing and break the tensorization invariant
    "slot tokens == real card count, plus k unpositioned identities, k ≤ hidden slots of the zone".
    ``visible_unplaced`` is the count ``mask_for`` finds of "the opponent's cards with visible identity that were **not**
    placed by the ledger (``revealed``)": ``unanchored_identities`` already subtracted the cards the ledger placed,
    so only the rest is subtracted here. Measured on 2026-09-03, 0.30% of zone calls over-reported on an evaluation set.
    """

    visible = dict(visible_unplaced or {})
    rows: list[tuple[int, int, int]] = []
    for (controller, location, code), copies in (counts or {}).items():
        if int(controller) == int(viewer):
            continue
        if int(location) & C.LOCATION_OVERLAY:
            # Xyz materials hang under a face-up host; they are public card by card and take no slot of their own.
            continue
        key = (int(controller), int(location), int(code))
        spare = max(0, int(copies)) - int(visible.get(key, 0))
        rows.extend([key] * max(0, spare))
    rows.sort()
    return tuple(rows)


def mask_for(
    snapshot: StateSnapshot,
    player: int,
    revealed: frozenset[tuple] | set[tuple] | None = None,
    *,
    unpositioned=None,
) -> StateSnapshot:
    """The part of ``snapshot`` that ``player`` is entitled to see.

    Visibility is **conditioned on events**, not only on location. The moment a card is activated
    (``MSG_CHAINING``) it is public information, and from then on the opponent of course knows what it is,
    even if it is still in the hand. ``revealed`` is the set of these "disclosed" codes.

    This used to be missing: ``HIDDEN_FOR_OPPONENT`` judged by location only, and a card the opponent activated from the hand
    (a hand trap, say) is still at ``LOCATION_HAND``, so the whole card was erased. Measured on a corpus,
    **40.2%** of the opponent's chain links had no source card in our state block, against 0.5% on our side.
    The result: at resolution we could see what the opponent activated, **but not when deciding in the response window**,
    which is exactly when "should I chain to it" is judged.

    This does not loosen the information state, it **corrects** it: activation being public is the rule itself;
    hiding public information and leaking hidden information are both wrong observations.
    Undisclosed cards are zeroed as before.
    """
    revealed = revealed or frozenset()
    out: list[CardState] = []
    own_deck: list[CardState] = []
    opponent_unordered: dict[tuple[int, int], list[CardState]] = {}
    visible_unplaced: dict[tuple[int, int, int], int] = {}
    for card in snapshot.cards:
        opponent = card.controller != player
        exact_reveal = (
            card.controller,
            card.location,
            card.sequence,
            card.code,
        ) in revealed
        card = _replace_card(
            card,
            public=bool(card.controller != player and exact_reveal),
        )
        # Identity visibility **asks only identity_visible**; this only decides how to blank an invisible card.
        visible = (not opponent) or identity_visible(card, player, revealed)
        if opponent and visible and card.code and not exact_reveal:
            # the opponent's, identity visible, and not placed by the ledger: position known, not in the unpositioned list
            key = (int(card.controller), int(card.location), int(card.code))
            visible_unplaced[key] = visible_unplaced.get(key, 0) + 1
        if opponent and card.location in CANONICAL_HIDDEN_FOR_OPPONENT:
            # The identity stays or goes by identity_visible; the sequence never stays: that index is
            # the engine's internal order, invisible to the other side, and `reset_sequence` changes it
            # silently. Canonicalization happens below in one place.
            opponent_unordered.setdefault(
                (card.controller, card.location), []
            ).append(_canonical_hidden_row(card, visible))
            continue
        if opponent and not visible:
            # In the hand even the position means nothing; set cards on the field keep their positions (the opponent knows a card is there).
            out.append(_blank(
                card,
                keep_position=card.location not in HIDDEN_FOR_OPPONENT,
            ))
            continue
        if not opponent and card.location == C.LOCATION_DECK:
            own_deck.append(card)
            continue
        out.append(card)
    # the opponent's count-only zones: whether an identity survived is decided
    # above, but the slot never survives.  Named cards take the low slots in a
    # canonical (code, position) order and the unknown remainder follows, which
    # is the assignment a client -- holding a multiset and a count, never an
    # index -- can also produce.  The zone size is unchanged, so OPPCNT still
    # counts every row.
    for key in sorted(opponent_unordered):
        zone = opponent_unordered[key]
        named = sorted(
            (c for c in zone if c.code), key=lambda c: (c.code, c.position)
        )
        unknown = [c for c in zone if not c.code]
        for i, card in enumerate(named + unknown):
            out.append(_replace_card(card, sequence=i))
    # our own deck: the contents are tracked, the shuffle order is not, so the
    # sequence numbers are reassigned in a canonical (code-sorted) order
    for i, card in enumerate(sorted(own_deck, key=lambda c: (c.code, c.position))):
        out.append(
            CardState(
                controller=card.controller,
                location=card.location,
                sequence=i,
                code=card.code,
                position=0,
                type=card.type,
            )
        )
    # ``core_info`` still names the raw engine slot inside the zones the loop
    # above canonicalized.  A public effect reference into one of them is
    # admitted by identity (:func:`public_effect_origin`), so it is rewritten
    # onto the canonical row that identity now occupies and the raw slot never
    # leaves.
    effect_slots = canonical_hidden_slots(out, player)
    revealed_identities = {
        (controller, location, code)
        for controller, location, _sequence, code in revealed
    }
    normalized_revealed = frozenset(
        (card.controller, card.location, card.sequence, card.code)
        for card in out
        if card.code and (
            (
                card.location in CANONICAL_HIDDEN_FOR_OPPONENT
                and (
                    (card.controller, card.location, card.code)
                    in revealed_identities
                    or (
                        card.location == C.LOCATION_EXTRA
                        and not card.face_down
                    )
                )
            )
            or (
                card.location not in CANONICAL_HIDDEN_FOR_OPPONENT
                and (
                    card.controller,
                    card.location,
                    card.sequence,
                    card.code,
                ) in revealed
            )
        )
    )
    return StateSnapshot(
        turn=snapshot.turn,
        turn_player=snapshot.turn_player,
        phase=snapshot.phase,
        lp=snapshot.lp,
        cards=out,
        counts=dict(snapshot.counts),
        view=player,
        revealed=normalized_revealed,
        revealed_unpositioned=_unpositioned_rows(
            unpositioned, player, visible_unplaced),
        core_info=_mask_core_info(
            snapshot.core_info,
            snapshot,
            player,
            normalized_revealed,
            effect_slots,
        ),
    )


def _normalized_effect_row(row) -> dict:
    """Give an effect row its packed ``*_info_location`` fields.

    ``RegisteredEffect.as_dict`` already carries them; a hand-built row or a
    row round-tripped through a narrower writer may only carry the components.
    """

    out = dict(row)
    for prefix in EFFECT_REFERENCE_FIELDS:
        key = f"{prefix}_info_location"
        if out.get(key):
            continue
        out[key] = (
            int(out.get(f"{prefix}_controler", 0) or 0)
            | (int(out.get(f"{prefix}_location", 0) or 0) << 8)
            | (int(out.get(f"{prefix}_sequence", 0) or 0) << 16)
            | (int(out.get(f"{prefix}_position", 0) or 0) << 24)
        )
    return out


def canonical_hidden_slots(
    cards, player: int
) -> dict[tuple[int, int, int], int]:
    """Canonical slot per identity inside the opponent's canonicalized zones.

    :func:`mask_for` builds this from the rows it just canonicalized so effect
    references can be placed on them; :func:`leaks` rebuilds it from a masked
    snapshot to audit that they were.  The lowest slot wins: duplicates of one
    code are addressed by identity, which is the only thing the slot in those
    zones ever stood for.
    """

    slots: dict[tuple[int, int, int], int] = {}
    for card in cards:
        if (
            card.code
            and card.controller != player
            and card.location in CANONICAL_HIDDEN_FOR_OPPONENT
        ):
            slots.setdefault(
                (card.controller, card.location, card.code), card.sequence
            )
    return slots


#: :func:`_required_effect_slot` answer for a reference outside the zones
#: :func:`mask_for` canonicalizes; it keeps the coordinate it came with.
_KEEP_EFFECT_SLOT = -1


def _required_effect_slot(
    row, prefix: str, player: int, slots
) -> int | None:
    """The slot one effect reference must name.

    :data:`_KEEP_EFFECT_SLOT` when the reference is outside the canonicalized
    hidden zones -- the opponent's hand, deck and extra deck are the only
    places whose slot is not addressable by both seats.  ``None`` when it does
    point into one of them and no masked row carries that identity: fail-closed
    for the masker, a leak for the audit.
    """

    packed = int(row.get(f"{prefix}_info_location", 0) or 0)
    controller = packed & 0xFF
    location = (packed >> 8) & 0xFF
    if controller == player or location not in CANONICAL_HIDDEN_FOR_OPPONENT:
        return _KEEP_EFFECT_SLOT
    return slots.get(
        (controller, location, int(row.get(f"{prefix}_code", 0) or 0))
    )


def _canonicalize_effect_slot(
    row: dict, prefix: str, player: int, slots
) -> bool:
    """Move one effect reference off the raw engine slot, in place.

    A reference admitted inside a canonicalized zone was admitted by identity
    (:func:`public_effect_origin`), so it must name the canonical row that
    identity was given rather than the engine's private index.  Returns
    ``False`` for the fail-closed case, where the caller drops the row rather
    than publish a raw hidden slot.
    """

    sequence = _required_effect_slot(row, prefix, player, slots)
    if sequence is None:
        return False
    if sequence == _KEEP_EFFECT_SLOT:
        return True
    packed = int(row.get(f"{prefix}_info_location", 0) or 0)
    row[f"{prefix}_info_location"] = (packed & ~(0xFF << 16)) | (sequence << 16)
    row[f"{prefix}_sequence"] = sequence
    return True


def _masked_effect_row(source, player: int, revealed, canonical_slots) -> dict | None:
    """One ``query_effect_info`` row as this seat may see it, or ``None``.

    Three rules, in order: the row survives only if its origin card is public
    (:func:`_effect_origin_is_public`); a named reference whose card is not
    public loses its whole field group; and a surviving reference into a
    canonicalized hidden zone is moved onto its canonical slot.  The last one
    is what lets a revealed opponent hand card carry its effects without its
    engine hand slot escaping alongside them.
    """

    row = _normalized_effect_row(source)
    if not _effect_origin_is_public(row, revealed):
        return None
    for prefix, keys in EFFECT_REFERENCE_FIELDS.items():
        if not _effect_reference_is_public(row, prefix, revealed):
            for key in keys:
                row[key] = 0
        elif not _canonicalize_effect_slot(row, prefix, player, canonical_slots):
            return None
    return row


def effect_block_leaks(effects, revealed, player: int, slots) -> list[str]:
    """Rows of an active-effect block whose evidence is a hidden card.

    Separated from :func:`leaks` so the tensorizer and the ledger audit can
    ask the same question of a block they built themselves.  ``slots`` is
    :func:`canonical_hidden_slots` over the rows the block belongs to: the
    audit re-checks both halves of :func:`_masked_effect_row`, that a named
    card is public and that a card inside a canonicalized zone is named at its
    canonical slot rather than at the engine's private index.
    """

    problems: list[str] = []
    for index, source in enumerate(effects or ()):
        row = _normalized_effect_row(source)
        if not _effect_origin_is_public(row, revealed):
            problems.append(
                f"active effect[{index}] id={row.get('id', '?')} has a hidden "
                f"origin (handler code {row.get('handler_code', 0)}, owner code "
                f"{row.get('owner_code', 0)})"
            )
            continue
        for prefix in EFFECT_REFERENCE_FIELDS:
            if not _effect_reference_is_public(row, prefix, revealed):
                problems.append(
                    f"active effect[{index}] names a hidden {prefix} card "
                    f"{row.get(f'{prefix}_code', 0)}"
                )
                continue
            required = _required_effect_slot(row, prefix, player, slots)
            packed = int(row.get(f"{prefix}_info_location", 0) or 0)
            if required is None or (
                required != _KEEP_EFFECT_SLOT
                and required != (packed >> 16) & 0xFF
            ):
                problems.append(
                    f"active effect[{index}] names {prefix} card "
                    f"{row.get(f'{prefix}_code', 0)} at raw hidden slot "
                    f"{(packed >> 16) & 0xFF}"
                )
    return problems


def _mask_core_info(
    info,
    snapshot: StateSnapshot,
    player: int,
    revealed,
    canonical_slots: Mapping[tuple[int, int, int], int] | None = None,
) -> dict | None:
    """Filter the raw core dump to the same identity view as the board.

    The core API is intentionally god-view (it is also used for diagnostics),
    so relation endpoints and source-card effects must be filtered before they
    enter a player's training sample.
    """
    if not info:
        return None
    info = info.as_dict() if hasattr(info, "as_dict") else dict(info)

    def visible(at: int, code: int) -> bool:
        at = int(at or 0)
        controller = at & 0xFF
        location = (at >> 8) & 0xFF
        if controller != player and location in CANONICAL_HIDDEN_FOR_OPPONENT:
            # The card rows were canonicalized above, but privileged core_info
            # still names raw engine slots.  Dropping these endpoints is safer
            # than pretending raw sequence 7 maps to canonical sequence 0.
            return False
        card = CardState(
            controller=controller,
            location=location,
            sequence=(at >> 16) & 0xFF,
            position=(at >> 24) & 0x0F,
            code=int(code or 0),
        )
        return identity_visible(card, player, revealed)

    # The effect block is *public* state, not per-viewer state: it is filtered
    # by :func:`public_effect_origin`, not by ``visible``.  A god-view engine
    # registers effects a deployment client could never observe (hand traps'
    # condition effects, unactivated set cards), and admitting those for the
    # seat that happens to own them would make self-play inputs asymmetric
    # with the online ones.  ``leaks`` re-checks exactly this rule.
    slots = {} if canonical_slots is None else canonical_slots
    effects = []
    for source in info.get("effects", ()):
        row = _masked_effect_row(source, player, revealed, slots)
        if row is not None:
            effects.append(row)

    out = dict(info)
    out["effects"] = effects
    out["cards"] = [
        row for row in info.get("cards", ())
        if visible(row.get("info_location", 0), row.get("code", 0))
    ]
    out["relations"] = [
        row for row in info.get("relations", ())
        if visible(row.get("source_info", 0), row.get("source_code", 0))
        and visible(row.get("target_info", 0), row.get("target_code", 0))
    ]
    out["setcodes"] = [
        row for row in info.get("setcodes", ())
        if visible(row.get("info_location", 0), row.get("code", 0))
    ]
    return out


def supervision_leaks(
    diff: "StateDiff",
    player: int,
    revealed: frozenset[tuple] | set[tuple] | None = None,
) -> list[str]:
    """Codes in the supervision targets that ``player`` should not know.

    **The observation mask and the loss mask must share one source**: this function shares
    :func:`identity_visible` with :func:`mask_for`. Resolution deltas are computed on the **raw (omniscient) snapshot**
    (``diff_snapshots`` takes the raw snapshot of ``capture()``), while observations are masked;
    with two different sources, "what the model cannot see" could become a supervision target,
    which is both an information leak and teaching the model to guess the unknowable.

    An empty list means every supervision target of this resolution lies within ``player``'s information state.
    """
    revealed = revealed or frozenset()
    problems: list[str] = []

    def _visible(controller, location, sequence, code, position=0, type_=0):
        # `face_down` is a property derived from position, not a field; pass position.
        return identity_visible(
            CardState(
                controller=controller, location=location, sequence=sequence,
                code=code, position=position, type=type_,
            ),
            player, revealed,
        )

    # **Check only the supervision surface that is really serialized**, not the omniscient intermediate `diff`.
    # Draws are deliberately not checked here: `diff.draws` carries codes, but the writer only writes
    # `dr_player`/`dr_count`/`dr_link` (writer.py:174-175, 355-357), and
    # **codes are never written**. Checking `diff` would report every opponent draw of every game as a leak;
    # the first version did exactly that and the tests immediately raised 5 false positives. A checker must judge
    # **the product**, not an intermediate, or the gate would reject a correct corpus forever.
    # Codes of moves are **already masked by viewpoint on the writing side**: `generate._move_code_visible` judges each one,
    # and `writer.mv_code` writes 0 for invisible ones, with the same `identity_visible` of this module.
    # So this **no longer judges `diff`**: `diff` is computed on the raw snapshot and always carries real codes,
    # and judging it would keep reporting "already fixed" things as leaks.
    #
    # This lesson appeared a third time in another guise: (1) checking the codes of `diff.draws` (the writer never writes them);
    # (2) taking the exemption set at the wrong moment; (3) the checker still watching an intermediate after the writer was fixed.
    # **A checker must judge the product.** `_visible` is kept for later product-based rechecks.
    del _visible
    return problems


#: Unpositioned identities can only be in these zones; Xyz materials hang under a face-up host and are public card by card.
_UNPOSITIONED_ZONES = frozenset({
    C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE,
    C.LOCATION_GRAVE, C.LOCATION_REMOVED, C.LOCATION_EXTRA,
})


def _unpositioned_leaks(snapshot: StateSnapshot) -> list[str]:
    """``revealed_unpositioned`` may hold only identities proven by the ledger, and never too many.

    This list bypasses per-card masking, so it needs its own assertion: every entry must be the **opponent's**
    card (one's own cards are visible anyway and need not take this list), in a legal zone, and the entries of one
    ``(controller, zone, code)`` may not exceed how many copies of that code the zone really has; over-reporting
    gives the observer a card from nothing.
    """

    viewer = snapshot.view
    rows = snapshot.revealed_unpositioned or ()
    problems: list[str] = []
    seen: dict[tuple[int, int, int], int] = {}
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 3 \
                or any(type(value) is not int for value in row):
            problems.append(f"revealed_unpositioned row is malformed: {row!r}")
            continue
        controller, location, code = row
        if controller not in (0, 1):
            problems.append(
                f"revealed_unpositioned names seat {controller}")
            continue
        if not code:
            problems.append("revealed_unpositioned carries an empty code")
            continue
        if location not in _UNPOSITIONED_ZONES:
            # Xyz materials and illegal zones should not appear in this list: saying which entry is much more useful
            # than letting it fall into the capacity error "this zone cannot hold it".
            problems.append(
                f"revealed_unpositioned location {location:#x} is not a zone")
            continue
        seen[(controller, location, code)] = \
            seen.get((controller, location, code), 0) + 1
    truth: dict[tuple[int, int, int], int] = {}
    for card in snapshot.cards:
        key = (int(card.controller), int(card.location), int(card.code))
        if card.code:
            truth[key] = truth.get(key, 0) + 1
    per_zone: dict[tuple[int, int], int] = {}
    for key, copies in seen.items():
        controller, location, code = key
        if controller == viewer:
            # one's own cards are visible card by card and should not take this list; one here means it was filled wrongly
            problems.append(
                f"revealed_unpositioned repeats the viewer's own card {code}")
            continue
        per_zone[(controller, location)] = \
            per_zone.get((controller, location), 0) + copies
        held = truth.get(key, 0)
        room = max(_hidden_copies(snapshot, controller, location),
                   int(snapshot.counts.get((controller, location), 0)))
        if copies > held + room:
            problems.append(
                f"revealed_unpositioned claims {copies} of {code} in "
                f"zone {location}, more than that zone holds")
    # Per-zone totals: k unpositioned identities can only sit in the zone's hidden slots; tensorization rejects
    # by this invariant, and the audit must report it first. A zone reporting only a count (the opponent's deck) has no card rows
    # after masking; its capacity comes from counts, the same calibration as above.
    for (controller, location), copies in per_zone.items():
        room = max(_hidden_copies(snapshot, controller, location),
                   int(snapshot.counts.get((controller, location), 0)))
        if copies > room:
            problems.append(
                f"revealed_unpositioned places {copies} identities in zone "
                f"{location} with only {room} hidden slots")
    return problems


def _hidden_copies(snapshot: StateSnapshot, controller: int, location: int) -> int:
    """The number of slots of the zone whose identity is masked; unpositioned identities can only sit in these."""

    return sum(1 for card in snapshot.cards
               if int(card.controller) == controller
               and int(card.location) == location and not card.code)


def leaks(
    snapshot: StateSnapshot,
    revealed: frozenset[tuple] | set[tuple] | None = None,
) -> list[str]:
    """Everything in a masked snapshot that its viewer must not know.

    Returns an empty list for a clean snapshot; the strings are meant to be
    read straight out of a failing assertion.

    ``revealed`` are the disclosed codes (activated, revealed). The assertion states the **correct** rule:
    blanking applies only to **undisclosed** cards; a disclosed card appearing with its identity in the opponent's view is not a leak,
    hiding it would be the wrong observation. Missing this would let the leak assertion undo the "activation is public" fix.
    """
    if snapshot.view is None:
        return ["snapshot has not been masked"]
    problems_unpositioned = _unpositioned_leaks(snapshot)
    if problems_unpositioned:
        return problems_unpositioned
    # A masked snapshot owns its canonical disclosure coordinates.  Accepting a
    # caller's earlier raw-slot set would undo HAND/DECK/EXTRA normalization.
    if revealed is not None and frozenset(revealed) != snapshot.revealed:
        return ["explicit revealed set differs from masked snapshot authority"]
    revealed = snapshot.revealed
    player = snapshot.view
    problems: list[str] = []
    seen_deck_order = []
    # ``mask_for`` canonicalizes the slot inside the opponent's count-only
    # zones, so the coordinates in ``revealed`` -- captured before that -- no
    # longer address the same rows.  In those zones the exemption is by
    # identity, which is the only thing the slot ever stood for; everywhere
    # else it stays the exact instance.
    revealed_identities = {
        (controller, location, code)
        for controller, location, _sequence, code in revealed
    }

    def exempt(card) -> bool:
        if card.location in CANONICAL_HIDDEN_FOR_OPPONENT:
            return (card.controller, card.location, card.code) in (
                revealed_identities
            )
        return (
            card.controller, card.location, card.sequence, card.code
        ) in revealed
    for card in snapshot.cards:
        opponent = card.controller != player
        if (
            opponent
            and card.location in HIDDEN_FOR_OPPONENT
            and card.code
            and not exempt(card)
            and not (
                card.location == C.LOCATION_EXTRA
                and not card.face_down
                and card.type & C.TYPE_PENDULUM
            )
        ):
            problems.append(
                f"opponent {location_name(card.location)}[{card.sequence}] "
                f"carries code {card.code}"
            )
        # A set card that **was disclosed** is not a leak: when a card is activated from the hand the engine's position
        # is still face-down, yet it has shown its face. This branch and the one above must use the same exemptions;
        # fixing only one would let the other undo the fix, which is exactly where it got stuck once.
        if (
            opponent
            and card.face_down
            and card.code
            and not exempt(card)
        ):
            problems.append(
                f"opponent face-down {location_name(card.location)}"
                f"[{card.sequence}] carries code {card.code}"
            )
        if not opponent and card.location == C.LOCATION_DECK:
            seen_deck_order.append(card.code)
    if seen_deck_order != sorted(seen_deck_order):
        problems.append("own deck is not in canonical order (shuffle order leaked)")
    if snapshot.core_info:
        problems.extend(
            effect_block_leaks(
                snapshot.core_info.get("effects", ()),
                revealed,
                player,
                canonical_hidden_slots(snapshot.cards, player),
            )
        )
    return problems


# -- diffs -----------------------------------------------------------------


@dataclass(frozen=True)
class Move:
    """One ``MSG_MOVE``: a card changed zone, slot or position."""

    code: int
    from_controller: int
    from_location: int
    from_sequence: int
    from_position: int
    to_controller: int
    to_location: int
    to_sequence: int
    to_position: int
    reason: int
    #: within the resolution window of which chain link this move happened. 0 = not in any link's window
    #: (the activation phase: flipping set cards face-up, paying costs and so on). See :func:`parse_moves`.
    link: int = 0
    trace_index: int = 0

    @property
    def appeared(self) -> bool:
        """Nothing was there before -- a token, or a card built out of nowhere."""
        return self.from_location == 0

    @property
    def vanished(self) -> bool:
        return self.to_location == 0

    def as_row(self) -> dict:
        return {
            "code": self.code,
            "from": [
                self.from_controller,
                self.from_location,
                self.from_sequence,
                self.from_position,
            ],
            "to": [
                self.to_controller,
                self.to_location,
                self.to_sequence,
                self.to_position,
            ],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Draw:
    """One ``MSG_DRAW``: cards left the deck for a hand.

    The core does *not* emit ``MSG_MOVE`` for a draw, so the move log alone
    cannot account for the deck and hand counts; every round-trip failure
    measured before this was added came from exactly this omission.  A draw is
    also a chance node -- which cards arrive is not a function of the state the
    model can see -- so it is kept as its own record rather than folded into
    the moves.
    """

    player: int
    codes: tuple[int, ...]
    link: int = 0
    trace_index: int = 0

    @property
    def count(self) -> int:
        return len(self.codes)


#: Activation negation can skip SOLVED (processor.cpp, solve_chain case 1).
#: Effect disabling still emits SOLVED; neither kind alone proves resolution.
CHAIN_CHAINING, CHAIN_SOLVED, CHAIN_NEGATED, CHAIN_DISABLED = 0, 1, 2, 3
#: The whole-chain END is a boundary even when a link's SOLVED was skipped.
CHAIN_SOLVING = 4

CHAIN_KIND_NAMES = ("chaining", "solved", "negated", "disabled", "solving")


@dataclass(frozen=True)
class ChainEvent:
    """A chain message.

    ``MSG_CHAINING`` (``processor.cpp:4068``) carries every field:
    the code, the packed location of ``get_info_location()``, the activating player, the effect description string and the link number.
    ``MSG_CHAIN_SOLVED`` / ``_NEGATED`` / ``_DISABLED`` carry only the link number;
    the other fields stay 0 and are matched to the earlier ``chaining`` by link number.
    """

    kind: int
    link: int
    code: int = 0
    at: int = 0
    desc: int = 0
    player: int = 0
    trace_index: int = 0


#: The four messages of LP changes have different meanings: battle damage / effect damage / recovery / paid cost.
#: Cards tell them apart ("take no battle damage"), so a resolution header recording only the difference cannot learn this layer.
#: The four bodies are the same (``player(u8) + amount(u32)``, ``operations.cpp:528/598/677``),
#: and the kind is told by the message number; battle damage and effect damage both use ``MSG_DAMAGE``
#: and are split again by "whether it falls between ``MSG_DAMAGE_STEP_START`` and ``_END``".
LP_DAMAGE, LP_RECOVER, LP_PAY_COST, LP_UPDATE = 0, 1, 2, 3
LP_KIND_NAMES = ("damage", "recover", "pay_cost", "update")


@dataclass(frozen=True)
class LpEvent:
    kind: int
    player: int
    amount: int
    link: int = 0
    #: inside the damage step (used to split battle damage from effect damage for ``MSG_DAMAGE``)
    in_damage_step: bool = False
    trace_index: int = 0


# Public result messages that are inputs, never settlement targets.  These ids
# form a compact model-side vocabulary independent of ygopro-core's MSG_* ids.
PUB_CONFIRM = 1
PUB_RANDOM = 2
PUB_COIN = 3
PUB_DICE = 4
PUB_ANNOUNCE_RACE = 5
PUB_ANNOUNCE_ATTRIB = 6
PUB_ANNOUNCE_CODE = 7
PUB_ANNOUNCE_NUMBER = 8
PUB_ANNOUNCE_ZONE = 9
PUB_OPSELECTED = 10
PUB_SHUFFLE_DECK = 11
PUB_SHUFFLE_HAND = 12
PUB_SHUFFLE_EXTRA = 13
PUB_SHUFFLE_SET = 14
PUB_HAND_RESULT = 15
PUB_DECK_TOP = 16
PUB_EQUIP = 17
PUB_UNEQUIP = 18
PUB_CARD_TARGET = 19
PUB_CANCEL_TARGET = 20
PUB_FIELD_DISABLED = 21
PUB_ADD_COUNTER = 22
PUB_REMOVE_COUNTER = 23
PUB_HINT_CARD = 24
PUB_CARD_HINT = 25
PUB_PLAYER_HINT = 26
PUB_SWAP_GRAVE_DECK = 27
PUB_REVERSE_DECK = 28
PUB_DRAW_REVEAL = 29
PUB_ATTACK_DISABLED = 30
PUB_MISSED_EFFECT = 31
PUB_SUMMONING = 32
PUB_SPSUMMONING = 33
PUB_FLIPSUMMONING = 34
PUB_SET = 35
PUB_NEW_TURN = 36
PUB_SUMMONED = 37
PUB_SPSUMMONED = 38
PUB_FLIPSUMMONED = 39
PUB_DAMAGE_STEP_START = 40


@dataclass(frozen=True)
class PublicEvent:
    """One result the network server makes visible after an action.

    ``audience`` is a two-bit seat mask (bit 0 = player 0, bit 1 = player 1).
    The in-process core exposes one omniscient message buffer, while the real
    server routes or masks some messages per seat.  Keeping that routing in the
    corpus prevents a seat-specific item from reading an opponent-only
    confirmation.
    """

    kind: int
    player: int = 2
    code: int = 0
    at: int = 0
    target: int = 0
    value: int = 0
    detail: int = 0
    link: int = 0
    audience: int = 3
    #: Ordinal in the action interval's raw core message stream.  This is the
    #: shared clock used to interleave public disclosures with private prompts.
    trace_index: int = 0


@dataclass(frozen=True)
class Attack:
    """One ``MSG_ATTACK``: who attacks whom.

    The body (``processor.cpp:2697``) = the attacker's ``get_info_location()``,
    followed by the target's when there is an attack target. A direct attack has no second field.

    The engine's ``query_field_card`` **cannot see** this: the battle context is processor state
    (``core.attacker`` / ``core.attack_target``), not a QUERY field of a card.
    A chain window in the damage step is a decision point, and then "who attacks whom" is information the player sees and must know,
    so it can only be recorded from the message stream.
    """

    attacker: int
    target: int = 0
    trace_index: int = 0


@dataclass(frozen=True)
class ChainTarget:
    """One ``MSG_BECOME_TARGET``: chain link ``link`` targeted some card.

    The body (three call sites such as ``libduel.cpp:3134``) is ``count(u8)`` plus count
    u32 packed locations of ``get_info_location()``; all three actually write one card.
    **The message has no code**: it gives a location reference, not an identity, so when the opponent's face-down card becomes a target
    what it is is naturally not exposed.

    Targeting and non-targeting are two mechanisms: a targeting effect whose target leaves before resolution fizzles,
    a non-targeting one does not. To learn the difference the model must see "which targets this link took".
    """

    link: int
    at: int
    trace_index: int = 0


@dataclass(frozen=True)
class PosChange:
    code: int
    controller: int
    location: int
    sequence: int
    previous: int
    current: int
    link: int = 0
    trace_index: int = 0

    def as_row(self) -> dict:
        return {
            "code": self.code,
            "at": [self.controller, self.location, self.sequence],
            "previous": self.previous,
            "current": self.current,
        }


@dataclass
class StateDiff:
    """What one action did to the board."""

    lp_delta: tuple[int, int] = (0, 0)
    moves: list[Move] = field(default_factory=list)
    pos_changes: list[PosChange] = field(default_factory=list)
    draws: list[Draw] = field(default_factory=list)
    #: cards whose own effects went from live to negated over the interval
    negated: list[tuple[int, int, int, int]] = field(default_factory=list)
    #: codes present afterwards that were nowhere before (tokens, mostly)
    appeared: list[int] = field(default_factory=list)
    vanished: list[int] = field(default_factory=list)
    #: (player, location) -> change in occupancy
    count_delta: dict[tuple[int, int], int] = field(default_factory=dict)
    #: the chain structure within the interval: who activated which effect at which link, which links were negated
    chain: list[ChainEvent] = field(default_factory=list)
    #: per-link targeting records. Whether a link "targets" is whether it has an entry here,
    #: without a separate column: when a derived column disagrees with the data there is nothing to decide by.
    targets: list[ChainTarget] = field(default_factory=list)
    #: attack declarations within the interval: the attacker and (optionally) the attack target
    attacks: list[Attack] = field(default_factory=list)
    #: LP events with reason semantics (`lp_delta` has only the net difference and cannot learn battle/effect/cost)
    lp_events: list[LpEvent] = field(default_factory=list)
    #: Server-visible disclosures.  They enter the token stream as inputs and
    #: are excluded from settlement loss.
    public_events: list[PublicEvent] = field(default_factory=list)
    turn_delta: int = 0
    phase_after: int = 0
    #: a random node fired inside the interval, so the result is not a function
    #: of the state and the action alone
    chance: bool = False
    chance_msgs: tuple[str, ...] = ()
    #: three-way: 0 deterministic / 1 true randomness / 2 hidden information. `chance` is its binary degenerate form,
    #: kept only for compatibility with older corpora.
    determinism: int = 0

    @property
    def empty(self) -> bool:
        return not (
            self.moves
            or self.draws
            or self.pos_changes
            or self.negated
            or self.public_events
            or any(self.count_delta.values())
            or self.lp_delta != (0, 0)
        )


def parse_chain(messages) -> list[ChainEvent]:
    """Read the chain structure of a message interval.

    Without this column, "which effects really took effect this turn" could only be approximated by "activated and the activating card
    did not get ``STATUS_DISABLED``", which cannot tell "the card's own effect was negated" from "this link was negated".
    """
    out: list[ChainEvent] = []
    for trace_index, message in enumerate(messages):
        if message.msg == C.MSG_CHAINING and len(message.payload) >= 16:
            code, at = struct.unpack_from("<II", message.payload, 0)
            player = message.payload[8]
            (desc,) = struct.unpack_from("<I", message.payload, 11)
            out.append(
                ChainEvent(
                    kind=CHAIN_CHAINING,
                    link=message.payload[15],
                    code=code,
                    at=at,
                    desc=desc,
                    player=player,
                    trace_index=trace_index,
                )
            )
        elif message.msg in _CHAIN_RESULT and message.payload:
            out.append(
                ChainEvent(kind=_CHAIN_RESULT[message.msg], link=message.payload[0],
                           trace_index=trace_index)
            )
    return out


_CHAIN_RESULT = {
    C.MSG_CHAIN_SOLVED: CHAIN_SOLVED,
    C.MSG_CHAIN_NEGATED: CHAIN_NEGATED,
    C.MSG_CHAIN_DISABLED: CHAIN_DISABLED,
    C.MSG_CHAIN_SOLVING: CHAIN_SOLVING,
}


def chain_link_after_message(
    link: int, msg: int, payload: bytes, *, include_chaining: bool = True,
) -> int:
    """Update public attribution without treating a negation as completion.

    Targets/LP/public events include activation callbacks; moves/draws/position
    changes use only resolution windows. END always clears both contexts,
    including an activation-negated link for which the core omitted SOLVED.
    """
    if msg in (C.MSG_CHAIN_SOLVED, C.MSG_CHAIN_END):
        return 0
    if msg == C.MSG_CHAIN_SOLVING and payload:
        return int(payload[0])
    if include_chaining and msg == C.MSG_CHAINING and len(payload) >= 16:
        return int(payload[15])
    return link


class PublicChainContext:
    """Chain attribution carried across the boundaries of parsed message intervals.

    The interval parsers start every interval at link 0 and outside a damage
    step. A network client instead keeps the active chain link, the settlement
    link and the damage-step flag from one packet to the next. An interval that
    begins inside a chain, such as the messages after a target chosen while a
    link resolves, therefore needs the carried state: :meth:`retag` gives every
    event of such an interval the attribution the client computes.
    """

    def __init__(self) -> None:
        self.chain_link = 0
        self.settlement_link = 0
        self.in_damage_step = False

    def retag(self, diff: "StateDiff", messages) -> "StateDiff":
        chain, settlement, damage = [], [], []
        for message in messages:
            msg, body = int(message.msg), bytes(message.payload)
            self.chain_link = chain_link_after_message(self.chain_link, msg, body)
            self.settlement_link = chain_link_after_message(self.settlement_link, msg, body, include_chaining=False)
            if msg == C.MSG_DAMAGE_STEP_START:
                self.in_damage_step = True
            chain.append(self.chain_link)
            settlement.append(self.settlement_link)
            damage.append(self.in_damage_step)
            if msg == C.MSG_DAMAGE_STEP_END:
                self.in_damage_step = False
            if msg in (C.MSG_START, C.MSG_NEW_TURN):
                # A chain or damage step cannot cross a game or turn boundary.
                self.chain_link = self.settlement_link = 0
                self.in_damage_step = False
        by_settlement = lambda rows: [replace(row, link=settlement[row.trace_index]) for row in rows]
        diff.moves, diff.pos_changes, diff.draws = (by_settlement(diff.moves), by_settlement(diff.pos_changes),
                                                   by_settlement(diff.draws))
        diff.public_events = [replace(row, link=chain[row.trace_index]) for row in diff.public_events]
        diff.targets = [replace(row, link=chain[row.trace_index]) for row in diff.targets]
        diff.lp_events = [replace(row, link=chain[row.trace_index], in_damage_step=damage[row.trace_index])
                          for row in diff.lp_events]
        return diff


def parse_targets(messages) -> list[ChainTarget]:
    """Per-link targeting records.

    Targets may be declared at two moments: when the link is built (the effect's target callback, right after
    ``MSG_CHAINING(k)``) or at resolution (a few effects target only after ``MSG_CHAIN_SOLVING(k)``).
    So the current link number is "the link number of the latest CHAINING or SOLVING",
    ``MSG_CHAIN_SOLVED`` or ``MSG_CHAIN_END`` clears the attribution.
    """
    out: list[ChainTarget] = []
    link = 0
    for trace_index, message in enumerate(messages):
        link = chain_link_after_message(link, message.msg, message.payload)
        if message.msg == C.MSG_BECOME_TARGET and message.payload:
            count = message.payload[0]
            for i in range(count):
                offset = 1 + i * 4
                if offset + 4 > len(message.payload):
                    break
                (at,) = struct.unpack_from("<I", message.payload, offset)
                out.append(ChainTarget(link=link, at=at,
                                       trace_index=trace_index))
    return out


_LP_MSG = {}


def _lp_msg_map():
    if not _LP_MSG:
        _LP_MSG.update({
            C.MSG_DAMAGE: LP_DAMAGE,
            C.MSG_RECOVER: LP_RECOVER,
            C.MSG_PAY_LPCOST: LP_PAY_COST,
            C.MSG_LPUPDATE: LP_UPDATE,
        })
    return _LP_MSG


def parse_lp_events(messages) -> list[LpEvent]:
    """LP changes with reasons and chain links."""
    table = _lp_msg_map()
    out: list[LpEvent] = []
    link = 0
    in_damage_step = False
    for trace_index, message in enumerate(messages):
        link = chain_link_after_message(link, message.msg, message.payload)
        if message.msg == C.MSG_DAMAGE_STEP_START:
            in_damage_step = True
        elif message.msg == C.MSG_DAMAGE_STEP_END:
            in_damage_step = False
        elif message.msg in table and len(message.payload) >= 5:
            (amount,) = struct.unpack_from("<I", message.payload, 1)
            out.append(
                LpEvent(kind=table[message.msg], player=message.payload[0],
                        amount=amount, link=link, in_damage_step=in_damage_step,
                        trace_index=trace_index)
            )
    return out


def parse_attacks(messages) -> list[Attack]:
    """Attack declarations within the interval. ``MSG_ATTACK`` is followed by 1 or 2 packed locations."""
    out: list[Attack] = []
    for trace_index, message in enumerate(messages):
        if message.msg != C.MSG_ATTACK or len(message.payload) < 4:
            continue
        (attacker,) = struct.unpack_from("<I", message.payload, 0)
        target = 0
        if len(message.payload) >= 8:
            (target,) = struct.unpack_from("<I", message.payload, 4)
        out.append(Attack(attacker=attacker, target=target,
                          trace_index=trace_index))
    return out


def _public_card_records(payload: bytes, head: int, count: int):
    """Decode the core's seven-byte ``code + location`` card records."""
    out = []
    off = head
    for _ in range(count):
        if off + 7 > len(payload):
            break
        (code,) = struct.unpack_from("<I", payload, off)
        ctrl, loc, seq = payload[off + 4: off + 7]
        out.append((code & 0x7FFFFFFF, ctrl | (loc << 8) | (seq << 16)))
        off += 7
    return out


def parse_public_events(messages) -> list[PublicEvent]:
    """Extract server-visible results in message order.

    ``MSG_SELECT_*`` is intentionally absent.  ``single_duel.cpp`` sends those
    prompts only to the answering player.  Feeding the raw in-process prompt to
    the other seat would expose private candidates that a deployed client never
    receives.  Public consequences and declarations are retained below.
    """
    out: list[PublicEvent] = []
    link = 0
    for trace_index, message in enumerate(messages):
        event_start = len(out)
        msg, body = message.msg, message.payload
        link = chain_link_after_message(link, msg, body)
        if msg in (C.MSG_CONFIRM_DECKTOP, C.MSG_CONFIRM_EXTRATOP) and len(body) >= 2:
            player, count = body[0], body[1]
            for code, at in _public_card_records(body, 2, count):
                out.append(PublicEvent(PUB_CONFIRM, player, code, at,
                                       link=link, audience=3))
        elif msg == C.MSG_CONFIRM_CARDS and len(body) >= 3:
            player, count = body[0], body[2]
            records = _public_card_records(body, 3, count)
            # The server sends a deck confirmation only to the requested seat;
            # confirmations from every other zone are broadcast.
            in_deck = bool(
                records
                and ((records[0][1] >> 8) & 0xFF) == C.LOCATION_DECK
            )
            audience = (1 << player) if in_deck else 3
            for code, at in records:
                out.append(PublicEvent(PUB_CONFIRM, player, code, at,
                                       link=link, audience=audience))
        elif msg == C.MSG_RANDOM_SELECTED and len(body) >= 2:
            player, count = body[0], body[1]
            # single_duel.cpp sends it to players[player] and re-sends it to
            # players[1]: seat 0 never receives seat 1's selection
            # (netduel/host_view.py keeps the quirk).
            audience = 3 if player == 0 else 2
            for k in range(count):
                off = 2 + 4 * k
                if off + 4 <= len(body):
                    (at,) = struct.unpack_from("<I", body, off)
                    out.append(PublicEvent(PUB_RANDOM, player, at=at,
                                           link=link, audience=audience))
        elif msg in (C.MSG_TOSS_COIN, C.MSG_TOSS_DICE) and len(body) >= 2:
            player, count = body[0], body[1]
            kind = PUB_COIN if msg == C.MSG_TOSS_COIN else PUB_DICE
            for value in body[2:2 + count]:
                out.append(PublicEvent(kind, player, value=int(value),
                                       link=link, audience=3))
        elif msg == C.MSG_HINT and len(body) >= 6:
            hint, player = body[0], body[1]
            (value,) = struct.unpack_from("<I", body, 2)
            kinds = {
                C.HINT_OPSELECTED: PUB_OPSELECTED,
                C.HINT_RACE: PUB_ANNOUNCE_RACE,
                C.HINT_ATTRIB: PUB_ANNOUNCE_ATTRIB,
                C.HINT_CODE: PUB_ANNOUNCE_CODE,
                C.HINT_NUMBER: PUB_ANNOUNCE_NUMBER,
                C.HINT_ZONE: PUB_ANNOUNCE_ZONE,
                C.HINT_CARD: PUB_HINT_CARD,
            }
            kind = kinds.get(hint)
            if kind is not None:
                code = value if kind in (PUB_ANNOUNCE_CODE, PUB_HINT_CARD) else 0
                # single_duel.cpp routes these declarations only to the other
                # player. HINT_CARD is the sole broadcast member of this
                # family; the chooser already knows every other answer.
                audience = 3 if hint == C.HINT_CARD else (1 << (1 - player))
                out.append(PublicEvent(kind, player, code=code, value=value,
                                       link=link, audience=audience))
        elif msg in (
            C.MSG_SHUFFLE_DECK,
            C.MSG_SHUFFLE_HAND,
            C.MSG_SHUFFLE_EXTRA,
            C.MSG_SHUFFLE_SET_CARD,
        ):
            kinds = {
                C.MSG_SHUFFLE_DECK: PUB_SHUFFLE_DECK,
                C.MSG_SHUFFLE_HAND: PUB_SHUFFLE_HAND,
                C.MSG_SHUFFLE_EXTRA: PUB_SHUFFLE_EXTRA,
                C.MSG_SHUFFLE_SET_CARD: PUB_SHUFFLE_SET,
            }
            player = body[0] if body and msg != C.MSG_SHUFFLE_SET_CARD else 2
            out.append(PublicEvent(kinds[msg], player, link=link, audience=3))
        elif msg == C.MSG_HAND_RES and body:
            out.append(PublicEvent(PUB_HAND_RESULT, value=body[0],
                                   link=link, audience=3))
        elif msg == C.MSG_DECK_TOP and len(body) >= 6:
            player, sequence = body[0], body[1]
            (raw_code,) = struct.unpack_from("<I", body, 2)
            # The high bit is the public/reversed marker.  Otherwise the
            # broadcast carries an intentionally hidden identity.
            code = (raw_code & 0x7FFFFFFF) if raw_code & 0x80000000 else 0
            at = player | (C.LOCATION_DECK << 8) | (sequence << 16)
            out.append(PublicEvent(PUB_DECK_TOP, player, code=code, at=at,
                                   link=link, audience=3))
        elif msg == C.MSG_DRAW and len(body) >= 2:
            player, count = body[0], body[1]
            for k in range(count):
                off = 2 + 4 * k
                if off + 4 > len(body):
                    break
                (raw_code,) = struct.unpack_from("<I", body, off)
                code = raw_code & 0x7FFFFFFF
                # The drawing player receives the identity. The opponent only
                # receives it when the core marks that identity public.
                audience = 3 if raw_code & 0x80000000 else (1 << player)
                out.append(PublicEvent(PUB_DRAW_REVEAL, player, code=code,
                                       link=link, audience=audience))
        elif msg in (
            C.MSG_SUMMONING, C.MSG_SPSUMMONING, C.MSG_FLIPSUMMONING,
        ) and len(body) >= 8:
            code, at = struct.unpack_from("<II", body, 0)
            kind = {
                C.MSG_SUMMONING: PUB_SUMMONING,
                C.MSG_SPSUMMONING: PUB_SPSUMMONING,
                C.MSG_FLIPSUMMONING: PUB_FLIPSUMMONING,
            }[msg]
            out.append(PublicEvent(
                kind, player=at & 0xFF, code=code & 0x7FFFFFFF,
                at=at, link=link, audience=3,
            ))
        elif msg == C.MSG_SET and len(body) >= 8:
            code, at = struct.unpack_from("<II", body, 0)
            # Setting is public, but a face-down identity is not. The tracker
            # needs only the location; code deliberately stays zero.
            out.append(PublicEvent(
                PUB_SET, player=at & 0xFF, at=at,
                link=link, audience=3,
            ))
        elif msg == C.MSG_NEW_TURN:
            player = int(body[0]) if body else 2
            out.append(PublicEvent(
                PUB_NEW_TURN, player=player, link=0, audience=3,
            ))
        elif msg in (
            C.MSG_SUMMONED, C.MSG_SPSUMMONED, C.MSG_FLIPSUMMONED,
            C.MSG_DAMAGE_STEP_START,
        ):
            kind = {
                C.MSG_SUMMONED: PUB_SUMMONED,
                C.MSG_SPSUMMONED: PUB_SPSUMMONED,
                C.MSG_FLIPSUMMONED: PUB_FLIPSUMMONED,
                C.MSG_DAMAGE_STEP_START: PUB_DAMAGE_STEP_START,
            }[msg]
            out.append(PublicEvent(kind, link=link, audience=3))
        elif msg in (C.MSG_EQUIP, C.MSG_CARD_TARGET, C.MSG_CANCEL_TARGET) and len(body) >= 8:
            source, target = struct.unpack_from("<II", body, 0)
            kind = {
                C.MSG_EQUIP: PUB_EQUIP,
                C.MSG_CARD_TARGET: PUB_CARD_TARGET,
                C.MSG_CANCEL_TARGET: PUB_CANCEL_TARGET,
            }[msg]
            out.append(PublicEvent(kind, at=source, target=target,
                                   link=link, audience=3))
        elif msg == C.MSG_UNEQUIP and len(body) >= 4:
            (source,) = struct.unpack_from("<I", body, 0)
            out.append(PublicEvent(PUB_UNEQUIP, at=source,
                                   link=link, audience=3))
        elif msg == C.MSG_FIELD_DISABLED and len(body) >= 4:
            (value,) = struct.unpack_from("<I", body, 0)
            out.append(PublicEvent(PUB_FIELD_DISABLED, value=value,
                                   link=link, audience=3))
        elif msg in (C.MSG_ADD_COUNTER, C.MSG_REMOVE_COUNTER) and len(body) >= 7:
            counter_type = struct.unpack_from("<H", body, 0)[0]
            at = body[2] | (body[3] << 8) | (body[4] << 16)
            count = struct.unpack_from("<H", body, 5)[0]
            kind = PUB_ADD_COUNTER if msg == C.MSG_ADD_COUNTER else PUB_REMOVE_COUNTER
            out.append(PublicEvent(kind, at=at, value=count, detail=counter_type,
                                   link=link, audience=3))
        elif msg == C.MSG_CARD_HINT and len(body) >= 9:
            (at,) = struct.unpack_from("<I", body, 0)
            hint_type = body[4]
            (value,) = struct.unpack_from("<I", body, 5)
            # CHINT_CARD stores another card's code as its value.
            code = value if hint_type == 2 else 0
            out.append(PublicEvent(PUB_CARD_HINT, code=code, at=at,
                                   value=value, detail=hint_type,
                                   link=link, audience=3))
        elif msg == C.MSG_PLAYER_HINT and len(body) >= 6:
            player, hint_type = body[0], body[1]
            (value,) = struct.unpack_from("<I", body, 2)
            out.append(PublicEvent(PUB_PLAYER_HINT, player=player, value=value,
                                   detail=hint_type, link=link, audience=3))
        elif msg == C.MSG_SWAP_GRAVE_DECK and body:
            out.append(PublicEvent(PUB_SWAP_GRAVE_DECK, player=body[0],
                                   link=link, audience=3))
        elif msg == C.MSG_REVERSE_DECK:
            out.append(PublicEvent(PUB_REVERSE_DECK, link=link, audience=3))
        elif msg == C.MSG_ATTACK_DISABLED:
            out.append(PublicEvent(PUB_ATTACK_DISABLED, link=link, audience=3))
        elif msg == C.MSG_MISSED_EFFECT and len(body) >= 8:
            at, code = struct.unpack_from("<II", body, 0)
            out.append(PublicEvent(PUB_MISSED_EFFECT, code=code, at=at,
                                   link=link, audience=3))
        for index in range(event_start, len(out)):
            out[index] = replace(out[index], trace_index=trace_index)
    return out


def parse_moves(messages) -> tuple[list[Move], list[PosChange], list[Draw]]:
    """Pull the card-movement record out of a message interval.

    Every event is tagged with the chain link whose resolution window it fell
    in.  The window is ``MSG_CHAIN_SOLVING(k)`` (``processor.cpp:4399``) to
    ``MSG_CHAIN_SOLVED(k)`` (``:4503``) or the whole-chain ``MSG_CHAIN_END``;
    a later SOLVING replaces the active link. Link 0 means the event happened
    outside any window, which is where activation-time movement lives -- a Set
    trap turning face-up, a cost being paid.

    ``MSG_CHAIN_NEGATED`` is deliberately *not* a boundary: ``negate_chain``
    (``operations.cpp:31``) fires while some *other* effect is resolving, so it
    appears outside link k's own window and using it as a boundary would
    misattribute everything after it. Activation negation can skip SOLVED;
    effect disabling does not. END prevents subsequent activation-time events
    from inheriting the last negated link's attribution.

    Without this tag the per-link split is unrecoverable: the movement list and
    the chain list each keep their own message order, and the interleaving
    between them -- which is what carries the attribution -- is gone.
    """
    moves: list[Move] = []
    changes: list[PosChange] = []
    draws: list[Draw] = []
    link = 0
    for trace_index, message in enumerate(messages):
        link = chain_link_after_message(
            link, message.msg, message.payload, include_chaining=False,
        )
        if message.msg == C.MSG_MOVE:
            code, prev, cur, reason = struct.unpack_from("<IIII", message.payload, 0)
            moves.append(
                Move(
                    code=code,
                    from_controller=prev & 0xFF,
                    from_location=(prev >> 8) & 0xFF,
                    from_sequence=(prev >> 16) & 0xFF,
                    from_position=(prev >> 24) & 0xFF,
                    to_controller=cur & 0xFF,
                    to_location=(cur >> 8) & 0xFF,
                    to_sequence=(cur >> 16) & 0xFF,
                    to_position=(cur >> 24) & 0xFF,
                    reason=reason,
                    link=link,
                    trace_index=trace_index,
                )
            )
        elif message.msg == C.MSG_POS_CHANGE:
            (code,) = struct.unpack_from("<I", message.payload, 0)
            body = message.payload
            changes.append(
                PosChange(
                    code=code,
                    controller=body[4],
                    location=body[5],
                    sequence=body[6],
                    previous=body[7],
                    current=body[8],
                    link=link,
                    trace_index=trace_index,
                )
            )
        elif message.msg == C.MSG_DRAW:
            body = message.payload
            count = body[1]
            codes = struct.unpack_from(f"<{count}I", body, 2) if count else ()
            # the top bit marks a card drawn face down to the opponent's view
            draws.append(
                Draw(player=body[0],
                     codes=tuple(c & 0x7FFFFFFF for c in codes),
                     link=link, trace_index=trace_index)
            )
    return moves, changes, draws


def diff_snapshots(
    before: StateSnapshot, after: StateSnapshot, messages=()
) -> StateDiff:
    """The settlement target: what changed between two snapshots."""
    moves, changes, draws = parse_moves(messages)
    chance = tuple(
        sorted({m.name for m in messages if m.msg in _CHANCE_MSGS})
    )

    before_keys = before.by_key()
    after_keys = after.by_key()
    negated = []
    for key, card in after_keys.items():
        was = before_keys.get(key)
        if card.negated and (was is None or not was.negated or was.code != card.code):
            negated.append((card.controller, card.location, card.sequence, card.code))

    before_codes = _code_bag(before)
    after_codes = _code_bag(after)
    appeared = sorted((after_codes - before_codes).elements())
    vanished = sorted((before_codes - after_codes).elements())

    count_delta = {}
    for key in set(before.counts) | set(after.counts):
        delta = after.counts.get(key, 0) - before.counts.get(key, 0)
        if delta:
            count_delta[key] = delta

    return StateDiff(
        lp_delta=(after.lp[0] - before.lp[0], after.lp[1] - before.lp[1]),
        moves=moves,
        pos_changes=changes,
        draws=draws,
        negated=negated,
        appeared=appeared,
        vanished=vanished,
        count_delta=count_delta,
        chain=parse_chain(messages),
        targets=parse_targets(messages),
        attacks=parse_attacks(messages),
        lp_events=parse_lp_events(messages),
        public_events=parse_public_events(messages),
        turn_delta=after.turn - before.turn,
        phase_after=after.phase,
        chance=bool(chance),
        chance_msgs=chance,
        determinism=determinism_of(messages),
    )


def _code_bag(snapshot: StateSnapshot):
    from collections import Counter

    bag = Counter()
    for card in snapshot.cards:
        if card.code:
            bag[card.code] += 1
        for code in card.overlay:
            bag[code] += 1
    return bag


#: True randomness: the result is **not** a function of state and action, and replaying the same position may not reproduce it.
_RANDOM_MSGS = frozenset(
    {
        C.MSG_TOSS_COIN,
        C.MSG_TOSS_DICE,
        C.MSG_RANDOM_SELECTED,
        C.MSG_SHUFFLE_DECK,
        C.MSG_SHUFFLE_HAND,
        C.MSG_SHUFFLE_EXTRA,
        C.MSG_SHUFFLE_SET_CARD,
    }
)

#: The result is actually determined (the deck order was fixed at the start), but **this seat cannot see it**,
#: so it is equally unpredictable from this observation. Kept apart from true randomness:
#: no amount of information helps the former, while the latter becomes predictable with the information; for a world model they differ.
_HIDDEN_INFO_MSGS = frozenset(
    {
        C.MSG_DRAW,
        C.MSG_CONFIRM_DECKTOP,
        C.MSG_CONFIRM_EXTRATOP,
    }
)

_CHANCE_MSGS = _RANDOM_MSGS | _HIDDEN_INFO_MSGS

#: values of the three-way flag
DET_DETERMINISTIC, DET_RANDOM, DET_HIDDEN = 0, 1, 2


def determinism_of(messages) -> int:
    """Split an interval into deterministic / true randomness / hidden information.

    There used to be one `chance` boolean, lumping "a coin flip" together with "a draw". For a world model these are
    two things: true randomness can **never** be predicted, while a draw is incomplete information that becomes predictable with it.
    Evaluation is only meaningful split by the three kinds; otherwise the "uncertain" lump mixes two entirely different failures.
    True randomness takes precedence: an interval with both counts as random (the stronger unpredictability).
    """
    kinds = {m.msg for m in messages}
    if kinds & _RANDOM_MSGS:
        return DET_RANDOM
    if kinds & _HIDDEN_INFO_MSGS:
        return DET_HIDDEN
    return DET_DETERMINISTIC


def roundtrip_occupancy(before: StateSnapshot, diff: StateDiff) -> dict:
    """Occupancy implied by replaying public count-changing events.

    The settlement target is only worth training on if it *determines* the next
    board.  Moves, draws and zone-wide operations must be applied in their
    shared wire order: ``MSG_SWAP_GRAVE_DECK`` is not accompanied by one move
    per card, and applying an activation-cost move after rather than before the
    swap changes the answer.  Overlay (Xyz material) traffic is excluded because
    a material is not a zone occupant.
    """
    occupancy: dict[tuple[int, int], int] = {}
    for (player, location), n in before.counts.items():
        occupancy[(player, location)] = n

    events = []
    events.extend(
        (int(move.trace_index), 0, "move", move) for move in diff.moves
    )
    events.extend(
        (int(draw.trace_index), 1, "draw", draw) for draw in diff.draws
    )
    events.extend(
        (int(event.trace_index), 2, "swap", event)
        for event in diff.public_events
        if int(event.kind) == PUB_SWAP_GRAVE_DECK
    )
    for _trace, _priority, kind, event in sorted(events):
        if kind == "move":
            if (event.from_location
                    and not (event.from_location & C.LOCATION_OVERLAY)):
                key = (event.from_controller, event.from_location)
                occupancy[key] = occupancy.get(key, 0) - 1
            if (event.to_location
                    and not (event.to_location & C.LOCATION_OVERLAY)):
                key = (event.to_controller, event.to_location)
                occupancy[key] = occupancy.get(key, 0) + 1
        elif kind == "draw":
            deck = (event.player, C.LOCATION_DECK)
            hand = (event.player, C.LOCATION_HAND)
            occupancy[deck] = occupancy.get(deck, 0) - event.count
            occupancy[hand] = occupancy.get(hand, 0) + event.count
        else:
            deck = (event.player, C.LOCATION_DECK)
            grave = (event.player, C.LOCATION_GRAVE)
            occupancy[deck], occupancy[grave] = (
                occupancy.get(grave, 0), occupancy.get(deck, 0)
            )
    return {k: v for k, v in occupancy.items()}
