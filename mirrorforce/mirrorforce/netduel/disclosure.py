"""Public identity ledger: who has seen which card, as both sides know it.

"Disclosure is input" is a hard rule of this project: once a card has been activated, revealed or confirmed, its
identity is public to both sides and **must never be anonymized again**. This module is the only implementation of
that rule in the repository. The corpus generator (``worldmodel/engine.py``) and the online bridge
(``netduel/board.py``) both use it, because both feed the same visibility predicate
(``worldmodel/state.identity_visible``); two copies would drift apart silently.

It lives in ``netduel`` rather than ``worldmodel``: ``worldmodel.state`` already depends on
``netduel.board``, so the reverse import would be circular. This module depends only on ``netduel.constants``.

Three changes of ledger v2
-------------

1. **A card of unknown identity leaving a zone no longer forgets the whole zone.** The old ledger's
   ``_forget_zone_for_viewer`` cleared every proven identity of that zone. The lower bound a spectator can always
   derive is much tighter: one card leaving lowers the guaranteed count of each card code by at most one. See :meth:`_depart_unidentified`.
2. **Slot tracking for the vector zones.** The ``previous`` / ``current`` location words of ``MSG_MOVE``
   are **not erased** when the server forwards the message (``gframe/single_duel.cpp:1011-1035`` only zeroes the
   ``code``), and the reordering rules of ``field::remove_card`` / ``field::add_card``
   are deterministic (``ygopro-core/field.cpp:204-249`` and ``:161-171``, together with
   ``reset_sequence``, ``field.cpp:1010``). So an observer can follow slots, and an identity pinned to another
   slot need not be lost because an unrelated card left.
3. **Category constraints.** A card searched from the deck to the hand / set from the deck by an effect has no public
   identity, but the activating card is public (``MSG_CHAINING``) and its text limits the set it can fetch. The ledger
   records that limit as a constraint on the slot and hands it to the sampler.

All three use only messages the observer legitimately receives; none of them reads the truth.
"""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from . import constants as C

__all__ = [
    "DISCLOSURE_LEDGER_SCHEMA",
    "DISCLOSURE_LEDGER_SCHEMA_KEY",
    "LEGACY_DISCLOSURE_LEDGER_SCHEMA",
    "ledger_schema_of",
    "ledger_stamp",
    "require_current_ledger_schema",
    "CategoryConstraint",
    "SLOT_HISTORY_DERIVABLE_STATUS_MASK",
    "SlotHistory",
    "CategoryRestrictions",
    "DisclosureLedger",
    "resolve_disclosure",
]

#: Ledger calibration version. A change changes the output of ``mask_for``, and with it the tensorized inputs and
#: history, so every artifact with provenance stamps it in, and data of mixed versions must never be joined silently.
#:
#: Schema 4 (2026-09-03) handled ``MSG_SWAP``. In every game with a swap of field cards, every later ``MSG_MOVE``
#: through those two zones was booked under the wrong card code since schema 3, so the ledger differs from schema 4 card by card.
#: This is not a backward-compatible extra column: the same public message stream resolves to a **different
#: disclosed set**, so the version must change and schema 3 artifacts fail closed in :func:`require_current_ledger_schema`.
#:
#: Schema 5 (2026-09-26): when a card of unknown identity reveals its code only as it leaves, and the known copies of that
#: card in the zone are all anchored to other slots, the known count is no longer lowered by one (before, the anchors
#: stayed but the count reached zero, resolution skipped the anchors and that slot's identity was lost). The same public message stream resolves to a disclosed set different from the previous schema.
DISCLOSURE_LEDGER_SCHEMA = "mirrorforce_disclosure_ledger/v5"

#: v1 is the version that "forgets the whole zone when a card of unknown identity leaves". An artifact **without**
#: this field was exported by v1: the field was added with v2, so a missing field means v1, never "unknown".
LEGACY_DISCLOSURE_LEDGER_SCHEMA = "mirrorforce_disclosure_ledger/v1"

#: Field name of the stamp in provenance. Writers use :func:`ledger_stamp`, readers use
#: :func:`ledger_schema_of` or :func:`require_current_ledger_schema`;
#: all three share this one name. A literal and a default of its own in each artifact is exactly how mixed versions
#: get joined silently.
DISCLOSURE_LEDGER_SCHEMA_KEY = "disclosure_ledger_schema"


def ledger_stamp() -> dict[str, str]:
    """Stamp the ledger calibration version of this build into provenance.

    Writers expand ``**ledger_stamp()`` into their own provenance dictionary, so the field name and its value are
    defined in this one place.
    """

    return {DISCLOSURE_LEDGER_SCHEMA_KEY: DISCLOSURE_LEDGER_SCHEMA}


def ledger_schema_of(
    provenance,
    *,
    default: str = LEGACY_DISCLOSURE_LEDGER_SCHEMA,
) -> str:
    """Which ledger version a provenance record was exported with.

    A missing field means ``default`` (v1 by default): the field was added with v2, so "absent"
    is a definite answer, not an unknown.
    """

    if not isinstance(provenance, Mapping):
        return default
    return str(provenance.get(DISCLOSURE_LEDGER_SCHEMA_KEY, default))


def require_current_ledger_schema(
    provenance,
    *,
    where: str,
    error: type[Exception] = ValueError,
) -> str:
    """Check that a provenance record has the calibration of this build; raise ``error`` otherwise.

    For readers of content derived from ``mask_for``: the ledger calibration decides which opponent cards count as
    disclosed, and data of two calibrations joined together give labels and features different meanings.
    """

    found = ledger_schema_of(provenance)
    if found != DISCLOSURE_LEDGER_SCHEMA:
        raise error(
            f"{where}: disclosure-ledger calibration differs from this build: "
            f"{found!r} != {DISCLOSURE_LEDGER_SCHEMA!r}"
        )
    return found

#: "Vector zones": ``field`` stores them in a ``std::vector``; removing a card calls ``reset_sequence``, which
#: shifts every later sequence down, and inserting one shifts every sequence from the insertion point up
#: (``ygopro-core/field.cpp:161-249``, ``field.cpp:1010``). The shift is deterministic,
#: and ``MSG_MOVE`` carries the leaving / arriving sequence, so an observer can follow it.
_LIST_LOCATIONS = frozenset({
    C.LOCATION_DECK,
    C.LOCATION_HAND,
    C.LOCATION_GRAVE,
    C.LOCATION_REMOVED,
    C.LOCATION_EXTRA,
})

#: Fixed zones: ``list_mzone`` / ``list_szone`` are fixed arrays; removing a card only zeroes its slot and
#: sequences do not move (``field.cpp:207-212``). In these two zones a count **means something only when it is
#: pinned to a slot**: a statement "this zone holds an X" without knowing which slot makes
#: :func:`resolve_disclosure` fill one slot in ascending sequence order, which is position information made up
#: from nothing. So once a fixed zone loses its slot anchor, the matching count is dropped too.
_FIXED_LOCATIONS = frozenset({C.LOCATION_MZONE, C.LOCATION_SZONE})


@dataclass(frozen=True)
class CategoryConstraint:
    """A public constraint on the identity of one hidden card.

    ``sequence is None`` means the anchor is lost (after a hand shuffle, say), but "this zone still holds a card
    whose code is in ``codes``" still holds.
    """

    #: zone of the card (``LOCATION_HAND`` / ``LOCATION_SZONE`` / ...)
    location: int
    #: sequence; ``None`` means only the zone is known, not the slot
    sequence: int | None
    #: the admissible set: the identity must be one of these codes
    codes: frozenset[int]
    #: code of the activating card the constraint comes from (public, given by ``MSG_CHAINING``)
    source_code: int

    def as_dict(self) -> dict:
        return {
            "location": int(self.location),
            "sequence": None if self.sequence is None else int(self.sequence),
            "codes": sorted(int(code) for code in self.codes),
            "source_code": int(self.source_code),
        }


#: The status bits of the face-down slot history that **can be derived from public messages**.
#:
#: Only ``STATUS_SET_TURN`` (``ygopro-core/common.h:200``): it is set when a card is set
#: (``operations.cpp:2472`` and ``:2600``), cleared by the end-of-turn reset loop
#: (``processor.cpp:3705``) and cleared when the card leaves (``operations.cpp:4656``). All three
#: match public messages, so "was this card set this turn" is a public fact.
#:
#: No other status bit can be derived: they record what this **particular card** went through
#: (``STATUS_EFFECT_ENABLED``, ``STATUS_SUMMON_TURN``,
#: ``STATUS_ATTACK_CANCELED`` and so on), and the identity and history of a face-down card are hidden from observers.
#: Consumers (``common/public_particles.py``) must treat the bits outside the mask as zero, never as
#: "known to be zero"; see :class:`SlotHistory`.
SLOT_HISTORY_DERIVABLE_STATUS_MASK = 0x0010  # STATUS_SET_TURN


@dataclass(frozen=True)
class SlotHistory:
    """The public history of one face-down slot. **One record per slot**, a shape kept for the C++ side.

    The fields match the arguments of ``Debug.SetCardState`` one to one, and have the shape of
    ``PublicSlotHistoryRow`` in ``common/public_particles.py``:
    ``(location, sequence, status, turnid, turn_counter)``.

    The three fields are not equally reliable, and consumers must read them separately:

    ``turnid``
        **Exact.** The turn in which the card now in this slot arrived. ``field::add_card`` writes
        ``infos.turn_id`` into ``pcard->turnid``, and ``infos.turn_id`` is incremented just before
        ``MSG_NEW_TURN`` (``processor.cpp:3756-3759``), so counting
        ``MSG_NEW_TURN`` keeps it aligned.
    ``status``
        **Only the bits in :data:`SLOT_HISTORY_DERIVABLE_STATUS_MASK` mean something.**
        Every other bit is zero, meaning "not derivable", not "known to be zero".
        Measured residual: on the multi-deck corpus 5 of 4,922 slots (0.10%) have this bit too low:
        the path at ``operations.cpp:4936`` also sets it when it turns a card face-down again, and outwardly that
        shows only as one ``MSG_POS_CHANGE``, which public messages cannot tell apart from an ordinary flip face-down.
        The ledger is **conservative** here (records 0); a card the realizer writes back lacks one bit, so the
        engine is more permissive about trap / quick-play activation conditions, not stricter. On pilot100 the residual is 0.

    ``turn_counter``
        **Not derivable; always 0.** In the engine only a script's ``Card.SetTurnCounter``
        changes it (``libcard.cpp:1408``), and a new card starts at 0 (``card.h:178``). Which card changes
        it depends on the card's identity, which is hidden for a face-down card. 0 is the truth for a newly placed slot,
        and an underestimate for a slot whose counter a script changed.
    """

    location: int
    sequence: int
    status: int
    turnid: int
    turn_counter: int

    def as_row(self) -> tuple[int, int, int, int, int]:
        """The ``PublicSlotHistoryRow`` shape, fed directly to ``public_particles``."""

        return (int(self.location), int(self.sequence), int(self.status),
                int(self.turnid), int(self.turn_counter))


class CategoryRestrictions:
    """Activating card code -> the set of card codes its effect can fetch.

    This is an **external, explicit** table; the ledger carries no entries by default. The reason is the direction of
    the error: a wrong constraint makes the sampler reject a legal hypothetical position as unrealizable, which is
    worse than not recording the constraint. So nothing here guesses "same archetype by setcode"; only tables the
    caller provides, checked against the card text entry by entry, are used. :meth:`from_setcodes` is likewise a
    mechanical expansion of "given rules + card database -> set of codes"; the rules remain the caller's responsibility.

    The table itself is public information (both sides have the same card database), so using it does not break the deployment observation contract.
    """

    __slots__ = ("_table",)

    def __init__(self, table: Mapping[int, frozenset[int]] | None = None) -> None:
        self._table: dict[int, frozenset[int]] = {}
        for code, allowed in (table or {}).items():
            codes = frozenset(int(value) & 0x7FFFFFFF for value in allowed)
            codes = frozenset(code for code in codes if code)
            if codes:
                self._table[int(code) & 0x7FFFFFFF] = codes

    def __bool__(self) -> bool:
        return bool(self._table)

    def __len__(self) -> int:
        return len(self._table)

    def get(self, source_code: int) -> frozenset[int] | None:
        return self._table.get(int(source_code) & 0x7FFFFFFF)

    def as_dict(self) -> dict[int, list[int]]:
        return {code: sorted(allowed) for code, allowed in sorted(self._table.items())}

    @classmethod
    def from_setcodes(
        cls,
        rules: Mapping[int, Mapping],
        pool: Mapping[int, object],
    ) -> "CategoryRestrictions":
        """Expand "archetype + card type" rules into card-code sets over the card database.

        ``rules`` looks like ``{activating code: {"setcodes": [0x??], "type_mask": ...,
        "type_value": ...}}``; ``pool`` maps ``code -> an object with setcodes/type attributes``;
        the ``cards`` of :class:`mirrorforce.netduel.cards.CardPool` work as is.
        ``setcodes`` are compared by ygopro's 16-bit segments (``write_setcode``):
        the high 4 bits are the sub-archetype, aligned by the mask rule before comparing.
        """

        table: dict[int, frozenset[int]] = {}
        for source, rule in rules.items():
            wanted = tuple(int(value) & 0xFFFF for value in rule.get("setcodes", ()))
            type_mask = int(rule.get("type_mask", 0))
            type_value = int(rule.get("type_value", 0))
            explicit = frozenset(int(value) for value in rule.get("codes", ()))
            allowed = set(explicit)
            selective = bool(wanted or type_mask)
            for code, data in pool.items():
                card_type = int(getattr(data, "type", 0) or 0)
                if type_mask and (card_type & type_mask) != type_value:
                    continue
                if wanted:
                    have = tuple(
                        int(value) & 0xFFFF
                        for value in (getattr(data, "setcodes", ()) or ())
                    )
                    if not any(_setcode_matches(one, other)
                               for one in have for other in wanted):
                        continue
                elif not selective:
                    continue
                allowed.add(int(code))
            if allowed:
                table[int(source)] = frozenset(allowed)
        return cls(table)


def _setcode_matches(have: int, wanted: int) -> bool:
    """The 16-bit comparison of ``check_setcode`` (``ygopro-core/card_data.h:25-29``):
    the low 12 bits (archetype) must be equal, the high 4 bits (sub-archetype) are matched as a mask, and an empty entry never matches."""

    return bool(have) and (have & 0x0FFF) == (wanted & 0x0FFF) and (
        have & wanted & 0xF000) == (wanted & 0xF000)


class DisclosureLedger:
    """Bookkeeping of public identities. **The corpus side and the online bridge share this one implementation.**

    "Disclosure is input" requires that once a card has been activated, revealed or confirmed, its identity is
    public to both sides and is never anonymized again. The ledger consumes only public messages, and the standard
    client and the MD adapter run the same code, so the input columns derived from it do not break the deployment observation contract.

    Two layers of bookkeeping
    --------

    * **Multiset**: ``(controller, zone, code) -> count``, a lower bound "this zone holds at least this many copies
      of this code". This layer is unaffected by sequence shifts and shuffles.
    * **Slot anchors**: ``(controller, zone) -> {sequence: code}``, a proof that "this slot is this card".
      Anchors follow moves by the deterministic reordering rules of ``field``; when they cannot follow (a hand or
      deck shuffle), only the anchors are cleared and the multiset layer stays.

    The split between the layers decides the leave rule (:meth:`_depart_unidentified`): when a card of unknown
    identity leaves a zone, **the identities pinned to other slots stay valid**; only the part of the lower bound
    "not known to be in which slot" drops by one. The old version cleared the whole zone here and lost already proven
    hand identities with it.
    """

    #: see :data:`DISCLOSURE_LEDGER_SCHEMA`
    SCHEMA = DISCLOSURE_LEDGER_SCHEMA

    __slots__ = (
        "_viewer_counts", "_slots", "_fresh", "_faceup_extra", "_claims",
        "_restrictions", "_last_chaining", "_slot_turns", "_turn",
        "_chain_stack",
    )

    def __init__(self, restrictions: CategoryRestrictions | Mapping | None = None
                 ) -> None:
        self._viewer_counts = (Counter(), Counter())
        self._slots: tuple[dict, dict] = ({}, {})
        self._fresh: tuple[dict, dict] = ({}, {})
        self._faceup_extra = (Counter(), Counter())
        self._claims: tuple[dict, dict] = ({}, {})
        if restrictions is None:
            self._restrictions = CategoryRestrictions()
        elif isinstance(restrictions, CategoryRestrictions):
            self._restrictions = restrictions
        else:
            self._restrictions = CategoryRestrictions(restrictions)
        self._last_chaining = 0
        #: Code of the activating card of each link of the current chain; index 0 is chain link 1. A chain **resolves
        #: backwards**, so the link resolving now must be given by ``MSG_CHAIN_SOLVING``;
        #: the last ``MSG_CHAINING`` cannot stand in for it.
        self._chain_stack: list[int] = []
        #: ``(controller, zone) -> {sequence: [arrival turn, was it set when it arrived, is it face-down now]}``.
        #: The face-down slot history is a fact **both sides can see** (location words and positions are not erased),
        #: so it is not kept per observer: one record per slot. Face-up field slots keep their records too:
        #: ``pcard->turnid`` records the turn the card **arrived**, and flipping does not change it, so dropping the
        #: record on a flip face-up would record the flip turn as turnid when the card is flipped face-down again.
        self._slot_turns: dict[tuple[int, int], dict[int, list[int]]] = {}
        #: Turns counted so far. Just before ``MSG_NEW_TURN`` the engine increments ``infos.turn_id``,
        #: so this count is aligned with ``pcard->turnid``.
        self._turn = 0

    # -- read-only views ----------------------------------------------------

    @property
    def counts(self) -> Counter:
        """Omniscient lower bound retained for diagnostics and old reports."""

        result: Counter = Counter()
        for viewer_counts in self._viewer_counts:
            for key, count in viewer_counts.items():
                result[key] = max(result.get(key, 0), int(count))
        return result

    def known_counts(self, viewer: int) -> Counter:
        if viewer not in (0, 1):
            raise ValueError("disclosure viewer must be seat zero or one")
        return Counter(self._viewer_counts[int(viewer)])

    def known_code_at(
        self, viewer: int, controller: int, location: int, sequence: int
    ) -> int:
        """Return the exact code proved at this slot, or zero.

        Overlay slots use ``host_sequence << 8 | material_index`` internally;
        a host's location alone does not identify one of its materials.
        """

        if viewer not in (0, 1):
            raise ValueError("disclosure viewer must be seat zero or one")
        slots = self._slots[int(viewer)].get((int(controller), int(location)))
        if not slots:
            return 0
        return int(slots.get(int(sequence), 0))

    def known_slots(self, viewer: int, controller: int, location: int
                    ) -> dict[int, int]:
        """``sequence -> code``: the proven slot identities of this zone (for diagnostics and tests)."""

        if viewer not in (0, 1):
            raise ValueError("disclosure viewer must be seat zero or one")
        slots = self._slots[int(viewer)].get((int(controller), int(location)))
        return dict(slots or {})

    def faceup_extra_counts(self, viewer: int) -> Counter:
        if viewer not in (0, 1):
            raise ValueError("disclosure viewer must be seat zero or one")
        return Counter(self._faceup_extra[int(viewer)])

    def category_constraints(
        self, viewer: int, controller: int | None = None
    ) -> tuple[CategoryConstraint, ...]:
        """The category constraints that still hold for this observer on one controller's hidden cards."""

        if viewer not in (0, 1):
            raise ValueError("disclosure viewer must be seat zero or one")
        out: list[CategoryConstraint] = []
        for (zone_controller, location), claims in self._claims[int(viewer)].items():
            if controller is not None and zone_controller != int(controller):
                continue
            for sequence, codes, source in claims:
                out.append(CategoryConstraint(
                    location=int(location),
                    sequence=None if sequence is None else int(sequence),
                    codes=codes,
                    source_code=int(source),
                ))
        out.sort(key=lambda item: (item.location,
                                   -1 if item.sequence is None else item.sequence,
                                   item.source_code, sorted(item.codes)))
        return tuple(out)

    def fresh_slots(self, viewer: int, controller: int) -> tuple[tuple[int, int], ...]:
        """The ``(zone, sequence)`` pairs of ``controller``'s "fresh" slots: a card of unknown identity arrived there
        and the slot's identity has not been disclosed since (see :meth:`_slot_arrive`).

        The ledger holds that the card in such a slot is none of the instances proven before, so when the slot later
        shows its identity, :meth:`disclose` books **one more copy**. So the cards of :meth:`unanchored_identities`,
        known by identity but not by position, can only be in hidden slots of this zone that are **not** fresh; the
        sampler must keep to this, or a hypothetical position showing one of them in such a slot makes the ledger count one more.
        """

        if viewer not in (0, 1) or controller not in (0, 1):
            raise ValueError("disclosure viewer and controller must be seat zero or one")
        return tuple(sorted((int(location), int(sequence))
                            for (zone_controller, location), sequences in self._fresh[int(viewer)].items()
                            if zone_controller == int(controller) and location in _FIXED_LOCATIONS
                            for sequence in sequences))

    def unanchored_identities(self, viewer: int, resolved=None) -> Counter:
        """Cards this observer knows **which** they are but not **in which slot**.

        Returns ``(controller, zone) -> code -> count`` flattened into a ``Counter`` keyed by
        ``(controller, zone, code)``, valued by count: the multiset lower bound minus the part still pinned to slots.

        Two sources, matching two things the model must read separately:

        * **Hand**: after a hand shuffle (``field::shuffle``) positions mean nothing, but "they hold this card in
          hand" still holds. :func:`resolve_disclosure` used to pin these counts to the lowest slots in ascending
          sequence order, which is **made-up position**.
        * **Face-down field cards**: after the set cards are shuffled (``MSG_SHUFFLE_SET_CARD``), the observer knows
          which cards these slots hold, but not which is where.

        In both cases "what" is public and "where" is not. The caller (``mask_for``)
        puts them into ``StateSnapshot.revealed_unpositioned`` without positions.
        """

        if viewer not in (0, 1):
            raise ValueError("disclosure viewer must be seat zero or one")
        placed: Counter = Counter()
        if resolved is not None:
            # The cards that :func:`resolve_disclosure` actually gave a position. Use these rather than
            # "the number of slots the ledger pinned": a binding may not match this snapshot (it happens when a
            # zone is rebuilt reporting counts only); such a card got no position at all and still belongs to
            # "known what, unknown where"; it must not be dropped.
            for controller, location, _sequence, code in resolved:
                placed[(int(controller), int(location), int(code))] += 1
        else:
            for (controller, location), slots in self._slots[int(viewer)].items():
                for code in (slots or {}).values():
                    placed[(int(controller), int(location), int(code))] += 1
        out: Counter = Counter()
        for key, count in self._viewer_counts[int(viewer)].items():
            spare = int(count) - int(placed.get(key, 0))
            if spare > 0:
                out[key] = spare
        return out

    def slot_history(self, controller: int | None = None
                     ) -> tuple[SlotHistory, ...]:
        """Public history of the face-down slots, sorted by ``(zone, sequence)``.

        Alongside :meth:`category_constraints`, it reaches the sampler through the same public-evidence path.
        One record per slot still face-down; with ``controller`` given, only that side's.
        """

        out: list[SlotHistory] = []
        for (zone_controller, location), slots in self._slot_turns.items():
            if controller is not None and zone_controller != int(controller):
                continue
            for sequence, record in slots.items():
                turnid, placed_by_set, facedown = record
                if not facedown:
                    continue  # a face-up field slot is not a set card; it only keeps its turnid
                status = 0
                # STATUS_SET_TURN holds only "within the turn the card was set": the end-of-turn reset loop
                # clears it (``processor.cpp:3705``).
                if placed_by_set and turnid == self._turn:
                    status |= SLOT_HISTORY_DERIVABLE_STATUS_MASK
                out.append(SlotHistory(
                    location=int(location),
                    sequence=int(sequence),
                    status=status,
                    turnid=int(turnid),
                    turn_counter=0,
                ))
        out.sort(key=lambda row: (row.location, row.sequence))
        return tuple(out)

    def slot_history_rows(self, controller: int | None = None
                          ) -> tuple[tuple[int, int, int, int, int], ...]:
        """The ``PublicSlotHistoryRow`` form of :meth:`slot_history`."""

        return tuple(row.as_row() for row in self.slot_history(controller))

    # -- maintaining the face-down slot history ------------------------------------

    def observe_new_turn(self, body: bytes | None = None) -> None:
        """``MSG_NEW_TURN``: one more turn, aligned with ``infos.turn_id``."""

        del body  # only "another turn passed" matters; whose turn it is does not affect slot history
        self._turn += 1

    def _slot_history_depart(self, zone: tuple[int, int], sequence: int) -> None:
        slots = self._slot_turns.get(zone)
        if slots is None:
            return
        slots.pop(int(sequence), None)
        if zone[1] in _LIST_LOCATIONS:
            self._slot_turns[zone] = {
                (slot - 1 if slot > int(sequence) else slot): record
                for slot, record in slots.items()}
        if not self._slot_turns.get(zone):
            self._slot_turns.pop(zone, None)

    def _slot_history_arrive(self, zone: tuple[int, int], sequence: int,
                             position: int) -> None:
        """A card arrives in this slot. Face-down creates/replaces the record; face-up removes it."""

        slots = self._slot_turns.get(zone, {})
        if zone[1] in _LIST_LOCATIONS:
            slots = {(slot + 1 if slot >= int(sequence) else slot): record
                     for slot, record in slots.items()}
        else:
            slots.pop(int(sequence), None)
        facedown = bool(position & C.POS_FACEDOWN) and not bool(
            position & C.POS_REVEAL)
        # ``placed_by_set`` always starts at 0: arriving face-down is not being set (an effect can special summon a
        # monster in face-down defense position, which does not set ``STATUS_SET_TURN``). Only ``MSG_SET``
        # is public evidence of a set; see :meth:`observe_set`.
        if facedown or int(zone[1]) in _FIXED_LOCATIONS:
            slots[int(sequence)] = [self._turn, 0, 1 if facedown else 0]
        if slots:
            self._slot_turns[zone] = slots
        else:
            self._slot_turns.pop(zone, None)

    def observe_set(self, body: bytes) -> None:
        """``MSG_SET``: ``code(4) location word(4)`` (``operations.cpp:2406-2408``).

        This is the only public evidence of a set and the only source of ``STATUS_SET_TURN``: the core writes this
        message right after both places that set the bit (``operations.cpp:2472`` with ``:2484``,
        ``:2600`` with ``:2612``), and the server broadcasts it to both sides with only the ``code`` zeroed
        (``single_duel.cpp:1051-1058``). Arriving face-down by itself is not a set: an effect can special summon a
        monster in face-down defense position, which does not set the bit.
        """

        if len(body) < 8:
            return
        (_code, at) = struct.unpack_from("<II", body, 0)
        controller = int(at) & 0xFF
        location = (int(at) >> 8) & 0xFF
        sequence = (int(at) >> 16) & 0xFF
        slots = self._slot_turns.get((controller, location))
        if slots is None:
            return
        record = slots.get(sequence)
        if record is not None:
            record[0] = self._turn
            # Only sets in the spell & trap zone set ``STATUS_SET_TURN`` (``operations.cpp:2472``
            # and ``:2600``, both ``move_to_field(..., LOCATION_SZONE, ...)``).
            # A face-down monster summon sends ``MSG_SET`` as well (``:2406``), but that path sets
            # ``STATUS_SUMMON_TURN`` and ``STATUS_CANNOT_CHANGE_FORM``,
            # not ``STATUS_SET_TURN``; that bit is read only by trap and quick-play activation conditions
            # (``effect.cpp:200-202``).
            record[1] = 1 if location == C.LOCATION_SZONE else 0
            record[2] = 1

    def observe_pos_change(self, body: bytes) -> None:
        """``MSG_POS_CHANGE``: a flip is the only message that publicly announces a change of position.

        Payload ``code(4) player(1) location(1) sequence(1) prev(1) new(1)``,
        parsed the same way as in ``netduel/board.py``. Flipping face-up removes the record (the slot is no
        longer a set card); flipping face-down again creates a new one, but **does not set** ``STATUS_SET_TURN``:
        an effect flipping a card face-down is not a set.
        """

        if len(body) < 9 or body[4] not in (0, 1):
            return
        controller = int(body[4])
        location = int(body[5]) & 0x7F
        sequence = int(body[6])
        new_position = int(body[8]) & 0x0F
        if location not in _FIXED_LOCATIONS:
            return
        zone = (controller, location)
        slots = self._slot_turns.get(zone, {})
        record = slots.get(sequence)
        facedown = 1 if (new_position & C.POS_FACEDOWN) else 0
        if record is None:
            # We never saw this slot's card arrive (the ledger joined mid-game, say), so the flip is taken as
            # the arrival turn: an underestimate, marked as not set.
            slots[sequence] = [self._turn, 0, facedown]
        else:
            # ``pcard->turnid`` records the arrival turn and a flip does not change it; a flip is not a set either,
            # so ``STATUS_SET_TURN`` is cleared with it.
            record[1] = 0
            record[2] = facedown
        self._slot_turns[zone] = slots

    # -- writing -----------------------------------------------------------

    def disclose_faceup_extra(
        self,
        controller: int,
        code: int,
        *,
        copies: int = 1,
        audience: int = 0b11,
    ) -> None:
        """Seed a checked public face-up Extra Deck multiplicity lower bound."""

        controller = int(controller)
        code = int(code) & 0x7FFFFFFF
        copies = max(0, int(copies))
        audience = int(audience) & 0b11
        if not code or not copies or not audience:
            return
        for viewer in (0, 1):
            if not (audience & (1 << viewer)):
                continue
            key = (controller, code)
            self._faceup_extra[viewer][key] = max(
                int(self._faceup_extra[viewer].get(key, 0)), copies
            )
            known_key = (controller, C.LOCATION_EXTRA, code)
            self._viewer_counts[viewer][known_key] = max(
                int(self._viewer_counts[viewer].get(known_key, 0)), copies
            )

    def clear(self) -> None:
        for viewer in (0, 1):
            self._viewer_counts[viewer].clear()
            self._slots[viewer].clear()
            self._fresh[viewer].clear()
            self._faceup_extra[viewer].clear()
            self._claims[viewer].clear()
        self._last_chaining = 0
        self._chain_stack.clear()
        self._slot_turns.clear()
        self._turn = 0

    def disclose(
        self,
        controller: int,
        location: int,
        code: int,
        *,
        sequence: int | None = None,
        audience: int = 0b11,
    ) -> None:
        """Raise each entitled viewer's known multiplicity idempotently.

        A coordinate-free disclosure proves at least one copy.  A disclosure
        carrying a sequence anchors one distinct instance to that slot; seeing
        the same slot twice cannot reveal a second copy.  Losing the anchors
        (a shuffle we cannot follow) keeps the multiset lower bound.
        """

        controller = int(controller)
        location = int(location)
        code = int(code) & 0x7FFFFFFF
        audience = int(audience) & 0b11
        if location & C.LOCATION_OVERLAY:
            # Confirm/chaining coordinates do not carry a material index.
            # Only MOVE's fourth location byte can anchor an overlay card.
            sequence = None
        if not code or not audience:
            return
        key = (controller, location, code)
        zone = (controller, location)
        for viewer in (0, 1):
            if not (audience & (1 << viewer)):
                continue
            counts = self._viewer_counts[viewer]
            if sequence is None:
                counts[key] = max(int(counts.get(key, 0)), 1)
                continue
            slots = self._slots[viewer].setdefault(zone, {})
            fresh = self._fresh[viewer].get(zone)
            arrived = bool(fresh) and int(sequence) in fresh
            slots[int(sequence)] = code
            if arrived:
                fresh.discard(int(sequence))
            distinct = sum(1 for value in slots.values() if value == code)
            if arrived:
                # This slot holds the **newly arrived** card (see :meth:`_slot_arrive`), which is none of the
                # instances proven before, so the lower bound really grows by one. After a hand shuffle the anchors
                # are gone and ``distinct`` counts only this slot; on its own it would count a second searched
                # copy of the same card as the same card, which is how two Hornet Drones once collapsed into one.
                #
                counts[key] = max(int(counts.get(key, 0)) + 1, distinct)
            else:
                counts[key] = max(int(counts.get(key, 0)), distinct)
            # An exact identity makes the category constraints on the same slot redundant.
            self._drop_claims_at(viewer, zone, int(sequence))

    # -- maintaining slot anchors ------------------------------------------

    def _untrack_zone(self, viewer: int, controller: int, location: int) -> None:
        """A shuffle: positions are scrambled and anchors are void; the multiset layer stays.

        Counts of the fixed zones mean something only when pinned to a slot (see :data:`_FIXED_LOCATIONS`), so
        those two zones lose their counts too.
        """

        zone = (int(controller), int(location))
        self._slots[viewer].pop(zone, None)
        # After a shuffle a sequence no longer points at the card it did, so "which slot is fresh" means nothing either.
        self._fresh[viewer].pop(zone, None)
        if int(location) in _FIXED_LOCATIONS:
            self._drop_zone_counts(viewer, zone)
            self._claims[viewer].pop(zone, None)
            return
        claims = self._claims[viewer].get(zone)
        if claims:
            self._claims[viewer][zone] = [
                (None, codes, source) for _sequence, codes, source in claims]

    def _drop_zone_counts(self, viewer: int, zone: tuple[int, int]) -> None:
        counts = self._viewer_counts[viewer]
        for key in [key for key in counts
                    if key[0] == zone[0] and key[1] == zone[1]]:
            del counts[key]
        if zone[1] == C.LOCATION_EXTRA:
            for key in [key for key in self._faceup_extra[viewer]
                        if key[0] == zone[0]]:
                del self._faceup_extra[viewer][key]

    def _forget_zone_for_viewer(
        self, viewer: int, controller: int, location: int
    ) -> None:
        """Forget the whole zone. Used only when **membership really changed and the ledger cannot follow** (graveyard
        and deck swapped, set cards shuffled). A card of unknown identity leaving does **not** take this path; it takes
        :meth:`_depart_unidentified`."""

        zone = (int(controller), int(location))
        self._drop_zone_counts(viewer, zone)
        self._slots[viewer].pop(zone, None)
        self._fresh[viewer].pop(zone, None)
        self._claims[viewer].pop(zone, None)

    def forget_zone(self, controller: int, location: int) -> None:
        controller, location = int(controller), int(location)
        for viewer in (0, 1):
            self._forget_zone_for_viewer(viewer, controller, location)

    def _anchored(self, viewer: int, zone: tuple[int, int]) -> Counter:
        slots = self._slots[viewer].get(zone)
        return Counter(slots.values()) if slots else Counter()

    def _slot_depart(self, viewer: int, zone: tuple[int, int], sequence: int
                     ) -> int:
        """A card leaves slot ``sequence`` of ``zone``; returns the proven code of that slot.

        A vector zone shifts every later anchor down by one, following ``field::remove_card`` + ``reset_sequence``;
        a fixed zone only removes this slot.
        """

        slots = self._slots[viewer].get(zone)
        fresh = self._fresh[viewer].get(zone)
        if slots is None and fresh is None:
            return 0
        code = int(slots.pop(int(sequence), 0)) if slots is not None else 0
        if fresh is not None:
            fresh.discard(int(sequence))
        if zone[1] in _LIST_LOCATIONS or zone[1] & C.LOCATION_OVERLAY:
            def later(slot):
                return slot > int(sequence) and (not zone[1] & C.LOCATION_OVERLAY
                                                 or slot >> 8 == int(sequence) >> 8)
            if slots is not None:
                shifted = {}
                for slot, value in slots.items():
                    shifted[slot - 1 if later(slot) else slot] = value
                self._slots[viewer][zone] = shifted
            if fresh is not None:
                self._fresh[viewer][zone] = {
                    slot - 1 if later(slot) else slot for slot in fresh}
        self._shift_claims(viewer, zone, int(sequence), departed=True)
        return code

    def _slot_arrive(self, viewer: int, zone: tuple[int, int], sequence: int,
                     code: int) -> None:
        """A card arrives in slot ``sequence`` of ``zone``; ``code`` 0 means the identity is unknown."""

        slots = self._slots[viewer].get(zone)
        if slots is None:
            slots = {}
        fresh = set(self._fresh[viewer].get(zone) or ())
        if zone[1] in _LIST_LOCATIONS or zone[1] & C.LOCATION_OVERLAY:
            def later(slot):
                return slot >= int(sequence) and (not zone[1] & C.LOCATION_OVERLAY
                                                  or slot >> 8 == int(sequence) >> 8)
            shifted = {}
            for slot, value in slots.items():
                shifted[slot + 1 if later(slot) else slot] = value
            slots = shifted
            fresh = {slot + 1 if later(slot) else slot for slot in fresh}
        else:
            slots.pop(int(sequence), None)
            fresh.discard(int(sequence))
        self._shift_claims(viewer, zone, int(sequence), departed=False)
        if code:
            slots[int(sequence)] = int(code)
            fresh.discard(int(sequence))
            self._drop_claims_at(viewer, zone, int(sequence))
        else:
            # A card of unknown identity arrives in this slot. It is none of the instances proven before,
            # so if a later message shows this slot's identity, that is evidence of **one more** copy,
            # not double booking (see :meth:`disclose`).
            fresh.add(int(sequence))
        self._slots[viewer][zone] = slots
        self._fresh[viewer][zone] = fresh

    def _lower_counts_after_a_blind_departure(
            self, viewer: int, zone: tuple[int, int]) -> None:
        """What remains of the multiset layer after a card of unknown identity leaves ``zone``.

        The lower bound a spectator can always derive: the card that left takes away at most **one** guaranteed copy
        of one code, so every code's lower bound drops by one; the cards pinned to other slots cannot have left and
        Fixed zones have no legal "slot unknown" bookkeeping (see :data:`_FIXED_LOCATIONS`),
        so those two zones fall back to the anchor counts.

        There are two callers: an unidentified card leaving (:meth:`_depart_unidentified`) and the unrecognized side
        of a cross-zone swap (:meth:`_swap_for_viewer`). Both apply the same rule, and two copies would drift
        apart. The anchors must already be updated to the state after the departure.
        """

        anchored = self._anchored(viewer, zone)
        counts = self._viewer_counts[viewer]
        fixed = zone[1] in _FIXED_LOCATIONS
        for key in [key for key in counts if (key[0], key[1]) == zone]:
            floor = int(anchored.get(key[2], 0))
            new = floor if fixed else max(floor, int(counts[key]) - 1)
            if new <= 0:
                del counts[key]
            else:
                counts[key] = new

    def _depart_unidentified(self, viewer: int, zone: tuple[int, int],
                             sequence: int) -> None:
        """A card of unknown identity leaves ``zone``: the core rule of this ledger.

        The lower bound a spectator can always derive: the card that left takes away at most **one** guaranteed copy
        of one code, so every code's lower bound drops by one. The identities pinned to other slots cannot have left
        and stay (``anchored``). The old ledger cleared the whole zone here and erased proven identities with it.

        Fixed zones have no legal "slot unknown" bookkeeping (see :data:`_FIXED_LOCATIONS`),
        so those two zones fall back to the anchor counts.
        """

        fresh_hand = zone[1] == C.LOCATION_HAND and sequence in self._fresh[viewer].get(zone, ())
        pinned = self._slot_depart(viewer, zone, sequence)
        counts = self._viewer_counts[viewer]
        if pinned:
            # Which card left is actually proven: lower only that code.
            key = (zone[0], zone[1], pinned)
            if counts.get(key, 0) > 0:
                counts[key] -= 1
                if counts[key] <= 0:
                    del counts[key]
            self._drop_unanchored_claims(viewer, zone, departed_code=pinned)
            return
        if fresh_hand:
            # This publicly tracked arrival cannot be any member of the old
            # shuffled group. Neither its identities nor its old unpositioned
            # category facts are weakened when the fresh card leaves hidden.
            return
        self._lower_counts_after_a_blind_departure(viewer, zone)
        if zone[1] == C.LOCATION_EXTRA:
            for key in list(self._faceup_extra[viewer]):
                if key[0] != zone[0]:
                    continue
                new = max(0, int(self._faceup_extra[viewer][key]) - 1)
                capped = int(counts.get((zone[0], C.LOCATION_EXTRA, key[1]), 0))
                new = min(new, capped)
                if new <= 0:
                    del self._faceup_extra[viewer][key]
                else:
                    self._faceup_extra[viewer][key] = new
        self._drop_unanchored_claims(viewer, zone, departed_code=None)

    # -- category constraints ------------------------------------------------

    def _add_claim(self, viewer: int, zone: tuple[int, int], sequence: int | None,
                   codes: frozenset[int], source: int) -> None:
        if not codes:
            return
        self._claims[viewer].setdefault(zone, []).append(
            (sequence, codes, int(source)))

    def _drop_claims_at(self, viewer: int, zone: tuple[int, int], sequence: int
                        ) -> None:
        claims = self._claims[viewer].get(zone)
        if not claims:
            return
        kept = [item for item in claims if item[0] != sequence]
        if kept:
            self._claims[viewer][zone] = kept
        else:
            self._claims[viewer].pop(zone, None)

    def _shift_claims(self, viewer: int, zone: tuple[int, int], sequence: int,
                      *, departed: bool) -> None:
        claims = self._claims[viewer].get(zone)
        if not claims:
            return
        out = []
        for slot, codes, source in claims:
            if slot is None:
                out.append((None, codes, source))
                continue
            if departed:
                if slot == sequence:
                    continue  # this slot left; its constraint goes with it
                if zone[1] in _LIST_LOCATIONS and slot > sequence:
                    slot -= 1
            elif zone[1] in _LIST_LOCATIONS and slot >= sequence:
                slot += 1
            out.append((slot, codes, source))
        if out:
            self._claims[viewer][zone] = out
        else:
            self._claims[viewer].pop(zone, None)

    def _drop_unanchored_claims(self, viewer: int, zone: tuple[int, int], *,
                                departed_code: int | None) -> None:
        """Whether the constraints that lost their anchors still hold after a departure.

        Constraints with anchors are unaffected: another slot left. For those without anchors (after a hand shuffle),
        the card that left may be the constrained one, so they are all dropped, unless we know which code left and
        that code is not in the constraint's admissible set.
        """

        claims = self._claims[viewer].get(zone)
        if not claims:
            return
        out = []
        for slot, codes, source in claims:
            if slot is not None:
                out.append((slot, codes, source))
                continue
            if departed_code and departed_code not in codes:
                out.append((slot, codes, source))
        if out:
            self._claims[viewer][zone] = out
        else:
            self._claims[viewer].pop(zone, None)

    def _record_restriction(self, viewer: int, controller: int, location: int,
                            sequence: int) -> None:
        allowed = self._restrictions.get(self._last_chaining)
        if not allowed:
            return
        self._add_claim(viewer, (int(controller), int(location)), int(sequence),
                        allowed, self._last_chaining)

    # -- visibility predicate ------------------------------------------------

    @staticmethod
    def _source_exact_at(value: int, viewer: int) -> bool:
        controller = int(value) & 0xFF
        location = (int(value) >> 8) & 0xFF
        position = (int(value) >> 24) & 0xFF
        if location & C.LOCATION_OVERLAY:
            return True
        if controller == viewer:
            # A player knows their deck multiset, never the shuffled raw slot.
            return location != C.LOCATION_DECK
        if location in (
            C.LOCATION_DECK,
            C.LOCATION_HAND,
            C.LOCATION_EXTRA,
        ):
            return False
        return not bool(position & C.POS_FACEDOWN)

    @staticmethod
    def _destination_visible_at(value: int, viewer: int) -> bool:
        controller = int(value) & 0xFF
        location = (int(value) >> 8) & 0xFF
        position = (int(value) >> 24) & 0xFF
        if controller == viewer:
            # single_duel sends players[current_controller] the full code first.
            return True
        hidden_facedown = bool(position & C.POS_FACEDOWN) and not bool(
            position & C.POS_REVEAL
        )
        if not (location & (C.LOCATION_GRAVE | C.LOCATION_OVERLAY)) and (
            location & (C.LOCATION_DECK | C.LOCATION_HAND) or hidden_facedown
        ):
            return False
        return True

    # -- moves ---------------------------------------------------------------

    def observe_move(
        self,
        previous: int,
        current: int,
        code: int,
        *,
        viewer: int | None = None,
    ) -> None:
        """A card changes zones.

        ``previous`` / ``current`` are location words that the server **does not erase** (the ``MSG_MOVE`` branch
        of ``single_duel.cpp`` only zeroes ``code``), so both sequences are public to every observer.
        Whether ``code`` was erased and whether this observer can recognize the card is decided by the predicate
        below and the slot anchors; an unrecognized card takes
        :meth:`_depart_unidentified`, **no longer forgetting the whole zone**.
        """

        code = int(code) & 0x7FFFFFFF
        if not previous:
            return
        from_controller = int(previous) & 0xFF
        from_location = (int(previous) >> 8) & 0xFF
        from_sequence = (int(previous) >> 16) & 0xFF
        from_position = (int(previous) >> 24) & 0xFF
        from_zone = (from_controller, from_location)
        to_controller = int(current) & 0xFF if current else 0
        to_location = (int(current) >> 8) & 0xFF if current else 0
        to_sequence = (int(current) >> 16) & 0xFF if current else 0
        to_position = (int(current) >> 24) & 0xFF if current else 0
        to_zone = (to_controller, to_location)
        # An overlay coordinate names both its host and its material index.
        # Hosts moving/switching control invalidate those coordinates; the
        # public wire identities remain usable without stale host anchors.
        # A summon may attach materials before its host leaves Extra. Any
        # host move/list insertion reuses host indices, not only MZONE moves.
        # Retire those material anchors before consulting a later MOVE; an old
        # Extra host must not override a new material's public wire identity.
        for controller, location in (from_zone, to_zone):
            if location and not location & C.LOCATION_OVERLAY:
                self.forget_zone(controller, location | C.LOCATION_OVERLAY)
        # The face-down slot history is a fact both sides can see, independent of the observer, so it is updated
        # once before the per-observer bookkeeping. The server does not erase location words or positions
        # (``single_duel.cpp:1011-1035`` only zeroes ``code``), so this step uses public information
        # throughout.
        self._slot_history_depart(from_zone, from_sequence)
        if current:
            self._slot_history_arrive(to_zone, to_sequence, to_position)
        viewers = (int(viewer),) if viewer in (0, 1) else (0, 1)
        for target_viewer in viewers:
            self._move_for_viewer(
                target_viewer,
                wire_code=code,
                wire_routed_to=viewer,
                previous=previous,
                current=current,
                from_zone=from_zone,
                from_sequence=from_sequence,
                from_position=from_position,
                to_zone=to_zone,
                to_sequence=to_sequence,
                to_position=to_position,
            )

    def _identified_code_for(
        self, viewer: int, *, wire_code: int, wire_routed_to: int | None,
        previous: int, current: int, from_zone: tuple[int, int],
        from_sequence: int,
    ) -> int:
        """Can this observer tell which card left? Returns its code if so, else 0.

        Three independent reasons:

        1. the slot was pinned before (a slot anchor, the observer's own public reasoning);
        2. the source zone is open (face-up field / graveyard / face-up banished / one's own non-deck zones);
        3. the destination is open (which is exactly how the server decides whether to send ``code``).

        An anchor takes precedence over the ``code`` on the wire: anchors do not depend on the truth, so two different
        truths behind the same public message stream yield the same ledger.
        """

        pinned = self.known_code_at(
            viewer, from_zone[0], from_zone[1], from_sequence)
        if pinned:
            return int(pinned)
        identified = (
            self._source_exact_at(previous, viewer)
            or (bool(current) and self._destination_visible_at(current, viewer))
            or (wire_routed_to == viewer and bool(wire_code))
        )
        return int(wire_code) if identified else 0

    def _move_for_viewer(
        self, viewer: int, *, wire_code: int, wire_routed_to: int | None,
        previous: int, current: int, from_zone: tuple[int, int],
        from_sequence: int, from_position: int, to_zone: tuple[int, int],
        to_sequence: int, to_position: int,
    ) -> None:
        """Book one move in the ledger of **one** observer."""

        if from_zone[1] & C.LOCATION_OVERLAY:
            from_sequence = (from_sequence << 8) | from_position
        if to_zone[1] & C.LOCATION_OVERLAY:
            to_sequence = (to_sequence << 8) | to_position
        # Private deck-sort markers erase coordinates, not already-known
        # membership. Carrying the old anchor into the reordered deck can
        # subsequently fabricate a different card in the opponent's hand.
        within_deck = from_zone == to_zone and from_zone[1] == C.LOCATION_DECK
        if within_deck and not wire_code and from_sequence == to_sequence:
            self._untrack_zone(viewer, from_zone[0], from_zone[1])
            return
        known_code = self._identified_code_for(
            viewer, wire_code=wire_code, wire_routed_to=wire_routed_to,
            previous=previous, current=current, from_zone=from_zone,
            from_sequence=from_sequence)
        if not known_code:
            if within_deck:
                self._untrack_zone(viewer, from_zone[0], from_zone[1])
                return
            self._depart_unidentified(viewer, from_zone, from_sequence)
            if current:
                # The destination's sequence shift is public; failing to follow it would misalign that zone's anchors.
                self._slot_arrive(viewer, to_zone, to_sequence, 0)
                self._maybe_record_restriction(
                    viewer, from_zone[1], to_zone[0], to_zone[1],
                    to_sequence, to_position)
            return

        counts = self._viewer_counts[viewer]
        from_key = (from_zone[0], from_zone[1], known_code)
        # Is the card that left one of those this observer knows: the slot was proven to be it; or the zone still
        # holds known but unanchored copies of that card (which one left is unknown, the lower bound drops by one).
        # When the known copies are all pinned to other slots, the card that left was one not recognized before
        # that showed its code only as it left; those copies cannot have left and the lower bound stays; the same principle as :meth:`_depart_unidentified`.
        slots = self._slots[viewer].get(from_zone) or {}
        fresh_hand = from_zone[1] == C.LOCATION_HAND and from_sequence in self._fresh[viewer].get(from_zone, ())
        known_departed = slots.get(from_sequence) == known_code or (not fresh_hand and int(counts.get(from_key, 0)) > sum(
            1 for value in slots.values() if value == known_code))
        self._adjust_faceup_extra(viewer, from_zone[0], from_zone[1],
                                  from_position, known_code, delta=-1)
        self._slot_depart(viewer, from_zone, from_sequence)
        if not fresh_hand:
            self._drop_unanchored_claims(
                viewer, from_zone, departed_code=known_code)
        if known_departed and counts.get(from_key, 0) > 0:
            counts[from_key] -= 1
            if counts[from_key] <= 0:
                del counts[from_key]
        if not current:
            return
        counts[(to_zone[0], to_zone[1], known_code)] += 1
        self._slot_arrive(viewer, to_zone, to_sequence, known_code)
        self._adjust_faceup_extra(viewer, to_zone[0], to_zone[1],
                                  to_position, known_code, delta=+1)

    def _adjust_faceup_extra(self, viewer: int, controller: int, location: int,
                             position: int, code: int, *, delta: int) -> None:
        """Changes of the **face-up** column of the Extra Deck. Face-down extra monsters are not counted."""

        if location != C.LOCATION_EXTRA or bool(position & C.POS_FACEDOWN):
            return
        key = (controller, code)
        counter = self._faceup_extra[viewer]
        counter[key] = int(counter.get(key, 0)) + int(delta)
        if counter[key] <= 0:
            del counter[key]

    def _maybe_record_restriction(self, viewer: int, from_location: int,
                                  to_controller: int, to_location: int,
                                  to_sequence: int, to_position: int) -> None:
        """A card taken from the deck by an effect without its identity being public: record the slot's category constraint.

        Two destinations: to the hand (search) or set on the field. The activating card's identity comes from
        ``MSG_CHAINING`` (public); the admissible set comes from the external card-text table.
        """

        if from_location != C.LOCATION_DECK or not self._last_chaining:
            return
        if to_location == C.LOCATION_HAND:
            self._record_restriction(viewer, to_controller, to_location,
                                     to_sequence)
        elif to_location in _FIXED_LOCATIONS and (to_position & C.POS_FACEDOWN):
            self._record_restriction(viewer, to_controller, to_location,
                                     to_sequence)

    # -- message parsing. Both sides feed raw payloads in; this is the only parser. ---------------

    def observe_chaining(self, body: bytes) -> None:
        """``MSG_CHAINING``: an activation is public wherever it comes from (a hand trap activated from the hand is public too)."""
        if len(body) < 16:
            return
        code, at = struct.unpack_from("<II", body, 0)
        self._last_chaining = int(code) & 0x7FFFFFFF
        self._chain_stack.append(self._last_chaining)
        self.disclose(
            at & 0xFF,
            (at >> 8) & 0xFF,
            int(code),
            sequence=(at >> 16) & 0xFF,
            audience=0b11,
        )

    def observe_chain_solving(self, body: bytes) -> None:
        """``MSG_CHAIN_SOLVING``: the payload is the number (from 1) of the chain link resolving.

        ``processor.cpp:4399-4400`` writes ``cait->chain_count``. A chain resolves backwards,
        so only this message tells which card is resolving now; taking the last ``MSG_CHAINING``
        instead would credit the search of chain link 1 to the activating card of link 2 and book the category constraint wrongly.
        """

        if not body:
            return
        index = int(body[0])
        if 1 <= index <= len(self._chain_stack):
            self._last_chaining = self._chain_stack[index - 1]

    def observe_chain_end(self) -> None:
        """``MSG_CHAIN_END``: the chain is over and the stack is empty."""

        self._chain_stack.clear()
        self._last_chaining = 0

    def observe_confirm(self, body: bytes, msg: int) -> None:
        """``MSG_CONFIRM_CARDS`` / ``_DECKTOP`` / ``_EXTRATOP``: revealing makes a card public.

        7 bytes per card: ``code(4) controler(1) location(1) sequence(1)``.
        ``MSG_DECK_TOP`` is deliberately not handled: it gives only the controler and a code with a high flag bit,
        which is not enough for a zone, and it refers to a card in the deck.
        """
        head = 3 if msg == C.MSG_CONFIRM_CARDS else 2
        if len(body) <= head:
            return
        count, offset = body[head - 1], head
        records = []
        for _ in range(count):
            if offset + 7 > len(body):
                break
            (code,) = struct.unpack_from("<I", body, offset)
            records.append((
                body[offset + 4],
                body[offset + 5],
                body[offset + 6],
                code & 0x7FFFFFFF,
            ))
            offset += 7
        if not records:
            return
        # single_duel.cpp sends a deck MSG_CONFIRM_CARDS only to body[0],
        # while every other confirmation family/zone is broadcast.
        deck_owner_only = (
            msg == C.MSG_CONFIRM_CARDS
            and records[0][1] == C.LOCATION_DECK
        )
        audience = (1 << int(body[0])) if deck_owner_only else 0b11
        for controller, location, sequence, code in records:
            self.disclose(
                controller,
                location,
                code,
                sequence=sequence,
                audience=audience,
            )

    def observe_move_message(
        self, body: bytes, *, viewer: int | None = None
    ) -> None:
        if len(body) < 16:
            return
        code, previous, current, _reason = struct.unpack_from("<IIII", body, 0)
        self.observe_move(previous, current, code, viewer=viewer)

    def observe_swap(self, body: bytes) -> None:
        """``MSG_SWAP``: two field cards **swap** places (``field.cpp:384-450``).

        Payload ``code1(4) location word1(4) code2(4) location word2(4)``; both location words are from **before**
        the swap (``field.cpp:388`` reads ``get_info_location()`` first). The core sends it only when
        ``s1 == new_sequence1 && s2 == new_sequence2``, that is, when the two cards **exchange
        coordinates**; when sequences change otherwise it sends two ``MSG_MOVE`` instead (``field.cpp:448-470``),
        which take :meth:`observe_move`. Both sources require ``LOCATION_ONFIELD``:
        ``Duel.SwapSequence`` is limited to the same controller's monster zones (``libduel.cpp:921-934``), and
        a control swap to the monster zones of both sides (``operations.cpp:1108-1122``).

        **Not handling this message corrupts the ledger for good.** A slot anchor proves "this slot is this card";
        after a swap the anchor stays put, and every later ``MSG_MOVE`` through those two slots would be booked by
        :meth:`_identified_code_for` under the wrong code: the card that really left is never deducted,
        so its lower bound stays in the zone forever, finally showing up as
        ``revealed_unpositioned`` putting 2 identities into a zone with only 1 hidden slot left
        (Elfnote against CenturIon, seed 4105, turn 21).

        **The ``code`` on the wire is not evidence.** ``single_duel.cpp:1060-1074`` forwards this message
        to both sides unchanged, not one byte erased (unlike the ``MSG_MOVE`` branch, which zeroes ``code`` by the
        destination's visibility), so "received a non-zero ``code``" does not mean this observer may know it.
        So only two public reasons count here: the slot was pinned before (a slot anchor), or the location word itself
        is public (face-up on the field, or the destination belongs to the observer); exactly the first
        two reasons of :meth:`observe_move`, without the third, "the message is distributed per observer". That is also
        why this method has no ``viewer`` parameter.
        """

        if len(body) < 16:
            return
        code1, first, code2, second = struct.unpack_from("<IIII", body, 0)
        left = (int(first) & 0xFF, (int(first) >> 8) & 0xFF)
        left_sequence = (int(first) >> 16) & 0xFF
        right = (int(second) & 0xFF, (int(second) >> 8) & 0xFF)
        right_sequence = (int(second) >> 16) & 0xFF
        if left[1] not in _FIXED_LOCATIONS or right[1] not in _FIXED_LOCATIONS:
            # ``swap_card`` has already returned before this point and writes not a single byte.
            return
        if left == right and left_sequence == right_sequence:
            return
        for controller, location in {left, right}:
            if location == C.LOCATION_MZONE:
                self.forget_zone(controller, C.LOCATION_MZONE | C.LOCATION_OVERLAY)
        # After the swap each card sits at the other's coordinates, and **the position goes with the card** (``swap_card``
        # does not touch ``current.position``), so the destination location word combines the other's coordinates with its own position.
        left_destination = (int(second) & 0x00FFFFFF) | (int(first) & 0xFF000000)
        right_destination = (int(first) & 0x00FFFFFF) | (int(second) & 0xFF000000)
        self._swap_slot_history(left, left_sequence, right, right_sequence)
        for viewer in (0, 1):
            self._swap_for_viewer(
                viewer,
                left=left, left_sequence=left_sequence,
                right=right, right_sequence=right_sequence,
                known_left=self._identified_code_for(
                    viewer, wire_code=int(code1) & 0x7FFFFFFF,
                    wire_routed_to=None, previous=int(first),
                    current=left_destination, from_zone=left,
                    from_sequence=left_sequence),
                known_right=self._identified_code_for(
                    viewer, wire_code=int(code2) & 0x7FFFFFFF,
                    wire_routed_to=None, previous=int(second),
                    current=right_destination, from_zone=right,
                    from_sequence=right_sequence),
            )

    def _swap_for_viewer(self, viewer: int, *, left: tuple[int, int],
                         left_sequence: int, right: tuple[int, int],
                         right_sequence: int, known_left: int,
                         known_right: int) -> None:
        """Book one swap in the ledger of **one** observer.

        The order matters: the multiset layer must be computed **before the anchors are swapped**, because the
        unrecognized side falls back to the anchor count from "before the swap" (the same rule as the fixed-zone
        branch of :meth:`_depart_unidentified`); computing it after would count the card just swapped in as one that was always there.
        """

        if left != right:
            # Only a cross-zone swap (a control change) really takes a card out of a zone; a same-zone swap leaves the multiset alone.
            for zone, known in ((left, known_left), (right, known_right)):
                if not known:
                    self._lower_counts_after_a_blind_departure(viewer, zone)
            for source, target, known in ((left, right, known_left),
                                          (right, left, known_right)):
                if known:
                    self._retag_zone_count(viewer, source, target, known)
                self._drop_unanchored_claims(
                    viewer, source, departed_code=known or None)
        left_state = self._take_slot_state(viewer, left, left_sequence)
        right_state = self._take_slot_state(viewer, right, right_sequence)
        self._put_slot_state(viewer, right, right_sequence, left_state)
        self._put_slot_state(viewer, left, left_sequence, right_state)

    def _retag_zone_count(self, viewer: int, source: tuple[int, int],
                          target: tuple[int, int], code: int) -> None:
        """A card of proven identity moves from zone ``source`` to zone ``target``."""

        counts = self._viewer_counts[viewer]
        from_key = (source[0], source[1], int(code))
        if counts.get(from_key, 0) > 0:
            counts[from_key] -= 1
            if counts[from_key] <= 0:
                del counts[from_key]
        counts[(target[0], target[1], int(code))] += 1

    def _take_slot_state(self, viewer: int, zone: tuple[int, int],
                         sequence: int) -> tuple[int, bool, tuple]:
        """Take a slot's anchor, fresh mark and category constraints off as a whole.

        ``fresh`` **goes with the card** and cannot be derived again at the destination: it means "this slot holds a
        card that arrived and whose identity is still undisclosed, none of the instances proven before". After a swap
        the card is still that card, so the mark moves with it to the new slot; setting it anew there would make a later
        confirmation count it as **one more copy** (see the ``arrived`` branch of :meth:`disclose`).
        """

        slots = dict(self._slots[viewer].get(zone) or {})
        code = int(slots.pop(int(sequence), 0))
        fresh = set(self._fresh[viewer].get(zone) or ())
        was_fresh = int(sequence) in fresh
        fresh.discard(int(sequence))
        self._slots[viewer][zone] = slots
        self._fresh[viewer][zone] = fresh
        claims = self._claims[viewer].get(zone) or []
        taken = tuple(item for item in claims if item[0] == int(sequence))
        kept = [item for item in claims if item[0] != int(sequence)]
        if kept:
            self._claims[viewer][zone] = kept
        else:
            self._claims[viewer].pop(zone, None)
        return code, was_fresh, taken

    def _put_slot_state(self, viewer: int, zone: tuple[int, int], sequence: int,
                        state: tuple[int, bool, tuple]) -> None:
        """The reverse of :meth:`_take_slot_state`; written back in the same order as :meth:`_slot_arrive`
        (delete, then write, so the slot comes last in the dictionary)."""

        code, was_fresh, claims = state
        slots = dict(self._slots[viewer].get(zone) or {})
        fresh = set(self._fresh[viewer].get(zone) or ())
        slots.pop(int(sequence), None)
        fresh.discard(int(sequence))
        if code:
            slots[int(sequence)] = int(code)
        if was_fresh:
            fresh.add(int(sequence))
        self._slots[viewer][zone] = slots
        self._fresh[viewer][zone] = fresh
        for _sequence, codes, source in claims:
            self._add_claim(viewer, zone, int(sequence), codes, source)

    def _swap_slot_history(self, left: tuple[int, int], left_sequence: int,
                           right: tuple[int, int], right_sequence: int) -> None:
        """The face-down slot history goes with the card: ``pcard->turnid`` and "was it set" are properties of the
        card, not of the slot, and a swap resets neither."""

        by_zone: dict[tuple[int, int], dict[int, list[int]]] = {}
        for zone in (left, right):
            if zone not in by_zone:
                by_zone[zone] = self._slot_turns.get(zone) or {}
        left_record = by_zone[left].pop(int(left_sequence), None)
        right_record = by_zone[right].pop(int(right_sequence), None)
        if right_record is not None:
            by_zone[left][int(left_sequence)] = right_record
        if left_record is not None:
            by_zone[right][int(right_sequence)] = left_record
        for zone, slots in by_zone.items():
            if slots:
                self._slot_turns[zone] = slots
            else:
                self._slot_turns.pop(zone, None)

    def observe_draw(
        self, body: bytes, *, viewer: int | None = None, hand_start: int | None = None
    ) -> None:
        """Move viewer-routed draw identities from DECK knowledge into HAND.

        A draw takes from the top of the deck (``field::draw`` uses ``list_main.back()``) and lands at the end of
        the hand (``add_card`` does ``push_back`` for ``LOCATION_HAND``), so:

        * anchors already in the hand **do not shift** and need not be voided;
        * when the caller has the public hand size, new slots start at hand_start; a hidden draw is recorded as
          fresh and a public draw may create a real anchor. A caller without the public count still creates no positions;
        * the deck side takes the top cards; the ledger does not know the deck size and cannot follow anchors,
          so deck anchors are cleared and the multiset drops by the lower bound "every code minus n".
        """

        if len(body) < 2 or body[0] not in (0, 1):
            return
        player = int(body[0])
        count = int(body[1])
        if hand_start is not None and (type(hand_start) is not int or not 0 <= hand_start <= 255
                                      or hand_start + count > 255 or len(body) != 2 + 4 * count):
            raise ValueError("draw slot tracking needs the exact public pre-draw hand count and complete packet")
        raw_codes = []
        for index in range(count):
            offset = 2 + 4 * index
            raw = (
                struct.unpack_from("<I", body, offset)[0]
                if offset + 4 <= len(body) else 0
            )
            raw_codes.append(int(raw))
        viewers = (int(viewer),) if viewer in (0, 1) else (0, 1)
        deck_zone = (player, C.LOCATION_DECK)
        for target_viewer in viewers:
            known = []
            for raw in raw_codes:
                if viewer == target_viewer:
                    # The network body was already routed/redacted.
                    code = raw & 0x7FFFFFFF if raw else 0
                elif target_viewer == player:
                    code = raw & 0x7FFFFFFF
                else:
                    code = raw & 0x7FFFFFFF if raw & 0x80000000 else 0
                known.append(code)
            deck_counts = self._viewer_counts[target_viewer]
            # Deck-top anchors cannot be followed (the deck size is unknown): clear the anchors first, then compute the lower bound.
            self._slots[target_viewer].pop(deck_zone, None)
            self._claims[target_viewer].pop(deck_zone, None)
            unknown = sum(1 for code in known if not code)
            for code in known:
                if not code:
                    continue
                key = (player, C.LOCATION_DECK, code)
                if deck_counts.get(key, 0) > 0:
                    deck_counts[key] -= 1
                    if deck_counts[key] <= 0:
                        del deck_counts[key]
            if unknown:
                for key in [key for key in deck_counts
                            if key[0] == player and key[1] == C.LOCATION_DECK]:
                    new = int(deck_counts[key]) - unknown
                    if new <= 0:
                        del deck_counts[key]
                    else:
                        deck_counts[key] = new
            for code, copies in Counter(known).items():
                if code:
                    deck_counts[(player, C.LOCATION_HAND, code)] += copies
            if hand_start is not None:
                for offset, code in enumerate(known):
                    self._slot_arrive(target_viewer, (player, C.LOCATION_HAND), hand_start + offset, code)

    def observe_deck_top(self, body: bytes) -> None:
        """Only POS_REVEAL/high-bit deck-top identities are public knowledge.

        The second byte of ``MSG_DECK_TOP`` is the **offset from the top**, not the raw sequence of
        ``list_main`` (``ygopro-core/libduel.cpp:997-1004`` writes ``count``,
        ``field.cpp:999-1005`` and ``libcard.cpp:3311-3314`` write 0). Taking it as a
        sequence would pin the anchor in the wrong coordinate system, so this books a disclosure **without coordinates**.
        """

        if len(body) < 6 or body[0] not in (0, 1):
            return
        player = int(body[0])
        raw = struct.unpack_from("<I", body, 2)[0]
        if raw & 0x80000000:
            self.disclose(
                player,
                C.LOCATION_DECK,
                raw & 0x7FFFFFFF,
                sequence=None,
                audience=0b11,
            )

    def observe_reverse_deck(self) -> None:
        for viewer in (0, 1):
            for player in (0, 1):
                self._untrack_zone(viewer, player, C.LOCATION_DECK)

    def observe_shuffle_set_card(self, body: bytes) -> None:
        """Set-card shuffles destroy identity knowledge for affected zones.

        The payload is ``loc(1) count(1)`` followed by ``count`` location words from before the shuffle, then ``count``
        location words of Xyz materials (``ygopro-core/libduel.cpp:1612-1633``). Counts of fixed zones
        mean something only when pinned to slots, so the counts are dropped here too.
        """

        if len(body) < 2:
            return
        count = int(body[1])
        affected = set()
        participants: list[tuple[int, int, int]] = []
        for index in range(count):
            offset = 2 + 4 * index
            if offset + 4 > len(body):
                break
            packed = struct.unpack_from("<I", body, offset)[0]
            if packed:
                affected.add((packed & 0xFF, (packed >> 8) & 0x7F))
                participants.append((packed & 0xFF, (packed >> 8) & 0x7F,
                                     (packed >> 16) & 0xFF))
        for controller, location in affected:
            self._shuffle_set_identities(controller, location, participants)
        self._shuffle_slot_history(participants)

    def _shuffle_set_identities(self, controller: int, location: int,
                                participants) -> None:
        """Shuffling set cards: positions are scrambled, but **which cards** they are has not changed.

        This used to forget the whole zone. That threw away real information: the observer sees which slots take part
        in the shuffle, and if all of them had pinned identities before, they still know afterwards "these slots hold
        exactly these cards, only not which is where".

        So the identities become **one category constraint per slot**, whose admissible set is the multiset of that group.
        The sampler's system of distinct representatives (Hall's condition) expresses exactly "the k slots hold some
        permutation of these k cards". The slot bindings are still cleared, so :func:`resolve_disclosure` gives these slots
        no positions; the positions really are unknown.

        If any slot had no pinned identity before the shuffle, the group cannot be described and the whole zone is forgotten.
        """

        zone = (int(controller), int(location))
        sequences = [sequence for one, other, sequence in participants
                     if (one, other) == zone]
        for viewer in (0, 1):
            slots = self._slots[viewer].get(zone) or {}
            known = [slots.get(sequence) for sequence in sequences]
            if not sequences or any(not code for code in known):
                self._forget_zone_for_viewer(viewer, controller, location)
                continue
            allowed = frozenset(int(code) for code in known)
            self._slots[viewer].pop(zone, None)
            self._claims[viewer].pop(zone, None)
            for sequence in sequences:
                # Every slot is constrained by the same multiset; distinct representatives give each its own card.
                self._add_claim(viewer, zone, int(sequence), allowed, 0)

    def _shuffle_slot_history(self, participants) -> None:
        """What happens to the history of the slots taking part in a set-card shuffle.

        ``MSG_SHUFFLE_SET_CARD`` reports the location words of the cards **before** the shuffle; the new positions
        ``seq[i]`` are not in the message (``ygopro-core/libduel.cpp:1612-1624``), so
        "which card went to which slot" cannot be followed. But a shuffle only permutes these slots, so the
        **multiset** of histories is unchanged: if every participating slot has the same history, the permutation is the
        identity and the records can stay; if any differs, nobody can say which went where, and those slots are all voided.

        Only the participating slots are touched: set cards that did not take part stay where they were with their histories.
        """

        by_zone: dict[tuple[int, int], list[int]] = {}
        for controller, location, sequence in participants:
            by_zone.setdefault((controller, location), []).append(sequence)
        for zone, sequences in by_zone.items():
            slots = self._slot_turns.get(zone)
            if not slots:
                continue
            records = [slots.get(sequence) for sequence in sequences]
            known = [record for record in records if record is not None]
            if len(known) == len(records) and known and all(
                    record == known[0] for record in known):
                continue  # the permutation is the identity; keep them
            for sequence in sequences:
                slots.pop(sequence, None)
            if not slots:
                self._slot_turns.pop(zone, None)

    def observe_shuffle(self, msg: int, body: bytes) -> None:
        """A shuffle retires raw slots, never already-public name counts.

        Grave/deck exchange changes membership and remains the conservative
        exception: those two zone multisets are cleared and rebuilt by moves.
        """
        if not body:
            return
        player = int(body[0])
        if msg in (
            C.MSG_SHUFFLE_DECK,
            C.MSG_SHUFFLE_HAND,
            C.MSG_SHUFFLE_EXTRA,
        ):
            location = {
                C.MSG_SHUFFLE_DECK: C.LOCATION_DECK,
                C.MSG_SHUFFLE_HAND: C.LOCATION_HAND,
                C.MSG_SHUFFLE_EXTRA: C.LOCATION_EXTRA,
            }[msg]
            for viewer in (0, 1):
                self._untrack_zone(viewer, player, location)
            self.forget_zone(player, location | C.LOCATION_OVERLAY)
        elif msg == C.MSG_SWAP_GRAVE_DECK:
            self.forget_zone(player, C.LOCATION_DECK)
            self.forget_zone(player, C.LOCATION_GRAVE)

    def resolve(self, snapshot, viewer: int | None = None) -> frozenset[tuple]:
        if viewer is None:
            return resolve_disclosure(snapshot, self.counts)
        if viewer not in (0, 1):
            raise ValueError("disclosure viewer must be seat zero or one")
        # Audience was fixed at ingest time and moves with knowledge.  Do not
        # infer it again from the card's current location: doing that launders
        # an owner-only DECK confirmation after the card moves into HAND.
        return resolve_disclosure(
            snapshot,
            self._viewer_counts[int(viewer)],
            anchors=self._slots[int(viewer)],
            place_unanchored=False,
        )


def resolve_disclosure(snapshot, disclosed, viewer: int | None = None,
                       *, anchors=None,
                       place_unanchored: bool = True) -> frozenset[tuple]:
    """Resolve "(controller, zone, code) -> disclosed count" into a concrete set of instance coordinates.

    ``DuelDriver.disclosed`` books only a multiset, because hand / deck / graveyard are vectors and
    ``field::remove_card`` silently reorders sequences (see the notes in ``engine.py``). Which sequences they land on
    is arbitrary in terms of information (instances of one code are indistinguishable anyway), so this assigns them
    canonically in **ascending sequence order** instead of reading the engine's true identity map. Excess counts
    are capped by the zone's real count: disclosing the same card twice never also discloses one that was not.

    ``snapshot`` must be the **unmasked** raw snapshot; a masked hand has no codes left to group by.

    ``DisclosureLedger.resolve`` passes an audience-filtered counter.  The
    legacy ``viewer`` argument remains for direct callers with an unfiltered
    triple-key counter; new code must preserve audience at ingest/move time.
    """
    if not disclosed:
        return frozenset()
    groups: dict[tuple[int, int, int], list[int]] = {}
    for card in snapshot.cards:
        if not card.code:
            continue
        groups.setdefault(
            (card.controller, card.location, card.code), []
        ).append(card.sequence)
    out = set()
    remaining = {key: max(int(count), 0) for key, count in disclosed.items()}
    taken: set[tuple[int, int, int]] = set()
    # First pass: slot bindings. Whatever the ledger proved "this slot is this card" goes back to that slot. Positions
    # in the fixed zones are meaningful, and picking any slot in ascending order would move the evidence to another
    # slot; a ``revealed_slot_identity`` violation once came from exactly that: the ledger bound spell & trap slot 1,
    # resolution booked it at slot 0, and slot 1 was then treated as permutable.
    for (controller, location), slots in (anchors or {}).items():
        for sequence, code in (slots or {}).items():
            key = (int(controller), int(location), int(code))
            if remaining.get(key, 0) <= 0:
                continue
            if int(sequence) not in groups.get(key, ()):
                continue  # the binding does not match this snapshot; leave it to the second pass
            out.add((int(controller), int(location), int(sequence), int(code)))
            taken.add((int(controller), int(location), int(sequence)))
            remaining[key] -= 1
    # What to do with the unbound ones depends on whether the caller can take "an identity without a position".
    #
    # ``place_unanchored=False`` (the ledger's per-observer path feeding ``mask_for``): assign no
    # positions. Pinning them to the lowest slots in ascending order was **made-up position**: the observer
    # knows "they hold this card", not which one it is. This part instead goes through
    # :meth:`DisclosureLedger.unanchored_identities` into
    # ``StateSnapshot.revealed_unpositioned``, expressing "what" and "where" separately.
    #
    # ``place_unanchored=True`` (the default, for old callers that only take coordinates): still assign
    # canonically in ascending sequence order. Instances of one code are indistinguishable, so the assignment is
    # arbitrary in terms of information; except in the fixed zones, whose slots have identities, where picking one reports position that is not known.
    if not place_unanchored:
        return frozenset(out)
    for (controller, location, code), count in remaining.items():
        if count <= 0 or location in _FIXED_LOCATIONS:
            continue
        if viewer is not None and controller != viewer \
                and location == C.LOCATION_DECK:
            continue
        sequences = [sequence
                     for sequence in sorted(groups.get(
                         (controller, location, code), ()))
                     if (controller, location, sequence) not in taken]
        for sequence in sequences[:count]:
            out.add((controller, location, sequence, code))
    return frozenset(out)
