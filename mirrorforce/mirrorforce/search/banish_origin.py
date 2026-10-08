"""Public sources of face-down banished slots: which slots hold cards banished face-down from the Extra Deck.

``Debug.PermuteHidden`` splits face-down banished slots into two kinds by their **current occupant**: slots holding a
main-deck card take cards only from the main-deck pool, slots holding an extra monster only from the extra pool (core patch
``mirrorforce/patches/ygopro-core-permute-hidden-extra.patch``). The sampler must know each slot's kind
when sampling, or the engine rejects the sampled assignment.

This fact is **public**. The server does not erase the two location words of ``MSG_MOVE`` (``previous`` / ``current``)
(the ``MSG_MOVE`` branch of ``single_duel.cpp`` only zeroes ``code``), so
"a card was banished face-down from a face-down Extra Deck position" is visible to both sides; only which card it was is not.
This module reads only these two location words and never ``code``; the same public message stream with different hidden truths
gives the same source table (a test pins this).

Why it is not in the disclosure ledger
--------------------

:class:`~mirrorforce.netduel.disclosure.DisclosureLedger` records **identities** (which card a slot
holds, how many known cards a zone has); this records **kinds** (whether a slot holds a main-deck card
or an extra monster), and neither implies the other. The ledger already tracks face-down banished cards as hidden slots, but its slot history
has only ``turnid`` and ``STATUS_SET_TURN``, no source. Like the ledger, this module hangs on
:class:`~mirrorforce.worldmodel.engine.DuelDriver` and is rolled back with core snapshots.

Sequence tracking
--------

``list_remove`` is a vector zone: after ``field::remove_card`` deletes a slot, every later slot shifts down
by one (``reset_sequence`` in ``field.cpp``); ``field::add_card`` appends at the end.
This module uses the same shift rules as the ledger's ``_slot_history_arrive`` / ``_slot_history_depart``,
so the ``(zone, sequence)`` keys both give refer to the same slot.
"""

from __future__ import annotations

import struct

from ..netduel import constants as C

__all__ = ["BanishOriginTracker", "extra_origin_slots_from_messages"]


def _unpack_location(value: int) -> tuple[int, int, int, int]:
    """Split a location word: ``(controller, zone, sequence, position)``, the same calibration as the disclosure ledger."""

    value = int(value)
    return (value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF,
            (value >> 24) & 0xFF)


class BanishOriginTracker:
    """Track, per controller, whether each face-down banished slot holds a card banished face-down from the Extra Deck.

    Consumes only ``MSG_MOVE``. A slot's record is a boolean: ``True`` means the card was **face-down** when it left the
    Extra Deck (a face-down card in the Extra Deck is always an extra monster; a face-up pendulum monster is a main-deck card and,
    once banished face-down, goes through the main-deck pool), ``False`` means it came from elsewhere.
    """

    def __init__(self) -> None:
        self._slots: tuple[dict[int, bool], dict[int, bool]] = ({}, {})

    # -- messages --------------------------------------------------------------

    def observe(self, msg: int, body: bytes) -> None:
        """Feed one raw message; only ``MSG_MOVE`` matters."""

        if int(msg) != C.MSG_MOVE or len(body) < 16:
            return
        _code, previous, current, _reason = struct.unpack_from("<IIII", body, 0)
        self.observe_move(previous, current)

    def observe_move(self, previous: int, current: int) -> None:
        """A card changes zones. Uses only the two location words, never the code."""

        if previous:
            from_controller, from_location, from_sequence, from_position = (
                _unpack_location(previous))
            if from_location == C.LOCATION_REMOVED:
                self._depart(from_controller, from_sequence)
        else:
            from_location, from_position = 0, 0
        if not current:
            return
        # The destination position is not recorded: whether the slot is now face-up or face-down is decided by the snapshot; only the source is recorded here
        to_controller, to_location, to_sequence, _to_position = (
            _unpack_location(current))
        if to_location != C.LOCATION_REMOVED:
            return
        from_extra_facedown = bool(
            from_location == C.LOCATION_EXTRA
            and (from_position & C.POS_FACEDOWN))
        self._arrive(to_controller, to_sequence, from_extra_facedown)

    def _depart(self, controller: int, sequence: int) -> None:
        if controller not in (0, 1):
            return
        slots = self._slots[controller]
        slots.pop(int(sequence), None)
        shifted = {(slot - 1 if slot > int(sequence) else slot): flag
                   for slot, flag in slots.items()}
        slots.clear()
        slots.update(shifted)

    def _arrive(self, controller: int, sequence: int, from_extra: bool) -> None:
        if controller not in (0, 1):
            return
        shifted = {(slot + 1 if slot >= int(sequence) else slot): flag
                   for slot, flag in self._slots[controller].items()}
        shifted[int(sequence)] = bool(from_extra)
        self._slots[controller].clear()
        self._slots[controller].update(shifted)

    # -- queries --------------------------------------------------------------

    def extra_origin_slots(self, controller: int) -> frozenset[tuple[int, int]]:
        """The face-down banished slots of ``controller`` that were banished face-down from the Extra Deck.

        Returns a set of ``(LOCATION_REMOVED, sequence)``, the same shape as the elements of
        :attr:`~mirrorforce.search.belief.Evidence.facedown_slot_keys`.
        """

        if controller not in (0, 1):
            raise ValueError(f"controller must be seat 0 or 1, got {controller!r}")
        return frozenset(
            (C.LOCATION_REMOVED, int(sequence))
            for sequence, flag in self._slots[controller].items() if flag)

    def tracked_slots(self, controller: int) -> int:
        """The number of records of ``controller``'s current face-down banished slots, for diagnostics only."""

        return len(self._slots[int(controller)])

    def clear(self) -> None:
        for slots in self._slots:
            slots.clear()


def extra_origin_slots_from_messages(messages, controller: int
                                     ) -> frozenset[tuple[int, int]]:
    """Compute ``controller``'s extra-source slots directly from a ``(msg, body)`` message stream."""

    tracker = BanishOriginTracker()
    for msg, body in messages:
        tracker.observe(int(msg), bytes(body))
    return tracker.extra_origin_slots(controller)
