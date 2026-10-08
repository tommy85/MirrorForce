"""Trim the particle-side board to the shape of the online path (parity alignment).

Why trim the rich side instead of enriching the online side
------------------------------

At deployment the policy reads **the online copy**. If the policy stream in search read a richer copy, ExIt distillation would
distil an information gap that does not exist at deployment into the policy. So both sides must have the same shape bit for bit, and alignment
can only bring particles down to the online side, never the reverse.

**The operational definition of parity is "what the online path produces"**:
the output shape of ``snapshot_from_shadow(ShadowBoard)`` is the target. A differential gate
measured two surpluses:

1. **Xyz materials**. The particle-side Xyz monster carries ``overlay=(material code,)``; the online side is empty.
   The cause is not masking: the online refresh flags **never include** ``QUERY_OVERLAY_CARD``
   (``netduel/board.py:62`` records the refresh set of ``single_duel.h:37-42``),
   and a real client's shadow board never queries which cards the materials are.
2. **Face-up cards in the opponent's Extra Deck zone** (pendulum cards returned face-up to the Extra Deck). Their identity **really is
   public**, and ``mask_for`` is right to keep it; it is the online path that drops it: the shadow board
   never received that code.

Both are "publicly derivable, but missing from the online explicit columns". The model has not really lost them: activations and moves
are public history, and the history tokens of the sequence still carry them. So trimming does not make the model dumber, it only makes
search time and deployment time read the same explicit columns.

Whether to add these two columns to the online side (making both sides richer) is a decision for the next round of
input changes, not for this line. If they are added one day,
**both sides must be added together**, and this module becomes obsolete.

Usage
----

Apply on the particle path, **before** ``mask_for``::

    full = state.capture(driver)
    full = parity_filter(full, viewer)
    snap = public_view(full, viewer, driver.disclosure)

Before ``mask_for`` rather than after, so that once "that Extra Deck card becomes unknown",
``mask_for`` itself rearranges the slots of count-only zones by its canonicalization rule, which is exactly what the online
path does. Rearranging after masking would write the canonicalization rule a second time.
"""

from __future__ import annotations

from dataclasses import replace

from ..netduel import constants as C

__all__ = ["parity_filter"]


def parity_filter(snapshot, player: int):
    """Trim ``snapshot`` to what the online path can represent.

    Idempotent: applying it again to a snapshot the online path produced gives the input back (no Xyz materials to clear,
    the opponent's Extra Deck zone is already anonymous). A test pins this as the evidence
    of "not trimming too much": an implementation that trims too much keeps changing things on the second application.
    """
    cards = []
    for card in snapshot.cards:
        opponent = card.controller != player
        # 1) Xyz materials: the online refresh flags do not query them, so neither side carries them
        if card.overlay:
            card = replace(card, overlay=())
        # 2) The opponent's Extra Deck zone: the online side never receives codes, so treat them all as unknown.
        #    Erase the identities right here and leave the canonical rearrangement of count-only zones to mask_for.
        if opponent and card.location == C.LOCATION_EXTRA and card.code:
            card = replace(card, code=0, position=0, type=0, level=0, rank=0,
                           attribute=0, race=0, attack=0, defense=0, status=0,
                           lscale=0, rscale=0, link=0, link_marker=0,
                           counters=(), hidden=True)
        cards.append(card)
    return replace(snapshot, cards=tuple(cards))
