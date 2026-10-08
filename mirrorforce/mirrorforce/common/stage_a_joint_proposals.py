"""The public proposal law shared by joint-belief fitting and live search.

No native engine, hidden layout reader or target-label module is used here.
Caller-selected recipes are public hypotheses, not actual opponent deck IDs.
The legacy shape-only sampler receives zero identities for unknown slots.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import random

from ..netduel import constants as C
from ..search.belief import evidence_from_snapshot
from ..search.particles import HiddenLayout, particles_from_evidence
from ..worldmodel.state import StateSnapshot, leaks
from .stage_a_joint_belief import JointRecipe

INFORMATION_SET_SEARCH = True
PROPOSAL_LAW = 'existing_public_constraint_sampler/v1'


class JointProposalError(ValueError):
    pass


def evidence_for_public_snapshot(public: StateSnapshot, viewer: int, recipe: JointRecipe, *,
                                 extra_origin_slots=(), categories=()):
    """Derive inventory from one already published view and an explicit recipe."""
    if type(viewer) is not int or viewer not in (0, 1) or not isinstance(public, StateSnapshot) \
            or public.view != viewer or leaks(public):
        raise JointProposalError('joint belief requires a leak-audited published observer snapshot')
    player = 1-viewer
    main = Counter({c: n for c, n, extra in zip(recipe.codes, recipe.copies, recipe.extra) if not extra})
    extra = Counter({c: n for c, n, is_extra in zip(recipe.codes, recipe.copies, recipe.extra) if is_extra})
    public_extra = tuple(int(card.code) for card in public.cards if card.controller == player
                         and card.location == C.LOCATION_EXTRA and card.code and main.get(int(card.code), 0))
    return evidence_from_snapshot(public, player=player, deck_list=main,
        revealed=frozenset(tuple(row) for row in public.revealed),
        extra_codes=frozenset(extra), public_extra=public_extra, extra_list=extra,
        extra_origin_slots=frozenset(tuple(row) for row in extra_origin_slots), categories=tuple(categories))


def proposal_bank(public, evidence, recipe: JointRecipe, *, player: int, count: int, seed: int, sampler=None):
    """Same seeded complete-layout draws as the fitted target pipeline.

    Only ``public.revealed`` is read. Duplicate draws are retained with their
    original multiplicity; deduplication would change the proposal law.
    ``sampler`` draws the hidden cards in place of the evidence's own belief
    sampler (``particles_from_evidence``).
    """
    if type(count) is not int or not 1 <= count <= 1024 or player not in (0, 1):
        raise ValueError('joint belief needs a bounded proposal bank and one observer opponent')
    evidence.check()
    shape = HiddenLayout(hand=(0,)*evidence.hand_size, deck=(0,)*evidence.deck_size,
        facedown=tuple((int(l), int(s), 0) for l, s in evidence.facedown_slot_keys),
        extra=(0,)*evidence.extra_facedown_size)
    layouts = [layout for _, layout, _ in particles_from_evidence(evidence, count, seed=seed, truth=shape,
        revealed=public.revealed, player=player, shape_only=True, sampler=sampler,
        extra_codes=frozenset(c for c, extra in zip(recipe.codes, recipe.extra) if extra))]
    # Keep the original seed domain as part of the fitted q-law. Its historical
    # name does NOT make this stream depend on a label or a real shuffle seed.
    rng = random.Random(('joint-label-extra', seed).__repr__())
    complete = []
    for layout in layouts:
        if layout.extra is None:
            if evidence.extra_in_slots or evidence.extra_slot_keys:
                raise JointProposalError('legacy proposal omitted a split extra inventory')
            cards = list((evidence.unknown_extra_pool() + evidence.disclosed_extra_deck).elements())
            if len(cards) != evidence.extra_facedown_size:
                raise JointProposalError('public extra inventory does not fill the declared zone')
            rng.shuffle(cards)
            layout = replace(layout, extra=tuple(cards))
        complete.append(layout)
    return complete
