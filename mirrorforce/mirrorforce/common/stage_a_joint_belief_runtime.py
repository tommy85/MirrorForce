"""Frozen joint belief over the SAME public proposal law used for fitting.

This probability adapter owns no engine or policy memory and grants no search
admission. The client still binds a bank to its public root/epoch and validates
each native realization. Failed realizations retain their original mass.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..netduel import constants as C
from ..search.particles import HiddenLayout
from .sidecar_io import require_sha256
from .stage_a_belief import recipe_card_features
from .stage_a_joint_belief import (JointRecipe, PublicFieldSlots, layout_counts,
                                    layout_log_weights, load_head)
from .stage_a_joint_proposals import PROPOSAL_LAW, evidence_for_public_snapshot, proposal_bank
from .world_model_contract import WorldLatent

INFORMATION_SET_SEARCH = True
SCHEMA = 'mirrorforce_public_joint_particle_bank/v1'


@dataclass(frozen=True)
class ZoneClaim:
    """A public fact the ledger does not keep: the opponent's ``zones`` (hand, deck, extra deck) together hold at
    least one of ``codes`` (an activation was legal only so; the client's follower derives it)."""
    zones: tuple[int, ...]
    codes: frozenset[int]

    def honored_by(self, layout) -> bool:
        cards = {C.LOCATION_HAND: layout.hand, C.LOCATION_DECK: layout.deck, C.LOCATION_EXTRA: layout.extra or ()}
        return any(code in self.codes for zone in self.zones for code in cards[zone])


#: proposals drawn per kept one when zone claims filter the draws
_CLAIM_POOL = 16


@dataclass(frozen=True)
class JointDraw:
    """Immutable assignments; exposing a layout returns a fresh mutable wrapper."""
    hand: tuple[int, ...]
    deck: tuple[int, ...]
    facedown: tuple[tuple[int, int, int], ...]
    extra: tuple[int, ...]

    @classmethod
    def from_layout(cls, layout):
        if layout.extra is None:
            raise ValueError('joint draw must explicitly assign its remaining extra deck')
        return cls(tuple(layout.hand), tuple(layout.deck), tuple(tuple(row) for row in layout.facedown), tuple(layout.extra))

    def layout(self):
        return HiddenLayout(hand=self.hand, deck=self.deck, facedown=self.facedown, extra=self.extra)


@dataclass(frozen=True)
class JointParticleBank:
    viewer: int
    recipe: JointRecipe
    draws: tuple[JointDraw, ...]
    probabilities: tuple[float, ...]
    log_probabilities: tuple[float, ...]
    field_keys: tuple[tuple[int, int], ...]
    zone_sizes: tuple[int, ...]
    seed: int
    power: float
    head_sha256: str | None
    proposal_law: str = PROPOSAL_LAW
    schema: str = SCHEMA

    def __post_init__(self):
        if not self.draws or len(self.draws) != len(self.probabilities) or len(self.draws) != len(self.log_probabilities) \
                or any(not math.isfinite(p) or p < 0 for p in self.probabilities) \
                or any(not math.isfinite(p) for p in self.log_probabilities) \
                or not math.isclose(math.fsum(self.probabilities), 1., rel_tol=0, abs_tol=1e-12):
            raise ValueError('joint bank needs finite normalized weights for every proposal draw')

    @property
    def effective_sample_size(self):
        return 1. / math.fsum(p*p for p in self.probabilities)

    def covered_mass(self, realized_indices) -> float:
        """No renormalization after failures, truncation, or native rejection."""
        indices = tuple(realized_indices)
        if len(indices) != len(set(indices)) or any(type(i) is not int or not 0 <= i < len(self.draws) for i in indices):
            raise ValueError('coverage needs distinct indices in the original proposal bank')
        return math.fsum(self.probabilities[i] for i in indices)


def _public_draws(public, recipe, *, viewer, count, seed, extra_origin_slots, categories):
    """The seeded public proposals. Zone claims among ``categories`` keep only the proposals that honor them, in draw
    order, from a pool of ``_CLAIM_POOL`` times the count (without any, the draws are the fitted law's own)."""
    if type(seed) is not int:
        raise ValueError('proposal randomness must be an explicit independent integer seed')
    claims = tuple(item for item in categories if isinstance(item, ZoneClaim))
    evidence = evidence_for_public_snapshot(public, viewer, recipe, extra_origin_slots=extra_origin_slots,
        categories=tuple(item for item in categories if not isinstance(item, ZoneClaim)))
    layouts = proposal_bank(public, evidence, recipe, player=1-viewer, count=count * _CLAIM_POOL if claims else count,
                            seed=seed)
    if claims:
        layouts = [layout for layout in layouts if all(claim.honored_by(layout) for claim in claims)][:count]
        if len(layouts) < count:
            raise ValueError('public zone claims left fewer proposals than the bank needs')
    keys = tuple(sorted((l, s) for l, s in evidence.facedown_slot_keys if l in (C.LOCATION_MZONE, C.LOCATION_SZONE)))
    sizes = (evidence.hand_size, evidence.deck_size, len(keys), evidence.extra_facedown_size)
    for layout in layouts:
        counts = layout_counts(layout, recipe)
        if tuple(sum(row[z] for row in counts) for z in range(4)) != sizes:
            raise ValueError('complete proposal sizes differ from public evidence')
    return layouts, keys, sizes


def _bank(layouts, keys, sizes, recipe, *, viewer, seed, logits, power, head_sha256):
    if logits.ndim != 1 or len(logits) != len(layouts) or not bool(torch.isfinite(logits).all()):
        raise ValueError('joint proposal scores must be finite and cover every original draw')
    logp = logits.double().log_softmax(-1)
    return JointParticleBank(viewer, recipe, tuple(JointDraw.from_layout(x) for x in layouts),
        tuple(logp.exp().cpu().tolist()), tuple(logp.cpu().tolist()), keys, sizes, seed, power, head_sha256)


def uniform_joint_bank(public, recipe: JointRecipe, *, viewer: int, count: int, seed: int,
                       extra_origin_slots=(), categories=()) -> JointParticleBank:
    """Public constrained baseline; no learned model or hidden layout needed."""
    layouts, keys, sizes = _public_draws(public, recipe, viewer=viewer, count=count, seed=seed,
        extra_origin_slots=extra_origin_slots, categories=categories)
    return _bank(layouts, keys, sizes, recipe, viewer=viewer, seed=seed,
                 logits=torch.zeros(len(layouts), dtype=torch.float64), power=0., head_sha256=None)


def belief_joint_bank(public, recipe: JointRecipe, belief, latent: WorldLatent | None, *, viewer: int, count: int,
                      seed: int, power: float, extra_origin_slots=(), categories=()) -> JointParticleBank:
    """``uniform_joint_bank``'s draws weighted by a loaded model's own opponent-hand belief.

    The draws are exactly the uniform bank's (``_public_draws``: the same public
    proposals, proposal law and ``categories``, among them the face-down field
    claims, passed through unchanged), so ``ClientParticleAdapter`` accepts the
    bank. Each draw's weight is ``stage_a_belief.belief_log_weights`` of its
    opponent hand, read from ``belief`` (a ``ParticleBelief``, e.g.
    ``stage_a_belief.own_head_belief``) at ``latent``, the public latent of the
    actor's own forward at the root: the weighting the rollout teacher's
    particles take. Without a belief, or at power 0, it is the uniform bank,
    byte for byte. Sampling happens before the head is read and never sees it.
    """
    if belief is None or power == 0:
        return uniform_joint_bank(public, recipe, viewer=viewer, count=count, seed=seed,
                                  extra_origin_slots=extra_origin_slots, categories=categories)
    main = tuple((code, copies) for code, copies, extra in zip(recipe.codes, recipe.copies, recipe.extra) if not extra)
    if tuple(zip(belief.recipe.codes, belief.recipe.copies)) != main:
        raise ValueError("the belief reads another opponent main deck than the bank's recipe")
    if latent is None or latent.tokens.shape[0] != 1 or int(latent.observer[0]) != viewer:
        raise ValueError("a belief bank reads the viewer's own public latent at one root")
    layouts, keys, sizes = _public_draws(public, recipe, viewer=viewer, count=count, seed=seed,
                                         extra_origin_slots=extra_origin_slots, categories=categories)
    logits = torch.tensor(belief.log_weights(latent, [layout.hand for layout in layouts], power=power),
                          dtype=torch.float64)
    return _bank(layouts, keys, sizes, recipe, viewer=viewer, seed=seed, logits=logits, power=float(power),
                 head_sha256=belief.sha256)


class JointParticleBelief:
    """Head-only root readout. A load receipt must name the actual frozen parent."""

    def __init__(self, head, recipe, cards, *, head_sha256: str, fitted: dict, power: float):
        require_sha256(head_sha256, 'joint belief head')
        if type(power) not in (int, float) or not math.isfinite(power) or not 0 <= power <= 1:
            raise ValueError('joint weighting power must be explicitly selected in [0,1]')
        if fitted.get('loss') != 'whole_layout_ranking_contrastive/v1' \
                or fitted.get('field_age_law') != 'unknown/v1' \
                or fitted.get('proposal_law', PROPOSAL_LAW) != PROPOSAL_LAW:
            raise ValueError('joint inference law differs from the fitted head contract')
        if cards.shape != (len(recipe.codes), head.config.card_dim) or not bool(torch.isfinite(cards).all()):
            raise ValueError('joint semantic features differ from head and recipe')
        self.head = head.to(cards.device).float().eval().requires_grad_(False)
        self.recipe, self.head_sha256, self.power = recipe, head_sha256, float(power)
        self.cards = cards.detach().float().clone()
        self.copies = torch.tensor(recipe.copies, dtype=torch.long, device=cards.device)
        self.extra = torch.tensor(recipe.extra, dtype=torch.bool, device=cards.device)

    @torch.inference_mode()
    def propose(self, public, latent: WorldLatent, *, count: int, seed: int,
                extra_origin_slots=(), categories=()) -> JointParticleBank:
        latent.validate()
        if latent.tokens.shape[0] != 1 or latent.tokens.device != self.cards.device:
            raise ValueError('joint belief reads exactly one public root on its own device')
        viewer = int(latent.observer[0])
        # Sampling happens before scoring and has no access to latent/model
        # outputs. All draws, including duplicates, remain in the finite bank.
        layouts, keys, sizes = _public_draws(public, self.recipe, viewer=viewer, count=count, seed=seed,
            extra_origin_slots=extra_origin_slots, categories=categories)
        if self.power == 0:
            logits = self.cards.new_zeros(len(layouts))
        else:
            positions = torch.tensor(keys, dtype=torch.long, device=self.cards.device).reshape(1, len(keys), 2)
            slots = PublicFieldSlots(positions[..., 0], positions[..., 1],
                torch.full((1, len(keys)), -1, dtype=torch.long, device=self.cards.device),
                torch.ones(1, len(keys), dtype=torch.bool, device=self.cards.device))
            state = WorldLatent(latent.tokens.float(), latent.padding_mask, latent.observer)
            with torch.autocast(device_type=self.cards.device.type, enabled=False):
                output = self.head(state, self.cards, self.copies, self.extra,
                    torch.tensor([sizes], dtype=torch.long, device=self.cards.device), slots)
                logits = layout_log_weights(output, layouts, self.recipe, keys, power=self.power)
        return _bank(layouts, keys, sizes, self.recipe, viewer=viewer, seed=seed,
                     logits=logits, power=self.power, head_sha256=self.head_sha256)


def joint_particle_belief(path, checksum, *, main, extra, card_effect_encoder, store, device,
                         evaluation_load, power: float) -> JointParticleBelief:
    """Bind head, declared public recipe, semantic table and parent load receipt.

    This does not verify native history or promote the parent's search flags.
    V1 artifacts with no explicit proposal_law retain their original fixed law;
    unknown new loss/age/proposal profiles are rejected instead of guessed.
    """
    fingerprint = store.artifact_fingerprint
    if card_effect_encoder.artifact_fingerprint != fingerprint:
        raise ValueError('belief encoder and semantic store differ')
    head, recipe, fitted = load_head(path, checksum,
        checkpoint_sha256=evaluation_load.checkpoint_sha256, artifact_fingerprint=fingerprint)
    if fitted['training_provenance_sha256'] != evaluation_load.training_provenance_sha256 \
            or recipe != JointRecipe.from_decks(main, extra):
        raise ValueError('joint belief parent provenance or explicit recipe differs')
    with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        cards = recipe_card_features(card_effect_encoder, store, recipe, device)
    return JointParticleBelief(head, recipe, cards, head_sha256=checksum, fitted=fitted, power=power)
