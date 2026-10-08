"""Public-latent belief potentials for hands, decks and hidden field slots.

The model adds energy terms to COMPLETE public-constrained layout proposals;
it does not independently sample each marginal. Therefore a card cannot be
created twice by separate hand/field predictions. The existing evidence
sampler owns hard constraints and card conservation. Labels and reference
engine state are deliberately absent from this deployment module.

The old hand-only head/artifact stays readable through stage_a_belief. This
new schema is explicit and cannot silently replace that head or a policy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import io
import math
import re

import torch
from torch import Tensor, nn

from ..netduel import constants as C
from ..search.particles import HiddenLayout
from .deferred_checks import DeferredChecks, duplicated_keys
from .sidecar_io import digest, read_regular
from .world_model_contract import WorldLatent

INFORMATION_SET_SEARCH = True
CONFIG_SCHEMA = "mirrorforce_stage_a_joint_zone_belief/v1"
HEAD_SCHEMA = "mirrorforce_stage_a_joint_zone_belief_head/v1"
ZONES = ("hand", "deck", "field_facedown", "extra")
COUNT_CLASSES = 4


@dataclass(frozen=True)
class JointBeliefConfig:
    d_model: int = 768
    card_dim: int = 256
    heads: int = 12
    hidden_dim: int = 768
    schema: str = CONFIG_SCHEMA

    def __post_init__(self):
        if self.schema != CONFIG_SCHEMA or any(type(getattr(self, k)) is not int or getattr(self, k) <= 0
                for k in ("d_model", "card_dim", "heads", "hidden_dim")) or self.d_model % self.heads:
            raise ValueError("invalid joint belief configuration")


@dataclass(frozen=True)
class JointRecipe:
    """A publicly selected recipe hypothesis, never the server's hidden list."""
    codes: tuple[int, ...]
    copies: tuple[int, ...]
    extra: tuple[bool, ...]

    def __post_init__(self):
        if (not self.codes or self.codes != tuple(sorted(set(self.codes)))
                or len(self.codes) != len(self.copies) or len(self.codes) != len(self.extra)
                or any(type(c) is not int or c <= 0 for c in self.codes)
                or any(type(n) is not int or not 1 <= n <= 3 for n in self.copies)
                or any(type(b) is not bool for b in self.extra)):
            raise ValueError("joint recipe needs ascending distinct codes, copy bounds and public home zones")

    @classmethod
    def from_decks(cls, main, extra=()):
        main, extra = Counter(main), Counter(extra)
        if set(main) & set(extra):
            raise ValueError("one printed identity cannot have both main and extra home zones")
        combined = main + extra
        codes = tuple(sorted(combined))
        return cls(codes, tuple(combined[c] for c in codes), tuple(c in extra for c in codes))

    def counts(self, codes) -> list[int]:
        counts = Counter(codes)
        if set(counts) - set(self.codes) or any(counts[c] > n for c, n in zip(self.codes, self.copies)):
            raise ValueError("layout identity or count is outside the proposed recipe")
        return [counts[c] for c in self.codes]


@dataclass(frozen=True)
class PublicFieldSlots:
    """Observable field coordinates and known set age; no identity/target input."""
    locations: Tensor       # [batch, slots], MZONE or SZONE
    sequences: Tensor       # [batch, slots], 0..7
    age_turns: Tensor       # [batch, slots], -1 unknown or nonnegative age
    valid: Tensor          # [batch, slots], excludes padding

    def validate(self, batch: int, device):
        checks = DeferredChecks()
        self.add_checks(checks, batch, device)
        checks.raise_first()

    def add_checks(self, checks: DeferredChecks, batch: int, device) -> bool:
        """Append this layout's ordered checks; False means raise now.

        Conditions, messages and order match the former per-condition and
        per-row validation. Valid MZONE/SZONE coordinates with sequence 0..7
        map injectively to ``location * 8 + sequence``, so duplicate
        detection is one sort per row instead of a Python loop.
        """
        values = (self.locations, self.sequences, self.age_turns, self.valid)
        shape = self.locations.shape
        if not checks.host(len(shape) != 2 or shape[0] != batch or shape[1] > 16
                           or any(t.shape != shape or t.device != device for t in values)
                           or self.valid.dtype != torch.bool
                           or any(t.dtype != torch.long for t in values[:3]), "public field slot layout differs"):
            return False
        coordinates = "invalid public field coordinates or age"
        checks.device(self.valid & ~((self.locations == C.LOCATION_MZONE) | (self.locations == C.LOCATION_SZONE)),
                      coordinates)
        checks.device(self.valid & ((self.sequences < 0) | (self.sequences > 7)), coordinates)
        checks.device(self.valid & ((self.age_turns < -1) | (self.age_turns > 65535)), coordinates)
        for t in values[:3]:
            checks.device(~self.valid & (t != 0), "padded public field slots must be zero")
        checks.device(duplicated_keys(self.locations * 8 + self.sequences, self.valid), "duplicate public field slot")
        return True


@dataclass(frozen=True)
class JointBeliefOutput:
    count_adjustments: Tensor  # [batch, card, zone, 4]; NOT independent samples
    slot_adjustments: Tensor   # [batch, field slot, card]


class OpponentJointBelief(nn.Module):
    """Shared card/slot attention over the existing public latent.

    All final readouts start at zero, preserving the evidence proposal law.
    Copy counts, home zones and visible zone sizes describe public metadata
    and a selected hypothetical recipe, not true hidden card locations.
    """
    def __init__(self, config: JointBeliefConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.state_norm = nn.LayerNorm(d)
        self.card_query = nn.Sequential(nn.Linear(config.card_dim, d), nn.GELU(), nn.LayerNorm(d))
        self.copies = nn.Embedding(COUNT_CLASSES, d)
        self.home = nn.Embedding(2, d)
        self.zone_sizes = nn.Linear(len(ZONES), d)
        self.location = nn.Embedding(2, d)
        self.sequence = nn.Embedding(8, d)
        self.age = nn.Linear(2, d)
        self.attention = nn.MultiheadAttention(d, config.heads, batch_first=True)
        self.count = nn.Sequential(nn.Linear(2*d, config.hidden_dim), nn.GELU(), nn.LayerNorm(config.hidden_dim),
                                   nn.Linear(config.hidden_dim, len(ZONES)*COUNT_CLASSES))
        self.slot_query = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, d))
        self.slot_key = nn.Linear(d, d)
        for layer in (self.count[-1], self.slot_query[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, latent: WorldLatent, cards: Tensor, copies: Tensor, extra: Tensor,
                zone_sizes: Tensor, slots: PublicFieldSlots) -> JointBeliefOutput:
        latent.validate()
        batch, _, d = latent.tokens.shape
        device = latent.tokens.device
        # Same conditions/messages/order as before; device flags share one sync.
        checks, inputs = DeferredChecks(), "joint belief public inputs differ from configuration"
        if not checks.host(d != self.config.d_model or cards.ndim != 2 or cards.shape[1] != self.config.card_dim
                           or not cards.shape[0] or not cards.is_floating_point()
                           or copies.shape != cards.shape[:1] or copies.dtype != torch.long
                           or extra.shape != copies.shape or extra.dtype != torch.bool
                           or zone_sizes.shape != (batch, len(ZONES)) or zone_sizes.dtype != torch.long
                           or any(t.device != device for t in (cards, copies, extra, zone_sizes)), inputs):
            checks.raise_first()
        checks.device(~torch.isfinite(cards), inputs)
        checks.device((copies < 1) | (copies > 3), inputs)
        checks.device((zone_sizes < 0) | (zone_sizes > 255), inputs)
        if slots.add_checks(checks, batch, device):
            checks.device(slots.valid.sum(-1) != zone_sizes[:, 2],
                          "visible facedown size differs from the field slot layout")
        checks.raise_first()
        state = self.state_norm(latent.tokens.masked_fill(latent.padding_mask.unsqueeze(-1), 0))
        # Copies and slot sequences are checked above; the clamps only keep a deferred refusal in range.
        card_query = self.card_query(cards.to(state.dtype)) + self.copies(copies.clamp(0, COUNT_CLASSES - 1)) \
            + self.home(extra.long())
        card_query = card_query.unsqueeze(0).expand(batch, -1, -1)
        sizes = self.zone_sizes(zone_sizes.to(state.dtype) / 60.).unsqueeze(1)
        ages = torch.stack(((slots.age_turns.clamp_min(0).float() / 32.).to(state.dtype),
                            (slots.age_turns >= 0).to(state.dtype)), -1)
        slot_query = (self.location((slots.locations == C.LOCATION_SZONE).long())
                      + self.sequence(slots.sequences.clamp(0, 7)) + self.age(ages))
        query = torch.cat((card_query, slot_query), 1) + sizes
        attended, _ = self.attention(query, state, state, key_padding_mask=latent.padding_mask, need_weights=False)
        combined = torch.cat((query, attended), -1)
        nc = cards.shape[0]
        counts = self.count(combined[:, :nc]).reshape(batch, nc, len(ZONES), COUNT_CLASSES)
        impossible = torch.arange(COUNT_CLASSES, device=device) > copies[:, None]
        counts = counts.masked_fill(impossible[None, :, None, :], -1e4)
        fields = torch.einsum("bsd,bcd->bsc", self.slot_query(combined[:, nc:]), self.slot_key(card_query)) / math.sqrt(d)
        fields = fields.masked_fill(~slots.valid.unsqueeze(-1), 0)
        return JointBeliefOutput(counts, fields)


def supported_log_probabilities(adjustments: Tensor, prior: Tensor) -> Tensor:
    """Tilt a diagnostic marginal without ever adding mass to a hard zero."""
    if adjustments.shape != prior.shape or not prior.is_floating_point() \
            or not bool(torch.isfinite(prior).all()) or bool((prior < 0).any()) \
            or not bool(torch.isfinite(adjustments).all()) or bool((prior.sum(-1) <= 0).any()):
        raise ValueError("belief marginal needs finite adjustments and nonempty nonnegative support")
    log_prior = prior.float().masked_fill(prior <= 0, 1).log().masked_fill(prior <= 0, -torch.inf)
    return torch.log_softmax(adjustments.float() + log_prior, -1)


def layout_counts(layout: HiddenLayout, recipe: JointRecipe) -> list[list[int]]:
    """Count all four predicted zones; banished cards still consume copies."""
    if any(recipe.extra) and layout.extra is None:
        raise ValueError("a complete joint layout must explicitly assign remaining extra cards")
    if any(l not in (C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_REMOVED) or type(s) is not int or s < 0
           for l, s, _ in layout.facedown):
        raise ValueError("invalid hidden layout coordinates")
    groups = (layout.hand, layout.deck,
              tuple(c for l, _, c in layout.facedown if l in (C.LOCATION_MZONE, C.LOCATION_SZONE)),
              tuple(layout.extra or ()))
    recipe.counts(list(layout.hand) + list(layout.deck) + [c for _, _, c in layout.facedown] + list(layout.extra or ()))
    rows = [recipe.counts(group) for group in groups]
    return [[rows[z][card] for z in range(len(ZONES))] for card in range(len(recipe.codes))]


def layout_log_weights(output: JointBeliefOutput, layouts, recipe: JointRecipe,
                       slot_keys: tuple[tuple[int, int], ...], *, power: float = 1.) -> Tensor:
    """Energy of already public-constrained COMPLETE proposals, not new draws.

    Using these weights with a finite proposal bank approximates the tilted
    joint law. It is not an exact sampler or a normalized joint likelihood.
    Zero readouts/power preserve all evidence-proposal weights equally.
    """
    if type(power) not in (int, float) or not math.isfinite(power) or not 0 <= power <= 1 or not layouts:
        raise ValueError("joint weighting needs proposals and a finite power in [0,1]")
    count, slots = output.count_adjustments, output.slot_adjustments
    if count.shape != (1, len(recipe.codes), len(ZONES), COUNT_CLASSES) \
            or slots.shape != (1, len(slot_keys), len(recipe.codes)) \
            or not bool(torch.isfinite(count).all()) or not bool(torch.isfinite(slots).all()) \
            or len(set(slot_keys)) != len(slot_keys):
        raise ValueError("joint potential shape, values or slot keys differ")
    indices = {code: i for i, code in enumerate(recipe.codes)}
    values = []
    for layout in layouts:
        fields = {(l, s): c for l, s, c in layout.facedown if l in (C.LOCATION_MZONE, C.LOCATION_SZONE)}
        if len(fields) != sum(l in (C.LOCATION_MZONE, C.LOCATION_SZONE) for l, _, _ in layout.facedown) \
                or set(fields) != set(slot_keys):
            raise ValueError("layout field identities do not cover exactly the public field slots")
        counts = torch.tensor(layout_counts(layout, recipe), dtype=torch.long, device=count.device)
        score = count[0].gather(-1, counts.unsqueeze(-1)).sum()
        for slot, key in enumerate(slot_keys):
            score = score + slots[0, slot, indices[fields[key]]]
        values.append(score * power)
    return torch.stack(values)


def head_payload(head: OpponentJointBelief, recipe: JointRecipe, fitted: dict) -> dict:
    """Separate sidecar bound to its frozen backbone; never overwrite it."""
    if not isinstance(fitted, dict) or any(not re.fullmatch(r"[0-9a-f]{64}", str(fitted.get(k, "")))
            for k in ("checkpoint_sha256", "training_provenance_sha256", "artifact_fingerprint")):
        raise ValueError("joint head requires backbone/provenance/semantic artifact bindings")
    return {"schema": HEAD_SCHEMA, "config": asdict(head.config), "recipe": asdict(recipe),
            "fitted": fitted, "weights": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}


def load_head(path, checksum, *, checkpoint_sha256: str, artifact_fingerprint: str):
    raw = read_regular(path, "joint belief head")
    if digest(raw) != checksum:
        raise ValueError("joint belief checksum differs")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"schema", "config", "recipe", "fitted", "weights"} \
            or payload["schema"] != HEAD_SCHEMA:
        raise ValueError("unsupported joint belief head artifact")
    fitted = payload["fitted"]
    if fitted.get("checkpoint_sha256") != checkpoint_sha256 or fitted.get("artifact_fingerprint") != artifact_fingerprint:
        raise ValueError("joint belief backbone or semantic artifact differs")
    recipe = JointRecipe(**{k: tuple(v) for k, v in payload["recipe"].items()})
    head = OpponentJointBelief(JointBeliefConfig(**payload["config"]))
    head.load_state_dict(payload["weights"], strict=True)
    if any(not bool(torch.isfinite(t).all()) for t in head.state_dict().values()):
        raise ValueError("non-finite joint belief weights")
    return head.eval(), recipe, fitted
