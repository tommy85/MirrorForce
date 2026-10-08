"""Data contracts between the state trunk and the root readout.

A latent is the observer's encoded information state; candidates are encoded
actions of the current real menu; a prediction scores them. Latent tokens need
not correspond one-to-one to physical cards. WDL values use the observer's
perspective. The learned transition model these types once also served was
removed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .deferred_checks import refuse


def _finite_float(value: Tensor, name: str) -> None:
    message = f"{name} must contain finite floating-point values"
    if not value.is_floating_point():
        raise ValueError(message)
    refuse(~torch.isfinite(value), message)


@dataclass(frozen=True)
class WorldLatent:
    tokens: Tensor                 # [batch, latent slots, width]
    padding_mask: Tensor           # [batch, latent slots], True = ignored
    observer: Tensor               # [batch], fixed absolute seat 0 or 1

    def validate(self) -> None:
        if self.tokens.ndim != 3 or min(self.tokens.shape) <= 0:
            raise ValueError("world latent must be nonempty [batch, slots, width]")
        _finite_float(self.tokens, "world latent")
        if self.padding_mask.shape != self.tokens.shape[:2] \
                or self.padding_mask.dtype != torch.bool:
            raise ValueError("world latent padding shape or dtype differs")
        refuse(self.padding_mask.all(dim=1), "each world latent needs a readable slot")
        if self.observer.shape != self.tokens.shape[:1] \
                or self.observer.dtype not in (torch.int32, torch.int64):
            raise ValueError("world observer must be a batch of integer seats")
        refuse((self.observer < 0) | (self.observer > 1), "world observer must be seat 0 or 1")
        if len({self.tokens.device, self.padding_mask.device, self.observer.device}) != 1:
            raise ValueError("world latent tensors must share a device")

    def detached(self) -> "WorldLatent":
        """Build a stop-gradient target; callers still own target provenance."""
        return WorldLatent(self.tokens.detach(), self.padding_mask.detach(),
                           self.observer.detach())


@dataclass(frozen=True)
class WorldCandidates:
    """Encoded structural candidates, independently generated from observations.

    This mask describes padding, never ground-truth future legality. Real-root
    action selection separately applies the server's authoritative legal mask.
    Future planning must obtain candidates and legality without a simulator.
    A zero-length candidate axis requests value-only prediction; it is not
    evidence that the latent represents a terminal state.
    """

    features: Tensor               # [batch, candidates, action width]
    padding_mask: Tensor           # [batch, candidates]

    def validate(self, latent: WorldLatent) -> None:
        if self.features.ndim != 3 or self.features.shape[0] <= 0 \
                or self.features.shape[2] <= 0 \
                or self.features.shape[0] != latent.tokens.shape[0]:
            raise ValueError("world candidate batch or feature width differs")
        _finite_float(self.features, "world candidates")
        if self.padding_mask.dtype != torch.bool \
                or self.padding_mask.shape != self.features.shape[:2]:
            raise ValueError("world candidate padding shape or dtype differs")
        if self.features.shape[1] and bool(self.padding_mask.all(dim=1).any()):
            raise ValueError("candidate rows cannot all be padding")
        if self.features.device != latent.tokens.device \
                or self.padding_mask.device != latent.tokens.device:
            raise ValueError("world candidates must share the latent device")


@dataclass(frozen=True)
class WorldPrediction:
    policy_logits: Tensor          # [batch, candidates]
    value_logits: Tensor           # [batch, 3]: loss, draw, win
    legality_logits: Tensor | None = None
    q_value_logits: Tensor | None = None  # [batch, candidates, 3], observer WDL

    def validate(self, latent: WorldLatent, candidates: WorldCandidates) -> None:
        latent.validate()
        candidates.validate(latent)
        if self.policy_logits.shape != candidates.padding_mask.shape \
                or self.value_logits.shape != (latent.tokens.shape[0], 3):
            raise ValueError("world prediction shapes differ from their inputs")
        values = [self.policy_logits, self.value_logits]
        if self.legality_logits is not None:
            if self.legality_logits.shape != self.policy_logits.shape:
                raise ValueError("world legality logits must align with candidates")
            values.append(self.legality_logits)
        if self.q_value_logits is not None:
            if self.q_value_logits.shape != (*self.policy_logits.shape, 3):
                raise ValueError("world Q logits must align with candidates and WDL")
            values.append(self.q_value_logits)
        for value in values:
            _finite_float(value, "world prediction")
            if value.device != latent.tokens.device:
                raise ValueError("world prediction device differs")
