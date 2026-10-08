"""Opponent hand belief read from the Stage A public latent.

For every card of the opponent's main-deck recipe the belief says how many
copies the opponent holds in hand now, as four count classes (0 to 3). Public
evidence alone already fixes a belief: the unknown hand slots filled uniformly
from the unknown pool, the law the rollout probe draws its particles from. The
head does not relearn that counting; it adjusts it. Its outputs are additive
log adjustments to the uniform evidence belief of each card, zero before any
training, read from the acting player's public latent (``StageAGroundingModel``
tokens: the visible board, the history and the memory of earlier turns) and
the public recipe. So a head can learn what a player's behaviour tells, such
as an opponent that let a chance for a hand trap go by holding it less often.
The real hand is a loss target only, owned by ``stage_a_belief_labels``.

The head reads the latent as it is and trains on it detached, so fitting it
changes nothing the policy computes. A fitted head is its own artifact, named
by its checksum and by the checkpoint whose latents it was fitted on; a
rollout probe may weigh its information-set particles with it
(``particle_weights``).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import io
import math

import torch
from torch import Tensor, nn

from .sidecar_io import digest, read_regular
from .world_model_contract import WorldLatent


BELIEF_COUNT_CLASSES = 4
STAGE_A_BELIEF_SCHEMA = "mirrorforce_stage_a_opponent_hand_belief/v2"
STAGE_A_BELIEF_HEAD_SCHEMA = "mirrorforce_stage_a_opponent_hand_belief_head/v2"


@dataclass(frozen=True)
class BeliefRecipe:
    """The opponent's main deck: its distinct codes in ascending order and the copies of each."""

    codes: tuple[int, ...]
    copies: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.codes or len(self.codes) != len(self.copies) or list(self.codes) != sorted(set(self.codes)) \
                or any(type(code) is not int or code <= 0 for code in self.codes) \
                or any(type(count) is not int or not 1 <= count < BELIEF_COUNT_CLASSES for count in self.copies):
            raise ValueError("a belief recipe names distinct ascending codes with 1 to 3 copies each")

    @classmethod
    def from_main(cls, main) -> "BeliefRecipe":
        counts = Counter(int(code) for code in main)
        return cls(tuple(sorted(counts)), tuple(counts[code] for code in sorted(counts)))

    def counts(self, hand) -> list[int]:
        """The recipe-ordered copies of each code in ``hand``; a card outside the recipe is refused."""
        held = Counter(int(code) for code in hand)
        if set(held) - set(self.codes):
            raise ValueError("a hand holds a card outside the opponent's recipe")
        return [held.get(code, 0) for code in self.codes]


@dataclass(frozen=True)
class StageABeliefConfig:
    d_model: int = 768
    card_dim: int = 256
    heads: int = 12
    hidden_dim: int = 768
    schema: str = STAGE_A_BELIEF_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STAGE_A_BELIEF_SCHEMA:
            raise ValueError("unsupported belief head configuration schema")
        for name in ("d_model", "card_dim", "heads", "hidden_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"belief head {name} must be a positive integer")
        if self.d_model % self.heads:
            raise ValueError("belief head width must be divisible by its heads")


def _masked_logit(dtype: torch.dtype) -> float:
    return float(max(-1e9, torch.finfo(dtype).min / 2))


class OpponentHandBelief(nn.Module):
    """One query per recipe card attends over the public latent; its readout adjusts the card's four count
    classes. The last layer starts at zero: an untrained head is the uniform evidence belief."""

    def __init__(self, config: StageABeliefConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.state_norm = nn.LayerNorm(d)
        self.card_query = nn.Sequential(nn.Linear(config.card_dim, d), nn.GELU(), nn.LayerNorm(d))
        self.copies = nn.Embedding(BELIEF_COUNT_CLASSES, d)
        self.attention = nn.MultiheadAttention(d, config.heads, batch_first=True)
        self.count = nn.Sequential(nn.Linear(2 * d, config.hidden_dim), nn.GELU(), nn.LayerNorm(config.hidden_dim),
                                   nn.Linear(config.hidden_dim, BELIEF_COUNT_CLASSES))
        nn.init.zeros_(self.count[-1].weight)
        nn.init.zeros_(self.count[-1].bias)

    def forward(self, latent: WorldLatent, cards: Tensor, copies: Tensor) -> Tensor:
        """Log adjustments ``[decisions, recipe cards, 4]``; a class above a card's copies is masked out.

        ``cards`` are the recipe cards' public semantic encodings ``[recipe cards, card_dim]`` and
        ``copies`` their recipe copies.
        """
        latent.validate()
        if latent.tokens.shape[-1] != self.config.d_model or cards.ndim != 2 \
                or cards.shape[-1] != self.config.card_dim or copies.shape != cards.shape[:1] \
                or copies.dtype != torch.long or bool(((copies < 1) | (copies >= BELIEF_COUNT_CLASSES)).any()):
            raise ValueError("belief head inputs differ from its configuration")
        state = self.state_norm(latent.tokens.masked_fill(latent.padding_mask.unsqueeze(-1), 0))
        query = (self.card_query(cards.to(state.dtype)) + self.copies(copies).to(state.dtype))
        query = query.unsqueeze(0).expand(state.shape[0], -1, -1)
        attended, _ = self.attention(query, state, state, key_padding_mask=latent.padding_mask, need_weights=False)
        logits = self.count(torch.cat((query, attended), -1))
        classes = torch.arange(BELIEF_COUNT_CLASSES, device=logits.device)
        return logits.masked_fill(classes > copies.unsqueeze(-1), _masked_logit(logits.dtype))


def recipe_card_features(card_effect_encoder, store, recipe: BeliefRecipe, device) -> Tensor:
    """The recipe cards' public semantic encodings, the same rows the state encoder reads for them."""
    rows = torch.as_tensor(store.card_rows_for_codes(list(recipe.codes)), dtype=torch.long, device=device)
    effects, mask = store.padded_effect_rows_for_cards(list(recipe.codes))
    if bool((rows <= 0).any()):
        raise ValueError("a recipe card has no semantic row")
    encoding = card_effect_encoder.encode_card_rows(rows, torch.as_tensor(effects, dtype=torch.long, device=device),
                                                    torch.as_tensor(mask, dtype=torch.bool, device=device))
    return encoding.representation


def adjusted_log_probabilities(adjustments: Tensor, uniform: Tensor) -> Tensor:
    """The belief's log probabilities: the uniform evidence belief ``uniform`` ``[..., 4]`` tilted by the head's
    ``adjustments`` and normalized per card. A count the evidence rules out keeps (practically) no mass."""
    return torch.log_softmax(adjustments.float() + uniform.float().clamp_min(1e-12).log(), -1)


def belief_log_weights(adjustments, hands, recipe: BeliefRecipe, *, power: float = 1.0) -> list[float]:
    """The log importance weights that move proposals drawn from public evidence toward the belief.

    ``hands`` are the proposals' opponent hands, drawn from the uniform evidence
    belief; ``adjustments`` ``[recipe cards, 4]`` are the head's at the root.
    The belief is that law tilted by the adjustments, so a proposal's log weight
    is ``power`` times the sum of its cards' adjustments: zero adjustments weigh
    every hand alike, and every proposal stays consistent with the evidence.

    The one weighting of public proposals by a belief head: the rollout
    teacher's particles (``particle_weights``) and the play-time client's joint
    bank (``stage_a_joint_belief_runtime.belief_joint_bank``) both read it.
    """
    if type(power) not in (int, float) or not 0 <= power <= 1 or not hands:
        raise ValueError("particle weighting needs proposals and a power in [0, 1]")
    table = torch.as_tensor(adjustments, dtype=torch.float64)
    if table.shape != (len(recipe.codes), BELIEF_COUNT_CLASSES):
        raise ValueError("belief adjustments must cover the recipe's count classes")
    return [power * sum(float(table[index, count]) for index, count in enumerate(recipe.counts(hand))) for hand in hands]


def particle_weights(adjustments, hands, recipe: BeliefRecipe, *, power: float = 1.0) -> list[float]:
    """``belief_log_weights`` as importance weights, the largest one 1."""
    logs = belief_log_weights(adjustments, hands, recipe, power=power)
    top = max(logs)
    return [math.exp(value - top) for value in logs]


def choose_weighted(weights, k: int, rng) -> list[int]:
    """``k`` distinct proposal indices, drawn by weight without replacement (Efraimidis and Spirakis)."""
    if type(k) is not int or not 0 < k <= len(weights) or any(not value >= 0 for value in weights) \
            or sum(value > 0 for value in weights) < k:
        raise ValueError("weighted choice needs k proposals of positive weight")
    keys = [(math.log(rng.random() or 1e-300) / value, index) for index, value in enumerate(weights) if value > 0]
    return sorted(index for _, index in sorted(keys, reverse=True)[:k])


def effective_sample_size(weights) -> float:
    total = sum(weights)
    return total * total / sum(value * value for value in weights)


class ParticleBelief:
    """A fitted head ready to weigh a root's particles: its recipe cards encoded once, on the runtime's device."""

    def __init__(self, head: OpponentHandBelief, recipe: BeliefRecipe, card_effect_encoder, store, device, *,
                 sha256: str, fitted: dict) -> None:
        self.head, self.recipe, self.sha256, self.fitted = head.to(device).eval(), recipe, sha256, fitted
        with torch.no_grad():
            self.cards = recipe_card_features(card_effect_encoder, store, recipe, device).float()
        self.copies = torch.tensor(recipe.copies, dtype=torch.long, device=device)

    @torch.no_grad()
    def adjustments(self, latent: WorldLatent) -> list[list[float]]:
        """The head's log adjustments at one root: per recipe card, to holding 0 to 3 copies."""
        if latent.tokens.shape[0] != 1:
            raise ValueError("a particle belief reads one root")
        latent = WorldLatent(latent.tokens.float(), latent.padding_mask, latent.observer)
        return self.head(latent, self.cards, self.copies)[0].float().tolist()

    def log_weights(self, latent: WorldLatent, hands, *, power: float) -> list[float]:
        """``belief_log_weights`` of proposed opponent hands, read from the head at one root's public latent."""
        return belief_log_weights(self.adjustments(latent), hands, self.recipe, power=power)

    def record(self) -> dict:
        """What a root that drew particles with this belief names of it: the head's checksum (a checkpoint's own
        head is named by the checkpoint) and, for a checkpoint's own head, the digest of the head's state."""
        return {"head_sha256": self.sha256,
                **({"head_state_sha256": self.fitted["head_state_sha256"]}
                   if self.fitted.get("source") == OWN_HEAD_SOURCE else {})}


#: How a belief built from a loaded checkpoint's own head names its source (``own_head_belief``).
OWN_HEAD_SOURCE = "the checkpoint's own belief head"


def head_state_sha256(head: OpponentHandBelief) -> str:
    """The digest of a belief head's state: every tensor by name, dtype, shape and bytes."""
    import hashlib
    import json

    out = hashlib.sha256()
    for name, value in sorted(head.state_dict().items()):
        value = value.detach().cpu().contiguous()
        out.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        out.update(value.reshape(-1).view(torch.uint8).numpy().tobytes() if value.numel() else b"")
    return out.hexdigest()


def own_head_belief(model, *, main, store, device, checkpoint_sha256) -> ParticleBelief:
    """The loaded checkpoint's own belief head (the auxiliary head a stage trains) for weighing particles against an
    opponent that plays ``main``: no separately fitted head. Named by the checkpoint and the digest of the head's
    state; refused for a checkpoint without one."""
    if getattr(model, "belief", None) is None:
        raise ValueError("this checkpoint has no belief head of its own")
    return ParticleBelief(model.belief, BeliefRecipe.from_main(main), model.backbone.card_effect_encoder, store,
                          device, sha256=checkpoint_sha256,
                          fitted={"source": OWN_HEAD_SOURCE, "head_state_sha256": head_state_sha256(model.belief)})


def particle_belief(path, checksum, *, main, card_effect_encoder, store, device) -> ParticleBelief:
    """A fitted head for weighing particles against an opponent that plays the deck ``main``."""
    head, recipe, fitted = load_head(path, checksum)
    if recipe != BeliefRecipe.from_main(main):
        raise ValueError("the belief head was fitted for another opponent deck")
    return ParticleBelief(head, recipe, card_effect_encoder, store, device, sha256=checksum, fitted=fitted)


def head_payload(head: OpponentHandBelief, recipe: BeliefRecipe, fitted: dict) -> dict:
    """A fitted head's weights with its configuration, recipe and fit record, for ``torch.save``."""
    return {"schema": STAGE_A_BELIEF_HEAD_SCHEMA, "config": asdict(head.config),
            "recipe": {"codes": list(recipe.codes), "copies": list(recipe.copies)}, "fitted": fitted,
            "weights": {name: value.detach().cpu() for name, value in head.state_dict().items()}}


def load_head(path, checksum) -> tuple[OpponentHandBelief, BeliefRecipe, dict]:
    """A fitted head checked against its checksum, with its recipe and fit record."""
    raw = read_regular(path, "belief head")
    if digest(raw) != checksum:
        raise ValueError("belief head checksum differs")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"schema", "config", "recipe", "fitted", "weights"} \
            or payload["schema"] != STAGE_A_BELIEF_HEAD_SCHEMA:
        raise ValueError("unsupported belief head artifact")
    head = OpponentHandBelief(StageABeliefConfig(**payload["config"]))
    head.load_state_dict(payload["weights"], strict=True)
    recipe = BeliefRecipe(tuple(payload["recipe"]["codes"]), tuple(payload["recipe"]["copies"]))
    return head.eval(), recipe, payload["fitted"]
