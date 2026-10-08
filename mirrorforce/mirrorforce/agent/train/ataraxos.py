"""Arm A's training recipe: the Ataraxos "dynamically damped self-play" update, ported for the trainer.

Reference (read-only clone, never imported): ``ataraxos/stratego`` at 92db29e --
``pyengine/core/rl.py`` (``RLConfig`` lines 48-107, ``train()`` lines 484-615, ``power_schedule`` line 756),
``pyengine/core/buffer.py`` (``add_post_act`` lines 133-190, ``process_data`` lines 194-241) and
``pyengine/networks/exponential_weighted_average.py`` (EMA). Sokota et al., "Scalable decision-making for games of
imperfect information", Nature 2026, doi:10.1038/s41586-026-11036-y.

- Schedules: ``x = clamp(coef / (s + 1) ** decay, floor, ceil)`` for the learning rate and the magnet temperature.
  Their ``s`` is the training iteration (about 4.9M samples each: 208B steps over 42.4k iterations); ours is
  ``samples seen / samples_per_iteration`` (rescaling by samples consumed, design section 10d).
- Targets: each seat's chain of its own decisions. The target of a decision is the one-hot outcome when that seat's
  action ended the game (the other seat's last decision gets the mirrored outcome: "solipsistic" terminal rows),
  else the softmax of the seat's value at its next decision. Returns are TD(lambda=0.8) on the categorical value
  distributions, advantages GAE(lambda=0.5) on the scalar win - loss. Computed once per update from the values the
  actor stored, before any optimizer step (their ``process_data``). Differences, both forced by the setting: seats
  do not alternate decisions (chains follow each seat), and a fixed 32-step segment ends with a bootstrap from the
  next observation's value for the acting seat and its win/loss mirror for the other (they wait for ready rows).
- Advantage filter: ``threshold = max(quantile(|A|, 0.75), 0.01)`` over the valid decisions of the batch on each
  learner device (per rank, as theirs); kept = valid and ``|A| >= threshold``; the mask applies to the whole loss.
- Loss per kept decision: ``policy_coef * PPO-clip(0.2) + temperature * KL(pi || uniform over legal)
  + vf_coef * CE(categorical return, value head) + kl_coef * KL(pi || pi_behaviour)``, means over kept decisions.
- EMA of the parameters, decay 0.999, updated once per update.

Value distributions are ordered (win, draw, loss), the order of policy_net's value head.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

NEG = -1e8  # logits at or below this are illegal rows (policy_net writes -1e9)


@dataclasses.dataclass(frozen=True)
class AtaraxosConfig:
    clip_range: float = 0.2
    ema_decay: float = 0.999
    td_lambda: float = 0.8
    gae_lambda: float = 0.5
    vf_coef: float = 1.0
    policy_coef: float = 1.0
    temperature_coef: float = 0.05
    temperature_ceil: float = 0.1
    temperature_floor: float = 0.001
    temperature_decay: float = 0.3
    kl_coef: float = 0.1
    adv_filt_rate: float = 0.75
    adv_filt_thresh: float = 0.01
    lr_coef: float = 0.5
    lr_decay: float = 1.1
    lr_ceil: float = 1e-4
    lr_floor: float = 5e-6
    max_grad_norm: float = 0.267
    adam_eps: float = 1e-8
    samples_per_iteration: float = 4.9e6
    """their samples per training iteration; our schedule step is samples seen / this"""
    magnet: str = "uniform_legal"
    """the magnet policy rho: "uniform_legal" (uniform over the prompt's legal options: the released Stratego code's
    uniform_magnet=True, and the doudizhu recipe per staged step; user decision 2026-10-02, since the game engine already
    stages an action into prompts) or "magnet_card_grouped/v1" (the paper's eq. 6, S3, uniform over movable pieces
    then over their moves; here over the menu's source cards, then that card's options; a registered alternative)"""


def power_schedule(coef, step, decay, ceil, floor):
    """``rl.py:756``: coef / (step + 1) ** decay, clamped to [floor, ceil] (floats or arrays)."""
    x = coef / ((step + 1) ** decay)
    return jnp.minimum(jnp.maximum(x, floor), ceil)


def schedule_step(samples_seen, cfg: AtaraxosConfig):
    return samples_seen / cfg.samples_per_iteration


def learning_rate(samples_seen, cfg: AtaraxosConfig):
    return power_schedule(cfg.lr_coef, schedule_step(samples_seen, cfg), cfg.lr_decay, cfg.lr_ceil, cfg.lr_floor)


def temperature(samples_seen, cfg: AtaraxosConfig):
    return power_schedule(cfg.temperature_coef, schedule_step(samples_seen, cfg), cfg.temperature_decay,
                          cfg.temperature_ceil, cfg.temperature_floor)


def outcome_one_hot(reward):
    """(win, draw, loss) of the player whose perspective ``reward`` is in."""
    return jnp.stack([reward > 0, reward == 0, reward < 0], axis=-1).astype(jnp.float32)


def mirror(dist):
    """The other player's (win, draw, loss) in a zero-sum game."""
    return dist[..., ::-1]


def scalar(dist):
    return dist[..., 0] - dist[..., 2]


def targets_by_seat(dists, rewards, dones, next_dones, mains, next_dist, cfg: AtaraxosConfig):
    """Categorical TD(lambda) returns [T, B, 3] and scalar GAE advantages [T, B] along each seat's own decisions.

    ``dists`` [T, B, 3]: the acting seat's value distribution at each decision (actor's); ``rewards`` [T, B]: the
    outcome after the step, from the acting seat's view (nonzero only at game ends); ``dones`` [T, B]: the step is
    the environment's reset transition (not a decision); ``next_dones`` [T, B]: the game ended after this step's
    action; ``mains`` [T, B]: the acting seat is the main seat; ``next_dist`` [B, 3]: the main seat's value
    distribution at the observation after the segment (bootstrap; the other seat takes its mirror)."""

    def step(carry, x):
        (dist_main, av_main, aa_main), (dist_other, av_other, aa_other) = carry
        dist, reward, done, next_done, main = x
        end = next_done & ~done
        result = outcome_one_hot(reward)  # the acting seat's outcome
        acting_end = jnp.where(main[:, None], result, mirror(result))  # ... seen by the main seat
        zero3, zero = jnp.zeros_like(dist), jnp.zeros_like(reward)
        # a game end resets both chains: the acting seat's target is its outcome, the other's last decision gets
        # the mirrored outcome (their solipsistic terminal rows)
        dist_main = jnp.where(end[:, None], acting_end, dist_main)
        dist_other = jnp.where(end[:, None], mirror(acting_end), dist_other)
        av_main = jnp.where(end[:, None], zero3, av_main)
        av_other = jnp.where(end[:, None], zero3, av_other)
        aa_main = jnp.where(end, zero, aa_main)
        aa_other = jnp.where(end, zero, aa_other)
        target = jnp.where(main[:, None], dist_main, dist_other)
        av_next = jnp.where(main[:, None], av_main, av_other)
        aa_next = jnp.where(main, aa_main, aa_other)
        av = target - dist + cfg.td_lambda * av_next
        aa = scalar(target) - scalar(dist) + cfg.gae_lambda * aa_next
        decision = ~done
        update_main = decision & main
        update_other = decision & ~main
        dist_main = jnp.where(update_main[:, None], dist, dist_main)
        av_main = jnp.where(update_main[:, None], av, av_main)
        aa_main = jnp.where(update_main, aa, aa_main)
        dist_other = jnp.where(update_other[:, None], dist, dist_other)
        av_other = jnp.where(update_other[:, None], av, av_other)
        aa_other = jnp.where(update_other, aa, aa_other)
        returns = jnp.where(decision[:, None], dist + av, 0.0)
        advantages = jnp.where(decision, aa, 0.0)
        return ((dist_main, av_main, aa_main), (dist_other, av_other, aa_other)), (returns, advantages)

    zero3 = jnp.zeros_like(next_dist)
    zero = jnp.zeros(next_dist.shape[:-1], next_dist.dtype)
    init = ((next_dist, zero3, zero), (mirror(next_dist), zero3, zero))
    _, (returns, advantages) = jax.lax.scan(step, init, (dists, rewards, dones, next_dones, mains), reverse=True)
    return returns, advantages


def advantage_mask(advantages, valid, cfg: AtaraxosConfig):
    """``buffer.py:240``: keep valid decisions with |A| >= max(quantile(|A| over valid, rate), floor)."""
    magnitude = jnp.abs(advantages)
    threshold = jnp.maximum(jnp.nanquantile(jnp.where(valid, magnitude, jnp.nan), cfg.adv_filt_rate),
                            cfg.adv_filt_thresh)
    return valid & (magnitude >= threshold), threshold


def magnet_log_probs(legal, groups):
    """log rho over a menu (``legal`` [N, A], ``groups`` [N, A] each option's source card row + 1, 0 = none):
    uniform over the groups among the legal options, then uniform within the group; an option without a source card
    is its own group. Illegal options get 0."""
    index = jnp.arange(legal.shape[-1])
    key = jnp.where(groups > 0, groups, 1000 + index)  # card rows are below 1000
    same = (key[..., :, None] == key[..., None, :]) & legal[..., None, :]
    in_group = jnp.maximum(same.sum(-1), 1).astype(jnp.float32)  # legal options sharing a's group
    count = jnp.maximum(jnp.where(legal, 1.0 / in_group, 0.0).sum(-1, keepdims=True), 1.0)  # number of groups
    return jnp.where(legal, -jnp.log(count) - jnp.log(in_group), 0.0)


def loss_terms(new_logits, new_value_logits, behaviour_logits, actions, advantages, returns, kept,
               temperature_now, cfg: AtaraxosConfig, denominator=None, groups=None):
    """``rl.py:520-575``, masked to the kept decisions. Returns the total loss and its terms (means over kept;
    ``denominator`` replaces the kept count under gradient accumulation)."""
    legal = behaviour_logits > NEG
    log_probs = jax.nn.log_softmax(jnp.where(legal, new_logits, -jnp.inf), axis=-1)
    log_probs = jnp.where(legal, log_probs, 0.0)
    old_log_probs = jax.nn.log_softmax(jnp.where(legal, behaviour_logits, -jnp.inf), axis=-1)
    old_log_probs = jnp.where(legal, old_log_probs, 0.0)
    probs = jnp.where(legal, jnp.exp(log_probs), 0.0)
    log_prob = jnp.take_along_axis(log_probs, actions[:, None], axis=-1)[:, 0]
    old_log_prob = jnp.take_along_axis(old_log_probs, actions[:, None], axis=-1)[:, 0]
    ratio = jnp.exp(log_prob - old_log_prob)
    policy = -jnp.minimum(advantages * ratio,
                          advantages * jnp.clip(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range))
    kl = (probs * (log_probs - old_log_probs)).sum(-1)
    entropy = -(probs * log_probs).sum(-1)
    if cfg.magnet == "magnet_card_grouped/v1":
        if groups is None:
            raise ValueError("the card-grouped magnet needs each option's source card")
        magnet_kl = (probs * (log_probs - magnet_log_probs(legal, groups))).sum(-1)  # KL(pi || rho)
    elif cfg.magnet == "uniform_legal":
        legal_count = legal.sum(-1).astype(jnp.float32)
        magnet_kl = -entropy + jnp.log(jnp.maximum(legal_count, 1.0))  # KL(pi || uniform over legal)
    else:
        raise ValueError(f"unknown magnet {cfg.magnet}")
    value = -(returns * jax.nn.log_softmax(new_value_logits.astype(jnp.float32), axis=-1)).sum(-1)
    weight = kept.astype(jnp.float32)
    count = weight.sum() if denominator is None else denominator
    mean = lambda x: (x * weight).sum() / jnp.maximum(count, 1.0)
    terms = {"policy": mean(policy), "magnet_kl": mean(magnet_kl), "value": mean(value), "kl": mean(kl),
             "entropy": mean(entropy), "clip_fraction": mean((jnp.abs(ratio - 1) > cfg.clip_range).astype(jnp.float32))}
    total = (cfg.policy_coef * terms["policy"] + temperature_now * terms["magnet_kl"]
             + cfg.vf_coef * terms["value"] + cfg.kl_coef * terms["kl"])
    return total, terms


def ema_update(ema, params, decay):
    """``exponential_weighted_average.py``: ema <- decay * ema + (1 - decay) * params (once per update)."""
    return jax.tree_util.tree_map(lambda e, p: decay * e + (1.0 - decay) * p, ema, params)
