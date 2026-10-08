"""Current-root particles drawn under the model's own count belief: uniform public proposals, resampled.

The uniform current-root bank draws hidden layouts uniformly among those consistent with the public record. This
law draws ``factor`` times as many of the same seeded uniform proposals, weighs each by how well its hidden-card
counts agree with the policy's count-belief head at the same pending observation, and resamples ``count`` of them
with equal weights. The head predicts, per public candidate card and hidden location, a distribution over 0..3+
copies. A layout's log weight is the sum over candidates and the decision-relevant locations (hand, face-down
monster, face-down spell/trap, face-down banished) of log head probability minus log proposal frequency of its
count, the proposal frequency being estimated from the drawn proposals themselves (add-one-half smoothed). The
deck and extra deck are left out: their composition follows from the rest.

Everything is a deterministic function of the public view, the seed and the head output, so admission recomputes
the bank and compares it exactly. Only public inputs and the head's output at the viewer's own pending observation
are read; the head itself was trained on engine truth as a training-only target.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from .current_root_protocol import COUNT_BANK_LAW, COUNT_NATIVE_PROOF

LOCATIONS = (1, 2, 3, 4, 6, 7)  # the head's output order: deck, hand, face-down monster/spell-trap/banished, extra
WEIGHED = (2, 3, 4, 6)
ZONE_LOCATION = {0x04: 3, 0x08: 4, 0x20: 6}  # a face-down row's zone -> the head's location
FACTOR = 8
SMOOTHING = 0.5
RESAMPLE_LAW = "systematic-PCG64-SeedSequence(seed,count-head-resample)/v1"


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check_belief(belief):
    """The service's ``public_belief`` payload: distinct public codes x six locations x four count logits."""
    if not isinstance(belief, dict) or set(belief) != {"codes", "locations", "logits"} \
            or belief["locations"] != list(LOCATIONS) or not isinstance(belief["codes"], list) \
            or any(type(code) is not int or code <= 0 for code in belief["codes"]) \
            or len(set(belief["codes"])) != len(belief["codes"]):
        raise ValueError("count belief needs exact distinct public raw codes and the six head locations")
    logits = np.asarray(belief["logits"], np.float64).reshape(len(belief["codes"]), len(LOCATIONS), 4) \
        if belief["codes"] else np.zeros((0, len(LOCATIONS), 4))
    if not np.isfinite(logits).all():
        raise ValueError("count belief logits must be finite")
    return logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)


def layout_counts(layout):
    """Copies per (code, head location) among the decision-relevant hidden places of one layout."""
    counts = {}
    for code in layout.hand:
        counts[code, 2] = counts.get((code, 2), 0) + 1
    for zone, _sequence, code in layout.facedown:
        location = ZONE_LOCATION.get(zone)
        if location is not None:
            counts[code, location] = counts.get((code, location), 0) + 1
    return counts


def log_weights(layouts, belief):
    """Head-over-proposal count log ratios, summed per layout (see the module docstring)."""
    head = check_belief(belief)
    if not layouts:
        raise ValueError("count-head weights need proposals")
    columns = [LOCATIONS.index(location) for location in WEIGHED]
    observed = np.zeros((len(layouts), len(belief["codes"]), len(WEIGHED)), np.int64)
    index = {code: i for i, code in enumerate(belief["codes"])}
    for row, layout in enumerate(layouts):
        for (code, location), copies in layout_counts(layout).items():
            if code in index:
                observed[row, index[code], WEIGHED.index(location)] = min(copies, 3)
    frequency = np.stack([(observed == n).sum(0) for n in range(4)], -1) + SMOOTHING  # [codes, places, 4]
    proposal = np.log(frequency / frequency.sum(-1, keepdims=True))
    target = head[:, columns, :]
    codes, places = np.meshgrid(np.arange(len(belief["codes"])), np.arange(len(WEIGHED)), indexing="ij")
    ratio = target - proposal
    return np.array([ratio[codes, places, observed[row]].sum() for row in range(len(layouts))], np.float64)


def systematic(log_w, count, seed):
    """``count`` indices, each drawn in proportion to exp(log_w), from one seeded uniform offset."""
    shifted = np.exp(log_w - log_w.max())
    cumulative = np.cumsum(shifted / shifted.sum())
    cumulative[-1] = 1.0
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed & 0xffffffffffffffff, 0x6b6c])))
    points = (rng.random() + np.arange(count)) / count
    return [int(i) for i in np.searchsorted(cumulative, points, side="right")]


def effective_sample_size(log_w):
    w = np.exp(log_w - log_w.max())
    return float(w.sum() ** 2 / (w * w).sum())


def resample(public, recipe, *, viewer, count, seed, belief, obs_sha256, extra_origin_slots, categories):
    """The resampled layouts with their public keys and sizes, and the record that binds them."""
    from ..common.stage_a_joint_belief_runtime import _public_draws
    if type(count) is not int or not 1 <= count <= 128:
        raise ValueError("count-head resampling needs a finite positive particle count")
    layouts, keys, sizes = _public_draws(public, recipe, viewer=viewer, count=count * FACTOR, seed=seed,
                                         extra_origin_slots=extra_origin_slots, categories=categories)
    log_w = log_weights(layouts, belief)
    selected = systematic(log_w, count, seed)
    record = {"schema": COUNT_NATIVE_PROOF, "law": COUNT_BANK_LAW, "obs_sha256": obs_sha256, "seed": seed,
              "count": count, "proposals": len(layouts), "factor": FACTOR, "smoothing": SMOOTHING,
              "weighed_locations": list(WEIGHED), "resample_law": RESAMPLE_LAW, "selected": selected,
              "effective_sample_size": effective_sample_size(log_w), "distinct": len(set(selected)),
              "belief": belief, "belief_sha256": _sha(belief), "search_admission": False}
    return [layouts[i] for i in selected], keys, sizes, record


def bank(public, recipe, *, viewer, count, seed, belief, obs_sha256, extra_origin_slots=(), categories=()):
    """An equal-weight current-root bank of the resampled layouts, and its proof."""
    from ..common.stage_a_joint_belief_runtime import JointDraw, JointParticleBank
    layouts, keys, sizes, record = resample(public, recipe, viewer=viewer, count=count, seed=seed, belief=belief,
        obs_sha256=obs_sha256, extra_origin_slots=extra_origin_slots, categories=categories)
    probabilities = tuple(1 / count for _ in range(count))
    return JointParticleBank(viewer, recipe, tuple(JointDraw.from_layout(x) for x in layouts), probabilities,
                             tuple(math.log(p) for p in probabilities), keys, sizes, seed, 0., None,
                             proposal_law=COUNT_BANK_LAW), record


def admit(public, joint_bank, record, *, extra_origin_slots, categories):
    """Recompute the bank from its public view, seed and head output; it must be identical."""
    from ..common.stage_a_joint_belief_runtime import JointDraw
    if type(record) is not dict or record.get("schema") != COUNT_NATIVE_PROOF or record.get("law") != COUNT_BANK_LAW \
            or joint_bank.proposal_law != COUNT_BANK_LAW or record.get("seed") != joint_bank.seed \
            or record.get("count") != len(joint_bank.draws) or joint_bank.power != 0 \
            or joint_bank.head_sha256 is not None or record.get("belief_sha256") != _sha(record.get("belief")):
        raise ValueError("count-head bank differs from its binding record")
    layouts, keys, sizes, again = resample(public, joint_bank.recipe, viewer=joint_bank.viewer,
        count=record["count"], seed=record["seed"], belief=record["belief"], obs_sha256=record["obs_sha256"],
        extra_origin_slots=extra_origin_slots, categories=categories)
    if again != record or keys != joint_bank.field_keys or sizes != joint_bank.zone_sizes \
            or tuple(JointDraw.from_layout(x) for x in layouts) != joint_bank.draws:
        raise ValueError("count-head bank is not the recomputed resampling of its public proposals")
