"""Deadline-bounded direct AR draws for a single public root.

Every lane is a free-generated sequence, not a reweighted uniform proposal.
This low-level bridge confers no checkpoint, public-law or native admission.
The caller must bind all three, then realize the complete original bank.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from .belief_distribution_eval import (
    DistributionBudgetExceeded, _decoder_geometry, _digest, _geometry,
)
from .search.belief_ar_law import masked_log_probabilities

SCHEMA = "mirrorforce_direct_public_ar_draws/v1"
RNG_LAW = "PCG64-SeedSequence-spawn-per-original-lane/v1"
EXECUTION_LAW = "same-root-fixed-batch-distinct-generated-prefixes/v1"


def draw_bank(law, decoder, *, count, seed, deadline, clock=time.monotonic):
    """Return all original IID draws or raise; a partial bank is never reweighted.

    Feasibility is recomputed from each generated prefix, never a teacher mask.
    All support, decoder and final-layout work is inside the caller's deadline.
    A slow/unknown support query propagates; no repair, fallback law or redraw.
    """
    if type(count) is not int or not 1 <= count <= 128 \
            or type(seed) is not int or not 0 <= seed < 2 ** 64 \
            or type(deadline) not in (int, float) or not math.isfinite(deadline) or not callable(clock):
        raise ValueError("direct AR draws need a bounded count, uint64 seed and finite deadline")

    def check():
        if clock() >= deadline:
            raise DistributionBudgetExceeded("AR root deadline exhausted; no partial bank accepted")

    check()
    geometry = _geometry(law.codes, [[s.location, s.sequence] for s in law.fields], law.hand_size)
    if type(law.length) is not int or law.length != len(law.fields) + law.hand_size:
        raise ValueError("AR law length differs from complete public geometry")
    identity = json.loads(json.dumps(decoder.bind_distribution(law.codes, law.length), allow_nan=False))
    _decoder_geometry(identity, geometry)
    if type(identity.get("fixed_batch_size")) is not int or count > identity["fixed_batch_size"] \
            or not callable(getattr(decoder, "prefix_batch", None)):
        raise ValueError("direct AR draws require an explicitly bound multi-prefix decoder capacity")
    streams = [np.random.Generator(np.random.PCG64(child))
               for child in np.random.SeedSequence(seed).spawn(count)]
    prefixes, log_probabilities = [()] * count, [0.] * count
    masks = {}
    mask_chain = []
    calls = 0
    for _ in range(law.length):
        check()
        current = []
        for prefix in prefixes:
            check()
            if prefix not in masks:
                mask = np.array(law.mask(prefix), copy=True)
                check()
                if mask.dtype != bool or mask.shape != (len(law.codes),) or not mask.any():
                    raise ValueError("AR public support needs a nonempty boolean full-vocabulary mask")
                mask.setflags(write=False)
                masks[prefix] = mask
            current.append(masks[prefix])
        check()
        logits = np.asarray(decoder.prefix_batch(tuple(prefixes)))
        check()
        calls += 1
        if logits.shape != (count, len(law.codes)) or not np.isfinite(logits).all():
            raise ValueError("batched AR logits differ from the entire original bank")
        mask_chain.append([mask.tolist() for mask in current])
        for lane, (mask, values, rng) in enumerate(zip(current, logits, streams)):
            check()
            logp = masked_log_probabilities(values, mask)
            probabilities = np.exp(logp)
            if np.any(probabilities[mask] == 0) or not np.isfinite(probabilities).all() \
                    or not math.isclose(float(probabilities.sum()), 1., rel_tol=0, abs_tol=1e-12):
                raise FloatingPointError("AR probability underflow/normalization; support cannot be removed")
            index = int(rng.choice(len(law.codes), p=probabilities))
            prefixes[lane] += (law.codes[index],)
            log_probabilities[lane] += float(logp[index])
    for prefix in prefixes:
        check()
        law.layout(prefix)
        check()
    result = {"schema": SCHEMA, "execution_law": EXECUTION_LAW, "rng_law": RNG_LAW,
              "geometry": geometry, "seed": seed, "count": count,
              "sequences": [list(prefix) for prefix in prefixes],
              "sequence_log_probabilities": log_probabilities,
              "monte_carlo_weights": [1. / count] * count,
              "decoder_identity": identity, "decoder_calls": calls,
              "unique_public_prefixes": len(masks), "support_mask_sha256": _digest(mask_chain),
              "duplicates_retained": True, "partial_bank": False,
              "teacher_channels": False, "search_admission": False}
    result["sha256"] = _digest(result)
    check()
    return result
