"""Immutable same-forward public features owned by one pending observation.

No policy call, recurrent state, RNG, target, hidden layout or engine is kept.
Bytes-backed arrays cannot be made writable by a consumer or a cloned session.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np

SCHEMA = "mirrorforce_pending_public_ar_features/v1"
BLOCKS = ("state", "turn", "chunks", "memory", "candidates")
KEYS = frozenset(key for block in BLOCKS for key in (block, "candidate_valid" if block == "candidates"
                                                 else block + "_valid"))


def _owned(array):
    value = np.ascontiguousarray(array)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True, init=False, eq=False, repr=False)
class PendingPublicFeatures:
    def __init__(self, features, candidates, *, obs_sha256):
        if type(obs_sha256) is not str or len(obs_sha256) != 64 \
                or any(c not in "0123456789abcdef" for c in obs_sha256) \
                or type(features) is not dict or set(features) != KEYS:
            raise ValueError("AR feature cache needs only complete public blocks and its exact observation SHA")
        host = {key: _owned(value) for key, value in features.items()}
        for block in BLOCKS:
            values = host[block]
            valid = host["candidate_valid" if block == "candidates" else block + "_valid"]
            if values.ndim != 2 or values.shape[1] < 1 \
                    or (values.dtype.kind != "f" and str(values.dtype) != "bfloat16") \
                    or not np.isfinite(values).all() or valid.dtype != bool or valid.shape != values.shape[:1]:
                raise ValueError("AR feature blocks require finite complete floating arrays and boolean masks")
        candidates = _owned(candidates)
        if candidates.ndim != 2 or candidates.shape[1] != 3 or candidates.dtype.kind not in "iu" \
                or np.any(candidates[:, :2] < 0) or np.any(candidates[:, :2] > 255) \
                or not np.array_equal(candidates[:, 2] > 0, host["candidate_valid"]):
            raise ValueError("AR features differ from the same observation's public candidate rows")
        digest = hashlib.sha256()
        for key, value in sorted({**host, "observation_candidates": candidates}.items()):
            header = json.dumps([key, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            digest.update(value.tobytes())
        object.__setattr__(self, "_features", MappingProxyType(host))
        object.__setattr__(self, "_candidates", candidates)
        object.__setattr__(self, "_obs_sha256", obs_sha256)
        object.__setattr__(self, "_feature_sha256", digest.hexdigest())

    def __deepcopy__(self, memo):
        return self  # all reachable arrays are immutable owned bytes

    def read(self, *, obs_sha256):
        if obs_sha256 != self._obs_sha256:
            raise ValueError("AR features belong to another pending observation")
        return dict(self._features), self._candidates

    @property
    def identity(self):
        return {"schema": SCHEMA, "obs_sha256": self._obs_sha256,
                "public_feature_sha256": self._feature_sha256,
                "nbytes": sum(array.nbytes for array in self._features.values()) + self._candidates.nbytes,
                "shapes": {key: list(value.shape) for key, value in self._features.items()},
                "source": "same-public-forward", "teacher_channels": False}
