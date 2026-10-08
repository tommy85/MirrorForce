"""Fixed-shape, public-only AR scoring bridge for offline distribution diagnostics.

All public feature slots are retained. Teacher targets and cached teacher masks
are not accepted here. This is a reference decoder bridge, not a policy-service
or learned-search admission; the caller must bind the loaded head checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np

from mirrorforce.agent.model.belief_ar import PUBLIC_KEYS, admit_public


def _parameter_snapshot(params):
    """Own every leaf before JAX transfer; jnp.asarray may share a writable NumPy buffer."""
    digest = hashlib.sha256()
    def copy(value, path):
        if isinstance(value, Mapping):
            if not value or any(type(key) is not str for key in value):
                raise ValueError("decoder parameter mappings need nonempty named fields")
            return {key: copy(value[key], [*path, key]) for key in sorted(value)}
        host = np.array(value, copy=True)
        if (host.dtype.kind not in "biuf" and str(host.dtype) != "bfloat16") or not np.isfinite(host).all():
            raise ValueError("decoder parameters must be finite numeric arrays")
        device = jnp.asarray(host)
        actual = np.asarray(device)
        raw = np.ascontiguousarray(actual).tobytes()
        if actual.shape != host.shape or actual.dtype != host.dtype or raw != np.ascontiguousarray(host).tobytes():
            raise ValueError("decoder parameter device transfer changed dtype, shape or bits")
        header = json.dumps([path, str(host.dtype), list(host.shape), len(raw)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(raw)
        return device
    if not isinstance(params, Mapping):
        raise ValueError("decoder parameters need their named model tree")
    parameters = copy(params, [])
    return parameters, digest.hexdigest()


@dataclass(frozen=True, init=False, eq=False, repr=False)
class PublicDecoderSnapshot:
    """One owned head and one JIT callable, reusable only with the exact creating model/parameter owners.

    This is not a checkpoint admission. Reuse is per complete input shape/dtype/device signature;
    compact rows of different shapes may compile separate executables. Batch lanes still repeat
    ONE public row and ONE prefix; this does not implement multi-prefix/row batching.
    """

    def __init__(self, model, params):
        model_config, owned_model = deepcopy(asdict(model.config)), deepcopy(model)
        parameters, parameter_digest = _parameter_snapshot(params)

        def decoder_apply(parameters, public, previous):
            return owned_model.apply({"params": parameters}, public, previous, deterministic=True)

        for name, value in (("_owner_model", model), ("_owner_params", params), ("_model_config", model_config),
                            ("_parameters", parameters), ("_parameter_digest", parameter_digest),
                            ("_executor", jax.jit(decoder_apply))):
            object.__setattr__(self, name, value)

    def _bind(self, model, params):
        if model is not self._owner_model or params is not self._owner_params \
                or asdict(model.config) != self._model_config:
            raise ValueError("shared decoder snapshot belongs to a different head/model/config owner")

    @property
    def identity(self):
        return {"model": deepcopy(self._model_config), "actual_parameter_tree_sha256": self._parameter_digest,
                "checkpoint_admitted": False}

    def _call(self, public, previous):
        return self._executor(self._parameters, public, previous)


class PublicDecoderScorer:
    def __init__(self, model, params, public, *, codes, target_length, batch_size, snapshot=None):
        if snapshot is not None:
            if not isinstance(snapshot, PublicDecoderSnapshot):
                raise ValueError("shared decoder snapshot needs its typed immutable head owner")
            snapshot._bind(model, params)
        if not isinstance(public, dict) or set(public) != PUBLIC_KEYS:
            raise ValueError("distribution decoder accepts only explicit public features, never targets/teacher masks")
        if not isinstance(codes, (list, tuple)) or any(type(c) is not int or c <= 0 for c in codes) \
                or list(codes) != sorted(set(codes)) or type(target_length) is not int or target_length < 0 \
                or type(batch_size) is not int or not 1 <= batch_size <= 128:
            raise ValueError("distribution decoder needs canonical public codes, length and fixed batch size")
        # Copy once before device transfer, so later edits to a loader's cache
        # cannot mutate the scored information set. Nothing is truncated.
        host = {key: np.array(value, copy=True) for key, value in public.items()}
        n, v = host["slot_valid"].shape, host["candidate_valid"].shape
        if len(n) != 1 or len(v) != 1 or not 0 <= target_length <= n[0] \
                or len(codes) > v[0] or target_length and not codes \
                or not np.array_equal(host["slot_valid"], np.arange(n[0]) < target_length) \
                or not np.array_equal(host["candidate_valid"], np.arange(v[0]) < len(codes)):
            raise ValueError("distribution decoder vocabulary/target geometry must match all real slots followed by padding")
        previous = np.full((batch_size, n[0]), -1, np.int32)
        expanded = {key: np.broadcast_to(value, (batch_size, *value.shape)) for key, value in host.items()}
        admit_public(expanded, previous, model.config)
        self._codes, self._length, self._slots, self._batch_size = tuple(codes), target_length, n[0], batch_size
        self._vocabulary_capacity = v[0]
        self._index = {code: i for i, code in enumerate(codes)}
        features = {key: jnp.asarray(value) for key, value in expanded.items()}
        if any(value.shape != expanded[key].shape or value.dtype != expanded[key].dtype for key, value in features.items()):
            raise ValueError("decoder public device transfer changed registered dtype or shape")
        snapshot = PublicDecoderSnapshot(model, params) if snapshot is None else snapshot
        self._call = lambda previous: snapshot._call(features, previous)
        digest = hashlib.sha256()
        for key, value in sorted(host.items()):
            digest.update(json.dumps([key, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        self._identity = {"schema": "mirrorforce_public_ar_distribution_decoder/v2",
            "public_feature_sha256": digest.hexdigest(), "codes": list(codes), "target_length": target_length,
            "fixed_batch_size": batch_size, "model": asdict(model.config), "deterministic": True,
            "full_public_shapes": {key: list(value.shape) for key, value in host.items()},
            "actual_parameter_tree_sha256": snapshot.identity["actual_parameter_tree_sha256"],
            "parameter_digest_law": "sorted-named-tree; length-prefixed-path-dtype-shape-bytes/v1",
            "checkpoint_admitted": False,
            "parameter_provenance": "owned parameter snapshot only; checkpoint/data/source admission still required",
            "teacher_channels": False, "search_admission": False}

    @property
    def codes(self):
        return self._codes

    @property
    def length(self):
        return self._length

    @property
    def identity(self):
        return deepcopy(self._identity)

    def bind_distribution(self, codes, length):
        if tuple(codes) != self._codes or type(length) is not int or length != self._length:
            raise ValueError("public decoder vocabulary/target length differs from the distribution law")
        return self.identity

    def __call__(self, prefix):
        if not isinstance(prefix, tuple) or len(prefix) >= self._length \
                or any(type(code) is not int or code not in self._index for code in prefix):
            raise ValueError("distribution decoder needs a generated canonical-code prefix before the next real slot")
        previous = np.full((self._batch_size, self._slots), -1, np.int32)
        for i, code in enumerate(prefix):
            previous[:, i + 1] = self._index[code]
        logits = np.asarray(self._call(jnp.asarray(previous)))
        expected = (self._batch_size, self._slots, self._vocabulary_capacity)
        if logits.shape != expected or not np.isfinite(logits).all():
            raise FloatingPointError("distribution decoder returned invalid logits")
        return np.asarray(logits[0, len(prefix), :len(self._codes)], np.float64)

    def sequence_logits(self, sequence):
        """Score all prefixes of ONE complete sequence in one fixed-shape causal forward.

        This is a scoring API, not a sampler or a marginal probability. It returns
        unnormalized logits; the caller still applies the public law's mask at
        EACH prefix. Every batch lane repeats the same sequence. A deployment
        must separately prove bit equality to ``self(sequence[:t])`` on its own
        backend; CPU evidence alone does not admit a GPU execution change.
        """
        if not isinstance(sequence, tuple) or len(sequence) != self._length \
                or any(type(code) is not int or code not in self._index for code in sequence):
            raise ValueError("complete-sequence scoring needs every canonical code, including the final token")
        if not sequence:
            return np.empty((0, len(self._codes)), np.float64)
        previous = np.full((self._batch_size, self._slots), -1, np.int32)
        previous[:, 1:self._length] = [self._index[code] for code in sequence[:-1]]
        logits = np.asarray(self._call(jnp.asarray(previous)))
        expected = (self._batch_size, self._slots, self._vocabulary_capacity)
        if logits.shape != expected or not np.isfinite(logits).all():
            raise FloatingPointError("complete-sequence decoder returned invalid logits")
        return np.asarray(logits[0, :self._length, :len(self._codes)], np.float64)

    def prefix_batch(self, prefixes):
        """One public root, several independently generated prefixes, one fixed-batch forward.

        Real lanes retain their order and may be at different decoding positions.
        Unused lanes repeat the final real prefix; no public feature or target slot
        is shortened. This is an explicit new execution path, requiring its own
        backend numerical proof before online admission.
        """
        if not isinstance(prefixes, (list, tuple)) or not 1 <= len(prefixes) <= self._batch_size \
                or any(not isinstance(prefix, tuple) or len(prefix) >= self._length
                       or any(type(code) is not int or code not in self._index for code in prefix)
                       for prefix in prefixes):
            raise ValueError("batched decoder needs 1..fixed_batch_size valid generated prefixes")
        padded = tuple(prefixes) + (prefixes[-1],) * (self._batch_size - len(prefixes))
        previous = np.full((self._batch_size, self._slots), -1, np.int32)
        for lane, prefix in enumerate(padded):
            previous[lane, 1:len(prefix) + 1] = [self._index[code] for code in prefix]
        logits = np.asarray(self._call(jnp.asarray(previous)))
        expected = (self._batch_size, self._slots, self._vocabulary_capacity)
        if logits.shape != expected or not np.isfinite(logits).all():
            raise FloatingPointError("batched decoder returned invalid logits")
        return np.stack([np.asarray(logits[lane, len(prefix), :len(self._codes)], np.float64)
                         for lane, prefix in enumerate(prefixes)])


__all__ = ["PublicDecoderScorer", "PublicDecoderSnapshot"]
