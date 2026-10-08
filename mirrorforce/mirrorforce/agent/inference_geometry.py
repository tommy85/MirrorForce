"""Exact opt-in public inference geometry; safe for metadata-only consumers.

This module deliberately imports neither JAX nor any policy/native module.
The profile changes inference execution geometry, never training or weights.
"""
from __future__ import annotations

from types import MappingProxyType


FIXED_PUBLIC_B64 = MappingProxyType({
    "schema": "mirrorforce_inference_geometry/v1",
    "law": "shared-policy-public-features;fixed-full-window/v1",
    "batch_size": 64,
    "menu_rows": 192,
    "event_rows": 512,
    "chunk_event_rows": 64,
    "chunk_slots": 48,
    "memory_slots": 64,
    "compute_dtype": "bfloat16",
    "matmul_precision": "default",
    "padding": "repeat-last-valid-row/v1",
})

# A separately versioned candidate, not an implicit resizing of a v1 service.
# Metadata/Policy support is NOT search admission: the serving CLI and numeric
# contract still admit only their independently verified existing modes.
FIXED_PUBLIC_B8 = MappingProxyType({**FIXED_PUBLIC_B64,
    "schema": "mirrorforce_inference_geometry/v2", "batch_size": 8})


def validate_geometry(value):
    """Require the complete explicit profile and return an independent JSON dict.

    None is not a profile: callers implement their unchanged legacy default
    before calling this validator. No smaller shape or precision fallback.
    """
    if not isinstance(value, (dict, type(FIXED_PUBLIC_B64))) or not any(
            set(value) == set(profile) and all(type(value[key]) is type(expected) and value[key] == expected
                                             for key, expected in profile.items())
            for profile in (FIXED_PUBLIC_B64, FIXED_PUBLIC_B8)):
        raise ValueError("public inference requires an exact fixed public geometry and its explicit version")
    return dict(value)


__all__ = ["FIXED_PUBLIC_B64", "FIXED_PUBLIC_B8", "validate_geometry"]
