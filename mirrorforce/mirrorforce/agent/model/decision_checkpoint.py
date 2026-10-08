"""Strict, opt-in migration from the legacy structured architecture."""
from collections.abc import Mapping

import flax.serialization
import numpy as np


NEW_ROOTS = (
    ("params", "StructuredEncoder_0", "decision_roles"),
    ("params", "decision_actor"),
)


def restore_decision_checkpoint(template, data, allow_legacy=False):
    target = flax.serialization.to_state_dict(template)
    source = flax.serialization.msgpack_restore(data)
    # The trainer saves an empty bookkeeping collection even without BatchNorm.
    # It carries no learned state; nonempty unexpected collections still fail.
    if "batch_stats" not in target and source.get("batch_stats") == {}:
        source.pop("batch_stats")
    if target.get("batch_stats") == {} and "batch_stats" not in source:
        source["batch_stats"] = {}
    added = []

    def merge(expected, saved, path=()):
        if isinstance(expected, Mapping):
            if not isinstance(saved, Mapping):
                raise ValueError(f"checkpoint node type mismatch at {'/'.join(path)}")
            extras = set(saved) - set(expected)
            if extras:
                raise ValueError(f"unexpected checkpoint keys at {'/'.join(path)}: {sorted(extras)}")
            result = {}
            for key, value in expected.items():
                child = (*path, key)
                if key not in saved:
                    if allow_legacy and child in NEW_ROOTS:
                        added.append("/".join(child))
                        result[key] = value
                    else:
                        raise ValueError(f"missing checkpoint key: {'/'.join(child)}")
                else:
                    result[key] = merge(value, saved[key], child)
            return result
        if np.shape(expected) != np.shape(saved):
            raise ValueError(
                f"checkpoint shape mismatch at {'/'.join(path)}: "
                f"{np.shape(saved)} != {np.shape(expected)}"
            )
        if hasattr(expected, "dtype") and np.dtype(expected.dtype) != np.dtype(saved.dtype):
            raise ValueError(f"checkpoint dtype mismatch at {'/'.join(path)}")
        return saved

    merged = merge(target, source)
    if added and set(added) != {"/".join(path) for path in NEW_ROOTS}:
        raise ValueError("partial decision checkpoint; refusing to reinitialize a missing branch")
    return flax.serialization.from_state_dict(template, merged), added
