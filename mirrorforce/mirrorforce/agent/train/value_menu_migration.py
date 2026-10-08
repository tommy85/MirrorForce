"""One declared, additive legacy -> value-menu migration; no optimizer/EMA reset or permissive tree merge."""
from __future__ import annotations

import copy

import flax
import numpy as np

ACTOR_ROOTS = (("readout", "value_menu"),)
CRITIC_ROOTS = (("menu_tokens",), ("value_menu",))
LAW = "zero_output_value_menu/v1"


def legacy_config(config):
    out = dict(config)
    out.pop("value_menu", None)
    return out


def check_identity(receipt, expected):
    """Only false/absent -> true is allowed; every other architecture and critic field stays exact."""
    target, old = copy.deepcopy(expected["model"]), copy.deepcopy(receipt.get("model", {}))
    if target.get("architecture") != "policy_net" or target.get("args", {}).get("value_menu") is not True:
        raise ValueError("value-menu migration needs a policy_net target with value_menu enabled")
    if old.get("args", {}).get("value_menu", False) is not False:
        raise ValueError("value-menu migration requires a legacy source")
    target["args"], old["args"] = legacy_config(target["args"]), legacy_config(old.get("args", {}))
    if target != old:
        raise ValueError("value-menu migration changes another model field")
    if ("critic" in receipt) != ("critic" in expected):
        raise ValueError("value-menu migration cannot add, remove or reset the critic")
    if "critic" in expected:
        target, old = copy.deepcopy(expected["critic"]), copy.deepcopy(receipt["critic"])
        if target["model"].get("value_menu") is not True or old["model"].get("value_menu", False) is not False:
            raise ValueError("value-menu migration needs legacy and enabled critic configs")
        target["model"], old["model"] = legacy_config(target["model"]), legacy_config(old["model"])
        if target != old:
            raise ValueError("value-menu migration changes another critic field")


def _get(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def extend_tree(saved, fresh, roots, path=()):
    """Preserve every old leaf exactly; fill ONLY whole named additions from a fresh matching template.

    The same named paths recur in Adam mu/nu and masked optimizer partitions. Scalar counts, hyperparameters,
    existing moments, parameter shape and dtype may not change. Unexpected or partially present modules fail.
    """
    if isinstance(fresh, dict):
        if not isinstance(saved, dict) or set(saved) - set(fresh):
            raise ValueError(f"value-menu tree mismatch at {path}")
        out = {}
        for key, value in fresh.items():
            child = path + (key,)
            if key in saved:
                out[key] = extend_tree(saved[key], value, roots, child)
            elif any(child[-len(root):] == root for root in roots):
                out[key] = copy.deepcopy(value)
            else:
                raise ValueError(f"value-menu migration missing old leaf at {child}")
        return out
    a, b = np.asarray(saved), np.asarray(fresh)
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError(f"value-menu leaf shape/dtype mismatch at {path}")
    return saved


def extend_state(raw, template, *, critic=False):
    fresh = flax.serialization.to_state_dict(template.replace(constants={}))
    roots = CRITIC_ROOTS if critic else ACTOR_ROOTS
    if int(np.asarray(fresh["step"])) != 0:
        raise ValueError("value-menu migration requires a fresh optimizer template")
    if type(fresh["step"]) is int:
        # TrainState.create uses weakly typed Python 0, whereas a real jitted
        # update persists an int32 array. Match that placeholder to the saved
        # scalar representation; the saved counter itself is never cast/reset.
        saved_step = np.asarray(raw.get("step"))
        if saved_step.shape != () or saved_step.dtype not in (np.dtype("int32"), np.dtype("int64")) \
                or int(saved_step) < 0:
            raise ValueError("value-menu migration needs a nonnegative signed integer step scalar")
        fresh["step"] = np.zeros((), dtype=saved_step.dtype)
    for root in roots:
        try:
            _get(raw["params"], root)
        except KeyError:
            pass
        else:
            raise ValueError("value-menu migration found an existing new module")
        _get(fresh["params"], root)
    projection = _get(fresh["params"], ("value_menu",) if critic else ACTOR_ROOTS[0])["out"]["kernel"]
    if np.asarray(projection).any():
        raise ValueError("value-menu residual projection must initialize to zero")
    return extend_tree(raw, fresh, roots)


def extend_ema(raw, template):
    # Virtual constant history for new parameters: old EMA leaves remain untouched, and the new projection is zero.
    fresh = flax.serialization.to_state_dict(template)
    return extend_tree(raw, fresh, ACTOR_ROOTS)
