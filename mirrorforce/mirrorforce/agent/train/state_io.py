"""Training-state checkpoints for the trainer: parameters, Adam state, learner keys and counters.

The payload is flax msgpack of ``{"state": TrainState without constants, "learner_keys", "counters"}``; the
frozen semantic constants are not stored (they are rebuilt from the semantic file, whose digest the receipt
names). Files and receipts follow ``checkpoint_store`` (the design notes). With ``actors`` the
payload also holds every actor thread's state at the top of the next rollout (its environments' exported states,
observations, recurrent states, key and counters), so a resume continues the same games; the receipt says which.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import flax
import jax
import numpy as np

from mirrorforce.agent.train import checkpoint_store

SCHEMA = "mirrorforce_training_checkpoint/v1"
COUNTERS = ("global_step", "learner_update")


def file_digest(path) -> str:
    return checkpoint_store.file_sha256(path)


def tree_digest(directory, pattern="*") -> str:
    """A file's SHA-256, or for a directory the SHA-256 over 'relative path NUL file sha256 newline' of the files matching ``pattern``, in path order."""
    root = Path(directory)
    if root.is_file():
        return file_digest(root)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob(pattern) if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + file_digest(path).encode() + b"\n")
    return digest.hexdigest()


def save(directory, state, learner_keys, counters, receipt_fields, actors=None, ema=None, critic=None) -> str:
    """Write one checkpoint from an unreplicated ``state`` (and the actor threads' snapshots); returns the payload
    SHA-256."""
    if set(counters) != set(COUNTERS):
        raise ValueError("counters must be exactly " + ", ".join(COUNTERS))
    target = {"state": flax.serialization.to_state_dict(jax.device_get(state.replace(constants={}))),
              "learner_keys": np.asarray(jax.device_get(learner_keys)),
              "counters": {name: int(counters[name]) for name in COUNTERS}}
    if actors is not None:
        target["actors"] = {str(i): actor for i, actor in enumerate(actors)}
    if ema is not None:  # arm A: the EMA of the parameters
        target["ema"] = flax.serialization.to_state_dict(jax.device_get(ema))
    if critic is not None:  # A0's central critic (its own parameters and Adam state)
        target["critic"] = flax.serialization.to_state_dict(jax.device_get(critic.replace(constants={})))
    payload = flax.serialization.msgpack_serialize(target)
    fields = {**receipt_fields, "schema": SCHEMA, "counters": target["counters"],
              "env_state_restored_on_resume": actors is not None, "ema": ema is not None, "critic_state": critic is not None}
    return checkpoint_store.write_checkpoint(directory, payload, fields)


def restore(path, template_state, learner_key_count, expected, *, add_value_menu=False):
    """State, learner keys, counters and receipt from checkpoint ``path`` (``<dir>/<sha256>.ckpt``).

    ``checkpoint_store.verify_resume`` checks the payload against its name and receipt, this host's arrival
    receipt and every ``expected`` receipt value before anything is deserialized."""
    path = Path(path)
    if path.suffix != ".ckpt":
        raise ValueError("resume from a <sha256>.ckpt payload")
    checked = {k: v for k, v in expected.items() if not add_value_menu or k not in ("model", "critic")}
    receipt = checkpoint_store.verify_resume(path.parent, path.stem, expected={**checked, "schema": SCHEMA})
    raw = flax.serialization.msgpack_restore(path.read_bytes())
    if add_value_menu:
        from . import value_menu_migration as migration
        migration.check_identity(receipt, expected)
        if "critic" in expected and (not isinstance(raw.get("critic"), dict)
                                     or not isinstance(raw["critic"].get("params"), dict)):
            raise ValueError("value-menu migration requires the existing critic payload")
        if "ema" in raw:
            # EMA must describe exactly the old parameter tree, not a separately/partially migrated model.
            migration.extend_tree(raw["ema"], raw["state"]["params"], ())
        raw["state"] = migration.extend_state(raw["state"], template_state)
        if "ema" in raw:
            raw["ema"] = migration.extend_ema(raw["ema"], template_state.params)
    state = flax.serialization.from_state_dict(template_state.replace(constants={}), raw["state"])
    keys = np.asarray(raw["learner_keys"])
    if learner_key_count is not None and keys.shape != (learner_key_count, 2):  # None: a declared layout change
        raise ValueError("the checkpoint's learner keys do not match the learner devices")
    if {name: int(raw["counters"][name]) for name in COUNTERS} != receipt["counters"]:
        raise ValueError("the payload counters differ from the receipt")
    actors = None
    if "actors" in raw:
        actors = [raw["actors"][str(i)] for i in range(len(raw["actors"]))]
        if not receipt.get("env_state_restored_on_resume"):
            raise ValueError("the payload holds actor states its receipt does not declare")
    ema = None
    if "ema" in raw:
        ema = flax.serialization.from_state_dict(template_state.params, raw["ema"])
    return state.replace(constants=template_state.constants), keys, receipt["counters"], receipt, actors, ema


def restore_extra(path, key, template, *, add_value_menu=False):
    """A further part of a checkpoint payload (A0's critic state), or None when the payload has none; the payload is
    verified by ``restore`` first in the trainer's resume path."""
    raw = flax.serialization.msgpack_restore(Path(path).read_bytes())
    if add_value_menu and key in raw:
        if key != "critic":
            raise ValueError("only the central critic supports the value-menu extra-state migration")
        from . import value_menu_migration as migration
        raw[key] = migration.extend_state(raw[key], template, critic=True)
    return flax.serialization.from_state_dict(template, raw[key]) if key in raw else None
