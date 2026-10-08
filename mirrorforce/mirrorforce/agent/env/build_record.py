"""The build record (``BUILD.json``) of a built ``duel_native`` module, checked against the files beside it.

``cxx/duelpool/build.py`` writes the record next to its outputs: the module, the core library ``libmfcore.so`` and Lua,
each with its SHA-256, plus the source and core commits. A loader refuses a module whose record is missing, does not
list the module and the core library, or whose listed files differ from their digests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_build_record(module_path) -> dict:
    record_path = Path(module_path).with_name("BUILD.json")
    if not record_path.is_file():
        raise ImportError(f"no BUILD.json next to {module_path}")
    record = json.loads(record_path.read_text())
    outputs = record.get("outputs", {})
    if Path(module_path).name not in outputs or "libmfcore.so" not in outputs or "core_commit" not in record:
        raise ImportError(f"{record_path} does not list the module, libmfcore.so and the core commit")
    for name, digest in outputs.items():
        if hashlib.sha256((record_path.parent / name).read_bytes()).hexdigest() != digest:
            raise ImportError(f"{name} differs from its digest in {record_path}")
    return record


def core_identity(record: dict) -> dict:
    """The part of the record a checkpoint identity carries: which core the environment ran."""
    return {"core_commit": record["core_commit"], "core_tree": record.get("core_tree"),
            "libmfcore_sha256": record["outputs"]["libmfcore.so"]}
