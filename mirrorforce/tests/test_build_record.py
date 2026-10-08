"""The native loader's build record check (mirrorforce.agent.env.build_record): stdlib only, no built module."""
from __future__ import annotations

import hashlib
import json

import pytest

from mirrorforce.agent.env.build_record import core_identity, load_build_record

MODULE = "duel_native.cpython-311-x86_64-linux-gnu.so"


def build(tmp_path, **override):
    files = {MODULE: b"module", "libmfcore.so": b"core", "liblua5.3.so.0": b"lua"}
    for name, body in files.items():
        (tmp_path / name).write_bytes(body)
    record = {"core_commit": "d6de93c", "core_tree": "a578734c",
              "outputs": {name: hashlib.sha256(body).hexdigest() for name, body in files.items()}}
    record.update(override)
    (tmp_path / "BUILD.json").write_text(json.dumps(record))
    return tmp_path / MODULE


def test_a_complete_record_loads_and_names_the_core(tmp_path):
    record = load_build_record(build(tmp_path))
    assert core_identity(record) == {"core_commit": "d6de93c", "core_tree": "a578734c",
                                     "libmfcore_sha256": hashlib.sha256(b"core").hexdigest()}


def test_a_changed_core_library_is_refused(tmp_path):
    module = build(tmp_path)
    (tmp_path / "libmfcore.so").write_bytes(b"another core")
    with pytest.raises(ImportError, match="libmfcore.so differs"):
        load_build_record(module)


def test_a_missing_record_or_core_entry_is_refused(tmp_path):
    module = build(tmp_path)
    (tmp_path / "BUILD.json").unlink()
    with pytest.raises(ImportError, match="no BUILD.json"):
        load_build_record(module)
    module = build(tmp_path, outputs={MODULE: hashlib.sha256(b"module").hexdigest()})
    with pytest.raises(ImportError, match="does not list"):
        load_build_record(module)
