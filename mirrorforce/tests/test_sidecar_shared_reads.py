"""One scope may share a parsed canonical artifact among readers that only read it."""
import pytest

from mirrorforce.common import sidecar_io
from mirrorforce.common.sidecar_io import (SidecarIOError, canonical, digest, parse_verified_json, shared_json_reads,
    value_digest)


def test_readers_in_one_scope_share_one_parse_and_its_checksum(monkeypatch):
    value = {"reports": [{"row": index, "p": index / 7} for index in range(50)], "schema": "x"}
    raw, other = canonical(value), digest({"other": 1})
    encoded = []
    original = sidecar_io.canonical
    monkeypatch.setattr(sidecar_io, "canonical", lambda *args, **kwargs: encoded.append(1) or original(*args, **kwargs))
    with shared_json_reads():
        first, second = parse_verified_json(raw, digest(raw)), parse_verified_json(raw, digest(raw))
        assert first is second and first == value
        assert value_digest(first) == digest(raw) and value_digest(second) == digest(raw)
        assert encoded == []  # The digest is the verified checksum, not another encoding.
        assert value_digest({"other": 1}) == other and len(encoded) == 1  # Unshared values are encoded.
    assert len(encoded) == 2  # The one proof on exit.


def test_outside_a_scope_every_reader_gets_its_own_parse():
    raw = canonical({"a": [1, 2]})
    first, second = parse_verified_json(raw, digest(raw)), parse_verified_json(raw, digest(raw))
    assert first == second and first is not second
    first["a"].append(3)
    assert value_digest(first) == digest({"a": [1, 2, 3]})


def test_a_reader_that_changes_a_shared_value_fails_the_whole_scope():
    raw = canonical({"a": [1, 2]})
    with pytest.raises(ValueError, match="changed by a reader"):
        with shared_json_reads(error=ValueError):
            parse_verified_json(raw, digest(raw))["a"].append(3)
            # A later reader sees the change, but the scope cannot end cleanly.
            assert parse_verified_json(raw, digest(raw))["a"] == [1, 2, 3]


def test_only_canonical_bytes_can_be_shared():
    raw = b'{"b": 1, "a": 2}'
    with pytest.raises(SidecarIOError, match="not canonical"):
        with shared_json_reads():
            assert value_digest(parse_verified_json(raw, digest(raw))) == digest(raw)


def test_a_failure_inside_the_scope_is_not_replaced_by_the_exit_proof():
    raw = b'{"b": 1, "a": 2}'
    with pytest.raises(KeyError):
        with shared_json_reads():
            parse_verified_json(raw, digest(raw))
            raise KeyError("original failure")
    # The scope is closed again: readers no longer share.
    assert parse_verified_json(raw, digest(raw)) is not parse_verified_json(raw, digest(raw))
