"""Canonical-JSON, digest and immutable-publication primitives for sidecars.

Two sidecar families now share these rules: the candidate-only offline search
targets in :mod:`mirrorforce.common.search_target_sidecar` and the training-eligible
self-play targets in :mod:`mirrorforce.common.selfplay_search_sidecar`.  The
content-addressed naming, the ``O_NOFOLLOW`` read, the hard-link publish and
the strict field-set check are load-bearing security properties; they are
written once here rather than copied.

Every function takes the caller's own error class so each sidecar keeps a
single, specific exception type in its public contract.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "GIT_OBJECT_PATTERN",
    "SHA256_PATTERN",
    "SidecarIOError",
    "assert_directory",
    "canonical",
    "digest",
    "exact_fields",
    "finite",
    "fsync_directory",
    "git",
    "load_canonical_document",
    "ordered_unique_strings",
    "plain_nonnegative",
    "prepare_output",
    "publish_immutable",
    "read_regular",
    "require_sha256",
    "safe_relative",
    "sha256_file",
    "sha256_tree",
]

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class SidecarIOError(RuntimeError):
    """Default error for a malformed or unsafe sidecar artifact."""


def canonical(value: Any, *, newline: bool = False, error=SidecarIOError) -> bytes:
    """Encode ``value`` as the one byte string this family calls canonical."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise error(f"value is not canonical JSON: {exc}") from exc
    return encoded + (b"\n" if newline else b"")


def digest(value: bytes | Any, *, error=SidecarIOError) -> str:
    """SHA-256 of raw bytes, or of ``value``'s canonical encoding."""

    encoded = value if isinstance(value, bytes) else canonical(value, error=error)
    return hashlib.sha256(encoded).hexdigest()


_SHARED_READS: contextvars.ContextVar = contextvars.ContextVar("shared_json_reads", default=None)


@contextmanager
def shared_json_reads(*, error=SidecarIOError):
    """Parse each verified canonical JSON artifact once inside one scope.

    For a caller whose checks read the same content-addressed evidence
    several times, such as the native verifier checking one game. Every
    reader still verifies the bytes it reads; only equal bytes share one
    parsed value, whose digest is the checksum of those bytes. Readers must
    not change it: on exit each shared value is encoded once more and must
    hash to the bytes it came from, or the whole scope fails.
    """
    memo: dict = {"parsed": {}, "by_id": {}}
    token = _SHARED_READS.set(memo)
    try:
        yield
    finally:
        _SHARED_READS.reset(token)
    for checksum, value in memo["parsed"].items():
        if digest(value, error=error) != checksum:
            raise error("a shared JSON artifact was not canonical or was changed by a reader")


def parse_verified_json(raw: bytes, checksum: str) -> Any:
    """``json.loads`` of bytes already verified against ``checksum``, shared in a scope."""
    memo = _SHARED_READS.get()
    if memo is None:
        return json.loads(raw)
    if checksum not in memo["parsed"]:
        value = memo["parsed"][checksum] = json.loads(raw)
        memo["by_id"][id(value)] = (value, checksum)
    return memo["parsed"][checksum]


def value_digest(value: Any, *, error=SidecarIOError) -> str:
    """``digest(value)``; a value shared by :func:`shared_json_reads` is not encoded again."""
    memo = _SHARED_READS.get()
    row = None if memo is None else memo["by_id"].get(id(value))
    if row is not None and row[0] is value:
        return row[1]  # Proven when the scope ends.
    return digest(value, error=error)


def require_sha256(value: Any, where: str, *, error=SidecarIOError) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise error(f"{where} must be a lowercase SHA-256")
    return value


def exact_fields(
    value: Any, keys: set[str], where: str, *, error=SidecarIOError
) -> Mapping[str, Any]:
    """Require an object whose key set is exactly ``keys``.

    A schema that only checks required keys silently accepts an added field,
    which is how a firewall flag gets bypassed.
    """

    if not isinstance(value, Mapping):
        raise error(f"{where} must be an object")
    missing, extra = keys - set(value), set(value) - keys
    if missing or extra:
        raise error(
            f"{where} fields changed; missing={sorted(missing)}, "
            f"unknown={sorted(extra)}"
        )
    return value


def plain_nonnegative(
    value: Any, where: str, *, positive: bool = False, error=SidecarIOError
) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise error(f"{where} must be a {qualifier} integer")
    return value


def finite(value: Any, where: str, *, error=SidecarIOError) -> float:
    if type(value) not in (int, float):
        raise error(f"{where} must be a plain real number")
    result = float(value)
    if not math.isfinite(result):
        raise error(f"{where} must be finite")
    return 0.0 if result == 0 else result


def safe_relative(value: Any, where: str, *, error=SidecarIOError) -> Path:
    if type(value) is not str:
        raise error(f"{where} must be a relative path")
    relative = Path(value)
    if (
        not value
        or relative.is_absolute()
        or ".." in relative.parts
        or value != relative.as_posix()
        or any(part in ("", ".") for part in relative.parts)
    ):
        raise error(f"{where} is not a canonical relative path")
    return relative


def assert_directory(path: Path, *, error=SidecarIOError) -> None:
    if path.is_symlink() or not path.is_dir():
        raise error(f"unsafe output directory {path}")


def prepare_output(root: Path, *, error=SidecarIOError) -> tuple[Path, Path]:
    """Create and validate the ``shards``/``manifests`` layout under ``root``."""

    try:
        root.mkdir(parents=True, exist_ok=True)
        shards = root / "shards"
        manifests = root / "manifests"
        shards.mkdir(exist_ok=True)
        manifests.mkdir(exist_ok=True)
    except OSError as exc:
        raise error("cannot create sidecar output directories") from exc
    for path in (root, shards, manifests):
        assert_directory(path, error=error)
    return shards, manifests


def read_regular(path: Path, where: str, *, error=SidecarIOError) -> bytes:
    """Read a regular file without following a symlink or racing a rewrite."""

    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise error(f"{where} is not a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        identity = lambda row: (
            row.st_dev,
            row.st_ino,
            row.st_mode,
            row.st_nlink,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )
        data = b"".join(chunks)
        if (
            identity(before) != identity(after)
            or identity(after) != identity(current)
            or len(data) != before.st_size
        ):
            raise error(f"{where} changed while being read")
        return data
    except error:
        raise
    except OSError as exc:
        raise error(f"cannot read {where}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_immutable(path: Path, encoded: bytes, *, error=SidecarIOError) -> None:
    """Link content-addressed bytes into place, or prove they already match."""

    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short sidecar write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if read_regular(path, "existing immutable sidecar", error=error) != encoded:
                raise error("refusing to overwrite a different immutable sidecar")
        fsync_directory(path.parent)
    except error:
        raise
    except OSError as exc:
        raise error("cannot publish immutable sidecar") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def load_canonical_document(
    path: Path, where: str, *, error=SidecarIOError
) -> tuple[Mapping[str, Any], bytes]:
    """Read a JSON object and prove its bytes are the canonical encoding."""

    encoded = read_regular(path, where, error=error)
    try:
        value = json.loads(encoded.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise error(f"cannot parse {where}") from exc
    if not isinstance(value, Mapping) or canonical(
        value, newline=True, error=error
    ) != encoded:
        raise error(f"{where} is not canonical JSON")
    return value, encoded


def git(root: Path, *args: str, error=SidecarIOError) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise error("cannot collect source git identity") from exc


def sha256_file(path: Path, *, error=SidecarIOError) -> str:
    value = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                value.update(chunk)
    except OSError as exc:
        raise error(f"cannot hash file {path}") from exc
    return value.hexdigest()


def sha256_tree(root: Path, *, error=SidecarIOError) -> str:
    """Hash a directory tree by relative path and bytes, ignoring ``.git``.

    Byte-identical to ``winbot_capture._sha256_tree`` so a self-play engine
    identity and a capture ledger name the same script tree the same way.
    """

    root = Path(root).resolve()
    if not root.is_dir():
        raise error(f"script root is not a directory: {root}")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )
    if not files:
        raise error(f"script root is empty: {root}")
    value = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        value.update(len(relative).to_bytes(4, "little"))
        value.update(relative)
        value.update(bytes.fromhex(sha256_file(path, error=error)))
    return value.hexdigest()


def ordered_unique_strings(
    values: Sequence[str], where: str, *, limit: int, error=SidecarIOError
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise error(f"{where} must be a sequence")
    rows = tuple(values)
    if not rows or len(rows) > limit:
        raise error(f"{where} count is invalid")
    if any(type(row) is not str or not row or len(row) > 8192 for row in rows):
        raise error(f"{where} contains an invalid entry")
    if len(set(rows)) != len(rows):
        raise error(f"{where} contains duplicates")
    return rows
