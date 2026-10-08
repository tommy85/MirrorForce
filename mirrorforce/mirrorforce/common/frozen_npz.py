"""Byte-deterministic NPZ writing for frozen, content-addressed artifacts.

``numpy.savez`` stamps the wall-clock time into every zip entry, so the same
arrays written twice give different bytes and different SHA-256s.  This writer
fixes the entry order, timestamps, permissions and compression level: the same
arrays always produce the same file, which ``numpy.load`` reads as usual.
"""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["atomic_write", "npz_bytes", "write_npz"]


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    # np.ascontiguousarray would turn a 0-d array (a JSON ``meta``) into shape (1,).
    np.lib.format.write_array(buffer, np.array(array, order="C"), allow_pickle=False)
    return buffer.getvalue()


def npz_bytes(arrays: Sequence[tuple[str, np.ndarray]]) -> bytes:
    """The NPZ bytes of ``(name, array)`` pairs, in the given order."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, array in arrays:
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, _npy_bytes(array), compresslevel=6)
    return buffer.getvalue()


def atomic_write(path: str | Path, data: bytes) -> str:
    """Write ``data`` through a temporary file and return its SHA-256."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, target)
    return hashlib.sha256(data).hexdigest()


def write_npz(path: str | Path, arrays: Sequence[tuple[str, np.ndarray]]) -> str:
    """Write a deterministic NPZ atomically and return its SHA-256."""
    return atomic_write(path, npz_bytes(arrays))
