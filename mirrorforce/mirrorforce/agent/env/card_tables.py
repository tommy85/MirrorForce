"""Static per-card tables of a run beside its frozen semantics: B1's card extras, deck-library genericity and
effect-unit evidence, row-aligned with the run's code list (row 0 the unknown card, row i the code on line i, as
``init_module`` numbers card ids). Facts only, no learned weights; the model reads them as constants.

Arrays (``mirrorforce_card_tables/v1``):

* ``setcode_bag`` [N, 32] float32 and ``link_arrows`` [N, 8] float32: the exact card channel (``common/card_exact.py``):
  the signed-hash setcode bag and the eight printed link arrows, from the run's card database (an artwork variant
  takes its base card's row);
* ``genericity`` [N, 1] float32: the deck-type document frequency of the card in the run's registered announce
  library (the share of deck types with at least one recipe holding it, main or extra) -- the statistic the announce
  law's staple table counts;
* ``effect_unit_bits`` [N, 14] uint16: for each printed description ordinal, the Lua effect units of the frozen
  semantics that provably carry it (the ``literal_own_setdescription_every_use_accounted/v1``, through the same unit
  split the semantics were built with); 0 means unproven and the model pools every unit. ``effect_unit_reasons``
  [N, 14] uint8 keeps why, for audits.

The file is ``card-tables-<sha256>.npz`` (byte-deterministic, named by its bytes); it binds the code list, the card
database, the semantics file (with its script manifest recomputed), the builder and the announce tables by digest.
``tools/mf_runtime_card_tables.py`` builds it (from the exact card channel and the evidence law, outside the tree); this module reads, identifies and injects it, and ``load`` refuses a renamed, edited or mismatched file.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

SCHEMA = "mirrorforce_card_tables/v1"
NAME = re.compile(r"card-tables-([0-9a-f]{64})\.npz")
ORDINALS = 14       # actions.cc UnpackDesc rejects description ordinals from 14 on
MAX_UNITS = 16      # the frozen semantics' effect slots
LIBRARY_LAW = "deck_type_document_frequency_main_or_extra/v1"
EVIDENCE_LAW = "literal_own_setdescription_every_use_accounted/v1"
MODEL_ARRAYS = ("setcode_bag", "link_arrows", "genericity", "effect_unit_bits")
SHAPES = {"setcode_bag": (32, np.float32), "link_arrows": (8, np.float32), "genericity": (1, np.float32),
          "effect_unit_bits": (ORDINALS, np.uint16), "effect_unit_reasons": (ORDINALS, np.uint8)}


def read_code_list(path):
    """The codes of a code list in card-id order; every line must be ``<code> <has_script>`` (the env numbers ids by
    line, so a blank or malformed line would shift every later id)."""
    codes = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) < 2 or not fields[0].isdigit() or int(fields[0]) <= 0:
            raise ValueError(f"{path}:{number}: not a '<code> <has_script>' line")
        codes.append(int(fields[0]))
    if len(codes) != len(set(codes)):
        raise ValueError(f"{path}: duplicate codes")
    return codes


def code_list_digest(codes):
    """The digest the semantics metadata and cleanba use for a code list."""
    return hashlib.sha256("\n".join(str(code) for code in codes).encode()).hexdigest()


def encode(arrays, metadata):
    """The file's bytes and its name: a zip of .npy entries in a fixed order with fixed timestamps and permissions, so
    the same tables always give the same bytes (``numpy.savez`` stamps the wall clock)."""
    import zipfile

    text = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, array in [("metadata", np.asarray(text))] + [(name, arrays[name]) for name in SHAPES]:
            entry = io.BytesIO()
            np.lib.format.write_array(entry, np.array(array, order="C"), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, entry.getvalue(), compresslevel=6)
    raw = buffer.getvalue()
    return raw, f"card-tables-{hashlib.sha256(raw).hexdigest()}.npz"


def write(arrays, metadata, out_dir):
    raw, name = encode(arrays, metadata)
    path = Path(out_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"{path} exists with other bytes")
    else:
        path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def load(path, code_list):
    """``(model arrays, metadata, sha256)`` of a card tables file for a run's code list; refuses a file whose name is
    not its bytes' digest, another schema or code list, or arrays off their declared shapes and laws."""
    path = Path(path)
    match = NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"{path.name} is not a content-addressed card tables file (card-tables-<sha256>.npz)")
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    if sha256 != match.group(1):
        raise ValueError(f"{path.name}: the file's bytes do not match its name")
    with np.load(io.BytesIO(raw), allow_pickle=False) as payload:
        if set(payload.files) != {"metadata", *SHAPES}:
            raise ValueError(f"{path.name}: unexpected arrays {sorted(payload.files)}")
        metadata = json.loads(str(payload["metadata"].item()))
        arrays = {name: np.asarray(payload[name]) for name in SHAPES}
    codes = read_code_list(code_list)
    if metadata.get("schema") != SCHEMA or metadata.get("learned_weights") is not False:
        raise ValueError(f"{path.name}: not {SCHEMA}")
    if metadata.get("code_list_sha256") != code_list_digest(codes):
        raise ValueError(f"{path.name} was built for another code list")
    for name, (width, dtype) in SHAPES.items():
        a = arrays[name]
        if a.shape != (len(codes) + 1, width) or a.dtype != dtype or a[0].any() and name != "effect_unit_reasons":
            raise ValueError(f"{path.name}: {name} has shape {a.shape} {a.dtype}, or a nonzero unknown row")
    if not np.isfinite(arrays["setcode_bag"]).all() or not np.isin(arrays["link_arrows"], (0.0, 1.0)).all() \
            or not ((arrays["genericity"] >= 0) & (arrays["genericity"] <= 1)).all() \
            or ((arrays["effect_unit_bits"] != 0) != (arrays["effect_unit_reasons"] == 0)).any():
        raise ValueError(f"{path.name}: values outside their laws")
    return {name: arrays[name] for name in MODEL_ARRAYS}, metadata, sha256


def identity(metadata, sha256):
    """What a checkpoint records of the run's card tables."""
    return {"schema": SCHEMA, "sha256": sha256, "code_list_sha256": metadata["code_list_sha256"],
            "semantics_sha256": metadata["semantics"]["sha256"],
            "announce_tables_sha256": metadata["library"]["announce_tables_sha256"]}


def inject(tree, tables):
    """Write the card tables into the model's constants: the one collection that declares all of ``MODEL_ARRAYS``,
    each with the declared shape; a model that declares none (or several) is refused."""
    import jax.numpy as jnp

    matches = []

    def visit(node):
        if isinstance(node, dict):
            if set(MODEL_ARRAYS) <= set(node):
                matches.append(node)
            for value in node.values():
                visit(value)

    visit(tree)
    if len(matches) != 1:
        raise ValueError(f"expected one model constant collection with the card tables {MODEL_ARRAYS}, "
                         f"found {len(matches)}")
    for name in MODEL_ARRAYS:
        if tuple(matches[0][name].shape) != tables[name].shape:
            raise ValueError(f"card table {name}: the model declares {tuple(matches[0][name].shape)}, the file "
                             f"holds {tables[name].shape}")
        matches[0][name] = jnp.asarray(tables[name])
