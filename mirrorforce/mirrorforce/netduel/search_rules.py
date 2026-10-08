"""Verify a play-time search engine against its registered checkpoint and asset stage.

This check runs before loading the follower core. The independently registered
STAGE digest pins the entire script list; neither an edited manifest nor extra
scripts may silently change the rules. The follower uses a separate core file
with the same bytes as the service's core. No engine, torch or JAX is imported.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Mapping

from .agent_wire import PROTOCOL
from .agent_core_upgrade import load_core_upgrade as _load_core_upgrade

SCHEMA = "mirrorforce_search_rules/v1"
STAGE_SCHEMA = "mirrorforce_reference_stage/v1"
INFORMATION_SET_SEARCH = True


class RulesMismatch(ValueError):
    """Search's actual files do not match the registered training rules."""


def load_core_upgrade(path, expected_sha256):
    """Public convenience entry alongside verify_rules; preserves the exact evidence bytes."""
    return _load_core_upgrade(path, expected_sha256)


@dataclass(frozen=True)
class RuleFiles:
    """Actual runtime paths, not a second set of files checked but never used."""

    core: Path
    service_core: Path
    stage: Path
    scripts: Path
    cards_db: Path
    code_list: Path
    semantics: Path
    card_tables: Path
    announce_tables: Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _match(actual: str, expected: object, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or actual != expected:
        raise RulesMismatch(f"{label} differs from its registered SHA-256")


def _relative(path: Path, root: Path) -> str:
    # Reject symlinks, including parent links: the measured tree is the tree loaded.
    path = path.absolute()
    try:
        relative = path.relative_to(root.absolute())
    except ValueError as exc:
        raise RulesMismatch(f"asset outside the registered stage: {path}") from exc
    current = root.absolute()
    if current.is_symlink():
        raise RulesMismatch(f"asset stage root is a symlink: {root}")
    for part in relative.parts:
        current /= part
        if part == ".." or current.is_symlink():
            raise RulesMismatch(f"asset escapes the registered stage: {path}")
    return relative.as_posix()


def _manifest(path: Path, expected: str) -> dict[str, str]:
    if path.is_symlink():
        raise RulesMismatch("STAGE.json must not be a symlink")
    raw = path.read_bytes()
    _match(hashlib.sha256(raw).hexdigest(), expected, "STAGE.json")
    stage = json.loads(raw)
    if stage.get("schema") != STAGE_SCHEMA or stage.get("copied_programs_executed") is not False:
        raise RulesMismatch("unknown asset stage schema")
    records = {}
    for row in stage["files"]:
        name = row["path"]
        relative = PurePosixPath(name)
        if (not name or name != relative.as_posix() or relative.is_absolute()
                or any(part in (".", "..", "") for part in name.split("/"))
                or "\\" in name or "\0" in name or name in records or row["mode"] not in ("100644", "100755")):
            raise RulesMismatch(f"invalid or duplicate staged file: {name!r}")
        records[name] = row["sha256"]
    return records


def _scripts(directory: Path, root: Path, records: Mapping[str, str]) -> dict:
    prefix = _relative(directory, root) + "/"
    expected = {name: sha for name, sha in records.items() if name.startswith(prefix)}
    if not expected or not directory.is_dir():
        raise RulesMismatch("the stage has no scripts at the runtime script path")
    actual = {}
    for path in sorted(directory.rglob("*")):
        name = _relative(path, root)
        if path.is_file():
            actual[name] = _sha(path)
        elif not path.is_dir():
            raise RulesMismatch(f"non-regular script asset: {path}")
    if actual != expected:
        changed = sorted(name for name in actual.keys() | expected.keys() if actual.get(name) != expected.get(name))
        raise RulesMismatch(f"script tree differs from STAGE.json: {changed[:5]}")
    body = json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()
    return {"files": len(actual), "file_map_sha256": hashlib.sha256(body).hexdigest()}


def verify_rules(files: RuleFiles, *, stage_sha256: str, checkpoint: Mapping, service: Mapping, core_upgrade=None) -> dict:
    """Verify rules before constructing a follower; return a content-addressable receipt.

    ``checkpoint`` is the identity from a verified checkpoint receipt (not model
    arrays). ``service`` is the registered live policy-service identity. The stage
    digest comes from the run configuration, not from the manifest being checked.
    This checks asset identity, not follower coverage or rollout correctness.
    """
    if service.get("protocol") != PROTOCOL:
        raise RulesMismatch("search requires a registered checkpoint service")
    from .agent_specialization_search import registered_service
    try:
        specialization=registered_service(service)
    except ValueError as exc:
        raise RulesMismatch("search actor binding refused: "+str(exc)) from exc
    if service.get("weights") != "iterate":
        raise RulesMismatch("play-time search must use the registered raw iterate weights")
    if files.core.samefile(files.service_core):
        raise RulesMismatch("the follower core must be a separate file, not a link to the service core")
    native = checkpoint.get("native_core", {})
    core_sha = _sha(files.core)
    _match(_sha(files.service_core), core_sha, "service core")
    build = service.get("native_build", {})
    _match(core_sha, build.get("outputs", {}).get("libmfcore.so"), "service build core")
    if core_upgrade is None:
        _match(core_sha, native.get("libmfcore_sha256"), "follower core")
        if not native.get("core_commit") or native.get("core_commit") != build.get("core_commit"):
            raise RulesMismatch("service core commit differs from the checkpoint")
    records = _manifest(files.stage, stage_sha256)
    root = files.stage.parent
    scripts = _scripts(files.scripts, root, records)
    tables = checkpoint.get("tables", {})
    expected = {
        "cards_db": tables.get("cards_db_sha256"), "code_list": tables.get("code_list_sha256"),
        "semantics": tables.get("semantic_file_sha256"),
        "card_tables": checkpoint.get("card_tables", {}).get("sha256"),
        "announce_tables": checkpoint.get("announce", {}).get("tables_file_sha256"),
    }
    service_keys = {"semantics": "semantic_file_sha256"}
    hashes = {}
    for key, sha in expected.items():
        path = getattr(files, key)
        name = _relative(path, root)
        actual = _sha(path)
        _match(actual, sha, key)
        _match(actual, service.get(service_keys.get(key, key + "_sha256")), "service " + key)
        _match(actual, records.get(name), "staged " + key)
        hashes[key] = actual
    receipt = {"schema": SCHEMA, "training_eligible": False, "core_sha256": core_sha,
               "core_commit": build["core_commit"], "stage_sha256": stage_sha256,
               "scripts": scripts, "assets": hashes}
    if specialization is not None:
        receipt['specialization_search']=specialization
    if core_upgrade is not None:
        from .agent_core_upgrade import check_core_upgrade
        try:
            receipt["core_upgrade"] = check_core_upgrade(core_upgrade, checkpoint=checkpoint, service=service,
                stage_sha256=stage_sha256, scripts=scripts, assets=hashes)
        except (ValueError, KeyError, TypeError) as exc:
            raise RulesMismatch("explicit serving-core upgrade refused: " + str(exc)) from exc
    return receipt
