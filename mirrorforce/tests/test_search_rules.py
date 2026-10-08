"""Search's rule identity checks, including wrong-but-self-consistent asset stages."""
from dataclasses import replace
import hashlib
import json
import os

import pytest

from mirrorforce.netduel import search_rules as R


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def rules(tmp_path):
    root = tmp_path / "assets"
    script = root / "project/script"
    script.mkdir(parents=True)
    assets = {"project/script/c1.lua": b"script", "project/script/utility.lua": b"helper",
              "project/cards.cdb": b"cards", "project/code_list.txt": b"codes",
              "project/semantics.npz": b"semantics", "tables/cards.npz": b"tables",
              "project/announce.json": b"announce"}
    for name, data in assets.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {"schema": R.STAGE_SCHEMA, "copied_programs_executed": False,
                "files": [{"path": name, "mode": "100644", "sha256": sha(raw)} for name, raw in assets.items()]}
    raw = json.dumps(manifest).encode()
    (root / "STAGE.json").write_bytes(raw)
    for name in ("follower.so", "service.so"):
        (tmp_path / name).write_bytes(b"core")
    files = R.RuleFiles(tmp_path / "follower.so", tmp_path / "service.so", root / "STAGE.json", script,
                        root / "project/cards.cdb", root / "project/code_list.txt", root / "project/semantics.npz",
                        root / "tables/cards.npz", root / "project/announce.json")
    checkpoint = {"native_core": {"core_commit": "core-commit", "libmfcore_sha256": sha(b"core")},
                  "tables": {"cards_db_sha256": sha(b"cards"), "code_list_sha256": sha(b"codes"),
                             "semantic_file_sha256": sha(b"semantics")},
                  "card_tables": {"sha256": sha(b"tables")}, "announce": {"tables_file_sha256": sha(b"announce")}}
    service = {"protocol": R.PROTOCOL, "backend": "checkpoint", "weights": "iterate",
               "native_build": {"core_commit": "core-commit", "outputs": {"libmfcore.so": sha(b"core")}},
               **checkpoint["tables"], "card_tables_sha256": sha(b"tables"), "announce_tables_sha256": sha(b"announce")}
    return files, {"stage_sha256": sha(raw), "checkpoint": checkpoint, "service": service}


def test_verified_files_produce_an_explicit_non_training_receipt(rules):
    files, kwargs = rules
    receipt = R.verify_rules(files, **kwargs)
    assert receipt["schema"] == R.SCHEMA and receipt["training_eligible"] is False
    assert receipt["core_sha256"] == sha(b"core")
    assert receipt["scripts"]["files"] == 2
    names = {"project/script/c1.lua": sha(b"script"), "project/script/utility.lua": sha(b"helper")}
    assert receipt["scripts"]["file_map_sha256"] == sha(json.dumps(names, sort_keys=True, separators=(",", ":")).encode())


@pytest.mark.parametrize("asset", ["core", "service_core", "cards_db", "code_list", "semantics", "card_tables", "announce_tables"])
def test_changed_runtime_file_is_refused(rules, asset):
    files, kwargs = rules
    getattr(files, asset).write_bytes(b"different")
    with pytest.raises(R.RulesMismatch, match="SHA-256"):
        R.verify_rules(files, **kwargs)


@pytest.mark.parametrize("change", ["edit", "remove", "extra", "symlink"])
def test_script_changes_are_refused(rules, change):
    files, kwargs = rules
    script = files.scripts / "c1.lua"
    if change == "edit":
        script.write_bytes(b"changed")
    elif change == "remove":
        script.unlink()
    elif change == "extra":
        (files.scripts / "extra.lua").write_bytes(b"extra")
    else:
        script.unlink()
        script.symlink_to(files.scripts / "utility.lua")
    with pytest.raises(R.RulesMismatch, match="script|escapes"):
        R.verify_rules(files, **kwargs)


def test_edited_manifest_cannot_bless_changed_scripts(rules):
    files, kwargs = rules
    (files.scripts / "c1.lua").write_bytes(b"new")
    stage = json.loads(files.stage.read_bytes())
    stage["files"][0]["sha256"] = sha(b"new")
    files.stage.write_text(json.dumps(stage))
    with pytest.raises(R.RulesMismatch, match="STAGE.json"):
        R.verify_rules(files, **kwargs)


@pytest.mark.parametrize("bad", ["../escape", "/escape", "project//bad", "project/script/c1.lua"])
def test_unsafe_or_duplicate_manifest_names_are_refused_even_when_pinned(rules, bad):
    files, kwargs = rules
    stage = json.loads(files.stage.read_bytes())
    stage["files"].append({"path": bad, "mode": "100644", "sha256": sha(b"new")})
    raw = json.dumps(stage).encode()
    files.stage.write_bytes(raw)
    kwargs["stage_sha256"] = sha(raw)
    with pytest.raises(R.RulesMismatch, match="invalid or duplicate"):
        R.verify_rules(files, **kwargs)


def test_a_second_correct_file_outside_the_stage_is_not_the_registered_runtime_asset(rules, tmp_path):
    files, kwargs = rules
    outside = tmp_path / "another.cdb"
    outside.write_bytes(files.cards_db.read_bytes())
    with pytest.raises(R.RulesMismatch, match="outside"):
        R.verify_rules(replace(files, cards_db=outside), **kwargs)


@pytest.mark.parametrize("link", ["same", "hard", "symbolic"])
def test_core_paths_must_not_alias_one_mapping(rules, tmp_path, link):
    files, kwargs = rules
    target = tmp_path / "alias.so"
    if link == "same":
        target = files.service_core
    elif link == "hard":
        os.link(files.service_core, target)
    else:
        target.symlink_to(files.service_core)
    with pytest.raises(R.RulesMismatch, match="separate file"):
        R.verify_rules(replace(files, core=target), **kwargs)


@pytest.mark.parametrize("field", ["weights", "backend", "cards_db_sha256", "native_build"])
def test_service_identity_must_match(rules, field):
    files, kwargs = rules
    kwargs["service"][field] = {} if field == "native_build" else "wrong"
    with pytest.raises(R.RulesMismatch):
        R.verify_rules(files, **kwargs)
