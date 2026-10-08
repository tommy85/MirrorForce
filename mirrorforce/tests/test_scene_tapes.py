"""Registered specialization scenes reproduce their original public tapes; unregistered changes fail."""
import hashlib
import json
from pathlib import Path

import pytest

from tools import mf_runtime_scene_tapes as T

REGISTRY = Path(__file__).resolve().parents[1] / "data/specialization/scenes.json"
PRIVILEGED_TARGET = True


def registry():
    return json.loads(REGISTRY.read_bytes())


def test_registry_lists_every_family_with_disjoint_heldout_seeds():
    scenes = registry()["scenes"]
    assert registry()["schema"] == T.SCHEMA and len(scenes) == 96
    assert {(s["family"], s.get("variant")) for s in scenes} == T.FAMILIES
    for family in T.FAMILIES:
        splits = {s["split"] for s in scenes if (s["family"], s.get("variant")) == family}
        assert splits == {"train", "heldout"}
    keys = [(s["family"], s.get("variant"), s["seed"]) for s in scenes]
    assert len(keys) == len(set(keys)) and all(len(s["tape_sha256"]) == 64 for s in scenes)


def one_per_family():
    seen, chosen = set(), []
    for index, scene in enumerate(registry()["scenes"]):
        key = (scene["family"], scene.get("variant"), bool(scene.get("finish_turn")))
        if key not in seen:
            seen.add(key)
            chosen.append(index)
    return chosen


def test_one_scene_of_every_family_reproduces_its_tape(tmp_path):
    pytest.importorskip("mirrorforce.effectinfo")
    assets = tmp_path / "assets.json"
    assets.write_text(json.dumps({"checkpoint": None}))
    chosen = one_per_family()
    try:
        assert T.main(["--registry", str(REGISTRY), "--assets", str(assets), "--out", str(tmp_path / "out"),
                       "--only", *map(str, chosen)]) == 0
    except OSError as exc:  # no rules-engine library in this environment
        pytest.skip(str(exc))
    config = json.loads((tmp_path / "out/export-config.json").read_bytes())
    assert len(config["tapes"]) == len(chosen) and config["assets"] == {"checkpoint": None}
    for tape in config["tapes"]:
        assert hashlib.sha256(Path(tape["path"]).read_bytes()).hexdigest() == tape["sha256"]
        assert "tape_sha256" not in tape and "decisions" not in tape


def test_a_changed_scene_fails_instead_of_being_redrawn(tmp_path):
    pytest.importorskip("mirrorforce.effectinfo")
    changed = registry()
    index = one_per_family()[0]
    changed["scenes"][index]["seed"] += 1
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(changed))
    assets = tmp_path / "assets.json"
    assets.write_text("{}")
    try:
        with pytest.raises(ValueError, match="registered"):
            T.main(["--registry", str(path), "--assets", str(assets), "--out", str(tmp_path / "out"),
                    "--only", str(index)])
    except OSError as exc:
        pytest.skip(str(exc))
    with pytest.raises(ValueError, match="unregistered scene family"):
        T.demonstrate(None, None, {"family": "sword", "variant": "other", "start_lp": 8000, "seed": 1})
