"""Demonstration tapes for the registered tactical scenes of the targeted (specialization) training.

``--registry`` lists the scenes (``data/specialization/scenes.json``): each has a family, a seed, a split, a starting
LP and its family's own fields. Every scene is played once in a fresh engine with its family's scripted
demonstration, its registered outcome is checked, and its public tape is written next to the private result (host
layouts and choices, kept for audit only). ``export-config.json`` then lists the tapes, with the pinned ``--assets``,
for ``mf_runtime_specialize export``. A scene that misses its outcome, or whose tape differs from its registered
``tape_sha256``, fails the run; scenes are never redrawn.

Families: ``sword`` and ``bomber`` finish lethal with a Link-4 finisher, ``main2_set`` deals its damage and sets the
defensive trap in Main Phase 2 (``finish_turn`` then ends the turn cleanly), ``bomber``/``linked_clear`` clears the
field before attacking, and ``backrow_timing`` keeps defensive sets until Main Phase 2 against a removal threat.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "mirrorforce_specialization_scene_registry/v1"
RECORD_KEYS = ("decisions", "tape_sha256")  # registry bookkeeping, not scene parameters
FAMILIES = {("sword", None), ("bomber", None), ("bomber", "linked_clear"), ("main2_set", None),
            ("backrow_timing", None)}


def demonstrate(core, deck, scene):
    """The family's scripted demonstration of one registered scene, after its outcome check."""
    from mirrorforce.agent.train.sky_combo_curriculum import BOMBER, demonstrate as combo
    from mirrorforce.agent.train.sky_bomber_clear import demonstrate_clear
    from mirrorforce.agent.train.sky_backrow_timing import demonstrate_timing
    family, variant, lp = scene["family"], scene.get("variant"), scene["start_lp"]
    if (family, variant) not in FAMILIES:
        raise ValueError("unregistered scene family " + repr((family, variant)))
    if family == "backrow_timing":
        result = demonstrate_timing(core, deck, {k: v for k, v in scene.items() if k not in RECORD_KEYS})
        ok = result["lp"][0] == lp - 1500 and len(result["own_backrow"]) == 2 \
            and all(x["phase"] == 256 for x in result["choices"] if x["player"] == 1 and x["action"]["act"] == 1)
    elif variant == "linked_clear":
        result = demonstrate_clear(core, deck, seed=scene["seed"], start_lp=lp,
                                   trigger_resources=scene.get("trigger_resources", True))
        ok = result["winner"] == 1 and not result["before_battle"]["opponent_field"]
    elif family == "main2_set":
        finish_turn = scene.get("finish_turn", False)
        result = combo(core, deck, seed=scene["seed"], start_lp=lp, finish_lethal=True, finish_main2=True,
                       finish_turn=finish_turn)
        ok = result["lp"][0] == lp - 7500 and (result["turn"] > 2 if finish_turn else result["winner"] is None)
    else:
        extra = {"finisher": BOMBER, "opponent": "set_raye"} if family == "bomber" else {}
        result = combo(core, deck, seed=scene["seed"], start_lp=lp, finish_lethal=True, **extra)
        ok = result["winner"] == 1
    if not ok:
        raise ValueError(f"scene {family}/{variant} seed {scene['seed']} missed its registered outcome")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True, help="the export's pinned assets (JSON object)")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--only", nargs="*", type=int, help="scene indices to play (default: all)")
    args = parser.parse_args(argv)
    from mirrorforce.effectinfo import get_effectinfo_core
    from mirrorforce.netduel.cards import load_ydk
    from mirrorforce.worldmodel.engine import DeckList
    registry = json.loads(args.registry.read_bytes())
    if registry.get("schema") != SCHEMA or not registry.get("scenes"):
        raise ValueError("a scene registry needs its schema and scenes")
    assets = json.loads(args.assets.read_bytes())
    main_deck, extra, _ = load_ydk(Path(__file__).resolve().parents[1] / registry["deck"])
    deck = DeckList("specialization-scenes", tuple(main_deck), tuple(extra))
    args.out.mkdir(parents=True, exist_ok=False)
    core = get_effectinfo_core()

    def save(name, value):
        path = args.out / name
        with path.open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
        return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    tapes = []
    for index, scene in enumerate(registry["scenes"]):
        if args.only and index not in args.only:
            continue
        result = demonstrate(core, deck, scene)
        save(f"private-{index:03d}.json", result)
        tape = save(f"public-{index:03d}.json", result["public_tape"])
        if "tape_sha256" in scene and tape["sha256"] != scene["tape_sha256"]:
            raise ValueError(f"scene {index} did not reproduce its registered public tape")
        tapes.append({**{k: v for k, v in scene.items() if k not in RECORD_KEYS}, **tape})
        print(json.dumps({"scene": index, "family": scene["family"], "variant": scene.get("variant"),
                          "split": scene["split"], "tape_sha256": tape["sha256"]}), flush=True)
    reference = save("export-config.json", {"assets": assets, "tapes": tapes})
    print(json.dumps({"tapes": len(tapes), "export_config": reference}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
