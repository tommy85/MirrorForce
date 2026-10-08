"""Independent A0 initial-parameter reconstruction audit; never a policy loader.

The two unused room embedding ranges are declared before reading checkpoints.
All 48,384 scalars are required, including values affected by FP32 EMA roundoff.
Passing this audit is not a behavior gate and does not authorize EMA deployment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np

SOURCE = "c04fee15de7dd2b9b748857f124fa0795b55c215"
ORIGINAL_JOB = "0da86a3bf2f36875924ad027d15bc36890939860698c6d989c1ba1fd83b7cd49"
ASSET_MANIFEST = "06a3c7b1b270a30e86ca7e23f9e0d0826b71d3665bbe96d37159e068307b0053"
ORIGINAL_SEED = 20261003
INIT_SEED = 3780174
DECAY = 0.999
FROZEN_PATHS = (("inputs", "room_format", "embedding"), ("inputs", "room_era", "embedding"))
FROZEN_LAW = "a0-room-zero-adam-no-weight-decay-rows-1-through-63/v1"
RECURRENCE_LAW = "fp32-multiply-multiply-add-once-per-a0-iteration/v1"
SCHEMA = "mirrorforce_initial_reconstruction_audit/v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_ref(ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("artifact reference requires exactly path and sha256")
    path = Path(ref["path"])
    if path.is_symlink() or not path.is_file():
        raise ValueError("artifact must be a regular, non-symlink file")
    raw = path.read_bytes()
    if digest(raw) != ref["sha256"]:
        raise ValueError("artifact checksum differs: " + str(path))
    return raw


def publish(directory, prefix, suffix, raw):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (prefix + "-" + digest(raw) + suffix)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError("immutable artifact collision") from None
    return {"path": str(path), "sha256": digest(raw)}


def flatten(tree, path=()):
    if isinstance(tree, dict):
        if not tree or any(not isinstance(key, str) or "/" in key for key in tree):
            raise ValueError("parameter trees require nonempty string-keyed dictionaries")
        result = {}
        for key in sorted(tree):
            result.update(flatten(tree[key], path + (key,)))
        return result
    value = np.asarray(tree)
    if value.dtype != np.dtype("float32") or not np.isfinite(value).all():
        raise ValueError("initial parameters must all be finite float32 arrays")
    return {path: value}


def array_identity(value):
    value = np.ascontiguousarray(value)
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": digest(value.tobytes())}


def tree_identity(tree):
    leaves = {"/".join(path): array_identity(value) for path, value in flatten(tree).items()}
    return {"sha256": digest(canonical(leaves)), "leaves": leaves,
            "scalar_count": sum(np.asarray(v).size for v in flatten(tree).values())}


def same_tree_layout(initial, other):
    left, right = flatten(initial), flatten(other)
    if left.keys() != right.keys():
        raise ValueError("full parameter tree paths differ")
    if any(left[path].shape != right[path].shape for path in left):
        raise ValueError("full parameter tree shapes differ")


def frozen_rows(tree):
    leaves = flatten(tree)
    rows = {}
    for path in FROZEN_PATHS:
        if path not in leaves or leaves[path].shape != (64, 384):
            raise ValueError("both complete 64 x 384 room embedding tables are required")
        rows["/".join(path)] = leaves[path][1:64]
    assert sum(value.size for value in rows.values()) == 48384
    return rows


def bit_mismatches(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        raise ValueError("bit comparison requires identical dtype and shape")
    return int(np.count_nonzero(np.ascontiguousarray(a).view(np.uint32)
                                != np.ascontiguousarray(b).view(np.uint32)))


def fp32_recurrence(initial, count, decay=DECAY):
    if type(count) is not int or not 0 < count <= 1000000 or decay != DECAY:
        raise ValueError("expected positive A0 iteration count and original decay")
    initial = np.asarray(initial)
    if initial.dtype != np.dtype("float32") or not np.isfinite(initial).all():
        raise ValueError("FP32 recurrence requires finite float32 inputs")
    result = initial.copy()
    for _ in range(count):
        # Separate ufuncs preserve the original eager JAX multiply/multiply/add;
        # no float64 recurrence and no algebraic simplification to initial.
        result = np.add(np.multiply(np.float32(decay), result),
                        np.multiply(np.float32(1.0 - decay), initial))
    return result


def checkpoint_count(payload, receipt):
    if receipt.get("schema") != "mirrorforce_training_checkpoint/v1" or receipt.get("ema") is not True:
        raise ValueError("expected an EMA-bearing original training checkpoint")
    counters = payload.get("counters")
    if not isinstance(counters, dict) or set(counters) != {"global_step", "learner_update"}:
        raise ValueError("missing exact payload counters")
    if any(type(v) is not int or v <= 0 for v in counters.values()) or counters != receipt.get("counters"):
        raise ValueError("payload counters must be positive and equal the receipt")
    config = receipt["config"]
    if (not {"room_format", "room_format_table"} <= config.keys()
            or config.get("room_format") is not None or config.get("room_format_table") is not None
            or config.get("recipe") != "ataraxos" or config.get("iteration_steps") != 200
            or config.get("iteration_decisions") != 4915200
            or config.get("ataraxos", {}).get("ema_decay") != DECAY
            or receipt.get("recipe", {}).get("ataraxos", {}).get("ema_decay") != DECAY):
        raise ValueError("checkpoint does not obey the declared A0 room/count/decay law")
    return counters["learner_update"]


def certify(initial, payload, receipt, model_identity):
    count = checkpoint_count(payload, receipt)
    if receipt.get("model") != model_identity:
        raise ValueError("reconstructed model identity differs from checkpoint")
    params, ema = payload["state"]["params"], payload["ema"]
    same_tree_layout(initial, params)
    same_tree_layout(initial, ema)
    rows0, rowsp, rowse = frozen_rows(initial), frozen_rows(params), frozen_rows(ema)
    rows = {}
    for path, start in rows0.items():
        replay = fp32_recurrence(start, count)
        rows[path] = {"row_start": 1, "row_stop_exclusive": 64, "scalar_count": start.size,
                      "initial": array_identity(start), "params": array_identity(rowsp[path]),
                      "historical_ema": array_identity(rowse[path]), "replayed_ema": array_identity(replay),
                      "initial_vs_params_bit_mismatches": bit_mismatches(start, rowsp[path]),
                      "replay_vs_ema_bit_mismatches": bit_mismatches(replay, rowse[path]),
                      "params_vs_ema_bit_mismatches": bit_mismatches(rowsp[path], rowse[path])}
    return {"iteration_count_from_payload": count, "decay": DECAY, "rows": rows,
            "frozen_scalar_count": 48384, "full_tree_layout_checked": True,
            "passed": all(r["initial_vs_params_bit_mismatches"] == 0
                          and r["replay_vs_ema_bit_mismatches"] == 0 for r in rows.values())}


def validate_original_job(raw):
    if digest(raw) != ORIGINAL_JOB:
        raise ValueError("reconstruction requires the pinned original A0 job")
    job = json.loads(raw)
    if job["module"] != "mirrorforce.agent.train.cleanba":
        raise ValueError("unexpected original trainer")
    if random.Random(ORIGINAL_SEED).randint(0, 100000000) != INIT_SEED:
        raise ValueError("Python seed derivation changed")
    return job


def reconstruct(config):
    """Run only in a fresh process importing the pinned old tree, never train()."""
    import os
    import sys
    import subprocess
    import flax
    import jax
    import jax.numpy as jnp
    from mirrorforce.agent.train import cleanba

    source = Path(config["original_source"]).resolve()
    trainer = Path(cleanba.__file__).resolve()
    if trainer != source / "mirrorforce/mirrorforce/agent/train/cleanba.py":
        raise ValueError("trainer resolved outside the pinned original source")
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("mirrorforce.") and filename and not Path(filename).resolve().is_relative_to(source):
            raise ValueError("mixed old/new model imports: " + name)

    def verify_source():
        command = [sys.executable, str(source / "mirrorforce/tools/localize_code.py"), "verify",
                   "--repo", str(source), "--expected-source", SOURCE]
        return json.loads(subprocess.check_output(command, text=True))

    source_before = verify_source()
    job = validate_original_job(read_ref(config["original_job"]))
    args = cleanba.tyro.cli(cleanba.Args, args=job["argv"])
    if args.seed != ORIGINAL_SEED or args.checkpoint or args.resume or args.embedding_file:
        raise ValueError("initialization must start from original seed without weights or resume")
    if args.room_format is not None or args.room_format_table is not None or args.ataraxos.ema_decay != DECAY:
        raise ValueError("original frozen-room law differs")
    baseline_ref = config["checkpoints"][0]
    baseline = json.loads(read_ref(baseline_ref["receipt"]))
    if baseline["job_sha256"] != ORIGINAL_JOB or baseline["initial_checkpoint"] is not None:
        raise ValueError("baseline receipt is not the original random initialization lineage")
    root_old = Path("/data/mirrorforce/run-a0-assets-06a3c7b1")
    root_new = Path(config["assets_root"]).resolve()
    read_ref({"path": str(root_new / "STAGE.json"), "sha256": ASSET_MANIFEST})
    assets = json.loads(subprocess.check_output(
        [sys.executable, str(source / "mirrorforce/tools/mf_runtime_reference_stage.py"),
         "check", "--tree", str(root_new)], text=True))
    for field in ("deck", "semantic_file", "code_list_file", "cards_db", "announce_tables", "card_tables"):
        setattr(args, field, str(root_new / Path(getattr(args, field)).relative_to(root_old)))
    for field, expected in (("semantic_file", baseline["tables"]["semantic_file_sha256"]),
                            ("code_list_file", baseline["tables"]["code_list_sha256"]),
                            ("cards_db", baseline["tables"]["cards_db_sha256"]),
                            ("announce_tables", baseline["announce"]["tables_file_sha256"]),
                            ("card_tables", baseline["card_tables"]["sha256"])):
        read_ref({"path": getattr(args, field), "sha256": expected})
    if cleanba.state_io.tree_digest(args.deck, "*.ydk") != baseline["pool"]["deck_tree_sha256"]:
        raise ValueError("original deck pool differs")
    if cleanba.state_io.file_digest(os.environ["MF_DUEL_NATIVE"]) != baseline["native_sha256"]:
        raise ValueError("original native module differs")
    if (jax.__version__, flax.__version__, np.__version__) != ("0.5.3", "0.10.4", "1.26.4"):
        raise ValueError("original JAX/Flax/NumPy versions required")
    if len(jax.local_devices()) != 1:
        raise ValueError("initialization diagnostic requires exactly one visible device")

    # Reproduce metadata and registration before env/model initialization.
    semantic_tables, args.semantic_shape, args.semantic_metadata = cleanba.load_structured_semantics(
        args.semantic_file, args.code_list_file)
    args.freeze_id = True
    args.batch_size = baseline["config"]["batch_size"]
    args.minibatch_size = baseline["config"]["minibatch_size"]
    identity = cleanba.checkpoint_identity(args)
    for field in ("model", "tables", "card_tables", "native_sha256", "native_core", "announce", "window_law"):
        if identity[field] != baseline[field]:
            raise ValueError("original initializer identity differs: " + field)
    predeclared = {"paths": [list(p) for p in FROZEN_PATHS], "start": 1, "stop": 64,
                   "total_scalars": 48384}
    declaration = publish(config["out"], "initial-reconstruction-predeclaration", ".json", canonical({
        "schema": "mirrorforce_initial_reconstruction_predeclaration/v1", "source": source_before,
        "assets": assets, "original_job": config["original_job"], "predeclared_rows": predeclared,
        "frozen_law": FROZEN_LAW, "recurrence_law": RECURRENCE_LAW,
        "config_sha256": digest(canonical(config)), "training_eligible": False}))
    print("INITIAL-PREDECLARATION " + json.dumps(declaration), flush=True)
    random.seed(ORIGINAL_SEED)
    seed = random.randint(0, 100000000)
    init_key = jax.random.PRNGKey(seed)
    random.seed(seed)
    args.real_seed = random.randint(0, 100000000)
    deck, names = cleanba.init_duel(args.env_id, "english", args.deck, args.code_list_file,
                                   return_deck_names=True, db_path=args.cards_db)
    args.deck_names = sorted(names)
    args.deck1, args.deck2 = args.deck1 or deck, args.deck2 or deck
    env = cleanba.make_env(args, 0, 2, 1)
    try:
        sample, _ = env.reset()
        sample = jax.tree.map(lambda x: jnp.asarray(x[:1]), sample)
    finally:
        env.close()
    agent = cleanba.create_agent(args)
    variables = flax.core.unfreeze(agent.init(init_key, sample, agent.init_rnn_state(1)))
    cleanba.inject_semantic_constants(variables, semantic_tables)
    tables = cleanba.card_tables_from_args(args)
    cleanba.card_tables.inject(variables, tables[0])
    variables = jax.device_get(variables)
    initial = variables["params"]
    artifact = publish(config["out"], "reconstructed-initial-variables", ".msgpack",
                       flax.serialization.msgpack_serialize(variables))
    checks = []
    for ref in config["checkpoints"]:
        receipt = json.loads(read_ref(ref["receipt"]))
        raw = read_ref(ref["payload"])
        if receipt["payload_sha256"] != digest(raw) or receipt["payload_bytes"] != len(raw):
            raise ValueError("payload and receipt binding differ")
        payload = flax.serialization.msgpack_restore(raw)
        result = certify(initial, payload, receipt, identity["model"])
        # Independently run the original eager JAX implementation, not only a
        # transcription, on ALL predeclared rows with the payload count.
        fixed = jax.tree.map(jnp.asarray, frozen_rows(initial))
        replay = fixed
        for _ in range(result["iteration_count_from_payload"]):
            replay = cleanba.ataraxos.ema_update(replay, fixed, DECAY)
        observed = frozen_rows(payload["ema"])
        mismatches = {path: bit_mismatches(np.asarray(value), observed[path]) for path, value in replay.items()}
        result["original_jax_recurrence_bit_mismatches"] = mismatches
        result["passed"] &= all(value == 0 for value in mismatches.values())
        checks.append({"payload": ref["payload"], "receipt": ref["receipt"], **result})
        del raw, payload
    source_after = verify_source()
    report = {"schema": SCHEMA, "source": source_before, "source_after": source_after,
              "original_job": config["original_job"], "original_seed": ORIGINAL_SEED, "init_seed": seed,
              "frozen_law": FROZEN_LAW, "recurrence_law": RECURRENCE_LAW,
              "predeclared_rows": predeclared, "predeclaration": declaration, "assets": assets,
              "reconstructed_variables": artifact, "initial_params": tree_identity(initial),
              "model": identity["model"], "checks": checks, "static_gate_passed": all(c["passed"] for c in checks),
              "behavior_gate_passed": False, "formal_ema_evaluation_eligible": False,
              "training_eligible": False, "policy_loader_changed": False,
              "runtime": {"jax": jax.__version__, "flax": flax.__version__, "numpy": np.__version__,
                          "device": str(jax.local_devices()[0]), "pid": os.getpid(),
                          "cpus": sorted(os.sched_getaffinity(0))},
              "config_sha256": digest(canonical(config))}
    reference = publish(config["out"], "initial-reconstruction-audit", ".json", canonical(report))
    print("INITIAL-RECONSTRUCTION " + json.dumps({**reference, "static_gate_passed": report["static_gate_passed"]}), flush=True)
    return 0 if report["static_gate_passed"] else 3
