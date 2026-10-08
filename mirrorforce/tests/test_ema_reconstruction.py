import copy

import numpy as np
import pytest

from mirrorforce.agent.train import ema_reconstruction as audit


def trees(count=140):
    rng = np.random.default_rng(7381)
    initial = {"inputs": {name: {"embedding": rng.normal(size=(64, 384)).astype(np.float32)}
                          for name in ("room_format", "room_era")},
               "another": {"kernel": rng.normal(size=(8, 8)).astype(np.float32)}}
    ema = copy.deepcopy(initial)
    for name in ("room_format", "room_era"):
        ema["inputs"][name]["embedding"] = audit.fp32_recurrence(initial["inputs"][name]["embedding"], count)
    payload = {"state": {"params": copy.deepcopy(initial)}, "ema": ema,
               "counters": {"learner_update": count, "global_step": count * 4915200}}
    receipt = {"schema": "mirrorforce_training_checkpoint/v1", "ema": True,
               "counters": payload["counters"].copy(), "model": {"a": 1},
               "config": {"room_format": None, "room_format_table": None, "recipe": "ataraxos",
                          "iteration_steps": 200, "iteration_decisions": 4915200,
                          "ataraxos": {"ema_decay": .999}},
               "recipe": {"ataraxos": {"ema_decay": .999}}}
    return initial, payload, receipt


@pytest.mark.parametrize("count", [2, 140])
def test_every_frozen_scalar_with_roundoff(count):
    initial, payload, receipt = trees(count)
    result = audit.certify(initial, payload, receipt, {"a": 1})
    assert result["passed"] and result["frozen_scalar_count"] == 48384
    assert sum(r["scalar_count"] for r in result["rows"].values()) == 48384
    assert sum(r["params_vs_ema_bit_mismatches"] for r in result["rows"].values()) > 0


@pytest.mark.parametrize("which", ["initial", "params", "ema"])
def test_single_bit_cannot_be_filtered_out(which):
    initial, payload, receipt = trees()
    tree = {"initial": initial, "params": payload["state"]["params"], "ema": payload["ema"]}[which]
    array = tree["inputs"]["room_era"]["embedding"]
    array.view(np.uint32)[63, 383] ^= np.uint32(1)
    assert not audit.certify(initial, payload, receipt, {"a": 1})["passed"]


@pytest.mark.parametrize("mutation", ["extra", "missing", "shape", "dtype", "nan", "partial"])
def test_entire_parameter_tree_required(mutation):
    initial, payload, receipt = trees()
    params = payload["state"]["params"]
    if mutation == "extra":
        params["extra"] = np.ones(1, np.float32)
    elif mutation == "missing":
        del params["another"]
    elif mutation == "shape":
        params["another"]["kernel"] = np.ones(3, np.float32)
    elif mutation == "dtype":
        params["another"]["kernel"] = params["another"]["kernel"].astype(np.float64)
    elif mutation == "nan":
        params["another"]["kernel"][0, 0] = np.nan
    else:
        params["inputs"]["room_format"]["embedding"] = params["inputs"]["room_format"]["embedding"][:32]
    with pytest.raises(ValueError):
        audit.certify(initial, payload, receipt, {"a": 1})


@pytest.mark.parametrize("count", [0, -1, True, 2.0, 1000001])
def test_counter_not_manual_epoch_or_adam_guess(count):
    with pytest.raises(ValueError):
        audit.fp32_recurrence(np.ones(3, np.float32), count)


@pytest.mark.parametrize("field,value", [("room_format", "md"), ("room_format_table", "other"),
                                         ("iteration_steps", 1), ("recipe", "ppo")])
def test_room_and_a0_law_fail_closed(field, value):
    initial, payload, receipt = trees()
    receipt["config"][field] = value
    with pytest.raises(ValueError):
        audit.certify(initial, payload, receipt, {"a": 1})


def test_receipt_counter_model_and_decay_bindings():
    initial, payload, receipt = trees()
    with pytest.raises(ValueError, match="model"):
        audit.certify(initial, payload, receipt, {"a": 2})
    receipt["counters"]["learner_update"] += 1
    with pytest.raises(ValueError, match="counters"):
        audit.certify(initial, payload, receipt, {"a": 1})
    with pytest.raises(ValueError):
        audit.fp32_recurrence(np.ones(1, np.float32), 2, .998)


def test_original_job_cannot_be_replaced():
    with pytest.raises(ValueError, match="pinned original"):
        audit.validate_original_job(b"{}")


def test_regular_ref_and_immutable_publication(tmp_path):
    ref = audit.publish(tmp_path, "test", ".json", b"{}")
    assert audit.read_ref(ref) == b"{}"
    assert audit.publish(tmp_path, "test", ".json", b"{}") == ref
    with pytest.raises(ValueError, match="checksum"):
        audit.read_ref({**ref, "sha256": "0" * 64})
    link = tmp_path / "link"
    link.symlink_to(ref["path"])
    with pytest.raises(ValueError, match="non-symlink"):
        audit.read_ref({**ref, "path": str(link)})
