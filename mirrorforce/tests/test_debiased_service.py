"""CPU-only service opt-in tests; no native/model forward or eligibility claim."""
import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mirrorforce.agent import train
from mirrorforce.agent.train import debiased_ema as D
from mirrorforce.agent.train import debiased_policy as P
from mirrorforce.agent.train import ema_reconstruction as audit
from tools import mf_runtime_policy_service as S


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    files = {}
    for name in ("checkpoint", "semantic", "cards", "codes", "announce"):
        files[name] = tmp_path / name
        files[name].write_bytes(name.encode())
    original = {"p": np.asarray([1, -.5], np.float32)}
    derived = {"p": np.asarray([2, 1], np.float32)}
    constants = object()
    variables = {"params": original, "constants": constants}
    receipt = {"model": {"args": {"dtype": "bfloat16"}},
               "recipe": {"ataraxos": {"magnet": "uniform_legal"}}, "native_sha256": "f" * 64}
    tree = audit.tree_identity(original)
    identity = {"schema": D.SCHEMA, "weights": D.WEIGHTS, "purpose": D.PURPOSE,
                "iterate_params": {key: tree[key] for key in ("sha256", "scalar_count")},
                "static_gate_passed": True, "behavior_gate_passed": False,
                "formal_ema_evaluation_eligible": False, "training_eligible": False}
    request = {"weights": D.WEIGHTS, "purpose": D.PURPOSE,
               "checkpoint": {"payload": {"path": str(files["checkpoint"]),
                                           "sha256": S.file_sha256(files["checkpoint"])},
                              "receipt": {"path": "/fixture-receipt.json", "sha256": "1" * 64}},
               "reconstruction": {"variables": {"path": "/fixture-variables", "sha256": "2" * 64},
                                  "audit": {"path": "/fixture-audit.json", "sha256": "3" * 64}}}
    ref = audit.publish(tmp_path, "request", ".json", audit.canonical(request))
    calls = []

    def load(checkpoint, which, **kwargs):
        calls.append((checkpoint, which, kwargs))
        dtype = kwargs.get("compute_dtype") or "bfloat16"
        return SimpleNamespace(config=SimpleNamespace(dtype=dtype)), variables, receipt

    class Policy:
        def __init__(self, agent, variables, receipt, *, matmul_precision):
            self.variables, self.max_options = variables, 192
            self.precision = matmul_precision

        def initial_state(self):
            return None

    fake = SimpleNamespace(load_policy=load, Policy=Policy,
                           native_setup=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(train, "policy_io", fake, raising=False)
    monkeypatch.setitem(sys.modules, "mirrorforce.agent.train.policy_io", fake)
    monkeypatch.setattr(D, "load_debiased_parameters", lambda **kwargs: (derived, copy.deepcopy(receipt), identity))
    return SimpleNamespace(files=files, original=original, derived=derived, variables=variables, receipt=receipt,
                           identity=identity, request=request, ref=ref, calls=calls, fake=fake)


def kwargs(runtime):
    f = runtime.files
    return dict(semantic_file=f["semantic"], code_list=f["codes"], card_tables=f["cards"],
                announce_tables=f["announce"], dormant_table=None)


@pytest.mark.parametrize("weights", ["iterate", "ema"])
@pytest.mark.parametrize("compute_dtype", [None, "float32"])
@pytest.mark.parametrize("batching", ["plain", "bucket", "constant"])
def test_original_identity_json_and_loader_arguments_are_unchanged(runtime, monkeypatch, weights, compute_dtype,
                                                                  batching):
    monkeypatch.setattr(P, "load_policy", lambda *a, **k: pytest.fail("ordinary weights invoked debiasing"))
    batch = {"plain": {}, "bucket": {"batch_sizes": (1, 4)}, "constant": {"constant_size": 64}}[batching]
    native = object()
    backend = S.CheckpointBackend(native, runtime.files["checkpoint"], weights,
                                  **kwargs(runtime), compute_dtype=compute_dtype, **batch)
    # Golden pre-opt-in identity: no new purpose, law, eligibility, or request keys.
    sha = lambda value: hashlib.sha256(value.encode()).hexdigest()
    expected = {"backend": "checkpoint", "checkpoint_sha256": sha("checkpoint"), "weights": weights,
                "receipt_sha256": hashlib.sha256(json.dumps(runtime.receipt, sort_keys=True).encode()).hexdigest(),
                "training_native_sha256": "f" * 64, "compute_dtype": compute_dtype or "bfloat16",
                "magnet": "uniform_legal", "semantic_file_sha256": sha("semantic"),
                "card_tables_sha256": sha("cards")}
    if compute_dtype is not None:
        expected["inference_precision"] = {"law": "explicit-inference-dtype/v1", "training_dtype": "bfloat16",
            "compute_dtype": "float32", "matmul_precision": "highest", "checkpoint_parameters_unchanged": True}
    if batching == "bucket":
        expected["search_batching"] = {"law": "valid-row-padding/v1", "sizes": [1, 4]}
    elif batching == "constant":
        expected["search_batching"] = {"law": "constant-batch-full-menu/v1", "size": 64,
                                       "menu_rows": 192, "applies_to": "all_forwards"}
    assert audit.canonical(backend.identity) == audit.canonical(expected)
    assert runtime.calls[0] == (str(runtime.files["checkpoint"]), weights, {
        "semantic_file": str(runtime.files["semantic"]), "code_list_file": str(runtime.files["codes"]),
        "card_tables_file": str(runtime.files["cards"]), "native": native, "compute_dtype": compute_dtype})
    assert backend.policy.variables is runtime.variables and backend.receipt is runtime.receipt


def test_adapter_only_replaces_params_after_normal_checks(runtime):
    agent, variables, receipt, identity = P.load_policy(str(runtime.files["checkpoint"]), runtime.ref,
        semantic_file="semantics", code_list_file="codes", card_tables_file="cards", native="native")
    assert variables is not runtime.variables and variables["params"] is runtime.derived
    assert variables["constants"] is runtime.variables["constants"]
    assert runtime.variables["params"] is runtime.original
    assert receipt == runtime.receipt and identity == runtime.identity and agent.config.dtype == "bfloat16"
    assert runtime.calls[0][1] == "iterate"
    assert runtime.calls[0][2] == {"semantic_file": "semantics", "code_list_file": "codes",
        "card_tables_file": "cards", "native": "native", "compute_dtype": None}


@pytest.mark.parametrize("fault", ["receipt", "params"])
def test_adapter_cross_checks_both_loader_results(runtime, monkeypatch, fault):
    receipt, variables = copy.deepcopy(runtime.receipt), copy.deepcopy(runtime.variables)
    if fault == "receipt":
        receipt["different"] = True
    else:
        variables["params"]["p"].view(np.uint32)[0] ^= np.uint32(1)
    monkeypatch.setattr(runtime.fake, "load_policy", lambda *a, **k: (None, variables, receipt))
    with pytest.raises(ValueError, match="different"):
        P.load_policy(str(runtime.files["checkpoint"]), runtime.ref,
                      semantic_file=None, code_list_file=None, card_tables_file=None)


@pytest.mark.parametrize("compute_dtype", [None, "float32"])
def test_opt_in_backend_names_derived_identity_without_relabeling_receipt(runtime, compute_dtype):
    backend = S.CheckpointBackend(None, runtime.files["checkpoint"], D.WEIGHTS,
        **kwargs(runtime), constant_size=64, compute_dtype=compute_dtype, debiased_request=runtime.ref)
    assert backend.identity["weights"] == D.WEIGHTS
    assert backend.identity["purpose"] == D.PURPOSE
    assert backend.identity["debiased_ema"] == runtime.identity
    assert backend.identity["debiased_ema_request_sha256"] == runtime.ref["sha256"]
    assert backend.identity["session_admission"]["debiased_ema_identity_sha256"] == audit.digest(
        audit.canonical(runtime.identity))
    assert backend.identity["search_batching"]["size"] == 64
    assert backend.policy.variables["params"] is runtime.derived
    assert backend.policy.variables["constants"] is runtime.variables["constants"]
    assert backend.receipt == runtime.receipt
    if compute_dtype is not None:
        assert backend.identity["inference_precision"]["checkpoint_parameters_unchanged"] is False
        assert backend.identity["inference_precision"]["checkpoint_artifact_unchanged"] is True


@pytest.mark.parametrize("weights,ref", [("iterate", {}), ("ema", {}), ("debiased_ema", None)])
def test_backend_refuses_implicit_or_mismatched_opt_in(runtime, weights, ref):
    with pytest.raises(ValueError, match="explicit request"):
        S.CheckpointBackend(None, runtime.files["checkpoint"], weights, **kwargs(runtime), debiased_request=ref)
    assert not runtime.calls


@pytest.mark.parametrize("fault", ["sha", "path", "purpose", "weights", "extra", "group", "ref"])
def test_request_fails_before_any_normal_policy_or_native_load(runtime, tmp_path, fault):
    request = copy.deepcopy(runtime.request)
    if fault == "path":
        request["checkpoint"]["payload"]["path"] = str(tmp_path / "other")
    elif fault == "purpose":
        request["purpose"] = "formal_evaluation"
    elif fault == "weights":
        request["weights"] = "ema"
    elif fault == "extra":
        request["manual_count"] = 200
    elif fault == "group":
        request["reconstruction"] = []
    elif fault == "ref":
        request["reconstruction"]["variables"]["path"] = "relative"
    ref = audit.publish(tmp_path, "changed", ".json", audit.canonical(request))
    if fault == "sha":
        ref["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        P.load_policy(str(runtime.files["checkpoint"]), ref,
                      semantic_file=None, code_list_file=None, card_tables_file=None)
    assert not runtime.calls


def test_opt_in_failure_does_not_fall_back_to_historical_ema_or_iterate(runtime, monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("registered static proof failed")

    monkeypatch.setattr(D, "load_debiased_parameters", refuse)
    with pytest.raises(ValueError, match="static proof failed"):
        S.CheckpointBackend(None, runtime.files["checkpoint"], D.WEIGHTS,
                            **kwargs(runtime), debiased_request=runtime.ref)
    assert not runtime.calls


def cli_base(tmp_path):
    return ["--selection", "greedy", "--cards-db", "/unused", "--code-list", "/unused",
            "--script-root", "/unused", "--announce-tables", "/unused", "--socket", str(tmp_path / "socket"),
            "--out", str(tmp_path / "out")]


@pytest.mark.parametrize("args", [["--weights", "debiased_ema"],
    ["--weights", "debiased_ema", "--backend", "uniform"],
    ["--debiased-ema-request", "/unused"], ["--debiased-ema-request-sha256", "0" * 64]])
def test_cli_rejects_missing_or_ordinary_request_before_native_load(tmp_path, monkeypatch, args):
    monkeypatch.setattr(S, "load_native", lambda *a: pytest.fail("invalid opt-in loaded native"))
    with pytest.raises(ValueError, match="request"):
        S.main(cli_base(tmp_path) + args)


def test_cli_default_remains_original_ema(tmp_path, monkeypatch):
    class ReachedNative(Exception):
        pass

    def check(args):
        assert args.weights == "ema" and args.debiased_ema_request is None
        assert args.debiased_ema_request_sha256 is None
        raise ReachedNative

    monkeypatch.setattr(S, "load_native", check)
    with pytest.raises(ReachedNative):
        S.main(cli_base(tmp_path) + ["--checkpoint", "/unused", "--semantic-file", "/unused"])


def test_cli_requires_full_pinned_opt_in_before_native(runtime, tmp_path, monkeypatch):
    class ReachedNative(Exception):
        pass

    def check(args):
        assert args.weights == D.WEIGHTS and str(args.debiased_ema_request) == runtime.ref["path"]
        assert args.debiased_ema_request_sha256 == runtime.ref["sha256"]
        raise ReachedNative

    monkeypatch.setattr(S, "load_native", check)
    args = cli_base(tmp_path) + ["--weights", D.WEIGHTS, "--checkpoint", str(runtime.files["checkpoint"]),
        "--semantic-file", str(runtime.files["semantic"]), "--debiased-ema-request", runtime.ref["path"],
        "--debiased-ema-request-sha256", runtime.ref["sha256"]]
    with pytest.raises(ReachedNative):
        S.main(args)
    with pytest.raises(ValueError, match="checksum"):
        S.main(args[:-1] + ["0" * 64])


@pytest.mark.parametrize("fault", ["missing", "purpose", "sha", "eligible", "proof", "law"])
@pytest.mark.parametrize("op", ["open", "open_stream"])
def test_existing_formal_clients_and_bad_opt_ins_cannot_create_diagnostic_sessions(runtime, fault, op):
    identity = {"weights": D.WEIGHTS, **P.service_identity(copy.deepcopy(runtime.identity), runtime.ref)}
    request = {"op": op, "seat": 0, "main": [1], "extra": [], "seed": 12,
               "purpose": D.PURPOSE,
               "debiased_ema_identity_sha256": identity["session_admission"]["debiased_ema_identity_sha256"]}
    if op == "open_stream":
        request.update(frames=[], messages=[])
    if fault == "missing":
        del request["purpose"]
        del request["debiased_ema_identity_sha256"]
    elif fault == "purpose":
        request["purpose"] = "formal_evaluation"
    elif fault == "sha":
        request["debiased_ema_identity_sha256"] = "0" * 64
    elif fault == "eligible":
        identity["debiased_ema"]["formal_ema_evaluation_eligible"] = True
    elif fault == "proof":
        identity["debiased_ema"]["static_gate_passed"] = False
    elif fault == "law":
        identity["session_admission"]["law"] = "other"
    service = S.Service(None, S.UniformBackend(), {}, "greedy", 1, identity,
                        client_factory=lambda *a, **k: pytest.fail("bad session reached native factory"))
    with pytest.raises(ValueError, match="behavior_diagnostic"):
        service.dispatch(request)
    assert service.sessions == {} and service.counter == 0


def test_explicit_behavior_session_opens_and_old_weights_need_no_handshake(runtime):
    identity = {"weights": D.WEIGHTS, **P.service_identity(runtime.identity, runtime.ref)}
    service = S.Service(None, S.UniformBackend(), {}, "greedy", 1, identity,
                        client_factory=lambda *a, **k: object())
    request = {"op": "open", "seat": 0, "main": [1], "extra": [], "seed": 12,
               "purpose": D.PURPOSE,
               "debiased_ema_identity_sha256": identity["session_admission"]["debiased_ema_identity_sha256"]}
    assert service.dispatch(request) == {"session": "s1"}
    for weights in ("ema", "iterate"):
        ordinary = S.Service(None, S.UniformBackend(), {}, "greedy", 1, {"weights": weights},
                             client_factory=lambda *a, **k: object())
        assert ordinary.dispatch({k: v for k, v in request.items()
                                  if k not in ("purpose", "debiased_ema_identity_sha256")}) == {"session": "s1"}


def test_diagnostic_open_stream_binds_handshake_and_keeps_old_framing_strict(runtime):
    identity = {"weights": D.WEIGHTS, **P.service_identity(runtime.identity, runtime.ref)}
    service = S.Service(None, S.UniformBackend(), {}, "greedy", 1, identity,
                        client_factory=lambda *a, **k: SimpleNamespace(prompt=lambda: None))
    request = {"op": "open_stream", "seat": 0, "main": [1], "extra": [], "seed": 12,
               "frames": [], "messages": [], "purpose": D.PURPOSE,
               "debiased_ema_identity_sha256": identity["session_admission"]["debiased_ema_identity_sha256"]}
    result = service.dispatch(request)
    assert result["session"] == "s1" and result["trace"] == []
    assert result["stream_sha256"] == audit.digest(audit.canonical(request))
    with pytest.raises(ValueError, match="only a fresh"):
        service.dispatch({**request, "host_hidden_seed": 123})
    assert service.counter == 1
    ordinary = S.Service(None, S.UniformBackend(), {}, "greedy", 1, {"weights": "iterate"})
    with pytest.raises(ValueError, match="only a fresh"):
        ordinary.dispatch(request)
    assert ordinary.counter == 0
