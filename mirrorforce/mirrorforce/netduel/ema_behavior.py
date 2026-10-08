"""Explicit public-client adapter for the debiased-EMA behavior diagnostic.

This module has no host, deal, hidden-card or training-data imports. Ordinary
``RemotePolicy`` clients and service defaults remain unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from .agent_policy import RemotePolicy
from mirrorforce.agent.train.debiased_policy import check_session

PURPOSE = "behavior_diagnostic"
FINITE_LAW = "behavior-all-backend-returns-and-initial-memory-finite/v1"
FINITE_IDENTITY = {"law": FINITE_LAW, "logits": True, "value": True,
                   "wdl": True, "recurrent_memory": True}


def finite_memory(state):
    """Check every recurrent leaf and return its exact typed-tree digest.

    No values are changed, clamped, rounded or sent to the policy. Empty and
    object-valued memory cannot certify a recurrent checkpoint.
    """
    h, leaves = hashlib.sha256(), 0

    def visit(value):
        nonlocal leaves
        if isinstance(value, dict):
            if not value or not all(isinstance(k, str) for k in value):
                raise ValueError("finite recurrent memory needs nonempty string-keyed trees")
            h.update(b"dict{")
            for key in sorted(value):
                h.update(json.dumps(key).encode())
                visit(value[key])
            h.update(b"}")
        elif isinstance(value, (list, tuple)):
            h.update(f"{type(value).__name__}:{len(value)}[".encode())
            for item in value:
                visit(item)
            h.update(b"]")
        else:
            array = np.asarray(value)
            if (not array.size or array.dtype.hasobject
                    or not (np.issubdtype(array.dtype, np.number) or str(array.dtype) == "bfloat16")
                    or not np.isfinite(array).all()):
                raise ValueError("nonfinite, empty or nonnumeric recurrent memory")
            header = json.dumps([str(array.dtype), array.shape], separators=(",", ":")).encode()
            h.update(len(header).to_bytes(8, "big") + header + array.tobytes())
            leaves += 1

    visit(state)
    if not leaves:
        raise ValueError("finite recurrent memory needs at least one numeric leaf")
    return h.hexdigest()


def check_forward(result):
    """Fail before emitting a decision if ANY network output is nonfinite."""
    state, logits, value, wdl = result
    memory = finite_memory(state)
    for label, raw in (("logits", logits), ("value", value), ("wdl", wdl)):
        array = np.asarray(raw)
        if (not array.size or array.dtype.hasobject
                or not (np.issubdtype(array.dtype, np.number) or str(array.dtype) == "bfloat16")
                or not np.isfinite(array).all()):
            raise ValueError("nonfinite or nonnumeric behavior " + label)
    return memory


class FiniteAuditBackend:
    """Opt-in guard around an unchanged backend; no alternative computation.

    Both single and batched returns are checked, including rebuild operations.
    The default service does not construct this wrapper.
    """

    def __init__(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def initial_state(self):
        state = self.backend.initial_state()
        finite_memory(state)
        return state

    def act(self, *args, **kwargs):
        result = self.backend.act(*args, **kwargs)
        check_forward(result)
        return result

    def act_batch(self, *args, **kwargs):
        results = self.backend.act_batch(*args, **kwargs)
        for result in results:
            check_forward(result)
        return results

    def act_batch_belief(self, *args, **kwargs):
        results = self.backend.act_batch_belief(*args, **kwargs)
        for result in results:
            check_forward(result[:4])
        return results


class BehaviorRemotePolicy(RemotePolicy):
    """Only a named diagnostic may add the debiased session handshake.

    The reference arm remains iterate: it does not receive candidate-purpose
    fields. Policy seeds are explicit and unrelated to privileged host seeds.
    """

    name = "ema-behavior-policy"

    def __init__(self, address, expected_identity, *, candidate, seed, timeout=120.0):
        if type(candidate) is not bool or type(seed) is not int or seed < 0:
            raise ValueError("behavior role and public policy seed must be explicit")
        identity = copy.deepcopy(expected_identity)
        if identity.get("behavior_finite_audit") != FINITE_IDENTITY:
            raise ValueError("behavior clients require the finite-output service opt-in")
        self.candidate = candidate
        self.handshake = {}
        if candidate:
            self.handshake = {"purpose": PURPOSE, "debiased_ema_identity_sha256":
                              identity.get("session_admission", {}).get("debiased_ema_identity_sha256")}
            if identity.get("weights") != "debiased_ema":
                raise ValueError("the behavior candidate must be explicitly debiased EMA")
            check_session(identity, self.handshake)
        elif identity.get("weights") != "iterate" or "debiased_ema" in identity:
            raise ValueError("the behavior reference must be the registered raw iterate")
        super().__init__(address, identity, seed=seed, timeout=timeout)

    def _call(self, request):
        if self.candidate and request.get("op") in ("open", "open_stream"):
            if set(request) & set(self.handshake):
                raise ValueError("a behavior request cannot override the registered handshake")
            request = {**request, **self.handshake}
        return super()._call(request)

    def report(self):
        return {**super().report(), "behavior_diagnostic": {
            "purpose": PURPOSE, "candidate": self.candidate, "policy_seed": self.seed,
            "finite_audit_law": FINITE_LAW}}
