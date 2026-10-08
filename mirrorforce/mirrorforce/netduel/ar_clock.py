"""Opt-in AR RPC clock, pinned inside the registered runtime head identity.

Absent ``request_clock`` retains the original five-second request protocol.
The new protocol carries the caller's same-host absolute search deadline twice
and requires exact equality: it creates no earlier sub-budget or renewed clock.
"""
from __future__ import annotations

import math

SCHEMA = "mirrorforce_ar_request_clock/v1"
LAW = "same-host-original-search-deadline-no-extension/v1"
LEGACY_MAX_SECONDS = 5.
OPT_IN_MAX_SECONDS = 9.
REQUEST_FIELDS = {"ar_request_clock", "original_deadline_monotonic"}


def contract(max_seconds):
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) \
            or max_seconds != OPT_IN_MAX_SECONDS:
        raise ValueError("the explicit AR request clock supports exactly nine seconds")
    return {"schema": SCHEMA, "law": LAW, "max_seconds": OPT_IN_MAX_SECONDS}


def from_head(head):
    """Return a fresh validated contract; a legacy head has no extra fields."""
    if type(head) is not dict:
        raise ValueError("AR request clock needs the registered runtime head identity")
    if "request_clock" not in head:
        return None
    value = head["request_clock"]
    if type(value) is not dict or set(value) != {"schema", "law", "max_seconds"} \
            or value != contract(value.get("max_seconds")):
        raise ValueError("AR runtime head has an unknown request clock")
    return contract(value["max_seconds"])


def bind(request, head):
    """Bind a request to the registered head without changing its deadline."""
    if REQUEST_FIELDS.intersection(request):
        raise ValueError("AR request clock fields are owned by the registered provider")
    value = from_head(head)
    if value is None:
        return dict(request)
    deadline = request.get("deadline_monotonic")
    _deadline(deadline)
    return {**request, "ar_request_clock": value, "original_deadline_monotonic": deadline}


def _deadline(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("AR proposals require a finite absolute same-host deadline")


def validate(request, head, *, now):
    """Reject mixed protocols/over-limit deadlines, preserving expired status."""
    value = from_head(head)
    expected_fields = {"op", "session", "expected_obs_sha256", "specification", "count", "seed",
                       "deadline_monotonic"} | (REQUEST_FIELDS if value is not None else set())
    if type(request) is not dict or set(request) != expected_fields:
        raise ValueError("AR request fields differ from the registered runtime clock")
    deadline = request["deadline_monotonic"]
    _deadline(deadline)
    _deadline(now)
    if value is not None:
        original = request["original_deadline_monotonic"]
        _deadline(original)
        supplied = request["ar_request_clock"]
        if type(supplied) is not dict or from_head({"request_clock": supplied}) != value \
                or deadline != original:
            raise ValueError("AR request changed its registered clock or original search deadline")
    maximum = LEGACY_MAX_SECONDS if value is None else value["max_seconds"]
    if deadline - now > maximum:
        raise ValueError("AR request exceeds the registered absolute-deadline upper limit")
    return deadline
