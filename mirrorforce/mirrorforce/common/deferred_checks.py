"""Ordered input checks with one host synchronization.

Model input validation used to call ``bool(tensor)`` once per condition and,
in some places, once per batch row. On CUDA every such call waits for the
device. This helper keeps the same conditions, the same messages and the same
first-failure order, but reads all device-side flags with one transfer.

Host-side (shape/dtype/device) failures are recorded in order too. A caller
stops adding checks after a host-side failure, because later tensor checks may
not be computable, and raises immediately; any earlier device flag still wins.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import threading

import torch

_SINK = threading.local()


@contextmanager
def deferred_into(sink):
    """Within the block, model value checks append ``(error factory, device flag)`` to the list ``sink`` in their
    order instead of reading the device; the caller reads every flag once (a PPO step's agreement, an actor step's
    read) and raises the first that fired. A host-side failure still raises at once, after the flags already in
    ``sink``, which keep their precedence. Code after a deferred check runs on the checked values, so every index
    such a check guards is clamped where it is used."""
    previous = getattr(_SINK, "value", None)
    _SINK.value = sink
    try:
        yield sink
    finally:
        _SINK.value = previous


def active_sink():
    return getattr(_SINK, "value", None)


def refuse(flag, message, error=ValueError):
    """Refuse with ``message`` when any element of the boolean tensor ``flag`` holds: at once (one host read), or
    deferred into the active sink."""
    sink = active_sink()
    if sink is None:
        if bool(flag.any()):
            raise error(message)
    else:
        sink.append((partial(error, message), flag.any()))


class DeferredChecks:
    def __init__(self):
        self._items = []

    def host(self, failed, message):
        """Record a Python-level condition; return True when it passed."""
        failed = bool(failed)
        self._items.append((failed, message))
        return not failed

    def device(self, failed, message):
        """Record a 0-d/any-shape boolean tensor; any True element fails."""
        if not torch.is_tensor(failed) or failed.dtype != torch.bool:
            raise TypeError("deferred device checks need boolean tensors")
        self._items.append((failed.any(), message))

    def raise_first(self, error=ValueError):
        sink = active_sink()
        if sink is not None:
            items, self._items = self._items, []
            for flag, message in items:
                if torch.is_tensor(flag):
                    sink.append((partial(error, message), flag))
                elif flag:
                    raise error(message)
            return
        tensors = [flag for flag, _ in self._items if torch.is_tensor(flag)]
        values = iter(torch.stack(tensors).tolist() if tensors else ())
        for flag, message in self._items:
            if (next(values) if torch.is_tensor(flag) else flag):
                raise error(message)
        self._items = []


def duplicated_keys(keys, valid):
    """Per row, whether any two valid entries share a key (keys must be >= 0)."""
    ordered = keys.masked_fill(~valid, -1).sort(-1).values
    return ((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] >= 0)).any(-1)
