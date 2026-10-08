"""The chain-link resolution a public message stream is in (native twin: ``DuelInstance::chain_resolution``).

0 outside a resolution; otherwise the stream's ordinal of the current ``MSG_CHAIN_SOLVING``, until the
matching ``MSG_CHAIN_SOLVED`` or ``MSG_CHAIN_END``. The host sends all three messages to both seats, so a
network client, the native vector frame and the Python engine driver derive the same value. An ordinal, not
the link number: link 1 of two chains in one phase are two resolutions.
"""
from __future__ import annotations

from . import constants as C


class ChainResolution:
    __slots__ = ("count", "current")

    def __init__(self) -> None:
        self.count = 0
        self.current = 0

    def observe(self, msg: int) -> None:
        if msg == C.MSG_CHAIN_SOLVING:
            self.count += 1
            self.current = self.count
        elif msg in (C.MSG_CHAIN_SOLVED, C.MSG_CHAIN_END):
            self.current = 0
