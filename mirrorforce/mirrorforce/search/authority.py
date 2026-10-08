"""Authority labels shared by snapshot-based simulator consumers.

These labels are part of the information-flow boundary, not descriptive
telemetry alone.  Search code must additionally possess an engine-minted
capability before a line carrying :data:`INFORMATION_SET_SEARCH` may be used.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "INFORMATION_SET_SEARCH",
    "PRIVILEGED_TARGET",
    "UNREALIZED_TRUE_HIDDEN",
    "SimulationAuthority",
]


class SimulationAuthority(str, Enum):
    """Where a simulator line's hidden state came from and who may use it."""

    #: The parked live duel still contains the opponent's real hidden state.
    #: It is only a source from which a sampled particle may be realized.
    UNREALIZED_TRUE_HIDDEN = "UNREALIZED_TRUE_HIDDEN"
    #: Hidden state was sampled from public evidence and realized in the core.
    INFORMATION_SET_SEARCH = "INFORMATION_SET_SEARCH"
    #: Full true state is intentionally used for an offline target/diagnostic.
    PRIVILEGED_TARGET = "PRIVILEGED_TARGET"


UNREALIZED_TRUE_HIDDEN = SimulationAuthority.UNREALIZED_TRUE_HIDDEN
INFORMATION_SET_SEARCH = SimulationAuthority.INFORMATION_SET_SEARCH
PRIVILEGED_TARGET = SimulationAuthority.PRIVILEGED_TARGET
