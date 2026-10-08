"""Play a ygopro duel over the LAN protocol with a pluggable policy.

The offline re-simulator (``mirrorforce.replay``) reads recorded response bytes
and recovers the action index; this package does the same mapping online, in
the other direction, against a live host.  It is also the first version of the
product's shadow duel: there is no engine on this side, so the board comes
entirely from the message stream (:mod:`.board`).
"""

from .client import DuelResult, NetDuelClient
from .policy import DecisionState, FirstPolicy, RandomPolicy, make_policy

__all__ = [
    "NetDuelClient",
    "DuelResult",
    "DecisionState",
    "RandomPolicy",
    "FirstPolicy",
    "make_policy",
]
