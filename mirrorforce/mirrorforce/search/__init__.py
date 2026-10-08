"""Engine search foundations: branching, state rebuilding, belief particles.

Three independent parts, in reverse order of dependency:

``forkbranch``
    O(1) snapshots of a **live** duel process with the copy-on-write of ``fork(2)``.
    Only for offline diagnostics and omniscient self-play experiments; see the deployment rule at the top of that module:
    policy-side search must **never** branch a real duel this way.

``rebuild``
    Rebuild a mid-game state from "public information + sampled hidden cards". This is the only legal particle source
    of policy-side search, and the most critical unknown of this line (fidelity).

``belief``
    A belief particle sampler for a closed environment: the decklist is known; hypergeometric sampling + collapse of disclosed cards.
"""

from __future__ import annotations

__all__ = ["belief", "forkbranch", "rebuild"]
