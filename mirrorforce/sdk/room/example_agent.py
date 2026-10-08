"""Replace choose(state) with your model; return an index into state.actions."""

import random

_rng = random.Random(20261003)


def choose(state):
    """Runnable uniform legal-action baseline; no model or rule engine required."""
    print("DECISION", state.turn, state.phase, state.lp,
          [(i, action.describe()) for i, action in enumerate(state.actions)], flush=True)
    return _rng.randrange(state.n)
