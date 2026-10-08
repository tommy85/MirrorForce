"""Policies for the network client.

A policy is anything with ``choose(state) -> int``.  The index it returns is an
index into ``state.actions``, which is built by :mod:`.actions` and ordered
exactly like ``ygoenv``'s ``legal_actions_`` -- the same convention the offline
re-simulator recovers from recorded responses.

``RandomPolicy`` and ``FirstPolicy`` exist to validate the channel end to end
without a model in the loop; ``FirstPolicy`` is the same "always take option 0"
rule as ygoenv's ``GreedyAI``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

__all__ = [
    "DecisionState",
    "Policy",
    "RandomPolicy",
    "FirstPolicy",
    "ScriptedPolicy",
    "make_policy",
    "POLICIES",
]


@dataclass
class DecisionState:
    """Everything the client can tell a policy about one decision."""

    msg: int
    player: int  # who the engine is asking (always us)
    our_player: int  # our duel-side index
    actions: list
    board: object = None
    turn: int = 0
    phase: int = 0
    lp: tuple = (0, 0)
    extra: dict = field(default_factory=dict)
    #: Give an **unmasked** ``StateSnapshot`` directly. With it, the scoring path no longer derives the board
    #: from ``ShadowBoard``: offline particles (`mirrorforce/search`) hold a
    #: ``DuelDriver``, and ``state.capture`` produces exactly this type, the same as the corpus side.
    #:
    #: **The masking rule is unaffected**: both paths go through ``worldmodel.state.mask_for`` alone.
    #: ``ShadowBoard`` was never a filter; it is the state container of the online path; filtering of message distribution
    #: happens on the ygopro server (``single_duel.cpp``), not on this path.
    snapshot: object = None
    #: the disclosure ledger (given here when ``board`` is empty)
    disclosure: object = None
    #: the turn player (given here when ``board`` is empty)
    turn_player: int = 0

    @property
    def n(self) -> int:
        return len(self.actions)


class Policy:
    name = "policy"

    def choose(self, state: DecisionState) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def on_duel_end(self, result) -> None:
        pass


class RandomPolicy(Policy):
    """Uniform over the legal actions."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, state: DecisionState) -> int:
        return self.rng.randrange(state.n)


class FirstPolicy(Policy):
    """Always the first option -- ygoenv's ``GreedyAI`` rule."""

    name = "first"

    def choose(self, state: DecisionState) -> int:
        return 0


class ScriptedPolicy(Policy):
    """Replays a fixed list of indices; used by the tests."""

    name = "scripted"

    def __init__(self, picks: list[int], fallback: int = 0):
        self.picks = list(picks)
        self.fallback = fallback
        self.pos = 0

    def choose(self, state: DecisionState) -> int:
        if self.pos < len(self.picks):
            idx = self.picks[self.pos]
            self.pos += 1
            return min(idx, state.n - 1)
        return min(self.fallback, state.n - 1)


POLICIES = {
    "random": RandomPolicy,
    "first": FirstPolicy,
}


def make_policy(name: str, seed: int = 0) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; have {sorted(POLICIES)}")
    cls = POLICIES[name]
    try:
        return cls(seed=seed)
    except TypeError:
        return cls()
