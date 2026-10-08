"""Update-equivalence search at a root (Ataraxos, Sokota et al., Nature 2026; their ``pyengine/core/search.py``): the
action values from depth-limited continuations and the root's KL-regularized mirror-descent step.

For each candidate root action, continuations (``RolloutPool`` with root actions; mirrorforce/agent/search/rollout.py)
roll out to depth d in the root player's own decisions with the policy for both players. A continuation's leaf value
is the network's value at its own decisions and the game's outcome, TD(lambda)-weighted as Ataraxos weights them:
at own decision i (0 the first after the root action) the value counts (1 - lambda) lambda^i, at the last one
(i = d - 1, the cut leaf) lambda^(d-1); a game that ends after the root player's t-th own decision counts its outcome
lambda^t (lambda = 1: only the leaf's value, or the outcome). The weights of one continuation sum to one. An action's
value is the mean leaf value of its continuations, from the root player's view.

The root step (``search_policy``): over the legal actions, softmax((logits + eta q + eta tau log magnet) / (1 + eta
tau)) with stepsize eta, temperature tau and a magnet policy (Ataraxos weights it uniform over each piece's moves;
here the caller's, uniform over the legal actions by default); ``uniform_magnet`` is their variant without the magnet
term, softmax((logits + eta q) / (1 + eta tau)). Their evaluation defaults: depth 10 plies (5 own decisions),
stepsize 10, temperature 0.001-0.006, lambda 1, 100-200 samples per action.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def balanced_actions(legal: int, samples: int) -> list:
    """The root action of each continuation: every legal action ``samples`` times (Ataraxos' deterministic sampling
    from the uniform policy over the legal actions)."""
    if legal < 1 or samples < 1:
        raise ValueError("a search needs a legal action and a sample per action")
    return [a for a in range(legal) for _ in range(samples)]


def value_weight(own_index: int, depth: int, td_lambda: float) -> float:
    """The weight of the network's value at the root player's own decision ``own_index`` (depth > 0)."""
    if depth <= 0 or not 0 <= own_index < depth:
        raise ValueError("a value counts at an own decision before the leaf depth")
    if td_lambda >= 1:
        return 1.0 if own_index == depth - 1 else 0.0
    if own_index < depth - 1:
        return (1 - td_lambda) * td_lambda ** own_index
    return td_lambda ** own_index


def outcome_weight(own_seen: int, td_lambda: float) -> float:
    """The weight of the outcome of a game that ended after ``own_seen`` own decisions of the root player."""
    return 1.0 if td_lambda >= 1 else td_lambda ** own_seen


def outcome(winner: int, root_player: int) -> float:
    """A finished game's value for the root player: 1 win, -1 loss, 0 draw."""
    if winner in (0, 1):
        return 1.0 if winner == root_player else -1.0
    if winner == 2:
        return 0.0
    raise ValueError(f"no outcome for winner {winner}")


class LeafValues:
    """Accumulates each continuation's leaf value from the values observed at the root player's own decisions (the
    caller evaluates the network there) and the continuation's result."""

    def __init__(self, depth: int, td_lambda: float):
        if depth < 0 or not 0 <= td_lambda <= 1:
            raise ValueError("depth >= 0 and 0 <= lambda <= 1")
        self.depth, self.td_lambda = depth, td_lambda
        self.leaf: Dict[Tuple[int, int], float] = defaultdict(float)
        self.errors = 0

    def wants_value(self, own: bool, own_index: int) -> bool:
        """Whether the value at this decision counts (only own decisions before the leaf depth, weighted)."""
        return bool(own) and self.depth > 0 and 0 <= own_index < self.depth and \
            value_weight(own_index, self.depth, self.td_lambda) > 0

    def add_value(self, root: int, continuation: int, own_index: int, value: float) -> None:
        self.leaf[(root, continuation)] += value_weight(own_index, self.depth, self.td_lambda) * float(value)

    def add_result(self, result: dict) -> Optional[float]:
        """Closes a continuation; returns its leaf value, or None when an env error ended it (no outcome)."""
        key = (int(result["root"]), int(result["continuation"]))
        if result["error"]:
            self.errors += 1
            self.leaf.pop(key, None)
            return None
        if not result["truncated"]:
            self.leaf[key] += outcome_weight(int(result["own_decisions"]), self.td_lambda) * \
                outcome(int(result["winner"]), int(result["root_player"]))
        return self.leaf[key]


def q_values(leaves: Iterable[Tuple[int, float]], legal: int) -> Tuple[np.ndarray, np.ndarray]:
    """(q, counts) over a root's legal actions from (root action, leaf value) pairs; q is 0 where no continuation
    finished (count 0)."""
    total, counts = np.zeros(legal), np.zeros(legal)
    for action, value in leaves:
        total[action] += value
        counts[action] += 1
    q = np.divide(total, counts, out=np.zeros(legal), where=counts > 0)
    return q, counts


def search_policy(q: np.ndarray, logits: np.ndarray, stepsize: float, temperature: float,
                  log_magnet: Optional[np.ndarray] = None, uniform_magnet: bool = False) -> np.ndarray:
    """The root's KL-regularized mirror-descent step over its legal actions (every array over the legal actions):
    softmax((logits + stepsize q + stepsize temperature log magnet) / (1 + stepsize temperature)); the magnet is
    uniform unless given. ``uniform_magnet``: softmax((logits + stepsize q) / (1 + temperature stepsize))."""
    q, logits = np.asarray(q, float), np.asarray(logits, float)
    if q.shape != logits.shape or q.ndim != 1 or not len(q):
        raise ValueError("q and logits are one value per legal action")
    if uniform_magnet:
        z = (logits + stepsize * q) / (1.0 + temperature * stepsize)
    else:
        if log_magnet is None:
            log_magnet = np.full(len(q), -np.log(len(q)))
        z = (logits + stepsize * q + temperature * stepsize * np.asarray(log_magnet, float)) / \
            (1.0 + temperature * stepsize)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()
