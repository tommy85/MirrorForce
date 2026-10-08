"""Calibration of the categorical value (win, draw, loss) against final outcomes, from the actor's own games.

Each decision's value is the behaviour policy's prediction before any training on that game (A0 trains on an
iteration's games only after they are collected), so the statistics are held out for the network that made them.
Per finished game, every recorded decision is scored against the outcome from its seat's view:

- Brier score (sum over the three classes of the squared error) and log loss;
- a reliability table of the predicted win probability (10 bins: count, mean prediction, observed win rate);
- both split by turn (1-2, 3-4, 5-8, 9+, from ``global_`` column 4) and by the decision's position within its turn
  (1, 2-4, 5-16, 17+).

``interval()`` returns the statistics since the last call (a CALIBRATION line); games still running are carried.
"""
from __future__ import annotations

import numpy as np

TURNS = ((1, 2), (3, 4), (5, 8), (9, 10 ** 9))
POSITIONS = ((1, 1), (2, 4), (5, 16), (17, 10 ** 9))
BINS = 10


def _bucket(value, ranges):
    for i, (low, high) in enumerate(ranges):
        if low <= value <= high:
            return i
    return len(ranges) - 1


class CalibrationMonitor:
    def __init__(self, envs: int):
        self.games = [[] for _ in range(envs)]  # per env: (p [3], seat, turn bucket, position bucket)
        self.last_turn = np.full(envs, -1)
        self.position = np.zeros(envs, np.int64)
        self._reset()

    def _reset(self):
        self.count = np.zeros((len(TURNS), len(POSITIONS)))
        self.brier = np.zeros((len(TURNS), len(POSITIONS)))
        self.logloss = np.zeros((len(TURNS), len(POSITIONS)))
        self.bins = np.zeros((BINS, 3))  # count, sum of predicted win, sum of wins
        self.finished = 0

    def record(self, obs, dists, seats, first):
        """One step's decisions: ``dists`` [B, 3] the acting seat's (win, draw, loss), ``seats`` [B] the acting seat,
        ``first`` [B] a new game starts at this decision (unfinished games before it are dropped)."""
        turn = np.asarray(obs["global_"])[:, 4].astype(np.int64)
        dists, seats, first = np.asarray(dists, np.float64), np.asarray(seats), np.asarray(first, bool)
        for env in range(len(self.games)):
            if first[env]:
                self.games[env] = []
                self.last_turn[env] = -1
            if turn[env] != self.last_turn[env]:
                self.last_turn[env], self.position[env] = turn[env], 0
            self.position[env] += 1
            self.games[env].append((dists[env], int(seats[env]), _bucket(int(turn[env]), TURNS),
                                    _bucket(int(self.position[env]), POSITIONS)))

    def finish(self, done, rewards, seats):
        """Games ending after this step: ``rewards`` [B] the outcome from the acting seat's view (``seats``)."""
        for env in np.flatnonzero(np.asarray(done, bool)):
            reward, seat = float(np.asarray(rewards)[env]), int(np.asarray(seats)[env])
            for p, s, tb, pb in self.games[env]:
                r = reward if s == seat else -reward
                truth = np.array([r > 0, r == 0, r < 0], np.float64)
                self.count[tb, pb] += 1
                self.brier[tb, pb] += float(((p - truth) ** 2).sum())
                self.logloss[tb, pb] -= float(np.log(max(float((p * truth).sum()), 1e-12)))
                b = min(int(p[0] * BINS), BINS - 1)
                self.bins[b] += (1.0, p[0], truth[0])
            self.games[env] = []
            self.finished += 1

    def interval(self):
        n = self.count.sum()
        mean = lambda a: round(float(a.sum() / n), 5) if n else None
        out = {"games": self.finished, "decisions": int(n), "brier": mean(self.brier), "logloss": mean(self.logloss),
               "by_turn": {f"{lo}-{hi if hi < 10 ** 9 else ''}": {
                   "n": int(self.count[i].sum()),
                   "brier": round(float(self.brier[i].sum() / self.count[i].sum()), 5) if self.count[i].sum() else None}
                   for i, (lo, hi) in enumerate(TURNS)},
               "by_position": {f"{lo}-{hi if hi < 10 ** 9 else ''}": {
                   "n": int(self.count[:, j].sum()),
                   "brier": round(float(self.brier[:, j].sum() / self.count[:, j].sum()), 5)
                   if self.count[:, j].sum() else None} for j, (lo, hi) in enumerate(POSITIONS)},
               "reliability": [[int(c), round(p / c, 4) if c else None, round(w / c, 4) if c else None]
                               for c, p, w in self.bins]}
        self._reset()
        return out
