"""The event window's overflow monitor: how long turns are, how often they leave the 512-row window and how many
64-row chunks they produce, overall and per deck.

Read from the observations and infos the actor already has (no environment change):

- completed turns: each closed window is delivered once to each observer; a turn is counted at its delivery to
  player 0 (``obs:closed_turn_meta_`` = (valid, turn, rows, chunks, turn player)); its length is
  ``chunks * chunk_rows + rows``;
- decisions: ``info:turn_events_dropped`` > 0 means the current turn has rows outside the window (all of them in
  chunks), ``info:turn_chunk_backlog`` > 0 means chunks are still queued for the deciding observer;
  ``info:closed_turn_backlog`` (builds that report it) > 0 means completed turns are still queued for it after this
  observation's deliveries (two per observation), so the decision reads a memory that lacks them.

``interval()`` returns the statistics since the last call (logged every log interval); ``cumulative()`` the
per-deck histograms since the start (written to a JSON file).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

BIN = 16  # rows per histogram bin


class WindowMonitor:
    def __init__(self, deck_names, window_rows: int, chunk_rows: int, chunk_cap: int):
        self.deck_names = list(deck_names)
        self.window_rows, self.chunk_rows, self.chunk_cap = window_rows, chunk_rows, chunk_cap
        self.bins = (window_rows + chunk_rows * chunk_cap) // BIN + 1
        self.histogram = np.zeros((len(self.deck_names), self.bins), np.int64)  # turn lengths by turn player's deck
        self.chunk_counts = np.zeros((len(self.deck_names), chunk_cap + 1), np.int64)
        self.reports_closed = False  # the build reports info:closed_turn_backlog
        self._reset_interval()

    def _reset_interval(self):
        self.lengths, self.chunks = [], []
        self.decisions = self.outside = self.backlog = 0
        self.closed_backlog, self.closed_backlog_max = 0, 0

    def observe(self, obs, info):
        """One actor step's observations (before the action) and infos."""
        meta = np.asarray(obs["closed_turn_meta_"]).astype(np.int64)
        to_play = np.asarray(info["to_play"]).astype(np.int64)
        counted = (meta[..., 0] > 0) & (to_play[:, None] == 0)
        if counted.any():
            lengths = meta[..., 3] * self.chunk_rows + meta[..., 2]
            seats = np.clip(meta[..., 4], 0, 1)
            decks = np.take_along_axis(np.asarray(info["deck"]).astype(np.int64), seats, axis=1)
            self.lengths.append(lengths[counted])
            self.chunks.append(meta[..., 3][counted])
            np.add.at(self.histogram, (decks[counted], np.minimum(lengths[counted] // BIN, self.bins - 1)), 1)
            np.add.at(self.chunk_counts, (decks[counted], np.minimum(meta[..., 3][counted], self.chunk_cap)), 1)
        self.decisions += to_play.size
        self.outside += int(np.count_nonzero(np.asarray(info["turn_events_dropped"]) > 0))
        self.backlog += int(np.count_nonzero(np.asarray(info["turn_chunk_backlog"]) > 0))
        if "closed_turn_backlog" in info:
            closed = np.asarray(info["closed_turn_backlog"])
            self.closed_backlog += int(np.count_nonzero(closed > 0))
            self.closed_backlog_max = max(self.closed_backlog_max, int(closed.max(initial=0)))
            self.reports_closed = True

    def interval(self):
        lengths = np.concatenate(self.lengths) if self.lengths else np.zeros(0, np.int64)
        chunks = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.int64)
        q = (lambda p: float(np.quantile(lengths, p))) if lengths.size else (lambda p: 0.0)
        out = {"turns": int(lengths.size), "rows_p50": q(0.5), "rows_p90": q(0.9), "rows_p99": q(0.99),
               "rows_max": int(lengths.max()) if lengths.size else 0,
               "turns_over_window": float(np.mean(chunks > 0)) if chunks.size else 0.0,
               "chunks_max": int(chunks.max()) if chunks.size else 0,
               "decisions": self.decisions,
               "decisions_outside_window": self.outside / max(self.decisions, 1),
               "decisions_with_backlog": self.backlog / max(self.decisions, 1),
               **({"decisions_with_closed_backlog": self.closed_backlog / max(self.decisions, 1),
                   "closed_backlog_max": self.closed_backlog_max} if self.reports_closed else {})}
        self._reset_interval()
        return out

    def cumulative(self):
        decks = {}
        for i, name in enumerate(self.deck_names):
            turns = int(self.histogram[i].sum())
            if turns:
                decks[name] = {"turns": turns, "rows_histogram": self.histogram[i].tolist(),
                               "chunk_counts": self.chunk_counts[i].tolist()}
        return {"bin_rows": BIN, "window_rows": self.window_rows, "chunk_rows": self.chunk_rows,
                "chunk_cap": self.chunk_cap, "decks": decks}

    def write(self, path):
        path = Path(path)
        temporary = path.with_name(path.name + ".part")
        temporary.write_text(json.dumps(self.cumulative(), sort_keys=True))
        os.replace(temporary, path)
