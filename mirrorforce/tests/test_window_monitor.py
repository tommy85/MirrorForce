"""The window monitor's counters: decisions outside the window, with a chunk backlog and with a closed-turn backlog."""
from __future__ import annotations

import numpy as np

from mirrorforce.agent.train.window_monitor import WindowMonitor


def step(closed=None):
    obs = {"closed_turn_meta_": np.zeros((3, 2, 5), np.int32)}
    info = {"to_play": np.array([0, 1, 0]), "deck": np.zeros((3, 2), np.int64),
            "turn_events_dropped": np.array([0, 3, 0]), "turn_chunk_backlog": np.array([0, 0, 1])}
    if closed is not None:
        info["closed_turn_backlog"] = np.asarray(closed)
    return obs, info


def test_the_closed_turn_backlog_is_counted_when_the_build_reports_it():
    monitor = WindowMonitor(["d"], 512, 64, 48)
    monitor.observe(*step())
    out = monitor.interval()
    assert out["decisions_outside_window"] == 1 / 3 and out["decisions_with_backlog"] == 1 / 3
    assert "decisions_with_closed_backlog" not in out  # a build without the counter
    monitor.observe(*step([0, 5, 0]))
    monitor.observe(*step([2, 0, 0]))
    out = monitor.interval()
    assert out["decisions_with_closed_backlog"] == 2 / 6 and out["closed_backlog_max"] == 5
    monitor.observe(*step([0, 0, 0]))
    out = monitor.interval()
    assert out["decisions_with_closed_backlog"] == 0 and out["closed_backlog_max"] == 0
