"""The value calibration monitor (mirrorforce/agent/train/calibration_monitor.py) against hand-computed scores."""
import numpy as np
import pytest

from mirrorforce.agent.train.calibration_monitor import CalibrationMonitor


def test_scores_against_the_outcome_from_each_seats_view():
    m = CalibrationMonitor(2)
    obs = lambda turns: {"global_": np.array([[0, 0, 0, 0, t] + [0] * 20 for t in turns], np.uint8)}
    # env 0: seat 0 predicts (0.7, 0.1, 0.2) at turn 1, seat 1 predicts (0.4, 0.2, 0.4) at turn 1 (second decision)
    m.record(obs([1, 3]), [[0.7, 0.1, 0.2], [0.5, 0.0, 0.5]], [0, 1], [True, True])
    m.record(obs([1, 3]), [[0.4, 0.2, 0.4], [0.5, 0.0, 0.5]], [1, 1], [False, False])
    # env 0 ends after a step by seat 1 with reward -1 for seat 1: seat 0 won
    m.finish([True, False], [-1.0, 0.0], [1, 1])
    report = m.interval()
    brier0 = (0.7 - 1) ** 2 + 0.1 ** 2 + 0.2 ** 2  # seat 0: win
    brier1 = 0.4 ** 2 + 0.2 ** 2 + (0.4 - 1) ** 2  # seat 1: loss
    assert report["games"] == 1 and report["decisions"] == 2
    assert report["brier"] == pytest.approx((brier0 + brier1) / 2, abs=1e-5)
    assert report["logloss"] == pytest.approx(-(np.log(0.7) + np.log(0.4)) / 2, abs=1e-5)
    assert report["by_turn"]["1-2"]["n"] == 2 and report["by_position"]["1-1"]["n"] == 1
    assert report["by_position"]["2-4"]["n"] == 1
    assert report["reliability"][7][0] == 1 and report["reliability"][7][2] == 1.0  # p_win 0.7 bin: a win
    assert report["reliability"][4][0] == 1 and report["reliability"][4][2] == 0.0  # p_win 0.4 bin: a loss
    m.finish([False, True], [0.0, 0.0], [1, 1])  # env 1: a draw, its two decisions carried across the interval
    assert m.interval()["decisions"] == 2
