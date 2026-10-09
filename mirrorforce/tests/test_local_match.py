"""The local match tool's schedule and summary (``tools/mf_runtime_local_match.py``)."""
import pytest

from tools.mf_runtime_local_match import schedule, summarize


def test_schedule_plays_each_deal_from_both_sides():
    plan = schedule(4, 100)
    assert plan == [(0, 100, True), (1, 100, False), (2, 101, True), (3, 101, False)]
    for games in (0, 3):
        with pytest.raises(ValueError):
            schedule(games, 0)


def test_summary_splits_by_who_moved_first_and_keeps_failures_apart():
    rows = [{"a_first": True, "error": "", "winner": "a"},
            {"a_first": True, "error": "", "winner": "draw"},
            {"a_first": False, "error": "", "winner": "b"},
            {"a_first": False, "error": "TimeoutError: no reply"}]
    out = summarize(rows)
    assert out["a_first"] == {"games": 2, "wins": 1, "losses": 0, "draws": 1, "win_rate": 0.75}
    assert out["a_second"] == {"games": 1, "wins": 0, "losses": 1, "draws": 0, "win_rate": 0.0}
    assert out["total"] == {"games": 3, "wins": 1, "losses": 1, "draws": 1, "win_rate": 0.5}
    assert out["failures"] == 1


def test_summary_of_only_failures_has_no_rate():
    out = summarize([{"a_first": True, "error": "boom"}])
    assert out["total"]["win_rate"] is None and out["failures"] == 1
