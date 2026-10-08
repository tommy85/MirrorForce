"""Untimed Search ON/OFF evaluation: the released on-demand search without its clocks.

The released ten-second profile (``final_search``) cuts a search at its prompt window and its turn-work
ledger, inside a 600-second room. This profile keeps every algorithmic setting of that search (triggers,
particles, rollout allocation, depth, root update) and removes only the time limits: the room clock is
effectively unlimited, a prompt may search until its registered rollouts complete, and the turn ledger no
longer closes search early. It answers how much the same search gains when time is not the constraint.

Only an isolated same-model evaluation registers it; it is never a room or human-play setting.
"""
from __future__ import annotations

LAW = 'untimed-same-model-evaluation/v1'
ROOM_SECONDS = 36000
PROMPT_SECONDS = 3600.
DUEL_SECONDS = 21600
HUMAN_WAIT_SECONDS = 36060.


def profile():
    return {'law': LAW, 'room_seconds': ROOM_SECONDS, 'prompt_seconds': PROMPT_SECONDS,
            'duel_seconds': DUEL_SECONDS}


def settings():
    """The released search settings with the clock caps raised; everything else is unchanged."""
    from . import final_search as final
    result = final.settings('on')
    finalize = result['finalize_seconds']
    result.update(seconds=PROMPT_SECONDS - finalize, total_seconds=PROMPT_SECONDS, untimed_profile=profile())
    result['on_demand'] = {**result['on_demand'], 'uncertain_seconds': PROMPT_SECONDS,
                           'lethal_seconds': PROMPT_SECONDS, 'turn_seconds': float(ROOM_SECONDS)}
    return result


def is_untimed(settings_value):
    return isinstance(settings_value, dict) and settings_value.get('untimed_profile') is not None


def clocks(settings_value):
    """Room, complete-turn, prompt and whole-duel limits that a registration must use."""
    if is_untimed(settings_value):
        return {'room_seconds': ROOM_SECONDS, 'complete_turn_seconds': float(ROOM_SECONDS),
                'prompt_seconds': PROMPT_SECONDS, 'duel_seconds': DUEL_SECONDS}
    return {'room_seconds': 600, 'complete_turn_seconds': 600., 'prompt_seconds': 40., 'duel_seconds': 1800}
