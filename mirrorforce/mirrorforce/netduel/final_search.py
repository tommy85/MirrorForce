"""Prepared final-human search switch; settings alone never confer admission.

The ten-second window starts with the original prompt and includes preparation
and the ordinary policy forward. Existing cleanup/send has a separate three-
second reserve. The public trigger remains a candidate, not a lethal proof.
Training and ordinary evaluation use pure policy; an explicitly registered
same-model diagnostic may measure this final-human configuration offline.
"""
from __future__ import annotations

LAW = 'final-human-on-demand-10s/v1'


def settings(search='off'):
    """Return no search object by default; ON still needs real model-bound gates."""
    if search not in ('off', 'on'):
        raise ValueError('explicit search=off|on required')
    if search == 'off':
        return None
    # Keep the pure-policy path free of search/follower construction or imports.
    from dataclasses import asdict
    from .agent_search_policy import SearchConfig
    from .agent_search_gate import GateConfig
    from .agent_stripe_batching import LAW as BUDGET_LAW
    from ..common.client_public_recipe import LAW as MIRROR_LAW

    config = SearchConfig(seconds=10., total_seconds=13., selection='greedy',
        budget_law=BUDGET_LAW, candidate_law='all-legal-common-bank/v1', particles=8,
        follower_recipe_law=MIRROR_LAW,
        on_demand=GateConfig(enabled=True, uncertain_seconds=13., lethal_seconds=13.))
    result = asdict(config)
    # Absence retains the released search variant; no experimental tactical arm.
    result.pop('tactical', None)
    result.pop('final_room_profile', None)
    result.pop('opponent_clear_profile', None)
    result.pop('opponent_known_profile', None)
    result.pop('untimed_profile', None)
    return result


def profile(search='off'):
    return {'law': LAW, 'search': search, 'settings': settings(search),
            'scope': 'final-human-play; explicit isolated same-model evaluation only',
            'search_window_seconds_including_preparation_and_policy': 10.,
            'cleanup_and_send_reserve_seconds': 3., 'whole_prompt_seconds': 13.,
            'turn_work_seconds': 30., 'training_eligible': False,
            'admitted': False, 'guaranteed_lethal': False,
            'admission': 'requires original model-bound native terminal, numeric, stripe and clock evidence'}


def bind_identity(identity):
    """Declare this client profile against its full separately registered identity.

    The service's historical request is retained as provenance, never rewritten
    to claim that its old budget or a different actor passed this runtime gate.
    """
    import hashlib
    import json
    if type(identity) is not dict or identity.get('settings') != settings('on'):
        raise ValueError('final-human profile needs its actual exact ten-second search identity')
    raw = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {'law': LAW, 'search_identity_sha256': hashlib.sha256(raw).hexdigest(),
            'service_request_preserved': True, 'production_admitted': False,
            'complete_game_clock_admitted': False, 'training_eligible': False}
