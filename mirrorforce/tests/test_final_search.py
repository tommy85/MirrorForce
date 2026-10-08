"""Final-human switch and actual gate deadlines, without an engine or GPU."""
import builtins
from dataclasses import replace

import pytest

from mirrorforce.netduel import final_search as F
from mirrorforce.netduel import agent_search_gate as G


def test_off_never_imports_or_constructs_a_search_follower(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if 'search_policy' in name or 'stripe_batching' in name:
            raise AssertionError('OFF constructed search dependencies')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    assert F.settings() is None and F.settings('off') is None
    assert F.profile()['admitted'] is False
    with pytest.raises(ValueError):
        F.settings('auto')


@pytest.mark.parametrize('lethal', [False, True])
def test_both_triggers_share_ten_seconds_including_preparation_then_three_cleanup(lethal):
    cfg = F.settings('on')
    gate = G.TurnSearchGate(G.GateConfig(**cfg['on_demand']))
    gate.begin_prompt(turn=1, started=100., search_deadline=137., response_deadline=140.)
    hint = G.PublicCombatHint(True, True, 8000, (1500,), frozenset({'attack'}))
    if lethal:
        hint = replace(hint, opponent_lp=1000)
    decision = gate.decide(logits=[9., 0.] if lethal else [0., 0.], rows=2, hint=hint, now=102.)
    assert decision.search and decision.reason == ('lethal_candidate' if lethal else 'uncertain')
    assert decision.search_deadline == 110. and decision.response_deadline == 113.
    assert decision.search_deadline - 102. == 8.  # Original preparation consumed two seconds.
    finished = gate.finish_prompt(now=112.5, delivered=True, cleanup_complete=True)
    assert finished['prompt_seconds'] == 12.5 and finished['turn_work_seconds'] == 12.5
    assert cfg['seconds'] == 10. and cfg['total_seconds'] == 13.
    assert cfg['finalize_seconds'] == cfg['response_margin'] == 3.


def test_final_profile_keeps_confident_and_singleton_fast_and_does_not_extend_clock():
    gate = G.TurnSearchGate(G.GateConfig(**F.settings('on')['on_demand']))
    hint = G.PublicCombatHint(True, True, 8000, (1500,), frozenset({'attack'}))
    gate.begin_prompt(turn=1, started=100., search_deadline=106., response_deadline=109.)
    assert gate.decide(logits=[10., 0.], rows=2, hint=hint, now=100.5).reason == 'confident'
    assert gate.decide(logits=[0.], rows=1, hint=hint, now=100.6).reason == 'singleton'
    decision = gate.decide(logits=[0., 0.], rows=2, hint=hint, now=101.)
    assert decision.search_deadline == 106. and decision.response_deadline == 109.


def test_profile_law_binds_the_full_actual_identity_without_relabelling_old_service_budget():
    identity={'settings':F.settings('on'),'actor':'first','service_budget':{'uncertain':8,'lethal':12}}
    first=F.bind_identity(identity)
    assert first['law']==F.LAW and not first['production_admitted'] and not first['complete_game_clock_admitted']
    second=F.bind_identity({**identity,'actor':'second'})
    assert first['search_identity_sha256']!=second['search_identity_sha256']
    identity['settings']['on_demand']['uncertain_seconds']=8.
    with pytest.raises(ValueError):F.bind_identity(identity)
