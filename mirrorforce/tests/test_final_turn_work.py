"""Final450 is an all-prompt work ledger, not 450 extra rollout seconds."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mirrorforce.netduel import agent_final_room_budget as B, final_search as F
from mirrorforce.netduel import agent_search_gate as G, agent_search_policy as P


HINT = G.PublicCombatHint(True, False, 8000, (), frozenset())


def gate():
    return G.TurnSearchGate(P.SearchConfig(**B.settings('on')).on_demand)


def accrue(ledger, seconds):
    """Many confident, non-search prompts, with substantial peer waits."""
    count = 0
    while seconds:
        work = min(10., seconds)
        start = 100. + count * 100.
        ledger.begin_prompt(turn=1, started=start, search_deadline=start+10., response_deadline=start+13.)
        choice = ledger.decide(logits=[0., 20.], rows=2, hint=HINT, now=start)
        assert not choice.search
        ledger.finish_prompt(now=start+work, delivered=True, cleanup_complete=True)
        seconds -= work
        count += 1
    return count


def begin(ledger, *, turn=1, left=600):
    start = max(100., ledger.last_stamp + 100.)
    clock = B.receive_time_limit(bytes([0,0])+left.to_bytes(2,'little'), game_id='game',
        prompt_index=100, received_at=start, room_seconds=600)
    budget = B.allocate(game_id='game', prompt_index=100, player=0, prompt_started=start,
        now=start, clock=clock, clock_delivery='before_prompt', previous_response_at=start-1.,
        room_verified=True, room_seconds=600, enabled=True)
    ledger.begin_prompt(turn=turn, started=start, search_deadline=budget.search_deadline,
        response_deadline=budget.original_response_deadline,
        **({'optional_response_deadline': budget.response_deadline} if budget.allow_optional else {}))
    return start


def choose(ledger, at):
    return ledger.decide(logits=[0.,0.], rows=2, hint=HINT, now=at)


def test_identity_versions_distinguish_450_from_30_and_the_no_cap_draft():
    config = P.SearchConfig(**B.settings('on'))
    assert config.on_demand.turn_seconds == B.TURN_WORK_SECONDS == 450.
    assert config.seconds == 10. and config.total_seconds == 13.
    assert config.clock_reserve == 150. and B.ROOM_SECONDS == 600
    assert B.LAW == 'final-human-room600-search10-floor150-turn450/v2'
    assert B.PROFILE['turn_work_scope'] == B.TURN_WORK_SCOPE
    assert G.gate_law(config.on_demand) == G.TURN_WORK450_LAW
    assert G.gate_law(G.GateConfig()) == G.LAW
    assert G.gate_law(replace(config.on_demand, turn_seconds=None)) == G.NO_TURN_CAP_LAW
    assert (P.current_identity_schema(config), P.current_root_schema(config), P.current_client_schema(config)) == (
        'mirrorforce_current_root_identity/final-room-turn450',
        'mirrorforce_current_root_search/v7-final-room-turn450',
        'mirrorforce_current_root_client/final-room-turn450')
    assert F.settings('on')['on_demand']['turn_seconds'] == 30.
    assert B.settings() is None and F.settings() is None
    old = deepcopy(B.settings('on'))
    old['final_room_profile']['law'] = 'final-human-room600-search10-floor150/v1'
    with pytest.raises(ValueError, match='exact and explicit'):
        P.SearchConfig(**old)
    for limit in (None, 30., 449., 451.):
        changed = deepcopy(B.settings('on'))
        changed['on_demand']['turn_seconds'] = limit
        with pytest.raises(ValueError):
            P.SearchConfig(**changed)


@pytest.mark.parametrize('prior,base,allowed,soft,hard,reason', [
    (440., 1., True, 7., 10., 'uncertain'),
    (446., .25, True, 1., 4., 'uncertain'),
    (446., 1., False, 1., 13., 'no_search_budget'),
    (447., 0., False, 0., 13., 'no_search_budget'),
    (449., 0., False, 0., 13., 'no_search_budget'),
    (450., 0., False, 0., 13., 'no_search_budget'),
    (451., 0., False, 0., 13., 'previous_overrun'),
])
def test_boundary_preserves_three_seconds_for_send_and_cleanup(prior,base,allowed,soft,hard,reason):
    ledger = gate()
    accrue(ledger, prior)
    assert ledger.spent == prior
    start = begin(ledger)
    decision = choose(ledger, start+base)
    assert decision.search is allowed and decision.reason == reason
    assert decision.spent_this_turn == prior+base
    assert decision.search_deadline == start+soft
    assert decision.response_deadline == start+hard
    if allowed:
        assert prior + decision.response_deadline-start <= 450.
        assert decision.response_deadline-decision.search_deadline >= 3.


def test_confident_base_inference_counts_across_45_prompts_but_peer_wait_does_not():
    ledger = gate()
    assert accrue(ledger, 450.) == 45
    assert ledger.last_stamp > 4500. and ledger.spent == 450.
    start = begin(ledger)
    decision = choose(ledger, start+.5)
    assert not decision.search and decision.spent_this_turn == 450.5
    # Required greedy response/cleanup still happens. It cannot be erased or
    # mislabeled as search, and its real overrun blocks further optional work.
    finished = ledger.finish_prompt(now=start+1., delivered=True, cleanup_complete=True)
    assert finished['turn_work_seconds'] == 451. and finished['over_turn_cap']
    assert finished['law'] == G.TURN_WORK450_LAW and ledger.search_stopped


def test_preparation_and_subdecisions_cannot_restart_a_turn_or_prompt_window():
    ledger = gate()
    accrue(ledger, 440.)
    start = begin(ledger)
    first = choose(ledger, start+2.)
    second = choose(ledger, start+6.)
    stopped = choose(ledger, start+7.)
    assert first.search and second.search and not stopped.search
    assert first.search_deadline == second.search_deadline == start+7.
    assert first.response_deadline == second.response_deadline == start+10.
    assert (first.spent_this_turn,second.spent_this_turn,stopped.spent_this_turn) == (442.,446.,447.)


def test_server150_floor_can_stop_search_before_the_450_work_limit():
    ledger = gate()
    accrue(ledger, 30.)
    start = begin(ledger, left=150)
    decision = choose(ledger, start+.5)
    assert not decision.search and decision.reason == 'no_search_budget'
    assert decision.response_deadline == start+13.
    finished = ledger.finish_prompt(now=start+1., delivered=True, cleanup_complete=True)
    assert finished['turn_work_seconds'] == 31. and not finished['over_turn_cap']


def test_near_floor_and_turn_work_limit_take_the_tighter_deadline():
    ledger = gate()
    accrue(ledger, 440.)
    start = begin(ledger, left=154)
    decision = choose(ledger, start+.25)
    assert decision.search and decision.search_deadline == start+1.
    assert decision.response_deadline == start+4.


def test_new_turn_resets_450_ledger_not_the_low_server_clock():
    ledger = gate()
    accrue(ledger, 451.)
    start = begin(ledger, turn=2, left=150)
    assert ledger.spent == 0. and not ledger.search_stopped
    assert not choose(ledger, start).search
    ledger.finish_prompt(now=start+1., delivered=True, cleanup_complete=True)
    start = begin(ledger, turn=3, left=600)
    decision = choose(ledger, start+.5)
    assert decision.search and decision.spent_this_turn == .5
    assert decision.search_deadline == start+10. and decision.response_deadline == start+13.
