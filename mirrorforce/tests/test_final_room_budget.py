"""Real-packet parsing and clock budgets; this is not a room integration gate."""
from dataclasses import replace
import pytest
from mirrorforce.netduel import agent_final_room_budget as B


def clock(seconds=600, *, at=100., player=0, game='game-a', prompt=0, room=600):
    raw=bytes([player,0])+seconds.to_bytes(2,'little')
    return B.receive_time_limit(raw,game_id=game,prompt_index=prompt,received_at=at,room_seconds=room)


def budget(**kwargs):
    args=dict(game_id='game-a',prompt_index=0,player=0,prompt_started=100.,now=102.,clock=clock(),
              clock_delivery='before_prompt',previous_response_at=90.,room_verified=True,room_seconds=600,enabled=True)
    args.update(kwargs)
    return B.allocate(**args)


def test_original_prompt_includes_preparation_and_preserves_cleanup():
    result=budget()
    assert result.allow_optional and result.remaining_seconds==598.
    assert result.search_deadline==110. and result.response_deadline==113.
    assert result.search_deadline-result.evaluated_at==8.
    assert result.response_deadline-result.search_deadline==3.


def test_floor_preserves_full_cleanup_not_only_the_rollout_window():
    result=budget(clock=clock(158))
    assert result.remaining_seconds==156.
    assert result.search_deadline==105. and result.response_deadline==108.
    assert 258.-result.response_deadline==150.
    assert result.response_deadline-result.search_deadline==3.


@pytest.mark.parametrize('seconds',[0,149,150,151,152,153,154,155])
def test_no_search_at_or_below_floor_or_without_complete_cleanup_reserve(seconds):
    result=budget(clock=clock(seconds))
    assert not result.allow_optional
    assert result.search_deadline<=result.evaluated_at
    assert not result.floor_enforced_for_optional
    # An ordinary response is still permitted if its actual deadline survives;
    # a zero clock does not become a fresh13-second response allowance.
    assert result.response_deadline==min(113.,100.+seconds-3.)


@pytest.mark.parametrize('sample,status',[(None,'missing'),(clock(game='old'),'other_game'),
    (clock(player=1),'other_player'),(clock(prompt=2),'stale_prompt'),(clock(at=103.),'future'),
    (clock(at=89.),'before_previous_response'),(clock(at=101.),'wrong_delivery_order'),
    (clock(450,room=450),'other_room_clock')])
def test_unknown_or_unrelated_clock_never_admits_optional_work(sample,status):
    result=budget(clock=sample)
    assert not result.allow_optional and result.clock_status==status
    assert result.remaining_seconds is None


def test_valid_earlier_clock_is_not_mechanically_classified_stale():
    result=budget(clock=clock(600,at=91.))
    assert result.allow_optional and result.clock_status=='valid' and result.remaining_seconds==589.


def test_prompt_then_clock_keeps_original_start_and_rejects_previous_prompt_clock():
    result=budget(clock=clock(at=101.),clock_delivery='prompt_then_clock')
    assert result.allow_optional and result.search_deadline==110. and result.response_deadline==113.
    assert not budget(clock=clock(at=99.),clock_delivery='prompt_then_clock').allow_optional


def test_new_clock_and_subdecision_cannot_extend_an_already_shorter_floor_deadline():
    first=budget(clock=clock(158))
    second=budget(clock=clock(600,at=103.),clock_delivery='prompt_then_clock',now=104.,previous=first)
    assert second.allow_optional and second.search_deadline==first.search_deadline==105.
    assert second.response_deadline==first.response_deadline==108.
    finished=budget(clock=clock(600,at=103.),clock_delivery='prompt_then_clock',now=106.,previous=second)
    assert not finished.allow_optional and finished.response_deadline==108.
    assert finished.search_deadline<=second.search_deadline
    missing=budget(clock=None,now=106.,previous=second)
    disabled=budget(enabled=False,now=106.,previous=second)
    for stopped in (missing,disabled):
        assert not stopped.allow_optional and stopped.search_deadline<=second.search_deadline
        assert stopped.response_deadline<=second.response_deadline
    with pytest.raises(ValueError):budget(previous=replace(first,game_id='another'))


def test_default_off_and_missing_clock_are_not_room_or_protocol_admission():
    assert B.prepared_profile()['search']=='off'
    assert not B.prepared_profile('on')['runtime_integrated']
    assert B.prepared_profile('on')['turn_work_cap_seconds'] == 450.
    assert B.prepared_profile('on')['turn_work_scope'] == B.TURN_WORK_SCOPE
    assert not budget(enabled=False).allow_optional
    assert not budget(clock=None).allow_optional
    with pytest.raises(ValueError):budget(room_verified=False)
    with pytest.raises(ValueError):budget(room_seconds=450)
    with pytest.raises(ValueError):budget(clock={'seconds':600})


@pytest.mark.parametrize('payload',[b'',b'\x00\x00\xff\xff',b'\x02\x00\x58\x02',b'\x00\x58\x02'])
def test_malformed_packets_are_errors_not_silent_clock_fallback(payload):
    with pytest.raises(ValueError):
        B.receive_time_limit(payload,game_id='game-a',prompt_index=0,received_at=100.,room_seconds=600)


def test_no_hidden_thirty_second_cap_and_no_model_value_input():
    first=budget()
    later=B.allocate(game_id='game-a',prompt_index=99,player=0,prompt_started=400.,now=402.,
        clock=clock(300,at=400.,prompt=99),clock_delivery='before_prompt',previous_response_at=395.,
        room_verified=True,room_seconds=600,enabled=True)
    assert first.allow_optional and later.allow_optional
    assert later.search_deadline==410. and later.response_deadline==413.
    with pytest.raises(TypeError):budget(model_win_probability=.99)
