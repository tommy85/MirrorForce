"""Strict current public counts/positions; no belief, hidden identity or native core."""
from copy import deepcopy
import struct
from types import SimpleNamespace as NS

import pytest

from mirrorforce.netduel import constants as C, agent_opponent_clear as O
from mirrorforce.netduel import agent_final_room_budget as F, agent_search_gate as G, agent_search_policy as P
from mirrorforce.netduel.board import ShadowBoard, ShadowCard


def example(viewer=0):
    board=ShadowBoard();board.start(viewer,[1]*40,[2]*15)
    client=NS(board=board,room_started=True,result=NS(our_player=viewer),board_problems=[])
    start=bytes([viewer,5])+struct.pack('<IIHHHH',8000,8000,40,15,40,15)
    return client,[(C.MSG_START,start),(C.MSG_SELECT_YESNO,bytes([viewer])+struct.pack('<I',30))]


def card(viewer,location,sequence=0,position=1,**kwargs):
    return ShadowCard(controller=1-viewer,location=location,sequence=sequence,
                      position=position,**({'code':123}|kwargs))


@pytest.mark.parametrize('viewer',[0,1])
@pytest.mark.parametrize('size',[0,1,2])
def test_hand_must_be_empty_even_when_every_hand_identity_is_known(viewer,size):
    client,messages=example(viewer)
    client.board.zones[(1-viewer,C.LOCATION_HAND)]=[card(viewer,C.LOCATION_HAND,i) for i in range(size)]
    class NoLedger:
        def __getattribute__(self,name):raise AssertionError('must not consult disclosed/hidden identities')
    client.board.disclosure=NoLedger()
    result=O.extract(client,messages)
    assert result['facts']['hand_count']==size and result['allow_search']==(size==0)
    assert result['reason']==('opponent_clear' if not size else 'opponent_hand_nonempty')


@pytest.mark.parametrize('viewer',[0,1])
@pytest.mark.parametrize('location,slot',[(C.LOCATION_MZONE,i) for i in range(7)]+[(C.LOCATION_SZONE,i) for i in range(8)])
@pytest.mark.parametrize('position,allowed',[(1,True),(4,True),(5,True),(2,False),(8,False),(10,False)])
def test_all_monster_spell_field_and_pendulum_slots_and_public_positions(viewer,location,slot,position,allowed):
    client,messages=example(viewer)
    client.board.zones[(1-viewer,location)]=[None]*slot+[card(viewer,location,slot,position)]
    result=O.extract(client,messages)
    assert result['allow_search'] is allowed
    if not allowed:assert result['reason']=='opponent_facedown_field'


@pytest.mark.parametrize('position',[0,3,6,7,9,15,16,-1,True])
def test_unknown_or_mixed_positions_fail_closed(position):
    client,messages=example()
    client.board.zones[(1,C.LOCATION_MZONE)]=[card(0,C.LOCATION_MZONE,position=position)]
    result=O.extract(client,messages)
    assert not result['allow_search'] and result['status']=='malformed_public_card'


@pytest.mark.parametrize('code',[0,-1,True,2**32,999000001,999000002,999000003,999000004])
def test_unknown_or_follower_placeholder_identity_cannot_look_public(code):
    client,messages=example()
    client.board.zones[(1,C.LOCATION_SZONE)]=[card(0,C.LOCATION_SZONE,code=code)]
    result=O.extract(client,messages)
    assert not result['allow_search'] and result['reason']=='opponent_field_unknown'


def test_facedown_identity_is_not_read_and_own_hidden_cards_do_not_block():
    client,messages=example()
    client.board.zones[(0,C.LOCATION_HAND)]=[ShadowCard(code=0,hidden=True)]
    client.board.zones[(0,C.LOCATION_MZONE)]=[ShadowCard(code=0,hidden=True,position=8)]
    assert O.extract(client,messages)['allow_search']
    hidden=card(0,C.LOCATION_MZONE,position=8,code=object(),hidden=True)
    client.board.zones[(1,C.LOCATION_MZONE)]=[hidden]
    assert O.extract(client,messages)['reason']=='opponent_facedown_field'
    hidden.position=1
    assert O.extract(client,messages)['reason']=='opponent_field_unknown'


@pytest.mark.parametrize('fault',['missing_hand','missing_field','none_hand','none_card','large_zone',
    'wrong_controller','wrong_location','wrong_sequence','wrong_viewer','unstarted','empty_board',
    'board_problem','slot_problem','mismatch','missing_stream','wrong_start','private_extra',
    'not_a_prompt','conflicting_position','unrepresented_occupied_slot','other_seat_prompt'])
def test_missing_contradictory_or_malformed_information_never_enables_search(fault):
    client,messages=example()
    if fault=='missing_hand':del client.board.zones[(1,C.LOCATION_HAND)]
    elif fault=='missing_field':del client.board.zones[(1,C.LOCATION_SZONE)]
    elif fault=='none_hand':client.board.zones[(1,C.LOCATION_HAND)]=None
    elif fault=='none_card':client.board.zones[(1,C.LOCATION_HAND)]=[None]
    elif fault=='large_zone':client.board.zones[(1,C.LOCATION_SZONE)]=[None]*9
    elif fault in ('wrong_controller','wrong_location','wrong_sequence'):
        value=card(0,C.LOCATION_MZONE)
        setattr(value,{'wrong_controller':'controller','wrong_location':'location','wrong_sequence':'sequence'}[fault],0)
        if fault=='wrong_sequence':value.sequence=1
        client.board.zones[(1,C.LOCATION_MZONE)]=[value]
    elif fault=='wrong_viewer':client.result.our_player=1
    elif fault=='unstarted':client.room_started=False
    elif fault=='empty_board':client.board=ShadowBoard()
    elif fault=='board_problem':client.board_problems=['bad public packet']
    elif fault=='slot_problem':client.board.slot_problems=['bad public slot']
    elif fault=='mismatch':client.board.mismatches=1
    elif fault=='missing_stream':messages=[]
    elif fault=='wrong_start':messages[0]=(C.MSG_START,b'')
    elif fault=='not_a_prompt':messages[-1]=(C.MSG_NEW_TURN,b'\0')
    elif fault=='other_seat_prompt':messages[-1]=(C.MSG_SELECT_YESNO,b'\1'+struct.pack('<I',30))
    elif fault=='conflicting_position':
        client.board.zones[(1,C.LOCATION_MZONE)]=[card(0,C.LOCATION_MZONE)]
        client.board.positions[(1,C.LOCATION_MZONE,0)]=8
    elif fault=='unrepresented_occupied_slot':client.board.positions[(1,C.LOCATION_SZONE,0)]=8
    else:client.board=NS(zones=client.board.zones,revision=1,private_truth='forbidden')
    result=O.extract(client,messages)
    assert not result['allow_search'] and result['reason']=='opponent_info_unknown'


def test_real_public_draw_move_flip_and_departure_update_the_gate():
    client,messages=example()
    def event(msg,body):
        client.board.apply(msg,body)
        messages.insert(-1,(msg,body))
        return O.extract(client,messages)
    assert O.extract(client,messages)['allow_search']
    assert event(C.MSG_DRAW,b'\1\1'+bytes(4))['reason']=='opponent_hand_nonempty'
    pack=lambda loc,pos: 1 | loc<<8 | pos<<24
    moved=struct.pack('<IIII',0,pack(C.LOCATION_HAND,10),pack(C.LOCATION_SZONE,8),0)
    assert event(C.MSG_MOVE,moved)['reason']=='opponent_facedown_field'
    # A public position change alone does not fabricate the identity blanked
    # by the earlier draw/set; it remains unknown until the public query.
    assert event(C.MSG_POS_CHANGE,struct.pack('<I',123)+bytes([1,C.LOCATION_SZONE,0,8,1]))['reason']=='opponent_field_unknown'
    card_now=client.board.zones[(1,C.LOCATION_SZONE)][0]
    card_now.code=123;card_now.hidden=False  # The corresponding public query's fields.
    assert O.extract(client,messages)['allow_search']
    assert event(C.MSG_MOVE,struct.pack('<IIII',123,pack(C.LOCATION_SZONE,1),pack(C.LOCATION_GRAVE,1),0))['allow_search']


@pytest.mark.parametrize('fault',['allow','reason','missing','extra','bool_count','bool_opponent','missing_field'])
def test_forged_witness_is_rejected_instead_of_reinterpreted(fault):
    client,messages=example();result=O.extract(client,messages)
    if fault=='allow':result['allow_search']=False
    elif fault=='reason':result['reason']='opponent_hand_nonempty'
    elif fault=='missing':del result['facts']
    elif fault=='extra':result['host_truth']={}
    elif fault=='bool_count':result['facts']['hand_count']=False
    elif fault=='bool_opponent':result['opponent']=True
    else:result['facts']['fields'].pop()
    with pytest.raises((ValueError,TypeError)):O.check(result)


def test_global_gate_precedes_both_uncertainty_and_public_lethal_hints():
    client,messages=example();client.board.zones[(1,C.LOCATION_HAND)]=[card(0,C.LOCATION_HAND)]
    witness=O.extract(client,messages)
    for logits in ([0.,0.],[20.,0.]):
        gate=G.TurnSearchGate(P.SearchConfig(**O.settings('on')).on_demand)
        gate.begin_prompt(turn=1,started=100.,search_deadline=110.,response_deadline=113.)
        hint=G.PublicCombatHint(True,True,100,(8000,),frozenset({'attack'}))
        decision=gate.decide(logits=logits,rows=2,hint=hint,now=101.,opponent_clear=witness)
        assert decision.lethal_candidate and decision.trigger=='lethal_candidate'
        assert not decision.search and decision.reason=='opponent_hand_nonempty'


def test_version_is_explicit_and_preserves450_room_clock_and_legacy_settings():
    old=F.settings('on');new=O.settings('on')
    assert 'opponent_clear_profile' not in old and old=={k:v for k,v in new.items() if k!='opponent_clear_profile'}
    assert P.search_settings(P.SearchConfig(**old))==old
    assert P.search_settings(P.SearchConfig(**new))==new
    assert P.current_identity_schema(P.SearchConfig(**old))=='mirrorforce_current_root_identity/final-room-turn450'
    assert P.current_identity_schema(P.SearchConfig(**new))=='mirrorforce_current_root_identity/opponent-clear'
    assert new['on_demand']['turn_seconds']==450. and new['seconds']==10. and new['total_seconds']==13.
    assert new['clock_reserve']==150. and O.settings() is None
    assert O.PROFILE['future_hidden_information_known'] is False and O.PROFILE['guaranteed_lethal'] is False
    wrong=deepcopy(new);wrong.pop('final_room_profile')
    with pytest.raises(ValueError):P.SearchConfig(**wrong)
