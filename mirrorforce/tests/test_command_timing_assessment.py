"""Synthetic scoring negatives, not evidence of learned model quality."""
import copy
import pytest
from mirrorforce.netduel import constants as C
from mirrorforce.netduel.actions import ActionAct as A, ActionPhase as P
from mirrorforce.agent.train.sky_command_timing import assess_command_scene,MAXX_C,RAYE,UPSTART,ENGAGE


def row(code=0,act=A.DIRECT_ATTACK,phase=C.PHASE_BATTLE_START,msg=C.MSG_SELECT_BATTLECMD,turn=2,**extra):
    return {'turn':turn,'turn_player':1 if turn%2==0 else 0,'phase':phase,'msg':msg,
            'action':{'code':code,'act':int(act),'spec':'h1'},**extra}


def assessment(variant):
    scene={'variant':variant,'start_lp':8000};decisions=[row()];draws=[];hand=[MAXX_C];lp=[6500,8000]
    if variant in ('own_turn_g_chain','opponent_turn_g_chain'):
        turn=2 if variant=='own_turn_g_chain' else 3
        decisions.insert(0,row(MAXX_C,A.ACTIVATE,msg=C.MSG_SELECT_CHAIN,turn=turn,
                              chain_sources=[{'code':RAYE,'controller':0}]))
        draws=[{'player':1,'turn':turn,'phase':C.PHASE_BATTLE_START if turn==2 else C.PHASE_MAIN1,'count':1}];hand=[]
    elif variant in ('main1_upstart','engage_direct'):
        decisions.insert(0,row(UPSTART if variant=='main1_upstart' else ENGAGE,A.ACTIVATE,
                              phase=C.PHASE_MAIN1,msg=C.MSG_SELECT_IDLECMD))
        if variant=='main1_upstart':lp[0]=7500
    elif variant=='main2_engage_value':
        bridge=row(act=A.NONE,phase=C.PHASE_MAIN1,msg=C.MSG_SELECT_IDLECMD,own_grave_spells=2)
        bridge['action']['phase']=int(P.BATTLE)
        decisions.insert(0,bridge)
        decisions.append(row(ENGAGE,A.ACTIVATE,phase=C.PHASE_MAIN2,msg=C.MSG_SELECT_IDLECMD,own_grave_spells=3))
        draws=[{'player':1,'turn':2,'phase':C.PHASE_MAIN2,'count':1}];lp[0]=7500
    return [scene,decisions,draws,hand,lp]


@pytest.mark.parametrize('variant',['idle_g_hold','own_turn_g_chain','opponent_turn_g_chain',
                                  'main1_upstart','engage_direct','main2_engage_value'])
def test_explicit_goals_and_legitimate_own_turn_g_or_main2_engage(variant):
    assert assess_command_scene(*assessment(variant))==[]


@pytest.mark.parametrize('variant,fault',[
 ('idle_g_hold','waste_g'),('own_turn_g_chain','free_command'),
 ('own_turn_g_chain','wrong_chain_owner'),
 ('main1_upstart','late'),('engage_direct','set_first'),
 ('engage_direct','backrow_activation'),('main2_engage_value','already_three'),
 ('main2_engage_value','still_two'),('main2_engage_value','no_draw'),('idle_g_hold','no_attack'),
 ('main1_upstart','no_damage')])
def test_actual_goal_violations_fail(variant,fault):
    args=copy.deepcopy(assessment(variant));scene,decisions,draws,hand,lp=args
    if fault=='waste_g':decisions.append(row(MAXX_C,A.ACTIVATE))
    elif fault=='free_command':decisions[0]['msg']=C.MSG_SELECT_BATTLECMD
    elif fault=='wrong_chain_owner':decisions[0]['chain_sources'][0]['controller']=1
    elif fault=='no_draw':draws.clear()
    elif fault=='late':decisions[0]['phase']=C.PHASE_MAIN2
    elif fault=='set_first':decisions.insert(0,row(ENGAGE,A.SET,phase=C.PHASE_MAIN1,msg=C.MSG_SELECT_IDLECMD))
    elif fault=='backrow_activation':decisions[0]['action']['spec']='s1'
    elif fault=='already_three':decisions[0]['own_grave_spells']=3
    elif fault=='still_two':decisions[-1]['own_grave_spells']=2
    elif fault=='no_attack':decisions.clear()
    elif fault=='no_damage':lp[0]=9000
    assert assess_command_scene(*args)


@pytest.mark.parametrize('turn',[1,3,7])
def test_proactive_opponent_turn_g_is_not_a_misplay_even_without_draws(turn):
    args=assessment('idle_g_hold')
    args[1].append(row(MAXX_C,A.ACTIVATE,msg=C.MSG_SELECT_CHAIN,turn=turn,chain_sources=[]))
    args[3].clear()
    assert assess_command_scene(*args)==[]


@pytest.mark.parametrize('turn',[2,6,8])
def test_free_g_on_our_turn_is_the_actual_no_waste_violation(turn):
    args=assessment('idle_g_hold');args[1].append(row(MAXX_C,A.ACTIVATE,turn=turn))
    assert any('on our turn' in x for x in assess_command_scene(*args))


@pytest.mark.parametrize('variant',['own_turn_g_chain','opponent_turn_g_chain'])
def test_positive_demonstration_does_not_force_g_over_alternative_interruption(variant):
    args=assessment(variant)
    args[1]=[d for d in args[1] if d['action']['code']!=MAXX_C]
    args[2].clear();args[3]=[MAXX_C]
    assert assess_command_scene(*args)==[]


def test_hand_absence_or_no_draw_does_not_alone_prove_wasted_g():
    args=assessment('idle_g_hold');args[3].clear()
    assert assess_command_scene(*args)==[]
    args=assessment('own_turn_g_chain');args[2].clear()
    assert assess_command_scene(*args)==[]


def test_missing_turn_owner_is_not_guessed_from_the_selected_action():
    args=assessment('own_turn_g_chain');del args[1][0]['turn_player']
    with pytest.raises(ValueError,match='turn owner'):assess_command_scene(*args)
