import pytest
from mirrorforce.effectinfo import get_effectinfo_core
from mirrorforce.netduel import constants as C
from mirrorforce.netduel.actions import ActionAct as A
from mirrorforce.agent.train.sky_command_timing import demonstrate_command_timing, MAXX_C, ENGAGE, UPSTART
from test_sky_combo_curriculum import deck


def run(variant):
    return demonstrate_command_timing(get_effectinfo_core(), deck(),
        {'family':'command_timing','variant':variant,'seed':2026125001,'start_lp':8000,
         'defenses':[98338152,84749824],'spare':14558127,'first_draw':97268402})


@pytest.mark.parametrize('variant,turn', [('own_turn_g_chain',2),('opponent_turn_g_chain',3)])
def test_g_is_used_in_response_to_a_real_opponent_special_summon_effect_and_draws(variant,turn):
    r=run(variant)
    plays=[x for x in r['choices'] if x['player']==1 and x['action']['code']==MAXX_C and x['action']['act']==A.ACTIVATE]
    assert len(plays)==1 and plays[0]['turn']==turn and plays[0]['msg']==C.MSG_SELECT_CHAIN
    assert any(d['player']==1 and d['turn']==turn and d['phase']>=C.PHASE_MAIN1 and d['count']>=1 for d in r['draws'])
    assert r['lp'][0]==6500


def test_no_opponent_action_means_no_wasted_free_battle_maxx_c():
    r=run('idle_g_hold')
    assert not any(x['player']==1 and x['action']['code']==MAXX_C and x['action']['act']==A.ACTIVATE for x in r['choices'])
    assert MAXX_C in r['own_hand'] and MAXX_C not in r['own_grave'] and r['lp'][0]==6500


@pytest.mark.parametrize('variant,code',[('main1_upstart',UPSTART),('engage_direct',ENGAGE)])
def test_useful_hand_spell_is_used_directly_in_main1_before_battle_or_sets(variant,code):
    r=run(variant)
    commands=[x for x in r['choices'] if x['player']==1 and x['msg']==C.MSG_SELECT_IDLECMD]
    first=commands[0]
    assert first['phase']==C.PHASE_MAIN1 and first['action']['code']==code and first['action']['act']==A.ACTIVATE
    assert first['action']['spec'].startswith('h')
    assert not any(x['action']['code']==code and x['action']['act']==A.SET for x in commands)
    assert r['lp'][0]<9000


def test_main2_engage_is_retained_when_battle_enables_its_extra_draw():
    r=run('main2_engage_value')
    engages=[x for x in r['choices'] if x['player']==1 and x['msg']==C.MSG_SELECT_IDLECMD
             and x['action']['code']==ENGAGE and x['action']['act']==A.ACTIVATE]
    assert len(engages)==1 and engages[0]['phase']==C.PHASE_MAIN2
    assert any(x['player']==1 and x['msg']==C.MSG_SELECT_YESNO and x['action']['code']==ENGAGE
               and x['action']['act']==A.ACTIVATE for x in r['choices'])
    assert any(d['player']==1 and d['turn']==2 and d['phase']==C.PHASE_MAIN2 for d in r['draws'])
