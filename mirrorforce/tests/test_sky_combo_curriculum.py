"""A played full-deck Halq/Bulb/Linkuriboh/Borrelsword chain, not learned weights."""
from pathlib import Path
import json

import pytest

from mirrorforce.effectinfo import get_effectinfo_core
from mirrorforce.netduel.cards import load_ydk
from mirrorforce.agent.train import sky_combo_curriculum as S
from mirrorforce.worldmodel.engine import DeckList

PRIVILEGED_TARGET = True


def deck():
    main,extra,_ = load_ydk(Path(__file__).parents[1]/'decks/stage-a/SkyStriker.ydk')
    return DeckList('independent-combo-curriculum',tuple(main),tuple(extra))


@pytest.mark.parametrize('seed,viewer',[(2026100501,0),(2026100502,1)])
def test_entire_link_chain_is_native_legal_in_an_ordinary_full_deck_game(seed,viewer):
    result = S.demonstrate(get_effectinfo_core(),deck(),seed=seed,viewer=viewer)
    assert [c['code'] for c in result['field']] == [S.BORRELSWORD], result
    assert all(code in result['grave'] for code in (S.HALQ,S.BULB,S.LINKURIBOH)), result
    assert not result['debug_board'] and result['ordinary_duel_rules'] and result['model_updates'] == 0
    assert set(result['public_tape']) == {'schema','viewer','own_recipe','opponent_recipe_mode',
                                        'messages','responses','terminal','winner','private_layouts'}
    assert result['public_tape']['private_layouts'] is False
    assert result['public_tape']['viewer'] == viewer
    assert len(result['public_tape']['own_recipe']['main']) == 40
    assert result['public_tape']['messages'][0][0] == 4  # actual MSG_START
    print('SKY-COMBO '+json.dumps(result,sort_keys=True))


@pytest.mark.parametrize('lp,win,remaining',[(7500,True,0),(8000,False,500)])
def test_link_chain_needs_a_second_attack_target_and_sufficient_damage(lp,win,remaining):
    result=S.demonstrate(get_effectinfo_core(),deck(),seed=2026100503,viewer=1,
                         start_lp=lp,finish_lethal=True)
    assert (result['winner']==1)==win,result
    assert result['lp'][0]==remaining,result
    assert result['turn']==2,result
    print('SKY-COMBO-LETHAL '+json.dumps(result,sort_keys=True))


@pytest.mark.parametrize('lp,win,remaining',[(1500,True,0),(2000,False,500)])
def test_bomber_full_deck_route_burns_a_normally_set_defender(lp,win,remaining):
    result=S.demonstrate(get_effectinfo_core(),deck(),seed=2026100513,viewer=1,
                         start_lp=lp,finish_lethal=True,finisher=S.BOMBER,opponent='set_raye')
    assert result['lp'][0]==remaining and (result['winner']==1)==win,result
    assert result['turn']==2 and result['ordinary_duel_rules'] and not result['debug_board']


def test_complete_combo_keeps_quick_spell_in_hand_until_main2():
    from mirrorforce.netduel import constants as C
    from mirrorforce.netduel.actions import ActionAct as A
    result=S.demonstrate(get_effectinfo_core(),deck(),seed=2026111001,viewer=1,
        start_lp=8000,finish_lethal=True,finish_main2=True)
    sets=[r for r in result['choices'] if r['player']==1 and r['action']['act']==A.SET]
    assert len(sets)==1 and sets[0]['action']['code']==98338152
    assert sets[0]['phase']==C.PHASE_MAIN2 and result['lp'][0]==500
    assert result['winner'] is None and not result['public_tape']['terminal']


def test_clean_main2_ends_turn_without_setting_spare_field_or_normal_spells():
    from mirrorforce.netduel import constants as C
    from mirrorforce.netduel.actions import ActionAct as A, ActionPhase as P
    result=S.demonstrate(get_effectinfo_core(),deck(),seed=2026113001,viewer=1,
        start_lp=8000,finish_lethal=True,finish_main2=True,finish_turn=True)
    own=[r for r in result['choices'] if r['player']==1]
    sets=[r for r in own if r['action']['act']==A.SET]
    assert len(sets)==1 and sets[0]['action']['code']==98338152
    assert sets[0]['phase']==C.PHASE_MAIN2
    assert any(r['action']['phase']==P.END for r in own)
    assert result['turn']>2 and result['lp'][0]==500
