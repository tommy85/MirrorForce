"""Bomber/Borrelsword choice, burn, linked-zone destruction and attack restriction.

These explicit offline puzzles verify labels; they are not model learning or
online admission. The first-turn battle option is deliberately declared.
"""
from dataclasses import asdict
import json

import pytest

from mirrorforce.effectinfo import get_effectinfo_core
from mirrorforce.netduel import constants as C
from mirrorforce.netduel.actions import ActionAct as A, ActionPhase as P
from mirrorforce.search.rebuild import BoardSpec, DUEL_ATTACK_FIRST_TURN, rebuild
from mirrorforce.worldmodel.engine import DuelConfig, DeckList
from mirrorforce.worldmodel.state import capture
from test_tactical_curriculum_native import placement, EndOfLine

PRIVILEGED_TARGET = True
BOMBER, SWORD, RAYE, GENE, HORNET, BULB, KURI = 5821478,85289965,26077387,69247929,52340444,67441435,41999284


def play(*, family, finisher=BOMBER, lp=1500, main_zone=False):
    core=get_effectinfo_core()
    cards=[placement(finisher,0,C.LOCATION_MZONE,2 if main_zone else 5)]
    if family=='burn':
        cards.append(placement(RAYE,1,C.LOCATION_MZONE,position=C.POS_FACEUP_DEFENSE))
    elif family=='clear':
        cards += [placement(GENE,1,C.LOCATION_MZONE),placement(RAYE,0,C.LOCATION_HAND)]
        if main_zone:
            cards += [placement(KURI,0,C.LOCATION_MZONE,5),placement(BULB,0,C.LOCATION_GRAVE)]
        else: cards.append(placement(HORNET,0,C.LOCATION_HAND,1))
    else: raise ValueError(family)
    for player in (0,1):
        cards.extend(placement(GENE,player,C.LOCATION_DECK,i,C.POS_FACEDOWN_DEFENSE) for i in range(3))
    spec=BoardSpec(lp=(8000,lp),draw_count=(0,0),duel_options=DUEL_ATTACK_FIRST_TURN,placements=tuple(cards))
    empty=DeckList('explicit-bomber-curriculum',(),())
    cfg=DuelConfig((empty,empty),seed=2026100504,start_hand=0,draw_count=0,
                   duel_options=(5<<16)|DUEL_ATTACK_FIRST_TURN,full_phase_menu=True,
                   full_card_sort_menu=True,auto_end_phase_discard=False)
    driver=rebuild(spec,seed=cfg.seed,config=cfg,core=core)
    choices=[]; activated=summoned=False
    def choose(prompt,live):
        nonlocal activated,summoned
        actions=prompt.actions
        assert not prompt.truncated and prompt.complete_menu
        picked=None
        if prompt.player==0 and prompt.msg==C.MSG_SELECT_IDLECMD:
            if family=='clear' and not activated:
                code=BULB if main_zone else HORNET
                picked=next(i for i,a in enumerate(actions) if a.act==A.ACTIVATE and a.code==code)
                activated=True
            elif family=='clear' and not summoned:
                picked=next(i for i,a in enumerate(actions) if a.act==A.SUMMON and a.code==RAYE)
                summoned=True
            else: picked=next(i for i,a in enumerate(actions) if a.phase==P.BATTLE)
        elif prompt.msg==C.MSG_SELECT_BATTLECMD:
            if family=='clear':
                assert not any(a.code==RAYE and a.act in (A.ATTACK,A.DIRECT_ATTACK) for a in actions)
            picked=next((i for i,a in enumerate(actions)
                         if a.code==finisher and a.act in (A.ATTACK,A.DIRECT_ATTACK)),None)
            if picked is None: raise EndOfLine()
        elif prompt.msg==C.MSG_SELECT_CARD:
            picked=next((i for i,a in enumerate(actions) if a.code==RAYE),None)
        elif prompt.msg==C.MSG_SELECT_PLACE:
            picked=next((i for i,a in enumerate(actions) if a.place==2),0) # linked MZ directly below left EMZ
        elif prompt.msg==C.MSG_SELECT_POSITION:
            picked=next((i for i,a in enumerate(actions) if a.position==C.POS_FACEUP_ATTACK),0)
        elif prompt.msg==C.MSG_SELECT_EFFECTYN:
            picked=next(i for i,a in enumerate(actions) if a.act==A.ACTIVATE)
        if picked is None:
            picked=next((i for i,a in enumerate(actions) if a.act==A.CANCEL or a.finish),None)
        if picked is None and len(actions)==1: picked=0
        assert picked is not None,(prompt.msg,[asdict(a) for a in actions])
        choices.append({'msg':prompt.msg,'player':prompt.player,'phase':prompt.phase,'choice':asdict(actions[picked])})
        return picked
    try:
        try:driver.run(choose,max_steps=20000,max_seconds=15)
        except EndOfLine:pass
        state=capture(driver)
        result={'winner':driver.winner,'lp':list(state.lp),'field':[c.code for c in state.zone(0,C.LOCATION_MZONE)],
                'own_grave':[c.code for c in state.zone(0,C.LOCATION_GRAVE)],
                'opponent_grave':[c.code for c in state.zone(1,C.LOCATION_GRAVE)],'choices':choices}
        print('BOMBER-CURRICULUM '+json.dumps(result,sort_keys=True))
        return result
    finally:driver.close()


@pytest.mark.parametrize('finisher,win',[(BOMBER,True),(SWORD,False)])
def test_original_attack_burn_beats_defense_where_swords_nonpiercing_attack_does_not(finisher,win):
    result=play(family='burn',finisher=finisher)
    assert (result['winner']==0)==win,result
    assert result['lp'][1]==(0 if win else 1500),result


@pytest.mark.parametrize('lp,win,remaining',[(3000,True,0),(4500,False,1500)])
def test_linked_special_summon_clears_main_zones_then_only_bomber_attacks(lp,win,remaining):
    result=play(family='clear',lp=lp)
    assert (result['winner']==0)==win and result['lp'][1]==remaining,result
    assert BOMBER in result['field'] and RAYE in result['field'] and GENE in result['opponent_grave'],result


def test_bomber_in_a_main_zone_destroys_itself_and_still_restricts_the_other_attackers():
    result=play(family='clear',lp=3000,main_zone=True)
    assert result['winner'] is None and result['lp'][1]==3000,result
    assert BOMBER in result['own_grave'] and BOMBER not in result['field'],result
    assert KURI in result['field'] and RAYE in result['field'],result
