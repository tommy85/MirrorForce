"""Independent ordinary-duel examples for delaying defensive sets until Main 2.

The opponent's optional battle-window Twin Twisters policy is a mechanical
counterexample, not a claim that an unknown backrow always contains removal.
Only public tapes may be exported to a learner; human replay orders are unused.
"""
from collections import Counter
from dataclasses import asdict, replace
import random

from ...netduel import constants as C
from ...netduel.actions import ActionAct as A, ActionPhase as P
from ...worldmodel.state import capture
from .sky_combo_curriculum import _PublicTapeHost, configuration, ComboComplete, SCHEMA, RAYE, BULB

HAYATE, SHIZUKU, ENGAGE, TWIN = 8491308, 90673288, 63166095, 43898403
DEFENSES = ((98338152,84749824),(98338152,41420027),(98338152,98338152))
HAND_TRAPS = (14558127,23434538,97268402,59438930)


def timing_configuration(deck, scene):
    defenses=tuple(scene['defenses']);spares=tuple(scene['spares']);draw=scene['first_draw']
    if defenses not in DEFENSES or len(spares)!=2 or len(set(spares))!=2 or any(x not in HAND_TRAPS for x in spares):
        raise ValueError('register defensive cards and two distinct passive hand resources')
    if draw not in HAND_TRAPS or draw in spares or type(scene.get('threat')) is not bool:
        raise ValueError('register an independent first draw and explicit opponent arm')
    cfg=configuration(deck,seed=scene['seed'],start_lp=scene.get('start_lp',8000),
        opening_hand=(RAYE,*defenses,*spares),first_draw=draw)
    # This is independently generated fixture setup, not a human's deck order.
    backrow=TWIN if scene['threat'] else 98338152
    hand=(backrow,BULB,14558127,23434538,97268402)
    if not Counter(hand)<=Counter(deck.main):raise ValueError('opponent fixture resources are absent')
    rest=list((Counter(deck.main)-Counter(hand)).elements());random.Random(scene['seed']^0x64A15).shuffle(rest)
    return replace(cfg,forced_deck_orders=(tuple(rest)+tuple(reversed(hand)),cfg.forced_deck_orders[1]))


class TimingTeacher:
    def __init__(self,scene,*,premature=False):
        self.scene=scene;self.premature=premature;self.choices=[]
        self.stage=0;self.attacked=False;self.changed=False;self.sets=Counter()
        self.opponent_set=False;self.intent=None;self.opponent_intent=None

    def __call__(self,prompt,driver):
        if prompt.turn>2 and prompt.player==1:raise ComboComplete()
        actions=prompt.actions;pick=None
        if prompt.player==0:
            if prompt.turn==1 and prompt.msg==C.MSG_SELECT_IDLECMD:
                if not self.opponent_set:
                    code=TWIN if self.scene['threat'] else 98338152
                    pick=next((i for i,a in enumerate(actions) if a.act==A.SET and a.code==code),None)
                    self.opponent_set=True
                else:pick=next((i for i,a in enumerate(actions) if a.phase==P.END),None)
            elif prompt.turn==2 and C.PHASE_BATTLE_START<=prompt.phase<C.PHASE_MAIN2 and prompt.msg==C.MSG_SELECT_CHAIN:
                targets=capture(driver).zone(1,C.LOCATION_SZONE)
                if self.scene['threat'] and len(targets)>=2:
                    pick=next((i for i,a in enumerate(actions) if a.act==A.ACTIVATE and a.code==TWIN),None)
                    if pick is not None:self.opponent_intent='twin'
            elif self.opponent_intent=='twin' and prompt.msg==C.MSG_SELECT_CARD:
                pick=next((i for i,a in enumerate(actions) if a.spec.startswith('os')),None)
                if pick is None:pick=next((i for i,a in enumerate(actions) if a.code==BULB),None)
        elif prompt.msg==C.MSG_SELECT_IDLECMD:
            if prompt.phase==C.PHASE_MAIN1:
                target=next((c for c in self.scene['defenses'] if self.sets[c]<Counter(self.scene['defenses'])[c]),None)
                if self.premature and self.stage>=1 and target is not None:
                    pick=next((i for i,a in enumerate(actions) if a.act==A.SET and a.code==target),None)
                    if pick is not None:self.sets[target]+=1
                elif self.stage==0:
                    pick=next((i for i,a in enumerate(actions) if a.act==A.SUMMON and a.code==RAYE),None)
                    if pick is not None:self.stage=1
                elif self.stage==1:
                    pick=next((i for i,a in enumerate(actions) if a.act==A.SPSUMMON and a.code==HAYATE),None)
                    if pick is not None:self.stage=2;self.intent=('material',RAYE)
                else:pick=next((i for i,a in enumerate(actions) if a.phase==P.BATTLE),None)
            elif prompt.phase==C.PHASE_MAIN2:
                if not self.changed:
                    pick=next((i for i,a in enumerate(actions) if a.act==A.SPSUMMON and a.code==SHIZUKU),None)
                    if pick is not None:self.changed=True;self.intent=('material',HAYATE)
                else:
                    target=next((c for c in self.scene['defenses'] if self.sets[c]<Counter(self.scene['defenses'])[c]),None)
                    if target is None:pick=next((i for i,a in enumerate(actions) if a.phase==P.END),None)
                    else:
                        pick=next((i for i,a in enumerate(actions) if a.act==A.SET and a.code==target),None)
                        if pick is not None:self.sets[target]+=1
            if pick is None:raise ValueError('registered timing command is not legal')
        elif prompt.msg==C.MSG_SELECT_BATTLECMD:
            if not self.attacked:
                pick=next((i for i,a in enumerate(actions) if a.act==A.DIRECT_ATTACK and a.code==HAYATE),None)
                if pick is not None:self.attacked=True
            else:pick=next((i for i,a in enumerate(actions) if a.phase==P.MAIN2),None)
            if pick is None:raise ValueError('registered direct attack or post-battle transition unavailable')
        elif prompt.msg in (C.MSG_SELECT_CHAIN,C.MSG_SELECT_EFFECTYN):
            pick=next((i for i,a in enumerate(actions) if a.act==A.ACTIVATE and a.code in (HAYATE,SHIZUKU)),None)
            if pick is not None:self.intent=('pick',ENGAGE if actions[pick].code==HAYATE else 52340444)
        elif prompt.msg in (C.MSG_SELECT_CARD,C.MSG_SELECT_UNSELECT_CARD,C.MSG_SELECT_SUM):
            if self.intent:
                wanted=self.intent[1]
                pick=next((i for i,a in enumerate(actions) if a.code==wanted and a.act!=A.CANCEL),None)
                if pick is None:pick=next((i for i,a in enumerate(actions) if a.finish),None)
        if pick is None and prompt.msg==C.MSG_SELECT_PLACE:pick=0
        if pick is None and prompt.msg==C.MSG_SELECT_POSITION:
            pick=next((i for i,a in enumerate(actions) if a.position==C.POS_FACEUP_ATTACK),0)
        if pick is None:pick=next((i for i,a in enumerate(actions) if a.act==A.CANCEL or a.finish),None)
        if pick is None and len(actions)==1:pick=0
        if pick is None:raise ValueError(f'unregistered timing response {prompt.msg}: {[a.describe() for a in actions]}')
        self.choices.append({'player':prompt.player,'response':len(driver.responses),'turn':prompt.turn,
            'phase':prompt.phase,'msg':prompt.msg,'index':pick,'action':asdict(actions[pick])})
        return pick


def demonstrate_timing(core,deck,scene,*,premature=False):
    cfg=timing_configuration(deck,scene);teacher=TimingTeacher(scene,premature=premature)
    with _PublicTapeHost(cfg,core,1) as driver:
        try:driver.run(teacher,max_seconds=30)
        except ComboComplete:pass
        state=capture(driver)
        return {'schema':SCHEMA+'#backrow-timing','scene':scene,'premature_control':premature,
            'lp':list(state.lp),'choices':teacher.choices,'turn':driver.turn,
            'own_backrow':[asdict(c) for c in state.zone(1,C.LOCATION_SZONE)],
            'training_eligible':False,'playing_strength_evidence':False,
            'public_tape':{'schema':SCHEMA+'#public-tape','viewer':1,
                'own_recipe':{'main':list(deck.main),'extra':list(deck.extra)},'opponent_recipe_mode':'mirror',
                'private_layouts':False,'messages':[[r[0],r[1:].hex()] for r in driver.public_packets],
                'responses':[r.hex() for r in driver.public_responses],
                'terminal':driver.finished,'winner':driver.winner}}
