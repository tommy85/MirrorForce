"""Ordinary-duel Bomber clear/direct-attack curriculum and resource controls."""
from dataclasses import asdict

from ...netduel import constants as C
from ...netduel.actions import ActionAct as A,ActionPhase as P
from ...worldmodel.state import capture
from .sky_combo_curriculum import (_PublicTapeHost,configuration,ComboTeacher,ComboComplete,
    BOMBER,HORNET,ASH,MULTIROLE,AREA_ZERO,RAYE,SCHEMA)

PRIVILEGED_TARGET=True


class ClearTeacher(ComboTeacher):
    def __init__(self,trigger=True):
        super().__init__(1,finish_lethal=True,finisher=BOMBER)
        self.opponent_stage=0
        self.trigger=trigger
        self.before_trigger=None
        self.before_battle=None
        if trigger:self.commands += [(A.ACTIVATE,MULTIROLE,0),(A.ACTIVATE,AREA_ZERO,0),
                                     (A.ACTIVATE,MULTIROLE,MULTIROLE*16)]

    def __call__(self,prompt,driver):
        actions=prompt.actions;index=None
        if prompt.player==1 and self.sword_attacks and prompt.msg==C.MSG_SELECT_IDLECMD:
            raise ComboComplete()
        if prompt.player==1 and self.sword_attacks and prompt.msg==C.MSG_SELECT_BATTLECMD:
            # Consume the real phase transition. A post-attack battle prompt
            # can be forced in ClientDuel and cannot be an unanswered prefix.
            index=next((i for i,a in enumerate(actions) if a.phase==P.MAIN2),None)
            if index is None:raise ValueError('registered post-attack main2 transition unavailable')
        elif prompt.player==0:
            if prompt.msg==C.MSG_SELECT_IDLECMD and self.opponent_stage<2:
                act,code=[(A.ACTIVATE,HORNET),(A.MSET,RAYE)][self.opponent_stage]
                index=next((i for i,a in enumerate(actions) if a.act==act and a.code==code),None)
                if index is None:raise ValueError('registered defending setup is unavailable')
                self.opponent_stage+=1
            elif prompt.msg==C.MSG_SELECT_PLACE:index=0
            if index is None:index=next((i for i,a in enumerate(actions) if a.act==A.CANCEL or a.phase==P.END or a.finish),None)
            if index is None and len(actions)==1:index=0
            if index is None:raise ValueError('unregistered defending response')
        elif prompt.msg==C.MSG_SELECT_PLACE and self.intent==('materials',BOMBER):
            index=next((i for i,a in enumerate(actions) if a.place in (6,7)),None)
            if index is None:raise ValueError('Bomber must be legally placed in an extra monster zone')
        elif prompt.msg==C.MSG_SELECT_PLACE and self.intent==('search',RAYE) and self.stage==len(self.commands):
            state=capture(driver)
            bomber=next(c for c in state.zone(1,C.LOCATION_MZONE) if c.code==BOMBER)
            if bomber.sequence not in (5,6) or bomber.link_marker!=135:
                raise ValueError('registered extra-zone Bomber or arrows differ')
            place=2 if bomber.sequence==5 else 4
            index=next((i for i,a in enumerate(actions) if a.place==place),None)
            if index is None:raise ValueError('no legal summon into the registered Bomber arrow')
            self.before_trigger=self.board(state)
        else:
            if prompt.player==1 and prompt.msg==C.MSG_SELECT_BATTLECMD and self.before_battle is None:
                self.before_battle=self.board(capture(driver))
            return super().__call__(prompt,driver)
        self.choices.append({'player':prompt.player,'response':len(driver.responses),'stage':self.stage,
            'turn':prompt.turn,'phase':prompt.phase,'msg':prompt.msg,'index':index,'action':asdict(actions[index])})
        return index

    @staticmethod
    def board(state):
        return {'opponent_field':[asdict(c) for c in state.zone(0,C.LOCATION_MZONE)],
                'own_field':[asdict(c) for c in state.zone(1,C.LOCATION_MZONE)],
                'opponent_backrow':[asdict(c) for c in state.zone(0,C.LOCATION_SZONE)],
                'own_hand':[c.code for c in state.zone(1,C.LOCATION_HAND)]}


def demonstrate_clear(core,deck,*,seed,start_lp=3000,trigger_resources=True):
    if type(trigger_resources) is not bool:raise ValueError('explicit trigger-resource arm required')
    kwargs={} if trigger_resources else {'opening_hand':(HORNET,ASH,98338152,41420027,99550630),'first_draw':ASH}
    cfg=configuration(deck,seed=seed,start_lp=start_lp,opponent='two_defenders',**kwargs)
    teacher=ClearTeacher(trigger_resources)
    with _PublicTapeHost(cfg,core,1) as driver:
        driver.record_messages=True
        try:driver.run(teacher,max_seconds=30)
        except ComboComplete:pass
        state=capture(driver)
        return {'schema':SCHEMA+'#bomber-clear','seed':seed,'start_lp':start_lp,
                'trigger_resources':trigger_resources,'ordinary_duel_rules':True,
                'defender_description':'face-up defense token and face-down defense Raye; no backrow',
                'before_trigger':teacher.before_trigger,'before_battle':teacher.before_battle,
                'lp':list(state.lp),'winner':driver.winner,'choices':teacher.choices,
                'field':[asdict(c) for c in state.zone(1,C.LOCATION_MZONE)],
                'opponent_field':[asdict(c) for c in state.zone(0,C.LOCATION_MZONE)],
                'training_eligible':False,'playing_strength_evidence':False,
                'public_tape':{'schema':SCHEMA+'#public-tape','viewer':1,
                    'own_recipe':{'main':list(deck.main),'extra':list(deck.extra)},
                    'opponent_recipe_mode':'mirror','private_layouts':False,
                    'messages':[[r[0],r[1:].hex()] for r in driver.public_packets],
                    'responses':[r.hex() for r in driver.public_responses],
                    'terminal':driver.finished,'winner':driver.winner}}
