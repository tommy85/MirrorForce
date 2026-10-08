"""Native mechanical demonstrations of the historical Sky Striker Link chain.

This is an offline curriculum producer, never an online policy. It starts an
ordinary full-deck duel, not a Debug board. The other seat deliberately passes:
these demonstrations teach legal mechanics, not robustness to interruption.
No result is admitted to a learner until its public-input export is verified.
"""
from collections import Counter
from dataclasses import asdict
import random
import struct

from ...netduel import constants as C
from ...netduel.actions import ActionAct as A, ActionPhase as P
from ...netduel.wire_projection import InitialRefreshCore, project_message
from ...worldmodel.engine import DuelConfig, DuelDriver
from ...worldmodel.state import capture

PRIVILEGED_TARGET = True
SCHEMA = 'mirrorforce_sky_halq_bulb_borrelsword_curriculum/v1'
HALQ, BULB, LINKURIBOH, BORRELSWORD = 50588353, 67441435, 41999284, 85289965
BOMBER = 5821478
HORNET, TOKEN, ASH, MULTIROLE, AREA_ZERO, RAYE = 52340444, 52340445, 14558127, 24010609, 50005218, 26077387


class ComboComplete(Exception):
    pass


class _PublicTapeHost(DuelDriver):
    """Project exactly once per native event, with the actual host query schedule."""
    def __init__(self, config, core, viewer):
        self.viewer = viewer
        self.public_packets = []
        self.public_responses = []
        super().__init__(config, InitialRefreshCore(core))

    def build(self):
        super().build()
        decks = self.config.decks
        start = bytes([C.MSG_START]) + struct.pack('<BBiiHHHH', self.viewer, 5,
            self.config.start_lp,self.config.start_lp,len(decks[0].main),len(decks[0].extra),
            len(decks[1].main),len(decks[1].extra))
        self.public_packets[:0] = [start,*self.core.initial_packets[self.viewer]]

    def _observe(self, message):
        super()._observe(message)
        self.public_packets.extend(project_message(self.core,self.pduel,message.msg,message.payload)[self.viewer])

    def _respond(self, data):
        if self._ctx.our_player == self.viewer:
            self.public_responses.append(bytes(data))
        super()._respond(data)


def configuration(deck, *, seed, viewer=1, start_lp=8000, opponent='pass', opening_set_code=70368879,
                  opening_hand=None, first_draw=None):
    if type(viewer) is not int or viewer not in (0, 1) or type(seed) is not int:
        raise ValueError('register the curriculum seat and seed')
    if opening_set_code not in (70368879,98338152):
        raise ValueError('unregistered opening spell')
    hand = (HORNET, ASH, MULTIROLE, AREA_ZERO, opening_set_code) if opening_hand is None else opening_hand
    if type(hand) is not tuple or len(hand)!=5 or any(type(code) is not int for code in hand):
        raise ValueError('an explicit fixture opening has exactly five declared cards')
    if not Counter(hand) <= Counter(deck.main):
        raise ValueError('the registered full deck does not contain the opening resources')
    rest = list((Counter(deck.main) - Counter(hand)).elements())
    rng = random.Random(seed)
    rng.shuffle(rest)
    # This fixture teaches the deck-summoned tuner route, not an opening in
    # which that unique tuner has already been drawn. Do not reject/retry a
    # seed after play: declare this resource constraint in the construction.
    if rest[-1] == BULB:
        index=next(i for i,code in enumerate(rest) if code != BULB)
        rest[index],rest[-1]=rest[-1],rest[index]
    if first_draw is not None:
        if type(first_draw) is not int or first_draw==BULB or first_draw not in rest:
            raise ValueError('the declared first draw must remain in the full deck and preserve the tuner route')
        index=rest.index(first_draw);rest[index],rest[-1]=rest[-1],rest[index]
    # Both the opening and the remaining order are declared fixture data. They
    # never cross into the policy input as a hidden-layout feature.
    own = tuple(rest) + tuple(reversed(hand))
    if opponent not in ('pass', 'set_raye','two_defenders') or (opponent != 'pass' and viewer != 1):
        raise ValueError('the registered opponent fixture needs its ordinary first turn')
    other = list(deck.main)
    rng.shuffle(other)
    if opponent == 'set_raye':
        other.remove(RAYE)
        other.append(RAYE)
    elif opponent=='two_defenders':
        other.remove(RAYE);other.remove(HORNET)
        other.extend((RAYE,HORNET))
    orders = (own, tuple(other)) if viewer == 0 else (tuple(other), own)
    return DuelConfig((deck, deck), seed=seed, start_lp=start_lp, full_phase_menu=True,
                      full_card_sort_menu=True, auto_end_phase_discard=False,
                      forced_deck_orders=orders)


class ComboTeacher:
    """Finite, stage-bound choices from the native legal menu only."""
    def __init__(self, viewer=1, *, finish_lethal=False, finisher=BORRELSWORD, opponent='pass',
                 finish_main2=False, finish_turn=False):
        if finisher not in (BORRELSWORD, BOMBER):
            raise ValueError('unregistered curriculum finisher')
        self.viewer, self.finish_lethal = viewer, finish_lethal
        self.finisher, self.opponent = finisher, opponent
        self.finish_main2 = finish_main2
        if finish_turn and not finish_main2:
            raise ValueError('finish-turn curriculum requires the nonlethal main2 fixture')
        self.finish_turn = finish_turn
        self.main2_set = False
        self.opponent_set = False
        self.stage = 0
        self.intent = None
        self.choices = []
        self.material_choices = []
        self.sword_attacks = self.raye_attacks = 0
        self.flipped = False
        self.commands = [(A.ACTIVATE,HORNET,None), (A.SUMMON,ASH,None),
                         (A.SPSUMMON,HALQ,None), (A.SPSUMMON,LINKURIBOH,None),
                         (A.ACTIVATE,BULB,None), (A.SPSUMMON,finisher,None)]
        if finish_lethal and finisher == BORRELSWORD:
            self.commands += [(A.ACTIVATE,MULTIROLE,0), (A.ACTIVATE,AREA_ZERO,0),
                              (A.ACTIVATE,MULTIROLE,MULTIROLE*16)]

    def __call__(self, prompt, driver):
        if prompt.truncated or not prompt.complete_menu:
            raise ValueError('curriculum requires a complete native menu')
        actions = prompt.actions
        picked = None
        if self.finish_turn and self.main2_set and prompt.turn > 2 and prompt.player == self.viewer:
            raise ComboComplete()
        if prompt.player != self.viewer:
            if self.opponent == 'set_raye' and not self.opponent_set and prompt.msg == C.MSG_SELECT_IDLECMD:
                picked = next((i for i,a in enumerate(actions) if a.act == A.MSET and a.code == RAYE), None)
                if picked is None: raise ValueError('registered defending monster is not legally settable')
                self.opponent_set = True
            elif prompt.msg == C.MSG_SELECT_PLACE:
                picked = 0
            if picked is None:
                picked = next((i for i,a in enumerate(actions) if a.act == A.CANCEL or a.phase == P.END or a.finish), None)
            if picked is None and len(actions) == 1: picked = 0
            if picked is None:
                raise ValueError('unopposed fixture needs an explicit opponent choice it did not register')
        elif prompt.msg == C.MSG_SELECT_IDLECMD:
            if self.finish_main2 and prompt.phase == C.PHASE_MAIN2:
                if self.main2_set:
                    if not self.finish_turn: raise ComboComplete()
                    picked = next((i for i,a in enumerate(actions) if a.phase == P.END), None)
                    if picked is None: raise ValueError('registered clean turn end is not legal')
                else:
                    picked = next((i for i,a in enumerate(actions) if a.act == A.SET and a.code == 98338152), None)
                    if picked is None: raise ValueError('registered main2 set is not legal')
                    self.main2_set = True
                    self.intent = ('set',98338152)
            elif self.stage == len(self.commands):
                if not self.finish_lethal: raise ComboComplete()
                picked = next((i for i,a in enumerate(actions) if a.phase == P.BATTLE), None)
                if picked is None: raise ValueError('the ordinary duel has no legal battle phase here')
            else:
                act,code,desc = self.commands[self.stage]
                picked = next((i for i,a in enumerate(actions)
                               if a.act == act and a.code == code and (desc is None or a.desc == desc)), None)
                if picked is None:
                    raise ValueError(f'combo stage {self.stage} is not legal: {[(a.act,a.code,a.desc) for a in actions]}')
                self.intent = ('materials',code) if act == A.SPSUMMON else ('activation',code)
                self.stage += 1
        elif prompt.msg in (C.MSG_SELECT_EFFECTYN, C.MSG_SELECT_CHAIN):
            for i,a in enumerate(actions):
                if a.act == A.ACTIVATE and a.code in (HALQ, AREA_ZERO, BOMBER):
                    picked = i
                    self.intent = ('search', BULB if a.code == HALQ else RAYE)
                    break
        elif prompt.msg in (C.MSG_SELECT_UNSELECT_CARD, C.MSG_SELECT_SUM, C.MSG_SELECT_TRIBUTE):
            if self.intent and self.intent[0] == 'materials':
                target = self.intent[1]
                wanted = {HALQ:{ASH,TOKEN}, LINKURIBOH:{BULB},
                          BORRELSWORD:{HALQ,LINKURIBOH,BULB}, BOMBER:{HALQ,LINKURIBOH,BULB}}[target]
                picked = next((i for i,a in enumerate(actions) if a.finish), None)
                if picked is None:
                    picked = next((i for i,a in enumerate(actions) if a.code in wanted and a.act != A.CANCEL), None)
                if picked is not None:
                    self.material_choices.append({'summon':target,'choice':asdict(actions[picked])})
        elif prompt.msg == C.MSG_SELECT_CARD:
            wanted = None
            if self.intent and self.intent[0] == 'search': wanted = self.intent[1]
            elif self.intent == ('activation',MULTIROLE): wanted = AREA_ZERO
            elif self.intent == ('flip',BORRELSWORD): wanted = RAYE
            elif self.intent == ('attack',BOMBER): wanted = RAYE
            if wanted is not None:
                picked = next((i for i,a in enumerate(actions) if a.code == wanted), None)
        elif prompt.msg == C.MSG_SELECT_PLACE:
            picked = 0
        elif prompt.msg == C.MSG_SELECT_POSITION:
            picked = next((i for i,a in enumerate(actions) if a.position == C.POS_FACEUP_ATTACK), 0)
        elif prompt.msg == C.MSG_SELECT_BATTLECMD and self.finish_lethal:
            if self.finisher == BOMBER:
                if self.sword_attacks: raise ComboComplete()
                picked = next((i for i,a in enumerate(actions)
                               if a.code == BOMBER and a.act in (A.ATTACK,A.DIRECT_ATTACK)), None)
                if picked is None: raise ValueError('registered Bomber attack is not legal')
                self.sword_attacks += 1
                self.intent = ('attack',BOMBER)
                self.choices.append({'player':prompt.player,'response':len(driver.responses),'stage':self.stage,
                                     'turn':prompt.turn,'phase':prompt.phase,'msg':prompt.msg,
                                     'index':picked,'action':asdict(actions[picked])})
                return picked
            wanted = RAYE if self.raye_attacks == 0 else BORRELSWORD if self.sword_attacks == 0 else None
            if wanted is not None:
                picked = next((i for i,a in enumerate(actions) if a.code == wanted and a.act == A.DIRECT_ATTACK), None)
                if picked is not None:
                    self.raye_attacks += wanted == RAYE
                    self.sword_attacks += wanted == BORRELSWORD
            elif not self.flipped:
                picked = next((i for i,a in enumerate(actions) if a.code == BORRELSWORD and a.act == A.ACTIVATE), None)
                if picked is not None:
                    self.flipped = True
                    self.intent = ('flip',BORRELSWORD)
            elif self.sword_attacks == 1:
                picked = next((i for i,a in enumerate(actions) if a.code == BORRELSWORD and a.act == A.DIRECT_ATTACK), None)
                if picked is not None: self.sword_attacks += 1
            else:
                if self.finish_main2:
                    picked = next((i for i,a in enumerate(actions) if a.phase == P.MAIN2), None)
                    if picked is None: raise ValueError('registered main2 transition is not legal')
                else: raise ComboComplete()
        if picked is None:
            picked = next((i for i,a in enumerate(actions) if a.act == A.CANCEL or a.finish), None)
        if picked is None and len(actions) == 1: picked = 0
        if picked is None:
            raise ValueError(f'unregistered combo follow-up {prompt.msg} / {self.intent}: {[asdict(a) for a in actions]}')
        self.choices.append({'player':prompt.player,'response':len(driver.responses),'stage':self.stage,
                             'turn':prompt.turn,'phase':prompt.phase,'msg':prompt.msg,
                             'index':picked,'action':asdict(actions[picked])})
        return picked


def demonstrate(core, deck, *, seed, viewer=1, start_lp=8000, finish_lethal=False,
                finisher=BORRELSWORD, opponent='pass', finish_main2=False, finish_turn=False):
    cfg = configuration(deck, seed=seed, viewer=viewer, start_lp=start_lp, opponent=opponent,
                        opening_set_code=98338152 if finish_main2 else 70368879)
    teacher = ComboTeacher(viewer, finish_lethal=finish_lethal,finisher=finisher,opponent=opponent,
                           finish_main2=finish_main2, finish_turn=finish_turn)
    with _PublicTapeHost(cfg,core,viewer) as driver:
        driver.record_messages = True
        try: driver.run(teacher, max_steps=30000, max_seconds=30)
        except ComboComplete: pass
        state = capture(driver)
        if teacher.stage != len(teacher.commands):
            raise ValueError('the complete registered Link chain was not executed')
        return {'schema':SCHEMA,'seed':seed,'viewer':viewer,'start_lp':start_lp,
                'ordinary_duel_rules':True,'debug_board':False,'opponent':'explicit-mechanism-'+opponent,
                'finisher':finisher,
                'commands_complete':teacher.stage,'choices':teacher.choices,'materials':teacher.material_choices,
                'field':[asdict(c) for c in state.zone(viewer,C.LOCATION_MZONE)],
                'grave':[c.code for c in state.zone(viewer,C.LOCATION_GRAVE)],
                'lp':list(state.lp),'winner':driver.winner,'turn':driver.turn,
                'responses':[bytes(r).hex() for r in driver.responses],
                'public_tape':{'schema':SCHEMA+'#public-tape','viewer':viewer,
                    'own_recipe':{'main':list(deck.main),'extra':list(deck.extra)},
                    'opponent_recipe_mode':'mirror',
                    'messages':[[raw[0],raw[1:].hex()] for raw in driver.public_packets],
                    'responses':[raw.hex() for raw in driver.public_responses],
                    'terminal':driver.finished,'winner':driver.winner,
                    'private_layouts':False},
                'training_eligible':False,'model_updates':0,'playing_strength_evidence':False}
