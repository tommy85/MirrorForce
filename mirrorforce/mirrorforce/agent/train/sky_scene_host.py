"""Privileged scene construction/scoring, isolated from the public player."""
from dataclasses import asdict
from collections import Counter
import ctypes
import struct

from ...netduel import constants as C
from ...worldmodel.state import capture
from .sky_combo_curriculum import _PublicTapeHost, configuration, ComboTeacher, BOMBER, BORRELSWORD

PRIVILEGED_TARGET = True


class SceneBoundary(Exception):
    pass


class AutonomousHost(_PublicTapeHost):
    def __init__(self, cfg, core, answer, *, last_turn=2):
        if type(last_turn) is not int or last_turn not in (2,3):
            raise ValueError('register the scene boundary explicitly')
        super().__init__(cfg, core, 1)
        self.answer_public = answer
        self.cursor = 0
        self.choices = []
        self.own_prompts = 0
        self.last_turn = last_turn
        self.chain_sources = []
        self.draws = []

    def own_grave_spell_count(self):
        # QUERY_CODE reads printed data only and does not update the engine's
        # q_cache. Full capture here would alter later incremental public
        # refresh packets (and break the independent tape/host equivalence).
        buf=ctypes.create_string_buffer(4096)
        size=self.core.query_field_card(self.pduel,self.viewer,C.LOCATION_GRAVE,C.QUERY_CODE,buf)
        if not 0<=size<=len(buf):raise ValueError('invalid public grave code query size')
        offset=count=0;pool=self.core.card_pool()
        while offset<size:
            if size-offset<12:raise ValueError('truncated public grave code query')
            length,flags,code=struct.unpack_from('<III',buf,offset)
            if length!=12 or flags!=C.QUERY_CODE or code not in pool.cards:
                raise ValueError('invalid public grave code query record')
            count+=bool(pool.cards[code].type & 0x2);offset+=length
        return count

    def _observe(self, message):
        super()._observe(message)
        if message.msg == C.MSG_CHAINING:
            self.chain_sources.append({'code':int.from_bytes(message.payload[:4],'little'),
                                       'controller':message.payload[4]})
        elif message.msg == C.MSG_CHAIN_END:
            self.chain_sources.clear()
        elif message.msg == C.MSG_DRAW:
            self.draws.append({'turn':self.turn,'phase':self.phase,'player':message.payload[0],
                              'count':message.payload[1],'chain_sources':[dict(x) for x in self.chain_sources]})
        if self.turn > self.last_turn and not (self.last_turn==3 and self.turn==4):
            raise SceneBoundary('registered first own turn ended')

    def _answer(self, message, responder):
        body = message.payload
        viewer = body[1] if message.msg == C.MSG_SELECT_SUM else body[0]
        if viewer == self.viewer:
            if self.last_turn==3 and self.turn==4:
                pending=self.parse_prompt(message.msg,body)
                if pending.selector is not None and len(pending.selector.options())>1:
                    raise SceneBoundary('registered follow-up ended at the next own decision')
            if self._replay:
                raise ValueError('autonomous host cannot consume a teacher tape')
            if self.own_prompts >= 300:
                raise SceneBoundary('own prompt budget exhausted')
            packets = [[raw[0],raw[1:].hex()] for raw in self.public_packets[self.cursor:]]
            result = self.answer_public({'packets':packets})
            if result.get('public_inputs_only') is not True or result.get('teacher_forcing') is not False:
                raise ValueError('autonomous response provenance missing')
            response = bytes.fromhex(result['response'])
            if not 0 < len(response) <= 255:
                raise ValueError('invalid actual public response')
            spell_count=self.own_grave_spell_count()
            self.choices.extend({**row, 'turn':self.turn, 'phase':self.phase,'turn_player':self.turn_player,
                                 'response':len(self.responses),'own_grave_spells':spell_count,
                                 'chain_sources':[dict(x) for x in self.chain_sources]}
                                for row in result['decisions'])
            self.cursor = len(self.public_packets)
            self.own_prompts += 1
            # The core validates this actual response. No teacher selector or
            # host legal action index is exposed to the policy endpoint.
            self._replay = [response]
        super()._answer(message, responder)


def run_scene(core, deck, scene, answer, *, max_seconds=900):
    family = scene['family']
    if family not in ('sword','bomber','main2_set','backrow_timing','command_timing'):
        raise ValueError('unregistered scene family')
    variant=scene.get('variant')
    if family=='command_timing':
        from .sky_command_timing import VARIANTS, command_configuration, CommandTeacher, assess_command_scene
        if variant not in VARIANTS:raise ValueError('unregistered command-timing variant')
    elif variant not in (None,'linked_clear') or variant=='linked_clear' and family!='bomber':
        raise ValueError('unregistered scene variant')
    opponent = 'two_defenders' if variant=='linked_clear' else 'set_raye' if family == 'bomber' else 'pass'
    if family=='command_timing':
        cfg=command_configuration(deck,scene);opponent_only=CommandTeacher(scene)
    elif family=='backrow_timing':
        from .sky_backrow_timing import timing_configuration, TimingTeacher
        cfg=timing_configuration(deck,scene);opponent_only=TimingTeacher(scene)
    else:
        cfg = configuration(deck, seed=scene['seed'], viewer=1, start_lp=scene['start_lp'],
            opponent=opponent, opening_set_code=98338152 if family == 'main2_set' else 70368879)
    if variant=='linked_clear':
        if scene.get('trigger_resources') is not True:
            raise ValueError('this autonomous clear fixture needs its registered positive resources')
        from .sky_bomber_clear import ClearTeacher
        opponent_only=ClearTeacher()
    elif family not in ('backrow_timing','command_timing'):
        opponent_only = ComboTeacher(1, opponent=opponent)
    def respond(prompt, driver):
        if prompt.player == 1:
            raise ValueError('policy seat unexpectedly reached the teacher')
        return opponent_only(prompt, driver)
    with AutonomousHost(cfg, core, answer, last_turn=3 if family=='command_timing' else 2) as driver:
        boundary = None
        try:
            driver.run(respond, max_steps=30000, max_seconds=max_seconds)
        except SceneBoundary as exc:
            boundary = str(exc)
        state = capture(driver)
        finisher = BOMBER if family == 'bomber' else BORRELSWORD
        from ...netduel.actions import ActionAct as A
        summoned = any(row['action'].get('code') == finisher and
                       row['action'].get('act') == int(A.SPSUMMON) for row in driver.choices)
        main2_set = any(row['phase'] == C.PHASE_MAIN2 and
                       row['action'].get('code') == 98338152 and
                       row['action'].get('act') == int(A.SET) for row in driver.choices)
        damage = scene['start_lp'] - state.lp[0]
        canonical_success=(driver.winner == 1 and summoned) if family != 'main2_set' else (
            damage >= 7500 and main2_set and summoned)
        success = (driver.winner == 1) if family != 'main2_set' else (damage >= 7500 and main2_set)
        if family=='backrow_timing':
            target=Counter(scene['defenses'])
            delayed=Counter(row['action'].get('code') for row in driver.choices
                            if row['action'].get('act')==int(A.SET) and row['phase']==C.PHASE_MAIN2)
            premature=any(row['action'].get('act')==int(A.SET) and row['phase']!=C.PHASE_MAIN2
                          for row in driver.choices)
            attacked=any(row['action'].get('act') in (int(A.ATTACK),int(A.DIRECT_ATTACK)) for row in driver.choices)
            kept=Counter(card.code for card in state.zone(1,C.LOCATION_SZONE))
            success=not premature and (driver.winner==1 or attacked and damage>=1500 and target<=delayed and target<=kept)
            canonical_success=success
        command_issues=None
        if family=='command_timing':
            command_issues=assess_command_scene(scene,driver.choices,driver.draws,
                [card.code for card in state.zone(1,C.LOCATION_HAND)],list(state.lp))
            success=canonical_success=not command_issues
        return {'scene':scene, 'success':bool(success), 'winner':driver.winner,
                'canonical_route_success':bool(canonical_success),
                'goal_law':('registered-command-timing-own-turn-g/v2' if family=='command_timing'
                            else 'natural-lethal-or-registered-post-battle-set/v2'),
                'terminal':driver.finished, 'turn':driver.turn, 'boundary':boundary,
                'lp':list(state.lp), 'opponent_lp_reduction':damage,
                'finisher_selected':summoned, 'main2_set':main2_set,
                'own_prompts':driver.own_prompts, 'decisions':driver.choices,
                'responses':[r.hex() for r in driver.public_responses],
                'field':[asdict(card) for card in state.zone(1,C.LOCATION_MZONE)],
                'opponent_monsters_remaining':len(state.zone(0,C.LOCATION_MZONE)),
                'own_backrow':[asdict(card) for card in state.zone(1,C.LOCATION_SZONE)],
                'own_hand':[card.code for card in state.zone(1,C.LOCATION_HAND)],
                'draws':driver.draws,'command_issues':command_issues,
                'teacher_forcing':False, 'playing_strength_evidence':False}
