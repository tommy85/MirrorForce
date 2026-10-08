"""Independent normal-game demonstrations for handtrap and utility timing.

No recorded human hands, seeds or responses are consumed.  The engine owns
the fixture; only projected public tapes may reach the learner.  In particular
own-turn Maxx C and Main-2 Engage are not universally forbidden.
"""
from collections import Counter
from dataclasses import asdict, replace
import random

from ...netduel import constants as C
from ...netduel.actions import ActionAct as A, ActionPhase as P
from ...worldmodel.state import capture
from .sky_combo_curriculum import _PublicTapeHost, configuration, ComboComplete, SCHEMA, RAYE, BULB, HORNET, TOKEN
from .sky_backrow_timing import TimingTeacher, HAYATE, SHIZUKU, ENGAGE

MAXX_C, UPSTART, ASH, VEILER, WIDOW, WARNING = 23434538, 70368879, 14558127, 97268402, 98338152, 84749824
VARIANTS = ('idle_g_hold', 'own_turn_g_chain', 'opponent_turn_g_chain',
            'main1_upstart', 'engage_direct', 'main2_engage_value')


def assess_command_scene(scene, decisions, draws, own_hand, lp):
    """Check real outcomes and command order, not agreement with a teacher."""
    variant=scene.get('variant')
    if variant not in VARIANTS:
        raise ValueError('unregistered command-timing assessment')
    issues=[]
    activations=[x for x in decisions if x['action'].get('act')==int(A.ACTIVATE)]
    g=[x for x in activations if x['action'].get('code')==MAXX_C]
    for used in g:
        if type(used.get('turn_player')) is not int or used['turn_player'] not in (0,1):
            raise ValueError('Maxx C timing needs the host-recorded turn owner')
        # User's no-waste rule concerns OUR turn only. Proactive opponent-turn
        # G is legal strategy, even when it deters summons and draws nothing.
        if used['turn_player']==0:
            continue
        responding=(used['msg']==C.MSG_SELECT_CHAIN and any(
            x.get('controller')==0 and x.get('code')==RAYE for x in used.get('chain_sources',[])))
        if not responding:
            issues.append('Maxx C was spent on our turn without the registered opponent special-summon action')
    # Positive demonstrations prove a useful G line exists; they are NOT a
    # mandate to choose G over another legal interruption. Nor does an absent
    # hand card alone prove a wasted activation (it may have paid another cost).
    commands=[x for x in decisions if x['msg']==C.MSG_SELECT_IDLECMD]
    if variant in ('main1_upstart','engage_direct'):
        code=UPSTART if variant=='main1_upstart' else ENGAGE
        used=[x for x in commands if x['action'].get('code')==code and x['action'].get('act')==int(A.ACTIVATE)]
        if not used or any(x['phase']!=C.PHASE_MAIN1 or not x['action'].get('spec','').startswith('h') for x in used):
            issues.append('the registered utility was not used directly from hand in Main 1')
        if any(x['action'].get('code')==code and x['action'].get('act')==int(A.SET) for x in decisions):
            issues.append('the utility was unnecessarily set before activation')
    if variant=='main2_engage_value':
        used=[x for x in commands if x['action'].get('code')==ENGAGE and x['action'].get('act')==int(A.ACTIVATE)]
        bridge=[x for x in commands if x['action'].get('phase')==int(P.BATTLE)]
        if (not used or any(x['phase']!=C.PHASE_MAIN2 or x.get('own_grave_spells',0)<3 for x in used)
                or not bridge or bridge[0].get('own_grave_spells')!=2):
            issues.append('Main 2 Engage did not follow the two-to-three grave-spell battle setup')
        if not any(x['player']==1 and x['turn']==2 and x['phase']==C.PHASE_MAIN2 and x['count']>0 for x in draws):
            issues.append('the registered delayed Engage extra draw was not obtained')
    if not any(x['action'].get('act') in (int(A.ATTACK),int(A.DIRECT_ATTACK)) for x in decisions):
        issues.append('the registered useful attack was skipped')
    # Upstart deliberately gives the opponent 1000 LP before a 1500 attack.
    minimum=500 if variant in ('main1_upstart','main2_engage_value') else 1500
    if scene.get('start_lp',8000)-lp[0]<minimum:
        issues.append('the registered net attack damage was not achieved')
    return issues


def command_configuration(deck, scene):
    variant = scene['variant']
    if variant not in VARIANTS:
        raise ValueError('unknown independently registered command-timing variant')
    defenses = tuple(scene.get('defenses', (WIDOW, WARNING)))
    if defenses not in ((WIDOW, WARNING), (WIDOW, 41420027), (WIDOW, WIDOW)):
        raise ValueError('declare one supported defensive pair')
    spare = scene.get('spare', ASH)
    draw = scene.get('first_draw', VEILER)
    if spare not in (ASH, VEILER, 59438930) or draw not in (ASH, VEILER, 59438930):
        raise ValueError('register passive visible hand variation')
    if variant == 'main2_engage_value':
        hand = (UPSTART, HORNET, ENGAGE, MAXX_C, WIDOW)
    elif variant == 'main1_upstart':
        hand = (UPSTART, RAYE, *defenses, MAXX_C)
    elif variant == 'engage_direct':
        hand = (ENGAGE, RAYE, *defenses, MAXX_C)
    else:
        hand = (RAYE, MAXX_C, *defenses, spare)
    cfg = configuration(deck, seed=scene['seed'], start_lp=scene.get('start_lp', 8000),
                        opening_hand=hand, first_draw=draw)
    opponent_hand = (RAYE, BULB, ASH, VEILER, MAXX_C)
    if not Counter(opponent_hand) <= Counter(deck.main):
        raise ValueError('opponent demonstration resources absent from full deck')
    rest = list((Counter(deck.main) - Counter(opponent_hand)).elements())
    random.Random(scene['seed'] ^ 0x617B3).shuffle(rest)
    return replace(cfg, forced_deck_orders=(tuple(rest) + tuple(reversed(opponent_hand)), cfg.forced_deck_orders[1]))


class CommandTeacher(TimingTeacher):
    def __init__(self, scene):
        normalized = {**scene, 'defenses': list(scene.get('defenses', (WIDOW, WARNING)))}
        if scene['variant'] == 'main2_engage_value':
            normalized['defenses'] = [WIDOW]
        super().__init__(normalized)
        self.variant = scene['variant']
        self.prefix = ([UPSTART, HORNET] if self.variant == 'main2_engage_value' else
                       [UPSTART] if self.variant == 'main1_upstart' else
                       [ENGAGE] if self.variant == 'engage_direct' else [])
        self.prefix_at = 0
        self.engage_late = False
        self.opponent_summoned = False
        self.opponent_raye_activated = False
        self.g_used = False

    def record(self, prompt, driver, index):
        self.choices.append({'player': prompt.player, 'response': len(driver.responses),
                             'turn': prompt.turn, 'phase': prompt.phase, 'msg': prompt.msg,
                             'index': index, 'action': asdict(prompt.actions[index])})
        return index

    @staticmethod
    def idle_answer(actions):
        return next((i for i, a in enumerate(actions) if a.act == A.CANCEL or a.finish), None)

    def __call__(self, prompt, driver):
        actions = prompt.actions
        if prompt.player == 1 and prompt.turn >= 4 and len(actions) > 1:
            raise ComboComplete()
        if prompt.player == 0:
            pick = None
            summon_turn = 1 if self.variant == 'own_turn_g_chain' else 3
            wants_raye = self.variant in ('own_turn_g_chain', 'opponent_turn_g_chain')
            if wants_raye and prompt.turn == summon_turn and prompt.msg == C.MSG_SELECT_IDLECMD and not self.opponent_summoned:
                pick = next((i for i, a in enumerate(actions) if a.act == A.SUMMON and a.code == RAYE), None)
                if pick is not None:
                    self.opponent_summoned = True
            activation_window = (self.variant == 'own_turn_g_chain' and prompt.turn == 2 and
                                 C.PHASE_BATTLE_START <= prompt.phase < C.PHASE_MAIN2 or
                                 self.variant == 'opponent_turn_g_chain' and prompt.turn == 3)
            if pick is None and activation_window and self.opponent_summoned and not self.opponent_raye_activated:
                pick = next((i for i, a in enumerate(actions) if a.act == A.ACTIVATE and a.code == RAYE), None)
                if pick is not None:
                    self.opponent_raye_activated = True
            if pick is None and self.opponent_raye_activated and prompt.msg in (C.MSG_SELECT_CARD, C.MSG_SELECT_UNSELECT_CARD):
                pick = next((i for i, a in enumerate(actions) if a.code == SHIZUKU and a.act != A.CANCEL), None)
            if pick is None and prompt.msg == C.MSG_SELECT_PLACE:
                pick = 0
            if pick is None and prompt.msg == C.MSG_SELECT_POSITION:
                pick = next((i for i, a in enumerate(actions) if a.position == C.POS_FACEUP_ATTACK), 0)
            if pick is None and prompt.msg == C.MSG_SELECT_IDLECMD:
                pick = next((i for i, a in enumerate(actions) if a.phase == P.END), None)
            if pick is None:
                pick = self.idle_answer(actions)
            if pick is None and len(actions) == 1:
                pick = 0
            if pick is None:
                raise ValueError(f'unregistered opponent command-timing answer {prompt.msg}')
            return self.record(prompt, driver, pick)
        if self.opponent_raye_activated and not self.g_used and prompt.msg == C.MSG_SELECT_CHAIN:
            pick = next((i for i, a in enumerate(actions) if a.act == A.ACTIVATE and a.code == MAXX_C), None)
            if pick is not None:
                self.g_used = True
                return self.record(prompt, driver, pick)
        if prompt.turn != 2:
            pick = self.idle_answer(actions)
            if pick is None and len(actions) == 1:
                pick = 0
            if pick is None:
                raise ValueError('unexpected model prompt outside the declared action turn')
            return self.record(prompt, driver, pick)
        if self.engage_late and prompt.msg == C.MSG_SELECT_YESNO:
            pick = next((i for i, a in enumerate(actions) if a.act == A.ACTIVATE and a.code == ENGAGE), None)
            if pick is not None:
                return self.record(prompt, driver, pick)
        if self.attacked and prompt.phase == C.PHASE_BATTLE_START and prompt.msg == C.MSG_SELECT_YESNO:
            # Hayate asks for direct-attack confirmation when the opponent
            # controls the Shizuku summoned by Raye in the positive G case.
            pick = next((i for i, a in enumerate(actions) if a.act == A.ACTIVATE and a.desc == 31), None)
            if pick is not None:
                return self.record(prompt, driver, pick)
        if prompt.msg == C.MSG_SELECT_IDLECMD and prompt.phase == C.PHASE_MAIN1 and self.prefix_at < len(self.prefix):
            code = self.prefix[self.prefix_at]
            pick = next((i for i, a in enumerate(actions) if a.act == A.ACTIVATE and a.code == code and a.spec.startswith('h')), None)
            if pick is None:
                raise ValueError('the registered main1 utility activation is not legal from hand')
            self.prefix_at += 1
            if code == ENGAGE:
                self.intent = ('pick', HORNET)
            if code == HORNET:
                self.stage = 1  # token instead of normal-summoned Raye feeds Hayate
            return self.record(prompt, driver, pick)
        if self.variant == 'main2_engage_value' and prompt.msg == C.MSG_SELECT_IDLECMD and prompt.phase == C.PHASE_MAIN2 and not self.engage_late:
            pick = next((i for i, a in enumerate(actions) if a.act == A.ACTIVATE and a.code == ENGAGE and a.spec.startswith('h')), None)
            if pick is None:
                raise ValueError('the registered post-battle Engage is not legal')
            self.engage_late = True
            self.intent = ('pick', HORNET)
            return self.record(prompt, driver, pick)
        pick = super().__call__(prompt, driver)
        if self.variant == 'main2_engage_value' and actions[pick].act == A.SPSUMMON and actions[pick].code == HAYATE:
            self.intent = ('material', TOKEN)
        if self.variant == 'main2_engage_value' and actions[pick].act == A.ACTIVATE and actions[pick].code == SHIZUKU:
            # Hornet is already in the grave in this route, so Shizuku cannot
            # search another copy of it. The unused Anchor name remains legal.
            self.intent = ('pick', WIDOW)
        return pick


def demonstrate_command_timing(core, deck, scene):
    cfg = command_configuration(deck, scene)
    teacher = CommandTeacher(scene)
    draws = []

    class Host(_PublicTapeHost):
        def _observe(self, message):
            super()._observe(message)
            if message.msg == C.MSG_DRAW:
                draws.append({'turn': self.turn, 'phase': self.phase, 'player': message.payload[0],
                              'count': message.payload[1]})

    with Host(cfg, core, 1) as driver:
        try:
            driver.run(teacher, max_steps=30000, max_seconds=30)
        except ComboComplete:
            pass
        if driver.fallbacks or driver.truncations:
            raise ValueError('command-timing fixture needs every complete legal native menu')
        state = capture(driver)
        return {'schema': SCHEMA + '#command-timing', 'scene': scene, 'choices': teacher.choices,
                'draws': draws, 'lp': list(state.lp), 'turn': driver.turn,
                'own_backrow': [asdict(c) for c in state.zone(1, C.LOCATION_SZONE)],
                'own_hand': [c.code for c in state.zone(1, C.LOCATION_HAND)],
                'own_grave': [c.code for c in state.zone(1, C.LOCATION_GRAVE)],
                'training_eligible': False, 'playing_strength_evidence': False,
                'public_tape': {'schema': SCHEMA + '#public-tape', 'viewer': 1,
                    'own_recipe': {'main': list(deck.main), 'extra': list(deck.extra)},
                    'opponent_recipe_mode': 'mirror', 'private_layouts': False,
                    'messages': [[p[0], p[1:].hex()] for p in driver.public_packets],
                    'responses': [r.hex() for r in driver.public_responses],
                    'terminal': driver.finished, 'winner': driver.winner}}
