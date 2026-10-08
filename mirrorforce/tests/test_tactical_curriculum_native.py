"""Constructed tactical curriculum: native outcomes, not a deployed policy.

Only this offline oracle constructs a full board. The first-turn battle flag
is explicit; these are fresh puzzles, never reconstructions of a user's game.
Pairs share a board and differ in the line. A passing test does NOT mean the
network learned the line or that the online tactical solver was admitted.
"""
from dataclasses import asdict
import json

import pytest

from mirrorforce.effectinfo import get_effectinfo_core
from mirrorforce.netduel import constants as C
from mirrorforce.netduel.actions import ActionAct as A, ActionPhase as P
from mirrorforce.search.rebuild import BoardSpec, Placement, DUEL_ATTACK_FIRST_TURN, rebuild
from mirrorforce.worldmodel.engine import DeckList, DuelConfig
from mirrorforce.worldmodel.state import capture

PRIVILEGED_TARGET = True
BORRELSWORD, RAYE, GENE = 85289965, 26077387, 69247929
ENGAGE, TWIN, SHARK, AFTERBURNERS, KAGARI = 63166095, 43898403, 51227866, 99550630, 63288573


class EndOfLine(Exception):
    pass


def placement(code, player, location, sequence=0, position=C.POS_FACEUP_ATTACK):
    return Placement(code, player, player, location, sequence, position, proc=True)


def borrelsword_spec(opponent_lp=8000):
    cards = [placement(BORRELSWORD, 0, C.LOCATION_MZONE, 5),
             placement(RAYE, 0, C.LOCATION_MZONE),
             placement(GENE, 1, C.LOCATION_MZONE)]
    # No draw or deck-dependent effect is used. Keep decks nonempty nevertheless.
    for player in (0, 1):
        cards.extend(placement(GENE, player, C.LOCATION_DECK, i,
                               C.POS_FACEDOWN_DEFENSE) for i in range(3))
    return BoardSpec(lp=(8000, opponent_lp), draw_count=(0, 0),
                     duel_options=DUEL_ATTACK_FIRST_TURN, placements=tuple(cards))


def run_borrelsword(core, *, early_flip=False, accept_buff=True, opponent_lp=8000):
    empty = DeckList('constructed-borrelsword-curriculum', (), ())
    spec = borrelsword_spec(opponent_lp)
    config = DuelConfig((empty, empty), seed=2026100401, start_hand=0, draw_count=0,
                       duel_options=(5 << 16) | DUEL_ATTACK_FIRST_TURN,
                       full_phase_menu=True, full_card_sort_menu=True,
                       auto_end_phase_discard=False)
    driver = rebuild(spec, seed=config.seed, config=config, core=core)
    driver.record_messages = True
    sword_attacks = weak_attacks = 0
    flipped = weak_finished = False
    selected_for = None
    choices = []

    def choose(prompt, live):
        nonlocal sword_attacks, weak_attacks, flipped, weak_finished, selected_for
        actions = prompt.actions
        assert not prompt.truncated and prompt.complete_menu
        if prompt.phase == C.PHASE_MAIN2 or prompt.turn > 1:
            raise EndOfLine()
        # An attack command is not yet damage: changing the attacker to defense
        # in its attack-response window would cancel that attack.
        if prompt.msg == C.MSG_SELECT_BATTLECMD and weak_attacks == 1:
            weak_finished = True
        flip_now = not flipped and (early_flip or weak_finished)
        chosen = None
        if prompt.msg == C.MSG_SELECT_EFFECTYN:
            # The only optional trigger in this exact board is Sword's ATK gain.
            chosen = next(i for i, a in enumerate(actions)
                          if a.act == (A.ACTIVATE if accept_buff else A.CANCEL))
        elif prompt.msg == C.MSG_SELECT_CHAIN:
            for i, action in enumerate(actions):
                if action.act != A.ACTIVATE or action.code != BORRELSWORD:
                    continue
                if action.desc == BORRELSWORD * 16 + 1 and accept_buff:
                    chosen = i
                elif action.desc == BORRELSWORD * 16 and flip_now:
                    chosen, flipped, selected_for = i, True, 'flip'
        elif prompt.msg == C.MSG_SELECT_IDLECMD:
            if flip_now:
                chosen = next((i for i, a in enumerate(actions)
                               if a.act == A.ACTIVATE and a.code == BORRELSWORD
                               and a.desc == BORRELSWORD * 16), None)
                if chosen is not None:
                    flipped, selected_for = True, 'flip'
            if chosen is None:
                chosen = next(i for i, a in enumerate(actions) if a.phase == P.BATTLE)
        elif prompt.msg == C.MSG_SELECT_BATTLECMD:
            want = BORRELSWORD if sword_attacks == 0 else RAYE if not early_flip and weak_attacks == 0 else None
            if want is not None:
                chosen = next((i for i, a in enumerate(actions)
                               if a.code == want and a.act in (A.ATTACK, A.DIRECT_ATTACK)), None)
                if chosen is not None:
                    sword_attacks += want == BORRELSWORD
                    weak_attacks += want == RAYE
                    selected_for = 'attack'
            if chosen is None and flip_now:
                chosen = next((i for i, a in enumerate(actions)
                               if a.act == A.ACTIVATE and a.code == BORRELSWORD
                               and a.desc == BORRELSWORD * 16), None)
                if chosen is not None:
                    flipped, selected_for = True, 'flip'
            if chosen is None and flipped:
                chosen = next((i for i, a in enumerate(actions)
                               if a.code == BORRELSWORD and a.act in (A.ATTACK, A.DIRECT_ATTACK)), None)
                if chosen is not None:
                    sword_attacks += 1
                    selected_for = 'attack'
            if chosen is None:
                raise EndOfLine()
        elif prompt.msg == C.MSG_SELECT_CARD:
            want = RAYE if selected_for == 'flip' else GENE
            chosen = next((i for i, a in enumerate(actions) if a.code == want), None)
        if chosen is None:
            chosen = next((i for i, a in enumerate(actions) if a.act == A.CANCEL), None)
        if chosen is None and len(actions) == 1:
            chosen = 0
        assert chosen is not None, (prompt.msg, selected_for, [asdict(a) for a in actions])
        choices.append({'player': prompt.player, 'msg': prompt.msg, 'phase': prompt.phase,
                        'action': asdict(actions[chosen])})
        return chosen

    try:
        try:
            driver.run(choose, max_steps=10000, max_seconds=20)
        except EndOfLine:
            pass
        state = capture(driver)
        return {'winner': driver.winner, 'finished': driver.finished, 'lp': state.lp,
                'sword_attacks': sword_attacks, 'weak_attacks': weak_attacks,
                'flipped': flipped, 'choices': choices,
                'native_responses': [bytes(r).hex() for r in driver.responses],
                'training_updates': 0, 'search_admission': False,
                'construction': 'fresh_puzzle_first_turn_battle_explicit'}
    finally:
        driver.close()


@pytest.mark.parametrize('early_flip,accept_buff,opponent_lp,win,remaining', [
    (False, True, 8000, True, 0),
    (True, True, 8000, False, 1000),
    (False, False, 8000, False, 2500),
    (False, True, 8600, False, 100),
])
def test_borrelsword_order_and_attack_effect_are_both_needed(
        early_flip, accept_buff, opponent_lp, win, remaining):
    result = run_borrelsword(get_effectinfo_core(), early_flip=early_flip,
                            accept_buff=accept_buff, opponent_lp=opponent_lp)
    assert (result['winner'] == 0) == win, result
    assert result['sword_attacks'] == 2, result
    assert (result['lp'][1] <= 0 if win else result['lp'][1] == remaining), result
    print('BORRELSWORD-CURRICULUM '+json.dumps(result, sort_keys=True))


def run_set_timing(core, *, family, early_set):
    """A declared opponent response policy, not a claim over every defense.

    Opponent's pre-Set Twin has discard fodder. Raye's grave effect is triggered
    by Afterburners in Main Phase, not a battle's restricted Damage Step.
    """
    if family == 'engage_twin':
        cards = [placement(ENGAGE, 0, C.LOCATION_HAND),
                 placement(RAYE, 0, C.LOCATION_DECK),
                 placement(TWIN, 1, C.LOCATION_SZONE, position=C.POS_FACEDOWN_DEFENSE),
                 placement(GENE, 1, C.LOCATION_HAND)]
    elif family == 'shark_raye':
        cards = [placement(SHARK, 0, C.LOCATION_HAND),
                 placement(AFTERBURNERS, 0, C.LOCATION_HAND, 1),
                 placement(KAGARI, 1, C.LOCATION_MZONE),
                 placement(RAYE, 1, C.LOCATION_GRAVE)]
    else:
        raise ValueError(family)
    for player in (0, 1):
        cards.extend(placement(GENE, player, C.LOCATION_DECK, i + 1,
                               C.POS_FACEDOWN_DEFENSE) for i in range(3))
    spec = BoardSpec(lp=(8000, 8000), draw_count=(0, 0), placements=tuple(cards))
    empty = DeckList('constructed-set-timing-curriculum', (), ())
    cfg = DuelConfig((empty, empty), seed=2026100402, start_hand=0, draw_count=0,
                     full_phase_menu=True, full_card_sort_menu=True, auto_end_phase_discard=False)
    driver = rebuild(spec, seed=cfg.seed, config=cfg, core=core)
    set_done = initial_play = response_play = raye_trigger = False
    choices = []

    def choose(prompt, live):
        nonlocal set_done, initial_play, response_play, raye_trigger
        actions = prompt.actions
        chosen = None
        assert not prompt.truncated and prompt.complete_menu
        if prompt.msg == C.MSG_SELECT_IDLECMD and prompt.player == 0:
            if early_set and not set_done:
                wanted = ENGAGE if family == 'engage_twin' else SHARK
                chosen = next(i for i, a in enumerate(actions) if a.act == A.SET and a.code == wanted)
                set_done = True
            elif not initial_play and (family == 'shark_raye' or not early_set):
                wanted = AFTERBURNERS if family == 'shark_raye' else ENGAGE
                chosen = next(i for i, a in enumerate(actions) if a.act == A.ACTIVATE and a.code == wanted)
                initial_play = True
            else:
                raise EndOfLine()
        elif prompt.msg == C.MSG_SELECT_EFFECTYN and family == 'shark_raye' and prompt.player == 1:
            # This board has only Raye's optional graveyard trigger. The core's
            # EFFECTYN description can be the generic special-summon hint.
            chosen = next(i for i, a in enumerate(actions) if a.act == A.ACTIVATE)
            raye_trigger = True
        elif prompt.msg in (C.MSG_SELECT_CHAIN, C.MSG_SELECT_EFFECTYN):
            for i, a in enumerate(actions):
                if a.act != A.ACTIVATE:
                    continue
                if family == 'engage_twin' and prompt.player == 1 and a.code == TWIN:
                    chosen, response_play = i, True
                elif family == 'shark_raye' and prompt.player == 1 and a.desc == RAYE * 16 + 1:
                    chosen, raye_trigger = i, True
                elif family == 'shark_raye' and prompt.player == 0 and a.code == SHARK and raye_trigger:
                    chosen, response_play = i, True
        elif prompt.msg == C.MSG_SELECT_CARD:
            if family == 'engage_twin':
                wanted = RAYE if prompt.player == 0 else GENE
                chosen = next((i for i, a in enumerate(actions) if a.code == wanted), None)
                if chosen is None and prompt.player == 1:
                    # A facedown Engage is not identified to the opponent.
                    chosen = next((i for i, a in enumerate(actions) if a.spec == 'os1'), None)
            else:
                wanted = RAYE if response_play else KAGARI
                chosen = next((i for i, a in enumerate(actions) if a.code == wanted), None)
        elif prompt.msg in (C.MSG_SELECT_PLACE, C.MSG_SELECT_POSITION):
            chosen = 0
        if chosen is None:
            chosen = next((i for i, a in enumerate(actions) if a.act == A.CANCEL or a.finish), None)
        if chosen is None and len(actions) == 1:
            chosen = 0
        assert chosen is not None, (prompt.msg, prompt.player, [asdict(a) for a in actions])
        choices.append({'player': prompt.player, 'msg': prompt.msg, 'action': asdict(actions[chosen])})
        return chosen

    try:
        with pytest.raises(EndOfLine):
            driver.run(choose, max_steps=10000, max_seconds=20)
        state = capture(driver)
        codes = lambda player, location: [c.code for c in state.zone(player, location)]
        return {'own_hand': codes(0, C.LOCATION_HAND), 'own_grave': codes(0, C.LOCATION_GRAVE),
                'opponent_field': codes(1, C.LOCATION_MZONE),
                'opponent_grave': codes(1, C.LOCATION_GRAVE),
                'opponent_removed': codes(1, C.LOCATION_REMOVED),
                'response_play': response_play, 'raye_trigger': raye_trigger,
                'choices': choices, 'native_responses': [bytes(r).hex() for r in driver.responses],
                'construction': 'fresh_puzzle_explicit_private_opponent',
                'training_updates': 0, 'search_admission': False}
    finally:
        driver.close()


@pytest.mark.parametrize('early_set', [False, True])
def test_twin_destroys_set_engage_but_does_not_negate_activated_engage(early_set):
    result = run_set_timing(get_effectinfo_core(), family='engage_twin', early_set=early_set)
    assert result['response_play'], result
    assert ENGAGE in result['own_grave'], result
    assert (RAYE in result['own_hand']) == (not early_set), result
    assert GENE in result['opponent_grave'], result  # actual discard cost
    print('SET-CURRICULUM '+json.dumps(result, sort_keys=True))


@pytest.mark.parametrize('early_set', [False, True])
def test_shark_kept_in_hand_answers_raye_but_newly_set_shark_cannot(early_set):
    result = run_set_timing(get_effectinfo_core(), family='shark_raye', early_set=early_set)
    assert result['raye_trigger'], result
    assert result['response_play'] == (not early_set), result
    assert (RAYE in result['opponent_field']) == early_set, result
    assert (RAYE in result['opponent_removed']) == (not early_set), result
    print('SET-CURRICULUM '+json.dumps(result, sort_keys=True))
