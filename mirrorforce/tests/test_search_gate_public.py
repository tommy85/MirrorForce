"""Only the live public board/menu may create tactical trigger flags."""
from types import SimpleNamespace

import pytest

from mirrorforce.netduel import constants as C, agent_search_gate as G
from mirrorforce.netduel.actions import ActionAct, LegalAction, SelectResult, Selector
from mirrorforce.netduel.board import ShadowBoard, ShadowCard


def surface(codes=(85289965, 26077387), *, lp=7500, phase=C.PHASE_MAIN1):
    board = ShadowBoard()
    board.our_player = board.turn_player = 1
    board.phase = phase
    board.zones[(1, C.LOCATION_MZONE)] = [
        ShadowCard(code=code, controller=1, location=C.LOCATION_MZONE, sequence=i,
            type=C.TYPE_MONSTER | (C.TYPE_LINK if code in (85289965, 5821478) else 0),
            attack=3000 if code in (85289965, 5821478) else 1500, position=C.POS_FACEUP_ATTACK)
        for i, code in enumerate(codes)]
    client = SimpleNamespace(board=board, result=SimpleNamespace(our_player=1), _lp=[lp, 8000])
    parsed = SelectResult(C.MSG_SELECT_IDLECMD, 1, selector=Selector(C.MSG_SELECT_IDLECMD, 1,
        [LegalAction(act=ActionAct.ACTIVATE, code=codes[0])], lambda *_: None))
    return client, parsed


def test_public_borrelsword_and_body_trigger_while_current_sum_is_only_4500():
    client, parsed = surface()
    witness = G.public_combat_witness(client, parsed)
    hint = G.hint_from_witness(witness)
    assert sum(hint.visible_own_attacks) == 4500 < hint.opponent_lp
    assert hint.tactical_candidates == frozenset({'borrelsword-double-attack-defense-body'})
    assert G.lethal_candidate(hint, low_lp_threshold=3000)


def test_link_monster_is_not_a_convertible_defense_body():
    client, parsed = surface((85289965, 5821478))
    hint = G.hint_from_witness(G.public_combat_witness(client, parsed))
    assert 'borrelsword-double-attack-defense-body' not in hint.tactical_candidates


def test_hidden_monster_fields_and_both_hands_are_not_read_even_for_a_trigger():
    client, parsed = surface()
    class Hidden:
        hidden = True
        def __getattr__(self, name):
            raise AssertionError('hidden field accessed: ' + name)
    client.board.zones[(0, C.LOCATION_MZONE)] = [Hidden()]
    client.board.zones[(0, C.LOCATION_HAND)] = [Hidden()]
    client.board.zones[(1, C.LOCATION_HAND)] = [Hidden()]
    first = G.public_combat_witness(client, parsed)
    client.board.zones[(0, C.LOCATION_MZONE)] = [Hidden(), Hidden()]
    assert G.public_combat_witness(client, parsed) == first


def test_bomber_activation_trigger_in_main2_and_on_opponent_turn():
    client, parsed = surface((5821478,), phase=C.PHASE_MAIN2)
    client.board.turn_player = 0
    hint = G.hint_from_witness(G.public_combat_witness(client, parsed))
    assert not hint.our_turn and not hint.battle_window
    assert hint.tactical_candidates == frozenset({'bomber-burn-or-clear'})
    assert G.lethal_candidate(hint, low_lp_threshold=3000)


def test_summonable_finisher_uses_only_a_received_legal_action():
    client, parsed = surface((26077387,))
    parsed.selector._actions = [LegalAction(act=ActionAct.SPSUMMON, code=85289965)]
    hint = G.hint_from_witness(G.public_combat_witness(client, parsed))
    assert 'borrelsword-double-attack-defense-body' in hint.tactical_candidates


def test_quiet_setup_and_unproven_combo_remain_explicit_coverage_gaps():
    client, parsed = surface((26077387,))
    hint = G.hint_from_witness(G.public_combat_witness(client, parsed))
    assert not hint.tactical_candidates and not G.lethal_candidate(hint, low_lp_threshold=3000)


def test_witness_rejects_private_extra_fields_and_concealed_card_rows():
    client, parsed = surface()
    witness = G.public_combat_witness(client, parsed)
    with pytest.raises(ValueError, match='private'):
        G.hint_from_witness({**witness, 'opponent_hand': [123]})
    witness['visible_monsters'][0][-1] = C.POS_FACEDOWN_DEFENSE
    with pytest.raises(ValueError, match='concealed'):
        G.hint_from_witness(witness)
