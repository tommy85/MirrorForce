"""CPU-only admission and full prompt wall-clock accounting; no runtime install."""
from dataclasses import replace
import math

import pytest

from mirrorforce.netduel import agent_search_gate as G


def hint(**changes):
    base = G.PublicCombatHint(True, True, 8000, (1500,), frozenset({"attack"}))
    return replace(base, **changes)


def begin(gate, *, turn=1, started=100., search=137., response=140.):
    gate.begin_prompt(turn=turn, started=started, search_deadline=search, response_deadline=response)


def decide(gate, *, logits=(0., 0.), now=100.5, public=None):
    return gate.decide(logits=logits, rows=len(logits), hint=public or hint(), now=now)


def test_complete_menu_entropy_margin_ties_and_shift_invariance():
    logits = [2., 2., -5.]
    result = G.policy_confidence(logits, rows=3)
    shifted = G.policy_confidence([v + 900 for v in logits], rows=3)
    assert result == shifted and result.raw_greedy_index == 0 and result.top_two_margin == 0
    assert .6 < result.normalized_entropy < .7
    assert logits == [2., 2., -5.]
    assert G.policy_confidence([0.] * 128, rows=128).normalized_entropy == pytest.approx(1.)
    extreme = G.policy_confidence([1e308, -1e308], rows=2)
    assert (extreme.normalized_entropy, extreme.top_two_margin) == (0., 1.)


@pytest.mark.parametrize("logits,rows", [([], 0), ([0., 1.], 1), ([0., math.nan], 2),
                                        ([0., math.inf], 2), ([False, 1.], 2), ([0.], True)])
def test_no_masking_dropping_or_sanitizing_original_legal_rows(logits, rows):
    with pytest.raises(ValueError, match="complete finite legal menu"):
        G.policy_confidence(logits, rows=rows)


def test_default_off_preserves_original_greedy_even_with_lethal_hint():
    gate = G.TurnSearchGate()
    begin(gate)
    pick = decide(gate, logits=(9., 10.), public=hint(opponent_lp=100))
    assert not pick.search and pick.reason == "disabled" and pick.confidence.raw_greedy_index == 1
    assert pick.response_deadline == 140.


def test_uncertainty_uses_entropy_or_margin_and_high_confidence_skips():
    gate = G.TurnSearchGate(G.GateConfig(enabled=True))
    begin(gate)
    confident = decide(gate, logits=(0., 9.))
    assert not confident.search and confident.reason == "confident"
    uncertain = decide(gate, logits=(1., 1.), now=101.)
    assert uncertain.search and uncertain.reason == "uncertain"
    assert uncertain.search_deadline == 105. and uncertain.response_deadline == 108.
    assert uncertain.spent_this_turn == 1.


def test_visible_damage_potential_does_not_depend_on_low_confidence():
    gate = G.TurnSearchGate(G.GateConfig(enabled=True))
    begin(gate)
    pick = decide(gate, logits=(99., 0.), public=hint(opponent_lp=3000, visible_own_attacks=(3000,)))
    assert pick.search and pick.reason == "lethal_candidate"
    assert pick.confidence.top_probability == pytest.approx(1.)
    assert pick.search_deadline == 109. and pick.response_deadline == 112.


def test_public_borrelsword_flag_covers_7500_line_beyond_current_attack_sum():
    public = hint(opponent_lp=7500, visible_own_attacks=(3000, 1500),
                  tactical_candidates=frozenset({"borrelsword-double-attack-defense-body"}))
    assert sum(public.visible_own_attacks) < public.opponent_lp
    assert G.lethal_candidate(public, low_lp_threshold=3000)
    gate = G.TurnSearchGate(G.GateConfig(enabled=True))
    begin(gate)
    pick = decide(gate, logits=(30., 0.), public=public)
    assert pick.search and pick.lethal_candidate
    # The output admits ordinary current-root search; it contains no WIN,
    # damage proof, tactical action choice, or synthesized response bytes.
    assert not hasattr(pick, "winner") and not hasattr(pick, "response")


def test_public_bomber_trigger_can_be_outside_own_battle_window():
    public = hint(our_turn=False, battle_window=False, visible_own_attacks=(),
                  tactical_candidates=frozenset({"bomber-burn-or-clear"}))
    assert G.lethal_candidate(public, low_lp_threshold=3000)
    assert not G.lethal_candidate(replace(public, tactical_candidates=frozenset()), low_lp_threshold=3000)
    assert not G.lethal_candidate(replace(public, opponent_lp=0), low_lp_threshold=3000)


def test_singleton_is_fast_even_with_public_tactical_candidate():
    gate = G.TurnSearchGate(G.GateConfig(enabled=True))
    begin(gate)
    pick = decide(gate, logits=(0.,), public=hint(opponent_lp=100))
    assert not pick.search and pick.reason == "singleton" and pick.confidence.raw_greedy_index == 0


def test_proposed_one_second_cap_cannot_hide_existing_three_second_cleanup():
    gate = G.TurnSearchGate(G.GateConfig(enabled=True, uncertain_seconds=1.))
    begin(gate)
    pick = decide(gate)
    assert not pick.search and pick.reason == "no_search_budget"
    assert pick.response_deadline == 140.  # Original safe RPC/response still completes.


def test_original_forward_preparation_and_cleanup_all_consume_the_turn():
    gate = G.TurnSearchGate(G.GateConfig(enabled=True))
    begin(gate)
    # Original forward+root preparation use 2s before admission.
    pick = decide(gate, now=102.)
    assert pick.search and pick.search_deadline == 105.
    # Cleanup and actual send happen after rollout; all six seconds count.
    first = gate.finish_prompt(now=106., delivered=True, cleanup_complete=True)
    assert first["prompt_seconds"] == 6. and first["turn_work_seconds"] == 6.
    # Pure-policy decisions also charge their original RPC, without charging
    # unrelated peer wait from 106 to 120.
    begin(gate, started=120., search=157., response=160.)
    assert not decide(gate, logits=(100., 0.), now=121.).search
    second = gate.finish_prompt(now=143., delivered=True, cleanup_complete=True)
    assert second["turn_work_seconds"] == 29.
    begin(gate, started=145., search=182., response=185.)
    blocked = decide(gate, now=145.2, public=hint(opponent_lp=100))
    assert not blocked.search and blocked.reason == "no_search_budget"
    assert blocked.spent_this_turn == pytest.approx(29.2)


def test_subdecisions_cannot_refresh_or_upgrade_original_prompt_deadline():
    gate = G.TurnSearchGate(G.GateConfig(enabled=True))
    begin(gate)
    first = decide(gate)
    second = decide(gate, now=102., public=hint(opponent_lp=100))
    assert second.search_deadline == first.search_deadline == 105.
    assert second.response_deadline == first.response_deadline == 108.
    expired = decide(gate, now=105., public=hint(opponent_lp=100))
    assert not expired.search and expired.reason == "no_search_budget"


def test_received_public_deadline_is_never_extended():
    gate = G.TurnSearchGate(G.GateConfig(enabled=True))
    begin(gate, search=102., response=104.)
    pick = decide(gate, public=hint(opponent_lp=100))
    assert pick.search and pick.search_deadline == 101. and pick.response_deadline == 104.


def test_failed_cleanup_is_charged_and_cannot_be_reused_as_safe_fallback():
    gate = G.TurnSearchGate(G.GateConfig(enabled=True))
    begin(gate)
    decide(gate)
    report = gate.finish_prompt(now=141., delivered=False, cleanup_complete=False)
    assert report["turn_work_seconds"] == 41. and report["over_turn_cap"]
    assert report["over_response_deadline"] and report["over_admitted_prompt_cap"]
    with pytest.raises(ValueError, match="lifecycle"):
        begin(gate, turn=2, started=150., search=160., response=165.)


def test_new_turn_resets_only_after_previous_prompt_completed():
    gate = G.TurnSearchGate(G.GateConfig(enabled=True))
    begin(gate)
    with pytest.raises(ValueError, match="lifecycle"):
        begin(gate, turn=2)
    gate.finish_prompt(now=120., delivered=True, cleanup_complete=True)
    begin(gate, turn=2, started=150., search=187., response=190.)
    assert decide(gate, now=150.5).spent_this_turn == .5
    with pytest.raises(ValueError, match="backwards"):
        gate.finish_prompt(now=150., delivered=True, cleanup_complete=True)


def test_expired_search_deadline_keeps_original_response_deadline_for_fast_skip():
    gate = G.TurnSearchGate(G.GateConfig(enabled=True))
    begin(gate, search=99., response=110.)
    pick = decide(gate)
    assert not pick.search and pick.reason == 'no_search_budget' and pick.response_deadline == 110.


def test_successful_send_cleanup_overrun_blocks_later_search_without_undoing_response():
    gate = G.TurnSearchGate(G.GateConfig(enabled=True))
    begin(gate)
    assert decide(gate).search
    done = gate.finish_prompt(now=109., delivered=True, cleanup_complete=True)
    assert done['over_admitted_prompt_cap'] and not gate.failed and gate.search_stopped
    begin(gate, turn=1, started=150., search=187., response=190.)
    pick = decide(gate, now=150.1, public=hint(opponent_lp=100))
    assert not pick.search and pick.reason == 'previous_overrun'
    gate.finish_prompt(now=151., delivered=True, cleanup_complete=True)
    begin(gate, turn=2, started=200., search=237., response=240.)
    assert decide(gate, now=200.1, public=hint(opponent_lp=100)).search


@pytest.mark.parametrize("change", [{"enabled": 1}, {"entropy_threshold": 1.1}, {"margin_threshold": -1},
                                    {"uncertain_seconds": float("nan")}, {"lethal_seconds": 1.},
                                    {"turn_seconds": 3.}, {"finalize_seconds": 0.}, {"low_lp_threshold": True}])
def test_invalid_config_cannot_silently_relax_the_gate(change):
    with pytest.raises(ValueError):
        G.GateConfig(**change)


def test_public_hint_schema_has_no_oracle_or_hidden_truth_input():
    with pytest.raises(TypeError):
        G.PublicCombatHint(True, True, 8000, (), frozenset(), opponent_hand=[1, 2])
    with pytest.raises(ValueError, match="public combat"):
        hint(tactical_candidates=frozenset({"teacher-says-win"}))
    # Detection coverage remains narrow: no public tactical flag means no
    # assertion about a deeper combo starting from this quiet high-LP board.
    assert not G.lethal_candidate(hint(), low_lp_threshold=3000)
