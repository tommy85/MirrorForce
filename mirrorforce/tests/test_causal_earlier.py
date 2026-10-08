"""Full-root uniqueness and earlier-prefix refutation are separate, exact claims."""
import copy
import json
from pathlib import Path

import pytest

from mirrorforce.netduel import causal_pass as P, causal_entropy as E
from mirrorforce.netduel.causal_proof import sha256
from mirrorforce.netduel.conditioned_particles import PublicConditionedCapacityParticles
from test_causal_pass import fixture as pass_fixture
from test_causal_pass_prefix import fixture as prefix_fixture, check, rebind


def fixture():
    data = json.loads((Path(__file__).parent / "fixtures/earlier-pass-contradiction.json").read_text())
    return data["certificate"], data["profile"]


def test_actual_fourth_root_has_a_later_known_set_but_an_earlier_unique_opening_contradiction():
    c, profile = fixture()
    before = copy.deepcopy(c)
    assert check(c, profile)["whole_candidate_impossible"] and c == before
    assert c["law"] == P.EARLY_LAW and c["native_submitted_responses"] == 1
    assert c["history_record"]["own_responses"] == [[20, "ffffffff"], [29, "ffffffff"], [43, "04000100"]]
    assert c["mismatch"]["packet_index"] == 21 and c["received_prefix"][-1].startswith("1200")
    assert PublicConditionedCapacityParticles(profile).identity["negative_certificate"][-1] == P.EARLY_LAW


def test_older_profiles_do_not_acquire_v3_suffix_semantics():
    for old_law, old_profile in ((P.LAW, pass_fixture()[1]), (P.PREFIX_LAW, prefix_fixture()[1])):
        c, _ = fixture()
        c.update(law=old_law, entropy_profile_sha256=sha256(old_profile))
        with pytest.raises(ValueError):
            check(c, old_profile)
    c, profile = fixture()
    for old_law in (P.LAW, P.PREFIX_LAW):
        c["law"] = old_law
        with pytest.raises(ValueError, match="versions"):
            check(c, profile)


def test_reuses_unchanged_old_prefix_proofs_only_by_explicit_new_profile():
    _, profile = fixture()
    for c in (pass_fixture()[0], prefix_fixture()[0]):
        c.update(law=P.EARLY_LAW, entropy_profile_sha256=sha256(profile))
        assert check(c, profile)["whole_candidate_impossible"]
    for source in (pass_fixture()[1], prefix_fixture()[1], profile):
        with pytest.raises(ValueError, match="unchanged"):
            E.earlier_contradiction_profile(source)


@pytest.mark.parametrize("raw", ["020300000000", "4600", "820001"])
def test_later_suffix_is_bound_but_not_claimed_to_be_executed_or_deterministic(raw):
    c, profile = fixture()
    c["received_prefix"][44] = raw  # after contradiction; no token permutation/death/birth in the complete graph
    with pytest.raises(ValueError):
        check(c, profile)  # even a harmless suffix rewrite must be reflected in full tape/history bindings
    rebind(c)
    assert check(c, profile)["whole_candidate_impossible"]


@pytest.mark.parametrize("mutation", [
    lambda c: c["history_record"]["own_responses"].append([45, "000800"]),  # pending was answered
    lambda c: c["history_record"]["own_responses"].append([46, "000800"]),
    lambda c: c["history_record"]["own_responses"].insert(0, [-1, "ffffffff"]),
    lambda c: c["history_record"]["own_responses"].insert(1, [20, "ffffffff"]),
    lambda c: c["history_record"]["own_responses"].reverse(),
    lambda c: c["history_record"]["own_responses"][2].__setitem__(0, True),
    lambda c: c["history_record"]["own_responses"][2].__setitem__(1, "FF"),
    lambda c: c["history_record"]["own_responses"][2].__setitem__(1, ""),
    lambda c: c["history_record"]["own_responses"][0].__setitem__(1, "04000100"),
    lambda c: c["history_record"]["own_responses"].pop(0),
    lambda c: c["submitted_trace"][0].update(response_hex="04000100"),
    lambda c: c["submitted_trace"][0].update(received_cursor=44),
    lambda c: c["native_prefix"].insert(-1, "3200"),  # a MOVE before contradiction is not covered
    lambda c: c["native_prefix"].insert(-1, "0b00"),  # native executed/answered IDLE is not a pass
    lambda c: c["native_prefix"].insert(-1, "8200"),  # chance before contradiction
    lambda c: c["received_prefix"].__setitem__(19, "3200"),
    lambda c: c.update(opponent_submitted_responses=1),
    lambda c: c.update(branches=1), lambda c: c.update(choices=1),
    lambda c: c.update(native_submitted_responses=4, own_answered=4),
    lambda c: c["history_record"]["permutations"].append({"packet": 44}),
    lambda c: c["history_record"]["dead"].append(1),
    lambda c: c["history_record"]["tokens"][0].update(created_at=44),
    lambda c: c["history_record"]["root_slots"].pop(),
    lambda c: c["history_record"]["root_slots"][0].__setitem__(3, c["history_record"]["root_slots"][1][3]),
    lambda c: c["plan_record"]["births"].append([1]),
    lambda c: c["plan_record"]["openings"][0][2].pop(),
])
def test_rehashed_nonunique_opening_or_unreviewed_executed_prefix_is_not_a_negative(mutation):
    c, profile = fixture()
    mutation(c)
    rebind(c)
    with pytest.raises(ValueError):
        check(c, profile)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(scope=P.PREFIX_SCOPE),
    lambda p: p["own_chain_pass_lemma"].update(law=P.PREFIX_LAW),
    lambda p: p["own_chain_pass_lemma"].update(suffix="deterministic"),
    lambda p: p["own_chain_pass_lemma"].update(opening="one-guessed-opening"),
    lambda p: p["own_chain_pass_lemma"].update(native_messages=[2, 11, 16, 40, 41, 90]),
    lambda p: p.update(recipe_sha256="0" * 64),
])
def test_profile_cannot_claim_general_action_determinism_or_guess_an_opening(mutation):
    c, profile = fixture()
    mutation(profile)
    c["entropy_profile_sha256"] = sha256(profile)
    with pytest.raises(ValueError):
        check(c, profile)
