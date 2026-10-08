"""v2 certifies an earlier contradiction, without claiming a later menu was executed."""
import copy
import hashlib
import json
from pathlib import Path
import struct

import pytest

from mirrorforce.netduel import causal_pass as P, causal_entropy as E
from mirrorforce.netduel.causal_rejection import verify_negative_certificate
from mirrorforce.netduel.causal_proof import sha256
from mirrorforce.netduel.conditioned_particles import PublicConditionedCapacityParticles
from test_causal_pass import fixture as old_fixture


def fixture():
    row = json.loads((Path(__file__).parent / "fixtures/own-chain-pass-prefix-negative.json").read_text())
    return row["certificate"], row["profile"]


def check(c, profile):
    return verify_negative_certificate(c, entropy_profile=profile,
        **{key: c[key] for key in ("hypothesis_sha256", "history_sha256", "source_sha256")})


def rebind(c):
    """Adversarial records get consistent hashes, so semantic guards must catch them."""
    received = [bytes.fromhex(raw) for raw in c["received_prefix"]]
    h = c["history_record"]
    h["public_prefix_sha256"] = hashlib.sha256(b"".join(struct.pack("<I", len(raw)) + raw for raw in received)).hexdigest()
    h["pending"] = c["received_prefix"][-1]
    c["history_sha256"] = c["plan_record"]["history_sha256"] = sha256(h)


def test_actual_third_root_keeps_whole_tape_but_only_one_submitted_pass_before_contradiction():
    c, profile = fixture()
    before = copy.deepcopy(c)
    assert check(c, profile)["whole_candidate_impossible"] and c == before
    assert c["law"] == P.PREFIX_LAW and c["native_submitted_responses"] == 1
    assert c["history_record"]["own_responses"] == [[20, "ffffffff"], [29, "ffffffff"]]
    assert c["mismatch"]["packet_index"] == 21 and c["received_prefix"][-1].startswith("0b00")
    assert PublicConditionedCapacityParticles(profile).identity["negative_certificate"][-1] == P.PREFIX_LAW


def test_v1_stays_closed_for_third_root_even_with_valid_old_profile_and_rebound_digest():
    c, _ = fixture()
    _, profile, _ = old_fixture()
    c.update(law=P.LAW, entropy_profile_sha256=sha256(profile))
    with pytest.raises(ValueError, match="unreviewed"):
        check(c, profile)


def test_old_second_root_is_still_accepted_by_exact_old_profile_and_explicit_v2():
    c, profile, _ = old_fixture()
    assert check(c, profile)["whole_candidate_impossible"]
    _, new = fixture()
    c.update(law=P.PREFIX_LAW, entropy_profile_sha256=sha256(new))
    assert check(c, new)["whole_candidate_impossible"]
    for changed in (profile, new):
        with pytest.raises(ValueError, match="unchanged"):
            E.own_pass_prefix_profile(changed)


@pytest.mark.parametrize("mutation", [
    lambda c: c["received_prefix"].__setitem__(31, "03"),  # opponent waiting, not an own pass
    lambda c: c["received_prefix"].__setitem__(31, "3200"),  # executed MOVE
    lambda c: c["received_prefix"].__setitem__(31, "3600"),  # executed SUMMONING
    lambda c: c["received_prefix"].__setitem__(31, "4600"),  # executed CHAINING
    lambda c: c["received_prefix"].__setitem__(31, "2000"),  # shuffle/chance
    lambda c: c["received_prefix"].__setitem__(31, "2801"),  # later turn
    lambda c: c["received_prefix"].__setitem__(30, "290800"),  # beyond first main phase
    lambda c: c["received_prefix"].__setitem__(31, c["received_prefix"][-1]),  # earlier IDLE, therefore not pending
    lambda c: c["received_prefix"].__setitem__(-1, "0b01"),  # other player's pending menu
    lambda c: c["received_prefix"].__setitem__(-1, "0b00"),  # truncated own pending menu
    lambda c: c["history_record"]["own_responses"][1].__setitem__(1, "00000000"),
    lambda c: c["history_record"]["own_responses"].pop(),
    lambda c: c["history_record"]["own_responses"].pop(0),
    lambda c: c["history_record"]["own_responses"][0].__setitem__(0, 29),
    lambda c: c["history_record"]["own_responses"].append([43, "ffffffff"]),  # IDLE was answered
    lambda c: c["submitted_trace"][0].update(received_cursor=30),
    lambda c: c["submitted_trace"][0].update(response_hex="00000000"),
    lambda c: c.update(opponent_submitted_responses=1),
    lambda c: c.update(branches=1), lambda c: c.update(choices=1),
    lambda c: c["history_record"]["permutations"].append({"packet": 31}),
    lambda c: c["history_record"]["dead"].append(1),
    lambda c: c["plan_record"]["births"].append([1]),
])
def test_rehashed_action_opponent_cut_or_response_suffix_is_still_not_a_negative(mutation):
    c, profile = fixture()
    mutation(c)
    rebind(c)
    with pytest.raises(ValueError):
        check(c, profile)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(scope=P.SCOPE),
    lambda p: p["own_chain_pass_lemma"].update(law=P.LAW),
    lambda p: p["own_chain_pass_lemma"].update(pending_idle="arbitrary-action"),
    lambda p: p["own_chain_pass_lemma"].update(received_scope="whole-history-deterministic"),
    lambda p: p["own_chain_pass_lemma"].update(native_messages=[2, 11, 16, 40, 41, 90]),
    lambda p: p.update(math_members={}),
])
def test_prefix_profile_cannot_silently_widen_native_or_suffix_scope(mutation):
    c, profile = fixture()
    mutation(profile)
    c["entropy_profile_sha256"] = sha256(profile)
    with pytest.raises(ValueError):
        check(c, profile)
