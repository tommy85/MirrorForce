"""An actual own-pass counterexample is narrowly certified; unsupported traces remain unknown."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mirrorforce.netduel import causal_pass as P, causal_entropy as E
from mirrorforce.netduel.causal_rejection import verify_negative_certificate, verify_entropy_profile
from mirrorforce.netduel.causal_proof import sha256
from mirrorforce.netduel.conditioned_particles import PublicConditionedCapacityParticles


def fixture():
    row = json.loads((Path(__file__).parent / "fixtures/own-chain-pass-negative.json").read_text())
    certificate = row["certificate"]
    binding = {k: certificate[k] for k in ("hypothesis_sha256", "history_sha256", "source_sha256")}
    return certificate, row["profile"], binding


def test_actual_record_has_full_pass_trace_original_tape_and_explicit_new_profile():
    c, profile, binding = fixture()
    before = copy.deepcopy(c)
    result = verify_negative_certificate(c, **binding, entropy_profile=profile)
    assert result["whole_candidate_impossible"] and not result["native_replay_rerun"]
    assert c == before and c["law"] == P.LAW and c["own_answered"] == 1
    assert c["submitted_trace"] == [{"native_index": 6, "player": 0, "received_cursor": 21,
        "response_hex": "ffffffff", "prompt_hex": c["native_prefix"][6]}]
    provider = PublicConditionedCapacityParticles(profile)
    assert provider.identity["negative_certificate"][-1] == P.LAW
    assert provider.identity["entropy_profile_sha256"] == sha256(profile)


@pytest.mark.parametrize("change", [
    lambda c: c.update(law="unique-opening-no-response-deterministic-prefix/v1"),
    lambda c: c.update(opponent_submitted_responses=1), lambda c: c.update(branches=1),
    lambda c: c.update(choices=1), lambda c: c.update(native_submitted_responses=0),
    lambda c: c.update(own_answered=0), lambda c: c.update(native_prefix=c["native_prefix"][:-1]),
    lambda c: c["native_prefix"].insert(0, "3200"), lambda c: c["native_prefix"].insert(0, "2801"),
    lambda c: c["native_prefix"].insert(0, "290800"), lambda c: c["native_prefix"].insert(0, "2d"),
    lambda c: c["native_prefix"].insert(0, "5a000100000000"),
    lambda c: c["received_prefix"].__setitem__(0, "03"),
    lambda c: c["submitted_trace"][0].update(player=1),
    lambda c: c["submitted_trace"][0].update(player=False),
    lambda c: c["submitted_trace"][0].update(response_hex="00000000"),
    lambda c: c["submitted_trace"][0].update(received_cursor=22),
    lambda c: c["submitted_trace"][0].update(native_index=7),
    lambda c: c["submitted_trace"].clear(),
    lambda c: c["mismatch"].update(synthetic_packet_hex="290100"),
    lambda c: c["mismatch"].update(packet_index=100000),
    lambda c: c["history_record"]["own_responses"][0].__setitem__(1, "00000000"),
])
def test_any_unobserved_response_action_randomness_or_tape_rewrite_invalidates_negative(change):
    c, profile, binding = fixture()
    change(c)
    with pytest.raises(ValueError):
        verify_negative_certificate(c, **binding, entropy_profile=profile)


def test_forced_chain_pass_is_not_a_legal_fixed_response_even_if_response_bytes_are_minus_one():
    c, _, _ = fixture()
    prompt = bytearray.fromhex(c["submitted_trace"][0]["prompt_hex"])
    P._chain(bytes(prompt), player=0, passing=True)
    prompt[13] = 1  # full per-row CHAIN_FORCED flag; not inferred from spe_count
    with pytest.raises(ValueError, match="forced"):
        P._chain(bytes(prompt), player=0, passing=True)


@pytest.mark.parametrize("mutation", [lambda p: p.update(scope="pinned-stage-pre-first-response/v1"),
    lambda p: p["own_chain_pass_lemma"].update(native_messages=[2, 16, 40, 41, 90, 50]),
    lambda p: p["own_chain_pass_lemma"].update(base_profile_sha256="0" * 64),
    lambda p: p.update(recipe_sha256="0" * 64), lambda p: p["rules"].update(draw_count=2),
    lambda p: p.update(math_members={})])
def test_new_pass_scope_cannot_reuse_old_profile_or_silently_expand_its_whitelist(mutation):
    c, profile, binding = fixture()
    mutation(profile)
    c["entropy_profile_sha256"] = sha256(profile)
    with pytest.raises(ValueError):
        verify_negative_certificate(c, **binding, entropy_profile=profile)


@pytest.mark.parametrize("mutation", [lambda w: w.native_trace.insert(0, "3200"),
    lambda w: w.native_trace.insert(0, "2d"), lambda w: setattr(w, "branches", 1),
    lambda w: setattr(w, "choices", [object()]), lambda w: setattr(w, "submitted_total", 2),
    lambda w: w.submitted_trace[0].update(response_hex="00000000"),
    lambda w: w.submitted_trace[0].update(received_cursor=99)])
def test_uncovered_producer_scope_returns_unknown_without_asking_runtime_to_certify_it(monkeypatch, mutation):
    c, profile, _ = fixture()
    h = SimpleNamespace(record=lambda: c["history_record"])
    plan = SimpleNamespace(births=(), root_sha256=c["hypothesis_sha256"], history_sha256=c["history_sha256"],
                           record=lambda: c["plan_record"])
    witness = SimpleNamespace(answered=1, submitted_total=1, branches=0, choices=[], script_errors=[],
        last_counterexample=c["mismatch"], native_trace=c["native_prefix"], submitted_trace=c["submitted_trace"],
        expected=[bytes.fromhex(raw) for raw in c["received_prefix"]])
    mutation(witness)
    monkeypatch.setattr(E, "runtime_scope", lambda *_: pytest.fail("unmodeled trace reached runtime certification"))
    assert E.certify_own_pass_counterexample(None, h, c["layout"], plan, witness,
                                           entropy_profile=profile, source_sha256=c["source_sha256"]) is None


def test_new_profile_still_binds_full_reviewed_runtime_and_original_base_profile():
    c, profile, _ = fixture()
    verify_entropy_profile(profile, core_sha256=c["actual_core_sha256"],
                           scripts_sha256=c["actual_scripts_sha256"], core_commit=c["core_commit"])
    with pytest.raises(ValueError, match="unchanged"):
        E.own_pass_profile(profile)  # no repeated or transitive silent widening
