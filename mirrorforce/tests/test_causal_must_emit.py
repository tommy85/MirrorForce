"""Standalone pure MaxC lemma: cuts are not waved away by a failed witness."""
import copy
import json
from pathlib import Path
import struct
import time

import pytest

from mirrorforce.netduel import causal_must_emit as M
from mirrorforce.netduel.causal_history import PublicPositionHistory
from mirrorforce.netduel.causal_proof import sha256
from mirrorforce.netduel.causal_rejection import verify_negative_certificate, verify_entropy_profile

FIXTURE = Path(__file__).parent / "fixtures/maxxc-must-emit-public.json"


@pytest.fixture
def case():
    data = json.loads(FIXTURE.read_text())
    profile = M.prototype_profile(data.pop("base_profile"))
    base = profile["base"]
    assets = {"core_sha256": base["core_library_sha256s"][0], "scripts_sha256": base["scripts_sha256"],
              "lua_sha256": base["lua_sha256"], "database_sha256": M.DATABASE_SHA256,
              "core_commit": base["core_commit"]}
    return {key: data[key] for key in ("history_record", "layout", "received_prefix", "public_recipe")} | {
        "profile": profile, "actual_assets": assets, "actual_setup": copy.deepcopy(M.STANDARD_SETUP),
        "source_sha256": "a" * 64}


def verify(c, case):
    return M.verify_certificate(c, profile=case["profile"], actual_assets=case["actual_assets"],
        actual_setup=case["actual_setup"],
        hypothesis_sha256=sha256(case["layout"]), history_sha256=sha256(case["history_record"]),
        source_sha256=case["source_sha256"])


def rebuild(case):
    """Test-only public compiler: alter evidence, not a trusted plan color."""
    recipe = case["public_recipe"]
    h = PublicPositionHistory(0, recipe["main"], recipe["extra"],
                              recipe["main"], recipe["extra"], card_types={})
    responses = dict(case["history_record"]["own_responses"])
    for i, raw in enumerate(case["received_prefix"]):
        h.feed(bytes.fromhex(raw))
        if h.pending is not None and i + 1 < len(case["received_prefix"]):
            h.respond(bytes.fromhex(responses[i]))
    record = json.loads(json.dumps(h.record()))
    record["catalog_sha256"] = M.CATALOG_SHA256  # test source catalogue; no original object mutation
    case["history_record"] = record
    return h


def test_real_derived_sixteenth_public_graph_later_own_shuffle_gets_specialized_certificate(case):
    before = copy.deepcopy(case)
    c = M.certify(**case)
    assert c is not None and verify(c, case)["whole_candidate_impossible"]
    assert case == before
    assert len(c["history_record"]["permutations"]) == 1
    assert len(c["history_record"]["tokens"]) == 144 and len(c["history_record"]["root_slots"]) == 110
    assert c["window"]["handler"] == {"kind": "unchanged-root-token", "token": 93,
                                      "code": M.MAXXC, "coordinate": [1, 2, 2]}
    assert c["window"]["received_packet_index"] == 21
    assert "plan_record" not in c and "opening_unique" not in c


def test_old_entropy_and_negative_verifiers_do_not_accept_or_broaden_for_new_profile(case):
    c = M.certify(**case)
    with pytest.raises(ValueError):
        verify_entropy_profile(case["profile"], core_sha256=case["actual_assets"]["core_sha256"],
                               scripts_sha256=case["actual_assets"]["scripts_sha256"])
    with pytest.raises(ValueError):
        verify_negative_certificate(c, hypothesis_sha256=c["hypothesis_sha256"],
            history_sha256=c["history_sha256"], source_sha256=c["source_sha256"],
            entropy_profile=case["profile"]["base"])


@pytest.mark.parametrize("field", ["core_sha256", "scripts_sha256", "lua_sha256", "database_sha256", "core_commit"])
def test_actual_asset_drift_is_unknown_not_a_new_ruleset(case, field):
    case["actual_assets"][field] = "0" * len(case["actual_assets"][field])
    assert M.certify(**case) is None


@pytest.mark.parametrize("kind", ["simple_ai", "custom_start", "custom_puzzle", "forced_pending",
                                  "forced_linked", "forced_random", "missing_rules", "boolean_counts"])
def test_same_assets_and_default_profile_do_not_substitute_for_actual_runtime_setup(case, kind):
    assets, profile = copy.deepcopy(case["actual_assets"]), copy.deepcopy(case["profile"])
    setup = case["actual_setup"]
    if kind == "simple_ai":
        setup["rules"]["duel_options"] |= 0x40  # DUEL_SIMPLE_AI, not reflected by asset hashes
    elif kind == "custom_start":
        setup["rules"]["start_hand"] = 6
    elif kind == "custom_puzzle":
        setup["construction"] = "custom-puzzle/v1"
    elif kind == "forced_pending":
        setup["forced_activation_state"][0] = 1
    elif kind == "forced_linked":
        setup["forced_activation_state"][1] = 1
    elif kind == "forced_random":
        setup["forced_random_state"][0] = 1
    elif kind == "missing_rules":
        del setup["rules"]
    else:
        setup["forced_activation_state"] = [False, False]
    assert case["actual_assets"] == assets and case["profile"] == profile
    assert M.certify(**case) is None


@pytest.mark.parametrize("kind", ["card", "helper", "core", "recipe", "simple_ai", "lp", "init_forbid", "init_cost"])
def test_mutated_initialization_closure_cannot_be_approved_by_caller_flags(case, kind):
    p = case["profile"]
    if kind in ("card", "init_forbid", "init_cost"):
        p["initialization"]["card_scripts"]["c23434538.lua"] = "0" * 64
        p["initialization"]["claimed_zero_forbids"] = True
        p["initialization"]["claimed_cost_payable"] = True
    elif kind == "helper":
        p["initialization"]["helpers"]["procedure.lua"] = "0" * 64
    elif kind == "core":
        p["base"]["core_source_files"]["effect.cpp"] = "0" * 64
    elif kind == "recipe":
        p["base"]["recipe_sha256"] = "0" * 64
    elif kind == "simple_ai":
        p["base"]["rules"]["duel_options"] |= 0x100
    elif kind == "lp":
        p["base"]["rules"]["start_lp"] = 4000
    assert M.certify(**case) is None


@pytest.mark.parametrize("kind", ["own_action", "wrong_phase", "extra_waiting", "early_chance",
                                  "spent_count", "forced_chain", "custom_start", "different_viewer"])
def test_nonfresh_nonpass_or_uncovered_window_remains_unknown(case, kind):
    packets = case["received_prefix"]
    if kind == "own_action":
        case["history_record"]["own_responses"][0][1] = "00000000"
    elif kind == "wrong_phase":
        packets[11] = "290400"
    elif kind == "extra_waiting":
        packets[19] = "03"
    elif kind == "early_chance":
        packets[19] = "82000101"
    elif kind == "spent_count":
        # An earlier activation event is not silently treated as fresh count=1.
        packets[19] = "4600000000000000000000000000000000"
    elif kind == "forced_chain":
        raw = bytearray.fromhex(packets[20])
        raw[13] = 1
        packets[20] = raw.hex()
    elif kind == "custom_start":
        packets[0] = packets[0].replace("401f0000", "a00f0000")
    else:
        case["history_record"]["viewer"] = 1
    assert M.certify(**case) is None


def test_public_fact_can_bind_old_initial_hand_token_after_a_later_anonymous_hand_cut(case):
    # Reveal initial token93 before the later cut; no chosen solver colors are inputs.
    packets = case["received_prefix"]
    last = len(packets) - 1
    case["history_record"]["own_responses"].append([last, "ffffffff"])
    confirm = bytes([31, 0, 0, 1]) + struct.pack("<I", M.MAXXC) + bytes([1, 2, 2])
    packets += [confirm.hex(), (bytes([33, 1, 5]) + bytes(20)).hex(), packets[-1]]
    rebuild(case)
    c = M.certify(**case)
    assert c is not None and verify(c, case)["whole_candidate_impossible"]
    assert c["window"]["handler"]["kind"] == "public-fact"


def test_opponent_hand_deck_cut_admits_no_initial_maxc_completion_so_chosen_root_maxc_is_not_universal(case):
    packets = case["received_prefix"]
    last = len(packets) - 1
    case["history_record"]["own_responses"].append([last, "ffffffff"])
    pending = packets[-1]
    move = bytes([50]) + bytes(4) + bytes([1, 2, 0, 10, 1, 1, 0, 8]) + bytes(4)
    packets += [move.hex()] * 5 + ["2001", (bytes([90, 1, 5]) + bytes(20)).hex(), pending]
    h = rebuild(case)
    assert len(h.permutations) == 2 and M.certify(**case) is None
    from mirrorforce.netduel.agent_causal_plan import _problem, _Solver
    record, _, domains, constraints = _problem(h, case["layout"])
    initial_hand = next(pool["tokens"][-5:] for pool in record["pools"] if tuple(pool["pool"]) == (1, 1))
    for token in initial_hand:
        domains[token - 1] -= {M.MAXXC}
    colors = _Solver(constraints, len(domains), 193, 100000, time.monotonic() + 10, time.monotonic).search(domains)
    assert all(colors[token - 1] != M.MAXXC for token in initial_hand)


@pytest.mark.parametrize("kind", ["root", "graph", "catalog", "window", "source", "truncated", "birth", "death"])
def test_tampered_certificate_or_full_suffix_is_not_accepted(case, kind):
    c = M.certify(**case)
    if kind == "root":
        c["layout"][0][3] += 1
    elif kind == "graph":
        c["history_record"]["root_slots"][0][3] = 93
    elif kind == "catalog":
        c["history_record"]["catalog_sha256"] = "0" * 64
    elif kind == "window":
        c["window"]["handler"]["token"] = 94
    elif kind == "source":
        c["source_sha256"] = "0" * 64
    elif kind == "truncated":
        c["received_prefix"] = c["received_prefix"][:22]
    elif kind == "birth":
        c["history_record"]["pools"].append({"pool": [-1, 145], "tokens": [145], "counts": [[52340445, 1]]})
    else:
        c["history_record"]["dead"] = [93]
    with pytest.raises(ValueError):
        verify(c, case)
