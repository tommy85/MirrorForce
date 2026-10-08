"""Pure admission/negative-certificate record checks; no native or file I/O."""
from __future__ import annotations

import math
from collections import Counter

from .causal_proof import sha256, verify_natural_root_proof
from . import causal_pass as PASS

LAW = "public-capacity-uniform-certified-natural-prefix/v1"
BASE_LAW = "public-world-and-position-capacity-uniform/v1"
ADMISSION_SCHEMA = "mirrorforce_conditioned_causal_admission/v1"
SEQUENCE_LAW = "independent-sha256-counter-capacity-proposal-prefix/v1"
PREDICATE = "complete-natural-prefix-witness-or-certified-whole-root-impossibility/v1"
NEGATIVE_SCHEMA = "mirrorforce_certified_natural_infeasible/v1"
NEGATIVE_LAW = "unique-opening-no-response-deterministic-prefix/v1"
ENTROPY_SCHEMA = "mirrorforce_stage_entropy_audit/v1"


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("conditional admission requires canonical SHA-256 bindings")
    return value


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("conditional admission requires finite nonnegative counters/times")
    return value


def _proposal_layout(row, index):
    if not isinstance(row, dict) or row.get("ordinal") != index or type(row.get("ordinal")) is not int \
            or not isinstance(row.get("layout"), list) or sha256(row["layout"]) != row.get("layout_sha256"):
        raise ValueError("conditional admission changed or renumbered a proposal")
    layout = row["layout"]
    if any(not isinstance(r, list) or len(r) != 4 or any(type(v) is not int for v in r) for r in layout) \
            or layout != sorted(layout) or len({tuple(r[:3]) for r in layout}) != len(layout):
        raise ValueError("conditional proposal layout is not canonical and complete by coordinate")
    if not layout or any(p not in (0, 1) or z not in (1, 2, 4, 8, 16, 32, 64)
            or not 0 <= s < (7 if z == 4 else 8 if z == 8 else 255)
            or not 0 < code < 0x80000000 or 999000001 <= code <= 999000004 for p, z, s, code in layout):
        raise ValueError("conditional proposal contains invalid coordinates or placeholder identities")
    return layout


def verify_entropy_profile(profile, *, core_sha256, scripts_sha256, core_commit=None):
    """Check a separately reviewed static audit against actual verify_rules identities.

    This function does not scan scripts and cannot manufacture an entropy
    audit. The caller supplies the actual core/script identities, not values
    copied from the profile being checked.
    """
    from . import causal_profile as profile_bundle
    if profile_bundle.is_bundle(profile):
        profile_bundle.verify_bundle(profile)
        verify_entropy_profile(profile["legacy_profile"], core_sha256=core_sha256,
                               scripts_sha256=scripts_sha256, core_commit=core_commit)
        return sha256(profile)
    _hash(core_sha256)
    _hash(scripts_sha256)
    if not isinstance(profile, dict) or profile.get("schema") != ENTROPY_SCHEMA \
            or profile.get("scope") not in ("pinned-stage-pre-first-response/v1", *PASS.SCOPES) \
            or profile.get("result") != "passed_static_entropy_audit" \
            or profile.get("script_digest_law") != "verify_rules-stage-relative-file-map-json/v1" \
            or profile.get("stage_script_prefix") != "project/script" \
            or profile.get("script_file_manifest_sha256") != "4729224b778e70cd71393ee4a22e4b85e855d5893e8691c9ba16a44b13190810" \
            or core_sha256 not in profile.get("core_library_sha256s", []) \
            or profile.get("scripts_sha256") != scripts_sha256 \
            or (core_commit is not None and profile.get("core_commit") != core_commit) \
            or profile.get("uncovered_native_rng") != [] or profile.get("script_entropy_findings") != [] \
            or profile.get("unclassified_global_access") != [] or profile.get("unclassified_math_access") != [] \
            or profile.get("unclassified_iteration_sites") != [] or profile.get("core_source_archive_verified") is not True \
            or profile.get("lua_sha256") != "99d643c45f9f501eb4d6d89e17b698b94fec0de11e459b9947fdfbe8902ddf52" \
            or profile.get("native_chance_messages") != [32, 33, 36, 39, 81, 130, 131]:
        raise ValueError("unreviewed or mismatched deterministic-prefix entropy profile")
    if type(profile.get("lua_files")) is not int or profile["lua_files"] < 1 \
            or not isinstance(profile.get("core_source_files"), dict) or not profile["core_source_files"]:
        raise ValueError("entropy audit lacks complete native/script source evidence")
    for digest in profile["core_source_files"].values():
        _hash(digest)
    if profile["scope"] in PASS.SCOPES:
        PASS.verify_profile(profile)
    return sha256(profile)


def verify_negative_certificate(certificate, *, hypothesis_sha256, history_sha256, source_sha256, entropy_profile):
    try:
        return _verify_negative_certificate(certificate, hypothesis_sha256=hypothesis_sha256,
            history_sha256=history_sha256, source_sha256=source_sha256, entropy_profile=entropy_profile)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError("malformed negative certificate record") from exc


def _verify_negative_certificate(certificate, *, hypothesis_sha256, history_sha256, source_sha256, entropy_profile):
    from . import causal_profile as profile_bundle
    if profile_bundle.is_bundle(entropy_profile):
        return _verify_certificate(certificate, hypothesis_sha256=hypothesis_sha256,
            history_sha256=history_sha256, source_sha256=source_sha256, profile=entropy_profile)
    for value in (hypothesis_sha256, history_sha256, source_sha256):
        _hash(value)
    c = certificate
    if not isinstance(c, dict) or c.get("schema") != NEGATIVE_SCHEMA or c.get("law") not in (NEGATIVE_LAW, *PASS.LAWS) \
            or c.get("whole_candidate_impossible") is not True \
            or c.get("script_errors") != [] \
            or (c.get("hypothesis_sha256"), c.get("history_sha256"), c.get("source_sha256")) \
                != (hypothesis_sha256, history_sha256, source_sha256) \
            or c.get("entropy_profile_sha256") != sha256(entropy_profile):
        raise ValueError("negative record is not a scoped whole-candidate certificate")
    verify_entropy_profile(entropy_profile, core_sha256=c.get("actual_core_sha256"),
                           scripts_sha256=c.get("actual_scripts_sha256"), core_commit=c.get("core_commit"))
    if c.get("actual_lua_sha256") != entropy_profile.get("lua_sha256"):
        raise ValueError("negative certificate has not bound the audited actual Lua runtime")
    pass_lemma = c["law"] in PASS.LAWS
    for key in ("cuts", "births", "dead", "branches", "choices"):
        if type(c.get(key)) is not int or c[key] != 0:
            raise ValueError("negative certificate contains an unresolved historical or response branch")
    response_count = c.get("native_submitted_responses")
    if type(response_count) is not int or not (1 <= response_count <= 3 if pass_lemma else response_count == 0) \
            or type(c.get("own_answered")) is not int or c["own_answered"] != response_count:
        raise ValueError("negative certificate contains a response outside its explicitly registered lemma")
    if type(c.get("tokens")) is not int or c["tokens"] < 1 or type(c.get("root_tokens")) is not int \
            or c.get("root_tokens") != c["tokens"] \
            or c.get("opening_unique") is not True:
        raise ValueError("negative certificate has not proved a unique complete opening")
    if sha256(c.get("layout")) != hypothesis_sha256:
        raise ValueError("negative certificate changed the proposed complete layout")
    history, plan = c.get("history_record"), c.get("plan_record")
    if not isinstance(history, dict) or not isinstance(plan, dict) or sha256(history) != history_sha256 \
            or history.get("schema") != "mirrorforce_public_position_history/v1" \
            or history.get("law") != "public-position-tokens-and-anonymous-permutation-cuts/v1" \
            or history.get("permutations") != [] or history.get("dead") != [] \
            or plan.get("schema") != "mirrorforce_causal_opening_plan/v1" \
            or plan.get("law") != "bounded-public-capacity-witness; unchanged-root-proposal/v1" \
            or plan.get("history_sha256") != history_sha256 or plan.get("root_sha256") != hypothesis_sha256 \
            or plan.get("cuts") not in ([], ()) or plan.get("births") not in ([], ()) \
            or plan.get("native_replay_admitted") is not False or plan.get("own_deck_applied") is not False:
        raise ValueError("negative certificate lacks the original no-cut/no-birth public graph and plan")
    tokens, pools, root_slots = history.get("tokens"), history.get("pools"), history.get("root_slots")
    colors = plan.get("token_codes")
    if not all(isinstance(value, (list, tuple)) for value in (tokens, pools, root_slots, colors)) \
            or len(tokens) != c["tokens"] or len(colors) != len(tokens) \
            or {t.get("token") for t in tokens} != set(range(1, len(tokens) + 1)) \
            or any(t.get("created_at") != -1 for t in tokens) \
            or any(type(code) is not int or not 0 < code < 0x80000000 or 999000001 <= code <= 999000004 for code in colors):
        raise ValueError("negative certificate does not identify only original opening tokens")
    opening = plan.get("openings")
    if not isinstance(opening, (list, tuple)) or len(opening) != 4 or len(pools) != 4:
        raise ValueError("negative certificate must bind all four initial recipe pools")
    ids, zones = [], set()
    for pool, row in zip(pools, opening):
        if not isinstance(pool, dict) or not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError("malformed initial token pool in negative certificate")
        p, z, members, codes = row
        if type(p) is not int or p not in (0, 1) or type(z) is not int or z not in (1, 64) \
                or (p, z) in zones or list(pool.get("pool", ())) != [p, z] \
                or list(pool.get("tokens", ())) != list(members) or len(members) != len(codes) \
                or any(type(t) is not int or not 1 <= t <= len(tokens) for t in members) \
                or list(codes) != [colors[t - 1] for t in members] \
                or Counter(codes) != Counter(dict(pool.get("counts", ()))):
            raise ValueError("negative certificate changed an opening recipe or its token assignment")
        zones.add((p, z))
        ids.extend(members)
    if sorted(ids) != list(range(1, len(tokens) + 1)) or len(root_slots) != len(tokens) \
            or any(not isinstance(row, (list, tuple)) or len(row) != 4 for row in root_slots) \
            or sorted(row[3] for row in root_slots) != sorted(ids):
        raise ValueError("some opening identity is latent, duplicated, dead or missing at the proposed root")
    expected_root = [[*row, colors[row[3] - 1]] for row in root_slots]
    expected_layout = [[row[0], row[1], row[2], row[4]] for row in expected_root]
    if [list(row) for row in plan.get("root", ())] != expected_root or expected_layout != c["layout"] \
            or any(colors[fact["token"] - 1] != fact["code"] for fact in history.get("facts", ())):
        raise ValueError("the complete root does not uniquely fix every initial identity and public fact")
    messages = c.get("native_prefix")
    if not isinstance(messages, list) or not messages:
        raise ValueError("negative certificate lacks the full pre-response native prefix")
    for raw in messages:
        if not isinstance(raw, str) or len(raw) < 2:
            raise ValueError("negative native prefix must use exact hex messages")
        data = bytes.fromhex(raw)
        if data.hex() != raw or data[0] in entropy_profile["native_chance_messages"]:
            raise ValueError("a random-capable prefix is not a deterministic impossibility proof")
    mismatch = c.get("mismatch")
    if not isinstance(mismatch, dict) or mismatch.get("native_message_hex") != messages[-1] \
            or type(mismatch.get("packet_index")) is not int or mismatch["packet_index"] < 0 \
            or not isinstance(mismatch.get("received_packet_hex"), str) \
            or not isinstance(mismatch.get("synthetic_packet_hex"), str) \
            or mismatch["received_packet_hex"] == mismatch["synthetic_packet_hex"]:
        raise ValueError("negative certificate does not end at an actual deterministic wire contradiction")
    for key in ("received_packet_hex", "synthetic_packet_hex"):
        raw = mismatch[key]
        if not raw or bytes.fromhex(raw).hex() != raw:
            raise ValueError("wire contradiction must contain exact canonical nonempty packet bytes")
    for key in ("own_answered", "native_submitted_responses", "branches", "choices"):
        expected = response_count if key in ("own_answered", "native_submitted_responses") else 0
        if type(mismatch.get(key)) is not int or mismatch[key] != expected:
            raise ValueError("the wire contradiction followed an unresolved choice or submitted response")
    if pass_lemma:
        PASS.verify_trace(c, entropy_profile)
    return {"record_consistent": True, "native_replay_rerun": False, "whole_candidate_impossible": True}


def _verify_certificate(certificate, *, hypothesis_sha256, history_sha256, source_sha256, profile):
    from . import causal_profile as profile_bundle
    from . import causal_must_emit as M
    for value in (hypothesis_sha256, history_sha256, source_sha256):
        _hash(value)
    profile_bundle.verify_bundle(profile)
    if not isinstance(certificate, dict) or certificate.get("registered_profile_sha256") != sha256(profile) \
            or source_sha256 != profile["producer_source_sha256"]:
        raise ValueError("negative certificate differs from the external registered profile/source")
    attestation = certificate.get("runtime_attestation")
    profile_bundle.verify_runtime_attestation(attestation, profile=profile, hypothesis_sha256=hypothesis_sha256,
        history_sha256=history_sha256, source_sha256=source_sha256)
    if certificate.get("law") == M.LAW:
        result = M.verify_certificate(certificate, profile=profile["must_emit_profile"],
            actual_assets=attestation["actual_assets"], actual_setup=attestation["boundaries"][-1]["actual_setup"],
            hypothesis_sha256=hypothesis_sha256, history_sha256=history_sha256, source_sha256=source_sha256)
        profile_bundle.verify_native_support(certificate)
        return result
    # An explicit registration may reuse the unchanged legacy theorem,
    # but not loosen the legacy checker or replace its embedded profile SHA.
    return verify_negative_certificate(certificate, hypothesis_sha256=hypothesis_sha256,
        history_sha256=history_sha256, source_sha256=source_sha256, entropy_profile=profile["legacy_profile"])


def check_conditioned_admission(audit, *, requested_count, proposal_seed, world_sha256, obs_sha256,
                                history_sha256, source_sha256, entropy_profile_sha256, require_complete=True):
    """Return accepted layout SHA order, after checking the full original prefix."""
    if not isinstance(audit, dict) or (audit.get("schema"), audit.get("law"), audit.get("base_law"),
            audit.get("sequence_law"), audit.get("acceptance_predicate")) != (
                ADMISSION_SCHEMA, LAW, BASE_LAW, SEQUENCE_LAW, PREDICATE) \
            or audit.get("requested_count") != requested_count or audit.get("proposal_seed") != proposal_seed \
            or audit.get("proposal_limit") != 128 or type(requested_count) is not int or not 1 <= requested_count <= 128 \
            or type(proposal_seed) is not int:
        raise ValueError("conditional admission differs from its fixed registered proposal law")
    for key, expected in (("world_sha256", world_sha256), ("obs_sha256", obs_sha256),
                          ("history_sha256", history_sha256), ("source_sha256", source_sha256)):
        if expected is not None:
            _hash(expected)
            if audit.get(key) != expected:
                raise ValueError("conditional admission bank binding differs: " + key)
    entropy = audit.get("entropy_profile")
    if (sha256(entropy) if entropy else None) != entropy_profile_sha256:
        raise ValueError("conditional admission entropy profile differs from registration")
    from . import causal_profile as profile_bundle
    if profile_bundle.is_bundle(entropy):
        profile_bundle.verify_bundle(entropy)
        if audit.get("source_sha256") != entropy["producer_source_sha256"]:
            raise ValueError("admission did not preserve the external pre-registered producer source digest")
    for key in ("proposal_seconds", "admission_seconds", "total_seconds", "rejection_rate"):
        _number(audit.get(key))
    if not math.isclose(audit["total_seconds"], audit["proposal_seconds"] + audit["admission_seconds"],
                        rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("conditional admission omits time from its shared budget")
    if audit.get("status") == "proposal_failed":
        # Construction failure can preserve fewer than 128 draws, but it can
        # never qualify a search bank. Do not invent missing original draws.
        rows = audit.get("proposals")
        if require_complete or not isinstance(rows, list) or len(rows) > 128 \
                or type(audit.get("generated_proposals")) is not int or audit["generated_proposals"] != len(rows) \
                or audit.get("requested_proposals") != 128 or audit.get("attempted") != 0 \
                or audit.get("accepted_ordinals") != [] or audit.get("rejected") != 0 or audit.get("unknown") != 0 \
                or audit.get("cap_reached") is not False or audit["rejection_rate"] != 0 \
                or audit["admission_seconds"] != 0 or not isinstance(audit.get("failure"), str):
            raise ValueError("failed proposal construction is not a complete admission bank")
        layouts = [_proposal_layout(row, index) for index, row in enumerate(rows)]
        if audit.get("proposal_sequence_sha256") != sha256(layouts) \
                or any(row.get("status") != "not_attempted" or row.get("proof") is not None
                       or row.get("seconds") != 0 for row in rows):
            raise ValueError("failed construction changed or attempted a partial original prefix")
        return ()
    sampler = audit.get("sampler")
    if not isinstance(sampler, dict) or sampler.get("schema") != "mirrorforce_causal_root_sampler/v1" \
            or sampler.get("law") != BASE_LAW or sampler.get("history_sha256") != history_sha256 \
            or sampler.get("world_sha256") != world_sha256 or type(sampler.get("distinct_root_layouts")) is not int \
            or sampler["distinct_root_layouts"] <= 0 \
            or sampler.get("sampling") != "integer-multinomial; sha256-counter-unbiased/v1" \
            or sampler.get("proposal_retries") != 0 or type(sampler.get("proposal_retries")) is not int \
            or sampler.get("native_replay_admitted") is not False \
            or sampler.get("historical_witness_multiplicity_counted") is not False:
        raise ValueError("conditional admission lacks its exact distinct-root capacity prior")
    rows = audit.get("proposals")
    if not isinstance(rows, list) or len(rows) != 128:
        raise ValueError("conditional admission must preserve its entire original proposal prefix")
    layouts, accepted, attempted, rejected, unknown = [], [], 0, 0, 0
    stopped = False
    for index, row in enumerate(rows):
        layout = _proposal_layout(row, index)
        layouts.append(layout)
        _number(row.get("seconds"))
        status = row.get("status")
        if status == "not_attempted":
            stopped = True
            if row.get("proof") is not None or row["seconds"] != 0:
                raise ValueError("an unattempted proposal has fabricated evidence or elapsed work")
            continue
        if stopped:
            raise ValueError("conditional admission skipped an earlier registered proposal")
        attempted += 1
        if status == "accepted":
            verify_natural_root_proof(row.get("proof"), hypothesis_sha256=row["layout_sha256"],
                                      history_sha256=history_sha256, source_sha256=source_sha256)
            accepted.append(index)
        elif status == "certified_infeasible":
            verify_negative_certificate(row.get("proof"), hypothesis_sha256=row["layout_sha256"],
                history_sha256=history_sha256, source_sha256=source_sha256, entropy_profile=entropy)
            rejected += 1
        elif status in ("unknown", "error"):
            if require_complete:
                raise ValueError("an unknown/error proposal cannot be silently rejected or replaced")
            unknown += int(status == "unknown")
            stopped = True
        else:
            raise ValueError("conditional admission has an unfinished or unknown verdict")
    if audit.get("proposal_sequence_sha256") != sha256(layouts) or audit.get("attempted") != attempted \
            or audit.get("accepted_ordinals") != accepted or audit.get("rejected") != rejected \
            or audit.get("unknown") != unknown or audit.get("cap_reached") is not (attempted == 128) \
            or not math.isclose(audit["rejection_rate"], rejected / attempted if attempted else 0, abs_tol=1e-12):
        raise ValueError("conditional admission counters or unchanged proposal prefix differ")
    if require_complete and (audit.get("status") != "complete" or len(accepted) != requested_count \
                             or attempted != accepted[-1] + 1):
        raise ValueError("conditional bank was not completely filled before candidate action values")
    return tuple(rows[index]["layout_sha256"] for index in accepted)
