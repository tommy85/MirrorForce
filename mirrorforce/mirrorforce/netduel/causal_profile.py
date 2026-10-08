"""Pure explicit admission registration and runtime-attestation checks.

The self-inclusive producer source map is supplied by the external frozen-run
registration, never computed from the certificate that is being checked.
Existing v0--v3 profiles and their certificates retain their original law.
"""
from __future__ import annotations

import copy
import struct

from . import causal_must_emit as M
from . import causal_pass as PASS
from .causal_proof import sha256

SCHEMA = "mirrorforce_natural_admission_profile/v4"
SOURCE_LAW = "external-registered-eight-file-producer-sha256-map/v1"
RUNTIME_SCHEMA = "mirrorforce_measured_fresh_standard_hypothesis/v1"
RUNTIME_LAW = "owned-fresh-standard-build-and-before-after-native-queue-reads/v1"
SOURCE_FILES = (
    "netduel/causal_replay.py", "netduel/causal_must_emit_runtime.py",
    "netduel/causal_profile.py", "netduel/causal_must_emit.py",
    "worldmodel/engine.py", "puzzle/core.py", "netduel/wire_projection.py", "netduel/host_view.py",
)
CONSTRUCTOR_SOURCES = {
    "worldmodel/engine.py": "70c941390e799afdd0e700a6f6e56165be4629e74fa6bd366ab014e72b9bcb37",
    "puzzle/core.py": "4dbea8df736ec1f5342acc5a554acff4a8735f671a4a0caf836115a493147849",
    "netduel/wire_projection.py": "904b6749d849c1073015c0a374c24df4230c7ec1d1aa1814cfe31a3617ed62e8",
    "netduel/host_view.py": "da6daf19a2e68c35345097e8c0b1f3862a3e74c530e47efa61721d2a6c1ba094",
}
CONSTRUCTOR_SCRIPTS = {**M.INITIALIZATION["card_scripts"], **M.INITIALIZATION["helpers"],
    "constant.lua": "01c949dad2eff92dde457258086f1ec0117888e794703cc679b64d0b6f1a8a6f"}
OPAQUE_FOLLOWER_CHUNKS = ("mf-client-history-registry.lua", "mirrorforce-client-sync.lua")
RECIPE_CARD_CODES = tuple(sorted(int(name[1:-4]) for name in M.INITIALIZATION["card_scripts"]))
RELEVANT_CARD_CODES = tuple(sorted((*RECIPE_CARD_CODES, 52340445)))
# Semantic fields from Core._query_card_data over the pinned 24bc database;
# no raw ctypes padding is hashed or compared.
RELEVANT_CARD_DATA_SHA256 = "9e3ee28f06e5ccacca4a47f98e0043112e8adad59cfb01be6936c8578279bc85"


def is_bundle(profile):
    return isinstance(profile, dict) and profile.get("schema") == SCHEMA


def verify_source_map(source_map, expected_sha256):
    M._digest(expected_sha256)
    if not isinstance(source_map, dict) or set(source_map) != set(SOURCE_FILES) \
            or any(M._digest(value) != value for value in source_map.values()) \
            or any(source_map[name] != value for name, value in CONSTRUCTOR_SOURCES.items()) \
            or sha256(source_map) != expected_sha256:
        raise ValueError("producer source map differs from its external registered complete digest")
    return expected_sha256


def production_bundle(legacy_profile, *, producer_source_map):
    """Create an external registration artifact after the entire source is frozen."""
    PASS.verify_profile(legacy_profile)
    if legacy_profile.get("scope") != PASS.EARLY_SCOPE:
        raise ValueError("requires the unchanged explicit v3 legacy profile")
    base = {key: copy.deepcopy(value) for key, value in legacy_profile.items() if key != "own_chain_pass_lemma"}
    base.update(scope="pinned-stage-pre-first-response/v1", qualification=PASS.BASE_QUALIFICATION)
    source_sha = sha256(producer_source_map)
    verify_source_map(producer_source_map, source_sha)
    return {"schema": SCHEMA, "legacy_profile": copy.deepcopy(legacy_profile),
            "must_emit_profile": M.production_profile(base), "runtime_law": RUNTIME_LAW,
            "source_binding_law": SOURCE_LAW, "producer_source_map": copy.deepcopy(producer_source_map),
            "producer_source_sha256": source_sha}


def verify_bundle(profile):
    if not is_bundle(profile):
        raise ValueError("an explicit separately registered profile bundle is required")
    try:
        expected = production_bundle(profile["legacy_profile"], producer_source_map=profile["producer_source_map"])
        if sha256(profile) != sha256(expected):
            raise ValueError("profile changed its separately scoped rules, constructor or source registration")
        return sha256(profile)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("malformed explicit profile bundle") from exc


def legacy_profile(profile):
    if is_bundle(profile):
        verify_bundle(profile)
        return profile["legacy_profile"]
    return profile


def verify_script_reads(events, *, expected=CONSTRUCTOR_SCRIPTS):
    """Actual callback-cache get/write events, not a declaration of unused names."""
    if not isinstance(events, list) or not events:
        raise ValueError("lacks actual fresh-native script read evidence")
    read = set()
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("path"), str) \
                or not event["path"].startswith("./script/"):
            raise ValueError("native script read evidence has a noncanonical path")
        name = event["path"][len("./script/"):]
        if name not in expected or name in OPAQUE_FOLLOWER_CHUNKS:
            raise ValueError("fresh native actually accessed a script outside its audited closure: " + event["path"])
        if event.get("operation") == "get":
            if type(event.get("hit")) is not bool or event.get("sha256") != (expected[name] if event["hit"] else None):
                raise ValueError("actual native script cache read changed a returned buffer")
            read.add(name)
        elif event.get("operation") == "set":
            if event.get("sha256") != expected[name]:
                raise ValueError("actual native script cache write changed loaded rule bytes")
        else:
            raise ValueError("native script audit contains an unsupported cache mutation")
    if not set(CONSTRUCTOR_SCRIPTS) <= read:
        raise ValueError("fresh native did not actually read every constructor card/helper script")


def verify_runtime_attestation(record, *, profile, hypothesis_sha256, history_sha256, source_sha256):
    profile_sha = verify_bundle(profile)
    verify_source_map(profile["producer_source_map"], source_sha256)
    if not isinstance(record, dict) or record.get("schema") != RUNTIME_SCHEMA or record.get("law") != RUNTIME_LAW \
            or record.get("registered_profile_sha256") != profile_sha \
            or record.get("source_binding_law") != SOURCE_LAW \
            or record.get("source_sha256") != source_sha256 \
            or record.get("source_map") != profile["producer_source_map"] \
            or record.get("hypothesis_sha256") != hypothesis_sha256 \
            or record.get("history_sha256") != history_sha256:
        raise ValueError("runtime attestation does not bind this externally registered source/root/history")
    verify_source_map(record["source_map"], source_sha256)
    assets = record.get("actual_assets")
    boundaries = record.get("boundaries")
    if not isinstance(boundaries, list) or len(boundaries) != 2:
        raise ValueError("runtime attestation lacks both actual fresh-build and counterexample boundaries")
    for boundary, phase in zip(boundaries, ("after-fresh-build-before-process", "idle-native-counterexample")):
        if not isinstance(boundary, dict) or boundary.get("phase") != phase \
                or boundary.get("queue_query_status") != [0, 0] \
                or any(type(x) is not int for x in boundary["queue_query_status"]) \
                or boundary.get("constructor_scripts") != CONSTRUCTOR_SCRIPTS \
                or boundary.get("resolver_law") != "actual-resolved-files-and-loaded-cache-match-audited-tree/v1":
            raise ValueError("runtime boundary lacks measured queues and actual constructor script/cache identities")
        M.verify_profile(profile["must_emit_profile"], actual_assets=assets, actual_setup=boundary.get("actual_setup"))
        cached_codes = boundary.get("cached_relevant_card_codes")
        if boundary.get("relevant_card_data_sha256") != RELEVANT_CARD_DATA_SHA256 \
                or not isinstance(cached_codes, list) or any(type(code) is not int for code in cached_codes) \
                or cached_codes != sorted(set(cached_codes)) \
                or not set(RECIPE_CARD_CODES) <= set(cached_codes) <= set(RELEVANT_CARD_CODES):
            raise ValueError("runtime boundary lacks actual relevant DB and cached CardData field agreement")
        M._digest(boundary.get("loaded_script_cache_sha256"))
        opaque = boundary.get("unread_follower_chunks")
        if not isinstance(opaque, dict) or not set(opaque) <= set(OPAQUE_FOLLOWER_CHUNKS):
            raise ValueError("runtime boundary invented unreviewed opaque follower cache chunks")
        for digest in opaque.values():
            M._digest(digest)
        verify_script_reads(boundary.get("script_read_events"))
    if boundaries[0]["unread_follower_chunks"] != boundaries[1]["unread_follower_chunks"] \
            or boundaries[0]["script_read_events"] != boundaries[1]["script_read_events"][:len(boundaries[0]["script_read_events"])]:
        raise ValueError("fresh witness changed opaque follower buffers or its actual script-read history")
    construction = record.get("construction")
    if not isinstance(construction, dict) or construction.get("driver") != "exact-DuelDriver" \
            or construction.get("core_proxy") != "exact-InitialRefreshCore" \
            or construction.get("fresh_handle") is not True or construction.get("distinct_from_follower") is not True \
            or sha256(construction.get("rules")) != sha256(M.RULES) \
            or sha256(construction.get("follower_rules")) != sha256(M.RULES) \
            or construction.get("public_recipe_sha256s") != [profile["legacy_profile"]["recipe_sha256"]] * 2 \
            or construction.get("config_sha256") != sha256(record.get("actual_config")):
        raise ValueError("runtime attestation did not use its exact fresh standard constructor and actual rules")
    config = record["actual_config"]
    if not isinstance(config, dict) or any(type(config.get(key)) is not int or config[key] != value for key, value in M.RULES.items()):
        raise ValueError("runtime certificate substituted default rules for its measured construction config")
    from .agent_public_recipe import declare
    decks, orders = config.get("decks"), config.get("forced_deck_orders")
    if not isinstance(decks, (list, tuple)) or len(decks) != 2 or not isinstance(orders, (list, tuple)) \
            or len(orders) != 2 or config.get("forced_core_seeds") is not None:
        raise ValueError("runtime construction does not bind both actual complete hypothetical decks")
    for deck, order in zip(decks, orders):
        if not isinstance(deck, dict) or declare(deck.get("main"), deck.get("extra"))["sha256"] \
                != profile["legacy_profile"]["recipe_sha256"] or sha256(order) != sha256(deck["main"]):
            raise ValueError("runtime construction changed its complete recipe or actual deck order")
    return {"record_consistent": True, "runtime_rerun": False}


def verify_native_support(certificate):
    """Check the real failed witness record, separately from the universal lemma."""
    native = [PASS._hex(raw) for raw in certificate.get("native_prefix", ())]
    own = certificate.get("submitted_trace")
    mismatch, window = certificate.get("mismatch"), certificate.get("window")
    if not native or any(raw[0] not in (2, 16, 40, 41, 90) for raw in native) \
            or [raw for raw in native if raw[0] == 40] != [bytes([40, 0])] \
            or [raw for raw in native if raw[0] == 41] != [bytes([41, 1, 0])] \
            or certificate.get("script_errors") != [] \
            or not isinstance(own, list) or len(own) != 1 or not isinstance(mismatch, dict) or not isinstance(window, dict):
        raise ValueError("MaxC certificate lacks its real first-DRAW native counterexample")
    prompts = [i for i, raw in enumerate(native) if raw[0] == 16]
    if len(prompts) != 2 or prompts[-1] != len(native) - 1:
        raise ValueError("native support includes an extra historical prompt")
    draws = [raw for raw in native if raw[0] == 90]
    if len(draws) != 2 or [raw[1:3] for raw in draws] != [bytes([0, 5]), bytes([1, 5])] \
            or any(len(raw) != 23 for raw in draws):
        raise ValueError("native support is not the actual initial two five-card draws")
    PASS._chain(native[prompts[0]], player=0, passing=True)
    PASS._chain(native[-1], player=1, passing=False)
    if not any(struct.unpack_from("<I", native[-1], 14 + 14*i)[0] == M.MAXXC
               and native[-1][18 + 14*i:20 + 14*i] == bytes([1, 2])
               for i in range(native[-1][2])):
        raise ValueError("required real opponent HAND MaxC in the emitted CHAIN menu")
    event = own[0]
    received = certificate["received_prefix"]
    if any(type(mismatch.get(key)) is not int for key in
            ("packet_index", "own_answered", "native_submitted_responses", "branches", "choices")):
        raise ValueError("native counterexample counters must be exact integers")
    if event != {"native_index": prompts[0], "player": 0, "prompt_hex": native[prompts[0]].hex(),
                 "received_cursor": window["own_pass_packet_index"] + 1, "response_hex": "ffffffff"} \
            or received[window["own_pass_packet_index"]] != native[prompts[0]].hex() \
            or mismatch != {"packet_index": window["received_packet_index"], "received_packet_hex": "290200",
                "synthetic_packet_hex": "03", "native_message_hex": native[-1].hex(),
                "own_answered": 1, "native_submitted_responses": 1, "branches": 0, "choices": 0}:
        raise ValueError("native response or first wire mismatch differs from the universal window")
