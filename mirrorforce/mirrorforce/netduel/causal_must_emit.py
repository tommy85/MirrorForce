"""Independent prototype: a Maxx C necessary-window lemma, not a v3 extension.

No caller-provided 'zero forbids' or chosen witness color establishes legality.
The fixed initialization closure and a directly forced original HAND token
establish a necessary opponent window for *every* historical completion.
A completion that differs sooner already cannot reproduce the received tape;
one matching the first own pass must emit WAITING before the next phase.

The standalone profile is not an admission opt-in. Its separate production
profile requires the bundle and the measured fresh-native runtime bridge.
This module neither executes Lua nor claims bytewise determinism of residual decks.
"""
from __future__ import annotations

import copy

from .causal_history import PublicPositionHistory
from .causal_proof import sha256
from .causal_pass import _chain, _hex
from .agent_public_recipe import checked

INFORMATION_SET_SEARCH = True
SCHEMA = "mirrorforce_maxxc_must_emit_certificate/v1"
LAW = "pinned-maxxc-necessary-first-draw-opponent-window/v1"
PROFILE_SCHEMA = "mirrorforce_maxxc_initialization_profile/v1"
PRODUCTION_PROFILE_SCHEMA = "mirrorforce_maxxc_initialization_profile/v2"
BASE_SHA256 = "d044496993aeadbfcd6d17784bfeee6ced08210ded8d0cc9696b01bc9a816902"
CATALOG_SHA256 = "a2d566efa5feb683b2d64c3adeae91537128c1aa7a47ea4180a4e2bad0bf68ec"
DATABASE_SHA256 = "24bc1c9bf75e3a3c5990976a128e51a48819f1bb7ab21495d264017ab6795afa"
MAXXC = 23434538
RULES = {"start_lp": 8000, "start_hand": 5, "draw_count": 1, "duel_options": 5 << 16}
STANDARD_SETUP = {
    "construction": "fresh-standard-complete-decks; no-puzzle-load-or-native-state-injection/v1",
    "rules": RULES,
    "forced_activation_state": [0, 0],
    "forced_random_state": [0, 0, 0],
}
INITIALIZATION = {
    "card_scripts": {
        "c2857636.lua": "0521eea1cb930ce6268685a39c496072d0fab78b4cf558b09729dfc6e887ecff",
        "c5821478.lua": "599e9a24c98ea1f70151668c47071078109f03306c4de4b3851c9ac17f84e0ac",
        "c8491308.lua": "6a2d4e81a7bec1677a09df5abefbd991c93ad3cfdc02435dd9535e7fe02198d8",
        "c14558127.lua": "0afff1fadd6be7e6301e4d72d5457eab179c3aa966e4d3a3ac909ef25a4b0667",
        "c23434538.lua": "997212b26aa01d52096a7bbc03757220a43db36a065e8f405d81dcc8c999e651",
        "c24010609.lua": "27f03e6a51e37d90390f8d70347078c8799450647acd9336311433da262ebcd2",
        "c25955749.lua": "5ce6ee72cf69421818852c9710ea4a4878e8915b4ad65bdeacbc6138c653b5ee",
        "c26077387.lua": "e3265c0490ac261176ad042df422be87105d332fc648b514a102584b5d6ad262",
        "c32807846.lua": "3d937f799a214f4cbc509efdac88121ae2f4c785785ae655fc42ef142cad84b3",
        "c35726888.lua": "abb31c76a0ba4d4ac5dd94c86e24f4915a3af6ca7009f732613936539dae098c",
        "c38342335.lua": "8f4448a406c3b2e9679e2fb5f77113ee808cd30953a5913976e71c6fd2595175",
        "c41420027.lua": "8bb8ca806b3bb31135fd9cd464d20ca81b5a8c191be934e6b326abea7f51321e",
        "c41999284.lua": "25567857971ca5d5286dd8a132bb3c9c2a0d9c0c42866e4dcf0625cfc689c95d",
        "c43898403.lua": "b7e1d9abd93bd325fcf6325319a2b6b3a98fb69ddf58b5518434c56c1e099608",
        "c50005218.lua": "1bf0a8f81edf435e26057997760867b839e1f7198bc8fe044f0a67cc7b456666",
        "c50588353.lua": "60ba2ddbb81b3c54cb1665b18dee833006a054c955ba98e73b46c72fec5419a3",
        "c51227866.lua": "91194f55481a7b2860cd5949a7ccb851cbc0bab3ea04cc3e8f59c2501ac472b9",
        "c52340444.lua": "21239cf0b7640664337149486d954194a94eb31917adec46346e24c3abe073c1",
        "c59438930.lua": "b254cacbb6e6793269e29f59322bb402be0dbaccef3cd93991777b926dfe48c7",
        "c63166095.lua": "97e5711a03d338de4f79336f62ac19e7bce33bf98842f2d7ac7a7b43fcd18c60",
        "c63288573.lua": "4fb04f94bb40d1a87280ccca3e15b40ecbe97a8633e4272366b3e9c185f705e2",
        "c67441435.lua": "2f01a34786e4be0230eb25edaf7bd0d421bf8092d6f93de3fcede850b0eccb02",
        "c70368879.lua": "88ff484344b06c33dc05707cf0a173f6a98722a5ea857b473ae3718f93eb7936",
        "c73594093.lua": "38b54f483fb7980a8f7b98fe3f157577ad7eb812320b9b71f1ce1b400e813837",
        "c75452921.lua": "5b33047897d90769b87904fcac745203339b17863b8ba2ac3cab83bfa1c748ea",
        "c84749824.lua": "b76ea470a28b93ded7a2be5e5f7bee919d585a10b691d387e6d45b6ffa1f76de",
        "c85289965.lua": "c497b8861592a5078fd21a3eb93d1bdde690fb0cfc8c5788ec355c90b00dceb5",
        "c90673288.lua": "d3d68447edec0ab9bdca394cb0ddfc4b577268e0ff38d60d813aa4a90b43c9b7",
        "c97268402.lua": "d283f6744376f17e4a71c64526dde0df7331f4625c393427f479ff725b295f4a",
        "c97616504.lua": "e9d7a6463031d084f94dc6e2aaf791e8890d2812c4c5eff325fced2ccebb3ad9",
        "c98338152.lua": "8a9e8f3846798b6bdb6eb7086476c3ddbfcca22b0b56d7e006f399fc1e20d5bb",
        "c99550630.lua": "60f4244316467ad69513fadc2a0e354f207af6fc08b603ff19f532699c1f3893"
    },
    "helpers": {
        "procedure.lua": "aae9067432e47ab1082ae27ff5dbf5a2d9688950d3af40f333d0859b3493d3b0",
        "utility.lua": "65fc51fa6e1f37581a4c04e4b7c5cd3b4c0dc0ba78119c01d30241bc41cb86fc"
    },
    "core_sites": {
        "interpreter.cpp": "register_card initial_effect, before field placement",
        "libcard.cpp": "RegisterEffect; EnableReviveLimit; SetSPSummonOnce; IsAbleToGraveAsCost",
        "libeffect.cpp": "SetCountLimit; SetType",
        "card.cpp": "apply_field_effect; enable_field_effect; add_effect; is_capable_send_to_grave_as_cost; destination_redirect",
        "effect.cpp": "check_count_limit; is_activateable; is_action_check; is_activate_ready; is_chainable",
        "field.cpp": "add_effect; is_player_can_send_to_grave",
        "processor.cpp": "process_turn; process_phase_event",
        "playerop.cpp": "select_chain: mandatory wire prompt before any opponent response",
        "ocgapi.cpp": "new_card; start_duel",
        "operations.cpp": "draw: initial five list_main.back cards"
    },
    "global_side_effect": "SetSPSummonOnce idempotent OR GLOBALFLAG_SPSUMMON_ONCE=0x200; no deck-reverse flag",
    "initial_effects": "self registration only; initial zones DECK/EXTRA; no active initial forbids, activation taxes, redirects or chain limits",
    "dry_run_closure": {
        "c43898403": "Twin cost chk0 queries IsDiscardable; target chk0 queries empty ONFIELD; no discard or registration",
        "c51227866": "Shark condition queries empty MZONE; target chk0 queries empty GRAVE; no operation",
        "c52340444": "Hornet condition/count query empty MZONE/GRAVE; target chk0 queries usable zone and token summon capability only",
        "c98338152": "Widow condition queries empty MZONE; target chk0 finds no MZONE target; no category change or operation",
        "c97268402": "Veiler condition returns false in DRAW before cost/target callbacks",
        "c23434538": "MaxC cost chk0 is IsAbleToGraveAsCost only; no cost payment or count-limit consumption",
        "other_cards": "normal spells fail speed1; HAND traps lack HAND permission; Raye range MZONE absent; Ash/Ghost event is CHAINING; no phase triggers or FREE_CHAIN continuous effects",
        "query_native": "libduel 2657/2710/3084 matching queries, libcard2541 discard capability; empty field/grave means target filters cannot register effects",
        "hornet_native": "libduel4474 reads token database metadata, field3368 assigns temporary card data then clears; no token initialization, summon or effect registration",
        "temporary_state": "effect401 resets cost_checked; effect243 saves/restores LP and reason state; field3312/2294/2385 only query summon restrictions/counters"
    },
    "main_hand": "MaxC has unconditional HAND FREE_CHAIN QuickO; fresh count=1; grave cost check succeeds",
    "helper_effects": "EnableReviveLimit is self SINGLE; LinkProcedure is EXTRA SPSUMMON_PROC; factories not operations",
    "limitation": "32 registered scripts and their initialization/legality closure only; no general Lua determinism"
}
START = bytes.fromhex("040005401f0000401f000028000f0028000f00")


def prototype_profile(base):
    """Explicit separate profile; never mutates or broadens any existing audit."""
    from .causal_rejection import verify_entropy_profile
    if sha256(base) != BASE_SHA256:
        raise ValueError("must-emit prototype requires the unchanged reviewed base audit")
    verify_entropy_profile(base, core_sha256=base["core_library_sha256s"][0],
                           scripts_sha256=base["scripts_sha256"], core_commit=base["core_commit"])
    return {"schema": PROFILE_SCHEMA, "law": LAW, "base": copy.deepcopy(base),
            "initialization": copy.deepcopy(INITIALIZATION),
            "catalog_sha256": CATALOG_SHA256, "database_sha256": DATABASE_SHA256,
            "standard_setup": copy.deepcopy(STANDARD_SETUP),
            "deployment": "standalone-prototype; no-production-admission-hook/v1"}


def production_profile(base):
    """Explicit opt-in artifact; the standalone profile remains a separate schema."""
    profile = prototype_profile(base)
    profile.update(schema=PRODUCTION_PROFILE_SCHEMA,
                   deployment="trusted-fresh-standard-constructor-and-measured-native-state/v1")
    return profile


def verify_profile(profile, *, actual_assets, actual_setup):
    """Actual setup is a mandatory independent runtime binding, not profile defaults.

    A future trusted producer must establish the actual standard construction,
    read the complete actual rules and query native forced-state queues. This
    standalone pure checker does not itself observe a running engine. It does
    not accept caller claims about absence of forbids, costs or chain limits;
    those follow from the fixed initialization and dry-run closure instead.
    """
    factory = production_profile if isinstance(profile, dict) and profile.get("schema") == PRODUCTION_PROFILE_SCHEMA \
        else prototype_profile
    if not isinstance(profile, dict) or sha256(profile) != sha256(factory(profile.get("base"))):
        raise ValueError("must-emit profile changed its fixed initialization closure")
    base = profile["base"]
    required = {"core_sha256": base["core_library_sha256s"][0],
                "scripts_sha256": base["scripts_sha256"], "lua_sha256": base["lua_sha256"],
                "database_sha256": DATABASE_SHA256, "core_commit": base["core_commit"]}
    if actual_assets != required:
        raise ValueError("must-emit asset identities differ from the fixed audited rules")
    if sha256(actual_setup) != sha256(STANDARD_SETUP):
        raise ValueError("must-emit actual construction/rules/forced-state differs from its standard opening")
    return sha256(profile)


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("must-emit bindings require canonical SHA-256")
    return value


def _history(record, received, recipe):
    """Recompile the full public tape; no caller-invented token ancestry.

    Card types are consulted by PublicPositionHistory only for token births or
    deaths. Both are conservatively unsupported here, including in the bound
    suffix. Thus an empty type map is sufficient to independently reconstruct
    every accepted graph field other than the fixed registered catalogue hash.
    """
    recipe = checked(recipe)
    if recipe["sha256"] != "7413d0f0b10070c60853ead43c121a202c0e091059bc8666914137f4d4100e70":
        raise ValueError("must-emit changed its complete registered mirror recipe")
    if not isinstance(record, dict) or record.get("viewer") != 0 or type(record.get("viewer")) is not int \
            or record.get("catalog_sha256") != CATALOG_SHA256 or record.get("dead") != []:
        raise ValueError("must-emit requires the registered player0 public graph without deaths")
    own = record.get("own_responses")
    if not isinstance(own, list):
        raise ValueError("must-emit needs the complete actual own response tape")
    answers, previous = {}, -1
    for row in own:
        if not isinstance(row, (list, tuple)) or len(row) != 2 or type(row[0]) is not int \
                or not previous < row[0] < len(received) - 1:
            raise ValueError("must-emit response cursors must be canonical and increasing")
        answers[row[0]] = _hex(row[1])
        previous = row[0]
    history = PublicPositionHistory(0, recipe["main"], recipe["extra"],
        recipe["main"], recipe["extra"], card_types={})
    used = []
    for index, raw in enumerate(received):
        history.feed(raw)
        if history.pending is not None and index + 1 < len(received):
            if index not in answers:
                raise ValueError("must-emit full public tape omits an own response")
            history.respond(answers[index])
            used.append(index)
        elif index in answers:
            raise ValueError("must-emit response does not answer a public own prompt")
    rebuilt = history.record()
    if used != list(answers) or rebuilt["pending"] is None or len(rebuilt["pools"]) != 4 \
            or rebuilt["dead"] or any(t["created_at"] != -1 for t in rebuilt["tokens"][:110]):
        raise ValueError("must-emit full history contains an unsupported birth/death or missing root")
    # Comparison normalizes only tuple/list JSON representation and the
    # separately pinned catalogue digest. It does not relax any graph field.
    without_catalog = lambda value: {key: item for key, item in value.items() if key != "catalog_sha256"}
    if sha256(without_catalog(rebuilt)) != sha256(without_catalog(record)):
        raise ValueError("must-emit token graph differs from the actual complete public tape")
    return history


def _necessary_window(history, received, layout):
    """A sufficient universal atom; never read a solver's chosen coloring."""
    record = history.record()
    first_phase2 = next((i for i, raw in enumerate(received) if raw == b"\x29\x02\x00"), None)
    if first_phase2 is None:
        raise ValueError("must-emit lacks the observed first STANDBY transition")
    prefix = received[:first_phase2 + 1]
    if prefix[0] != START or any(raw[0] not in (2, 4, 6, 7, 16, 40, 41, 90) for raw in prefix):
        raise ValueError("must-emit prefix contains a nonstandard start, action, WAITING or random event")
    if [raw for raw in prefix if raw[0] == 4] != [START] \
            or [raw for raw in prefix if raw[0] == 40] != [b"\x28\x00"] \
            or [raw for raw in prefix if raw[0] == 41] != [b"\x29\x01\x00", b"\x29\x02\x00"]:
        raise ValueError("must-emit covers only the fresh first turn DRAW-to-STANDBY window")
    draws = [raw for raw in prefix if raw[0] == 90]
    if len(draws) != 2 or [raw[1:3] for raw in draws] != [b"\x00\x05", b"\x01\x05"] \
            or any(len(raw) != 23 for raw in draws) or draws[1][3:] != bytes(20):
        raise ValueError("must-emit covers exactly the standard private initial five-card draws")
    prompts = [i for i, raw in enumerate(prefix) if raw[0] == 16]
    if prompts != [first_phase2 - 1] \
            or [list(row) for row in record["own_responses"] if row[0] < first_phase2] \
                != [[first_phase2 - 1, "ffffffff"]]:
        raise ValueError("must-emit needs exactly the actual first own CHAIN pass, no intervening response")
    _chain(prefix[first_phase2 - 1], player=0, passing=True)
    if prefix.index(b"\x28\x00") >= prefix.index(b"\x29\x01\x00") \
            or any(prefix.index(raw) >= prefix.index(b"\x28\x00") for raw in draws):
        raise ValueError("must-emit draw, turn and phase ordering differs from the fresh opening")

    if not isinstance(layout, list) or not layout or any(
            not isinstance(row, (list, tuple)) or len(row) != 4 or any(type(x) is not int for x in row)
            for row in layout):
        raise ValueError("must-emit needs a canonical unchanged complete root layout")
    if layout != sorted(layout) or len({tuple(row[:3]) for row in layout}) != len(layout):
        raise ValueError("must-emit root coordinates are repeated or noncanonical")
    slots = {tuple(row[:3]): row[3] for row in record["root_slots"]}
    if {tuple(row[:3]) for row in layout} != set(slots):
        raise ValueError("must-emit proposal omits or adds a public root coordinate")
    pools = {tuple(row["pool"]): dict(row["counts"]) for row in record["pools"]}
    tokens = {row["token"]: row for row in record["tokens"]}
    forced = dict(history.known)
    bindings = {token: {"kind": "public-fact", "token": token, "code": code}
                for token, code in forced.items()}
    for player, zone, seq, code in layout:
        token = slots[player, zone, seq]
        allowed = {c for pool in tokens[token]["pools"] for c in pools[tuple(pool)]}
        if code not in allowed or token in forced and forced[token] != code:
            raise ValueError("must-emit root conflicts with a public fact or declared recipe")
        forced[token] = code
        bindings.setdefault(token, {"kind": "unchanged-root-token", "token": token, "code": code,
                                    "coordinate": [player, zone, seq]})
    main = next(row for row in record["pools"] if tuple(row["pool"]) == (1, 1))
    # Standard draw uses the last five original main-deck position tokens.
    candidates = [token for token in reversed(main["tokens"][-5:]) if forced.get(token) == MAXXC]
    if not candidates:
        raise ValueError("no original opponent HAND token is forced to MaxC in every completion")
    return {"received_packet_index": first_phase2, "received_packet_hex": "290200",
            "must_emit_packet_hex": "03", "own_pass_packet_index": first_phase2 - 1,
            "handler": bindings[candidates[0]],
            "universal_basis": "direct-original-token public fact or unchanged-root equality; no witness colors/v1"}


def verify_certificate(certificate, *, profile, actual_assets, actual_setup,
                       hypothesis_sha256, history_sha256, source_sha256):
    """Pure independent verifier. Asset identities must come from actual verified assets."""
    try:
        profile_sha = verify_profile(profile, actual_assets=actual_assets, actual_setup=actual_setup)
        for digest in (hypothesis_sha256, history_sha256, source_sha256):
            _digest(digest)
        c = certificate
        if not isinstance(c, dict) or c.get("schema") != SCHEMA or c.get("law") != LAW \
                or c.get("whole_candidate_impossible") is not True \
                or c.get("scope") != "MaxC necessary window only; not general deterministic replay/v1" \
                or c.get("source_sha256") != source_sha256 or c.get("profile_sha256") != profile_sha \
                or c.get("actual_assets") != actual_assets \
                or sha256(c.get("actual_setup")) != sha256(actual_setup) \
                or c.get("hypothesis_sha256") != hypothesis_sha256 or sha256(c.get("layout")) != hypothesis_sha256 \
                or c.get("history_sha256") != history_sha256 or sha256(c.get("history_record")) != history_sha256:
            raise ValueError("must-emit certificate changed a bound record or its independent scope")
        received = [_hex(raw) for raw in c.get("received_prefix", ())]
        if not received:
            raise ValueError("must-emit requires the full original received public tape")
        history = _history(c["history_record"], received, c["public_recipe"])
        window = _necessary_window(history, received, c["layout"])
        if c.get("window") != window:
            raise ValueError("must-emit certificate changed the necessary-window derivation")
        return {"record_consistent": True, "native_replay_rerun": False,
                "whole_candidate_impossible": True, "scope": LAW}
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError("malformed must-emit prototype certificate") from exc


def certify(*, history_record, received_prefix, layout, public_recipe, profile, actual_assets, actual_setup, source_sha256):
    """Return a scoped prototype certificate or None (unknown), never guessed rejection."""
    try:
        profile_sha = verify_profile(profile, actual_assets=actual_assets, actual_setup=actual_setup)
        received = [_hex(raw) for raw in received_prefix]
        history = _history(history_record, received, public_recipe)
        window = _necessary_window(history, received, layout)
        certificate = {"schema": SCHEMA, "law": LAW, "whole_candidate_impossible": True,
            "scope": "MaxC necessary window only; not general deterministic replay/v1",
            "source_sha256": _digest(source_sha256), "profile_sha256": profile_sha,
            "actual_assets": copy.deepcopy(actual_assets), "actual_setup": copy.deepcopy(actual_setup),
            "hypothesis_sha256": sha256(layout),
            "history_sha256": sha256(history_record), "history_record": copy.deepcopy(history_record),
            "received_prefix": list(received_prefix), "layout": copy.deepcopy(layout),
            "public_recipe": copy.deepcopy(public_recipe), "window": window}
        verify_certificate(certificate, profile=profile, actual_assets=actual_assets, actual_setup=actual_setup,
            hypothesis_sha256=certificate["hypothesis_sha256"], history_sha256=certificate["history_sha256"],
            source_sha256=source_sha256)
        return certificate
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError):
        return None
