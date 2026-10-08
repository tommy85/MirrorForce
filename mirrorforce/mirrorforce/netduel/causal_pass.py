"""Pure versioned early-opening CHAIN-pass lemmas; no general response determinism."""
from collections import Counter
import hashlib
import struct

from .causal_proof import sha256

LAW = "unique-opening-known-own-chain-pass-deterministic-prefix/v1"
SCOPE = "pinned-stage-empty-opening-own-chain-pass/v1"
NATIVE_MESSAGES = (2, 16, 40, 41, 90)
RECEIVED_MESSAGES = (2, 3, 4, 6, 7, 16, 40, 41, 90)
LEMMA = {
    "law": LAW, "base_profile_sha256": "d044496993aeadbfcd6d17784bfeee6ced08210ded8d0cc9696b01bc9a816902",
    "native_messages": list(NATIVE_MESSAGES), "received_messages": list(RECEIVED_MESSAGES),
    "start": "standard-empty-field; first-turn-player0; initial-five-cards-each/v1",
    "responses": "one-to-three-known-own-select-chain-minus-one; zero-opponent-responses/v1",
    "counterexample": "extra-opponent-select-chain-waiting-before-received-next-phase/v1",
    "playerop_sha256": "bfa381ff02e364346d29624689d69614aca7ebb2a0cb18cff5f05138df4cc46b",
    "source_lemma": "field::select_chain -1 returns without executing an effect after rejecting CHAIN_FORCED",
    "uncovered_message_or_response": "unknown; never a negative certificate/v1",
}
PREFIX_LAW = "unique-opening-known-own-chain-pass-deterministic-prefix/v2"
PREFIX_SCOPE = "pinned-stage-empty-opening-own-chain-pass/v2"
PREFIX_LEMMA = {
    **LEMMA, "law": PREFIX_LAW,
    "received_scope": "deterministic-prefix-through-mismatch; full-original-tape-sha-bound/v1",
    "own_responses": "exact-actual-own-prefix-before-mismatch; later-known-passes-retained/v1",
    "received_suffix": "only-passive-opening-refresh-phase-and-known-own-chain-passes; pending-own-idle-allowed/v1",
    "pending_idle": "one-final-unanswered-own-select-idlecmd-only; not-a-native-executed-action/v1",
}
EARLY_LAW = "unique-opening-certified-earlier-own-pass-contradiction/v3"
EARLY_SCOPE = "pinned-stage-unique-opening-earlier-own-pass-contradiction/v3"
EARLY_LEMMA = {
    **LEMMA, "law": EARLY_LAW,
    "opening": "complete-current-root-bijection-to-all-original-tokens; no-cuts-births-deaths/v1",
    "initial_pools": "registered-mirror-main40-extra15-for-each-player; not-current-zone-shapes/v1",
    "received_scope": "deterministic-prefix-through-first-mismatch; full-original-tape-sha-bound/v1",
    "own_responses": "full-canonical-ordered-tape; exact-actual-prefix-before-mismatch/v1",
    "suffix": "bound-but-neither-executed-nor-claimed-deterministic; cannot-repair-earlier-contradiction/v1",
}
SCOPES = (SCOPE, PREFIX_SCOPE, EARLY_SCOPE)
LAWS = (LAW, PREFIX_LAW, EARLY_LAW)
BASE_QUALIFICATION = ("negative only before any submitted response, zero cuts/births/deaths and unique opening;"
                      " full native (not observer-masked) chance markers inspected; not general Lua determinism")


def verify_profile(profile):
    lemma = {SCOPE: LEMMA, PREFIX_SCOPE: PREFIX_LEMMA, EARLY_SCOPE: EARLY_LEMMA}.get(profile.get("scope"))
    if lemma is None or profile.get("own_chain_pass_lemma") != lemma \
            or profile.get("core_source_files", {}).get("playerop.cpp") != LEMMA["playerop_sha256"] \
            or profile.get("recipe_sha256") != "7413d0f0b10070c60853ead43c121a202c0e091059bc8666914137f4d4100e70" \
            or profile.get("rules") != {"start_lp": 8000, "start_hand": 5, "draw_count": 1, "duel_options": 5 << 16}:
        raise ValueError("own-pass entropy profile lacks its explicitly reviewed empty-opening lemma")
    base = {key: value for key, value in profile.items() if key != "own_chain_pass_lemma"}
    base.update(scope="pinned-stage-pre-first-response/v1", qualification=BASE_QUALIFICATION)
    if sha256(base) != LEMMA["base_profile_sha256"]:
        raise ValueError("own-pass extension changed the complete original reviewed entropy audit")


def _hex(raw):
    if not isinstance(raw, str) or not raw:
        raise ValueError("own-pass proof needs canonical nonempty message bytes")
    value = bytes.fromhex(raw)
    if value.hex() != raw:
        raise ValueError("own-pass proof needs canonical message hex")
    return value


def _chain(raw, *, player, passing):
    if len(raw) < 12 or raw[0] != 16 or raw[1] != player or len(raw) != 12 + 14 * raw[2]:
        raise ValueError("own-pass proof contains a different or truncated native CHAIN prompt")
    if passing and any(raw[12 + 14 * i + 1] for i in range(raw[2])):
        raise ValueError("a forced CHAIN cannot be certified as a legal pass")


def verify_trace(certificate, profile):
    verify_profile(profile)
    c, h = certificate, certificate["history_record"]
    prefix_scope, early_scope = profile["scope"] == PREFIX_SCOPE, profile["scope"] == EARLY_SCOPE
    if c.get("law") != profile["own_chain_pass_lemma"]["law"]:
        raise ValueError("own-pass certificate and explicit profile versions differ")
    viewer = h.get("viewer")
    if type(viewer) is not int or viewer not in (0, 1):
        raise ValueError("own-pass certificate has no exact observing player")
    count = c["native_submitted_responses"]
    if type(count) is not int or not 1 <= count <= 3 or c.get("own_answered") != count \
            or c.get("opponent_submitted_responses") != 0 or type(c.get("opponent_submitted_responses")) is not int:
        raise ValueError("own-pass certificate includes an unobserved response or too many opening passes")
    if early_scope:
        _initial_recipe(c["plan_record"], profile)
    else:
        slots = c["layout"]
        expected = {(p, 1): 35 for p in (0, 1)} | {(p, 2): 5 for p in (0, 1)} | {(p, 64): 15 for p in (0, 1)}
        if Counter((row[0], row[1]) for row in slots) != Counter(expected):
            raise ValueError("own-pass lemma only covers the registered standard empty-field opening")
    native = [_hex(raw) for raw in c["native_prefix"]]
    received = [_hex(raw) for raw in c.get("received_prefix", ())]
    mismatch = c["mismatch"]
    at = mismatch["packet_index"]
    if type(at) is not int or not 0 <= at < len(received):
        raise ValueError("own-pass mismatch is outside the original received tape")
    checked_received = received[:at + 1] if prefix_scope or early_scope else received
    if not received or any(raw[0] not in NATIVE_MESSAGES for raw in native) \
            or any(raw[0] not in RECEIVED_MESSAGES for raw in checked_received):
        raise ValueError("own-pass prefix contains an unreviewed action, random event or message")
    public_digest = hashlib.sha256(b"".join(struct.pack("<I", len(raw)) + raw for raw in received)).hexdigest()
    if public_digest != h.get("public_prefix_sha256") or received[-1].hex() != h.get("pending"):
        raise ValueError("own-pass certificate does not preserve the entire original received tape")
    turns = [raw for raw in native if raw[0] == 40]
    phases = [raw for raw in native if raw[0] == 41]
    draws = [raw for raw in native if raw[0] == 90]
    if turns != [bytes([40, 0])] or not phases \
            or phases != [bytes([41, phase, 0]) for phase in (1, 2, 4)][:len(phases)] \
            or len(draws) != 2 or [raw[1:3] for raw in draws] != [bytes([0, 5]), bytes([1, 5])] \
            or any(len(raw) != 23 for raw in draws):
        raise ValueError("own-pass prefix is not the standard first draw/standby/main opening")
    if at >= len(received) or received[at].hex() != mismatch["received_packet_hex"] \
            or received[at][0] != 41 or mismatch["synthetic_packet_hex"] != "03":
        raise ValueError("own-pass lemma only proves an extra opponent CHAIN before an observed next phase")
    _chain(native[-1], player=1 - viewer, passing=False)
    events, own = c.get("submitted_trace"), h.get("own_responses")
    if early_scope:
        own = _earlier_response_prefix(received, own, mismatch=at)
    elif prefix_scope:
        own = _known_pass_prefix(received, own, viewer=viewer, mismatch=at)
    if not isinstance(events, list) or len(events) != count or not isinstance(own, list) or len(own) != count:
        raise ValueError("own-pass proof omits submitted responses or the original own tape")
    native_prompts = [index for index, raw in enumerate(native[:-1]) if raw[0] == 16]
    if len(native_prompts) != count:
        raise ValueError("some historical native prompt has no known own response")
    previous_cursor = -1
    for index, (event, fixed) in enumerate(zip(events, own)):
        ni, cursor = event.get("native_index"), event.get("received_cursor")
        if type(ni) is not int or ni != native_prompts[index] or type(cursor) is not int \
                or not previous_cursor < cursor <= at or type(event.get("player")) is not int or event["player"] != viewer \
                or event.get("response_hex") != "ffffffff" or not isinstance(fixed, (list, tuple)) \
                or list(fixed) != [cursor - 1, "ffffffff"] \
                or event.get("prompt_hex") != native[ni].hex() or received[cursor - 1] != native[ni]:
            raise ValueError("own-pass response/prompt/cursor differs from the original exact own tape")
        _chain(native[ni], player=viewer, passing=True)
        previous_cursor = cursor


def _initial_recipe(plan, profile):
    """The outer certificate checker proves root-token/opening-token bijection.

    This checks the initial game's recipe, not where those very same tokens
    reside now. Ordinary later MOVE events do not change the unique opening.
    """
    from .agent_public_recipe import declare
    openings = plan.get("openings")
    if not isinstance(openings, (list, tuple)) or len(openings) != 4:
        raise ValueError("earlier contradiction lacks all four initial recipe pools")
    pools = {}
    for row in openings:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError("malformed initial pool in earlier contradiction")
        player, zone, tokens, codes = row
        size = 40 if zone == 1 else 15 if zone == 64 else None
        if type(player) is not int or player not in (0, 1) or type(zone) is not int \
                or (player, zone) in pools or size is None or len(tokens) != size or len(codes) != size:
            raise ValueError("earlier contradiction changed an initial 40/15 pool")
        pools[player, zone] = codes
    for player in (0, 1):
        if declare(pools[player, 1], pools[player, 64])["sha256"] != profile["recipe_sha256"]:
            raise ValueError("earlier contradiction changed the registered initial recipe")


def _earlier_response_prefix(received, own, *, mismatch):
    """Bind the full actual tape, but certify only responses before contradiction.

    A later actual action is evidence in the complete history, not an action
    executed by this failed native witness and not a determinism claim about
    the suffix. No possible suffix can change a mismatched earlier packet.
    """
    if not isinstance(own, list):
        raise ValueError("earlier contradiction lacks the original complete own response tape")
    previous = -1
    for row in own:
        if not isinstance(row, (list, tuple)) or len(row) != 2 or type(row[0]) is not int \
                or not previous < row[0] < len(received) - 1:
            raise ValueError("full own response cursors must be canonical, increasing and before pending")
        _hex(row[1])
        previous = row[0]
    return [row for row in own if row[0] < mismatch]


def _known_pass_prefix(received, own, *, viewer, mismatch):
    """The suffix is bound, not simulated or claimed deterministic by this prefix proof.

    Still reject every executed-action, opponent prompt, chance event or
    non-opening suffix. Only a final unanswered IDLE menu may be new here.
    A contradiction before the next actual own pass cannot be repaired by
    that later pass; it is not a response submitted to the failed witness.
    """
    if not isinstance(own, list) or not 1 <= len(own) <= 3:
        raise ValueError("prefix scope needs the complete bounded known-own-pass tape")
    if any(raw[0] not in (2, 6, 7, 16, 41) for raw in received[mismatch + 1:-1]) \
            or received[-1][0] not in (11, 16) or len(received[-1]) < 2 or received[-1][1] != viewer:
        raise ValueError("own-pass suffix contains an executed action, opponent event or non-opening menu")
    phases = [raw for raw in received if raw[0] == 41]
    if phases != [bytes([41, phase, 0]) for phase in (1, 2, 4)][:len(phases)]:
        raise ValueError("own-pass suffix extends outside the initial draw/standby/main phases")
    prompts = [i for i, raw in enumerate(received[:-1]) if raw[0] in (11, 16)]
    if any(not isinstance(row, (list, tuple)) or len(row) != 2 or type(row[0]) is not int for row in own) \
            or [list(row) for row in own] != [[i, "ffffffff"] for i in prompts]:
        raise ValueError("the original full own tape omits a prompt or contains an executed non-pass action")
    for i in prompts:
        _chain(received[i], player=viewer, passing=True)
    if received[-1][0] == 16:
        _chain(received[-1], player=viewer, passing=False)
    else:
        # Merely parse the unsubmitted wire menu; never execute an IDLE action.
        raw, cursor = received[-1], 2
        for group in range(6):
            if cursor >= len(raw):
                raise ValueError("truncated unanswered own IDLE menu")
            cursor += 1 + raw[cursor] * (11 if group == 5 else 7)
        if cursor + 3 != len(raw) or any(flag not in (0, 1) for flag in raw[cursor:]):
            raise ValueError("malformed unanswered own IDLE menu")
    return [row for row in own if row[0] < mismatch]
