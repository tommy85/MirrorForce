"""Pure structural verification of causal replay records, never a native rerun.

Safe in an evaluator/host process: stdlib only, no Core, environment, model,
filesystem or network imports. Artifact hashes authenticate the original
execution elsewhere; these checks establish a record's declared consistency.
"""
from __future__ import annotations

import hashlib
import json

PROOF_SCHEMA = "mirrorforce_natural_causal_root/v1"
LAW = "natural-complete-causal-witness/v1"
ACTIVATION_LEGALITY = "natural-menu-only; blank-follower-force-not-installed/v1"
RESPONSE_LAW = "native-acceptance-no-retry-and-exact-received-prefix/v1"
WITNESS_LAW = "seeded-first-fully-consistent; root-layout-unchanged/v1"
PLAN_SCHEMA = "mirrorforce_causal_opening_plan/v1"
PLAN_LAW = "bounded-public-capacity-witness; unchanged-root-proposal/v1"
HISTORY_LAW = "public-position-tokens-and-anonymous-permutation-cuts/v1"
SHUFFLE_SCHEMA = "mirrorforce_consumed_shuffle_plan/v1"


def sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("causal proof needs canonical SHA-256 identities")
    return value


def _integer(value, maximum, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("causal proof integer is outside its declared range")
    return value


def _sequence(value):
    if not isinstance(value, (list, tuple)):
        raise ValueError("causal proof needs explicit sequences")
    return value


def producer_contract():
    return {"schema": PROOF_SCHEMA, "initialization": LAW, "activation_legality": ACTIVATION_LEGALITY,
            "response_witness_law": RESPONSE_LAW, "witness_selection": WITNESS_LAW,
            "shuffle_abi": 1, "scope": "detached-complete-hypothesis-only/v1",
            "verification": "record-self-consistency; not native re-execution/v1"}


def verify_natural_root_proof(proof, *, hypothesis_sha256, history_sha256, source_sha256):
    try:
        return _verify(proof, hypothesis_sha256=hypothesis_sha256,
                       history_sha256=history_sha256, source_sha256=source_sha256)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("malformed causal replay proof structure") from exc


def _verify(proof, *, hypothesis_sha256, history_sha256, source_sha256):
    binding = tuple(_sha(x) for x in (hypothesis_sha256, history_sha256, source_sha256))
    if not isinstance(proof, dict) or any(proof.get(key) != value for key, value in producer_contract().items()
                                         if key in ("schema", "initialization", "activation_legality",
                                                    "response_witness_law", "witness_selection")) \
            or any(proof.get(key) is not True for key in ("public_stream_equal", "root_menu_equal",
                                                         "root_layout_equal", "own_deck_applied")) \
            or proof.get("training_eligible") is not False \
            or (proof.get("hypothesis_sha256"), proof.get("history_sha256")) != binding[:2]:
        raise ValueError("natural causal replay declaration differs from the bank")
    for key in ("stream_sha256", "public_prefix_sha256"):
        _sha(proof.get(key))
    graph, plan = proof.get("causal_plan"), proof.get("causal_plan_record")
    if not isinstance(graph, dict) or not isinstance(plan, dict) \
            or any(record.get("schema") != PLAN_SCHEMA or record.get("law") != PLAN_LAW
                   or record.get("history_sha256") != history_sha256 or record.get("root_sha256") != hypothesis_sha256
                   or record.get("native_replay_admitted") is not False for record in (graph, plan)) \
            or plan.get("history_law") != HISTORY_LAW or plan.get("own_deck_applied") is not False \
            or graph.get("plan_sha256") != sha256(plan):
        raise ValueError("causal graph/plan identity differs from its exact root and history")
    colors = _sequence(plan.get("token_codes"))
    for code in colors:
        _integer(code, 0x7fffffff, 1)
        if 999000001 <= code <= 999000004:
            raise ValueError("causal proof contains a placeholder identity")
    roots = _sequence(plan.get("root"))
    layout = []
    for row in roots:
        if len(_sequence(row)) != 5:
            raise ValueError("causal root row has the wrong shape")
        p, z, s, token, code = row
        _integer(p, 1)
        _integer(s, 254)
        _integer(token, len(colors), 1)
        if z not in (1, 2, 4, 8, 16, 32, 64) or type(z) is not int or code != colors[token - 1]:
            raise ValueError("causal root row has an invalid zone or token identity")
        if (z == 4 and s >= 7) or (z == 8 and s >= 8):
            raise ValueError("causal field position exceeds its physical zone")
        layout.append([p, z, s, code])
    if layout != sorted(layout) or len({tuple(r[:3]) for r in layout}) != len(layout) \
            or sha256(layout) != hypothesis_sha256 or graph.get("tokens") != len(colors) \
            or graph.get("root_cards") != len(roots):
        raise ValueError("causal root coverage or layout hash differs")
    cuts, cut_records = _sequence(plan.get("cuts")), _sequence(proof.get("causal_cuts"))
    if len(cuts) != len(cut_records) or graph.get("cuts") != len(cuts) or len(cuts) > 4096:
        raise ValueError("causal cut count differs from its graph")
    words = [1, len(cuts)]
    ordered_tokens = []
    lineage, live = {}, set()
    opening_zones = set()
    for opening in _sequence(plan.get("openings")):
        if len(_sequence(opening)) != 4:
            raise ValueError("causal opening shape differs")
        p, z, tokens, codes = opening
        _integer(p, 1)
        if type(z) is not int or z not in (1, 64) or (p, z) in opening_zones:
            raise ValueError("causal opening pool repeats or has an invalid zone")
        opening_zones.add((p, z))
        if len(_sequence(tokens)) != len(_sequence(codes)):
            raise ValueError("causal opening code/token counts differ")
        for token, code in zip(tokens, codes):
            _integer(token, len(colors), 1)
            if token in lineage or code != colors[token - 1]:
                raise ValueError("causal opening repeats or changes a source token")
            lineage[token] = token
            live.add(token)
    if opening_zones != {(0, 1), (0, 64), (1, 1), (1, 64)}:
        raise ValueError("causal opening omits a public recipe pool")
    for birth in _sequence(plan.get("births")):
        if len(_sequence(birth)) != 2:
            raise ValueError("causal birth shape differs")
        token, code = birth
        _integer(token, len(colors), 1)
        if token in lineage or code != colors[token - 1]:
            raise ValueError("causal birth repeats or changes a source token")
        lineage[token] = token
        live.add(token)
    total = 0
    for cut, history_cut in zip(cuts, cut_records):
        if len(_sequence(cut)) != 4 or not isinstance(history_cut, dict):
            raise ValueError("causal cut certificate has the wrong shape")
        packet, msg, after, matched = cut
        _integer(packet, 0xffffffff)
        after, matched = _sequence(after), _sequence(matched)
        old = _sequence(history_cut.get("before"))
        positions = _sequence(history_cut.get("positions"))
        if history_cut.get("packet") != packet or history_cut.get("msg") != msg \
                or list(_sequence(history_cut.get("after"))) != list(after) \
                or len(positions) != len(old) or len(old) != len(after) or len(matched) != len(after) \
                or len(set(old)) != len(old) or len(set(after)) != len(after) \
                or len(set(matched)) != len(matched) or set(matched) != set(old):
            raise ValueError("causal cut is not a complete declared token bijection")
        for token in (*old, *after, *matched):
            _integer(token, len(colors), 1)
        for new, source in zip(after, matched):
            if colors[new - 1] != colors[source - 1]:
                raise ValueError("a causal cut changes its card identity")
        if not set(old) <= live or set(after) & lineage.keys():
            raise ValueError("causal cut consumes a retired token or recreates an existing token")
        for new, source in zip(after, matched):
            lineage[new] = lineage[source]
        live.difference_update(old)
        live.update(after)
        for pos in positions:
            if len(_sequence(pos)) != 3:
                raise ValueError("causal cut position shape differs")
            for value in pos:
                _integer(value, 255)
        zones = {tuple(pos[:2]) for pos in positions}
        if len(zones) > 1 or len({tuple(pos) for pos in positions}) != len(positions):
            raise ValueError("one shuffle cannot repeat a slot or span players/zones")
        if positions:
            p, z = zones.pop()
            if "player" in history_cut and (history_cut["player"], history_cut.get("location")) != (p, z):
                raise ValueError("causal cut metadata and participating positions differ")
        else:
            p, z = history_cut.get("player"), history_cut.get("location")
            if msg != 39 or z != 64:
                raise ValueError("an empty causal event must explicitly name a player's Extra shuffle")
        _integer(p, 1)
        n = len(positions)
        total += n
        if total > 262144 or n > 255 or (msg == 36 and (z not in (4, 8) or not 2 <= n <= 5)) \
                or msg != 36 and (msg, z) not in ((32, 1), (33, 2), (39, 64)):
            raise ValueError("causal shuffle message/zone/capacity is invalid")
        order = sorted(range(n), key=lambda i: positions[i][2])
        sequences = [positions[i][2] for i in order]
        if (msg == 36 and max(sequences) > 4) or msg != 36 and sequences != list(range(n)):
            raise ValueError("causal shuffle sequence bounds differ")
        sources = [old[i] for i in order]
        targets = [after[i] for i in order]
        selected = dict(zip(after, matched))
        permutation = [sources.index(selected[token]) for token in targets]
        codes = [colors[token - 1] for token in sources]
        words.extend([msg, p, z, n, *sequences, *permutation, *codes])
        ordered_tokens.append((msg, p, z, sequences, permutation, sources, targets))
    if proof.get("shuffle_words") != words:
        raise ValueError("native shuffle words differ from the certified causal token permutation")
    dead = _sequence(proof.get("causal_dead_tokens"))
    for token in dead:
        _integer(token, len(colors), 1)
    root_tokens = {row[3] for row in roots}
    if len(root_tokens) != len(roots) or len(set(dead)) != len(dead) or root_tokens & set(dead) \
            or live != root_tokens | set(dead) or set(lineage) != set(range(1, len(colors) + 1)) \
            or len({lineage[t] for t in live}) != len(live):
        raise ValueError("causal graph does not preserve each opening/born entity exactly once")
    registration = proof.get("shuffle_registration")
    if not isinstance(registration, dict) or registration.get("schema") != SHUFFLE_SCHEMA \
            or registration.get("scope") != "detached-complete-hypothesis-only/v1" \
            or registration.get("native_replay_admitted") is not False \
            or registration.get("source_sha256") != source_sha256 \
            or registration.get("history_sha256") != history_sha256 or registration.get("root_sha256") != hypothesis_sha256 \
            or registration.get("events") != len(cuts) or registration.get("words_sha256") != sha256(words):
        raise ValueError("native shuffle installation words or bank scope binding differ")
    receipt = _sequence(proof.get("shuffle_receipts"))
    if len(receipt) < 9:
        raise ValueError("native shuffle receipt header is incomplete")
    for word in receipt:
        _integer(word, 0xffffffffffffffff)
    if list(receipt[:5]) != [1, 2, len(cuts), len(cuts), 0]:
        raise ValueError("native shuffle plan was not completely and successfully finished")
    cursor, aliases, uid_origins = 9, {}, {}
    for index, (msg, p, z, sequences, permutation, sources, targets) in enumerate(ordered_tokens):
        n = len(sequences)
        if cursor + 6 + 3 * n > len(receipt):
            raise ValueError("native shuffle receipt record is truncated")
        ordinal, flavor, rmsg, rp, rz, count = receipt[cursor:cursor + 6]
        if (ordinal, rmsg, rp, rz, count) != (index, msg, p, z, n) \
                or flavor not in ((3, 4) if msg == 36 else (1, 2)):
            raise ValueError("native shuffle receipt names another event or flavor")
        seq = receipt[cursor + 6:cursor + 6 + n]
        before = receipt[cursor + 6 + n:cursor + 6 + 2 * n]
        after = receipt[cursor + 6 + 2 * n:cursor + 6 + 3 * n]
        if list(seq) != sequences or len(set(before)) != n or any(uid == 0 for uid in before) \
                or len(set(after)) != n or set(before) != set(after) \
                or list(after) != [before[i] for i in permutation]:
            raise ValueError("native shuffle receipt is not the exact complete UID permutation")
        for token, uid in zip(sources, before):
            if token in aliases and aliases[token] != uid:
                raise ValueError("native shuffle receipt changed an existing token's hypothetical UID")
            origin = lineage[token]
            if uid in uid_origins and uid_origins[uid] != origin:
                raise ValueError("native shuffle receipts reuse a UID for two distinct opening/born cards")
            uid_origins[uid] = origin
            aliases[token] = uid
        for token, uid in zip(targets, after):
            if token in aliases:
                raise ValueError("native shuffle receipt reuses an already created output token")
            aliases[token] = uid
        cursor += 6 + 3 * n
    final = ordered_tokens[-1] if ordered_tokens else None
    last = [final[0], final[1], final[2], len(final[3])] if final else [0, 0, 0, 0]
    if cursor != len(receipt) or list(receipt[5:9]) != last:
        raise ValueError("native shuffle receipt has extra records or inconsistent final event metadata")
    return {"record_consistent": True, "native_replay_rerun": False,
            "history_sha256": history_sha256, "hypothesis_sha256": hypothesis_sha256,
            "source_sha256": source_sha256, "events": len(cuts)}
