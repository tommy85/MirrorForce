"""Pinned-stage entropy audit and narrowly scoped deterministic-prefix negative proofs.

This is NOT a general Lua sandbox or a theorem prover for arbitrary scripts.
The auditable profile binds the exact core sources/library, full script tree,
public mirror recipe and standard room rules. Unclassified access fails the
audit; unclassified runtime histories remain unknown, never rejected.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import io
from pathlib import Path
import re
import subprocess
import tarfile
import weakref

from .causal_proof import sha256
from .causal_rejection import ENTROPY_SCHEMA, NEGATIVE_SCHEMA, NEGATIVE_LAW, verify_entropy_profile, verify_negative_certificate
from . import causal_pass as PASS

INFORMATION_SET_SEARCH = True
CORE_COMMIT = "d8433787852bcad858a4abc8258bf4f235d6c55f"
CORE_LIBRARY = "9db20725c3a9bd8c4ed398a934886e3d6f0537fa5ef6bcdf243a85eb564c53f8"
LUA_LIBRARY = "99d643c45f9f501eb4d6d89e17b698b94fec0de11e459b9947fdfbe8902ddf52"
SCRIPT_MANIFEST = "4729224b778e70cd71393ee4a22e4b85e855d5893e8691c9ba16a44b13190810"
CHANCE_MESSAGES = (32, 33, 36, 39, 81, 130, 131)
MATH_MEMBERS = frozenset(("abs", "ceil", "floor", "log", "max", "min"))
FORBIDDEN = frozenset(("random", "randomseed", "os", "io", "package", "require", "load", "loadstring",
                       "loadfile", "dofile", "_ENV", "getfenv", "setfenv", "rawget", "rawset", "debug",
                       "collectgarbage", "LoadScript", "LoadCardScript", "tostring", "string", "print",
                       "Debug", "setmetatable", "tonumber"))
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_LONG = re.compile(r"\[(=*)\[")
_RUNTIME = weakref.WeakKeyDictionary()
ITERATIONS = {
    ("c39000945.lua", 121): "userdata-key-order-independent-readonly-integer-sum-after-resolved-operation",
    ("c77189532.lua", 45): "integer-array-of-native-effect-multiple-returns",
    ("c16684346.lua", 32): "integer-array-of-native-effect-multiple-returns",
    ("c41570943.lua", 56): "integer-array-of-native-effect-multiple-returns",
    ("c41570943.lua", 78): "integer-array-of-native-effect-multiple-returns",
    ("c41570943.lua", 101): "integer-array-of-native-effect-multiple-returns",
    ("c78579058.lua", 61): "integer-array-of-native-effect-multiple-returns",
    ("c78579058.lua", 83): "integer-array-of-native-effect-multiple-returns",
    ("c78579058.lua", 106): "integer-array-of-native-effect-multiple-returns",
    ("c58809685.lua", 55): "next-used-only-as-table-nonemptiness-test; integer-level-keys",
    ("c58809685.lua", 58): "next-used-only-as-table-nonemptiness-test; integer-level-keys",
    ("procedure.lua", 386): "integer-array-of-native-effect-multiple-returns",
    ("procedure.lua", 449): "integer-array-of-native-effect-multiple-returns",
    ("procedure.lua", 896): "explicit-number-checked-material-code-keys",
    ("procedure.lua", 1025): "explicit-number-checked-material-code-keys",
    ("procedure.lua", 2151): "integer-array-of-native-effect-multiple-returns",
    ("procedure.lua", 2173): "integer-array-of-native-effect-multiple-returns",
    ("procedure.lua", 2194): "integer-array-of-native-effect-multiple-returns",
    ("utility.lua", 508): "material-card-code-keys; pinned-integer-call-sites",
    ("utility.lua", 1785): "native-event-code-keys; not-called-before-any-response-in-pinned-Sky",
}


def _tokens(source):
    """Small fail-closed lexical scanner: comments/strings are not executable identifiers."""
    i, line, out = 0, 1, []
    while i < len(source):
        char = source[i]
        if char.isspace():
            line += char == "\n"
            i += 1
            continue
        comment = source.startswith("--", i)
        at = i + 2 if comment else i
        long = _LONG.match(source, at)
        if long:
            ending = "]" + long.group(1) + "]"
            end = source.find(ending, long.end())
            if end < 0:
                raise ValueError("unterminated long Lua string/comment")
            if not comment:
                out.append(("string", source[long.end():end], line))
            line += source[i:end + len(ending)].count("\n")
            i = end + len(ending)
            continue
        if comment:
            end = source.find("\n", i)
            i = len(source) if end < 0 else end
            continue
        if char in ("'", '"'):
            start, at = i, i + 1
            while at < len(source) and source[at] != char:
                at += 2 if source[at] == "\\" else 1
            if at >= len(source):
                raise ValueError("unterminated Lua quoted string")
            # For the permitted _G["c"..integer] dispatch, escaped or other
            # string forms are conservatively not classified as that prefix.
            out.append(("string", source[start + 1:at], line))
            line += source[start:at + 1].count("\n")
            i = at + 1
            continue
        match = _IDENTIFIER.match(source, i)
        if match:
            out.append(("identifier", match.group(), line))
            i = match.end()
            continue
        symbol = ".." if source.startswith("..", i) else char
        out.append(("symbol", symbol, line))
        i += len(symbol)
    return out


def scan_lua(source):
    tokens = _tokens(source)
    findings, globals_, math_, permitted_globals = [], [], [], []
    members = Counter()
    iteration_sites = []
    for i, (kind, name, line) in enumerate(tokens):
        if kind != "identifier":
            continue
        if name in FORBIDDEN:
            findings.append({"line": line, "identifier": name})
        if name in ("pairs", "next"):
            iteration_sites.append({"line": line, "identifier": name})
        if name == "math":
            following = [t[1] for t in tokens[i + 1:i + 4]]
            if len(following) != 3 or following[0] != "." or following[1] not in MATH_MEMBERS or following[2] != "(":
                math_.append({"line": line, "tokens": following})
            else:
                members[following[1]] += 1
        if name == "_G":
            following = tokens[i + 1:i + 12]
            prefix = len(following) >= 5 and following[0][1] == "[" and following[1][:2] == ("string", "c") \
                     and following[2][1] == ".."
            words = [t[1] for t in following]
            simple = prefix and following[3][:2] == ("identifier", "code") and words[4] == "]"
            card = prefix and following[3][0] == "identifier" and words[4:9] == [":", "GetCode", "(", ")", "]"]
            if simple or card:
                permitted_globals.append({"line": line, "kind": "card-table-c-prefix"})
            else:
                globals_.append({"line": line, "tokens": words})
    return {"script_entropy_findings": findings, "unclassified_global_access": globals_,
            "unclassified_math_access": math_, "math_members": dict(members),
            "permitted_card_table_dispatch": permitted_globals, "iteration_sites": iteration_sites}


def script_tree_digest(path, stage_script_prefix="project/script"):
    """Exact verify_rules._scripts map law, including its explicit stage-relative prefix."""
    if Path(path).is_symlink():
        raise ValueError("entropy profile refuses a symlinked script root")
    root = Path(path).resolve()
    if not isinstance(stage_script_prefix, str) or not stage_script_prefix or stage_script_prefix.startswith("/") \
            or any(part in ("", ".", "..") for part in stage_script_prefix.split("/")) or "\\" in stage_script_prefix:
        raise ValueError("entropy profile needs a canonical explicit stage-relative script prefix")
    rows, mapping = [], {}
    for file in sorted(root.rglob("*")):
        if file.is_symlink() or not file.is_file() and not file.is_dir():
            raise ValueError("entropy profile refuses non-regular or symlinked script assets")
        if not file.is_file():
            continue
        relative = file.relative_to(root).as_posix()
        raw = file.read_bytes()
        hashed = hashlib.sha256(raw).hexdigest()
        mapping[stage_script_prefix + "/" + relative] = hashed
        rows.append((relative, hashed))
    if not rows:
        raise ValueError("entropy audit needs the complete nonempty pinned script tree")
    return sha256(mapping), rows


def audit_stage(core_source, scripts, *, core_commit, core_library_sha256s, recipe_sha256, rules,
                stage_script_prefix="project/script"):
    """Generate the review artifact, never silently approve arbitrary native source.

    The source-to-marker mapping below was individually read in d843378. A
    different commit requires a new source review, not a permissive grep.
    """
    if core_commit != CORE_COMMIT:
        raise ValueError("this audit's native RNG call graph names only its reviewed core revision")
    root = Path(core_source)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir())
             if p.is_file() and p.suffix in (".cpp", ".h", ".c")}
    needed = {"mtrandom.h", "duel.cpp", "field.cpp", "libgroup.cpp", "libduel.cpp", "operations.cpp",
              "interpreter.cpp", "ocgapi.cpp", "playerop.cpp"}
    if not needed <= files.keys():
        raise ValueError("native entropy review sources are incomplete")
    archive = subprocess.check_output(["git", "-C", str(root), "archive", "--format=tar", CORE_COMMIT])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        committed = {member.name: hashlib.sha256(tree.extractfile(member).read()).hexdigest()
                     for member in tree.getmembers() if member.isfile() and Path(member.name).suffix in (".cpp", ".h", ".c")}
    if files != committed or set(core_library_sha256s) != {CORE_LIBRARY}:
        raise ValueError("entropy audit sources/library differ from the reviewed d843 archive and formal BUILD")
    script_sha, script_rows = script_tree_digest(scripts, stage_script_prefix)
    if sha256(script_rows) != SCRIPT_MANIFEST:
        raise ValueError("iteration classifications bind the entire reviewed script manifest, not reusable line numbers")
    combined = {key: [] for key in ("script_entropy_findings", "unclassified_global_access", "unclassified_math_access",
                                   "permitted_card_table_dispatch", "iteration_sites")}
    members, lua_files = Counter(), 0
    for name, _ in script_rows:
        if not name.endswith(".lua"):
            continue
        scanned = scan_lua((Path(scripts) / name).read_text(encoding="utf-8"))
        lua_files += 1
        members.update(scanned["math_members"])
        for key in combined:
            combined[key].extend({"file": name, **row} for row in scanned[key])
    failed = any(combined[key] for key in ("script_entropy_findings", "unclassified_global_access", "unclassified_math_access"))
    unclassified_iterations = []
    for row in combined["iteration_sites"]:
        classification = ITERATIONS.get((row["file"], row["line"]))
        if classification is None:
            unclassified_iterations.append(row)
        else:
            row["classification"] = classification
    failed |= bool(unclassified_iterations)
    return {"schema": ENTROPY_SCHEMA, "scope": "pinned-stage-pre-first-response/v1",
            "result": "failed_static_entropy_audit" if failed else "passed_static_entropy_audit",
            "core_commit": core_commit, "core_library_sha256s": list(core_library_sha256s), "core_source_files": files,
            "core_source_archive_verified": True, "lua_sha256": LUA_LIBRARY,
            "lua_hash_seed": "time-and-address-dependent; never assumed fixed",
            "scripts_sha256": script_sha, "script_files": len(script_rows), "lua_files": lua_files,
            "stage_script_prefix": stage_script_prefix,
            "script_digest_law": "verify_rules-stage-relative-file-map-json/v1",
            "script_file_manifest_sha256": sha256(script_rows), "recipe_sha256": recipe_sha256, "rules": rules,
            "native_chance_messages": list(CHANCE_MESSAGES), "uncovered_native_rng": [],
            "native_rng_map": [
                {"entry": "field::shuffle / mtrandom::shuffle_vector", "source": "field.cpp:995-1060", "messages": [32, 33, 39]},
                {"entry": "Duel.ShuffleSetCard / duel::get_next_integer", "source": "libduel.cpp:1593-1670", "messages": [36]},
                {"entry": "Group.RandomSelect / duel::get_next_integer", "source": "libgroup.cpp:330-386", "messages": [81]},
                {"entry": "field::toss_coin / duel::get_next_outcome", "source": "operations.cpp:6439-6504", "messages": [130]},
                {"entry": "field::toss_dice / duel::get_next_outcome", "source": "operations.cpp:6507-6562", "messages": [131]}],
            "seed_scope": "create_duel only; no future-seed API is called during the certified prefix",
            "lua_libraries": "base,string,utf8,table,math; unsafe io/loadfile/dofile disabled in interpreter",
            "lua_lexical_law": "closed-math-calls; no dynamic-load/global-env/entropy access; card-c-prefix-only/v1",
            "math_members": dict(members), **combined,
            "unclassified_iteration_sites": unclassified_iterations,
            "qualification": "negative only before any submitted response, zero cuts/births/deaths and unique opening;"
                             " full native (not observer-masked) chance markers inspected; not general Lua determinism"}


def runtime_scope(root, profile):
    """Bind the reviewed artifact to the actual already-verified hypothesis assets."""
    core = root.owner.core
    identity = sha256(profile)
    cached = _RUNTIME.get(core)
    if cached is not None and cached[0] == identity:
        result = cached[1]
        library, scripts = result["actual_core_sha256"], result["actual_scripts_sha256"]
    else:
        if len(core.script_dirs) != 1:
            raise ValueError("negative entropy certificates require exactly the registered pinned script directory")
        library = hashlib.sha256(Path(core.lib_path).read_bytes()).hexdigest()
        scripts, _ = script_tree_digest(core.script_dirs[0], profile.get("stage_script_prefix"))
        mapped = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                  if len(line.split()) >= 6 and "liblua5.3.so" in line.split()[-1]}
        if not mapped or any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != LUA_LIBRARY for p in mapped):
            raise ValueError("hypothesis Lua runtime differs from the audited hash-seed/iteration implementation")
        result = {"actual_core_sha256": library, "actual_scripts_sha256": scripts, "core_commit": CORE_COMMIT,
                  "actual_lua_sha256": LUA_LIBRARY}
        _RUNTIME[core] = (identity, result)
    verify_entropy_profile(profile, core_sha256=library, scripts_sha256=scripts, core_commit=CORE_COMMIT)
    rules = {"start_lp": 8000, "start_hand": 5, "draw_count": 1, "duel_options": 5 << 16,
             **dict(root.owner.follower.rules)}
    from .agent_public_recipe import declare
    own = root.owner.follower.decks[root.owner.follower.viewer]
    if profile.get("recipe_sha256") != declare(own.main, own.extra)["sha256"] or profile.get("rules") != rules:
        raise ValueError("negative entropy profile belongs to another public recipe or room rule set")
    return result


def certify_initial_counterexample(root, history, layout, plan, witness, *, entropy_profile, source_sha256):
    record = history.record()
    mismatch = witness.last_counterexample
    if record["permutations"] or record["dead"] or plan.births or len(record["tokens"]) != len(record["root_slots"]) \
            or witness.answered or witness.submitted_total or witness.branches or witness.choices or witness.script_errors \
            or not mismatch or not mismatch.get("native_message_hex") or not mismatch.get("received_packet_hex") \
            or not witness.native_before_response \
            or any(bytes.fromhex(raw)[0] in CHANCE_MESSAGES for raw in witness.native_before_response):
        return None
    scope = runtime_scope(root, entropy_profile)
    certificate = {"schema": NEGATIVE_SCHEMA, "law": NEGATIVE_LAW, "whole_candidate_impossible": True,
        "hypothesis_sha256": plan.root_sha256, "history_sha256": plan.history_sha256, "source_sha256": source_sha256,
        "entropy_profile_sha256": sha256(entropy_profile), **scope, "opening_unique": True,
        "cuts": 0, "births": 0, "dead": 0, "tokens": len(record["tokens"]), "root_tokens": len(record["root_slots"]),
        "own_answered": witness.answered, "native_submitted_responses": witness.submitted_total,
        "branches": witness.branches, "choices": len(witness.choices), "native_prefix": list(witness.native_before_response),
        "mismatch": dict(mismatch), "layout": [list(row) for row in layout],
        "history_record": record, "plan_record": plan.record(), "script_errors": []}
    verify_negative_certificate(certificate, hypothesis_sha256=plan.root_sha256, history_sha256=plan.history_sha256,
                                source_sha256=source_sha256, entropy_profile=entropy_profile)
    return certificate


def own_pass_profile(base):
    """Explicitly derive, never silently broaden, the one reviewed fixed-stage profile."""
    if sha256(base) != PASS.LEMMA["base_profile_sha256"]:
        raise ValueError("the own-pass extension requires the unchanged separately reviewed base audit")
    verify_entropy_profile(base, core_sha256=CORE_LIBRARY, scripts_sha256=base["scripts_sha256"], core_commit=CORE_COMMIT)
    return {**copy.deepcopy(base), "scope": PASS.SCOPE, "own_chain_pass_lemma": copy.deepcopy(PASS.LEMMA),
            "qualification": "unique empty-field opening, fixed actual own CHAIN passes only;"
                             " no opponent responses or chance/action messages; not arbitrary Lua determinism"}


def own_pass_prefix_profile(base):
    """A separate v2 opt-in; never changes a v1 profile or its checker semantics."""
    result = own_pass_profile(base)
    result.update(scope=PASS.PREFIX_SCOPE, own_chain_pass_lemma=copy.deepcopy(PASS.PREFIX_LEMMA),
        qualification="unique empty-field opening; certify only the native/received prefix through contradiction;"
                      " full tape/layout bound; later actual own CHAIN passes and an unanswered own IDLE menu only")
    PASS.verify_profile(result)
    return result


def earlier_contradiction_profile(base):
    """Independent v3 opt-in: unique opening plus an already-refuted prefix."""
    result = own_pass_profile(base)
    result.update(scope=PASS.EARLY_SCOPE, own_chain_pass_lemma=copy.deepcopy(PASS.EARLY_LEMMA),
        qualification="complete current root uniquely fixes every original opening token;"
                      " a certified earlier own-pass prefix contradiction excludes every later continuation;"
                      " later actual actions are bound evidence, not a claimed deterministic native prefix")
    PASS.verify_profile(result)
    return result


def certify_own_pass_counterexample(root, history, layout, plan, witness, *, entropy_profile, source_sha256):
    """Classify only the failed native prefix, under its explicit versioned opening lemma."""
    if not isinstance(entropy_profile, dict) or entropy_profile.get("scope") not in PASS.SCOPES:
        return None
    PASS.verify_profile(entropy_profile)
    record, mismatch = history.record(), witness.last_counterexample
    if record["permutations"] or record["dead"] or plan.births or len(record["tokens"]) != len(record["root_slots"]) \
            or not 1 <= witness.answered <= 3 or witness.submitted_total != witness.answered \
            or witness.branches or witness.choices or witness.script_errors or not mismatch:
        return None
    certificate = {"schema": NEGATIVE_SCHEMA, "law": entropy_profile["own_chain_pass_lemma"]["law"], "whole_candidate_impossible": True,
        "hypothesis_sha256": plan.root_sha256, "history_sha256": plan.history_sha256, "source_sha256": source_sha256,
        "entropy_profile_sha256": sha256(entropy_profile), "opening_unique": True,
        "cuts": 0, "births": 0, "dead": 0, "tokens": len(record["tokens"]), "root_tokens": len(record["root_slots"]),
        "own_answered": witness.answered, "native_submitted_responses": witness.submitted_total,
        "opponent_submitted_responses": 0, "branches": witness.branches, "choices": len(witness.choices),
        "native_prefix": list(witness.native_trace), "submitted_trace": copy.deepcopy(witness.submitted_trace),
        "received_prefix": [p.hex() for p in witness.expected], "mismatch": dict(mismatch),
        "layout": [list(row) for row in layout], "history_record": record, "plan_record": plan.record(), "script_errors": []}
    try:
        # A merely unmodeled event remains unknown. Runtime identity errors,
        # by contrast, are hard errors and are intentionally outside this block.
        PASS.verify_trace(certificate, entropy_profile)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return None
    certificate.update(runtime_scope(root, entropy_profile))
    verify_negative_certificate(certificate, hypothesis_sha256=plan.root_sha256, history_sha256=plan.history_sha256,
                                source_sha256=source_sha256, entropy_profile=entropy_profile)
    return certificate
