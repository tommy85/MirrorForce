"""Trusted only bridge from owned fresh native construction to pure records.

No runtime setup dictionary is accepted from a caller. A live bank capability
owns the source/assets, and only this bridge can issue an admissible certificate
after building the exact natural driver and reading both native queue boundaries.
Offline verification remains record consistency, not a claim to rerun an engine.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import MutableMapping
from dataclasses import asdict
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import weakref

from . import causal_profile as P
from . import causal_must_emit as M
from .causal_proof import sha256
from .agent_public_recipe import declare
from .wire_projection import InitialRefreshCore
from ..puzzle.core import CardData, Core
from ..common.client_root import RootEnvelope
from ..worldmodel.engine import DuelConfig, DuelDriver

_BANKS = weakref.WeakKeyDictionary()
_BUILDS = weakref.WeakKeyDictionary()
_BUILD_METHOD = DuelDriver.build
_RESOLVER_METHOD = Core._resolve_script
_READER_METHOD = Core._read_script
_PROXY_START = InitialRefreshCore.start_duel
_CORE_METHODS = {name: getattr(Core, name) for name in (
    "reset_session", "create_duel", "set_player_info", "new_card", "start_duel", "process",
    "get_message", "set_responseb", "_query_card_data", "_read_card", "_read_script", "_resolve_script")}
_DRIVER_METHODS = {name: getattr(DuelDriver, name) for name in ("build", "_respond", "_observe")}


class RuntimeRefusal(ValueError):
    """No certificate may be issued; the current bank must fail without replacement."""


class RuntimeUncovered(RuntimeRefusal):
    """Actual native execution left the reviewed closure; classify the bank as unknown."""


def _verify_reads(events, *, expected=P.CONSTRUCTOR_SCRIPTS):
    try:
        P.verify_script_reads(events, expected=expected)
    except ValueError as exc:
        raise RuntimeUncovered(str(exc)) from exc


class _CacheReadAudit(MutableMapping):
    """Forward to the exact original mapping/buffers; do not replace native callbacks.

    Core._read_script dynamically calls self._script_cache.get even for a hit.
    Only that operation and writes are logged. Never raise from a ctypes
    reader callback: the owned Python boundary checks the completed log.
    """
    def __init__(self, original):
        self.original, self.events = original, []

    def __getitem__(self, key):
        return self.original[key]

    def __iter__(self):
        return iter(self.original)

    def __len__(self):
        return len(self.original)

    def get(self, key, default=None):
        value = self.original.get(key, default)
        hit = key in self.original
        self.events.append({"operation": "get", "path": key, "hit": hit,
                            "sha256": hashlib.sha256(bytes(value)).hexdigest() if hit else None})
        return value

    def __setitem__(self, key, value):
        self.events.append({"operation": "set", "path": key, "sha256": hashlib.sha256(bytes(value)).hexdigest()})
        self.original[key] = value

    def __delitem__(self, key):
        self.events.append({"operation": "delete", "path": key})
        del self.original[key]


def source_map():
    root = Path(__file__).resolve().parent.parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in P.SOURCE_FILES}


def verify_registered_source(profile, *, source_sha256=None):
    P.verify_bundle(profile)
    expected = profile["producer_source_sha256"]
    if source_sha256 is not None and source_sha256 != expected:
        raise RuntimeRefusal("bank source digest is not the externally registered complete producer map")
    actual = source_map()
    P.verify_source_map(actual, expected)
    if actual != profile["producer_source_map"]:
        raise RuntimeRefusal("actual producer bytes drifted from the prior external registration")
    return actual


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _asset_snapshot(core, profile):
    from .causal_entropy import script_tree_digest
    if type(core) is not Core or len(core.script_dirs) != 1:
        raise RuntimeRefusal("requires the exact registered Core and one audited script directory")
    script_root = Path(core.script_dirs[0]).resolve()
    script_sha, rows = script_tree_digest(core.script_dirs[0])
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
             if len(line.split()) >= 6 and line.split()[-1].startswith("/")}
    if str(Path(core.lib_path).resolve()) not in paths:
        raise RuntimeRefusal("the actual native Core library is not the registered mapped file")
    lua = {path for path in paths if "liblua5.3.so" in path}
    if not lua or {_sha_file(path) for path in lua} != {profile["legacy_profile"]["lua_sha256"]}:
        raise RuntimeRefusal("mapped Lua runtimes differ from the registered profile")
    databases = {row[1]: Path(row[2]).resolve() for row in core._db.execute("PRAGMA database_list")}
    if databases.get("main") != Path(core.db_path).resolve():
        raise RuntimeRefusal("the live SQLite reader and registered database path differ")
    assets = {"core_sha256": _sha_file(core.lib_path), "scripts_sha256": script_sha,
              "lua_sha256": next(iter({_sha_file(path) for path in lua})),
              "database_sha256": _sha_file(core.db_path), "core_commit": profile["legacy_profile"]["core_commit"]}
    base = profile["legacy_profile"]
    expected = {"core_sha256": base["core_library_sha256s"][0], "scripts_sha256": base["scripts_sha256"],
                "lua_sha256": base["lua_sha256"], "database_sha256": M.DATABASE_SHA256,
                "core_commit": base["core_commit"]}
    if assets != expected:
        raise RuntimeRefusal("actual native/scripts/Lua/database bytes differ from the registered assets")
    kinds = {int(code): int(card.type) for code, card in core.card_pool().cards.items()}
    catalog = hashlib.sha256(json.dumps(sorted(kinds.items())).encode()).hexdigest()
    if catalog != M.CATALOG_SHA256:
        raise RuntimeRefusal("the actual public catalogue differs from the separately pinned graph catalogue")
    return {"assets": assets, "script_root": script_root, "script_map": dict(rows), "catalog_sha256": catalog}


def _card_fields(card):
    if type(card) is not CardData:
        raise RuntimeRefusal("actual card reader cache contains an unexpected metadata structure")
    return {name: [int(item) for item in getattr(card, name)] if hasattr(kind, "_length_") else int(getattr(card, name))
            for name, kind in CardData._fields_}


def _card_data_snapshot(core, *, require_constructor):
    # Query through the unchanged actual Core decoder; compare semantic
    # fields, never unspecified ctypes padding and never clear any cache.
    fields, cached = {}, []
    for code in P.RELEVANT_CARD_CODES:
        actual = core._query_card_data(code)
        if actual is None or code in core._known_missing:
            raise RuntimeRefusal("registered constructor/token card is absent from the actual DB reader")
        value = _card_fields(actual)
        fields[str(code)] = value
        if code in core._card_cache:
            if _card_fields(core._card_cache[code]) != value:
                raise RuntimeRefusal("actual cached CardData fields differ from the live pinned DB: " + str(code))
            cached.append(code)
        elif require_constructor and code in P.RECIPE_CARD_CODES:
            raise RuntimeRefusal("fresh construction did not load required actual CardData: " + str(code))
    digest = sha256(fields)
    if digest != P.RELEVANT_CARD_DATA_SHA256:
        raise RuntimeRefusal("actual decoded constructor/token metadata differs from the pinned semantic field digest")
    return {"relevant_card_data_sha256": digest, "cached_relevant_card_codes": cached}


def _cache_snapshot(core, assets, *, require_constructor):
    if type(core) is not Core or Core._resolve_script is not _RESOLVER_METHOD \
            or Core._read_script is not _READER_METHOD \
            or getattr(core._resolve_script, "__func__", None) is not _RESOLVER_METHOD \
            or getattr(core._read_script, "__func__", None) is not _READER_METHOD \
            or len(core.script_dirs) != 1 or Path(core.script_dirs[0]).resolve() != assets["script_root"]:
        raise RuntimeRefusal("actual Core reader/resolver or script root was replaced")
    if any(getattr(Core, name) is not method or getattr(getattr(core, name), "__func__", None) is not method
           for name, method in _CORE_METHODS.items()):
        raise RuntimeRefusal("actual native construction/processing/reader method was replaced")
    expected = assets["script_map"]
    for name in P.CONSTRUCTOR_SCRIPTS:
        resolved = core._resolve_script("./script/" + name)
        if resolved is None or resolved.resolve() != assets["script_root"] / name \
                or expected.get(name) != P.CONSTRUCTOR_SCRIPTS[name] \
                or _sha_file(resolved) != P.CONSTRUCTOR_SCRIPTS[name]:
            raise RuntimeRefusal("actual script resolution escapes or differs from the audited constructor tree")
    loaded, opaque = {}, {}
    for path, buffer in core._script_cache.items():
        if not isinstance(path, str) or not path.startswith("./script/") \
                or "/" in path[len("./script/"):]:
            raise RuntimeRefusal("native script cache contains an unregistered loader path")
        name = path[len("./script/"):]
        resolved = core._resolve_script(path)
        digest = hashlib.sha256(bytes(buffer)).hexdigest()
        if name in P.OPAQUE_FOLLOWER_CHUNKS and resolved is None and name not in expected:
            opaque[name] = digest
            continue
        if resolved is None or resolved.resolve() != assets["script_root"] / name or expected.get(name) != digest:
            raise RuntimeRefusal(f"actual loaded script cache differs from the audited file-map bytes: {path}; "
                                 f"file={expected.get(name)} cache={digest} resolved={resolved}")
        loaded[name] = digest
    constructor = {name: loaded[name] for name in P.CONSTRUCTOR_SCRIPTS if name in loaded}
    if require_constructor and constructor != P.CONSTRUCTOR_SCRIPTS:
        raise RuntimeRefusal("fresh native construction did not actually load all 32 cards and three helpers")
    return {"constructor_scripts": constructor, "loaded_script_cache_sha256": sha256(loaded),
            "unread_follower_chunks": opaque,
            **_card_data_snapshot(core, require_constructor=require_constructor),
            "resolver_law": "actual-resolved-files-and-loaded-cache-match-audited-tree/v1"}


def _active(root, profile=None):
    if type(root) is not RootEnvelope:
        raise RuntimeRefusal("requires this client's exact owned root")
    root._check()
    bank = _BANKS.get(root)
    if root.branch_session is None or bank is None or bank["branch"] is not root.branch_session \
            or profile is not None and sha256(profile) != bank["profile_sha256"]:
        raise RuntimeRefusal("no live owned bank capability matches this root and profile")
    return bank


@contextmanager
def bank_scope(root, profile, *, source_sha256, history_sha256):
    if type(root) is not RootEnvelope or root.branch_session is None or root in _BANKS:
        raise RuntimeRefusal("bank scope requires a fresh owned branch lease")
    source = verify_registered_source(profile, source_sha256=source_sha256)
    assets = _asset_snapshot(root.owner.core, profile)
    _cache_snapshot(root.owner.core, assets, require_constructor=False)
    bank = {"profile": copy.deepcopy(profile), "profile_sha256": sha256(profile), "source_map": source,
            "source_sha256": source_sha256, "history_sha256": history_sha256,
            "assets": assets, "branch": root.branch_session, "issued": set()}
    _BANKS[root] = bank
    try:
        try:
            yield
        except BaseException as original:
            try:
                # A typed unknown may later choose the raw policy. No fresh
                # constructor is required before a first plan-budget failure,
                # but every already loaded source/asset must remain exact.
                verify_bank_assets(root, require_constructor=False)
            except BaseException as postcheck:
                from .conditioned_particles import NaturalFeasibilityUnknown
                if isinstance(original, NaturalFeasibilityUnknown):
                    raise  # revoke any possible fallback after failed cleanup
                # Both are hard. Preserve the first cause and retain the
                # additional postcheck error in the enclosing bank report.
                original.bank_postcheck_error = f"{type(postcheck).__name__}: {postcheck}"
            raise
        else:
            verify_bank_assets(root)
    finally:
        _BANKS.pop(root, None)


def verify_bank_assets(root, *, require_constructor=True):
    bank = _active(root)
    verify_registered_source(bank["profile"], source_sha256=bank["source_sha256"])
    current = _asset_snapshot(root.owner.core, bank["profile"])
    if current != bank["assets"]:
        raise RuntimeRefusal("assets changed during this admission bank")
    _cache_snapshot(root.owner.core, current, require_constructor=require_constructor)


def verify_restored_bank(root, profile):
    """Live-only prerequisite for a bank-unknown capability, after rollback."""
    if type(root) is not RootEnvelope or root.branch_session is not None or root in _BANKS \
            or any(ticket["root"] is root for ticket in _BUILDS.values()):
        raise RuntimeRefusal("bank fallback still has live branch, reader, or native capabilities")
    root._check()
    verify_registered_source(profile)
    assets = _asset_snapshot(root.owner.core, profile)
    _cache_snapshot(root.owner.core, assets, require_constructor=False)
    root.snapshot.verify()
    return {"root_snapshot_sha256": root.snapshot.digest}


def _rules(witness):
    driver, follower = witness.driver, witness.root.owner.follower
    actual = {name: getattr(driver.config, name) for name in M.RULES}
    if any(type(value) is not int for value in actual.values()) or actual != M.RULES \
            or sha256(follower.rules) != sha256(actual):
        raise RuntimeRefusal("actual full driver/follower rules are not the explicit standard room rules")
    return actual


def _queues(driver):
    result, statuses = {}, []
    for name, length in (("activation", 2), ("random", 3)):
        function = getattr(driver.core._lib, "duel_forced_" + name + "_state", None)
        if function is None:
            raise RuntimeRefusal("native forced-state query ABI is missing")
        function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
        function.restype = ctypes.c_int32
        values = (ctypes.c_int32 * length)()
        status = function(driver.pduel, values)
        if status != 0:
            raise RuntimeRefusal("native forced-state query failed at an owned idle boundary")
        statuses.append(int(status))
        result["forced_" + name + "_state"] = list(values)
    return result, statuses


def _boundary(witness, bank, phase):
    if any(getattr(DuelDriver, name) is not method \
            or getattr(getattr(witness.driver, name), "__func__", None) is not method
            for name, method in _DRIVER_METHODS.items()):
        raise RuntimeRefusal("actual natural driver method was replaced before the native boundary")
    ticket = _BUILDS.get(witness.driver)
    if ticket is None or witness.root.owner.core._script_cache is not ticket["reader_audit"]:
        raise RuntimeRefusal("fresh witness lacks its live unmodified actual script-reader audit")
    _verify_reads(ticket["reader_audit"].events)
    queues, statuses = _queues(witness.driver)
    # The construction label is justified by build_fresh's closed call path,
    # not copied from a caller-provided/default setup record.
    setup = {"construction": "fresh-standard-complete-decks; no-puzzle-load-or-native-state-injection/v1",
             "rules": _rules(witness), **queues}
    M.verify_profile(bank["profile"]["must_emit_profile"], actual_assets=bank["assets"]["assets"], actual_setup=setup)
    return {"phase": phase, "actual_setup": setup, "queue_query_status": statuses,
            "script_read_events": copy.deepcopy(ticket["reader_audit"].events),
            **_cache_snapshot(witness.root.owner.core, bank["assets"], require_constructor=True)}


def build_fresh(witness, profile):
    """The sole capability-minting path; invokes the exact normal builder itself."""
    from .causal_replay import _Witness, _config
    if type(witness) is not _Witness:
        raise RuntimeRefusal("cannot attest a custom witness/puzzle constructor")
    root, driver = witness.root, witness.driver
    bank = _active(root, profile)
    witness._check()
    verify_registered_source(profile, source_sha256=witness.source_sha256)
    if type(driver) is not DuelDriver or type(driver.config) is not DuelConfig \
            or DuelDriver.build is not _BUILD_METHOD or getattr(driver.build, "__func__", None) is not _BUILD_METHOD \
            or type(driver.core) is not InitialRefreshCore or InitialRefreshCore.start_duel is not _PROXY_START \
            or driver.core._core is not root.owner.core or driver.pduel is not None or driver.steps != 0 \
            or driver.responses or driver.messages or driver in _BUILDS:
        raise RuntimeRefusal("requires a never-built exact natural DuelDriver with the exact core proxy")
    config = asdict(driver.config)
    if sha256(config) != sha256(asdict(_config(root, witness.plan, witness.seed))) \
            or witness.plan.history_sha256 != bank["history_sha256"] \
            or witness.source_sha256 != bank["source_sha256"]:
        raise RuntimeRefusal("fresh constructor config/plan differs from the registered candidate")
    rules = _rules(witness)
    recipes = [declare(deck.main, deck.extra)["sha256"] for deck in driver.config.decks]
    if recipes != [profile["legacy_profile"]["recipe_sha256"]] * 2:
        raise RuntimeRefusal("fresh constructor recipes differ from the initialization closure")
    _cache_snapshot(root.owner.core, bank["assets"], require_constructor=False)
    original_cache = root.owner.core._script_cache
    if type(original_cache) is not dict:
        raise RuntimeRefusal("fresh witness cannot nest or replace another script-cache audit")
    reader_audit = _CacheReadAudit(original_cache)
    ticket = {"root": root, "branch": root.branch_session, "pduel": None, "built": False,
        "reader_audit": reader_audit, "original_cache": original_cache,
        "opaque_before": _cache_snapshot(root.owner.core, bank["assets"], require_constructor=False)["unread_follower_chunks"],
        "config": config, "source_sha256": bank["source_sha256"], "profile_sha256": bank["profile_sha256"],
        "history_sha256": witness.plan.history_sha256, "hypothesis_sha256": witness.plan.root_sha256,
        "construction": {"driver": "exact-DuelDriver", "core_proxy": "exact-InitialRefreshCore",
            "fresh_handle": True, "distinct_from_follower": True, "rules": rules,
            "follower_rules": copy.deepcopy(root.owner.follower.rules), "public_recipe_sha256s": recipes,
            "config_sha256": sha256(config)}}
    _BUILDS[driver] = ticket
    root.owner.core._script_cache = reader_audit
    driver.build()  # new create_duel -> set_player_info -> real cards -> start_duel; no puzzle/force APIs
    if not driver.pduel or driver.pduel == root.owner.follower.local.pduel:
        raise RuntimeRefusal("natural construction did not create a distinct fresh hypothetical native handle")
    ticket.update(pduel=driver.pduel, before=_boundary(witness, bank, "after-fresh-build-before-process"), built=True)


def discard_build(driver):
    ticket = _BUILDS.pop(driver, None)
    if ticket is None:
        return
    core = ticket["root"].owner.core
    try:
        if core._script_cache is not ticket["reader_audit"]:
            raise RuntimeRefusal("fresh witness replaced the script-cache audit before cleanup")
        bank = _BANKS.get(ticket["root"])
        if bank is not None:
            current = _cache_snapshot(core, bank["assets"], require_constructor=ticket["built"])
            if current["unread_follower_chunks"] != ticket["opaque_before"]:
                raise RuntimeRefusal("fresh witness changed preexisting opaque follower buffers")
            if ticket["built"]:
                _verify_reads(ticket["reader_audit"].events, expected=bank["assets"]["script_map"])
    finally:
        core._script_cache = ticket["original_cache"]


def _attestation(witness, profile):
    bank = _active(witness.root, profile)
    ticket = _BUILDS.get(witness.driver)
    if ticket is None or not ticket["built"] or ticket["root"] is not witness.root or ticket["branch"] is not witness.root.branch_session \
            or ticket["pduel"] != witness.driver.pduel or ticket["config"] != asdict(witness.driver.config) \
            or ticket["history_sha256"] != witness.plan.history_sha256 \
            or ticket["hypothesis_sha256"] != witness.plan.root_sha256 \
            or ticket["source_sha256"] != bank["source_sha256"]:
        raise RuntimeRefusal("counterexample has no matching live fresh-construction capability")
    witness._check()
    source = verify_registered_source(profile, source_sha256=bank["source_sha256"])
    after = _boundary(witness, bank, "idle-native-counterexample")
    record = {"schema": P.RUNTIME_SCHEMA, "law": P.RUNTIME_LAW,
        "registered_profile_sha256": bank["profile_sha256"], "source_binding_law": P.SOURCE_LAW,
        "source_sha256": bank["source_sha256"], "source_map": source,
        "history_sha256": ticket["history_sha256"], "hypothesis_sha256": ticket["hypothesis_sha256"],
        "actual_assets": copy.deepcopy(bank["assets"]["assets"]), "actual_config": copy.deepcopy(ticket["config"]),
        "construction": copy.deepcopy(ticket["construction"]), "boundaries": [copy.deepcopy(ticket["before"]), after]}
    P.verify_runtime_attestation(record, profile=profile, hypothesis_sha256=ticket["hypothesis_sha256"],
        history_sha256=bank["history_sha256"], source_sha256=bank["source_sha256"])
    return record


def certify_counterexample(witness, history, layout, *, profile, legacy_certificate=None):
    """Use real measured setup and a live ticket; never accept setup values from callers."""
    bank = _active(witness.root, profile)
    attestation = _attestation(witness, profile)
    certificate = copy.deepcopy(legacy_certificate)
    if certificate is None:
        if witness.branches or witness.choices or witness.script_errors or witness.answered != 1 \
                or witness.submitted_total != 1 or not witness.last_counterexample:
            return None
        certificate = M.certify(history_record=history.record(),
            received_prefix=[raw.hex() for raw in witness.expected], layout=[list(row) for row in layout],
            public_recipe=witness.public, profile=profile["must_emit_profile"],
            actual_assets=attestation["actual_assets"], actual_setup=attestation["boundaries"][-1]["actual_setup"],
            source_sha256=bank["source_sha256"])
        if certificate is None:
            return None
        certificate.update(native_prefix=list(witness.native_trace), submitted_trace=copy.deepcopy(witness.submitted_trace),
                           mismatch=copy.deepcopy(witness.last_counterexample), script_errors=list(witness.script_errors))
        try:
            P.verify_native_support(certificate)
        except (ValueError, KeyError, TypeError, IndexError):
            return None
    certificate.update(registered_profile_sha256=bank["profile_sha256"], runtime_attestation=attestation)
    from .causal_rejection import verify_negative_certificate
    verify_negative_certificate(certificate, hypothesis_sha256=witness.plan.root_sha256,
        history_sha256=bank["history_sha256"], source_sha256=bank["source_sha256"], entropy_profile=profile)
    bank["issued"].add(sha256(certificate))
    return certificate


def verify_issued_certificate(root, certificate, *, profile, hypothesis_sha256, history_sha256, source_sha256):
    """Live producer check in addition to the independent pure record checker."""
    bank = _active(root, profile)
    if source_sha256 != bank["source_sha256"] or history_sha256 != bank["history_sha256"] \
            or certificate.get("hypothesis_sha256") != hypothesis_sha256 \
            or sha256(certificate) not in bank["issued"]:
        raise RuntimeRefusal("negative record was not issued by this bank's measured fresh-native capability")
