"""Fixed, public history observers registered at the opening of every followed duel.

Some cards read state the duel keeps from its opening on: a summon count, an activity counter, a flag set when a
monster is destroyed by battle. Their scripts register that bookkeeping in ``initial_effect``, so a card that first
exists as a hidden placeholder, and gets its identity later (a particle, a revealed card), would miss everything
before. The registry compiles the pinned original script tables of every such card of the deck pools, then registers
only their card-independent history callbacks at the opening. It never accepts a deck, a hidden identity or a
server-state input, creates no card and runs no initial_effect. Extra bookkeeping does not certify a search root.
"""

from __future__ import annotations

import ctypes
import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..netduel import host_view, wire_projection
from ..search.snap_engine import snapshot_api

INFORMATION_SET_SEARCH = True
PROFILE = "opening_global_watchers/v1"
#: Triple Tactics Talent, Nibiru, Typhon, Triple Tactics Thrust, Favorite HERO Flame Wingman, Droll & Lock Bird
TARGETS = (25311006, 27204311, 93039339, 35269904, 13243124, 94145021)
_TYPE_MONSTER, _TYPE_NORMAL, _TYPE_EFFECT = 0x1, 0x10, 0x20
#: what a script's initial_effect registers for the whole duel (the text a covered card's script carries)
GLOBAL_MARKERS = ("global_check", "AddCustomActivityCounter", "Effect.GlobalEffect")
SCRIPT_SHA256 = {
    "constant.lua": "01c949dad2eff92dde457258086f1ec0117888e794703cc679b64d0b6f1a8a6f",
    "utility.lua": "65fc51fa6e1f37581a4c04e4b7c5cd3b4c0dc0ba78119c01d30241bc41cb86fc",
    "procedure.lua": "aae9067432e47ab1082ae27ff5dbf5a2d9688950d3af40f333d0859b3493d3b0",
    "c25311006.lua": "7b2434f0e84f2cb61b9d3cc84ae58dc8bd66a3008f1cf4023ec67e5c6ad59177",
    "c27204311.lua": "457256bcf0e22118b679a571f74337da23e3dfaf744c32bae163597be0233ca9",
    "c93039339.lua": "f0e32b44583e279e634edb4cd63344af8207b41465684d700d8f00154eb0378e",
    "c35269904.lua": "f2a80df7f5c43aa5a19f0cb9c9815b9f0d631ab1699e30e029ccd8619e5546e2",
    "c13243124.lua": "c95f212a2c70d91c76bb240f8193b1e83a2fe51c39d0817fb7850d5d004c5940",
    "c94145021.lua": "c280468b08f71f25a4e126157a300632393294db291c91ecb6b337be472d7bc1",
}
WATCHERS = r"""
if not mf_global_watchers_registered then
    if Duel.GetTurnCount()~=0 then error('history registry needs the opening boundary') end
    mf_global_watchers_registered=true
    Duel.AddCustomActivityCounter(25311006,ACTIVITY_CHAIN,c25311006.chainfilter)
    if not c27204311.global_check then
        c27204311.global_check=true
        local e=Effect.GlobalEffect()
        e:SetType(EFFECT_TYPE_FIELD+EFFECT_TYPE_CONTINUOUS)
        e:SetCode(EVENT_SUMMON_SUCCESS)
        e:SetOperation(c27204311.checkop)
        Duel.RegisterEffect(e,0)
        local e2=e:Clone()
        e2:SetCode(EVENT_SPSUMMON_SUCCESS)
        Duel.RegisterEffect(e2,0)
    end
    if not c93039339.global_check then
        c93039339.global_check=true
        local e=Effect.GlobalEffect()
        e:SetType(EFFECT_TYPE_FIELD+EFFECT_TYPE_CONTINUOUS)
        e:SetCode(EVENT_SPSUMMON_SUCCESS)
        e:SetOperation(c93039339.chk)
        Duel.RegisterEffect(e,0)
    end
    Duel.AddCustomActivityCounter(35269904,ACTIVITY_CHAIN,aux.FilterBoolFunction(aux.NOT(Effect.IsActiveType),TYPE_MONSTER))
    if not c13243124.global_check then
        c13243124.global_check=true
        local e=Effect.GlobalEffect()
        e:SetType(EFFECT_TYPE_FIELD+EFFECT_TYPE_CONTINUOUS)
        e:SetCode(EVENT_DESTROYED)
        e:SetOperation(c13243124.checkop)
        Duel.RegisterEffect(e,0)
    end
    if not c94145021.global_check then
        c94145021.global_check=true
        local e=Effect.GlobalEffect()
        e:SetType(EFFECT_TYPE_FIELD+EFFECT_TYPE_CONTINUOUS)
        e:SetCode(EVENT_TO_HAND)
        e:SetCondition(c94145021.regcon)
        e:SetOperation(c94145021.regop)
        Duel.RegisterEffect(e,0)
    end
end
"""


class HistoryRegistryError(RuntimeError):
    """Rules or installation timing do not match this fixed history profile."""


@dataclass(frozen=True)
class HistoryRegistryReceipt:
    profile: str
    script_sha256: tuple[tuple[str, str], ...]
    registry_sha256: str
    rule_artifact_sha256: tuple[tuple[str, str], ...]
    protocol: str


def table_source(code: int, source: str) -> str:
    """Mirror the original core's script-table setup, without card creation."""
    if code not in TARGETS:
        raise HistoryRegistryError("script is not in the fixed history registry")
    return f"""
if c{code} == nil then
    c{code}={{}}
    setmetatable(c{code},Card)
    c{code}.__index=c{code}
    local previous_table,previous_code=self_table,self_code
    self_table,self_code=c{code},{code}
    do
{source}
    end
    self_table,self_code=previous_table,previous_code
end
"""


def _sources(core) -> dict[str, bytes]:
    """Bind both file and already cached readers to the approved rule bytes."""
    sources = {}
    for name, expected in SCRIPT_SHA256.items():
        key = "./script/" + name
        path = core._resolve_script(key)
        if path is None:
            raise HistoryRegistryError("missing pinned history script: " + name)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise HistoryRegistryError("unreadable history script: " + name) from exc
        cached = core._script_cache.get(key)
        if (hashlib.sha256(payload).hexdigest() != expected
                or cached is not None and hashlib.sha256(bytes(cached)).hexdigest() != expected):
            raise HistoryRegistryError("history script fingerprint mismatch: " + name)
        sources[name] = payload
    return sources


def _execute(core, duel: int, source: str) -> None:
    if not all(hasattr(core._lib, name) for name in ("duel_snapshot", "duel_rollback", "duel_snapshot_free", "duel_arena_extent")):
        raise HistoryRegistryError("history registry requires the complete snapshot API")
    api = snapshot_api(core)
    saved = api.duel_snapshot(duel)
    if not saved:
        raise HistoryRegistryError("cannot snapshot history registry installation")
    raw = source.encode("utf-8")
    name = "./script/mf-client-history-registry.lua"
    core._script_cache[name] = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
    first = len(core.log)
    try:
        ok = core.preload_script(duel, name)
        if not ok or len(core.log) != first:
            if api.duel_rollback(duel, saved) != 0:
                raise HistoryRegistryError("cannot roll back history registry installation")
            raise HistoryRegistryError("history registry installation failed: " + " | ".join(core.log[first:]))
    finally:
        api.duel_snapshot_free(saved)


def load_tables(core, duel: int) -> None:
    """Versioned table loader retained for the independent timing diagnostic."""
    sources = _sources(core)
    _execute(core, duel, "\n".join(table_source(code, sources[f"c{code}.lua"].decode("utf-8"))
                                   for code in TARGETS))


def uncovered_global_cards(core, codes) -> tuple[int, ...]:
    """The cards among ``codes`` whose script registers duel-long bookkeeping the registry does not cover. A normal
    monster has no script and registers none; any other card whose script cannot be read is reported too."""
    missing = []
    for code in sorted(set(int(code) for code in codes)):
        if code in TARGETS:
            continue
        path = core._resolve_script(f"./script/c{code}.lua")
        if path is None:
            data = core.card_pool().cards.get(code)
            if data is None or data.type & (_TYPE_MONSTER | _TYPE_NORMAL | _TYPE_EFFECT) != _TYPE_MONSTER | _TYPE_NORMAL:
                missing.append(code)
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            missing.append(code)
            continue
        if any(marker in text for marker in GLOBAL_MARKERS):
            missing.append(code)
    return tuple(missing)


def install_history_registry(core, duel: int) -> HistoryRegistryReceipt:
    """Install at opening, or idempotently revisit an already installed registry."""
    sources = _sources(core)
    source = "\n".join(table_source(code, sources[f"c{code}.lua"].decode("utf-8")) for code in TARGETS) + WATCHERS
    artifacts = (("core", core.lib_path), ("database", core.db_path),
                 ("host_view", host_view.__file__), ("wire_projection", wire_projection.__file__))
    try:
        hashes = tuple((name, hashlib.sha256(Path(path).read_bytes()).hexdigest()) for name, path in artifacts)
    except OSError as exc:
        raise HistoryRegistryError("cannot bind the history registry rule artifacts") from exc
    _execute(core, duel, source)
    return HistoryRegistryReceipt(PROFILE, tuple(sorted(SCRIPT_SHA256.items())),
                                  hashlib.sha256(source.encode("utf-8")).hexdigest(),
                                  hashes, wire_projection.PROJECTION_SCHEMA)
