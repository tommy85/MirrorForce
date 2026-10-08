"""Scenarios for the weak script-group tests (``test_core_script_group_gc.py``), one per process.

A core with weak script temporaries (ygopro-core branch ``mirrorforce-lane132-script-gc``) lets Lua's collector
free a group a library function returned once no Lua value refers to it, instead of keeping every group of a script
call until the outermost return. Each scenario builds a small board through the Python ctypes core, runs Lua
through it and prints one JSON line. A core refusal (``ygopro-core refusal [<reason>]``, a C++ exception) cannot
cross ctypes and ends the process: the tests read that from the exit status and stderr, which is why every scenario
runs in a process of its own.

    python core_group_gc_scenarios.py <scenario> <core library> [<arena noise pages>]

Scenarios:
  burst           one script call churning through 300,000 temporary groups (production core)
  forced_gc       a full collection inside the callbacks of every library function that fills a result group,
                  or keeps its group argument, while it calls back into Lua (diagnostic core: Debug.CollectGarbage)
  holds           label objects, KeepAlive, Group.__gc called by a script, and the audit catching a structure
                  that keeps a weak group without holding it (diagnostic core)
  released_group  a script using a group released at the end of an earlier call (expected: refusal)
  redirect_event  a monster destroyed under EFFECT_TO_GRAVE_REDIRECT_CB (the Crystal Beast rule): the
                  redirect operation's event group must be the sent group (the core used to pass the send's
                  internal record cast to a group, which a weak-group core then read as garbage)
  gc_sequence     an effect churning through temporary groups around a card selection it yields for; the
                  per-process() group counters, Lua heap and messages, straight through and again after a rollback
                  to a snapshot taken at the selection (mid-burst, the call suspended)
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import sys
from pathlib import Path
import tempfile

from mirrorforce.netduel import constants as C
from mirrorforce.netduel.actions import ActionAct
from mirrorforce.puzzle.core import PROCESSOR_BUFFER_LEN, PROCESSOR_END, PROCESSOR_FLAG, Core
from mirrorforce.puzzle.single import RESPONSE_REQUIRED, SinglePuzzle
from mirrorforce.search.snap_engine import snapshot_api

URABY, KOJIKOCY, GALAXY_SERPENT = 1784619, 1184620, 11066358            # Normal Lv4, Normal Lv4, Tuner Lv2
GAIA_KNIGHT, GEM_KNIGHT_PEARL, LINK_SPIDER = 97204936, 71594310, 98978921  # Synchro Lv6, Xyz R4, Link-1
TYPHOON, POT_OF_GREED = 5318639, 55144522
STATS = ("groups", "call_groups", "weak", "collected", "pending", "weak_slots", "lua_bytes")

BOARD = f"""Debug.ReloadFieldBegin(DUEL_ATTACK_FIRST_TURN+DUEL_SIMPLE_AI,5)
Debug.SetPlayerInfo(0,8000,0,0)
Debug.SetPlayerInfo(1,8000,0,0)
mf_trigger=Debug.AddCard({KOJIKOCY},0,0,LOCATION_HAND,0,POS_FACEDOWN)
Debug.AddCard({URABY},0,0,LOCATION_HAND,1,POS_FACEDOWN)
Debug.AddCard({TYPHOON},0,0,LOCATION_HAND,2,POS_FACEDOWN)
Debug.AddCard({POT_OF_GREED},0,0,LOCATION_HAND,3,POS_FACEDOWN)
Debug.AddCard({URABY},0,0,LOCATION_MZONE,0,POS_FACEUP_ATTACK)
Debug.AddCard({KOJIKOCY},0,0,LOCATION_MZONE,1,POS_FACEUP_ATTACK)
Debug.AddCard({GALAXY_SERPENT},0,0,LOCATION_MZONE,2,POS_FACEUP_ATTACK)
Debug.AddCard({GAIA_KNIGHT},0,0,LOCATION_EXTRA,0,POS_FACEDOWN)
Debug.AddCard({GEM_KNIGHT_PEARL},0,0,LOCATION_EXTRA,1,POS_FACEDOWN)
Debug.AddCard({LINK_SPIDER},0,0,LOCATION_EXTRA,2,POS_FACEDOWN)
Debug.AddCard({URABY},1,1,LOCATION_MZONE,0,POS_FACEUP_ATTACK)
Debug.AddCard({URABY},1,1,LOCATION_DECK,0,POS_FACEDOWN)
Debug.AddCard({URABY},0,0,LOCATION_DECK,0,POS_FACEDOWN)
Debug.ReloadFieldEnd()
-- run f inside a script call (call depth 1), the way any effect condition or card filter runs
function mf_in_call(f)
  local done=false
  Duel.GetMatchingGroupCount(function(c) if not done then done=true f() end return false end,0,LOCATION_MZONE,0,nil)
end
function mf_churn(n)
  for i=1,n do local g=Group.CreateGroup()+Group.CreateGroup() end
end
"""


class Stop(Exception):
    pass


class Session:
    """A puzzle board on one core, with the weak-group counters and the audit export bound."""

    def __init__(self, core_path: Path, workdir: Path, extra_lua: str = ""):
        self.core = Core(lib_path=core_path)
        self.lib = snapshot_api(self.core)
        self.lib.duel_group_stats.restype = ctypes.c_int32
        self.lib.duel_group_stats.argtypes = [ctypes.c_ssize_t, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int32]
        self.lib.duel_group_audit.restype = ctypes.c_int32
        self.lib.duel_group_audit.argtypes = [ctypes.c_ssize_t, ctypes.c_char_p, ctypes.c_int32]
        path = workdir / "board.lua"
        path.write_text(BOARD + extra_lua, encoding="utf-8")
        self.puzzle = SinglePuzzle(path, core=self.core)
        self.puzzle.load()
        assert self.puzzle.run_result.loaded and not self.core.log, self.core.log
        self.pduel = self.puzzle.pduel

    def stats(self) -> dict:
        out = (ctypes.c_uint64 * len(STATS))()
        assert self.lib.duel_group_stats(self.pduel, out, len(STATS)) == len(STATS)
        return dict(zip(STATS, out))

    def audit(self) -> tuple[int, str]:
        buf = ctypes.create_string_buffer(256)
        return self.lib.duel_group_audit(self.pduel, buf, len(buf)), buf.value.decode()

    def arena(self) -> int:
        return self.lib.duel_arena_extent(self.pduel)

    def run_lua(self, source: str) -> list[str]:
        """Run a chunk at call depth 0 (preload_script); returns the core log it wrote."""
        name = "./script/mf-group-gc.lua"
        raw = source.encode()
        self.core._script_cache[name] = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
        before = len(self.core.log)
        assert self.core.preload_script(self.pduel, name), self.core.log[before:]
        return self.core.log[before:]

    def drive(self, policy, stop_before=None, max_steps=600):
        """process() until the policy raises Stop; one record per process() call (messages digest and counters).

        ``stop_before`` names a prompt message type: the drive returns just before answering the first one, with
        that message, so the caller can snapshot the duel there."""
        if not self.puzzle.run_result.started:
            self.puzzle.start()
        records = []
        for _ in range(max_steps):
            raw = self.core.process(self.pduel)
            data = b""
            messages = []
            if raw & PROCESSOR_BUFFER_LEN:
                messages = self.puzzle._drain()
                data = b"".join(bytes([m.msg]) + bytes(m.payload) for m in messages)
            records.append({"messages": hashlib.sha256(data).hexdigest(), **self.stats()})
            for message in messages:
                self.puzzle._observe(message)
                if message.msg == C.MSG_RETRY:
                    raise RuntimeError("the core rejected the last response")
                if message.msg in RESPONSE_REQUIRED:
                    if stop_before is not None and message.msg == stop_before:
                        return records, message
                    try:
                        self.puzzle._answer(message, policy)
                    except Stop:
                        return records, None
            if raw & PROCESSOR_FLAG == PROCESSOR_END:
                break
        raise RuntimeError("the duel did not reach the policy's stop")


def activate_once(code):
    """Activate `code`'s effect at the first idle menu, stop at the next one; decline chains; else option 0."""
    state = {"activated": False}

    def policy(selector, actions, puzzle):
        if selector.msg == C.MSG_SELECT_IDLECMD:
            if state["activated"]:
                raise Stop()
            state["activated"] = True
            return next(i for i, a in enumerate(actions) if a.act == ActionAct.ACTIVATE and a.code == code)
        if selector.msg == C.MSG_SELECT_CHAIN:
            return next(i for i, a in enumerate(actions) if a.act == ActionAct.CANCEL)
        return 0
    return policy


def ignition(operation: str, target: str = "nil", flags: str = "0") -> str:
    """Give the trigger card in hand an ignition effect with this operation (and optional target)."""
    return f"""
local mf_e=Effect.CreateEffect(mf_trigger)
mf_e:SetType(EFFECT_TYPE_IGNITION)
mf_e:SetRange(LOCATION_HAND)
mf_e:SetProperty({flags})
if {target} then mf_e:SetTarget({target}) end
mf_e:SetOperation({operation})
mf_trigger:RegisterEffect(mf_e)
"""


def scenario_burst(session: Session) -> dict:
    before, stats_before = session.arena(), session.stats()
    log = session.run_lua("mf_in_call(function() local mz=Duel.GetFieldGroup(0,LOCATION_MZONE,0) "
                          "for i=1,100000 do local g=Group.FromCards(mz:GetFirst())+Group.CreateGroup() "
                          "local h=g:Filter(aux.TRUE,nil) end end)")
    return {"log": log, "arena_growth": session.arena() - before, "before": stats_before, "after": session.stats(),
            "audit": session.audit()}


FORCED_GC = """
function mf_gc() mf_churn(200) Debug.CollectGarbage() end
local function collected() local _,_,_,n,p=Debug.GroupStats() return n+p end
function mf_check(name, f)
  local before=collected()
  local ok,err=pcall(f)
  Debug.Message("MFGC "..name.." "..tostring(ok).." "..(collected()-before).." "..tostring(err))
end
function mf_gcfilter(c) mf_gc() return true end
function mf_gcvalue(c) mf_gc() return 1 end
"""

NON_INTERACTIVE = """mf_in_call(function()
  local mz=Duel.GetFieldGroup(0,LOCATION_MZONE,0)
  local n=mz:GetCount()
  mf_check("group_filter", function()
    local r=mz:Filter(mf_gcfilter,Group.FromCards(mz:GetFirst())) assert(r:GetCount()==n-1) end)
  mf_check("group_get_min_group", function()
    local r,v=mz:GetMinGroup(function(c) mf_gcfilter(c) return c:GetLevel() end) assert(r:GetCount()==1 and v==2) end)
  mf_check("group_get_max_group", function()
    local r,v=mz:GetMaxGroup(function(c) mf_gcfilter(c) return c:GetLevel() end) assert(r:GetCount()==2 and v==4) end)
  mf_check("duel_get_matching_group", function()
    local r=Duel.GetMatchingGroup(mf_gcfilter,0,LOCATION_MZONE,0,Group.FromCards(mz:GetFirst())) assert(r:GetCount()==n-1) end)
  mf_check("duel_get_matching_count", function()
    assert(Duel.GetMatchingGroupCount(mf_gcfilter,0,LOCATION_MZONE,0,Group.FromCards(mz:GetFirst()))==n-1) end)
  local gaia=Duel.GetFirstMatchingCard(Card.IsCode,0,LOCATION_EXTRA,0,nil,%(gaia)d)
  local pearl=Duel.GetFirstMatchingCard(Card.IsCode,0,LOCATION_EXTRA,0,nil,%(pearl)d)
  local spider=Duel.GetFirstMatchingCard(Card.IsCode,0,LOCATION_EXTRA,0,nil,%(spider)d)
  local function wrapped(tbl, key, f)
    local real=tbl[key] tbl[key]=function(...) mf_gc() return real(...) end
    local ok,err=pcall(f) tbl[key]=real assert(ok,err)
  end
  mf_check("card_is_synchro_summonable", function()
    wrapped(Card, "IsNotTuner", function() assert(gaia:IsSynchroSummonable(nil,mz:Filter(aux.TRUE,nil))) end) end)
  mf_check("card_is_xyz_summonable", function()
    wrapped(Duel, "CheckXyzMaterial", function()
      assert(pearl:IsXyzSummonable(mz:Filter(Card.IsLevel,nil,4))) end) end)
  mf_check("card_is_link_summonable", function()
    wrapped(Card, "IsCanBeLinkMaterial", function()
      assert(spider:IsLinkSummonable(Group.FromCards(mz:GetFirst()))) end) end)
end)
""" % {"gaia": GAIA_KNIGHT, "pearl": GEM_KNIGHT_PEARL, "spider": LINK_SPIDER}

INTERACTIVE = """function(e,tp)
  local mz=Duel.GetFieldGroup(tp,LOCATION_MZONE,0)
  local n=mz:GetCount()
  mf_check("duel_get_target_count", function()
    assert(Duel.GetTargetCount(mf_gcfilter,tp,LOCATION_MZONE,0,Group.FromCards(mz:GetFirst()))==n-1) end)
  mf_check("duel_select_matching_cards", function()
    local r=Duel.SelectMatchingCard(tp,mf_gcfilter,tp,LOCATION_MZONE,0,1,1,Group.FromCards(mz:GetFirst()))
    assert(r:GetCount()==1 and not r:IsContains(mz:GetFirst())) end)
  mf_check("group_filter_select", function()
    local r=mz:FilterSelect(tp,mf_gcfilter,1,1,Group.FromCards(mz:GetFirst()))
    assert(r:GetCount()==1 and not r:IsContains(mz:GetFirst())) end)
  mf_check("group_select_with_sum_equal", function()
    local r=mz:Filter(aux.TRUE,nil):SelectWithSumEqual(tp,mf_gcvalue,1,1,1) assert(r:GetCount()==1) end)
  mf_check("group_select_with_sum_greater", function()
    local r=mz:Filter(aux.TRUE,nil):SelectWithSumGreater(tp,mf_gcvalue,1) assert(r:GetCount()==1) end)
  local gaia=Duel.GetFirstMatchingCard(Card.IsCode,tp,LOCATION_EXTRA,0,nil,%(gaia)d)
  mf_check("duel_select_synchro_material", function()
    local r=Duel.SelectSynchroMaterial(tp,gaia,nil,aux.NonTuner(nil),1,5,nil,mz:Filter(aux.TRUE,nil))
    Debug.CollectGarbage() assert(r:GetCount()==2, r:GetCount()) end)
  mf_check("duel_select_tuner_material", function()
    local tuner=mz:Filter(Card.IsType,nil,TYPE_TUNER):GetFirst()
    local r=Duel.SelectTunerMaterial(tp,gaia,tuner,nil,aux.NonTuner(nil),1,5,mz:Filter(aux.TRUE,nil))
    Debug.CollectGarbage() assert(r:GetCount()==2, r:GetCount()) end)
  mf_check("duel_discard_hand", function()
    assert(Duel.DiscardHand(tp,mf_gcfilter,1,1,REASON_EFFECT+REASON_DISCARD,Group.FromCards(e:GetHandler()))==1) end)
  mf_check("duel_sets", function()
    local spells=Duel.GetMatchingGroup(Card.IsType,tp,LOCATION_HAND,0,nil,TYPE_SPELL)
    assert(Duel.SSet(tp,spells:Filter(aux.TRUE,nil))==spells:GetCount()) end)
end""" % {"gaia": GAIA_KNIGHT}

TARGETING = """function(e,tp,eg,ep,ev,re,r,rp,chk)
  if chk==0 then return true end
  local mz=Duel.GetFieldGroup(tp,LOCATION_MZONE,0)
  mf_check("duel_select_target", function()
    local r=Duel.SelectTarget(tp,mf_gcfilter,tp,LOCATION_MZONE,0,1,1,Group.FromCards(mz:GetFirst()))
    assert(r:GetCount()==1 and not r:IsContains(mz:GetFirst())) end)
end"""


def parse_checks(log: list[str]) -> dict:
    checks = {}
    for line in log:
        if line.startswith("MFGC "):
            _, name, ok, collected, err = line.split(" ", 4)
            checks[name] = {"ok": ok == "true", "collected": int(collected), "error": None if ok == "true" else err}
    return checks


def scenario_forced_gc(session: Session) -> dict:
    session.run_lua(FORCED_GC)
    log = session.run_lua(NON_INTERACTIVE)
    before = len(session.core.log)
    session.drive(activate_once(KOJIKOCY))
    log += session.core.log[before:]
    return {"checks": parse_checks(log), "other_log": [line for line in log if not line.startswith("MFGC ")],
            "audit": session.audit()}


HOLDS = """mf_in_call(function()
  local mz=Duel.GetFieldGroup(0,LOCATION_MZONE,0)
  local n=mz:GetCount()
  local function gc() mf_churn(200) Debug.CollectGarbage() end
  local e=Effect.CreateEffect(mz:GetFirst())
  e:SetLabelObject(mz:Filter(aux.TRUE,nil))
  gc()
  assert(e:GetLabelObject():GetCount()==n, "the label object's group did not survive")
  local v,w=Debug.GroupAudit() assert(v==0, w)
  mf_kept=mz:Filter(aux.TRUE,nil) mf_kept:KeepAlive()
  mf_deleted=mz:Filter(aux.TRUE,nil) mf_deleted:KeepAlive() mf_deleted:DeleteGroup()
  gc()
  local t=mz:Filter(aux.TRUE,nil)
  Group.__gc(t) Group.__gc(mz:GetFirst()) Group.__gc(e) Group.__gc(1)
  gc()
  assert(t:GetCount()==n, "a script calling Group.__gc freed a live group")
  local _,_,weak_before=Debug.GroupStats()
  local u=mz:Filter(aux.TRUE,nil)
  local _,_,weak_after=Debug.GroupStats()
  assert(weak_after==weak_before+1, "a library result did not turn weak")
  Debug.KeepGroupUnheld(u)
  mf_violations,mf_what=Debug.GroupAudit()
  Debug.KeepGroupUnheld(nil)
  v,w=Debug.GroupAudit() assert(v==0, w)
end)
assert(mf_violations==1, "the audit missed a weak group kept without a hold: "..mf_violations)
assert(mf_what:find("core.limit_syn",1,true), mf_what)
assert(mf_kept:GetCount()==3, "a kept-alive group did not outlive its call")
"""


def scenario_holds(session: Session) -> dict:
    log = session.run_lua(HOLDS)
    return {"log": log, "audit": session.audit(), "stats": session.stats()}


def scenario_released_group(session: Session) -> dict:
    session.run_lua("mf_in_call(function() mf_stale=Group.CreateGroup() end)")
    print(json.dumps({"reached": "before the stale use"}), flush=True)
    session.run_lua("mf_in_call(function() local n=mf_stale:GetCount() end)")
    return {"reached": "after the stale use"}


REDIRECT = f"""
local mf_victim=Duel.GetFirstMatchingCard(Card.IsCode,0,LOCATION_MZONE,0,nil,{URABY})
local mf_r=Effect.CreateEffect(mf_victim)
mf_r:SetType(EFFECT_TYPE_SINGLE)
mf_r:SetCode(EFFECT_TO_GRAVE_REDIRECT_CB)
mf_r:SetProperty(EFFECT_FLAG_UNCOPYABLE)
mf_r:SetCondition(function(e)
  local c=e:GetHandler() return c:IsFaceup() and c:IsLocation(LOCATION_MZONE) and c:IsReason(REASON_DESTROY)
end)
mf_r:SetOperation(function(e,tp,eg,ep,ev,re,r,rp)
  Debug.Message("MFEG "..tostring(eg and eg:GetCount()).." "..tostring(eg and eg:IsContains(e:GetHandler())))
end)
mf_victim:RegisterEffect(mf_r)
""" + ignition("function(e,tp) Duel.Destroy(Duel.GetFirstMatchingCard(Card.IsCode,tp,LOCATION_MZONE,0,nil,%d),"
               "REASON_EFFECT) end" % URABY)


def scenario_redirect_event(session: Session) -> dict:
    session.drive(activate_once(KOJIKOCY))
    zone = [c.code for c in session.puzzle.zone(0, C.LOCATION_SZONE) if c]
    return {"log": session.core.log, "szone": zone, "audit": session.audit()}


GC_SEQUENCE = """function(e,tp)
  local c=e:GetHandler()
  local mz=Duel.GetFieldGroup(tp,LOCATION_MZONE,0)
  mf_keep={}
  for i=1,30000 do
    local g=mz:Filter(aux.TRUE,nil)+Group.FromCards(c)
    if i%300==0 then mf_keep[#mf_keep+1]=g end
  end
  local sel=Duel.SelectMatchingCard(tp,aux.TRUE,tp,LOCATION_MZONE,0,1,1,nil)
  for i=1,30000 do local g=(mz-sel)+Group.CreateGroup() end
  for _,g in ipairs(mf_keep) do assert(g:GetCount()==mz:GetCount()+1, "a kept group changed") end
  Duel.SendtoGrave(sel,REASON_EFFECT)
end"""


def scenario_gc_sequence(session: Session) -> dict:
    policy = activate_once(KOJIKOCY)
    head, prompt = session.drive(policy, stop_before=C.MSG_SELECT_CARD)
    assert prompt is not None, "the effect never asked for its card"
    snap = session.lib.duel_snapshot(session.pduel)
    at_snapshot = session.stats()
    try:
        session.puzzle._answer(prompt, policy)
        straight, _ = session.drive(policy)
        assert session.lib.duel_rollback(session.pduel, snap) == 0
        after_rollback = session.stats()
        session.puzzle._answer(prompt, policy)  # the same policy: it activated once and stops at the next menu
        replayed, _ = session.drive(policy)
    finally:
        session.lib.duel_snapshot_free(snap)
    return {"head": head, "at_snapshot": at_snapshot, "after_rollback": after_rollback, "straight": straight,
            "replayed": replayed, "log": session.core.log, "audit": session.audit()}


SCENARIOS = {
    "burst": (scenario_burst, ""),
    "forced_gc": (scenario_forced_gc, ignition(INTERACTIVE, TARGETING, "EFFECT_FLAG_CARD_TARGET")),
    "holds": (scenario_holds, ""),
    "released_group": (scenario_released_group, ""),
    "redirect_event": (scenario_redirect_event, REDIRECT),
    "gc_sequence": (scenario_gc_sequence, ignition(GC_SEQUENCE)),
}


def main(argv):
    name, core_path = argv[1], Path(argv[2])
    noise_pages = int(argv[3]) if len(argv) > 3 else 0
    # mapped pages before the duel's arena move where it lands: another process, another layout
    kept = mmap.mmap(-1, 4096 * noise_pages) if noise_pages else None
    run, extra = SCENARIOS[name]
    with tempfile.TemporaryDirectory() as workdir:
        session = Session(core_path, Path(workdir), extra)
        try:
            result = run(session)
        finally:
            session.puzzle.close()
    del kept
    print(json.dumps({"scenario": name, **result}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
