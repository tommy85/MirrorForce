"""Native counterexamples for late identity replacement in a public-flow core.

These fixtures are an offline audit, not a deployment synchronization backend.
MST/Cosmic Cyclone select an unknown set card through their real scripts.
Moving a different card object into that slot after selection is not in-place
identity hydration: the chain still owns the selected original object.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from mirrorforce.effectinfo import DEFAULT_EFFECTINFO_LIB, get_effectinfo_core
from mirrorforce.netduel import constants as C
from mirrorforce.netduel.actions import ActionAct
from mirrorforce.puzzle.core import DEFAULT_DB, DEFAULT_SCRIPTS
from mirrorforce.puzzle.single import SinglePuzzle
from mirrorforce.search.snap_engine import snapshot_api

PRIVILEGED_TARGET = True
MST, COSMIC, OLD_SET, REVEALED_SET, VANILLA = 5318639, 8267140, 63166095, 99550630, 21615956
ENGINE_READY = (Path(DEFAULT_EFFECTINFO_LIB).is_file() and Path(DEFAULT_DB).is_file()
                and Path(DEFAULT_SCRIPTS).is_dir())


class _Complete(Exception):
    pass


class _RecordedPuzzle(SinglePuzzle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = []

    def _observe(self, message):
        super()._observe(message)
        self.messages.append((message.msg, bytes(message.payload)))


def _lua(puzzle, source: str) -> None:
    raw = source.encode()
    name = "./script/mf-blank-sync-contract.lua"
    puzzle.core._script_cache[name] = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
    before = len(puzzle.core.log)
    assert puzzle.core.preload_script(puzzle.pduel, name), puzzle.core.log[before:]
    assert puzzle.core.log[before:] == []


def _permute(puzzle, *, after_target: bool) -> None:
    """Swap the two real identities; optionally inspect the live chain's pointer."""
    prefix = """
local tg=Duel.GetChainInfo(1,CHAININFO_TARGET_CARDS)
assert(tg and #tg==1)
mf_original_target=tg:GetFirst()
assert(mf_original_target==Duel.GetFieldCard(1,LOCATION_SZONE,0))
""" if after_target else ""
    suffix = """
assert(tg:GetFirst()==mf_original_target)
assert(mf_original_target:IsLocation(LOCATION_DECK))
assert(tg:GetFirst()~=Duel.GetFieldCard(1,LOCATION_SZONE,0))
""" if after_target else ""
    _lua(puzzle, prefix + f"""
local deck={{}}
for i=0,Duel.GetFieldGroupCount(1,LOCATION_DECK,0)-1 do
  local c=Duel.GetFieldCard(1,LOCATION_DECK,i)
  deck[#deck+1]=c:GetOriginalCode()=={REVEALED_SET} and {OLD_SET} or c:GetOriginalCode()
end
assert(Debug.PermuteHidden(1,{{}},deck,{{LOCATION_SZONE,0,{REVEALED_SET}}}))
assert(Duel.GetFieldCard(1,LOCATION_SZONE,0):IsCode({REVEALED_SET}))
""" + suffix)


def _run_target_case(tmp_path: Path, timing: str, source_code: int, destination: int):
    path = tmp_path / ("target-" + timing + ".lua")
    path.write_text(f"""
Debug.SetPlayerInfo(0,8000,0,0)
Debug.SetPlayerInfo(1,8000,0,0)
Debug.AddCard({source_code},0,0,LOCATION_HAND,0,POS_FACEDOWN)
Debug.AddCard({source_code},0,0,LOCATION_HAND,1,POS_FACEDOWN)
Debug.AddCard({OLD_SET},1,1,LOCATION_SZONE,0,POS_FACEDOWN_DEFENSE)
Debug.AddCard({REVEALED_SET},1,1,LOCATION_DECK,0,POS_FACEDOWN_DEFENSE)
for i=1,10 do
  Debug.AddCard({VANILLA},0,0,LOCATION_DECK,0,POS_FACEDOWN_DEFENSE)
  Debug.AddCard({VANILLA},1,1,LOCATION_DECK,0,POS_FACEDOWN_DEFENSE)
end
Debug.ReloadFieldEnd()
""", encoding="utf-8")
    core = get_effectinfo_core()
    with _RecordedPuzzle(path, core=core, options=5 << 16) as puzzle:
        assert puzzle.run_result.loaded and not core.log
        changed, activated = False, False
        if timing == "before_target":
            _permute(puzzle, after_target=False)
            changed = True

        def choose(selector, actions, live):
            nonlocal activated, changed
            if selector.msg == C.MSG_SELECT_IDLECMD:
                if activated:
                    raise _Complete()
                index = next(i for i, action in enumerate(actions)
                             if action.act == ActionAct.ACTIVATE and action.code == source_code)
                activated = True
                return index
            if selector.msg == C.MSG_SELECT_CARD:
                return next(i for i, action in enumerate(actions) if action.code in (OLD_SET, REVEALED_SET))
            if selector.msg == C.MSG_SELECT_CHAIN:
                if timing in ("after_target", "after_target_rollback") and not changed and activated:
                    api = snapshot_api(core)
                    saved = api.duel_snapshot(live.pduel) if timing == "after_target_rollback" else None
                    if timing == "after_target_rollback":
                        assert saved
                    try:
                        _permute(live, after_target=True)
                    finally:
                        if saved:
                            try:
                                assert api.duel_rollback(live.pduel, saved) == 0
                            finally:
                                api.duel_snapshot_free(saved)
                    changed = True
                return next(i for i, action in enumerate(actions) if action.describe() == "cancel")
            return 0

        with pytest.raises(_Complete):
            puzzle.play(choose, max_steps=1000)
        assert activated and changed == (timing != "control")
        assert any(msg == C.MSG_CHAINING for msg, _ in puzzle.messages)
        assert any(msg == C.MSG_CHAIN_SOLVED for msg, _ in puzzle.messages)
        _lua(puzzle, f"""
local moved=Duel.GetFieldGroup(1,{destination},0)
assert(#moved==1)
assert(moved:GetFirst():IsCode({REVEALED_SET if timing == 'before_target' else OLD_SET}))
""")
        if timing == "after_target":
            _lua(puzzle, f"assert(Duel.GetFieldCard(1,LOCATION_SZONE,0):IsCode({REVEALED_SET}))")
        else:
            _lua(puzzle, "assert(Duel.GetFieldCard(1,LOCATION_SZONE,0)==nil)")
        return puzzle.messages


@pytest.mark.skipif(not ENGINE_READY, reason="needs the pinned effectinfo core, database and scripts")
@pytest.mark.parametrize("source_code,destination", [(MST, C.LOCATION_GRAVE), (COSMIC, C.LOCATION_REMOVED)])
def test_replacing_a_live_chain_target_by_permutation_does_not_hydrate_its_identity(tmp_path, source_code, destination):
    """Real script resolution follows the old card into the deck after a late swap.

    The before-target control removes the newly assigned set card, proving
    that the identities are individually executable. The after-target case
    leaves that set card intact and removes a card out of the deck instead.
    This is a counterexample to unsafe swap-based hydration, not a regression
    claim about ClientSync (which currently refuses/does not handle this case).
    """
    control = _run_target_case(tmp_path, "control", source_code, destination)
    before = _run_target_case(tmp_path, "before_target", source_code, destination)
    after = _run_target_case(tmp_path, "after_target", source_code, destination)
    restored = _run_target_case(tmp_path, "after_target_rollback", source_code, destination)
    assert control != before and before != after
    assert restored == control
    def public_chain(messages):
        return [(msg, body) for msg, body in messages
                if msg in (C.MSG_CHAINING, C.MSG_CHAINED, C.MSG_BECOME_TARGET)]
    assert len(public_chain(control)) >= 3
    assert public_chain(control) == public_chain(before) == public_chain(after)
