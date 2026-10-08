"""Read-only material copies of the summon rules must not outlive their use (ygopro-core 2408c043).

* Duel.CheckMustMaterial copies its card or group into a read-only group that
  the script-group release never frees; the check must delete it itself. The
  unfixed core leaks one group per call (an era-pool gate game exhausted its
  256 MiB duel arena this way).
* Duel.SynchroSummon/XyzSummon/LinkSummon store read-only copies of their
  material limits in the core state. When the summon rule takes the Pendulum
  Summon branch it never reads them; the unfixed core then leaves them in place,
  and the player's next Synchro Summon from the idle menu is silently limited to
  the old material group.

The core under test is the rebuilt Python-engine copy (a new file; the old core
is never replaced); ``MF_REBUILT_PYCORE_LIB`` points at any other build.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.netduel.actions import ActionAct
from mirrorforce.puzzle.core import DEFAULT_DB, DEFAULT_SCRIPTS, Core
from mirrorforce.puzzle.single import SinglePuzzle
from mirrorforce.search.snap_engine import snapshot_api

CORE = Path(os.environ.get("MF_REBUILT_PYCORE_LIB", "/path/to/workspace/build-deps/priv-mfenv/core-pycore-2408c043.so"))
URABY, KOJIKOCY, FROSTOSAURUS = 1784619, 1184620, 6631034          # Normal monsters: Level 4, 4, 6
GALAXY_SERPENT, GAIA_KNIGHT = 11066358, 97204936                     # Level 2 Tuner; Level 6 Synchro
CLEAR_WING_FAST_DRAGON, DRAGONPIT_MAGICIAN = 90036274, 51531505      # Synchro Pendulum scale 4; scale 8


@pytest.fixture(scope="module")
def core():
    if not (CORE.is_file() and Path(DEFAULT_DB).is_file() and Path(DEFAULT_SCRIPTS).is_dir()):
        pytest.skip("needs the rebuilt core, the card database and scripts")
    return Core(lib_path=CORE)


def puzzle_file(tmp_path, body):
    path = tmp_path / "summon-groups.lua"
    path.write_text("Debug.ReloadFieldBegin(DUEL_ATTACK_FIRST_TURN+DUEL_SIMPLE_AI,5)\n"
                    "Debug.SetPlayerInfo(0,8000,0,0)\nDebug.SetPlayerInfo(1,8000,0,0)\n" + body, encoding="utf-8")
    return path


def run_lua(puzzle, source):
    name = "./script/mf-summon-groups.lua"
    raw = source.encode()
    puzzle.core._script_cache[name] = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
    before = len(puzzle.core.log)
    assert puzzle.core.preload_script(puzzle.pduel, name) and puzzle.core.log[before:] == []


def test_must_material_checks_free_their_copies(core, tmp_path):
    body = (f"Debug.AddCard({URABY},0,0,LOCATION_MZONE,0,POS_FACEUP_ATTACK)\n"
            f"Debug.AddCard({URABY},1,1,LOCATION_MZONE,0,POS_FACEUP_ATTACK)\nDebug.ReloadFieldEnd()\n")
    with SinglePuzzle(puzzle_file(tmp_path, body), core=core) as puzzle:
        arena = snapshot_api(core).duel_arena_extent
        run_lua(puzzle, "mf_c=Duel.GetFieldCard(0,LOCATION_MZONE,0) mf_g=Group.FromCards(mf_c)")
        before = arena(puzzle.pduel)
        run_lua(puzzle, "for i=1,40000 do "
                        "assert(Duel.CheckMustMaterial(0,mf_c,EFFECT_MUST_BE_SMATERIAL)) "
                        "assert(Duel.CheckMustMaterial(0,mf_g,EFFECT_MUST_BE_XMATERIAL)) end")
        grown = arena(puzzle.pduel) - before
        # 80k leaked groups would take well over 10 MiB; the check itself allocates nothing that survives.
        assert grown < (2 << 20), f"the duel arena grew by {grown} bytes"


class Stop(Exception):
    pass


def test_pendulum_branch_of_a_synchro_summon_releases_its_material_limit(core, tmp_path):
    # A test-only ignition effect calls Duel.SynchroSummon with a one-card material limit on the Synchro
    # Pendulum in the left Pendulum Zone. Its Synchro procedure only works from the Extra Deck, so the
    # summon rule takes the Pendulum Summon branch (Frostosaurus, Level 6, between scales 4 and 8). The
    # player then Synchro Summons Gaia Knight from the idle menu with Galaxy Serpent and Uraby, which
    # the stale limit (the trigger card only) would forbid.
    body = (f"local trig=Debug.AddCard({KOJIKOCY},0,0,LOCATION_HAND,0,POS_FACEDOWN)\n"
            f"Debug.AddCard({FROSTOSAURUS},0,0,LOCATION_HAND,1,POS_FACEDOWN)\n"
            f"Debug.AddCard({CLEAR_WING_FAST_DRAGON},0,0,LOCATION_PZONE,0,POS_FACEUP)\n"
            f"Debug.AddCard({DRAGONPIT_MAGICIAN},0,0,LOCATION_PZONE,1,POS_FACEUP)\n"
            f"Debug.AddCard({GALAXY_SERPENT},0,0,LOCATION_MZONE,0,POS_FACEUP_ATTACK)\n"
            f"Debug.AddCard({URABY},0,0,LOCATION_MZONE,1,POS_FACEUP_ATTACK)\n"
            f"Debug.AddCard({GAIA_KNIGHT},0,0,LOCATION_EXTRA,0,POS_FACEDOWN)\n"
            f"Debug.AddCard({URABY},1,1,LOCATION_MZONE,0,POS_FACEUP_ATTACK)\n"
            "Debug.ReloadFieldEnd()\n"
            "local e=Effect.CreateEffect(trig)\ne:SetType(EFFECT_TYPE_IGNITION)\ne:SetRange(LOCATION_HAND)\n"
            "e:SetOperation(function(e,tp)\n"
            "  Duel.SynchroSummon(tp,Duel.GetFieldCard(tp,LOCATION_PZONE,0),nil,Group.FromCards(e:GetHandler()))\n"
            "end)\ntrig:RegisterEffect(e)\n")
    plan = iter([(ActionAct.ACTIVATE, KOJIKOCY), (ActionAct.SPSUMMON, GAIA_KNIGHT)])

    def policy(selector, actions, puzzle):
        if selector.msg == C.MSG_SELECT_IDLECMD:
            step = next(plan, None)
            if step is None:
                raise Stop()
            return next(i for i, a in enumerate(actions) if (a.act, a.code) == step)
        if selector.msg == C.MSG_SELECT_CHAIN:
            return next(i for i, a in enumerate(actions) if a.act == ActionAct.CANCEL)
        return 0

    with SinglePuzzle(puzzle_file(tmp_path, body), core=core) as puzzle:
        with pytest.raises(Stop):
            puzzle.play(policy=policy, max_steps=400)
        monsters = {card.code for card in puzzle.zone(0, C.LOCATION_MZONE) if card}
    assert FROSTOSAURUS in monsters, "the Pendulum Summon branch did not run"
    assert GAIA_KNIGHT in monsters, "the next Synchro Summon was still limited to the old material group"
