"""The core's forced activation (plan 5.11): the local duel takes an activation the opponent made as legal, however
its hidden cards stand; inside that link the opponent's blank placeholders pass the script's card filters."""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from mirrorforce.effectinfo import DEFAULT_EFFECTINFO_LIB, get_effectinfo_core
from mirrorforce.netduel import constants as C
from mirrorforce.netduel.actions import ActionAct
from mirrorforce.puzzle.core import DEFAULT_DB, DEFAULT_SCRIPTS
from mirrorforce.search.snap_engine import snapshot_api
from mirrorforce.common.client_shadow import BLANK_MAIN, install_blanks
from test_blank_engine_sync_contract import _Complete, _lua, _RecordedPuzzle

PRIVILEGED_TARGET = True
READY = (Path(DEFAULT_EFFECTINFO_LIB).is_file() and Path(DEFAULT_DB).is_file()
         and Path(DEFAULT_SCRIPTS).is_dir())
#: Sky Striker Mobilize - Engage!: its activation needs a "Sky Striker" card in the deck to add; Sky Striker Mecha -
#: Widow Anchor: an effect monster on the field to target
ENGAGE, WIDOW_ANCHOR, VANILLA = 63166095, 98338152, 21615956
SKY_STRIKER = "function(c) return c:IsSetCard(0x115) end"


@pytest.fixture
def core():
    if not READY:
        pytest.skip("needs the pinned effectinfo core, database and scripts")
    core = get_effectinfo_core()
    if core._query_card_data(BLANK_MAIN) is None:
        install_blanks(core)
    return core


def _puzzle(tmp_path, core, deck):
    """Engage and Widow Anchor in the first player's hand, ``deck`` as its deck (no "Sky Striker" card in it)."""
    path = tmp_path / "forced.lua"
    path.write_text("Debug.SetPlayerInfo(0,8000,0,0)\nDebug.SetPlayerInfo(1,8000,0,0)\n"
                    + f"Debug.AddCard({ENGAGE},0,0,LOCATION_HAND,0,POS_FACEDOWN)\n"
                    + f"Debug.AddCard({WIDOW_ANCHOR},0,0,LOCATION_HAND,1,POS_FACEDOWN)\n"
                    + "".join(f"Debug.AddCard({code},0,0,LOCATION_DECK,0,POS_FACEDOWN_DEFENSE)\n" for code in deck)
                    + f"for i=1,5 do Debug.AddCard({VANILLA},1,1,LOCATION_DECK,0,POS_FACEDOWN_DEFENSE) end\n"
                    + "Debug.ReloadFieldEnd()\n", encoding="utf-8")
    return _RecordedPuzzle(path, core=core, options=5 << 16)


def _activatable(slot):
    return f"Duel.GetFieldCard(0,LOCATION_HAND,{slot}):GetActivateEffect():IsActivatable(0)"


def _state(lib, pduel):
    state = (ctypes.c_int32 * 2)()
    assert lib.duel_forced_activation_state(pduel, state) == 0
    return list(state)


def _force(lib, pduel, player, code, description=0):
    return lib.duel_force_activation(pduel, player, ctypes.c_uint32(code), ctypes.c_uint32(description))


def test_a_forced_activation_passes_its_script_checks_only_for_the_named_effect_and_rolls_back(core, tmp_path):
    lib = core._lib
    with _puzzle(tmp_path, core, [VANILLA] * 6) as puzzle:
        pduel = ctypes.c_void_p(puzzle.pduel)
        api = snapshot_api(core)
        saved = api.duel_snapshot(puzzle.pduel)
        try:
            # No "Sky Striker" card to add, no monster to target: both target checks fail.
            _lua(puzzle, f"assert(not {_activatable(0)}) assert(not {_activatable(1)})")
            # Named as the stream shows it: another controller, or another description, names nothing here.
            assert _force(lib, pduel, 1, ENGAGE) == 0 and _force(lib, pduel, 0, ENGAGE, 1) == 0
            _lua(puzzle, f"assert(not {_activatable(0)})")
            # The controller, the code and the description (Engage's activation has none): that effect only.
            assert _force(lib, pduel, 0, ENGAGE) == 0
            _lua(puzzle, f"assert({_activatable(0)}) assert(not {_activatable(1)})")
            assert _state(lib, pduel) == [3, 0]
            # The same activation named again is still one mark.
            assert _force(lib, pduel, 0, ENGAGE) == 0 and _state(lib, pduel) == [3, 0]
            # Duel state: a rollback takes the pending forces back.
            assert api.duel_rollback(puzzle.pduel, saved) == 0
            assert _state(lib, pduel) == [0, 0]
            _lua(puzzle, f"assert(not {_activatable(0)})")
            assert _force(lib, pduel, 0, WIDOW_ANCHOR) == 0
            _lua(puzzle, f"assert({_activatable(1)})")
            assert lib.duel_clear_forced_activations(pduel) == 1 and _state(lib, pduel) == [0, 0]
            _lua(puzzle, f"assert(not {_activatable(1)})")
            assert _force(lib, 0, 0, ENGAGE) == -1 and _force(lib, pduel, 2, ENGAGE) == -1
        finally:
            api.duel_snapshot_free(saved)


def test_a_forced_link_lets_its_players_blanks_pass_the_scripts_filters_until_it_is_solved(core, tmp_path):
    lib = core._lib
    with _puzzle(tmp_path, core, [VANILLA] * 4 + [BLANK_MAIN] * 2) as puzzle:
        assert puzzle.run_result.loaded and not core.log
        pduel = ctypes.c_void_p(puzzle.pduel)
        # Outside a forced link a blank is what it is: no "Sky Striker" card in the deck.
        _lua(puzzle, f"assert(not Duel.IsExistingMatchingCard({SKY_STRIKER},0,LOCATION_DECK,0,1,nil))")
        assert _force(lib, pduel, 0, ENGAGE) == 0
        seen = {"activated": False, "offered": None, "linked": None}

        def choose(selector, actions, live):
            if selector.msg == C.MSG_SELECT_IDLECMD:
                if seen["activated"]:
                    raise _Complete()
                seen["activated"] = True
                return next(i for i, action in enumerate(actions)
                            if action.act == ActionAct.ACTIVATE and action.code == ENGAGE)
            if selector.msg == C.MSG_SELECT_CHAIN:
                return next(i for i, action in enumerate(actions) if action.describe() == "cancel")
            if selector.msg == C.MSG_SELECT_CARD:
                # The search's own filter, inside the forced link (chained: no longer pending): the deck's blanks
                # stand for the card taken.
                seen["linked"] = _state(lib, pduel)
                seen["offered"] = sorted({action.code for action in actions if action.code})
                return next(i for i, action in enumerate(actions) if action.code == BLANK_MAIN)
            return 0

        with pytest.raises(_Complete):
            puzzle.play(choose, max_steps=1000)
        assert seen == {"activated": True, "offered": [BLANK_MAIN], "linked": [0, 1]}
        assert any(msg == C.MSG_CHAIN_SOLVED for msg, _ in puzzle.messages)
        # The link is solved: the blank still in the deck is a blank again, and the one taken is in the hand.
        assert _state(lib, pduel) == [0, 0]
        _lua(puzzle, f"assert(not Duel.IsExistingMatchingCard({SKY_STRIKER},0,LOCATION_DECK,0,1,nil))"
                     f" assert(Duel.IsExistingMatchingCard(Card.IsCode,0,LOCATION_HAND,0,1,nil,{BLANK_MAIN}))")
