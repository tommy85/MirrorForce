"""Random outcomes the blank follower reads from the received packets and forces in its local core."""

import ctypes
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from mirrorforce.effectinfo import DEFAULT_EFFECTINFO_LIB, get_effectinfo_core
from mirrorforce.netduel import constants as C
from mirrorforce.puzzle.core import DEFAULT_DB, DEFAULT_SCRIPTS
from mirrorforce.search.snap_engine import snapshot_api
from mirrorforce.common.client_sync import ClientSync, _Divergence
from test_blank_engine_sync_contract import _RecordedPuzzle, _lua

PRIVILEGED_TARGET = True
READY = (Path(DEFAULT_EFFECTINFO_LIB).is_file() and Path(DEFAULT_DB).is_file()
         and Path(DEFAULT_SCRIPTS).is_dir())
VANILLA, OTHER = 21615956, 89631139
VIEWER = 0


def move(code, before, after):
    return bytes([C.MSG_MOVE]) + struct.pack("<I", code) + bytes([*before, 0, *after, 0]) + struct.pack("<I", 0)


def selected(player, *places):
    return bytes([C.MSG_RANDOM_SELECTED, player, len(places)]) + b"".join(bytes([*place, 0]) for place in places)


def follower(packets, local_random=()):
    return SimpleNamespace(viewer=VIEWER, packets=list(packets), local_random=list(local_random),
                           _random_selection=lambda *args: ClientSync._random_selection(sync, *args),
                           _code_shown_after=lambda *args: ClientSync._code_shown_after(sync, *args))


sync = None


def evidence(packets, divergence, start=0, forced=None, local_random=()):
    global sync
    sync = follower(packets, local_random)
    return ClientSync._random_evidence(sync, divergence, start, forced)


def test_toss_results_come_from_every_received_toss_of_the_batch_in_order():
    coin = bytes([C.MSG_TOSS_COIN, 1, 2, 1, 0])
    dice = bytes([C.MSG_TOSS_DICE, 0, 1, 5])
    packets = [bytes([C.MSG_WAITING]), coin, dice]
    divergence = _Divergence(2, dice, bytes([C.MSG_TOSS_DICE, 0, 1, 3]))
    assert evidence(packets, divergence) == ((1, 0, 5), ())
    # The same outcome again is no new evidence: the divergence is not a random one.
    assert evidence(packets, divergence, forced=((1, 0, 5), ())) is None
    assert evidence(packets, _Divergence(2, dice, bytes([C.MSG_TOSS_COIN, 0, 1, 1]))) is None


def test_a_received_selection_names_its_card_by_the_code_its_place_shows_next():
    shown = selected(1, (VIEWER, C.LOCATION_HAND, 3))
    local = selected(1, (VIEWER, C.LOCATION_HAND, 1))
    packets = [shown, bytes([C.MSG_WAITING]), move(VANILLA, (VIEWER, C.LOCATION_HAND, 3), (VIEWER, C.LOCATION_GRAVE, 0))]
    outcome = evidence(packets, _Divergence(0, shown, local), local_random=[(0, local[1:], True)])
    assert outcome == ((), (((0, VANILLA),),))
    # A card whose identity the stream never shows is named by its place.
    hidden = selected(1, (1, C.LOCATION_HAND, 3))
    outcome = evidence([hidden], _Divergence(0, hidden, selected(1, (1, C.LOCATION_HAND, 0))),
                       local_random=[(0, selected(1, (1, C.LOCATION_HAND, 0))[1:], True)])
    assert outcome == ((), (((1 | C.LOCATION_HAND << 8 | 3 << 16, 0),),))


def test_an_unsent_selection_is_read_from_the_chosen_cards_move_and_earlier_ones_keep_their_local_result():
    earlier = selected(1, (1, C.LOCATION_GRAVE, 4))[1:]
    unsent = selected(1, (VIEWER, C.LOCATION_DECK, 15))[1:]
    real = move(VANILLA, (VIEWER, C.LOCATION_DECK, 32), (VIEWER, C.LOCATION_HAND, 2))
    local = move(OTHER, (VIEWER, C.LOCATION_DECK, 11), (VIEWER, C.LOCATION_HAND, 2))
    outcome = evidence([real], _Divergence(0, real, local), local_random=[(0, earlier, True), (0, unsent, False)])
    assert outcome == ((), (((1 | C.LOCATION_GRAVE << 8 | 4 << 16, 0),), ((0, VANILLA),)))
    # A selection whose packet showed its outcome is compared where it is received, never through a later move.
    assert evidence([real], _Divergence(0, real, local), local_random=[(0, unsent, True)]) is None
    # Moves out of different zones are not one selection coming out differently.
    other_zone = move(OTHER, (VIEWER, C.LOCATION_HAND, 1), (VIEWER, C.LOCATION_GRAVE, 0))
    assert evidence([real], _Divergence(0, real, other_zone), local_random=[(0, unsent, False)]) is None
    assert evidence([real], _Divergence(0, real, local)) is None


@pytest.fixture
def core():
    if not READY:
        pytest.skip("needs native core, database and scripts")
    result = get_effectinfo_core()
    if not hasattr(result._lib, "duel_force_random_select"):
        pytest.skip("needs the core's forced random outcomes")
    return result


def test_the_core_takes_a_forced_selection_by_code_or_place_and_rolls_it_back(core, tmp_path):
    path = tmp_path / "forced.lua"
    path.write_text(f"""
Debug.SetPlayerInfo(0,8000,0,0)
Debug.SetPlayerInfo(1,8000,0,0)
for i=1,6 do Debug.AddCard({VANILLA},0,0,LOCATION_DECK,0,POS_FACEDOWN_DEFENSE) end
Debug.AddCard({OTHER},0,0,LOCATION_DECK,0,POS_FACEDOWN_DEFENSE)
for i=1,3 do Debug.AddCard({VANILLA},1,1,LOCATION_DECK,0,POS_FACEDOWN_DEFENSE) end
Debug.ReloadFieldEnd()
""", encoding="utf-8")
    lib = core._lib
    pick = "mf_g=Duel.GetFieldGroup(0,LOCATION_DECK,0) mf_s=mf_g:RandomSelect(1,1):GetFirst() " \
           "mf_code=mf_s:GetCode() mf_seq=mf_s:GetSequence()"
    state = (ctypes.c_int32 * 3)()
    with _RecordedPuzzle(path, core=core, options=5 << 16) as puzzle:
        pduel = ctypes.c_void_p(puzzle.pduel)
        api = snapshot_api(core)
        saved = api.duel_snapshot(puzzle.pduel)
        try:
            # By code: the one copy of OTHER, wherever it lies.
            assert lib.duel_force_random_select(pduel, 1, (ctypes.c_uint32 * 1)(0), (ctypes.c_uint32 * 1)(OTHER)) == 0
            _lua(puzzle, pick + f" assert(mf_code=={OTHER})")
            # By place: deck sequence 2.
            place = 0 | C.LOCATION_DECK << 8 | 2 << 16
            assert lib.duel_force_random_select(pduel, 1, (ctypes.c_uint32 * 1)(place), None) == 0
            _lua(puzzle, pick + " assert(mf_seq==2)")
            # A card outside the group is a miss; the core then keeps its own selection.
            outside = 1 | C.LOCATION_DECK << 8 | 0 << 16
            assert lib.duel_force_random_select(pduel, 1, (ctypes.c_uint32 * 1)(outside), None) == 0
            _lua(puzzle, pick)
            assert lib.duel_forced_random_state(pduel, state) == 0 and list(state) == [0, 0, 1]
            # Pending entries are duel state: a rollback takes them back.
            assert lib.duel_force_random_outcomes(pduel, 2, (ctypes.c_uint8 * 2)(1, 0)) == 0
            assert api.duel_rollback(puzzle.pduel, saved) == 0
            assert lib.duel_forced_random_state(pduel, state) == 0 and list(state) == [0, 0, 0]
            assert lib.duel_force_random_outcomes(pduel, 2, (ctypes.c_uint8 * 2)(1, 0)) == 0
            assert lib.duel_clear_forced_random(pduel) == 0
            assert lib.duel_forced_random_state(pduel, state) == 0 and list(state) == [0, 0, 0]
        finally:
            api.duel_snapshot_free(saved)
