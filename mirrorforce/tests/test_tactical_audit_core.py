"""Actual native audit ABI, covered reads, unchanged outcomes and rollback.

Run with MF_EFFECTINFO_LIB naming the independent tactical-audit core build.
Only these constructed native fixtures use unmasked host handles.
"""
import ctypes
import hashlib

import pytest

from mirrorforce.effectinfo import get_effectinfo_core
from mirrorforce.netduel import constants as C
from mirrorforce.search.snap_engine import snapshot_api
from mirrorforce.common.client_entity_map import _read

PRIVILEGED_TARGET = True


@pytest.fixture
def fixture():
    core = get_effectinfo_core()
    lib = core._lib
    if not hasattr(lib,'duel_tactical_audit_begin'):
        pytest.skip('set MF_EFFECTINFO_LIB to the independent tactical-audit core')
    lib.duel_tactical_audit_begin.argtypes = [ctypes.c_ssize_t,ctypes.POINTER(ctypes.c_uint64),ctypes.c_int32]
    lib.duel_tactical_audit_begin.restype = ctypes.c_int32
    lib.duel_tactical_audit_state.argtypes = [ctypes.c_ssize_t,ctypes.POINTER(ctypes.c_uint64),ctypes.c_int32]
    lib.duel_tactical_audit_state.restype = ctypes.c_int32
    lib.duel_tactical_audit_end.argtypes = [ctypes.c_ssize_t]
    lib.duel_tactical_audit_end.restype = ctypes.c_int32
    core.reset_session()
    duel = core.create_duel([731,3,5,7,11,13,17,19])
    assert duel
    for player in (0,1):
        core.set_player_info(duel,player,8000,0,0)
    for sequence in (0,1):
        core.new_card(duel,48305365,0,0,C.LOCATION_MZONE,sequence,C.POS_FACEUP_ATTACK)
        core.new_card(duel,21615956,1,1,C.LOCATION_DECK,sequence,C.POS_FACEDOWN_DEFENSE)
    assert not core.log
    unknown = tuple(row.uid for row in _read(lib,duel) if row.controller == 1 and row.location == C.LOCATION_DECK)
    assert len(unknown) == 2
    try:
        yield core,lib,duel,unknown
    finally:
        core.end_duel(duel)


def state(lib,duel):
    out = (ctypes.c_uint64*8)()
    assert lib.duel_tactical_audit_state(duel,out,8) == 0
    return tuple(out)


def begin(lib,duel,uids):
    data = (ctypes.c_uint64*len(uids))(*uids)
    return lib.duel_tactical_audit_begin(duel,data,len(uids))


def lua(core,duel,text):
    raw = text.encode()
    key = './script/tactical-audit-fixture-'+hashlib.sha256(raw).hexdigest()+'.lua'
    core._script_cache[key] = (ctypes.c_ubyte*len(raw)).from_buffer_copy(raw)
    assert core.preload_script(duel,key) == 1
    assert not core.log


def messages(core,duel):
    buf = ctypes.create_string_buffer(4<<20)
    length = core.get_message(duel,buf)
    return bytes(buf.raw[:length])


def test_default_is_dormant_and_reads_do_not_enable_it(fixture):
    core,lib,duel,_ = fixture
    assert state(lib,duel) == (1,0,0,0,0,0,0,0)
    lua(core,duel,'local c=Duel.GetDecktopGroup(1,1):GetFirst(); assert(c:GetCode()==21615956)')
    assert state(lib,duel) == (1,0,0,0,0,0,0,0)


@pytest.mark.parametrize('bad', ['zero','duplicate','nonlive','too_many'])
def test_invalid_begin_is_atomic_and_does_not_change_existing_counters(fixture,bad):
    _,lib,duel,uids = fixture
    before = state(lib,duel)
    values = {'zero':[0],'duplicate':[uids[0],uids[0]],'nonlive':[2**63],'too_many':list(range(1,514))}[bad]
    assert begin(lib,duel,values) == (-4 if bad == 'nonlive' else -2)
    assert state(lib,duel) == before


def test_unlisted_cards_are_not_mislabeled_as_unknown_reads(fixture):
    core,lib,duel,uids = fixture
    assert begin(lib,duel,list(reversed(uids))) == 0
    lua(core,duel,'local g=Duel.GetFieldGroup(0,LOCATION_MZONE,0); assert(g:GetCount()==2); assert(g:GetFirst():GetCode()==48305365)')
    assert state(lib,duel) == (1,1,2,0,0,0,0,0)
    before = state(lib,duel)
    assert begin(lib,duel,[]) == -3
    assert state(lib,duel) == before


def test_lua_unknown_card_and_group_reads_are_recorded_without_revealing_ids(fixture):
    core,lib,duel,uids = fixture
    assert begin(lib,duel,uids) == 0
    lua(core,duel,'local c=Duel.GetDecktopGroup(1,1):GetFirst(); assert(c:GetCode()==21615956)')
    values = state(lib,duel)
    assert values[:3] == (1,1,2)
    assert values[3] > 0 and values[4] > 0 and values[6] > 0 and values[7] == 0
    assert lib.duel_tactical_audit_end(duel) == 0
    ended = state(lib,duel)
    assert ended == (1,0,*values[2:])
    assert lib.duel_tactical_audit_end(duel) == -3
    assert state(lib,duel) == ended


def test_random_use_is_recorded_even_with_no_unknown_cards(fixture):
    core,lib,duel,_ = fixture
    assert begin(lib,duel,[]) == 0
    lua(core,duel,'local g=Duel.GetFieldGroup(0,LOCATION_MZONE,0); assert(g:RandomSelect(0,1):GetCount()==1)')
    assert state(lib,duel)[:7] == (1,1,0,0,0,0,0)
    assert state(lib,duel)[7] > 0


def test_effect_metadata_keeps_its_unknown_owner_dependency(fixture):
    core,lib,duel,uids = fixture
    assert begin(lib,duel,uids) == 0
    lua(core,duel,'local c=Duel.GetDecktopGroup(1,1):GetFirst(); local e=Effect.CreateEffect(c); e:SetType(EFFECT_TYPE_SINGLE); assert(e:GetType()==EFFECT_TYPE_SINGLE)')
    assert state(lib,duel)[5] > 0


def test_native_identity_query_is_covered_even_without_lua(fixture):
    core,lib,duel,uids = fixture
    assert begin(lib,duel,uids) == 0
    buf = ctypes.create_string_buffer(1<<20)
    assert core.query_field_card(duel,1,C.LOCATION_DECK,C.QUERY_CODE,buf) > 0
    after = state(lib,duel)
    assert after[3:6] == (0,0,0) and after[6] > 0 and after[7] == 0


def test_snapshot_restores_audit_configuration_counters_and_natural_random_result(fixture):
    core,lib,duel,uids = fixture
    api = snapshot_api(core)
    assert begin(lib,duel,uids) == 0
    baseline = state(lib,duel)
    # Remove construction messages before taking the native snapshot.
    messages(core,duel)
    saved = api.duel_snapshot(duel)
    assert saved
    body = 'local g=Duel.GetFieldGroup(0,LOCATION_MZONE,0); local r=g:RandomSelect(0,1); assert(r:GetCount()==1)'
    try:
        lua(core,duel,body)
        first = messages(core,duel)
        first_state = state(lib,duel)
        assert first_state[7] > 0
        assert api.duel_rollback(duel,saved) == 0
        assert state(lib,duel) == baseline
        # Disable recording only; the same engine draws and emits the same bytes.
        assert lib.duel_tactical_audit_end(duel) == 0
        lua(core,duel,body)
        assert messages(core,duel) == first
        assert state(lib,duel)[3:] == (0,0,0,0,0)
    finally:
        assert api.duel_rollback(duel,saved) == 0
        api.duel_snapshot_free(saved)
    assert state(lib,duel) == baseline


def test_state_is_read_only_and_bad_buffers_leave_output_untouched(fixture):
    _,lib,duel,uids = fixture
    assert begin(lib,duel,uids) == 0
    before = state(lib,duel)
    out = (ctypes.c_uint64*8)(*([923]*8))
    assert lib.duel_tactical_audit_state(duel,out,7) == -2
    assert list(out) == [923]*8
    assert lib.duel_tactical_audit_state(0,out,8) == -1
    assert list(out) == [923]*8
    assert state(lib,duel) == before
