"""Public recipe copy/provenance contracts; mocked native reads are not a real-game proof."""
import ctypes
import struct
from types import SimpleNamespace as NS

import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.netduel.agent_public_recipe import declare
from mirrorforce.common import client_public_recipe as P
from mirrorforce.common.client_shadow import BLANK_MAIN, BLANK_EXTRA, BlankClientSync


def recipe():
    return P.PublicRecipe.checked(declare([10, 10, 20], [30]))


def row(uid, code, owner=1, provenance=None, token=False):
    return (uid, owner, code, code if provenance is None else provenance, token)


def test_declaration_is_order_free_but_retains_all_copies():
    a=P.PublicRecipe.checked(declare([20,10,10],[30]))
    assert a==recipe() and a.main==(10,10,20) and a.remaining((),opponent=1)=={10:2,20:1,30:1}


@pytest.mark.parametrize('main,extra',[([10]*4,[]),([10],[10])])
def test_bad_copy_or_home_zone_declaration_rejected(main,extra):
    with pytest.raises(ValueError):P.PublicRecipe.checked(declare(main,extra))


def test_remaining_counts_original_owner_once_and_excludes_tokens():
    rows=(row(1,10),row(2,10,owner=0),row(3,999,token=True))
    assert recipe().remaining(rows,opponent=1)=={10:1,20:1,30:1}
    with pytest.raises(P.PublicRecipeError,match='duplicate'):
        recipe().remaining(rows+(rows[0],),opponent=1)


def test_repeated_hydrate_of_same_physical_card_does_not_consume_another_copy():
    rows=(row(1,10),row(2,10),row(3,BLANK_MAIN))
    assert recipe().permits(rows,opponent=1,target_uid=1,code=10,location=C.LOCATION_HAND)
    assert not recipe().permits(rows,opponent=1,target_uid=3,code=10,location=C.LOCATION_DECK)
    assert not recipe().permits(rows,opponent=1,target_uid=3,code=30,location=C.LOCATION_HAND)
    assert recipe().permits(rows,opponent=1,target_uid=3,code=30,location=C.LOCATION_EXTRA)


@pytest.mark.parametrize('rows',[(row(1,10),row(2,10),row(3,10)),(row(1,20,provenance=10),),
                                (row(1,99),),(row(1,10,provenance=0),)])
def test_exhausted_or_unproved_recreated_identities_rejected(rows):
    with pytest.raises(P.PublicRecipeError):recipe().remaining(rows,opponent=1)


def test_restore_recomputes_copy_budget_without_persistent_mutable_inventory():
    original=(row(1,10),row(2,BLANK_MAIN))
    hypothesis=(row(1,10),row(2,10))
    assert recipe().remaining(original,opponent=1)[10]==1
    assert 10 not in recipe().remaining(hypothesis,opponent=1)
    assert recipe().remaining(original,opponent=1)[10]==1


def native_fixture(monkeypatch):
    # owner!=controller and an opponent-owned overlay under our host are intentional.
    entities=(NS(uid=1,owner=1,controller=0,location=4,sequence=0,overlay_parent=0,overlay_ordinal=0),
              NS(uid=2,owner=1,controller=2,location=128,sequence=0,overlay_parent=1,overlay_ordinal=0),
              NS(uid=3,owner=0,controller=1,location=2,sequence=0,overlay_parent=0,overlay_ordinal=0),
              NS(uid=4,owner=1,controller=1,location=4,sequence=1,overlay_parent=0,overlay_ordinal=0),
              NS(uid=5,owner=1,controller=1,location=1,sequence=0,overlay_parent=0,overlay_ordinal=0),
              NS(uid=6,owner=1,controller=1,location=64,sequence=0,overlay_parent=0,overlay_ordinal=0),
              NS(uid=7,owner=1,controller=2,location=0,sequence=0,overlay_parent=0,overlay_ordinal=0),
              NS(uid=8,owner=1,controller=1,location=4,sequence=2,overlay_parent=0,overlay_ordinal=0))
    from mirrorforce.common import client_origin_receipt as O,client_shadow as S
    monkeypatch.setattr(O,'entities',lambda _f:entities)
    codes={(0,4,0):10,(1,4,1):999,(1,1,0):BLANK_MAIN,(1,64,0):BLANK_EXTRA,(1,4,2):20}
    reads=[]
    def printed(_core,_duel,p,l,s):
        reads.append((p,l,s));return codes[(p,l,s)]
    monkeypatch.setattr(S,'card_code',printed)
    def query(_duel,p,l,s,flag,buf,_cache):
        assert (p,l,s,flag)==(0,4,0,C.QUERY_OVERLAY_CARD)
        data=struct.pack('<IIII',16,flag,1,10)
        ctypes.memmove(buf,data,len(data));return len(data)
    mutations=tuple(NS(uid=uid,code=10,kind='hydrate') for uid in (1,2))
    core=NS(_lib=NS(query_card=query),card_pool=lambda:NS(cards={10:NS(type=1,alias=123),20:NS(type=1),
                                                            999:NS(type=0x4000)}))
    opening=tuple(NS(uid=uid,owner=1,placeholder=1 if uid!=6 else 2) for uid in (1,2,5,6))
    events=(NS(kind='opening',after=NS(entities=opening)),)
    f=NS(core=core,local=NS(pduel=123),viewer=0,receipt_state=NS(accepted=(NS(mutations=mutations),),pending=(),
         replay_events=events),
         public_opponent_recipe=recipe())
    return f,reads,codes


def test_native_inventory_includes_overlay_owner_and_ignores_foreign_controlled_card(monkeypatch):
    f,reads,_=native_fixture(monkeypatch)
    rows,_=P.native_rows(f)
    assert recipe().remaining(rows,opponent=1)=={20:1,30:1}
    assert (1,2,0) not in reads  # never query an originally own private card
    assert not P.permits_native(f,1,1,0,10)
    assert P.permits_native(f,0,4,0,10)  # same UID is not a third copy
    assert P.permits_native(f,1,4,1,999)  # an existing created token confirms only itself
    assert not P.permits_native(f,1,4,1,20)  # no recipe hydration onto a new created object
    assert (7,1,0,0,True) in rows  # unplaced creation is not guessed or queried
    assert (8,1,20,None,True) in rows  # a non-token CreateCard is also a proved non-opening birth
    assert recipe().remaining(rows,opponent=1)[20]==1


def test_opening_recipe_domain_is_required_and_an_original_cannot_silently_disappear(monkeypatch):
    f,_reads,_codes=native_fixture(monkeypatch)
    original=f.receipt_state.replay_events
    f.receipt_state.replay_events=()
    with pytest.raises(P.PublicRecipeError,match='opening'):P.native_rows(f)
    f.receipt_state.replay_events=original
    from mirrorforce.common import client_origin_receipt as O
    rows=O.entities(f)
    monkeypatch.setattr(O,'entities',lambda _f:tuple(row for row in rows if row.uid!=6))
    with pytest.raises(P.PublicRecipeError,match='disappeared'):P.native_rows(f)


def test_native_recreation_refused_and_restored_native_state_recomputed(monkeypatch):
    f,_reads,codes=native_fixture(monkeypatch)
    before=P.native_rows(f)[0]
    codes[(0,4,0)]=20
    with pytest.raises(P.PublicRecipeError,match='identity'):
        recipe().remaining(P.native_rows(f)[0],opponent=1)
    codes[(0,4,0)]=10
    assert P.native_rows(f)[0]==before


@pytest.mark.parametrize('method',['_hydrate','_hypothesize','_align_identity'])
def test_real_write_entrypoints_reject_before_any_native_mutation(monkeypatch,method):
    from mirrorforce.common import client_shadow as S
    f,_reads,_codes=native_fixture(monkeypatch)
    sync=object.__new__(BlankClientSync)
    sync.public_opponent_recipe=f.public_opponent_recipe
    sync.viewer=f.viewer;sync.core=f.core;sync.local=f.local;sync.receipt_state=f.receipt_state
    sync.record_origins=False
    writes=[]
    monkeypatch.setattr(S,'hydrate',lambda *_a:writes.append('hydrate'))
    monkeypatch.setattr(S,'blank',lambda *_a:writes.append('blank'))
    with pytest.raises(P.PublicRecipeError,match='remaining'):
        if method=='_align_identity':sync._align_identity(1,0,BLANK_MAIN,10)
        else:getattr(sync,method)(1,1,0,10)
    assert not writes


def test_old_default_has_no_recipe_queries_or_changed_write(monkeypatch):
    from mirrorforce.common import client_shadow as S
    sync=object.__new__(BlankClientSync);sync.public_opponent_recipe=None
    sync.record_origins=False;sync.core='old';sync.local=NS(pduel='duel')
    calls=[]
    monkeypatch.setattr(S,'hydrate',lambda *a:calls.append(a) or True)
    assert sync._hydrate(1,1,0,10)
    assert calls==[('old','duel',1,1,0,10)]


def test_search_optin_is_explicit_and_old_default_identity_unchanged():
    from mirrorforce.netduel.agent_search_policy import SearchConfig,search_settings
    from mirrorforce.netduel.agent_search_gate import GateConfig
    from mirrorforce.netduel.agent_stripe_batching import LAW
    assert 'follower_recipe_law' not in search_settings(SearchConfig())
    with pytest.raises(ValueError,match='explicit'):SearchConfig(follower_recipe_law=P.LAW)
    with pytest.raises(ValueError,match='explicit'):
        SearchConfig(follower_recipe_law='just-a-flag',on_demand=GateConfig(enabled=True))
    c=SearchConfig(seconds=9,total_seconds=12,particles=8,selection='greedy',budget_law=LAW,
                   candidate_law='all-legal-common-bank/v1',on_demand=GateConfig(enabled=True),
                   follower_recipe_law=P.LAW)
    assert search_settings(c)['follower_recipe_law']==P.LAW
