"""Viewer-scoped current facts, including known-but-unpositioned copies."""
from copy import deepcopy
import struct

import pytest

from mirrorforce.netduel import constants as C, agent_opponent_known as K, agent_opponent_clear as O
from mirrorforce.netduel.disclosure import DisclosureLedger
from test_opponent_clear import example,card


def hidden(client,viewer,location,codes):
    client.board.zones[(1-viewer,location)]=[
        card(viewer,location,i,8,code=object(),hidden=True) for i in range(len(codes))]
    for seq,code in enumerate(codes):
        if code:client.board.disclosure.disclose(1-viewer,location,code,sequence=seq)


@pytest.mark.parametrize('viewer',[0,1])
@pytest.mark.parametrize('location',[C.LOCATION_HAND,C.LOCATION_MZONE,C.LOCATION_SZONE])
@pytest.mark.parametrize('codes',[[],[123],[123,123,123],[123,456],[123,None]])
def test_complete_current_identities_are_required_for_each_zone(viewer,location,codes):
    client,messages=example(viewer);hidden(client,viewer,location,codes)
    value=K.extract(client,messages)
    assert value['allow_search']==all(codes)
    assert not value['future_hidden_information_known'] and not value['guaranteed_lethal']
    if codes:assert not O.extract(client,messages)['allow_search']


@pytest.mark.parametrize('viewer',[0,1])
def test_real_confirm_then_hand_shuffle_keeps_multiset_but_never_invents_positions(viewer):
    client,messages=example(viewer);hidden(client,viewer,C.LOCATION_HAND,[None,None,None])
    opponent=1-viewer;ledger=client.board.disclosure
    body=bytes([viewer,0,3])+b''.join(struct.pack('<IBBB',code,opponent,C.LOCATION_HAND,seq)
                                    for seq,code in enumerate([123,123,456]))
    ledger.observe_confirm(body,C.MSG_CONFIRM_CARDS)
    assert K.extract(client,messages)['allow_search']
    ledger.observe_shuffle(C.MSG_SHUFFLE_HAND,bytes([opponent,3])+bytes(12))
    before=ledger.known_slots(viewer,opponent,C.LOCATION_HAND)
    value=K.extract(client,messages)
    assert value['allow_search'] and value['position_uncertainty_remaining']
    assert value['zones'][0]['counts']==[[123,2],[456,1]]
    assert value['zones'][0]['anchors']==[] and before=={}
    assert ledger.known_slots(viewer,opponent,C.LOCATION_HAND)=={}


@pytest.mark.parametrize('location',[C.LOCATION_MZONE,C.LOCATION_SZONE])
def test_real_field_shuffle_retains_copy_constrained_position_orbit(location):
    client,messages=example();hidden(client,0,location,[123,456])
    places=[1 | location<<8 | seq<<16 | 8<<24 for seq in (0,1)]
    client.board.disclosure.observe_shuffle_set_card(bytes([location,2])+struct.pack('<II',*places)+bytes(8))
    value=K.extract(client,messages)
    zone=next(z for z in value['zones'] if z['location']==location)
    assert value['allow_search'] and value['position_uncertainty_remaining']
    assert zone['anchors']==[] and zone['counts']==[[123,1],[456,1]]
    assert [c['codes'] for c in zone['categories']]==[[123,456],[123,456]]


def test_another_viewers_private_knowledge_does_not_qualify_and_omniscient_counts_never_read():
    class ScopedLedger(DisclosureLedger):
        @property
        def counts(self):raise AssertionError('omniscient property is forbidden')
    client,messages=example();client.board.disclosure=ScopedLedger()
    hidden(client,0,C.LOCATION_HAND,[None])
    client.board.disclosure.disclose(1,C.LOCATION_HAND,123,sequence=0,audience=2)
    assert K.extract(client,messages)['reason']=='opponent_identities_unknown'
    client.board.disclosure.disclose(1,C.LOCATION_HAND,123,sequence=0,audience=1)
    assert K.extract(client,messages)['allow_search']


def test_concealed_shadow_code_is_never_even_read(monkeypatch):
    from mirrorforce.netduel.board import ShadowCard
    client,messages=example()
    for location in K.PROFILE['locations']:hidden(client,0,location,[123])
    original=ShadowCard.__getattribute__
    def read(card,name):
        if name=='code' and original(card,'hidden'):
            raise AssertionError('hidden shadow identity must not be inspected')
        return original(card,name)
    monkeypatch.setattr(ShadowCard,'__getattribute__',read)
    assert K.extract(client,messages)['allow_search']


def test_visible_and_ledger_facts_are_max_not_sum_and_distinct_public_slots_keep_copies():
    client,messages=example();loc=C.LOCATION_MZONE
    client.board.zones[(1,loc)]=[card(0,loc,i,1,code=123) for i in range(3)]
    ledger=client.board.disclosure
    ledger.disclose(1,loc,123,sequence=0)
    assert K.extract(client,messages)['allow_search']
    for i in range(3):ledger.disclose(1,loc,123,sequence=i)
    assert K.extract(client,messages)['allow_search']


def test_stale_departure_deck_shuffle_and_new_unknown_draw_cannot_reuse_hand_proof():
    client,messages=example();hidden(client,0,C.LOCATION_HAND,[123])
    board=client.board
    move=struct.pack('<IIII',123,1 | C.LOCATION_HAND<<8 | 8<<24,
                     1 | C.LOCATION_DECK<<8 | 8<<24,0)
    board.apply(C.MSG_MOVE,move)
    board.apply(C.MSG_SHUFFLE_DECK,b'\1')
    board.apply(C.MSG_DRAW,b'\1\1'+bytes(4))
    assert len(board.zones[(1,C.LOCATION_HAND)])==1
    assert K.extract(client,messages)['reason']=='opponent_identities_unknown'


def test_old_unanchored_identity_cannot_identify_new_fresh_field_slot():
    client,messages=example();hidden(client,0,C.LOCATION_SZONE,[None])
    ledger=client.board.disclosure
    ledger.disclose(1,C.LOCATION_SZONE,123)
    ledger._fresh[0][(1,C.LOCATION_SZONE)]={0}
    assert K.extract(client,messages)['reason']=='opponent_identities_unknown'
    ledger.disclose(1,C.LOCATION_SZONE,123,sequence=0)
    # A truly new known instance adds a copy. Stale old-zone knowledge now
    # exceeds the real count and must not be silently truncated.
    assert K.extract(client,messages)['reason']=='opponent_knowledge_inconsistent'


@pytest.mark.parametrize('fault',['excess','anchor_count','anchor_empty','visible_conflict','fresh_anchor',
    'category_wrong','category_duplicate','category_no_matching','malformed_hand','placeholder'])
def test_inconsistent_public_knowledge_fails_closed(fault):
    client,messages=example();loc=C.LOCATION_SZONE;hidden(client,0,loc,[123,456])
    ledger=client.board.disclosure
    if fault=='excess':ledger._viewer_counts[0][(1,loc,123)]+=1
    elif fault=='anchor_count':ledger._viewer_counts[0][(1,loc,123)]=0
    elif fault=='anchor_empty':ledger._slots[0][(1,loc)][7]=123
    elif fault=='visible_conflict':
        c=client.board.zones[(1,loc)][0];c.position=1;c.hidden=False;c.code=789
    elif fault=='fresh_anchor':ledger._fresh[0][(1,loc)]={0}
    elif fault=='category_wrong':ledger._claims[0][(1,loc)]=[(0,frozenset([789]),0)]
    elif fault=='category_duplicate':ledger._claims[0][(1,loc)]=[(0,frozenset([123]),0)]*2
    elif fault=='category_no_matching':
        ledger._slots[0].pop((1,loc));ledger._claims[0][(1,loc)]=[(i,frozenset([123]),0) for i in (0,1)]
    elif fault=='placeholder':
        ledger._slots[0][(1,loc)][0]=999000001
        ledger._viewer_counts[0][(1,loc,123)]=0
        ledger._viewer_counts[0][(1,loc,999000001)]=1
    else:
        hidden(client,0,C.LOCATION_HAND,[123]);client.board.zones[(1,C.LOCATION_HAND)][0].sequence=1
    assert K.extract(client,messages)['reason']=='opponent_knowledge_inconsistent'


@pytest.mark.parametrize('codes',[(123,),(123,456)])
def test_category_constraints_are_not_identity_disclosures(codes):
    client,messages=example();hidden(client,0,C.LOCATION_HAND,[None])
    client.board.disclosure._claims[0][(1,C.LOCATION_HAND)]=[(0,frozenset(codes),0)]
    assert K.extract(client,messages)['reason']=='opponent_identities_unknown'


@pytest.mark.parametrize('fault',['allow','uncertainty','future','proof','extra','visible_hidden','counts','missing'])
def test_closed_witness_cannot_forge_admission(fault):
    client,messages=example();hidden(client,0,C.LOCATION_HAND,[123,456])
    value=deepcopy(K.extract(client,messages))
    if fault=='allow':value['allow_search']=False
    elif fault=='uncertainty':value['position_uncertainty_remaining']=True
    elif fault=='future':value['future_hidden_information_known']=True
    elif fault=='proof':value['guaranteed_lethal']=True
    elif fault=='extra':value['host_truth']={}
    elif fault=='visible_hidden':value['zones'][0]['visible']=[[0,123]]
    elif fault=='counts':value['zones'][0]['counts']=[[123,2],[456,1]]
    else:value['zones'].pop()
    with pytest.raises(ValueError):K.check(value)
