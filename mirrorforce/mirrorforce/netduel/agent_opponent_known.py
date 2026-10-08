"""Current-viewer identity coverage, not positional knowledge or lethal proof.

Only the wire shadow and viewer-scoped disclosure ledger are consulted. A full
known multiset can pass after a shuffle; no synthetic slot anchors are created.
Decks, future draws and random effects remain information-set uncertainty.
"""
from __future__ import annotations

from collections import Counter

from . import constants as C, agent_opponent_clear as OC
from .board import ShadowCard
from .disclosure import DisclosureLedger, DISCLOSURE_LEDGER_SCHEMA

LAW = 'public-opponent-current-identities-known/v1'
SCHEMA = LAW + '#witness'
PROFILE = {'law': LAW, 'source': 'current-viewer-disclosure-and-wire-shadow/v1',
    'locations': [C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE],
    'identity_coverage': 'complete-current-zone-multiset',
    'unanchored_positions': 'retain-public-constrained-sampling',
    'missing_information': 'disable-optional-search',
    'future_hidden_information_known': False, 'guaranteed_lethal': False}
BLOCKED_REASONS = frozenset(('opponent_identities_unknown', 'opponent_knowledge_inconsistent',
                             'opponent_info_unknown'))
_FIELDS = {'schema','law','public_shape','ledger_schema','status','zones','allow_search','reason',
           'position_uncertainty_remaining','future_hidden_information_known','guaranteed_lethal'}
_ZONE_FIELDS = {'location','occupied','public_cards','visible','anchors','counts','unanchored','fresh','categories'}


class KnowledgeInvalid(ValueError):
    """Contradictory/malformed public facts; never catches native/inference errors."""


def validate_profile(profile):
    if type(profile) is not dict or profile != PROFILE:
        raise ValueError('opponent-known search requires its exact viewer-scoped profile')
    return {**PROFILE, 'locations': list(PROFILE['locations'])}


def settings(search='off'):
    from .agent_final_room_budget import settings as room_settings
    result = room_settings(search)
    if result is not None:
        result['opponent_known_profile'] = validate_profile(PROFILE)
    return result


def _code(value):
    return type(value) is int and 0 < value < 2**32 and value not in OC._PLACEHOLDERS


def _pairs(rows, *, counts=False):
    if type(rows) is not list:
        raise KnowledgeInvalid('public pairs must be an ordered list')
    result = {};previous = -1
    for row in rows:
        if type(row) is not list or len(row) != 2:
            raise KnowledgeInvalid('invalid public pair')
        key,value = row
        if type(key) is not int or key <= previous or (counts and not _code(key)) \
                or type(value) is not int or (value <= 0 if counts else not _code(value)):
            raise KnowledgeInvalid('invalid or duplicated public identity/count')
        previous=key;result[key]=value
    return Counter(result) if counts else result


def _matching(domains, counts):
    """Existence only: never return a matching or turn it into a public anchor."""
    copies=[code for code,n in sorted(counts.items()) for _ in range(n)]
    if len(domains) != len(copies):return False
    owned={}
    def augment(row,seen):
        for index,code in enumerate(copies):
            if index in seen or code not in domains[row]:continue
            seen.add(index)
            if index not in owned or augment(owned[index],seen):
                owned[index]=row;return True
        return False
    return all(augment(row,set()) for row in sorted(range(len(domains)),key=lambda i:len(domains[i])))


def _zone_result(zone, shape):
    if type(zone) is not dict or set(zone) != _ZONE_FIELDS:
        raise KnowledgeInvalid('incomplete public zone evidence')
    location=zone['location'];facts=shape['facts']
    if type(location) is not int or location not in PROFILE['locations']:
        raise KnowledgeInvalid('wrong public zone')
    expected=list(range(facts['hand_count'])) if location==C.LOCATION_HAND else [
        row['sequence'] for row in next(z for z in facts['fields'] if z['location']==location)['cards']]
    if type(zone['occupied']) is not list or any(type(x) is not int for x in zone['occupied']) \
            or zone['occupied'] != expected:
        raise KnowledgeInvalid('identity coverage changed current public occupancy')
    occupied=set(expected)
    visible=_pairs(zone['visible']);anchors=_pairs(zone['anchors'])
    counts=_pairs(zone['counts'],counts=True);unanchored=_pairs(zone['unanchored'],counts=True)
    if set(visible)-occupied or set(anchors)-occupied:
        raise KnowledgeInvalid('identity anchored outside current occupied slots')
    metadata=zone['public_cards']
    if type(metadata) is not list or len(metadata)!=len(expected):
        raise KnowledgeInvalid('missing public card metadata')
    for seq,row in zip(expected,metadata):
        if type(row) is not dict or set(row)!={'sequence','position','hidden'} \
                or type(row['sequence']) is not int or row['sequence']!=seq \
                or type(row['position']) is not int or row['position'] not in (0,1,2,4,5,8,10) \
                or type(row['hidden']) is not bool:
            raise KnowledgeInvalid('malformed public card metadata')
        if seq in visible and (row['hidden'] or not row['position'] & C.POS_FACEUP
                               or row['position'] & C.POS_FACEDOWN):
            raise KnowledgeInvalid('concealed card code cannot become a visible identity')
    if location!=C.LOCATION_HAND:
        field=next(z for z in facts['fields'] if z['location']==location)
        if metadata!=[{k:r[k] for k in ('sequence','position','hidden')} for r in field['cards']] \
                or any(r['sequence'] in visible and not r['identified_faceup'] for r in field['cards']):
            raise KnowledgeInvalid('identity evidence changed received public field metadata')
    anchor_counts=Counter(anchors.values())
    if any(anchor_counts[code]>counts[code] for code in anchor_counts) \
            or unanchored != counts-anchor_counts:
        raise KnowledgeInvalid('ledger slots/counts/unanchored disagree')
    merged=dict(anchors)
    for sequence,code in visible.items():
        if sequence in merged and merged[sequence]!=code:
            raise KnowledgeInvalid('visible identity contradicts current public anchor')
        merged[sequence]=code
    # Overlapping public observations prove a per-code MAX, never addition.
    lower=counts | Counter(merged.values())
    if sum(lower.values())>len(occupied):
        raise KnowledgeInvalid('public identity lower bounds exceed current zone count')
    fresh=zone['fresh']
    if type(fresh) is not list or any(type(x) is not int for x in fresh) \
            or fresh!=sorted(set(fresh)) or set(fresh)-occupied \
            or (location==C.LOCATION_HAND and fresh) or set(fresh)&set(anchors):
        raise KnowledgeInvalid('invalid fresh-slot/anchor evidence')
    categories=zone['categories']
    if type(categories) is not list:raise KnowledgeInvalid('missing public category constraints')
    domains={seq:set(lower) for seq in occupied-set(merged)}
    anonymous=[];positioned=set()
    for row in categories:
        if type(row) is not dict or set(row)!={'sequence','codes'}:
            raise KnowledgeInvalid('invalid public category claim')
        seq,codes=row['sequence'],row['codes']
        if type(codes) is not list or not codes or any(not _code(code) for code in codes) \
                or codes!=sorted(set(codes)):
            raise KnowledgeInvalid('invalid category identity domain')
        if seq is None:anonymous.append(set(codes));continue
        if type(seq) is not int or seq not in occupied or seq in positioned:
            raise KnowledgeInvalid('invalid category position')
        positioned.add(seq)
        if seq in merged:
            if merged[seq] not in codes:raise KnowledgeInvalid('public identity violates category')
        else:domains[seq]&=set(codes)
    complete=sum(lower.values())==len(occupied) and not (set(fresh)-set(visible))
    remaining=lower-Counter(merged.values())
    if complete:
        # Coordinate-free claims constrain distinct unresolved cards, not
        # specific positions. Only unconstrained interchangeable slots are
        # used in this feasibility test; no chosen positions escape it.
        free=sorted(set(domains)-positioned)
        if len(anonymous)>len(free):raise KnowledgeInvalid('ambiguous category coverage')
        for seq,allowed in zip(free,anonymous):domains[seq]&=allowed
        if not _matching(list(domains.values()),remaining):
            raise KnowledgeInvalid('known copies cannot satisfy current public constraints')
    return complete, bool(remaining)


def _derive(witness):
    shape=OC.check(witness['public_shape'])
    status=witness['status']
    if status=='public_info_unknown':
        if witness['zones'] is not None or witness['ledger_schema'] is not None:
            raise ValueError('missing public facts cannot invent ledger coverage')
        return False,'opponent_info_unknown',False
    if status=='knowledge_inconsistent':
        if shape['status']!='complete' or witness['zones'] is not None \
                or witness['ledger_schema']!=DISCLOSURE_LEDGER_SCHEMA:
            raise ValueError('invalid inconsistent-public-knowledge witness')
        return False,'opponent_knowledge_inconsistent',False
    if status!='complete' or shape['status']!='complete' \
            or witness['ledger_schema']!=DISCLOSURE_LEDGER_SCHEMA \
            or type(witness['zones']) is not list or len(witness['zones'])!=3 \
            or [zone.get('location') for zone in witness['zones'] if type(zone) is dict]!=PROFILE['locations']:
        raise ValueError('opponent-known witness lacks current viewer-scoped evidence')
    results=[_zone_result(zone,shape) for zone in witness['zones']]
    allowed=all(row[0] for row in results)
    return allowed,'opponent_identities_known' if allowed else 'opponent_identities_unknown',any(r[1] for r in results)


def check(witness):
    if type(witness) is not dict or set(witness)!=_FIELDS or witness['schema']!=SCHEMA \
            or witness['law']!=LAW or type(witness['allow_search']) is not bool \
            or type(witness['position_uncertainty_remaining']) is not bool \
            or witness['future_hidden_information_known'] is not False or witness['guaranteed_lethal'] is not False:
        raise ValueError('opponent-known witness changed its closed public-only contract')
    allowed,reason,uncertain=_derive(witness)
    if (witness['allow_search'],witness['reason'],witness['position_uncertainty_remaining'])!=(allowed,reason,uncertain):
        raise ValueError('opponent-known witness misstates current identity coverage')
    return witness


def extract(client, public_messages):
    shape=OC.extract(client,public_messages)
    witness={'schema':SCHEMA,'law':LAW,'public_shape':shape,'ledger_schema':None,
        'status':'public_info_unknown','zones':None,'allow_search':False,'reason':'opponent_info_unknown',
        'position_uncertainty_remaining':False,'future_hidden_information_known':False,'guaranteed_lethal':False}
    if shape['status']!='complete':return check(witness)
    board=client.board;ledger=getattr(board,'disclosure',None)
    if not isinstance(ledger,DisclosureLedger) or ledger.SCHEMA!=DISCLOSURE_LEDGER_SCHEMA:return check(witness)
    viewer,opponent=shape['viewer'],shape['opponent']
    # Never use ledger.counts: that property merges BOTH players' knowledge.
    counts=ledger.known_counts(viewer);unanchored=ledger.unanchored_identities(viewer)
    fresh=ledger.fresh_slots(viewer,opponent);categories=ledger.category_constraints(viewer,opponent)
    witness['ledger_schema']=DISCLOSURE_LEDGER_SCHEMA
    zones=[]
    try:
        for location in PROFILE['locations']:
            cards=board.zones[(opponent,location)];visible=[];occupied=[];metadata=[]
            for sequence,card in enumerate(cards):
                if card is None:continue
                if type(card) is not ShadowCard or type(card.controller) is not int or card.controller!=opponent \
                        or type(card.location) is not int or card.location!=location \
                        or type(card.sequence) is not int or card.sequence!=sequence \
                        or type(card.position) is not int or card.position not in (0,1,2,4,5,8,10) \
                        or type(card.hidden) is not bool:
                    raise KnowledgeInvalid('malformed public card coordinates')
                occupied.append(sequence)
                metadata.append({'sequence':sequence,'position':card.position,'hidden':card.hidden})
                # Hidden HAND/face-down code is NEVER read, even when nonzero.
                if not card.hidden and card.position & C.POS_FACEUP and not card.position & C.POS_FACEDOWN:
                    if _code(card.code):visible.append([sequence,card.code])
            def zone_counts(source):
                return [[code,n] for (player,loc,code),n in sorted(source.items())
                        if player==opponent and loc==location and n!=0]
            zones.append({'location':location,'occupied':occupied,'public_cards':metadata,'visible':visible,
                'anchors':[[seq,code] for seq,code in sorted(ledger.known_slots(viewer,opponent,location).items())],
                'counts':zone_counts(counts),'unanchored':zone_counts(unanchored),
                'fresh':[seq for loc,seq in fresh if loc==location],
                'categories':[{'sequence':claim.sequence,'codes':sorted(claim.codes)}
                              for claim in categories if claim.location==location]})
        witness.update(status='complete',zones=zones)
        allowed,reason,uncertain=_derive(witness)
        witness.update(allow_search=allowed,reason=reason,position_uncertainty_remaining=uncertain)
    except KnowledgeInvalid:
        witness.update(status='knowledge_inconsistent',zones=None,allow_search=False,
            reason='opponent_knowledge_inconsistent',position_uncertainty_remaining=False)
    return check(witness)


def check_binding(witness, public_messages, *, packet_index, viewer):
    check(witness)
    OC.check_binding(witness['public_shape'],public_messages,packet_index=packet_index,viewer=viewer)
    return witness
