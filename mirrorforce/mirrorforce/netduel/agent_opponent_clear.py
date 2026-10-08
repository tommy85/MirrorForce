"""Strict current-public-information gate; never a lethal proof or a belief query.

An empty opponent hand and entirely face-up, identified field are required.
Known/revealed hand cards and known face-down cards are deliberately NOT exempt.
The remaining decks, future draws and chance branches remain unknown.
"""
from __future__ import annotations

import hashlib

from . import constants as C
from .board import ShadowBoard, ShadowCard

LAW = 'public-opponent-empty-hand-no-facedown-field/v1'
SCHEMA = LAW + '#witness'
PROFILE = {'law': LAW, 'source': 'current-wire-shadow-board/v1', 'opponent_hand_count': 0,
           'field_locations': [C.LOCATION_MZONE, C.LOCATION_SZONE], 'known_concealed_exempt': False,
           'missing_information': 'disable-optional-search', 'future_hidden_information_known': False,
           'guaranteed_lethal': False}
# Wire follower sentinels, never actual identified face-up cards. This module
# must not import the native follower just to test public card validity.
_PLACEHOLDERS = frozenset((999000001, 999000002, 999000003, 999000004))
_STATUS = frozenset(('complete', 'missing_public_stream', 'wrong_public_seat', 'uninitialized_board',
                     'board_integrity_problem', 'missing_public_zone', 'malformed_public_zone',
                     'malformed_public_card'))
_FIELDS = {'schema', 'law', 'viewer', 'opponent', 'public_messages_seen', 'prompt_sha256',
           'board_revision', 'status', 'facts', 'allow_search', 'reason'}


def validate_profile(profile):
    if type(profile) is not dict or profile != PROFILE:
        raise ValueError('opponent-clear search requires its exact strict public-only profile')
    return {**PROFILE, 'field_locations': list(PROFILE['field_locations'])}


def settings(search='off'):
    from .agent_final_room_budget import settings as room_settings
    result = room_settings(search)
    if result is not None:
        result['opponent_clear_profile'] = validate_profile(PROFILE)
    return result


def _reason(witness):
    if witness['status'] != 'complete':
        return 'opponent_info_unknown'
    facts = witness['facts']
    if facts['hand_count']:
        return 'opponent_hand_nonempty'
    cards = [card for zone in facts['fields'] for card in zone['cards']]
    if any(card['position'] & C.POS_FACEDOWN for card in cards):
        return 'opponent_facedown_field'
    if any(card['hidden'] or not card['identified_faceup'] for card in cards):
        return 'opponent_field_unknown'
    return 'opponent_clear'


def check(witness):
    if type(witness) is not dict or set(witness) != _FIELDS or witness['schema'] != SCHEMA \
            or witness['law'] != LAW or witness['status'] not in _STATUS \
            or type(witness['allow_search']) is not bool \
            or type(witness['public_messages_seen']) is not int or witness['public_messages_seen'] < 0 \
            or type(witness['board_revision']) is not int or witness['board_revision'] < 0:
        raise ValueError('opponent-clear witness is incomplete or belongs to another law')
    viewer = witness['viewer']
    if viewer is not None and (type(viewer) is not int or viewer not in (0, 1)) \
            or viewer is not None and type(witness['opponent']) is not int \
            or witness['opponent'] != (None if viewer is None else 1-viewer):
        raise ValueError('opponent-clear witness changed its public observer')
    sha = witness['prompt_sha256']
    if sha is not None and (type(sha) is not str or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)):
        raise ValueError('opponent-clear witness lost its received-prompt binding')
    facts = witness['facts']
    if witness['status'] != 'complete':
        if facts is not None:
            raise ValueError('incomplete public information cannot invent clear-field facts')
    else:
        if viewer is None or sha is None or witness['public_messages_seen'] < 2 or witness['board_revision'] < 1 \
                or type(facts) is not dict or set(facts) != {'hand_count', 'fields'} \
                or type(facts['hand_count']) is not int or not 0 <= facts['hand_count'] <= 256 \
                or type(facts['fields']) is not list or len(facts['fields']) != 2:
            raise ValueError('opponent-clear witness lacks complete public counts/field slots')
        for zone, location, capacity in zip(facts['fields'], (C.LOCATION_MZONE,C.LOCATION_SZONE), (7,8)):
            if type(zone) is not dict or set(zone) != {'location','slots','cards'} or zone['location'] != location \
                    or type(zone['slots']) is not int or not 0 <= zone['slots'] <= capacity \
                    or type(zone['cards']) is not list:
                raise ValueError('invalid public field zone')
            previous = -1
            for card in zone['cards']:
                if type(card) is not dict or set(card) != {'sequence','position','hidden','identified_faceup'} \
                        or type(card['sequence']) is not int or not previous < card['sequence'] < zone['slots'] \
                        or type(card['position']) is not int or card['position'] not in (1,2,4,5,8,10) \
                        or type(card['hidden']) is not bool or type(card['identified_faceup']) is not bool \
                        or card['identified_faceup'] and (card['hidden'] or card['position'] & C.POS_FACEDOWN):
                    raise ValueError('invalid public occupied field slot')
                previous = card['sequence']
    reason = _reason(witness)
    if witness['reason'] != reason or witness['allow_search'] != (reason == 'opponent_clear'):
        raise ValueError('concealed/unknown public state cannot be relabeled as clear')
    return witness


def extract(client, public_messages):
    """Read counts/positions only; never consult disclosure knowledge or a core.

    Missing or malformed public state disables optional work. No native,
    transport, sampler or inference exception is caught/downgraded here.
    """
    viewer = getattr(getattr(client,'result',None),'our_player',None)
    if type(viewer) is not int or viewer not in (0,1):viewer=None
    board = getattr(client,'board',None)
    revision = getattr(board,'revision',0)
    if type(revision) is not int or revision < 0:revision=0
    witness = {'schema':SCHEMA, 'law':LAW, 'viewer':viewer, 'opponent':None if viewer is None else 1-viewer,
        'public_messages_seen':len(public_messages) if type(public_messages) in (list,tuple) else 0,
        'prompt_sha256':None, 'board_revision':revision, 'status':'missing_public_stream',
        'facts':None, 'allow_search':False, 'reason':'opponent_info_unknown'}
    def finish(status, facts=None):
        witness.update(status=status,facts=facts)
        witness['reason']=_reason(witness);witness['allow_search']=witness['reason']=='opponent_clear'
        return check(witness)
    if type(public_messages) not in (list,tuple) or len(public_messages)<2:
        return finish('missing_public_stream')
    from .client import SELECT_MESSAGES
    first,last=public_messages[0],public_messages[-1]
    if any(type(row) not in (list,tuple) or len(row)!=2 or type(row[0]) is not int
           or not 0<=row[0]<=255 or type(row[1]) is not bytes for row in (first,last)) \
            or first[0]!=C.MSG_START or len(first[1])!=18 or last[0] not in SELECT_MESSAGES:
        return finish('missing_public_stream')
    witness['prompt_sha256']=hashlib.sha256(bytes([last[0]])+last[1]).hexdigest()
    player_offset=1 if last[0]==C.MSG_SELECT_SUM else 0
    if viewer is None or first[1][0]&15 != viewer or type(board) is not ShadowBoard \
            or type(board.our_player) is not int or board.our_player!=viewer \
            or len(last[1])<=player_offset or last[1][player_offset]!=viewer:
        return finish('wrong_public_seat')
    if getattr(client,'room_started',False) is not True or not revision or not board.our_deck_start:
        return finish('uninitialized_board')
    if type(getattr(client,'board_problems',None)) is not list or client.board_problems \
            or type(board.slot_problems) is not list or board.slot_problems \
            or type(board.mismatches) is not int or board.mismatches!=0 or type(board.positions) is not dict:
        return finish('board_integrity_problem')
    opponent=1-viewer
    if type(board.zones) is not dict or any((opponent,loc) not in board.zones
                                           for loc in (C.LOCATION_HAND,C.LOCATION_MZONE,C.LOCATION_SZONE)):
        return finish('missing_public_zone')
    hand=board.zones[(opponent,C.LOCATION_HAND)]
    if type(hand) is not list or len(hand)>256 or any(type(card) is not ShadowCard for card in hand):
        return finish('malformed_public_zone')
    fields=[]
    for location,capacity in ((C.LOCATION_MZONE,7),(C.LOCATION_SZONE,8)):
        cards=board.zones[(opponent,location)]
        if type(cards) is not list or len(cards)>capacity:
            return finish('malformed_public_zone')
        rows=[]
        for sequence,card in enumerate(cards):
            if card is None:continue
            if type(card) is not ShadowCard or type(card.controller) is not int or card.controller!=opponent \
                    or type(card.location) is not int or card.location!=location \
                    or type(card.sequence) is not int or card.sequence!=sequence \
                    or type(card.position) is not int or card.position not in (1,2,4,5,8,10) \
                    or type(card.hidden) is not bool:
                return finish('malformed_public_card')
            tracked=board.positions.get((opponent,location,sequence),card.position)
            if type(tracked) is not int or tracked!=card.position:
                return finish('malformed_public_card')
            # No code read for hidden/face-down cards. A stale/unknown/sentinel
            # identity on an apparently face-up card cannot make it clear.
            identified=False
            if not card.hidden and not card.position & C.POS_FACEDOWN:
                identified=type(card.code) is int and 0<card.code<2**32 and card.code not in _PLACEHOLDERS
            rows.append({'sequence':sequence,'position':card.position,'hidden':card.hidden,
                         'identified_faceup':identified})
        fields.append({'location':location,'slots':len(cards),'cards':rows})
    for place in board.positions:
        if type(place) is not tuple or len(place)!=3 or any(type(x) is not int for x in place):
            return finish('malformed_public_card')
        player,location,sequence=place
        if player==opponent and location in (C.LOCATION_MZONE,C.LOCATION_SZONE):
            cards=board.zones[(player,location)]
            if not 0<=sequence<len(cards) or cards[sequence] is None:
                return finish('malformed_public_card')
    return finish('complete',{'hand_count':len(hand),'fields':fields})


def check_binding(witness, public_messages, *, packet_index, viewer):
    """Bind a closed witness to the actual recorded own prompt, not another game/seat."""
    check(witness)
    msg,body=public_messages[packet_index]
    raw=bytes([msg])+bytes.fromhex(body)
    if witness['viewer']!=viewer or witness['public_messages_seen']!=packet_index+1 \
            or witness['prompt_sha256']!=hashlib.sha256(raw).hexdigest():
        raise ValueError('opponent-clear evidence differs from the original received prompt')
    return witness
