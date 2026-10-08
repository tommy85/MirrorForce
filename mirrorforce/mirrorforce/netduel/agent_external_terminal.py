"""Opt-in server terminal boundaries; never rule-engine terminal proofs."""
import hashlib
import json

from . import constants as C
from ..common.client_sync import REFRESH_MESSAGES

LAW = 'received-server-surrender-cancel-follower/v1'
NETWORK_LAW = 'received-server-external-cancel-follower/v1'
REASONS = {C.WIN_REASON_SURRENDER: 'server_surrender', C.WIN_REASON_TIMEUP: 'server_timeup',
           C.WIN_REASON_DISCONNECT: 'server_disconnect'}


def reasons(law):
    if law == LAW:
        return (C.WIN_REASON_SURRENDER,)
    if law == NETWORK_LAW:
        return tuple(REASONS)
    raise ValueError('unknown external terminal law')


def digest(messages):
    return hashlib.sha256(json.dumps(messages, separators=(',', ':')).encode()).hexdigest()


def boundary(capture, owner, committed, *, law=LAW):
    """Retain the exact unsynchronized tail without trying to manufacture WIN."""
    owner._ensure_open()  # a pre-existing poison/follower failure remains fatal
    if owner.pending_packets:
        raise ValueError('external terminal has undelivered follower input')
    messages = [[int(m), bytes(b).hex()] for m, b in capture]
    if not messages:
        raise ValueError('external terminal has no received capture')
    packet = bytes([messages[-1][0]]) + bytes.fromhex(messages[-1][1])
    if len(packet) != 3 or packet[0] != C.MSG_WIN or packet[1] not in (0, 1) or packet[2] not in reasons(law):
        raise ValueError('not an exact received server terminal under the registered law')
    raw = tuple(bytes([m]) + bytes.fromhex(b) for m, b in messages[:-1])
    if owner.journal != raw:
        raise ValueError('external terminal lost the original follower prefix')
    follower = owner.follower
    packets = [p for p in raw if p[0] not in REFRESH_MESSAGES and p[0] != C.MSG_START]
    if follower.packets != packets or not 0 <= follower.cursor <= len(packets):
        raise ValueError('external terminal lost the follower cursor or packet tail')
    return {'law': law, 'terminal_index': len(messages) - 1, 'packet': packet.hex(),
            'capture_sha256': digest(messages), 'winner': packet[1], 'reason': packet[2],
            'sent_committed': committed, 'native_terminal_verified': False,
            'follower_scope': 'validated_owned_roots_only', 'follower_cursor': follower.cursor,
            'cancelled_unverified_tail': [p.hex() for p in packets[follower.cursor:]]}


def check(report, record, *, allowed=False, allow_network=False):
    evidence = report.get('external_terminal')
    if evidence is None:
        if record.get('win_reason') in REASONS:
            raise ValueError('server terminal lacks an explicit external boundary')
        return False
    if type(evidence) is not dict or set(evidence) != {
            'law', 'terminal_index', 'packet', 'capture_sha256', 'winner', 'reason', 'sent_committed',
            'native_terminal_verified', 'follower_scope', 'follower_cursor', 'cancelled_unverified_tail'}:
        raise ValueError('external server terminal is not explicitly admitted')
    law = evidence['law']
    if not (law == LAW and allowed or law == NETWORK_LAW and allow_network):
        raise ValueError('external server terminal law is not explicitly admitted')
    messages = report.get('public_messages', [])
    if not messages or any(m == C.MSG_WIN for m, _ in messages[:-1]):
        raise ValueError('external terminal has a missing or duplicate WIN')
    packet = bytes([messages[-1][0]]) + bytes.fromhex(messages[-1][1])
    packets = [bytes([m]) + bytes.fromhex(b) for m, b in messages[:-1]
               if m not in REFRESH_MESSAGES and m != C.MSG_START]
    cursor = evidence['follower_cursor']
    if len(packet) != 3 or packet[0] != C.MSG_WIN or packet[1] not in (0, 1) or packet[2] not in reasons(law) \
            or evidence['terminal_index'] != len(messages) - 1 \
            or evidence['packet'] != packet.hex() or evidence['capture_sha256'] != digest(messages) \
            or evidence['winner'] != packet[1] or evidence['reason'] != packet[2] \
            or record.get('winner') != packet[1] or record.get('win_reason') != packet[2] or record.get('error') \
            or evidence['sent_committed'] != report.get('sent_committed') \
            or evidence['sent_committed'] != record.get('auto_responses') \
            or evidence['native_terminal_verified'] is not False \
            or evidence['follower_scope'] != 'validated_owned_roots_only' \
            or type(cursor) is not int or not 0 <= cursor <= len(packets) \
            or evidence['cancelled_unverified_tail'] != [p.hex() for p in packets[cursor:]] \
            or report.get('failures') != []:
        raise ValueError('server terminal boundary changed its exact evidence or hid a failure')
    return True
