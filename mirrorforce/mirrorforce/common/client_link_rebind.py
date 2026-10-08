"""Public intrinsic Link-program evidence for an owned blank follower.

This certificate contains LOCAL UIDs and public packet coordinates, never a
server hidden layout. It authorizes a candidate replay, not search admission.
SUMMONING is an announcement, explicitly not a successful summon.
"""
from __future__ import annotations

from dataclasses import dataclass
import struct

from ..netduel import constants as C
from ..netduel.board import ShadowBoard
from .client_origin_receipt import PublicMutation, PublicSource
from .client_sync import _STILL_PACKETS

INFORMATION_SET_SEARCH = True
MARKER = "public_link_menu_rebind_unvalidated"
LINK_MATERIAL = 0x10000008


def _public_before_origin(follower, cursor):
    """Observation-only projection from original bytes, NOT a native rebuild.

    DisclosureLedger need not pin identities of already-visible field cards.
    Reuse the actual client board instead of treating that absence as unknown,
    or querying a native identity and pretending the wire had disclosed it.
    """
    board = ShadowBoard()
    own = follower.decks[follower.viewer]
    board.start(follower.viewer, list(own.main), list(own.extra))
    board.deck_count = [len(d.main) for d in follower.decks]
    board.extra_count = [len(d.extra) for d in follower.decks]
    end = follower.receipt_packet_raw_indices[cursor - 1] + 1 if cursor else 0
    for packet in follower.receipt_wire_packets[:end]:
        board.apply(packet[0], packet[1:])
    return board


@dataclass(frozen=True)
class LinkProgram:
    reveal: tuple[int, int, int, int]
    extra_uid: int
    material_uids: tuple[int, ...]
    sources: tuple[PublicSource, ...]
    available_after: int

    def receipt(self):
        return PublicMutation("public_link_program", ("link/v1", self.extra_uid, self.material_uids,
            self.available_after), self.reveal[:3], self.extra_uid, self.reveal[3], False,
            self.sources, (MARKER,))


def certify_link(follower, choice):
    """Require known face-up own-controlled materials and one exact Extra move.

    The visible material transitions may span several already-received opaque
    prompts, but only the existing IDLE origin is borrowed. No new lease or
    guessed Extra order is created. Other costs, moves and windows fail closed.
    """
    if (not follower.public_action_rebind or not follower.record_origins or choice.origin is None
            or choice.prompt.msg != C.MSG_SELECT_IDLECMD or choice.prompt.payload[0] == follower.viewer
            or choice.fixes is not None):
        return None
    origin = follower.receipt_saved.get(choice.origin[0])
    current = follower.receipt_saved.get(choice.saved[0])
    if origin is None or current is None or current.cursor != choice.begin + 1:
        return None
    if any(p[0] not in _STILL_PACKETS
           for p in follower.packets[origin.cursor:choice.begin + 1]):
        return None
    player = 1 - follower.viewer
    before = {(r.controller, r.location, r.sequence): r for r in origin.entities if r.location in (4, 64)}
    after = {(r.controller, r.location, r.sequence): r for r in current.entities if r.location in (4, 64)}
    public = None
    materials, sources, extra = [], [], None

    def stable(coordinate):
        a, b = before.get(coordinate), after.get(coordinate)
        return a if a is not None and b is not None and a.uid == b.uid and a.placeholder == b.placeholder else None

    def source(index, role):
        return PublicSource(index, follower.receipt_packet_raw_indices[index], follower.packets[index], role)

    for index in range(choice.begin + 1, len(follower.packets)):
        packet = follower.packets[index]
        if packet[0] in (C.MSG_WAITING, C.MSG_HINT):
            continue
        if extra is not None:
            if len(packet) != 9 or packet[0] != C.MSG_SPSUMMONING \
                    or packet[1:5] != extra[1:5] or packet[5:9] != extra[9:13]:
                return None
            sources.append(source(index, "program_summoning_not_success"))
            reveal = (player, C.LOCATION_EXTRA, extra[7], struct.unpack_from("<I", extra, 1)[0] & 0x7fffffff)
            return LinkProgram(reveal, stable(reveal[:3]).uid, tuple(materials), tuple(sources), index + 1)
        if len(packet) != 17 or packet[0] != C.MSG_MOVE:
            return None
        code = struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff
        previous, destination = tuple(packet[5:9]), tuple(packet[9:13])
        reason = struct.unpack_from("<I", packet, 13)[0]
        if previous[0] != player or not code:
            return None
        if previous[1] == C.LOCATION_MZONE:
            row = stable(previous[:3])
            if public is None:
                public = _public_before_origin(follower, origin.cursor)
            zone = public.zone(player, C.LOCATION_MZONE)
            seen = zone[previous[2]] if previous[2] < len(zone) else None
            if (row is None or row.placeholder or row.uid in materials or not previous[3] & C.POS_FACEUP
                    or reason != LINK_MATERIAL or destination[1] not in (0, C.LOCATION_GRAVE, C.LOCATION_REMOVED)
                    or seen is None or seen.hidden or seen.code != code or seen.position != previous[3]):
                return None
            materials.append(row.uid)
            sources.append(source(index, "program_material"))
        elif previous[1] == C.LOCATION_EXTRA:
            row = stable(previous[:3])
            data = follower.core.card_pool().cards.get(code)
            if (not materials or row is None or row.placeholder != 2 or not previous[3] & C.POS_FACEDOWN
                    or destination[0] != player or destination[1] != C.LOCATION_MZONE
                    or not destination[3] & C.POS_FACEUP or data is None or not data.type & C.TYPE_LINK
                    or reason != 0x800):  # native intrinsic special summon, not another cost/effect
                return None
            extra = packet
            sources.append(source(index, "program_identity"))
        else:
            return None
    return None
