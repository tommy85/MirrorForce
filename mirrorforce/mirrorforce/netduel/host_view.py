"""The host's per-player routing and masking rules, as pure functions.

``SingleDuel::Analyze`` in ``ygopro-client/gframe/single_duel.cpp`` is the one
place that decides what a network client sees: the core writes god-view
messages into a buffer, and the host takes them apart one by one, decides
who receives each message and zeroes fields before sending. A local session
that must hand a client the same bytes as a network host needs exactly this
function's output, so it is ported here as ``(msg, payload) -> bytes per
player``: no socket, no engine, no state.

It is ported, not re-derived from an understanding of the protocol, so two
upstream quirks are kept because clients compare byte for byte:

* ``MSG_CARD_SELECTED`` is sent to nobody (``single_duel.cpp:1224`` only
  advances the pointer);
* ``MSG_RANDOM_SELECTED`` goes to ``players[player]`` and is then re-sent to
  ``players[1]`` (``single_duel.cpp:1230``), not ``players[1 - player]``; with
  player 1, seat 0 receives nothing and seat 1 receives it twice.

**Refresh packets (``MSG_UPDATE_DATA`` / ``MSG_UPDATE_CARD``) do not come from
the core.** The host queries ``query_field_card`` / ``query_card`` around some
messages. Only a live core can answer, so this module splits the work:

* :func:`refreshes` says which queries a message triggers, before or after it
  is sent; it reads the message only;
* :func:`mask_update_data` / :func:`mask_update_card` say what the opponent's
  copy of a query result loses; they read the query bytes only.

The caller issues the queries (see :mod:`.wire_projection`).

Source: ``ygopro-client/gframe/single_duel.cpp`` (``Analyze`` at lines
632-1480, ``Refresh*`` at 1548-1672) and ``ygopro-client/gframe/netserver.h``
(``ShouldHideFacedownCode`` / ``StripRevealFlag``). Ported from branch
``sync-validation-20260904`` without behavior changes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable

from . import constants as C

__all__ = [
    "BOTH",
    "Delivery",
    "NOBODY",
    "Refresh",
    "REFRESH_ALL_ZONES",
    "client_stream",
    "deliver",
    "mask_update_card",
    "mask_update_data",
    "refreshes",
    "should_hide_facedown_code",
    "strip_reveal_flag",
]

POS_FACEDOWN = C.POS_FACEDOWN
POS_FACEUP = C.POS_FACEUP
#: ``POS_REVEAL`` is the transient "face-down but shown in this step" bit (``ygopro-core/common.h:83``).
POS_REVEAL = C.POS_REVEAL

#: The shortest ``query_card`` segment: a 4-byte length and a 4-byte flag.
LEN_HEADER = 8
#: Offset of the position word (``QUERY_POSITION``'s info_location) from the segment start:
#: length(4) + flag(4) + code(4). ``card::get_infos`` writes fields in flag-bit order,
#: ``QUERY_CODE`` and ``QUERY_POSITION`` are the first two bits, and every host refresh
#: sets both (``flag |= QUERY_CODE | QUERY_POSITION``), so the offset is constant.
_POSITION_OFFSET = 12

BOTH = (0, 1)
NOBODY: tuple[int, ...] = ()


def should_hide_facedown_code(position: int) -> bool:
    """``NetServer::ShouldHideFacedownCode`` (``netserver.h:37``)."""
    return bool(position & POS_FACEDOWN) and not (position & POS_REVEAL)


def strip_reveal_flag(payload: bytearray, offset: int) -> int:
    """``NetServer::StripRevealFlag`` (``netserver.h:41``), in place.

    Clears ``POS_REVEAL`` in the high byte of the info_location word at
    ``offset`` and returns the cleared position byte. Upstream rewrites the
    buffer before routing, so both players receive the cleared copy.
    """
    (info,) = struct.unpack_from("<I", payload, offset)
    info &= ~(POS_REVEAL << 24)
    struct.pack_into("<I", payload, offset, info)
    return (info >> 24) & 0xFF


@dataclass(frozen=True)
class Refresh:
    """One host refresh query issued before or after a message.

    ``kind`` selects the query and its mask: ``mzone`` / ``szone`` / ``hand``
    / ``grave`` / ``extra`` use ``query_field_card`` (``MSG_UPDATE_DATA``);
    ``single`` uses ``query_card`` (``MSG_UPDATE_CARD``).
    """

    kind: str
    player: int
    flag: int
    use_cache: int = 1
    location: int = 0
    sequence: int = 0


#: ``RefreshMzone(0); RefreshMzone(1); RefreshSzone(0); RefreshSzone(1);
#: RefreshHand(0); RefreshHand(1);`` around idle/battle menus, new turns, new
#: phases and some chain steps. Default flags are ``single_duel.h``'s default arguments.
_DEFAULT_MZONE_FLAG = 0x881fff
_DEFAULT_SZONE_FLAG = 0x681fff
_DEFAULT_HAND_FLAG = 0x681fff
_DEFAULT_GRAVE_FLAG = 0x81fff
_DEFAULT_EXTRA_FLAG = 0xe81fff
_DEFAULT_SINGLE_FLAG = 0xf81fff

REFRESH_ALL_ZONES: tuple[Refresh, ...] = (
    Refresh("mzone", 0, _DEFAULT_MZONE_FLAG),
    Refresh("mzone", 1, _DEFAULT_MZONE_FLAG),
    Refresh("szone", 0, _DEFAULT_SZONE_FLAG),
    Refresh("szone", 1, _DEFAULT_SZONE_FLAG),
    Refresh("hand", 0, _DEFAULT_HAND_FLAG),
    Refresh("hand", 1, _DEFAULT_HAND_FLAG),
)

_REFRESH_FIELD_ONLY: tuple[Refresh, ...] = REFRESH_ALL_ZONES[:4]
_REFRESH_MZONE_ONLY: tuple[Refresh, ...] = REFRESH_ALL_ZONES[:2]


@dataclass(frozen=True)
class Delivery:
    """How one god-view message is routed.

    ``payloads`` maps a seat to the complete bytes it receives (msg id
    included); a missing seat receives nothing. ``counts`` records how many
    times a seat receives it, because ``MSG_RANDOM_SELECTED`` reaches seat 1
    twice.
    """

    payloads: dict[int, bytes]
    counts: dict[int, int]
    before: tuple[Refresh, ...] = ()
    after: tuple[Refresh, ...] = ()

    def for_seat(self, seat: int) -> bytes | None:
        return self.payloads.get(seat)


def _both(raw: bytes, *, before=(), after=()) -> Delivery:
    return Delivery({0: raw, 1: raw}, {0: 1, 1: 1}, before, after)


def _only(seat: int, raw: bytes, *, before=(), after=()) -> Delivery:
    return Delivery({seat: raw}, {seat: 1}, before, after)


def _nobody(*, before=(), after=()) -> Delivery:
    return Delivery({}, {}, before, after)


def _split(viewer_full: bytes, other: bytes, viewer: int) -> dict[int, bytes]:
    return {viewer: viewer_full, 1 - viewer: other}


def _zero_at(raw: bytes, offset: int) -> bytes:
    out = bytearray(raw)
    struct.pack_into("<i", out, offset, 0)
    return bytes(out)


def deliver(msg: int, payload: bytes) -> Delivery:
    """Route and mask one god-view message ``(msg, payload)``.

    Returned bytes include the msg id, as a client reads them from its socket.
    """
    raw = bytes([msg]) + bytes(payload)

    # Hints and the end of the duel.
    if msg == C.MSG_HINT:
        hint_type = payload[0]
        player = payload[1]
        if hint_type in (1, 2, 3, 5):
            return _only(player, raw)
        if hint_type in (4, 6, 7, 8, 9, 11):
            return _only(1 - player, raw)
        if hint_type == 10:
            return _both(raw)
        return _nobody()
    if msg == C.MSG_WIN:
        return _both(raw)
    if msg == C.MSG_RETRY:
        # Sent to the last responder, which is host state rather than a message
        # field. Replays that reproduce never contain it.
        return _nobody()

    # Selection prompts reach only the asked player.
    if msg in (C.MSG_SELECT_BATTLECMD, C.MSG_SELECT_IDLECMD):
        return _only(payload[0], raw, before=REFRESH_ALL_ZONES)
    if msg in (C.MSG_SELECT_EFFECTYN, C.MSG_SELECT_YESNO, C.MSG_SELECT_OPTION,
               C.MSG_SELECT_CHAIN, C.MSG_SELECT_PLACE, C.MSG_SELECT_DISFIELD,
               C.MSG_SELECT_POSITION, C.MSG_SELECT_COUNTER, C.MSG_SORT_CARD):
        return _only(payload[0], raw)
    if msg == C.MSG_SELECT_SUM:
        return _only(payload[1], raw)
    if msg in (C.MSG_SELECT_CARD, C.MSG_SELECT_TRIBUTE):
        player = payload[0]
        return _only(player, _hide_foreign_codes(raw, 1 + 4, player, blocks=1))
    if msg == C.MSG_SELECT_UNSELECT_CARD:
        player = payload[0]
        return _only(player, _hide_foreign_codes(raw, 1 + 5, player, blocks=2))

    # Public confirmations.
    if msg in (C.MSG_CONFIRM_DECKTOP, C.MSG_CONFIRM_EXTRATOP):
        return _both(raw)
    if msg == C.MSG_CONFIRM_CARDS:
        player = payload[0]
        # player(1) + reserved(1) + count(1) + count * (code(4) c(1) l(1) s(1));
        # upstream's ``pbuf[5]`` is the location of the first card.
        location = payload[8] if len(payload) > 8 else 0
        if location != C.LOCATION_DECK:
            return _both(raw)
        return _only(player, raw)

    # Shuffles.
    if msg == C.MSG_SHUFFLE_DECK:
        return _both(raw)
    if msg in (C.MSG_SHUFFLE_HAND, C.MSG_SHUFFLE_EXTRA):
        player = payload[0]
        count = payload[1]
        blind = bytearray(raw)
        for i in range(count):
            struct.pack_into("<i", blind, 1 + 2 + i * 4, 0)
        after = ((Refresh("hand", player, 0x781fff, 0),)
                 if msg == C.MSG_SHUFFLE_HAND
                 else (Refresh("extra", player, _DEFAULT_EXTRA_FLAG),))
        return Delivery(_split(raw, bytes(blind), player), {0: 1, 1: 1}, (), after)
    if msg == C.MSG_REFRESH_DECK or msg == C.MSG_REVERSE_DECK:
        return _both(raw)
    if msg == C.MSG_SWAP_GRAVE_DECK:
        return _both(raw, after=(Refresh("grave", payload[0], _DEFAULT_GRAVE_FLAG),))
    if msg == C.MSG_DECK_TOP:
        return _both(raw)
    if msg == C.MSG_SHUFFLE_SET_CARD:
        location = payload[0]
        after = ((Refresh("mzone", 0, 0x181fff, 0), Refresh("mzone", 1, 0x181fff, 0))
                 if location == C.LOCATION_MZONE
                 else (Refresh("szone", 0, 0x181fff, 0), Refresh("szone", 1, 0x181fff, 0)))
        return _both(raw, after=after)

    # Turns and phases.
    if msg == C.MSG_NEW_TURN:
        return _both(raw, before=REFRESH_ALL_ZONES)
    if msg == C.MSG_NEW_PHASE:
        return _both(raw, after=REFRESH_ALL_ZONES)

    # Moves.
    if msg == C.MSG_MOVE:
        return _deliver_move(raw, payload)
    if msg == C.MSG_POS_CHANGE:
        # code(4), controller, location, sequence, previous/current position.
        cc, cl, cs, pp, cp = payload[4:9]
        after = ((Refresh("single", cc, _DEFAULT_SINGLE_FLAG, 0, cl, cs),)
                 if (pp & POS_FACEDOWN) and (cp & POS_FACEUP) else ())
        return _both(raw, after=after)
    if msg == C.MSG_SET:
        return _both(_zero_at(raw, 1))
    if msg == C.MSG_SWAP:
        c1, l1, s1 = payload[4], payload[5], payload[6]
        c2, l2, s2 = payload[12], payload[13], payload[14]
        return _both(raw, after=(
            Refresh("single", c1, _DEFAULT_SINGLE_FLAG, 0, l1, s1),
            Refresh("single", c2, _DEFAULT_SINGLE_FLAG, 0, l2, s2),
        ))

    # Summons.
    if msg == C.MSG_FIELD_DISABLED:
        return _both(raw)
    if msg == C.MSG_SUMMONING:
        return _both(raw)
    if msg in (C.MSG_SUMMONED, C.MSG_SPSUMMONED, C.MSG_FLIPSUMMONED):
        return _both(raw, after=_REFRESH_FIELD_ONLY)
    if msg == C.MSG_SPSUMMONING:
        out = bytearray(raw)
        cc = payload[4]
        cp = payload[7]
        hide = should_hide_facedown_code(cp)
        strip_reveal_flag(out, 1 + 4)
        blind = bytearray(out)
        if hide:
            struct.pack_into("<i", blind, 1, 0)
        return Delivery(_split(bytes(out), bytes(blind), cc), {0: 1, 1: 1})
    if msg == C.MSG_FLIPSUMMONING:
        return _both(raw, before=(Refresh("single", payload[4], _DEFAULT_SINGLE_FLAG, 0,
                                          payload[5], payload[6]),))

    # Chains.
    if msg == C.MSG_CHAINING:
        return _both(raw)
    if msg in (C.MSG_CHAINED, C.MSG_CHAIN_SOLVED, C.MSG_CHAIN_END):
        return _both(raw, after=REFRESH_ALL_ZONES)
    if msg in (C.MSG_CHAIN_SOLVING, C.MSG_CHAIN_NEGATED, C.MSG_CHAIN_DISABLED):
        return _both(raw)

    # Selected and targeted cards.
    if msg == C.MSG_CARD_SELECTED:
        # Upstream sends it to nobody (single_duel.cpp:1224).
        return _nobody()
    if msg == C.MSG_RANDOM_SELECTED:
        # Upstream: players[player], then ReSendToPlayer(players[1]).
        player = payload[0]
        counts: dict[int, int] = {}
        counts[player] = counts.get(player, 0) + 1
        counts[1] = counts.get(1, 0) + 1
        return Delivery({seat: raw for seat in counts}, counts)
    if msg == C.MSG_BECOME_TARGET:
        return _both(raw)

    # Draws.
    if msg == C.MSG_DRAW:
        player = payload[0]
        count = payload[1]
        blind = bytearray(raw)
        for i in range(count):
            off = 1 + 2 + i * 4
            if not (blind[off + 3] & 0x80):
                struct.pack_into("<i", blind, off, 0)
        return Delivery(_split(raw, bytes(blind), player), {0: 1, 1: 1})

    # Messages for one player only.
    if msg == C.MSG_MISSED_EFFECT:
        return _only(payload[0], raw)
    if msg in (C.MSG_ROCK_PAPER_SCISSORS, C.MSG_ANNOUNCE_RACE,
               C.MSG_ANNOUNCE_ATTRIB, C.MSG_ANNOUNCE_CARD,
               C.MSG_ANNOUNCE_NUMBER):
        return _only(payload[0], raw)

    # Damage step.
    if msg in (C.MSG_DAMAGE_STEP_START, C.MSG_DAMAGE_STEP_END):
        return _both(raw, after=_REFRESH_MZONE_ONLY)

    # Everything else is broadcast.
    if msg in (C.MSG_DAMAGE, C.MSG_RECOVER, C.MSG_EQUIP, C.MSG_LPUPDATE,
               C.MSG_UNEQUIP, C.MSG_CARD_TARGET, C.MSG_CANCEL_TARGET,
               C.MSG_PAY_LPCOST, C.MSG_ADD_COUNTER, C.MSG_REMOVE_COUNTER,
               C.MSG_ATTACK, C.MSG_BATTLE, C.MSG_ATTACK_DISABLED,
               C.MSG_TOSS_COIN, C.MSG_TOSS_DICE, C.MSG_HAND_RES,
               C.MSG_CARD_HINT, C.MSG_PLAYER_HINT):
        return _both(raw)
    if msg == C.MSG_MATCH_KILL:
        # Forwarded only in match mode; a single duel sends it to nobody.
        return _nobody()

    # A msg outside the switch is dropped by the host; do not guess.
    return _nobody()


def _deliver_move(raw: bytes, payload: bytes) -> Delivery:
    """``MSG_MOVE`` (``single_duel.cpp:1011-1034``)."""
    out = bytearray(raw)
    pc = payload[4]
    pl = payload[5]
    cc = payload[8]
    cl = payload[9]
    cs = payload[10]
    cp = payload[11]
    hide_code = should_hide_facedown_code(cp)
    if cl & C.LOCATION_ONFIELD:
        strip_reveal_flag(out, 1 + 8)
    blind = bytearray(out)
    if (not (cl & (C.LOCATION_GRAVE | C.LOCATION_OVERLAY))
            and ((cl & (C.LOCATION_DECK | C.LOCATION_HAND)) or hide_code)):
        struct.pack_into("<i", blind, 1, 0)
    after: tuple[Refresh, ...] = ()
    if cl != 0 and not (cl & C.LOCATION_OVERLAY) and (cl != pl or pc != cc):
        after = (Refresh("single", cc, _DEFAULT_SINGLE_FLAG, 0, cl, cs),)
    return Delivery({cc: bytes(out), 1 - cc: bytes(blind)}, {0: 1, 1: 1}, (), after)


def _hide_foreign_codes(raw: bytes, start: int, player: int, *, blocks: int) -> bytes:
    """``MSG_SELECT_CARD`` and relatives: zero the codes of cards the asked player does not control.

    Each block is a ``count`` byte followed by ``count`` entries of ``code(4)
    c(1) l(1) s(1) ss(1)``; ``SELECT_UNSELECT_CARD`` has two blocks.
    """
    out = bytearray(raw)
    off = start
    for _ in range(blocks):
        count = out[off]
        off += 1
        for _card in range(count):
            if out[off + 4] != player:
                struct.pack_into("<i", out, off, 0)
            off += 8
    return bytes(out)


def refreshes(msg: int, payload: bytes) -> tuple[tuple[Refresh, ...], tuple[Refresh, ...]]:
    """The refresh queries this message triggers before and after it is sent."""
    d = deliver(msg, payload)
    return d.before, d.after


def _iter_segments(body: bytes, start: int):
    """Walk ``query_field_card`` output, yielding ``(offset, segment length)`` per card."""
    pos = start
    total = len(body)
    while pos < total:
        if pos + 4 > total:
            raise ValueError("truncated query segment length")
        (clen,) = struct.unpack_from("<i", body, pos)
        if clen < 4 or pos + clen > total:
            raise ValueError("invalid query segment length")
        if clen > LEN_HEADER and clen < _POSITION_OFFSET + 4:
            raise ValueError("query segment is missing code/position")
        yield pos, clen
        pos += clen
    return


def mask_update_data(kind: str, player: int, location: int, query: bytes) -> dict[int, bytes]:
    """Masks of ``RefreshMzone`` / ``RefreshSzone`` / ``RefreshHand`` / ``RefreshGrave`` / ``RefreshExtra``.

    ``query`` is the complete ``query_field_card`` result, without the
    three-byte ``MSG_UPDATE_DATA`` header. Returns seat -> complete message bytes.
    """
    head = bytes([C.MSG_UPDATE_DATA, player, location])
    if kind in ("mzone", "szone"):
        full = bytearray(head + query)
        hidden: list[tuple[int, int]] = []
        for pos, clen in _iter_segments(bytes(full), len(head)):
            if clen <= LEN_HEADER:
                continue
            position = full[pos + _POSITION_OFFSET + 3]
            hide = should_hide_facedown_code(position)
            strip_reveal_flag(full, pos + _POSITION_OFFSET)
            if hide:
                hidden.append((pos, clen))
        blind = bytearray(full)
        for pos, clen in hidden:
            # Upstream zeroes after the length field (memset(qbuf, 0, clen - 4));
            # the length stays so the client can walk past the segment.
            blind[pos + 4:pos + clen] = b"\x00" * (clen - 4)
        return {player: bytes(full), 1 - player: bytes(blind)}
    if kind == "hand":
        full = head + query
        blind = bytearray(full)
        for pos, clen in _iter_segments(full, len(head)):
            if clen <= LEN_HEADER:
                continue
            position = blind[pos + _POSITION_OFFSET + 3]
            if not (position & POS_FACEUP):
                blind[pos + 4:pos + clen] = b"\x00" * (clen - 4)
        return {player: bytes(full), 1 - player: bytes(blind)}
    if kind == "grave":
        full = head + query
        return {0: full, 1: full}
    if kind == "extra":
        return {player: head + query}
    raise ValueError(f"unknown refresh kind {kind!r}")


def mask_update_card(player: int, location: int, sequence: int, query: bytes) -> dict[int, bytes]:
    """Mask of ``RefreshSingle`` (``single_duel.cpp:1641``).

    ``query`` is the ``query_card`` result without the four-byte header. When
    hidden, the opponent's copy is a minimal segment with only ``QUERY_CODE |
    QUERY_POSITION`` and a zero code.
    """
    head = bytes([C.MSG_UPDATE_CARD, player, location, sequence])
    if len(query) <= LEN_HEADER:
        return {player: head + query}
    full = bytearray(head + query)
    at = len(head) + _POSITION_OFFSET
    position = full[at + 3]
    hide = bool(position & POS_FACEDOWN)
    if location & C.LOCATION_ONFIELD:
        hide = should_hide_facedown_code(position)
        strip_reveal_flag(full, at)
    mine = bytes(full)
    if not hide:
        return {player: mine, 1 - player: mine}
    blind = bytearray(head)
    blind += struct.pack("<i", 16)
    blind += struct.pack("<I", C.QUERY_CODE | C.QUERY_POSITION)
    blind += b"\x00\x00\x00\x00"
    blind += bytes(full[at:at + 4])
    return {player: mine, 1 - player: bytes(blind)}


def client_stream(messages: Iterable[tuple[int, bytes]], viewer: int) -> list[bytes]:
    """Project a whole god-view message stream to the bytes ``viewer`` receives.

    Routing only: refresh packets need a live core, and the caller adds them
    using :func:`refreshes`.
    """
    out: list[bytes] = []
    for msg, payload in messages:
        d = deliver(msg, payload)
        raw = d.payloads.get(viewer)
        if raw is None:
            continue
        out.extend([raw] * d.counts.get(viewer, 1))
    return out
