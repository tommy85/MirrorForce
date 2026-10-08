"""Split one core message buffer into individual messages.

``get_message`` hands back everything the core wrote since the last call, which
is usually several messages concatenated with no length prefix: the reader is
expected to know how long each one is.  Over the network the *host* does that
split (``single_duel.cpp``) and forwards one message per ``STOC_GAME_MSG``, so
``mirrorforce.netduel`` never had to; a headless single-mode driver reads the
core directly and does.

The advance rules below are a port of ``SingleMode::SinglePlayAnalyze``
(``ygopro-client/gframe/single_mode.cpp``), which is the client half of the same
pinned engine, so the two agree by construction.  ``ygoenv`` does *not* do this
today -- ``YGOProEnvImpl::handle_message`` jumps ``dp_ = dl_`` for every message
it does not itself decode, which throws away whatever follows in the buffer.
That is harmless when the env only wants the prompt at the end of the buffer,
and wrong for anything that has to see every message.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from ..netduel import constants as C

__all__ = ["Message", "UnknownMessage", "split_messages"]


class UnknownMessage(RuntimeError):
    """A message id with no known length; the rest of the buffer is unreadable."""


@dataclass(frozen=True)
class Message:
    """One core message: its id and the payload after the id byte."""

    msg: int
    payload: bytes

    @property
    def name(self) -> str:
        return MESSAGE_NAMES.get(self.msg, f"MSG_{self.msg}")


MESSAGE_NAMES = {
    value: name
    for name, value in vars(C).items()
    if name.startswith("MSG_") and isinstance(value, int)
}

# messages whose payload is a fixed number of bytes
_FIXED = {
    C.MSG_RETRY: 0,
    C.MSG_HINT: 6,
    C.MSG_WIN: 2,
    C.MSG_SELECT_EFFECTYN: 13,
    C.MSG_SELECT_YESNO: 5,
    C.MSG_SELECT_PLACE: 6,
    C.MSG_SELECT_DISFIELD: 6,
    C.MSG_SELECT_POSITION: 6,
    C.MSG_SHUFFLE_DECK: 1,
    C.MSG_REFRESH_DECK: 1,
    C.MSG_SWAP_GRAVE_DECK: 1,
    C.MSG_REVERSE_DECK: 0,
    C.MSG_DECK_TOP: 6,
    C.MSG_NEW_TURN: 1,
    C.MSG_NEW_PHASE: 2,
    C.MSG_MOVE: 16,
    C.MSG_POS_CHANGE: 9,
    C.MSG_SET: 8,
    C.MSG_SWAP: 16,
    C.MSG_FIELD_DISABLED: 4,
    C.MSG_SUMMONING: 8,
    C.MSG_SUMMONED: 0,
    C.MSG_SPSUMMONING: 8,
    C.MSG_SPSUMMONED: 0,
    C.MSG_FLIPSUMMONING: 8,
    C.MSG_FLIPSUMMONED: 0,
    C.MSG_CHAINING: 16,
    C.MSG_CHAINED: 1,
    C.MSG_CHAIN_SOLVING: 1,
    C.MSG_CHAIN_SOLVED: 1,
    C.MSG_CHAIN_END: 0,
    C.MSG_CHAIN_NEGATED: 1,
    C.MSG_CHAIN_DISABLED: 1,
    C.MSG_DAMAGE: 5,
    C.MSG_RECOVER: 5,
    C.MSG_EQUIP: 8,
    C.MSG_LPUPDATE: 5,
    C.MSG_UNEQUIP: 4,
    C.MSG_CARD_TARGET: 8,
    C.MSG_CANCEL_TARGET: 8,
    C.MSG_PAY_LPCOST: 5,
    C.MSG_ADD_COUNTER: 7,
    C.MSG_REMOVE_COUNTER: 7,
    C.MSG_ATTACK: 8,
    C.MSG_BATTLE: 26,
    C.MSG_ATTACK_DISABLED: 0,
    C.MSG_DAMAGE_STEP_START: 0,
    C.MSG_DAMAGE_STEP_END: 0,
    C.MSG_MISSED_EFFECT: 8,
    C.MSG_ROCK_PAPER_SCISSORS: 1,
    C.MSG_HAND_RES: 1,
    C.MSG_ANNOUNCE_RACE: 6,
    C.MSG_ANNOUNCE_ATTRIB: 6,
    C.MSG_CARD_HINT: 9,
    C.MSG_PLAYER_HINT: 6,
    C.MSG_MATCH_KILL: 4,
}

# messages laid out as: `head` fixed bytes, then repeated
# (count byte, count * stride bytes) blocks, then `tail` fixed bytes
_COUNTED = {
    C.MSG_SELECT_BATTLECMD: (1, [11, 8], 2),
    C.MSG_SELECT_IDLECMD: (1, [7, 7, 7, 7, 7, 11], 3),
    C.MSG_SELECT_OPTION: (1, [4], 0),
    C.MSG_SELECT_CARD: (4, [8], 0),
    C.MSG_SELECT_TRIBUTE: (4, [8], 0),
    C.MSG_SELECT_UNSELECT_CARD: (5, [8, 8], 0),
    C.MSG_SELECT_COUNTER: (5, [9], 0),
    C.MSG_SORT_CARD: (1, [7], 0),
    C.MSG_SORT_CHAIN: (1, [7], 0),
    C.MSG_CONFIRM_DECKTOP: (1, [7], 0),
    C.MSG_CONFIRM_EXTRATOP: (1, [7], 0),
    C.MSG_CONFIRM_CARDS: (2, [7], 0),
    C.MSG_SHUFFLE_HAND: (1, [4], 0),
    C.MSG_SHUFFLE_EXTRA: (1, [4], 0),
    C.MSG_SHUFFLE_SET_CARD: (1, [8], 0),
    C.MSG_CARD_SELECTED: (1, [4], 0),
    C.MSG_RANDOM_SELECTED: (1, [4], 0),
    C.MSG_BECOME_TARGET: (0, [4], 0),
    C.MSG_DRAW: (1, [4], 0),
    C.MSG_TOSS_COIN: (1, [1], 0),
    C.MSG_TOSS_DICE: (1, [1], 0),
    C.MSG_ANNOUNCE_CARD: (1, [4], 0),
    C.MSG_ANNOUNCE_NUMBER: (1, [4], 0),
}


def _len_select_chain(buf: bytes, pos: int) -> int:
    # player, count, then 9 fixed bytes and count * 14
    count = buf[pos + 1]
    return 2 + 9 + count * 14


def _len_select_sum(buf: bytes, pos: int) -> int:
    # mode, player, 6 fixed bytes, then two counted lists of 11-byte entries
    p = pos + 8
    for _ in range(2):
        count = buf[p]
        p += 1 + count * 11
    return p - pos


def _len_tag_swap(buf: bytes, pos: int) -> int:
    # player, deck count, hand count, ... (single_mode.cpp: pbuf[2]*4 + pbuf[4]*4 + 9)
    return buf[pos + 2] * 4 + buf[pos + 4] * 4 + 9


def _len_reload_field(buf: bytes, pos: int) -> int:
    p = pos + 1  # duel rule
    for _ in range(2):
        p += 4  # lp
        for _ in range(7):  # monster zones
            if buf[p]:
                p += 2
            p += 1
        for _ in range(8):  # spell/trap zones
            if buf[p]:
                p += 1
            p += 1
        p += 6  # deck / hand / grave / removed / extra / extra-summonable counts
    count = buf[p]
    p += 1 + count * 15
    return p - pos


def _len_cstring16(buf: bytes, pos: int) -> int:
    # uint16 byte length, then that many bytes, then a terminator
    (n,) = struct.unpack_from("<H", buf, pos)
    return 2 + n + 1


_VARIABLE = {
    C.MSG_SELECT_CHAIN: _len_select_chain,
    C.MSG_SELECT_SUM: _len_select_sum,
    C.MSG_TAG_SWAP: _len_tag_swap,
    C.MSG_RELOAD_FIELD: _len_reload_field,
    C.MSG_AI_NAME: _len_cstring16,
    C.MSG_SHOW_HINT: _len_cstring16,
}


def _payload_length(msg: int, buf: bytes, pos: int) -> int:
    if msg in _FIXED:
        return _FIXED[msg]
    if msg in _COUNTED:
        head, strides, tail = _COUNTED[msg]
        p = pos + head
        for stride in strides:
            count = buf[p]
            p += 1 + count * stride
        return p + tail - pos
    handler = _VARIABLE.get(msg)
    if handler is not None:
        return handler(buf, pos)
    raise UnknownMessage(
        f"no length rule for message {MESSAGE_NAMES.get(msg, msg)} ({msg})"
    )


def split_messages(buf: bytes) -> list[Message]:
    """Walk one ``get_message`` buffer into the messages it concatenates."""
    out: list[Message] = []
    pos = 0
    end = len(buf)
    while pos < end:
        msg = buf[pos]
        pos += 1
        length = _payload_length(msg, buf, pos)
        if pos + length > end:
            raise UnknownMessage(
                f"{MESSAGE_NAMES.get(msg, msg)} wants {length} bytes at {pos}, "
                f"buffer holds {end - pos}"
            )
        out.append(Message(msg, buf[pos : pos + length]))
        pos += length
    return out
