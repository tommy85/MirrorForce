"""Wire layer of the ygopro LAN duel protocol.

Framing (``gframe/netserver.cpp:169-184`` and ``gframe/duelclient.cpp``): every
packet on the TCP stream is

    uint16 packet_len (little endian)   # 1 + len(payload)
    uint8  opcode
    byte   payload[packet_len - 1]

The same framing is used in both directions.  Opcode tables and struct layouts
below come from ``gframe/network.h``; ``PRO_VERSION`` from ``gframe/config.h:29``.

Nothing here knows about duel semantics -- see ``msgparse``/``actions`` for the
``STOC_GAME_MSG`` payloads, which are the raw ygopro-core message buffers.
"""

from __future__ import annotations

import socket
import struct

__all__ = [
    "PRO_VERSION",
    "CTOS",
    "STOC",
    "PLAYERCHANGE",
    "ERRMSG",
    "NETPLAYER_TYPE_OBSERVER",
    "HostInfo",
    "encode_name",
    "decode_name",
    "pack_packet",
    "PacketStream",
    "ProtocolError",
]

# gframe/config.h:29
PRO_VERSION = 0x1362

MAX_DATA_SIZE = 0xFFFE  # network.h: UINT16_MAX - 1
NETPLAYER_TYPE_OBSERVER = 7


class ProtocolError(RuntimeError):
    """The peer sent something the client cannot make sense of."""


class CTOS:
    """Client -> server opcodes (network.h:250-270)."""

    RESPONSE = 0x01  # raw response bytes for set_responseb
    UPDATE_DECK = 0x02  # uint32 mainc, uint32 sidec, uint32[mainc + sidec]
    HAND_RESULT = 0x03  # uint8 (1 scissors, 2 rock, 3 paper)
    TP_RESULT = 0x04  # uint8 (1 = go first)
    PLAYER_INFO = 0x10  # uint16 name[20]
    CREATE_GAME = 0x11
    JOIN_GAME = 0x12  # uint16 version, pad2, uint32 gameid, uint16 pass[20]
    LEAVE_GAME = 0x13
    SURRENDER = 0x14
    TIME_CONFIRM = 0x15
    CHAT = 0x16
    EXTERNAL_ADDRESS = 0x17
    HS_TODUELIST = 0x20
    HS_TOOBSERVER = 0x21
    HS_READY = 0x22
    HS_NOTREADY = 0x23
    HS_KICK = 0x24
    HS_START = 0x25
    REQUEST_FIELD = 0x30


class STOC:
    """Server -> client opcodes (network.h:272-295)."""

    GAME_MSG = 0x01  # payload == one ygopro-core message buffer
    ERROR_MSG = 0x02  # uint8 msg, pad3, uint32 code
    SELECT_HAND = 0x03
    SELECT_TP = 0x04
    HAND_RESULT = 0x05  # uint8 res1, uint8 res2
    TP_RESULT = 0x06
    CHANGE_SIDE = 0x07
    WAITING_SIDE = 0x08
    DECK_COUNT = 0x09
    CREATE_GAME = 0x11
    JOIN_GAME = 0x12  # HostInfo
    TYPE_CHANGE = 0x13  # uint8: pos | (host << 4)
    LEAVE_GAME = 0x14
    DUEL_START = 0x15
    DUEL_END = 0x16
    REPLAY = 0x17
    TIME_LIMIT = 0x18  # uint8 player, pad1, uint16 left_time
    CHAT = 0x19
    HS_PLAYER_ENTER = 0x20
    HS_PLAYER_CHANGE = 0x21
    HS_WATCH_CHANGE = 0x22
    TEAMMATE_SURRENDER = 0x23
    FIELD_FINISH = 0x30


class PLAYERCHANGE:
    OBSERVE = 0x8
    READY = 0x9
    NOTREADY = 0xA
    LEAVE = 0xB


class ERRMSG:
    JOINERROR = 0x1
    DECKERROR = 0x2
    SIDEERROR = 0x3
    VERERROR = 0x4


_STOC_NAMES = {v: k for k, v in vars(STOC).items() if isinstance(v, int)}
_CTOS_NAMES = {v: k for k, v in vars(CTOS).items() if isinstance(v, int)}


def stoc_name(op: int) -> str:
    return _STOC_NAMES.get(op, f"STOC_0x{op:02x}")


def ctos_name(op: int) -> str:
    return _CTOS_NAMES.get(op, f"CTOS_0x{op:02x}")


_HOST_INFO = struct.Struct("<IBBBBB3xiBBH")


class HostInfo:
    """network.h:27-41; 20 bytes with the compiler's 3 byte hole made explicit."""

    __slots__ = (
        "lflist",
        "rule",
        "mode",
        "duel_rule",
        "no_check_deck",
        "no_shuffle_deck",
        "start_lp",
        "start_hand",
        "draw_count",
        "time_limit",
    )

    def __init__(self, *values):
        for name, value in zip(self.__slots__, values):
            setattr(self, name, value)

    @classmethod
    def unpack(cls, data: bytes) -> "HostInfo":
        if len(data) < _HOST_INFO.size:
            raise ProtocolError(
                f"STOC_JOIN_GAME payload is {len(data)} bytes, need {_HOST_INFO.size}"
            )
        return cls(*_HOST_INFO.unpack_from(data))

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__slots__}

    def __repr__(self) -> str:
        return f"HostInfo({self.as_dict()})"


def encode_name(name: str, units: int = 20) -> bytes:
    """UTF-16LE, NUL terminated, fixed width -- the client's ``CopyWideString``."""
    codes = [ord(c) for c in name][: units - 1]
    codes += [0] * (units - len(codes))
    return struct.pack(f"<{units}H", *codes)


def decode_name(raw: bytes) -> str:
    text = raw.decode("utf-16-le", errors="replace")
    end = text.find("\x00")
    return text[:end] if end >= 0 else text


def pack_packet(opcode: int, payload: bytes = b"") -> bytes:
    body = bytes([opcode]) + payload
    if len(body) > MAX_DATA_SIZE:
        raise ProtocolError(f"packet of {len(body)} bytes exceeds the protocol limit")
    return struct.pack("<H", len(body)) + body


class PacketStream:
    """Length-prefixed packet framing over a blocking socket."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._buf = bytearray()
        self.bytes_in = 0
        self.bytes_out = 0

    def send(self, opcode: int, payload: bytes = b"") -> None:
        data = pack_packet(opcode, payload)
        self.sock.sendall(data)
        self.bytes_out += len(data)

    def recv(self, timeout: float | None = None) -> tuple[int, bytes]:
        """Next ``(opcode, payload)``; raises on EOF or timeout."""
        self.sock.settimeout(timeout)
        while True:
            if len(self._buf) >= 2:
                (length,) = struct.unpack_from("<H", self._buf)
                if length == 0:
                    raise ProtocolError("zero-length packet")
                if len(self._buf) >= 2 + length:
                    body = bytes(self._buf[2 : 2 + length])
                    del self._buf[: 2 + length]
                    return body[0], body[1:]
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("connection closed by the host")
            self._buf += chunk
            self.bytes_in += len(chunk)

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


# -- payload builders ------------------------------------------------------


def player_info(name: str) -> bytes:
    return encode_name(name)


def join_game(version: int = PRO_VERSION, gameid: int = 0, password: str = "") -> bytes:
    # uint16 version + 2 padding bytes + uint32 gameid + uint16 pass[20]
    return struct.pack("<H2xI", version, gameid) + encode_name(password)


def update_deck(main: list[int], side: list[int] | None = None) -> bytes:
    """``mainc`` covers main + extra; the host splits them by card type.

    ``DeckManager::LoadDeck`` decides what belongs in the extra deck, so the
    caller passes main and extra concatenated, in that order.
    """
    side = side or []
    body = struct.pack("<II", len(main), len(side))
    body += struct.pack(f"<{len(main) + len(side)}I", *(list(main) + list(side)))
    return body


def hand_result(res: int) -> bytes:
    return bytes([res])


def tp_result(res: int) -> bytes:
    return bytes([res])


def response(data: bytes) -> bytes:
    return bytes(data)
