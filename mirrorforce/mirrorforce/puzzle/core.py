"""ctypes binding for the pinned ygopro-core, driven headless.

``ygopro-core`` is a plain C ABI (``ocgapi.h``): three callbacks are installed
process-wide, then every duel is a handle passed back into ``process`` /
``get_message`` / ``set_responseb``.  Nothing in that API needs a window, a
socket or a deck, which is what makes single-mode puzzles loadable from Python.

Which shared object
-------------------

The default is the same object ``ygoenv`` links, so a puzzle that loads here
loads in the environment too; ``MF_OCGCORE_LIB`` overrides it.  Loading it with
``ctypes`` does *not* run its module initialiser, so ``ygoenv``'s own readers --
which serve scripts only out of a preloaded map and raise on an unknown card --
are never installed and the readers below take their place.  That difference is
the whole reason puzzles need a separate entry point; see
``mirrorforce/v1/pv/pv12-results.md``.

The readers are global to the process, so only one :class:`Core` may be live at
a time.  :func:`get_core` returns the singleton.
"""

from __future__ import annotations

import ctypes
import os
import sqlite3
from pathlib import Path

__all__ = [
    "CardData",
    "Core",
    "CoreError",
    "DEFAULT_PUZZLE_DIR",
    "EDOPRO_SHIM_FAITHFUL",
    "EDOPRO_SHIM_NAME",
    "EDOPRO_SHIM_RULE_CHANGING",
    "EDOPRO_SHIM_SOURCE",
    "PROCESSOR_BUFFER_LEN",
    "PROCESSOR_END",
    "PROCESSOR_FLAG",
    "PROCESSOR_WAITING",
    "SIZE_MESSAGE_BUFFER",
    "SIZE_QUERY_BUFFER",
    "SIZE_RETURN_VALUE",
    "get_core",
]

# common.h
PROCESSOR_BUFFER_LEN = 0x0FFFFFFF
PROCESSOR_FLAG = 0xF0000000
PROCESSOR_WAITING = 0x10000000
PROCESSOR_END = 0x20000000
SIZE_MESSAGE_BUFFER = 0x2000
SIZE_QUERY_BUFFER = 0x4000
SIZE_RETURN_VALUE = 256
SIZE_SETCODE = 16

TYPE_LINK = 0x4000000

DEFAULT_LIB = os.environ.get(
    "MF_OCGCORE_LIB", "/path/to/workspace/build-deps/priv/ygopro_ygoenv.so"
)
DEFAULT_RUN = Path(os.environ.get("MF_YGOPRO_RUN", "/path/to/workspace/ygopro-client-run"))
DEFAULT_DB = Path(os.environ.get("MF_YGOPRO_DB", str(DEFAULT_RUN / "cards.cdb")))
DEFAULT_SCRIPTS = Path(
    os.environ.get("MF_YGOPRO_SCRIPTS", str(DEFAULT_RUN / "script"))
)
#: archived community puzzle packs; see puzzles/SOURCES.md
DEFAULT_PUZZLE_DIR = Path(os.environ.get("MF_PUZZLE_DIR", "/path/to/workspace/puzzles"))

# ``alias`` in the cdb means two different things and the core now wants them in
# two fields; this is the heuristic ygoenv uses to split them
# (ygoenv/ygopro/ygopro.h, ``db_query_card_data``).
_ARTWORK_OFFSET = 20
_BLACK_LUSTER_SOLDIER_2 = 5405695

#: ``Card.Type`` is EDOPro's name for our ``Card.GetType`` -- the same method,
#: so puzzles recovered by this half of the shim run under unchanged rules.
EDOPRO_SHIM_FAITHFUL = ("Card.Type",)

#: The ``DUEL_*`` names are EDOPro duel-mode flags our core has no equivalent
#: for.  Defining them as 0 turns the mode *off*: a Speed Duel puzzle then
#: loads and plays, but with 5 monster zones instead of 3 and the Master Rule
#: 2020 summoning rules, which is not what its author wrote it against.  A
#: puzzle that needs one of these is playable but **not rule-faithful**.
EDOPRO_SHIM_RULE_CHANGING = ("DUEL_MODE_SPEED", "DUEL_1_FIELD", "DUEL_1ST_TURN_DRAW")

#: served out of the script cache, so nothing is written to disk
EDOPRO_SHIM_NAME = "./script/mirrorforce-edopro-shim.lua"
EDOPRO_SHIM_SOURCE = b"""\
-- MirrorForce compatibility shim for puzzles written against EDOPro.
-- Loaded into the duel's Lua state before the puzzle script when
-- SinglePuzzle(edopro_shim=True); see mirrorforce/v1/pv/pv12-expansion.md.
DUEL_MODE_SPEED = 0
DUEL_1_FIELD = 0
DUEL_1ST_TURN_DRAW = 0
if Card and not Card.Type then Card.Type = Card.GetType end
"""


class CoreError(RuntimeError):
    """A failure the core reported through the message handler."""


class CardData(ctypes.Structure):
    """``card_data`` from ``ygopro-core/card_data.h`` (80 bytes, asserted there)."""

    _fields_ = [
        ("code", ctypes.c_uint32),
        ("alias", ctypes.c_uint32),
        ("setcode", ctypes.c_uint16 * SIZE_SETCODE),
        ("type", ctypes.c_uint32),
        ("level", ctypes.c_uint32),
        ("attribute", ctypes.c_uint32),
        ("race", ctypes.c_uint32),
        ("attack", ctypes.c_int32),
        ("defense", ctypes.c_int32),
        ("lscale", ctypes.c_uint32),
        ("rscale", ctypes.c_uint32),
        ("link_marker", ctypes.c_uint32),
        ("rule_code", ctypes.c_uint32),
    ]


assert ctypes.sizeof(CardData) == 80, "card_data layout drifted from the core"

_SCRIPT_READER = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)
)
_CARD_READER = ctypes.CFUNCTYPE(
    ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(CardData)
)
_MESSAGE_HANDLER = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32)


class Core:
    """The pinned core with file-backed script and database-backed card readers."""

    def __init__(
        self,
        lib_path: str | os.PathLike = DEFAULT_LIB,
        db_path: str | os.PathLike = DEFAULT_DB,
        script_dirs: list[str | os.PathLike] | None = None,
        mode: int = ctypes.RTLD_GLOBAL,
    ) -> None:
        self.lib_path = str(lib_path)
        self.db_path = str(db_path)
        self.script_dirs = [Path(p) for p in (script_dirs or [DEFAULT_SCRIPTS])]
        if not Path(self.lib_path).is_file():
            raise FileNotFoundError(f"core library not found: {self.lib_path}")
        if not Path(self.db_path).is_file():
            raise FileNotFoundError(f"card database not found: {self.db_path}")

        # RTLD_GLOBAL is the default because that is how the object has always
        # been loaded here.  Pass ``mode=ctypes.RTLD_LOCAL`` when loading a
        # *second* core into the same process: two objects built from the same
        # sources export the same ocgapi names, and whichever reached the global
        # namespace first then answers the other one's internal calls -- the
        # duel silently runs on the wrong readers instead of failing.
        self._lib = ctypes.CDLL(self.lib_path, mode=mode)
        self._declare()

        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        # the core keeps the pointer the script reader returned until it has
        # compiled the chunk, so the buffer has to outlive the callback
        self._script_cache: dict[str, ctypes.Array] = {}
        self._card_cache: dict[int, CardData] = {}
        # codes the database lacks, remembered across ``reset_session`` so a
        # cache hit still reports the puzzle as incompatible
        self._known_missing: set[int] = set()
        self._card_pool = None

        #: card codes the puzzle asked for that the pinned cdb does not have
        self.missing_codes: set[int] = set()
        #: every line the core sent to the message handler this session
        self.log: list[str] = []

        self._logbuf = ctypes.create_string_buffer(1024)
        # keep the trampolines alive for the process lifetime
        self._cb_script = _SCRIPT_READER(self._read_script)
        self._cb_card = _CARD_READER(self._read_card)
        self._cb_message = _MESSAGE_HANDLER(self._handle_message)
        self._lib.set_script_reader(self._cb_script)
        self._lib.set_card_reader(self._cb_card)
        self._lib.set_message_handler(self._cb_message)

    # -- ctypes plumbing ---------------------------------------------------

    def _declare(self) -> None:
        lib = self._lib
        p = ctypes.c_void_p
        lib.create_duel.restype = p
        lib.create_duel.argtypes = [ctypes.c_uint32]
        lib.create_duel_v2.restype = p
        lib.create_duel_v2.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        lib.start_duel.argtypes = [p, ctypes.c_uint32]
        lib.end_duel.argtypes = [p]
        lib.set_player_info.argtypes = [p] + [ctypes.c_int32] * 4
        lib.preload_script.restype = ctypes.c_int32
        lib.preload_script.argtypes = [p, ctypes.c_char_p]
        lib.new_card.argtypes = [p, ctypes.c_uint32] + [ctypes.c_uint8] * 5
        lib.process.restype = ctypes.c_uint32
        lib.process.argtypes = [p]
        lib.get_message.restype = ctypes.c_int32
        lib.get_message.argtypes = [p, ctypes.c_char_p]
        lib.get_log_message.argtypes = [p, ctypes.c_char_p]
        lib.set_responsei.argtypes = [p, ctypes.c_int32]
        lib.set_responseb.argtypes = [p, ctypes.c_char_p]
        lib.query_card.restype = ctypes.c_int32
        lib.query_card.argtypes = [
            p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8,
            ctypes.c_uint32, ctypes.c_char_p, ctypes.c_int32,
        ]
        lib.query_field_card.restype = ctypes.c_int32
        lib.query_field_card.argtypes = [
            p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint32,
            ctypes.c_char_p, ctypes.c_int32,
        ]
        lib.query_field_count.restype = ctypes.c_int32
        lib.query_field_count.argtypes = [p, ctypes.c_uint8, ctypes.c_uint8]
        lib.query_field_info.restype = ctypes.c_int32
        lib.query_field_info.argtypes = [p, ctypes.c_char_p]

    # -- the three global callbacks ---------------------------------------

    def _resolve_script(self, path: str) -> Path | None:
        """``./script/cXXXX.lua`` and puzzle paths, against the search dirs."""
        candidate = Path(path)
        if candidate.is_file():
            return candidate
        name = candidate.name
        for root in self.script_dirs:
            hit = root / name
            if hit.is_file():
                return hit
        return None

    def _read_script(self, name, lenptr):
        path = name.decode("utf8", "replace")
        cached = self._script_cache.get(path)
        if cached is not None:
            lenptr[0] = len(cached)
            return ctypes.cast(cached, ctypes.c_void_p).value
        resolved = self._resolve_script(path)
        if resolved is None:
            lenptr[0] = 0
            return None
        data = resolved.read_bytes()
        buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        self._script_cache[path] = buf
        lenptr[0] = len(data)
        return ctypes.cast(buf, ctypes.c_void_p).value

    def _query_card_data(self, code: int) -> CardData | None:
        row = self._db.execute(
            "SELECT alias,setcode,type,atk,def,level,race,attribute "
            "FROM datas WHERE id=?",
            (code,),
        ).fetchone()
        if row is None:
            return None
        alias, setcode, ctype, atk, cdef, level, race, attribute = row
        card = CardData()
        card.code = code
        artwork_variant = (
            alias != 0
            and code != _BLACK_LUSTER_SOLDIER_2
            and alias < code + _ARTWORK_OFFSET
            and code < alias + _ARTWORK_OFFSET
        )
        card.alias = alias if artwork_variant else 0
        card.rule_code = 0 if artwork_variant else alias
        i = 0
        value = setcode & 0xFFFFFFFFFFFFFFFF
        while value and i < SIZE_SETCODE:
            if value & 0xFFFF:
                card.setcode[i] = value & 0xFFFF
                i += 1
            value >>= 16
        card.type = ctype
        card.level = level & 0xFF
        card.lscale = (level >> 24) & 0xFF
        card.rscale = (level >> 16) & 0xFF
        card.attack = atk
        if ctype & TYPE_LINK:
            card.link_marker = cdef
            card.defense = 0
        else:
            card.defense = cdef
            card.link_marker = 0
        card.race = race
        card.attribute = attribute
        return card

    def _read_card(self, code, out):
        code = int(code)
        card = self._card_cache.get(code)
        if card is None:
            card = self._query_card_data(code)
            if card is None:
                # A puzzle written against a different card pool.  ygoenv's
                # reader raises here; raising through a ctypes callback would
                # unwind C++ frames, so record it and hand the core a blank
                # card -- the caller checks ``missing_codes`` afterwards.
                self._known_missing.add(code)
                card = CardData()
                card.code = code
            self._card_cache[code] = card
        if code in self._known_missing:
            self.missing_codes.add(code)
        ctypes.memmove(out, ctypes.byref(card), ctypes.sizeof(CardData))
        return 0

    def _handle_message(self, pduel, msg_type):
        self._lib.get_log_message(ctypes.c_void_p(pduel), self._logbuf)
        self.log.append(self._logbuf.value.decode("utf8", "replace"))
        return 0

    # -- ocgapi ------------------------------------------------------------

    def create_duel(self, seed_sequence) -> int:
        seeds = (ctypes.c_uint32 * 8)(*seed_sequence)
        return self._lib.create_duel_v2(seeds)

    def set_player_info(self, pduel, playerid, lp, startcount, drawcount) -> None:
        self._lib.set_player_info(pduel, playerid, lp, startcount, drawcount)

    def preload_script(self, pduel, path) -> int:
        return self._lib.preload_script(pduel, str(path).encode("utf8"))

    def preload_edopro_shim(self, pduel) -> int:
        """Define the EDOPro-only globals, in this duel's Lua state only.

        ``preload_script`` compiles into ``((duel*)pduel)->lua``, so loading
        this before the puzzle leaves the names defined for it and for nothing
        else.  The source is served out of the script cache, so no file is
        written and the real script directory is untouched.
        """
        if EDOPRO_SHIM_NAME not in self._script_cache:
            data = EDOPRO_SHIM_SOURCE
            self._script_cache[EDOPRO_SHIM_NAME] = (
                ctypes.c_ubyte * len(data)
            ).from_buffer_copy(data)
        return self.preload_script(pduel, EDOPRO_SHIM_NAME)

    def new_card(
        self,
        pduel,
        code: int,
        owner: int,
        playerid: int,
        location: int,
        sequence: int,
        position: int,
    ) -> None:
        """Place one card before ``start_duel``.

        A puzzle builds its board from Lua, but a *normal* duel is built by
        calling this once per deck card, which is what ``load_deck`` in
        ``ygoenv/ygopro/ygopro.h`` does.  Same entry point, so a duel assembled
        here is the duel the environment would have assembled.
        """
        self._lib.new_card(pduel, code, owner, playerid, location, sequence, position)

    def start_duel(self, pduel, options=0) -> None:
        self._lib.start_duel(pduel, options)

    def end_duel(self, pduel) -> None:
        self._lib.end_duel(pduel)

    def process(self, pduel) -> int:
        return self._lib.process(pduel)

    def get_message(self, pduel, buf) -> int:
        return self._lib.get_message(pduel, buf)

    def set_responseb(self, pduel, data: bytes) -> None:
        # duel::set_responseb memcpy's a fixed SIZE_RETURN_VALUE bytes out of
        # the pointer, so a shorter buffer would be read past its end
        if len(data) > SIZE_RETURN_VALUE:
            raise ValueError(f"response of {len(data)} bytes exceeds the return buffer")
        self._lib.set_responseb(pduel, data.ljust(SIZE_RETURN_VALUE, b"\x00"))

    def set_responsei(self, pduel, value: int) -> None:
        self._lib.set_responsei(pduel, value)

    def query_field_card(self, pduel, player, location, flags, buf) -> int:
        return self._lib.query_field_card(pduel, player, location, flags, buf, 0)

    def query_field_count(self, pduel, player, location) -> int:
        return self._lib.query_field_count(pduel, player, location)

    def query_field_info(self, pduel, buf) -> int:
        return self._lib.query_field_info(pduel, buf)

    # -- session bookkeeping ----------------------------------------------

    def reset_session(self) -> None:
        """Forget the previous puzzle's errors and missing cards."""
        self.missing_codes.clear()
        self.log.clear()

    def card_pool(self):
        """The whole database, for prompts that ask the player to name a card.

        ``MSG_ANNOUNCE_CARD`` is answered by searching the card pool rather than
        the board, so the answer can be any card in the game.  Built once per
        process on first use (~30 ms for the 14,968-row database).
        """
        if self._card_pool is None:
            from ..netduel.cards import CardPool

            self._card_pool = CardPool(self.db_path)
        return self._card_pool


_CORE: Core | None = None


def get_core(**kwargs) -> Core:
    """The process-wide core.

    ``set_script_reader`` and friends are global in ``ocgapi.cpp``, so a second
    :class:`Core` would silently steal the first one's readers.  Callers share
    this one instead.
    """
    global _CORE
    if _CORE is None:
        _CORE = Core(**kwargs)
    return _CORE
