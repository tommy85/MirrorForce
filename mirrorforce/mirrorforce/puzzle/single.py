"""Load and play a ygopro single-mode puzzle headlessly.

A puzzle is a Lua script that calls the ``Debug`` library to place cards, set
life points and hand the turn to the player, instead of the two decks a normal
duel starts from.  The client loads one with the sequence in
``SingleMode::SinglePlayThread`` (``ygopro-client/gframe/single_mode.cpp``);
this is that sequence with the Irrlicht half removed:

1. install the script / card / message callbacks (``puzzle.core``);
2. ``create_duel_v2`` with a seed sequence;
3. ``set_player_info`` for both players with the *defaults* -- the puzzle's own
   ``Debug.SetPlayerInfo`` overwrites them, and a puzzle that omits the call
   inherits these;
4. ``preload_script`` on the puzzle file -- this is where the board is built,
   and where a missing card or a Lua error shows up;
5. drain ``get_message`` once: the ``Debug.SetAIName`` / ``Debug.ShowHint``
   output was written before the duel started;
6. ``start_duel``, then ``process`` / ``get_message`` until ``MSG_WIN`` or the
   processor reports ``PROCESSOR_END``.

Step 4 is the one that matters for compatibility: the puzzle names card codes
directly, so a code the pinned database does not carry loads no script and the
core reports it through the message handler.
"""

from __future__ import annotations

import ctypes
import random
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from ..netduel import constants as C
from ..netduel.actions import (
    ActionAct,
    ActionPhase,
    LegalAction,
    SelectContext,
    parse_select,
)
from ..netduel.board import parse_query_segments
from .core import (
    EDOPRO_SHIM_FAITHFUL,
    EDOPRO_SHIM_RULE_CHANGING,
    PROCESSOR_BUFFER_LEN,
    PROCESSOR_END,
    PROCESSOR_FLAG,
    SIZE_QUERY_BUFFER,
    Core,
    get_core,
)
from .messages import Message, split_messages

__all__ = [
    "Decision",
    "FieldInfo",
    "PuzzleError",
    "PuzzleRun",
    "RESPONSE_REQUIRED",
    "SinglePuzzle",
    "ZoneCard",
    "edopro_shim_symbols",
    "first_policy",
    "random_policy",
    "scripted_policy",
    "solution_policy",
]

# single_mode.cpp: the client's defaults, before the puzzle's SetPlayerInfo
DEFAULT_START_LP = 8000
DEFAULT_START_HAND = 5
DEFAULT_DRAW_COUNT = 1

# single_mode.cpp SinglePlayReload; everything except QUERY_REASON_CARD
FULL_QUERY_FLAGS = 0xFFDFFF

QUERY_LOCATIONS = (
    C.LOCATION_MZONE,
    C.LOCATION_SZONE,
    C.LOCATION_HAND,
    C.LOCATION_DECK,
    C.LOCATION_EXTRA,
    C.LOCATION_GRAVE,
    C.LOCATION_REMOVED,
)

# the prompts ``SingleMode::SinglePlayAnalyze`` blocks on: every message after
# which it waits for ``SingleMode::SetResponse`` before walking on
RESPONSE_REQUIRED = frozenset(
    {
        C.MSG_SELECT_BATTLECMD,
        C.MSG_SELECT_IDLECMD,
        C.MSG_SELECT_EFFECTYN,
        C.MSG_SELECT_YESNO,
        C.MSG_SELECT_OPTION,
        C.MSG_SELECT_CARD,
        C.MSG_SELECT_TRIBUTE,
        C.MSG_SELECT_UNSELECT_CARD,
        C.MSG_SELECT_CHAIN,
        C.MSG_SELECT_PLACE,
        C.MSG_SELECT_DISFIELD,
        C.MSG_SELECT_POSITION,
        C.MSG_SELECT_COUNTER,
        C.MSG_SELECT_SUM,
        C.MSG_SORT_CARD,
        C.MSG_SORT_CHAIN,
        C.MSG_ROCK_PAPER_SCISSORS,
        C.MSG_ANNOUNCE_RACE,
        C.MSG_ANNOUNCE_ATTRIB,
        C.MSG_ANNOUNCE_CARD,
        C.MSG_ANNOUNCE_NUMBER,
    }
)

LOCATION_NAMES = {
    C.LOCATION_DECK: "deck",
    C.LOCATION_HAND: "hand",
    C.LOCATION_MZONE: "mzone",
    C.LOCATION_SZONE: "szone",
    C.LOCATION_GRAVE: "grave",
    C.LOCATION_REMOVED: "removed",
    C.LOCATION_EXTRA: "extra",
}


class PuzzleError(RuntimeError):
    """The puzzle could not be loaded or played to a conclusion."""


_COMMENT_BLOCK = re.compile(r"(?s)--\[\[.*?\]\]")


def edopro_shim_symbols(path: str | Path) -> list[str]:
    """Which EDOPro-only names a puzzle script references, in shim order.

    Comments are stripped first: every puzzle in the archive carries a prose
    header and many carry a written solution, and an English sentence must not
    count as a use of the API.
    """
    text = Path(path).read_text(encoding="utf8", errors="replace")
    text = _COMMENT_BLOCK.sub("", text)
    text = "\n".join(line.split("--", 1)[0] for line in text.splitlines())
    found = []
    for name in EDOPRO_SHIM_RULE_CHANGING:
        if re.search(rf"\b{re.escape(name)}\b", text):
            found.append(name)
    for name in EDOPRO_SHIM_FAITHFUL:
        # "Card.Type" is a method call on some card variable, not a global
        method = name.split(".", 1)[1]
        if re.search(rf"[:.]{re.escape(method)}\s*\(", text):
            found.append(name)
    return found


@dataclass
class ZoneCard:
    """One card the engine reported for a zone slot."""

    player: int
    location: int
    sequence: int
    code: int = 0
    position: int = 0
    level: int = 0
    rank: int = 0
    attack: int = 0
    defense: int = 0
    overlay: list = field(default_factory=list)

    @property
    def location_name(self) -> str:
        return LOCATION_NAMES.get(self.location, hex(self.location))

    def __str__(self) -> str:
        base = f"p{self.player} {self.location_name}[{self.sequence}] {self.code}"
        if self.overlay:
            base += f" xyz{self.overlay}"
        return base


@dataclass
class FieldInfo:
    """``query_field_info`` -- life points and per-zone occupancy."""

    duel_rule: int = 0
    lp: tuple = (0, 0)
    mzone: tuple = ((), ())
    szone: tuple = ((), ())
    counts: tuple = ((), ())  # deck, hand, grave, removed, extra, extra_p


@dataclass
class Decision:
    """One prompt the policy answered."""

    msg: int
    player: int
    n_options: int
    index: int
    action: str


@dataclass
class PuzzleRun:
    """Everything one puzzle session produced."""

    path: str
    loaded: bool = False
    started: bool = False
    hints: list = field(default_factory=list)
    ai_name: str = ""
    missing_codes: list = field(default_factory=list)
    core_log: list = field(default_factory=list)
    messages: list = field(default_factory=list)  # (name, count)
    decisions: list = field(default_factory=list)
    auto_responses: int = 0
    winner: int | None = None
    win_reason: int | None = None
    turns: int = 0
    error: str = ""
    #: EDOPro-only names this puzzle uses, empty unless the shim was enabled
    shim_symbols: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Loaded and played with nothing reported through the message handler."""
        return self.loaded and not self.missing_codes and not self.core_log

    @property
    def rules_unfaithful(self) -> bool:
        """The shim had to switch a duel mode off to make this puzzle load.

        True means the board is playable but the rules are not the ones the
        puzzle was written against, so it must not be used as a reward task --
        a wrong-rules puzzle teaches wrong rules.  The ``Card.Type`` half of
        the shim is an exact alias and does not set this.
        """
        return any(name in EDOPRO_SHIM_RULE_CHANGING for name in self.shim_symbols)

    @property
    def player_won(self) -> bool:
        return self.winner == 0


class SinglePuzzle:
    """One puzzle, from ``preload_script`` to ``MSG_WIN``."""

    def __init__(
        self,
        path: str | Path,
        core: Core | None = None,
        seed: int = 0x9E3779B9,
        options: int = 0,
        max_options: int = 64,
        edopro_shim: bool = False,
    ) -> None:
        self.path = Path(path)
        # off by default: the shim defines duel-mode flags as 0, which makes a
        # puzzle written for another mode load under *our* rules instead of
        # failing loudly.  Callers ask for it, and check ``rules_unfaithful``.
        self.edopro_shim = edopro_shim
        self.core = core if core is not None else get_core()
        # the puzzle's script directory is searched too, so a puzzle that pulls
        # in a helper next to itself still loads.  It joins the core's search
        # roots only for this puzzle's lifetime (load() .. close()): the core is
        # process-global, and a root left behind changes what every later duel
        # in the process resolves -- the reanalysis locator, for one, refuses a
        # core with more than one capture script tree.
        self._script_dir = self.path.parent
        self._script_dir_added = False
        self.seed = seed
        self.options = options
        self.pduel = None
        self.run_result = PuzzleRun(path=str(self.path))
        self._rng = random.Random(seed)
        # MSG_ANNOUNCE_CARD asks the player to name a card, so the answer has
        # to be searched over the whole database rather than the board
        self.ctx = SelectContext(
            max_options=max_options, rng=self._rng, card_pool=self.core.card_pool()
        )
        self._msgbuf = None
        self._querybuf = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> SinglePuzzle:
        self.load()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self.pduel is not None:
            self.core.end_duel(self.pduel)
            self.pduel = None
        if self._script_dir_added:
            self._script_dir_added = False
            try:
                self.core.script_dirs.remove(self._script_dir)
            except ValueError:
                pass  # someone else already dropped it; nothing to undo

    def load(self) -> PuzzleRun:
        """Steps 1-5: build the board and read back what the puzzle announced."""
        if not self.path.is_file():
            raise PuzzleError(f"puzzle not found: {self.path}")
        if self._script_dir not in self.core.script_dirs:
            self.core.script_dirs.append(self._script_dir)
            self._script_dir_added = True
        self.core.reset_session()
        self._msgbuf = ctypes.create_string_buffer(0x10000)
        self._querybuf = ctypes.create_string_buffer(SIZE_QUERY_BUFFER)

        rng = random.Random(self.seed)
        seeds = [rng.getrandbits(32) for _ in range(8)]
        self.pduel = self.core.create_duel(seeds)
        for player in (0, 1):
            self.core.set_player_info(
                self.pduel,
                player,
                DEFAULT_START_LP,
                DEFAULT_START_HAND,
                DEFAULT_DRAW_COUNT,
            )
        result = self.run_result
        if self.edopro_shim:
            if not self.core.preload_edopro_shim(self.pduel):
                raise PuzzleError("the EDOPro compatibility shim failed to compile")
            result.shim_symbols = edopro_shim_symbols(self.path)
        rc = self.core.preload_script(self.pduel, self.path)
        result.loaded = bool(rc)
        if not result.loaded:
            result.error = "preload_script rejected the puzzle script"
        for message in self._drain():
            self._observe(message)
        result.missing_codes = sorted(self.core.missing_codes)
        result.core_log = list(self.core.log)
        return result

    def start(self) -> None:
        if self.pduel is None:
            raise PuzzleError("load() first")
        self.core.start_duel(self.pduel, self.options)
        self.run_result.started = True

    # -- reading the board -------------------------------------------------

    def _drain(self) -> list[Message]:
        length = self.core.get_message(self.pduel, self._msgbuf)
        if length <= 0:
            return []
        return split_messages(self._msgbuf.raw[:length])

    def field_info(self) -> FieldInfo:
        """``query_field_info``, which writes an ``MSG_RELOAD_FIELD`` body."""
        length = self.core.query_field_info(self.pduel, self._querybuf)
        raw = self._querybuf.raw[:length]
        # skip the leading MSG_RELOAD_FIELD id
        pos = 1
        info = FieldInfo(duel_rule=raw[pos])
        pos += 1
        lp, mzone, szone, counts = [], [], [], []
        for _ in range(2):
            (player_lp,) = struct.unpack_from("<i", raw, pos)
            pos += 4
            lp.append(player_lp)
            zones = []
            for _ in range(7):
                if raw[pos]:
                    zones.append((raw[pos + 1], raw[pos + 2]))
                    pos += 3
                else:
                    zones.append(None)
                    pos += 1
            mzone.append(tuple(zones))
            zones = []
            for _ in range(8):
                if raw[pos]:
                    zones.append(raw[pos + 1])
                    pos += 2
                else:
                    zones.append(None)
                    pos += 1
            szone.append(tuple(zones))
            counts.append(tuple(raw[pos : pos + 6]))
            pos += 6
        info.lp = tuple(lp)
        info.mzone = tuple(mzone)
        info.szone = tuple(szone)
        info.counts = tuple(counts)
        return info

    def zone(self, player: int, location: int) -> list[ZoneCard | None]:
        """``query_field_card`` for one zone, decoded into slots."""
        length = self.core.query_field_card(
            self.pduel, player, location, FULL_QUERY_FLAGS, self._querybuf
        )
        segments = parse_query_segments(self._querybuf.raw[:length])
        out: list[ZoneCard | None] = []
        for sequence, fields in enumerate(segments):
            if not fields or not fields.get("code"):
                out.append(None)
                continue
            # QUERY_POSITION writes card::get_public_info_location(), which
            # packs controller | location<<8 | sequence<<16 | position<<24;
            # POS_REVEAL can be OR-ed into the top byte, hence the 0x0f mask
            info_location = fields.get("info_location", 0)
            out.append(
                ZoneCard(
                    player=player,
                    location=location,
                    sequence=sequence,
                    code=fields.get("code", 0),
                    position=(info_location >> 24) & 0x0F,
                    level=fields.get("level", 0),
                    rank=fields.get("rank", 0),
                    attack=fields.get("attack", 0),
                    defense=fields.get("defense", 0),
                    overlay=fields.get("overlay", []),
                )
            )
        return out

    def board(self) -> list[ZoneCard]:
        """Every card the engine can see, in a stable order."""
        out: list[ZoneCard] = []
        for player in (0, 1):
            for location in QUERY_LOCATIONS:
                for card in self.zone(player, location):
                    if card is not None:
                        out.append(card)
        return out

    # -- playing -----------------------------------------------------------

    def _observe(self, message: Message) -> None:
        result = self.run_result
        result.messages.append(message.name)
        msg, body = message.msg, message.payload
        if msg == C.MSG_AI_NAME:
            (n,) = struct.unpack_from("<H", body, 0)
            result.ai_name = body[2 : 2 + n].decode("utf8", "replace")
        elif msg == C.MSG_SHOW_HINT:
            (n,) = struct.unpack_from("<H", body, 0)
            result.hints.append(body[2 : 2 + n].decode("utf8", "replace"))
        elif msg == C.MSG_NEW_TURN:
            result.turns += 1
        elif msg == C.MSG_NEW_PHASE:
            (self.ctx.current_phase,) = struct.unpack_from("<H", body, 0)
        elif msg == C.MSG_HINT:
            hint_type, _player = body[0], body[1]
            (value,) = struct.unpack_from("<I", body, 2)
            if hint_type == C.HINT_SELECTMSG and value == 501:
                self.ctx.discard_hand = True
        elif msg == C.MSG_WIN:
            result.winner = body[0]
            result.win_reason = body[1]

    def _answer(self, message: Message, policy) -> None:
        msg, body = message.msg, message.payload
        result = self.run_result
        # in single mode the local side answers for whichever player the core
        # is asking about, so the prompt's own player is "us" for this prompt
        if body:
            self.ctx.our_player = body[0] if msg != C.MSG_SELECT_SUM else body[1]
        select = parse_select(msg, body, self.ctx)
        if select.auto_response is not None:
            result.auto_responses += 1
            self.core.set_responseb(self.pduel, select.auto_response)
            return
        selector = select.selector
        if selector is None:
            raise PuzzleError(f"{message.name} produced neither answer nor selector")
        while True:
            actions = selector.options()
            if not actions:
                raise PuzzleError(f"{message.name} offers no legal action")
            index = int(policy(selector, actions, self))
            if not 0 <= index < len(actions):
                raise PuzzleError(
                    f"policy chose {index} of {len(actions)} for {message.name}"
                )
            result.decisions.append(
                Decision(
                    msg=msg,
                    player=selector.player,
                    n_options=len(actions),
                    index=index,
                    action=actions[index].describe(),
                )
            )
            data = selector.choose(index)
            if data is not None:
                self.core.set_responseb(self.pduel, data)
                return

    def play(self, policy=None, max_steps: int = 4000) -> PuzzleRun:
        """Run the duel to ``MSG_WIN`` (or until the processor ends).

        ``PROCESSOR_WAITING`` is deliberately not used as the cue to answer.
        ``PROCESSOR_WAIT`` (``processor.cpp``) raises the same flag with an
        empty buffer purely to let a client redraw after ``MSG_CONFIRM_CARDS``,
        and a prompt is not always the last message in its buffer.  The client
        answers on encountering a prompt *during the walk*
        (``SingleMode::SinglePlayAnalyze``) and stops only on ``MSG_WIN``; so
        does this.
        """
        if policy is None:
            policy = first_policy
        if not self.run_result.started:
            self.start()
        result = self.run_result
        for _ in range(max_steps):
            raw = self.core.process(self.pduel)
            flag = raw & PROCESSOR_FLAG
            length = raw & PROCESSOR_BUFFER_LEN
            if length:
                for message in self._drain():
                    self._observe(message)
                    if message.msg == C.MSG_RETRY:
                        raise PuzzleError("the core rejected the last response")
                    if message.msg in RESPONSE_REQUIRED:
                        self._answer(message, policy)
            if result.winner is not None:
                break
            if flag == PROCESSOR_END:
                break
        else:
            result.error = f"no conclusion after {max_steps} process() calls"
        result.core_log = list(self.core.log)
        result.missing_codes = sorted(self.core.missing_codes)
        return result


# -- policies ---------------------------------------------------------------


def first_policy(selector, actions: list[LegalAction], puzzle: SinglePuzzle) -> int:
    """Always the first option; enough to walk a puzzle to its turn end."""
    return 0


def random_policy(seed: int = 0):
    """Uniform over the legal actions -- the search a lethal puzzle needs."""
    rng = random.Random(seed)

    def choose(selector, actions, puzzle):
        return rng.randrange(len(actions))

    return choose


def scripted_policy(indices, fallback=first_policy):
    """Replay a recorded solution: one index per decision, then ``fallback``."""
    remaining = list(indices)

    def choose(selector, actions, puzzle):
        if remaining:
            return remaining.pop(0)
        return fallback(selector, actions, puzzle)

    return choose


_ACT_NAMES = {a.name.lower(): a for a in ActionAct}
_PHASE_NAMES = {p.name.lower(): p for p in ActionPhase}


def _matches(action: LegalAction, constraints: dict) -> bool:
    for key, want in constraints.items():
        if key == "act":
            if action.act != _ACT_NAMES[want]:
                return False
        elif key == "phase":
            if action.phase != _PHASE_NAMES[want]:
                return False
        elif key == "finish":
            if bool(action.finish) != (want in ("1", "true", "yes")):
                return False
        elif key in ("code", "position", "place", "effect", "number", "response"):
            if getattr(action, key) != int(want):
                return False
        elif key == "spec":
            if action.spec != want:
                return False
        else:
            raise PuzzleError(f"unknown solution constraint {key!r}")
    return True


def solution_policy(steps, fallback=None, decline_chains: bool = True):
    """Play a written-down solution, one constraint set per decision.

    A step is either an explicit index, or ``"key=value,key=value"`` naming the
    action to take -- ``"act=summon,code=48305365"``, ``"act=direct_attack"``,
    ``"phase=battle"``, ``"place=1"``.  The first legal action satisfying every
    constraint is chosen; if none does, the run stops with the options listed,
    which is what makes a solution that stopped working readable.  ``"any"``
    takes the first option, for the prompts a solution does not care about
    (where to place a monster, an empty chain).

    ``decline_chains`` answers every ``MSG_SELECT_CHAIN`` that still offers a
    choice with "no chain", without spending a step.  A puzzle's opponent holds
    open traps that the core offers to *us* -- single mode has no second
    player -- and a solution that never chains should not have to say so at
    every window.  Set it false for a solution that chains on purpose.
    """
    remaining = list(steps)

    def choose(selector, actions, puzzle):
        if decline_chains and selector.msg == C.MSG_SELECT_CHAIN:
            for index, action in enumerate(actions):
                if action.act == ActionAct.CANCEL:
                    return index
        if not remaining:
            if fallback is not None:
                return fallback(selector, actions, puzzle)
            raise PuzzleError(
                "solution exhausted but the core still asks: "
                + ", ".join(a.describe() for a in actions)
            )
        step = remaining.pop(0)
        if isinstance(step, int):
            return step
        if step == "any":
            return 0
        constraints = dict(
            part.split("=", 1) for part in step.split(",") if "=" in part
        )
        for index, action in enumerate(actions):
            if _matches(action, constraints):
                return index
        raise PuzzleError(
            f"solution step {step!r} matches none of: "
            + ", ".join(a.describe() for a in actions)
        )

    return choose
