"""A whole duel, headless and re-runnable, over the pinned core.

``mirrorforce.puzzle`` drives *puzzles*: a Lua script builds the board and the
local side answers for whoever the core asks.  The world model needs the other
shape -- two real decks, two independent policies, and the ability to reach a
mid-duel state twice.

Re-running the same state is the whole trick.  ``ocgapi`` has no way to copy a
live duel, so a branch is taken by rebuilding the duel from its seed and
feeding back the responses that were recorded up to the branch point.  The core
is a pure function of (seed sequence, card order, response stream), so the
rebuilt duel is byte-identical to the original at that point; from there the
branch answers differently and the two futures can be compared.  Everything the
generator does -- the forced-pass next menu, alternative-action settlements,
chain-order probes -- is that one mechanism.

Two details make the replay exact rather than approximate:

* **every** response is recorded, including the ones the action layer answers
  by itself (an empty chain, a counter placement, the end-phase discard).  Some
  of those consume the shared RNG, so regenerating them would only agree if the
  RNG were in the same state; feeding the bytes back verbatim needs no such
  assumption.
* the main deck is shuffled in Python from the duel seed rather than by the
  core, so the card order is part of the recorded configuration.

``PROCESSOR_WAITING`` is deliberately not used as the cue to answer -- see the
note in ``mirrorforce/puzzle/single.py``; this loop answers on encountering a
prompt while walking the buffer, exactly as the reference client does.
"""

from __future__ import annotations

import ctypes
import random
import struct
import time as _time
from dataclasses import dataclass, field
from pathlib import Path

from ..netduel import constants as C
from ..netduel.actions import (
    LegalAction,
    PromptSelection,
    SelectContext,
    UnsupportedMessage,
    parse_select,
    track_command_follow_up,
)
from ..puzzle.core import (
    PROCESSOR_BUFFER_LEN,
    PROCESSOR_END,
    PROCESSOR_FLAG,
    Core,
    get_core,
)
from ..puzzle.messages import Message, split_messages
from ..puzzle.single import RESPONSE_REQUIRED
from ..search.banish_origin import BanishOriginTracker
from .state import DisclosureLedger

__all__ = [
    "CHANCE_MSGS",
    "DECISION_MSGS",
    "DeckList",
    "DuelConfig",
    "DuelDriver",
    "DuelError",
    "MR5_DUEL_OPTIONS",
    "Prompt",
    "StopAt",
    "StopDuel",
    "first_responder",
    "load_ydk",
    "mixed_responder",
    "pass_responder",
    "random_responder",
]

#: ygoenv builds duel options as ``(rules << 16)``; rules 5 is Master Rule 5
MR5_DUEL_OPTIONS = 5 << 16

#: the three prompts that present an *action menu* rather than a sub-selection
#: inside an action already chosen.  These are the world model's decision
#: points: their options are exactly the "card instance x action type" grammar
#: the enumerator reproduces, plus the phase / chain-pass globals.
DECISION_MSGS = frozenset(
    {C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD, C.MSG_SELECT_CHAIN}
)

POS_FACEDOWN_DEFENSE = C.POS_FACEDOWN_DEFENSE


#: Messages that "reveal" cards. Every payload format was checked against the ygopro-core source;
#: they all carry code + controler + location + sequence per card, enough to locate the instance.
_CONFIRM_MSGS = (
    C.MSG_CONFIRM_CARDS, C.MSG_CONFIRM_DECKTOP, C.MSG_CONFIRM_EXTRATOP,
)


class DuelError(RuntimeError):
    """The duel could not be built or driven."""


class DuelDeadlineExceeded(DuelError):
    """This duel exceeded the per-duel wall clock and is dropped as unfinished.

    It can only trigger between steps; an infinite loop inside a single `core.process()` never reaches here
    and needs a process-level wall clock.
    """

    def __init__(self, message: str, seconds: float, steps: int, turn: int,
                 *, max_seconds: float | None = None):
        super().__init__(message)
        self.seconds = seconds
        self.steps = steps
        self.turn = turn
        # Preserve the exact allowance; callers must not infer it from rounded
        # diagnostic text when distinguishing their own budget from a failure.
        self.max_seconds = max_seconds


class SubSelectionUnresolved(DuelError):
    """Sub-choices did not converge within ``max_sub_rounds`` rounds; this duel is dropped as unfinished.

    Carries ``msg`` / ``turn`` / ``codes`` so the coverage ledger can book by (deck, sub-prompt type, triggering card).
    Dropped duels are now a normal path; if they cluster on one card or one scene, that is a coverage hole dug out
    silently: dropping 0.01% uniformly does not matter, dropping 100% of one card is a training blind spot.
    """

    def __init__(self, message: str, msg: int, turn: int, codes: tuple[int, ...]):
        super().__init__(message)
        self.msg = msg
        self.turn = turn
        self.codes = codes


class StopDuel(Exception):
    """Raised by a responder to end the walk early, keeping the duel alive."""


@dataclass(frozen=True)
class DeckList:
    """One deck, as ``new_card`` wants it."""

    name: str
    main: tuple[int, ...]
    extra: tuple[int, ...]

    @property
    def codes(self) -> frozenset[int]:
        return frozenset(self.main) | frozenset(self.extra)


def load_ydk(path: str | Path, name: str | None = None) -> DeckList:
    """Read a ``.ydk``; the side deck is ignored, as it is in ygoenv."""
    main: list[int] = []
    extra: list[int] = []
    bucket: list[int] | None = None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("#main"):
            bucket = main
            continue
        if line.startswith("#extra"):
            bucket = extra
            continue
        if line.startswith("!side"):
            bucket = None
            continue
        if not line or line.startswith(("#", "!")):
            continue
        if bucket is not None and line.isdigit():
            bucket.append(int(line))
    if not main:
        raise DuelError(f"{path}: no main deck")
    return DeckList(
        name=name or Path(path).stem, main=tuple(main), extra=tuple(extra)
    )


@dataclass(frozen=True)
class DuelConfig:
    """Everything the duel is a function of.

    Two duels with the same config and the same response stream are the same
    duel, which is what makes a branch reproducible.
    """

    decks: tuple[DeckList, DeckList]
    seed: int
    start_lp: int = 8000
    start_hand: int = 5
    draw_count: int = 1
    duel_options: int = MR5_DUEL_OPTIONS
    #: Deck-building environment. ocgcore itself does not consume the ban list; rooms check decks before the duel, but it changes
    #: the prior over the opponent's hidden cards and the policy distribution, so it is still a model input per duel. Only cards
    #: deviating from the default limit of 3 are listed; the id is the YGOPro lflist hash, used only for provenance.
    banlist_id: int = 0
    banlist: tuple[tuple[int, int], ...] = ()
    #: cap on the option list the action layer builds for ``MSG_ANNOUNCE_CARD``
    #: (the only prompt whose candidate list is truncated rather than complete)
    max_options: int = 128
    #: cap on sub-selection rounds within a single prompt. A multi-select keeps
    #: re-prompting until ``selector.choose`` returns data; if the selection can
    #: never be satisfied, that loop spins forever in user space — full CPU, zero
    #: syscalls, no output. Five workers wedged this way for 49–80 minutes each on
    #: 2026-08-23 and held the last 120 chunks of trajA hostage. A legitimate
    #: selection needs tens of rounds, never hundreds, so this only ever fires on
    #: a duel that was never going to finish.
    max_sub_rounds: int = 512
    #: External WinBot can choose end phase directly while battle/main 2 is
    #: also legal.  Legacy ygoenv corpora intentionally omit that action;
    #: exact live-teacher replay opts into the complete engine phase menu.
    full_phase_menu: bool = False
    #: Keep legacy ygoenv's random end-phase discard by default.  Exact
    #: external-teacher replay disables it and consumes the recorded choices.
    auto_end_phase_discard: bool = True
    full_card_sort_menu: bool = False
    #: See ``SelectContext.commit_command_selections``; a replay must use the recording client's value.
    commit_command_selections: bool = False
    #: rebuild of a duel some *other* host dealt: exact per-player main-deck
    #: orders (as a forward ``new_card`` loop wants them) and the 8-word
    #: sequence that host passed to ``create_duel_v2``.  Unset, the driver
    #: deals for itself from ``seed`` as before.  search/hostdeal.py derives
    #: these for the patched ygopro host the collect workers duel on.
    forced_deck_orders: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    forced_core_seeds: tuple[int, ...] | None = None

    def deck_orders(self) -> tuple[list[int], list[int]]:
        """The shuffled main decks, derived from the seed.

        Done here rather than in the core so that the card order is part of the
        configuration and a rebuild reproduces it.
        """
        if self.forced_deck_orders is not None:
            return (list(self.forced_deck_orders[0]),
                    list(self.forced_deck_orders[1]))
        rng = random.Random(("deckorder", self.seed).__repr__())
        out = []
        for deck in self.decks:
            main = list(deck.main)
            rng.shuffle(main)
            out.append(main)
        return out[0], out[1]


@dataclass
class Prompt:
    """One point at which the core stopped and asked somebody something."""

    index: int  #: ordinal over every response the duel consumes
    msg: int
    player: int
    turn: int
    turn_player: int
    phase: int
    actions: list[LegalAction] = field(default_factory=list)
    note: str = ""
    truncated: bool = False  #: the option list hit ``max_options``
    #: The parser proved that no legal phase transition was hidden for legacy
    #: ygoenv compatibility.  Distinct from size truncation.
    complete_menu: bool = True
    #: The chooser's progress through a card-selection prompt at this round
    #: (``Selector.selection``); ``None`` for every other prompt.  The
    #: tensorizer refuses a card-selection prompt without it.
    selection: PromptSelection | None = None

    @property
    def is_decision(self) -> bool:
        """An action menu, as opposed to a sub-selection inside one action."""
        return self.msg in DECISION_MSGS

    @property
    def name(self) -> str:
        return _MSG_NAMES.get(self.msg, f"MSG_{self.msg}")


_MSG_NAMES = {
    value: name
    for name, value in vars(C).items()
    if name.startswith("MSG_") and isinstance(value, int)
}

#: messages that make the following state change depend on the engine's RNG or
#: reveal information no deterministic model could have predicted
CHANCE_MSGS = frozenset(
    {
        C.MSG_DRAW,
        C.MSG_TOSS_COIN,
        C.MSG_TOSS_DICE,
        C.MSG_ROCK_PAPER_SCISSORS,
        C.MSG_RANDOM_SELECTED,
        C.MSG_SHUFFLE_DECK,
        C.MSG_SHUFFLE_HAND,
        C.MSG_SHUFFLE_EXTRA,
        C.MSG_SHUFFLE_SET_CARD,
        C.MSG_CONFIRM_DECKTOP,
        C.MSG_CONFIRM_EXTRATOP,
    }
)


def _selection_of(selector) -> PromptSelection | None:
    """A selector's card-selection context; a duck-typed agent grammar selector may have none."""
    method = getattr(selector, "selection", None)
    return method() if method is not None else None


class DuelDriver:
    """One live duel; drives the core and hands prompts to a responder.

    The responder is called as ``responder(prompt, driver) -> int`` for every
    prompt the action layer does not answer itself, and may raise
    :class:`StopDuel` to freeze the duel where it stands (used to stop a branch
    at the first prompt it cares about).
    """

    def parse_prompt(self, msg: int, payload: bytes):
        """Parser hook for consumers requiring the complete agent grammar."""
        return parse_select(msg, payload, self._ctx)

    def __init__(self, config: DuelConfig, core: Core | None = None):
        self.config = config
        self.core = core if core is not None else get_core()
        self.pduel = None
        #: every payload handed to ``set_responseb``, in order
        self.responses: list[bytes] = []
        #: messages seen since :meth:`clear_log`
        self.messages: list[Message] = []
        self.record_messages = False
        self.turn = 0
        self.turn_player = 0
        #: the seat that took turn 1.  ``start_duel`` in ``ocgapi.cpp`` ends
        #: with ``add_process(PROCESSOR_TURN, 0, ...)``, so this is always seat
        #: 0 -- recorded rather than assumed, because every win rate in this
        #: project is now reported split by going first / going second.
        self.first_player: int | None = None
        self.phase = 0
        self.winner: int | None = None
        self.win_reason: int | None = None
        self.finished = False
        self.steps = 0
        self.prompt_count = 0
        self.truncations = 0
        # Structurally zero since the fallback answer was removed (every prompt
        # is parsed or the game fails); kept so snapshot formats and admission
        # checks keep their fields.
        self.fallbacks = 0
        self.last_response_context = "none"
        self._pending_response_context = ""
        #: Disclosed cards: ``(controller, zone, code) -> number of cards with disclosed identity in that zone``.
        #: The moment a card is activated (``MSG_CHAINING``) it is public information, and from then on both sides know
        #: what it is, even if it is still in the hand. ``mask_for`` uses it for visibility "conditioned on events",
        #: not only on location.
        #:
        #: **Codes alone are not enough.** With codes only, the opponent activating an Ash Blossom would also expose
        #: the other two copies **in their deck**, a real leak. It actually happened:
        #: the leak assertion reported ``opponent face-down deck[13] carries code``.
        #:
        #: **Instance coordinates like (controller, zone, sequence, code) do not work either.** Hand / deck / graveyard
        #: are vectors: ``field::remove_card`` (``field.cpp:204``) calls ``reset_sequence`` on removal, shifting every later
        #: sequence down, and **these shifts send no** ``MSG_MOVE``. So "the opponent plays hand card 0" would silently
        #: misalign a previously disclosed hand card 3 and anonymize it again: disclosed information turning unknown, the
        #: opposite of "disclosure is input". So the ledger books the **multiset count** by (controller, zone, code):
        #: instances of one code are indistinguishable anyway, and neither sequence shifts nor shuffles affect the count.
        #: Which sequences they map to is resolved canonically against the current board by :func:`state.resolve_disclosure`
        #: at masking time.
        #:
        self.disclosure = DisclosureLedger()
        #: Public sources of face-down banished slots (main-deck card or extra monster), kept beside the ledger, also using only
        #: public location words and rolled back with snapshots; the sampler uses it to decide which pool each face-down banished slot draws from
        self.banish_origin = BanishOriginTracker()
        self._replay: list[bytes] = []
        #: optional observer called for every *replay-answered* selector
        #: prompt as ``on_replay_prompt(prompt, driver)``, before the recorded
        #: response is fed.  The search server uses it to grow a scorer
        #: policy's stream through the replayed prefix exactly as the online
        #: actor grew its own: one observation per prompt the actor answered,
        #: none for auto-responses, none for the pending decision itself.
        self.on_replay_prompt = None
        self._rng = random.Random(("select", config.seed).__repr__())
        self._msgbuf = ctypes.create_string_buffer(0x10000)
        self._ctx = SelectContext(
            max_options=config.max_options,
            rng=self._rng,
            card_pool=self.core.card_pool(),
            full_phase_menu=config.full_phase_menu,
            auto_end_phase_discard=config.auto_end_phase_discard,
            full_card_sort_menu=config.full_card_sort_menu,
            commit_command_selections=config.commit_command_selections,
        )

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "DuelDriver":
        self.build()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    #: the driver's mutable python state, for the R4 snapshot engine: the
    #: core rolls back by memcpy, and this is the exact set of fields that
    #: must roll back with it.  Everything else on the instance is config,
    #: OS handles or scratch.  ``save_pystate`` asserts the instance holds
    #: no attribute outside these two sets, so a future field lands in one
    #: of them deliberately instead of silently diverging a rollback.
    _PYSTATE_FIELDS = (
        "_answering_msg", "_answering_payload", "_ctx",
        "_pending_response_context", "_replay",
        "_rng", "banish_origin", "disclosure", "fallbacks", "finished",
        "first_player",
        "last_response_context", "messages", "phase", "prompt_count",
        "responses", "steps", "truncations", "turn", "turn_player",
        "win_reason", "winner",
    )
    _PYSTATE_EXEMPT = (
        "config", "core", "pduel", "_msgbuf", "on_replay_prompt",
        "record_messages", "_infobuf", "_querybuf",
    )

    @staticmethod
    def _copy_field(name: str, value):
        """Copy mutable rollback state; share only checked immutable leaves.

        ``_ctx.card_pool`` is the whole card database -- a read-only lookup
        deepcopy was spending 120ms per call on (measured; it made every
        snapshot-engine activation cost more than a GPU model forward).
        The pool is shared by reference. Message lists also get their own
        container while exact frozen Message(int, bytes) leaves can be shared.
        Subclasses, extra fields and mutable payloads keep the deep-copy path.
        """
        import copy

        if name == "_ctx" and value is not None:
            clone = copy.copy(value)
            clone.rng = copy.deepcopy(value.rng)
            clone.known_codes = list(value.known_codes)
            return clone
        if name == "messages" and type(value) is list \
                and Message.__dataclass_params__.frozen \
                and Message.__dataclass_fields__.keys() == {"msg", "payload"} \
                and all(type(message) is Message and vars(message).keys() == {"msg", "payload"}
                        and type(message.msg) is int and type(message.payload) is bytes for message in value):
            return value.copy()
        return copy.deepcopy(value)

    def save_pystate(self) -> dict:
        """A copy of every field a rollback must restore."""
        unknown = (set(vars(self))
                   - set(self._PYSTATE_FIELDS) - set(self._PYSTATE_EXEMPT))
        if unknown:
            raise DuelError(
                f"DuelDriver has unclassified mutable fields {sorted(unknown)}: "
                "add them to _PYSTATE_FIELDS (rolled back) or _PYSTATE_EXEMPT "
                "(configuration / handles / temporary buffers), otherwise snapshot rollback drifts silently")
        return {name: self._copy_field(name, getattr(self, name))
                for name in self._PYSTATE_FIELDS if hasattr(self, name)}

    def restore_pystate(self, state: dict, *, consume: bool = False) -> None:
        """Put a saved state back; ``consume=True`` hands its objects over when the caller drops the dict."""
        for name, value in state.items():
            setattr(self, name, value if consume else self._copy_field(name, value))

    def build(self) -> "DuelDriver":
        """``create_duel`` -> ``new_card`` per deck card -> ``start_duel``."""
        self.core.reset_session()
        if self.config.forced_core_seeds is not None:
            seeds = list(self.config.forced_core_seeds)
        else:
            seed_rng = random.Random(("seeds", self.config.seed).__repr__())
            seeds = [seed_rng.getrandbits(32) for _ in range(8)]
        self.pduel = self.core.create_duel(seeds)
        mains = self.config.deck_orders()
        for player, deck in enumerate(self.config.decks):
            self.core.set_player_info(
                self.pduel,
                player,
                self.config.start_lp,
                self.config.start_hand,
                self.config.draw_count,
            )
            for code in mains[player]:
                self.core.new_card(
                    self.pduel, code, player, player, C.LOCATION_DECK, 0,
                    POS_FACEDOWN_DEFENSE,
                )
            # ygoenv adds the extra deck back to front; keep that, so the extra
            # zone sequence numbers line up with the environment's
            for code in reversed(deck.extra):
                self.core.new_card(
                    self.pduel, code, player, player, C.LOCATION_EXTRA, 0,
                    POS_FACEDOWN_DEFENSE,
                )
        # A missing/broken card script does not make new_card fail.  The core
        # logs the Lua error and keeps the card as a silent blank, which would
        # generate plausible-looking but false legality/state labels.  Corpus
        # generation must fail the shard loudly instead.
        if self.core.log:
            first = self.core.log[0].replace("\n", " ")[:240]
            raise DuelError(f"core card-load log: {first}")
        self.core.start_duel(self.pduel, self.config.duel_options)
        return self

    def close(self) -> None:
        if self.pduel is not None:
            self.core.end_duel(self.pduel)
            self.pduel = None

    # -- driving -----------------------------------------------------------

    def clear_log(self) -> None:
        self.messages = []

    def _observe(self, message: Message) -> None:
        if self.record_messages:
            self.messages.append(message)
        msg, body = message.msg, message.payload
        track_command_follow_up(self._ctx, msg, body)
        if msg == C.MSG_NEW_TURN:
            self.disclosure.observe_new_turn(body)
            self.turn += 1
            self.turn_player = body[0]
            if self.first_player is None:
                self.first_player = self.turn_player
        elif msg == C.MSG_NEW_PHASE:
            (self.phase,) = struct.unpack_from("<H", body, 0)
            self._ctx.current_phase = self.phase
        elif msg == C.MSG_HINT:
            hint_type = body[0]
            (value,) = struct.unpack_from("<I", body, 2)
            if hint_type == C.HINT_SELECTMSG and value == 501:
                self._ctx.discard_hand = True
        elif msg == C.MSG_WIN:
            self.winner = body[0]
            self.win_reason = body[1]
        elif msg == C.MSG_POS_CHANGE:
            self.disclosure.observe_pos_change(body)
        elif msg == C.MSG_SET:
            self.disclosure.observe_set(body)
        elif msg == C.MSG_SWAP:
            self.disclosure.observe_swap(body)
        elif msg == C.MSG_CHAINING:
            self.disclosure.observe_chaining(body)
        elif msg == C.MSG_CHAIN_SOLVING:
            self.disclosure.observe_chain_solving(body)
        elif msg == C.MSG_CHAIN_END:
            self.disclosure.observe_chain_end()
        elif msg == C.MSG_MOVE:
            self.disclosure.observe_move_message(body)
            self.banish_origin.observe(msg, body)
        elif msg == C.MSG_DRAW:
            self.disclosure.observe_draw(body)
        elif msg == C.MSG_DECK_TOP:
            self.disclosure.observe_deck_top(body)
        elif msg == C.MSG_REVERSE_DECK:
            self.disclosure.observe_reverse_deck()
        elif msg == C.MSG_SHUFFLE_SET_CARD:
            self.disclosure.observe_shuffle_set_card(body)
        elif msg in (C.MSG_SHUFFLE_DECK, C.MSG_SHUFFLE_HAND, C.MSG_SHUFFLE_EXTRA,
                     C.MSG_SWAP_GRAVE_DECK):
            self.disclosure.observe_shuffle(msg, body)
        elif msg in _CONFIRM_MSGS:
            self.disclosure.observe_confirm(body, msg)
        elif msg == C.MSG_RETRY:
            raise DuelError(
                "the core rejected the last response: "
                + self.last_response_context
            )

    def _respond(self, data: bytes) -> None:
        self.responses.append(bytes(data))
        name = _MSG_NAMES.get(getattr(self, "_answering_msg", -1),
                              getattr(self, "_answering_msg", -1))
        self.last_response_context = (
            f"{name} data={bytes(data).hex()} {self._pending_response_context}"
        ).strip()
        self.core.set_responseb(self.pduel, data)

    def _answer(self, message: Message, responder) -> None:
        msg, body = message.msg, message.payload
        self._answering_msg = msg
        #: raw payload of the prompt being answered: the R4 snapshot engine
        #: re-enters ``_answer`` from a rolled-back core, and rebuilding the
        #: selector needs the original message bytes
        self._answering_payload = bytes(body)
        self._pending_response_context = ""
        if body:
            self._ctx.our_player = (
                body[0] if msg != C.MSG_SELECT_SUM else body[1]
            )
        try:
            result = self.parse_prompt(msg, body)
        except UnsupportedMessage as exc:
            # No fallback answer: an unsupported prompt ends the game as an error.
            raise DuelError(f"{_MSG_NAMES.get(msg, msg)}: {exc}") from exc

        index = len(self.responses)
        if result.auto_response is not None:
            self.prompt_count += 1
            if self._replay:
                self._respond(self._replay.pop(0))
            else:
                self._respond(result.auto_response)
            return

        selector = result.selector
        if selector is None:
            raise DuelError(f"{_MSG_NAMES.get(msg, msg)} gave neither answer nor selector")

        # A recorded response covers the whole prompt, sub-selections included,
        # so a replayed prompt never reaches the responder at all.
        if self._replay:
            self.prompt_count += 1
            if self.on_replay_prompt is not None:
                actions = selector.options()
                if actions:
                    self.on_replay_prompt(Prompt(
                        index=index, msg=msg, player=selector.player,
                        turn=self.turn, turn_player=self.turn_player,
                        phase=self.phase, actions=actions, note=result.note,
                        complete_menu=bool(getattr(result, "complete_menu", True)),
                        selection=_selection_of(selector),
                    ), self)
            self._pending_response_context = "replay"
            self._respond(self._replay.pop(0))
            return

        chosen_actions = []
        for _round in range(self.config.max_sub_rounds):
            actions = selector.options()
            if not actions:
                raise DuelError(f"{_MSG_NAMES.get(msg, msg)} offers no legal action")
            truncated = (
                msg == C.MSG_ANNOUNCE_CARD and len(actions) >= self.config.max_options
            )
            if truncated:
                self.truncations += 1
            prompt = Prompt(
                index=index,
                msg=msg,
                player=selector.player,
                turn=self.turn,
                turn_player=self.turn_player,
                phase=self.phase,
                actions=actions,
                note=result.note,
                truncated=truncated,
                complete_menu=bool(getattr(result, "complete_menu", True)),
                selection=_selection_of(selector),
            )
            self.prompt_count += 1
            choice = int(responder(prompt, self))
            if not 0 <= choice < len(actions):
                raise DuelError(
                    f"responder chose {choice} of {len(actions)} for {prompt.name}"
                )
            selected = actions[choice]
            describe = getattr(selected, "describe", None)
            chosen_actions.append(
                describe() if callable(describe) else repr(selected)
            )
            data = selector.choose(choice)
            if data is not None:
                prefix = (result.note + " ") if result.note else ""
                self._pending_response_context = (
                    prefix + "choices=" + ",".join(chosen_actions)
                )
                self._respond(data)
                return
        # Getting here means sub-choices did not converge within max_sub_rounds. Raise DuelError instead of spinning on:
        # the caller records it as this duel's error and drops the duel, so the chunk can continue,
        # instead of the whole worker spinning forever and holding up every remaining chunk.
        # The triggering codes go along for the coverage ledger to aggregate by card: dropped duels must be visible, otherwise they are silent training blind spots.
        codes = tuple(sorted({
            int(getattr(a, "code", 0)) for a in actions if getattr(a, "code", 0)
        }))
        raise SubSelectionUnresolved(
            f"sub-choices of {_MSG_NAMES.get(msg, msg)} did not finish in {self.config.max_sub_rounds} "
            f"rounds (message {index}, turn {self.turn}); dropped as an unfinished duel",
            msg=msg, turn=self.turn, codes=codes,
        )

    def run(
        self,
        responder,
        max_steps: int = 40000,
        max_seconds: float | None = None,
    ) -> "DuelDriver":
        """Walk the duel until it ends, the responder stops it, or a cap trips.

        ``max_seconds`` is the **per-duel wall clock**, checked between steps; on timeout it raises
        `DuelDeadlineExceeded` (a subclass of `DuelError`; the generator records it as a dropped duel as usual).

        **It only stops runaways between steps, not an infinite loop inside a single `core.process()`.**
        During an infinite loop in the core the interpreter never gets control back (even `kill -INT` does nothing),
        and this check is never reached. That case **can only be handled by a process-level wall clock**:
        run the work driving the core in a killable child process, or use an external watchdog that kills on frozen syscalls.


        """
        if self.pduel is None:
            raise DuelError("build() first")
        started = _time.monotonic()
        for _ in range(max_steps):
            raw = self.core.process(self.pduel)
            self.steps += 1
            if raw & PROCESSOR_BUFFER_LEN:
                length = self.core.get_message(self.pduel, self._msgbuf)
                for message in split_messages(self._msgbuf.raw[:length]):
                    self._observe(message)
                    if message.msg in RESPONSE_REQUIRED:
                        self._answer(message, responder)
            if self.winner is not None:
                self.finished = True
                break
            if (raw & PROCESSOR_FLAG) == PROCESSOR_END:
                self.finished = True
                break
            if max_seconds is not None:
                elapsed = _time.monotonic() - started
                if elapsed > max_seconds:
                    raise DuelDeadlineExceeded(
                        f"this duel took {elapsed:.1f}s, more than the per-duel wall clock of {max_seconds:.1f}s "
                        f"(step {self.steps}, turn {self.turn}); dropped as an unfinished duel",
                        seconds=elapsed, steps=self.steps, turn=self.turn,
                        max_seconds=max_seconds,
                    )
        return self

    # -- branching ---------------------------------------------------------

    def fork(self, upto: int) -> "DuelDriver":
        """A fresh duel replayed to just before response ``upto``.

        The returned driver is at the same state this one was at when it had
        consumed ``upto`` responses; the next prompt reaches its responder.
        """
        branch = DuelDriver(self.config, self.core)
        branch.build()
        branch._replay = [bytes(r) for r in self.responses[:upto]]
        return branch

    def replay_remaining(self) -> int:
        return len(self._replay)


def first_responder(prompt: Prompt, driver: DuelDriver) -> int:
    return 0


def random_responder(seed: int):
    """Uniform over the options; the state-diversity policy for generation."""
    rng = random.Random(("responder", seed).__repr__())

    def choose(prompt: Prompt, driver: DuelDriver) -> int:
        return rng.randrange(len(prompt.actions))

    return choose


def mixed_responder(seed: int, greedy_prob: float = 0.35):
    """Random, but sometimes the first option.

    ygoenv orders the idle menu summon-first, so "index 0" is the built-in
    greedy heuristic; mixing it in pushes duels further into developed boards
    than uniform play reaches, which is where the interesting menus are.
    """
    rng = random.Random(("mixed", seed).__repr__())

    def choose(prompt: Prompt, driver: DuelDriver) -> int:
        if rng.random() < greedy_prob:
            return 0
        return rng.randrange(len(prompt.actions))

    return choose


def pass_responder(player: int, inner):
    """``inner``, except that ``player`` declines every optional response.

    This is the "opponent is forced to pass" condition: at a chain window the
    passing side takes the cancel option whenever the core offers one, so the
    next menu we see is a function of our own state and action alone.  Forced
    chains (the core offers no cancel) are still answered by ``inner``, because
    there is no passing them.
    """

    def choose(prompt: Prompt, driver: DuelDriver) -> int:
        if prompt.player == player and prompt.msg == C.MSG_SELECT_CHAIN:
            for i, action in enumerate(prompt.actions):
                if action.act.name == "CANCEL":
                    return i
        if prompt.player == player and prompt.msg in (
            C.MSG_SELECT_EFFECTYN,
            C.MSG_SELECT_YESNO,
        ):
            for i, action in enumerate(prompt.actions):
                if action.act.name == "CANCEL":
                    return i
        return inner(prompt, driver)

    return choose


@dataclass
class StopAt:
    """Responder wrapper that stops the walk at the first matching prompt.

    ``captured`` holds that prompt; the duel stays alive and un-answered, so the
    state at the prompt can be queried afterwards.
    """

    predicate: object
    inner: object
    captured: Prompt | None = field(default=None, init=False)

    def __call__(self, prompt: Prompt, driver: DuelDriver) -> int:
        if self.predicate(prompt):
            self.captured = prompt
            raise StopDuel
        return self.inner(prompt, driver)
