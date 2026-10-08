"""A local core that follows a network duel from what one seat receives.

A client sees what the host sends its seat: every public message, its own
draws and prompts, and ``MSG_WAITING`` while the opponent decides. Searching
from a client needs a core in the same state, so :class:`ClientSync` plays the
same duel in a local core and keeps it on the real one:

* the local duel starts like the real one: the same decks and rules, the
  seat's real opening hand on top of its deck, the opponent's hand and both
  deck orders one arbitrary assignment of the cards the seat cannot see;
* every message the local core writes is routed and masked for the seat as
  the host does it (:func:`~mirrorforce.netduel.host_view.deliver`) and must
  equal the next packet the seat received. Host refreshes are not compared:
  their omitted fields follow the host's query cache, not the game;
* the seat's own prompts must arrive byte for byte and are answered with the
  seat's real responses;
* an opponent prompt shows up as ``MSG_WAITING``; its answer is the first legal
  answer of the local prompt whose consequences match the real packets, tried
  from a snapshot one after another;
* a card the real packets reveal from a hidden place (the seat's own draws, an
  opponent's activation, move or confirmation) that the local core holds
  elsewhere is permuted into that place (``Debug.PermuteHidden``) and the core
  batch that revealed it is played again. A reveal inside an opponent prompt's
  answer is made before the prompt was written, so the prompt lists the card.

The local duel is a candidate history, not a certificate of search equivalence.
Matching packets does not prove that private bookkeeping or object relations
survived a hidden-object permutation. Engine bookkeeping comes from continuous
execution, but unsupported reveals and hidden histories still require rejection.
The opt-in blank hydration experiment is in ``client_shadow``.

Deployment-side: the input is the seat's packets and responses only. The local
core's hidden cards are a hypothesis this module made, not the opponent's.
"""

from __future__ import annotations

import ctypes
import io
import pickle
import random
import struct
from collections import Counter
from dataclasses import dataclass, field

from ..immutable import ImmutableRecord
from ..netduel import constants as C
from ..netduel import host_view
from ..netduel.wire_projection import _WAITING_PROMPTS
from ..puzzle.core import PROCESSOR_BUFFER_LEN, PROCESSOR_END, PROCESSOR_FLAG
from ..puzzle.messages import Message, split_messages
from ..puzzle.single import RESPONSE_REQUIRED
from ..search import particles as P
from ..search.snap_engine import snapshot_api
from ..worldmodel.engine import _MSG_NAMES, DeckList, DuelConfig, DuelDriver, DuelError

__all__ = ["ClientSync", "SyncError", "SyncStats", "REFRESH_MESSAGES"]

#: Host query results; they are not game messages and are not compared.
REFRESH_MESSAGES = frozenset({C.MSG_UPDATE_DATA, C.MSG_UPDATE_CARD})
#: Places whose card identities the seat cannot see on the opponent's side.
_HIDDEN_PLACES = frozenset({C.LOCATION_HAND, C.LOCATION_DECK, C.LOCATION_EXTRA})
#: Packets that move no card: a batch of only these left every hand slot where it was.
_STILL_PACKETS = frozenset({C.MSG_WAITING, C.MSG_HINT, C.MSG_NEW_PHASE, C.MSG_NEW_TURN})


class SyncError(RuntimeError):
    """The local duel cannot follow the real one from here."""


class RevealConflict(SyncError):
    """A reveal names a hidden place where the local duel holds another known card."""


class _Divergence(Exception):
    """A local packet differs from the real one at ``index`` of the seat's stream."""

    def __init__(self, index: int, real: bytes, local: bytes):
        super().__init__("divergence at packet %d" % index)
        self.index, self.real, self.local = index, real, local


class GroupAsked(SyncError):
    """A group operation already asked the seat about its first card: the order belongs before the operation."""


class GroupHeld(SyncError):
    """An operation in progress keeps a card set of its own that the core does not sort again (the opening draw):
    no group order can be set while it runs."""


class _GroupBeforeBatch(_Divergence):
    """The group order a divergence shows belongs before this batch: the operation asked the seat in the batch before
    (a zone prompt names one card and a card that needs none may come first), so only a run from before that prompt
    takes the cards in the server's order."""

    def __init__(self, divergence: _Divergence, uids):
        super().__init__(divergence.index, divergence.real, divergence.local)
        self.uids = tuple(uids)


class _EvidenceConflict(_Divergence):
    """The received packets show a hidden card at a batch start place where this branch holds another known card."""

    def __init__(self, index: int, reveal):
        super().__init__(index, b"", b"")
        self.reveal = reveal


class _Unread(_Divergence):
    """No answer of an opponent prompt is what the received packets show on this branch."""

    def __init__(self, index: int, msg: int, reason: str):
        super().__init__(index, b"", b"")
        self.msg, self.reason = msg, reason


class _Frontier(Exception):
    """The local duel wrote more than the seat has received so far."""


@dataclass
class SyncStats:
    batches: int = 0
    own_prompts: int = 0
    opponent_prompts: int = 0
    answers_tried: int = 0
    answers_ranked: int = 0
    replays: int = 0
    reveal_fixes: int = 0
    regenerated: int = 0
    replaced_cards: int = 0
    forced_shuffles: int = 0
    declined_prompts: int = 0
    alignment_blanked: int = 0
    alignment_hydrated: int = 0
    alignment_refused: int = 0
    #: revealed cards whose place at the batch start the packets do not tell, so no batch start takes them
    reveals_untraced: int = 0
    #: hidden cards and set proxies written at batch starts from the received packets before the batch ran
    evidence_placed: int = 0
    random_forced: int = 0
    groups_ordered: int = 0
    #: WAITINGs, invisible to the order-tolerant follower (plan 5.10): the server's skipped, and the local duel's that
    #: consumed nothing
    order_skipped: int = 0
    order_local: int = 0
    #: select-message hints only one side sends (the server's skipped, the local duel's that consumed nothing)
    hint_skipped: int = 0
    hint_local: int = 0
    #: category facts several hidden zones could hold (the proxy stands in the first)
    categories_open: int = 0
    #: opponent activations the local core was made to take as legal (plan 5.11)
    activations_forced: int = 0
    counts: Counter = field(default_factory=Counter)
    #: opponent prompts whose answer was read from the received packets, and those left to the answer search
    answers_read: Counter = field(default_factory=Counter)
    answers_searched: Counter = field(default_factory=Counter)
    #: single pass: prompts no answer of which the packets show
    answers_unread: Counter = field(default_factory=Counter)
    #: prompts with several answers proved through the packets to the seat's next prompt (an open decision)
    answers_open: Counter = field(default_factory=Counter)
    #: single pass: hidden picks whose allowed classes (a placeholder, known cards) the packets did not tell apart
    picks_equivalent: int = 0


@dataclass(frozen=True)
class FrozenPyState(ImmutableRecord):
    """A saved rollback state of the local duel's Python fields, frozen as bytes.

    A follower keeps one or two per open opponent choice (up to
    ``max_choices``). Frozen, a search root shares them by reference and
    digests each once, instead of copying and hashing every saved state at
    every root. The card pool is one shared read-only object: it is held out
    of the bytes and put back when the state is thawed.
    """

    blob: bytes

    @classmethod
    def freeze(cls, state: dict, pool) -> "FrozenPyState":
        buffer = io.BytesIO()
        pickler = pickle.Pickler(buffer, protocol=4)
        pickler.persistent_id = lambda value: "card-pool" if value is pool else None
        pickler.dump(state)
        return cls(buffer.getvalue())

    def thaw(self, pool) -> dict:
        """A fresh copy of the state, its card pool references bound to ``pool``."""
        unpickler = pickle.Unpickler(io.BytesIO(self.blob))

        def load(key):
            if key != "card-pool":
                raise SyncError("unknown shared object in a frozen duel state")
            return pool
        unpickler.persistent_load = load
        return unpickler.load()


class _Choice:
    """An opponent prompt the search answered: its snapshot, answers, and where it began."""

    def __init__(self, saved, origin, prompt: Message, answers: list[bytes], begin: int):
        self.saved, self.origin, self.prompt, self.answers, self.begin = saved, origin, prompt, answers, begin
        self.next = 0
        self.fixes = None
        #: why no answer was read, when none was
        self.unread = ""
        #: another alignment of the stream: the receipt of an omitted opponent pass that may own the
        #: WAITING this prompt took, and once the pass took it, that packet (``client_phase_pass``)
        self.pass_receipt = self.pass_taken = None
        #: every answer is proved to lead through the received packets to the seat's next prompt: the packets do
        #: not tell them apart there, and a later packet refuting the one taken takes the next
        self.open = False
        #: why the answer taken before the current one was given up (a single pass reports it with its failure)
        self.refuted = ""


class _FollowingDuel(DuelDriver):
    """A local duel whose every message must reproduce the seat's next packets."""

    _PYSTATE_EXEMPT = DuelDriver._PYSTATE_EXEMPT + ("follower",)

    def __init__(self, config: DuelConfig, core, follower: "ClientSync"):
        super().__init__(config, core)
        self.follower = follower
        self.disclosure = follower._new_disclosure()

    def _observe(self, message: Message) -> None:
        super()._observe(message)
        self.follower._match(message)


def _run_lua(driver: DuelDriver, source: str, name: str = "./script/mirrorforce-client-sync.lua") -> None:
    """Run a Lua chunk in the local duel, as particles are written (``preload_script``)."""
    core = driver.core
    raw = source.encode("utf-8")
    core.log.clear()
    core._script_cache[name] = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
    if not core.preload_script(driver.pduel, name) or core.log:
        raise SyncError("the local core refused a sync script: " + " | ".join(core.log[:2]))


def _prompt_player(message: Message) -> int:
    body = message.payload
    return body[1] if message.msg == C.MSG_SELECT_SUM else body[0]


def _reveals(packet: bytes, viewer: int) -> list[tuple[int, int, int, int]]:
    """``(controller, location, sequence, code)`` of cards a real packet shows in places the seat cannot see."""
    if not packet:
        return []
    msg, body = packet[0], packet[1:]
    out = []
    if msg == C.MSG_DRAW and len(body) >= 2:
        player, count = body[0], body[1]
        codes = [struct.unpack_from("<I", body, 2 + 4 * i)[0] & 0x7FFFFFFF for i in range(count)]
        # A draw takes the top card each time: the first drawn is the top now.
        if all(codes):
            out.append((player, C.LOCATION_DECK, -1, tuple(codes)))
    elif msg == C.MSG_MOVE and len(body) >= 12:
        code = struct.unpack_from("<I", body, 0)[0] & 0x7FFFFFFF
        controller, location, sequence = body[4], body[5], body[6]
        if code and location in _HIDDEN_PLACES and location != C.LOCATION_DECK and controller != viewer:
            out.append((controller, location, sequence, code))
        elif code and location == C.LOCATION_DECK and controller == viewer:
            # The seat does not know its own shuffled deck's order either: a card leaving it face up (a milled
            # top card, a searched or excavated card) shows which card stood at that deck place.
            out.append((controller, location, sequence, code))
    elif msg == C.MSG_CHAINING and len(body) >= 8:
        code = struct.unpack_from("<I", body, 0)[0] & 0x7FFFFFFF
        controller, location, sequence = body[4], body[5], body[6]
        if code and controller != viewer and location == C.LOCATION_HAND:
            out.append((controller, location, sequence, code))
    elif msg == C.MSG_CONFIRM_CARDS and len(body) >= 3:
        count = body[2]
        for i in range(count):
            off = 3 + 7 * i
            code = struct.unpack_from("<I", body, off)[0] & 0x7FFFFFFF
            controller, location, sequence = body[off + 4], body[off + 5], body[off + 6]
            if code and location in _HIDDEN_PLACES and (controller != viewer or location == C.LOCATION_DECK):
                out.append((controller, location, sequence, code))
    elif msg == C.MSG_CONFIRM_DECKTOP and len(body) >= 2:
        # Excavated deck-top cards are shown to both players, each with its deck place.
        for i in range(body[1]):
            off = 2 + 7 * i
            if len(body) < off + 7:
                break
            code = struct.unpack_from("<I", body, off)[0] & 0x7FFFFFFF
            controller, location, sequence = body[off + 4], body[off + 5], body[off + 6]
            if code and location == C.LOCATION_DECK:
                out.append((controller, location, sequence, code))
    return out


def _comparable(packet: bytes, viewer: int) -> bytes:
    """A packet as far as the local duel must reproduce it.

    A card leaving a deck names its deck place, which depends on the deck's
    order. That order is hidden from both players and is shuffled after a
    search, so the local duel keeps its own. Its face-down position only
    tells whether that card once went back to the deck, a trace of objects the
    local duel permutes by deck place, so it is not compared either. A move's
    reason is not compared (user, 2026-09-26: the local duel must reproduce
    the board, not the core's internal account of why a card moved): where
    the reason matters to the rules, a trigger that reads it, the packets
    after it differ. A card
    confirmation lists its cards in the order of the core's own card objects,
    which the local duel created in another order, so the list is compared as
    a set; so is a random selection, whose chosen deck cards are named by deck
    places and positions of that same hidden order. A select-message hint
    naming a system string only names the text a client shows above the
    prompt that follows it; a server whose scripts phrase that prompt with
    another string (2026-10-06 league server: 551 where the pinned scripts
    write 573) shows the same duel, and the prompt itself stays exact, so the
    string is not compared (a hint naming a card stays exact). The same
    server announces a select-unselect prompt's upper bound as 99 where the
    pinned core writes the script's own bound (3 at packet 665 of the first
    human final game); every card offered, both lists and the bound below
    stay exact, and the seat's index answer means the same card on both, so
    that one byte is not compared. Everything else stays exact.
    """
    if _selectmsg_hint(packet):
        return packet[:3] + b"\x00\x00\x00\x00"
    if packet and packet[0] == C.MSG_SELECT_UNSELECT_CARD and len(packet) >= 7:
        return packet[:5] + b"\x00" + packet[6:]
    if packet and packet[0] == C.MSG_MOVE and len(packet) >= 17:
        packet = packet[:13] + b"\xff\xff\xff\xff" + packet[17:]
        if packet[6] == C.LOCATION_DECK:
            return packet[:7] + b"\x00\x00" + packet[9:]
    if packet and packet[0] == C.MSG_CONFIRM_CARDS and len(packet) >= 4 and len(packet) == 4 + 7 * packet[3]:
        return packet[:4] + b"".join(sorted(packet[offset:offset + 7] for offset in range(4, len(packet), 7)))
    if packet and packet[0] == C.MSG_RANDOM_SELECTED and len(packet) >= 3 and len(packet) == 3 + 4 * packet[2]:
        places = (packet[offset:offset + 4] for offset in range(3, len(packet), 4))
        return packet[:3] + b"".join(sorted(place[:2] + b"\x00\x00" if place[1] == C.LOCATION_DECK else place
                                            for place in places))
    return packet


def _shuffle_orders(packet: bytes, viewer: int):
    """``(location, codes)`` a real packet tells the seat about a shuffle it must reproduce, if any.

    The seat reads its own hand after ``MSG_SHUFFLE_HAND``; its draws name its
    deck's top cards, which a shuffle earlier in the same core batch decided.
    """
    msg, body = packet[0], packet[1:]
    if msg == C.MSG_SHUFFLE_HAND and body[0] == viewer:
        count = body[1]
        return C.LOCATION_HAND, [struct.unpack_from("<I", body, 2 + 4 * i)[0] & 0x7FFFFFFF for i in range(count)]
    if msg == C.MSG_DRAW and body[0] == viewer:
        count = body[1]
        return C.LOCATION_DECK, [struct.unpack_from("<I", body, 2 + 4 * i)[0] & 0x7FFFFFFF for i in range(count)]
    return None


def _boundary(packet: bytes, viewer: int) -> bool:
    """A packet at which somebody decides: the opponent (``MSG_WAITING``) or the seat (its prompt)."""
    if packet[0] == C.MSG_WAITING:
        return True
    if packet[0] in RESPONSE_REQUIRED:
        return (packet[2] if packet[0] == C.MSG_SELECT_SUM else packet[1]) == viewer
    return False


#: consecutive core calls that write nothing the seat sees before a batch is refused (a few hints are normal)
_UNSEEN_CALLS = 64


def _selectmsg_hint(packet: bytes) -> bool:
    """A select-message hint naming a system string: the text a client shows above the prompt that follows it,
    nothing on the board. The 2026-10-06 league server (EDOPro) omits some of the ones the pinned core writes before
    a card selection, and phrases others with another string. A select-message hint that names a card (the zone
    prompt of a group names the card it places) is not one: it carries an identity and stays exact."""
    return len(packet) == 7 and packet[0] == C.MSG_HINT and packet[1] == C.HINT_SELECTMSG \
        and struct.unpack_from("<I", packet, 3)[0] < 0x10000


def _order_packet(packet: bytes) -> bool:
    """Whether a packet only shows the order in which the players were asked: the opponent asked something
    (``WAITING``). It changes nothing on the board, and a hidden card can change it (an optional trigger whose
    target is in a deck asks a player the other duel does not ask). The hints the seat receives before its own
    prompts are not order: they belong to the prompt (the zone prompt of a group names the card it places)."""
    return packet[:1] == bytes([C.MSG_WAITING])


class ClientSync:
    """Follow one seat of a real duel in a local core; see the module docstring.

    Feed every game packet the seat receives with :meth:`receive` and every
    response it sends with :meth:`respond`; :meth:`advance` then runs the local
    duel through all of it and returns the local prompt the seat must answer
    next (byte-identical to the real one), or ``None`` once the duel ended.
    """

    def __init__(self, *, viewer: int, decks: tuple[DeckList, DeckList], core, seed: int = 0,
                 start_lp: int = 8000, start_hand: int = 5, draw_count: int = 1, duel_options: int | None = None,
                 max_answers: int = 512, max_fixes: int = 8, max_seed_tries: int = 256, max_choices: int = 96, max_tries: int = 4000):
        if viewer not in (0, 1) or len(decks) != 2:
            raise ValueError("a client follows one of two seats with both decks known")
        self.viewer, self.decks, self.core, self.seed = viewer, tuple(decks), core, seed
        self.rules = {"start_lp": start_lp, "start_hand": start_hand, "draw_count": draw_count}
        if duel_options is not None:
            self.rules["duel_options"] = duel_options
        # ``max_seed_tries`` bounds the rounds of random outcomes one batch may take from the received packets.
        self.max_answers, self.max_fixes, self.max_seed_tries = max_answers, max_fixes, max_seed_tries
        #: the local core's own ``MSG_RANDOM_SELECTED`` of the running batch: (cursor, payload, whether the packet
        #: itself shows the outcome -- the host sends it to this seat and no chosen card lies in a deck)
        self.local_random: list[tuple[int, bytes, bool]] = []
        self.max_choices, self.max_tries = max_choices, max_tries
        self.spent = 0
        self.packets: list[bytes] = []
        self.cursor = 0
        self.own: list[bytes] = []
        self.answered = 0
        self.local: _FollowingDuel | None = None
        self.api = None
        self.parked: Message | None = None
        #: the state before the core batch that wrote the parked prompt, where hidden cards are placed
        #: when no answer of the prompt fits
        self.origin = None
        self.last_local: Message | None = None
        #: the opponent prompts the search answered, latest last (at most ``max_choices``)
        self.choices: list[_Choice] = []
        self.stats = SyncStats()

    # -- input ---------------------------------------------------------------

    def receive(self, packet: bytes) -> None:
        """One ``STOC_GAME_MSG`` payload as the seat received it."""
        packet = bytes(packet)
        if packet and packet[0] not in REFRESH_MESSAGES and packet[0] != C.MSG_START:
            self.packets.append(packet)

    def respond(self, data: bytes) -> None:
        """The seat's response to its latest prompt, as it was sent."""
        self.own.append(bytes(data))

    def close(self) -> None:
        """Release all retained snapshots before destroying the local duel."""
        self._drop_origin()
        for choice in self.choices:
            self._release(choice)
        self.choices.clear()
        if self.local is not None:
            self.local.close()

    # -- following -----------------------------------------------------------

    def advance(self) -> Message | None:
        """Run the local duel through everything received; return the seat's pending prompt, or None at the end.

        A search over the opponent's answers: every opponent prompt is a choice
        with a snapshot and the answers not tried yet. A divergence anywhere
        later (even after the seat's own next answer: the opponent chose the
        Battle or the End Phase, and only the phase after both passed shows
        which) goes back to the latest choice with an answer left and plays on
        from there, replaying the seat's own answers. A choice whose answers
        are spent is written again with the hidden cards its real answer named
        placed where it began, then given up.
        """
        if self.local is None:
            self._start()
        self.spent = 0
        while True:
            if self.parked is None:
                try:
                    self.parked, self.origin = self._run()
                except (_Divergence, _Frontier) as divergence:
                    self._backtrack(divergence)
                    continue
                if self.parked is None:
                    return None
            prompt = self.parked
            if _prompt_player(prompt) == self.viewer:
                if self.answered == len(self.own):
                    return prompt
                self._respond(prompt, self.own[self.answered])
                self.answered += 1
                self.stats.own_prompts += 1
                self._own_answered()
                self.parked = None
            else:
                self.stats.opponent_prompts += 1
                saved = self._save()
                choice = _Choice(saved, self.origin, prompt, [], self._prompt_begin())
                self.origin = None
                self.choices.append(choice)
                if len(self.choices) > self.max_choices:
                    self._release(self.choices.pop(0))
                self._prepare_choice(choice)
                if not choice.answers:
                    self._backtrack(_Unread(choice.begin, choice.prompt.msg, choice.unread))
                    continue
                self._try(choice)

    def _prepare_choice(self, choice: "_Choice") -> None:
        """Prepare one owned opponent prompt before enumerating its answers."""
        choice.answers = self._choice_answers(choice)

    def _choice_answers(self, choice: "_Choice") -> list[bytes]:
        """The answers of an opponent prompt, in the order they are tried."""
        return self._ranked(choice.prompt, choice.saved)

    def _try(self, choice: "_Choice") -> None:
        """Play the choice's next answer from its snapshot; the answer's successor is found by the next run."""
        self.spent += 1
        if self.spent > self.max_tries:
            raise SyncError("the local duel spent %d answers without following the real one (at packet %d of %d; "
                            "most frequent divergences %s)" % (self.spent, self.cursor, len(self.packets),
                                                               self.stats.counts.most_common(3)))
        self._load(choice.saved)
        data = choice.answers[choice.next]
        choice.next += 1
        self.stats.answers_tried += 1
        self._respond(choice.prompt, data)
        self.parked = None

    def _backtrack(self, divergence) -> None:
        """Go back to the latest choice with an answer left (writing spent ones again with hidden cards placed)."""
        if isinstance(divergence, _Divergence):
            self.stats.counts[(divergence.index, divergence.real.hex()[:40], divergence.local.hex()[:40])] += 1
        self._drop_origin()
        self.parked = None
        while self.choices:
            choice = self.choices[-1]
            if choice.next < len(choice.answers):
                self._try(choice)
                return
            if self._rewrite(choice):
                self._try(choice)
                return
            self._release(self.choices.pop())
        raise self._unfollowable(divergence)

    @staticmethod
    def _unfollowable(divergence) -> SyncError:
        """The failure a divergence leaves when nothing is left to go back to."""
        if isinstance(divergence, _Unread):
            return SyncError("the local duel cannot follow the real one: no answer of the opponent's %s at packet %d "
                             "is what the packets show (%s)" % (_MSG_NAMES.get(divergence.msg, divergence.msg),
                                                                divergence.index, divergence.reason))
        if isinstance(divergence, _EvidenceConflict):
            return SyncError("the local duel cannot follow the real one: the packets from %d show %s where the local "
                             "duel holds another known card" % (divergence.index, divergence.reveal))
        if isinstance(divergence, _Divergence):
            return SyncError("the local duel cannot follow the real one: at packet %d the local %s differs from the "
                             "real %s" % (divergence.index, divergence.local.hex()[:48], divergence.real.hex()[:48]))
        return SyncError("the local duel wrote more than the real duel sent")

    def _rewrite(self, choice: "_Choice") -> bool:
        """Write a spent choice's prompt again with the next placement of hidden cards; False when none is left."""
        if choice.origin is None:
            return False
        if choice.fixes is None:
            self._load(choice.origin)
            choice.fixes = list(self._fixes(choice.begin))
        while choice.fixes:
            fix = choice.fixes.pop(0)
            self._load(choice.origin)
            fixed = None
            try:
                self._apply(fix)
                fixed = self._save()
                prompt = self._fixed_batch(fixed, receipt_origin=choice.origin)
            except (_Divergence, _Frontier, P.ParticleError, SyncError):
                continue
            finally:
                self._free(fixed)
            if prompt is None or self.cursor - 1 != choice.begin or _prompt_player(prompt) == self.viewer:
                continue
            self.stats.regenerated += 1
            replacement = self._save()
            previous = choice.saved
            choice.saved, choice.prompt = replacement, prompt
            self._free(previous)
            choice.answers, choice.next = self._choice_answers(choice), 0
            if choice.answers:
                return True
        return False

    def _drop_origin(self) -> None:
        self._free(self.origin)
        self.origin = None

    def _own_answered(self) -> None:
        """The seat answered its parked prompt: the state before the batch that wrote it is no longer needed."""
        self._drop_origin()

    def _release(self, choice: "_Choice") -> None:
        self._free(choice.saved)
        self._free(choice.origin)

    def _run(self):
        """Run batches to the next prompt: ``(prompt, the state before the batch that wrote it)``, or (None, None)."""
        while not self.local.finished:
            saved = self._save()
            try:
                prompt = self._fixed_batch(saved)
            except BaseException:
                self._free(saved)
                raise
            if prompt is not None:
                return prompt, saved
            self._free(saved)
        return None, None

    def _fixes(self, begin: int):
        """Placements to try where a prompt began, from the real packets of its answer.

        First the cards the packets reveal; then, for each opponent card that
        left a hidden hand place face-down (which card it was stays hidden), each
        other card that could have been there.
        """
        reveals, hidden = [], []
        boundaries = 0
        for packet in self.packets[begin:begin + 64]:
            if _boundary(packet, self.viewer):
                boundaries += 1
                if boundaries > 2:
                    break
            reveals.extend(_reveals(packet, self.viewer))
            if packet[0] == C.MSG_MOVE and len(packet) >= 13:
                code = struct.unpack_from("<I", packet, 1)[0] & 0x7FFFFFFF
                controller, location, sequence = packet[5], packet[6], packet[7]
                if not code and controller != self.viewer and location == C.LOCATION_HAND:
                    hidden.append((controller, location, sequence, packet[10], packet[11]))
        if reveals:
            yield tuple(reveals)
        for controller, location, sequence, destination, place in hidden:
            layout = P.read_hidden_layout(self.local, controller)
            for code in sorted(set(layout.hand) | set(layout.deck)):
                if self._can_be_set(code, destination, place):
                    yield tuple(reveals) + ((controller, location, sequence, code),)

    def _can_be_set(self, code: int, location: int, sequence: int) -> bool:
        """Whether a card of this name can lie face-down where a hidden hand card went."""
        card = self.local._ctx.card_pool.cards.get(code)
        if card is None:
            return True
        kind = card.type
        if location == C.LOCATION_MZONE:
            return bool(kind & C.TYPE_MONSTER)
        if location == C.LOCATION_SZONE:
            if sequence == 5:
                return bool(kind & C.TYPE_FIELD)
            return bool(kind & (C.TYPE_SPELL | C.TYPE_TRAP)) and not kind & C.TYPE_FIELD
        return True

    def _apply(self, fix) -> None:
        for reveal in fix:
            self._place(*reveal)
            self.stats.reveal_fixes += 1

    def _new_disclosure(self):
        from ..netduel.disclosure import DisclosureLedger
        return DisclosureLedger()

    def _start(self) -> None:
        opening = self._opening_hand()
        own, other = self.decks[self.viewer], self.decks[1 - self.viewer]
        rest = list((Counter(own.main) - Counter(opening)).elements())
        if Counter(opening) - Counter(own.main):
            raise SyncError("the seat's opening hand is not in its deck")
        rng = random.Random(("client-sync", self.seed).__repr__())
        rng.shuffle(rest)
        orders = [None, None]
        # The first card drawn is the top of the deck, the last entry of the order.
        orders[self.viewer] = tuple(rest + list(reversed(opening)))
        theirs = list(other.main)
        rng.shuffle(theirs)
        orders[1 - self.viewer] = tuple(theirs)
        config = DuelConfig(self.decks, seed=self.seed, full_phase_menu=True, auto_end_phase_discard=False,
                            full_card_sort_menu=True, forced_deck_orders=(orders[0], orders[1]), **self.rules)
        self.local = _FollowingDuel(config, self.core, self)
        self.local.record_messages = True
        self.local.build()
        self.api = snapshot_api(self.core)

    def _opening_hand(self) -> list[int]:
        for packet in self.packets:
            if packet[0] == C.MSG_DRAW and packet[1] == self.viewer:
                count = packet[2]
                return [struct.unpack_from("<I", packet, 3 + 4 * i)[0] & 0x7FFFFFFF for i in range(count)]
        raise SyncError("the seat has not received its opening draw yet")

    def _match(self, message: Message) -> None:
        """Check one local message against the seat's next packets (called from the local duel's observer)."""
        self.last_local = message
        msg, payload = message.msg, message.payload
        if msg == C.MSG_RANDOM_SELECTED and len(payload) >= 2 and len(payload) == 2 + 4 * payload[1]:
            sent = host_view.deliver(msg, payload).payloads.get(self.viewer) is not None
            decked = any(payload[3 + 4 * i] == C.LOCATION_DECK for i in range(payload[1]))
            self.local_random.append((self.cursor, bytes(payload), sent and not decked))
        expected = []
        if msg in _WAITING_PROMPTS and _prompt_player(message) != self.viewer:
            expected.append(bytes([C.MSG_WAITING]))
        delivery = host_view.deliver(msg, payload)
        raw = delivery.payloads.get(self.viewer)
        if raw is not None:
            expected.extend([raw] * delivery.counts.get(self.viewer, 1))
        for packet in expected:
            while True:
                if self.cursor >= len(self.packets):
                    raise _Frontier()
                real = self.packets[self.cursor]
                if _selectmsg_hint(packet) != _selectmsg_hint(real):
                    # A select-message hint one side sends and the other does not shows the same duel.
                    if _selectmsg_hint(packet):
                        self.stats.hint_local += 1
                        break
                    self.cursor += 1
                    self.stats.hint_skipped += 1
                    continue
                if self._order_tolerant:
                    # Only the board and the seat's prompts must agree (plan 5.10): the WAITINGs, which show only the
                    # order in which the opponent was asked, are invisible on both sides. The local duel's consume
                    # nothing; the server's are skipped.
                    if _order_packet(packet):
                        self.stats.order_local += 1
                        break
                    if _order_packet(real):
                        self.cursor += 1
                        self.stats.order_skipped += 1
                        continue
                if _comparable(real, self.viewer) == _comparable(packet, self.viewer) \
                        or self._shuffled_deck_insertion(real, packet) \
                        or self._equivalent(real, packet, message):
                    self.cursor += 1
                    break
                raise _Divergence(self.cursor, real, packet)

    def _shuffled_deck_insertion(self, real: bytes, local: bytes) -> bool:
        """An insertion coordinate immediately erased by a received shuffle.

        Some server scripts send to deck-top then explicitly shuffle; the
        pinned script sends to DECKSHUFFLE then explicitly shuffles. Both
        reveal the same move, but initially insert at different deck slots.
        Do not generally ignore a destination deck position: require the
        *next* received non-refresh packet to shuffle that exact deck, with
        no draw, reveal or decision between. The ordinary matcher must still
        reproduce that shuffle and every later packet before admitting the
        seat's next root. No wire/history bytes or engine cards are changed.
        """
        if len(real) != 17 or len(local) != 17 or real[0] != C.MSG_MOVE or local[0] != C.MSG_MOVE \
                or real[6] == C.LOCATION_DECK or real[10] != C.LOCATION_DECK \
                or real[12] != C.POS_FACEDOWN or real[11] == local[11] \
                or real[:11] != local[:11] or real[12:] != local[12:] \
                or self.cursor + 1 >= len(self.packets):
            return False
        return real[9] in (0, 1) and self.packets[self.cursor + 1] == bytes([C.MSG_SHUFFLE_DECK, real[9]])

    def _equivalent(self, real: bytes, local: bytes, message: Message) -> bool:
        """Whether a local packet that differs from the received one still shows the seat the same duel."""
        return False

    def _prompt_begin(self) -> int:
        """The first received packet an opponent prompt just written can explain: its WAITING, which the exact
        follower has just matched; the order-tolerant follower's cursor (the WAITING consumed nothing)."""
        return self.cursor if self._order_tolerant else self.cursor - 1

    # -- snapshots ------------------------------------------------------------

    def _save(self):
        snap = self.api.duel_snapshot(self.local.pduel)
        if not snap:
            raise SyncError("the local core could not take a snapshot")
        return snap, FrozenPyState.freeze(self.local.save_pystate(), self.core.card_pool()), self.cursor, self.answered

    def _load(self, saved) -> None:
        snap, state, cursor, answered = saved
        if self.api.duel_rollback(self.local.pduel, snap) != 0:
            raise SyncError("the local core could not roll back")
        self.local.restore_pystate(state.thaw(self.core.card_pool()), consume=True)
        self.cursor, self.answered = cursor, answered

    def _free(self, saved) -> None:
        if saved is not None and saved[0]:
            self.api.duel_snapshot_free(saved[0])

    # -- running --------------------------------------------------------------

    def _batch(self) -> Message | None:
        """One core batch; the prompt it ends with, or None (the duel went on, or ended)."""
        local = self.local
        raw = local.core.process(local.pduel)
        local.steps += 1
        self.stats.batches += 1
        prompt = None
        if raw & PROCESSOR_BUFFER_LEN:
            length = local.core.get_message(local.pduel, local._msgbuf)
            for message in self._observed_messages(split_messages(local._msgbuf.raw[:length])):
                local._observe(message)
                if message.msg in RESPONSE_REQUIRED:
                    prompt = message
        if prompt is None and (local.winner is not None or (raw & PROCESSOR_FLAG) == PROCESSOR_END):
            local.finished = True
        return prompt

    def _observed_messages(self, messages):
        """Whole-batch observation hook; default preserves every native byte."""
        return messages

    def _fixed_batch(self, saved, *, receipt_origin=None) -> Message | None:
        """One batch; on a divergence at a reveal, the revealed cards are placed and the batch played again.

        Each replay starts from ``saved`` with every card placed so far, so a
        batch that shows several hidden cards one after another keeps them all.
        A reveal whose cards already sit where the batch started (a shuffle in
        the batch moved them) is retried with the seat's shuffle order forced.
        """
        placed = replaced = rounds = 0
        kept, forced, randomness, grouped = [], None, None, []
        origin = (saved[2], self._traced_sizes())
        placements, orders = self._batch_evidence(origin)
        targets = frozenset(reveal[:3] for reveal in placements)
        for reveal in placements:
            # Everything the received packets already show is written before the batch runs, once. Each is a card of
            # its own: none is taken from where another goes.
            try:
                if self._place(*reveal, keep=targets - {reveal[:3]}):
                    kept.append(reveal)
                    self.stats.evidence_placed += 1
            except RevealConflict:
                raise _EvidenceConflict(saved[2], reveal) from None
        for order in orders:
            self._force_shuffle(order)
        self._before_batch()

        def restart(divergence):
            """Back to the batch start with every kept card; a kept card that no longer fits is a divergence."""
            self._load(saved)
            try:
                for reveal in kept:
                    self._place(*reveal, keep=frozenset(other[:3] for other in kept) - {reveal[:3]})
            except RevealConflict:
                raise divergence from None
            for order in orders:
                self._force_shuffle(order)
            if forced is not None:
                self._force_shuffle(forced)
            if randomness is not None:
                self._force_random(randomness)
            self._before_batch()
            for uids in grouped:
                self._order_cards(uids)

        while True:
            self.local_random = []
            try:
                start, unseen = self.cursor, 0
                prompt = self._batch()
                while prompt is None and not self.local.finished and (
                        self.cursor == start and self._unseen_calls_join
                        or any(not shown for _cursor, _payload, shown in self.local_random)):
                    # A core call that wrote nothing this seat sees (a hint to the opponent) is no batch of its own:
                    # a phase's trigger collection writes such a hint before its prompt, and a hidden card placed
                    # after the collection comes too late for it. The real core never runs on unseen for long: a
                    # local core that does left the real duel's course.
                    unseen = unseen + 1 if self.cursor == start else 0
                    if unseen > _UNSEEN_CALLS:
                        raise SyncError("the local core wrote nothing the seat sees for %d calls" % _UNSEEN_CALLS)
                    # A random selection whose packet does not show the outcome (the host does not send it to this
                    # seat, or it chose deck cards, whose places are not comparable): only the chosen cards' later
                    # moves show it. The batch runs on to the next prompt, so that a divergence there comes back to
                    # before the selection.
                    prompt = self._batch()
                if forced is not None or orders:
                    # An order the batch did not use must not reach a later shuffle.
                    self._force_shuffle(None)
                if randomness is not None:
                    self._force_random(None)
                return prompt
            except _Divergence as divergence:
                if divergence.local == bytes([C.MSG_WAITING]) and replaced < self.max_fixes:
                    # The local opponent was asked something the real one was not: a hidden card of its
                    # hand gave it the chance, so that card was not in its real hand.
                    enabling = self._enabling_places(self.last_local)
                    if enabling:
                        restart(divergence)
                        replaced += 1
                        self._replace(1 - self.viewer, enabling, replaced)
                        self.stats.replaced_cards += 1
                        self.stats.replays += 1
                        continue
                outcome = self._random_evidence(divergence, saved[2], randomness) if rounds < self.max_seed_tries \
                    else None
                if outcome is not None:
                    # A coin, a die or a random selection came out differently: the received packets show how the
                    # real one came out, and the batch is played again with that outcome.
                    randomness = outcome
                    rounds += 1
                    restart(divergence)
                    self.stats.random_forced += 1
                    self.stats.replays += 1
                    continue
                shown = self._group_order_shown(divergence, saved[2])
                if shown is not None:
                    # One group operation processed its cards in another order than the server did: the order is
                    # the core's own, not a rule. The batch is played again with these cards in the server's order.
                    restart(divergence)
                    uids = self._start_uids(saved[2], shown)
                    if uids is None or uids in grouped or len(grouped) >= self.max_fixes:
                        raise
                    try:
                        self._order_cards(uids)
                    except GroupAsked:
                        raise _GroupBeforeBatch(divergence, uids) from None
                    grouped.append(uids)
                    self.stats.groups_ordered += 1
                    self.stats.replays += 1
                    continue
                if not self._late_reveals:
                    raise  # everything the packets show was written before the batch ran
                reveals = [reveal for reveal in self._reveals_ahead(divergence.index, origin=origin)
                           if reveal not in kept]
                order = _shuffle_orders(divergence.real, self.viewer) or self._deck_order_shown(divergence)
                if not reveals and order is None:
                    raise
                restart(divergence)
                try:
                    new = [reveal for reveal in reveals if self._place(*reveal)] if placed < self.max_fixes else []
                except RevealConflict:
                    # The local duel holds another known card where the real one showed this card: an earlier
                    # hidden choice went another way (the opponent took another card from its deck), which the
                    # search revisits.
                    raise divergence from None
                if new:
                    placed += 1
                    kept.extend(new)
                    self.stats.reveal_fixes += len(new)
                elif order is not None and forced is None:
                    forced = order
                    self._force_shuffle(order)
                    self.stats.forced_shuffles += 1
                else:
                    # Not a hidden card out of place: the divergence is a choice somebody made differently.
                    raise
                self.stats.replays += 1

    def _enabling_places(self, message: Message) -> list[int]:
        """The opponent's hand places whose cards a local prompt of the opponent offers to use."""
        if message is None or message.msg not in RESPONSE_REQUIRED or _prompt_player(message) == self.viewer:
            return []
        state = self.local.save_pystate()
        try:
            self.local._ctx.our_player = _prompt_player(message)
            result = self.local.parse_prompt(message.msg, message.payload)
            actions = result.selector.options() if result.selector is not None else []
        finally:
            self.local.restore_pystate(state)
        places = set()
        for action in actions:
            spec = getattr(action, "spec", "")
            if spec.startswith("h") and spec[1:].isdigit():
                places.add(int(spec[1:]) - 1)
        return sorted(places)

    def _replace(self, controller: int, places: list[int], attempt: int) -> None:
        """Trade the cards at these hand places for deck cards of other names (a different choice each attempt)."""
        layout = P.read_hidden_layout(self.local, controller)
        hand, deck = list(layout.hand), list(layout.deck)
        names = {hand[place] for place in places if place < len(hand)}
        spare = [index for index, code in enumerate(deck) if code not in names]
        rng = random.Random(repr(("client-sync-replace", self.seed, self.cursor, attempt)))
        rng.shuffle(spare)
        for place, index in zip((place for place in places if place < len(hand)), spare):
            hand[place], deck[index] = deck[index], hand[place]
        P.apply_particle(self.local, controller, P.HiddenLayout(hand=tuple(hand), deck=tuple(deck), facedown=()))

    def _force_shuffle(self, order) -> None:
        """Make the next shuffle of a hand or deck leave the order the real duel showed.

        ``(location, codes)`` is the seat's, ``(player, location, codes)`` either
        player's; a deck order names its top cards, the first the top. None
        clears every order.
        """
        if order is None:
            targets = [(player, location, ()) for player in (0, 1) for location in (C.LOCATION_HAND, C.LOCATION_DECK)]
        else:
            targets = [tuple(order) if len(order) == 3 else (self.viewer, *order)]
        lines = ["if not Debug.ForceShuffle(%d,%d,{%s}) then error('ForceShuffle refused') end"
                 % (player, location, ",".join(str(int(code)) for code in codes)) for player, location, codes in targets]
        _run_lua(self.local, "\n".join(lines) + "\n")

    def _deck_order_shown(self, divergence):
        """``(player, LOCATION_DECK, codes)`` a shuffle earlier in this batch must leave, when the real duel shows a
        deck card the local one does not have in the same deck place: the local duel moved another card out of the
        same place, or excavated other top cards. The cards the local shuffle left above that place stay; the real
        card, placed in the deck by the reveal fixes, joins them. None for any other divergence."""
        real, local = divergence.real, divergence.local
        if not real or not local or real[0] != local[0]:
            return None
        if real[0] == C.MSG_CONFIRM_DECKTOP and len(real) >= 3 and real[1:3] == local[1:3]:
            codes = [struct.unpack_from("<I", real, 3 + 7 * i)[0] & 0x7FFFFFFF for i in range(real[2])]
            return (real[1], C.LOCATION_DECK, codes) if all(codes) else None
        if real[0] != C.MSG_MOVE or len(real) < 17 or len(local) < 17 or real[5:8] != local[5:8] \
                or real[6] != C.LOCATION_DECK:
            return None
        code = struct.unpack_from("<I", real, 1)[0] & 0x7FFFFFFF
        player, sequence = real[5], real[7]
        # The local core has already moved its card: the cards above the place are now the top ``depth`` cards.
        deck = P.read_hidden_layout(self.local, player).deck
        depth = len(deck) - sequence
        if not code or not 0 <= depth <= len(deck):
            return None
        return player, C.LOCATION_DECK, [deck[len(deck) - 1 - i] for i in range(depth)] + [code]

    #: whether a divergence may place the hidden cards and shuffle orders it shows and run the batch again (else
    #: they must all have been written before the batch ran)
    _late_reveals = True
    #: whether a core call that wrote nothing the seat sees joins the next call's batch (else it is a batch of its
    #: own, and a divergence after it goes back only to the state it left)
    _unseen_calls_join = False
    #: whether the WAITINGs may differ (plan 5.10; the single-pass follower)
    _order_tolerant = False

    def _batch_evidence(self, origin):
        """Hidden cards to write where the batch starting at ``origin`` begins, and shuffle orders to force,
        before it runs; none here."""
        return [], []

    def _group_order_shown(self, divergence, begin: int):
        """The two cards of one group operation the divergence shows in another order, ``((packet, place), ...)`` in
        the server's order; None here (a subclass reads them)."""
        return None

    def _start_uids(self, begin: int, shown):
        """The local objects the ``(packet, place)`` pairs name, where the batch starting at ``begin`` began."""
        raise NotImplementedError

    def _order_cards(self, uids) -> None:
        """Make the local core process these objects in this order when they meet in one group."""
        raise NotImplementedError

    def _before_batch(self) -> None:
        """What a batch start puts back into the local core that its saved state may lack; nothing here."""

    def _traced_sizes(self) -> dict:
        """Zone sizes a subclass follows reveals back to the batch start with; none here."""
        return {}

    def _reveals_ahead(self, index: int, limit: int = 64, origin=None) -> list[tuple[int, int, int, int]]:
        """Reveals in the real packets from a divergence through the next decision's consequences.

        A local duel that lacks a hidden card diverges before the card shows:
        the opponent is asked to chain (``MSG_WAITING``) only if its hand holds a
        card it can chain, and the card shows after the answer. So the reveals
        are read from the divergence to the second decision boundary after it.
        """
        out, boundaries = [], 0
        for packet in self.packets[index:index + limit]:
            if _boundary(packet, self.viewer):
                boundaries += 1
                if boundaries > 1 and packet[0] != C.MSG_WAITING or boundaries > 2:
                    # The seat's own prompt still lists its deck cards; nothing after it.
                    out.extend(_reveals(packet, self.viewer))
                    break
            out.extend(_reveals(packet, self.viewer))
        return out

    def _place(self, controller, location, sequence, code, keep=frozenset()) -> bool:
        """Put ``code`` (a tuple of codes for the deck top) at a hidden place of the local duel; False if it is there.
        ``keep``: the places the other cards of the same evidence go to, which are distinct cards (this follower
        writes identities in place and moves no object, so it needs none)."""
        layout = P.read_hidden_layout(self.local, controller)
        hand, deck = list(layout.hand), list(layout.deck)
        if location == C.LOCATION_DECK and sequence == -1:
            wanted = list(code)
            top = [deck[len(deck) - 1 - i] for i in range(len(wanted))] if len(deck) >= len(wanted) else None
            if top == wanted:
                return False
            for depth, card in enumerate(wanted):
                index = len(deck) - 1 - depth
                if deck[index] == card:
                    continue
                donor = next((i for i in range(index - 1, -1, -1) if deck[i] == card), None)
                if donor is None:
                    donor_hand = next((i for i, value in enumerate(hand) if value == card), None)
                    if donor_hand is None or controller == self.viewer:
                        raise SyncError("a drawn card is nowhere in the local duel's hidden cards")
                    hand[donor_hand], deck[index] = deck[index], hand[donor_hand]
                else:
                    deck[donor], deck[index] = deck[index], deck[donor]
        elif location == C.LOCATION_HAND and sequence >= len(hand):
            # A hand place the batch has not filled yet: the card is drawn in it, from the deck top.
            return self._place(controller, C.LOCATION_DECK, -1, tuple(deck[len(deck) - 1 - i]
                               for i in range(sequence - len(hand))) + (code,))
        elif location == C.LOCATION_HAND:
            if hand[sequence] == code:
                return False
            donor = next((i for i in range(len(deck)) if deck[i] == code), None)
            if donor is not None:
                deck[donor], hand[sequence] = hand[sequence], deck[donor]
            else:
                donor = next((i for i in range(len(hand)) if hand[i] == code and i != sequence), None)
                if donor is None:
                    raise SyncError("a revealed hand card is nowhere in the local duel's hidden cards")
                hand[donor], hand[sequence] = hand[sequence], hand[donor]
        elif location == C.LOCATION_DECK:
            if sequence >= len(deck) or deck[sequence] == code:
                return False
            donor = next((i for i in range(len(deck)) if deck[i] == code and i != sequence), None)
            if donor is None:
                donor_hand = next((i for i, value in enumerate(hand) if value == code), None)
                if donor_hand is None or controller == self.viewer:
                    raise SyncError("a revealed deck card is nowhere in the local duel's hidden cards")
                hand[donor_hand], deck[sequence] = deck[sequence], hand[donor_hand]
            else:
                deck[donor], deck[sequence] = deck[sequence], deck[donor]
        else:
            return False
        P.apply_particle(self.local, controller,
                         P.HiddenLayout(hand=tuple(hand), deck=tuple(deck), facedown=()))
        return True

    def _random_evidence(self, divergence: _Divergence, start: int, forced):
        """The random outcomes a batch must take, read from the received packets, when a random outcome explains
        the divergence: ``(tosses, selections)`` extending ``forced``; None otherwise.

        ``tosses`` are the coin and die results of the batch in order (the
        received ``MSG_TOSS_COIN``/``MSG_TOSS_DICE`` carry them). ``selections``
        are ``Group.RandomSelect`` results in order, each the local locations
        of the chosen cards: the selections the local core made before the
        divergent one keep their local result, the divergent one takes the
        real one. The real result is the received ``MSG_RANDOM_SELECTED``, or,
        when that packet does not show it (the host does not send it to this
        seat, or deck places are not comparable), the chosen card's move after
        it.
        """
        real, local = divergence.real, divergence.local
        tosses_msgs = (C.MSG_TOSS_COIN, C.MSG_TOSS_DICE)
        tosses, selections = forced if forced is not None else ((), ())
        if real[:1] and real[0] in tosses_msgs and local[:1] == real[:1]:
            found = []
            for packet in self.packets[start:divergence.index + 1]:
                if packet[0] in tosses_msgs and len(packet) >= 3 and len(packet) == 3 + packet[2]:
                    found.extend(packet[3:])
            outcome = (tuple(found), selections)
            return outcome if outcome != forced else None
        if not self.local_random:
            return None
        cursor, payload, shown = self.local_random[-1]
        chosen = self._random_selection(divergence, cursor, payload, shown)
        if chosen is None:
            return None
        # The selections before the divergent one came out as the local ones did; they are named where they were.
        earlier = tuple(tuple((struct.unpack_from("<I", entry, 2 + 4 * i)[0] & 0xFFFFFF, 0) for i in range(entry[1]))
                        for _cursor, entry, _shown in self.local_random[:-1])
        outcome = (tosses, earlier + (chosen,))
        return outcome if outcome != forced else None

    def _random_selection(self, divergence: _Divergence, cursor: int, payload: bytes, shown: bool):
        """The cards the real random selection chose, as ``(local location, code)`` pairs (a card with a public
        identity is named by its code, since a shuffle may come between the selection and the card's move; copies
        are not told apart), or None when this divergence is not that selection coming out differently."""
        real, local = divergence.real, divergence.local
        if real[:1] == bytes([C.MSG_RANDOM_SELECTED]) and local[:1] == real[:1] and cursor == divergence.index:
            count = real[2] if len(real) >= 3 else 0
            if not count or len(real) != 3 + 4 * count:
                return None
            places = [tuple(real[3 + 4 * i:6 + 4 * i]) for i in range(count)]
            shown = [self._code_shown_after(divergence.index, place) for place in places]
        elif (not shown and payload[1] == 1 and real[:1] == local[:1] == bytes([C.MSG_MOVE]) and len(real) >= 13
              and len(local) >= 13 and real[1:5] != local[1:5] and tuple(real[5:7]) == tuple(local[5:7])):
            # The selection's packet did not show its outcome, and the cards the two duels then move out of the
            # same zone differ: the received move names the card the real selection chose. (Forcing it and
            # diverging the same way again gives no new outcome, and the divergence goes to the ordinary handling.)
            places = [tuple(real[5:8])]
            shown = [struct.unpack_from("<I", real, 1)[0] & 0x7FFFFFFF]
        else:
            return None
        chosen = []
        for (controller, location, sequence), code in zip(places, shown):
            chosen.append((0, code) if code else (controller | location << 8 | sequence << 16, 0))
        return tuple(chosen)

    def _code_shown_after(self, index: int, place, limit: int = 32) -> int:
        """The code of the first card the received packets move with a code out of ``place``'s zone after
        ``index`` (the exact place first), or 0."""
        moves = [packet for packet in self.packets[index + 1:index + 1 + limit]
                 if packet[0] == C.MSG_MOVE and len(packet) >= 13 and struct.unpack_from("<I", packet, 1)[0] & 0x7FFFFFFF]
        for packet in moves:
            if tuple(packet[5:8]) == tuple(place):
                return struct.unpack_from("<I", packet, 1)[0] & 0x7FFFFFFF
        for packet in moves:
            if tuple(packet[5:7]) == tuple(place[:2]):
                return struct.unpack_from("<I", packet, 1)[0] & 0x7FFFFFFF
        return 0

    def _force_random(self, outcome) -> None:
        """Make the batch's coin and die tosses and random selections come out as ``outcome``; None clears."""
        lib = self.core._lib
        pduel = self.local.pduel
        if lib.duel_clear_forced_random(ctypes.c_void_p(pduel)) != 0:
            raise SyncError("the local core refused to clear forced random outcomes")
        if outcome is None:
            return
        tosses, selections = outcome
        if tosses:
            values = (ctypes.c_uint8 * len(tosses))(*tosses)
            if lib.duel_force_random_outcomes(ctypes.c_void_p(pduel), len(tosses), values) != 0:
                raise SyncError("the local core refused forced toss results")
        for chosen in selections:
            places = (ctypes.c_uint32 * len(chosen))(*(place for place, _code in chosen))
            codes = (ctypes.c_uint32 * len(chosen))(*(code for _place, code in chosen))
            if lib.duel_force_random_select(ctypes.c_void_p(pduel), len(chosen), places, codes) != 0:
                raise SyncError("the local core refused a forced random selection")

    # -- answering -------------------------------------------------------------

    def _respond(self, message: Message, data: bytes) -> None:
        local = self.local
        local._answering_msg, local._answering_payload = message.msg, bytes(message.payload)
        local._respond(data)

    def _ranked(self, prompt: Message, saved, batches: int = 2) -> list[bytes]:
        """A prompt's answers, the one whose own batches follow the real packets furthest first.

        The opponent's answer never reaches the seat, but its consequences do:
        the batches it triggers write the packets the seat receives next. So
        each answer is played from the prompt for a couple of batches and kept
        with how much of the real stream it reproduced. Trying them in that
        order is what keeps a wrong answer high up from making every answer of
        every prompt below it be tried again.
        """
        answers = self._answers(prompt)
        if len(answers) < 2:
            return answers
        base, scored = saved[2], []
        for index, data in enumerate(answers):
            reached, followed, after = saved[2], False, None
            try:
                self._load(saved)
                self._respond(prompt, data)
                for _ in range(batches):
                    after = self._save()
                    try:
                        if self._fixed_batch(after) is not None or self.local.finished:
                            break
                    finally:
                        self._free(after)
                        after = None
                reached, followed = self.cursor, True
            except _Divergence as divergence:
                reached = divergence.index
            except (_Frontier, SyncError, DuelError, P.ParticleError):
                pass
            finally:
                self._free(after)
                if not followed:
                    self._force_shuffle(None)
            scored.append((base - reached, not followed, index, data))
        self._load(saved)
        scored.sort()
        self.stats.answers_ranked += len(answers)
        return [data for _, _, _, data in scored]

    def _answers(self, message: Message) -> list[bytes]:
        """Every complete answer of a local prompt, walking its selection rounds (at most ``max_answers``)."""
        local = self.local
        state = local.save_pystate()
        try:
            local._ctx.our_player = _prompt_player(message)
            first = local.parse_prompt(message.msg, message.payload)
            if first.auto_response is not None:
                return [bytes(first.auto_response)]
            out, seen, stack = [], set(), [()]
            while stack and len(out) < self.max_answers:
                prefix = stack.pop()
                if len(prefix) > local.config.max_sub_rounds:
                    continue
                selector = local.parse_prompt(message.msg, message.payload).selector
                data = None
                for choice in prefix:
                    data = selector.choose(choice)
                if data is not None:
                    data = bytes(data)
                    if data not in seen:
                        seen.add(data)
                        out.append(data)
                    continue
                options = selector.options()
                stack.extend(prefix + (index,) for index in reversed(range(len(options))))
            return out
        finally:
            local.restore_pystate(state)
