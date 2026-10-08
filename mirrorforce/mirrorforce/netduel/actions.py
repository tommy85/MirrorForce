"""Legal actions and response encoding, from the raw core message buffers.

This is the online twin of ``mirrorforce/mirrorforce/v1/replay/resim.py``.  Offline, the
re-simulator reads a recorded response and searches the action list for the one
that would have produced those bytes; online we go the other way -- pick an
action, encode the bytes -- but both sides must agree on *which list*, or an
index means different things to the policy and to the engine.

So this module is a port of the ``MSG_SELECT_*`` / ``MSG_ANNOUNCE_*`` branches
of ``YGOProEnvImpl::handle_message`` (``ygoenv/ygopro/ygopro.h``): same order of
candidates, same spec strings, same response bytes.  The message buffers arrive
over the wire inside ``STOC_GAME_MSG`` and are byte-identical to the ones ygoenv
reads locally, because the host links the same pinned core.

Two behaviours of the offline env are reproduced here on purpose:

* messages the env answers without asking a policy (an empty chain prompt, card
  sorting, counter placement, the end-phase hand discard) are answered the same
  way, so the two paths make the same number of decisions;
* multi-card selections are decomposed into single choices exactly as
  ``init_multi_select`` does, reusing :class:`~mirrorforce.replay.actions.MultiSelect`.
"""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass, field
from enum import IntEnum

from ..replay.actions import MultiSelect, encode_responsei, encode_selection
from . import constants as C

__all__ = [
    "COMMAND_COMMITMENT_LAW",
    "ActionAct",
    "ActionPhase",
    "LegalAction",
    "PromptSelection",
    "Reader",
    "SELECTION_ROLE_BY_MSG",
    "SelectionRole",
    "Selector",
    "SelectResult",
    "SelectContext",
    "UnsupportedMessage",
    "ls_to_spec",
    "spec_to_ls",
    "unpack_desc",
    "flag_to_usable_places",
    "parse_select",
    "track_command_follow_up",
]

#: Recorded wherever a run opts into ``SelectContext.commit_command_selections``.
COMMAND_COMMITMENT_LAW = "command_follow_up_selection_without_cancel/v1"

DESCRIPTION_LIMIT = 10000
CARD_EFFECT_OFFSET = 10010

OPCODE_ADD = 0x40000000
OPCODE_SUB = 0x40000001
OPCODE_MUL = 0x40000002
OPCODE_DIV = 0x40000003
OPCODE_AND = 0x40000004
OPCODE_OR = 0x40000005
OPCODE_NEG = 0x40000006
OPCODE_NOT = 0x40000007
OPCODE_ISCODE = 0x40000100
OPCODE_ISSETCARD = 0x40000101
OPCODE_ISTYPE = 0x40000102
OPCODE_ISRACE = 0x40000103
OPCODE_ISATTRIBUTE = 0x40000104


class UnsupportedMessage(RuntimeError):
    """A prompt the ported action space does not cover."""


class ActionAct(IntEnum):
    NONE = 0
    SET = 1
    REPO = 2
    SPSUMMON = 3
    SUMMON = 4
    MSET = 5
    ATTACK = 6
    DIRECT_ATTACK = 7
    ACTIVATE = 8
    CANCEL = 9


class ActionPhase(IntEnum):
    NONE = 0
    BATTLE = 1
    MAIN2 = 2
    END = 3


# ActionPlace (ygopro.h:989): 1..7 own monster zones, 8..15 own spell/trap
# zones, 16..22 opponent monster zones, 23..30 opponent spell/trap zones.
PLACE_MZONE1 = 1
PLACE_SZONE1 = 8
PLACE_OP_MZONE1 = 16
PLACE_OP_SZONE1 = 23


@dataclass
class LegalAction:
    """One entry of the action list, mirroring ``ygopro.h``'s ``LegalAction``."""

    spec: str = ""
    act: ActionAct = ActionAct.NONE
    phase: ActionPhase = ActionPhase.NONE
    finish: bool = False
    position: int = 0
    effect: int = -1
    number: int = 0
    place: int = 0
    attribute: int = 0
    race: int = 0
    code: int = 0  # raw card code behind the action, 0 when unknown
    response: int = 0  # MSG_ANNOUNCE_CARD answers with the code itself
    msg: int = 0
    #: The **full** description value given by the engine, not split by `unpack_desc`.
    #: `effect` keeps only the string index n of the low four bits, and n is not the effect index
    #: (measured inconsistent on 35.55% of activation candidates), while the dropped `desc >> 4` is exactly
    #: the lookup key of Stringids carrying another card's code. Effect identity is always looked up in desc_map with this raw value.
    desc: int = 0

    def describe(self) -> str:
        bits = []
        if self.spec:
            bits.append(self.spec)
        if self.act != ActionAct.NONE:
            bits.append(self.act.name.lower())
        if self.phase != ActionPhase.NONE:
            bits.append("to_" + self.phase.name.lower())
        if self.finish:
            bits.append("finish")
        if self.position:
            bits.append(f"pos{self.position}")
        if self.place:
            bits.append(f"place{self.place}")
        if self.number:
            bits.append(f"n{self.number}")
        if self.attribute:
            bits.append(f"attr{self.attribute}")
        if self.race:
            bits.append(f"race{self.race}")
        if self.effect >= 0:
            bits.append(f"eff{self.effect}")
        if self.response:
            bits.append(f"code{self.response}")
        return "|".join(bits) or "?"


class SelectionRole(IntEnum):
    """What the chooser's already-selected cards are in a card-selection prompt."""

    NONE = 0
    GENERIC = 1  # MSG_SELECT_CARD, MSG_SELECT_UNSELECT_CARD
    MATERIAL = 2  # MSG_SELECT_SUM
    TRIBUTE = 3  # MSG_SELECT_TRIBUTE


#: The card-selection prompts that carry a :class:`PromptSelection`, with the
#: role of the cards selected in them.  Every other prompt carries none.
SELECTION_ROLE_BY_MSG = {
    C.MSG_SELECT_CARD: SelectionRole.GENERIC,
    C.MSG_SELECT_UNSELECT_CARD: SelectionRole.GENERIC,
    C.MSG_SELECT_SUM: SelectionRole.MATERIAL,
    C.MSG_SELECT_TRIBUTE: SelectionRole.TRIBUTE,
}


@dataclass(frozen=True)
class PromptSelection:
    """The chooser's own progress through one card-selection prompt.

    A multi-card selection is answered in rounds, one menu per round, and the
    menu alone does not say what the earlier rounds picked.  This is that
    context, as the selector that owns the prompt knows it:

    * ``cards`` is every card the prompt names -- its candidates, then the
      ``MSG_SELECT_SUM`` must-select cards or the ``MSG_SELECT_UNSELECT_CARD``
      already-selected list -- as ``(spec, code)`` rows.  The tensorizer binds
      them to public slots once, so a card keeps its slot in every round;
    * ``selected`` indexes the chooser's current selection in ``cards``: the
      picks so far (plus the must-select cards of a sum, which the core fixed);
    * ``minimum``/``maximum`` bound the selection's size in the same units as
      ``selected``.  A tribute prompt's minimum is the core's, a *release
      weight* (``MultiSelect`` compares it to the summed weights);
    * ``value_target``/``value_low``/``value_high`` are the weighted total the
      prompt compares against its target: for ``MSG_SELECT_SUM`` the target
      value and the selected cards' smallest and largest possible sums (a card
      may count as either of two values), for ``MSG_SELECT_TRIBUTE`` the
      tribute minimum and the selected release weight; zero otherwise.

    Only the chooser's own selector produces it; an opponent's rounds never
    reach this player.
    """

    role: SelectionRole
    minimum: int
    maximum: int
    cards: tuple[LegalAction, ...]
    selected: tuple[int, ...]
    value_target: int = 0
    value_low: int = 0
    value_high: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.role, SelectionRole) or self.role is SelectionRole.NONE:
            raise ValueError("a prompt selection needs a card-selection role")
        numbers = (self.minimum, self.maximum, self.value_target, self.value_low, self.value_high)
        if any(type(value) is not int or value < 0 for value in numbers):
            raise ValueError("prompt selection counts must be non-negative integers")
        # Distinct card specs are a property of the message, checked where the
        # cards are bound to slots (the tensorizer), not while parsing it.
        if len(set(self.selected)) != len(self.selected) or any(
            type(index) is not int or not 0 <= index < len(self.cards) for index in self.selected
        ):
            raise ValueError("prompt selection indexes cards it does not name")

    def payload(self) -> dict:
        """The plain-data form the native ``public_model_input`` reads."""
        return {
            "role": int(self.role),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "cards": [{"spec": str(card.spec), "code": int(card.code)} for card in self.cards],
            "selected": list(self.selected),
            "value_target": self.value_target,
            "value_low": self.value_low,
            "value_high": self.value_high,
        }


def _selection_cards(msg: int, specs, codes) -> tuple[LegalAction, ...]:
    return tuple(LegalAction(spec=str(spec), code=int(code), msg=msg) for spec, code in zip(specs, codes))


class Reader:
    """Little-endian cursor over one core message buffer."""

    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.pos = pos
        # (code, controller, location, sequence) tuples the prompt listed; the
        # engine wrote them from the true board, so they cross-check the shadow
        self.cards: list = []

    def note_card(self, code: int, controller: int, loc: int, seq: int) -> None:
        if code:
            self.cards.append((code, controller, loc, seq))

    @property
    def left(self) -> int:
        return len(self.data) - self.pos

    def _take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise UnsupportedMessage(
                f"message buffer truncated: need {n} bytes at {self.pos}, "
                f"have {self.left}"
            )
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def u8(self) -> int:
        return self._take(1)[0]

    def u16(self) -> int:
        return struct.unpack("<H", self._take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self._take(4))[0]

    def skip(self, n: int) -> None:
        self._take(n)


def ls_to_spec(loc: int, seq: int, pos: int, opponent: bool = False) -> str:
    """ygopro.h:439 -- the textual card reference the action space keys on."""
    if loc & C.LOCATION_HAND:
        spec = "h"
    elif loc & C.LOCATION_MZONE:
        spec = "m"
    elif loc & C.LOCATION_SZONE:
        spec = "s"
    elif loc & C.LOCATION_GRAVE:
        spec = "g"
    elif loc & C.LOCATION_REMOVED:
        spec = "r"
    elif loc & C.LOCATION_EXTRA:
        spec = "x"
    else:
        spec = ""
    spec += str(seq + 1)
    if loc & C.LOCATION_OVERLAY:
        spec += chr(ord("a") + pos)
    return ("o" + spec) if opponent else spec


_LOC_OF_LETTER = {
    "h": C.LOCATION_HAND,
    "m": C.LOCATION_MZONE,
    "s": C.LOCATION_SZONE,
    "g": C.LOCATION_GRAVE,
    "r": C.LOCATION_REMOVED,
    "x": C.LOCATION_EXTRA,
}


def spec_to_ls(spec: str) -> tuple[bool, int, int, int]:
    """Inverse of :func:`ls_to_spec`: ``(opponent, location, sequence, overlay)``."""
    opponent = spec.startswith("o")
    rest = spec[1:] if opponent else spec
    loc = _LOC_OF_LETTER.get(rest[:1], 0)
    rest = rest[1:]
    digits = ""
    while rest and rest[0].isdigit():
        digits += rest[0]
        rest = rest[1:]
    seq = int(digits) - 1 if digits else 0
    overlay = ord(rest[0]) - ord("a") if rest else 0
    if rest:
        loc |= C.LOCATION_OVERLAY
    return opponent, loc, seq, overlay


def unpack_desc(code: int, desc: int) -> tuple[int, int]:
    """ygopro.h: split an effect description into ``(card code, effect index)``."""
    if desc < DESCRIPTION_LIMIT:
        # A generic system description does not encode a card identity.
        # Keep the identity supplied by the player-visible native prompt.
        return code, desc
    code_ = desc >> 4
    idx = desc & 0xF
    if idx >= 14:
        raise UnsupportedMessage(f"invalid effect index {idx} in desc {desc}")
    return code_, idx + CARD_EFFECT_OFFSET


def flag_to_usable_places(flag: int, reverse: bool = False) -> list[int]:
    """ygopro.h:1024 -- a zone bitmask becomes an ordered list of ActionPlace."""
    bases = (PLACE_MZONE1, PLACE_SZONE1, PLACE_OP_MZONE1, PLACE_OP_SZONE1)
    places = []
    for j in range(4):
        value = (flag >> (j * 8)) & 0xFF
        for i in range(8):
            avail = (value & (1 << i)) == 0
            if reverse:
                avail = not avail
            if avail:
                places.append(bases[j] + i)
    return places


def place_to_ls(place: int, player: int) -> tuple[int, int, int]:
    """``(controller, location, sequence)`` for a chosen ActionPlace."""
    if PLACE_MZONE1 <= place <= PLACE_MZONE1 + 6:
        return player, C.LOCATION_MZONE, place - PLACE_MZONE1
    if PLACE_SZONE1 <= place <= PLACE_SZONE1 + 7:
        return player, C.LOCATION_SZONE, place - PLACE_SZONE1
    if PLACE_OP_MZONE1 <= place <= PLACE_OP_MZONE1 + 6:
        return 1 - player, C.LOCATION_MZONE, place - PLACE_OP_MZONE1
    if PLACE_OP_SZONE1 <= place <= PLACE_OP_SZONE1 + 7:
        return 1 - player, C.LOCATION_SZONE, place - PLACE_OP_SZONE1
    raise UnsupportedMessage(f"unmapped place {place}")


def read_cardlist_spec(
    r: Reader, player: int, extra: bool = False, extra8: bool = False
) -> list[tuple[int, str, int]]:
    """ygopro.h: ``(code, spec, data)`` triples used by idle / battle menus."""
    out = []
    for _ in range(r.u8()):
        code = r.u32()
        controller = r.u8()
        loc = r.u8()
        seq = r.u8()
        data = 0
        if extra:
            data = r.u8() if extra8 else r.u32()
        r.note_card(code, controller, loc, seq)
        out.append((code, ls_to_spec(loc, seq, 0, player != controller), data))
    return out


# -- select_sum combination search (ygopro.h:92-134) ------------------------


def get_sum_params(param: int) -> list[int]:
    """The value(s) one card may contribute to a ``MSG_SELECT_SUM`` total.

    Port of ``field::get_sum_params`` (``ygopro-core/field.cpp``): the two
    halves are 16 bits each, *unless* the top bit of the high half is set, in
    which case the whole word is one 31-bit value and there is no second
    option.  Reading only ``param & 0xff`` -- which is what ygoenv's
    ``handle_message`` does -- truncates every value above 255.
    """
    op1 = param & 0xFFFF
    op2 = (param >> 16) & 0xFFFF
    if op2 & 0x8000:
        op1 = param & 0x7FFFFFFF
        op2 = 0
    return [value for value in (op1, op2) if value > 0]


def _sum_to2(weights: list[list[int]], ind: list[int], i: int, r: int) -> bool:
    if r <= 0:
        return False
    w = weights[ind[i]]
    if not w:
        # a card that can contribute nothing can never complete a sum
        return False
    if i == len(ind) - 1:
        return r in w
    if len(w) == 1:
        return _sum_to2(weights, ind, i + 1, r - w[0])
    return _sum_to2(weights, ind, i + 1, r - w[0]) or _sum_to2(
        weights, ind, i + 1, r - w[1]
    )


def combinations_with_weight2(
    weights: list[list[int]], r: int, mandatory: list[list[int]] = ()
) -> list[list[int]]:
    """Which subsets of ``weights`` complete a sum of exactly ``r``.

    ``mandatory`` are the ``MSG_SELECT_SUM`` must-select cards.  They are part
    of every candidate sum and, like the selectable ones, may contribute either
    of their two values -- ``select_sum_check1`` in ``ygopro-core`` walks
    must-select and selected cards through the same recursion.  Subtracting one
    fixed value per must-select card instead pins it to its first option and
    loses every line that needs the second.

    The returned indices are into ``weights`` alone.  An empty list is a legal
    answer when the must-select cards already reach ``r`` by themselves; it is
    only offered when there are must-select cards, so callers with none keep
    the old "at least one card" behaviour.
    """
    from itertools import combinations

    mandatory = list(mandatory)
    pool = mandatory + list(weights)
    base = list(range(len(mandatory)))
    offset = len(mandatory)
    n = len(weights)
    results = []
    if mandatory and _sum_to2(pool, base, 0, r):
        results.append([])
    for k in range(1, n + 1):
        for comb in combinations(range(n), k):
            ind = base + [offset + i for i in comb]
            if _sum_to2(pool, ind, 0, r):
                results.append(list(comb))
    return results


def combinations_with_sum_limit(
    weights: list[list[int]], r: int, mandatory: list[list[int]] = ()
) -> list[list[int]]:
    """Port ``select_with_sum_limit`` mode 1 (``max == 0``).

    A set is legal when some choice of each card's value can reach ``r`` and
    removing the smallest minimum contribution would fall below ``r``. This is
    the core's exact ``mx < acc || sum - mn >= acc`` rejection test.
    """
    from itertools import combinations

    mandatory = list(mandatory)
    out: list[list[int]] = []
    for count in range(0, len(weights) + 1):
        for comb in combinations(range(len(weights)), count):
            rows = mandatory + [weights[index] for index in comb]
            if not rows or any(not row for row in rows):
                continue
            minimums = [min(row) for row in rows]
            maximum = sum(max(row) for row in rows)
            if maximum >= r and sum(minimums) - min(minimums) < r:
                out.append(list(comb))
    return out


# -- announce card ----------------------------------------------------------


def parse_codes_from_opcodes(opcodes: list[int]) -> list[int] | None:
    """The plain "one of these card names" filter; ``None`` if it is richer."""
    n = len(opcodes)
    if n < 2 or ((n - 2) % 3) != 0 or opcodes[1] != OPCODE_ISCODE:
        return None
    codes = [opcodes[0]]
    for i in range(2, n, 3):
        if opcodes[i + 1] != OPCODE_ISCODE or opcodes[i + 2] != OPCODE_OR:
            return None
        codes.append(opcodes[i])
    return codes


def _check_setcode(setcode: int, value: int) -> bool:
    settype = value & 0x0FFF
    setsubtype = value & 0xF000
    return bool(
        setcode
        and (setcode & 0x0FFF) == settype
        and (setcode & setsubtype) == setsubtype
    )


def card_is_declarable(cd, opcodes: list[int]) -> bool:
    """Port of ``card_is_declarable`` (ygopro.h:185): the core's RPN filter."""
    if cd.alias:
        return False
    stack: list[int] = []

    def pop2():
        if len(stack) < 2:
            return None
        rhs = stack.pop()
        lhs = stack.pop()
        return lhs, rhs

    def pop1():
        return stack.pop() if stack else None

    for op in opcodes:
        if op == OPCODE_ADD:
            v = pop2()
            if v:
                stack.append(v[0] + v[1])
        elif op == OPCODE_SUB:
            v = pop2()
            if v:
                stack.append(v[0] - v[1])
        elif op == OPCODE_MUL:
            v = pop2()
            if v:
                stack.append(v[0] * v[1])
        elif op == OPCODE_DIV:
            v = pop2()
            if v:
                stack.append(v[0] // v[1] if v[1] else 0)
        elif op == OPCODE_AND:
            v = pop2()
            if v:
                stack.append(int(bool(v[0]) and bool(v[1])))
        elif op == OPCODE_OR:
            v = pop2()
            if v:
                stack.append(int(bool(v[0]) or bool(v[1])))
        elif op == OPCODE_NEG:
            v = pop1()
            if v is not None:
                stack.append(-v)
        elif op == OPCODE_NOT:
            v = pop1()
            if v is not None:
                stack.append(int(not v))
        elif op == OPCODE_ISCODE:
            v = pop1()
            if v is not None:
                stack.append(int(cd.code == (v & 0xFFFFFFFF)))
        elif op == OPCODE_ISSETCARD:
            v = pop1()
            if v is not None:
                stack.append(
                    int(any(_check_setcode(s, v & 0xFFFFFFFF) for s in cd.setcodes))
                )
        elif op == OPCODE_ISTYPE:
            v = pop1()
            if v is not None:
                stack.append(cd.type & (v & 0xFFFFFFFF))
        elif op == OPCODE_ISRACE:
            v = pop1()
            if v is not None:
                stack.append(cd.race & (v & 0xFFFFFFFF))
        elif op == OPCODE_ISATTRIBUTE:
            v = pop1()
            if v is not None:
                stack.append(cd.attribute & (v & 0xFFFFFFFF))
        else:
            stack.append(op if op < 0x80000000 else op - 0x100000000)
    if len(stack) != 1 or stack[-1] == 0:
        return False
    if cd.rule_code or (cd.type & 0x4000):  # TYPE_TOKEN
        return False
    return True


# -- selectors --------------------------------------------------------------


class Selector:
    """One core request; yields one or more choice points, then the response.

    ``options()`` is the action list for the current sub-step; ``choose(idx)``
    either returns the finished response bytes or ``None``, meaning the
    selection continues with a fresh ``options()``.  Non multi-select prompts
    always finish on the first choice.
    """

    _selection: PromptSelection | None = None

    def __init__(self, msg: int, player: int, actions: list[LegalAction], encode,
                 *, selection: PromptSelection | None = None):
        self.msg = msg
        self.player = player
        self._actions = actions
        self._encode = encode
        self._selection = selection
        for a in actions:
            a.msg = msg

    def options(self) -> list[LegalAction]:
        return self._actions

    def choose(self, idx: int) -> bytes | None:
        if not 0 <= idx < len(self._actions):
            raise IndexError(f"action {idx} out of range ({len(self._actions)})")
        return self._encode(idx, self._actions[idx])

    def selection(self) -> PromptSelection | None:
        """The card-selection context of the current round; ``None`` for any other prompt."""
        return self._selection


class MultiSelector(Selector):
    """``init_multi_select``: one recorded selection, many single choices.

    ``must`` are the ``MSG_SELECT_SUM`` must-select cards; ``values`` and
    ``must_values`` are the sum's per-card values (``get_sum_params``), and
    ``value_target`` its target.  They only feed :meth:`selection`.
    """

    def __init__(self, msg: int, player: int, ms: MultiSelect, codes: list[int],
                 *, cancelable: bool = False, must: tuple[LegalAction, ...] = (),
                 values: list[list[int]] | None = None, must_values: list[list[int]] | None = None,
                 value_target: int = 0):
        self.msg = msg
        self.player = player
        self.ms = ms
        self.codes = codes  # per spec index, for observation only
        self.cancelable = cancelable
        self.must = tuple(must)
        self.values = [list(row) for row in values or ()]
        self.must_values = [list(row) for row in must_values or ()]
        self.value_target = int(value_target)
        if len(self.must) != len(self.must_values) or (self.values and len(self.values) != len(ms.specs)):
            raise UnsupportedMessage("select_sum values disagree with its cards")

    def selection(self) -> PromptSelection:
        ms = self.ms
        role = SELECTION_ROLE_BY_MSG[self.msg]
        if len(ms.specs) != len(self.codes):
            raise UnsupportedMessage("selection context candidate identities disagree")
        cards = _selection_cards(self.msg, ms.specs, self.codes) + self.must
        selected = tuple(range(len(ms.specs), len(cards))) + tuple(ms.r_idxs)
        target = low = high = 0
        if role is SelectionRole.MATERIAL:
            rows = self.must_values + [self.values[index] for index in ms.r_idxs]
            target = self.value_target
            low = sum(min(row) for row in rows if row)
            high = sum(max(row) for row in rows if row)
        elif role is SelectionRole.TRIBUTE:
            target = ms.min
            low = high = sum(ms.weights[index] for index in ms.r_idxs)
        return PromptSelection(role, len(self.must) + ms.min, len(self.must) + ms.max, cards, selected,
                               target, low, high)

    def _can_cancel(self) -> bool:
        # Cancellation is a complete wire response, not a selected-card list.
        # Offer it before the first pick only: a later cancellation would erase
        # internal choices that cannot be reconstructed from the wire's -1.
        return self.cancelable and not self.ms.done and not self.ms.r_idxs

    def options(self) -> list[LegalAction]:
        out = []
        for spec in self.ms.options():
            if spec is None:
                out.append(LegalAction(finish=True, msg=self.msg))
            else:
                j = self.ms.spec2idx.get(spec, -1)
                code = self.codes[j] if 0 <= j < len(self.codes) else 0
                out.append(LegalAction(spec=spec, code=code, msg=self.msg))
        if self._can_cancel():
            out.append(LegalAction(act=ActionAct.CANCEL, msg=self.msg))
        return out

    def choose(self, idx: int) -> bytes | None:
        if self._can_cancel() and idx == len(self.ms.options()):
            self.ms.done = True
            self.ms.response = encode_responsei(-1)
            return self.ms.response
        self.ms.step(idx)
        return self.ms.response if self.ms.done else None


class PlaceSelector(Selector):
    """Select ``count`` distinct zones and emit the core's 3-byte triplets."""

    def __init__(self, msg: int, player: int, places: list[int], count: int):
        self.msg = msg
        self.player = player
        self.places = list(places)
        self.count = int(count)
        self.selected_mask = 0
        self.response = bytearray()

    def _mask(self, place: int) -> int:
        controller, location, sequence = place_to_ls(place, self.player)
        shift = sequence + (0 if controller == self.player else 16)
        shift += 0 if location == C.LOCATION_MZONE else 8
        mask = 1 << shift
        # The two Extra Monster Zones are shared across players. Match
        # field::select_place's duplicate suppression exactly.
        if mask & (1 << 5):
            mask |= 1 << 22
        if mask & (1 << 6):
            mask |= 1 << 21
        return mask

    def options(self) -> list[LegalAction]:
        return [
            LegalAction(place=place, msg=self.msg)
            for place in self.places
            if not (self._mask(place) & self.selected_mask)
        ]

    def choose(self, idx: int) -> bytes | None:
        actions = self.options()
        if not 0 <= idx < len(actions):
            raise IndexError(f"action {idx} out of range ({len(actions)})")
        place = actions[idx].place
        self.selected_mask |= self._mask(place)
        self.places.remove(place)
        self.response.extend(place_to_ls(place, self.player))
        return bytes(self.response) if len(self.response) == self.count * 3 else None


class OrderSelector(Selector):
    """Choose a complete permutation one remaining card at a time.

    ``MSG_SORT_CHAIN`` expects exactly ``count`` bytes containing every
    original row index once.  Exposing each next row as a normal selector round
    keeps the legal menu ragged and lets recurrent policies record the same
    forced final row as every other multi-round prompt.
    """

    def __init__(
        self,
        msg: int,
        player: int,
        cards: list[tuple[int, int, int, int]],
    ) -> None:
        self.msg = int(msg)
        self.player = int(player)
        self.cards = list(cards)
        self.remaining = list(range(len(cards)))
        self.order = bytearray()

    def options(self) -> list[LegalAction]:
        actions = []
        for original in self.remaining:
            code, controller, location, sequence = self.cards[original]
            actions.append(LegalAction(
                spec=ls_to_spec(
                    location,
                    sequence,
                    0,
                    controller != self.player,
                ),
                number=original,
                code=code,
                msg=self.msg,
            ))
        return actions

    def choose(self, idx: int) -> bytes | None:
        actions = self.options()
        if not 0 <= idx < len(actions):
            raise IndexError(f"action {idx} out of range ({len(actions)})")
        original = self.remaining.pop(idx)
        self.order.append(original)
        return bytes(self.order) if not self.remaining else None


class CardOrderSelector(OrderSelector):
    """Sort cards by destination rank, with the core's explicit cancel option.

    Unlike SORT_CHAIN, SORT_CARD serializes ``destination[original_index]``.
    Selection rounds still choose the next card in destination order.
    """

    def options(self) -> list[LegalAction]:
        rows = super().options()
        if self.remaining and not self.order:
            rows.append(LegalAction(act=ActionAct.CANCEL, msg=self.msg))
        return rows

    def choose(self, idx: int) -> bytes | None:
        if not 0 <= idx < len(self.options()):
            raise IndexError('card sort choice is outside the current menu')
        if not self.order and idx == len(self.remaining):
            self.remaining.clear()
            return b'\xff'
        original = self.remaining.pop(idx)
        self.order.append(original)
        if self.remaining:
            return None
        destinations = bytearray(len(self.cards))
        for rank, original in enumerate(self.order):
            destinations[original] = rank
        return bytes(destinations)


class BitmaskSelector(Selector):
    """Select several attribute/race bits, then emit their OR mask."""

    def __init__(self, msg: int, player: int, bits: list[int], count: int,
                 attribute: bool):
        self.msg = msg
        self.player = player
        self.bits = list(bits)
        self.count = int(count)
        self.attribute = bool(attribute)
        self.selected = 0
        self.picks = 0

    def options(self) -> list[LegalAction]:
        if self.attribute:
            return [
                LegalAction(attribute=bit, msg=self.msg)
                for bit in self.bits if not (bit & self.selected)
            ]
        return [
            LegalAction(race=bit, msg=self.msg)
            for bit in self.bits if not (bit & self.selected)
        ]

    def choose(self, idx: int) -> bytes | None:
        actions = self.options()
        if not 0 <= idx < len(actions):
            raise IndexError(f"action {idx} out of range ({len(actions)})")
        bit = actions[idx].attribute if self.attribute else actions[idx].race
        self.selected |= bit
        self.picks += 1
        return encode_responsei(self.selected) if self.picks == self.count else None


@dataclass
class SelectContext:
    """Everything a prompt may need beyond its own bytes."""

    our_player: int = 0
    current_phase: int = 0
    discard_hand: bool = False
    max_options: int = 24
    rng: random.Random = field(default_factory=random.Random)
    card_pool: object = None  # netduel.cards.CardPool, for MSG_ANNOUNCE_CARD
    known_codes: list = field(default_factory=list)
    #: WindBot may legally skip directly to the end phase even when battle or
    #: main phase 2 is also available.  The historical ygoenv action space
    #: hides that second phase action; exact external-teacher capture enables
    #: the complete engine menu explicitly without changing legacy corpora.
    full_phase_menu: bool = False
    #: ygoenv discards random hand cards automatically at the end phase.  A
    #: live external teacher made a real private selection, so capture and
    #: replay expose that selection instead of fabricating ygoenv's answer.
    auto_end_phase_discard: bool = True
    #: Exact arbitrary-deck capture must retain real card-order choices. The
    #: default preserves historical corpora that automatically decline sorting.
    full_card_sort_menu: bool = False
    #: A card or tribute selection that completes this player's own idle or
    #: battle command (the attack target before the declaration, tributes,
    #: summon materials) returns the core to that same command menu when
    #: cancelled, with nothing public in between.  The cancel adds no reachable
    #: position, only a way to loop until the host clock ends; opted-in clients
    #: do not offer it. Historical menus keep it; both drivers carry the flag.
    commit_command_selections: bool = False
    #: Maintained by :func:`track_command_follow_up`: the player whose command
    #: answer has so far produced only hints and its own follow-up selections.
    command_player: int | None = None


@dataclass
class SelectResult:
    """Either an immediate answer or a selector that needs a policy.

    ``cards`` carries the ``(code, controller, location, sequence)`` tuples the
    prompt itself listed.  The engine writes those from the true board, so they
    are a free ground-truth sample to check the shadow board against -- see
    ``ShadowBoard.cross_check``.  Codes of the opponent's cards arrive as 0
    (the host blanks them), so only non-zero entries carry identity.
    """

    msg: int
    player: int
    auto_response: bytes | None = None
    selector: Selector | None = None
    note: str = ""
    cards: list = field(default_factory=list)
    #: True only when the parser exposed every phase action carried by the
    #: engine message.  Legacy ygoenv compatibility may hide END when another
    #: phase transition is also available; exact-menu consumers reject that
    #: representation even though it is not a size truncation.
    complete_menu: bool = True


#: What the core writes between an idle/battle command answer and the
#: selections that complete it; the host adds nothing there.  Any other message
#: is a public consequence: a declaration, a summon, a chain or a new phase.
_COMMAND_FOLLOW_UP = frozenset({
    C.MSG_HINT, C.MSG_SELECT_YESNO, C.MSG_SELECT_OPTION, C.MSG_SELECT_CARD, C.MSG_SELECT_TRIBUTE,
    C.MSG_SELECT_UNSELECT_CARD, C.MSG_SELECT_SUM, C.MSG_SELECT_PLACE, C.MSG_SELECT_POSITION,
})


def track_command_follow_up(ctx: SelectContext, msg: int, body: bytes) -> None:
    """Update ``ctx.command_player``; call for every game message, in order, before parsing it."""
    if msg in (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD):
        ctx.command_player = body[0]
    elif msg not in _COMMAND_FOLLOW_UP:
        ctx.command_player = None


def _encode_i(value: int):
    def enc(idx, action):
        return encode_responsei(value)

    return enc


def parse_select(msg: int, payload: bytes, ctx: SelectContext) -> SelectResult:
    """Turn one ``MSG_SELECT_*`` / ``MSG_ANNOUNCE_*`` buffer into a decision.

    ``payload`` excludes the leading message id byte.
    """
    r = Reader(payload)

    if msg == C.MSG_SELECT_BATTLECMD:
        player = r.u8()
        activatable = read_cardlist_spec(r, player, True)
        attackable = read_cardlist_spec(r, player, True, True)
        to_m2 = bool(r.u8())
        to_ep = bool(r.u8())
        actions: list[LegalAction] = []
        for code_t, spec, desc in activatable:
            code = code_t & 0x7FFFFFFF
            code_d, eff_idx = unpack_desc(code, desc)
            if desc == 0:
                code_d = code
            actions.append(
                LegalAction(spec=spec, act=ActionAct.ACTIVATE, effect=eff_idx, desc=desc, code=code_d)
            )
        for code, spec, data in attackable:
            act = ActionAct.DIRECT_ATTACK if (data & 0x1) else ActionAct.ATTACK
            actions.append(LegalAction(spec=spec, act=act, code=code))
        if to_m2:
            actions.append(LegalAction(phase=ActionPhase.MAIN2))
        if to_ep and (ctx.full_phase_menu or not to_m2):
            actions.append(LegalAction(phase=ActionPhase.END))
        n_act, n_atk = len(activatable), len(attackable)
        listed = list(r.cards)

        def enc(idx, action):
            if idx < n_act:
                return encode_responsei(idx << 16)
            if idx < n_act + n_atk:
                return encode_responsei(((idx - n_act) << 16) + 1)
            if action.phase == ActionPhase.END and to_ep:
                return encode_responsei(3)
            if action.phase == ActionPhase.MAIN2 and to_m2:
                return encode_responsei(2)
            raise UnsupportedMessage("invalid battle command option")

        return SelectResult(
            msg,
            player,
            selector=Selector(msg, player, actions, enc),
            cards=listed,
            complete_menu=bool(ctx.full_phase_menu or not (to_m2 and to_ep)),
        )

    if msg == C.MSG_SELECT_IDLECMD:
        player = r.u8()
        summonable = read_cardlist_spec(r, player)
        spsummon = read_cardlist_spec(r, player)
        repos = read_cardlist_spec(r, player)
        idle_mset = read_cardlist_spec(r, player)
        idle_set = read_cardlist_spec(r, player)
        idle_activate = read_cardlist_spec(r, player, True)
        to_bp = bool(r.u8())
        to_ep = bool(r.u8())
        r.u8()  # can_shuffle
        actions = []
        groups = (
            (summonable, ActionAct.SUMMON, 0),
            (spsummon, ActionAct.SPSUMMON, 1),
            (repos, ActionAct.REPO, 2),
            (idle_mset, ActionAct.MSET, 3),
            (idle_set, ActionAct.SET, 4),
        )
        offsets = {}
        for cards, act, sub in groups:
            offsets[act] = len(actions)
            for code, spec, _ in cards:
                actions.append(LegalAction(spec=spec, act=act, code=code))
        offsets[ActionAct.ACTIVATE] = len(actions)
        for code_t, spec, desc in idle_activate:
            code = code_t & 0x7FFFFFFF
            code_d, eff_idx = unpack_desc(code, desc)
            if desc == 0:
                code_d = code
            actions.append(
                LegalAction(spec=spec, act=ActionAct.ACTIVATE, effect=eff_idx, desc=desc, code=code_d)
            )
        if to_bp:
            actions.append(LegalAction(phase=ActionPhase.BATTLE))
        if to_ep and (ctx.full_phase_menu or not to_bp):
            actions.append(LegalAction(phase=ActionPhase.END))
        subcode = {
            ActionAct.SUMMON: 0,
            ActionAct.SPSUMMON: 1,
            ActionAct.REPO: 2,
            ActionAct.MSET: 3,
            ActionAct.SET: 4,
            ActionAct.ACTIVATE: 5,
        }

        def enc(idx, action):
            if action.phase == ActionPhase.BATTLE:
                return encode_responsei(6)
            if action.phase == ActionPhase.END:
                return encode_responsei(7)
            sub = subcode[action.act]
            return encode_responsei(((idx - offsets[action.act]) << 16) + sub)

        return SelectResult(
            msg,
            player,
            selector=Selector(msg, player, actions, enc),
            cards=list(r.cards),
            complete_menu=bool(ctx.full_phase_menu or not (to_bp and to_ep)),
        )

    if msg == C.MSG_SELECT_UNSELECT_CARD:
        player = r.u8()
        finishable = bool(r.u8())
        r.u8()  # cancelable
        min_ = r.u8()
        max_ = r.u8()
        select_size = r.u8()
        specs = []
        codes = []
        for _ in range(select_size):
            codes.append(r.u32())
            controller = r.u8()
            loc = r.u8()
            seq = r.u8()
            pos = r.u8()
            r.note_card(codes[-1], controller, loc, seq)
            specs.append(ls_to_spec(loc, seq, pos, controller != player))
        # The core runs this prompt's rounds itself and lists the cards chosen
        # so far; unselecting one is never offered, but they are the selection.
        unselect_size = r.u8()
        chosen_specs, chosen_codes = [], []
        for _ in range(unselect_size):
            chosen_codes.append(r.u32())
            controller = r.u8()
            loc = r.u8()
            seq = r.u8()
            pos = r.u8()
            chosen_specs.append(ls_to_spec(loc, seq, pos, controller != player))
        actions = [LegalAction(spec=s, code=c) for s, c in zip(specs, codes)]
        if finishable:
            actions.append(LegalAction(finish=True))
        selection = PromptSelection(
            SelectionRole.GENERIC, min_, max_,
            _selection_cards(msg, specs + chosen_specs, codes + chosen_codes),
            tuple(range(select_size, select_size + unselect_size)),
        )

        def enc(idx, action):
            if action.finish:
                return encode_responsei(-1)
            return bytes([1, idx])

        return SelectResult(msg, player, selector=Selector(msg, player, actions, enc, selection=selection))

    if msg in (C.MSG_SELECT_CARD, C.MSG_SELECT_TRIBUTE):
        player = r.u8()
        cancelable = bool(r.u8()) and not (ctx.commit_command_selections and ctx.command_player == player)
        min_ = r.u8()
        max_ = r.u8()
        size = r.u8()
        specs, codes = [], []
        release_params = [1] * size
        if msg == C.MSG_SELECT_CARD:
            for _ in range(size):
                codes.append(r.u32())
                controller = r.u8()
                loc = r.u8()
                seq = r.u8()
                pos = r.u8()
                r.note_card(codes[-1], controller, loc, seq)
                specs.append(ls_to_spec(loc, seq, pos, controller != player))
        else:
            release_params = []
            for _ in range(size):
                codes.append(r.u32())
                controller = r.u8()
                loc = r.u8()
                seq = r.u8()
                release_params.append(r.u8())
                r.note_card(codes[-1], controller, loc, seq)
                specs.append(ls_to_spec(loc, seq, 0, controller != player))

        if msg == C.MSG_SELECT_CARD and ctx.discard_hand:
            ctx.discard_hand = False
            if ctx.current_phase == C.PHASE_END and ctx.auto_end_phase_discard:
                comb = list(range(size))
                ctx.rng.shuffle(comb)
                return SelectResult(
                    msg,
                    player,
                    auto_response=encode_selection(sorted(comb[:min_])[:min_]),
                    note="end-phase hand discard",
                )

        ms = MultiSelect(
            min=min_, max=max_, specs=specs, weights=release_params
        )
        return SelectResult(
            msg,
            player,
            selector=MultiSelector(msg, player, ms, codes, cancelable=cancelable),
            cards=list(r.cards),
        )

    if msg == C.MSG_SELECT_SUM:
        mode = r.u8()
        player = r.u8()
        val = r.u32()
        min_ = r.u8()
        max_ = r.u8()
        must_size = r.u8()
        if mode not in (0, 1):
            raise UnsupportedMessage(f"select_sum mode {mode} not implemented")
        # ``min`` is an int32 written through write_buffer8: operations.cpp
        # queues the process with ``min - (mcount - 1)``, so two must-select
        # cards and a minimum of zero arrive here as 0xff.  Read it back as
        # signed and clamp, or the selection looks like it needs 255 cards.
        if min_ > 0x7F:
            min_ -= 0x100
        min_ = max(min_, 0)
        must_levels = []
        must_cards = []
        for _ in range(must_size):
            must_code = r.u32()
            controller = r.u8()
            loc = r.u8()
            seq = r.u8()
            must_levels.append(get_sum_params(r.u32()))
            must_cards.append(LegalAction(spec=ls_to_spec(loc, seq, 0, controller != player),
                                          code=must_code, msg=msg))
        select_size = r.u8()
        specs, codes, params = [], [], []
        for _ in range(select_size):
            codes.append(r.u32())
            controller = r.u8()
            loc = r.u8()
            seq = r.u8()
            params.append(r.u32())
            r.note_card(codes[-1], controller, loc, seq)
            specs.append(ls_to_spec(loc, seq, 0, controller != player))
        card_levels = [get_sum_params(p) for p in params]
        if mode == 0:
            combs = [
                sorted(c)
                for c in combinations_with_weight2(card_levels, val, must_levels)
                # Core validates the response count against min/max in addition
                # to the exact weighted sum. ``c`` contains optional cards only;
                # the response encoder adds the mandatory count separately.
                if min_ <= len(c) <= max_
            ]
        else:
            combs = [
                sorted(c)
                for c in combinations_with_sum_limit(
                    card_levels, val, must_levels
                )
            ]
        ms = MultiSelect(
            min=min_, max=max_, must=must_size, specs=specs, mode=1, combs=combs
        )
        return SelectResult(
            msg,
            player,
            selector=MultiSelector(msg, player, ms, codes, must=tuple(must_cards), values=card_levels,
                                   must_values=must_levels, value_target=val),
            cards=list(r.cards),
            note=(
                f"sum mode={mode} value={val} min={min_} max={max_} "
                f"must={must_levels} optional={card_levels}"
            ),
        )

    if msg == C.MSG_SELECT_CHAIN:
        player = r.u8()
        size = r.u8()
        spe_count = r.u8()
        r.skip(8)  # hint_timing, other_timing
        forced = False
        entries = []
        for _ in range(size):
            r.u8()  # flag
            forced |= r.u8() != 0  # per-entry since ygopro-core #753
            code = r.u32()
            c = r.u8()
            loc = r.u8()
            seq = r.u8()
            pos = r.u8()
            desc = r.u32()
            r.note_card(code, c, loc, seq)
            entries.append((code, ls_to_spec(loc, seq, pos, c != player), desc))
        if size == 0 and spe_count == 0:
            return SelectResult(
                msg, player, auto_response=encode_responsei(-1), note="empty chain"
            )
        actions = []
        for code, spec, desc in entries:
            code_d, eff_idx = unpack_desc(code, desc)
            if desc == 0:
                code_d = code
            actions.append(
                LegalAction(spec=spec, act=ActionAct.ACTIVATE, effect=eff_idx, desc=desc, code=code_d)
            )
        if not forced:
            actions.append(LegalAction(act=ActionAct.CANCEL))

        def enc(idx, action):
            if action.act == ActionAct.CANCEL:
                return encode_responsei(0 if forced else -1)
            return encode_responsei(idx)

        return SelectResult(
            msg,
            player,
            selector=Selector(msg, player, actions, enc),
            cards=list(r.cards),
        )

    if msg == C.MSG_SELECT_YESNO:
        player = r.u8()
        desc = r.u32()
        if desc == 0:
            raise UnsupportedMessage("unknown desc 0 in select_yesno")
        code, eff_idx = unpack_desc(0, desc)
        actions = [
            LegalAction(act=ActionAct.ACTIVATE, effect=eff_idx, desc=desc, code=code),
            LegalAction(act=ActionAct.CANCEL),
        ]
        return SelectResult(
            msg,
            player,
            selector=Selector(
                msg, player, actions, lambda idx, a: encode_responsei(1 if idx == 0 else 0)
            ),
        )

    if msg == C.MSG_SELECT_EFFECTYN:
        player = r.u8()
        code = r.u32()
        ct = r.u8()
        loc = r.u8()
        seq = r.u8()
        pos = r.u8()
        desc = r.u32()
        r.note_card(code, ct, loc, seq)
        spec = ls_to_spec(loc, seq, pos, ct != player)
        code_d, eff_idx = unpack_desc(code, desc)
        if desc == 0:
            code_d = code
        actions = [
            LegalAction(spec=spec, act=ActionAct.ACTIVATE, effect=eff_idx, desc=desc, code=code_d),
            LegalAction(act=ActionAct.CANCEL),
        ]
        return SelectResult(
            msg,
            player,
            selector=Selector(
                msg, player, actions, lambda idx, a: encode_responsei(1 if idx == 0 else 0)
            ),
            cards=list(r.cards),
        )

    if msg == C.MSG_SELECT_OPTION:
        player = r.u8()
        size = r.u8()
        actions = []
        for index in range(size):
            desc = r.u32()
            code, eff_idx = unpack_desc(0, desc)
            # A missing label does not make the engine's option illegal.
            # Retain an opaque ordinal without fabricating a card identity.
            actions.append(
                LegalAction(act=ActionAct.ACTIVATE, effect=eff_idx if desc else -1,
                            desc=desc, code=code, number=index if not desc else 0)
            )
        return SelectResult(
            msg,
            player,
            selector=Selector(msg, player, actions, lambda idx, a: encode_responsei(idx)),
        )

    if msg in (C.MSG_SELECT_PLACE, C.MSG_SELECT_DISFIELD):
        player = r.u8()
        count = r.u8() or 1
        flag = r.u32()
        places = flag_to_usable_places(flag)
        if count > len(places):
            raise UnsupportedMessage(
                f"select place count {count} exceeds {len(places)} options"
            )
        if count > 1:
            return SelectResult(
                msg, player, selector=PlaceSelector(msg, player, places, count)
            )
        actions = [LegalAction(place=p) for p in places]

        def enc(idx, action):
            plr, loc, seq = place_to_ls(action.place, player)
            return bytes([plr, loc, seq])

        return SelectResult(msg, player, selector=Selector(msg, player, actions, enc))

    if msg == C.MSG_SELECT_POSITION:
        player = r.u8()
        code = r.u32()
        valid_pos = r.u8()
        actions = [
            LegalAction(position=pos, code=code)
            for pos in (
                C.POS_FACEUP_ATTACK,
                C.POS_FACEDOWN_ATTACK,
                C.POS_FACEUP_DEFENSE,
                C.POS_FACEDOWN_DEFENSE,
            )
            if valid_pos & pos
        ]
        return SelectResult(
            msg,
            player,
            selector=Selector(
                msg, player, actions, lambda idx, a: encode_responsei(a.position)
            ),
        )

    if msg == C.MSG_SELECT_COUNTER:
        # The declared auto-response law for any card count: in listed order
        # each card gives min(remaining, its counters), a u16 per card. For
        # every legal prompt these are the bytes the former two-card formula
        # and the removed greedy fallback produced.
        player = r.u8()
        r.u16()  # counter type
        remaining = r.u16()
        count = r.u8()
        resp = bytearray()
        for _ in range(count):
            r.u32()
            r.u8()
            r.u8()
            r.u8()
            take = min(remaining, r.u16() & 0xFFFF)
            resp += struct.pack("<H", take)
            remaining -= take
        return SelectResult(
            msg, player, auto_response=bytes(resp), note="counter placement"
        )

    if msg == C.MSG_SORT_CHAIN:
        player = r.u8()
        count = r.u8()
        cards = []
        for _ in range(count):
            code = r.u32() & 0x7FFFFFFF
            controller = r.u8()
            location = r.u8()
            sequence = r.u8()
            r.note_card(code, controller, location, sequence)
            cards.append((code, controller, location, sequence))
        if not cards:
            raise UnsupportedMessage("sort chain offers no rows")
        return SelectResult(
            msg,
            player,
            selector=OrderSelector(msg, player, cards),
            cards=list(r.cards),
        )

    if msg == C.MSG_SORT_CARD:
        if ctx.full_card_sort_menu:
            player, count = r.u8(), r.u8()
            cards = []
            for _ in range(count):
                code, controller, location, sequence = r.u32() & 0x7FFFFFFF, r.u8(), r.u8(), r.u8()
                r.note_card(code, controller, location, sequence)
                cards.append((code, controller, location, sequence))
            if cards:
                return SelectResult(msg, player, selector=CardOrderSelector(msg, player, cards), cards=list(r.cards))
        # ygoenv never reorders; 0xff is the core's "cancel the sort"
        return SelectResult(
            msg, ctx.our_player, auto_response=bytes([255]), note="sort declined"
        )

    if msg == C.MSG_ANNOUNCE_NUMBER:
        player = r.u8()
        count = r.u8()
        actions = []
        for _ in range(count):
            # The 12-button UI is only a fast path.  The protocol accepts any
            # int32 option and answers with its *index*; larger values use the
            # client's combo box (Tierra's pathological seed exposes 100).
            actions.append(LegalAction(number=r.i32()))
        return SelectResult(
            msg,
            player,
            selector=Selector(msg, player, actions, lambda idx, a: encode_responsei(idx)),
        )

    if msg in (C.MSG_ANNOUNCE_ATTRIB, C.MSG_ANNOUNCE_RACE):
        player = r.u8()
        count = r.u8()
        flag = r.u32()
        if msg == C.MSG_ANNOUNCE_ATTRIB:
            bits = [1 << i for i in range(7) if flag & (1 << i)]
            if count > len(bits):
                raise UnsupportedMessage(
                    f"announce count {count} exceeds {len(bits)} attributes"
                )
            if count > 1:
                return SelectResult(
                    msg, player,
                    selector=BitmaskSelector(msg, player, bits, count, True),
                )
            actions = [LegalAction(attribute=bit) for bit in bits]
            enc = lambda idx, a: encode_responsei(a.attribute)  # noqa: E731
        else:
            bits = [
                1 << i for i in range(C.RACES_COUNT) if flag & (1 << i)
            ]
            if count > len(bits):
                raise UnsupportedMessage(
                    f"announce count {count} exceeds {len(bits)} races"
                )
            if count > 1:
                return SelectResult(
                    msg, player,
                    selector=BitmaskSelector(msg, player, bits, count, False),
                )
            actions = [LegalAction(race=bit) for bit in bits]
            enc = lambda idx, a: encode_responsei(a.race)  # noqa: E731
        return SelectResult(msg, player, selector=Selector(msg, player, actions, enc))

    if msg == C.MSG_ANNOUNCE_CARD:
        player = r.u8()
        count = r.u8()
        opcodes = [r.u32() for _ in range(count)]
        codes = parse_codes_from_opcodes(opcodes)
        note = "declared card names"
        if codes is None:
            if ctx.card_pool is None:
                raise UnsupportedMessage(
                    "announce_card needs a card pool for the general filter"
                )
            codes = ctx.card_pool.declarable(
                opcodes, ctx.known_codes, limit=ctx.max_options
            )
            note = "declarable filter over the card pool"
        if not codes:
            raise UnsupportedMessage(f"no declarable card for filter {opcodes}")
        actions = [LegalAction(code=c, response=c) for c in codes]
        return SelectResult(
            msg,
            player,
            selector=Selector(
                msg, player, actions, lambda idx, a: encode_responsei(a.response)
            ),
            note=note,
        )

    if msg == C.MSG_ROCK_PAPER_SCISSORS:
        # Transmission Gear can invoke this during a duel.  Auto-answering 1
        # made both seats choose Rock forever, producing thousands of tie
        # results in one settlement interval.  It is a real three-way private
        # choice; the core broadcasts MSG_HAND_RES after both answers.
        player = r.u8()
        actions = [LegalAction(number=value, response=value) for value in (1, 2, 3)]
        return SelectResult(
            msg, player,
            selector=Selector(
                msg, player, actions,
                lambda idx, action: encode_responsei(action.response),
            ),
            note="rock-paper-scissors",
        )

    raise UnsupportedMessage(f"no action mapping for message {msg}")
