"""Response codec and a reference model of the ygoenv action decomposition.

The re-simulator recovers an *action index* from the raw response bytes a
replay stores.  The authoritative implementation lives in the C++ driver
(``ygoenv/ygopro/ygopro.h``, the ``resim_*`` methods), which replays every
candidate action's callback and compares the bytes it would have produced.

This module carries the parts that do not need the duel engine:

* the response wire format (``set_responsei`` / ``set_responseb`` payloads and
  the per-message lengths the replay writer stores),
* a pure-Python model of ``init_multi_select`` / ``handle_multi_select`` /
  ``_callback_multi_select``, which is what turns one recorded multi-card
  selection into the sequence of single choices the env asks for.

Keeping the model here makes the decomposition testable without a compiled
``ygoenv`` and documents the convention the C++ side implements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "MSG_NAMES",
    "MSG_SELECT_BATTLECMD",
    "MSG_SELECT_IDLECMD",
    "MSG_SELECT_EFFECTYN",
    "MSG_SELECT_YESNO",
    "MSG_SELECT_OPTION",
    "MSG_SELECT_CARD",
    "MSG_SELECT_CHAIN",
    "MSG_SELECT_PLACE",
    "MSG_SELECT_POSITION",
    "MSG_SELECT_TRIBUTE",
    "MSG_SELECT_COUNTER",
    "MSG_SELECT_SUM",
    "MSG_SELECT_DISFIELD",
    "MSG_SORT_CARD",
    "MSG_SELECT_UNSELECT_CARD",
    "MULTI_SELECT_MSGS",
    "DecomposeError",
    "encode_responsei",
    "decode_responsei",
    "encode_selection",
    "decode_selection",
    "responseb_len",
    "MultiSelect",
    "decompose_selection",
]

# ygopro-core common.h
MSG_SELECT_BATTLECMD = 10
MSG_SELECT_IDLECMD = 11
MSG_SELECT_EFFECTYN = 12
MSG_SELECT_YESNO = 13
MSG_SELECT_OPTION = 14
MSG_SELECT_CARD = 15
MSG_SELECT_CHAIN = 16
MSG_SELECT_PLACE = 18
MSG_SELECT_POSITION = 19
MSG_SELECT_TRIBUTE = 20
MSG_SELECT_COUNTER = 22
MSG_SELECT_SUM = 23
MSG_SELECT_DISFIELD = 24
MSG_SORT_CARD = 25
MSG_SELECT_UNSELECT_CARD = 26
MSG_ANNOUNCE_ATTRIB = 141
MSG_ANNOUNCE_NUMBER = 143
MSG_ANNOUNCE_CARD = 142

MSG_NAMES = {
    MSG_SELECT_BATTLECMD: "select_battlecmd",
    MSG_SELECT_IDLECMD: "select_idlecmd",
    MSG_SELECT_EFFECTYN: "select_effectyn",
    MSG_SELECT_YESNO: "select_yesno",
    MSG_SELECT_OPTION: "select_option",
    MSG_SELECT_CARD: "select_card",
    MSG_SELECT_CHAIN: "select_chain",
    MSG_SELECT_PLACE: "select_place",
    MSG_SELECT_POSITION: "select_position",
    MSG_SELECT_TRIBUTE: "select_tribute",
    MSG_SELECT_COUNTER: "select_counter",
    MSG_SELECT_SUM: "select_sum",
    MSG_SELECT_DISFIELD: "select_disfield",
    MSG_SORT_CARD: "sort_card",
    MSG_SELECT_UNSELECT_CARD: "select_unselect_card",
    MSG_ANNOUNCE_ATTRIB: "announce_attrib",
    MSG_ANNOUNCE_NUMBER: "announce_number",
    MSG_ANNOUNCE_CARD: "announce_card",
}

# the messages ygoenv answers by decomposing one selection into single choices
MULTI_SELECT_MSGS = frozenset(
    {MSG_SELECT_CARD, MSG_SELECT_TRIBUTE, MSG_SELECT_SUM}
)


class DecomposeError(ValueError):
    """The recorded selection cannot be produced by the env decomposition."""


def encode_responsei(value: int) -> bytes:
    """4 little-endian bytes, the payload ``set_responsei`` writes."""
    return int(value).to_bytes(4, "little", signed=value < 0)


def decode_responsei(data: bytes) -> int:
    if len(data) != 4:
        raise ValueError(f"responsei payload must be 4 bytes, got {len(data)}")
    return int.from_bytes(data, "little", signed=True)


def encode_selection(indices: list[int], must: int = 0) -> bytes:
    """``[count, <must zeros>, index...]``, as ``select_with_sum_limit`` reads it.

    ``count`` covers the automatically selected cards too, which is why the
    ``must`` leading entries are present but carry no information.
    """
    if must < 0:
        raise ValueError("must must not be negative")
    return bytes([len(indices) + must] + [0] * must + list(indices))


def decode_selection(data: bytes, must: int = 0) -> list[int]:
    if not data:
        raise ValueError("empty selection response")
    count = data[0]
    if len(data) < count + 1:
        raise ValueError(
            f"selection response declares {count} entries, only {len(data) - 1} present"
        )
    if must > count:
        raise ValueError(f"must ({must}) exceeds the declared count ({count})")
    return list(data[1 + must : 1 + count])


def responseb_len(msg: int, buf: bytes, n_counters: int = 0) -> int:
    """Number of bytes the replay writer stores for a ``set_responseb`` payload.

    Mirrors ``YGOProEnvImpl::YGO_SetResponseb`` in ``ygopro.h``; the reference
    client writes the same lengths.
    """
    if msg == MSG_SORT_CARD:
        return 1
    if msg == MSG_SELECT_COUNTER:
        return 2 * n_counters
    if msg in (MSG_SELECT_PLACE, MSG_SELECT_DISFIELD):
        return 3
    if not buf:
        raise ValueError("empty responseb payload")
    return buf[0] + 1


@dataclass
class MultiSelect:
    """Model of the ygoenv multi select state machine.

    ``mode`` 0 covers ``MSG_SELECT_CARD`` / ``MSG_SELECT_TRIBUTE``. ``weights``
    defaults to one per card; weighted tribute prompts use each card's
    ``release_param``. A synthetic ``finish`` action appears once the selected
    weight reaches ``min``. ``mode`` 1 covers
    ``MSG_SELECT_SUM``: only the cards that start a still-viable combination
    are offered and the selection ends by itself when a combination runs out,
    so there is no ``finish`` action.
    """

    min: int
    max: int
    must: int = 0
    specs: list[str] = field(default_factory=list)
    mode: int = 0
    combs: list[list[int]] = field(default_factory=list)
    weights: list[int] = field(default_factory=list)

    idx: int = 0
    r_idxs: list[int] = field(default_factory=list)
    spec2idx: dict[str, int] = field(default_factory=dict)
    done: bool = False
    response: bytes | None = None

    def __post_init__(self) -> None:
        self.spec2idx = {spec: j for j, spec in enumerate(self.specs)}
        if not self.weights:
            self.weights = [1] * len(self.specs)
        if len(self.weights) != len(self.specs):
            raise ValueError(
                f"weights has {len(self.weights)} rows for {len(self.specs)} specs"
            )
        if self.mode != 0:
            self.combs = [list(c) for c in self.combs]

    def _selected_weight(self) -> int:
        return sum(self.weights[index] for index in self.r_idxs)

    def _mode0_viable(self, index: int) -> bool:
        """Can choosing ``index`` still reach the required tribute weight?"""
        selected = len(self.r_idxs) + 1
        if selected > self.max:
            return False
        total = self._selected_weight() + self.weights[index]
        if total >= self.min:
            return True
        slots = self.max - selected
        remaining = sorted(
            (
                self.weights[j]
                for j in self.spec2idx.values()
                if j != index
            ),
            reverse=True,
        )
        return total + sum(remaining[:slots]) >= self.min

    # -- offered actions ---------------------------------------------------
    def options(self) -> list[str | None]:
        """Legal actions of the current sub-step; ``None`` is the finish action."""
        if self.done:
            return []
        if self.mode == 0:
            opts: list[str | None] = [
                spec for spec in self.specs
                if spec in self.spec2idx
                and self._mode0_viable(self.spec2idx[spec])
            ]
            if self._selected_weight() >= self.min:
                opts.append(None)
            return opts
        firsts = sorted({c[0] for c in self.combs if c})
        opts: list[str | None] = [self.specs[i] for i in firsts]
        if any(not c for c in self.combs):
            # an empty combination means the must-select cards already reach
            # the target on their own, so "select nothing more" is a legal
            # answer; it can only be present before the first pick, because
            # _step_mode1 finishes the moment a combination runs out
            opts.append(None)
        return opts

    # -- transitions -------------------------------------------------------
    def step(self, option: int) -> None:
        opts = self.options()
        if not 0 <= option < len(opts):
            raise DecomposeError(f"option {option} out of range ({len(opts)} offered)")
        spec = opts[option]
        if self.mode == 0:
            self._step_mode0(spec)
        else:
            self._step_mode1(spec)

    def _step_mode0(self, spec: str | None) -> None:
        if spec is not None:
            self.r_idxs.append(self.spec2idx[spec])
        # handle_multi_select() forces the finish once max cards are picked.
        finish = spec is None or len(self.r_idxs) >= self.max
        if finish:
            if self._selected_weight() < self.min:
                raise DecomposeError(
                    f"selection weight {self._selected_weight()} < min {self.min}"
                )
            self.done = True
            self.response = encode_selection(self.r_idxs)
        else:
            self.idx = len(self.r_idxs)
            if spec is not None:
                del self.spec2idx[spec]

    def _step_mode1(self, spec: str | None) -> None:
        if spec is None:
            self.done = True
            self.response = encode_selection(self.r_idxs, self.must)
            return
        idx = self.spec2idx[spec]
        self.r_idxs.append(idx)
        rest: list[list[int]] = []
        for comb in self.combs:
            if comb and comb[0] == idx:
                comb = comb[1:]
                if not comb:
                    self.done = True
                    self.response = encode_selection(self.r_idxs, self.must)
                    return
                rest.append(comb)
        self.idx += 1
        self.combs = rest


def decompose_selection(
    ms: MultiSelect, response: bytes, ordered: bool = True
) -> list[int]:
    """Sequence of option indices that reproduces ``response``.

    ``ordered`` replays the recorded pick order; with ``ordered=False`` the
    selection is replayed in ascending card index order, which is the only
    order ``MSG_SELECT_SUM`` can produce.  Raises :class:`DecomposeError` when
    the recorded selection is not reachable, which is the signal that the game
    has to be dropped rather than silently mislabelled.
    """
    target = decode_selection(response, ms.must if ms.mode != 0 else 0)
    if not ordered:
        target = sorted(target)
    picks: list[int] = []
    for _ in range(len(target) + 2):
        if ms.done:
            break
        opts = ms.options()
        want = target[len(ms.r_idxs)] if len(ms.r_idxs) < len(target) else None
        choice = None
        for i, spec in enumerate(opts):
            if want is None:
                if spec is None:
                    choice = i
                    break
            elif spec is not None and ms.spec2idx.get(spec) == want:
                choice = i
                break
        if choice is None:
            raise DecomposeError(
                f"no option leads to card {want!r} after {ms.r_idxs} "
                f"(offered {opts})"
            )
        picks.append(choice)
        ms.step(choice)
    if not ms.done:
        raise DecomposeError(f"selection {target} never completed")
    got = decode_selection(ms.response or b"", ms.must if ms.mode != 0 else 0)
    if sorted(got) != sorted(target):
        raise DecomposeError(f"decomposition produced {got}, recorded {target}")
    return picks
