"""Strict opt-in ABI for a detached hypothesis' consumed shuffle plan.

No actor observation or original host object belongs at this boundary. The
caller owns the hypothetical duel and records the source/history/root SHA.
Installing this low-level plan alone is not search admission: received wire,
root World/menu and legal response witnesses still need their separate proofs.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

from . import constants as C
from .agent_causal_plan import verify
from ..common.sidecar_io import digest, require_sha256

INFORMATION_SET_SEARCH = True
SCHEMA = "mirrorforce_consumed_shuffle_plan/v1"
PROCESSOR_REPLAY_ERROR = 0x40000000
MAX_ENTRIES, MAX_PARTICIPANTS = 4096, 262144


class ShufflePlanError(RuntimeError):
    pass


class ShufflePlanReplayError(ShufflePlanError):
    """The native processor set its sticky replay fault; no buffer was drained."""
    def __init__(self, state):
        self.native_state = tuple(state)
        super().__init__("native hypothetical shuffle-plan replay failed: " + repr(self.native_state))


@dataclass(frozen=True)
class ShuffleStep:
    message: int
    player: int
    location: int
    sequences: tuple
    destination_from_source: tuple
    source_codes: tuple

    def words(self):
        n = len(self.sequences)
        values = (self.message, self.player, self.location, n, *self.sequences,
                  *self.destination_from_source, *self.source_codes)
        if any(type(v) is not int or not 0 <= v <= 0xffffffff for v in values) \
                or self.player not in (0, 1) or len(self.destination_from_source) != n or len(self.source_codes) != n \
                or set(self.destination_from_source) != set(range(n)) \
                or tuple(sorted(set(self.sequences))) != self.sequences \
                or any(not 0 < code < 0x80000000 or 999000001 <= code <= 999000004 for code in self.source_codes):
            raise ShufflePlanError("malformed exact shuffle permutation or source identity")
        if self.message == C.MSG_SHUFFLE_SET_CARD:
            valid = self.location in (C.LOCATION_MZONE, C.LOCATION_SZONE) and 2 <= n <= 5 \
                    and all(sequence <= 4 for sequence in self.sequences)
        else:
            valid = (self.message, self.location) in ((C.MSG_SHUFFLE_HAND, C.LOCATION_HAND),
                (C.MSG_SHUFFLE_DECK, C.LOCATION_DECK), (C.MSG_SHUFFLE_EXTRA, C.LOCATION_EXTRA)) \
                and int(self.message != C.MSG_SHUFFLE_EXTRA) <= n <= 255 and self.sequences == tuple(range(n))
        if not valid:
            raise ShufflePlanError("shuffle event has unsupported message/zone/participant bounds")
        return values


def steps_from_causal_plan(history, layout, plan):
    proof = verify(history, layout, plan)
    steps = []
    for cut, (_, _, after, matched) in zip(history.permutations, plan.cuts):
        zones = {place[:2] for place in cut.positions}
        if len(zones) > 1 or zones and zones != {(cut.player, cut.location)}:
            raise ShufflePlanError("one native shuffle cannot mix players or zones")
        player, location = cut.player, cut.location
        old_place = dict(zip(cut.before, cut.positions))
        selected = dict(zip(after, matched))
        by_position = sorted(zip(cut.positions, cut.before, cut.after))
        sequences = tuple(place[2] for place, _, _ in by_position)
        permutation = tuple(sequences.index(old_place[selected[token]][2]) for _, _, token in by_position)
        codes = tuple(plan.token_codes[token - 1] for _, token, _ in by_position)
        step = ShuffleStep(cut.msg, player, location, sequences, permutation, codes)
        step.words()
        steps.append(step)
    return tuple(steps), proof


class ShufflePlanAPI:
    def __init__(self, core):
        self.core, self.lib = core, core._lib
        names = ("install", "state", "receipts", "finish")
        if any(not hasattr(self.lib, "duel_shuffle_plan_" + name) for name in names):
            raise ShufflePlanError("this isolated core lacks the consumed shuffle-plan ABI")
        for name in names:
            getattr(self.lib, "duel_shuffle_plan_" + name).restype = ctypes.c_int32
        self.lib.duel_shuffle_plan_install.argtypes = [ctypes.c_ssize_t, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int32]
        self.lib.duel_shuffle_plan_state.argtypes = [ctypes.c_ssize_t, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int32]
        self.lib.duel_shuffle_plan_receipts.argtypes = [ctypes.c_ssize_t, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int32]
        self.lib.duel_shuffle_plan_finish.argtypes = [ctypes.c_ssize_t]

    def install(self, pduel, steps, *, source_sha256, history_sha256, root_sha256):
        for value in (source_sha256, history_sha256, root_sha256):
            require_sha256(value, "hypothetical shuffle scope", error=ShufflePlanError)
        if not isinstance(steps, tuple) or len(steps) > MAX_ENTRIES or any(type(step) is not ShuffleStep for step in steps) \
                or sum(len(step.sequences) for step in steps) > MAX_PARTICIPANTS:
            raise ShufflePlanError("a closed bounded tuple of shuffle entries is required")
        words = (1, len(steps), *(word for step in steps for word in step.words()))
        raw = (ctypes.c_uint32 * len(words))(*words)
        result = self.lib.duel_shuffle_plan_install(pduel, raw, len(words))
        if result != 0:
            raise ShufflePlanError("native shuffle plan installation refused: " + str(result))
        return {"schema": SCHEMA, "source_sha256": source_sha256, "history_sha256": history_sha256,
                "root_sha256": root_sha256, "words_sha256": digest(words), "events": len(steps),
                "scope": "detached-complete-hypothesis-only/v1", "native_replay_admitted": False}

    def state(self, pduel):
        out = (ctypes.c_uint32 * 9)()
        if self.lib.duel_shuffle_plan_state(pduel, out, 9) != 0 or out[0] != 1:
            raise ShufflePlanError("cannot read the hypothetical shuffle-plan state")
        return tuple(out)

    def receipts(self, pduel):
        n = self.lib.duel_shuffle_plan_receipts(pduel, None, 0)
        if not 9 <= n <= 9 + 6 * MAX_ENTRIES + 3 * MAX_PARTICIPANTS:
            raise ShufflePlanError("invalid shuffle-plan receipt size")
        out = (ctypes.c_uint64 * n)()
        if self.lib.duel_shuffle_plan_receipts(pduel, out, n) != n or tuple(out[:9]) != self.state(pduel):
            raise ShufflePlanError("shuffle receipts differ from the same idle native boundary")
        return tuple(out)

    def process(self, pduel):
        raw = self.core.process(pduel)
        if raw & PROCESSOR_REPLAY_ERROR:
            # Do not drain/project the partial message buffer as accepted data.
            raise ShufflePlanReplayError(self.state(pduel))
        return raw

    def finish(self, pduel):
        result = self.lib.duel_shuffle_plan_finish(pduel)
        if result != 0:
            raise ShufflePlanError("native shuffle plan cannot finish: " + str(result) + " " + repr(self.state(pduel)))
        return self.receipts(pduel)
