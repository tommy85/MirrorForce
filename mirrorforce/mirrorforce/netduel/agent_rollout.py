"""Batched continuations of a search root in a local engine (the design notes).

A root is a prompt of the searching seat where a local engine stands exactly as the server's game (the follower's
root), with the opponent's hidden cards filled by one hypothetical position per start. Every continuation ("line")
takes one legal row of the root, then both seats play by their policies to the root seat's ``depth``-th own decision
after the root, or to the game's end:

- The engine is a ``DuelDriver`` whose class is switched, for the search, to a parking subclass: every message is
  projected to the two seats' packets (``netduel.wire_projection.project_message``, the host's own routing and
  masking), and every prompt parks the line (its core snapshot and driver state are kept) until the prompted seat's
  response bytes arrive. No prompt is answered by the local parser.
- Each seat's observation and choice come from the policy service (``tools/mf_runtime_policy_service.py``):
  the root seat's session is a clone of its real session at the root (the root row applied as the line's first
  step); the opponent's session comes from ``opponent_session(start)`` (option A: built from that position's own
  past). All lines advance in lockstep; each round sends every waiting line's prompted seat in one ``rollout_step``.
- Leaf values are Ataraxos' TD(lambda) estimate (``agent/search/update.py``: ``value_weight``, ``outcome_weight``,
  ``outcome``): the root seat's win - loss at its own decisions and the game's result.

Rows get equal numbers of lines (``lines_per_row``) spread over the starts in turn; ``q_values`` averages each row's
leaf values weighted by its starts' weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
import math
import time

from mirrorforce.puzzle.messages import Message
from mirrorforce.worldmodel.engine import DuelDeadlineExceeded
from mirrorforce.agent.search.update import outcome, outcome_weight, value_weight
from . import agent_wire as W
from .wire_projection import project_message

SCHEMA = "mirrorforce_rollout/v1"
MAX_ITEMS = 256
BOUNDED_LANES_LAW = "bounded-common-stripe-lanes/v1"
REPLAY_FILTERED_LANES = 8


class RolloutError(RuntimeError):
    """A continuation could not be played: the root or a line is not what the search requires."""


class RolloutBudgetExceeded(RolloutError):
    """Only the declared clock expired; never a transport, rule or replay failure."""


class EnvLawViolation(RolloutError):
    """A continuation reached a state the environment's rules handle and the local core does not
    (``illegal_activation_withdrawal/v1``): the line is void and counted."""


class _Park(Exception):
    pass


_PARKING = {}


def parking(base):
    """The parking subclass of a driver class (one per class)."""
    if base in _PARKING:
        return _PARKING[base]

    class Parking(base):
        #: the running line's per-seat packet lists (a class attribute: a driver's own fields are its rollback state)
        sink = None

        def _observe(self, message):
            ended = self.winner is not None
            super()._observe(message)
            if ended:
                return  # a host sends nothing after MSG_WIN
            packets = project_message(self.core, self.pduel, message.msg, message.payload)
            for seat in (0, 1):
                Parking.sink[seat].extend(packets[seat])

        def _answer(self, message, responder):
            if not self._replay:
                self._answering_msg, self._answering_payload = message.msg, bytes(message.payload)
                raise _Park()
            super()._answer(message, responder)

    Parking.__name__ = "Parking" + base.__name__
    _PARKING[base] = Parking
    return Parking


@dataclass
class Line:
    index: int
    row: int
    start: int
    sessions: dict
    outbox: dict = field(default_factory=lambda: {0: [], 1: []})
    snap: object = None
    pystate: dict = None
    message: Message = None
    seat: int = None
    own_decisions: int = 0
    leaf: float = 0.0
    end: str = None
    value: float = None
    driver: object = None


@dataclass(frozen=True)
class RestoredRoot:
    """A producer-owned particle engine and its own native root prompt.

    RootLines owns line snapshots and cloned service sessions, NOT these duel
    handles. The producer keeps all handles alive until RootLines.close().
    """
    driver: object
    message: Message
    map_root_response: Callable[[bytes], bytes] | None = None


def prompt_player(message):
    return message.payload[1] if message.msg == 23 else message.payload[0]  # MSG_SELECT_SUM names the player second


class RootLines:
    """The lines of one root. ``restore(start)`` puts ``driver`` at the root prompt ``root_message`` with that
    start's hypothetical position written, or returns a ``RestoredRoot`` for a
    producer-owned replay engine. Each line retains that engine, so snapshots
    never cross duel handles. ``max_live_lines`` explicitly limits a common
    stripe to bounded chunks, closing each chunk before opening the next;
    only completion of all candidate rows returns values. The default keeps
    the original all-lines-at-once scheduling. ``api`` snapshots and rolls back the core;
    ``starts`` are the hypothetical-position keys the lines take in turn."""

    def __init__(self, *, sock, driver, api, restore, starts, root_message, own_session, opponent_session, seat,
                 rows, lines_per_row, depth, td_lambda, seed, max_seconds=60.0, candidate_rows=None,
                 seed_law="independent-lines/v1", max_live_lines=None):
        if depth < 1 or not 0 <= td_lambda <= 1 or lines_per_row < 1 or rows < 2 or not starts:
            raise ValueError("a search root needs two rows, a line per row, a depth and 0 <= lambda <= 1")
        if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("a search needs a finite positive time allowance")
        self.sock, self.driver, self.api, self.restore, self.starts = sock, driver, api, restore, list(starts)
        self.root_message, self.own_session, self.opponent_session = root_message, own_session, opponent_session
        self.seat, self.rows, self.depth, self.td_lambda, self.seed = seat, rows, depth, td_lambda, seed
        if seed_law not in ("independent-lines/v1", "common-stripe/v1") \
                or seed_law == "common-stripe/v1" and (lines_per_row != 1 or len(self.starts) != 1):
            raise ValueError("common stripe seeds require one fixed start and one line per candidate")
        self.seed_law = seed_law
        if max_live_lines is not None and (type(max_live_lines) is not int or not 1 <= max_live_lines <= MAX_ITEMS
                or seed_law != "common-stripe/v1"):
            raise ValueError("bounded live lines require a positive common-stripe lane count <=256")
        self.max_live_lines = max_live_lines
        self._closed_sessions = set()
        self.candidate_rows = tuple(range(rows)) if candidate_rows is None else tuple(candidate_rows)
        if len(self.candidate_rows) != rows or len(set(self.candidate_rows)) != rows \
                or any(type(index) is not int or index < 0 for index in self.candidate_rows):
            raise ValueError("one distinct original service row is required for each searched candidate")
        self.max_seconds = max_seconds
        self._deadline = None
        self.closed = False
        self.lines = []
        number = 0
        for row in range(rows):
            for _ in range(lines_per_row):
                self.lines.append(Line(len(self.lines), row, self.starts[number % len(self.starts)], {}))
                number += 1
        self.stats = {"lines": len(self.lines), "rounds": 0, "items": 0, "service_seconds": 0.0, "engine_seconds": 0.0,
                      "void": 0, "void_logs": []}
        if max_live_lines is not None:
            self.stats.update(resource_law=BOUNDED_LANES_LAW, max_live_lines=max_live_lines,
                              peak_owned_sessions=0, chunks=0)

    # -- engine -------------------------------------------------------------------------------------------------

    def _remaining(self, phase):
        if self._deadline is None:  # Direct single-engine unit probes, outside run().
            return self.max_seconds
        remaining = min(self.max_seconds, self._deadline - time.monotonic())
        if remaining <= 0:
            raise RolloutBudgetExceeded("the search's time ran out before " + phase)
        return remaining

    def _run(self, line, response, *, at_root):
        """Answer the line's parked prompt with ``response`` and run to the next prompt (parked) or the end."""
        api = self.api
        began = time.perf_counter()
        if at_root:
            restored = self.restore(line.start)
            if restored is not None and not isinstance(restored, RestoredRoot):
                raise RolloutError("particle restore must return its owned engine/prompt, or use the legacy driver")
            line.driver = self.driver if restored is None else restored.driver
            message = self.root_message if restored is None else restored.message
            if restored is not None and restored.map_root_response is not None:
                if not callable(restored.map_root_response):
                    raise RolloutError("root response mapper is not callable")
                response = restored.map_root_response(bytes(response))
                if type(response) is not bytes or not response:
                    raise RolloutError("root response mapper did not return immutable wire bytes")
        else:
            message = line.message
        driver = line.driver
        if driver is None or not driver.pduel:
            raise RolloutError("line lost its producer-owned particle engine")
        base = type(driver)
        driver.__class__ = parking(base)
        logged = len(driver.core.log)
        try:
            if not at_root:
                if api.duel_rollback(driver.pduel, line.snap) != 0:
                    raise RolloutError("the core could not roll back to a parked line")
                state, line.pystate = line.pystate, None
                driver.restore_pystate(state, consume=True)
            type(driver).sink = line.outbox
            driver._replay = [bytes(response)]
            try:
                driver._answer(message, None)
                deadline = self._deadline
                allowance = self._remaining("native advance")
                try:
                    driver.run(None, max_steps=200000, max_seconds=allowance)
                except DuelDeadlineExceeded as exc:
                    # Capture the time BEFORE diagnostics/cleanup: neither may
                    # turn an early rule failure into an expired search budget.
                    caught_at = time.monotonic()
                    self._script_errors(driver, logged)
                    if (type(exc) is DuelDeadlineExceeded
                            and type(deadline) in (int, float) and math.isfinite(deadline)
                            and self._deadline == deadline
                            and math.isfinite(caught_at) and caught_at >= deadline
                            and type(exc.max_seconds) in (int, float)
                            and exc.max_seconds == allowance
                            and type(exc.seconds) in (int, float) and math.isfinite(exc.seconds)
                            and exc.seconds > allowance):
                        raise RolloutBudgetExceeded(
                            "the search's time ran out during native advance") from exc
                    raise
            except _Park:
                self._script_errors(driver, logged)
                self._release(line)
                snap = api.duel_snapshot(driver.pduel)
                if not snap:
                    raise RolloutError("the core could not snapshot a parked line")
                line.snap, line.pystate = snap, driver.save_pystate()
                line.message = Message(int(driver._answering_msg), bytes(driver._answering_payload))
                line.seat = int(prompt_player(line.message))
                return
            except Exception:
                self._script_errors(driver, logged)
                raise
            self._script_errors(driver, logged)
            if not driver.finished or driver.winner not in (0, 1, 2):
                self._remaining("native advance completion")
                raise RolloutError("a continuation stopped without a prompt or a result")
            line.leaf += outcome_weight(line.own_decisions, self.td_lambda) * outcome(int(driver.winner), self.seat)
            line.end, line.value = "terminal", line.leaf
            self._release(line)
        finally:
            driver.__class__ = base
            self.stats["engine_seconds"] += time.perf_counter() - began

    @staticmethod
    def _script_errors(driver, logged):
        """A script error in a continuation is a state the environment ends by its own rules (the illegal
        activation withdrawal) or as a fatal error, never one it plays on: the line is void."""
        errors = driver.core.log[logged:]
        if errors:
            del driver.core.log[logged:]
            raise EnvLawViolation("; ".join(errors)[:500])

    def _release(self, line):
        if line.snap:
            self.api.duel_snapshot_free(line.snap)
        line.snap = None

    # -- service ------------------------------------------------------------------------------------------------

    def _call(self, payload):
        began = time.perf_counter()
        try:
            return W.call(self.sock, payload)
        finally:
            self.stats["service_seconds"] += time.perf_counter() - began

    def _own_values(self, line, decisions):
        for decision in decisions:
            win, _, loss = decision["wdl"]
            line.leaf += value_weight(line.own_decisions, self.depth, self.td_lambda) * (win - loss)
            line.own_decisions += 1

    def run(self, deadline=None):
        """Drain this owner's batches alone (the original serial execution law)."""
        batches = self.batches(deadline=deadline)
        try:
            try:
                request = next(batches)
            except StopIteration as completed:
                return completed.value
            while True:
                reply = self._call(request)
                try:
                    request = batches.send(reply)
                except StopIteration as completed:
                    return completed.value
        finally:
            batches.close()

    def batches(self, deadline=None):
        """Play every line to its end; returns ``{row: [(start, leaf value)]}``. A line the env's rules would end
        (``EnvLawViolation``) is void and counted, never valued; any void line
        makes the whole root incomplete rather than silently reweighting actions.

        Yield only rollout_step requests, accepting their matching full replies.
        A multiplexing driver may combine these requests without taking over
        engine state, memory, RNG, clone ownership or whole-root certification.
        The driver must close this generator AND the owner, even on failure.
        Clone/close RPCs still use this owner's _call; the driver accounts for
        the wall time of its shared rollout RPC exactly once.
        """
        if self.closed:
            raise RolloutError("cannot run a closed continuation owner")
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError("the absolute search deadline must be finite")
        self._deadline = time.monotonic() + self.max_seconds
        if deadline is not None:
            self._deadline = min(self._deadline, deadline)
        self._remaining("session initialization")
        if self.max_live_lines is None:
            yield from self._line_batches(self.lines)
        else:
            for begin in range(0, len(self.lines), self.max_live_lines):
                lines = self.lines[begin:begin + self.max_live_lines]
                self.stats["chunks"] += 1
                try:
                    yield from self._line_batches(lines)
                finally:
                    self._close_lines(lines)
        out = {}
        for line in self.lines:
            if line.end != "void":
                out.setdefault(line.row, []).append((line.start, line.value))
        if self.stats["void"] or any(len(out.get(row, ())) != sum(line.row == row for line in self.lines)
                                     for row in range(self.rows)):
            raise RolloutError("root rollout is incomplete; invalid lines cannot change per-action particle budgets")
        self._remaining("complete rollout certification")
        return out

    def _line_batches(self, lines):
        """A bounded lane group; partial groups never bypass whole-root certification."""
        root_items = []
        for line in lines:
            self._remaining("own-session clone")
            clone = self._call({"op": "clone", "session": self.own_session,
                                "seed": (self.seed * 1_000_003 + (0 if self.seed_law == "common-stripe/v1"
                                                                else line.index)) & 0x7FFFFFFF})["session"]
            # Register returned resources BEFORE checking time, so close() can
            # clean a successful but overdue request as well as ordinary work.
            line.sessions[self.seat] = clone
            self._resource_peak(lines)
            self._remaining("opponent-session clone")
            line.sessions[1 - self.seat] = self.opponent_session(line.start, line.index)
            self._resource_peak(lines)
            self._remaining("next line initialization")
            root_items.append((line, {"session": clone, "index": self.candidate_rows[line.row], "messages": [],
                                      "stop_after": self.depth}))
        pending = root_items
        while pending:
            self._remaining("next rollout round")
            self.stats["rounds"] += 1
            for begin in range(0, len(pending), MAX_ITEMS):
                self._remaining("next rollout chunk")
                chunk = pending[begin:begin + MAX_ITEMS]
                self.stats["items"] += len(chunk)
                reply = yield {"op": "rollout_step", "items": [item for _, item in chunk]}
                results = reply["items"]
                if len(results) != len(chunk):
                    raise RolloutError("the service dropped or added rollout items")
                self._remaining("rollout result processing")
                for (line, item), result in zip(chunk, results):
                    at_root = "index" in item
                    if item["session"] == line.sessions[self.seat]:
                        self._own_values(line, result["decisions"])
                    if result.get("cut"):
                        line.end, line.value = "cut", line.leaf
                        self._release(line)
                        continue
                    try:
                        self._run(line, bytes.fromhex(result["response"]), at_root=at_root)
                    except EnvLawViolation as exc:
                        line.end, line.value = "void", None
                        self.stats["void"] += 1
                        self.stats["void_logs"].append(str(exc))
                        self._release(line)
                    self._remaining("next native line")
            pending = []
            for line in lines:
                if line.end is not None:
                    continue
                seat = line.seat
                packets, line.outbox[seat] = line.outbox[seat], []
                item = {"session": line.sessions[seat], "messages": [[p[0], bytes(p[1:]).hex()] for p in packets]}
                if seat == self.seat:
                    item["stop_after"] = self.depth - line.own_decisions
                    if item["stop_after"] < 1:
                        raise RolloutError("a line passed its depth")
                pending.append((line, item))

    def _resource_peak(self, lines):
        if self.max_live_lines is not None:
            peak = len({session for line in lines for session in line.sessions.values()})
            self.stats["peak_owned_sessions"] = max(self.stats["peak_owned_sessions"], peak)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._close_lines(self.lines)

    def _close_lines(self, lines):
        errors = []
        for line in lines:
            try:
                self._release(line)
            except BaseException as exc:
                errors.append(exc)
        sessions = {session for line in lines for session in line.sessions.values()} - self._closed_sessions
        for session in sessions:
            try:
                self._call({"op": "close", "session": session, "messages": []})
                self._closed_sessions.add(session)
            except BaseException as exc:
                errors.append(exc)
        if self.max_live_lines is not None:
            for line in lines:
                line.sessions = {seat: name for seat, name in line.sessions.items() if name not in self._closed_sessions}
                if not line.sessions and line.snap is None:
                    line.pystate, line.message, line.driver = None, None, None
                    line.outbox = {0: [], 1: []}
        if errors:
            self.stats["cleanup_errors"] = [f"{type(exc).__name__}: {exc}" for exc in errors]
            raise RolloutError("continuation cleanup failed after attempting every owned snapshot/session") from errors[0]


def q_values(returns, rows, weights):
    """Per row: the mean of its lines' leaf values, each weighted by its start's weight; None for a row whose lines
    were all void."""
    q = []
    for row in range(rows):
        pairs = returns.get(row, [])
        total = math.fsum(weights[start] for start, _ in pairs)
        q.append(None if not pairs else math.fsum(weights[start] * value for start, value in pairs) / total)
    return q
