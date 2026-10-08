"""O(1) snapshots and branches of a live duel through the copy-on-write of ``fork(2)``.

Background
----

``ocgapi`` has no snapshot interface. :meth:`DuelDriver.fork` in ``worldmodel/engine.py``
uses **replay**: rebuild the duel from the seed, then feed the recorded
answers back one by one. It is exact, but the cost of each branch grows with the depth of the decision point: one
mid-game expansion costs replaying the whole game from the start.

This module takes the other road: the process itself is the snapshot. The core is a ``.so`` loaded into the Python
process with ctypes, and the whole duel state (``duel*``, card objects, effect tables, mt19937) lives on this process's heap,
so ``fork(2)`` is one kernel page-table copy: the branch cost does not depend on depth, only on the pages the branch
actually dirties.

**Deployment rule (hard constraint)**
----------------------

**Policy-side search must never fork the real duel process.** The real duel process holds the opponent's real hand, set cards
and deck order; forking it feeds illegal information straight into the search tree, and no win rate obtained that way stands.

There are only two legal uses of this module:

1. **offline diagnostics / ablations**: measuring search depth, value-head quality and probe metrics under omniscience;
2. **advancing particles**: a particle is built by :mod:`mirrorforce.search.rebuild` from public information +
   sampled hidden cards; once built it is a legal world "omniscient to us", and forking
   branches on it leaks nothing. **This is the form online search really uses.**

Online decisions must **never** call this module on the ``DuelDriver`` of a real duel.

Why the fork must happen inside the responder
--------------------------------

When the core is parked on a prompt, the return buffer of ``duel::set_responseb`` still holds the **previous**
answer. Calling ``process()`` again from outside at that moment makes the core rerun this step with those stale bytes,
without an error, silently producing a wrong branch (measured in the prototype). So the fork point can only be
the body of ``responder(prompt, driver)``: after ``fork`` the child **returns** the index it chooses from the responder,
and the driver calls ``selector.choose`` as usual, not one step more or less.

:class:`SearchSession` implements this constraint: it is the responder and the branch
controller at once, one object with three roles (host / zygote / branch).

fork safety
---------

``fork(2)`` copies only the calling thread. The host must be:

* **single-threaded**. Thread pools of torch / numpy, ``ThreadPoolExecutor`` and logging's
  ``QueueListener`` all make the child inherit a lock nobody will unlock again.
  :func:`assert_fork_safe` checks before every branch.
* **holding no mutable cross-process handles**. ``puzzle.core.Core`` holds a sqlite3 connection:
  parent and child reading the same fd would trample each other's offsets. So the child **must never touch the database**:
  :func:`prewarm_card_cache` reads the whole cdb into ``Core._card_cache`` before branching,
  after which ``_read_card`` always hits the cache, including token codes scripts create on the fly.
* **writing only its own fds in the child**, then ``os._exit`` at once, bypassing atexit / GC / buffer flushes.

Determinism
------

Branch determinism comes from three pieces of state, all copied by fork: the core's mt19937 (the seed sequence of
``create_duel_v2``), the driver's ``random.Random`` (sub-choices and ``MSG_ANNOUNCE_*``), and
the continuation policy's own rng. Running the same branch twice gives byte-identical trajectories, measured by
:meth:`SearchSession.verify_determinism`.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import socket
import struct
import sys
import threading
import time
from dataclasses import asdict, dataclass, field

__all__ = [
    "BranchOutcome",
    "BranchSpec",
    "BranchNotPreparedError",
    "ForkSafetyError",
    "SearchSession",
    "Snapshot",
    "assert_fork_safe",
    "greedy_index",
    "prewarm_card_cache",
    "read_private_dirty_kb",
    "read_rss_kb",
    "rollout_policy",
]


class ForkSafetyError(RuntimeError):
    """The host process state is not fit for ``fork(2)``."""


class _BranchDepthReached(Exception):
    """The branch used up its budget; a normal finish."""


class BranchNotPreparedError(RuntimeError):
    """The branch wants to touch the engine before finishing its permutation: a hard rule, refused outright.

    Right after a replay-style particle is forked, the process holds **the opponent's real hidden cards**.
    The hidden zones must be permuted into a belief sample before any query or evaluation is allowed. This exception is where
    that rule is enforced in the responder.
    """


class _ReturnFromResponder(Exception):
    """Internal signal: the branch child must return an action index from the responder.

    After ``fork`` the child must **return from the responder**, but at that moment it is already inside the loop of
    :meth:`SearchSession.expand` or ``_serve``, several stack frames away.
    Raising this signal gets back to the catch point in :meth:`SearchSession.__call__`, which then ``return``s as is.
    """

    def __init__(self, action: int):
        super().__init__(action)
        self.action = action


# -- process introspection ----------------------------------------------------------------


def _smaps_rollup() -> dict:
    out: dict[str, int] = {}
    try:
        with open("/proc/self/smaps_rollup") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                value = value.strip()
                if value.endswith("kB"):
                    out[key] = int(value[:-2].strip())
    except OSError:
        pass
    return out


def read_private_dirty_kb() -> int:
    """Pages this process owns exclusively and has dirtied, in kB.

    After a fork every page of the child is a shared clean page; each page written turns into
    ``Private_Dirty``. So "the real memory increase of a branch over the host" = the growth of the child's
    ``Private_Dirty`` during the branch. RSS cannot be used: pages shared by parent and child count once on each side.
    """
    return _smaps_rollup().get("Private_Dirty", 0)


def read_rss_kb() -> int:
    return _smaps_rollup().get("Rss", 0)


def assert_fork_safe() -> None:
    """The host must be single-threaded; forking with threads takes along locks other threads hold."""
    alive = [t for t in threading.enumerate() if t.is_alive()]
    if len(alive) > 1:
        names = ", ".join(t.name for t in alive)
        raise ForkSafetyError(
            f"the host process has {len(alive)} live threads ({names}); fork copies only the calling thread, "
            "and locks held by the others can never be released in the child"
        )


def prewarm_card_cache(core) -> int:
    """Read the whole cdb into the core's card data cache so children never touch sqlite.

    Returns the number of cards cached. About 15,000 rows, 90 ms, 1.2 MB, once per process.
    """
    from ..puzzle.core import CardData

    scratch = CardData()
    rows = core._db.execute("SELECT id FROM datas").fetchall()
    for (code,) in rows:
        if code not in core._card_cache:
            core._read_card(code, ctypes.byref(scratch))
    core.missing_codes.clear()  # prewarming is not "this duel lacks cards"
    return len(core._card_cache)


# -- branch descriptions and results ----------------------------------------------------------


@dataclass
class BranchSpec:
    """What a branch does."""

    #: which option to choose on the prompt where the fork happens
    action: int
    #: seed of the continuation policy afterwards
    policy_seed: int = 0
    #: stop after this many decisions; 0 = play to the end
    max_decisions: int = 0
    #: a label, returned as is for the caller to match
    tag: str = ""


@dataclass
class BranchOutcome:
    """What a branch sends back when it finishes."""

    action: int
    tag: str = ""
    ok: bool = False
    error: str = ""
    winner: int | None = None
    win_reason: int | None = None
    turn: int = 0
    decisions: int = 0
    steps: int = 0
    #: sha256 of every ``set_responseb`` payload within the branch (one criterion of byte-identical trajectories)
    response_digest: str = ""
    #: sha256 of every core message within the branch (likewise, covering the core's output side)
    message_digest: str = ""
    #: pages dirtied by the branch since the fork, kB
    private_dirty_kb: int = 0
    rss_kb: int = 0
    #: interval from the host calling ``fork()`` to the child getting the CPU (same CLOCK_MONOTONIC)
    visible_latency_ns: int = 0
    seconds: float = 0.0
    #: what the branch brings back itself (:meth:`SearchSession.report`). Must be JSON-serializable.
    payload: dict = field(default_factory=dict)

    @property
    def decisions_per_second(self) -> float:
        return self.decisions / self.seconds if self.seconds > 0 else 0.0


# -- continuation policies ----------------------------------------------------------------


def rollout_policy(seed: int):
    """The default continuation policy within a branch: uniformly random over options.

    The same seed on the same branch gives the same trajectory twice; giving the same seed when comparing different actions means common
    random numbers, with smaller variance, so this is the default for comparing branches.
    """
    import random as _random

    rng = _random.Random(("mf-rollout", seed).__repr__())

    def choose(prompt, driver) -> int:
        return rng.randrange(len(prompt.actions))

    return choose


def greedy_index(prompt, driver) -> int:
    """Always the first option. ygoenv's menu puts summons first, so this is the built-in greedy policy."""
    return 0


# -- digests --------------------------------------------------------------------


def _hash_messages(messages) -> str:
    digest = hashlib.sha256()
    for message in messages:
        digest.update(struct.pack("<H", message.msg))
        digest.update(message.payload)
    return digest.hexdigest()


def _hash_responses(responses) -> str:
    digest = hashlib.sha256()
    for payload in responses:
        digest.update(payload)
    return digest.hexdigest()


def _write_json(write_fd: int, obj) -> None:
    try:
        os.write(write_fd, json.dumps(obj).encode())
    finally:
        try:
            os.close(write_fd)
        except OSError:
            pass


def _read_all(read_fd: int) -> bytes:
    chunks = []
    while True:
        data = os.read(read_fd, 1 << 16)
        if not data:
            break
        chunks.append(data)
    os.close(read_fd)
    return b"".join(chunks)


# -- session: responder and branch controller in one ----------------------------------------


@dataclass
class _BranchRole:
    spec: BranchSpec
    write_fd: int
    policy: object
    measure: bool
    t_fork: int
    #: run once in the child, after ``fork`` and before the responder returns, with the signature
    #: ``prepare(session, spec, prompt)``. Particle permutation hangs here: this is the only place that can keep
    #: "no query / evaluation before the permutation completes". Returning an integer makes it this step's
    #: action index, overriding ``spec.action``; useful when a branch picks its action from the permuted board.
    prepare: object = None
    #: whether ``prepare`` has finished. Called by the responder before it finishes: raise outright.
    prepared: bool = False
    #: the action index returned by ``prepare``, overriding ``spec.action``
    action_override: int | None = None
    dirty0: int = 0
    base_responses: int = 0
    calls: int = 0
    started: float = 0.0


class SearchSession:
    """A duel's responder, and at the same time its branch controller.

    Usage::

        session = SearchSession(driver, base_responder=random_responder(1))
        session.arm(lambda prompt: prompt.msg == C.MSG_SELECT_IDLECMD,
                    on_point=my_expand_callback)
        session.run(max_steps=200000, max_seconds=120)

    ``on_point(prompt, session)`` is called **on the host**, on the target prompt; inside it you may
    call :meth:`expand` / :meth:`snapshot` / :meth:`verify_determinism`, then
    return the index the host itself chooses. Branch children never reach ``on_point``: they return their own action
    directly from the responder, continue the duel, report and ``_exit``.
    """

    def __init__(self, driver, base_responder, *, record_messages: bool = True):
        self.driver = driver
        self.base_responder = base_responder
        self.record_messages = record_messages
        self._match = None
        self._on_point = None
        self._armed = False
        self._role: _BranchRole | None = None
        #: what a branch brings back; always empty on the host
        self._payload: dict = {}
        #: host side: the time each ``os.fork()`` took to return (ns)
        self.fork_return_ns: list[int] = []

    def report(self, key: str, value) -> None:
        """Record a result to bring back to the host from within a branch. A no-op on the host.

        A branch runs in its own process, and the host sees nothing it changes; the only exit is the JSON
        written back to the pipe when it finishes. Evaluation results go through here; do not count on shared memory.
        """
        if self._role is not None:
            self._payload[key] = value

    # -- assembly --------------------------------------------------------------

    def arm(self, match, on_point) -> "SearchSession":
        """Hand control to ``on_point`` on the first prompt where ``match(prompt)`` is true."""
        self._match = match
        self._on_point = on_point
        self._armed = True
        return self

    # -- responder ---------------------------------------------------------

    def __call__(self, prompt, driver) -> int:
        role = self._role
        if role is not None:
            if role.prepare is not None and not role.prepared:
                raise BranchNotPreparedError(
                    "the branch was asked for a decision before permuting its hidden zones: the process still holds the opponent's "
                    "real hidden cards, and any query / evaluation counts as a leak")
            role.calls += 1
            if role.spec.max_decisions and role.calls > role.spec.max_decisions:
                raise _BranchDepthReached
            return role.policy(prompt, driver)
        if self._armed and self._match is not None and self._match(prompt):
            self._armed = False
            try:
                choice = self._on_point(prompt, self)
            except _ReturnFromResponder as signal:
                # Branch child: forked deep inside _on_point, it returns its own action from here,
                # and the driver continues with selector.choose, not one step more or less.
                return signal.action
            return int(choice)
        return self.base_responder(prompt, driver)

    # -- the host's driver --------------------------------------------------------

    def run(self, *, max_steps: int = 200000, max_seconds: float | None = 180.0):
        """Play this duel. **Branch children never return from here**; they ``_exit`` inside it."""
        from ..worldmodel.engine import StopDuel

        try:
            self.driver.run(self, max_steps=max_steps, max_seconds=max_seconds)
        except StopDuel:
            if self._role is not None:
                self._finish_branch(None)
            return self.driver
        except _BranchDepthReached:
            self._finish_branch(None)
        except BaseException as exc:  # noqa: BLE001
            if self._role is not None:
                self._finish_branch(exc)
            raise
        else:
            if self._role is not None:
                self._finish_branch(None)
        return self.driver

    # -- the branch role ----------------------------------------------------------

    def _become_branch(self, role: _BranchRole, prompt=None) -> None:
        self._role = role
        self._armed = False
        role.dirty0 = read_private_dirty_kb() if role.measure else 0
        role.base_responses = len(self.driver.responses)
        if self.record_messages:
            self.driver.record_messages = True
            self.driver.messages = []
        # Permutation before everything. If it raises, the branch finishes with the error; never "run first and see".
        if role.prepare is not None:
            chosen = role.prepare(self, role.spec, prompt)
            role.prepared = True
            if isinstance(chosen, int):
                role.action_override = chosen
        role.started = time.perf_counter()

    def _finish_branch(self, exc) -> None:
        """Branch finish: measure, send back, exit. **Does not return.**"""
        role = self._role
        assert role is not None
        driver = self.driver
        outcome = BranchOutcome(action=role.spec.action, tag=role.spec.tag)
        outcome.seconds = time.perf_counter() - role.started
        outcome.ok = exc is None
        if exc is not None:
            outcome.error = f"{type(exc).__name__}: {exc}"[:400]
        outcome.decisions = role.calls
        outcome.winner = driver.winner
        outcome.win_reason = driver.win_reason
        outcome.turn = driver.turn
        outcome.steps = driver.steps
        outcome.response_digest = _hash_responses(
            driver.responses[role.base_responses:]
        )
        if self.record_messages:
            outcome.message_digest = _hash_messages(driver.messages)
        outcome.payload = self._payload
        if role.measure:
            outcome.private_dirty_kb = read_private_dirty_kb() - role.dirty0
            outcome.rss_kb = read_rss_kb()
        outcome.visible_latency_ns = role.t_fork
        _write_json(role.write_fd, asdict(outcome))
        os._exit(0)

    # -- one-shot expansion --------------------------------------------------------

    def expand(self, prompt, specs, *, inner_policy=None, max_seconds: float = 30.0,
               measure: bool = True, parallel: int = 0, prepare=None):
        """Expand several branches at once on the current prompt. **Only call from ``on_point``.**

        ``parallel`` caps the children running at once; 0 = fork them all together. On CPU-only
        machines this decides directly how many cores are used.

        ``prepare(session, spec)`` runs once **in the child**, after the fork and before the responder
        returns. The hidden-zone permutation of replay-style particles hangs here; see
        :class:`BranchNotPreparedError`.

        Returns ``(outcomes, fork_return_ns)``.
        """
        assert_fork_safe()
        specs = list(specs)
        fan = parallel if parallel > 0 else max(1, len(specs))
        outcomes: list[BranchOutcome] = []
        latencies: list[int] = []
        pending: list[tuple[int, int, BranchSpec]] = []

        def reap_one() -> None:
            pid, read_fd, spec = pending.pop(0)
            blob = _read_all(read_fd)
            os.waitpid(pid, 0)
            if blob:
                outcomes.append(BranchOutcome(**json.loads(blob)))
            else:
                outcomes.append(BranchOutcome(
                    action=spec.action, tag=spec.tag,
                    error="the child sent back no result (most likely it crashed in the core)"))

        for spec in specs:
            while len(pending) >= fan:
                reap_one()
            read_fd, write_fd = os.pipe()
            policy = inner_policy or rollout_policy(spec.policy_seed)
            sys.stdout.flush()
            sys.stderr.flush()
            t_fork = time.perf_counter_ns()
            pid = os.fork()
            if pid == 0:
                t_child = time.perf_counter_ns()
                os.close(read_fd)
                for _p, other_fd, _s in pending:
                    try:
                        os.close(other_fd)
                    except OSError:
                        pass
                role = _BranchRole(
                    spec=spec, write_fd=write_fd, policy=policy,
                    measure=measure, t_fork=t_child - t_fork, prepare=prepare,
                )
                self._become_branch(role, prompt)
                # Return this action from the responder; the driver continues as usual,
                # and when it ends the finish in run() writes the result to the pipe and _exits.
                raise _ReturnFromResponder(
                    role.action_override
                    if role.action_override is not None else spec.action)
            latencies.append(time.perf_counter_ns() - t_fork)
            os.close(write_fd)
            pending.append((pid, read_fd, spec))

        while pending:
            reap_one()
        self.fork_return_ns.extend(latencies)
        return outcomes, latencies

    def verify_determinism(self, prompt, action: int, *, repeats: int = 2,
                           policy_seed: int = 0, max_seconds: float = 30.0):
        """Run the same branch ``repeats`` times and compare the trajectories byte by byte.

        The criterion is that both the answer digest and the message digest are equal: the answer digest covers every
        byte we feed the core, the message digest every byte the core emits; only both equal means identical trajectories.
        """
        specs = [BranchSpec(action=action, policy_seed=policy_seed, tag=f"rep{i}")
                 for i in range(repeats)]
        outcomes, _ = self.expand(prompt, specs, max_seconds=max_seconds,
                                  measure=False)
        outcomes.sort(key=lambda o: o.tag)
        identical = (
            len({o.response_digest for o in outcomes}) == 1
            and len({o.message_digest for o in outcomes}) == 1
            and all(o.ok for o in outcomes)
        )
        return identical, outcomes

    # -- resident snapshots ----------------------------------------------------------

    def snapshot(self, prompt, *, inner_policy=None) -> "Snapshot":
        """Freeze a snapshot on the current prompt that can be branched again and again. **Only call from ``on_point``.**

        :meth:`expand` expands the current prompt once and is done; tree search needs **the same node
        to be expandable again** (new priors, deeper budgets, other opponent samples). This forks at the decision point
        a "zygote" that does nothing but park on that prompt waiting for commands; each
        :meth:`Snapshot.branch` makes the zygote fork a grandchild to run. The host itself moves on as usual.

        The price is one sleeping process per live snapshot. :meth:`Snapshot.close` it when done.
        """
        assert_fork_safe()
        host_sock, child_sock = socket.socketpair()
        sys.stdout.flush()
        sys.stderr.flush()
        t_fork = time.perf_counter_ns()
        pid = os.fork()
        if pid == 0:
            host_sock.close()
            self._serve(child_sock, inner_policy)  # only grandchildren come out of here
        fork_ns = time.perf_counter_ns() - t_fork
        child_sock.close()
        return Snapshot(_sock=host_sock, _pid=pid, fork_ns=fork_ns)

    def _serve(self, sock: socket.socket, inner_policy) -> None:
        """The zygote: parked on the prompt, forking grandchildren on command. Grandchildren ``raise`` out of here."""
        live: list[int] = []
        while True:
            try:
                msg, fds, _flags, _addr = socket.recv_fds(sock, 1 << 16, 1)
            except OSError:
                break
            if not msg:
                break
            request = json.loads(msg)
            if request.get("op") != "branch":
                break
            write_fd = fds[0] if fds else None
            spec = BranchSpec(**request["spec"])
            policy = inner_policy or rollout_policy(spec.policy_seed)
            t_fork = time.perf_counter_ns()
            pid = os.fork()
            if pid == 0:
                t_child = time.perf_counter_ns()
                sock.close()
                role = _BranchRole(
                    spec=spec, write_fd=write_fd, policy=policy,
                    measure=bool(request.get("measure", True)),
                    t_fork=t_child - t_fork,
                )
                self._become_branch(role)
                raise _ReturnFromResponder(spec.action)
            if write_fd is not None:
                os.close(write_fd)
            live.append(pid)
            for done in [p for p in live if os.waitpid(p, os.WNOHANG)[0]]:
                live.remove(done)
        for pid in live:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        os._exit(0)


@dataclass
class Snapshot:
    """A frozen process parked at a decision point that can be branched again and again."""

    _sock: socket.socket
    _pid: int
    closed: bool = False
    #: the time building this snapshot took (host-side ``fork()`` return, ns)
    fork_ns: int = 0

    def branch(self, spec: BranchSpec, *, max_seconds: float = 30.0,
               measure: bool = True) -> BranchOutcome:
        if self.closed:
            raise ForkSafetyError("the snapshot is closed")
        read_fd, write_fd = os.pipe()
        try:
            socket.send_fds(
                self._sock,
                [json.dumps({"op": "branch", "spec": asdict(spec),
                             "max_seconds": max_seconds,
                             "measure": measure}).encode()],
                [write_fd],
            )
        finally:
            os.close(write_fd)
        blob = _read_all(read_fd)
        if not blob:
            return BranchOutcome(action=spec.action, tag=spec.tag,
                                 error="the zygote sent back no result")
        return BranchOutcome(**json.loads(blob))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self._sock.sendall(json.dumps({"op": "close"}).encode())
        except OSError:
            pass
        self._sock.close()
        try:
            os.waitpid(self._pid, 0)
        except ChildProcessError:
            pass
