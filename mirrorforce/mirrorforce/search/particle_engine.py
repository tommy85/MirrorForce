"""Connect replay-style particles to the ``ParticleEngine`` protocol of BPTS.

The planner lives in a separate ``search`` package; it only knows
the six methods ``reconstruct / menu / action_kind / branch / advance / outcome``,
plus the two of ``Nets``. This module implements those six methods on the real engine.

Composite handles
--------

To BPTS a handle is the pair **(engine branch state, policy-stream branch state)**. Both live in
**the same forked process**; the handle itself is only a proxy on the host side:

* engine branch state = the ``duel*`` in that process, parked at some decision point;
* policy-stream branch state = ``WorldModelPolicy.fork_for_search()`` in the same process,
  fed the messages along the way as the engine advances.

Why it must be the same process: the policy stream must consume the messages of **this hypothetical line**, and those
messages exist only in this branch's process. Keeping the policy stream on the host and shipping messages out would move
every branch's full message stream across processes: slower, and one more place to go wrong.

Why a handle is a **live process** rather than a one-shot task: the planner calls ``branch`` on the same handle again and
again (once per action of the root menu), and continues to ``branch`` on child nodes. So each handle
is a process **parked at a decision point waiting for commands**, and ``branch`` makes it fork a grandchild. This is
the shape of :class:`~mirrorforce.search.forkbranch.Snapshot`, except that these processes must also
answer menus, priors and values, so they carry a small RPC loop.

Three layers of processes
--------

1. **Host**: replays the real prefix to the decision point, once only (12.5–22.0 ms).
2. **Root zygote**: the process the host forks at the decision point, **holding the opponent's real hidden cards**.
   It accepts only ``reconstruct`` and ``close``; ``menu`` / ``prior`` / ``value`` are always
   refused. That refusal is where the first deployment rule lands at the process level: the true state is never evaluated.
3. **Particles and their descendants**: processes forked from the root zygote that **permute the hidden zones first, then park**.
   Every handle the planner sees is in this layer.

How ``branch`` and ``advance`` divide the work
--------------------------------

In the protocol ``branch(h, a)`` makes a child handle and ``advance(h, opp)`` pushes it to the next point of interest.
This implementation does the "push" inside ``branch``: the grandchild returns action ``a`` from the responder and then keeps
running until our next decision point / the end of our turn / the end of the duel before parking for commands. ``advance`` then only
reports faithfully where it parked. The behavior observable by the planner is the same (after ``advance``, ``menu`` /
``outcome`` are in place), but one process round trip is saved.

Opponent windows inside the grandchild are answered by the opponent policy, acting on the opponent's view of **this particle**,
whose sampled hand is real to them. The opponent policy is installed at construction; the ``opponent_policy``
``advance`` receives is only checked to be the same one, and a different one raises instead of silently using the old one.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from dataclasses import dataclass

from ..netduel import constants as C
from ..worldmodel.engine import DuelConfig, DuelDriver, StopDuel
from . import particles as P
from .forkbranch import assert_fork_safe

__all__ = [
    "HandleNets",
    "MCNets",
    "ParticleEngineError",
    "ParticleHandle",
    "ReplayParticleEngine",
    "StubScorer",
    "WorldModelScorer",
    "action_kind",
]

#: Our action-menu prompts. Sub-choices (which slot, which target) are not decision points of the planner
#: and are answered by the default policy.
DECISION_MSGS = frozenset(
    {C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD, C.MSG_SELECT_CHAIN})

#: Names used by the planner's ``must_include_kinds``, taken from the verb part of the
#: ``spec|act|...`` format of ``LegalAction.describe()``.
_KIND_OF = {
    "activate": "activate",
    "summon": "summon", "spsummon": "summon", "mset": "summon",
    "set": "set", "repo": "repo",
    "attack": "attack", "direct_attack": "attack",
    "to_battle": "phase", "to_main2": "phase", "to_end": "phase",
}


def action_kind(action: str) -> str:
    """``'h1|activate|eff10010'`` -> ``'activate'``; anything unrecognized becomes ``other``.

    The planner uses it to guarantee that actions whose "existence the menu reveals" are not hidden by the prior
    (the deceptive-line guarantee of the planner's design), so an unseen verb is better classified as ``other``
    than guessed into a possibly wrong category.

    **Scanning must start at segment 0.** Phase changes are bare strings **without a spec prefix**
    (``to_battle`` / ``to_main2`` / ``to_end``); scanning only ``[1:]`` would classify them all as
    ``other``, and ``"phase"`` in ``must_include_kinds`` could never be enforced,
    while "holding back and entering the Battle Phase" is exactly the line that guarantee is meant to save. The spec segments (``h1`` / ``m2`` / ``x3``)
    do not intersect the verb table, so scanning from 0 cannot misclassify.
    """
    for part in str(action).split("|"):
        if part in _KIND_OF:
            return _KIND_OF[part]
    return "other"


class ParticleEngineError(RuntimeError):
    pass


# -- scorers (the in-process side of Nets) ---------------------------------------------


class StubScorer:
    """A deterministic stub scorer: the value reads the LP difference, the prior a stable hash of the action string.

    It exists to test **the wiring between planner and engine** on its own. Neither depends on a model, so passing with the stub
    shows that handles, branching, telemetry, the root-menu guard and "with particle = truth the plan equals the omniscient
    rollout" are all correct.

    Both must be **deterministic and discriminating**: a constant-zero value makes all Q equal and argmax degenerates
    into "the first item of the menu", and such a comparison tests nothing. The LP difference is a quantity the engine
    already has and that really changes along a line, enough to rank different branches.

    For the real ``WorldModelPolicy`` binding: its scoring path consumes ``ShadowBoard`` (the online
    message stream after host filtering),
    not ``DuelDriver``; an adapter layer is missing in between.
    """

    def __init__(self, viewer: int | None = None):
        self.viewer = viewer

    def prior(self, prompt, driver) -> dict:
        menu = [a.describe() for a in prompt.actions]
        if not menu:
            return {}
        # stable, order-independent and identical across processes: the built-in hash cannot be used (PYTHONHASHSEED is random)
        import zlib

        raw = {a: (zlib.crc32(a.encode()) % 1000) / 1000.0 for a in menu}
        total = sum(raw.values()) or 1.0
        return {a: v / total for a, v in raw.items()}

    def value(self, prompt, driver) -> float:
        import ctypes

        from ..puzzle.core import SIZE_QUERY_BUFFER
        from .rebuild import _field_lp

        buf = ctypes.create_string_buffer(SIZE_QUERY_BUFFER)
        length = driver.core.query_field_info(driver.pduel, buf)
        lp = _field_lp(buf.raw[:length])
        ours = lp[self.viewer] if self.viewer is not None else lp[0]
        theirs = lp[1 - self.viewer] if self.viewer is not None else lp[1]
        return max(-1.0, min(1.0, (ours - theirs) / 8000.0))


class WorldModelScorer:
    """Score particles with this handle's own policy-stream branch.

    A handle is the pair (engine branch state, policy-stream branch state), both halves in one process; this scorer runs in
    that process, so it reads the policy stream of **this hypothetical line**, not the host's.

    The board comes directly from ``state.capture(driver)``, not through ``ShadowBoard``;
    ``DecisionState.snapshot`` is the opening for this path. The masking rule is unaffected: both paths
    go through ``worldmodel.state.mask_for`` alone (see `wmplay._score`). This was accepted by
    a differential gate: on 120 decision points,
    the particle-side snapshot and the online path's representation were field-identical and token-identical.

    Before reaching ``_score`` the board goes through :func:`~mirrorforce.search.parity.parity_filter`
    down to the online calibration (see :meth:`_state`). The differential gate runs the same path.

    **Not yet run against a real checkpoint.** The differential gate covers the half "the same state and the same token
    string go in"; "whether the scores the model gives for that token string are right" waits for weights.
    """

    def __init__(self, policy, viewer: int):
        #: the product of one ``WorldModelPolicy.fork_for_search()``
        self.policy = policy
        self.viewer = viewer

    def _state(self, prompt, driver):
        from ..netduel.policy import DecisionState
        from ..worldmodel import state as _state_mod

        from .parity import parity_filter

        # Parity alignment must happen here, before ``mask_for``: ``_score`` does
        # ``mask_for`` internally, so the ``snapshot`` handed to it must already be **reduced to the online calibration**.
        # The particle-side capture carries more than the online ``ShadowBoard`` (Xyz materials,
        # the opponent's Extra Deck identities); not trimming it lets search read inputs unavailable at deployment, and ExIt would
        # distil that gap into the policy too. The differential gate verifies exactly that the trimmed
        # copy equals the online one token by token.
        return DecisionState(
            msg=int(prompt.msg), player=int(prompt.player),
            our_player=self.viewer, actions=list(prompt.actions),
            board=None, turn=int(driver.turn), phase=int(driver.phase),
            snapshot=parity_filter(_state_mod.capture(driver), self.viewer),
            disclosure=driver.disclosure,
            turn_player=int(driver.turn_player),
        )

    def _scores(self, prompt, driver):
        import time as _time

        t0 = _time.perf_counter()
        state = self._state(prompt, driver)
        t1 = _time.perf_counter()
        out = self.policy._score(state)
        # temporary cost telemetry (smoke run 16+): each forked child logs
        # its own walls so the tree phase stops being a black box --
        # state build (engine capture + parity filter) and model separately
        print(f"[score-cost] pid={os.getpid()} state={t1 - t0:.2f}s "
              f"model={_time.perf_counter() - t1:.2f}s",
              file=sys.stderr, flush=True)
        return out

    def prior(self, prompt, driver) -> dict:
        return self.prior_from(self._scores(prompt, driver), prompt)

    def prior_from(self, scored, prompt) -> dict:
        """The prior, from an already-computed ``_scores`` result.

        One forward yields policy, value and cells together; the worker
        caches it per parked node, so reading prior *and* value costs one
        model call instead of two.
        """
        import numpy as _np

        menu = [a.describe() for a in prompt.actions]
        scores, _cells, _point, _value = scored
        if scores is None or not _np.isfinite(scores).any():
            share = 1.0 / len(menu) if menu else 0.0
            return {a: share for a in menu}
        scores = _np.asarray(scores, dtype=_np.float64)
        scores[~_np.isfinite(scores)] = -1e9
        shifted = scores - float(_np.max(scores))
        weights = _np.exp(shifted)
        total = float(weights.sum()) or 1.0
        return {a: float(w) / total for a, w in zip(menu, weights)}

    def value(self, prompt, driver) -> float:
        """The value head's scalar reading: **win − loss**, the same calibration as production.

        The fourth return value of ``wmplay._score`` is the **tuple of class probabilities** after the value head's softmax,
        not a scalar; ``float(it)`` raises ``TypeError``. Production takes
        ``value[0] - value[1]`` (``self.values.append`` in `wmplay.py`), and this
        must agree: reading it another way changes the quantity search maximizes.
        """
        if prompt is None:
            return 0.0
        return self.value_from(self._scores(prompt, driver))

    def value_from(self, scored) -> float:
        """The scalar value from an already-computed ``_scores`` result."""
        _scores, _cells, _point, value = scored
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        seq = tuple(float(v) for v in value)
        if not seq:
            return 0.0
        win = seq[0]
        loss = seq[1] if len(seq) > 1 else 0.0
        return max(-1.0, min(1.0, win - loss))


# -- host-side handle proxy --------------------------------------------------------


@dataclass
class ParticleHandle:
    """A proxy for a particle process parked at some point.

    ``status`` / ``menu`` / ``outcome_value`` are sent back when the process parks and cached on the host,
    so reading them needs no RPC.
    """

    _sock: socket.socket
    _pid: int
    tag: str = ""
    status: str = "decision"
    menu: tuple = ()
    outcome_value: float = 0.0
    turn: int = 0
    error: str = ""
    closed: bool = False
    via_action: str = ""
    #: the true-state root zygote. It holds the opponent's real hidden cards and allows only reconstruct and close.
    is_root: bool = False

    def call(self, request: dict, fds=()) -> dict:
        if self.closed:
            raise ParticleEngineError("the handle is closed")
        if self.is_root and request.get("op") not in (
                "reconstruct", "omniscient", "close"):
            raise ParticleEngineError(
                f"the root zygote holds the opponent's real hidden cards and does not accept {request.get('op')!r}: "
                "every evaluation must happen on a permuted particle")
        payload = json.dumps(request).encode()
        if fds:
            socket.send_fds(self._sock, [payload], list(fds))
        else:
            self._sock.sendall(payload)
        chunks: list[bytes] = []
        while True:
            data = self._sock.recv(1 << 16)
            if not data:
                raise ParticleEngineError(
                    f"the particle process disconnected on {request.get('op')!r}")
            chunks.append(data)
            try:
                return json.loads(b"".join(chunks))
            except json.JSONDecodeError:
                continue

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self._sock.sendall(json.dumps({"op": "close"}).encode())
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        try:
            os.waitpid(self._pid, 0)
        except (ChildProcessError, OSError):
            pass


class MCNets:
    """G3 floor group: leaf value = seeded random rollouts played to the **engine's real end of the duel**.

    Nothing learned: a uniform prior, and the value is the sample mean of real results. It is two things at once:
    a mechanical check (needs no network at all; it runs as soon as the particle adapter exists) and the G3 floor reference
    (a learned value head is worth something only if it beats this).

    ``rollouts`` is how many lines are sampled per leaf. One is enough for the floor, high-variance but unbiased; more
    trades time for variance.
    """

    def __init__(self, seed: int = 0, rollouts: int = 1):
        self.seed = int(seed)
        self.rollouts = int(rollouts)
        self.prior_calls = 0
        self.value_calls = 0
        self._counter = 0

    def policy_prior(self, handle: "ParticleHandle"):
        self.prior_calls += 1
        menu = tuple(handle.menu)
        if not menu:
            return {}
        share = 1.0 / len(menu)
        return {a: share for a in menu}

    def value(self, handle: "ParticleHandle") -> float:
        self.value_calls += 1
        self._counter += 1
        reply = handle.call({"op": "rollout",
                             "seed": self.seed + self._counter,
                             "n": self.rollouts})
        if reply.get("error"):
            raise ParticleEngineError(f"rollout failed: {reply['error']}")
        return float(reply["outcome"])


class HandleNets:
    """The ``Nets`` protocol: both prior and value are read from the handle's own policy-stream branch.

    The planner passes the handle in, so nothing here needs to know about the model: the scores are computed in that process,
    because the policy-stream branch is there.
    """

    def __init__(self):
        self.prior_calls = 0
        self.value_calls = 0

    def policy_prior(self, handle: ParticleHandle):
        import time as _t

        self.prior_calls += 1
        t0 = _t.perf_counter()
        reply = handle.call({"op": "prior"})
        print(f"[op-cost] prior {_t.perf_counter() - t0:.2f}s",
              file=sys.stderr, flush=True)
        if reply.get("error"):
            raise ParticleEngineError(f"prior evaluation failed: {reply['error']}")
        return {a: float(p) for a, p in reply["prior"].items()}

    def value(self, handle: ParticleHandle) -> float:
        import time as _t

        self.value_calls += 1
        t0 = _t.perf_counter()
        reply = handle.call({"op": "value"})
        print(f"[op-cost] value {_t.perf_counter() - t0:.2f}s",
              file=sys.stderr, flush=True)
        if reply.get("error"):
            raise ParticleEngineError(f"value evaluation failed: {reply['error']}")
        return float(reply["value"])


# -- particle processes -----------------------------------------------------------------


class _Worker:
    """A particle process: runs the engine, parks at points of interest, serves RPC.

    The host never enters this class; it only runs in forked children.
    """

    def __init__(self, sock, driver, *, viewer, opponent_policy,
                 default_policy, scorer, tag, on_first_park=None):
        self.sock = sock
        self.driver = driver
        self.viewer = viewer
        self.opponent_policy = opponent_policy
        self.default_policy = default_policy
        self.scorer = scorer
        self.tag = tag
        self.on_first_park = on_first_park
        self.status = "decision"
        self.prompt = None
        self.menu: tuple = ()
        self.outcome = 0.0
        self.fatal = ""
        self.children: list[int] = []

    # -- wrapping up -------------------------------------------------------------

    def finish(self, error: str = "") -> None:
        """After the duel ends (or fails), park at the end waiting for commands. **Never returns.**

        Every process that becomes a worker must get here. A child continues the duel by **returning**
        an action index from the responder, so when it finishes it unwinds all the way back up the host's call stack, and
        would then go on executing host code. This method (with the takeover check in
        :meth:`ReplayParticleEngine.open`) is the only exit of that path.
        """
        if error:
            self.fatal = error
        winner = self.driver.winner
        if winner is not None:
            self.status = "terminal"
            self.outcome = 1.0 if winner == self.viewer else -1.0
        elif self.fatal:
            self.status = "error"
        else:
            self.status = "turn_end"
        self.menu = ()
        self.prompt = None
        self.park()
        os._exit(0)

    def responder(self, prompt, driver) -> int:
        if prompt.player != self.viewer:
            return self.opponent_policy(prompt, driver)
        if prompt.msg not in DECISION_MSGS:
            return self.default_policy(prompt, driver)
        # Our action menu. The turn is no longer ours = this line finished our turn.
        self.prompt = prompt
        self.menu = tuple(a.describe() for a in prompt.actions)
        self.status = ("decision" if driver.turn_player == self.viewer
                       else "turn_end")
        return self.park()

    # -- parking to serve commands ----------------------------------------------------

    def park(self) -> int:
        """Serve commands until some ``branch`` turns **this process** into that branch.

        The "handle" the host sees is the process parked here. ``branch`` forks: the parent stays
        parked here (so the same handle can be expanded again and again), and the child returns the action index from here
        and the engine continues.
        """
        if self.on_first_park is not None:
            hook, self.on_first_park = self.on_first_park, None
            hook(self)
        while True:
            try:
                message, fds, _flags, _addr = socket.recv_fds(
                    self.sock, 1 << 16, 1)
            except OSError:
                os._exit(0)
            if not message:
                os._exit(0)
            request = json.loads(message)
            op = request.get("op")
            if op == "close":
                self._reap()
                os._exit(0)
            if op == "state":
                self._send({"status": self.status, "menu": list(self.menu),
                            "outcome": self.outcome, "turn": self.driver.turn,
                            "error": self.fatal})
            elif op == "prior":
                self._send(self._score("prior"))
            elif op == "value":
                self._send(self._send_value())
            elif op == "rollout":
                self._send(self._rollout(int(request.get("seed", 0)),
                                         int(request.get("n", 1))))
            elif op in ("branch", "reconstruct", "omniscient"):
                index = self._spawn(request, fds)
                if index is not None:
                    return index          # this process is that branch now
            else:
                self._send({"error": f"unknown command {op!r}"})

    def _spawn(self, request: dict, fds) -> int | None:
        """Fork a child. The parent replies with the pid and stays parked; the child returns the action index."""
        child_fd = fds[0] if fds else None
        if child_fd is None:
            self._send({"error": f"{request['op']} carries no socket for the child"})
            return None
        op = request["op"]
        if op == "branch":
            action = request["action"]
            if action not in self.menu:
                os.close(child_fd)
                self._send({"error": f"action {action!r} is not in the menu: {self.menu}"})
                return None
            index = self.menu.index(action)
        else:  # reconstruct / omniscient: the child settles first, then parks at the same point
            index = None
        pid = os.fork()
        if pid == 0:
            self.sock.close()
            self.sock = socket.socket(fileno=child_fd)
            self.children = []
            self.tag = request.get("tag", self.tag)
            # the child's game state moves on; the parent's cached score
            # would be an answer about a position this process left behind
            self._score_cache = None
            if op in ("reconstruct", "omniscient"):
                if op == "reconstruct":
                    # First deployment rule: permutation before everything. On failure park with the error; never continue.
                    try:
                        self._apply(request)
                    except BaseException as exc:  # noqa: BLE001
                        self.fatal = f"{type(exc).__name__}: {exc}"[:200]
                        self.status = "error"
                        self.menu = ()
                        self.park()
                        os._exit(0)
                # park at the same point again: the menu is unchanged (our legal actions do not depend on the opponent's hidden cards),
                # but this process is now a handle that may be evaluated.
                return self.park()
            return index
        os.close(child_fd)
        self.children.append(pid)
        self._send({"pid": pid})
        return None

    def _apply(self, request: dict) -> None:
        layout = P.HiddenLayout(
            hand=tuple(request["hand"]),
            deck=tuple(request["deck"]),
            facedown=tuple(tuple(x) for x in request["facedown"]),
        )
        P.apply_particle(self.driver, 1 - self.viewer, layout,
                         slot_history=tuple(
                             tuple(x) for x in request.get("slot_history", ())))

    def _send_value(self) -> dict:
        return self._score("value")

    def _rollout(self, seed: int, n: int) -> dict:
        """A seeded random rollout to the end; returns the real result from our side.

        Nothing learned: the continuation policy and the opponent both use seeded random play, and the leaf is **the real end
        given by the engine**, not a network's estimate. This is the G3 floor reference.

        Each rollout runs **in its own fork**: this process is a handle still to be expanded again and again,
        and playing it to the end would destroy it. The grandchild writes its result to a pipe and exits; this process stays put.
        """
        from ..worldmodel.engine import random_responder

        if self.status == "terminal":
            return {"outcome": self.outcome, "n": 0, "terminal": True}
        # The continuation policy defaults to seeded random play (the definition of the G3 floor group). A scorer may provide its own
        # continuation policy: the formal G3 groups require all four values to be V^π **under the same π**, otherwise
        # the variable "leaf source" would be confounded with "under which policy the value was computed". Without one,
        # play stays random and the floor reading of `probe_mc` does not change by a single byte.
        make_policy = getattr(self.scorer, "rollout_policy", None)
        results = []
        for i in range(max(1, n)):
            read_fd, write_fd = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(read_fd)
                value = 0.0
                try:
                    policy = (make_policy(seed * 1009 + i, self.driver)
                              if make_policy is not None
                              else random_responder(seed * 1009 + i))
                    self.driver.run(policy, max_steps=200000, max_seconds=60)
                    winner = self.driver.winner
                    if winner is not None:
                        value = 1.0 if winner == self.viewer else -1.0
                except BaseException:  # noqa: BLE001
                    value = 0.0       # a game that cannot finish counts as a draw, so it cannot bias the mean
                try:
                    os.write(write_fd, json.dumps({"v": value}).encode())
                finally:
                    os.close(write_fd)
                os._exit(0)
            os.close(write_fd)
            chunks = []
            while True:
                data = os.read(read_fd, 4096)
                if not data:
                    break
                chunks.append(data)
            os.close(read_fd)
            os.waitpid(pid, 0)
            blob = b"".join(chunks)
            results.append(json.loads(blob)["v"] if blob else 0.0)
        return {"outcome": sum(results) / len(results), "n": len(results)}

    def _score(self, want: str) -> dict:
        try:
            if self.prompt is None and want == "prior":
                return {"prior": {}}
            if self.prompt is None:
                return {"value": float(self.scorer.value(None, self.driver))}
            if not hasattr(self.scorer, "_scores"):
                # heuristic scorers (G3 floor, unit fixtures) have no shared
                # forward to cache; call them the plain way
                if want == "prior":
                    return {"prior": dict(self.scorer.prior(self.prompt,
                                                            self.driver))}
                return {"value": float(self.scorer.value(self.prompt,
                                                         self.driver))}
            # one forward serves both heads: a parked node's state never
            # changes, so the result is cached for the process's lifetime
            # (cleared in freshly forked children, whose state moved on)
            scored = getattr(self, "_score_cache", None)
            if scored is None:
                scored = self.scorer._scores(self.prompt, self.driver)
                self._score_cache = scored
            if want == "prior":
                return {"prior": dict(self.scorer.prior_from(scored,
                                                             self.prompt))}
            return {"value": float(self.scorer.value_from(scored))}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"[:200]}

    def _send(self, payload: dict) -> None:
        self.sock.sendall(json.dumps(payload).encode())

    def _reap(self) -> None:
        for pid in self.children:
            try:
                os.waitpid(pid, os.WNOHANG)
            except (ChildProcessError, OSError):
                pass


# -- engine ---------------------------------------------------------------------


class ReplayParticleEngine:
    """An implementation of the ``ParticleEngine`` protocol with particles from replay + fork + permutation.

    Lifecycle::

        engine = ReplayParticleEngine(config, responses, upto, viewer=0, core=core)
        engine.open()                       # replay once, build the root zygote
        planner = BPTS(engine, HandleNets(), cfg, opponent_policy=opp)
        result = planner.plan(completions)  # completions = [HiddenLayout, ...]
        engine.close()                      # reap the whole process tree
    """

    def __init__(self, config: DuelConfig, responses, upto: int, *,
                 viewer: int, core=None, opponent_policy=None,
                 default_policy=None, scorer=None):
        self.config = config
        self.responses = [bytes(r) for r in responses[:upto]]
        self.upto = upto
        self.viewer = viewer
        self.core = core
        self.opponent_policy = opponent_policy or (lambda prompt, drv: 0)
        self.default_policy = default_policy or (lambda prompt, drv: 0)
        self.scorer = scorer if scorer is not None else StubScorer(viewer)
        self.root: ParticleHandle | None = None
        self.handles: list[ParticleHandle] = []
        self.root_menu: tuple = ()
        #: diagnostic counters, for comparison with the planner's Telemetry
        self.stats = {"reconstruct": 0, "branch": 0, "advance": 0}

    # -- building the root zygote ----------------------------------------------------------

    def open(self) -> "ReplayParticleEngine":
        """Replay the real prefix to the decision point and fork the root zygote there.

        The root zygote holds the opponent's **real** hidden cards, so it accepts only ``reconstruct``;
        :meth:`ParticleHandle.call` refuses any evaluation of it.

        Once a process becomes a worker it can never return to host code. A child continues the duel by
        **returning** an action index from the responder, so when the duel ends it unwinds the stack back
        here; ``cell["worker"]`` is the takeover check at that moment, which goes straight into
        :meth:`_Worker.finish` and never returns.
        """
        assert_fork_safe()
        from ..puzzle.core import get_core

        core = self.core if self.core is not None else get_core()
        self.core = core
        driver = DuelDriver(self.config, core).build()
        driver._replay = list(self.responses)
        cell: dict = {}

        def at_decision(prompt, drv) -> int:
            worker = cell.get("worker")
            if worker is not None:
                return worker.responder(prompt, drv)
            if prompt.player != self.viewer or prompt.msg not in DECISION_MSGS:
                return self.default_policy(prompt, drv)
            cell["menu"] = tuple(a.describe() for a in prompt.actions)
            host_sock, child_sock = socket.socketpair()
            pid = os.fork()
            if pid == 0:
                host_sock.close()
                child = _Worker(
                    child_sock, drv, viewer=self.viewer,
                    opponent_policy=self.opponent_policy,
                    default_policy=self.default_policy,
                    scorer=self.scorer, tag="root")
                child.prompt = prompt
                child.menu = cell["menu"]
                child.status = "decision"
                cell["worker"] = child
                return child.park()
            child_sock.close()
            cell["handle"] = ParticleHandle(
                _sock=host_sock, _pid=pid, tag="root",
                menu=cell["menu"], is_root=True)
            raise StopDuel

        error = ""
        try:
            driver.run(at_decision, max_steps=200000, max_seconds=180)
        except StopDuel:
            pass
        except BaseException as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"[:200]
        worker = cell.get("worker")
        if worker is not None:
            # This process is the root zygote (a forked child): its life ends here.
            # finish normally ends with _exit(0) inside park; but the parent may shut down first
            # (planning failed -> engine.close() -> broken pipe), and finish's notification raises.
            # Either way the child must not leak back into the caller's world: the caller may be a
            # server main loop reading stdin, and one more reader corrupts the whole pipe
            # (as actually happened in a smoke run). The error goes to stderr first, then the process must die.
            try:
                if error:
                    print(f"[particle-root] replay error: {error}",
                          file=sys.stderr, flush=True)
                worker.finish(error)
            except BaseException as exc:  # noqa: BLE001
                print(f"[particle-root] finish failed: "
                      f"{type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
            finally:
                os._exit(0)
        if error:
            driver.close()
            raise ParticleEngineError(f"error while replaying to the decision point: {error}")
        if "handle" not in cell:
            driver.close()
            raise ParticleEngineError("the replay did not park at one of our decision points")
        self.root = cell["handle"]
        self.root_menu = cell["menu"]
        self.driver = driver
        return self

    # -- ParticleEngine protocol ------------------------------------------------

    def reconstruct(self, completion) -> ParticleHandle:
        """Fork a particle from the root zygote: permute the hidden zones first, then park at the same decision point."""
        if self.root is None:
            raise ParticleEngineError("call open() first")
        layout, history = _unpack(completion)
        handle = self._spawn(self.root, {
            "op": "reconstruct",
            "tag": f"p{self.stats['reconstruct']}",
            "hand": list(layout.hand),
            "deck": list(layout.deck),
            "facedown": [list(x) for x in layout.facedown],
            "slot_history": [list(x) for x in history],
        })
        self.stats["reconstruct"] += 1
        self.handles.append(handle)
        if handle.status == "error":
            raise ParticleEngineError(f"particle permutation failed: {handle.error}")
        return handle

    def reconstruct_omniscient(self) -> ParticleHandle:
        """**Offline diagnostics only**: no permutation; the true state itself is the particle.

        This is the reference of "omniscient rollout", which the planner-level form of G2 is compared against: the action
        planned with a true-state particle must equal the one planned directly on the real state.

        Its handle holds the opponent's **real hidden cards**, so online decisions must never use it,
        and no search result obtained from it may serve as a policy target. The name is this long so that
        misuse is obvious in review.
        """
        if self.root is None:
            raise ParticleEngineError("call open() first")
        handle = self._spawn(self.root, {"op": "omniscient", "tag": "truth"})
        self.handles.append(handle)
        return handle

    def menu(self, handle: ParticleHandle):
        return list(handle.menu)

    def action_kind(self, action) -> str:
        return action_kind(action)

    def branch(self, handle: ParticleHandle, action) -> ParticleHandle:
        child = self._spawn(handle, {"op": "branch", "action": action,
                                     "tag": f"{handle.tag}>{action}"})
        child.via_action = str(action)
        self.stats["branch"] += 1
        self.handles.append(child)
        return child

    def advance(self, handle: ParticleHandle, opponent_policy) -> str:
        """Report wherever the handle parked; the push happens inside ``branch`` (see the module notes)."""
        if (opponent_policy is not None
                and opponent_policy is not self.opponent_policy):
            raise ParticleEngineError(
                "the opponent policy given to advance is not the one installed at construction; "
                "the opponent answers inside the particle, and switching midway makes the opponent inconsistent along the line")
        self.stats["advance"] += 1
        if handle.status == "error":
            raise ParticleEngineError(f"branch error: {handle.error}")
        return handle.status

    def outcome(self, handle: ParticleHandle) -> float:
        return float(handle.outcome_value)

    # -- wrapping up --------------------------------------------------------------

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.close()
        self.handles = []
        if self.root is not None:
            self.root.close()
            self.root = None
        driver = getattr(self, "driver", None)
        if driver is not None:
            driver.close()
            self.driver = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # -- internals --------------------------------------------------------------

    #: hard ceiling on live particle processes per engine.  The lifecycle
    #: already bounds them (a plan's tree is menu x particles x depth, and
    #: close() reaps everything), but a bound that is merely implied by
    #: correct code elsewhere is not a bound -- a planner bug that forgot
    #: to stop expanding would fork until the box drowned.  Ordinary plans
    #: hold a few dozen handles; hitting this is always a defect upstream.
    MAX_LIVE_HANDLES = 256

    def _spawn(self, parent: ParticleHandle, request: dict) -> ParticleHandle:
        import time as _t

        live = sum(1 for h in self.handles if not h.closed)
        if live >= self.MAX_LIVE_HANDLES:
            raise ParticleEngineError(
                f"{live} particle processes still alive, more than the engine limit of "
                f"{self.MAX_LIVE_HANDLES}: the upstream planner is expanding without bound")
        t0 = _t.perf_counter()
        assert_fork_safe()
        host_sock, child_sock = socket.socketpair()
        try:
            reply = parent.call(request, fds=[child_sock.fileno()])
        finally:
            child_sock.close()
        if reply.get("error"):
            host_sock.close()
            raise ParticleEngineError(reply["error"])
        handle = ParticleHandle(_sock=host_sock, _pid=int(reply["pid"]),
                                tag=request.get("tag", ""))
        state = handle.call({"op": "state"})
        print(f"[op-cost] {request['op']} {_t.perf_counter() - t0:.2f}s "
              f"tag={request.get('tag', '')[:40]}",
              file=sys.stderr, flush=True)
        handle.status = state["status"]
        handle.menu = tuple(state["menu"])
        handle.outcome_value = float(state["outcome"])
        handle.turn = int(state["turn"])
        handle.error = state.get("error", "")
        return handle


def _unpack(completion):
    """``completion`` may be a ``HiddenLayout`` or ``(layout, slot history)``."""
    if isinstance(completion, P.HiddenLayout):
        return completion, ()
    layout, history = completion
    return layout, tuple(history)
