"""Snapshot particle engine: a single-process, fork-free ``ParticleEngine`` implementation.

The fork engine uses "process = parked handle" to copy core state that cannot be cloned; once the core can
save itself (``duel_snapshot``/``duel_rollback``),
a handle reduces to plain data: **core snapshot + driver Python state + prompt text + sub-choice
path + the line's policy-stream branch**. Only one core state is live at any time; to continue on a handle,
roll back to its snapshot and restore its Python state.

Three calibrations aligned bit for bit with the fork engine (acceptance gate 2 takes it as the reference truth):

* parking points = the responder layer: our ``DECISION_MSGS`` prompts (with their later sub-choice rounds)
  park as handles; opponent prompts are answered by ``opponent_policy``, our non-decision prompts by
  ``default_policy`` (both default to "choose item 0");
* sub-choice rounds do not touch the core: one recorded response covers the whole prompt, so core snapshots are taken only at prompt
  entry, and the sub-choice tree is rebuilt by "scripted re-entry into ``_answer``": the same entry snapshot + a longer
  ``subpath``;
* the policy stream forks from the parent line at ``branch`` entry: observations the parent already made are inherited, those not yet made
  are not borrowed in advance, the same timing as copying by process fork.

The rules are unchanged: the root handle holds the opponent's real hidden cards and accepts only ``reconstruct``; a particle line
is never scored before its permutation completes (scorers hang only on particle lines).
"""

from __future__ import annotations

import ctypes
import hashlib
from dataclasses import dataclass, field
from typing import Any

import sys

from ..netduel.actions import LegalAction
from ..puzzle.messages import Message
from ..worldmodel.engine import DECISION_MSGS, DuelError, Prompt, StopDuel
from . import particles as P
from .authority import (INFORMATION_SET_SEARCH, UNREALIZED_TRUE_HIDDEN,
                        SimulationAuthority)
from .particle_engine import ParticleEngineError, WorldModelScorer

_ROOT_SUBPATH_DOMAIN = b"mirrorforce-root-subpath-v1\0"
_EMPTY_ROOT_SUBPATH_SHA256 = hashlib.sha256(
    _ROOT_SUBPATH_DOMAIN + (0).to_bytes(8, "little")
).hexdigest()

__all__ = [
    "INFORMATION_SET_SEARCH",
    "UNREALIZED_TRUE_HIDDEN",
    "ParticleProvenance",
    "ParticleRealizationError",
    "UnrestoredSearchRootError",
    "SnapshotParticleSpec",
    "SimulationAuthority",
    "SnapshotParticleEngine",
    "SnapLine",
    "SnapNets",
    "snapshot_api",
]


class ParticleRealizationError(ParticleEngineError):
    """A particle was rejected and the original live root was restored."""


class UnrestoredSearchRootError(ParticleEngineError):
    """The live duel is unsafe to continue after a failed rollback/cleanup."""


def snapshot_api(core):
    """Bind the R4 exports off the loaded core, or raise on an old build."""
    lib = core._lib
    if not hasattr(lib, "duel_snapshot"):
        raise ParticleEngineError(
            "this core has no snapshot exports (duel_snapshot); the snapshot engine needs "
            "the rebuilt libygopro-core")
    lib.duel_snapshot.restype = ctypes.c_void_p
    lib.duel_snapshot.argtypes = [ctypes.c_ssize_t]
    lib.duel_rollback.restype = ctypes.c_int32
    lib.duel_rollback.argtypes = [ctypes.c_ssize_t, ctypes.c_void_p]
    lib.duel_snapshot_free.restype = None
    lib.duel_snapshot_free.argtypes = [ctypes.c_void_p]
    lib.duel_arena_extent.restype = ctypes.c_int64
    lib.duel_arena_extent.argtypes = [ctypes.c_ssize_t]
    # Optional R5 surface.  Old snapshot builds remain usable for legacy
    # completions and deck-order-only search; a requested future RNG particle
    # fails closed in reconstruct() when this symbol is absent.
    if hasattr(lib, "duel_set_future_seed"):
        lib.duel_set_future_seed.restype = ctypes.c_int32
        lib.duel_set_future_seed.argtypes = [
            ctypes.c_ssize_t,
            ctypes.POINTER(ctypes.c_uint32),
        ]
    return lib


class _Park(Exception):
    """Internal control flow: the walk stopped at the next parking point (carrying the parsed prompt)."""

    def __init__(self, prompt, sub_round: bool):
        super().__init__("park")
        self.prompt = prompt
        self.sub_round = sub_round


class _RootPromptMatched(Exception):
    """Internal control flow: ``root_subpath`` reached its exact root round."""

    def __init__(self, prompt: Prompt, round_index: int):
        super().__init__("root prompt matched")
        self.prompt = prompt
        self.round_index = int(round_index)


@dataclass(frozen=True)
class ParticleProvenance:
    """Auditable identity of the sampled hidden layout realized for a line.

    The digest names the *sampled assignment*, not the live duel's hidden
    state.  Children retain this exact frozen object so telemetry cannot lose
    which particle authorized a branch.
    """

    particle_id: str
    hidden_player: int
    assignment_sha256: str
    slot_history_entries: int
    #: Per-seat full hidden-layout digests.  The legacy four-field constructor
    #: remains valid, but carries no deck-order permission until the engine
    #: fills these fields while realizing a particle.
    player_assignment_sha256: tuple[tuple[int, str], ...] = ()
    #: Seats whose complete current main-deck order was explicitly realized.
    deck_order_players: tuple[int, ...] = ()
    #: Hash of an independently supplied eight-word future core seed.  The raw
    #: seed is deliberately absent from descendants and telemetry.
    future_rng_sha256: str = ""
    future_rng_words: int = 0
    #: Immutable selector lineage at the original parked root.  ``SnapLine``'s
    #: ordinary ``subpath`` may grow within a response and resets at the next
    #: response; this digest/length pair never changes across descendants.
    root_subpath_sha256: str = _EMPTY_ROOT_SUBPATH_SHA256
    root_subpath_length: int = 0


@dataclass(frozen=True)
class SnapshotParticleSpec:
    """Complete realization request for one information-set particle.

    ``opponent_layout`` is the historical required particle.  Supplying
    ``viewer_layout`` additionally samples the observer's own unknown deck
    order (while retaining their known hand/facedown identities), which is
    required before a viewer draw may be treated as particle-determined.

    ``future_core_seeds`` is an independent eight-word seed stream sampled by
    the information-set planner.  It replaces the live duel's future PRNG only
    after both layouts have been realized and before a capability is minted.
    """

    particle_id: str
    opponent_layout: P.HiddenLayout
    opponent_slot_history: tuple = ()
    viewer_layout: P.HiddenLayout | None = None
    viewer_slot_history: tuple = ()
    future_core_seeds: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not str(self.particle_id):
            raise ParticleEngineError("particle_id may not be blank")
        if not isinstance(self.opponent_layout, P.HiddenLayout):
            raise ParticleEngineError("opponent_layout must be HiddenLayout")
        if self.viewer_layout is not None and not isinstance(
            self.viewer_layout, P.HiddenLayout
        ):
            raise ParticleEngineError("viewer_layout must be HiddenLayout or None")
        if self.future_core_seeds is not None:
            seeds = tuple(self.future_core_seeds)
            if len(seeds) != 8:
                raise ParticleEngineError(
                    "future_core_seeds must contain exactly eight uint32 words"
                )
            if any(
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed < 0
                or seed > 0xFFFFFFFF
                for seed in seeds
            ):
                raise ParticleEngineError(
                    "future_core_seeds entries must be uint32 integers"
                )


@dataclass
class SnapLine:
    """A handle: a hypothetical line that can be rolled back to at any time."""

    tag: str
    status: str                    # decision | turn_end | terminal | error
    menu: tuple = ()
    outcome_value: float = 0.0
    error: str = ""
    turn: int = 0
    snap: int | None = None        # core snapshot handle (ctypes void_p int)
    pystate: dict | None = None
    message: Message | None = None  # the pending prompt's raw message
    prompt: Any = None              # parsed Prompt (actions carry the menu)
    # Engine-internal replay cursor.  It may encode private selector choices:
    # never pass a SnapLine or this field to an evaluator.  The future
    # PublicDecisionInput boundary must omit it entirely.
    root_subpath: tuple = ()        # immutable selector path at search entry
    subpath: tuple = ()             # sub-round choices already fixed
    policy: Any = None              # this line's stream branch (scoring)
    authority: SimulationAuthority = UNREALIZED_TRUE_HIDDEN
    particle_provenance: ParticleProvenance | None = None
    via_action: str = ""
    closed: bool = False
    _scored: Any = None             # cached scorer._scores result
    _children: list = field(default_factory=list)
    #: Opaque, engine-minted authority.  The engine also records the exact line
    #: identity, so copying a valid line does not copy its permission.
    _capability: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class _LineCapabilityRecord:
    """Immutable state bound to one exact registered ``SnapLine`` object."""

    token: object
    root_subpath: tuple[int, ...]
    subpath: tuple[int, ...]
    provenance: ParticleProvenance
    provenance_signature: tuple


class SnapNets:
    """The ``Nets`` protocol: prior and value read the line's own policy-stream branch **in this process**.

    Isomorphic to the fork engine's ``HandleNets``; one ``_scores`` forward serves both heads
    (once a handle is parked its state no longer changes, so the cache is always valid).
    """

    def __init__(self, engine: "SnapshotParticleEngine"):
        self.engine = engine
        self.prior_calls = 0
        self.value_calls = 0

    def _scorer(self, line: SnapLine):
        self.engine._require_search_authority(line, "score")
        return self.engine.scorer_factory(line)

    def policy_prior(self, line: SnapLine):
        self.prior_calls += 1
        self.engine._require_search_authority(line, "score")
        if line.prompt is None:
            return {}
        scorer = self._scorer(line)
        self.engine._activate(line)
        if hasattr(scorer, "_scores"):
            # world-model scorer: one forward serves both heads, cached per handle
            if line._scored is None:
                line._scored = scorer._scores(line.prompt, self.engine.driver)
            return dict(scorer.prior_from(line._scored, line.prompt))
        return dict(scorer.prior(line.prompt, self.engine.driver))

    def value(self, line: SnapLine) -> float:
        self.value_calls += 1
        self.engine._require_search_authority(line, "score")
        if line.status == "terminal":
            return float(line.outcome_value)
        if line.prompt is None:
            return 0.0
        scorer = self._scorer(line)
        self.engine._activate(line)
        if hasattr(scorer, "_scores"):
            if line._scored is None:
                line._scored = scorer._scores(line.prompt, self.engine.driver)
            return float(scorer.value_from(line._scored))
        return float(scorer.value(line.prompt, self.engine.driver))


class SnapshotParticleEngine:
    """The ``ParticleEngine`` protocol: reconstruct / menu / branch / advance / outcome.

    ``driver`` must already be parked on a prompt awaiting a decision (the shape after ``at_pending`` raises ``StopDuel``:
    the core waits for a response, ``_answering_msg/_answering_payload`` are in place).
    The engine does not own the driver's lifetime: the mirror duel persists across plans, and ``close()`` only
    releases this plan's snapshots.
    """

    #: hard cap on live handles per plan, the same discipline as the fork engine
    MAX_LIVE_LINES = 256

    def __init__(self, driver, prompt, *, viewer: int, base_policy=None,
                 scorer_factory=None, opponent_policy=None,
                 default_policy=None, max_seconds: float = 60.0,
                 park_all_viewer_prompts: bool = False,
                 park_all_prompts: bool = False,
                 root_subpath: tuple[int, ...] = ()):
        from .particle_engine import StubScorer

        self.driver = driver
        self.viewer = int(viewer)
        self.base_policy = base_policy
        stub = None if base_policy is not None else StubScorer(int(viewer))
        self.scorer_factory = scorer_factory or (
            (lambda line: WorldModelScorer(line.policy, int(viewer)))
            if base_policy is not None else (lambda line: stub))
        self.opponent_policy = opponent_policy or (lambda p, d: 0)
        self.default_policy = default_policy or (lambda p, d: 0)
        # The legacy BPTS curriculum only parked top-level DECISION_MSGS and
        # delegated later selector messages to ``default_policy``.  A full
        # menu-level MCTS must keep target/tribute/material/position/etc. as
        # explicit viewer nodes instead of silently choosing option zero.
        # Keep the old behaviour as the default; the strict v2 adapter opts in.
        self.park_all_viewer_prompts = bool(park_all_viewer_prompts)
        # Stronger opt-in used by information-set MCTS: both players' main and
        # private selector prompts become explicit nodes.  It takes precedence
        # over both fallback policies.  Default=False preserves the historical
        # BPTS/SnapNets contract byte-for-byte.
        self.park_all_prompts = bool(park_all_prompts)
        self.max_seconds = float(max_seconds)
        self.api = snapshot_api(driver.core)
        self.stats = {"reconstruct": 0, "branch": 0, "advance": 0,
                      "rollback": 0}
        self._active: SnapLine | None = None
        self._poisoned = False
        self._poison_reason = ""
        self._closed = False
        # id(line) -> exact opaque token.  A dataclass copy carrying the same
        # fields/token is still unauthorized because it was never registered.
        self._line_capabilities: dict[int, _LineCapabilityRecord] = {}
        self.lines: list[SnapLine] = []
        if driver._answering_msg is None:
            raise ParticleEngineError("the driver is not on a prompt awaiting a decision")
        root_subpath = _checked_root_subpath(root_subpath)
        self._root_subpath = root_subpath
        self._root_subpath_sha256 = _root_subpath_digest(root_subpath)
        if int(getattr(prompt, "player", -1)) != self.viewer:
            raise ParticleEngineError(
                f"root prompt actor {getattr(prompt, 'player', None)!r} "
                f"differs from viewer {self.viewer}")
        if int(getattr(prompt, "msg", -1)) != int(driver._answering_msg):
            raise ParticleEngineError(
                f"root prompt msg {getattr(prompt, 'msg', None)!r} differs from the pending message "
                f"{int(driver._answering_msg)}")
        if not tuple(getattr(prompt, "actions", ())):
            raise ParticleEngineError("the root prompt has no legal action")
        pystate = driver.save_pystate()
        snap = self._snapshot()
        try:
            message = Message(msg=int(driver._answering_msg),
                              payload=bytes(driver._answering_payload))
            fresh_prompt = self._validate_root_prompt(
                message, prompt, root_subpath, snap=snap, pystate=pystate)
            self.root = SnapLine(
                tag="root", status="decision",
                menu=tuple(a.describe() for a in fresh_prompt.actions),
                pystate=pystate, snap=snap, message=message,
                prompt=fresh_prompt, root_subpath=root_subpath,
                subpath=root_subpath, policy=None,
                turn=int(fresh_prompt.turn),
                authority=UNREALIZED_TRUE_HIDDEN,
                particle_provenance=None, _capability=None)
        except BaseException:
            self.api.duel_snapshot_free(snap)
            raise
        self._active = self.root
        self.root_menu = self.root.menu

    def _validate_root_prompt(self, message, prompt, root_subpath, *, snap, pystate):
        """Replay the immutable selector prefix and require the exact root row.

        A multi-round selector lives only inside :meth:`DuelDriver._answer`.
        Parking by raising out of round N loses that local selector, so the
        snapshot line must retain rounds ``[:N]`` and rebuild them from the raw
        message on every branch.  Validation runs against the just-created root
        snapshot, then unconditionally rolls both core and Python state back.
        """

        cursor = {"round": 0}

        def replay_path(candidate, _driver) -> int:
            round_index = cursor["round"]
            cursor["round"] = round_index + 1
            if int(getattr(candidate, "player", -1)) != self.viewer:
                raise ParticleEngineError(
                    f"root_subpath round {round_index} actor "
                    f"{getattr(candidate, 'player', None)!r} != viewer {self.viewer}")
            if int(getattr(candidate, "msg", -1)) != int(message.msg):
                raise ParticleEngineError(
                    f"root_subpath round {round_index} msg "
                    f"{getattr(candidate, 'msg', None)!r} != {int(message.msg)}")
            if round_index < len(root_subpath):
                index = root_subpath[round_index]
                actions = tuple(getattr(candidate, "actions", ()))
                if not 0 <= index < len(actions):
                    raise ParticleEngineError(
                        f"root_subpath[{round_index}]={index} is beyond that round's menu length "
                        f"{len(actions)}")
                return index

            # The root round is exactly the next callback after consuming the
            # supplied prefix.  Never scan for a matching menu: two selector
            # rounds may expose byte-identical rows.
            _validate_root_prompt_fields(prompt, candidate)
            raise _RootPromptMatched(candidate, round_index)

        fresh_prompt = None
        try:
            self.driver._answer(message, replay_path)
        except _RootPromptMatched as matched:
            if matched.round_index != len(root_subpath):
                raise ParticleEngineError(
                    "root_subpath replay reached the wrong selector round")
            fresh_prompt = matched.prompt
        except ParticleEngineError:
            raise
        except Exception as exc:
            raise ParticleEngineError(
                f"root_subpath replay failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            self._rollback_or_poison(
                snap, "duel_rollback failed after checking root_subpath")
            self.driver.restore_pystate(pystate)
            self.stats["rollback"] += 1
            self._active = None
        if fresh_prompt is None:
            raise ParticleEngineError(
                "root_subpath consumed the whole response without reaching the given root prompt")
        return fresh_prompt

    # -- core state round trips --------------------------------------------------------

    def _ensure_usable(self, operation: str) -> None:
        if getattr(self, "_poisoned", False):
            raise ParticleEngineError(
                f"the snapshot engine is poisoned ({self._poison_reason}); refusing {operation}")
        if getattr(self, "_closed", False):
            raise ParticleEngineError(f"the snapshot engine is closed; refusing {operation}")

    def _release_snapshot_handles(self) -> None:
        """Free every owned snapshot without changing core or Python state."""

        freed = set()
        for line in getattr(self, "lines", ()):
            if line.snap is not None and line.snap not in freed:
                self.api.duel_snapshot_free(line.snap)
                freed.add(line.snap)
            line.snap = None
            line.closed = True
        root = getattr(self, "root", None)
        if root is not None:
            if root.snap is not None and root.snap not in freed:
                self.api.duel_snapshot_free(root.snap)
            root.snap = None
            root.closed = True
        self._line_capabilities.clear()
        self.lines = []
        self._active = None

    def _poison(self, reason: str) -> None:
        """Retire an engine whose core state can no longer be synchronized."""

        if self._poisoned:
            return
        self._poisoned = True
        self._poison_reason = str(reason)
        self._closed = True
        # Snapshot buffers live on the system heap and remain safe to free even
        # when the duel rollback failed.  Restoring Python state here would be
        # dishonest: the live core is explicitly in an unknown state.
        self._release_snapshot_handles()

    def _rollback_or_poison(self, snap: int, context: str) -> None:
        try:
            rc = int(self.api.duel_rollback(self.driver.pduel, snap))
        except Exception as exc:
            reason = f"{context}: {type(exc).__name__}: {exc}"
            self._poison(reason)
            raise UnrestoredSearchRootError(reason) from exc
        if rc != 0:
            reason = f"{context} (rc={rc})"
            self._poison(reason)
            raise UnrestoredSearchRootError(reason)

    def _snapshot(self) -> int:
        self._ensure_usable("snapshot")
        snap = self.api.duel_snapshot(self.driver.pduel)
        if not snap:
            raise ParticleEngineError("duel_snapshot failed (arena limit exceeded?)")
        return snap

    def _activate(self, line: SnapLine) -> None:
        self._ensure_usable("activate")
        if self._active is line:
            return
        if line.snap is None or line.pystate is None:
            raise ParticleEngineError(
                f"handle {line.tag} is terminal ({line.status}) and cannot be activated")
        self._rollback_or_poison(line.snap, "duel_rollback failed")
        self.driver.restore_pystate(line.pystate)
        self.stats["rollback"] += 1
        self._active = line

    # -- ParticleEngine protocol ------------------------------------------------

    def reconstruct(self, completion) -> SnapLine:
        self._ensure_usable("reconstruct")
        request = _unpack(completion)
        if self.root.root_subpath != self._root_subpath \
                or self.root.subpath != self._root_subpath \
                or _root_subpath_digest(self.root.root_subpath) \
                != self._root_subpath_sha256:
            raise ParticleEngineError(
                "the immutable root_subpath was altered before the particle was realized")
        self._activate(self.root)
        # First rule: permutation before any read. A failed permutation fails the whole plan (half a particle is
        # more dangerous than none).
        particle_id = (
            str(request.particle_id)
            if request.particle_id is not None
            else f"p{self.stats['reconstruct']}"
        )
        hidden_player = 1 - self.viewer
        snap = None
        try:
            assignments = [
                (
                    hidden_player,
                    request.opponent_layout,
                    tuple(request.opponent_slot_history),
                )
            ]
            if request.viewer_layout is not None:
                assignments.append(
                    (
                        self.viewer,
                        request.viewer_layout,
                        tuple(request.viewer_slot_history),
                    )
                )
            for player, layout, history in assignments:
                P.apply_particle(
                    self.driver,
                    player,
                    layout,
                    slot_history=history,
                )
            if request.future_core_seeds is not None:
                self._set_future_seed(request.future_core_seeds)
            # Save Python state before allocating a core snapshot: if field
            # inventory validation fails, there is no snapshot handle to leak.
            pystate = self.driver.save_pystate()
            snap = self._snapshot()
        except Exception as exc:
            # apply_particle may have changed part of the true-hidden root
            # before rejecting an invalid layout.  Never leave that half-realized
            # state active for a later reconstruct call.
            self._restore_root_after_failed_realization()
            if isinstance(exc, P.ParticleError):
                # The public sampler may propose a layout the core cannot
                # realize. Convert only after successful core + Python rollback;
                # higher layers can then skip this search, not the real game.
                raise ParticleRealizationError(str(exc)) from exc
            raise
        player_digests = tuple(
            (player, _layout_digest(layout))
            for player, layout, _history in assignments
        )
        future_rng_sha256 = (
            _seed_digest(request.future_core_seeds)
            if request.future_core_seeds is not None
            else ""
        )
        provenance = ParticleProvenance(
            particle_id=particle_id,
            hidden_player=hidden_player,
            assignment_sha256=_assignment_digest(player_digests),
            slot_history_entries=sum(len(history) for _, _, history in assignments),
            player_assignment_sha256=player_digests,
            deck_order_players=tuple(sorted(player for player, _, _ in assignments)),
            future_rng_sha256=future_rng_sha256,
            future_rng_words=(8 if future_rng_sha256 else 0),
            root_subpath_sha256=self._root_subpath_sha256,
            root_subpath_length=len(self._root_subpath),
        )
        capability = object()
        line = SnapLine(
            tag=particle_id, status="decision",
            menu=self.root.menu, snap=snap,
            pystate=pystate, message=self.root.message,
            prompt=self.root.prompt,
            root_subpath=self._root_subpath,
            subpath=self._root_subpath,
            policy=(self.base_policy.fork_for_search()
                    if self.base_policy is not None else None),
            authority=INFORMATION_SET_SEARCH,
            particle_provenance=provenance, _capability=capability,
            turn=self.root.turn)
        self.stats["reconstruct"] += 1
        try:
            self._register(line)
        except Exception:
            if snap is not None:
                self.api.duel_snapshot_free(snap)
            self._restore_root_after_failed_realization()
            raise
        # The permutation touched the core; the root snapshot is still as it was before; the active line is now the particle line
        self._active = line
        return line

    def _set_future_seed(self, seeds) -> None:
        self._ensure_usable("set future seed")
        if not hasattr(self.api, "duel_set_future_seed"):
            raise ParticleEngineError(
                "future RNG particle requested, but this core lacks "
                "duel_set_future_seed"
            )
        words = tuple(int(seed) for seed in seeds)
        if len(words) != 8 or any(seed < 0 or seed > 0xFFFFFFFF for seed in words):
            raise ParticleEngineError(
                "future RNG particle requires exactly eight uint32 words"
            )
        raw = (ctypes.c_uint32 * 8)(*words)
        if self.api.duel_set_future_seed(self.driver.pduel, raw) != 0:
            raise ParticleEngineError("duel_set_future_seed failed")

    def menu(self, line: SnapLine):
        self._ensure_usable("read menu")
        return list(line.menu)

    def action_kind(self, action) -> str:
        self._ensure_usable("read action kind")
        from .particle_engine import action_kind

        return action_kind(action)

    def branch(self, parent: SnapLine, action) -> SnapLine:
        self._require_search_authority(parent, "branch")
        if action not in parent.menu:
            raise ParticleEngineError(
                f"action {action!r} is not in the menu: {parent.menu}")
        return self.branch_index(parent, parent.menu.index(action))

    def branch_index(self, parent: SnapLine, index: int) -> SnapLine:
        """Branch by the exact prompt row, preserving duplicate descriptions.

        ``LegalAction.describe()`` is presentation text, not a unique action
        identity.  The ragged-menu adapter therefore binds an action to its
        selector row and calls this method.  Existing string callers retain
        their historical first-match behaviour through :meth:`branch`.
        """

        self._require_search_authority(parent, "branch")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ParticleEngineError(
                f"the action index must be an int, got {type(index).__name__}")
        if not 0 <= index < len(parent.menu):
            raise ParticleEngineError(
                f"action index {index} is beyond the menu length {len(parent.menu)}")
        child = self._walk(parent, index)
        child.via_action = str(parent.menu[index])
        self.stats["branch"] += 1
        self._register(child)
        return child

    def advance(self, line: SnapLine, opponent_policy) -> str:
        self._require_search_authority(line, "advance")
        if (opponent_policy is not None
                and opponent_policy is not self.opponent_policy):
            raise ParticleEngineError(
                "the opponent policy given to advance is not the one installed at construction")
        self.stats["advance"] += 1
        if line.status == "error":
            raise ParticleEngineError(f"branch error: {line.error}")
        return line.status

    def outcome(self, line: SnapLine) -> float:
        self._require_search_authority(line, "score")
        return float(line.outcome_value)

    # -- walking ---------------------------------------------------------------

    def _walk(self, parent: SnapLine, index: int) -> SnapLine:
        """Walk from ``parent`` by index ``index`` to the next parking point."""
        self._require_search_authority(parent, "branch")
        self._activate(parent)
        driver = self.driver
        policy = (parent.policy.fork_for_search()
                  if parent.policy is not None else None)
        taken = {"n": 0}
        parked: dict = {}

        def scripted(prompt, drv) -> int:
            # within the current prompt: replay the sub-choices already made, then this action; the sub-choice
            # round after it is a new parking point
            k = taken["n"]
            taken["n"] += 1
            if k < len(parent.subpath):
                return int(parent.subpath[k])
            if k == len(parent.subpath):
                return index
            raise _Park(prompt, sub_round=True)

        def walker(prompt, drv) -> int:
            if self.park_all_prompts:
                raise _Park(prompt, sub_round=not prompt.is_decision)
            if prompt.player != self.viewer:
                return int(self.opponent_policy(prompt, drv))
            if self.park_all_viewer_prompts:
                raise _Park(prompt, sub_round=not prompt.is_decision)
            if prompt.msg not in DECISION_MSGS:
                return int(self.default_policy(prompt, drv))
            raise _Park(prompt, sub_round=False)

        tag = f"{parent.tag}>{index}"
        try:
            try:
                driver._answer(parent.message, scripted)
            except _Park as park:
                # the next sub-choice round of the same prompt: the core has not moved; the snapshot is the parent's
                child = SnapLine(
                    tag=tag, status=parent.status,
                    menu=tuple(a.describe() for a in park.prompt.actions),
                    snap=parent.snap, pystate=parent.pystate,
                    message=parent.message, prompt=park.prompt,
                    root_subpath=parent.root_subpath,
                    subpath=parent.subpath + (index,) + tuple(
                        _extra_choices(taken["n"], parent.subpath, index)),
                    policy=policy, turn=parent.turn,
                    authority=parent.authority,
                    particle_provenance=parent.particle_provenance,
                    _capability=parent._capability)
                # The bytes are shared with the parent, but the logical parked
                # selector is now the child.  Marking it active forces a real
                # rollback before a sibling reuses the parent; otherwise the
                # live core may already be beyond that prompt while _activate
                # incorrectly returns early.
                self._active = child
                return child
            # the prompt is answered (set_responseb happened); walk on to the next parking point
            driver.run(walker, max_steps=200000, max_seconds=self.max_seconds)
        except _Park as park:
            snap_ = self._snapshot()
            child = SnapLine(
                tag=tag,
                # Under the full-prompt protocol actor comes from the prompt,
                # not from whose turn it is.  In particular, an opponent chain
                # response during our turn is a real decision node, not the old
                # catch-all ``turn_end`` status.
                status=(
                    "decision"
                    if self.park_all_prompts
                    or driver.turn_player == self.viewer
                    else "turn_end"
                ),
                menu=tuple(a.describe() for a in park.prompt.actions),
                snap=snap_, pystate=driver.save_pystate(),
                message=Message(msg=int(driver._answering_msg),
                                payload=bytes(driver._answering_payload)),
                prompt=park.prompt, root_subpath=parent.root_subpath,
                subpath=(), policy=policy,
                turn=int(driver.turn), authority=parent.authority,
                particle_provenance=parent.particle_provenance,
                _capability=parent._capability)
            self._active = child
            return child
        except StopDuel:
            pass
        except DuelError as exc:
            self._active = None
            return SnapLine(tag=tag, status="error",
                            error=f"{type(exc).__name__}: {exc}"[:200],
                            policy=policy, turn=int(driver.turn),
                            root_subpath=parent.root_subpath,
                            authority=parent.authority,
                            particle_provenance=parent.particle_provenance,
                            _capability=parent._capability)
        # the duel ran to its natural end: the terminal semantics match the fork engine's finish()
        self._active = None
        winner = driver.winner
        if winner is not None:
            return SnapLine(
                tag=tag, status="terminal",
                outcome_value=(
                    0.0
                    if winner == 2
                    else (1.0 if winner == self.viewer else -1.0)
                ),
                policy=policy, turn=int(driver.turn),
                root_subpath=parent.root_subpath,
                authority=parent.authority,
                particle_provenance=parent.particle_provenance,
                _capability=parent._capability)
        return SnapLine(tag=tag, status="turn_end", policy=policy,
                        prompt=None, turn=int(driver.turn),
                        root_subpath=parent.root_subpath,
                        authority=parent.authority,
                        particle_provenance=parent.particle_provenance,
                        _capability=parent._capability)

    # -- bookkeeping and wrapping up ---------------------------------------------------------

    def _register(self, line: SnapLine) -> None:
        self._ensure_usable("register line")
        live = sum(1 for l in self.lines if not l.closed)
        if live >= self.MAX_LIVE_LINES:
            raise ParticleEngineError(
                f"{live} live handles, more than the engine limit of {self.MAX_LIVE_LINES}: "
                "the upstream planner is expanding without bound")
        provenance = line.particle_provenance
        if (line.authority is not INFORMATION_SET_SEARCH
                or not isinstance(provenance, ParticleProvenance)
                or line._capability is None):
            raise ParticleEngineError(
                f"handle {line.tag} has no search authority from a realized particle; refusing to register")
        if id(line) in self._line_capabilities:
            raise ParticleEngineError(f"handle {line.tag} is already registered")
        root_subpath = _checked_root_subpath(line.root_subpath)
        subpath = _checked_root_subpath(line.subpath)
        if (provenance.root_subpath_length != len(root_subpath)
                or provenance.root_subpath_sha256
                != _root_subpath_digest(root_subpath)):
            raise ParticleEngineError(
                f"the root_subpath of handle {line.tag} disagrees with the particle provenance")
        self.lines.append(line)
        self._line_capabilities[id(line)] = _LineCapabilityRecord(
            token=line._capability,
            root_subpath=tuple(root_subpath),
            subpath=tuple(subpath),
            provenance=provenance,
            provenance_signature=_provenance_signature(provenance),
        )

    def _require_search_authority(self, line: SnapLine, operation: str) -> None:
        """Fail closed unless ``line`` is an engine-minted realized particle."""
        self._ensure_usable(operation)
        if not isinstance(line, SnapLine):
            raise ParticleEngineError(
                f"{operation} needs a SnapLine, got {type(line).__name__}")
        registered = self._line_capabilities.get(id(line))
        provenance = line.particle_provenance
        lineage_ok = False
        if registered is not None and isinstance(provenance, ParticleProvenance):
            try:
                root_subpath = _checked_root_subpath(line.root_subpath)
                subpath = _checked_root_subpath(line.subpath)
                lineage_ok = (
                    root_subpath == registered.root_subpath
                    and subpath == registered.subpath
                    and provenance is registered.provenance
                    and _provenance_signature(provenance)
                    == registered.provenance_signature
                    and provenance.root_subpath_length == len(root_subpath)
                    and provenance.root_subpath_sha256
                    == _root_subpath_digest(root_subpath)
                )
            except (ParticleEngineError, TypeError, ValueError, OverflowError):
                lineage_ok = False
        if (line.closed
                or line.authority is not INFORMATION_SET_SEARCH
                or not isinstance(provenance, ParticleProvenance)
                or line._capability is None
                or registered is None
                or registered.token is not line._capability
                or not lineage_ok):
            authority = getattr(line.authority, "value", repr(line.authority))
            raise ParticleEngineError(
                f"handle {line.tag} has no search capability from a realized particle; "
                f"authority={authority}, refusing {operation}")

    def _restore_root_after_failed_realization(self) -> None:
        """Return the live duel to the only safe state after realize failure."""
        self._ensure_usable("restore failed realization")
        self._active = None
        if self.root.snap is None or self.root.pystate is None:
            raise UnrestoredSearchRootError("the root snapshot cannot be restored after a failed particle realization")
        self._rollback_or_poison(
            self.root.snap, "duel_rollback of the root snapshot failed after a failed particle realization")
        try:
            self.driver.restore_pystate(self.root.pystate)
        except Exception as exc:
            reason = "particle failure restored the core but not Python root state"
            self._poison(reason)
            raise UnrestoredSearchRootError(reason) from exc
        self.stats["rollback"] += 1
        self._active = self.root

    def close(self) -> None:
        if self._poisoned or self._closed:
            return
        if self.root.snap is not None:
            # the end of planning brings the mirror duel back onto the real trajectory (permutations and hypothetical lines all vanish)
            self._rollback_or_poison(
                self.root.snap, "duel_rollback of the root snapshot failed at the end of planning")
            self.driver.restore_pystate(self.root.pystate)
        self._release_snapshot_handles()
        self._closed = True


def _extra_choices(calls: int, subpath, index: int):
    """The number of calls scripted had consumed when ``_Park`` interrupted it, only for robustness checks."""
    # scripted parks after len(subpath)+1 calls; there is no extra chosen item
    del calls, subpath, index
    return ()


_ROOT_PROMPT_FIELDS = (
    "index",
    "turn",
    "turn_player",
    "phase",
    "msg",
    "player",
    "complete_menu",
    "truncated",
    "note",
    # the chooser's card-selection context at this round (R3); a replayed
    # root must reproduce it exactly like the menu
    "selection",
)

_LEGAL_ACTION_FIELDS = (
    "spec",
    "act",
    "phase",
    "finish",
    "position",
    "effect",
    "number",
    "place",
    "attribute",
    "race",
    "code",
    "response",
    "msg",
    "desc",
)


def _legal_action_signature(action, *, row: int) -> tuple:
    if type(action) is not LegalAction:
        raise ParticleEngineError(
            f"root prompt action[{row}] must be a LegalAction, got "
            f"{type(action).__name__}")
    actual_fields = set(vars(action))
    expected_fields = set(_LEGAL_ACTION_FIELDS)
    if actual_fields != expected_fields:
        raise ParticleEngineError(
            f"root prompt action[{row}] schema drift: missing="
            f"{sorted(expected_fields - actual_fields)} unknown="
            f"{sorted(actual_fields - expected_fields)}")
    return tuple(getattr(action, field) for field in _LEGAL_ACTION_FIELDS)


def _validate_root_prompt_fields(supplied, fresh: Prompt) -> None:
    """Require the caller's description to match one freshly replayed prompt."""

    if type(supplied) is not Prompt:
        raise ParticleEngineError(
            f"the root prompt must be a Prompt, got {type(supplied).__name__}")
    if type(fresh) is not Prompt:
        raise ParticleEngineError(
            f"the core replay returned a non-Prompt: {type(fresh).__name__}")
    expected_fields = set(_ROOT_PROMPT_FIELDS) | {"actions"}
    for label, candidate in (("given", supplied), ("core replay", fresh)):
        actual_fields = set(vars(candidate))
        if actual_fields != expected_fields:
            raise ParticleEngineError(
                f"root prompt {label} Prompt schema drift: missing="
                f"{sorted(expected_fields - actual_fields)} unknown="
                f"{sorted(actual_fields - expected_fields)}")
    for field in _ROOT_PROMPT_FIELDS:
        supplied_value = getattr(supplied, field)
        fresh_value = getattr(fresh, field)
        if supplied_value != fresh_value:
            raise ParticleEngineError(
                f"root prompt {field} drifted: given {supplied_value!r}, "
                f"core replay {fresh_value!r}")
    supplied_actions = tuple(
        _legal_action_signature(action, row=row)
        for row, action in enumerate(supplied.actions)
    )
    fresh_actions = tuple(
        _legal_action_signature(action, row=row)
        for row, action in enumerate(fresh.actions)
    )
    if supplied_actions != fresh_actions:
        raise ParticleEngineError(
            "the ordered LegalAction fields of the root prompt differ from the core replay")


def _checked_root_subpath(value) -> tuple[int, ...]:
    """Return one immutable, non-negative selector prefix or fail closed."""

    if not isinstance(value, tuple):
        raise ParticleEngineError(
            f"root_subpath must be an immutable tuple, got {type(value).__name__}")
    for round_index, index in enumerate(value):
        if isinstance(index, bool) or not isinstance(index, int) \
                or index < 0 or index > 0xFFFFFFFFFFFFFFFF:
            raise ParticleEngineError(
                f"root_subpath[{round_index}] must be a non-negative int, got {index!r}")
    return value


def _root_subpath_digest(value) -> str:
    path = _checked_root_subpath(value)
    body = _ROOT_SUBPATH_DOMAIN + len(path).to_bytes(8, "little")
    for index in path:
        body += int(index).to_bytes(8, "little", signed=False)
    return hashlib.sha256(body).hexdigest()


def _provenance_signature(provenance: ParticleProvenance) -> tuple:
    """Snapshot all immutable provenance fields for registry validation."""

    return (
        str(provenance.particle_id),
        int(provenance.hidden_player),
        str(provenance.assignment_sha256),
        int(provenance.slot_history_entries),
        tuple(
            (int(player), str(digest))
            for player, digest in provenance.player_assignment_sha256
        ),
        tuple(int(player) for player in provenance.deck_order_players),
        str(provenance.future_rng_sha256),
        int(provenance.future_rng_words),
        str(provenance.root_subpath_sha256),
        int(provenance.root_subpath_length),
    )


@dataclass(frozen=True)
class _RealizationRequest:
    opponent_layout: P.HiddenLayout
    opponent_slot_history: tuple
    particle_id: str | None
    viewer_layout: P.HiddenLayout | None = None
    viewer_slot_history: tuple = ()
    future_core_seeds: tuple[int, ...] | None = None


def _unpack(completion) -> _RealizationRequest:
    if isinstance(completion, SnapshotParticleSpec):
        return _RealizationRequest(
            opponent_layout=completion.opponent_layout,
            opponent_slot_history=tuple(completion.opponent_slot_history),
            particle_id=str(completion.particle_id),
            viewer_layout=completion.viewer_layout,
            viewer_slot_history=tuple(completion.viewer_slot_history),
            future_core_seeds=(
                tuple(completion.future_core_seeds)
                if completion.future_core_seeds is not None
                else None
            ),
        )
    if isinstance(completion, P.HiddenLayout):
        return _RealizationRequest(completion, (), None)
    if len(completion) == 2:
        layout, history = completion
        return _RealizationRequest(layout, tuple(history), None)
    if len(completion) == 3:
        tag, layout, history = completion
        return _RealizationRequest(layout, tuple(history), str(tag))
    raise ParticleEngineError(
        "a particle completion must be a SnapshotParticleSpec, HiddenLayout, "
        "(layout, history) or (tag, layout, history)")


def _layout_digest(layout: P.HiddenLayout) -> str:
    """Stable identifier for sampled assignment provenance (never live truth)."""
    body = repr(layout.assignment()).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _assignment_digest(player_digests) -> str:
    body = repr(tuple(player_digests)).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _seed_digest(seeds) -> str:
    body = b"mirrorforce-future-core-rng-v1\0" + b"".join(
        int(seed).to_bytes(4, "little", signed=False) for seed in seeds
    )
    return hashlib.sha256(body).hexdigest()
