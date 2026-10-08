"""Local transaction envelopes for a permanent blank/fact client core.

This is a transaction boundary, not a search-admission certificate. The owner
must stop external native consumers before attaching the controller; all real
receive/advance/respond calls then go through it. Shared-core native calls and
reader caches are serialized. Arriving packets are journaled outside rollback.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import math
import pickle
import random
import sys
import threading
import time
import weakref
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Callable

from ..immutable import ImmutableRecord
from ..netduel import host_view
from ..netduel import constants as C
from ..puzzle.core import PROCESSOR_BUFFER_LEN, PROCESSOR_END, PROCESSOR_FLAG
from ..puzzle.messages import split_messages
from ..puzzle.single import RESPONSE_REQUIRED
from ..search.snap_engine import snapshot_api
from ..worldmodel.engine import DuelDriver
from .client_shadow import BlankClientSync, hydrate, reorder_deck, reorder_hand
from .client_sync import FrozenPyState, _prompt_player

SCHEMA = "client-local-root/v1"
OPAQUE_SCHEMA = "client-opaque-public-decision/v1"
INFORMATION_SET_SEARCH = True
SEARCH_READY = False


class RootError(RuntimeError):
    pass


class RootBusy(RootError):
    pass


class StaleRoot(RootError):
    pass


class RootTimeout(RootError):
    pass


class SnapshotOwnershipError(RootError):
    pass


class RestoreMismatch(RootError):
    pass


_CORE_LOCKS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_FOLLOWER_OWNERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_OWNERS_LOCK = threading.Lock()
_GUARDED = ("receive", "respond", "advance", "close", "_free", "_release")
_ALIASES = ("core", "local", "api")
_SYNC_FIELDS = frozenset((
    "viewer", "decks", "seed", "rules", "max_answers", "max_fixes", "max_seed_tries",
    "max_choices", "max_tries", "spent", "packets", "cursor", "own", "answered", "local_random",
    "parked", "origin", "last_local", "choices", "stats", "failure", "prompt_maps",
    "semantic_remaps", "history_registry", "history_registry_receipt", "unresolved_history",
    "record_origins", "record_replay", "phase_pass_audit", "public_action_rebind", "public_action_target_witness",
    "public_action_witness_main", "hidden_hand_rebind", "hidden_target_deferral", "declined_prompt_audit", "public_identities_only", "public_board", "public_started", "public_board_problems", "public_identities", "read_answers", "reading_only", "single_pass", "category_constraints", "answer_states", "order_facts", "order_rewinds", "order_live", "order_saved", "hidden_target_origin", "construction", "constructions_tried", "own_origin", "hidden_target_skip", "hidden_target_pool", "phase_pass_consume", "phase_pass_ambiguous", "phase_pass_kept",
    "last_response_own", "deferred_action", "receipt_wire_packets",
    "receipt_packet_raw_indices", "receipt_state", "receipt_saved", "public_opponent_recipe",
))


class _AdvanceBoundary:
    """One real, locked advance; never a search branch or a reusable capability."""

    def __init__(self, owner):
        self.owner, self.thread = owner, threading.get_ident()
        self.active, self.phase = True, "idle"
        self.frames, self.requests = [], []
        self.history_markers = ()
        self.history_proofs = ()


def _advance_boundary(follower, token=None):
    owner = _FOLLOWER_OWNERS.get(follower)
    scope = getattr(owner, "_advance_scope", None)
    if (owner is None or type(scope) is not _AdvanceBoundary or not scope.active
            or scope.owner is not owner or scope.thread != threading.get_ident()
            or token is not None and token is not scope or owner.follower is not follower
            or not owner._busy or owner._root is not None or not owner._native_lock.locked()):
        raise SnapshotOwnershipError("no current owned real-advance boundary")
    return scope


def _bind_started_follower(follower):
    scope = _advance_boundary(follower)
    owner = scope.owner
    if (scope.phase != "idle" or follower.core is not owner.core or follower.local is None
            or follower.local.core is not owner.core or follower.local.follower is not follower
            or not follower.local.pduel):
        raise SnapshotOwnershipError("invalid started follower binding at real advance")
    pair = (follower.local, follower.local.pduel)
    if owner._entity_binding is not None and owner._entity_binding != pair:
        raise SnapshotOwnershipError("real advance replaced an already owned native duel")
    owner._entity_binding = pair


_RECORD_DIGESTS: dict[int, tuple] = {}


def _deeply_immutable(value) -> bool:
    if value is None or type(value) in (str, bytes, int, float, bool) or isinstance(value, (Enum, ImmutableRecord)):
        return True  # a nested record is checked when its own digest is taken
    if isinstance(value, (tuple, frozenset)):
        return all(_deeply_immutable(item) for item in value)
    return is_dataclass(value) and not isinstance(value, type) and value.__dataclass_params__.frozen \
        and all(_deeply_immutable(getattr(value, item.name)) for item in fields(value))


def _record_digest(value, pool):
    key = id(value)
    row = _RECORD_DIGESTS.get(key)
    if row is not None and row[0]() is value:
        return row[1]
    if not is_dataclass(value) or not value.__dataclass_params__.frozen \
            or not all(_deeply_immutable(getattr(value, item.name)) for item in fields(value)):
        raise RootError(f"immutable record holds mutable state: {type(value).__qualname__}")
    body = tuple((item.name, _canonical(getattr(value, item.name), {}, pool)) for item in fields(value))
    # Interned: equal records then share one digest object, so the pickled root
    # digest never depends on which of two equal records a state holds.
    digest = sys.intern(hashlib.sha256(pickle.dumps(body, protocol=4)).hexdigest())

    def drop(ref, key=key):
        if _RECORD_DIGESTS.get(key, (None,))[0] is ref:
            del _RECORD_DIGESTS[key]
    _RECORD_DIGESTS[key] = (weakref.ref(value, drop), digest)
    return digest


def _canonical(value, seen=None, immutable_pool=None):
    """Stable values AND mutable-alias topology; no pointer addresses in digests."""
    seen = {} if seen is None else seen
    if value is immutable_pool and immutable_pool is not None:
        return ("shared-readonly-card-pool",)
    if value is None or type(value) in (str, bytes, int, float, bool):
        return (type(value).__name__, value)
    if isinstance(value, Enum):
        return (type(value).__module__, type(value).__qualname__, value.value)
    if isinstance(value, ImmutableRecord):
        return ("immutable", type(value).__module__, type(value).__qualname__, _record_digest(value, immutable_pool))
    if isinstance(value, tuple):
        return ("tuple", tuple(_canonical(x, seen, immutable_pool) for x in value))
    if isinstance(value, frozenset):
        return ("frozenset", tuple(sorted((_canonical(x, {}, immutable_pool) for x in value), key=repr)))
    if id(value) in seen:
        return ("ref", seen[id(value)])
    number = len(seen)
    seen[id(value)] = number
    kind = (type(value).__module__, type(value).__qualname__)
    if isinstance(value, random.Random):
        body = value.getstate()
    elif isinstance(value, dict):
        body = tuple((_canonical(k, seen, immutable_pool), _canonical(value[k], seen, immutable_pool))
                     for k in sorted(value, key=lambda key: repr(_canonical(key))))
    elif isinstance(value, list):
        body = tuple(_canonical(x, seen, immutable_pool) for x in value)
    elif isinstance(value, set):
        body = tuple(sorted((_canonical(x, {}, immutable_pool) for x in value), key=repr))
    elif is_dataclass(value):
        body = tuple((f.name, _canonical(getattr(value, f.name), seen, immutable_pool)) for f in fields(value))
    elif any("__slots__" in vars(cls) for cls in type(value).__mro__):
        names = set()
        for cls in type(value).__mro__:
            slots = vars(cls).get("__slots__", ())
            names.update((slots,) if isinstance(slots, str) else slots)
        body = tuple((name, _canonical(getattr(value, name), seen, immutable_pool))
                     for name in sorted(names - {"__dict__", "__weakref__"}) if hasattr(value, name))
    elif hasattr(value, "__dict__") and not callable(value):
        body = _canonical(vars(value), seen, immutable_pool)
    else:
        raise RootError(f"unsupported mutable root state: {kind}")
    return ("object", number, kind, body)


def _digest(value, pool=None):
    return hashlib.sha256(pickle.dumps(_canonical(value, immutable_pool=pool), protocol=4)).hexdigest()


def _pool_digest(pool):
    """The shared card database's content, compared before and after a root or branch in this process.

    The pool is one large read-only graph (about 15,000 cards). Its pickled
    bytes keep every value, type and alias in it, and take some 25 times less
    time than the canonical form, which is only needed for digests that must
    agree across processes.
    """
    return hashlib.sha256(pickle.dumps(vars(pool), protocol=4)).hexdigest()


def _copy_card_pool(pool):
    """Detach the static database graph without Python's per-field deepcopy walk.

    This round trip uses ONLY bytes just produced from our own trusted pool,
    never an artifact or a received pickle. It keeps card/list/dictionary
    aliases inside the clone, but shares no mutable object with the parent.
    Branch entry/exit still fingerprint the entire graph; this is not a shared
    mutable-pool shortcut. Nonstandard pool classes retain their copy hooks.
    """
    from ..netduel.cards import CardPool
    if type(pool) is not CardPool:
        return copy.deepcopy(pool)
    return pickle.loads(pickle.dumps(pool, protocol=4))


def _clone(value, pool=None, replacement=None):
    memo = {} if pool is None else {id(pool): pool if replacement is None else replacement}
    return copy.deepcopy(value, memo)


def _driver_state(driver):
    # Keep the existing unknown-field guard, but copy the graph jointly below:
    # independent deepcopy of _rng and _ctx.rng would break their alias.
    driver.save_pystate()
    return {name: getattr(driver, name) for name in (*driver._PYSTATE_FIELDS, "record_messages", "config")
            if hasattr(driver, name)}


def _mutable_tokens(value, seen=None, *, require_detached=False):
    """Reject shallow container copies and detached-but-shared Torch storage."""
    seen = set() if seen is None else seen
    if value is None or isinstance(value, (str, bytes, int, float, bool, Enum)) or id(value) in seen:
        return set()
    seen.add(id(value))
    out = set() if isinstance(value, (tuple, frozenset)) else {("object", id(value))}
    if hasattr(value, "untyped_storage") and callable(value.untyped_storage):
        if require_detached and (value.requires_grad or value.grad_fn is not None):
            raise SnapshotOwnershipError("state hook snapshot retains an autograd graph; use detach().clone()")
        storage = value.untyped_storage()
        out.add(("tensor-storage", str(value.device), getattr(storage, "_cdata", storage.data_ptr())))
        return out
    children = (list(value.keys()) + list(value.values()) if isinstance(value, dict) else
                list(value) if isinstance(value, (list, tuple, set, frozenset)) else
                [getattr(value, f.name) for f in fields(value)] if is_dataclass(value) else
                list(vars(value).values()) if hasattr(value, "__dict__") else [])
    for child in children:
        out |= _mutable_tokens(child, seen, require_detached=require_detached)
    return out


@dataclass(frozen=True)
class StateHook:
    """Owner-supplied snapshot for model memory, published observations or caches.

    clone must allocate independent mutable leaves (Tensor.detach is NOT a
    clone). digest describes semantic values. quiesce waits for asynchronous
    consumers; restore must update the actual owner, not just a local variable.
    Ownership does not grant information authority: live client memory may
    inherit only its own observations; a simulated opponent's memory must be
    derived from public information/the particle, never omniscient opponent
    memory copied from self-play. The inference adapter must enforce that law.
    """

    capture: Callable[[], Any]
    clone: Callable[[Any], Any]
    restore: Callable[[Any], None]
    digest: Callable[[Any], str]
    quiesce: Callable[[], None]

    def owned_copy(self, value):
        result = self.clone(value)
        if _mutable_tokens(value) & _mutable_tokens(result, require_detached=True):
            raise SnapshotOwnershipError("state hook clone aliases mutable state or tensor storage")
        if self.digest(value) != self.digest(result):
            raise SnapshotOwnershipError("state hook clone changed its value")
        return result


class _ReaderState:
    """Restore game-dependent reader buffers; move branch errors to audit logs."""

    def __init__(self, core):
        self.core = core
        self.maps = {}
        for name in ("_script_cache", "_card_cache"):
            original = getattr(core, name)
            self.maps[name] = (original, {key: (value, bytes(value)) for key, value in original.items()})
        self.sets = {name: (getattr(core, name), set(getattr(core, name)))
                     for name in ("_known_missing", "missing_codes")}
        self.log_object, self.log = core.log, list(core.log)
        self.pool = core.card_pool()
        self.pool_digest = _pool_digest(self.pool)

    def restore(self, audit):
        current = list(self.core.log)
        if current != self.log:
            audit.append({"kind": "branch-core-log", "entries":
                          current[len(self.log):] if current[:len(self.log)] == self.log else current})
        for name, (original, rows) in self.maps.items():
            original.clear()
            for key, (value, raw) in rows.items():
                ctypes.memmove(ctypes.addressof(value), raw, len(raw))
                original[key] = value
            setattr(self.core, name, original)
        for name, (original, values) in self.sets.items():
            original.clear()
            original.update(values)
            setattr(self.core, name, original)
        self.log_object[:] = self.log
        self.core.log = self.log_object
        if self.core.card_pool() is not self.pool or _pool_digest(self.pool) != self.pool_digest:
            raise RestoreMismatch("shared read-only card pool was mutated outside the branch")


class _Snapshot:
    def __init__(self, api, duel):
        self.api, self.duel, self.closed = api, duel, False
        self.handle = api.duel_snapshot(duel)
        if not self.handle:
            raise RootError("native root snapshot failed")
        extent = int(api.duel_arena_extent(duel))
        length = ctypes.c_size_t.from_address(int(self.handle)).value
        if not 0 < length == extent <= 256 * 1024 * 1024:
            self.close()
            raise RootError("unsupported snapshot layout/extent")
        self.digest = hashlib.sha256(ctypes.string_at(int(self.handle) + ctypes.sizeof(ctypes.c_size_t), length)).hexdigest()

    def restore(self):
        if self.closed:
            raise SnapshotOwnershipError("snapshot already released")
        if self.api.duel_rollback(self.duel, self.handle) != 0:
            raise RestoreMismatch("native rollback failed")

    def verify(self):
        current = _Snapshot(self.api, self.duel)
        try:
            if current.digest != self.digest:
                raise RestoreMismatch("native arena differs after rollback")
        finally:
            current.close()

    def close(self):
        if not self.closed:
            self.closed = True
            self.api.duel_snapshot_free(self.handle)


def _saved_handles(follower):
    saved = [follower.origin, getattr(follower, "hidden_target_origin", None), getattr(follower, "own_origin", None)]
    saved += [item for choice in follower.choices for item in (choice.saved, choice.origin)]
    saved += list(getattr(follower, "answer_states", ()))
    construction = getattr(follower, "construction", None)
    if construction is not None:
        saved += list(construction.candidates) + list(construction.held().values())
    return {item[0]: item for item in saved if item is not None}


class _CatchupTransaction:
    """Rollback a deferred real-history catch-up without rolling back input.

    Existing follower snapshots are borrowed until commit, so failed replay
    can restore the exact choice graph.  New handles are owned here until they
    either enter the committed graph or are freed during rollback.
    """

    def __init__(self, owner):
        self.owner, self.follower = owner, owner.follower
        self.host = owner._capture_host()
        self.host_digest = _digest(self.host, owner.core.card_pool())
        self.snapshot = _Snapshot(snapshot_api(owner.core), self.follower.local.pduel)
        self.reader = _ReaderState(owner.core)
        self.audit = list(owner.audit)
        self.protected = _saved_handles(self.follower)
        self.created, self.pending = {}, {}
        self.original_save, self.original_free = self.follower._save, self.follower._free
        self.save_override = vars(self.follower).get("_save", None)
        self.free_override = vars(self.follower).get("_free", None)

        def saved():
            value = self.original_save()
            self.created[value[0]] = value
            return value

        def freed(value):
            if value is None:
                return
            handle = value[0]
            if handle in self.protected:
                self.pending[handle] = self.protected[handle]
                return
            self.created.pop(handle, None)
            self.original_free(value)

        self.follower._save, self.follower._free = saved, freed
        self.finished = False

    def _restore_methods(self):
        if self.save_override is None:
            vars(self.follower).pop("_save", None)
        else:
            self.follower._save = self.save_override
        if self.free_override is None:
            vars(self.follower).pop("_free", None)
        else:
            self.follower._free = self.free_override

    def commit(self):
        if self.finished:
            raise SnapshotOwnershipError("deferred catch-up transaction already finished")
        self._restore_methods()
        live = _saved_handles(self.follower)
        for handle, saved in self.pending.items():
            if handle not in live:
                self.original_free(saved)
        self.snapshot.close()
        self.finished = True

    def rollback(self):
        if self.finished:
            return
        self._restore_methods()
        failure = None
        try:
            self.snapshot.restore()
            self.owner._restore_host(self.host)
            self.reader.restore([])  # candidate/failed replay details never enter owner audit
            self.owner.audit[:] = self.audit
            for saved in tuple(self.created.values()):
                self.original_free(saved)
            self.snapshot.verify()
            if _digest(self.owner._capture_host(), self.owner.core.card_pool()) != self.host_digest:
                raise RestoreMismatch("deferred catch-up did not restore follower/driver state")
        except BaseException as exc:
            failure = exc
        finally:
            self.snapshot.close()
            self.finished = True
        if failure is not None:
            raise failure


class OpaqueRoot:
    """A public server decision with no native root or model-state capability."""

    schema, search_ready, native_ready = OPAQUE_SCHEMA, False, False
    _IMMUTABLE = frozenset(("owner", "thread_id", "epoch", "server_prompt", "selector_path",
                            "host_digest", "native_digest", "candidate_set_sha256", "candidate_count"))

    def __setattr__(self, name, value):
        if name in self._IMMUTABLE and name in vars(self):
            raise SnapshotOwnershipError("opaque decision metadata is immutable: " + name)
        object.__setattr__(self, name, value)

    def __init__(self, owner):
        state = owner.follower.deferred_action
        self.owner, self.thread_id, self.epoch = owner, threading.get_ident(), owner.epoch
        self.server_prompt, self.selector_path = bytes(state.server_prompt), ()
        self.candidate_set_sha256, self.candidate_count = state.candidate_set_sha256, state.candidate_count
        self.host_digest = _digest(owner._capture_host(), owner.core.card_pool())
        probe = _Snapshot(snapshot_api(owner.core), owner.follower.local.pduel)
        try:
            self.native_digest = probe.digest
        finally:
            probe.close()
        self.closed = self.used = False

    def _require_owner_thread(self):
        if threading.get_ident() != self.thread_id:
            raise SnapshotOwnershipError("opaque decision belongs to its creating thread")

    def close(self):
        self._require_owner_thread()
        if self.closed:
            return
        if self.owner._opaque_root is not self:
            raise SnapshotOwnershipError("only the exact opaque decision may be released")
        self.closed = True
        self.owner._opaque_root = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class ClientRootController:
    """Exclusive local owner; inbound journal/epoch are deliberately not rolled back."""

    def __init__(self, follower: BlankClientSync, *, quiesce: Callable[[], None],
                 native_idle: Callable[[], bool], hooks: dict[str, StateHook] | None = None,
                 on_packet: Callable[[bytes], None] | None = None, clock=time.monotonic):
        if not isinstance(follower, BlankClientSync):
            raise RootError("only permanent blank/fact followers are accepted")
        quiesce()
        if not native_idle():
            raise SnapshotOwnershipError("external native/asynchronous consumers still own mutable state")
        with _OWNERS_LOCK:
            if follower in _FOLLOWER_OWNERS:
                raise SnapshotOwnershipError("follower already has a root owner")
            # Not reentrant: another controller on the SAME thread must not
            # enter while this duel borrows shared callback/script caches.
            self._native_lock = _CORE_LOCKS.setdefault(follower.core, threading.Lock())
            _FOLLOWER_OWNERS[follower] = self
        self.follower, self.core = follower, follower.core
        self._entity_binding = ((follower.local, follower.local.pduel)
                                if follower.local is not None else None)
        self.quiesce, self.native_idle, self.hooks = quiesce, native_idle, dict(hooks or {})
        self.on_packet, self.clock = on_packet, clock
        self._input_lock = threading.RLock()
        self._inbox, self._journal = deque(), []
        self._epoch, self._busy, self._root, self._opaque_root = 0, False, None, None
        self._issued_roots = weakref.WeakValueDictionary()
        self._issued_opaque_roots = weakref.WeakValueDictionary()
        self._closed, self._poison_reason = False, None
        self._quiescent = False
        self._advance_scope = None
        self.audit = []
        self._original = {name: getattr(follower, name) for name in _GUARDED}
        self._overrides = {name: vars(follower).get(name) for name in _GUARDED if name in vars(follower)}
        follower.receive, follower.advance, follower.close = self.receive, self.advance, self.close
        follower.respond = self._reject_direct_response

    def _install_free_guards(self):
        self.follower._free = lambda saved: self._guard_follower_free("_free", saved)
        self.follower._release = lambda choice: self._guard_follower_free("_release", choice)

    def _remove_free_guards(self):
        for name in ("_free", "_release"):
            if name in self._overrides:
                setattr(self.follower, name, self._overrides[name])
            elif name in vars(self.follower):
                delattr(self.follower, name)

    @property
    def epoch(self):
        with self._input_lock:
            return self._epoch

    @property
    def journal(self):
        with self._input_lock:
            return tuple(self._journal)

    @property
    def pending_packets(self):
        with self._input_lock:
            return tuple(self._inbox)

    def _ensure_open(self):
        if self._closed or self._poison_reason or self.follower.failure:
            raise RootError("local root owner is closed or poisoned: " + str(self._poison_reason or self.follower.failure))

    @contextmanager
    def _native_operation(self):
        if not self._native_lock.acquire(False):
            raise RootBusy("shared core is in another operation or transaction")
        try:
            yield
        finally:
            self._native_lock.release()

    def _poison(self, reason):
        self._poison_reason = str(reason)
        self.follower.failure = "root transaction poisoned: " + str(reason)
        self.audit.append({"kind": "poison", "reason": str(reason)})

    def _quiesce(self):
        self._quiescent = False
        quiet = False
        try:
            self.quiesce()
            for hook in self.hooks.values():
                hook.quiesce()
            if not self.native_idle():
                raise RootBusy("native/asynchronous consumers have not released branch state")
            quiet = True
            self._quiescent = True
        finally:
            if not quiet:
                self._poison("quiescence failed; snapshots retained until explicit safe abort")

    def _guard_follower_free(self, method, value):
        raise SnapshotOwnershipError("real follower origin/choice snapshots are borrowed and frozen")

    def _reject_direct_response(self, _data):
        raise RootError("real responses require commit_real_response with a current root epoch")

    def receive(self, packet):
        packet = bytes(packet)
        with self._input_lock:
            if self._closed:
                raise RootError("closed receiver; transport must be detached before close")
            self._journal.append(packet)
            self._inbox.append(packet)
            self._epoch += 1
        if self._root is None and self._opaque_root is None and not self._busy \
                and not self._poison_reason and self._native_lock.acquire(False):
            try:
                if self._root is None and self._opaque_root is None and not self._busy:
                    self._drain()
            finally:
                self._native_lock.release()

    def _drain(self):
        with self._input_lock:
            while self._inbox:
                packet = self._inbox[0]
                delivered = False
                try:
                    self._original["receive"](packet)
                    if self.on_packet is not None:
                        self.on_packet(packet)
                    delivered = True
                finally:
                    if not delivered:
                        self._poison("real packet delivery failed; packet retained in inbox")
                self._inbox.popleft()

    def advance(self):
        self._ensure_open()
        if self._root is not None or self._opaque_root is not None or self._busy:
            raise RootBusy("real advance is frozen during search or another operation")
        if not self._native_lock.acquire(False):
            raise RootBusy("shared core is in a transaction")
        try:
            if self._root is not None or self._opaque_root is not None or self._busy:
                raise RootBusy("real advance is frozen during search")
            self._busy = True
            self._quiesce()
            self._drain()
            self._quiesce()
            with self._input_lock:
                self._epoch += 1
            if getattr(self.follower, "record_origins", False):
                self._advance_scope = _AdvanceBoundary(self)
            transaction = None
            deferred = getattr(self.follower, "deferred_action", None)
            if deferred is not None and len(self.follower.own) > deferred.own_index:
                transaction = _CatchupTransaction(self)
            try:
                result = self._original["advance"]()
                if transaction is not None:
                    transaction.commit()
            except BaseException as exc:
                if transaction is not None:
                    try:
                        transaction.rollback()
                    except BaseException as restore:
                        self._poison("deferred catch-up restoration failed: " + str(restore))
                        raise restore from exc
                    self._poison("deferred catch-up rejected: " + str(exc))
                raise
            if self._entity_binding is None and self.follower.local is not None:
                self._entity_binding = (self.follower.local, self.follower.local.pduel)
            return result
        finally:
            if self._advance_scope is not None:
                self._advance_scope.active = False
                self._advance_scope.phase = "closed"
                self._advance_scope = None
            self._busy = False
            self._native_lock.release()

    def _capture_host(self):
        unknown = set(vars(self.follower)) - _SYNC_FIELDS - set(_ALIASES) - set(_GUARDED)
        if unknown:
            raise RootError("unclassified follower fields: " + repr(sorted(unknown)))
        raw = {"sync": {name: getattr(self.follower, name) for name in _SYNC_FIELDS if hasattr(self.follower, name)},
               "driver": _driver_state(self.follower.local)}
        return _clone(raw, self.core.card_pool())

    def _restore_host(self, saved):
        restored = _clone(saved, self.core.card_pool())
        for name, value in restored["sync"].items():
            setattr(self.follower, name, value)
        self.follower.local.restore_pystate(restored["driver"], consume=True)

    def capture_root(self, *, selector_path=(), max_seconds=30.0):
        self._ensure_open()
        if getattr(self.follower, "deferred_action", None) is not None:
            raise RootError("deferred public decision is opaque and cannot become a native RootEnvelope")
        if self._root is not None or self._opaque_root is not None or self._busy:
            raise RootBusy("another root or operation already freezes this follower")
        if any(type(index) is not int or index < 0 for index in selector_path):
            raise RootError("selector path must contain nonnegative integer choices")
        if not math.isfinite(max_seconds) or max_seconds <= 0 or not self._native_lock.acquire(False):
            raise RootBusy("invalid budget or shared core busy")
        success = False
        try:
            if self._root is not None or self._opaque_root is not None or self._busy:
                raise RootBusy("another root already freezes this follower")
            self._busy = True
            self._drain()
            self._quiesce()
            sync = self.follower
            if (sync.local is None or sync.local.finished or sync.parked is None
                    or _prompt_player(sync.parked) != sync.viewer or sync.answered != len(sync.own)
                    or sync.cursor != len(sync.packets) or sync.cursor == 0):
                raise RootError("follower is not at a fully consumed, unanswered own prompt")
            if sync.local.on_replay_prompt is not None:
                raise SnapshotOwnershipError("detach real model callbacks; use explicit state hooks for branch memory")
            self._install_free_guards()
            root = RootEnvelope(self, tuple(selector_path), float(max_seconds))
            self._root, success = root, True
            self._issued_roots[id(root)] = root
            return root
        finally:
            self._busy = False
            if not success:
                self._remove_free_guards()
                self._native_lock.release()

    def capture_opaque_root(self):
        """Freeze only a received public menu; never expose native/model state."""
        self._ensure_open()
        if self.hooks:
            raise RootError("opaque public decisions cannot attach policy/model/cache state hooks")
        if self._root is not None or self._opaque_root is not None or self._busy \
                or not self._native_lock.acquire(False):
            raise RootBusy("another root/operation exists or the shared core is busy")
        try:
            self._busy = True
            self._drain()
            self._quiesce()
            state = getattr(self.follower, "deferred_action", None)
            if state is None or len(self.follower.own) != state.own_index:
                raise RootError("follower has no unanswered deferred public prompt")
            root = OpaqueRoot(self)
            self._opaque_root = root
            self._issued_opaque_roots[id(root)] = root
            return root
        finally:
            self._busy = False
            self._native_lock.release()

    def commit_opaque_response(self, root, data, *, validate_original, send):
        """Commit one real response without pretending the native follower is at this prompt."""
        self._ensure_open()
        if type(root) is not OpaqueRoot:
            raise SnapshotOwnershipError("opaque response requires an exact OpaqueRoot")
        root._require_owner_thread()
        if (root.owner is not self or self._issued_opaque_roots.get(id(root)) is not root
                or root.used or not root.closed or self._opaque_root is not None):
            raise SnapshotOwnershipError("response needs this owner's released, unused opaque decision")
        with self._native_operation():
            with self._input_lock:
                if self._epoch != root.epoch or self._inbox:
                    raise StaleRoot("new public input invalidated this opaque decision")
            if _digest(self._capture_host(), self.core.card_pool()) != root.host_digest:
                self._poison("host changed before opaque response commit")
                raise RestoreMismatch("host changed after opaque decision release")
            probe = _Snapshot(snapshot_api(self.core), self.follower.local.pduel)
            try:
                if probe.digest != root.native_digest:
                    self._poison("native state changed before opaque response commit")
                    raise RestoreMismatch("native state changed after opaque decision release")
            finally:
                probe.close()
            self._busy = True
            try:
                data = bytes(data)
                if not validate_original(root.server_prompt, (), data):
                    raise RootError("response does not belong to the original opaque server menu")
                with self._input_lock:
                    if self._epoch != root.epoch or self._inbox:
                        raise StaleRoot("public input arrived while validating the opaque response")
                    root.used = True
                sent = False
                try:
                    send(data)
                    self._original["respond"](data)  # ledger only; native replay is deliberately delayed
                    with self._input_lock:
                        self._epoch += 1
                    sent = True
                finally:
                    if not sent:
                        self._poison("opaque real response send failed; delivery is indeterminate")
            finally:
                self._busy = False
                if not self._poison_reason:
                    self._drain()

    def commit_real_response(self, root, data, *, validate_original, send):
        self._ensure_open()
        if type(root) is not RootEnvelope:
            raise SnapshotOwnershipError("native response requires an exact RootEnvelope")
        root._require_owner_thread()
        if (root.owner is not self or self._issued_roots.get(id(root)) is not root
                or root.used or not root.closed or self._root is not None):
            raise SnapshotOwnershipError("response needs this owner's released, unused root")
        with self._native_operation():
            with self._input_lock:
                if self._epoch != root.epoch or self._inbox:
                    raise StaleRoot("new public input or state invalidated this decision")
            if _digest(self._capture_host(), self.core.card_pool()) != root.host_digest:
                self._poison("host changed outside its owner before response commit")
                raise RestoreMismatch("host changed after root release")
            probe = _Snapshot(root.api, root.duel)
            try:
                if probe.digest != root.snapshot.digest:
                    self._poison("native state changed outside its owner before response commit")
                    raise RestoreMismatch("native state changed after root release")
            finally:
                probe.close()
            for name, hook in self.hooks.items():
                hook.quiesce()
                if hook.digest(hook.capture()) != root.hook_digests[name]:
                    self._poison("model/cache state changed before response commit")
                    raise RestoreMismatch("state hook changed after root release: " + name)
            self._busy = True
            try:
                data = bytes(data)
                if not validate_original(root.server_prompt, root.selector_path, data):
                    raise RootError("response does not belong to the original server menu")
                with self._input_lock:
                    if self._epoch != root.epoch or self._inbox:
                        raise StaleRoot("public input arrived while validating the original response")
                    # Linearization point: later input is queued, not dropped,
                    # but cannot retract an already committed transport send.
                    root.used = True
                sent = False
                try:
                    send(data)
                    self._original["respond"](data)
                    with self._input_lock:
                        self._epoch += 1
                    sent = True
                finally:
                    if not sent:
                        self._poison("real response send failed; delivery is indeterminate")
            finally:
                self._busy = False
                if not self._poison_reason:
                    self._drain()

    def _uninstall(self):
        for name in _GUARDED:
            if name in self._overrides:
                setattr(self.follower, name, self._overrides[name])
            elif name in vars(self.follower):
                delattr(self.follower, name)
        self._closed = True
        with _OWNERS_LOCK:
            _FOLLOWER_OWNERS.pop(self.follower, None)

    def detach(self):
        """Hand a healthy live follower back after its transport consumers pause."""
        self._ensure_open()
        if self._root is not None or self._opaque_root is not None or self._busy:
            raise RootBusy("cannot hand off a follower with a live transaction")
        with self._native_operation():
            self._quiesce()
            self._drain()
            self._quiesce()
            self._uninstall()
        return self.follower

    def close(self):
        if self._closed:
            return
        if self._busy:
            raise RootBusy("cannot close during a real input/response operation")
        if self._root is not None:
            if self._root.branch_session is not None:
                raise RootBusy("cannot close a follower with an active branch")
            self._root.close()
        if self._opaque_root is not None:
            self._opaque_root.close()
        with self._native_operation():
            self._quiesce()
            if not self._poison_reason:
                self._drain()
            self._quiesce()
            self._uninstall()
            self._original["close"]()


class RootEnvelope:
    """Owns its root snapshot; origin/choice snapshots remain follower-owned."""

    schema, search_ready = SCHEMA, False
    _IMMUTABLE = frozenset(("owner", "thread_id", "epoch", "deadline", "selector_path", "host_digest",
                            "duel", "message", "server_prompt", "api", "snapshot", "reader", "host",
                            "hook_states", "hook_digests", "_bindings", "_borrowed_graph"))

    def __setattr__(self, name, value):
        if name in self._IMMUTABLE and name in vars(self):
            raise SnapshotOwnershipError("root metadata is immutable: " + name)
        object.__setattr__(self, name, value)

    def __init__(self, owner, selector_path, max_seconds):
        self.owner, self.thread_id = owner, threading.get_ident()
        self.epoch, self.deadline = owner.epoch, owner.clock() + max_seconds
        self.selector_path = selector_path
        self.closed = self.used = False
        self.branch_session = None
        self.hook_states = {}
        for name, hook in owner.hooks.items():
            hook.quiesce()
            self.hook_states[name] = hook.owned_copy(hook.capture())
        self.hook_digests = {name: owner.hooks[name].digest(value) for name, value in self.hook_states.items()}
        self.host = owner._capture_host()
        self.host_digest = _digest(self.host, owner.core.card_pool())
        self._borrowed_graph = _clone({name: self.host["sync"][name] for name in ("origin", "choices")}, owner.core.card_pool())
        saved = [self._borrowed_graph["origin"]]
        saved += [item for choice in self._borrowed_graph["choices"] for item in (choice.saved, choice.origin)]
        handles = [item[0] for item in saved if item is not None]
        if any(not handle for handle in handles) or len(set(handles)) != len(handles):
            raise SnapshotOwnershipError("follower snapshot ownership graph has duplicate/invalid handles")
        self.reader = _ReaderState(owner.core)
        self.api = snapshot_api(owner.core)
        self.duel = owner.follower.local.pduel
        self.message = owner.follower.parked
        self.server_prompt = bytes(owner.follower.packets[owner.follower.cursor - 1])
        self._bindings = (owner.follower.local, owner.follower.api, owner.follower.core)
        self.snapshot = _Snapshot(self.api, self.duel)

    def _require_owner_thread(self):
        if threading.get_ident() != self.thread_id:
            raise SnapshotOwnershipError("root native access belongs to its creating thread")

    def _check(self):
        self._require_owner_thread()
        self.owner._ensure_open()
        if self.closed:
            raise SnapshotOwnershipError("root already released")
        if self.owner._root is not self:
            raise SnapshotOwnershipError("root is not registered with this owner")
        if self.owner.epoch != self.epoch:
            raise StaleRoot("newly received packets invalidated this root")
        if self.owner.clock() >= self.deadline:
            raise RootTimeout("root budget expired")

    def _restore(self):
        """Restoration is allowed after timeout or epoch change; never drop inbox."""
        self._require_owner_thread()
        if self.owner._root is not self:
            raise SnapshotOwnershipError("only the exact registered root may restore this follower")
        self.owner._quiesce()
        restored = False
        try:
            if _digest(self.host, self.owner.core.card_pool()) != self.host_digest:
                raise RestoreMismatch("saved host envelope was modified")
            self.snapshot.restore()
            self.owner.follower.local, self.owner.follower.api, self.owner.follower.core = self._bindings
            self.owner._restore_host(self.host)
            self.reader.restore(self.owner.audit)
            for name, hook in self.owner.hooks.items():
                # All hooks were quiesced together before native restoration.
                if hook.digest(self.hook_states[name]) != self.hook_digests[name]:
                    raise RestoreMismatch("saved state hook snapshot was modified: " + name)
                hook.restore(hook.owned_copy(self.hook_states[name]))
                if hook.digest(hook.capture()) != self.hook_digests[name]:
                    raise RestoreMismatch("state hook did not restore " + name)
            self.snapshot.verify()
            if _digest(self.owner._capture_host(), self.owner.core.card_pool()) != self.host_digest:
                raise RestoreMismatch("follower/driver/choice/ledger state did not restore")
            restored = True
        finally:
            if not restored:
                self.owner._poison("full local root restoration failed")
            elif self.owner._poison_reason:
                self.owner.follower.failure = "root transaction poisoned: " + self.owner._poison_reason

    @contextmanager
    def branch(self, materialize=None, *, max_seconds=None):
        """A revocable session from this root. ``materialize(session)`` writes a particle into the unanswered root
        prompt's own state before the session is ready: the prompt the core already asked and its menu stay as they
        are, so the seat's decision here is the server's whatever the particle guessed (plan 5.13)."""
        self._check()
        if self.branch_session is not None:
            raise RootBusy("particles must return to the blank root before another begins")
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
            raise RootError("branch budget must be finite and positive")
        self._restore()
        session = None
        try:
            session = BranchSession(self, max_seconds)
            self.branch_session = session
            session.prepare(materialize)
            yield session
        finally:
            try:
                if session is not None:
                    session.close()
            finally:
                if session is None or not session.active:
                    self.branch_session = None
                    if not self.closed:
                        self._restore()
                else:
                    self.owner._poison("branch still has active consumers; cleanup is deferred")

    def abort(self):
        """After a failed quiescence, the owner first stops consumers, then retries."""
        self._require_owner_thread()
        if self.closed:
            return
        if self.owner._root is not self:
            raise SnapshotOwnershipError("only the exact registered root may abort")
        self.owner._quiesce()
        if self.branch_session is not None:
            self.branch_session.close()
            self.branch_session = None
        self.close()

    def close(self):
        self._require_owner_thread()
        if self.closed:
            return
        if self.owner._root is not self:
            raise SnapshotOwnershipError("only the exact registered root may release native ownership")
        if self.branch_session is not None:
            raise RootBusy("close branch scopes before their root")
        self.owner._quiesce()
        try:
            self._restore()
        finally:
            # _restore has another quiescence boundary. If that one fails,
            # retain this snapshot AND lock until a later explicit safe abort.
            if self.owner._quiescent:
                self.closed = True
                self.snapshot.close()
                if self.owner._poison_reason:
                    # Even a broken host restore must retain the original
                    # ownership graph so retiring the follower frees each once.
                    borrowed = _clone(self._borrowed_graph, self.owner.core.card_pool())
                    self.owner.follower.origin = borrowed["origin"]
                    self.owner.follower.choices = borrowed["choices"]
                self.owner._root = None
                self.owner._remove_free_guards()
                try:
                    if not self.owner._poison_reason:
                        self.owner._drain()
                finally:
                    self.owner._native_lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class _OwnedHandle(int):
    def __new__(cls, value, owner):
        obj = int.__new__(cls, value)
        obj.owner, obj.released = owner, False
        return obj


class _GuardedFunction:
    def __init__(self, function, call, check):
        self.function, self.call, self.check = function, call, check

    def __call__(self, *args):
        return self.call(*args)

    @property
    def argtypes(self):
        return self.function.argtypes

    @argtypes.setter
    def argtypes(self, value):
        self.check(cleanup=True)
        self.function.argtypes = value

    @property
    def restype(self):
        return self.function.restype

    @restype.setter
    def restype(self, value):
        self.check(cleanup=True)
        self.function.restype = value


class _BranchLibrary:
    _ALLOWED = frozenset(("process", "get_message", "get_log_message", "set_responsei", "set_responseb",
                          "preload_script", "query_card", "query_field_card", "query_field_count", "query_field_info",
                          "query_effect_info", "query_duel_state", "query_local_entity_map",
                          "duel_hydrate_card", "duel_reorder_deck_uids", "duel_reorder_zone_uids",
                          "duel_set_future_seed",
                          "duel_snapshot", "duel_rollback", "duel_snapshot_free", "duel_arena_extent"))

    def __init__(self, session):
        self.session, self.functions = session, {}

    def __getattr__(self, name):
        if name not in self._ALLOWED:
            raise AttributeError("branch does not own native lifecycle/global readers: " + name)
        if name not in self.functions:
            function = getattr(self.session.root.owner.core._lib, name)
            self.functions[name] = _GuardedFunction(function, lambda *args: self.session._native(name, function, args), self.session._check)
        return self.functions[name]


class _BranchCore:
    def __init__(self, session):
        self.session = session
        self._lib = _BranchLibrary(session)
        self._card_pool = _copy_card_pool(session.root.owner.core.card_pool())

    def card_pool(self):
        self.session._check()
        return self._card_pool

    def __getattr__(self, name):
        self.session._check()
        if name in ("lib_path", "db_path", "script_dirs", "_script_cache", "_card_cache", "_known_missing",
                    "missing_codes", "log", "_resolve_script", "_query_card_data"):
            return getattr(self.session.root.owner.core, name)
        if name in ("process", "get_message", "preload_script", "query_field_count", "query_field_info", "set_responsei"):
            return getattr(self._lib, name)
        raise AttributeError("branch core operation is not exposed: " + name)

    def set_responseb(self, duel, data):
        if len(data) > 256:
            raise RootError("native response exceeds fixed buffer")
        self._lib.set_responseb(duel, bytes(data).ljust(256, b"\0"))

    def query_field_card(self, duel, player, location, flags, buf):
        return self._lib.query_field_card(duel, player, location, flags, buf, 0)

    def preload_script(self, duel, name):
        return self._lib.preload_script(duel, str(name).encode())


class _BranchDriver(DuelDriver):
    _PYSTATE_EXEMPT = DuelDriver._PYSTATE_EXEMPT + ("branch_session",)

    def build(self):
        raise SnapshotOwnershipError("branch borrows a live duel; it must not build one")

    def close(self):
        raise SnapshotOwnershipError("branch must not destroy the real duel")

    def _observe(self, message):
        self.branch_session._check()
        super()._observe(message)  # deliberately NOT _FollowingDuel._observe

    def _respond(self, data):
        self.branch_session._check()
        super()._respond(data)


@dataclass(frozen=True)
class BranchAction:
    owner: Any
    node: int
    index: int
    description: str


class BranchSession:
    """Revocable borrowed driver; no real receive/response/packet comparison path."""

    def __init__(self, root, max_seconds):
        self.root, self.active, self.ready, self.node = root, True, False, 0
        # Host-only lifetime stamp: native rollback may reuse a future cardid.
        # This stamp is intentionally not part of checkpoint-restored state.
        self.entity_generation = 0
        self.deadline = min(root.deadline, root.owner.clock() + max_seconds) if max_seconds is not None else root.deadline
        self.native_handles, self.stack, self._blank = [], [], None
        self.models = {name: hook.owned_copy(root.hook_states[name]) for name, hook in root.owner.hooks.items()}
        self.driver = _BranchDriver(root.owner.follower.local.config, _BranchCore(self))
        self.driver.branch_session = self
        self.driver.pduel = root.duel
        self.pool_digest = _pool_digest(self.driver.core._card_pool)
        self.message, self.path = root.message, root.selector_path
        self._restore_driver(root.host["driver"])

    def _restore_driver(self, state):
        if isinstance(state, FrozenPyState):
            clone = state.thaw(self.driver.core._card_pool)
        else:
            clone = _clone(state, self.root.owner.core.card_pool(), self.driver.core._card_pool)
        self.driver.restore_pystate(clone, consume=True)
        self.driver._answering_msg, self.driver._answering_payload = self.message.msg, self.message.payload

    def _check(self, cleanup=False):
        self.root._require_owner_thread()
        if not self.active:
            raise SnapshotOwnershipError("branch capability was revoked")
        if not cleanup:
            self.root._check()
            if self.root.owner.clock() >= self.deadline:
                raise RootTimeout("branch budget expired")

    def _native(self, name, function, args):
        self._check(cleanup=name in ("duel_snapshot_free", "duel_rollback"))
        if name == "duel_snapshot_free":
            handle = args[0]
            if not isinstance(handle, _OwnedHandle) or handle.owner is not self or handle.released:
                raise SnapshotOwnershipError("cannot free a borrowed, foreign or released native snapshot")
            handle.released = True
            return function(int(handle))
        if not args or args[0] != self.root.duel:
            raise SnapshotOwnershipError("branch native call names a different duel")
        if name in ("set_responsei", "set_responseb", "duel_set_future_seed") and not self.ready:
            raise RootError("responses and future RNG require a ready pending branch prompt")
        if name in ("duel_hydrate_card", "duel_reorder_deck_uids", "duel_reorder_zone_uids") and self.ready:
            raise RootError("a particle is written at the unanswered root, before the branch is ready")
        if name == "duel_reorder_zone_uids" and (len(args) < 3
                or args[1] != 1 - self.root.owner.follower.viewer or args[2] != C.LOCATION_HAND):
            raise RootError("the branch zone reorder capability is only its public opponent hand")
        if name == "duel_rollback":
            handle = args[1]
            if not isinstance(handle, _OwnedHandle) or handle.owner is not self or handle.released:
                raise SnapshotOwnershipError("cannot roll back an unowned native snapshot")
            result = function(args[0], int(handle))
            if result == 0:
                self.entity_generation += 1
            return result
        value = function(*args)
        if name == "duel_snapshot" and value:
            value = _OwnedHandle(value, self)
            self.native_handles.append(value)
        return value

    def prepare(self, materialize):
        # The blank root, kept for ``rewrite``: every particle of the session is written from exactly this state.
        lib = self.driver.core._lib
        snap = lib.duel_snapshot(self.driver.pduel)
        if not snap:
            raise RootError("the blank root could not be kept for the session's particles")
        self._blank = (snap, self.driver.save_pystate())
        if materialize is not None:
            materialize(self)
        self.driver._answering_msg, self.driver._answering_payload = self.message.msg, self.message.payload
        self.ready = True

    def rewrite(self, materialize):
        """Write another particle into the same unanswered root: back to the blank root the session began at (the
        native state, the driver's state and the root prompt), then ``materialize`` as ``prepare`` does. Every
        particle is still written at the unanswered root; what the session kept of the earlier ones (their own
        snapshots) stays theirs."""
        self._check()
        if self.stack:
            raise RootError("close the branch checkpoints before writing another particle")
        snap, state = self._blank
        if self.driver.core._lib.duel_rollback(self.driver.pduel, snap) != 0:
            raise RootError("the session could not return to its blank root")
        self.driver.restore_pystate(state)
        self.message, self.path = self.root.message, self.root.selector_path
        self.node += 1
        self.ready = False
        materialize(self)
        self.driver._answering_msg, self.driver._answering_payload = self.message.msg, self.message.payload
        self.ready = True

    def hydrate(self, player, location, sequence, code):
        """Give a root placeholder the particle's identity, in place: the same card object with its status, slot
        history and external effects, so the root prompt's stored menu still names it."""
        self._check()
        if self.ready:
            raise RootError("a particle is written at the unanswered root, before the branch is ready")
        return hydrate(self.driver.core, self.driver.pduel, player, location, sequence, code)

    def reorder_deck(self, player, uids):
        """Exact local-object deck order, at the unanswered root before the branch is ready."""
        self._check()
        reorder_deck(self.driver.core, self.driver.pduel, player, uids)

    def reorder_hand(self, player, uids):
        """Whole opponent-hand objects, at the unanswered root only; no pending-vector rewrite."""
        self._check()
        if self.ready or player != 1 - self.root.owner.follower.viewer:
            raise RootError("public hand order belongs to the opponent at the unanswered root")
        reorder_hand(self.driver.core, self.driver.pduel, player, uids)

    def _selector(self):
        self._check()
        if not self.ready:
            raise RootError("branch prompt is not ready yet")
        self.driver._ctx.our_player = _prompt_player(self.message)
        result = self.driver.parse_prompt(self.message.msg, self.message.payload)
        if result.auto_response is not None:
            if self.path:
                raise RootError("automatic prompt cannot have a selector prefix")
            return result, None
        selector = result.selector
        for choice in self.path:
            if selector.choose(choice) is not None:
                raise RootError("selector prefix already completed its response")
        return result, selector

    def menu(self):
        result, selector = self._selector()
        if result.auto_response is not None:
            return (), bytes(result.auto_response)
        return tuple(BranchAction(self, self.node, i, action.describe())
                     for i, action in enumerate(selector.options())), None

    def choose(self, action):
        self._check()
        if not isinstance(action, BranchAction) or action.owner is not self or action.node != self.node:
            raise SnapshotOwnershipError("action belongs to a stale or foreign branch node")
        _result, selector = self._selector()
        data = selector.choose(action.index)
        self.node += 1
        if data is None:
            self.path += (action.index,)
            return self.message
        return self.respond(bytes(data))

    def respond(self, data):
        self._check()
        if not self.ready or self.message is None:
            raise RootError("branch has no ready pending prompt")
        self.driver._answering_msg, self.driver._answering_payload = self.message.msg, self.message.payload
        self.driver._respond(bytes(data))
        self.path, self.node = (), self.node + 1
        while not self.driver.finished:
            _emitted, prompt = self._batch()
            if prompt is not None:
                self.message = prompt
                return prompt
        self.message = None
        return None

    def _batch(self, project=False):
        self._check()
        raw = self.driver.core.process(self.driver.pduel)
        self._check()
        self.driver.steps += 1
        emitted, prompt = [], None
        if raw & PROCESSOR_BUFFER_LEN:
            size = self.driver.core.get_message(self.driver.pduel, self.driver._msgbuf)
            for message in split_messages(self.driver._msgbuf.raw[:size]):
                self.driver._observe(message)
                if project:
                    viewer = self.root.owner.follower.viewer
                    if message.msg in RESPONSE_REQUIRED and _prompt_player(message) != viewer:
                        emitted.append(bytes([3]))
                    delivery = host_view.deliver(message.msg, message.payload)
                    if viewer in delivery.payloads:
                        emitted.extend([delivery.payloads[viewer]] * delivery.counts.get(viewer, 1))
                if message.msg in RESPONSE_REQUIRED:
                    prompt = message
        if self.driver.winner is not None or raw & PROCESSOR_FLAG == PROCESSOR_END:
            self.driver.finished = True
        return emitted, prompt

    def checkpoint(self):
        self._check()
        if not self.ready:
            raise RootError("cannot checkpoint an unmaterialized prompt")
        checkpoint = BranchCheckpoint(self)
        self.stack.append(checkpoint)
        return checkpoint

    def close(self):
        if not self.active:
            return
        self.root.owner._quiesce()
        try:
            pool_changed = _pool_digest(self.driver.core._card_pool) != self.pool_digest
            while self.stack:
                self.stack[-1].close()
            if pool_changed:
                raise RootError("branch changed its read-only card-data pool")
        finally:
            # Each checkpoint.close quiesces again. A later failure revokes
            # the earlier success: no still-borrowed resource may be freed.
            if self.root.owner._quiescent:
                for checkpoint in reversed(self.stack):
                    checkpoint.snapshot.close()
                    checkpoint.closed = True
                self.stack.clear()
                for handle in self.native_handles:
                    if not handle.released:
                        handle.released = True
                        self.root.api.duel_snapshot_free(int(handle))
                self.active = False
                self.driver.pduel = None


class BranchCheckpoint:
    def __init__(self, branch):
        self.branch, self.closed = branch, False
        self.driver = _clone(_driver_state(branch.driver), branch.driver.core._card_pool)
        self.driver_digest = _digest(self.driver, branch.driver.core._card_pool)
        self.message, self.path = branch.message, branch.path
        self.models = {name: hook.owned_copy(branch.models[name]) for name, hook in branch.root.owner.hooks.items()}
        self.reader = _ReaderState(branch.root.owner.core)
        self.snapshot = _Snapshot(branch.root.api, branch.root.duel)

    def close(self):
        if self.closed:
            return
        branch = self.branch
        branch._check(cleanup=True)
        if not branch.stack or branch.stack[-1] is not self:
            raise SnapshotOwnershipError("branch checkpoints must be restored in LIFO order")
        branch.root.owner._quiesce()
        restored = False
        try:
            self.snapshot.restore()
            branch.entity_generation += 1
            branch.message, branch.path = self.message, self.path
            branch.driver.restore_pystate(_clone(self.driver, branch.driver.core._card_pool), consume=True)
            branch.models = {name: hook.owned_copy(self.models[name]) for name, hook in branch.root.owner.hooks.items()}
            self.reader.restore(branch.root.owner.audit)
            self.snapshot.verify()
            if _digest(_driver_state(branch.driver), branch.driver.core._card_pool) != self.driver_digest:
                raise RestoreMismatch("branch driver did not restore with its native checkpoint")
            for name, hook in branch.root.owner.hooks.items():
                if hook.digest(branch.models[name]) != hook.digest(self.models[name]):
                    raise RestoreMismatch("branch model state did not restore: " + name)
            branch.node += 1
            restored = True
        finally:
            self.closed = True
            branch.stack.pop()
            self.snapshot.close()
            if not restored:
                branch.root.owner._poison("branch checkpoint restoration failed")

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
