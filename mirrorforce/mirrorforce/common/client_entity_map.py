"""Internal physical entities in an owned blank-client VM, never model inputs.

The public entry accepts only a registered controller, its current frozen
root, or its live branch. It never accepts a caller-supplied core/duel and
never uses a teacher/omniscient state API. The C ABI contains no card codes,
effects, RNG or Lua state. These local IDs are not knowledge of server cards.

This is cooperative in-process ownership, not a Python/ctypes sandbox. Native
callers remain responsible for supplying a valid buffer and serialized duel.
"""

from __future__ import annotations

import ctypes
import struct
import weakref
from dataclasses import dataclass, field

from .client_root import (
    BranchSession, ClientRootController, RootBusy, RootEnvelope,
    SnapshotOwnershipError, _FOLLOWER_OWNERS,
    _advance_boundary,
)
from .client_shadow import BlankClientSync

INFORMATION_SET_SEARCH = True
SCHEMA = "query_local_entity_map/v1"
ABI_VERSION = 1
_HEADER = struct.Struct("<4sHHII")
_ROW = struct.Struct("<QQBBBBIII")
_NO_SLOT = 0xffffffff
_MAX_BYTES = 4 * 1024 * 1024
_ZONES = frozenset((1, 2, 4, 8, 16, 32, 64))
_NAMESPACES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class EntityMapError(RuntimeError):
    pass


class EntityMapUnsupported(EntityMapError):
    pass


@dataclass(frozen=True)
class LocalEntity:
    uid: int
    overlay_parent: int
    owner: int
    controller: int
    location: int
    placeholder: int
    sequence: int
    overlay_ordinal: int


@dataclass(frozen=True)
class LocalEntityRef:
    """Opaque-context reference, not a public feature or a native write token."""

    _scope: object = field(repr=False)
    generation: int
    uid: int


@dataclass(frozen=True)
class LocalEntityMap:
    """An immutable capture. Numeric UID matching alone grants no authority.

    References are conservative: all are invalidated across a branch rollback
    (including surviving cards), and never transfer across roots/branches.
    Numeric UIDs still permit future, explicitly certified origin/root joins.
    """

    entities: tuple[LocalEntity, ...]
    _scope: object = field(repr=False)
    generation: int
    schema: str = SCHEMA

    def at(self, controller: int, location: int, sequence: int) -> LocalEntity:
        if location not in _ZONES:
            raise EntityMapError("location zero/overlay has no ordinary zone coordinate")
        matches = [row for row in self.entities if
                   (row.controller, row.location, row.sequence) == (controller, location, sequence)]
        if len(matches) != 1:
            raise EntityMapError("zone coordinate does not name exactly one local entity")
        return matches[0]

    def reference(self, uid: int) -> LocalEntityRef:
        if not any(row.uid == uid for row in self.entities):
            raise EntityMapError("UID is absent from this capture")
        return LocalEntityRef(self._scope, self.generation, uid)

    def resolve(self, reference: LocalEntityRef) -> LocalEntity:
        if (type(reference) is not LocalEntityRef or reference._scope is not self._scope
                or reference.generation != self.generation):
            raise SnapshotOwnershipError("entity reference belongs to a foreign or rolled-back scope")
        for row in self.entities:
            if row.uid == reference.uid:
                return row
        raise EntityMapError("referenced entity no longer exists in this capture")


def _parse(raw: bytes) -> tuple[LocalEntity, ...]:
    if len(raw) < _HEADER.size:
        raise EntityMapError("truncated local-entity header")
    magic, version, width, count, size = _HEADER.unpack_from(raw)
    if magic != b"LEM1" or version != ABI_VERSION or width != _ROW.size:
        raise EntityMapUnsupported("unknown local-entity ABI")
    if size != len(raw) or size != _HEADER.size + count * width or size > _MAX_BYTES:
        raise EntityMapError("local-entity count/length differs")
    entities = []
    previous, occupied = 0, set()
    for offset in range(_HEADER.size, size, width):
        uid, parent, owner, controller, location, placeholder, sequence, ordinal, reserved = _ROW.unpack_from(raw, offset)
        if uid <= previous or reserved or owner not in (0, 1, 2) or placeholder not in (0, 1, 2, 3, 4):
            raise EntityMapError("invalid local-entity identity/order/reserved field")
        if location == 0:
            # CreateToken sets its intended controller before placement/yield.
            valid = controller in (0, 1, 2) and sequence == _NO_SLOT and not parent and ordinal == _NO_SLOT
        elif location == 128:
            valid = controller == 2 and parent not in (0, uid) and ordinal == sequence and sequence <= 255
        else:
            valid = location in _ZONES and controller in (0, 1) and sequence <= 255 and not parent and ordinal == _NO_SLOT
            coordinate = (controller, location, sequence)
            if coordinate in occupied:
                valid = False
            occupied.add(coordinate)
        if not valid:
            raise EntityMapError("invalid local-entity coordinate/overlay metadata")
        previous = uid
        entities.append(LocalEntity(uid, parent, owner, controller, location, placeholder, sequence, ordinal))
    ids = {row.uid for row in entities}
    attached = set()
    for row in entities:
        if row.overlay_parent:
            key = (row.overlay_parent, row.overlay_ordinal)
            if row.overlay_parent not in ids or key in attached:
                raise EntityMapError("missing parent or duplicate overlay ordinal")
            attached.add(key)
    return tuple(entities)


def _read(lib, duel):
    try:
        query = lib.query_local_entity_map
    except AttributeError as exc:
        raise EntityMapUnsupported("core lacks query_local_entity_map/v1; no label fallback") from exc
    query.argtypes = [ctypes.c_ssize_t, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int32]
    query.restype = ctypes.c_int32
    size = query(duel, ABI_VERSION, None, 0)
    if size == -2:
        raise EntityMapUnsupported("core rejected local-entity ABI v1")
    if not _HEADER.size <= size <= _MAX_BYTES or (size - _HEADER.size) % _ROW.size:
        raise EntityMapError("invalid local-entity size probe: " + str(size))
    buf = ctypes.create_string_buffer(size)
    written = query(duel, ABI_VERSION, buf, size)
    if written != size:
        raise EntityMapError("local-entity read changed size or failed: " + str(written))
    return _parse(buf.raw)


def _validated_owner(source):
    if type(source) is ClientRootController:
        owner = source
    elif type(source) is RootEnvelope:
        source._check()
        owner = source.owner
        if source.branch_session is not None:
            raise RootBusy("root entities are not active while its branch borrows the arena")
    elif type(source) is BranchSession:
        source._check()
        owner = source.root.owner
        if source.root.branch_session is not source:
            raise SnapshotOwnershipError("branch is not the exact registered active session")
    else:
        raise SnapshotOwnershipError("entity map requires an owned controller/root/branch, never core/duel")
    owner._ensure_open()
    follower = owner.follower
    if (type(owner) is not ClientRootController or _FOLLOWER_OWNERS.get(follower) is not owner
            or not isinstance(follower, BlankClientSync) or follower.core is not owner.core
            or follower.local is None or not follower.local.pduel or follower.local.core is not owner.core
            or owner._entity_binding != (follower.local, follower.local.pduel)
            or getattr(follower.local, "follower", None) is not follower):
        raise SnapshotOwnershipError("blank follower/duel does not belong to this registered controller")
    root = owner._root
    if root is not None and (root.duel != follower.local.pduel or root._bindings !=
                            (follower.local, follower.api, follower.core)):
        raise SnapshotOwnershipError("frozen native/driver binding was replaced")
    return owner


def capture_entities(source) -> LocalEntityMap:
    """Read only this owner's currently active local entities; no raw handles.

    Controller calls during a root delegate to that root, never the borrowed
    branch. Call with the branch explicitly to read its hypothetical entities.
    A captured map is audit data, not a continuing write capability.
    """
    owner = _validated_owner(source)
    if type(source) is ClientRootController and owner._root is not None:
        return capture_entities(owner._root)
    if type(source) is ClientRootController:
        if owner._busy:
            raise RootBusy("real advance/input operation has not released the owner")
        with owner._native_operation():
            owner._quiesce()
            _validated_owner(source)
            rows = _read(owner.core._lib, owner.follower.local.pduel)
            generation = owner.epoch
    else:
        owner._quiesce()
        _validated_owner(source)
        source._check()
        branch = type(source) is BranchSession
        lib = source.driver.core._lib if branch else owner.core._lib
        rows = _read(lib, source.root.duel if branch else source.duel)
        source._check()
        generation = source.entity_generation if branch else source.epoch
    scope = _NAMESPACES.setdefault(source, object())
    return LocalEntityMap(rows, scope, generation)


def _capture_advance_entities(follower, token):
    """Receipt-only internal C boundary, under the original advance's lock.

    Never callable during process/observe/preload, never a permission to
    bypass the native query's own idle check, and not a public raw-duel API.
    """
    if token is None:
        raise SnapshotOwnershipError("receipt entity query requires its explicit advance token")
    scope = _advance_boundary(follower, token)
    if scope.phase != "idle":
        raise SnapshotOwnershipError("entity receipt queried inside a native/observer operation")
    scope.owner._quiesce()
    owner = _validated_owner(scope.owner)
    rows = _read(owner.core._lib, follower.local.pduel)
    _advance_boundary(follower, token)
    return rows
