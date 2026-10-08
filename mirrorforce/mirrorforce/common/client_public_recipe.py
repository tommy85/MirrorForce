"""Opt-in recipe inventory for the owned blank follower, not server knowledge.

Opening cards stay blank. Only the order-free public construction is known;
native trial identities consume its copies and rollback recomputes the budget.
QUERY_CODE reads printed data.code (not effect-driven get_code/alias). A native
recreation without matching owned hydration provenance is refused, not counted
as if its new displayed identity were an original card from this construction.
"""
from __future__ import annotations

from collections import Counter
import ctypes
from dataclasses import dataclass
import struct

from ..immutable import ImmutableRecord
from ..netduel import constants as C
from .client_sync import SyncError

INFORMATION_SET_SEARCH = True
LAW = "public-mirror-recipe;blank-base;owned-uid-copy-constraints/v1"
_ZONES = frozenset((1, 2, 4, 8, 16, 32, 64))
_TOKEN = 0x4000


class PublicRecipeError(SyncError):
    pass


@dataclass(frozen=True)
class PublicRecipe(ImmutableRecord):
    main: tuple[int, ...]
    extra: tuple[int, ...]
    sha256: str

    @classmethod
    def checked(cls, declaration):
        from ..netduel.agent_public_recipe import checked
        value = checked(declaration)
        if set(value['main']) & set(value['extra']) or any(
                copies > 3 for copies in Counter(value['main'] + value['extra']).values()):
            raise ValueError("public follower recipe needs disjoint home zones and at most three copies")
        return cls(tuple(value['main']), tuple(value['extra']), value['sha256'])

    def remaining(self, rows, *, opponent):
        """rows=(uid, original owner, printed code, provenance code, excluded birth).

        Each UID is counted once, wherever its controller/overlay position is.
        Unassigned blanks consume no *named* copy. Opening-certified non-recipe
        births (tokens/created copies) consume none. Provenance refers only to
        this restored local hypothesis journal, never actual server identities.
        """
        from .client_shadow import BLANK_CODES
        pool = Counter(self.main + self.extra)
        seen = set()
        for uid, owner, code, provenance, excluded_birth in rows:
            if uid in seen:
                raise PublicRecipeError("duplicate physical UID in public recipe inventory")
            seen.add(uid)
            if owner != opponent or code in BLANK_CODES or excluded_birth:
                continue
            if code != provenance or code not in pool:
                raise PublicRecipeError("original recipe identity is not proved for native UID (alias/recreation)")
            pool[code] -= 1
            if pool[code] < 0:
                raise PublicRecipeError("native hypothesis exceeds public recipe copies")
        return +pool

    def permits(self, rows, *, opponent, target_uid, code, location):
        from .client_shadow import BLANK_CODES
        if code in BLANK_CODES:
            return True
        target = [row for row in rows if row[0] == target_uid]
        if len(target) != 1:
            raise PublicRecipeError("recipe assignment has no unique physical target")
        if target[0][1] != opponent:
            return True  # original ownership, not current controller
        if target[0][4]:
            # A created object is not an original deck copy. Confirm only its
            # existing printed identity; do not hypothesize an arbitrary card.
            return target[0][2] == code
        self.remaining(rows, opponent=opponent)  # reject unexplained identities even when target is replaced
        if location in (C.LOCATION_HAND, C.LOCATION_DECK) and code not in self.main \
                or location == C.LOCATION_EXTRA and code not in self.extra:
            return False
        return self.remaining(tuple(row for row in rows if row[0] != target_uid),
                              opponent=opponent).get(code, 0) > 0


def _overlay_codes(follower, host):
    """Read only printed overlay codes in the owned local core; no q_cache fields."""
    flag = C.QUERY_OVERLAY_CARD
    buf = ctypes.create_string_buffer(4096)
    size = follower.core._lib.query_card(follower.local.pduel, host.controller, host.location,
                                        host.sequence, flag, buf, 0)
    if size < 12 or size > len(buf):
        raise PublicRecipeError("invalid local overlay recipe query")
    length, flags, count = struct.unpack_from('<III', buf)
    if length != size or flags != flag or size != 12 + 4 * count:
        raise PublicRecipeError("invalid local overlay recipe query shape")
    return struct.unpack_from('<' + 'I' * count, buf, 12)


def native_rows(follower):
    """Current restored native inventory, at an owned complete-C boundary only."""
    from .client_origin_receipt import entities
    from .client_shadow import BLANK_CODES, card_code
    rows = entities(follower)
    by_uid = {row.uid: row for row in rows}
    ledger = follower.receipt_state
    events = ledger.replay_events
    if not events or events[0].kind != 'opening':
        raise PublicRecipeError("recipe inventory lacks its certified native opening")
    opening = events[0].after.entities
    originals = {row.uid: row for row in opening if row.owner == 1 - follower.viewer
                 and row.placeholder in (1, 2)}
    recipe = follower.public_opponent_recipe
    if (sum(row.placeholder == 1 for row in originals.values()),
            sum(row.placeholder == 2 for row in originals.values())) != (len(recipe.main), len(recipe.extra)):
        raise PublicRecipeError("native opening UID counts differ from the complete public recipe")
    if set(originals) - set(by_uid):
        raise PublicRecipeError("an original recipe physical object disappeared without attribution")
    if any(by_uid[uid].owner != original.owner for uid, original in originals.items()):
        raise PublicRecipeError("native original ownership changed without recipe attribution")
    mutations = [m for receipt in ledger.accepted for m in receipt.mutations] + list(ledger.pending)
    provenance = {}
    for mutation in mutations:
        if mutation.uid and type(mutation.code) is int and mutation.kind in (
                'hydrate', 'hidden_target_hypothesis', 'category_proxy', 'public_align'):
            provenance[mutation.uid] = mutation.code
    cards = follower.core.card_pool().cards
    overlays, result = {}, []
    opponent = 1 - follower.viewer
    for row in rows:
        if row.owner != opponent:
            # Retain original owner for target lookup, without reading its private code.
            result.append((row.uid, row.owner, 0, 0, False))
            continue
        excluded_birth = row.uid not in originals
        if row.location == 0 and excluded_birth:
            # CreateToken/CreateCard can yield before placement. Its new UID
            # proves it was not one of the opening construction's card objects;
            # no inaccessible printed identity is read or guessed here.
            result.append((row.uid, row.owner, 0, 0, True))
            continue
        if row.location in _ZONES and row.controller in (0, 1):
            code = card_code(follower.core, follower.local.pduel, row.controller, row.location, row.sequence)
        elif row.location == C.LOCATION_OVERLAY:
            host = by_uid.get(row.overlay_parent)
            if host is None or host.location not in _ZONES or host.controller not in (0, 1):
                raise PublicRecipeError("overlay recipe ownership lacks a physical host")
            if host.uid not in overlays:
                overlays[host.uid] = _overlay_codes(follower, host)
            codes = overlays[host.uid]
            if row.overlay_ordinal >= len(codes):
                raise PublicRecipeError("overlay recipe ordinal differs from native entities")
            code = codes[row.overlay_ordinal]
        else:
            raise PublicRecipeError("unplaced native opponent identity has no proved recipe attribution")
        data = cards.get(code)
        if code not in BLANK_CODES and data is None:
            raise PublicRecipeError("recipe identity lacks pinned printed card data")
        if not excluded_birth and data is not None and data.type & _TOKEN:
            raise PublicRecipeError("an original recipe object changed to a token without attribution")
        result.append((row.uid, row.owner, code, provenance.get(row.uid), excluded_birth))
    return tuple(result), rows


def permits_native(follower, controller, location, sequence, code):
    recipe = getattr(follower, 'public_opponent_recipe', None)
    if recipe is None:
        return True
    rows, entities = native_rows(follower)
    matches = [row for row in entities if (row.controller, row.location, row.sequence) ==
               (controller, location, sequence)]
    if len(matches) != 1:
        raise PublicRecipeError("recipe assignment target is absent from the restored engine")
    return recipe.permits(rows, opponent=1 - follower.viewer, target_uid=matches[0].uid,
                          code=code, location=location)


def require_native(follower, controller, location, sequence, code):
    if not permits_native(follower, controller, location, sequence, code):
        raise PublicRecipeError("native assignment has no remaining public recipe copy in its home zone")
