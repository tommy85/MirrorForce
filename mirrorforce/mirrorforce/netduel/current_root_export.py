"""Mask only an owned, already realized current branch into the root-view ABI.

Current chain/lingering effects come from the live observer's common-public
incremental tracker, not a private effect query or reconstructed past. Native
card queries here inspect only this client's hypothetical engine. Their cache
effects must be rolled back by the particle owner before continuing a line.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json

from . import constants as C
from .current_root_protocol import PUBLIC_SEED_SCHEMA, PUBLIC_SEED_RPC_SCHEMA
from .current_root_view import CurrentRootView, SCHEMA, ZONES, FIELDS

INFORMATION_SET_SEARCH = True
_PUBLIC_FIELDS = {"chain", "chain_context", "card_status", "public_effects", "equips", "field_origins"}


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _common_public_anchors(seed):
    def place(value, *, field=False, nullable=False):
        if value is None and nullable:
            return
        if not isinstance(value, (list, tuple)) or len(value) != 3 or any(type(v) is not int for v in value) \
                or value[0] not in (0, 1) or value[1] not in (FIELDS if field else (*FIELDS, C.LOCATION_GRAVE, C.LOCATION_REMOVED)) \
                or not 0 <= value[2] < FIELDS.get(value[1], 256):
            raise ValueError("current common-public facts contain a private or invalid anchor")
    for row in seed["card_status"]:
        place(row["place"], field=True)
        place(row["source"], nullable=True)
    for pair in seed["equips"]:
        for row in pair:
            place(row, field=True)
    for row in seed["field_origins"]:
        place(row[:3], field=True)
    for link in seed["chain"]:
        # Origin is the publicly broadcast CHAINING coordinate, not a private
        # current anchor. Current source/targets/moved identities are stricter.
        place(link["source"], nullable=True)
        for target in link["targets"]:
            place(target, field=True)
        for moved in link["moved"]:
            place(moved["place"])
            place(moved["destination"])


def check_seed_reply(reply, *, session, obs_sha256, source_viewer):
    if not isinstance(reply, dict) or set(reply) != {"schema", "session", "obs_sha256", "seed", "seed_sha256"} \
            or reply["schema"] != PUBLIC_SEED_RPC_SCHEMA or reply["session"] != session \
            or reply["obs_sha256"] != obs_sha256 or _sha(reply["seed"]) != reply["seed_sha256"]:
        raise ValueError("current public seed differs from the owned pending RPC binding")
    seed = reply["seed"]
    if not isinstance(seed, dict) or set(seed) != {"schema", "source_viewer", "turn", "turn_player", "phase", "public_seed"} \
            or seed["schema"] != PUBLIC_SEED_SCHEMA or type(seed["source_viewer"]) is not int \
            or seed["source_viewer"] not in (0, 1) or seed["source_viewer"] != source_viewer \
            or type(seed["turn"]) is not int or not 1 <= seed["turn"] < 2**31 \
            or type(seed["turn_player"]) is not int or seed["turn_player"] not in (0, 1) \
            or type(seed["phase"]) is not int or seed["phase"] not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) \
            or not isinstance(seed["public_seed"], dict) or set(seed["public_seed"]) != _PUBLIC_FIELDS:
        raise ValueError("current public seed contains undeclared/private fields or another viewer")
    _common_public_anchors(seed["public_seed"])
    return json.loads(json.dumps(seed, allow_nan=False))


def build_current_view(branch, *, public_recipe, public_root_seed, root_id, root_hash, hypothesis_hash, viewer):
    from ..common.client_root import BranchSession
    from ..common.client_entity_map import capture_entities
    from ..worldmodel.state import capture
    if type(branch) is not BranchSession:
        raise ValueError("current root export requires a live owned particle branch")
    branch._check()
    source_viewer = branch.root.owner.follower.viewer
    if viewer != 1 - source_viewer or public_root_seed["source_viewer"] != source_viewer \
            or root_hash != branch.root.snapshot.digest:
        raise ValueError("current root export changed the root or observer direction")
    entities = capture_entities(branch).entities
    raw = capture(branch.driver)
    if (raw.turn, raw.turn_player, raw.phase) != tuple(public_root_seed[k] for k in ("turn", "turn_player", "phase")):
        raise ValueError("current hypothetical engine differs from the live public clock")
    return _view_from_current(raw, entities, public_recipe, public_root_seed["public_seed"],
                              root_id=root_id, root_hash=root_hash, hypothesis_hash=hypothesis_hash, viewer=viewer,
                              disclosure=branch.driver.disclosure)


def _view_from_current(raw, entities, recipe, public_seed, *, root_id, root_hash, hypothesis_hash, viewer, disclosure):
    """Pure encoder split out for negative two-hidden-world tests; not a public engine ownership entry."""
    from ..worldmodel.state import identity_visible, public_view
    seed = copy.deepcopy(public_seed)
    if set(seed) != _PUBLIC_FIELDS:
        raise ValueError("only the common-public current seed may cross observer seats")
    _common_public_anchors(seed)
    # Reuse the audience-filtered disclosure ledger, including identities whose
    # exact position was lost in a shuffle. public_view canonicalizes hand slots,
    # so consume only its unpositioned multiset here; retain native/wire slots
    # for the client and use resolve() for genuinely proved exact anchors.
    revealed = disclosure.resolve(raw, viewer)
    unpositioned = Counter(public_view(raw, viewer, disclosure).revealed_unpositioned)
    if any(p != 1 - viewer or loc not in (C.LOCATION_HAND, C.LOCATION_DECK, C.LOCATION_EXTRA, *FIELDS)
           for p, loc, code in unpositioned):
        raise ValueError("current-root observer cannot encode this unpositioned zone")
    seed["positioned"] = [list(row) for row in sorted(revealed)
                          if row[0] != viewer and row[1] not in (C.LOCATION_DECK, C.LOCATION_EXTRA)]
    seed["reveals"] = [list(row) for row in sorted(revealed)
                       if row[1] in (C.LOCATION_DECK, C.LOCATION_EXTRA)]
    seed["unpositioned"] = [[loc, code, count] for (p, loc, code), count in sorted(unpositioned.items())]
    seed["hand_group"] = [c.sequence for c in raw.cards
                          if c.controller != viewer and c.location == C.LOCATION_HAND
                          and not identity_visible(c, viewer, revealed)]
    # the public_view gives zone-level lower bounds, not a hidden permutation.
    # Keep them among the currently anonymous field cards at this root. This
    # immutable constraint is not installed as new native policy memory; the
    # next real root is rebuilt from the follower's updated disclosure ledger.
    # Never use raw codes to choose which unknown slot holds a known name.
    field_locations = sorted({loc for _, loc, _ in unpositioned if loc in FIELDS})
    seed["field_groups"] = [[loc, *sorted(c.sequence for c in raw.cards
        if c.controller != viewer and c.location == loc and not identity_visible(c, viewer, revealed))]
        for loc in field_locations]
    entity_by_place = {(e.controller, e.location, e.sequence): e for e in entities if e.location in ZONES}
    materials = {(e.overlay_parent, e.overlay_ordinal): e for e in entities if e.overlay_parent}
    equips = {tuple(pair[0]): tuple(pair[1]) for pair in seed["equips"]}
    if len(equips) != len(seed["equips"]):
        raise ValueError("duplicate common-public equip source")
    cards, material_owners = [[[] for _ in ZONES] for _ in range(2)], []
    for player in (0, 1):
        for z, location in enumerate(ZONES):
            zone = sorted((c for c in raw.cards if c.controller == player and c.location == location), key=lambda c: c.sequence)
            if len(zone) != raw.counts.get((player, location), 0) or len({c.sequence for c in zone}) != len(zone):
                raise ValueError("current branch card count/coordinates are inconsistent")
            if location not in FIELDS and [c.sequence for c in zone] != list(range(len(zone))):
                raise ValueError("current branch list zone is not contiguous")
            if location == C.LOCATION_EXTRA and player != viewer and any(c.position & C.POS_FACEUP for c in zone):
                raise ValueError("current-root observer does not support opponent face-up Extra Deck cards")
            output = [None] * (max((c.sequence for c in zone), default=-1) + 1)
            for c in zone:
                place = (player, location, c.sequence)
                entity = entity_by_place.get(place)
                if entity is None or entity.owner not in (0, 1) or entity.placeholder:
                    raise ValueError("current branch card is absent, unowned or still a placeholder")
                hidden = not identity_visible(c, viewer, revealed)
                code = 0 if hidden else int(c.code)
                # Hidden identities/stats are never taken from the hypothetical
                # other hand/deck. Current public relations have their own DTO.
                if hidden and c.counters:
                    raise ValueError("hidden-card counters need an explicit common-public counter exporter")
                static = location in (C.LOCATION_DECK, C.LOCATION_GRAVE, C.LOCATION_REMOVED, C.LOCATION_EXTRA)
                concealed_known = player != viewer and (location in (C.LOCATION_HAND, C.LOCATION_DECK, C.LOCATION_EXTRA)
                                                        or c.position & C.POS_FACEDOWN)
                masked_stats = hidden or concealed_known
                numeric = [0] * 4 if masked_stats else [int(c.level), int(c.rank), int(c.attack), int(c.defense)]
                equip = equips.get(place)
                packed_equip = 0 if equip is None else equip[0] | (equip[1] << 8) | (equip[2] << 16)
                # QUERY_EQUIP_CARD is get_info_location(): its high byte is
                # the target's current position, not part of its public
                # controller/location/sequence relation. The public seed
                # deliberately carries only those three coordinates; target
                # position is independently exported in the target card row.
                if equip is not None and (c.equip_card & 0x00ffffff) != packed_equip:
                    raise ValueError("common-public equip relation differs from current hypothetical engine")
                row = [code, player, location, c.sequence, int(c.position) & 15, *numeric, packed_equip,
                       list(c.overlay), [] if hidden else [list(x) for x in c.counters], entity.owner,
                       0 if masked_stats or static else int(c.status), 0 if masked_stats else int(c.lscale),
                       0 if masked_stats else int(c.rscale), 0 if masked_stats else int(c.link),
                       0 if masked_stats else int(c.link_marker), not masked_stats and not static]
                if player != viewer and location == C.LOCATION_EXTRA:
                    row[0] = 0  # Known exact Extra identities live only in reveals.
                for index, material_code in enumerate(c.overlay):
                    material = materials.get((entity.uid, index))
                    if material is None or material.owner not in (0, 1) or material.placeholder:
                        raise ValueError("current public material ownership is missing")
                    material_owners.append([player, location, c.sequence, index, int(material_code), material.owner])
                output[c.sequence] = row
            if location == C.LOCATION_DECK:
                # No physical deck index, status, stat or draw order reaches a policy.
                codes = sorted(c.code for c in zone) if player == viewer else [0] * len(zone)
                if player == viewer and Counter(codes) - Counter(recipe["main"]):
                    raise ValueError("hypothetical own deck exceeds its public recipe")
                output = [[int(code), player, location, i, C.POS_FACEDOWN_DEFENSE, 0, 0, 0, 0, 0, [], [],
                           player, 0, 0, 0, 0, 0, False] for i, code in enumerate(codes)]
            cards[player][z] = output
    return CurrentRootView.from_dict({"schema": SCHEMA, "root_id": root_id, "root_hash": root_hash,
        "hypothesis_hash": hypothesis_hash, "viewer": viewer, "turn": raw.turn, "turn_player": raw.turn_player,
        "phase": raw.phase, "lp": list(raw.lp), "cards": cards, "main": recipe["main"], "extra": recipe["extra"],
        "opponent_main": recipe["main"], "opponent_extra": recipe["extra"], "material_owners": material_owners,
        "public_seed": seed})
