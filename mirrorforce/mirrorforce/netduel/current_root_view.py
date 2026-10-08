"""A seat's legal view of an owned hypothetical current root, never an engine snapshot.

The caller masks the hypothetical engine for ``viewer`` and supplies only public
facts still known to that viewer. This boundary cannot authenticate where facts
came from: root ownership and the public-only builder belong to the search client.
It rejects ambiguous coordinates, concealed identities without an explicit public
fact, private deck order, extra fields, and inconsistent current state.

``cards[player][zone]`` uses DECK/HAND/MZONE/SZONE/GRAVE/REMOVED/EXTRA and the
existing native 19-field records. Own hand/extra keep engine/network slot order;
own deck is a sorted remaining multiset. ``public_seed`` uses ABSOLUTE player
coordinates, never engine entities. Missing historical facts are explicitly cold:
no past events, chunks, choices, arrival times, use counts or neural Memory.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import re

from . import constants as C

SCHEMA = "mirrorforce_current_root_view/v2"
HISTORY_LAW = "current_root_cold_masked_field_constraints/v2"
ZONES = (C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE,
         C.LOCATION_GRAVE, C.LOCATION_REMOVED, C.LOCATION_EXTRA)
FIELDS = {C.LOCATION_MZONE: 7, C.LOCATION_SZONE: 8}


def _keys(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"current root has missing/unknown fields; expected {sorted(allowed)}")


def _int(value, low=0, high=2**32 - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"current root integer outside [{low}, {high}]: {value!r}")
    return value


def _ints(value, width=None, low=0, high=2**32 - 1):
    if not isinstance(value, (list, tuple)) or (width is not None and len(value) != width):
        raise ValueError("current root has an invalid row width")
    return [_int(x, low, high) for x in value]


def _place(value, *, nullable=False):
    if value is None and nullable:
        return None
    p, loc, seq = _ints(value, 3)
    if p not in (0, 1) or loc not in ZONES or seq > 255 or (loc in FIELDS and seq >= FIELDS[loc]):
        raise ValueError("current root has an invalid card place")
    return [p, loc, seq]


def _pairs(value):
    pairs = [_ints(row, 2) for row in value]
    if len({row[0] for row in pairs}) != len(pairs) or any(n < 1 for _, n in pairs):
        raise ValueError("current root has invalid duplicate/zero counts")
    return sorted(pairs)


def _matrix(value, width):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("current root needs two seats")
    return [_ints(row, width) for row in value]


def _seed(raw, turn):
    allowed = {"positioned", "reveals", "unpositioned", "hand_group", "field_groups", "card_metadata", "turn_counts",
               "turn_ledger", "player_hints", "activations", "chain", "chain_context", "card_status",
               "public_effects", "field_origins", "card_targets", "equips"}
    _keys(raw, allowed)
    out = {name: [] for name in allowed - {"turn_counts", "turn_ledger", "chain_context"}}
    for name in ("positioned", "reveals"):
        for row in raw.get(name, []):
            row = _ints(row, 4)
            _place(row[:3])
            _int(row[3], 1)
            out[name].append(row)
        if len({tuple(r[:3]) for r in out[name]}) != len(out[name]):
            raise ValueError(f"duplicate {name} place")
    for row in raw.get("unpositioned", []):
        loc, code, count = _ints(row, 3, 1)
        if loc not in (C.LOCATION_HAND, C.LOCATION_DECK, C.LOCATION_EXTRA, *FIELDS):
            raise ValueError("invalid unpositioned location")
        out["unpositioned"].append([loc, code, count])
    if len({tuple(r[:2]) for r in out["unpositioned"]}) != len(out["unpositioned"]):
        raise ValueError("duplicate unpositioned identity")
    out["hand_group"] = _ints(raw.get("hand_group", []), high=255)
    if len(set(out["hand_group"])) != len(out["hand_group"]):
        raise ValueError("duplicate hand group place")
    for row in raw.get("field_groups", []):
        row = _ints(row)
        if len(row) < 2 or row[0] not in FIELDS or len(set(row[1:])) != len(row[1:]) \
                or any(seq >= FIELDS[row[0]] for seq in row[1:]):
            raise ValueError("invalid unpositioned field group")
        out["field_groups"].append([row[0], *sorted(row[1:])])
    if len({row[0] for row in out["field_groups"]}) != len(out["field_groups"]):
        raise ValueError("duplicate unpositioned field group location")
    out["field_groups"].sort()
    for item in raw.get("card_metadata", []):
        _keys(item, {"place", "arrival_turn", "arrival_kind", "arrived_from", "activations", "attacks",
                     "desc_hints", "hint_kind", "hint_value"}, {"place"})
        row = {"place": _place(item["place"]), "arrival_turn": _int(item.get("arrival_turn", -1), -1, turn),
               "arrival_kind": _int(item.get("arrival_kind", 0), high=5),
               "arrived_from": _int(item.get("arrived_from", 0), high=255),
               "desc_hints": _pairs(item.get("desc_hints", []))}
        for name in ("activations", "attacks", "hint_kind", "hint_value"):
            row[name] = _int(item.get(name, 0))
        out["card_metadata"].append(row)
    out["turn_counts"] = _matrix(raw.get("turn_counts", [[0] * 6] * 2), 6)
    out["turn_ledger"] = _matrix(raw.get("turn_ledger", [[0] * 8] * 2), 8)
    for row in raw.get("player_hints", []):
        p, desc, count = _ints(row, 3)
        _int(p, high=1)
        _int(count, 1)
        out["player_hints"].append([p, desc, count])
    for item in raw.get("activations", []):
        _keys(item, {"code", "desc", "turn", "resolved", "duel", "negated", "last_turn"}, {"code", "desc"})
        row = {"code": _int(item["code"], 1), "desc": _int(item["desc"]),
               "last_turn": _int(item.get("last_turn", turn), high=turn),
               "negated": _int(item.get("negated", 0))}
        for name in ("turn", "resolved", "duel"):
            row[name] = _ints(item.get(name, [0, 0]), 2)
        out["activations"].append(row)
    for item in raw.get("chain", []):
        _keys(item, {"link", "player", "code", "desc", "origin", "source", "state", "negated", "left",
                     "targets", "moved"}, {"link", "player", "code", "desc", "origin"})
        row = {"link": _int(item["link"], 1, 255), "player": _int(item["player"], high=1),
               "code": _int(item["code"], 1), "desc": _int(item["desc"]), "origin": _place(item["origin"]),
               "source": _place(item.get("source"), nullable=True), "state": _int(item.get("state", 0), 0, 4),
               "targets": [_place(p) for p in item.get("targets", [])], "moved": []}
        for name in ("negated", "left"):
            if type(item.get(name, False)) is not bool:
                raise ValueError("current chain flags must be bool")
            row[name] = item.get(name, False)
        for moved in item.get("moved", []):
            _keys(moved, {"place", "from_location", "control", "destination"}, {"place", "from_location", "control"})
            if type(moved["control"]) is not bool or moved["from_location"] not in ZONES:
                raise ValueError("invalid current chain moved fact")
            row["moved"].append({"place": _place(moved["place"]), "from_location": moved["from_location"],
                                  "control": moved["control"],
                                  "destination": _place(moved.get("destination", moved["place"]))})
        out["chain"].append(row)
    links = [r["link"] for r in out["chain"]]
    if links != sorted(set(links)):
        raise ValueError("current chain links must be unique and ordered")
    ctx = raw.get("chain_context", {})
    _keys(ctx, {"links", "solving", "resolution", "chain_link", "settlement_link", "in_damage_step"})
    out["chain_context"] = {k: _int(ctx.get(k, max(links, default=0) if k in ("links", "chain_link") else 0))
                            for k in ("links", "solving", "resolution", "chain_link", "settlement_link")}
    context = out["chain_context"]
    damage = ctx.get("in_damage_step", False)
    if type(damage) is not bool or max(links, default=0) > context["links"] \
            or context["resolution"] > context["solving"] \
            or max(context["chain_link"], context["settlement_link"]) > context["links"]:
        raise ValueError("inconsistent current chain context")
    context["in_damage_step"] = damage
    for item in raw.get("card_status", []):
        _keys(item, {"place", "source", "bits", "turn"}, {"place", "bits", "turn"})
        out["card_status"].append({"place": _place(item["place"]),
                                   "source": _place(item.get("source"), nullable=True),
                                   "bits": _int(item["bits"], 1, 63), "turn": _int(item["turn"], 1, turn)})
    for row in raw.get("public_effects", []):
        code, desc, p, kind, created = _ints(row, 5)
        _int(code, 1)
        _int(p, high=1)
        _int(kind, 1, 3)
        _int(created, 1, turn)
        out["public_effects"].append([code, desc, p, kind, created])
    for row in raw.get("field_origins", []):
        p, loc, seq, first_controller = _ints(row, 4)
        _place([p, loc, seq])
        _int(first_controller, high=1)
        if loc not in FIELDS:
            raise ValueError("field origin outside field")
        out["field_origins"].append([p, loc, seq, first_controller])
    for name in ("card_targets", "equips"):
        for row in raw.get(name, []):
            if len(row) != 2:
                raise ValueError("card relation needs two places")
            out[name].append([_place(row[0]), _place(row[1])])
    return out


@dataclass(frozen=True)
class CurrentRootView:
    """Validated, immutable wire value; ``to_dict`` always returns an independent copy."""
    _json: str

    @classmethod
    def from_dict(cls, value: dict) -> "CurrentRootView":
        required = {"schema", "root_id", "root_hash", "hypothesis_hash", "viewer", "lp", "turn", "turn_player",
                    "phase", "cards", "main", "extra", "opponent_main", "opponent_extra", "material_owners"}
        _keys(value, required | {"public_seed", "history_law"}, required)
        legacy_native = value["schema"] == "mirrorforce_current_root_view/v1"
        expected_history = "current_root_cold/v1" if legacy_native else HISTORY_LAW
        if value["schema"] not in (SCHEMA, "mirrorforce_current_root_view/v1") \
                or value.get("history_law", expected_history) != expected_history:
            raise ValueError("unsupported current root schema/history law")
        if not isinstance(value["root_id"], str) or not value["root_id"]:
            raise ValueError("current root needs a nonempty root_id")
        for name in ("root_hash", "hypothesis_hash"):
            if not isinstance(value[name], str) or re.fullmatch(r"[0-9a-f]{64}", value[name]) is None:
                raise ValueError(f"invalid {name}")
        viewer = _int(value["viewer"], high=1)
        turn = _int(value["turn"], 1, 2**31 - 1)
        out = {k: value[k] for k in ("schema", "root_id", "root_hash", "hypothesis_hash")}
        out["schema"] = SCHEMA  # native's v1 validation callback returns the canonical full DTO
        out.update(viewer=viewer, turn=turn, turn_player=_int(value["turn_player"], high=1),
                   lp=_ints(value["lp"], 2, -(2**31), 2**31 - 1), phase=_int(value["phase"], high=0xffff),
                   history_law=HISTORY_LAW)
        if out["phase"] not in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}:
            raise ValueError("current root has an unknown phase")
        for name in ("main", "extra", "opponent_main", "opponent_extra"):
            out[name] = sorted(_ints(value[name], low=1))
        if not out["main"] or not out["opponent_main"]:
            raise ValueError("current root needs explicit own and public opponent recipes")
        seed = out["public_seed"] = _seed(value.get("public_seed", {}), turn)
        # The existing native importer calls this validator before parsing its
        # three-zone ABI. A v1 label must never smuggle field knowledge through
        # that importer, where non-hand/deck counts historically meant Extra.
        if legacy_native and (seed["field_groups"] or any(row[0] in FIELDS for row in seed["unpositioned"])):
            raise ValueError("legacy native root projection cannot carry field knowledge")
        positioned = {tuple(r[:3]): r[3] for r in seed["positioned"]}
        reveals = {tuple(r[:3]): r[3] for r in seed["reveals"]}
        cards = value["cards"]
        if not isinstance(cards, (list, tuple)) or len(cards) != 2 or any(len(z) != len(ZONES) for z in cards):
            raise ValueError("current root requires two seats and seven zones")
        out["cards"] = [[], []]
        occupied = {}
        expected_materials = {}
        for p in (0, 1):
            for location, zone in zip(ZONES, cards[p]):
                if not isinstance(zone, (list, tuple)) or len(zone) > FIELDS.get(location, 255):
                    raise ValueError("current root zone size exceeds network coordinates")
                rows = []
                for seq, card in enumerate(zone):
                    if card is None:
                        if location not in FIELDS:
                            raise ValueError("current root list zone has a hole")
                        rows.append(None)
                        continue
                    if not isinstance(card, (list, tuple)) or len(card) != 19:
                        raise ValueError("a current root native card has 19 fields")
                    row = list(card)
                    for i in (*range(7), 9, *range(12, 18)):
                        _int(row[i])
                    _int(row[7], -(2**31), 2**31 - 1)
                    _int(row[8], -(2**31), 2**31 - 1)
                    if row[1:4] != [p, location, seq] or row[4] > 15 or row[12] not in (0, 1):
                        raise ValueError("current root card has inconsistent seat/slot/owner")
                    if type(row[18]) is not bool:
                        raise ValueError("stats_known must be bool")
                    row[10] = _ints(row[10], low=1)
                    row[11] = _pairs(row[11])
                    if row[10] and location not in (C.LOCATION_MZONE, C.LOCATION_EXTRA):
                        raise ValueError("materials outside monster/extra zones")
                    place = (p, location, seq)
                    hidden = p != viewer and (location in (C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_EXTRA)
                                               or row[4] & C.POS_FACEDOWN)
                    if hidden and row[0] and positioned.get(place) != row[0] and reveals.get(place) != row[0]:
                        raise ValueError("concealed identity lacks a current public fact")
                    if p != viewer and location in (C.LOCATION_DECK, C.LOCATION_EXTRA) and row[0]:
                        raise ValueError("opponent deck/extra rows must be anonymous; use explicit reveals")
                    if not row[0] and (any(row[i] for i in (5, 6, 7, 8, 13, 14, 15, 16, 17)) or row[18]):
                        raise ValueError("hidden card data contains private stats")
                    if not row[0] and not hidden:
                        raise ValueError("a visible current-root card is missing its identity")
                    if p == viewer and not row[0]:
                        raise ValueError("own current-root identities must be complete")
                    occupied[place] = row
                    for index, code in enumerate(row[10]):
                        expected_materials[(*place, index)] = code
                    rows.append(row)
                if p == viewer and location == C.LOCATION_DECK:
                    codes = [r[0] for r in rows]
                    if codes != sorted(codes):
                        raise ValueError("own deck must be a sorted remaining multiset, not a private order")
                    if Counter(codes) - Counter(out["main"]):
                        raise ValueError("own remaining deck exceeds its explicit recipe")
                out["cards"][p].append(rows)
        for name, known in (("positioned", positioned), ("reveals", reveals)):
            for place, code in known.items():
                if place not in occupied or (occupied[place][0] not in (0, code) and place[1] != C.LOCATION_DECK):
                    raise ValueError(f"{name} contradicts current card slot")
                if name == "positioned" and place[1] in (C.LOCATION_DECK, C.LOCATION_EXTRA):
                    raise ValueError("deck known positions belong in reveals")
        if any(place in reveals and reveals[place] != code for place, code in positioned.items()):
            raise ValueError("positioned and revealed identities disagree")
        own_fixed = Counter(code for (p, loc, _), code in reveals.items() if p == viewer and loc == C.LOCATION_DECK)
        if own_fixed - Counter(row[0] for row in out["cards"][viewer][0]):
            raise ValueError("known own deck positions exceed its remaining multiset")
        materials = []
        for row in value["material_owners"]:
            p, loc, seq, index, code, owner = _ints(row, 6)
            key = (p, loc, seq, index)
            if expected_materials.pop(key, None) != code or owner not in (0, 1):
                raise ValueError("material owner rows do not match current material slots")
            materials.append([p, loc, seq, index, code, owner])
        if expected_materials:
            raise ValueError("current root is missing material owners")
        out["material_owners"] = sorted(materials)
        def existing(place):
            if place is not None and tuple(place) not in occupied:
                raise ValueError("current public fact names an absent card")
        for name in ("card_metadata", "card_status"):
            if len({tuple(item["place"]) for item in seed[name]}) != len(seed[name]):
                raise ValueError(f"duplicate {name} place")
            for item in seed[name]:
                existing(item["place"])
                if item["place"][1] in (C.LOCATION_DECK, C.LOCATION_EXTRA):
                    raise ValueError("current card metadata/status requires a tracked card place")
                if name == "card_status":
                    if item["place"][1] not in FIELDS:
                        raise ValueError("current lingering card status must be on the field")
                    existing(item["source"])
        for row in seed["field_origins"]:
            existing(row[:3])
        for pair in seed["card_targets"]:
            for place in pair:
                existing(place)
        for source, target in seed["equips"]:
            existing(source)
            existing(target)
            if source[1] not in FIELDS or target[1] not in FIELDS \
                    or occupied[tuple(source)][9] != (target[0] | target[1] << 8 | target[2] << 16):
                raise ValueError("public equip seed contradicts current native card records")
        for row in occupied.values():
            if row[9]:
                target = [row[9] & 255, (row[9] >> 8) & 255, (row[9] >> 16) & 255]
                if row[2] not in FIELDS or target[1] not in FIELDS or row[9] >> 24:
                    raise ValueError("current equip relation must join field places")
                existing(target)
        for link in seed["chain"]:
            existing(link["source"])
            for place in link["targets"]:
                existing(place)
            for moved in link["moved"]:
                existing(moved["place"])
        opponent_hand = out["cards"][1 - viewer][1]
        for seq in seed["hand_group"]:
            if seq >= len(opponent_hand) or opponent_hand[seq][0] \
                    or (1 - viewer, C.LOCATION_HAND, seq) in positioned \
                    or (1 - viewer, C.LOCATION_HAND, seq) in reveals:
                raise ValueError("unpositioned hand group names an absent/known card")
        claimed = Counter()
        for loc, code, count in seed["unpositioned"]:
            claimed[loc] += count
        field_groups = {row[0]: row[1:] for row in seed["field_groups"]}
        if set(field_groups) != set(claimed).intersection(FIELDS):
            raise ValueError("field identities require exactly their explicit unknown-slot groups")
        for loc, group in field_groups.items():
            for seq in group:
                place = (1 - viewer, loc, seq)
                row = occupied.get(place)
                if row is None or row[0] or not row[4] & C.POS_FACEDOWN \
                        or place in positioned or place in reveals:
                    raise ValueError("unpositioned field group names an absent/known card")
        for loc, count in claimed.items():
            capacity = len(seed["hand_group"]) if loc == C.LOCATION_HAND else len(field_groups[loc]) if loc in FIELDS else sum(
                r is not None and not r[0] and (1 - viewer, loc, r[3]) not in reveals
                for r in out["cards"][1 - viewer][ZONES.index(loc)])
            if count > capacity:
                raise ValueError("unpositioned identities exceed unknown current slots")
        return cls(json.dumps(out, sort_keys=True, separators=(",", ":")))

    def to_dict(self) -> dict:
        return json.loads(self._json)

    def to_native_dict(self) -> dict:
        """Project to the trained observer's original three-zone vocabulary.

        Field identities without positions stay in this immutable root DTO and
        its hash, but are not assigned to any card row and are not a new model
        input. The old native importer dispatches non-hand/deck counts to Extra,
        so it MUST NOT receive field counts. The opponent policy therefore does
        not exploit this set knowledge; the engine and bank retain it.
        """
        projected = self.to_dict()
        projected["schema"] = "mirrorforce_current_root_view/v1"
        projected["history_law"] = "current_root_cold/v1"
        seed = projected["public_seed"]
        seed["unpositioned"] = [row for row in seed["unpositioned"] if row[0] not in FIELDS]
        del seed["field_groups"]
        return projected

    @property
    def viewer(self) -> int:
        return self.to_dict()["viewer"]

    @property
    def root_id(self) -> str:
        return self.to_dict()["root_id"]

    @property
    def root_hash(self) -> str:
        return self.to_dict()["root_hash"]

    @property
    def hypothesis_hash(self) -> str:
        return self.to_dict()["hypothesis_hash"]

    @property
    def lp(self) -> tuple[int, int]:
        return tuple(self.to_dict()["lp"])

    @property
    def turn(self) -> int:
        return self.to_dict()["turn"]

    @property
    def turn_player(self) -> int:
        return self.to_dict()["turn_player"]

    @property
    def phase(self) -> int:
        return self.to_dict()["phase"]

    def require_binding(self, *, root_id: str, root_hash: str, hypothesis_hash: str, viewer: int) -> None:
        data = self.to_dict()
        if any(data[k] != v for k, v in locals().items() if k in ("root_id", "root_hash", "hypothesis_hash", "viewer")):
            raise ValueError("current root/viewer binding mismatch")
