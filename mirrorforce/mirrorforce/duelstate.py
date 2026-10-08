"""Reader for ygopro-core's viewer-scoped ``query_duel_state`` v2 stream."""

from __future__ import annotations

import ctypes
import struct
from dataclasses import dataclass, field


VERSION = 2
HEADER = struct.Struct("<BBHI")
SECTION = struct.Struct("<BI")

SECTION_END = 0
SECTION_META = 1
SECTION_PLAYER = 2
SECTION_CARD = 3
SECTION_RELATION = 4
SECTION_EFFECT = 5
SECTION_EFFECT_OBJECT = 6

VIEW_PLAYER_0 = 0
VIEW_PLAYER_1 = 1
VIEW_SHARED_PUBLIC = 2
VIEW_OMNISCIENT_LABEL = 3

FLAG_TARGET_ENTITY_IDS = 0x1

CARD_IDENTITY_VISIBLE = 0x1
CARD_SEQUENCE_VISIBLE = 0x2
CARD_ANONYMOUS = 0x4
CARD_OVERLAY = 0x8
CARD_OWN_KNOWLEDGE = 0x10

_META = struct.Struct("<16I")
_PLAYER = struct.Struct("<Iiii12I")
_CARD = struct.Struct("<13I4i14I")
_RELATION = struct.Struct("<8I")
_EFFECT = struct.Struct("<7Ii24I")
_EFFECT_OBJECT = struct.Struct("<4I")
QUERY_BUFFER_SIZE = 0x40000
TRAINING_TARGET_SCHEMA = "duel-state-v2-target-entity-v1"


class DuelStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class DuelMeta:
    view: int
    requested_flags: int
    duel_rule: int
    duel_options: int
    turn_id: int
    phase: int
    turn_player: int
    priority_0: int
    priority_1: int
    can_shuffle: int
    chain_size: int
    win_player: int
    win_reason: int
    to_bp: int
    to_m2: int
    to_ep: int


@dataclass(frozen=True)
class PlayerState:
    player: int
    lp: int
    start_count: int
    draw_count: int
    used_location: int
    disabled_location: int
    extra_p_count: int
    szone_size: int
    deck_count: int
    hand_count: int
    mzone_slots: int
    szone_slots: int
    grave_count: int
    removed_count: int
    extra_count: int
    tag_extra_p_count: int


@dataclass(frozen=True)
class CardState:
    ref: int
    info_location: int
    printed_code: int
    current_code: int
    code2: int
    type: int
    level: int
    rank: int
    link: int
    lscale: int
    rscale: int
    attribute: int
    race: int
    attack: int
    defense: int
    base_attack: int
    base_defense: int
    owner: int
    summon_player: int
    summon_info: int
    status: int
    attack_announce_count: int
    direct_attackable: int
    announce_count: int
    attacked_count: int
    attack_all_target: int
    attack_controler: int
    turn_id: int
    turn_counter: int
    host_ref: int
    flags: int
    #: Target-only stable id inside one duel; zero unless explicitly requested.
    entity_id: int = 0

    @property
    def controller(self) -> int:
        return self.info_location & 0xFF

    @property
    def location(self) -> int:
        return (self.info_location >> 8) & 0xFF

    @property
    def sequence(self) -> int:
        return (self.info_location >> 16) & 0xFF

    @property
    def position(self) -> int:
        return (self.info_location >> 24) & 0xFF

    @property
    def identity_visible(self) -> bool:
        return bool(self.flags & CARD_IDENTITY_VISIBLE)

    @property
    def anonymous(self) -> bool:
        return bool(self.flags & CARD_ANONYMOUS)

    @property
    def overlay(self) -> bool:
        return bool(self.flags & CARD_OVERLAY)


@dataclass(frozen=True)
class RelationState:
    kind: int
    source_ref: int
    target_ref: int
    reset: int
    count: int
    aux0: int
    aux1: int
    reserved: int


@dataclass(frozen=True)
class EffectState:
    ref: int
    code: int
    type: int
    source_ref: int
    handler_ref: int
    description: int
    reset_flag: int
    reset_count: int
    count_code: int
    active_type: int
    category_low: int
    category_high: int
    flag_low: int
    flag_high: int
    flag2_low: int
    flag2_high: int
    s_range: int
    o_range: int
    range: int
    status: int
    container: int
    effect_owner: int
    target_player: int
    count_limit: int
    count_limit_max: int
    label_count: int
    object_type: int
    copy_id: int
    active_location: int
    active_sequence: int
    active_handler_ref: int
    last_handler_ref: int
    #: Target-only stable registration id; zero unless explicitly requested.
    entity_id: int = 0

    @property
    def category(self) -> int:
        return self.category_low | (self.category_high << 32)

    @property
    def flag(self) -> int:
        return self.flag_low | (self.flag_high << 32)

    @property
    def flag2(self) -> int:
        return self.flag2_low | (self.flag2_high << 32)


@dataclass(frozen=True)
class EffectObject:
    effect_ref: int
    object_type: int
    card_ref: int
    aux_ref: int


@dataclass(frozen=True)
class DuelState:
    version: int
    length: int
    meta: DuelMeta | None = None
    players: tuple[PlayerState, ...] = ()
    cards: tuple[CardState, ...] = ()
    relations: tuple[RelationState, ...] = ()
    effects: tuple[EffectState, ...] = ()
    effect_objects: tuple[EffectObject, ...] = ()
    unknown_sections: dict[int, bytes] = field(default_factory=dict)


def _records(payload: bytes, layout: struct.Struct, build, extension=None):
    if len(payload) < 4:
        raise DuelStateError("record section is shorter than its header")
    count, width = struct.unpack_from("<HH", payload, 0)
    if width < layout.size:
        raise DuelStateError(
            f"record is {width} bytes, reader needs {layout.size}"
        )
    if 4 + count * width > len(payload):
        raise DuelStateError("record section is truncated")
    out = []
    pos = 4
    for _ in range(count):
        values = layout.unpack_from(payload, pos)
        if extension is not None:
            values += tuple(extension(payload, pos, width))
        out.append(build(*values))
        pos += width
    return tuple(out)


def _card_extension(payload: bytes, pos: int, width: int):
    return struct.unpack_from("<Q", payload, pos + _CARD.size) if (
        width >= _CARD.size + 8
    ) else (0,)


def _effect_extension(payload: bytes, pos: int, width: int):
    return struct.unpack_from("<I", payload, pos + _EFFECT.size) if (
        width >= _EFFECT.size + 4
    ) else (0,)


def parse(raw: bytes) -> DuelState:
    if len(raw) < HEADER.size:
        raise DuelStateError("duel-state header is truncated")
    version, header_len, _reserved, total = HEADER.unpack_from(raw, 0)
    if version != VERSION:
        raise DuelStateError(f"duel-state version {version}, expected {VERSION}")
    if header_len < HEADER.size or total != len(raw):
        raise DuelStateError(
            f"invalid duel-state lengths: header={header_len}, total={total}, raw={len(raw)}"
        )
    pos = header_len
    meta = None
    players = ()
    cards = ()
    relations = ()
    effects = ()
    effect_objects = ()
    unknown: dict[int, bytes] = {}
    while pos + SECTION.size <= len(raw):
        section_id, length = SECTION.unpack_from(raw, pos)
        pos += SECTION.size
        end = pos + length
        if end > len(raw):
            raise DuelStateError(f"section {section_id} is truncated")
        payload = raw[pos:end]
        pos = end
        if section_id == SECTION_END:
            if length:
                raise DuelStateError("END section must be empty")
            break
        if section_id == SECTION_META:
            rows = _records(payload, _META, lambda *v: DuelMeta(*v))
            if len(rows) != 1:
                raise DuelStateError(f"META has {len(rows)} rows, expected 1")
            meta = rows[0]
        elif section_id == SECTION_PLAYER:
            players = _records(payload, _PLAYER, lambda *v: PlayerState(*v))
        elif section_id == SECTION_CARD:
            cards = _records(
                payload, _CARD, lambda *v: CardState(*v), _card_extension
            )
        elif section_id == SECTION_RELATION:
            relations = _records(
                payload, _RELATION, lambda *v: RelationState(*v)
            )
        elif section_id == SECTION_EFFECT:
            effects = _records(
                payload, _EFFECT, lambda *v: EffectState(*v), _effect_extension
            )
        elif section_id == SECTION_EFFECT_OBJECT:
            effect_objects = _records(
                payload, _EFFECT_OBJECT, lambda *v: EffectObject(*v)
            )
        else:
            unknown[section_id] = payload
    else:
        raise DuelStateError("duel-state stream has no END section")
    return DuelState(
        version=version, length=total, meta=meta,
        players=players, cards=cards, relations=relations, effects=effects,
        effect_objects=effect_objects, unknown_sections=unknown,
    )


def declare(lib: ctypes.CDLL) -> bool:
    try:
        fn = lib.query_duel_state
    except AttributeError:
        return False
    fn.restype = ctypes.c_int32
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint32,
        ctypes.c_char_p, ctypes.c_int32,
    ]
    return True


def supports(core) -> bool:
    return declare(core._lib)


def query_duel_state(core, pduel, view: int, flags: int = 0, buf=None) -> DuelState:
    if view not in (
        VIEW_PLAYER_0, VIEW_PLAYER_1,
        VIEW_SHARED_PUBLIC, VIEW_OMNISCIENT_LABEL,
    ):
        raise ValueError(f"invalid duel-state view: {view}")
    if not declare(core._lib):
        raise DuelStateError(f"{core.lib_path} has no query_duel_state")
    if buf is None:
        buf = ctypes.create_string_buffer(QUERY_BUFFER_SIZE)
    length = core._lib.query_duel_state(pduel, view, flags, buf, len(buf))
    if length <= 0:
        raise DuelStateError(
            f"query_duel_state returned {length} with {len(buf)}-byte buffer"
        )
    return parse(buf.raw[:length])


def as_training_targets(state: DuelState, legacy: dict | None = None) -> dict:
    """Project viewer-safe v2 rows onto the world-model target dictionary.

    ``query_effect_info`` still carries count-code/setcode sections not yet in
    v2.  ``legacy`` may supply those sections; CARD/RELATION/EFFECT always come
    from the richer viewer-scoped API.
    """
    if (state.meta is not None
            and state.meta.requested_flags & FLAG_TARGET_ENTITY_IDS
            and any(not card.entity_id for card in state.cards)):
        raise DuelStateError(
            "target entity ids were requested but the core returned a zero CARD id"
        )
    out = dict(legacy or {})
    cards = {card.ref: card for card in state.cards}
    legacy_cards = {
        (int(row.get("info_location", 0)), int(row.get("code", 0))): row
        for row in (legacy or {}).get("cards", ())
    }

    def card_code(ref: int) -> int:
        card = cards.get(int(ref))
        if card is None or not card.identity_visible:
            return 0
        return int(card.printed_code)

    def card_at(ref: int) -> int:
        card = cards.get(int(ref))
        return 0 if card is None else int(card.info_location)

    def card_entity(ref: int) -> int:
        card = cards.get(int(ref))
        return 0 if card is None else int(card.entity_id)

    relation_count: dict[int, int] = {}
    material_count: dict[int, int] = {}
    for relation in state.relations:
        relation_count[relation.source_ref] = (
            relation_count.get(relation.source_ref, 0) + 1
        )
        if relation.kind == 3:  # DUEL_STATE_REL_SUMMON_MATERIAL
            material_count[relation.source_ref] = (
                material_count.get(relation.source_ref, 0) + 1
            )

    out["state_schema"] = "duel-state-v2"
    out["target_schema"] = (
        TRAINING_TARGET_SCHEMA
        if state.meta is not None
        and state.meta.requested_flags & FLAG_TARGET_ENTITY_IDS
        else "duel-state-v2-snapshot-only"
    )
    out["players"] = [dict(player.__dict__) for player in state.players]
    out["cards"] = [
        {
            "ref": card.ref,
            "info_location": card.info_location,
            "code": card_code(card.ref),
            "printed_code": card_code(card.ref),
            "current_code": card.current_code if card.identity_visible else 0,
            "code2": card.code2 if card.identity_visible else 0,
            "lscale": card.lscale,
            "rscale": card.rscale,
            "owner": card.owner,
            "summon_player": card.summon_player,
            "summon_info": card.summon_info,
            "status": card.status,
            "attack_announce_count": card.attack_announce_count,
            "direct_attackable": card.direct_attackable,
            "announce_count": card.announce_count,
            "attacked_count": card.attacked_count,
            "attack_all_target": card.attack_all_target,
            "attack_controler": card.attack_controler,
            "material_count": material_count.get(card.ref, 0),
            "relation_count": relation_count.get(card.ref, 0),
            "indestructable_count": int(legacy_cards.get(
                (card.info_location, card_code(card.ref)), {}
            ).get("indestructable_count", 0)),
            "host_ref": card.host_ref,
            "flags": card.flags,
            # Label-factory metadata only.  It never becomes a token or model
            # target; it carries graph edges across CARD moves/reindexing.
            "entity_id": card.entity_id,
        }
        for card in state.cards
    ]
    out["relations"] = [
        {
            "kind": relation.kind,
            "source_ref": relation.source_ref,
            "source_info": card_at(relation.source_ref),
            "source_code": card_code(relation.source_ref),
            "target_ref": relation.target_ref,
            "target_info": card_at(relation.target_ref),
            "target_code": card_code(relation.target_ref),
            "source_entity_id": card_entity(relation.source_ref),
            "target_entity_id": card_entity(relation.target_ref),
            "reset": relation.reset,
            "count": relation.count,
            "aux0": relation.aux0,
            "aux1": relation.aux1,
        }
        for relation in state.relations
    ]
    objects = {obj.effect_ref: obj for obj in state.effect_objects}
    effects = []
    for effect in state.effects:
        row = dict(effect.__dict__)
        row.update({
            "id": effect.ref,
            "entity_id": effect.entity_id,
            "owner_code": card_code(effect.source_ref),
            "owner_info_location": card_at(effect.source_ref),
            "owner_entity_id": card_entity(effect.source_ref),
            "handler_code": card_code(effect.handler_ref),
            "handler_info_location": card_at(effect.handler_ref),
            "handler_entity_id": card_entity(effect.handler_ref),
            "active_handler_info_location": card_at(effect.active_handler_ref),
            "active_handler_entity_id": card_entity(effect.active_handler_ref),
            "last_handler_info_location": card_at(effect.last_handler_ref),
            "last_handler_entity_id": card_entity(effect.last_handler_ref),
            "category": effect.category,
            "flag": effect.flag,
            "flag2": effect.flag2,
        })
        obj = objects.get(effect.ref)
        if obj is not None:
            row.update({
                "label_object_type": obj.object_type,
                "label_object_info_location": card_at(obj.card_ref),
                "label_object_code": card_code(obj.card_ref),
                "label_object_entity_id": card_entity(obj.card_ref),
            })
        effects.append(row)
    out["effects"] = effects
    return out
