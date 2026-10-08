"""Read the core's ``query_effect_info`` dump: the exact "already applied" state.

``ygopro-core`` exposes the board through ``QUERY_*`` flags and the message
stream, and neither fully reaches runtime effects attached to players or cards.
A resolved Maxx "C" is three ``EFFECT_TYPE_CONTINUOUS`` effects sitting
in ``field::effects``, owned by a card in the graveyard; Droll & Lock Bird is
two restriction auras pointed at both players.  Nothing is queryable and --
because neither script sets ``EFFECT_FLAG_CLIENT_HINT`` -- nothing is even
announced.  From outside, the whole turn's worth of state shows up only as
menu entries that quietly stop appearing.

``query_effect_info`` is the read-only dump that closes that gap.  Its source
of truth is the private ``mirrorforce-ygopro-core`` repository at
commit ``d8592f0``; patch copies were retired once that repository became the
maintained core.  It walks existing public members, calls no Lua and mutates
nothing.  This module is the read side.

What comes back
---------------

``effects``
    Every registered effect carrying ``EFFECT_FLAG_FIELD_ONLY``, plus every
    runtime effect in a card's SINGLE/FIELD/EQUIP/TARGET/XMATERIAL container,
    plus sparse INITIAL rows whose mutable count/status/active-handler state is
    non-default.  Each
    record names its source card (passcode plus where that card is *now*,
    typically the graveyard), the player it was registered for, which players
    it targets, its remaining lifetime (``reset_flag`` / ``reset_count``) and
    its full flag words, runtime handler and card-valued ``label_object``.
    Unchanged INITIAL card-script semantics stay absent because the card/effect
    embedding already represents them.

``count_codes``
    The once-per-turn / once-per-duel / once-per-chain accounting.  "This
    player has already used Maxx \"C\" this turn" is
    ``info.count_used(0, 23434538)``.

``turn``, ``spsummon_once``, ``activity``
    The per-turn summon / attack tallies that feed "you may only special
    summon once per turn"-style conditions, the once-per-turn-per-name special
    summon table, and the script-registered custom activity counters.

Which shared object
-------------------

The dump only exists in a core built from the patch, so this module defaults to
its own build (``mirrorforce/build/effectinfo/``) rather than the object
``ygoenv`` links; ``MF_EFFECTINFO_LIB`` overrides it.  :func:`get_effectinfo_core`
returns a core bound to it, kept separate from the package singleton so that
nothing already holding that singleton has its core changed underneath it --
see that function for why two cores in one process need care.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ACTIVITY_NAMES",
    "COUNT_CHAIN",
    "COUNT_DUEL",
    "COUNT_TURN",
    "CONTAINER_NAMES",
    "ActivityCount",
    "CardStateInfo",
    "CountCode",
    "DEFAULT_EFFECTINFO_LIB",
    "DEFAULT_SCRIPT_OVERRIDES",
    "EFFECT_FLAG_EFFECT",
    "EffectInfo",
    "EffectInfoError",
    "RegisteredEffect",
    "RelationInfo",
    "SetCodeInfo",
    "SpSummonOnce",
    "PlayerState",
    "TurnCounters",
    "declare",
    "get_effectinfo_core",
    "expires_on_chain_end",
    "expires_this_turn",
    "flag_effect_code",
    "parse",
    "query_effect_info",
    "script_override_provenance",
    "supports",
]

DEFAULT_EFFECTINFO_LIB = os.environ.get(
    "MF_EFFECTINFO_LIB",
    "/path/to/mirrorforce/build/effectinfo/libygopro-core-effectinfo.so",
)
DEFAULT_SCRIPT_OVERRIDES = Path(os.environ.get(
    "MF_SCRIPT_OVERRIDES",
    str(Path(__file__).resolve().parents[1] / "script-overrides"),
))


def script_override_provenance(path: str | Path = DEFAULT_SCRIPT_OVERRIDES):
    """Content hashes for the exact runtime script overrides, if any."""
    root = Path(path)
    rows = []
    if root.is_dir():
        for script in sorted(root.glob("c*.lua")):
            rows.append({
                "name": script.name,
                "sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            })
    return rows

#: capacity handed to the core.  A busy board registers a few dozen runtime
#: effects at ~92 bytes each, so this is orders of magnitude of headroom; the
#: core returns 0 rather than overrunning if it ever is not.
QUERY_BUFFER_SIZE = 0x20000

VERSION = 1
HEADER = struct.Struct("<BBHI")
SECTION = struct.Struct("<BI")

SECTION_END = 0
SECTION_EFFECTS = 1
SECTION_COUNT_CODE = 2
SECTION_TURN = 3
SECTION_SPSUMMON_ONCE = 4
SECTION_ACTIVITY = 5
SECTION_PLAYER_STATE = 6
SECTION_CARD_STATE = 7
SECTION_RELATIONS = 8
SECTION_SETCODE = 9

_EFFECT = struct.Struct("<7Ii2I2Q4H6BH")
_EFFECT_EXT = struct.Struct("<7Ii2I2Q4H6BH5I")
_COUNT = struct.Struct("<BBIi")
_TURN_HEAD = struct.Struct("<HHBBBBBBHH")
_TURN_PLAYER = struct.Struct("<i8B")
_SPONCE = struct.Struct("<BII")
_ACTIVITY = struct.Struct("<BiHH")
_PLAYER = struct.Struct("<5I")
_CARD_STATE = struct.Struct("<15I")
_RELATION = struct.Struct("<7I")
_SETCODE = struct.Struct("<3I")

CONTAINER_NAMES = {
    1: "aura",
    2: "ignition",
    3: "activate",
    4: "trigger_o",
    5: "trigger_f",
    6: "quick_o",
    7: "quick_f",
    8: "continuous",
    9: "card_single",
    10: "card_field",
    11: "card_equip",
    12: "card_target",
    13: "card_xmaterial",
}

#: ``ACTIVITY_*`` in ``ygopro-core/common.h``
ACTIVITY_NAMES = {
    1: "summon",
    2: "normalsummon",
    3: "spsummon",
    4: "flipsummon",
    5: "attack",
    7: "chain",
}

COUNT_TURN = 0
COUNT_DUEL = 1
COUNT_CHAIN = 2
COUNT_TABLE_NAMES = {COUNT_TURN: "turn", COUNT_DUEL: "duel", COUNT_CHAIN: "chain"}

# effect.h
EFFECT_FLAG_FIELD_ONLY = 0x0008
EFFECT_FLAG_CANNOT_DISABLE = 0x0400
EFFECT_FLAG_PLAYER_TARGET = 0x0800
EFFECT_FLAG_OATH = 0x80000
EFFECT_FLAG_CLIENT_HINT = 0x4000000
EFFECT_FLAG_EFFECT = 0x20000000
MAX_CARD_ID = 0x0FFFFFFF

RESET_SELF_TURN = 0x10000000
RESET_OPPO_TURN = 0x20000000
RESET_PHASE = 0x40000000
RESET_CHAIN = 0x80000000

PHASE_END = 0x200

#: ``PLAYER_NONE`` in ``common.h``; also the count_code slot for effects with no
#: player attribution
PLAYER_NONE = 2


def flag_effect_code(code: int) -> int:
    """The passcode a ``RegisterFlagEffect`` marker's code encodes, else 0."""

    code = int(code)
    return (code & MAX_CARD_ID) if code & EFFECT_FLAG_EFFECT else 0


def expires_this_turn(reset_flag: int) -> bool:
    """``SetReset(RESET_PHASE + PHASE_END)`` -- gone at this end phase."""

    reset_flag = int(reset_flag)
    return bool(reset_flag & RESET_PHASE) and bool(reset_flag & PHASE_END)


def expires_on_chain_end(reset_flag: int) -> bool:
    """``SetReset(RESET_CHAIN)`` -- gone when the current chain finishes."""

    return bool(int(reset_flag) & RESET_CHAIN)


class EffectInfoError(RuntimeError):
    """The dump could not be produced or parsed."""


@dataclass(frozen=True)
class RegisteredEffect:
    """One ``EFFECT_FLAG_FIELD_ONLY`` effect registered on a player.

    ``owner_*`` describe the *source card* as it stands now -- for a resolved
    Maxx "C" that is the copy in the graveyard, which is the whole point: the
    graveyard list alone cannot tell a discarded copy from a resolved one.
    """

    id: int  #: ``effect::id``; unique and monotonic within the duel
    code: int
    type: int
    owner_code: int  #: source card passcode (``card::data.code``)
    owner_controler: int
    owner_location: int
    owner_sequence: int
    owner_position: int
    description: int
    reset_flag: int
    reset_count: int
    count_code: int
    active_type: int
    flag: int  #: ``effect::flag[0]``, the ``EFFECT_FLAG_*`` word
    flag2: int  #: ``effect::flag[1]``, the ``EFFECT_FLAG2_*`` word
    s_range: int
    o_range: int
    range: int
    status: int
    container: int
    effect_owner: int  #: the player the effect was registered for
    target_player: int  #: bit 0 -> player 0 is affected, bit 1 -> player 1
    count_limit: int
    count_limit_max: int
    label_count: int
    handler_controler: int = 0
    handler_location: int = 0
    handler_sequence: int = 0
    handler_position: int = 0
    handler_code: int = 0
    label_object_type: int = 0
    label_object_controler: int = 0
    label_object_location: int = 0
    label_object_sequence: int = 0
    label_object_position: int = 0
    label_object_code: int = 0

    @property
    def container_name(self) -> str:
        return CONTAINER_NAMES.get(self.container, f"container_{self.container}")

    @staticmethod
    def _pack_location(controler: int, location: int, sequence: int,
                       position: int) -> int:
        return (controler | (location << 8) | (sequence << 16)
                | (position << 24))

    @property
    def owner_info_location(self) -> int:
        return self._pack_location(
            self.owner_controler, self.owner_location,
            self.owner_sequence, self.owner_position,
        )

    @property
    def handler_info_location(self) -> int:
        return self._pack_location(
            self.handler_controler, self.handler_location,
            self.handler_sequence, self.handler_position,
        )

    @property
    def label_object_info_location(self) -> int:
        return self._pack_location(
            self.label_object_controler, self.label_object_location,
            self.label_object_sequence, self.label_object_position,
        )

    @property
    def is_flag_effect(self) -> bool:
        """A ``Duel.RegisterFlagEffect`` marker rather than a real effect."""
        return bool(self.code & EFFECT_FLAG_EFFECT)

    @property
    def flag_effect_code(self) -> int:
        """The passcode a flag effect's code encodes (0 if not a flag effect)."""
        return flag_effect_code(self.code)

    def affects(self, player: int) -> bool:
        """Is ``player`` inside the effect's target range?

        Restriction auras use ``EFFECT_FLAG_PLAYER_TARGET`` with ``s_range`` /
        ``o_range``; the core computes this in ``effect::is_target_player`` and
        the dump carries its answer, so no range arithmetic is redone here.
        """
        return bool(self.target_player & (1 << player))

    @property
    def expires_this_turn(self) -> bool:
        """``SetReset(RESET_PHASE + PHASE_END)`` -- gone at this end phase."""
        return expires_this_turn(self.reset_flag)

    @property
    def expires_on_chain_end(self) -> bool:
        return expires_on_chain_end(self.reset_flag)

    def as_dict(self) -> dict:
        out = dict(self.__dict__)
        out["container_name"] = self.container_name
        out["is_flag_effect"] = self.is_flag_effect
        out["flag_effect_code"] = self.flag_effect_code
        out["expires_this_turn"] = self.expires_this_turn
        out["owner_info_location"] = self.owner_info_location
        out["handler_info_location"] = self.handler_info_location
        out["label_object_info_location"] = self.label_object_info_location
        return out


@dataclass(frozen=True)
class CountCode:
    """One row of an ``effect_count_code`` table: a used activation right."""

    table: int  #: :data:`COUNT_TURN` / :data:`COUNT_DUEL` / :data:`COUNT_CHAIN`
    player: int  #: 0, 1, or :data:`PLAYER_NONE`
    code: int  #: the ``SetCountLimit`` count code, usually the card's passcode
    value: int  #: activations already spent

    @property
    def table_name(self) -> str:
        return COUNT_TABLE_NAMES.get(self.table, f"table_{self.table}")

    def as_dict(self) -> dict:
        return dict(self.__dict__, table_name=self.table_name)


@dataclass(frozen=True)
class SpSummonOnce:
    """One row of ``spsummon_once_map``: this name, special summoned already."""

    player: int
    code: int
    count: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ActivityCount:
    """One ``Duel.AddCustomActivityCounter`` counter, per player."""

    kind: int  #: ``ACTIVITY_*``
    counter_id: int
    count: tuple[int, int]

    @property
    def kind_name(self) -> str:
        return ACTIVITY_NAMES.get(self.kind, f"activity_{self.kind}")

    def as_dict(self) -> dict:
        return dict(self.__dict__, kind_name=self.kind_name)


@dataclass(frozen=True)
class PlayerState:
    """Exact player zone masks and capacity state from ``field::player``."""

    player: int
    used_location: int
    disabled_location: int
    extra_p_count: int
    szone_size: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class CardStateInfo:
    """Per-card state absent from the ordinary ``QUERY_*`` protocol."""

    info_location: int
    code: int
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
    material_count: int
    relation_count: int
    indestructible_count: int

    @property
    def controller(self) -> int:
        return self.info_location & 0xFF

    @property
    def location(self) -> int:
        return (self.info_location >> 8) & 0xFF

    @property
    def sequence(self) -> int:
        return (self.info_location >> 16) & 0xFF

    def as_dict(self) -> dict:
        return dict(self.__dict__, controller=self.controller,
                    location=self.location, sequence=self.sequence)


@dataclass(frozen=True)
class RelationInfo:
    """A persistent material or generic card-to-card relation."""

    kind: int
    source_info: int
    source_code: int
    target_info: int
    target_code: int
    reset: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class SetCodeInfo:
    """One current (possibly effect-modified) series code."""

    info_location: int
    code: int
    setcode: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class TurnCounters:
    """The per-turn tallies ``Duel.GetActivityCount`` serves to scripts."""

    turn_id: int
    phase: int
    turn_player: int
    current_player: int
    duel_rule: int
    deck_reversed: int
    remove_brainwashing: int
    turn_id_by_player: tuple[int, int]
    summon_count: tuple[int, int]
    extra_summon: tuple[int, int]
    summon_state_count: tuple[int, int]
    normalsummon_state_count: tuple[int, int]
    flipsummon_state_count: tuple[int, int]
    spsummon_state_count: tuple[int, int]
    attack_state_count: tuple[int, int]
    battle_phase_count: tuple[int, int]
    battled_count: tuple[int, int]

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class EffectInfo:
    """A whole ``query_effect_info`` dump, parsed."""

    version: int
    length: int
    effects: tuple[RegisteredEffect, ...] = ()
    count_codes: tuple[CountCode, ...] = ()
    turn: TurnCounters | None = None
    spsummon_once: tuple[SpSummonOnce, ...] = ()
    activity: tuple[ActivityCount, ...] = ()
    players: tuple[PlayerState, ...] = ()
    cards: tuple[CardStateInfo, ...] = ()
    relations: tuple[RelationInfo, ...] = ()
    setcodes: tuple[SetCodeInfo, ...] = ()
    #: sections this reader did not recognise, kept so a newer core can be read
    #: by an older reader without losing anything
    unknown_sections: dict[int, bytes] = field(default_factory=dict)

    # -- queries -----------------------------------------------------------

    def player_effects(self, player: int) -> tuple[RegisteredEffect, ...]:
        """Effects registered *by* ``player`` (``effect_owner``)."""
        return tuple(e for e in self.effects if e.effect_owner == player)

    def effects_affecting(self, player: int) -> tuple[RegisteredEffect, ...]:
        """Restriction auras whose target range covers ``player``."""
        return tuple(e for e in self.effects if e.affects(player))

    def effects_from(self, code: int) -> tuple[RegisteredEffect, ...]:
        """Everything a given source card has left applied."""
        return tuple(e for e in self.effects if e.owner_code == code)

    def count_used(self, player: int, code: int, table: int = COUNT_TURN) -> int:
        """Activations of ``code`` already spent by ``player``.

        ``SetCountLimit(1, 23434538)`` means "once per turn"; after Maxx "C" is
        chained this reads 1 for the activating player, and it stays 1 even if
        the activation is negated -- only ``EFFECT_COUNT_CODE_OATH`` counts are
        refunded (``processor.cpp``, the negate path).
        """
        for row in self.count_codes:
            if row.table == table and row.player == player and row.code == code:
                return row.value
        return 0

    def once_per_turn_used(self, player: int) -> dict[int, int]:
        return {
            row.code: row.value
            for row in self.count_codes
            if row.table == COUNT_TURN and row.player == player
        }

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "length": self.length,
            "effects": [e.as_dict() for e in self.effects],
            "count_codes": [c.as_dict() for c in self.count_codes],
            "turn": self.turn.as_dict() if self.turn else None,
            "spsummon_once": [s.as_dict() for s in self.spsummon_once],
            "activity": [a.as_dict() for a in self.activity],
            "players": [p.as_dict() for p in self.players],
            "cards": [c.as_dict() for c in self.cards],
            "relations": [r.as_dict() for r in self.relations],
            "setcodes": [s.as_dict() for s in self.setcodes],
            "unknown_sections": {k: v.hex() for k, v in self.unknown_sections.items()},
        }


# -- parsing ---------------------------------------------------------------


def _parse_effects(payload: bytes) -> tuple[RegisteredEffect, ...]:
    count, record_len = struct.unpack_from("<HH", payload, 0)
    if record_len < _EFFECT.size:
        raise EffectInfoError(
            f"effect record is {record_len} bytes, reader needs {_EFFECT.size}"
        )
    out = []
    pos = 4
    for _ in range(count):
        fields = _EFFECT.unpack_from(payload, pos)
        extension = ((0, 0, 0, 0, 0) if record_len < _EFFECT_EXT.size else
                     struct.unpack_from("<5I", payload, pos + _EFFECT.size))
        pos += record_len  # a longer record from a newer core: skip the tail
        (
            eid, code, etype, owner_code, owner_loc, description, reset_flag,
            reset_count, count_code, active_type, flag0, flag1,
            s_range, o_range, erange, status,
            container, effect_owner, target_player, count_limit,
            count_limit_max, label_count, _reserved,
        ) = fields
        (handler_loc, handler_code, label_object_type,
         label_object_loc, label_object_code) = extension
        out.append(
            RegisteredEffect(
                id=eid,
                code=code,
                type=etype,
                owner_code=owner_code,
                # card::get_info_location packing, same as MSG_CHAINING
                owner_controler=owner_loc & 0xFF,
                owner_location=(owner_loc >> 8) & 0xFF,
                owner_sequence=(owner_loc >> 16) & 0xFF,
                owner_position=(owner_loc >> 24) & 0xFF,
                description=description,
                reset_flag=reset_flag,
                reset_count=reset_count,
                count_code=count_code,
                active_type=active_type,
                flag=flag0,
                flag2=flag1,
                s_range=s_range,
                o_range=o_range,
                range=erange,
                status=status,
                container=container,
                effect_owner=effect_owner,
                target_player=target_player,
                count_limit=count_limit,
                count_limit_max=count_limit_max,
                label_count=label_count,
                handler_controler=handler_loc & 0xFF,
                handler_location=(handler_loc >> 8) & 0xFF,
                handler_sequence=(handler_loc >> 16) & 0xFF,
                handler_position=(handler_loc >> 24) & 0xFF,
                handler_code=handler_code,
                label_object_type=label_object_type,
                label_object_controler=label_object_loc & 0xFF,
                label_object_location=(label_object_loc >> 8) & 0xFF,
                label_object_sequence=(label_object_loc >> 16) & 0xFF,
                label_object_position=(label_object_loc >> 24) & 0xFF,
                label_object_code=label_object_code,
            )
        )
    return tuple(out)


def _parse_records(payload: bytes, layout: struct.Struct, build):
    count, record_len = struct.unpack_from("<HH", payload, 0)
    if record_len < layout.size:
        raise EffectInfoError(
            f"record is {record_len} bytes, reader needs {layout.size}"
        )
    out = []
    pos = 4
    for _ in range(count):
        out.append(build(*layout.unpack_from(payload, pos)))
        pos += record_len
    return tuple(out)


def _parse_turn(payload: bytes) -> TurnCounters:
    (
        turn_id, phase, turn_player, current_player, duel_rule, deck_reversed,
        remove_brainwashing, _pad, turn0, turn1,
    ) = _TURN_HEAD.unpack_from(payload, 0)
    per = []
    pos = _TURN_HEAD.size
    for _ in range(2):
        per.append(_TURN_PLAYER.unpack_from(payload, pos))
        pos += _TURN_PLAYER.size
    columns = tuple(zip(*per))
    return TurnCounters(
        turn_id=turn_id,
        phase=phase,
        turn_player=turn_player,
        current_player=current_player,
        duel_rule=duel_rule,
        deck_reversed=deck_reversed,
        remove_brainwashing=remove_brainwashing,
        turn_id_by_player=(turn0, turn1),
        summon_count=columns[0],
        extra_summon=columns[1],
        summon_state_count=columns[2],
        normalsummon_state_count=columns[3],
        flipsummon_state_count=columns[4],
        spsummon_state_count=columns[5],
        attack_state_count=columns[6],
        battle_phase_count=columns[7],
        battled_count=columns[8],
    )


def parse(raw: bytes) -> EffectInfo:
    """Turn a raw dump into an :class:`EffectInfo`."""
    if len(raw) < HEADER.size:
        raise EffectInfoError(f"dump is {len(raw)} bytes, shorter than its header")
    version, header_len, _reserved, total = HEADER.unpack_from(raw, 0)
    if version != VERSION:
        raise EffectInfoError(f"dump version {version}, reader speaks {VERSION}")
    if total > len(raw):
        raise EffectInfoError(f"dump claims {total} bytes, got {len(raw)}")
    raw = raw[:total]
    pos = header_len
    parsed: dict = {}
    unknown: dict[int, bytes] = {}
    while pos + SECTION.size <= len(raw):
        section_id, length = SECTION.unpack_from(raw, pos)
        pos += SECTION.size
        if section_id == SECTION_END:
            break
        payload = raw[pos : pos + length]
        if len(payload) != length:
            raise EffectInfoError(f"section {section_id} truncated")
        pos += length
        if section_id == SECTION_EFFECTS:
            parsed["effects"] = _parse_effects(payload)
        elif section_id == SECTION_COUNT_CODE:
            parsed["count_codes"] = _parse_records(
                payload, _COUNT,
                lambda table, player, code, value: CountCode(table, player, code, value),
            )
        elif section_id == SECTION_TURN:
            parsed["turn"] = _parse_turn(payload)
        elif section_id == SECTION_SPSUMMON_ONCE:
            parsed["spsummon_once"] = _parse_records(
                payload, _SPONCE,
                lambda player, code, count: SpSummonOnce(player, code, count),
            )
        elif section_id == SECTION_ACTIVITY:
            parsed["activity"] = _parse_records(
                payload, _ACTIVITY,
                lambda kind, cid, c0, c1: ActivityCount(kind, cid, (c0, c1)),
            )
        elif section_id == SECTION_PLAYER_STATE:
            parsed["players"] = _parse_records(
                payload, _PLAYER,
                lambda player, used, disabled, extra, szone:
                    PlayerState(player, used, disabled, extra, szone),
            )
        elif section_id == SECTION_CARD_STATE:
            parsed["cards"] = _parse_records(
                payload, _CARD_STATE,
                lambda *values: CardStateInfo(*values),
            )
        elif section_id == SECTION_RELATIONS:
            parsed["relations"] = _parse_records(
                payload, _RELATION,
                lambda kind, source_info, source_code, target_info, target_code,
                       reset, _reserved: RelationInfo(
                           kind, source_info, source_code, target_info,
                           target_code, reset,
                       ),
            )
        elif section_id == SECTION_SETCODE:
            parsed["setcodes"] = _parse_records(
                payload, _SETCODE,
                lambda info, code, setcode: SetCodeInfo(info, code, setcode),
            )
        else:
            unknown[section_id] = payload
    return EffectInfo(version=version, length=total, unknown_sections=unknown, **parsed)


# -- the ctypes side -------------------------------------------------------


def declare(lib: ctypes.CDLL) -> bool:
    """Give ``lib.query_effect_info`` its signature; False if it has none."""
    try:
        fn = lib.query_effect_info
    except AttributeError:
        return False
    fn.restype = ctypes.c_int32
    fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]
    return True


def supports(core) -> bool:
    """Does this :class:`~mirrorforce.puzzle.core.Core` carry the patch?"""
    return declare(core._lib)


def query_effect_info(core, pduel, buf=None) -> EffectInfo:
    """Dump and parse the effect state of a live duel.

    ``core`` is a :class:`~mirrorforce.puzzle.core.Core` whose shared object was
    built from the patch; ``pduel`` is the handle ``create_duel`` returned.
    Pass ``buf`` (a ``ctypes`` string buffer) to reuse one allocation across a
    walk -- the dump is taken at every decision point, so it is worth it.
    """
    if not declare(core._lib):
        raise EffectInfoError(
            f"{core.lib_path} has no query_effect_info; build it with "
            "mirrorforce/build/effectinfo/build-effectinfo-core.sh"
        )
    if buf is None:
        buf = ctypes.create_string_buffer(QUERY_BUFFER_SIZE)
    length = core._lib.query_effect_info(pduel, buf, len(buf))
    if length <= 0:
        raise EffectInfoError(
            f"query_effect_info returned {length} "
            f"(buffer of {len(buf)} bytes too small?)"
        )
    return parse(buf.raw[:length])


_CORE = None


def get_effectinfo_core(**kwargs):
    """A :class:`~mirrorforce.puzzle.core.Core` over the patched shared object.

    Deliberately *not* ``puzzle.core.get_core``: this is a second core in the
    same process, so nothing that already holds the package singleton has its
    core swapped out from under it.

    Two cores can only coexist because of how this one is built and loaded.
    Both objects define the same ``ocgapi`` names, and a shared library's calls
    to its own global functions go through the PLT, so the object that reached
    the global namespace first answers the other one's ``read_script`` /
    ``read_card``.  That failure is silent -- the duel runs, on somebody else's
    card reader -- and it was reproduced here before the fix.  Two things close
    it: the build hides every symbol except the ``ocgapi`` entry points, and
    this loads with ``RTLD_LOCAL`` so even those stay out of the global
    namespace.  Both are required; keep them together.

    The reader callbacks are still per-object statics, so there is exactly one
    of these per process, cached here.
    """
    global _CORE
    if _CORE is not None:
        return _CORE
    from .puzzle.core import Core, DEFAULT_SCRIPTS

    kwargs.setdefault("lib_path", DEFAULT_EFFECTINFO_LIB)
    kwargs.setdefault("mode", ctypes.RTLD_LOCAL)
    if "script_dirs" not in kwargs:
        overrides = Path(os.environ.get(
            "MF_SCRIPT_OVERRIDES", str(DEFAULT_SCRIPT_OVERRIDES)
        ))
        kwargs["script_dirs"] = (
            [overrides, DEFAULT_SCRIPTS]
            if script_override_provenance(overrides) else [DEFAULT_SCRIPTS]
        )
    if not Path(kwargs["lib_path"]).is_file():
        raise EffectInfoError(
            f"patched core not built: {kwargs['lib_path']}; run "
            "mirrorforce/build/effectinfo/build-effectinfo-core.sh"
        )
    core = Core(**kwargs)
    if not supports(core):
        raise EffectInfoError(
            f"{core.lib_path} has no query_effect_info; rebuild it with "
            "mirrorforce/build/effectinfo/build-effectinfo-core.sh"
        )
    _CORE = core
    return core
