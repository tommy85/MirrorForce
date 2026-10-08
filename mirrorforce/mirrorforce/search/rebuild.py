"""Rebuild a mid-game state from "public information + sampled hidden cards".

**This is the only legal particle source for policy-side search.** Deployment rule: search must never fork the real duel
process, which holds the opponent's real hand, set cards and deck order; forking it feeds illegal information into search.
Particles must be built from scratch as here: the public part is copied, the hidden part is sampled by
:mod:`mirrorforce.search.belief`.

Construction channel
--------

``ocgapi`` has no ``load_state``. The only construction entry is ``preload_script``: compile a piece of Lua
calling ``Debug.*`` into the duel's Lua state (``preload_script`` -> ``lua->load_script`` in ``ocgapi.cpp``),
which is exactly how ygopro loads single-player puzzles
(``SinglePlayThread`` in ``single_mode.cpp``). This module compiles a structured
:class:`BoardSpec` into that Lua.

Of the full Debug library (``ygopro-core/libdebug.cpp``, 11 entries) the 8 related to construction:
``ReloadFieldBegin`` / ``SetPlayerInfo`` / ``AddCard`` / ``PreSummon`` /
``PreEquip`` / ``PreSetTarget`` / ``PreAddCounter`` / ``ReloadFieldEnd``.

**Constructible**: LP, the cards of the seven zones and their positions (location, face-up/down), counters, equip relations,
effect target relations, Xyz materials, ``STATUS_PROC_COMPLETE``, ``summon_info`` (summon type
and source zone), deck order (``SEQ_DECKTOP`` push_back card by card), hands, each player's
``draw_count``, duel rules and options.

**Not constructible** (no entry in ``libdebug.cpp`` and no setter on the C side, measured):


* ``infos.turn_id`` / ``infos.phase`` / ``infos.turn_player``: a rebuilt duel always starts at
  turn 1, first player, draw phase. This module works around the turn player by **swapping seats**
  (mapping the real turn player to seat 0) and the phase with a one-off field effect skipping DP/SP,
  but ``turn_id`` itself cannot be changed.
* the three tables ``effect_count_code`` / ``_duel`` / ``_chain`` and ``spsummon_once_map``,
  i.e. "once per turn" bookkeeping. **This is the only hard gap of this line.**
* the activity counters in ``processor`` (``summon_count`` / ``attack_state_count`` /
  ``battled_count`` / ``battle_phase_count`` ...).
* state in the middle of a chain: search only expands at empty-chain decision points, so this one is not needed
  by design and is not a gap.

Lossy items are recorded as is in the ``lost`` field of :class:`BoardSpec`, for a key-by-key audit of the
rebuild.

Seat swapping
--------

Turn 1 of a rebuilt duel always belongs to seat 0. So ``describe`` maps the **real turn player** to seat 0 by default
and the other player to seat 1. Seat numbers in a rebuilt duel are names we choose anyway;
swapping changes nothing about the game, while not swapping would introduce the fatal error of a wrong turn player.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass, field

from ..netduel import constants as C
from ..netduel.board import parse_query_segments
from ..puzzle.core import SIZE_QUERY_BUFFER
from ..worldmodel.engine import DuelConfig, DuelDriver

__all__ = [
    "BoardSpec",
    "OPPONENT_HIDDEN_LOCATIONS",
    "Placement",
    "REBUILD_LOCATIONS",
    "RebuildError",
    "RebuiltDuel",
    "attach_engine_state",
    "attach_summon_info",
    "describe",
    "emit_lua",
    "fill_hidden",
    "hidden_slots",
    "leak_check",
    "rebuild",
]

#: Everything ``card::get_infos`` can write, except ``QUERY_REASON_CARD`` (a pointer,
#: meaningless outside the core). The same value as ``FULL_QUERY_FLAGS`` of ``puzzle/single.py``.
FULL_QUERY_FLAGS = 0xFFDFFF

#: Order of reading zones. The field first, then hand and stores: a host on the field must exist before its Xyz materials.
REBUILD_LOCATIONS = (
    C.LOCATION_MZONE,
    C.LOCATION_SZONE,
    C.LOCATION_HAND,
    C.LOCATION_GRAVE,
    C.LOCATION_REMOVED,
    C.LOCATION_EXTRA,
    C.LOCATION_DECK,
)

#: ``common.h``: ``field::add_card`` treats sequence as a mode for ``LOCATION_DECK``
SEQ_DECKTOP = 0
SEQ_DECKBOTTOM = 1
SEQ_DECKSHUFFLE = 2

#: ``common.h:196-227``. ``QUERY_STATUS`` writes only the three bits ``STATUS_DISABLED |
#: STATUS_FORBIDDEN | STATUS_PROC_COMPLETE`` (``card.cpp:333``);
#: the other 29 bits cannot be read and no Debug entry writes them; the 7th argument of ``Debug.AddCard``,
#: ``proc``, only handles the bit ``STATUS_PROC_COMPLETE``.
STATUS_DISABLED = 0x0001
STATUS_PROC_COMPLETE = 0x0008
STATUS_FORBIDDEN = 0x4000000
QUERY_STATUS_MASK = STATUS_DISABLED | STATUS_FORBIDDEN | STATUS_PROC_COMPLETE

#: ``card.h:424-428``: the bit fields of ``summon_info``. Main type 0xf0000000, subtype
#: 0x0f000000, **source zone 0x00ff0000**, custom type 0x0000ffff.
#: ``Debug.PreSummon(card, T, L)`` writes ``T | (L << 16)``, so to restore a
#: specific ``summon_info``, T must be ``info & DEFAULT_SUMMON_TYPE`` (cutting out
#: the 8 zone bits in the middle) and L ``(info >> 16) & 0xff``. The intuitive split of the low 16 bits as type and the high 16 bits
#: as zone is wrong: the main type is not in the low 16 bits at all.
SUMMON_VALUE_MAIN_TYPE = 0xF0000000
SUMMON_VALUE_SUB_TYPE = 0x0F000000
SUMMON_VALUE_CUSTOM_TYPE = 0x0000FFFF
DEFAULT_SUMMON_TYPE = (
    SUMMON_VALUE_MAIN_TYPE | SUMMON_VALUE_SUB_TYPE | SUMMON_VALUE_CUSTOM_TYPE
)

#: ``processor.cpp:2081`` / ``field.cpp:3662``: without this bit, "turn 1" of a rebuilt duel
#: may not enter the Battle Phase, while it may stand for turn 7. Puzzles always carry this bit, for the same reason.
DUEL_ATTACK_FIRST_TURN = 0x02

LOCATION_NAMES = {
    C.LOCATION_DECK: "deck", C.LOCATION_HAND: "hand", C.LOCATION_MZONE: "mzone",
    C.LOCATION_SZONE: "szone", C.LOCATION_GRAVE: "grave",
    C.LOCATION_REMOVED: "removed", C.LOCATION_EXTRA: "extra",
}


class RebuildError(RuntimeError):
    """Rebuilding the state failed."""


# -- structured board --------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    """Where a card goes in the rebuilt duel and what it looks like."""

    code: int
    owner: int
    controller: int
    location: int
    sequence: int
    position: int
    #: ``STATUS_PROC_COMPLETE``: this monster was "properly summoned".
    #: Without it, the legality of revival effects on this card changes: an error that changes the legal set.
    proc: bool = False
    #: codes of the Xyz materials, in the engine's order
    overlay: tuple[int, ...] = ()
    #: ``{counter type: count}``
    counters: tuple[tuple[int, int], ...] = ()
    #: equip target ``(controller, zone, sequence)``; ``None`` means this is not an equip card
    equip_to: tuple[int, int, int] | None = None
    #: effect targets ``[(controller, zone, sequence), ...]``
    targets: tuple[tuple[int, int, int], ...] = ()
    #: ``card::summon_info`` split into the two arguments of ``Debug.PreSummon``
    summon_type: int = 0
    summon_location: int = 0
    #: The following six have no construction entry and can only be set by ``Debug.SetCardState`` after the duel starts.
    #: ``None`` means "this one is unknown, leave it alone".
    status: int | None = None
    turnid: int | None = None
    turn_counter: int | None = None
    summon_player: int | None = None
    attacked_count: int | None = None
    announce_count: int | None = None
    attack_announce_count: int | None = None

    @property
    def location_name(self) -> str:
        return LOCATION_NAMES.get(self.location, hex(self.location))

    def key(self) -> tuple:
        return (self.controller, self.location, self.sequence)


@dataclass
class History:
    """The engine's turn-level history state, exactly covered by the six setters of :mod:`Debug`.

    Seats are already mapped by ``BoardSpec.seat_map``.
    """

    turn_id: int = 1
    turn_id_by_player: tuple[int, int] = (1, 0)
    #: ``[(table, player, code, value)]``; table 0=per turn 1=per duel 2=per chain
    count_codes: tuple[tuple[int, int, int, int], ...] = ()
    #: ``[(player, code, count)]``
    spsummon_once: tuple[tuple[int, int, int], ...] = ()
    #: ``[(kind, counter_id, count0, count1)]``
    activity: tuple[tuple[int, int, int, int], ...] = ()
    #: one per player ``(summon_count, extra_summon, summon_state, normalsummon_state,
    #: flipsummon_state, spsummon_state, attack_state, battle_phase, battled)``
    turn_counters: tuple[tuple[int, ...], tuple[int, ...]] = ((0,) * 9, (0,) * 9)


@dataclass
class BoardSpec:
    """A constructible mid-game state, plus a detailed list of "what cannot be constructed"."""

    lp: tuple[int, int]
    duel_rule: int = 5
    duel_options: int = 0
    draw_count: tuple[int, int] = (1, 1)
    placements: tuple[Placement, ...] = ()
    #: ``seat_map[real seat] = rebuilt seat``. By default the real turn player goes to seat 0.
    seat_map: tuple[int, int] = (0, 1)
    #: the target phase of the rebuild; ``build`` walks the rebuilt duel to it
    target_phase: int = C.PHASE_MAIN1
    #: the engine's "history state": once-per-turn bookkeeping, once-per-duel special summons, activity counters, turn count.
    #: These are cleared by ``process_turn`` the moment the duel starts, so they cannot be set in the construction script;
    #: :meth:`RebuiltDuel.build` sets them after the first ``process()``.
    #: ``None`` = do not set them (the old calibration, for comparison).
    history: "History | None" = None
    #: items that cannot be constructed and are only recorded. The audit compares exactly this, key by key.
    lost: dict = field(default_factory=dict)
    #: optional: the emission order of ``Debug.AddCard``, listed by ``Placement.key()``. The engine's effect
    #: containers are ordered by when cards entered their current zone (``field::add_card`` ->
    #: ``card::apply_field_effect``), and that is the order of the "can activate" rows of a menu; given this
    #: list, :func:`emit_lua` orders by it first, and cards not listed follow in the old zone order. Empty means
    #: the old calibration (zone order).
    placement_order: tuple[tuple[int, int, int], ...] = ()

    def to_json(self) -> dict:
        out = asdict(self)
        out["placements"] = [asdict(p) for p in self.placements]
        return out


# -- reading the board --------------------------------------------------------------------


def _decode_ref(value: int) -> tuple[int, int, int]:
    """The low three bytes of ``card::get_info_location``: controller / zone / sequence."""
    return (value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF)


def describe(driver: DuelDriver, *, effect_info=None, seat_map=None,
             target_phase: int | None = None) -> BoardSpec:
    """Read a live duel into a :class:`BoardSpec`.

    It reads from the **omniscient** view (the in-process core's ``query_field_card`` has no audience filter).
    Online search does not use it this way: there the opponent's hand and deck in ``placements`` are filled by
    :mod:`mirrorforce.search.belief`. This function exists to take the "omniscient rebuild" as the **upper bound**
    of fidelity: what cannot be rebuilt even omnisciently cannot be recovered by sampling either.
    """
    core, pduel = driver.core, driver.pduel
    buf = ctypes.create_string_buffer(SIZE_QUERY_BUFFER)

    turn_player = driver.turn_player
    if seat_map is None:
        # Real turn player -> seat 0. Turn 1 of a rebuilt duel always belongs to seat 0.
        seat_map = (0, 1) if turn_player == 0 else (1, 0)
    seat = tuple(seat_map)

    placements: list[Placement] = []
    for player in (0, 1):
        for location in REBUILD_LOCATIONS:
            length = core.query_field_card(pduel, player, location,
                                           FULL_QUERY_FLAGS, buf)
            if length <= 0:
                continue
            for sequence, fields in enumerate(parse_query_segments(buf.raw[:length])):
                if not fields or not fields.get("code"):
                    continue
                info = fields.get("info_location", 0)
                position = (info >> 24) & 0x0F
                status = fields.get("status", 0)
                equip = fields.get("equip_card")
                equip_to = None
                if equip:
                    ec, el, es = _decode_ref(equip)
                    equip_to = (seat[ec], el, es)
                targets = tuple(
                    (seat[c], l, s)
                    for c, l, s in (_decode_ref(v) for v in fields.get("targets", ()))
                )
                placements.append(Placement(
                    code=int(fields["code"]),
                    owner=seat[int(fields.get("owner", player))],
                    controller=seat[player],
                    location=location,
                    sequence=sequence,
                    position=position or C.POS_FACEDOWN_DEFENSE,
                    proc=bool(status & STATUS_PROC_COMPLETE),
                    overlay=tuple(int(x) for x in fields.get("overlay", ())),
                    counters=tuple(sorted(
                        (int(k), int(v))
                        for k, v in fields.get("counters", {}).items())),
                    equip_to=equip_to,
                    targets=targets,
                ))

    length = core.query_field_info(pduel, buf)
    raw = buf.raw[:length]
    lp = _field_lp(raw)

    spec = BoardSpec(
        lp=(lp[seat.index(0)], lp[seat.index(1)]),
        duel_rule=driver.config.duel_options >> 16 or 5,
        duel_options=(driver.config.duel_options & 0xFFFF) | DUEL_ATTACK_FIRST_TURN,
        draw_count=(driver.config.draw_count, driver.config.draw_count),
        placements=tuple(placements),
        seat_map=seat,
        target_phase=target_phase if target_phase is not None else driver.phase,
    )

    # lossy items: only recorded, not constructible
    spec.lost = {
        "turn_id": driver.turn,
        "phase": driver.phase,
        "turn_player_real": turn_player,
        "note": "turn_id/phase/turn_player have no Debug entry; turn_player is worked around by swapping seats, "
                "phase by a one-off skip of DP/SP; turn_id cannot be worked around",
    }
    if effect_info is not None:
        spec.lost["summon_info_source"] = "query_duel_state"
        spec.lost["count_codes"] = [c.as_dict() for c in effect_info.count_codes]
        spec.lost["spsummon_once"] = [s.as_dict() for s in effect_info.spsummon_once]
        spec.lost["activity"] = [a.as_dict() for a in effect_info.activity]
        if effect_info.turn is not None:
            spec.lost["turn_counters"] = effect_info.turn.as_dict()
    return spec


def _field_lp(raw: bytes) -> tuple[int, int]:
    """``query_field_info`` writes a ``MSG_RELOAD_FIELD`` body; its first two ints are the LP."""
    import struct

    pos = 2  # skip the MSG id and duel_rule
    out = []
    for _ in range(2):
        (value,) = struct.unpack_from("<i", raw, pos)
        out.append(value)
        pos += 4
        for _ in range(7):  # monster zones
            pos += 3 if raw[pos] else 1
        for _ in range(8):  # spell & trap zones
            pos += 2 if raw[pos] else 1
        pos += 6  # card counts of the zones
    return (out[0], out[1])


def attach_engine_state(spec: BoardSpec, duel_state, effect_info=None) -> BoardSpec:
    """Complete ``Placement`` and :class:`History` with the extended state API.

    The old ``QUERY_*`` protocol cannot give these: ``summon_info`` decides "how this monster came to the field",
    the 29 bits of ``status`` hold "summoned / set this turn / cannot change position / negated",
    and ``turnid`` is "in which turn this arrival happened". The legality of revival, material, position-change and
    banish-recovery effects reads them, and missing them is an error that changes the legal set.

    Only a core with the ``query_duel_state`` / ``query_effect_info`` patches can provide them.
    """
    seat = spec.seat_map
    by_key: dict[tuple, object] = {}
    for card in duel_state.cards:
        if card.overlay or not card.current_code:
            continue
        by_key[(seat[card.controller], card.location, card.sequence)] = card
    out = []
    for placement in spec.placements:
        card = by_key.get(placement.key())
        info = card.summon_info if card is not None else 0
        fields = dict(asdict(placement))
        fields.update(
            overlay=placement.overlay,
            counters=placement.counters,
            targets=placement.targets,
            summon_type=info & DEFAULT_SUMMON_TYPE,
            summon_location=(info >> 16) & 0xFF,
        )
        if card is not None:
            fields.update(
                status=card.status,
                turnid=card.turn_id,
                turn_counter=card.turn_counter,
                summon_player=(seat[card.summon_player]
                               if card.summon_player in (0, 1)
                               else card.summon_player),
                attacked_count=card.attacked_count,
                announce_count=card.announce_count,
                attack_announce_count=card.attack_announce_count,
            )
        out.append(Placement(**fields))
    spec.placements = tuple(out)
    if effect_info is not None:
        spec.history = _history_from(effect_info, seat)
    return spec


#: the old name, still used by older call sites
attach_summon_info = attach_engine_state


def _history_from(effect_info, seat) -> History:
    """Translate the four sections of ``query_effect_info`` into :class:`History`, swapping seats."""
    turn = effect_info.turn
    count_codes = tuple(
        # PLAYER_NONE(2) is not swapped; 0/1 are swapped by seat_map
        (row.table, seat[row.player] if row.player in (0, 1) else row.player,
         row.code, row.value)
        for row in effect_info.count_codes
    )
    spsummon_once = tuple(
        (seat[row.player], row.code, row.count)
        for row in effect_info.spsummon_once if row.player in (0, 1)
    )
    activity = tuple(
        # count is a per-player pair; swapping seats swaps the two numbers
        (row.kind, row.counter_id,
         row.count[seat.index(0)], row.count[seat.index(1)])
        for row in effect_info.activity
    )
    if turn is None:
        return History(count_codes=count_codes, spsummon_once=spsummon_once,
                       activity=activity)
    def per_player(rebuilt_seat: int) -> tuple[int, ...]:
        real = seat.index(rebuilt_seat)
        return (
            turn.summon_count[real], turn.extra_summon[real],
            turn.summon_state_count[real], turn.normalsummon_state_count[real],
            turn.flipsummon_state_count[real], turn.spsummon_state_count[real],
            turn.attack_state_count[real], turn.battle_phase_count[real],
            turn.battled_count[real],
        )
    return History(
        turn_id=turn.turn_id,
        turn_id_by_player=(turn.turn_id_by_player[seat.index(0)],
                           turn.turn_id_by_player[seat.index(1)]),
        count_codes=count_codes,
        spsummon_once=spsummon_once,
        activity=activity,
        turn_counters=(per_player(0), per_player(1)),
    )


# -- generating Lua ----------------------------------------------------------------

#: Skip the draw and standby phases only in this one rebuilt turn.
#:
#: Puzzles do the same with ``aux.BeginPuzzle()``, but it also registers an effect "set your own LP to 0 at the end of
#: the turn", which would make a rebuilt duel lose at the end of its first turn and rule out continuing.
#: Only the two effects ``EFFECT_SKIP_DP`` / ``EFFECT_SKIP_SP`` are taken here, with
#: ``RESET_PHASE+PHASE_END``, so normal draws resume from turn 2.
#:
#: Why skipping is required: turn 1 of a rebuilt duel really runs DP and SP and raises
#: ``EVENT_PHASE_START+PHASE_DRAW/STANDBY`` again. If the rebuilt position is already
#: past SP, this run is **extra**: "during your Standby Phase" maintenance effects would trigger twice.
#: Under MR5 turn 1 draws nothing anyway (``processor.cpp``: ``duel_rule<=2 || turn_id>1``),
#: so skipping DP does not change the number of draws.
_SKIP_PHASES_LUA = """\
local _e_dp=Effect.GlobalEffect()
_e_dp:SetType(EFFECT_TYPE_FIELD)
_e_dp:SetProperty(EFFECT_FLAG_PLAYER_TARGET)
_e_dp:SetCode(EFFECT_SKIP_DP)
_e_dp:SetTargetRange(1,0)
_e_dp:SetReset(RESET_PHASE+PHASE_END)
Duel.RegisterEffect(_e_dp,0)
local _e_sp=Effect.GlobalEffect()
_e_sp:SetType(EFFECT_TYPE_FIELD)
_e_sp:SetProperty(EFFECT_FLAG_PLAYER_TARGET)
_e_sp:SetCode(EFFECT_SKIP_SP)
_e_sp:SetTargetRange(1,0)
_e_sp:SetReset(RESET_PHASE+PHASE_END)
Duel.RegisterEffect(_e_sp,0)
"""


def _deck_sequence_arg(location: int, sequence: int) -> int:
    """How ``field::add_card`` reads its 5th argument for each zone.

    The deck is a vector, and ``sequence`` is a **mode**, not a slot: 0=top (push_back),
    1=bottom (insert at the front), 2=shuffle in (sets ``shuffle_deck_check``, and the order is gone).
    Hand / graveyard / banished / Extra Deck always push_back, whatever is passed; pass 0.
    Only the field (monster zones / spell & trap zones) uses real slots.
    """
    if location in (C.LOCATION_MZONE, C.LOCATION_SZONE):
        return sequence
    return SEQ_DECKTOP


def emit_lua(spec: BoardSpec) -> str:
    """Compile a :class:`BoardSpec` into the Lua that ``preload_script`` takes."""
    lines = [
        "-- MirrorForce search particle: generated by mirrorforce.search.rebuild",
        f"Debug.ReloadFieldBegin({spec.duel_options},{spec.duel_rule})",
        f"Debug.SetPlayerInfo(0,{spec.lp[0]},0,{spec.draw_count[0]})",
        f"Debug.SetPlayerInfo(1,{spec.lp[1]},0,{spec.draw_count[1]})",
        _SKIP_PHASES_LUA.rstrip(),
    ]

    # Ordering: the field before other zones; within a zone by sequence. The deck must be pushed back card by card
    # **from bottom to top**, and query gives the index order of list_main (the top is back), so ascending order works.
    # When ``spec.placement_order`` gives the arrival order, order by it first: in vector zones arrival order runs the same way as sequence,
    # so the pushed-back sequences do not get mixed up.
    order = {loc: i for i, loc in enumerate(REBUILD_LOCATIONS)}
    rank = {key: index for index, key in enumerate(spec.placement_order)}
    placements = sorted(
        spec.placements,
        key=lambda p: (rank.get(p.key(), len(rank)),
                       order.get(p.location, 99), p.controller, p.sequence),
    )

    names: dict[tuple, str] = {}
    for index, p in enumerate(placements):
        name = f"c{index}"
        names[p.key()] = name
        seq = _deck_sequence_arg(p.location, p.sequence)
        proc = ",true" if p.proc else ""
        lines.append(
            f"local {name}=Debug.AddCard({p.code},{p.owner},{p.controller},"
            f"{p.location},{seq},{p.position}{proc})"
        )
        # Xyz material: the host is already in this slot; another AddCard on the same slot is
        # caught by the is_location_useable failure branch of libdebug.cpp and attached with xyz_add.
        for code in p.overlay:
            lines.append(
                f"Debug.AddCard({code},{p.owner},{p.controller},"
                f"{C.LOCATION_MZONE},{p.sequence},{C.POS_FACEUP_ATTACK},true)"
            )

    for p in placements:
        name = names[p.key()]
        if p.summon_type:
            lines.append(
                f"Debug.PreSummon({name},{p.summon_type},{p.summon_location})")
        for ctype, count in p.counters:
            lines.append(f"Debug.PreAddCounter({name},{ctype},{count})")

    for p in placements:
        name = names[p.key()]
        if p.equip_to is not None:
            target = names.get(p.equip_to)
            if target:
                lines.append(f"Debug.PreEquip({name},{target})")
        for ref in p.targets:
            target = names.get(ref)
            if target:
                lines.append(f"Debug.PreSetTarget({name},{target})")

    lines.append("Debug.ReloadFieldEnd()")
    return "\n".join(lines) + "\n"


def emit_history_lua(spec: BoardSpec) -> str:
    """The second script: put the engine's turn-level history state back.

    **It must run after the duel starts and before the first decision point.** Step 0 of ``field::process_turn``
    clears, at the start of every turn, everything set here: the three once-per-turn tables,
    ``spsummon_once_map``, the activity counters, and status bits of each field card such as
    ``SUMMON_TURN`` / ``SET_TURN`` / ``CANNOT_CHANGE_FORM``; and
    ``start_duel`` ends by queuing ``PROCESSOR_TURN``. So values written in the construction script are erased
    in the first ``process()``, every one of them.

    The window is **after the first ``process()`` returns**: its ``MSG_NEW_TURN`` shows the clearing
    is done, while the first menu only appears after the third ``process()``.
    :meth:`RebuiltDuel.build` calls ``preload_script`` again in exactly that window.

    Cards are located by ``(controller, zone, sequence)`` (``Duel.GetFieldCard`` covers all seven zones),
    because the Lua locals of the first script are out of scope by then.
    """
    history = spec.history
    if history is None:
        return ""
    lines = [
        "-- MirrorForce: engine history state restored after the duel starts (see rebuild.emit_history_lua)",
        f"Debug.SetTurnInfo({history.turn_id},{history.turn_id_by_player[0]},"
        f"{history.turn_id_by_player[1]})",
    ]
    for player, counters in enumerate(history.turn_counters):
        lines.append(f"Debug.SetTurnCounters({player},"
                     + ",".join(str(int(v)) for v in counters) + ")")
    for table, player, code, value in history.count_codes:
        lines.append(f"Debug.SetCountCode({table},{player},{code},{value})")
    for player, code, count in history.spsummon_once:
        lines.append(f"Debug.SetSpSummonOnce({player},{code},{count})")
    for kind, counter_id, count0, count1 in history.activity:
        lines.append(
            f"Debug.SetActivityCount({kind},{counter_id},{count0},{count1})")

    seen: set[tuple] = set()
    for index, p in enumerate(spec.placements):
        if p.status is None or p.key() in seen:
            continue
        seen.add(p.key())
        args = ",".join(str(int(v if v is not None else 0)) for v in (
            p.status, p.turnid, p.turn_counter, p.summon_player,
            p.attacked_count, p.announce_count, p.attack_announce_count))
        name = f"h{index}"
        lines.append(
            f"local {name}=Duel.GetFieldCard({p.controller},{p.location},"
            f"{p.sequence}) if {name} then Debug.SetCardState({name},{args}) end")
    return "\n".join(lines) + "\n"


# -- building a live rebuilt duel ------------------------------------------------------


class RebuiltDuel(DuelDriver):
    """A duel built by ``preload_script`` instead of ``new_card``.

    Apart from :meth:`build`, everything (the answer loop, the disclosure ledger, ``fork``) is inherited from
    :class:`~mirrorforce.worldmodel.engine.DuelDriver`, so rebuilt and real duels use the same action layer
    and the same message parsing, and their legal menus can be compared row by row.
    """

    #: the generated Lua goes through the script cache and is not written to disk; the name must be fixed for cache hits
    SCRIPT_NAME = "./script/mirrorforce-r3-particle.lua"
    HISTORY_SCRIPT_NAME = "./script/mirrorforce-r3-history.lua"

    def __init__(self, spec: BoardSpec, config: DuelConfig, core=None,
                 script_name: str | None = None):
        super().__init__(config, core)
        self.spec = spec
        self.lua = emit_lua(spec)
        self.history_lua = emit_history_lua(spec)
        self.script_name = script_name or self.SCRIPT_NAME
        #: the batch of messages consumed by ``build()`` while restoring history state, for diagnostics
        self.history_messages: list[str] = []

    def build(self) -> "RebuiltDuel":
        core = self.core
        core.reset_session()
        import random

        seed_rng = random.Random(("seeds", self.config.seed).__repr__())
        seeds = [seed_rng.getrandbits(32) for _ in range(8)]
        self.pduel = core.create_duel(seeds)
        for player in (0, 1):
            core.set_player_info(self.pduel, player, self.spec.lp[player], 0,
                                 self.spec.draw_count[player])
        data = self.lua.encode("utf-8")
        core._script_cache[self.script_name] = (
            ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        if not core.preload_script(self.pduel, self.script_name):
            raise RebuildError(
                "preload_script refused the generated particle script; core log: "
                + " | ".join(core.log[:3])
            )
        if core.log:
            raise RebuildError("construction script error: " + core.log[0].replace("\n", " ")[:240])
        if core.missing_codes:
            raise RebuildError(f"cards missing from the cdb: {sorted(core.missing_codes)[:8]}")
        # Debug.ReloadFieldBegin already set rule and options; start_duel ORs them in,
        # so passing the same value is idempotent, and 0 would work too. Passing the same value keeps both paths explicitly equal.
        core.start_duel(self.pduel,
                        (self.spec.duel_rule << 16) | self.spec.duel_options)
        if self.history_lua:
            self._apply_history()
        return self

    def _apply_history(self) -> None:
        """Walk to the restore window, then compile the history-state script into the duel's Lua state."""
        self._apply_history_window()
        core = self.core
        data = self.history_lua.encode("utf-8")
        core._script_cache[self.HISTORY_SCRIPT_NAME] = (
            ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        if not core.preload_script(self.pduel, self.HISTORY_SCRIPT_NAME):
            raise RebuildError(
                "history-state script failed to compile; core log: " + " | ".join(core.log[:3]))
        if core.log:
            raise RebuildError(
                "history-state script error: " + core.log[0].replace("\n", " ")[:240])

    def _apply_history_window(self) -> None:
        """Advance the duel to the window "after the turn-start clearing, before the first menu".

        ``PROCESSOR_TURN``, queued at the end of ``start_duel``, runs step 0 of ``field::process_turn``,
        which clears once-per-turn bookkeeping, activity counters and the turn status bits of field cards,
        then writes ``MSG_NEW_TURN``; the core returns as soon as that is in the buffer. So the criterion of the window is
        **seeing ``MSG_NEW_TURN``**, not "having run ``process()`` once":
        with ``start_count > 0``, ``start_duel`` sends the opening draws first, and that batch of
        ``MSG_DRAW`` makes ``process()`` return early, before the clearing,
        where restoring would be wasted. A rebuilt duel sets ``start_count`` to 0, so measured it arrives
        on the first call; this still loops on the criterion so a changed opening hand size never fails silently.

        Reaching a prompt means the window assumption does not hold (there is no point to insert between the clearing
        and the menu); it is better to fail than to swallow the prompt.
        """
        from ..puzzle.core import PROCESSOR_BUFFER_LEN, PROCESSOR_END, PROCESSOR_FLAG
        from ..puzzle.messages import split_messages
        from ..puzzle.single import RESPONSE_REQUIRED

        core = self.core
        seen_new_turn = False
        for _ in range(64):
            raw = core.process(self.pduel)
            self.steps += 1
            if raw & PROCESSOR_BUFFER_LEN:
                length = core.get_message(self.pduel, self._msgbuf)
                for message in split_messages(self._msgbuf.raw[:length]):
                    self.history_messages.append(message.name)
                    if message.msg in RESPONSE_REQUIRED:
                        raise RebuildError(
                            f"a prompt {message.name} appeared in the history restore window: "
                            "there is no insertion point between the turn-start clearing and the first menu; refusing to continue")
                    self._observe(message)
                    if message.msg == C.MSG_NEW_TURN:
                        seen_new_turn = True
            if seen_new_turn:
                break
            if (raw & PROCESSOR_FLAG) == PROCESSOR_END:
                break
        if not seen_new_turn:
            raise RebuildError(
                "no MSG_NEW_TURN after the duel started; the restore window does not exist: "
                + ",".join(self.history_messages[:12]))

    def open_at(self, responder, *, max_steps: int = 20000,
                max_seconds: float = 30.0):
        """Walk to the first decision point of the rebuild's target phase.

        A rebuilt duel always starts at turn 1; ``_SKIP_PHASES_LUA`` skips DP/SP, so
        the default landing point is main phase 1. For main phase 2 or the Battle Phase, choose ``to_battle`` /
        ``to_m2`` in ``responder`` to walk there.
        """
        return self.run(responder, max_steps=max_steps, max_seconds=max_seconds)


def rebuild(spec: BoardSpec, *, seed: int, config: DuelConfig | None = None,
            core=None) -> RebuiltDuel:
    """Build a live rebuilt duel from a :class:`BoardSpec` and ``start_duel``.

    ``config`` only carries the seed and options: decks no longer go through ``new_card``, and the whole card order is
    in ``spec``. A config with empty decks is enough.
    """
    from ..worldmodel.engine import DeckList

    if config is None:
        empty = DeckList(name="particle", main=(), extra=())
        config = DuelConfig(
            decks=(empty, empty), seed=seed,
            start_lp=spec.lp[0], start_hand=0, draw_count=spec.draw_count[0],
            duel_options=(spec.duel_rule << 16) | spec.duel_options,
        )
    if core is None:
        from ..puzzle.core import get_core

        core = get_core()
    return RebuiltDuel(spec, config, core).build()


# -- belief particle -> constructible board ------------------------------------------------

#: Zones where the opponent's identities are invisible. The Extra Deck is not among them: in a closed environment the
#: list is known and the Extra Deck is not drawn from, so its order carries no information and "which cards remain" is public to both sides.
OPPONENT_HIDDEN_LOCATIONS = (C.LOCATION_HAND, C.LOCATION_DECK)


def hidden_slots(spec: BoardSpec, viewer_seat: int):
    """Slots of ``spec`` whose identity is invisible to ``viewer_seat``, grouped by zone.

    Three kinds: the opponent's hand, the opponent's main deck, and the opponent's set / face-down field cards. **Our own
    deck is included too**: we know which cards remain but not their order, so particles must reshuffle it, or search
    would plan around what we draw next.
    """
    opponent = 1 - viewer_seat
    out: dict[str, list[Placement]] = {
        "opp_hand": [], "opp_deck": [], "opp_facedown": [], "own_deck": [],
    }
    for p in spec.placements:
        if p.controller == viewer_seat:
            if p.location == C.LOCATION_DECK:
                out["own_deck"].append(p)
            continue
        if p.controller != opponent:
            continue
        if p.location == C.LOCATION_HAND:
            out["opp_hand"].append(p)
        elif p.location == C.LOCATION_DECK:
            out["opp_deck"].append(p)
        elif (p.location in (C.LOCATION_MZONE, C.LOCATION_SZONE)
              and not (p.position & C.POS_FACEUP)):
            out["opp_facedown"].append(p)
    for group in out.values():
        group.sort(key=lambda p: p.sequence)
    return out


def fill_hidden(spec: BoardSpec, *, viewer_seat: int, particle,
                disclosed_hand=(), rng=None) -> BoardSpec:
    """Replace the identities of ``spec`` invisible to ``viewer_seat`` with sampled ones from a particle.

    This is the **deployment-shaped** construction: in the incoming ``spec`` the opponent's hand / deck identities are the truth
    (because it was read omnisciently from a live duel); in the outgoing ``spec`` they have all been replaced by sampled
    values, except the cards in ``disclosed_hand`` (disclosure is input: what was public must never be
    anonymized again). When running online, the incoming ``spec`` holds only public information and
    this function does the same; offline an omniscient spec is used so that :func:`leak_check` can verify
    "whether any truth slipped through".

    ``disclosed_hand`` is a ``(code, count)`` sequence, filled at the front of the hand in **ascending sequence order**,
    the same convention as the canonical assignment of ``netduel.disclosure.resolve_disclosure``;
    both must agree, or the same ledger resolves to different instances in the two places.
    """
    rng = rng if rng is not None else __import__("random").Random()
    groups = hidden_slots(spec, viewer_seat)

    hand_codes = []
    for code, count in disclosed_hand:
        hand_codes.extend([int(code)] * int(count))
    hand_codes.extend(particle.hand)
    if len(hand_codes) != len(groups["opp_hand"]):
        raise RebuildError(
            f"the opponent's hand has {len(groups['opp_hand'])} slots, the particle gave {len(hand_codes)} cards")
    if len(particle.deck) != len(groups["opp_deck"]):
        raise RebuildError(
            f"the opponent's deck has {len(groups['opp_deck'])} slots, the particle gave {len(particle.deck)} cards")
    if len(particle.facedown) != len(groups["opp_facedown"]):
        raise RebuildError(
            f"the opponent's set cards have {len(groups['opp_facedown'])} slots, "
            f"the particle gave {len(particle.facedown)} cards")

    own_deck = [p.code for p in groups["own_deck"]]
    rng.shuffle(own_deck)

    replacement: dict[tuple, int] = {}
    for slot, code in zip(groups["opp_hand"], hand_codes):
        replacement[slot.key()] = int(code)
    for slot, code in zip(groups["opp_deck"], particle.deck):
        replacement[slot.key()] = int(code)
    for slot, code in zip(groups["opp_facedown"], particle.facedown):
        replacement[slot.key()] = int(code)
    for slot, code in zip(groups["own_deck"], own_deck):
        replacement[slot.key()] = int(code)

    placements = []
    for p in spec.placements:
        code = replacement.get(p.key())
        if code is None or code == p.code:
            placements.append(p)
            continue
        # For a slot whose identity changed, the items bound to the identity must be cleared with it: counters, equips,
        # effect targets and summon info describe the history of **that card**, which no longer holds for another card.
        placements.append(Placement(
            code=code, owner=p.owner, controller=p.controller,
            location=p.location, sequence=p.sequence, position=p.position,
            proc=p.proc, overlay=(), counters=(), equip_to=None, targets=(),
            summon_type=p.summon_type, summon_location=p.summon_location,
        ))
    return BoardSpec(
        lp=spec.lp, duel_rule=spec.duel_rule, duel_options=spec.duel_options,
        draw_count=spec.draw_count, placements=tuple(placements),
        seat_map=spec.seat_map, target_phase=spec.target_phase,
        lost=dict(spec.lost),
    )


def leak_check(truth: BoardSpec, particle_spec: BoardSpec, *, viewer_seat: int,
               disclosed_hand=()) -> dict:
    """How much truth "the observer should not know" is still left in a particle.

    The test compares **slot by slot**: each of the opponent's hidden slots where the particle's code equals the truth
    counts as a hit. A hit is not a leak: sampling hits the truth by chance, often when the pool is small.
    So the analytic baseline, the expected hit rate of a random guess, is given too. Only a measured hit rate significantly
    above the baseline means truth really slipped through.
    """
    from collections import Counter

    groups = hidden_slots(truth, viewer_seat)
    got = {p.key(): p.code for p in particle_spec.placements}
    disclosed = Counter()
    for code, count in disclosed_hand:
        disclosed[int(code)] += int(count)

    report = {}
    for name in ("opp_hand", "opp_deck", "opp_facedown", "own_deck"):
        slots = groups[name]
        if not slots:
            continue
        pool = Counter(p.code for p in slots)
        if name == "opp_hand":
            # the disclosed cards should match anyway; counted neither as hits nor in the baseline
            for code, count in disclosed.items():
                pool[code] -= count
            pool = +pool
        total = sum(pool.values())
        hits = 0
        checked = 0
        for slot in slots:
            code = got.get(slot.key())
            if code is None:
                continue
            if disclosed.get(slot.code, 0) and name == "opp_hand":
                continue
            checked += 1
            hits += int(code == slot.code)
        baseline = (sum(c * c for c in pool.values()) / (total * total)
                    if total else 0.0)
        report[name] = {
            "slots": checked,
            "hit_rate": hits / checked if checked else None,
            "chance_baseline": baseline,
        }
    return report
