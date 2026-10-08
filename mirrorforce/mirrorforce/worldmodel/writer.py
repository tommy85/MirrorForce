"""Sharded parquet output.

Both tables are one row per *decision point* or per *action*, with the parts
that vary in length kept as list columns rather than exploded into their own
rows.  A decision point carries ~180 candidates; writing 180 rows that repeat
the same board would multiply the corpus by two orders of magnitude for no
information, and the menu head consumes the whole candidate set at once anyway.

Parallel list-of-scalar columns are used in preference to lists of structs:
parquet stores them as plain dictionary-encoded pages, they read back as flat
numpy arrays with an offset vector, and no reader needs to understand a nested
schema.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import SCHEMA_VERSION
from .candidates import ActionKind
from .generate import GameResult, MenuSample, SettleSample

__all__ = [
    "HARDNEG_SCHEMA",
    "MENU_SCHEMA",
    "REPLAY_SCHEMA",
    "SETTLE_SCHEMA",
    "ShardWriter",
    "hardneg_batch",
    "menu_batch",
    "replay_batch",
    "settle_batch",
]

_KIND_INDEX = {kind.value: i for i, kind in enumerate(ActionKind)}

_u8 = pa.list_(pa.uint8())
_i8 = pa.list_(pa.int8())
_u32 = pa.list_(pa.uint32())
_i32 = pa.list_(pa.int32())
_bool = pa.list_(pa.bool_())
_str = pa.list_(pa.string())
_u32x2 = pa.list_(_u32)
_boolx2 = pa.list_(_bool)
_binary = pa.list_(pa.binary())

MENU_SCHEMA = pa.schema(
    [
        ("game_id", pa.string()),
        ("decision_id", pa.string()),
        ("parent_id", pa.string()),
        ("source", pa.string()),
        ("forced_pass", pa.bool_()),
        ("player", pa.uint8()),          # seat, 0 or 1
        ("first_player", pa.uint8()),    # the seat that took turn 1
        ("on_play", pa.bool_()),         # this row's player went first
        ("msg", pa.uint8()),
        ("turn", pa.uint16()),
        ("turn_player", pa.uint8()),
        ("phase", pa.uint16()),
        ("response_index", pa.int32()),
        ("decision_ordinal", pa.int32()),
        # Deck-building rule environment. banlist_code/limit list only entries that differ from the default of 3 copies.
        ("banlist_id", pa.uint32()),
        ("banlist_code", _u32),
        ("banlist_limit", _u8),
        ("deck_self", pa.string()),
        ("deck_opp", pa.string()),
        ("policy_self", pa.string()),
        ("policy_opp", pa.string()),
        ("outcome", pa.int8()),          # null on a fork
        ("steps_to_end", pa.int32()),    # null on a fork
        ("lp_self", pa.int32()),
        ("lp_opp", pa.int32()),
        # Exact query_effect_info state after the same information-state mask.
        # Old corpora lack this optional column and are read as empty state.
        ("core_info", pa.string()),
        # board
        ("st_controller", _u8),
        ("st_location", _u8),
        ("st_sequence", _u8),
        ("st_code", _u32),
        ("st_position", _u8),
        ("st_type", _u32),
        ("st_level", _u8),
        ("st_rank", _u8),
        ("st_lscale", _u8),
        ("st_rscale", _u8),
        ("st_attack", _i32),
        ("st_defense", _i32),
        ("st_status", _u32),
        ("st_overlay", _u8),
        ("st_hidden", _bool),
        # -- runtime attribute / race / Link rating / Link arrow bitmap. `QUERY_ATTRIBUTE` /
        # `QUERY_RACE` / `QUERY_LINK` / `QUERY_LINK_MARKER` are standard protocol fields
        # that `parse_query_segments` and `CardState` always decoded but never wrote into columns.
        # Extra Deck material checks (`IsLinkAttribute` / `IsLinkRace` / `IsLinkType`)
        # need exactly the **runtime** values of these three; the printed values in the frozen card vectors are not comparable columns.
        # `st_type` is already above; only its consumer was missing.
        # This card's identity is already public to **both** sides (activated / revealed / confirmed). It is a fact of the duel,
        # not a viewpoint quantity: the bit is the same on the same card in both seats' samples. True on the opponent's hand
        # = we know what it is; true on our hand = the opponent knows what this card of ours is,
        # and its concealment value is gone.
        ("st_public", _bool),
        ("st_attribute", _u32),
        ("st_race", _u32),
        ("st_link", _u8),            # Link rating; 0 for non-Link monsters
        ("st_link_marker", _u32),    # LINK_MARKER_* bitmap; effects can change it at runtime
        # -- completion (core audit + relation review). These fields were always decoded by `parse_query_segments`
        # and used to be dropped at the CardState/writer layer.
        # `QUERY_ALIAS` returns the **current card name after renaming**, not the second name of `get_another_code()`;
        # the old comment calling it "second card name" was wrong. The real second card name exists only in
        # `code2` of the modified core (`out["cards"]` of `duelstate.py`); the standard
        # protocol cannot query it, so it is a supervision target, not an actor input.
        ("st_alias", _u32),          # current card name (after renaming effects)
        ("st_base_attack", _i32),    # original ATK; the difference from st_attack = the modifier
        ("st_base_defense", _i32),
        ("st_owner", _i8),           # owner, ≠ controller
        ("st_equip_at", _u32),       # equip edge: whom this card is equipped to (packed location)
        # per-card counters: `ct_at` is the packed location of the card, `ct_type`/`ct_count` come in pairs
        ("ct_at", _u32),
        ("ct_type", _u32),
        ("ct_count", _u32),
        # persistent "targeted": `tgc_at` is the targeting card's location, `tgc_target` the target's location
        ("tgc_at", _u32),
        ("tgc_target", _u32),
        # Xyz material identities: `st_overlay` has only a count, but the resolution of detach effects depends on
        # what the materials are. `ov_at` is the packed location of the owning Xyz, `ov_code` the material's code.
        ("ov_at", _u32),
        ("ov_code", _u32),
        # candidates
        ("cd_kind", _u8),
        ("cd_location", _u8),
        ("cd_sequence", _i8),
        ("cd_code", _u32),
        ("cd_slot", _i8),
        ("cd_label", _bool),
        ("cd_neg", _u8),
        ("cd_unseen", _bool),
        ("cd_opponent", _bool),
        # The full desc given by the engine (0 on a miss). The effect slot is now looked up in desc_map,
        # no longer `desc & 0xF`: that is a string-table index, inconsistent with the effect index on 35.55% of
        # activation candidates, and truncation loses the lookup key of Stringids carrying another card's code.
        ("cd_desc", _u32),
        ("chosen_index", pa.int32()),
        ("menu_size", pa.int32()),
        ("n_missed", pa.int32()),
        ("n_duplicate", pa.int32()),
        ("n_system_desc", pa.int32()),
        ("truncated", pa.bool_()),
    ]
)

#: the hard-negative probe set, exploded one row per candidate.  These are the
#: questions the A-stage understanding probe is made of, so they are written as
#: their own table rather than left as a filter over ``menu``: an evaluator
#: should not have to know how to unpack a list column to ask them.
HARDNEG_SCHEMA = pa.schema(
    [
        ("game_id", pa.string()),
        ("decision_id", pa.string()),
        ("cand_index", pa.int32()),
        ("neg_class", pa.uint8()),
        ("player", pa.uint8()),
        ("on_play", pa.bool_()),
        ("msg", pa.uint8()),
        ("turn", pa.uint16()),
        ("phase", pa.uint16()),
        ("kind", pa.uint8()),
        ("code", pa.uint32()),
        ("slot", pa.int8()),
        ("location", pa.uint8()),
        ("sequence", pa.int8()),
        ("unseen", pa.bool_()),
        ("source", pa.string()),
        ("deck_self", pa.string()),
        ("deck_opp", pa.string()),
    ]
)

# One row per generated duel.  This is the label factory's native replay
# format, not a client .yrp: config + ordered deck recipe + the exact response
# byte stream is sufficient to reproduce every core message and regenerate
# future label schemas without replaying a learned policy.
REPLAY_SCHEMA = pa.schema(
    [
        ("game_id", pa.string()),
        ("ok", pa.bool_()),
        ("error", pa.string()),
        ("duel_seed", pa.int64()),
        ("deck0_name", pa.string()),
        ("deck0_main", _u32),
        ("deck0_extra", _u32),
        ("deck1_name", pa.string()),
        ("deck1_main", _u32),
        ("deck1_extra", _u32),
        ("policy0", pa.string()),
        ("policy1", pa.string()),
        ("start_lp", pa.int32()),
        ("start_hand", pa.uint8()),
        ("draw_count", pa.uint8()),
        ("duel_options", pa.uint64()),
        ("banlist_id", pa.uint32()),
        ("banlist_code", _u32),
        ("banlist_limit", _u8),
        ("max_options", pa.uint16()),
        ("max_sub_rounds", pa.uint16()),
        ("first_player", pa.int8()),
        ("winner", pa.int8()),
        ("turns", pa.uint16()),
        ("response_count", pa.int32()),
        ("responses", _binary),
        ("choice_count", pa.int32()),
        ("choice_response_index", _i32),
        ("choice_round_index", _u32),
        ("choice_msg", _u8),
        ("choice_player", _u8),
        ("choice_index", _u32),
        ("choice_action", _str),
    ]
)

SETTLE_SCHEMA = pa.schema(
    [
        ("game_id", pa.string()),
        ("decision_id", pa.string()),
        ("settle_id", pa.string()),
        ("source", pa.string()),
        ("on_trajectory", pa.bool_()),
        ("forced_pass", pa.bool_()),
        ("player", pa.uint8()),
        ("on_play", pa.bool_()),
        ("action_kind", pa.uint8()),
        ("action_spec", pa.string()),
        ("action_slot", pa.int8()),
        ("action_menu_index", pa.int32()),
        ("action_desc", pa.uint32()),
        ("action_code", pa.uint32()),
        ("action_at", pa.uint32()),
        ("lp_delta_0", pa.int32()),
        ("lp_delta_1", pa.int32()),
        ("mv_code", _u32),
        ("mv_from", _u32),   # controller | location<<8 | sequence<<16 | position<<24
        ("mv_to", _u32),
        ("mv_reason", _u32),
        # Within the resolution window of which chain link this event falls (0 = outside a window, the activation phase).
        # Without it a whole chain's deltas cannot be split back per link: the move columns and chain columns each keep
        # message order, but their interleaving, which is the attribution itself, is not in the corpus.
        ("mv_link", _u8),
        ("mv_trace", _u32),
        ("pc_code", _u32),
        ("pc_at", _u32),
        ("pc_prev", _u8),
        ("pc_cur", _u8),
        ("pc_link", _u8),
        ("pc_trace", _u32),
        # Draws could only be inferred from the residual of `cnt_delta`; per link they must be explicit:
        # the residual is the total of the whole interval and cannot be split per link. **Only player, count and link are written, no codes**:
        # what was drawn is a chance node, the resolution header only predicts the count anyway, and writing codes would just
        # add a leak surface.
        ("dr_player", _u8),
        ("dr_count", _u8),
        ("dr_link", _u8),
        ("dr_trace", _u32),
        ("ng_code", _u32),
        ("ng_at", _u32),
        ("appeared", _u32),
        ("vanished", _u32),
        ("cnt_key", _u32),   # player | location<<8
        ("cnt_delta", _i32),
        ("turn_delta", pa.int32()),
        ("phase_after", pa.uint16()),
        ("chance", pa.bool_()),
        ("chance_msgs", _str),
        # Three-way: 0 deterministic / 1 true randomness / 2 hidden information. `chance` is the binary degenerate form,
        # lumping "a coin flip" with "a draw"; for a world model they differ:
        # true randomness cannot be predicted with any amount of information, hidden information can once observed.
        ("determinism", pa.uint8()),
        # 0/missing = legacy compressed rule inference; 1 = public summon/set
        # declarations are present in pub_* and may replace move heuristics.
        ("public_rule_version", pa.uint8()),
        ("next_menu", _str),
        ("next_msg", pa.uint8()),
        ("next_player", pa.int8()),
        ("opponent_window", pa.bool_()),
        ("roundtrip_ok", pa.bool_()),
        ("reached_next", pa.bool_()),
        ("ended", pa.bool_()),
        ("lifo_group", pa.string()),
        ("sub_msg", _u8),
        ("sub_n", _u8),
        ("sub_choice", _u8),
        ("sub_desc", _str),
        ("sub_player", _u8),
        ("sub_code", _u32),
        ("sub_at", _u32),
        ("sub_value", _u32),
        # 0 = activation / procedure phase, 1 = the matching chain link is resolving. Without these two columns, non-targeting
        # choices made only at resolution would be moved up to right after ACT, leaking future conditions to earlier links.
        ("sub_stage", _u8),
        ("sub_link", _u8),
        ("sub_trace", _u32),
        # The complete candidate box belongs only to the answering side. The outer level aligns with sub prompts, the inner level is the
        # candidate multiset of that prompt. Truncated prompts with known=false must not enter set supervision.
        ("sub_cand_known", _bool),
        ("sub_cand_code", _u32x2),
        ("sub_cand_at", _u32x2),
        ("sub_cand_value", _u32x2),
        ("sub_cand_desc", _u32x2),
        ("sub_cand_finish", _boolx2),
        # Chain structure: kind 0 chaining / 1 solved / 2 negated / 3 disabled.
        # chaining rows carry the code, packed location, effect description string and activating player; the other three only the link number.
        ("ch_kind", _u8),
        ("ch_link", _u8),
        ("ch_code", _u32),
        ("ch_at", _u32),
        ("ch_desc", _u32),
        ("ch_player", _u8),
        ("ch_trace", _u32),
        # Per-link targeting: `tg_link` is the link number, `tg_at` a packed location reference.
        # `MSG_BECOME_TARGET` gives only the location, no code, so naturally there is no identity here.
        # Whether a link "targets" = whether it appears in tg_link, without a separate derived column.
        ("tg_link", _u8),
        ("tg_at", _u32),
        ("tg_trace", _u32),
        # Battle context: packed location references of the attacker and the attack target.
        # `query_field_card` cannot see this: it is processor state, not a card field.
        ("at_attacker", _u32),
        ("at_target", _u32),
        ("at_trace", _u32),
        # LP events with reasons: 0 damage / 1 recover / 2 pay_cost / 3 update,
        # plus "inside the damage step" to split battle damage from effect damage.
        # Recording only the net difference cannot learn distinctions like "take no battle damage".
        ("lp_kind", _u8),
        ("lp_player", _u8),
        ("lp_amount", _u32),
        ("lp_link", _u8),
        ("lp_battle", _bool),
        ("lp_trace", _u32),
        # Public disclosures are conditional inputs of later resolution, not targets the resolution header guesses. audience
        # is a seat bitmask, preventing the raw core's omniscient messages from leaking to the other seat.
        ("pub_kind", _u8),
        ("pub_player", _u8),
        ("pub_code", _u32),
        ("pub_at", _u32),
        ("pub_target", _u32),
        ("pub_value", _u32),
        ("pub_detail", _u32),
        ("pub_link", _u8),
        ("pub_audience", _u8),
        ("pub_trace", _u32),
    ]
)


def _pack_ls(controller: int, location: int, sequence: int, position: int = 0) -> int:
    return (
        (controller & 0xFF)
        | ((location & 0xFF) << 8)
        | ((sequence & 0xFF) << 16)
        | ((position & 0xFF) << 24)
    )


def menu_batch(samples: list[MenuSample]) -> pa.RecordBatch:
    cols: dict[str, list] = {name: [] for name in MENU_SCHEMA.names}
    for s in samples:
        cards = s.state.cards
        cands = s.candidates
        cols["game_id"].append(s.game_id)
        cols["decision_id"].append(s.decision_id)
        cols["parent_id"].append(s.parent_id or "")
        cols["source"].append(s.source)
        cols["forced_pass"].append(s.forced_pass)
        cols["player"].append(s.player)
        cols["first_player"].append(s.first_player)
        cols["on_play"].append(s.on_play)
        cols["msg"].append(s.msg)
        cols["turn"].append(s.turn)
        cols["turn_player"].append(s.turn_player)
        cols["phase"].append(s.phase)
        cols["response_index"].append(s.response_index)
        cols["decision_ordinal"].append(s.decision_ordinal)
        cols["banlist_id"].append(int(getattr(s, "banlist_id", 0)))
        banlist = tuple(getattr(s, "banlist", ()) or ())
        cols["banlist_code"].append([int(code) for code, _ in banlist])
        cols["banlist_limit"].append([int(limit) for _, limit in banlist])
        cols["deck_self"].append(s.deck_self)
        cols["deck_opp"].append(s.deck_opp)
        cols["policy_self"].append(s.policy_self)
        cols["policy_opp"].append(s.policy_opp)
        cols["outcome"].append(s.outcome)
        cols["steps_to_end"].append(s.steps_to_end)
        cols["lp_self"].append(s.state.lp[s.player])
        cols["lp_opp"].append(s.state.lp[1 - s.player])
        cols["core_info"].append(json.dumps(
            s.state.core_info or {}, sort_keys=True, separators=(",", ":")
        ))
        cols["st_controller"].append([c.controller for c in cards])
        cols["st_location"].append([c.location for c in cards])
        cols["st_sequence"].append([min(c.sequence, 255) for c in cards])
        cols["st_code"].append([c.code for c in cards])
        cols["st_position"].append([c.position for c in cards])
        cols["st_type"].append([c.type for c in cards])
        cols["st_level"].append([min(c.level, 255) for c in cards])
        cols["st_rank"].append([min(c.rank, 255) for c in cards])
        cols["st_lscale"].append([min(c.lscale, 255) for c in cards])
        cols["st_rscale"].append([min(c.rscale, 255) for c in cards])
        cols["st_attack"].append([c.attack for c in cards])
        cols["st_defense"].append([c.defense for c in cards])
        cols["st_status"].append([c.status for c in cards])
        cols["st_overlay"].append([len(c.overlay) for c in cards])
        cols["st_hidden"].append([c.hidden for c in cards])
        cols["st_public"].append([c.public for c in cards])
        cols["st_attribute"].append([c.attribute for c in cards])
        cols["st_race"].append([c.race for c in cards])
        cols["st_link"].append([min(c.link, 255) for c in cards])
        cols["st_link_marker"].append([c.link_marker for c in cards])
        cols["st_alias"].append([c.alias for c in cards])
        cols["st_base_attack"].append([c.base_attack for c in cards])
        cols["st_base_defense"].append([c.base_defense for c in cards])
        cols["st_owner"].append([max(-1, min(c.owner, 127)) for c in cards])
        cols["st_equip_at"].append([c.equip_card for c in cards])
        ct_at, ct_type, ct_count = [], [], []
        tgc_at, tgc_target = [], []
        for c in cards:
            at = _pack_ls(c.controller, c.location, c.sequence)
            for ctype, n in (c.counters or ()):
                ct_at.append(at); ct_type.append(int(ctype)); ct_count.append(int(n))
            for t in (c.targets or ()):
                tgc_at.append(at); tgc_target.append(int(t))
        cols["ct_at"].append(ct_at)
        cols["ct_type"].append(ct_type)
        cols["ct_count"].append(ct_count)
        cols["tgc_at"].append(tgc_at)
        cols["tgc_target"].append(tgc_target)
        ov_at, ov_code = [], []
        for c in cards:
            for material in (c.overlay or ()):
                ov_at.append(_pack_ls(c.controller, c.location, c.sequence))
                ov_code.append(int(material))
        cols["ov_at"].append(ov_at)
        cols["ov_code"].append(ov_code)
        cols["cd_kind"].append([_KIND_INDEX[c.kind.value] for c in cands])
        cols["cd_location"].append([c.location for c in cands])
        cols["cd_sequence"].append([max(-1, min(c.sequence, 127)) for c in cands])
        cols["cd_code"].append([c.code for c in cands])
        cols["cd_slot"].append([c.eff_slot for c in cands])
        cols["cd_label"].append(list(s.labels))
        cols["cd_neg"].append(list(s.neg_class))
        cols["cd_unseen"].append(list(s.unseen))
        cols["cd_opponent"].append([c.opponent for c in cands])
        cols["cd_desc"].append(list(getattr(s, "desc", None) or [0] * len(cands)))
        cols["chosen_index"].append(s.chosen_index)
        cols["menu_size"].append(s.menu_size)
        cols["n_missed"].append(s.n_missed)
        cols["n_duplicate"].append(s.n_duplicate)
        cols["n_system_desc"].append(s.n_system_desc)
        cols["truncated"].append(s.truncated)
    return pa.record_batch(
        [pa.array(cols[f.name], type=f.type) for f in MENU_SCHEMA],
        schema=MENU_SCHEMA,
    )


def settle_batch(samples: list[SettleSample]) -> pa.RecordBatch:
    cols: dict[str, list] = {name: [] for name in SETTLE_SCHEMA.names}
    for s in samples:
        d = s.diff
        key = s.action_key or ("", "", -1)
        cols["game_id"].append(s.game_id)
        cols["decision_id"].append(s.decision_id)
        cols["settle_id"].append(s.settle_id)
        cols["source"].append(s.source)
        cols["on_trajectory"].append(s.on_trajectory)
        cols["forced_pass"].append(s.forced_pass)
        cols["player"].append(s.player)
        cols["on_play"].append(s.on_play)
        cols["action_kind"].append(_KIND_INDEX.get(key[0], 255))
        cols["action_spec"].append(key[1])
        cols["action_slot"].append(max(-1, min(int(key[2]), 127)))
        cols["action_menu_index"].append(s.action_menu_index)
        cols["action_desc"].append(int(getattr(s, "action_desc", 0) or 0))
        cols["action_code"].append(int(getattr(s, "action_code", 0) or 0))
        cols["action_at"].append(int(getattr(s, "action_at", 0) or 0))
        cols["lp_delta_0"].append(d.lp_delta[0])
        cols["lp_delta_1"].append(d.lp_delta[1])
        # Viewpoint mask: an invisible move writes 0 (see SettleSample.mv_code_visible)
        vis = s.mv_code_visible
        cols["mv_code"].append([
            (m.code if (vis is None or (i < len(vis) and vis[i])) else 0)
            for i, m in enumerate(d.moves)
        ])
        cols["mv_from"].append(
            [
                _pack_ls(m.from_controller, m.from_location, m.from_sequence,
                         m.from_position)
                for m in d.moves
            ]
        )
        cols["mv_to"].append(
            [
                _pack_ls(m.to_controller, m.to_location, m.to_sequence, m.to_position)
                for m in d.moves
            ]
        )
        cols["mv_reason"].append([m.reason for m in d.moves])
        cols["mv_link"].append([min(getattr(m, "link", 0), 255) for m in d.moves])
        cols["mv_trace"].append([
            max(int(getattr(m, "trace_index", 0)), 0) for m in d.moves
        ])
        cols["pc_code"].append([p.code for p in d.pos_changes])
        cols["pc_at"].append(
            [_pack_ls(p.controller, p.location, p.sequence) for p in d.pos_changes]
        )
        cols["pc_prev"].append([p.previous for p in d.pos_changes])
        cols["pc_cur"].append([p.current for p in d.pos_changes])
        cols["pc_link"].append(
            [min(getattr(p, "link", 0), 255) for p in d.pos_changes]
        )
        cols["pc_trace"].append([
            max(int(getattr(p, "trace_index", 0)), 0) for p in d.pos_changes
        ])
        cols["dr_player"].append([w.player for w in d.draws])
        cols["dr_count"].append([min(w.count, 255) for w in d.draws])
        cols["dr_link"].append([min(getattr(w, "link", 0), 255) for w in d.draws])
        cols["dr_trace"].append([
            max(int(getattr(w, "trace_index", 0)), 0) for w in d.draws
        ])
        cols["ng_code"].append([n[3] for n in d.negated])
        cols["ng_at"].append([_pack_ls(n[0], n[1], n[2]) for n in d.negated])
        cols["appeared"].append(list(d.appeared))
        cols["vanished"].append(list(d.vanished))
        cols["cnt_key"].append(
            [(p & 0xFF) | ((loc & 0xFF) << 8) for (p, loc) in d.count_delta]
        )
        cols["cnt_delta"].append(list(d.count_delta.values()))
        cols["turn_delta"].append(d.turn_delta)
        cols["phase_after"].append(d.phase_after)
        cols["chance"].append(d.chance)
        cols["chance_msgs"].append(list(d.chance_msgs))
        cols["determinism"].append(getattr(d, "determinism", 0))
        cols["public_rule_version"].append(1)
        cols["next_menu"].append(
            [f"{k[0]}|{k[1]}|{k[2]}" for k in (s.next_menu or [])]
        )
        cols["next_msg"].append(s.next_msg)
        cols["next_player"].append(s.next_player)
        cols["opponent_window"].append(s.opponent_window)
        cols["roundtrip_ok"].append(s.roundtrip_ok)
        cols["reached_next"].append(s.reached_next)
        cols["ended"].append(s.ended)
        cols["lifo_group"].append(s.lifo_group)
        # Archive every raw prompt with its responder. Visibility belongs to
        # the seat-specific serializer, not this irreversible storage layer:
        # dropping non-activator prompts here also drops prompts that are
        # private to the *other* seat and therefore should be visible in that
        # seat's own training view.
        subs = list(s.sub_prompts)
        cols["sub_msg"].append([p.msg for p in subs])
        cols["sub_n"].append([min(p.n, 255) for p in subs])
        cols["sub_choice"].append([min(p.choice, 255) for p in subs])
        cols["sub_desc"].append([p.desc for p in subs])
        cols["sub_player"].append([p.player for p in subs])
        cols["sub_code"].append([p.code for p in subs])
        cols["sub_at"].append([p.at for p in subs])
        cols["sub_value"].append([p.value for p in subs])
        cols["sub_stage"].append([min(getattr(p, "stage", 0), 1)
                                  for p in subs])
        cols["sub_link"].append([min(getattr(p, "link", 0), 255)
                                 for p in subs])
        cols["sub_trace"].append([max(int(getattr(p, "trace_index", 0)), 0)
                                  for p in subs])
        cols["sub_cand_known"].append([not p.truncated for p in subs])
        cols["sub_cand_code"].append([
            [c.code for c in p.candidates] if not p.truncated else [] for p in subs
        ])
        cols["sub_cand_at"].append([
            [c.at for c in p.candidates] if not p.truncated else [] for p in subs
        ])
        cols["sub_cand_value"].append([
            [c.value for c in p.candidates] if not p.truncated else [] for p in subs
        ])
        cols["sub_cand_desc"].append([
            [c.desc for c in p.candidates] if not p.truncated else [] for p in subs
        ])
        cols["sub_cand_finish"].append([
            [c.finish for c in p.candidates] if not p.truncated else [] for p in subs
        ])
        chain = getattr(d, "chain", []) or []
        cols["ch_kind"].append([c.kind for c in chain])
        cols["ch_link"].append([min(c.link, 255) for c in chain])
        cols["ch_code"].append([c.code for c in chain])
        cols["ch_at"].append([c.at for c in chain])
        cols["ch_desc"].append([c.desc for c in chain])
        cols["ch_player"].append([c.player for c in chain])
        cols["ch_trace"].append([
            max(int(getattr(c, "trace_index", 0)), 0) for c in chain
        ])
        targets = getattr(d, "targets", []) or []
        cols["tg_link"].append([min(t.link, 255) for t in targets])
        cols["tg_at"].append([t.at for t in targets])
        cols["tg_trace"].append([
            max(int(getattr(t, "trace_index", 0)), 0) for t in targets
        ])
        attacks = getattr(d, "attacks", []) or []
        cols["at_attacker"].append([a.attacker for a in attacks])
        cols["at_target"].append([a.target for a in attacks])
        cols["at_trace"].append([
            max(int(getattr(a, "trace_index", 0)), 0) for a in attacks
        ])
        lp_events = getattr(d, "lp_events", []) or []
        cols["lp_kind"].append([e.kind for e in lp_events])
        cols["lp_player"].append([e.player for e in lp_events])
        cols["lp_amount"].append([e.amount for e in lp_events])
        cols["lp_link"].append([min(e.link, 255) for e in lp_events])
        cols["lp_battle"].append([e.in_damage_step for e in lp_events])
        cols["lp_trace"].append([
            max(int(getattr(e, "trace_index", 0)), 0) for e in lp_events
        ])
        public = getattr(d, "public_events", []) or []
        cols["pub_kind"].append([e.kind for e in public])
        cols["pub_player"].append([e.player for e in public])
        cols["pub_code"].append([e.code for e in public])
        cols["pub_at"].append([e.at for e in public])
        cols["pub_target"].append([e.target for e in public])
        cols["pub_value"].append([e.value for e in public])
        cols["pub_detail"].append([e.detail for e in public])
        cols["pub_link"].append([min(e.link, 255) for e in public])
        cols["pub_audience"].append([e.audience for e in public])
        cols["pub_trace"].append([max(int(getattr(e, "trace_index", 0)), 0)
                                  for e in public])
    return pa.record_batch(
        [pa.array(cols[f.name], type=f.type) for f in SETTLE_SCHEMA],
        schema=SETTLE_SCHEMA,
    )


def hardneg_batch(rows: list[tuple]) -> pa.RecordBatch:
    cols: dict[str, list] = {name: [] for name in HARDNEG_SCHEMA.names}
    for row in rows:
        for name, value in zip(HARDNEG_SCHEMA.names, row):
            cols[name].append(value)
    return pa.record_batch(
        [pa.array(cols[f.name], type=f.type) for f in HARDNEG_SCHEMA],
        schema=HARDNEG_SCHEMA,
    )


def replay_batch(results: list[GameResult]) -> pa.RecordBatch:
    cols: dict[str, list] = {name: [] for name in REPLAY_SCHEMA.names}
    for result in results:
        config = result.config
        if config is None:
            raise ValueError(f"{result.game_id}: replay result has no DuelConfig")
        deck0, deck1 = config.decks
        limit_codes = [int(code) for code, _ in config.banlist]
        limits = [int(limit) for _, limit in config.banlist]
        values = {
            "game_id": result.game_id,
            "ok": bool(result.ok),
            "error": result.error,
            "duel_seed": int(config.seed),
            "deck0_name": deck0.name,
            "deck0_main": list(deck0.main),
            "deck0_extra": list(deck0.extra),
            "deck1_name": deck1.name,
            "deck1_main": list(deck1.main),
            "deck1_extra": list(deck1.extra),
            "policy0": result.policies[0],
            "policy1": result.policies[1],
            "start_lp": int(config.start_lp),
            "start_hand": int(config.start_hand),
            "draw_count": int(config.draw_count),
            "duel_options": int(config.duel_options),
            "banlist_id": int(config.banlist_id),
            "banlist_code": limit_codes,
            "banlist_limit": limits,
            "max_options": int(config.max_options),
            "max_sub_rounds": int(config.max_sub_rounds),
            "first_player": result.first_player,
            "winner": result.winner,
            "turns": int(result.turns),
            "response_count": len(result.responses),
            "responses": list(result.responses),
            "choice_count": len(result.choices),
            "choice_response_index": [row.response_index for row in result.choices],
            "choice_round_index": [row.round_index for row in result.choices],
            "choice_msg": [row.msg for row in result.choices],
            "choice_player": [row.player for row in result.choices],
            "choice_index": [row.choice for row in result.choices],
            "choice_action": [row.action for row in result.choices],
        }
        for name in REPLAY_SCHEMA.names:
            cols[name].append(values[name])
    return pa.record_batch(
        [pa.array(cols[field.name], type=field.type) for field in REPLAY_SCHEMA],
        schema=REPLAY_SCHEMA,
    )


def hardneg_rows(sample: MenuSample) -> list[tuple]:
    """Every candidate of one decision point the engine refused for a reason."""
    out = []
    for i, (candidate, label, neg, unseen) in enumerate(
        zip(sample.candidates, sample.labels, sample.neg_class, sample.unseen)
    ):
        if label or neg == 0:
            continue
        out.append(
            (
                sample.game_id,
                sample.decision_id,
                i,
                neg,
                sample.player,
                sample.on_play,
                sample.msg,
                sample.turn,
                sample.phase,
                _KIND_INDEX[candidate.kind.value],
                candidate.code,
                max(-1, min(candidate.eff_slot, 127)),
                candidate.location,
                max(-1, min(candidate.sequence, 127)),
                unseen,
                sample.source,
                sample.deck_self,
                sample.deck_opp,
            )
        )
    return out


@dataclass
class ShardWriter:
    """Buffers rows and flushes them into numbered parquet shards."""

    out_dir: Path
    name: str
    schema: pa.Schema
    rows_per_shard: int = 20000
    compression: str = "zstd"

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list = []
        self._index = 0
        self.rows = 0
        self.shards: list[str] = []

    def add(self, samples: list) -> None:
        self._buffer.extend(samples)
        if len(self._buffer) >= self.rows_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        if self.schema is MENU_SCHEMA:
            batch = menu_batch(self._buffer)
        elif self.schema is SETTLE_SCHEMA:
            batch = settle_batch(self._buffer)
        elif self.schema is REPLAY_SCHEMA:
            batch = replay_batch(self._buffer)
        else:
            batch = hardneg_batch(self._buffer)
        filename = f"{self.name}-{self._index:05d}.parquet"
        pq.write_table(
            pa.Table.from_batches([batch]),
            self.out_dir / filename,
            compression=self.compression,
        )
        self.shards.append(filename)
        self.rows += batch.num_rows
        self._index += 1
        self._buffer = []

    def close(self) -> None:
        self.flush()


def write_manifest(out_dir: Path, payload: dict) -> dict:
    """Write ``manifest.json`` and hand back what was written."""
    path = Path(out_dir) / "manifest.json"
    payload["schema_version"] = SCHEMA_VERSION
    payload["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
