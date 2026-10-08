"""Drive a live duel with the PV-14 world model.

The model is not a policy that was trained to win; it is a sequence model of
duels, and the policy head is one readout among several.  Putting it on the
table is the first milestone of the "world model plays" chain, and the bar is a
pipeline that provably encodes the live board the way the corpus did -- not a
win rate.

**Nothing here re-implements the encoder.**  The one lesson worth carrying over
from PV-1 is that an observation encoder can be wrong in ways that cost win rate
without raising anything, and that only a differential check against the
original path finds it.  So the live board is pushed through exactly the corpus
code::

    ShadowBoard -> StateSnapshot -> mask_for -> enumerate_candidates
      -> MenuSample -> menu_batch -> MenuTable -> ItemBuilder -> collate -> model

Every step except the first is the same function the training corpus was built
with, imported read-only from ``worldmodel-train``.

Four things this has to get right that a naive bridge does not
--------------------------------------------------------------

1. **The deck and the opponent's hidden zones have to be in the snapshot.**
   The corpus captures *every* card and lets :func:`mask_for` decide what the
   player may see -- our deck becomes code-sorted rows, the opponent's deck,
   hand and extra become one blanked row each.  A snapshot that carries only
   the zones a client can see leaves the ``DECK`` tokens empty and all three
   ``OPPCNT`` tokens reporting zero, in every block, silently.  The shadow
   board knows both: ``our_remaining_deck()`` for ours, the zone lengths and
   the tracked counts for theirs.

2. **Life points come from the client, not the board.**  ``ShadowBoard`` has no
   ``lp``; the client does.  They appear in the ``STATE`` token, in every
   ``CARD``, in every ``SLOT`` and in the ``GLOBAL``, so getting them wrong is
   not a small error in one place.

3. **The context is the whole game.**  The whole-game trainer feeds one
   duel/seat as a single sequence -- no slicing, no recurrent carry -- so
   inference has to do the same or the model is asked at play time for a
   distribution it never saw.  :class:`WholeGameStream` keeps one append-only
   :class:`ItemBuilder` per duel and pushes each new segment through the
   model's own exact KV cache: ``prefill`` for the header plus the first state
   block, ``decode_state_block`` for one complete later state block,
   ``decode_causal`` for a run of action/event/disclosure tokens.  History is
   never recomputed, so the 400th decision of a duel costs what the 4th did.

4. **Top-level and private prompts use different heads.**
   ``MSG_SELECT_IDLECMD``, ``MSG_SELECT_BATTLECMD`` and ``MSG_SELECT_CHAIN``
   use the top-level policy grid.  Checkpoints with ``sub_policy`` also score
   target, option, tribute and other private prompts over the authoritative
   engine menu; older checkpoints retain an explicit counted fallback.

The engine's menu is the hard action boundary.  The live choice is a softmax
over the policy logits of those entries only; the legality head and value head
are diagnostics, and the settlement head is auxiliary training supervision.  No
world-model rollout, learned-legality penalty or value reranking participates
in the live choice.  The opponent's private prompts are never an input.

Why one decision is serialized twice
------------------------------------

``ItemBuilder.point`` emits a state block *and* the ``ACT`` token recording
what was done, in one call, because the corpus always knows the answer already.
A live agent does not: it has to read the state block, ask the model, and only
then know its own ``ACT``.  Rather than keep a second copy of the ACT-writing
rules here -- the exact thing this module exists not to do -- the point is
emitted twice: once with a blank ``ACT`` to obtain the state block, and once
through the builder's own ``act_override`` once the choice is known.  The state
block is a pure function of the same inputs, so the second pass reproduces it
token for token; that equality is asserted rather than assumed, and only the
one-token ``ACT`` is appended to the KV cache afterwards.
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter

import numpy as np

from . import constants as C
from .policy import DecisionState, Policy

__all__ = [
    "WorldModelPolicy", "WholeGameStream", "DecisionRecord",
    "snapshot_from_shadow", "config_from_checkpoint", "load_model",
    "add_worldmodel_arguments", "make_worldmodel_factory",
    "build_card_source", "random_model",
    "serializer_context", "main",
]

#: The prompts the corpus enumerated candidates for.  Anything else has no
#: policy head and must not be answered by argmax over one.
DECISION_MSGS = frozenset(
    {C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD, C.MSG_SELECT_CHAIN}
)

# Only messages that contribute an action, public transition or disclosure are
# retained between model decisions. In particular, an incoming private prompt
# for this client is not copied into history merely because this seat received
# it; its chosen result is represented by ACT/SUB or a later public message.
PUBLIC_HISTORY_MSGS = frozenset({
    C.MSG_HINT,
    C.MSG_CONFIRM_DECKTOP, C.MSG_CONFIRM_CARDS, C.MSG_CONFIRM_EXTRATOP,
    C.MSG_SHUFFLE_DECK, C.MSG_SHUFFLE_HAND, C.MSG_SHUFFLE_EXTRA,
    C.MSG_SHUFFLE_SET_CARD, C.MSG_SWAP_GRAVE_DECK, C.MSG_REVERSE_DECK,
    C.MSG_DECK_TOP, C.MSG_MOVE, C.MSG_POS_CHANGE, C.MSG_SET, C.MSG_SWAP,
    C.MSG_NEW_TURN, C.MSG_NEW_PHASE,
    C.MSG_SUMMONING, C.MSG_SUMMONED, C.MSG_SPSUMMONING, C.MSG_SPSUMMONED,
    C.MSG_FLIPSUMMONING, C.MSG_FLIPSUMMONED,
    C.MSG_CHAINING, C.MSG_CHAIN_SOLVING, C.MSG_CHAIN_SOLVED, C.MSG_CHAIN_END,
    C.MSG_CHAIN_NEGATED, C.MSG_CHAIN_DISABLED,
    C.MSG_RANDOM_SELECTED, C.MSG_BECOME_TARGET, C.MSG_DRAW,
    C.MSG_DAMAGE, C.MSG_RECOVER, C.MSG_EQUIP, C.MSG_LPUPDATE,
    C.MSG_UNEQUIP, C.MSG_CARD_TARGET, C.MSG_CANCEL_TARGET, C.MSG_PAY_LPCOST,
    C.MSG_ADD_COUNTER, C.MSG_REMOVE_COUNTER, C.MSG_ATTACK, C.MSG_BATTLE,
    C.MSG_ATTACK_DISABLED, C.MSG_DAMAGE_STEP_START, C.MSG_DAMAGE_STEP_END,
    C.MSG_MISSED_EFFECT, C.MSG_TOSS_COIN, C.MSG_TOSS_DICE, C.MSG_HAND_RES,
    C.MSG_CARD_HINT, C.MSG_PLAYER_HINT, C.MSG_FIELD_DISABLED,
})

_ZONES = (
    C.LOCATION_DECK,
    C.LOCATION_HAND,
    C.LOCATION_MZONE,
    C.LOCATION_SZONE,
    C.LOCATION_GRAVE,
    C.LOCATION_REMOVED,
    C.LOCATION_EXTRA,
)


def _card_state(cls, controller, location, sequence, code=0, position=0, **kw):
    code = int(code or 0)
    hidden = bool(kw.get("hidden", False))
    if hidden:
        if code:
            raise ValueError("hidden shadow card carries a public passcode")
        # A censored query row may still hold stale values from an earlier
        # refresh.  Preserve only public slot geometry/ownership; no printed or
        # per-instance identity field may cross the PUB boundary.
        return cls(
            controller=int(controller),
            location=int(location),
            sequence=int(sequence),
            code=0,
            position=int(position),
            owner=int(kw.get("owner", -1)),
            hidden=True,
        )
    return cls(
        controller=int(controller), location=int(location), sequence=int(sequence),
        code=code, position=int(position),
        type=int(kw.get("type", 0)), level=int(kw.get("level", 0)),
        rank=int(kw.get("rank", 0)), attribute=int(kw.get("attribute", 0)),
        race=int(kw.get("race", 0)), attack=int(kw.get("attack", 0)),
        defense=int(kw.get("defense", 0)), status=int(kw.get("status", 0)),
        lscale=int(kw.get("lscale", 0)), rscale=int(kw.get("rscale", 0)),
        link=int(kw.get("link", 0)), link_marker=int(kw.get("link_marker", 0)),
        overlay=tuple(int(x) for x in (kw.get("overlay") or ())),
        counters=tuple((int(t), int(n)) for t, n in sorted(
            (kw.get("counters") or {}).items())),
        alias=int(kw.get("alias", 0)), equip_card=int(kw.get("equip_card", 0)),
        owner=int(kw.get("owner", -1)),
        targets=tuple(int(x) for x in (kw.get("targets") or ())),
        base_attack=int(kw.get("base_attack", 0)),
        base_defense=int(kw.get("base_defense", 0)),
        setcode=tuple(int(x) for x in (kw.get("setcode") or ())),
    )


def _static_card_state(
    cls,
    data,
    controller: int,
    location: int,
    sequence: int,
    *,
    position: int,
):
    """One database-complete card row with no private runtime modifiers."""

    type_ = int(data.type)
    level = int(data.level)
    return _card_state(
        cls,
        controller,
        location,
        sequence,
        code=int(data.code),
        position=position,
        type=type_,
        level=(0 if type_ & (C.TYPE_XYZ | C.TYPE_LINK) else level),
        rank=(level if type_ & C.TYPE_XYZ else 0),
        link=(level if type_ & C.TYPE_LINK else 0),
        attribute=int(data.attribute),
        race=int(data.race),
        attack=int(data.attack),
        defense=int(data.defense),
        base_attack=int(data.attack),
        base_defense=int(data.defense),
        alias=int(data.alias),
        setcode=tuple(data.setcodes),
        lscale=int(data.lscale),
        rscale=int(data.rscale),
        link_marker=int(data.link_marker),
        owner=-1,
    )


def _project_opponent_hidden_zone(
    cls,
    card_pool,
    controller: int,
    location: int,
    count: int,
    wire_rows,
    known_counts,
    faceup_extra_counts,
    anchors=None,
):
    """Known multiset with proved anchors retained until the public mask.

    Unproved rows receive only temporary reconstruction coordinates; the
    disclosure resolver will not turn those into known physical identities.
    """

    wire_codes = Counter(
        int(card.code) for card in wire_rows if int(card.code or 0)
    )
    known_codes = Counter({
        int(code): max(0, int(copies))
        for (known_controller, known_location, code), copies in known_counts.items()
        if int(known_controller) == controller
        and int(known_location) == location
        and int(code)
    })
    wire_faceup = Counter(
        int(card.code) for card in wire_rows
        if int(card.code or 0) and not card.face_down
    )
    names = []
    for code in sorted(set(wire_codes) | set(known_codes)):
        copies = max(wire_codes[code], known_codes[code])
        faceup = min(
            copies,
            max(
                wire_faceup[code],
                int(faceup_extra_counts.get((controller, code), 0)),
            ),
        )
        names.extend([(code, True)] * faceup)
        names.extend([(code, False)] * (copies - faceup))
    names = names[:count]
    # Keep only anchors proved by this viewer's message ledger. Re-sorting
    # those rows before ledger.resolve() loses the proof and duplicates a
    # known member as anonymous-slot + unpositioned identity. mask_for() will
    # still remove the raw hidden-zone order before neural input is built.
    seated = {}
    remaining = list(names)
    for sequence, code in sorted((anchors or {}).items()):
        if not 0 <= int(sequence) < count:
            continue
        item = next((entry for entry in remaining if entry[0] == int(code)), None)
        if item is not None:
            seated[int(sequence)] = item
            remaining.remove(item)
    free = (sequence for sequence in range(count) if sequence not in seated)
    for item in remaining:
        seated[next(free)] = item
    rows = []
    for sequence, (code, faceup) in sorted(seated.items()):
        if card_pool is None or code not in card_pool.cards:
            raise ValueError(
                f"disclosed opponent zone code {code} is absent from cards.cdb"
            )
        data = card_pool.cards[code]
        position = (
            C.POS_FACEUP_DEFENSE
            if faceup
            else C.POS_FACEDOWN_DEFENSE
        )
        rows.append(_static_card_state(
            cls,
            data,
            controller,
            location,
            sequence,
            position=position,
        ))
    for sequence in range(count):
        if sequence in seated:
            continue
        rows.append(_card_state(
            cls,
            controller,
            location,
            sequence,
            position=C.POS_FACEDOWN_DEFENSE,
            owner=-1,
        ))
    return rows


def _turn_player(state) -> int:
    """The turn player: the online path reads it from ShadowBoard, offline particles from ``DecisionState``."""
    if state.board is not None:
        return int(getattr(state.board, "turn_player", 0))
    return int(getattr(state, "turn_player", 0))


def snapshot_from_shadow(board, our_player: int, lp, turn: int, card_pool=None):
    """The live board as the corpus would have captured it.

    Captures *everything*, including our deck and the opponent's hidden zones,
    and leaves the hiding to :func:`mask_for` -- the same division of labour the
    corpus uses, so the masking rule stays in one place.  Cards we have never
    been told the identity of go in with ``code=0``, which is what the host's
    blanked query segments mean and what ``mask_for`` would have produced
    anyway.
    """
    from mirrorforce.worldmodel.state import CardState, StateSnapshot

    cards: list = []
    counts: dict[tuple[int, int], int] = {}
    opp = 1 - our_player

    # The client submitted its own Extra Deck, so every remaining row is an
    # entitled identity.  Treat a missing or anonymous row as bridge drift,
    # not as a smaller deck: silently deriving the count from a broken zone is
    # what allowed an off-policy special summon to point at code zero.
    own_extra = list(board.zone(our_player, C.LOCATION_EXTRA))
    declared_own_extra = max(0, int(board.extra_count[our_player]))
    tracked_own_extra = [card for card in own_extra if card is not None]
    if len(tracked_own_extra) != declared_own_extra:
        raise ValueError(
            "shadow own-extra zone/count drift: "
            f"tracked={len(tracked_own_extra)} declared={declared_own_extra}"
        )
    anonymous_own_extra = [
        index
        for index, card in enumerate(own_extra)
        if card is None
        or int(getattr(card, "code", 0) or 0) == 0
        or bool(getattr(card, "hidden", False))
    ]
    if anonymous_own_extra:
        raise ValueError(
            "shadow own-extra identity drift at sequences "
            f"{anonymous_own_extra}"
        )
    own_extra_geometry_drift = [
        index
        for index, card in enumerate(own_extra)
        if card is not None and (
            int(getattr(card, "controller", -1)) != int(our_player)
            or int(getattr(card, "location", 0)) != C.LOCATION_EXTRA
            or int(getattr(card, "sequence", -1)) != index
        )
    ]
    if own_extra_geometry_drift:
        raise ValueError(
            "shadow own-extra geometry drift at sequences "
            f"{own_extra_geometry_drift}"
        )

    ledger = getattr(board, "disclosure", None)
    for player in (0, 1):
        for location in _ZONES:
            if location == C.LOCATION_DECK:
                continue
            zone = board.zone(player, location)
            n = 0
            for seq, sc in enumerate(zone):
                if sc is None:
                    continue
                n += 1
                data = card_pool.cards.get(int(sc.code)) if card_pool else None
                # Live values first, database only as the fallback for a card
                # the host never sent us.  A card's type is not a constant: a
                # pendulum monster sitting in a pendulum zone reports as a
                # SPELL, and the corpus captures that from the engine.  Reading
                # the printed type from the database instead disagrees on every
                # scale a pendulum deck sets -- and the type field is what the
                # CARD token is built from.
                live = bool(getattr(sc, "queried", False))
                position = int(getattr(sc, "position", 0) or 0)
                if bool(getattr(sc, "hidden", False)) and not position:
                    # Stock refresh packets blank a face-down segment including
                    # its position.  MSG_MOVE/MSG_POS_CHANGE are the public
                    # source of that slot state and ShadowBoard keeps it here.
                    position = int(getattr(board, "positions", {}).get(
                        (player, location, seq), 0
                    ))
                if player != our_player \
                        and location in (C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_REMOVED) \
                        and bool(getattr(sc, "hidden", False)) and ledger is not None:
                    known = ledger.known_code_at(our_player, player, location, seq)
                    if known:
                        # A censored refresh erases runtime fields, not a
                        # message-proven identity. Only the printed facts are
                        # recoverable; never reuse the old private modifiers.
                        # The face-down banished pile is the same case: a card
                        # seen face up and then banished face down keeps its
                        # ledger slot, but the host's single-card refresh after
                        # the move blanks it.
                        if card_pool is None or known not in card_pool.cards:
                            raise ValueError(f"disclosed field code {known} is absent from cards.cdb")
                        restored = _static_card_state(CardState, card_pool.cards[known],
                            player, location, seq, position=position)
                        restored.owner = int(getattr(sc, "owner", -1))
                        # Materials stay attached to a monster turned face down (the core detaches
                        # none on a position change) and each came in through a public MSG_MOVE.
                        restored.overlay = tuple(int(m.code) for m in board.materials.get((player, location, seq), ()))
                        cards.append(restored)
                        continue
                cards.append(_card_state(
                    CardState,
                    getattr(sc, "controller", player), location,
                    getattr(sc, "sequence", seq),
                    code=sc.code, position=position,
                    hidden=bool(getattr(sc, "hidden", False)),
                    type=(getattr(sc, "type", 0) or 0) if live else (data.type if data else 0),
                    level=getattr(sc, "level", 0), rank=getattr(sc, "rank", 0),
                    attribute=((getattr(sc, "attribute", 0) or 0) if live
                               else (data.attribute if data else 0)),
                    race=((getattr(sc, "race", 0) or 0) if live
                          else (data.race if data else 0)),
                    attack=getattr(sc, "attack", 0),
                    defense=getattr(sc, "defense", 0),
                    status=getattr(sc, "status", 0),
                    lscale=getattr(sc, "lscale", 0), rscale=getattr(sc, "rscale", 0),
                    link=getattr(sc, "link", 0),
                    link_marker=getattr(sc, "link_marker", 0),
                    overlay=[m.code for m in board.materials.get(
                        (player, location, seq), ())],
                    # counters live on the board, not the card: the host's
                    # refresh flags carry neither QUERY_COUNTERS nor
                    # QUERY_OVERLAY_CARD, so both come from the message stream
                    counters=(board.counters_of(player, location, seq)
                              if hasattr(board, "counters_of")
                              else getattr(sc, "counters", None)),
                    alias=getattr(sc, "alias", 0),
                    # Stock host refresh flags omit QUERY_EQUIP_CARD,
                    # QUERY_TARGET_CARD and QUERY_OWNER. ShadowBoard rebuilds
                    # the first two from public messages and carries owner
                    # forward from the card's first public zone.
                    equip_card=(board.equip_target_of(player, location, seq)
                                if hasattr(board, "equip_target_of")
                                else getattr(sc, "equip_card", 0)),
                    targets=(board.targets_of(player, location, seq)
                             if hasattr(board, "targets_of")
                             else getattr(sc, "targets", ())),
                    owner=getattr(sc, "owner", -1),
                    # Base stats follow the same rule as type/attribute/race:
                    # the engine's runtime value when the host sent one, the
                    # database only for a card it never described.  The printed
                    # columns are not interchangeable with the runtime ones --
                    # a "?" ATK is stored as -2 and a Link monster's def column
                    # holds its markers.
                    base_attack=((getattr(sc, "base_attack", 0) or 0) if live
                                 else (data.attack if data else 0)),
                    base_defense=((getattr(sc, "base_defense", 0) or 0) if live
                                  else (data.defense if data else 0)),
                    setcode=(data.setcodes if data else ()),
                ))
            counts[(player, location)] = n

    if ledger is not None and hasattr(ledger, "known_counts"):
        known_disclosures = ledger.known_counts(our_player)
        faceup_extra = ledger.faceup_extra_counts(our_player)
    elif ledger is not None:
        known_disclosures = getattr(ledger, "counts", {})
        faceup_extra = {}
    else:
        known_disclosures = {}
        faceup_extra = {}

    # our own deck: the multiset the shadow board maintains.  mask_for sorts and
    # renumbers these, so the order we hand over does not reach the tokens.
    own_deck = []
    for code, k in board.our_remaining_deck().items():
        for _ in range(max(0, k)):
            own_deck.append(code)
    declared_own_deck = max(0, int(board.deck_count[our_player]))
    if len(own_deck) != declared_own_deck:
        unresolved = board.unresolved_own_deck_departures() if hasattr(board, "unresolved_own_deck_departures") else 0
        raise ValueError(
            "shadow own-deck multiset/count drift: "
            f"tracked={len(own_deck)} declared={declared_own_deck} "
            f"unresolved_anonymous_departures={unresolved}"
        )
    for seq, code in enumerate(own_deck):
        data = card_pool.cards.get(int(code)) if card_pool else None
        cards.append(_card_state(
            CardState, our_player, C.LOCATION_DECK, seq, code=code,
            position=C.POS_FACEDOWN,
            type=data.type if data else 0, level=data.level if data else 0,
            attribute=data.attribute if data else 0,
            race=data.race if data else 0,
            attack=data.attack if data else 0,
            defense=data.defense if data else 0,
            base_attack=data.attack if data else 0,
            base_defense=data.defense if data else 0,
            alias=data.alias if data else 0,
            setcode=(data.setcodes if data else ()),
        ))
    counts[(our_player, C.LOCATION_DECK)] = len(own_deck)

    # Rebuild all opponent hidden zones from this viewer's known name multiset.
    # Existing wire rows are removed first: RefreshExtra may contain a face-up
    # Pendulum and appending ``extra_count`` more rows would duplicate it.
    for location, n in (
        (C.LOCATION_HAND, max(0, int(counts[(opp, C.LOCATION_HAND)]))),
        (C.LOCATION_DECK, max(0, int(board.deck_count[opp]))),
        (C.LOCATION_EXTRA, max(0, int(board.extra_count[opp]))),
    ):
        wire_rows = [
            card for card in cards
            if card.controller == opp and card.location == location
        ]
        cards = [
            card for card in cards
            if not (card.controller == opp and card.location == location)
        ]
        cards.extend(_project_opponent_hidden_zone(
            CardState,
            card_pool,
            opp,
            location,
            n,
            wire_rows,
            known_disclosures,
            faceup_extra,
            anchors=(ledger.known_slots(our_player, opp, location)
                     if ledger is not None and hasattr(ledger, "known_slots") else None),
        ))
        counts[(opp, location)] = n

    return StateSnapshot(
        turn=int(turn), turn_player=int(board.turn_player), phase=int(board.phase),
        lp=(int(lp[0]), int(lp[1])), cards=cards, counts=counts, view=int(our_player),
    )


# ---------------------------------------------------------------------------
# The whole-game token stream and its KV cache
# ---------------------------------------------------------------------------


def _segments(ints: np.ndarray):
    """Split one pending token slice into cache-appendable chunks.

    A chunk is either exactly one complete state block -- which
    ``decode_state_block`` appends atomically and bidirectionally, occupying a
    single structural step -- or a maximal run of action/event/disclosure
    tokens, which ``decode_causal`` appends at one structural step each.  That
    is the same partition ``model.structural_steps`` induces on a teacher-forced
    whole sequence, which is why the two paths agree.
    """
    from wmtok import T

    n = len(ints)
    if not n:
        return []
    blk = ints[:, 12].astype(np.int64)
    ttype = ints[:, 0].astype(np.int64)
    state_like = (ttype == T.STATE) | (ttype == T.CARD) | (ttype == T.PERSIST)
    bounds = [0]
    bounds.extend(i for i in range(1, n) if blk[i] != blk[i - 1])
    bounds.append(n)
    out: list[list] = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        is_state = bool(state_like[start:end].any())
        if not is_state and out and not out[-1][2]:
            # consecutive non-state blocks are one causal run: the structural
            # step advances per token, not per block id
            out[-1][1] = end
        else:
            out.append([start, end, is_state])
    return [tuple(chunk) for chunk in out]


def _blank_item(ints: np.ndarray, nums: np.ndarray):
    """One token slice as a standalone :class:`Item` with no supervision.

    ``collate`` reads a fixed set of attributes unconditionally, so those get
    correctly shaped empty arrays; everything else it probes with ``getattr``
    and skips when absent.  This is a slice adapter, not a second serializer:
    every value that reaches it was produced by ``ItemBuilder``.
    """
    from wmserialize import R_MAX
    from wmtok import EFFECT_SLOT_CARD, Item, LEGAL_W, N_EV_FIELD

    item = Item()
    item.ints = ints
    item.nums = nums
    item.meta = {}
    item.legal_pos = np.zeros((0, R_MAX), np.int32)
    empty3 = np.zeros((0, R_MAX, LEGAL_W), np.uint8)
    item.legal_target = empty3
    item.legal_mask = empty3.copy()
    item.legal_unseen = empty3.copy()
    item.legal_neg = empty3.copy()
    item.legal_muted = empty3.copy()
    item.legal_effect = np.full((0, R_MAX, LEGAL_W), EFFECT_SLOT_CARD, np.int8)
    item.legal_effect_card = np.zeros((0, R_MAX, LEGAL_W), np.int32)
    item.policy_pos = np.zeros(0, np.int32)
    item.policy_target = np.zeros(0, np.int32)
    item.value_pos = np.zeros(0, np.int32)
    item.value_target = np.zeros(0, np.int8)
    item.value_on_play = np.zeros(0, np.int8)
    item.value_steps = np.zeros(0, np.int32)
    item.ev_pos = np.zeros(0, np.int32)
    item.ev_class = np.zeros(0, np.int8)
    item.ev_det = np.zeros(0, np.int8)
    item.ev_ptr_target = np.zeros(0, np.int32)
    item.ev_field_target = np.zeros((0, N_EV_FIELD), np.int16)
    item.ev_reason_target = np.zeros((0, 32), np.int8)
    item.ev_block = np.zeros((0, 2), np.int32)
    item.ev_weight = np.zeros(0, np.float32)
    return item


class DecisionRecord:
    """One on-policy decision, kept so a learner can replay it exactly.

    ``point`` indexes the decision points of the finished whole-game item and
    ``group`` its sub-prompt queries, so one teacher-forced forward over that
    item reproduces every logit the collector sampled from.  No activation is
    stored and nothing can drift.
    """

    __slots__ = ("kind", "point", "group", "cells", "chosen", "logp",
                 "entropy", "value", "menu_size", "msg", "turn")

    def __init__(self, kind, point, cells, chosen, logp, entropy, value,
                 menu_size, msg, turn, group=-1):
        self.kind = kind              # "top" | "sub"
        self.point = int(point)       # decision-point ordinal, or -1
        self.group = int(group)       # sub-query ordinal, or -1
        self.cells = tuple(cells)     # (row, col) per menu entry, (-1,-1)=unmatched
        self.chosen = int(chosen)     # index into the engine menu
        self.logp = float(logp)
        self.entropy = float(entropy)
        self.value = value            # (win, lose, draw) probabilities or None
        self.menu_size = int(menu_size)
        self.msg = int(msg)
        self.turn = int(turn)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind, "point": self.point, "group": self.group,
            "cells": [[int(a), int(b)] for a, b in self.cells],
            "chosen": self.chosen, "logp": self.logp, "entropy": self.entropy,
            "value": None if self.value is None else [float(v) for v in self.value],
            "menu_size": self.menu_size, "msg": self.msg, "turn": self.turn,
        }


class WholeGameStream:
    """One duel/seat whole-game sequence, encoded incrementally.

    The builder is append-only for the whole duel, so the tokens the model sees
    are the tokens the whole-game trainer would have built for this seat.
    ``encoded`` marks how much of that buffer already lives in the KV cache;
    everything past it is pending and gets pushed through
    ``prefill``/``decode_state_block``/``decode_causal`` at the next readout.

    Set ``use_cache=False`` to take the reference path instead: re-encode the
    whole prefix with one ordinary forward every time.  That is quadratic and
    exists only so the cache has something to be equal to.
    """

    def __init__(self, model, cfg, ctx, emb_source, device="cuda",
                 use_cache: bool = True, seat: int = 0):
        self.model, self.cfg, self.ctx = model, cfg, ctx
        self.emb_source, self.device = emb_source, device
        self.use_cache = bool(use_cache)
        self.seat = int(seat)
        self.builder = None
        self.persist = None
        self.block = None
        self.kv = None
        self.encoded = 0
        self.stats: Counter = Counter()

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        self.builder = None
        self.persist = None
        self.block = None
        self.kv = None
        self.encoded = 0

    def started(self) -> bool:
        return self.builder is not None

    def fork(self) -> "WholeGameStream":
        """Search-branch copy of the stream (the design notes).

        Shares the model and the KV dict -- the cache is functional, every
        append builds a new dict and never mutates the old one, so the
        parent and any number of forks can extend the same snapshot
        independently (pinned by ``BranchPurityTest`` in tests_wm.py).
        Deep-copies the mutable serializer state, so tokens the branch
        emits never appear in the parent's stream.
        """
        twin = self.__class__(
            self.model, self.cfg, self.ctx, self.emb_source,
            device=self.device, use_cache=self.use_cache, seat=self.seat,
        )
        # The same as fork_for_search: ctx/cfg/emb_source are read-only shared parts;
        # climbing into them through the builder in a deepcopy copies the whole card pool again (0.1-0.5 s each time,
        # once per branch of the snapshot engine: all the remaining tree cost in one measured run).
        memo = {id(self.cfg): self.cfg, id(self.ctx): self.ctx,
                id(self.emb_source): self.emb_source,
                id(self.model): self.model}
        if isinstance(self.ctx, dict):
            for _v in self.ctx.values():
                memo.setdefault(id(_v), _v)
        twin.builder = copy.deepcopy(self.builder, memo)
        twin.persist = copy.deepcopy(self.persist, memo)
        twin.block = copy.deepcopy(self.block, memo)
        twin.kv = self.kv
        twin.encoded = self.encoded
        return twin

    def open(self, menu, row: int = 0) -> None:
        from wmserialize import ItemBuilder, make_chain_persist

        self.builder = ItemBuilder(self.cfg, self.ctx)
        self.builder.header(menu, row)
        self.persist = make_chain_persist(self.ctx)
        self.block = None
        self.kv = None
        self.encoded = 0

    @property
    def pending(self) -> int:
        return 0 if self.builder is None else self.builder.buf.n - self.encoded

    def tokens(self) -> int:
        return 0 if self.builder is None else self.builder.buf.n

    def finish(self, meta: dict):
        if self.builder is None:
            return None
        return self.builder.finish(dict(meta), allow_no_points=True)

    # -- appending ---------------------------------------------------------

    def public(self, messages, player: int) -> None:
        """Append the public wire history received since the last append."""
        if self.builder is None or self.block is None or not messages:
            return
        before = self.builder.buf.n
        self.builder.public_history(messages, self.block, player)
        self.stats["public_tokens"] += self.builder.buf.n - before

    def sync_rule_state(self, checkpoint) -> None:
        """Adopt the public rule memory the wire tracker maintains.

        ``PublicRuleTracker`` is the deployable source of truth for turn state
        and persistent-effect bookkeeping.  The builder's own message handling
        stays, but the tracker's view wins at every decision point, exactly as
        it did when every decision rebuilt a fresh window.
        """
        if self.builder is None or checkpoint is None:
            return
        self.builder.turn_state = copy.deepcopy(checkpoint["turn_state"])
        self.persist.restore(checkpoint["persist"])
        self.builder._live_current_link = int(checkpoint["current_link"])
        self.builder._live_chain_meta = {
            int(link): (int(meta[0]), int(meta[2]), int(meta[3]))
            for link, meta in checkpoint["chain_meta"].items()
        }

    def sub_token(self, segment) -> None:
        """Append one answered private prompt exactly as the corpus writes it.

        The token fields come out of ``wmserialize._sub_rows`` -- the function
        the corpus itself uses -- fed a one-row view of the choice just made, so
        a live ``SUB`` cannot drift away from a trained one.
        """
        from wmserialize import _sub_rows
        from wmtok import F_ON_PLAY, F_TURN_SELF, T

        candidate = segment["candidate"]
        player = int(segment["player"])
        base_flags = (F_ON_PLAY if player == 0 else 0) | (
            F_TURN_SELF if int(segment["turn_player"]) == player else 0
        )
        lists = {
            "sub_msg": [int(segment["msg"])],
            "sub_choice": [int(segment["choice"])],
            "sub_stage": [int(segment.get("stage", 0))],
            "sub_link": [int(segment.get("link", 0))],
            "sub_player": [player],
            "sub_at": [int(candidate.at)],
            "sub_code": [int(candidate.code)],
            "sub_value": [int(candidate.value)],
            "sub_n": [int(segment["n"])],
            "sub_trace": [-1],
        }
        for _stage, _link, kwargs in _sub_rows(
                lists, self.block, player, base_flags, self.ctx):
            kwargs = dict(kwargs)
            kwargs.pop("_trace", None)
            self.builder.buf.add(T.SUB, **kwargs)
        self.stats["sub_tokens"] += 1

    # -- one top-level decision -------------------------------------------

    def open_point(self, menu, row: int = 0):
        """Emit the decision's state block and read the heads off it.

        Returns ``(block, pending, out)`` where ``pending`` is the token the
        caller hands back to :meth:`close_point`.  The trailing blank ``ACT``
        deliberately stays out of the cache.
        """
        checkpoint = self.builder.checkpoint()
        start = self.builder.buf.n
        block = self.builder.point(
            menu, row, None, None, self.persist,
            supervise_legal=True, supervise_settle=False,
        )
        act = self.builder.buf.n - 1
        out = self._flush(stop=act, dp_point=len(self.builder.dp_pos) - 1)
        return block, (menu, row, checkpoint, start, act), out

    def close_point(self, pending, candidate: int = -1, verify: bool = True):
        """Rewrite the decision through the builder with the action we sent."""
        menu, row, checkpoint, start, act = pending
        buf = self.builder.buf
        before = buf.ints[start:act].copy() if verify else None
        self.builder.restore(checkpoint)
        block = self.builder.point(
            menu, row, None, None, self.persist,
            supervise_legal=True, supervise_settle=False,
            act_override=(int(candidate) if candidate >= 0 else None),
        )
        if buf.n - 1 != act:
            raise AssertionError(
                f"second pass of one decision emitted {buf.n - start} tokens, "
                f"first pass emitted {act + 1 - start}"
            )
        if verify and not np.array_equal(before, buf.ints[start:act]):
            raise AssertionError(
                "the state block is not reproducible across the two passes of "
                "one decision point; the KV cache would then hold tokens the "
                "sequence no longer contains"
            )
        # the state block is already cached; only the resolved ACT is pending
        self.encoded = act
        self.block = block
        return block

    # -- private prompts ---------------------------------------------------

    def sub_anchor(self) -> int:
        """The token a private prompt attaches to, or -1 when there is none.

        The anchor has to be inside the chunk about to be encoded, because that
        is the only place the readout can find its hidden state.  Trailing
        ``EV_END`` is skipped, matching the corpus anchor rule.
        """
        from wmtok import T

        if self.builder is None:
            return -1
        anchor = self.builder.buf.n - 1
        while anchor >= 0 and int(self.builder.buf.ints[anchor, 0]) == int(T.EV_END):
            anchor -= 1
        return anchor if anchor >= self.encoded else -1

    def score_private(self, anchor: int, stage: int, link: int, msg: int,
                      candidates, keep: bool = False):
        """Run the sub-policy head over the engine's real private candidates."""
        from wmserialize import resolve_effect_index
        from wmtok import EFFECT_SLOT_CARD

        builder = self.builder
        group = len(builder.subq_pos)
        n = len(candidates)
        builder.subq_pos.append(int(anchor))
        builder.subq_stage.append(1 if int(stage) else 0)
        builder.subq_link.append(min(max(int(link), 0), 15))
        builder.subq_has.append(1)
        builder.subq_msg.append(min(max(int(msg), 0), 63))
        builder.subq_known.append(1)
        for candidate in candidates:
            at = int(candidate.at)
            model_at = 0
            if at:
                model_at = (
                    (1 if (at & 0xFF) == self.seat else 2)
                    | (((at >> 8) & 0xFF) << 8)
                    | (((at >> 16) & 0xFF) << 16)
                )
            builder.subcand_group.append(group)
            builder.subcand_card.append(
                self.ctx["code_index"].get(int(candidate.code))
                if candidate.code else 0
            )
            builder.subcand_effect.append(
                resolve_effect_index(
                    self.ctx, int(candidate.code), int(candidate.desc)
                ) if candidate.code and candidate.desc else EFFECT_SLOT_CARD
            )
            builder.subcand_at.append(model_at)
            builder.subcand_value.append(int(candidate.value) & 0xFFFFFFFF)
            builder.subcand_finish.append(1 if candidate.finish else 0)
            builder.subcand_target.append(1)
            builder.subcand_chosen.append(0)
        out = self._flush(sub_group=group)
        logits = None if out is None else out.get("sub_policy")
        if not keep:
            self._drop_sub_query(group, n)
            group = -1
        if logits is None:
            return None, group
        return logits.float().cpu().numpy(), group

    def sub_rows_of(self, group: int) -> list[int]:
        return [k for k, value in enumerate(self.builder.subcand_group)
                if int(value) == group]

    def set_sub_chosen(self, group: int, chosen: int) -> None:
        for offset, k in enumerate(self.sub_rows_of(group)):
            self.builder.subcand_chosen[k] = 1 if offset == chosen else 0

    def _drop_sub_query(self, group: int, n: int) -> None:
        builder = self.builder
        for name in ("subq_pos", "subq_stage", "subq_link", "subq_has",
                     "subq_msg", "subq_known"):
            del getattr(builder, name)[group:]
        keep = len(builder.subcand_group) - n
        for name in ("subcand_group", "subcand_card", "subcand_effect",
                     "subcand_at", "subcand_value", "subcand_finish",
                     "subcand_target", "subcand_chosen"):
            del getattr(builder, name)[keep:]

    # -- the model ---------------------------------------------------------

    def _flush(self, stop: int | None = None, dp_point: int = -1,
               sub_group: int = -1):
        """Push pending tokens through the model; return the readout chunk."""
        import torch

        builder = self.builder
        end = builder.buf.n if stop is None else int(stop)
        if end <= self.encoded:
            return None
        if not self.use_cache:
            item = _blank_item(
                builder.buf.ints[:end].copy(), builder.buf.nums[:end].copy()
            )
            if dp_point >= 0:
                self._attach_point(item, dp_point, 0)
            if sub_group >= 0:
                self._attach_sub(item, sub_group, 0)
            with torch.no_grad():
                out = self.model(self._batch(item))
            self.encoded = end
            self.stats["full_forwards"] += 1
            return out

        ints = builder.buf.ints[self.encoded:end]
        nums = builder.buf.nums[self.encoded:end]
        chunks = _segments(ints)
        out = None
        for index, (a, b, is_state) in enumerate(chunks):
            last = index == len(chunks) - 1
            base = self.encoded + a
            item = _blank_item(ints[a:b].copy(), nums[a:b].copy())
            if last and dp_point >= 0:
                self._attach_point(item, dp_point, base)
            if last and sub_group >= 0:
                self._attach_sub(item, sub_group, base)
            batch = self._batch(item)
            with torch.no_grad():
                if self.kv is None:
                    out = self.model.prefill(batch)
                elif is_state:
                    out = self.model.decode_state_block(batch, self.kv)
                else:
                    out = self.model.decode_causal(batch, self.kv)
            self.kv = out["kv_cache"]
            self.stats["chunks"] += 1
            self.stats["state_chunks" if is_state else "causal_chunks"] += 1
        self.encoded = end
        self.stats["encoded_tokens"] = self.encoded
        return out

    def _attach_point(self, item, point: int, base: int) -> None:
        builder = self.builder
        pos = np.asarray(builder.dp_pos[point], dtype=np.int32).copy()
        pos[pos >= 0] -= base
        item.legal_pos = pos[None, :]
        item.legal_target = builder.dp_target[point][None, ...]
        item.legal_mask = builder.dp_mask[point][None, ...]
        item.legal_unseen = builder.dp_unseen[point][None, ...]
        item.legal_neg = builder.dp_neg[point][None, ...]
        item.legal_muted = builder.dp_muted[point][None, ...]
        item.legal_effect = builder.dp_effect[point][None, ...]
        item.legal_effect_card = builder.dp_effect_card[point][None, ...]
        item.policy_target = np.asarray([builder.dp_policy[point]], np.int32)
        item.value_target = np.asarray([builder.dp_value[point]], np.int8)
        item.value_on_play = np.asarray([builder.dp_on_play[point]], np.int8)
        item.value_steps = np.asarray([builder.dp_steps[point]], np.int32)
        glob = np.asarray([builder.dp_global[point] - base], np.int32)
        item.policy_pos = glob
        item.value_pos = glob

    def _attach_sub(self, item, group: int, base: int) -> None:
        builder = self.builder
        rows = self.sub_rows_of(group)
        item.subq_pos = np.asarray([builder.subq_pos[group] - base], np.int32)
        item.subq_stage = np.asarray([builder.subq_stage[group]], np.int8)
        item.subq_link = np.asarray([builder.subq_link[group]], np.int8)
        item.subq_has = np.asarray([builder.subq_has[group]], np.int8)
        item.subq_msg = np.asarray([builder.subq_msg[group]], np.int16)
        item.subq_known = np.asarray([builder.subq_known[group]], np.int8)
        item.subcand_group = np.zeros(len(rows), np.int32)
        item.subcand_card = np.asarray(
            [builder.subcand_card[k] for k in rows], np.int32)
        item.subcand_effect = np.asarray(
            [builder.subcand_effect[k] for k in rows], np.int8)
        item.subcand_at = np.asarray(
            [builder.subcand_at[k] for k in rows], np.uint32)
        item.subcand_value = np.asarray(
            [builder.subcand_value[k] for k in rows], np.uint32)
        item.subcand_finish = np.asarray(
            [builder.subcand_finish[k] for k in rows], np.int8)
        item.subcand_target = np.ones(len(rows), np.uint8)
        item.subcand_chosen = np.asarray(
            [builder.subcand_chosen[k] for k in rows], np.uint8)

    def _batch(self, item):
        from data import collate

        batch = collate([item], self.emb_source, len(item.ints))
        return {
            key: (value.to(self.device) if hasattr(value, "to") else value)
            for key, value in batch.items()
        }


# ---------------------------------------------------------------------------


class WorldModelPolicy(Policy):
    """Masked softmax over the policy head, restricted to the engine's menu."""

    name = "worldmodel"

    def __init__(self, model, cfg, ctx, texts, emb_source, card_pool,
                 device="cuda", fallback: str = "first", banlists=None,
                 sample: bool = False, temperature: float = 1.0,
                 rng=None, use_cache: bool = True, keep_item: bool = False,
                 verify_every: int = 64):
        self.model, self.cfg, self.ctx = model, cfg, ctx
        self.texts, self.emb_source, self.card_pool = texts, emb_source, card_pool
        self.device, self.fallback = device, fallback
        self.our_player = 0
        self.sample = bool(sample)
        self.temperature = float(temperature)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.keep_item = bool(keep_item)
        self.verify_every = int(verify_every)
        self.stream = WholeGameStream(
            model, cfg, ctx, emb_source, device=device, use_cache=use_cache
        )
        self._ordinal = 0
        self._public_messages: list = []
        self._since_action: list = []
        self._pending = None
        self._rule_tracker = None
        self.stats: Counter = Counter()
        self.values: list[float] = []
        #: hash -> Banlist or {card_code: 0/1/2}.  A nonzero room hash must be
        #: resolved; feeding a hash alone would teach no transferable meaning.
        self.banlists = dict(banlists or {})
        self.banlist_id = 0
        self.banlist: tuple[tuple[int, int], ...] = ()
        #: Last distribution in the exact order of ``DecisionState.actions``.
        #: Unmatched entries get probability zero; this is observability, not
        #: another action generator.
        self.last_action_probs: np.ndarray | None = None
        #: On-policy trace and finished sequence; only filled when ``keep_item``.
        self.decisions: list[DecisionRecord] = []
        self.item = None
        #: ``NetDuelClient`` appends every ``(msg, body)`` it receives here when
        #: the policy exposes the attribute.  This is the lossless capsule for
        #: our seat: with the duel seed it re-derives the whole duel, so a later
        #: schema change can be re-exported instead of replayed.
        self.capture: list = []

    # -- lifecycle ---------------------------------------------------------

    def bind(self, client) -> None:
        from wmserialize import PublicRuleTracker

        self.our_player = int(client.result.our_player)
        self._client = client
        self.stream.reset()
        self.stream.seat = self.our_player
        self._ordinal = 0
        self._pending = None
        self._public_messages.clear()
        self._since_action.clear()
        self.decisions = []
        self.item = None
        self.capture.clear()
        self._rule_tracker = PublicRuleTracker(self.ctx)
        self.last_action_probs = None
        self.last_belief = None
        host = getattr(client, "host_info", None)
        self.banlist_id = int(getattr(host, "lflist", 0) or 0)
        raw = self.banlists.get(self.banlist_id)
        if self.banlist_id and raw is None:
            raise ValueError(
                f"room banlist 0x{self.banlist_id:08x} is not loaded; "
                "resolve HostInfo.lflist against the host's lflist.conf"
            )
        limits = getattr(raw, "limits", raw) if raw is not None else {}
        self.banlist = tuple(sorted(
            (int(code), int(limit)) for code, limit in dict(limits).items()
            if 0 <= int(limit) < 3
        ))
        if not self.banlist:
            # A section with no rows still hashes to the seed constant, so an
            # "Unlimited" room reports a nonzero id with nothing behind it.
            # The serializer refuses that pairing, and rightly: a hash carries
            # no transferable meaning.  No restrictions is banlist 0.
            self.banlist_id = 0

    def fork_for_search(self) -> "WorldModelPolicy":
        """A branch of this policy for BPTS particle rollouts.

        The branch continues serializing *simulated* engine messages from a
        particle, so everything the serializer reads must be a private copy,
        while everything heavy or immutable is shared:

        - shared: weights, config, embeddings, card pool, banlist tables,
          and the KV dict (functional -- appends never mutate the parent);
        - deep-copied: the token stream, chain persist, rule tracker and
          pending-message buffers;
        - severed: episode records, capsule capture, values and stats.  A
          hypothetical line is not experience and must never be trained on
          or archived; the fork starts those containers empty and nothing
          ever reads them back.
        """
        twin = object.__new__(WorldModelPolicy)
        for name in ("model", "cfg", "ctx", "texts", "emb_source",
                     "card_pool", "device", "fallback", "temperature",
                     "banlists", "verify_every"):
            setattr(twin, name, getattr(self, name))
        twin.our_player = self.our_player
        twin.sample = False           # rollouts act greedily; search explores
        twin.rng = np.random.default_rng(0)
        twin.keep_item = False
        twin.stream = self.stream.fork()
        twin._ordinal = self._ordinal
        # The deep copy must avoid large shared objects reached through ctx / rule trackers (card_pool is
        # the whole card pool database: 120 ms each time, measured to drag each snapshot-engine node to 0.2-0.7 s).
        # The memo is preloaded with these instances: mutable state is copied, read-only lookup tables are shared.
        memo = {id(self.ctx): self.ctx, id(self.card_pool): self.card_pool,
                id(self.texts): self.texts,
                id(self.emb_source): self.emb_source}
        # The values of ctx (code_index/rule_code/desc_map/count_limit_index...) are
        # read-only tables from load time, and structures such as rule trackers hold references to them directly; the twin
        # shares the whole ctx with the parent policy anyway, and these values are shared by reference likewise
        if isinstance(self.ctx, dict):
            for _v in self.ctx.values():
                memo.setdefault(id(_v), _v)
        twin._public_messages = copy.deepcopy(self._public_messages, memo)
        twin._since_action = copy.deepcopy(self._since_action, memo)
        twin._pending = copy.deepcopy(self._pending, memo)
        twin._rule_tracker = copy.deepcopy(self._rule_tracker, memo)
        twin.stats = Counter()
        twin.values = []
        twin.banlist_id = self.banlist_id
        twin.banlist = self.banlist
        twin.last_action_probs = None
        twin.decisions = []
        twin.item = None
        twin.capture = []
        twin._client = None
        return twin

    def on_duel_end(self, result) -> None:
        if self.keep_item and self.stream.started():
            self.item = self.stream.finish({
                "game": "live", "seat": self.our_player,
                "kind": "trajectory", "deck": "",
            })
        self.stream.reset()
        self._public_messages.clear()
        self._since_action.clear()
        self._pending = None
        self._rule_tracker = None

    def observe_game_message(self, msg: int, body: bytes) -> None:
        """Keep exactly the public wire history received by this client."""
        from mirrorforce.puzzle.messages import Message

        if int(msg) not in PUBLIC_HISTORY_MSGS:
            return
        message = Message(int(msg), bytes(body))
        self._public_messages.append(message)
        self._since_action.append(message)
        self.stats["public_wire_messages"] += 1

    # -- one decision ------------------------------------------------------

    def choose(self, state: DecisionState) -> int:
        self.last_action_probs = None
        self.stats["prompts"] += 1
        top_level = state.msg in DECISION_MSGS
        if state.n <= 1:
            self.stats["trivial"] += 1
            if (not top_level
                    and getattr(self.model.cfg, "sub_policy", False)
                    and state.n == 1):
                self._commit_private_choice(state, 0)
            return 0
        if not top_level:
            return self._choose_private(state)
        try:
            scores, cells, point, value = self._score(state)
        except Exception as exc:
            self.stats["score_error"] += 1
            self.stats[f"err:{type(exc).__name__}"] += 1
            # A broken sequence cannot be repaired by answering anyway: the KV
            # cache and the builder would disagree from here on.  Fail loudly.
            raise
        if scores is None or not np.isfinite(scores).any():
            self.stats["no_scores" if scores is None else "all_unmatched"] += 1
            self._commit_choice(0)
            return 0
        scores[~np.isfinite(scores)] = -1e9
        probs = self._distribution(scores)
        best = self._pick(probs)
        self.last_action_probs = probs
        self._record("top", point, cells, best, probs, value, state)
        self._commit_choice(best)
        self.stats["model_decisions"] += 1
        return best

    def _choose_private(self, state: DecisionState) -> int:
        if getattr(self.model.cfg, "sub_policy", False):
            try:
                scores, group = self._score_private(state)
            except Exception as exc:
                self.stats["private_score_error"] += 1
                self.stats[f"private_err:{type(exc).__name__}"] += 1
                self._commit_private_choice(state, 0)
                return 0
            if scores is not None and len(scores) == state.n:
                scores = np.asarray(scores, dtype=np.float64)
                if not np.isfinite(scores).any():
                    self.stats["private_all_nonfinite"] += 1
                    self._commit_private_choice(state, 0, group)
                    return 0
                scores[~np.isfinite(scores)] = -1e9
                probs = self._distribution(scores)
                best = self._pick(probs)
                self.last_action_probs = probs
                self._record("sub", -1, (), best, probs, None, state, group)
                self._commit_private_choice(state, best, group)
                self.stats["private_model_decisions"] += 1
                return best
        self.stats["out_of_scope"] += 1
        self.stats[f"scope_msg_{state.msg}"] += 1
        fallback = 0 if self.fallback == "first" else state.n - 1
        self._commit_private_choice(state, fallback)
        return fallback

    def _distribution(self, scores: np.ndarray) -> np.ndarray:
        temperature = max(self.temperature, 1e-6)
        shifted = (scores - float(np.max(scores))) / temperature
        probs = np.exp(shifted)
        total = float(probs.sum())
        if not np.isfinite(total) or total <= 0:
            return np.full(len(scores), 1.0 / len(scores))
        return probs / total

    def _pick(self, probs: np.ndarray) -> int:
        forced = getattr(self, "forced_action", None)
        if forced is not None:
            # One-shot override for the live searcher: the action was chosen
            # by search, but it goes out through the normal scoring path so
            # the stream, the decision record and the recorded probabilities
            # stay exactly what the policy actually computed.  The learner
            # excludes these decisions from the PPO surrogate (they are
            # off-policy by construction) and teaches them by distillation.
            self.forced_action = None
            if 0 <= int(forced) < len(probs):
                return int(forced)
        if not self.sample:
            return int(np.argmax(probs))
        return int(self.rng.choice(len(probs), p=probs))

    def _record(self, kind, point, cells, chosen, probs, value, state,
                group: int = -1) -> None:
        if not self.keep_item:
            return
        safe = np.clip(probs, 1e-12, 1.0)
        self.decisions.append(DecisionRecord(
            kind=kind, point=point, cells=cells, chosen=chosen,
            logp=float(np.log(safe[chosen])),
            entropy=float(-(safe * np.log(safe)).sum()),
            value=value, menu_size=state.n, msg=state.msg, turn=state.turn,
            group=group,
        ))

    def _commit_choice(self, action_index: int) -> None:
        """Close the sequence: rewrite the decision with the action we sent."""
        pending, self._pending = self._pending, None
        if pending is None:
            return
        stream_pending, mapping = pending
        candidate = -1
        if 0 <= action_index < len(mapping):
            candidate = int(mapping[action_index])
        if candidate < 0:
            self.stats["history_action_unmatched"] += 1
        else:
            self.stats["history_action_recorded"] += 1
        n = self.stats["model_decisions"]
        self.stream.close_point(
            stream_pending, candidate,
            verify=(self.verify_every <= 1 or n < 8
                    or n % self.verify_every == 0),
        )
        self._since_action.clear()

    def _commit_private_choice(self, state: DecisionState, action_index: int,
                               group: int = -1) -> None:
        if not 0 <= int(action_index) < state.n:
            return
        if not self.stream.started() or self.stream.block is None:
            return
        from mirrorforce.worldmodel.generate import _sub_candidate, _sub_context

        stage, link = _sub_context(self._since_action)
        segment = {
            "candidate": _sub_candidate(
                state.actions[int(action_index)], self.our_player
            ),
            "choice": int(action_index),
            "n": int(state.n),
            "msg": int(state.msg),
            "player": int(self.our_player),
            "turn_player": _turn_player(state),
            "stage": stage,
            "link": link,
        }
        if group >= 0:
            self.stream.set_sub_chosen(group, int(action_index))
        self.stream.sub_token(segment)
        self.stats["private_history_recorded"] += 1

    # -- the encoder path --------------------------------------------------

    def _score(self, state: DecisionState):
        import pyarrow as pa
        from wmserialize import MenuTable, R_MAX
        from wmtok import LEGAL_W
        from mirrorforce.worldmodel.candidates import (
            ActionKind, candidate_index_for_action, enumerate_candidates,
        )
        from mirrorforce.worldmodel.state import mask_for, public_view
        from mirrorforce.worldmodel.writer import menu_batch

        board = state.board
        # One of two board sources, and **masking happens only in the mask_for below**:
        # the online path derives it from ShadowBoard, offline particles give a captured
        # StateSnapshot directly. Both have the same type, so the code after this does not branch at all.
        full = state.snapshot
        if full is None:
            full = snapshot_from_shadow(
                board, self.our_player, state.lp, state.turn, self.card_pool)
        # Cards the duel has published stay identified: mask_for un-blanks
        # anything the disclosure ledger resolves, and without it a hand trap
        # the opponent announced goes back to being an anonymous card in their
        # hand.  Resolution must be done against the **current** board: hand / deck are vectors, and the engine silently
        # reorders sequences when it removes a card without sending MSG_MOVE, so comparing with coordinates from the activation moment always mismatches.
        ledger = (state.disclosure if state.disclosure is not None
                  else getattr(board, "disclosure", None))
        # Both are needed: identities with positions go through ``revealed``, those without through
        # ``revealed_unpositioned``. Taking only the first would lose "still knowing which cards after a hand shuffle",
        # while the offline path takes them with ``public_view``, and the two would disagree.
        snap = (public_view(full, self.our_player, ledger)
                if ledger is not None
                else mask_for(full, self.our_player, None))
        cands = enumerate_candidates(snap, self.our_player, self.texts)
        if not cands:
            self.stats["no_candidates"] += 1
            return None, (), -1, None

        # candidate -> engine menu index, through the key the corpus matched on
        index_of: dict[tuple, int] = {}
        for i, c in enumerate(cands):
            index_of.setdefault(c.key, i)
        cand_of_action = [
            candidate_index_for_action(cands, index_of, action)
            for action in state.actions
        ]
        # The training corpus finds the Lua effect registration index from the engine's full desc through desc_map. The live path used
        # to write the whole column as 0, so the policy's per-effect action queries in real games all degenerated into whole-card
        # aggregate vectors. Only descs the authoritative menu has made public are filled into their candidates; candidates not
        # enumerated / not matched stay 0, and the serializer falls back safely to the whole card.
        descs = [
            ((int(c.code) << 4) | int(c.eff_slot))
            if c.kind is ActionKind.ACTIVATE_EFFECT
            and c.code and c.eff_slot >= 0 else 0
            for c in cands
        ]
        for action, candidate_index in zip(state.actions, cand_of_action):
            if candidate_index >= 0:
                descs[candidate_index] = int(getattr(action, "desc", 0) or 0)
        matched = sum(1 for x in cand_of_action if x >= 0)
        self.stats["menu_entries"] += state.n
        self.stats["menu_matched"] += matched
        if matched == 0:
            self.stats["menu_all_missed"] += 1
            return None, (), -1, None

        sample = self._sample(state, snap, cands, descs, self._ordinal)
        batch = menu_batch([sample])
        table = (
            pa.Table.from_batches([batch])
            if isinstance(batch, pa.RecordBatch) else batch
        )
        menu = MenuTable(table)

        public = tuple(self._public_messages)
        self._public_messages.clear()
        self._rule_tracker.observe(public, int(state.turn))
        if not self.stream.started():
            self.stream.open(menu, 0)
        else:
            self.stream.public(public, self.our_player)
        self.stream.sync_rule_state(self._rule_tracker.checkpoint())

        point = len(self.stream.builder.dp_pos)
        block, stream_pending, out = self.stream.open_point(menu, 0)
        self._pending = (stream_pending, cand_of_action)
        self._ordinal += 1
        if out is None or "policy" not in out:
            self.stats["no_readout"] += 1
            return None, (), -1, None
        logit = out["policy"][-1].float().cpu().numpy()
        legal = out["legal"][-1].float().cpu().numpy()
        value = out["value"][-1].float().softmax(-1).cpu().numpy()
        self.values.append(float(value[0] - value[1]))
        # Posterior of the opponent's hand (belief head, card row -> probability), used by the search server to weight particles.
        # On checkpoints without the head it stays None, and weighting falls back to uniform.
        belief = out.get("belief")
        self.last_belief = (belief[-1].float().sigmoid().cpu().numpy()
                            if belief is not None else None)

        scores = np.full(state.n, -np.inf, dtype=np.float64)
        cells: list[tuple[int, int]] = [(-1, -1)] * state.n
        c2o = np.asarray(block.cand_to_out)
        for a, ci in enumerate(cand_of_action):
            if ci < 0 or ci >= len(c2o):
                continue
            r, c = int(c2o[ci][0]), int(c2o[ci][1])
            if not (0 <= r < R_MAX and 0 <= c < LEGAL_W):
                self.stats["out_of_grid"] += 1
                continue
            if legal[r, c] <= 0:
                self.stats["model_says_illegal"] += 1
            # Every ``state.actions`` entry came from the authoritative engine
            # menu and is legal.  The learned legality head is measured here,
            # but it must never change the policy.
            scores[a] = float(logit[r, c])
            cells[a] = (r, c)
        return scores, tuple(cells), point, tuple(float(v) for v in value)

    def _score_private(self, state: DecisionState):
        """Score an actual private prompt with the supervised sub-policy head."""
        if not self.stream.started() or self.stream.block is None:
            return None, -1
        from mirrorforce.worldmodel.generate import _sub_candidate, _sub_context

        public = tuple(self._public_messages)
        self._public_messages.clear()
        if public:
            self._rule_tracker.observe(public, int(state.turn))
            self.stream.public(public, self.our_player)
        anchor = self.stream.sub_anchor()
        if anchor < 0:
            self.stats["private_no_anchor"] += 1
            return None, -1
        stage, link = _sub_context(self._since_action)
        candidates = [
            _sub_candidate(action, self.our_player) for action in state.actions
        ]
        return self.stream.score_private(
            anchor, stage, link, state.msg, candidates, keep=self.keep_item
        )

    def _sample(self, state, snap, cands, descs, ordinal: int):
        from mirrorforce.worldmodel.generate import MenuSample

        n = len(cands)
        return MenuSample(
            game_id="live", decision_id=f"live:{ordinal}", parent_id=None,
            source="trajectory", forced_pass=False,
            player=self.our_player, first_player=0,
            on_play=(self.our_player == 0),
            msg=int(state.msg), turn=int(state.turn),
            turn_player=_turn_player(state),
            phase=int(state.phase), response_index=0, decision_ordinal=ordinal,
            state=snap, candidates=cands,
            labels=[False] * n, neg_class=[0] * n, unseen=[False] * n,
            desc=list(descs), chosen_index=-1, menu_size=state.n,
            n_missed=0, n_duplicate=0, n_system_desc=0, truncated=False,
            banlist_id=self.banlist_id, banlist=self.banlist,
        )

    # -- reporting ---------------------------------------------------------

    def report(self) -> dict:
        s = dict(self.stats)
        s.update({f"stream_{k}": v for k, v in self.stream.stats.items()})
        me, mm = s.get("menu_entries", 0), s.get("menu_matched", 0)
        s["menu_match_rate"] = round(mm / me, 4) if me else 0.0
        p = s.get("prompts", 0)
        driven = s.get("model_decisions", 0) + s.get(
            "private_model_decisions", 0
        )
        s["model_driven_share"] = round(driven / p, 4) if p else 0.0
        if self.values:
            s["mean_value"] = round(float(np.mean(self.values)), 4)
        return s


# ---------------------------------------------------------------------------


def config_from_checkpoint(blob: dict, card_dim: int, fallback_max_len: int = 1024,
                           consequence_ablation: str = "none",
                           consequence_ablation_seed: int = 20260828):
    """Rebuild ``Config`` from the training checkpoint, without guessing.

    Old v0 checkpoints did not write ``pos_mode``; absence therefore means the
    historical ``absolute`` mode.  New checkpoints store it both at top level
    and inside ``cfg``.  A disagreement is corruption/config drift and must be
    loud rather than selecting one silently.
    """
    from model import Config

    saved = blob.get("cfg") or {}
    top = blob.get("pos_mode")
    inner = saved.get("pos_mode")
    if top is not None and inner is not None and top != inner:
        raise ValueError(
            f"checkpoint pos_mode disagrees: top-level={top!r}, cfg={inner!r}"
        )
    pos_mode = top or inner or "absolute"
    contract = blob.get("actor_input_contract")
    if contract not in (None, "bot-public-v1"):
        raise ValueError(
            f"checkpoint actor_input_contract={contract!r} is not deployable"
        )
    return Config(
        d_model=int(saved.get("d_model", 512)),
        n_layer=int(saved.get("n_layer", 12)),
        n_head=int(saved.get("n_head", 8)),
        max_len=int(saved.get("max_len", fallback_max_len)),
        card_dim=card_dim,
        pos_mode=pos_mode,
        attn_mode=saved.get("attn_mode") or None,
        rope_theta=float(saved.get("rope_theta", 10000.0)),
        state_aux=bool(saved.get("state_aux", False)),
        policy_identity=bool(saved.get("policy_identity", False)),
        policy_id_dropout=float(saved.get("policy_id_dropout", 0.5)),
        relation_endpoints=bool(saved.get("relation_endpoints", False)),
        recurrent_memory=bool(saved.get("recurrent_memory", False)),
        legal_effect_interaction_rank=int(
            saved.get("legal_effect_interaction_rank", 0)
        ),
        legal_effect_interaction_init=str(
            saved.get("legal_effect_interaction_init", "zero_gate")
        ),
        legal_family_presence=bool(
            saved.get("legal_family_presence", False)
        ),
        legal_family_presence_decode=bool(
            saved.get("legal_family_presence_decode", True)
        ),
        card_op_role_residual=bool(
            saved.get("card_op_role_residual", False)
        ),
        card_identity=bool(saved.get("card_identity", False)),
        card_identity_dropout=float(
            saved.get("card_identity_dropout", 0.25)
        ),
        card_identity_scale=float(saved.get("card_identity_scale", 1.0)),
        card_name_residual=bool(saved.get("card_name_residual", False)),
        card_name_dropout=float(saved.get("card_name_dropout", 0.25)),
        sub_policy=bool(saved.get("sub_policy", False)),
        # The distilled arm carries an extra permission readout.  It is gated
        # on the flag rather than on the presence of the tensor, so a
        # checkpoint that has one and a config that does not fails loudly at
        # ``load_state_dict`` instead of silently dropping the head.
        distill_flags=bool(saved.get("distill_flags", False)),
        # R2 explicit coupling.  Same rule as ``distill_flags``: rebuild from
        # the recorded flag, never from whether the tensors happen to be there,
        # so a coupled checkpoint loaded by an uncoupled config fails at
        # ``load_state_dict`` instead of quietly dropping z and scoring the
        # policy off the base path alone.
        consequence_query=bool(saved.get("consequence_query", False)),
        consequence_dim=int(saved.get("consequence_dim", 128)),
        consequence_layers=int(saved.get("consequence_layers", 1)),
        consequence_heads=int(saved.get("consequence_heads", 4)),
        consequence_candidates=int(saved.get("consequence_candidates", 16)),
        consequence_gate=float(saved.get("consequence_gate", 0.1)),
        consequence_policy_mixer=bool(
            saved.get("consequence_policy_mixer", False)
        ),
        # Same rebuild-from-flag rule as the rest: a belief-headed checkpoint
        # loaded by a config without it must fail loudly at load_state_dict.
        belief_head=bool(saved.get("belief_head", False)),
        # Never read from the checkpoint: a stale training flag would silently
        # cripple a live actor.  The intervention is an argument the caller has
        # to pass on purpose, and `rl.run` refuses it outright, so only the
        # evaluation path can ever turn it on.
        consequence_ablation=consequence_ablation,
        consequence_ablation_seed=consequence_ablation_seed,
    )


def build_card_source(zh_db: str, embed: str, effect_embeddings: str,
                      op_sequences: str, op_cache: str,
                      card_binding: str = "effect",
                      effect_text_embeddings: str = "",
                      card_name_embeddings: str = ""):
    """The frozen card tables, built exactly as ``load_model`` builds them."""
    import torch
    from model import FrozenTableProvider
    from wmtok import (
        CardFeatSource, CodeIndex, build_card_semantic_matrix,
        build_op_table, build_reference_table,
        load_effect_embeddings, load_effect_text_embeddings,
        load_whole_card_source,
    )
    from wmserialize import load_rule_codes
    from mirrorforce.cardfeat import load_op_sequences

    code_index = CodeIndex.from_cdb(zh_db)
    base = load_whole_card_source(embed, code_index)
    rule_code = load_rule_codes(zh_db)
    seq_index = load_op_sequences(op_sequences)
    op_table, op_vocab, _ = build_op_table(
        seq_index, code_index, cache_path=op_cache, binding=card_binding)
    code_matrix = load_effect_embeddings(
        effect_embeddings, code_index, binding=card_binding)
    effect_text_matrix, effect_text_meta = load_effect_text_embeddings(
        effect_text_embeddings, code_index, binding=card_binding
    )
    name_matrix = None
    if card_name_embeddings:
        name_source = load_whole_card_source(card_name_embeddings, code_index)
        name_matrix = name_source.matrix()
    ref_matrix = build_reference_table(
        seq_index, code_index,
        build_card_semantic_matrix(base.matrix(), code_matrix, effect_text_matrix),
        getattr(base, "meta", {}).get("named_references", {}),
        effect_text_meta.get("named_references", {}),
        binding=card_binding,
    )
    emb = CardFeatSource(
        code_index.codes, base.matrix(), code_matrix, op_table, op_vocab,
        identity_rows=code_index.identity_rows, ref_matrix=ref_matrix,
        effect_text_matrix=effect_text_matrix,
    )
    provider = FrozenTableProvider(
        torch.from_numpy(emb.matrix()), torch.from_numpy(code_matrix),
        torch.from_numpy(op_table.astype("int64")), op_vocab,
        torch.from_numpy(ref_matrix) if ref_matrix is not None else None,
        (torch.from_numpy(effect_text_matrix)
         if effect_text_matrix is not None else None),
        identity_rows=torch.from_numpy(code_index.identity_rows),
        name_matrix=(
            torch.from_numpy(name_matrix) if name_matrix is not None else None
        ))
    return emb, provider, code_index, rule_code


def random_model(config_path: str, card_dim: int, provider, device: str,
                 max_len: int, seed: int = 0):
    """An untrained model with the deployable geometry.

    This is what a pure-RL group starts from and what a plumbing/throughput
    smoke should use when no compatible checkpoint exists: the encoder, the
    card tables and every head are real, only the weights are not.  It carries
    no ``tokenizer`` field, so nothing is silently loaded across a serializer
    version change -- the loud failure that guard exists for stays loud.
    """
    import json

    import torch
    from model import Config, WorldModel

    saved = json.loads(open(config_path).read()) if config_path else {}
    torch.manual_seed(seed)
    cfg = Config(
        d_model=int(saved.get("d_model", 512)),
        n_layer=int(saved.get("n_layer", 8)),
        n_head=int(saved.get("n_head", 8)),
        max_len=max_len, card_dim=card_dim,
        pos_mode=saved.get("pos_mode", "structured_rope"),
        attn_mode=saved.get("attn_mode", "prefix_block"),
        state_aux=bool(saved.get("state_aux", True)),
        relation_endpoints=bool(saved.get("relation_endpoints", True)),
        recurrent_memory=bool(saved.get("recurrent_memory", False)),
        card_identity=bool(saved.get("card_identity", True)),
        card_identity_dropout=float(saved.get("card_identity_dropout", 0.0)),
        card_name_residual=bool(saved.get("card_name_residual", True)),
        card_name_dropout=float(saved.get("card_name_dropout", 0.0)),
        sub_policy=bool(saved.get("sub_policy", True)),
        flex_attention=bool(saved.get("flex_attention", device.startswith("cuda"))),
    )
    return WorldModel(cfg, provider).to(device).eval(), cfg


def load_model(ckpt: str, zh_db: str, embed: str, effect_embeddings: str,
               op_sequences: str, op_cache: str, device: str = "cuda",
               card_binding: str = "effect", max_len: int = 1024,
               effect_text_embeddings: str = "",
               card_name_embeddings: str = "", flex_attention=None,
               consequence_ablation: str = "none",
               consequence_ablation_seed: int = 20260828,
               enable_belief_head: bool = False):
    """Rebuild the model exactly as training built it.

    The three card matrices are registered non-persistently, so they are *not*
    in the checkpoint: build the provider from the same artifacts or
    ``load_state_dict`` fails on a shape mismatch that names ``op_embed``.
    That failure is the loud one; the quiet one is loading with fewer channels
    than the checkpoint was trained on.
    """
    import torch
    from model import FrozenTableProvider, WorldModel
    from wmtok import (
        CardFeatSource, CodeIndex, build_card_semantic_matrix,
        build_op_table, build_reference_table,
        load_effect_embeddings, load_effect_text_embeddings,
        load_whole_card_source,
    )
    from wmserialize import load_rule_codes
    from mirrorforce.cardfeat import load_op_sequences
    from wmtok import TOKENIZER_VERSION

    code_index = CodeIndex.from_cdb(zh_db)
    base = load_whole_card_source(embed, code_index)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    rule_code = load_rule_codes(zh_db)
    seq_index = load_op_sequences(op_sequences)
    op_table, op_vocab, _ = build_op_table(
        seq_index, code_index,
        cache_path=op_cache, binding=card_binding)
    code_matrix = load_effect_embeddings(effect_embeddings, code_index,
                                         binding=card_binding)
    effect_text_matrix, effect_text_meta = load_effect_text_embeddings(
        effect_text_embeddings, code_index, binding=card_binding
    )
    use_effect_text = "embed.cardenc.effect_text_proj.weight" in blob.get("model", {})
    if use_effect_text and effect_text_matrix is None:
        raise ValueError("checkpoint requires --effect-text-embeddings artifact")
    if not use_effect_text:
        effect_text_matrix = None
        effect_text_meta = {}
    use_name = "embed.cardenc.name_proj.weight" in blob.get("model", {})
    name_matrix = None
    if use_name:
        if not card_name_embeddings:
            raise ValueError("checkpoint requires card_name_embeddings artifact")
        name_source = load_whole_card_source(
            card_name_embeddings, code_index
        )
        name_meta = getattr(name_source, "meta", {})
        if (name_meta.get("identity_view") != "name"
                or name_meta.get("text_scope") != "card-name-only"):
            raise ValueError("invalid name-only card embedding artifact")
        name_matrix = name_source.matrix()
    use_refs = "embed.cardenc.ref_proj.weight" in blob.get("model", {})
    ref_matrix = build_reference_table(
        seq_index, code_index,
        build_card_semantic_matrix(base.matrix(), code_matrix, effect_text_matrix),
        getattr(base, "meta", {}).get("named_references", {}),
        effect_text_meta.get("named_references", {}),
        binding=card_binding,
    ) if use_refs else None
    emb = CardFeatSource(
        code_index.codes, base.matrix(), code_matrix, op_table, op_vocab,
        identity_rows=code_index.identity_rows, ref_matrix=ref_matrix,
        effect_text_matrix=effect_text_matrix,
    )
    saved_tokenizer = blob.get("tokenizer")
    if saved_tokenizer != TOKENIZER_VERSION:
        raise ValueError(
            f"checkpoint tokenizer={saved_tokenizer!r}, runtime="
            f"{TOKENIZER_VERSION!r}; use the matching code/data version"
        )
    mcfg = config_from_checkpoint(
        blob, emb.dim, max_len,
        consequence_ablation=consequence_ablation,
        consequence_ablation_seed=consequence_ablation_seed,
    )
    if consequence_ablation != "none" and not mcfg.consequence_query:
        # An uncoupled checkpoint under an intervention plays exactly like the
        # unablated one, and the batch would be written up as "the ablation
        # changed nothing" -- a silent false pass on the usability gate.
        raise ValueError(
            f"consequence_ablation={consequence_ablation!r} needs a checkpoint "
            "with consequence_query; this one has no z(s, a)"
        )
    if flex_attention is not None:
        mcfg.flex_attention = bool(flex_attention)
    provider = FrozenTableProvider(
        torch.from_numpy(emb.matrix()), torch.from_numpy(code_matrix),
        torch.from_numpy(op_table.astype("int64")), op_vocab,
        torch.from_numpy(ref_matrix) if ref_matrix is not None else None,
        (torch.from_numpy(effect_text_matrix)
         if effect_text_matrix is not None else None),
        identity_rows=torch.from_numpy(code_index.identity_rows),
        name_matrix=(
            torch.from_numpy(name_matrix) if name_matrix is not None else None
        ))
    model = WorldModel(mcfg, provider).to(device)
    if enable_belief_head and not mcfg.belief_head:
        # Upgrade path for a checkpoint written before the belief head
        # existed: build the head fresh, load everything else strictly, and
        # verify the missing set is *exactly* the belief modules -- any other
        # gap must still fail the way a strict load would.
        mcfg.belief_head = True
        model = WorldModel(mcfg, provider).to(device)
        result = model.load_state_dict(blob["model"], strict=False)
        missing = set(result.missing_keys)
        allowed = {k for k in model.state_dict()
                   if k.startswith(("belief_core", "head_belief",
                                    "belief_back", "belief_gate"))}
        if result.unexpected_keys or not missing <= allowed:
            raise RuntimeError(
                f"belief-head upgrade expected to add only {sorted(allowed)}, "
                f"got missing={sorted(missing)} "
                f"unexpected={sorted(result.unexpected_keys)}"
            )
    else:
        model.load_state_dict(blob["model"])
    model.eval()
    return model, emb, code_index, rule_code, blob.get("step")


# ---------------------------------------------------------------------------
# Turning a checkpoint into a runnable policy
# ---------------------------------------------------------------------------


def add_worldmodel_arguments(parser) -> None:
    """The artifacts a whole-game checkpoint needs to become a live policy."""
    group = parser.add_argument_group("world model")
    group.add_argument("--wm-checkpoint", default="")
    group.add_argument("--wm-zh-db", default="")
    group.add_argument("--wm-text-db", default="")
    group.add_argument("--wm-cover-db", default="")
    group.add_argument("--wm-embed", default="")
    group.add_argument("--wm-effect-embeddings", default="")
    group.add_argument("--wm-effect-text-embeddings", default="")
    group.add_argument("--wm-card-name-embeddings", default="")
    group.add_argument("--wm-op-sequences", default="")
    group.add_argument("--wm-op-cache", default="")
    group.add_argument("--wm-effect-blocks", default="")
    group.add_argument("--wm-desc-map", default="")
    group.add_argument("--wm-card-struct", default="")
    group.add_argument("--wm-config", default="",
                       help="whole_game_transformer.json, for --wm-random-init")
    group.add_argument("--wm-random-init", action="store_true",
                       help="untrained weights with the deployable geometry")
    group.add_argument("--wm-init-seed", type=int, default=0)
    group.add_argument("--wm-lflist", default="",
                       help="comma-separated lflist.conf paths to "
                            "resolve the room's banlist hash")
    group.add_argument("--wm-device", default="cuda")
    group.add_argument("--wm-card-binding", default="effect")
    group.add_argument("--wm-max-len", type=int, default=262144)
    group.add_argument("--wm-slot-floor", type=int, default=5)
    group.add_argument("--wm-fallback", default="first")
    group.add_argument("--wm-sample", action="store_true",
                       help="sample the masked softmax instead of taking argmax")
    # R2 usability gate.  Evaluation only: `rl.run` refuses to start when this
    # is set, so a training group cannot be silently ablated -- an ablated
    # trainee would learn around the intervention and the A/B would measure
    # nothing while looking healthy.
    group.add_argument(
        "--wm-consequence-ablation", default="none",
        choices=("none", "zero", "shuffle"),
        help="inference-time z(s, a) intervention; evaluation batches only",
    )
    group.add_argument("--wm-consequence-ablation-seed", type=int,
                       default=20260828,
                       help="seed for the shuffle intervention's derangement")
    group.add_argument("--wm-temperature", type=float, default=1.0)
    group.add_argument("--wm-no-kv-cache", action="store_true",
                       help="reference path: re-encode the whole prefix every time")
    group.add_argument("--wm-keep-item", action="store_true",
                       help="retain the whole-game item and the on-policy trace")
    group.add_argument("--wm-verify-every", type=int, default=64,
                       help="how often to re-check the two-pass state block")


def serializer_context(effect_blocks: str = "", card_struct: str = "",
                       desc_map: str = "") -> dict:
    """The optional serializer artifacts, loaded the way ``train.py`` loads them.

    The key names matter: ``emit_state_block`` looks up ``count_limit_index``,
    ``card_struct`` and ``desc_map`` by exactly these names, and a typo would
    silently degrade the tokens rather than fail.
    """
    from wmserialize import CountLimitIndex

    out: dict = {}
    if effect_blocks:
        from mirrorforce.cardfeat import load_effect_blocks

        out["count_limit_index"] = CountLimitIndex.from_effect_blocks(
            load_effect_blocks(effect_blocks)
        )
    if card_struct:
        from mirrorforce.cardfeat import load_card_struct

        out["card_struct"] = load_card_struct(card_struct)
    if desc_map:
        from mirrorforce.cardfeat import load_desc_map

        out["desc_map"] = load_desc_map(desc_map)
    return out


def make_worldmodel_factory(args, card_pool):
    """Build a per-duel :class:`WorldModelPolicy` factory over one loaded model.

    The model is loaded once and shared by every duel of the run: a fresh
    policy per duel keeps per-duel state isolated, while reloading 41M
    parameters and three card matrices per duel would dominate the clock.
    """
    from wmserialize import Cfg
    from mirrorforce.worldmodel.cardtext import CardTextIndex

    required = ["wm_zh_db", "wm_text_db", "wm_embed",
                "wm_effect_embeddings", "wm_op_sequences"]
    if not getattr(args, "wm_random_init", False):
        required.append("wm_checkpoint")
    missing = [name for name in required if not getattr(args, name, "")]
    if missing:
        raise ValueError(
            "the worldmodel policy needs " + ", ".join(
                "--" + name.replace("_", "-") for name in missing
            )
        )
    if getattr(args, "wm_random_init", False):
        emb, provider, code_index, rule_code = build_card_source(
            args.wm_zh_db, args.wm_embed, args.wm_effect_embeddings,
            args.wm_op_sequences, args.wm_op_cache,
            card_binding=args.wm_card_binding,
            effect_text_embeddings=args.wm_effect_text_embeddings,
            card_name_embeddings=args.wm_card_name_embeddings,
        )
        model, _cfg = random_model(
            args.wm_config, emb.dim, provider, args.wm_device,
            args.wm_max_len, seed=int(getattr(args, "wm_init_seed", 0)),
        )
        step = "random-init"
    else:
        model, emb, code_index, rule_code, step = load_model(
            args.wm_checkpoint, args.wm_zh_db, args.wm_embed,
            args.wm_effect_embeddings, args.wm_op_sequences, args.wm_op_cache,
            device=args.wm_device, card_binding=args.wm_card_binding,
            max_len=args.wm_max_len,
            effect_text_embeddings=args.wm_effect_text_embeddings,
            card_name_embeddings=args.wm_card_name_embeddings,
            consequence_ablation=str(
                getattr(args, "wm_consequence_ablation", "none")
            ),
            consequence_ablation_seed=int(
                getattr(args, "wm_consequence_ablation_seed", 20260828)
            ),
        )
    texts = CardTextIndex(
        args.wm_text_db,
        cover_dbs=[args.wm_cover_db] if args.wm_cover_db else [],
        slot_floor=args.wm_slot_floor,
    )
    ctx = {"code_index": code_index, "rule_code": rule_code}
    ctx.update(serializer_context(
        effect_blocks=getattr(args, "wm_effect_blocks", ""),
        card_struct=getattr(args, "wm_card_struct", ""),
        desc_map=getattr(args, "wm_desc_map", ""),
    ))
    # whole_game: one duel/seat is one sequence, exactly as the trainer builds
    # it.  max_points is irrelevant in that mode and is left at the ceiling so
    # a stray rolling-window path would fail loudly rather than silently trim.
    cfg = Cfg(max_len=args.wm_max_len, max_points=1 << 30,
              settle_mode="link", whole_game=True)
    # The room exposes only the selected banlist's 32-bit hash. A hash with no
    # per-card limits behind it would teach the model nothing transferable, so
    # ``bind`` refuses it; resolving it here against the host's own
    # ``lflist.conf`` is the intended way to satisfy that.
    banlists: dict = {}
    for path in (getattr(args, "wm_lflist", "") or "").split(","):
        if path.strip():
            from .banlist import load_lflist_map

            banlists.update(load_lflist_map([path.strip()]))

    def factory(seed: int = 0):
        return WorldModelPolicy(
            model, cfg, ctx, texts, emb, card_pool,
            device=args.wm_device, fallback=args.wm_fallback,
            banlists=banlists,
            sample=bool(getattr(args, "wm_sample", False)),
            temperature=float(getattr(args, "wm_temperature", 1.0)),
            rng=np.random.default_rng(seed),
            use_cache=not bool(getattr(args, "wm_no_kv_cache", False)),
            keep_item=bool(getattr(args, "wm_keep_item", False)),
            verify_every=int(getattr(args, "wm_verify_every", 64)),
        )

    factory.model = model
    factory.step = step
    factory.ctx = ctx
    factory.cfg = cfg
    factory.emb = emb
    factory.texts = texts
    factory.code_index = code_index
    return factory


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mirrorforce.netduel.wmplay",
                                 description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="no network, no checkpoint: board -> snapshot -> tokens")
    ap.add_argument("--load-only", action="store_true",
                    help="load the checkpoint and print its config, then exit")
    ap.add_argument("--agent-path", default="",
                    help="prepended to sys.path so wmserialize/model import")
    add_worldmodel_arguments(ap)
    args = ap.parse_args(argv)
    if args.agent_path:
        sys.path.insert(0, args.agent_path)
    if args.selftest:
        return _selftest()
    if args.load_only:
        from .cards import CardPool

        pool = CardPool(args.wm_zh_db) if args.wm_zh_db else None
        factory = make_worldmodel_factory(args, pool)
        cfg = factory.model.cfg
        print(f"step               : {factory.step}")
        print(f"d_model/n_layer    : {cfg.d_model}/{cfg.n_layer}")
        print(f"pos_mode/attn_mode : {cfg.pos_mode}/{cfg.attn_mode}")
        print(f"sub_policy         : {cfg.sub_policy}")
        print(f"parameters         : "
              f"{sum(p.numel() for p in factory.model.parameters()):,}")
        return 0
    print("play with: harness --policy worldmodel --wm-checkpoint ...",
          file=sys.stderr)
    return 0


def _selftest() -> int:
    """Offline: does a live board turn into the tokens the corpus would emit?"""
    from .board import ShadowBoard

    board = ShadowBoard()
    board.start(0, [4031928] * 40, [])
    board.deck_count = [37, 40]
    board.extra_count = [15, 15]
    snap = snapshot_from_shadow(board, 0, (8000, 8000), 1)
    deck = [c for c in snap.cards if c.location == C.LOCATION_DECK and c.controller == 0]
    opp_deck = [c for c in snap.cards if c.location == C.LOCATION_DECK and c.controller == 1]
    opp_extra = [c for c in snap.cards if c.location == C.LOCATION_EXTRA and c.controller == 1]
    print(f"our deck rows {len(deck)} (expect 40, none drawn yet)")
    print(f"opponent deck rows {len(opp_deck)} (expect 40)")
    print(f"opponent extra rows {len(opp_extra)} (expect 15)")
    assert len(deck) == 40 and len(opp_deck) == 40 and len(opp_extra) == 15
    assert all(c.code == 0 for c in opp_deck + opp_extra), "opponent zones must be blank"
    assert snap.lp == (8000, 8000)
    print("selftest ok")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
