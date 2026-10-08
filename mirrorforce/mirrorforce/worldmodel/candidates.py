"""The candidate enumerator, and how its output is scored against the engine.

The menu head's job is to say which of the moves a player could *name* are
moves they may actually *make*.  So the candidate set has to be produced
without consulting the rules: card instance x generic action type, plus the
handful of actions that belong to the turn rather than to a card.

    hand card       -> Normal Summon / Set as monster / Set / Special Summon /
                       activate the card / activate effect i
    monster on field-> change position / attack / activate the card /
                       activate effect i
    spell or trap   -> activate the card / activate effect i
    graveyard, banished, extra deck
                    -> Special Summon / activate the card / activate effect i
    own deck        -> code-addressed Special Summon / activate candidates;
                       deck order remains hidden
    the turn itself -> to Battle Phase / to Main Phase 2 / to End Phase /
                       decline the chain

``i`` runs over the card's printed effect clauses (:mod:`.cardtext`); nothing
here knows what any of them *do*.  The only rule the enumerator does encode is
where an action can physically originate -- you cannot attack with a card in
the graveyard -- because that is board grammar, not card knowledge, and it is
what keeps the candidate list a few hundred long instead of a few thousand.

Scoring
-------

A candidate is positive when the engine's menu contains a matching entry.  The
direction that matters is the other one: **every engine entry must match some
candidate**.  A candidate the engine never offers is just a negative and costs
nothing; an engine entry with no candidate is a hole in the action space, and
:func:`match_menu` reports each one so the gap can be named rather than
averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..netduel import constants as C
from ..netduel.actions import CARD_EFFECT_OFFSET, ActionAct, ActionPhase, ls_to_spec
from .cardtext import MAX_EFFECT_SLOTS, CardTextIndex
from .state import StateSnapshot

__all__ = [
    "ActionKind",
    "Candidate",
    "GLOBAL_KINDS",
    "MatchResult",
    "ORIGIN_LOCATIONS",
    "engine_key",
    "candidate_index_for_action",
    "enumerate_candidates",
    "match_menu",
    "spsummon_static_allowed",
]


class ActionKind(str, Enum):
    """The generic action types, named as a player would name them."""

    SUMMON = "summon"            # normal summon
    MSET = "mset"                # monster set
    SET = "set"                  # spell/trap set
    SPSUMMON = "spsummon"        # special summon
    REPO = "repo"                # change of battle position
    ATTACK = "attack"            # attack declaration
    ACTIVATE_CARD = "act_card"   # activate the card itself
    ACTIVATE_EFFECT = "act_eff"  # activate effect i
    TO_BATTLE = "to_battle"
    TO_MAIN2 = "to_main2"
    TO_END = "to_end"
    CHAIN_PASS = "chain_pass"


#: actions that belong to the turn, not to a card
GLOBAL_KINDS = (
    ActionKind.TO_BATTLE,
    ActionKind.TO_MAIN2,
    ActionKind.TO_END,
    ActionKind.CHAIN_PASS,
)

_ACT_ALL = frozenset(
    {
        C.LOCATION_HAND,
        C.LOCATION_MZONE,
        C.LOCATION_SZONE,
        C.LOCATION_GRAVE,
        C.LOCATION_REMOVED,
        C.LOCATION_EXTRA,
    }
)

#: where each card action can physically originate.  Deliberately generous --
#: a location listed here that never produces a legal action only adds
#: negatives, while one missing produces an unreachable engine entry.
ORIGIN_LOCATIONS: dict[ActionKind, frozenset[int]] = {
    # The monster zone counts too: "Red-Eyes Black Flare Dragon" (30079770) treated as a normal monster on the field can
    # normal summon itself once more, and the engine really offers a SUMMON starting from the monster zone in the menu.
    # This gap is unreachable with the meta deck pool; coverage-driven synthetic builds caught it
    # (1 missing of 19,069 menu entries).
    ActionKind.SUMMON: frozenset({C.LOCATION_HAND, C.LOCATION_MZONE}),
    ActionKind.MSET: frozenset({C.LOCATION_HAND}),
    ActionKind.SET: frozenset({C.LOCATION_HAND}),
    ActionKind.SPSUMMON: frozenset(
        {
            C.LOCATION_HAND,
            C.LOCATION_EXTRA,
            C.LOCATION_GRAVE,
            C.LOCATION_REMOVED,
            C.LOCATION_MZONE,
            C.LOCATION_SZONE,
        }
    ),
    ActionKind.REPO: frozenset({C.LOCATION_MZONE}),
    ActionKind.ATTACK: frozenset({C.LOCATION_MZONE}),
    ActionKind.ACTIVATE_CARD: _ACT_ALL,
    ActionKind.ACTIVATE_EFFECT: _ACT_ALL,
}

#: Zones of ours the enumerator walks. Own-deck order is canonicalized by
#: ``mask_for``; deck candidates are deduplicated and keyed by code, never by
#: that synthetic sequence.
OWN_ZONES = (
    C.LOCATION_DECK,
    C.LOCATION_HAND,
    C.LOCATION_MZONE,
    C.LOCATION_SZONE,
    C.LOCATION_GRAVE,
    C.LOCATION_REMOVED,
    C.LOCATION_EXTRA,
)

#: zones of the opponent that are public; only activations are enumerated there
OPPONENT_ZONES = (
    C.LOCATION_MZONE,
    C.LOCATION_SZONE,
    C.LOCATION_GRAVE,
    C.LOCATION_REMOVED,
)


def spsummon_static_allowed(card_type: int | None) -> bool:
    """Only a printed Monster Card can be a direct SPSUMMON action source.

    ``None`` stays allowed: an unknown public runtime code must remain
    conservative.  Printed type matters rather than the card's live query
    type, because a Pendulum Monster in a scale reports as a Spell while it
    remains a Monster Card for summon procedures.
    """
    return card_type is None or bool(int(card_type) & C.TYPE_MONSTER)


@dataclass(frozen=True)
class Candidate:
    """One thing a player could try, before anyone checks whether they may."""

    kind: ActionKind
    spec: str = ""
    code: int = 0
    eff_slot: int = -1
    location: int = 0
    sequence: int = -1
    opponent: bool = False
    described: bool = True

    @property
    def key(self) -> tuple:
        if self.location == C.LOCATION_DECK:
            return (self.kind.value, "#deck-code", self.eff_slot, self.code)
        return (self.kind.value, self.spec, self.eff_slot)

    def describe(self) -> str:
        if self.kind in GLOBAL_KINDS:
            return self.kind.value
        if self.kind is ActionKind.ACTIVATE_EFFECT:
            return f"{self.spec}|{self.kind.value}{self.eff_slot}"
        return f"{self.spec}|{self.kind.value}"


def enumerate_candidates(
    snapshot: StateSnapshot,
    player: int,
    texts: CardTextIndex,
    include_opponent: bool = True,
) -> list[Candidate]:
    """Every candidate at this state, in a stable order.

    ``snapshot`` should already be masked for ``player``; a card whose identity
    the mask stripped contributes no effect slots, because a player cannot name
    the effects of a card they cannot see.
    """
    out: list[Candidate] = []

    def n_effect_slots(code: int) -> int:
        known = getattr(texts, "known", None)
        if callable(known) and not known(code):
            return MAX_EFFECT_SLOTS
        return texts.n_slots(code)

    def described_slots(code: int) -> int:
        count = getattr(texts, "effect_count", None)
        return int(count(code)) if callable(count) else texts.n_slots(code)

    def printed_type(code: int) -> int | None:
        lookup = getattr(texts, "card_type", None)
        return lookup(code) if callable(lookup) else None

    zones: list[tuple[int, tuple[int, ...], bool]] = [(player, OWN_ZONES, False)]
    if include_opponent:
        zones.append((1 - player, OPPONENT_ZONES, True))

    for owner, locations, is_opponent in zones:
        seen_deck_codes: set[int] = set()
        for location in locations:
            cards = sorted(
                (
                    c
                    for c in snapshot.cards
                    if c.controller == owner and c.location == location
                ),
                key=lambda c: c.sequence,
            )
            for card in cards:
                if location == C.LOCATION_DECK:
                    if not card.code or card.code in seen_deck_codes:
                        continue
                    seen_deck_codes.add(card.code)
                spec = ls_to_spec(location, card.sequence, 0, is_opponent)
                kinds = (
                    (
                        ActionKind.SPSUMMON,
                        ActionKind.ACTIVATE_CARD,
                        ActionKind.ACTIVATE_EFFECT,
                    )
                    if location == C.LOCATION_DECK
                    else (ActionKind.ACTIVATE_CARD, ActionKind.ACTIVATE_EFFECT)
                    if is_opponent
                    else (
                        ActionKind.SUMMON,
                        ActionKind.MSET,
                        ActionKind.SET,
                        ActionKind.SPSUMMON,
                        ActionKind.REPO,
                        ActionKind.ATTACK,
                        ActionKind.ACTIVATE_CARD,
                        ActionKind.ACTIVATE_EFFECT,
                    )
                )
                n_slots = 0 if card.hidden else n_effect_slots(card.code)
                for kind in kinds:
                    if (location != C.LOCATION_DECK
                            and location not in ORIGIN_LOCATIONS[kind]):
                        continue
                    if (kind is ActionKind.SPSUMMON
                            and not spsummon_static_allowed(
                                printed_type(card.code)
                            )):
                        continue
                    if kind is ActionKind.ACTIVATE_EFFECT:
                        for slot in range(n_slots):
                            out.append(
                                Candidate(
                                    kind=kind,
                                    spec=spec,
                                    code=card.code,
                                    eff_slot=slot,
                                    location=location,
                                    sequence=card.sequence,
                                    opponent=is_opponent,
                                    described=slot < described_slots(card.code),
                                )
                            )
                    else:
                        out.append(
                            Candidate(
                                kind=kind,
                                spec=spec,
                                code=card.code,
                                location=location,
                                sequence=card.sequence,
                                opponent=is_opponent,
                            )
                        )
                # Overlay materials are not standalone CardState rows; their
                # public codes live on the host's ordered ``overlay`` tuple.
                # Core addresses one as e.g. ``m1a``. Give each material a
                # separate virtual SLOT row while preserving that menu spec.
                for material, material_code in enumerate(card.overlay):
                    if not material_code:
                        continue
                    material_spec = ls_to_spec(
                        C.LOCATION_MZONE | C.LOCATION_OVERLAY,
                        card.sequence, material, is_opponent,
                    )
                    virtual_sequence = card.sequence * 16 + material
                    out.append(Candidate(
                        kind=ActionKind.ACTIVATE_CARD,
                        spec=material_spec,
                        code=material_code,
                        location=C.LOCATION_OVERLAY,
                        sequence=virtual_sequence,
                        opponent=is_opponent,
                    ))
                    for slot in range(n_effect_slots(material_code)):
                        out.append(Candidate(
                            kind=ActionKind.ACTIVATE_EFFECT,
                            spec=material_spec,
                            code=material_code,
                            eff_slot=slot,
                            location=C.LOCATION_OVERLAY,
                            sequence=virtual_sequence,
                            opponent=is_opponent,
                            described=slot < described_slots(material_code),
                        ))
    for kind in GLOBAL_KINDS:
        out.append(Candidate(kind=kind))
    return out


# -- scoring ---------------------------------------------------------------

_ACT_TO_KIND = {
    ActionAct.SUMMON: ActionKind.SUMMON,
    ActionAct.MSET: ActionKind.MSET,
    ActionAct.SET: ActionKind.SET,
    ActionAct.SPSUMMON: ActionKind.SPSUMMON,
    ActionAct.REPO: ActionKind.REPO,
    ActionAct.ATTACK: ActionKind.ATTACK,
    ActionAct.DIRECT_ATTACK: ActionKind.ATTACK,
}

_PHASE_TO_KIND = {
    ActionPhase.BATTLE: ActionKind.TO_BATTLE,
    ActionPhase.MAIN2: ActionKind.TO_MAIN2,
    ActionPhase.END: ActionKind.TO_END,
}


def engine_key(action) -> tuple:
    """The candidate key one engine menu entry corresponds to.

    ``desc == 0`` reaches the action layer as ``effect == 0`` and means "this
    card, not one of its numbered effects".  A system-string description
    (``0 < effect < CARD_EFFECT_OFFSET``) is not a numbered effect either, so it
    maps onto the same key; that is an over-approximation and is counted
    separately by :func:`match_menu`.
    """
    if action.phase in _PHASE_TO_KIND:
        return (_PHASE_TO_KIND[action.phase].value, "", -1)
    if action.act == ActionAct.CANCEL:
        return (ActionKind.CHAIN_PASS.value, "", -1)
    if action.act == ActionAct.ACTIVATE:
        if action.effect >= CARD_EFFECT_OFFSET:
            key = (
                ActionKind.ACTIVATE_EFFECT.value,
                action.spec,
                action.effect - CARD_EFFECT_OFFSET,
            )
        else:
            key = (ActionKind.ACTIVATE_CARD.value, action.spec, -1)
    else:
        kind = _ACT_TO_KIND.get(action.act)
        if kind is None:
            return ("unmapped", action.spec, action.effect)
        key = (kind.value, action.spec, -1)
    # A bare ordinal denotes a card in the Deck. It restarts for every prompt,
    # so it is not the shuffled or canonical deck sequence. The prompt reveals
    # the code; use that stable, deployable key instead.
    if action.spec.isdigit():
        return (
            key[0], "#deck-code", key[2],
            int(getattr(action, "code", 0) or 0),
        )
    return key


def candidate_index_for_action(candidates, index_of: dict[tuple, int], action) -> int:
    """Map one engine action to the same candidate online and offline.

    A few scripts use an external or malformed ``Stringid`` whose low nibble
    is unrelated to this card's printed-effect axis. Exact matching fails in
    that case. If the source spec has exactly one effect candidate, binding it
    is unambiguous; with two or more, refusing to guess preserves the hard gate.
    """
    exact = index_of.get(engine_key(action))
    if exact is not None:
        return exact
    if action.act != ActionAct.ACTIVATE or action.effect < CARD_EFFECT_OFFSET:
        return -1
    matches = [
        index for index, candidate in enumerate(candidates)
        if candidate.kind is ActionKind.ACTIVATE_EFFECT
        and candidate.spec == action.spec
        and candidate.described
    ]
    return matches[0] if len(matches) == 1 else -1


@dataclass
class MatchResult:
    """The menu-head labels for one decision point."""

    labels: list[bool]
    #: index into the engine menu for each positive candidate, else -1
    menu_index: list[int]
    #: engine entries no candidate covered -- each one is an enumerator bug
    missed: list[tuple[tuple, str]]
    #: engine entries that landed on the same candidate as another entry
    duplicate: int = 0
    #: entries reached through the system-description over-approximation
    system_desc: int = 0
    #: The **full desc** value of each candidate. Ordinary per-effect candidates are determined by the static
    #: `Stringid(card_code, eff_slot)`; when they hit the menu the engine's original value overrides it,
    #: preserving exceptions such as "strings referencing another card" and system descriptions.
    #: Storing only `desc & 0xF` is not enough: that is the index n into the script's string table, and measured,
    #: n differs from the effect index 35.55% of the time (Infinite Impermanence / Evenly Matched: effect 0 is
    #: an ACTIVATE without description, and Stringid(code,0) hangs on effect 1),
    #: and truncating to the low four bits loses `desc >> 4`, so Stringids carrying another card's code cannot be looked up.
    #: The effect slot is always looked up in desc_map, never computed from the low four bits.
    desc: list[int] = field(default_factory=list)

    @property
    def n_positive(self) -> int:
        return sum(self.labels)


def match_menu(candidates: list[Candidate], actions) -> MatchResult:
    """Label every candidate against the engine's menu."""
    index_of: dict[tuple, int] = {}
    for i, candidate in enumerate(candidates):
        index_of.setdefault(candidate.key, i)

    labels = [False] * len(candidates)
    menu_index = [-1] * len(candidates)
    # Illegal / unmatched candidates need effect semantics too, so the legality head can ask "can this effect be used now".
    # They have no engine menu entry, but the full desc of an ordinary card effect is a public static quantity anyway:
    # aux.Stringid(code, n) == (code << 4) | n. Candidates that really hit are still overridden with
    # action.desc below, so strings of other cards and system descs are not wiped out by this default.
    descs = [
        ((int(c.code) << 4) | int(c.eff_slot))
        if c.kind is ActionKind.ACTIVATE_EFFECT
        and c.code and c.eff_slot >= 0 else 0
        for c in candidates
    ]
    missed: list[tuple[tuple, str]] = []
    duplicate = 0
    system_desc = 0
    for menu_pos, action in enumerate(actions):
        key = engine_key(action)
        if (
            action.act == ActionAct.ACTIVATE
            and 0 < action.effect < CARD_EFFECT_OFFSET
        ):
            system_desc += 1
        target = candidate_index_for_action(candidates, index_of, action)
        if target < 0:
            missed.append((key, action.describe()))
            continue
        if labels[target]:
            duplicate += 1
        labels[target] = True
        menu_index[target] = menu_pos
        descs[target] = int(getattr(action, "desc", 0) or 0)
    return MatchResult(
        labels=labels,
        menu_index=menu_index,
        missed=missed,
        duplicate=duplicate,
        system_desc=system_desc,
        desc=descs,
    )
