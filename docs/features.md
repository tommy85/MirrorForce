# Observation and features

The environment (`mirrorforce/cxx/duelpool`) turns the engine's state into one observation per seat at every
decision. Everything in it is derived from what a client of that seat receives from a YGOPro server; the opponent's
hidden cards and the deck order are never encoded. This document lists the observation and the design choices behind
it.

## Principles

1. **Client-visible only.** Each seat's facts come from per-viewer records of the message stream (the same records an
   online client reproduces from the server's messages). A card the seat cannot identify carries card id 0.
2. **Disclosure is permanent.** Once a card is activated, revealed or confirmed, its identity stays public for both
   sides; the public disclosure ledger keeps it attached to the right card even through moves and shuffles, and keeps
   identities known without a position (after a hand or set-card shuffle) separately from positioned ones.
3. **No made-up positions.** Hand and deck order are not observable, so they are canonicalized (own deck rows in card
   id order; sequences in the deck are never written).
4. **The same encoding everywhere.** Training, evaluation and online play use the same code, so a policy sees the same
   kind of input in self-play and in a room.

## Board

| Array | Shape | Contents |
|---|---|---|
| `cards_` | cards × 41 | One row per card of both players: card id, location, sequence, controller, position, attribute, race, level, counters, negated, current ATK/DEF, type bits |
| `card_status_` | cards × 6 | Public status: equip host and status source (as card references), equip count, the zone a card came from, status bits, turns since |
| `card_turn_` | cards × 8 | This turn's activations and attacks of the card, when and how it arrived, its counters and hints |
| `global_` | 25 | LP, phase, turn, whose turn, zone counts and other board totals, the room's format and era |
| `hand_limit_` | 2 × 4 | Each seat's hand limit, hand size and excess at the end phase |
| `own_recipe_`, `opponent_recipe_` | 80 × 6, 80 × 4 | The deck lists and how many copies of each card are still unaccounted for, as far as the public record shows |
| `unpositioned_` | 32 × 4 | Opponent identities known to be in a zone without a known position |
| `public_effects_` | 16 × 8 | Lingering player-level effects that are public (for example "draw when the opponent special summons") |

## History

| Array | Shape | Contents |
|---|---|---|
| `turn_events_`, `turn_event_refs_` | 512 × 27, 512 × 4 | The current turn's rows: public events and own decisions, with references to card rows |
| `turn_chunks_`, `turn_chunk_refs_`, `turn_chunk_meta_` | per decision | Rows of a long turn that left the 512-row window, delivered in chunks of 64 to be summarized |
| `closed_turns_`, `closed_turn_refs_`, `closed_turn_meta_` | 2 windows | A completed turn's rows, delivered once so the model can write its summary |
| `chain_` | 8 × 8 | The open chain: each link's card, controller and state |
| `turn_activations_` | 64 × 12 | This turn's activations and their outcomes (resolved, negated, disabled) |
| `turn_ledger_` | 2 × 16 | Per-seat counters of this turn: summons and activations, draws, searches, cards sent to the graveyard or banished, LP paid and damage taken |
| `player_hints_` | 16 × 5 | Hints the server sent with the current prompt |

Each event row holds: the kind of event and its subtype, the public event kind or (for an own decision) the prompt
message answered, the acting player relative to the viewer, the phase, the card row, the source and destination
(controller, location, sequence; never a deck sequence), the positions before and after, the full reason bits, an
amount (damage, LP), a count, the chain link, and for own decisions the effect description and the action kind.
Phase boundaries are not rows (every row carries its phase); a chain link is one row whose state is updated when it
resolves, is negated or disabled. When the window overflows, the oldest non-activation rows leave first.

## Decisions

| Array | Shape | Contents |
|---|---|---|
| `action_ir_` | options × 24 | Each legal option: action kind, phase change, position, effect description, counts and flags |
| `action_single_refs_` | options × 4 | Card rows the option names directly |
| `action_group_refs_`, `action_group_mask_` | options × 5 × 8 | Card groups for multi-card choices (materials, costs, targets), by role |
| `action_discard_` | options | The cards an end-phase move would discard |
| `selection_` | 12 | The prompt being answered: kind, counts, min/max, cancel availability |
| `candidates_` | options × 3 | The public candidate list for card declarations (card id and tier), also the cards the belief head scores |

Multi-card selections are decomposed into a sequence of single choices, so every decision is "pick one of the legal
options". A menu holds at most 192 options; a game is capped at 1,000 decision steps.

## Card semantics

Card identity is not a one-hot code. The frozen semantic table (`mirrorforce/agent/semantics`) gives each card 78 exact
database features (kind, stats, level/rank/link rating, pendulum scales, type, race and attribute flags) and up to 16
effect units parsed from its Lua script, each a 64-dimensional hashed bag of the script's semantic tokens (effect
category, code, range, conditions and operations). The table is computed once from `cards.cdb` and the scripts; the
model adds a learned identity residual per card. Card tables built alongside (`tools/mf_runtime_card_tables.py`) hold
static per-card extras such as link arrows and archetype codes.

## Belief targets

During training the environment also exports, for the acting seat, the true multiset of the opponent's hidden cards
(hand, set cards, unknown extra and deck). These targets train the belief head and never enter the observation.
