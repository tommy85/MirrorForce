# Model

The policy network is a card-level transformer implemented in JAX/Flax
(`mirrorforce/mirrorforce/agent/model/policy_net.py`). One forward pass reads one seat's observation at a decision and
returns a score for every legal action, a win/draw/loss estimate and a belief over the opponent's hidden cards.

## Overview

```
                ┌──────────────── card semantics (frozen tables + learned identity) ────────────────┐
card rows ──────┤                                                                                   ├─► card tokens ─┐
                └───────────────────────────────────────────────────────────────────────────────────┘                │
global / recipes / public effects / unpositioned identities / chain / hints ─────────────► state tokens ─────────────┤
                                                                                                                      ▼
                                                                        state block: 4 self-attention layers (d=384)
                                                                                                                      │
history rows of this turn ─► event embedding ─┐                                                                      │
summaries of earlier turns (≤64) ─────────────┼─► turn block: 2 causal layers ─► turn summary ───────────────────────┤
chunk summaries of a long turn (≤48) ─────────┘                                                                      ▼
                                                                        readout: 2 layers, state tokens attend to the turn
                                                                                                                      │
legal actions ─► action tokens ─► action queries ─cross-attention─► policy logits                                    │
value queries (4) ─► state tokens + legal menu ─────────────────► win / draw / loss                                  │
belief head ─────────────────────────────────────────────────────► opponent hidden-card composition (auxiliary)    ◄┘
```

## Card representation

Every card that appears on the board, in a hand or in a pile is a token. A token combines:

- **Dynamic columns** from the observation (`obs:cards_`, 41 columns per card): card id, location, sequence,
  controller, position, attribute, race, level, counters, negated flag, current ATK/DEF and type bits; plus the
  public status of the card (`obs:card_status_`: equip host, status source, the zone it came from, status bits,
  turns since) and its turn record (`obs:card_turn_`). Cards whose identity the seat cannot see carry id 0.
- **Frozen card semantics** (`mirrorforce/agent/semantics`): for every card in the database, 78 exact features from
  the card database (kind, stats, level/rank/link rating, scales, type, race and attribute flags) and up to 16
  effect units extracted from the card's Lua script, each hashed into a 64-dimensional vector of semantic tokens.
  The table covers 14,969 cards and is built once from the card database and the scripts.
- **A learned identity residual** per card id, so the network can learn card-specific behavior that the semantic
  features do not capture.

When an action or an event names a specific effect of a card ("activate effect 2 of card X"), the effect unit that the
script binds to that description is added to the row through a zero-initialized projection; a row without a proven
binding falls back to the pool of all the card's units.

## State block

The state tokens are the card tokens plus a handful of non-card tokens: global board features (`obs:global_`, LP,
phase, turn, counts, the room format), the hand-limit state, the deck recipes (`obs:own_recipe_`,
`obs:opponent_recipe_`: what each deck list still holds, as far as it is public), lingering public player effects
(`obs:public_effects_`), identities known without a position (`obs:unpositioned_`), the open chain
(`obs:chain_`) and selection hints (`obs:player_hints_`). Four self-attention layers (width 384, 8 heads,
feed-forward 1,536) mix them.

## History: this turn and earlier turns

The events of the current turn (`obs:turn_events_`, up to 512 rows of 27 columns, each with up to four references to
card rows) are embedded and passed through two causal transformer layers. Each row records one public event or one
own decision: the kind, the acting player, the phase, the card, where it came from and went to, positions, reasons,
amounts, the chain link, and for own decisions the prompt answered and the option chosen.

Earlier turns are compressed. When a turn ends, its rows are summarized into one token; the turn block reads the last
64 such summaries (with a recency embedding) before the current turn's rows. A very long turn that overflows the
512-row window keeps its older rows as chunk summaries (64 rows each, up to 48 per turn). The memory is carried per
seat between decisions, so a full game is processed incrementally.

## Readout and heads

Two readout layers let the state tokens attend to the turn's tokens, so the final state representation knows what
happened this turn.

- **Policy.** Each legal action becomes an action token built from its description (`obs:action_ir_`, 24 features:
  action kind, phase changes, positions, effect description and the like) and the card rows it references (up to four
  single references and five groups of up to eight members for multi-card choices). Action queries cross-attend to
  the state tokens and a small MLP gives one logit per action; illegal rows are masked. Menus of any size work the
  same way.
- **Value.** Four learned value queries attend to the state tokens and, through a residual attention, to the legal
  action tokens. The pooled result predicts a categorical win/draw/loss outcome; the scalar value is
  `P(win) − P(loss)`.
- **Belief.** An auxiliary head (width 128, 4 heads, 2 layers) predicts the composition of the opponent's hidden
  cards. It is trained with a small weight (0.1) and shapes the representation; the policy does not read its output.

## Sizes

| Part | Parameters |
|---|---|
| Card semantics (projections of the frozen tables, identity residual, effect gating) | 5.9M |
| Input encoders (card rows, event rows) | 2.7M |
| State block (4 layers) | 7.1M |
| Turn block (2 layers) | 3.5M |
| Readout layers and heads (policy, value) | 8.1M |
| Belief head | 0.5M |
| **Policy network total** | **28.1M** |
| Central critic (training only) | 9.4M |

Computation runs in bfloat16; parameters are kept in float32.

## Central critic

Training uses a separate, smaller network (width 256, 3 state layers) as a central critic. It encodes both seats'
observations, including the other seat's private view, and predicts the acting seat's win/draw/loss. Because it sees
hidden information it is used only to estimate advantages in the learner; it never acts and is not needed to play.
