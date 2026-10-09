# System architecture

MirrorForce has three parts: a training system that produces a policy by self-play, a serving system that plays
that policy in online rooms, and an optional play-time search module that sits between the two. This document gives
the overall picture; the other documents go into each part:

- [Model](model.md): the policy network, its heads and its sizes.
- [Observation and features](features.md): what the network sees and how it is encoded.
- [Training](training.md): the self-play recipe, the central critic and the distributed setup.
- [Play-time search](search.md): the follower engine and the information-set search built on it.

## Components

![MirrorForce architecture: training loop, online play and policy network](architecture.svg)

| Part | Code | Role |
|---|---|---|
| Rules engine | `third_party/ygopro-core` | The YGOPro core with deterministic effect order, thread-safe global state, whole-duel snapshots and hidden-card permutation |
| Environment | `mirrorforce/cxx/duelpool`, `mirrorforce/agent/env` | Hundreds of duels on threads in one process; builds each seat's observation from public information |
| Model | `mirrorforce/agent/model/policy_net.py` | Card-level transformer with policy, value and belief heads |
| Trainer | `mirrorforce/agent/train/cleanba.py`, `ataraxos.py` | Distributed JAX actor–learner with the regularized self-play recipe |
| Card semantics | `mirrorforce/agent/semantics`, `tools/mf_runtime_card_tables.py` | Frozen per-card tables built from the card database and Lua scripts |
| Policy service | `tools/mf_runtime_policy_service.py` | Loads a checkpoint and answers decisions over a Unix socket |
| Room client | `tools/mf_runtime_league_client.py`, `mirrorforce/netduel` | Speaks the YGOPro room protocol, tracks the public board and asks the service for each answer |
| Follower and search | `mirrorforce/common/client_*`, `mirrorforce/netduel/agent_search_policy.py` | Rebuilds the game locally from the server's messages and searches over hidden-card hypotheses |

## The information boundary

The single most important rule of the design is that **the policy only ever sees what a real client would receive**.

- The environment builds every observation from the messages a client of that seat would get from a YGOPro server:
  its own hand and deck list, the public zones of both players, the public event history and anything revealed.
  The opponent's hidden cards and the deck order never reach the policy's input.
- A public disclosure ledger (`mirrorforce/netduel/disclosure.py`) tracks which identities have become public and
  keeps them public (an activated or revealed card is never anonymized again), using only received messages.
- The central critic used in training may read hidden information, but it only estimates advantages; it never acts,
  and it is not part of a released policy.
- The follower used by play-time search rebuilds the duel from received messages only; hidden cards are placeholders,
  and the opponent's choices and random outcomes are inferred from later public messages.

The same observation code serves training and play, so a policy behaves the same in self-play and in an online room.

## Training data flow

1. Each actor process runs a batch of duels (Sky Striker mirror) in the C++ environment. Both seats are played by the
   same policy (the learner's latest parameters), sampling from its distribution.
2. Trajectories of both seats, with the central critic's values and the belief targets, are sent to the learner.
3. After an iteration of about 4.9 million decisions, the learner takes 200 optimizer steps with the training loss and
   updates the actors' parameters.
4. Checkpoints are content-addressed by SHA-256 and carry a receipt (configuration, source revision, engine digest,
   rule versions); resuming verifies them.

## Serving data flow

1. The policy service loads a checkpoint, its card tables and the card database, warms up the network, and listens
   on a Unix socket (`identity`, `open`, `decide`, `close`).
2. The room client joins a YGOPro-compatible room, uploads the deck and follows the game. For every prompt it builds
   the observation of its seat from the messages it received and asks the service for an answer.
3. The client keeps the room clock (time confirmations, surrender and time-out handling) independent of the model.

Room passwords are read only from a private credentials file and never appear on command lines or in logs.

## Play-time search

The search module is optional and off by default. When enabled, the client also runs a follower: a local copy of the
engine that replays every server message. At a decision the search samples hidden-card assignments consistent with
the public record, plays each candidate action forward in the local engine under those assignments, and scores the
results with the value head. See [search.md](search.md) for the follower, the sampling rules, the time budgets and the
differences between servers that the follower has to handle.
