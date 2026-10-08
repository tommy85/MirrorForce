# Play-time search

Play-time search is optional; the competition results were obtained with the plain policy. The module lets the agent
look ahead at a decision by playing candidate actions forward in a local copy of the engine under hypotheses about the
opponent's hidden cards. It has two parts: a **follower** that keeps a local engine in step with the server, and a
**root search** that uses it.

## Why a follower

An online client never has the engine state: the server sends messages, and only those a seat may see. To look ahead,
the client needs an engine that is in the same public state as the server. The follower
(`mirrorforce/mirrorforce/common/client_sync.py`, `client_shadow.py`, `client_root.py`) builds one:

- It starts a local duel with the client's own deck and placeholders for the opponent's cards, then replays every
  message the server sends.
- After each local step it compares the local engine's messages for this seat with the server's, byte for byte. The
  opponent's choices and random outcomes (shuffles, coin flips, draws) are not sent to us directly; the follower
  infers them from the later public messages, trying the candidates the local engine offers until its output matches.
- Hidden cards stay placeholders unless the public record identifies them. The local engine may be bent to agree
  with the server (for example, a placeholder treated as able to pay a cost), but it never invents an identity.

Comparison is byte for byte, so a divergence shows at the first differing message. A few fields legitimately
differ between server implementations and are handled as follows:

| Field | Handling |
|---|---|
| The string number of a select-message hint | Not compared; the prompt that follows is |
| A select-message hint sent by only one side | Passed over; any other hint must match |
| The upper bound of a select-unselect prompt | Not compared; the cards and the lower bound are |
| A chain prompt offering fewer optional activations than the local engine | Accepted when no chain is forced and every offered activation is a local one; the answer is mapped by record |

`tools/mf_runtime_follower_replay_audit.py` replays recorded games through the follower offline and reports the first
divergence of each game with the differing bytes. `tests/test_follower_recorded_games.py` replays recorded games as
regression tests, together with negative cases (a changed server message, a changed own answer, a record missing
answers).

## Root search

At a decision the search (`mirrorforce/mirrorforce/netduel/agent_search_policy.py`):

1. **Samples hypotheses** of the opponent's hidden cards at the current root. Every hypothesis honors the public
   record: the known deck list minus every located or disclosed card, the hand and set-card counts, identities known
   without a position, and category facts from public activations ("this card was searched by that effect"). Two
   proposal laws are available:
   - *uniform*: layouts drawn uniformly among those consistent with the record;
   - *count-head*: eight times as many uniform layouts, weighed by the policy's own count-belief head at the same
     observation (for each candidate card, the head's probability of its hand and face-down counts over their
     frequency among the drawn layouts), then resampled to equal weights. The bank is a deterministic function of
     the public view, the seed and the head output, and is recomputed before use.
2. **Writes each hypothesis into a snapshot** of the follower's engine with `Debug.PermuteHidden`, which exchanges
   hidden card objects without creating, destroying or re-registering anything. The opponent's memory starts empty at
   the root; the own seat keeps its real memory.
3. **Plays every legal root action** against the same bank of hypotheses and the same random streams, continuing
   with the policy for both seats for up to five own decisions or the end of the game, and scores each line with the
   value head (P(win) − P(loss)).
4. **Updates the root policy.** The searched actions share the prior's probability mass in proportion to
   softmax((Q + β·logit) / (α + β)), with α = 0.002 and β = 0.02; the agent plays the most probable action.

Two different hidden truths behind the same public stream produce byte-identical samples: the sampler never reads
hidden information.

## When it searches

The search runs on demand, when the policy is uncertain (normalized entropy of its choice at least 0.6, or a gap of
at most 0.2 between its two most probable actions) or when a lethal is publicly possible (the visible attackers
reach the opponent's LP, or the opponent has 3,000 LP or less). Otherwise the policy's own choice is sent.

Two time profiles are registered:

- **Room**: up to 10 seconds per prompt, including preparation and the policy's own forward pass, plus 3 seconds to
  send the answer, within the room clock.
- **Untimed** (evaluation only): no room clock and no per-prompt cut; every search completes its registered rollouts.

The follower and the rollouts run in the client process; every network forward pass, including those of the
rollouts, goes to the policy service, which batches them on the GPU.
