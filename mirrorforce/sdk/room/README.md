# Room SDK

A small standard-library client that connects your own agent to a YGOPro-compatible room (the classic LAN TCP
protocol, version `0x1362`). It needs no GPU, model weights or local rules engine: it uploads a deck, readies up,
handles the room clock and turns every server prompt into a list of legal actions for your policy.

Requirements: Python 3.10 or newer; only the standard library and the `mirrorforce.netduel` package of this
repository. Put a deck (`.ydk`) and a card database (`cards.cdb`) under `assets/`, or pass them with `--deck` and
`--cards`. Values in angle brackets below are yours to fill in.

## Connect

Check the lobby without starting a game:

```bash
python room_sdk.py --host <HOST> --port <PORT> --name <NAME> --password <ROOM PASSWORD> --probe
```

Play one game with the random baseline in `example_agent.py`:

```bash
python room_sdk.py --host <HOST> --port <PORT> --name <NAME> --password <ROOM PASSWORD>
```

Each run plays one game. `--timeout` bounds how long the client waits on the socket; it is not the room clock.
`--no-idle-timeout` waits without limit once connected and is refused in rooms that have a turn clock.
`--smoke-decisions N` plays N decisions and then surrenders, to check that your code can act.

## Your own policy

Copy `example_agent.py` to `my_agent.py` and implement one function:

```python
def choose(state):
    # score state.actions with your model
    return best_index
```

Then run with `--agent my_agent`. `choose(state)` must return a Python `int` with `0 <= index < state.n`. The index
refers only to the current prompt's `state.actions`. Multi-card selections (materials, targets, types) are split by
the SDK into sub-choices, each a separate call; a prompt with a single legal option is answered automatically.

| Field | Meaning |
|---|---|
| `state.actions` / `state.n` | Legal actions of the current prompt and their number |
| `action.describe()` | A readable description, for logs |
| `action.code`, `act`, `phase`, `spec`, `desc` | Card code, action type, phase change, location tag, effect description; 0 when unknown |
| `state.our_player` | Our player index in this duel: 0 goes first, 1 second |
| `state.turn`, `phase`, `lp` | Current turn, phase, and both players' LP |
| `state.board` | The board tracked from the messages this client received, with the public disclosure ledger |
| `state.extra['response_index']`, `['round_index']` | Our prompt number and the sub-choice number within a multi-selection |

Policies that keep their own history can use the `observe_game_message(msg, body)` callback of `NetDuelClient`.
All inputs come from this client's own connection; the server never sends the opponent's hidden cards.

## Troubleshooting

- `Connection refused` or a timeout: check the host and port; a room may need a moment to reopen after a game.
- Seat already taken: another client is playing in that room.
- `server error msg=4`: protocol version mismatch.
- `MSG_RETRY`: the rules engine rejected an answer; return only an index of the current legal actions.
