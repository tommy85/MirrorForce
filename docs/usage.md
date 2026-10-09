# Usage

How to build MirrorForce, train it, serve the released weights and connect it to a room.

## Requirements

- Linux x86-64, Python 3.11, and the packages in [`requirements.txt`](../requirements.txt) (JAX 0.5.3, Flax 0.10.4,
  optax, chex, distrax and others). Install the CUDA build of JAX for GPU training. The follower and some shared
  modules also use PyTorch (a CPU build is enough) and pyarrow.
- A C++17 compiler, Lua 5.3 and SQLite development files for the engine and the environment extension.
- A YGOPro card database (`cards.cdb`) and the Lua card scripts, from the YGOPro community repositories.

## Building

```bash
git clone --recursive <this repository>
cd MirrorForce

# 1. The rules engine as a plain shared library, used by the follower and the engine tests
MF_CORE_ROOT=third_party/ygopro-core MF_BUILD_ROOT=<output directory> MF_LUA_INC=<Lua 5.3 include directory> \
    bash mirrorforce/build/effectinfo/build-effectinfo-core.sh
export MF_EFFECTINFO_LIB=<output directory>/effectinfo/libygopro-core-effectinfo.so

# 2. The batched environment extension (sources are taken from git commits, so commit local changes first)
python3 mirrorforce/cxx/duelpool/build.py --rev <commit> --core third_party/ygopro-core --core-rev <core commit> \
    --deps <directory of dependency snapshots> --out <new output directory>
export MF_DUEL_NATIVE=<output directory>/duel_native.cpython-311-x86_64-linux-gnu.so
```

The dependency snapshots (pybind11, fmt, SQLiteCpp, concurrentqueue, Lua and others) and their digests are listed
in `mirrorforce/cxx/duelpool/deps.json`; the build checks every one before compiling.

Paths to local builds and data are passed through environment variables; set the ones your workflow needs:

| Variable | Meaning |
|---|---|
| `MF_DUEL_NATIVE` | The built environment extension (`duel_native…so`) |
| `MF_EFFECTINFO_LIB` | The engine library built in step 1, for the follower and the engine tests |
| `MF_YGOPRO_RUN` | A YGOPro client directory holding `cards.cdb` and `script/` |
| `MF_YGOPRO_DB` | The card database, when it is not under `MF_YGOPRO_RUN` |
| `MF_TEST_ASSETS`, `MF_SCRIPTED_RUN` | Card tables and a client directory for the engine-backed tests |

## Running

Run the commands below from `mirrorforce/`. Every tool documents its full set of options with `--help`; values in
angle brackets are yours to fill in.

**Train.** On one machine:

```bash
python -m mirrorforce.agent.train.cleanba --help
```

Across several machines, start one process per machine with the same coordinator:

```bash
python -m mirrorforce.agent.train.cleanba <training options> \
    --distributed --coordinator-address <HOST:PORT> --num-processes <N> --process-id <I>
```

The champion's full set of training options, and how to continue training from the released weights, are in
[training.md](training.md#the-champions-configuration).

**Serve a policy.**

```bash
python -m tools.mf_runtime_policy_service --checkpoint <checkpoint>.ckpt --weights iterate --selection greedy \
    --cards-db <cards.cdb> --code-list <code list> --script-root <script directory> \
    --announce-tables <announce tables> --semantic-file <semantic table> --card-tables <card tables> \
    --warmup-deck decks/stage-a/SkyStriker.ydk --opponent-mode mirror \
    --socket <socket path> --out <new output directory>
```

**Join a room.** The league client reads a JSON plan (rooms, series, policy socket, deck) and a credentials file
with mode 0600; room passwords are read only from that file and never appear on the command line or in logs.
`--prepare-only` validates everything without connecting:

```bash
python -m tools.mf_runtime_league_client --plan <plan.json> --plan-sha256 <plan digest> \
    --credentials-file <credentials.json> --out <new output directory> --prepare-only
```

**Use the released weights.** Download the release bundle, unpack it and point the service at its files:

```bash
python -m tools.mf_runtime_policy_service --checkpoint <bundle>/checkpoint/<sha256>.ckpt --weights iterate \
    --selection greedy --cards-db <bundle>/cards.cdb --code-list <bundle>/code_list.txt --script-root <bundle> \
    --announce-tables <bundle>/announce-tables-<sha256>.json --semantic-file <bundle>/frozen_semantics.npz \
    --card-tables <bundle>/card-tables-<sha256>.npz --warmup-deck decks/stage-a/SkyStriker.ydk --opponent-mode mirror \
    --socket <socket path> --out <new output directory>
```

`--script-root` is the directory that contains `script/`. The bundle's `MANIFEST.json` lists every file with its
SHA-256, and the service checks the card database, code list and tables against the checkpoint's receipt. We
checked that this code with the bundle gives bit-identical policy and value outputs to the code and files the
champion played with. Two policy services and two room clients in a YGOPro room play the agent against itself or against another
checkpoint.

**Audit the follower.** `tools/mf_runtime_follower_replay_audit.py` replays recorded online games through the follower
and reports the first difference between the server's messages and the local engine's, if any.

**Test.**

```bash
PYTHONPATH=. python -m pytest tests -q -p no:cacheprovider
```

Tests that need the compiled extension or card data skip themselves when those are missing; set the environment
variables above to run them.

## Search switch

Search is off by default: the policy service and the league client play the plain greedy
policy, as in the competition. The search module ships as a library
(`mirrorforce/mirrorforce/netduel/agent_search_policy.py`, with the follower in `mirrorforce/mirrorforce/common/`)
and is configured by registered settings rather than free parameters:

| Setting | Meaning |
|---|---|
| `final_search.settings("off")` | No search; returns `None` and never constructs the follower |
| `final_search.settings("on")` | On-demand search within the room clock (10 s per prompt, 8 hypotheses) |
| `untimed_search.settings()` | The same search without time limits, for evaluation only |

The hidden-card proposal is chosen with `make_particle_provider("public_current_root_uniform")` or
`make_particle_provider("public_current_root_count_head")` (resampled with the policy's count-belief head). A room
client that drives the search is not part of this release; [search.md](search.md) describes the follower,
the root search and when it triggers.

## Repository layout

| Path | Contents |
|---|---|
| `mirrorforce/mirrorforce/agent/` | Environment wrapper, model, trainer (`train/cleanba.py`, `train/ataraxos.py`), card-semantics builder |
| `mirrorforce/mirrorforce/netduel/` | YGOPro network protocol, room client, public disclosure ledger, play-time search policy |
| `mirrorforce/mirrorforce/common/` | Follower engine (`client_sync.py`, `client_shadow.py`, `client_root.py`) and shared artifact I/O |
| `mirrorforce/mirrorforce/search/`, `worldmodel/`, `puzzle/` | Belief sampler and particles, engine driver and state capture, ctypes engine loader |
| `mirrorforce/mirrorforce/cardrules/` | Card rule tables built from the card database and Lua scripts |
| `mirrorforce/cxx/duelpool/`, `mirrorforce/cxx/mfenv/` | Batched C++ duel environment, its build script and the public-history sources it compiles |
| `mirrorforce/script-overrides/` | Runtime-equivalent scripts for two cards whose stock scripts enumerate exponentially |
| `mirrorforce/tools/` | Policy service, room client, card-table builders, follower replay audit |
| `mirrorforce/tests/` | Unit and contract tests |
| `deck/` | The Sky Striker deck the released weights were trained on and played with |
| `replay/` | Replays of the human best-of-seven at the competition |
| `mirrorforce/decks/` | Decks (the mirror match uses `stage-a/SkyStriker.ydk`) |
| `mirrorforce/sdk/room/` | A minimal SDK for connecting your own agent to a room |
| `third_party/ygopro-core/` | The patched rules engine (git submodule) |
