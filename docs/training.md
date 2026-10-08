# Training

MirrorForce is trained from scratch by self-play, with no human data and no imitation of a scripted bot. The trainer
is `mirrorforce/mirrorforce/agent/train/cleanba.py` (a Cleanba-style distributed actor–learner in JAX) and the loss is
in `mirrorforce/mirrorforce/agent/train/ataraxos.py`. The recipe, which we call A0, adapts the regularized policy
gradient of Ataraxos (Sokota et al., *Scalable decision-making for games of imperfect information*, Nature 2026,
[doi:10.1038/s41586-026-11036-y](https://doi.org/10.1038/s41586-026-11036-y)) to Yu-Gi-Oh!.

## Self-play

- Both seats of every duel are played by the current policy, sampling from its distribution. The deck is the Sky
  Striker mirror; opening hands and decks are shuffled by the engine.
- Actor threads step batches of environments on threads of the C++ environment (5,120 environments in all in the
  champion run, 160 per actor thread) and send both seats' trajectories to the learner.
- One iteration collects about 4.9 million decisions; the learner then takes 200 optimizer steps and publishes the new
  parameters to the actors.

## Targets

Each seat's decisions form their own chain (the seats do not alternate decision by decision in Yu-Gi-Oh!).

- The value target of a decision is the categorical outcome (win, draw, loss) when that seat's action ended the game;
  otherwise TD(λ = 0.8) over the categorical value distributions predicted at the seat's later decisions.
- Advantages use GAE(λ = 0.5) on the scalar value `P(win) − P(loss)`, computed once per update from the values stored
  by the actors, before any optimizer step. Values come from the central critic.
- Segments have a fixed length; a segment boundary bootstraps from the next observation's value for the acting seat
  and its win/loss mirror for the other seat.

## Loss

For every kept decision:

```
loss = PPO-clip(0.2)                                   # policy improvement
     + temperature · KL(π ‖ uniform over legal actions) # the "magnet"
     + 1.0 · CE(categorical return, value head)        # value
     + 0.1 · KL(π ‖ π_behaviour)                       # stay close to the acting policy
     + 0.1 · belief loss                               # auxiliary: opponent hidden-card composition
```

- **Advantage filtering.** Per learner device and batch, only decisions with `|A| ≥ max(quantile(|A|, 0.75), 0.01)`
  are kept; the mask applies to the whole loss. Many decisions of a long game are forced or nearly so; filtering
  focuses the update on the decisions that move the value.
- **Magnet temperature and learning rate** follow power schedules `clamp(coef / (s + 1)^decay, floor, ceil)`,
  where `s` is the number of samples seen divided by the samples per iteration: temperature `0.05 / (s + 1)^0.3` in
  [0.001, 0.1], learning rate `0.5 / (s + 1)^1.1` in [5e-6, 1e-4]. Gradients are clipped at a norm of 0.267.
- **Parameter averaging.** An exponential moving average of the parameters (decay 0.999) is kept beside the iterate.
  Actors and the released champion use the iterate.

## Central critic

A separate network (width 256, 3 state layers, 9.4M parameters) reads both seats' observations at a decision,
including the other seat's private view, and predicts the acting seat's win/draw/loss. It is trained with the same
categorical value loss on λ = 1 returns and supplies the values for the advantages. Because it conditions on the full current state,
its advantage estimates are much less noisy than those of a value head that sees only public information. It is used
only inside the learner.

## Distributed setup

- Processes are started with `jax.distributed` (one per machine, `--distributed --coordinator-address <HOST:PORT>
  --num-processes <N> --process-id <I>`). Within a machine, some GPUs run actors and the others the learner and the
  critic.
- The champion was trained for about 3 days on 32 H20 GPUs, roughly 1,080 iterations.
- Checkpoints hold the iterate, the EMA, the optimizer state and the critic, and are content-addressed by SHA-256 with
  a receipt (configuration, source revision, engine digest, rule and observation versions). Resuming verifies the
  receipt and the card tables, and logs the configuration it resolved.

## The champion's configuration

The champion was trained from random weights with the options below, one process per machine. Fill in the values in
angle brackets and start one copy per machine with its own `--process-id`:

```bash
python -m mirrorforce.agent.train.cleanba --architecture policy_net --recipe ataraxos \
    --net.d 384 --net.heads 8 --net.ff 1536 --net.state-layers 4 --net.turn-layers 2 --net.readout-layers 2 \
    --net.event-hidden 768 --net.action-hidden 768 --net.readout-hidden 512 --net.semantic-dim 128 \
    --net.belief-width 128 --net.belief-heads 4 --net.belief-layers 2 --net.dtype bfloat16 --net.remat \
    --net.hand-limit --net.public-status --net.value-menu --belief-coef 0.1 --lp-shaping-coef 0 \
    --actor-blocks --public-opponent-recipe --export-both-seats \
    --critic --critic-td-lambda 1.0 --critic-zero-init --critic-advantages --critic-chunks 8 \
    --critic-model.value-menu \
    --iteration-decisions 4915200 --iteration-steps 200 --iteration-micro-segments 2 --iteration-threads 24 \
    --iteration-kept-rows --iteration-kept-chunk 128 --iteration-prefetch 2 \
    --actor-device-ids 0 1 2 3 --learner-device-ids 0 1 2 3 \
    --local-num-envs 160 --local-env-threads 12 --num-actor-threads 2 --num-steps 32 \
    --num-minibatches 1 --update-epochs 1 --gamma 1.0 --max-options 192 --max-steps 1000 --timeout 120 \
    --deck <deck directory> --deck-schedule cluster_uniform \
    --cards-db <cards.cdb> --code-list-file <code list> --semantic-file <semantic table> \
    --announce-tables <announce tables> --card-tables <card tables> \
    --total-timesteps <decisions to train> --save-interval 1 --keep-checkpoints 4 --keep-every 50 --eval-interval 0 \
    --ckpt-dir <checkpoint directory> --tb-dir <log directory> --window-stats-dir <statistics directory> \
    --run-name <name>__<YYYYMMDD> --seed <seed> \
    --distributed --coordinator-address <HOST:PORT> --num-processes <N> --process-id <I>
```

`MF_DUEL_NATIVE` names the environment extension. The environment reads the Lua card scripts from `script/` in the
working directory, so start the trainer from a directory that holds them (the weights bundle's directory does) with
`PYTHONPATH` set to the repository's `mirrorforce/` directory. The cross-machine collectives use NCCL, whose network settings (`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`
and so on) depend on your cluster. `--local-num-envs` is per actor thread; keep `--num-processes` × actor devices ×
`--num-actor-threads` × `--local-num-envs` at 5,120 when you change the number of machines or GPUs, so that one
iteration stays 30 updates of 163,840 decisions.

Start a new run without `--critic-advantages`: the advantages come from the policy's own value head while the critic
learns. Once the critic predicts held-out outcomes better than the value head, resume with `--critic-advantages
--advantages-switch-evidence <comparison JSON>`; the comparison is recorded in every later checkpoint receipt.

The trainer runs hundreds of threads and opens many files; raise the open-file limit before starting it, for example
`ulimit -n 1048576`.

**Continuing from the released weights.** On every machine, register the downloaded checkpoint once:

```bash
python -m mirrorforce.agent.train.checkpoint_store arrive --dir <bundle>/checkpoint --sha <sha256>
```

Then add `--resume <bundle>/checkpoint/<sha256>.ckpt --resume-topology-change --resume-env-change` and use the
bundle's files for the tables, the card database and the deck directory (`<bundle>/decks`). `--resume-env-change`
records that your build of the environment extension differs from ours. `--total-timesteps` counts the decisions to
train after the resumed iteration.

## Targeted scenario training

Self-play rarely visits some tactical lines, so the champion's training paused once, at iteration 1,000, for a
supervised pass on scripted scenes, and then continued with A0 from the updated actor.

- **Scenes.** `data/specialization/scenes.json` registers 96 scenes (62 for training, 34 held out, 3,784 decisions)
  in five families: a lethal line finished with Sword (`sword`), the same finished with Bomber (`bomber`), a board
  clear before the Bomber attack (`bomber`/`linked_clear`), dealing the damage and setting the defensive trap in Main
  Phase 2 (`main2_set`), and keeping defensive sets until Main Phase 2 against a removal threat (`backrow_timing`).
  `tools/mf_runtime_scene_tapes.py` plays each scene's scripted demonstration in the engine, checks its outcome
  and writes its public tape; every tape must match the digest registered for it.
- **Inputs.** `tools/mf_runtime_specialize.py export` replays each tape through the client-side environment and
  keeps only the observations a client of the demonstrated seat would have, with the demonstrated choices as labels.
- **Fit.** `tools/mf_runtime_specialize.py fit` fine-tunes the actor with its own Adam optimizer (learning rate
  1e-5): cross-entropy on the demonstrated choices, with four times the weight on main- and battle-phase command
  choices, plus 0.25 × KL(π‖π_parent) to the frozen parent on the same sequences. Whole scenes are drawn in
  proportions balanced across the families; each sequence rebuilds its memory from its first decision. The
  champion's pass took 448 updates.
- **Validation.** `tools/mf_runtime_scene_host.py` plays the held-out scenes against a policy service through the
  public client protocol (`tools/mf_runtime_scene_player.py`); the policy sees only its own client's messages.
- **Return to A0.** `mirrorforce/agent/train/policy_repair.py` writes the fine-tuned actor parameters into the A0
  checkpoint and keeps everything else of it: the A0 optimizer, the critic, the parameter average and the counters.
  A0 then resumes from that checkpoint.

## Evaluation during training

Checkpoints were evaluated with greedy action selection and no search, on fixed sets of deals played with both turn
orders: against a fixed reference agent (to place the run on an internal Elo scale) and against
WinBot. The curve is in the README.
