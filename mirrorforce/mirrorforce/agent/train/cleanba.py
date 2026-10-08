import os
import json
import signal
import sys
import traceback
import shutil
import collections
import concurrent.futures
import dataclasses
import queue
import resource
import random
import threading
import time
from datetime import datetime, timedelta, timezone
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from types import SimpleNamespace
from typing import List, NamedTuple, Optional, Literal
from functools import partial
from pathlib import Path

import mirrorforce.agent.env
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
import tyro
from rich.pretty import pprint

from mirrorforce.agent.utils import init_duel, load_embeddings
from mirrorforce.agent.rl.env import RecordEpisodeStatistics, EnvPreprocess
from mirrorforce.agent.model.agent import RNNAgent, ModelArgs
from mirrorforce.agent.model.reference_agent import ReferenceAgent, select_rows
from mirrorforce.agent.model.policy_net import BELIEF_LOCATIONS, PolicyNetConfig, belief_targets
from mirrorforce.agent.train import a0_buffer, a0_counters, ataraxos
from mirrorforce.agent.train.ataraxos import AtaraxosConfig
from mirrorforce.agent.model.decision import public_outcome_targets, outcome_loss
from mirrorforce.agent.model.decision_checkpoint import restore_decision_checkpoint
from mirrorforce.agent.train import state_io
from mirrorforce.agent.env import card_tables, room_format
from mirrorforce.agent.model.utils import masked_normalize, categorical_sample, TrainState
from mirrorforce.agent.model.eval import evaluate, battle
from mirrorforce.agent.model.switch import truncated_gae_sep as gae_sep_switch
from mirrorforce.agent.model import clipped_surrogate_pg_loss, mse_loss, entropy_loss, simple_policy_loss, \
    ach_loss, policy_gradient_loss, vtrace, vtrace_sep, truncated_gae, truncated_gae_sep


# Appended to, never replacing, the job's own XLA flags (command buffers and the like); the resolved value is
# printed at startup and written into every checkpoint receipt.
os.environ["XLA_FLAGS"] = " ".join(
    [os.environ.get("XLA_FLAGS", "").strip(), "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"]
).strip()


def deck_cluster_name(deck_name: str) -> str:
    if deck_name.startswith("cluster_") and "__" in deck_name:
        return deck_name.split("__", 1)[0]
    return deck_name


def lp_pair(obs):
    """Both players' LP from the structured observation global_ (P0=g0*256+g1, P1=g2*256+g3) -> (N,2)."""
    g = np.asarray(obs["global_"])
    if g.ndim == 1:
        g = g[None, :]
    elif g.ndim > 2:
        g = g.reshape(g.shape[0], -1)
    return np.stack([g[:, 0].astype(np.float64) * 256.0 + g[:, 1].astype(np.float64),
                     g[:, 2].astype(np.float64) * 256.0 + g[:, 3].astype(np.float64)], axis=1)


class RecordClusterEpisodeStatistics(RecordEpisodeStatistics):
    def update_stats_and_infos(self, *values):
        obs, rewards, terminated, truncated, infos = values
        infos["_cluster_terminated"] = terminated
        infos["_cluster_truncated"] = truncated
        return super().update_stats_and_infos(obs, rewards, terminated, truncated, infos)


#: Set by SIGTERM: the learner writes a checkpoint after its current update and stops.
STOP_REQUESTED = threading.Event()
# Exact resume: each actor thread's state at the top of its recent rollouts (thread id -> {update: snapshot}); the
# learner saves, with update U, every thread's snapshot for rollout U + 1.
ACTOR_SNAPSHOTS: dict = {}
SNAPSHOT_READY = threading.Condition()
SNAPSHOT_TIMEOUT = 900.0


def actor_snapshot(update, envs, next_obs, next_to_play, next_done, main_player, rstates, key, global_step,
                   actor_policy_version, next_labels=None, next_priv=None):
    extra = {} if next_labels is None else {"next_labels": np.array(next_labels)}
    if next_priv is not None:  # the other seat's view of the next decision (central critic)
        extra["next_priv"] = {k: np.array(v) for k, v in next_priv.items()}
    return {**extra, "update": int(update), "envs": list(envs.unwrapped.export_states()),
            "next_obs": {k: np.array(v) for k, v in next_obs.items() if v is not None},
            "next_obs_none": sorted(k for k, v in next_obs.items() if v is None),
            "next_to_play": np.array(next_to_play), "next_done": np.array(next_done, dtype=np.bool_),
            "main_player": np.array(main_player), "rstate": [np.array(x) for x in jax.tree.leaves(rstates)],
            "key": np.array(key), "global_step": int(global_step), "actor_policy_version": int(actor_policy_version)}


def publish_snapshot(thread_id, snapshot):
    with SNAPSHOT_READY:
        slot = ACTOR_SNAPSHOTS.setdefault(thread_id, {})
        slot[snapshot["update"]] = snapshot
        for old in [u for u in slot if u < snapshot["update"] - 3]:
            del slot[old]
        SNAPSHOT_READY.notify_all()


COMPILES = [0]


def _count_compile(event, duration, **_):
    if event == "/jax/core/compile/backend_compile_duration":
        COMPILES[0] += 1


jax.monitoring.register_event_duration_secs_listener(_count_compile)

MENU_BLOCK = 32  # event rows and chunk slots switch on the device (policy_net's turn-pass shapes)
MENU_KEYS = ("actions_", "action_ir_", "action_single_refs_", "action_group_refs_", "action_group_mask_",
             "candidates_", "action_discard_")


def menu_block(obs, max_options):
    """The step's menu block: 32 when every menu fits, else the full axis. Options are left-aligned and read from
    the observation alone (the model's validity column), so a resumed actor picks the same block."""
    options = int((np.asarray(obs["action_ir_"])[..., 0] > 0).sum(-1).max())
    return MENU_BLOCK if options <= MENU_BLOCK else max_options


def sliced_obs(obs, menu):
    out = dict(obs)
    for key in MENU_KEYS:
        if obs.get(key) is not None:
            out[key] = np.asarray(obs[key])[:, :menu]
    return out


class GuardMonitor:
    """illegal_activation_withdrawal/v1 counters per log interval (``info:illegal_activation``): withdrawn decisions
    and their cards, withdrawn decisions in an already shipped segment, and target shortfalls outside a frame."""

    def __init__(self):
        self.withdrawn, self.crossed, self.shortfalls = 0, 0, 0
        self.withdrawn_cards, self.shortfall_cards = {}, {}

    def interval(self):
        top = lambda d: dict(sorted(d.items(), key=lambda kv: -kv[1])[:8])
        out = {"withdrawn": self.withdrawn, "withdrawn_in_shipped_segment": self.crossed,
               "withdrawn_cards": top(self.withdrawn_cards), "shortfalls": self.shortfalls,
               "shortfall_cards": top(self.shortfall_cards)}
        self.__init__()
        return out


def mark_withdrawn(storage, info, monitor, recipe, previous=None):
    """Withdrawn decisions are not transitions: the environment restored the state before them and re-presents the
    first one's observation exactly (``info:illegal_activation`` [0] and [1]: decisions of players 0 and 1 withdrawn
    at this step, the most recent ones of that environment, this step's included). They are masked out of arm A's
    targets and losses here; the recurrent state needs no undo, since policy_net's deliveries are idempotent and the
    re-presented observation repeats the withdrawn one's. ``previous`` [T, envs]: the withdrawn flags of the previous
    update when its data are not trained yet (A0: within an iteration); decisions beyond it were trained already and
    are counted."""
    ill = np.asarray(info["illegal_activation"])
    count = ill[:, 0] + ill[:, 1]
    for env in np.flatnonzero(ill[:, 4] > 0):
        monitor.shortfalls += int(ill[env, 4])
        monitor.shortfall_cards[int(ill[env, 5])] = monitor.shortfall_cards.get(int(ill[env, 5]), 0) + 1
    for env in np.flatnonzero(count > 0):
        if recipe != "ataraxos":
            raise RuntimeError("withdrawn decisions (illegal_activation_withdrawal/v1) are masked by arm A only")
        n = int(count[env])
        monitor.withdrawn += n
        monitor.withdrawn_cards[int(ill[env, 2])] = monitor.withdrawn_cards.get(int(ill[env, 2]), 0) + n
        for back in range(1, n + 1):
            if back <= len(storage):
                storage[-back].withdrawn[env] = True
            elif previous is not None and back - len(storage) <= previous.shape[0]:
                previous[previous.shape[0] - (back - len(storage)), env] = True
            else:
                monitor.crossed += 1


ENV_IDENTITY = ("native_sha256", "native_core", "window_law", "observation_laws", "card_view_law")


def restore_on_process0(args, template, key_count, expected):
    """``state_io.restore`` of ``args.resume``. With several processes only process 0 reads (and verifies) the
    checkpoint, which lives on its host only; the others receive the state, the EMA, the first learner key, the
    counters and the receipt from it over the distributed runtime (a resume of several processes restarts their
    environments, so there are no actor states)."""
    if args.world_size == 1:
        return state_io.restore(args.resume, template, key_count, expected,
                                add_value_menu=getattr(args, "resume_add_value_menu", False))
    from jax.experimental import multihost_utils
    constants, template = template.constants, template.replace(constants={})  # the same in every process
    if args.local_rank == 0:
        state, keys, counters, receipt, _, ema = state_io.restore(
            args.resume, template, key_count, expected, add_value_menu=getattr(args, "resume_add_value_menu", False))
        if ema is None:
            raise ValueError(f"{Path(args.resume).name}: an A0 checkpoint without its EMA")
        text = np.frombuffer(json.dumps({"counters": counters, "receipt": receipt}).encode(), np.uint8)
        key = np.asarray(keys[:1])
    else:
        state, ema = template, jax.tree.map(np.zeros_like, template.params)
        text, key = np.zeros(0, np.uint8), np.zeros((1, 2), np.uint32)
    size = int(np.asarray(multihost_utils.broadcast_one_to_all(np.asarray([text.size], np.int32)))[0])
    if args.local_rank != 0:
        text = np.zeros(size, np.uint8)
    text, state, ema, key = multihost_utils.broadcast_one_to_all((text, state, ema, key))
    meta = json.loads(bytes(np.asarray(text, np.uint8)).decode())
    state, ema = jax.tree.map(np.asarray, state).replace(constants=constants), jax.tree.map(np.asarray, ema)
    return state, np.asarray(key), meta["counters"], meta["receipt"], None, ema


def resume_contract(args):
    """What a resume from this run's checkpoints restores, written into every receipt."""
    if args.iteration_decisions:  # A0: iterations are synchronous; one process keeps every environment
        exact = args.world_size == 1
        return {"actor_and_env_state": exact, "exact": exact, "iterations": True,
                "not_restored": [] if exact else ["environments and actor memory of every process (several "
                                                  "processes: a resume restarts them, --resume-topology-change)"],
                "bitwise_note": "bitwise identity on GPU also needs deterministic XLA ops (runtime.xla_flags)"}
    exact = not args.concurrency
    return {
        "actor_and_env_state": True,
        "exact": exact,
        "concurrency": args.concurrency,
        "not_restored": ["logging statistics (episode returns and lengths, deck, seat and decision counters)"],
        "async_difference": None if exact else (
            "the first two rollouts after a resume use the restored parameters; uninterrupted, the two-version "
            "lag gives them the parameters of the update before and of the restored update"),
        "bitwise_note": "bitwise identity on GPU also needs deterministic XLA ops (runtime.xla_flags)",
    }


def collect_snapshots(update, threads):
    """Every actor thread's snapshot for rollout ``update``, waiting for threads that have not reached it."""
    deadline = time.time() + SNAPSHOT_TIMEOUT
    with SNAPSHOT_READY:
        while not all(update in ACTOR_SNAPSHOTS.get(t, {}) for t in range(threads)):
            left = deadline - time.time()
            if left <= 0:
                raise RuntimeError(f"actor snapshots for rollout {update} did not arrive")
            SNAPSHOT_READY.wait(left)
        return [ACTOR_SNAPSHOTS[t][update] for t in range(threads)]
#: First calls of jitted functions run one at a time. Concurrent first calls on one
#: device let XLA autotune (synchronizing the GPU) while another thread captures a
#: CUDA graph, which invalidates the capture (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED).
FIRST_CALL_LOCK = threading.Lock()


def fatal_exit(where, exc_type, exc_value, exc_traceback):
    """Any uncaught exception ends the whole process; actor threads never outlive it."""
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print(f"FATAL_EXCEPTION in {where}; exiting the process", flush=True)
    sys.stderr.flush()
    os._exit(70)


SHARED_JIT = {}
SHARED_JIT_LOCK = threading.Lock()


def shared_jit(name, fn):
    """The process's single jitted function ``name``: the first actor thread's, for every thread (the threads build
    identical closures over the same configuration)."""
    with SHARED_JIT_LOCK:
        return SHARED_JIT.setdefault(name, fn)


def first_call_serialized(fn):
    """Run ``fn``'s first call (compilation and first execution) under ``FIRST_CALL_LOCK``."""
    state = {"done": False}

    def call(*args, **kwargs):
        if state["done"]:
            return fn(*args, **kwargs)
        with FIRST_CALL_LOCK:
            result = fn(*args, **kwargs)
            jax.block_until_ready(result)
            state["done"] = True
            return result
    return call


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    log_frequency: int = 10
    """the logging frequency of the model performance (in terms of `updates`)"""
    time_log_freq: int = 0
    """the logging frequency of the deck time statistics, 0 to disable"""
    cluster_stats_dir: str = ""
    iteration_decisions: int = 0
    """A0, strict Ataraxos iterations (design 10e): decisions collected with one behaviour policy per iteration (0:
    the per-update recipe), rounded to whole updates; then iteration_steps optimizer steps over them (one epoch),
    the EMA, a checkpoint and the next behaviour policy. lr and temperature follow the iteration count. Requires
    --recipe ataraxos"""
    iteration_steps: int = 200
    iteration_micro_segments: int = 12
    """segments (of num_steps decisions) per micro-batch on a learner device"""
    iteration_kept_rows: bool = False
    """A0: the per-decision passes and their backward run only on the decisions the |A| filter keeps (every loss term,
    the belief loss included, reads only those: belief_rows kept, in the identity); the memory recurrence still runs
    on every segment (mirrorforce/agent/train/a0_kept.py; exact against the full forward)"""
    iteration_kept_chunk: int = 64
    """A0 kept rows: rows per chunk of the per-decision passes (the capacity is the kept rate plus 15%, in chunks)"""
    iteration_threads: int = 16
    iteration_prefetch: int = 2
    """A0: minibatches built ahead of the optimizer steps (each by one builder thread; the steps use them in order)"""
    """host threads compressing and decompressing segments"""
    window_stats_dir: str = ""
    """where each actor thread writes its cumulative per-deck turn-length and chunk histograms (window monitor)"""
    """optional append-only terminal statistics directory"""
    save_interval: int = 400
    keep_checkpoints: int = 0
    """> 0: after each save keep only the payloads of the last this many checkpoints plus every --keep-every-th
    update (receipts stay); 0 keeps everything"""
    keep_every: int = 500
    """the frequency of saving the model (in terms of `updates`)"""
    checkpoint: Optional[str] = None
    resume: Optional[str] = None
    """a training checkpoint (<sha256>.ckpt) to continue: parameters, Adam state, learner keys and counters"""
    """the path to the model checkpoint to load"""
    decision_warm_start: bool = False
    """explicitly add zero-output decision branches to a legacy checkpoint"""
    decision_aux_coef: float = 0.0
    mirror_probability_percent: int = 0
    matched_probability_percent: int = 0
    pool_deck_uniform: int = 0
    """opponent-side pool draw: 0 = cluster-uniform, 1 = deck-uniform (weights each cluster by deck count)"""
    lp_shaping_coef: float = 0.0
    """immediate reward for the net LP differential inside a step (0 = off)"""
    lp_shaping_scale: float = 500.0
    """LP units that saturate the shaping term"""
    """weight of executed-action next-decision public-outcome prediction"""
    timeout: int = 600
    """the timeout of the environment step"""
    debug: bool = False
    """whether to run the script in debug mode"""

    tb_dir: Optional[str] = "runs"
    """the directory to save the tensorboard logs"""
    tb_offset: int = 0
    """the step offset of the tensorboard logs"""
    run_name: Optional[str] = None
    """the name of the tensorboard run"""
    ckpt_dir: str = "checkpoints"
    """the directory to save the model checkpoints"""

    # Algorithm specific arguments
    env_id: str = "Duel-v1"
    """the id of the environment"""
    deck: str = "../assets/deck"
    """the deck file to use"""
    deck1: Optional[str] = None
    """the deck file for the first player"""
    deck2: Optional[str] = None
    """the deck file for the second player"""
    deck_schedule: str = "cluster_uniform"
    """paired deck sampler: independent, cluster_uniform, or anchor_half"""
    anchor_deck: str = ""
    """the deck included in anchor_probability_percent of duels"""
    anchor_probability_percent: int = 0
    """percentage of duels containing exactly one anchor deck"""
    code_list_file: str = "code_list.txt"
    cards_db: Optional[str] = None
    """the card database; defaults to the package's assets/locale/en/cards.cdb"""
    """the code list file for card embeddings"""
    embedding_file: Optional[str] = None
    """the embedding file for card embeddings"""
    semantic_file: Optional[str] = "../assets/structured/frozen_semantics_v1.npz"
    """frozen exact-CDB and effect-level Lua semantic cache"""
    max_options: int = 64
    """the maximum number of options"""
    announce_tables: Optional[str] = None
    """the announce law's content-addressed tables (announce-tables-<sha256>.json); required"""
    announce_cap: int = 192
    """the announce law's candidate cap (a registered law parameter, at least 192 under announce_public_candidates/v2;
    tiers 1-4 are never truncated; must not exceed max_options)"""
    announce_room_format: Optional[str] = None
    """the room format the announce belief restricts the library to (None: every recipe)"""
    dormant_table: Optional[str] = None
    """the run's dormant-identity table (dormant-table-<sha256>.json): every duel first gets one inert object per table
    code, so the load-time duel-level registrations of card scripts no longer depend on the decks (belief-world
    search, S3/S6); part of the checkpoint identity when set"""
    room_format: Optional[str] = None
    """the room's public format: a key of --room-format-table (obs:global_ columns 23 and 24: its index and era,
    mirrorforce/agent/env/room_format.py); None: index 0 (unknown); part of the checkpoint identity when set"""
    room_format_table: Optional[str] = None
    """the registered room format table (room-formats-<sha256>.json) --room-format names a key of"""
    card_tables: Optional[str] = None
    """the run's card tables (card-tables-<sha256>.npz, mirrorforce/agent/env/card_tables.py): setcode bag, link
    arrows, genericity and effect-unit evidence, written into the model's constants; bound to --code-list-file and
    --semantic-file; part of the checkpoint identity when set"""
    public_opponent_recipe: bool = False
    """declare the opponent's decklist public (a mirror or an open-decklist mode): the observation carries it
    (obs:opponent_recipe_) and belief-world search may place particles drawn from it (search plan S1/S2); part of
    the checkpoint identity when set"""
    allow_unreviewed_public_effects: bool = False
    critic: bool = False
    """A0: train a central critic (mirrorforce/agent/model/critic.py) on both seats' observations alongside the policy;
    needs --export-both-seats"""
    critic_advantages: bool = False
    """A0: the policy's GAE(0.5) advantages come from the central critic's values (else the public value head's)"""
    critic_model: PolicyNetConfig = field(default_factory=lambda: PolicyNetConfig(
        d=256, heads=8, ff=1024, state_layers=3, turn_layers=1, readout_layers=1, semantic_dim=64, event_hidden=256,
        action_hidden=256, readout_hidden=256, belief_width=64, belief_heads=4, belief_layers=1, remat=True,
        dtype="bfloat16", value_menu=True))
    critic_mix_layers: int = 2
    critic_chunks: int = 4
    critic_td_lambda: Optional[float] = None
    """the critic's TD(lambda) for its own targets (None: the recipe's td_lambda); 1.0 gives Monte-Carlo outcomes
    for games that end within the iteration"""
    critic_zero_init: bool = False
    """the critic's output layer starts at zero (uniform first predictions)"""
    resume_reset_critic: bool = False
    """the resumed checkpoint's critic is dropped and a fresh one starts at this iteration boundary (its
    configuration may change; logged CRITIC-RESET, a0.critic_started)"""
    """the critic's forward and step over a device's minibatch in this many row chunks (gradients summed)"""
    resume_add_critic: bool = False
    advantages_switch_evidence: Optional[str] = None
    """a JSON file with the held-out calibration that justifies switching the policy's advantages to the central
    critic; required at that switch (ADVANTAGES-SWITCH), embedded in every later receipt (a0.advantages_switch)"""
    resume_declare: List[str] = field(default_factory=list)
    """dotted identity keys a resume declares changed (e.g. recipe.belief_rows when --iteration-kept-rows starts on
    a run without it): logged DECLARED-CHANGE with the old and new values, recorded in every later receipt"""
    resume_env_change: bool = False
    """the resumed checkpoint was trained on another environment build (native module, core, window or observation
    laws; the model's inputs are unchanged): declared, logged ENV-CHANGE with the old and new values and recorded in
    every later receipt (env_changes); environments restart"""
    resume_add_value_menu: bool = False
    """declare the one-time legacy -> zero-output legal-menu residual upgrade for public V and the existing
    critic. Keeps old parameters, Adam counts/moments, EMA, keys and counters; only new module leaves initialize.
    Does not imply a deployment restart or an environment-law change (declare --resume-env-change separately)."""
    """A0: the resumed checkpoint has no central critic (nor the both-seat export); the critic starts fresh at this
    iteration boundary (declared; logged CRITIC-ADDED and recorded as a0.critic_started in every later receipt)"""
    export_both_seats: bool = False
    """the env also exports the other seat's observation (priv: keys in info["priv"]) for a central critic;
    the acting seat's observation is unchanged; never a policy input"""
    """let pool duels hold cards outside the env's public effect table (public_effects/v1 is reviewed card by card;
    by default such a pool is refused): tests and audits of other pools, whose cards' lingering effects are then not
    shown; part of the checkpoint identity when set"""
    max_steps: int = 1000
    """maximum environment decisions before a draw"""
    n_history_actions: int = 32
    """the number of history actions to use"""
    greedy_reward: bool = False
    """whether to use greedy reward (faster kill higher reward)"""

    total_timesteps: int = 50000000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    local_num_envs: int = 128
    """the number of parallel game environments"""
    local_env_threads: Optional[int] = None
    """the number of threads to use for environment"""
    num_actor_threads: int = 2
    """the number of actor threads to use"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    collect_steps: Optional[int] = None
    """the number of steps to compute the advantages"""
    segment_length: Optional[int] = None
    """the length of the segment for training"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0
    """the discount factor gamma"""
    num_minibatches: int = 64
    """the number of mini-batches"""
    update_epochs: int = 1
    """the K epochs to update the policy"""
    switch: bool = False
    """Toggle the use of switch mechanism"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    burn_in_steps: Optional[int] = None
    """the number of burn-in steps for training (for R2D2)"""

    upgo: bool = False
    """Toggle the use of UPGO for advantages"""
    sep_value: bool = True
    """Whether separate value function computation for each player"""
    value: Literal["vtrace", "gae"] = "gae"
    """the method to learn the value function"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    c_clip_min: float = 0.001
    """the minimum value of the importance sampling clipping"""
    c_clip_max: float = 1.007
    """the maximum value of the importance sampling clipping"""
    rho_clip_min: float = 0.001
    """the minimum value of the importance sampling clipping"""
    rho_clip_max: float = 1.007
    """the maximum value of the importance sampling clipping"""

    ppo_clip: bool = True
    """whether to use the PPO clipping to replace V-Trace surrogate clipping"""
    clip_coef: float = 0.25
    """the PPO surrogate clipping coefficient"""
    dual_clip_coef: Optional[float] = 3.0
    """the dual surrogate clipping coefficient, typically 3.0"""
    spo_kld_max: Optional[float] = None
    """the maximum KLD for the SPO policy, typically 0.02"""
    logits_threshold: Optional[float] = None
    """the logits threshold for NeuRD and ACH, typically 2.0-6.0"""

    vloss_clip: Optional[float] = None
    """the value loss clipping coefficient"""

    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""

    m1: ModelArgs = field(default_factory=lambda: ModelArgs())
    """the model arguments for the agent"""
    architecture: Literal["fork", "policy_net"] = "fork"
    """the network: the fork's model (m1/m2) or the structure in JAX (; design section 8)"""
    net: PolicyNetConfig = field(default_factory=lambda: PolicyNetConfig(value_menu=True))
    """the policy_net configuration (architecture policy_net); its dtype field sets the compute dtype"""
    recipe: Literal["fork", "ataraxos"] = "fork"
    """the training update: the fork's PPO, or arm A's Ataraxos dynamically damped self-play (policy_net only;
    mirrorforce/agent/train/ataraxos.py; constants in --ataraxos.<field>, part of the checkpoint identity)"""
    ataraxos: AtaraxosConfig = field(default_factory=AtaraxosConfig)
    grad_accum: int = 1
    """micro-batches per optimizer step on each learner device (exact accumulation: every loss term of a micro-batch
    is normalized by the whole minibatch's count, so the summed gradients equal the minibatch's gradient); a
    hardware detail recorded in the receipt -- the recipe is num_minibatches optimizer steps of the global minibatch"""
    belief_coef: float = 0.0
    """policy_net: weight of the belief head's loss (engine-truth label:hidden_ per public candidate and hidden location;
    the head reads the state block through stop-gradient, so the policy is unaffected); > 0 turns the labels on"""
    actor_blocks: bool = False
    """actor inference on fixed-length menu blocks: each step's batch is sliced to menu {32, max_options} by its
    longest menu (design 8.3; event rows and chunk slots switch on the device inside policy_net); both shapes compile at
    warmup, any later compile is fatal"""
    m2: ModelArgs = field(default_factory=lambda: ModelArgs())
    """the model arguments for the eval agent"""

    actor_device_ids: List[int] = field(default_factory=lambda: [0, 1])
    """the device ids that actor workers will use"""
    learner_device_ids: List[int] = field(default_factory=lambda: [2, 3])
    """the device ids that learner workers will use"""
    distributed: bool = False
    """whether to use `jax.distirbuted`"""
    coordinator_address: str = ""
    """multi-process (M2): host:port of process 0's coordinator"""
    num_processes: int = 1
    process_id: int = 0
    resume_topology_change: bool = False
    """resume on another device layout (a declared non-exact resume at an iteration boundary): parameters, Adam
    state, EMA and the iteration counter carry over; environments, actor memory and learner keys start afresh"""
    concurrency: bool = True
    """whether to run the actor and learner concurrently"""
    bfloat16: bool = False
    """whether to use bfloat16 for the agent"""
    thread_affinity: bool = False
    """whether to use thread affinity for the environment"""

    eval_checkpoint: Optional[str] = None
    """the path to the model checkpoint to evaluate"""
    local_eval_episodes: int = 128
    """the number of episodes to evaluate the model"""
    eval_interval: int = 100
    """the number of iterations to evaluate the model"""

    # runtime arguments to be filled in
    local_batch_size: int = 0
    local_minibatch_size: int = 0
    world_size: int = 0
    local_rank: int = 0
    num_envs: int = 0
    batch_size: int = 0
    minibatch_size: int = 0
    num_updates: int = 0
    global_learner_decices: Optional[List[str]] = None
    actor_devices: Optional[List[str]] = None
    learner_devices: Optional[List[str]] = None
    num_embeddings: Optional[int] = None
    freeze_id: Optional[bool] = None
    semantic_shape: Optional[tuple[int, int, int, int]] = None
    semantic_metadata: Optional[dict] = None
    deck_names: Optional[List[str]] = None
    real_seed: Optional[int] = None


def make_env(args, seed, num_envs, num_threads, mode='self', thread_affinity_offset=-1, eval=False):
    if not args.thread_affinity:
        thread_affinity_offset = -1
    if thread_affinity_offset >= 0:
        print("Binding to thread offset", thread_affinity_offset)
    envs = mirrorforce.agent.env.make(
        task_id=args.env_id,
        env_type="gymnasium",
        num_envs=num_envs,
        num_threads=num_threads,
        thread_affinity_offset=thread_affinity_offset,
        seed=seed,
        deck1=args.deck1,
        deck2=args.deck2,
        deck_schedule=args.deck_schedule,
        anchor_deck=args.anchor_deck,
        anchor_probability_percent=args.anchor_probability_percent,
        mirror_probability_percent=args.mirror_probability_percent,
        matched_probability_percent=args.matched_probability_percent,
        pool_deck_uniform=args.pool_deck_uniform,
        max_options=args.max_options,
        max_steps=args.max_steps,
        n_history_actions=args.n_history_actions,
        async_reset=False,
        greedy_reward=args.greedy_reward,
        play_mode=mode,
        timeout=args.timeout,
        oppo_info=False,
        public_opponent_recipe=args.public_opponent_recipe,
        **({"allow_unreviewed_public_effects": True} if args.allow_unreviewed_public_effects else {}),
        belief_labels=args.belief_coef > 0,
        **({"export_both_seats": True} if args.export_both_seats else {}),
        **room_format.env_config(room_format.from_args(args)),
    )
    envs.num_envs = num_envs
    return envs


class Transition(NamedTuple):
    obs: list
    dones: list
    actions: list
    logits: list
    values: list
    rewards: list
    mains: list
    next_dones: list
    outcome_targets: list = None
    outcome_valid: list = None
    belief_labels: list = None
    value_dists: list = None
    withdrawn: list = None  # [envs] bool: a decision the environment withdrew afterwards (not a transition)


def create_agent(args, eval=False):
    if args.recipe == "ataraxos" and (args.architecture != "policy_net" or args.lp_shaping_coef):
        raise ValueError("arm A (--recipe ataraxos) runs policy_net on the outcome reward only (no LP shaping)")
    if args.architecture == "policy_net":
        if args.bfloat16 or args.decision_aux_coef or args.switch:
            raise ValueError("policy_net takes its dtype from --net.dtype and has no decision auxiliary or switch mode")
        return ReferenceAgent(args.net, args.semantic_shape)
    if eval:
        return RNNAgent(
            embedding_shape=args.num_embeddings,
            semantic_shape=args.semantic_shape,
            dtype=jnp.bfloat16 if args.bfloat16 else jnp.float32,
            param_dtype=jnp.float32,
            **asdict(args.m2),
        )
    else:
        return RNNAgent(
            embedding_shape=args.num_embeddings,
            semantic_shape=args.semantic_shape,
            dtype=jnp.bfloat16 if args.bfloat16 else jnp.float32,
            param_dtype=jnp.float32,
            switch=args.switch,
            freeze_id=args.freeze_id,
            **asdict(args.m1),
        )


def get_variables(agent_state):
    batch_stats = getattr(agent_state, "batch_stats", None)
    variables = {'params': agent_state.params}
    if batch_stats is not None:
        variables['batch_stats'] = batch_stats
    constants = getattr(agent_state, "constants", None)
    if constants is not None:
        variables['constants'] = constants
    return variables


def load_structured_semantics(path: str, code_list_file: str):
    cache_path = os.path.abspath(path)
    with np.load(cache_path, allow_pickle=False) as payload:
        required = {"cdb_exact", "lua_effects", "lua_mask", "metadata"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(
                f"semantic cache is missing arrays: {sorted(missing)}"
            )
        cdb_exact = np.asarray(payload["cdb_exact"], dtype=np.float32)
        lua_effects = np.asarray(payload["lua_effects"], dtype=np.float16)
        lua_mask = np.asarray(payload["lua_mask"], dtype=np.uint8)
        metadata = json.loads(str(payload["metadata"].item()))

    with open(code_list_file, "r", encoding="utf-8") as handle:
        codes = [int(line.split()[0]) for line in handle if line.split()]
    expected_rows = len(codes) + 1
    if cdb_exact.ndim != 2:
        raise ValueError(f"cdb_exact must be rank 2, got {cdb_exact.shape}")
    if lua_effects.ndim != 3:
        raise ValueError(
            f"lua_effects must be rank 3, got {lua_effects.shape}"
        )
    if lua_mask.shape != lua_effects.shape[:2]:
        raise ValueError(
            f"lua_mask shape {lua_mask.shape} does not match "
            f"lua_effects {lua_effects.shape}"
        )
    if cdb_exact.shape[0] != expected_rows or lua_effects.shape[0] != expected_rows:
        raise ValueError(
            f"semantic row count must be {expected_rows}, got "
            f"{cdb_exact.shape[0]} and {lua_effects.shape[0]}"
        )
    code_list_hash = __import__("hashlib").sha256(
        "\n".join(str(code) for code in codes).encode()
    ).hexdigest()
    if metadata.get("code_list_sha256") != code_list_hash:
        raise ValueError("semantic cache code_list hash does not match")
    semantic_shape = (
        cdb_exact.shape[0],
        cdb_exact.shape[1],
        lua_effects.shape[1],
        lua_effects.shape[2],
    )
    tables = {
        "cdb_exact": cdb_exact,
        "lua_effects": lua_effects,
        "lua_mask": lua_mask,
    }
    return tables, semantic_shape, metadata


def inject_semantic_constants(tree, tables):
    matches = []

    def visit(node):
        if not isinstance(node, dict):
            return
        if set(tables).issubset(node):
            matches.append(node)
        for value in node.values():
            visit(value)

    visit(tree)
    if len(matches) != 1:
        raise ValueError(
            f"expected one structured semantic constant collection, "
            f"found {len(matches)}"
        )
    destination = matches[0]
    for name, value in tables.items():
        if tuple(destination[name].shape) != tuple(value.shape):
            raise ValueError(
                f"semantic tensor {name} shape mismatch: "
                f"{destination[name].shape} != {value.shape}"
            )
        destination[name] = jnp.asarray(value)


def reshape_minibatch(
    x, multi_step, num_minibatches, num_steps, segment_length=None, key=None):
    # if segment_length is None,
    #   n_mb = num_minibatches
    #   if multi_step, from (num_steps, num_envs, ...)) to
    #     (n_mb, num_steps * (num_envs // n_mb), ...)
    #   else, from (num_envs, ...) to
    #     (n_mb, num_envs // n_mb, ...)
    # else,
    #   n_mb_t = num_steps // segment_length
    #   n_mb_e = num_minibatches // n_mb_t
    #   if multi_step, from (num_steps, num_envs, ...)) to
    #     (n_mb_e, n_mb_t, segment_length * (num_envs // n_mb_e), ...)
    #   else, from (num_envs, ...) to
    #     (n_mb_e, num_envs // n_mb_e, ...)
    if key is not None:
        x = jax.random.permutation(key, x, axis=1 if multi_step else 0)

    N = num_minibatches
    if segment_length is None:
        if multi_step:
            x = jnp.reshape(x, (num_steps, N, -1) + x.shape[2:])
            x = x.transpose(1, 0, *range(2, x.ndim))
            x = x.reshape(N, -1, *x.shape[3:])
        else:
            x = jnp.reshape(x, (N, -1) + x.shape[1:])
    else:
        M = segment_length
        Nt = num_steps // M
        Ne = N // Nt
        if multi_step:
            x = jnp.reshape(x, (Nt, M, Ne, -1) + x.shape[2:])
            x = x.transpose(2, 0, 1, *range(3, x.ndim))
            x = jnp.reshape(x, (Ne, Nt, -1) + x.shape[4:])
        else:
            x = jnp.reshape(x, (Ne, -1) + x.shape[1:])
    return x


def advantage_fn(
    args, next_v, values, rewards, next_dones, switch_or_mains, ratios=None, return_carry=False):
    if args.switch:
        if args.value == "vtrace" or args.sep_value or return_carry:
            raise NotImplementedError
        return gae_sep_switch(
            next_v, values, rewards, next_dones, switch_or_mains,
            args.gamma, args.gae_lambda, args.upgo)
    else:
        # TODO: TD(lambda) for multi-step
        if args.value == "gae":
            adv_fn = truncated_gae_sep if args.sep_value else truncated_gae
            return adv_fn(
                next_v, values, rewards, next_dones, switch_or_mains,
                args.gamma, args.gae_lambda, args.upgo, return_carry=return_carry)
        else:
            adv_fn = vtrace_sep if args.sep_value else vtrace
            if ratios is None:
                ratios = jnp.ones_like(values)
            return adv_fn(
                next_v, ratios, values, rewards, next_dones, switch_or_mains, args.gamma,
                args.rho_clip_min, args.rho_clip_max, args.c_clip_min, args.c_clip_max,
                args.upgo, return_carry=return_carry)


def rollout(key, args, rollout_queue, params_queue, writer, actor_device, learner_devices, device_thread_id,
            resume=None):
    """One actor thread, with its own actor device as the thread's default device.

    Host arrays (observations, rewards) reached the jitted helpers uncommitted, so they
    defaulted to the first local device; with a second actor device a helper compiled for
    one device was then handed buffers committed to the other ("Buffer passed to Execute()
    ... is on device cuda:1, but replica is assigned to device cuda:0").
    jax.default_device is thread-local, so each actor thread places its host inputs on its
    own device."""
    with jax.default_device(actor_device):
        return _rollout(key, args, rollout_queue, params_queue, writer, actor_device, learner_devices,
                        device_thread_id, resume)


def _rollout(
    key: jax.random.PRNGKey,
    args: Args,
    rollout_queue,
    params_queue,
    writer,
    actor_device,
    learner_devices,
    device_thread_id,
    resume=None,
):
    eval_mode = 'self' if args.eval_checkpoint else 'bot'
    if eval_mode != 'bot':
        eval_params = params_queue.get()

    local_seed = args.real_seed + device_thread_id * args.local_num_envs
    np.random.seed(local_seed)

    envs = make_env(
        args,
        local_seed,
        args.local_num_envs,
        args.local_env_threads,
        thread_affinity_offset=device_thread_id * args.local_env_threads,
    )
    envs = EnvPreprocess(envs, skip_mask=True)
    envs = (RecordClusterEpisodeStatistics(envs) if args.cluster_stats_dir
            else RecordEpisodeStatistics(envs))

    eval_envs = make_env(
        args,
        local_seed + 100000,
        args.local_eval_episodes,
        args.local_eval_episodes // 4, mode=eval_mode, eval=True)
    eval_envs = EnvPreprocess(eval_envs, skip_mask=True)
    eval_envs = RecordEpisodeStatistics(eval_envs)

    len_actor_device_ids = len(args.actor_device_ids)
    n_actors = args.num_actor_threads * len_actor_device_ids
    global_step = 0
    start_time = time.time()
    warmup_step = 0
    other_time = 0
    avg_ep_returns = deque(maxlen=1000)
    avg_win_rates = deque(maxlen=1000)
    anchor_episode_count = 0
    finished_episode_count = 0
    anchor_seat_counts = [0, 0]
    deck_cluster_samples: Counter[str] = Counter()
    decision_seat_counts = [0, 0]
    main_decision_count = 0
    opponent_decision_count = 0

    agent = create_agent(args)
    apply_fn = agent.apply
    eval_agent = create_agent(args, eval=eval_mode != 'bot')
    eval_apply_fn = eval_agent.apply

    @jax.jit
    def get_action(params, obs, rstate):
        rstate, logits = eval_apply_fn(params, obs, rstate)[:2]
        return rstate, logits.argmax(axis=1)

    @jax.jit
    def get_action_battle(params1, params2, obs, rstate1, rstate2, main, done):
        next_rstate1, logits1 = apply_fn(params1, obs, rstate1)[:2]
        next_rstate2, logits2 = eval_apply_fn(params2, obs, rstate2)[:2]
        logits = jnp.where(main[:, None], logits1, logits2)
        rstate1 = select_rows(main, next_rstate1, rstate1)
        rstate2 = select_rows(main, rstate2, next_rstate2)
        rstate1, rstate2 = select_rows(done, jax.tree.map(jnp.zeros_like, (rstate1, rstate2)), (rstate1, rstate2))
        return rstate1, rstate2, logits.argmax(axis=1)

    @jax.jit
    def sample_action(
        params, next_obs, rstate1, rstate2, main, done, key):
        # policy_net returns its delivery check as an output (checked on the host with the action): a host callback
        # inside the actor's program waits for the GIL that the other actor threads hold
        outputs = apply_fn(params, next_obs, (rstate1, rstate2), done, main,
                           **({"return_bad": True} if args.architecture == "policy_net" else {}))
        (rstate1, rstate2), logits, value = outputs[:3]
        bad = outputs[4] if args.architecture == "policy_net" else None
        value = jnp.squeeze(value, axis=-1)
        # policy_net's (win, draw, loss) distribution of the acting seat (arm A's categorical targets)
        value_dist = jax.nn.softmax(outputs[3], axis=-1) if args.architecture == "policy_net" else None
        action, key = categorical_sample(logits, key)
        return next_obs, done, main, rstate1, rstate2, action, logits, value, key, value_dist, bad

    @jax.jit
    def bootstrap_dist(params, next_obs, rstate1, rstate2, main, done):
        """The main seat's (win, draw, loss) at the observation after a segment (arm A's bootstrap)."""
        outputs = apply_fn(params, next_obs, (rstate1, rstate2), done, main, return_bad=True)
        dist = jax.nn.softmax(outputs[3], axis=-1)
        return jnp.where(main[:, None], dist, ataraxos.mirror(dist)), outputs[4]

    @jax.jit
    def compute_advantage_carry(
        next_value, values, rewards, next_dones, mains):
        return advantage_fn(
            args, next_value, values, rewards, next_dones, mains, return_carry=True)

    # one jitted function per process for all actor threads (jit specializes per device on first use there), so a
    # thread on a device another thread has compiled for reuses its executable instead of compiling again
    sample_action = shared_jit("sample_action", sample_action)
    bootstrap_dist = shared_jit("bootstrap_dist", bootstrap_dist)
    get_action = first_call_serialized(get_action)
    get_action_battle = first_call_serialized(get_action_battle)
    sample_action = first_call_serialized(sample_action)
    bootstrap_dist = first_call_serialized(bootstrap_dist)
    compute_advantage_carry = first_call_serialized(compute_advantage_carry)

    deck_names = args.deck_names
    cluster_recorder = None
    if args.cluster_stats_dir:
        from mirrorforce.agent.train.cluster_stats import Recorder
        cluster_recorder = Recorder(args.cluster_stats_run, device_thread_id, deck_names)
    deck_cluster_names = sorted(
        {deck_cluster_name(name) for name in deck_names}
    )
    deck_avg_times = {name: 0 for name in deck_names}
    deck_max_times = {name: 0 for name in deck_names}
    deck_time_count = {name: 0 for name in deck_names}

    # put data in the last index
    params_queue_get_time = deque(maxlen=10)
    rollout_time = deque(maxlen=10)
    actor_policy_version = 0
    next_obs, info = envs.reset()
    next_labels = np.asarray(info["label"]["hidden_"]) if args.belief_coef else None
    next_priv = info["priv"] if args.critic else None  # the other seat's view at the decision (central critic)
    next_to_play = info["to_play"]
    next_done = np.zeros(args.local_num_envs, dtype=np.bool_)
    next_rstate1 = next_rstate2 = agent.init_rnn_state(args.local_num_envs)

    eval_rstate1 = agent.init_rnn_state(args.local_eval_episodes)
    eval_rstate2 = eval_agent.init_rnn_state(args.local_eval_episodes)

    next_rstate1, next_rstate2, eval_rstate1, eval_rstate2 = \
        jax.device_put([next_rstate1, next_rstate2, eval_rstate1, eval_rstate2], actor_device)

    main_player = np.concatenate([
        np.zeros(args.local_num_envs // 2, dtype=np.int64),
        np.ones(args.local_num_envs // 2, dtype=np.int64)
    ])
    np.random.shuffle(main_player)
    guard_monitor = GuardMonitor()
    calibration = None
    if args.recipe == "ataraxos":
        from mirrorforce.agent.train.calibration_monitor import CalibrationMonitor
        calibration = CalibrationMonitor(args.local_num_envs)
    window_monitor = None
    if "turn_chunk_meta_" in next_obs:
        from mirrorforce.agent.train.window_monitor import WindowMonitor
        window_monitor = WindowMonitor(deck_names, np.asarray(next_obs["turn_events_"]).shape[1],
                                       np.asarray(next_obs["turn_chunks_"]).shape[2], args.net.chunk_slots)
    info_current = resume is None  # the reset's infos describe the restored environments only without a resume
    start_update = getattr(args, "actor_start_update", 1)  # after a topology change: the iteration's first update
    if resume is not None:  # exact resume: this thread's state at the top of rollout resume["update"]
        envs.unwrapped.import_states(list(resume["envs"]))
        next_obs = {k: np.asarray(v) for k, v in resume["next_obs"].items()}
        next_obs.update({k: None for k in resume["next_obs_none"]})
        if args.belief_coef:
            next_labels = np.asarray(resume["next_labels"])
        if args.critic:
            next_priv = {k: np.asarray(v) for k, v in resume["next_priv"].items()}
        next_to_play = np.asarray(resume["next_to_play"])
        next_done = np.asarray(resume["next_done"], dtype=np.bool_)
        main_player = np.asarray(resume["main_player"])
        structure = jax.tree.structure((next_rstate1, next_rstate2))
        if structure.num_leaves != len(resume["rstate"]):
            raise ValueError("the checkpoint's actor recurrent state does not match the model")
        next_rstate1, next_rstate2 = jax.device_put(
            jax.tree.unflatten(structure, [np.asarray(x) for x in resume["rstate"]]), actor_device)
        key = jax.device_put(np.asarray(resume["key"]), actor_device)
        global_step = int(resume["global_step"])
        actor_policy_version = int(resume["actor_policy_version"])
        start_update = int(resume["update"])
    start_step = 0
    storage = []

    init_rstates = []

    @jax.jit
    def prepare_data(storage: List[Transition]) -> Transition:
        return jax.tree.map(lambda *xs: jnp.stack(xs), *storage)

    prepare_data = first_call_serialized(prepare_data)

    blocks_warm = False
    priv_steps = []  # A0 with a central critic: the other seat's view at each stored decision
    previous_withdrawn = None  # A0: the previous update's withdrawn flags, while its data wait for training
    pack_pool = (concurrent.futures.ThreadPoolExecutor(max(1, args.iteration_threads // 4))
                 if args.iteration_decisions else None)

    @jax.jit
    def a0_targets(storage_small, next_dist):
        value_dists, rewards, not_decision, next_dones, mains = storage_small
        return ataraxos.targets_by_seat(value_dists, rewards, not_decision, next_dones, mains, next_dist,
                                        args.ataraxos)
    a0_targets = first_call_serialized(shared_jit("a0_targets", a0_targets))

    for update in range(start_update, args.num_updates + start_update + 1):
        publish_snapshot(device_thread_id, actor_snapshot(
            update, envs, next_obs, next_to_play, next_done, main_player, (next_rstate1, next_rstate2), key,
            global_step, actor_policy_version, next_labels, next_priv))
        if update == 10:
            start_time = time.time()
            warmup_step = global_step

        update_time_start = time.time()
        inference_time = 0
        env_time = 0
        params_queue_get_time_start = time.time()
        if args.iteration_decisions:  # A0: one behaviour policy per iteration
            if (update - 1) % args.iteration_updates == 0:
                params = params_queue.get()
                actor_policy_version += 1
                previous_withdrawn = None  # the previous iteration's data are trained
        elif args.concurrency:
            if update != start_update + 1:  # the second rollout reuses the first parameters (two-version lag)
                params = params_queue.get()
                # params["params"]["Encoder_0"]['Embed_0']["embedding"].block_until_ready()
                actor_policy_version += 1
        else:
            params = params_queue.get()
            actor_policy_version += 1
        params_queue_get_time.append(time.time() - params_queue_get_time_start)
        if args.actor_blocks and not blocks_warm:  # compile the declared shapes now (outputs discarded)
            for menu in (MENU_BLOCK, args.max_options):
                jax.block_until_ready(sample_action(
                    params, sliced_obs(next_obs, menu), next_rstate1, next_rstate2,
                    next_to_play == main_player, next_done, key))
            blocks_warm = True

        rollout_time_start = time.time()
        for k in range(start_step, args.collect_steps):
            if k % args.num_steps == 0:
                init_rstate1, init_rstate2 = jax.tree.map(
                    lambda x: x.copy(), (next_rstate1, next_rstate2))
                init_rstates.append((init_rstate1, init_rstate2))
            global_step += args.local_num_envs * n_actors * args.world_size

            main = next_to_play == main_player
            decision_seat_counts[0] += int(np.count_nonzero(next_to_play == 0))
            decision_seat_counts[1] += int(np.count_nonzero(next_to_play == 1))
            main_decision_count += int(np.count_nonzero(main))
            opponent_decision_count += int(main.size - np.count_nonzero(main))

            if window_monitor is not None and info_current:  # after a resume, from the first step's infos on
                window_monitor.observe(next_obs, info)
            inference_time_start = time.time()
            if args.actor_blocks:
                menu = menu_block(next_obs, args.max_options)
                _, cached_next_done, cached_main, next_rstate1, next_rstate2, action, logits, value, key, \
                    value_dist, bad = sample_action(params, sliced_obs(next_obs, menu), next_rstate1,
                                                    next_rstate2, main, next_done, key)
                cached_next_obs = next_obs  # the full observation is stored; the learner reads full shapes
                if menu < args.max_options:
                    logits = jnp.pad(logits, ((0, 0), (0, args.max_options - menu)), constant_values=-1e9)
            else:
                cached_next_obs, cached_next_done, cached_main, \
                    next_rstate1, next_rstate2, action, logits, value, key, value_dist, bad = sample_action(
                    params, next_obs, next_rstate1, next_rstate2, main, next_done, key)

            cpu_action = np.array(action)
            if bad is not None and np.asarray(bad).any():
                raise RuntimeError("policy_net: inconsistent chunk or closed-turn delivery at an actor step")
            acting_seats = np.asarray(next_to_play)
            if calibration is not None and info_current:
                calibration.record(next_obs, value_dist, acting_seats, next_done)
            inference_time += time.time() - inference_time_start

            _start = time.time()
            if args.lp_shaping_coef:
                lp_prev_pair = lp_pair(next_obs)
                acting_pair_seat = np.array(next_to_play, dtype=np.int64)
            cached_labels = next_labels
            if args.critic:
                priv_steps.append({f"priv:{k}": np.asarray(v) for k, v in next_priv.items()})
            next_obs, next_reward, next_done, info = envs.step(cpu_action)
            if args.critic:
                next_priv = info["priv"]
            if calibration is not None and info_current:
                calibration.finish(next_done, next_reward, acting_seats)
            next_labels = np.asarray(info["label"]["hidden_"]) if args.belief_coef else None
            if args.lp_shaping_coef:
                lp_cur_pair = lp_pair(next_obs)
                lost0 = np.maximum(lp_prev_pair[:, 0] - lp_cur_pair[:, 0], 0.0)
                lost1 = np.maximum(lp_prev_pair[:, 1] - lp_cur_pair[:, 1], 0.0)
                acting_lost = np.where(acting_pair_seat == 0, lost0, lost1)
                other_lost = np.where(acting_pair_seat == 0, lost1, lost0)
                shaping = args.lp_shaping_coef * np.clip(
                    (other_lost - acting_lost) / args.lp_shaping_scale, -1.0, 1.0)
                shaping = np.where(np.asarray(next_done, dtype=bool), 0.0, shaping)
                next_reward = (np.asarray(next_reward, dtype=np.float64) + shaping).astype(np.float32)
                lp_shaping_applied = float(np.mean(np.abs(shaping))) if shaping.size else 0.0
            next_to_play = info["to_play"]
            info_current = True
            env_time += time.time() - _start

            targets, target_valid = None, None
            if args.decision_aux_coef:
                targets, target_valid = public_outcome_targets(
                    cached_next_obs, next_obs, next_done, cached_next_done
                )
            storage.append(
                Transition(
                    obs=cached_next_obs,
                    dones=cached_next_done,
                    mains=cached_main,
                    actions=action,
                    logits=logits,
                    values=value,
                    rewards=next_reward,
                    next_dones=next_done,
                    outcome_targets=targets,
                    outcome_valid=target_valid,
                    belief_labels=cached_labels,
                    value_dists=value_dist if args.recipe == "ataraxos" else None,
                    withdrawn=np.zeros(args.local_num_envs, np.bool_) if args.recipe == "ataraxos" else None,
                )
            )
            mark_withdrawn(storage, info, guard_monitor, args.recipe, previous_withdrawn)

            for idx, d in enumerate(next_done):
                if not d:
                    continue
                cur_main = main[idx]
                if args.switch:
                    for j in reversed(range(len(storage) - 1)):
                        t = storage[j]
                        if t.next_dones[idx]:
                            # For OTK where player may not switch
                            break
                        if t.mains[idx] != cur_main:
                            t.next_dones[idx] = True
                            t.rewards[idx] = -next_reward[idx]
                            break
                
                if args.time_log_freq:
                    for i in range(2):
                        deck_time = info['step_time'][idx][i]
                        deck_name = deck_names[info['deck'][idx][i]]

                        time_count = deck_time_count[deck_name]
                        avg_time = deck_avg_times[deck_name]
                        avg_time = avg_time * (time_count / (time_count + 1)) + deck_time / (time_count + 1)
                        max_time = max(deck_time, deck_max_times[deck_name])
                        deck_avg_times[deck_name] = avg_time
                        deck_max_times[deck_name] = max_time
                        deck_time_count[deck_name] += 1
                        if deck_time_count[deck_name] % args.time_log_freq == 0:
                            print(f"Deck {deck_name}, avg: {avg_time * 1000:.2f}, max: {max_time * 1000:.2f}")

                episode_reward = info['r'][idx] * (1 if cur_main else -1)
                win = 1 if episode_reward > 0 else 0
                avg_ep_returns.append(episode_reward)
                avg_win_rates.append(win)
                anchor_seat = int(info["anchor_seat"][idx])
                finished_episode_count += 1
                if anchor_seat:
                    anchor_episode_count += 1
                    anchor_seat_counts[anchor_seat - 1] += 1
                for seat in range(2):
                    deck_name = deck_names[int(info["deck"][idx][seat])]
                    deck_cluster_samples[deck_cluster_name(deck_name)] += 1
                if cluster_recorder is not None:
                    acting_seat = int(main_player[idx]) if main[idx] else 1 - int(main_player[idx])
                    cluster_recorder.record(
                        env=idx, step=args.tb_offset + global_step, deck_indices=info["deck"][idx],
                        acting_seat=acting_seat, reward=float(next_reward[idx]),
                        terminated=info["_cluster_terminated"][idx],
                        truncated=info["_cluster_truncated"][idx], length=info["l"][idx],
                        selfplay=info["is_selfplay"][idx], win_reason=info["win_reason"][idx])

        if cluster_recorder is not None:
            cluster_recorder.flush()
        rollout_time.append(time.time() - rollout_time_start)

        start_step = args.collect_steps - args.num_steps

        next_main = main_player == next_to_play
        if args.collect_steps == args.num_steps:
            storage_t = storage
            storage = []
            if args.recipe == "ataraxos":
                next_data, bad = bootstrap_dist(params, next_obs, next_rstate1, next_rstate2, next_main, next_done)
                if np.asarray(bad).any():
                    raise RuntimeError("policy_net: inconsistent chunk or closed-turn delivery at a segment's bootstrap")
            else:
                next_data = (next_obs, next_main)
        else:
            storage_t = storage[:args.num_steps]
            storage = storage[args.num_steps:]
            values, rewards, next_dones, mains = prepare_data([
                (t.values, t.rewards, t.next_dones, t.mains) for t in storage])

            next_value = sample_action(
                params, next_obs, next_rstate1, next_rstate2, next_main, next_done, key)[7]

            next_value = jnp.where(next_main, next_value, -next_value)
            adv_carry = compute_advantage_carry(
                next_value, values, rewards, next_dones, mains)
            next_data = adv_carry

        if args.iteration_decisions:  # A0: this update's segments to the host, compressed
            withdrawn = np.stack([t.withdrawn for t in storage_t])  # flags stay live for later withdrawals
            small = [np.stack([np.asarray(getattr(t, name)) for t in storage_t])
                     for name in ("value_dists", "rewards", "dones", "next_dones", "mains")]
            small[2] = small[2] | withdrawn  # reset transitions and withdrawn decisions are not decisions
            returns, advantages = a0_targets(tuple(small), next_data)
            steps = []
            for i, t in enumerate(storage_t):
                step = {k: np.asarray(v) for k, v in t.obs.items() if v is not None}
                if args.critic:
                    step.update(priv_steps[i])
                step["logits_"] = np.asarray(t.logits, np.float32)
                if args.belief_coef:
                    step["labels_"] = np.asarray(t.belief_labels)
                steps.append(step)
            init_rstate = jax.tree.map(np.asarray, init_rstates.pop(0))
            layout, segments = a0_buffer.pack_update(
                steps, {"dones": np.stack([np.asarray(t.dones) for t in storage_t]), "mains": small[4],
                        "actions": np.stack([np.asarray(t.actions) for t in storage_t]).astype(np.int32),
                        "returns": np.asarray(returns), "advantages": np.asarray(advantages),
                        "rewards": small[1], "next_dones": small[3], "value_dists": small[0],
                        "bootstrap": np.asarray(next_data)},
                withdrawn, init_rstate, pack_pool)
            priv_steps = []
            previous_withdrawn = withdrawn
            sharded_storage, sharded_data = None, (layout, segments)
        else:
            partitioned_storage = jax.tree.map(
                lambda x: jnp.split(x, len(learner_devices), axis=1), prepare_data(storage_t))
            sharded_storage = []
            for x in partitioned_storage:
                if isinstance(x, dict):
                    x = {
                        k: jax.device_put_sharded(v, devices=learner_devices) if v is not None else None
                        for k, v in x.items()
                    }
                elif x is not None:
                    x = jax.device_put_sharded(x, devices=learner_devices)
                sharded_storage.append(x)
            sharded_storage = Transition(*sharded_storage)

            init_rstate = init_rstates.pop(0)
            sharded_data = jax.tree.map(lambda x: jax.device_put_sharded(
                    np.split(x, len(learner_devices)), devices=learner_devices),
                             (init_rstate, next_data))

        if args.eval_interval and update % args.eval_interval == 0:
            _start = time.time()
            if eval_mode == 'bot':
                predict_fn = lambda *x: get_action(params, *x)
                eval_return, eval_ep_len, eval_win_rate = evaluate(
                    eval_envs, args.local_eval_episodes, predict_fn, eval_rstate2)
            else:
                predict_fn = lambda *x: get_action_battle(params, eval_params, *x)
                eval_return, eval_ep_len, eval_win_rate = battle(
                    eval_envs, args.local_eval_episodes, predict_fn, eval_rstate1, eval_rstate2)
            eval_time = time.time() - _start
            other_time += eval_time
            eval_stats = np.array([eval_time, eval_return, eval_win_rate], dtype=np.float32)
        else:
            eval_stats = None

        payload = (
            global_step,
            update,
            sharded_storage,
            *sharded_data,
            np.mean(params_queue_get_time),
            eval_stats,
        )
        rollout_queue.put(payload)

        if update % args.log_frequency == 0:
            print("GUARD " + json.dumps({"thread": device_thread_id, "update": update, **guard_monitor.interval()}),
                  flush=True)
        if update % args.log_frequency == 0 and calibration is not None:
            print("CALIBRATION " + json.dumps({"thread": device_thread_id, "update": update, **calibration.interval()}),
                  flush=True)
        if update % args.log_frequency == 0 and window_monitor is not None:
            print("WINDOW " + json.dumps({"thread": device_thread_id, "update": update,
                                          **window_monitor.interval()}), flush=True)
            if args.window_stats_dir:
                Path(args.window_stats_dir).mkdir(parents=True, exist_ok=True)
                window_monitor.write(Path(args.window_stats_dir) / f"window-stats-{device_thread_id}.json")
        if update % args.log_frequency == 0:
            episode_lengths = np.asarray(envs.returned_episode_lengths)
            avg_episodic_return = (
                float(np.mean(avg_ep_returns)) if len(avg_ep_returns) else 0.0
            )
            avg_episodic_length = (
                float(np.mean(episode_lengths))
                if episode_lengths.size
                else 0.0
            )
            max_episode_length = (
                int(np.max(episode_lengths))
                if episode_lengths.size
                else 0
            )
            SPS = int((global_step - warmup_step) / (time.time() - start_time - other_time))
            SPS_update = int(args.batch_size / (time.time() - update_time_start))

            tb_global_step = args.tb_offset + global_step

            if device_thread_id == 0:
                print(
                    f"global_step={tb_global_step}, avg_return={avg_episodic_return:.4f}, avg_length={avg_episodic_length:.0f}"
                )
                time_now = datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S")
                print(
                    f"{time_now} SPS: {SPS}, update: {SPS_update}, "
                    f"rollout_time={rollout_time[-1]:.2f}, params_time={params_queue_get_time[-1]:.2f}, "
                    f"inference_time={inference_time:.2f}, env_time={env_time:.2f}"
                )
                if finished_episode_count:
                    print(
                        "deck_schedule: "
                        f"anchor_ratio={anchor_episode_count / finished_episode_count:.3f}, "
                        f"seat0={anchor_seat_counts[0]}, "
                        f"seat1={anchor_seat_counts[1]}, "
                        f"episodes={finished_episode_count}"
                    )
                    cluster_counts = np.asarray(
                        [
                            deck_cluster_samples[name]
                            for name in deck_cluster_names
                        ],
                        dtype=np.float64,
                    )
                    cluster_mean = float(cluster_counts.mean())
                    cluster_cv = (
                        float(cluster_counts.std() / cluster_mean)
                        if cluster_mean > 0
                        else 0.0
                    )
                    clusters_seen = int(np.count_nonzero(cluster_counts))
                    print(
                        "deck_sampling: "
                        f"schedule={args.deck_schedule}, "
                        f"clusters_seen={clusters_seen}/"
                        f"{len(deck_cluster_names)}, "
                        f"samples={int(cluster_counts.sum())}, "
                        f"cv={cluster_cv:.4f}, "
                        f"min={int(cluster_counts.min())}, "
                        f"max={int(cluster_counts.max())}"
                    )
                print(
                    "shared_policy_decisions: "
                    f"seat0={decision_seat_counts[0]}, "
                    f"seat1={decision_seat_counts[1]}, "
                    f"main={main_decision_count}, "
                    f"opponent={opponent_decision_count}"
                )
            writer.add_scalar("stats/rollout_time", np.mean(rollout_time), tb_global_step)
            writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, tb_global_step)
            writer.add_scalar("charts/avg_episodic_length", avg_episodic_length, tb_global_step)
            writer.add_scalar("charts/max_episode_length", max_episode_length, tb_global_step)
            writer.add_scalar("stats/params_queue_get_time", np.mean(params_queue_get_time), tb_global_step)
            writer.add_scalar("stats/inference_time", inference_time, tb_global_step)
            writer.add_scalar("stats/env_time", env_time, tb_global_step)
            writer.add_scalar("charts/SPS", SPS, tb_global_step)
            writer.add_scalar("charts/SPS_update", SPS_update, tb_global_step)
            if finished_episode_count:
                writer.add_scalar(
                    "charts/anchor_deck_ratio",
                    anchor_episode_count / finished_episode_count,
                    tb_global_step,
                )
                cluster_counts = np.asarray(
                    [
                        deck_cluster_samples[name]
                        for name in deck_cluster_names
                    ],
                    dtype=np.float64,
                )
                cluster_mean = float(cluster_counts.mean())
                writer.add_scalar(
                    "charts/deck_clusters_seen_ratio",
                    np.count_nonzero(cluster_counts)
                    / len(deck_cluster_names),
                    tb_global_step,
                )
                writer.add_scalar(
                    "charts/deck_cluster_sampling_cv",
                    (
                        float(cluster_counts.std() / cluster_mean)
                        if cluster_mean > 0
                        else 0.0
                    ),
                    tb_global_step,
                )
            decision_count = decision_seat_counts[0] + decision_seat_counts[1]
            if decision_count:
                writer.add_scalar(
                    "charts/seat0_decision_ratio",
                    decision_seat_counts[0] / decision_count,
                    tb_global_step,
                )

    if cluster_recorder is not None:
        cluster_recorder.close()


def jsonable(value):
    """A JSON-compatible copy of a configuration value (dataclasses as dicts, other objects as strings)."""
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def window_law(module, args):
    """The environment's event-window law; policy_net's chunk carry must hold the law's per-turn chunk cap."""
    law = {"window_rows": int(module.history_window_rows), "chunk_rows": int(module.history_chunk_rows),
           "chunk_cap": int(module.history_chunk_cap), "chunk_slots": int(module.history_chunk_slots)}
    if args.net.chunk_slots != law["chunk_cap"]:
        raise ValueError(f"--net.chunk-slots {args.net.chunk_slots} must equal the environment's chunk cap {law}")
    return law


def checkpoint_identity(args):
    """The receipt fields a resumed run must match: model, recipe, the tables the model reads and the announce law.

    Registers the run's announce law first (``announce_law.register_from_args``; the same law again is a no-op), so
    every env made afterwards uses it and a resume under another law is refused with the rest of the identity."""
    from mirrorforce.agent.env import dormant_law
    from mirrorforce.agent.env.announce_law import register_from_args
    from mirrorforce.agent.env.build_record import core_identity
    from mirrorforce.agent.env.duel import native as module, native_build
    announce = register_from_args(args)
    dormant = dormant_law.register_from_args(args)
    room = room_format.from_args(args)
    tables = card_tables_from_args(args)
    native = os.environ.get("MF_DUEL_NATIVE")
    return {
        "model": ({"architecture": "policy_net", "args": model_config_identity(args.net), "semantic_shape": jsonable(args.semantic_shape)}
                  if args.architecture == "policy_net" else
                  {"architecture": "decision-v1" if args.m1.decision_layers else "structured-v1",
                   "args": jsonable(args.m1), "semantic_shape": jsonable(args.semantic_shape),
                   "bfloat16": args.bfloat16}),
        "recipe": {"batch_size": args.batch_size, "minibatch_size": args.minibatch_size,
                   "num_minibatches": args.num_minibatches, "update_epochs": args.update_epochs,
                   "num_steps": args.num_steps, "learning_rate": args.learning_rate, "gamma": args.gamma,
                   "gae_lambda": args.gae_lambda, "value": args.value, "ent_coef": args.ent_coef,
                   "vf_coef": args.vf_coef, "decision_aux_coef": args.decision_aux_coef,
                   "lp_shaping_coef": args.lp_shaping_coef, "lp_shaping_scale": args.lp_shaping_scale,
                   "max_options": args.max_options, "max_steps": args.max_steps,
                   "step_limit_law": module.step_limit_law, "guard_laws": list(module.guard_laws),
                   **({"public_opponent_recipe": True} if args.public_opponent_recipe else {}),
                   **({"allow_unreviewed_public_effects": True} if args.allow_unreviewed_public_effects else {}),
                   **({"belief_rows": "kept"} if args.iteration_kept_rows else {}),
                   **({"update": "ataraxos", "ataraxos": jsonable(args.ataraxos)} if args.recipe == "ataraxos"
                      else {})},
        "tables": {"semantic_file_sha256": state_io.file_digest(args.semantic_file) if args.semantic_file else None,
                   "code_list_sha256": state_io.file_digest(args.code_list_file),
                   "cards_db_sha256": state_io.file_digest(args.cards_db) if args.cards_db else None},
        "native_sha256": state_io.file_digest(native) if native else None,
        "native_core": core_identity(native_build),
        **({"window_law": window_law(module, args)} if args.architecture == "policy_net" else {}),
        # the observation laws a network client reproduces (card view, pending source, ...): modules before the laws
        # read a fresh engine query and move the pending source on every command
        "card_view_law": getattr(module, "card_view_law", "fresh_query"),
        "observation_laws": list(getattr(module, "observation_laws", ())),
        "export_both_seats": args.export_both_seats,
        **({"critic": {"model": model_config_identity(args.critic_model), "mix_layers": args.critic_mix_layers,
                       **({"td_lambda": args.critic_td_lambda} if args.critic_td_lambda is not None else {}),
                       **({"zero_init": True} if args.critic_zero_init else {})}}
           if args.critic else {}),
        **({"public_effects_table": module.public_effects_table_sha256}
           if hasattr(module, "public_effects_table_sha256") else {}),
        "announce": announce,
        **({"dormant": dormant} if dormant else {}),
        **({"room_format": room} if room else {}),
        **({"card_tables": card_tables.identity(tables[1], tables[2])} if tables else {}),
    }


def model_config_identity(config):
    out = jsonable(config)
    if not out.get("value_menu", False):
        out.pop("value_menu", None)  # keep legacy receipt identity byte-for-byte
    return out


def card_tables_from_args(args):
    """``(model arrays, metadata, sha256)`` of the run's ``--card-tables``, checked against its code list and semantics
    file; None when the run has none."""
    if not args.card_tables:
        return None
    tables, metadata, sha256 = card_tables.load(args.card_tables, args.code_list_file)
    if not args.semantic_file or metadata["semantics"]["sha256"] != state_io.file_digest(args.semantic_file):
        raise ValueError(f"{Path(args.card_tables).name} was built for another semantics file than --semantic-file")
    return tables, metadata, sha256


def main():
    """Entry point: whole-process exit on any exception, SIGTERM stops after a checkpoint."""
    threading.excepthook = lambda hook: fatal_exit(
        f"thread {hook.thread.name if hook.thread else '?'}", hook.exc_type, hook.exc_value, hook.exc_traceback)
    signal.signal(signal.SIGTERM, lambda signum, frame: STOP_REQUESTED.set())
    try:
        train()
    except BaseException:
        fatal_exit("main", *sys.exc_info())
    sys.stdout.flush()
    sys.stderr.flush()
    # Actor threads block on their parameter queues once the learner stops.
    os._exit(0)


def train():
    args = tyro.cli(Args)
    # no core files: a crashing JAX process maps 100+ GB and a dump can fill a shared host's root filesystem
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if not args.concurrency and os.environ.get("JAX_PLATFORMS") == "cpu":
        # deterministic mode on CPU: one CPU before the backend starts, so XLA's CPU thread pool has one thread and
        # reductions keep their order however loaded the machine is (multi-thread Eigen is already off)
        cpu = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpu})
        print(f"DETERMINISTIC-CPU pinned to cpu {cpu}", flush=True)
    if not np.isfinite(args.lp_shaping_coef) or args.lp_shaping_coef < 0:
        raise ValueError("lp_shaping_coef must be finite and nonnegative")
    if not np.isfinite(args.lp_shaping_scale) or args.lp_shaping_scale <= 0:
        raise ValueError("lp_shaping_scale must be positive")
    if not np.isfinite(args.decision_aux_coef) or args.decision_aux_coef < 0:
        raise ValueError("decision_aux_coef must be finite and nonnegative")
    if (args.decision_aux_coef or args.decision_warm_start) and not args.m1.decision_layers:
        raise ValueError("decision auxiliary training/migration requires --m1.decision-layers")
    if args.m1.decision_layers and args.segment_length is not None:
        raise ValueError("decision v1 does not support the existing experimental segmented updater")
    if args.decision_warm_start and not args.checkpoint:
        raise ValueError("decision warm start requires a checkpoint")
    args.local_batch_size = int(args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    assert (
        args.local_num_envs % len(args.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)"
    assert (
        int(args.local_num_envs / len(args.learner_device_ids)) * args.num_actor_threads % args.num_minibatches == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"
    if args.distributed:
        if os.environ.get("JAX_PLATFORMS") == "cpu":  # CPU processes reduce across processes through gloo
            jax.config.update("jax_cpu_collectives_implementation", "gloo")
        local_ids = sorted(set(args.actor_device_ids) | set(args.learner_device_ids))
        jax.distributed.initialize(coordinator_address=args.coordinator_address, num_processes=args.num_processes,
                                   process_id=args.process_id, local_device_ids=local_ids)
        print("DISTRIBUTED " + json.dumps({"process": args.process_id, "processes": args.num_processes,
                                           "coordinator": args.coordinator_address, "local_device_ids": local_ids,
                                           "nccl": {k: v for k, v in sorted(os.environ.items())
                                                    if k.startswith(("NCCL_", "UCX_"))}}), flush=True)

    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        raise ValueError("JAX_COMPILATION_CACHE_DIR must name the run's persistent compilation cache")
    if len(args.actor_device_ids) > 1:
        # JAX 0.5.3's persistent cache hands a single-device executable compiled for
        # one actor device to the other actor device's thread ("Buffer passed to
        # Execute() ... is on device cuda:1, but replica is assigned to device
        # cuda:0"; trainer-0, 2026-10-01). With several actor devices, compile in memory.
        # JAX also reads JAX_COMPILATION_CACHE_DIR from the environment at import,
        # so leaving the directory unset is not enough: switch the cache off.
        jax.config.update("jax_enable_compilation_cache", False)
        print("PERSISTENT_COMPILATION_CACHE disabled: several actor devices", flush=True)
    else:
        jax.config.update("jax_compilation_cache_dir", cache_dir)

    args.world_size = jax.process_count()
    args.local_rank = jax.process_index()
    args.num_envs = args.local_num_envs * args.world_size * args.num_actor_threads * len(args.actor_device_ids)
    args.batch_size = args.local_batch_size * args.world_size
    args.minibatch_size = args.local_minibatch_size * args.world_size
    args.num_updates = args.total_timesteps // (args.local_batch_size * args.world_size)
    args.local_env_threads = args.local_env_threads or args.local_num_envs
    if args.segment_length is not None:
        assert args.num_steps % args.segment_length == 0, "num_steps must be divisible by segment_length"
    args.collect_steps = args.collect_steps or args.num_steps
    assert args.collect_steps >= args.num_steps, "collect_steps must be greater than or equal to num_steps"
    if args.iteration_decisions:
        if args.recipe != "ataraxos" or args.collect_steps != args.num_steps:
            raise ValueError("A0 iterations run arm A's recipe on whole segments (--recipe ataraxos, no collect_steps)")
        args.iteration_updates = max(1, round(args.iteration_decisions / args.batch_size))
        segments = args.iteration_updates * args.num_envs
        group = len(args.learner_device_ids) * args.iteration_micro_segments * args.world_size
        if segments % args.iteration_steps or (segments // args.iteration_steps) % group:
            raise ValueError(f"an iteration's {segments} segments must split into {args.iteration_steps} steps of a "
                             f"multiple of {group} (learner devices x micro-batch segments)")
        print("A0-ITERATION " + json.dumps({"updates": args.iteration_updates, "segments": segments,
                                            "decisions": segments * args.num_steps, "steps": args.iteration_steps,
                                            "segments_per_step": segments // args.iteration_steps}), flush=True)

    structured_enabled = args.m1.structured or args.m2.structured
    if structured_enabled:
        if not args.semantic_file:
            raise ValueError(
                "structured models require --semantic-file"
            )
        semantic_tables, semantic_shape, semantic_metadata = \
            load_structured_semantics(
                args.semantic_file, args.code_list_file
            )
        args.semantic_shape = semantic_shape
        args.semantic_metadata = semantic_metadata
        args.freeze_id = True
        embeddings = None
    elif args.embedding_file:
        embeddings = load_embeddings(args.embedding_file, args.code_list_file)
        embedding_shape = embeddings.shape
        args.num_embeddings = embedding_shape
        args.freeze_id = True if args.freeze_id is None else args.freeze_id
        semantic_tables = None
    else:
        embeddings = None
        embedding_shape = None
        semantic_tables = None

    local_devices = jax.local_devices()
    global_devices = jax.devices()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    actor_devices = [local_devices[d_id] for d_id in args.actor_device_ids]
    global_learner_decices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.world_size)
        for d_id in args.learner_device_ids
    ]
    global_main_devices = [
        global_devices[process_index * len(local_devices)]
        for process_index in range(args.world_size)
    ]
    print("global_learner_decices", global_learner_decices)
    args.global_learner_decices = [str(item) for item in global_learner_decices]
    args.actor_devices = [str(item) for item in actor_devices]
    args.learner_devices = [str(item) for item in learner_devices]
    pprint(args)

    if args.run_name is None:
        timestamp = int(time.time())
        run_name = f"{args.exp_name}__{args.seed}__{timestamp}"
    else:
        run_name = args.run_name
        timestamp = int(run_name.split("__")[-1])

    dummy_writer = SimpleNamespace()
    dummy_writer.add_scalar = lambda x, y, z: None
    dummy_writer.close = lambda: None

    if args.local_rank == 0 and not args.debug and args.tb_dir is not None:
        from tensorboardX import SummaryWriter
        tb_log_dir = f"{args.tb_dir}/{run_name}"
        writer = SummaryWriter(tb_log_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = dummy_writer

    identity = checkpoint_identity(args)
    receipt_base = {**identity,
                    "config": jsonable(vars(args)),
                    "pool": {"deck": args.deck, "deck_tree_sha256": state_io.tree_digest(args.deck, "*.ydk"),
                             "deck_schedule": args.deck_schedule},
                    "initial_checkpoint": args.checkpoint,
                    "job_sha256": os.environ.get("MF_RUNTIME_JOB_SHA256"),
                    "runtime": {"xla_flags": os.environ.get("XLA_FLAGS", ""), "jax": jax.__version__},
                    "optimizer": {"steps_per_update": args.num_minibatches * args.update_epochs,
                                  "global_minibatch": args.batch_size // args.num_minibatches,
                                  "grad_accum": args.grad_accum,
                                  "micro_batch_per_device": args.batch_size // args.num_minibatches
                                  // len(args.learner_device_ids) // args.grad_accum}}
    print("XLA-FLAGS-READBACK " + json.dumps(os.environ.get("XLA_FLAGS", "")), flush=True)
    parent_sha = None
    update_offset = 0

    # seeding
    random.seed(args.seed)
    seed = random.randint(0, int(1e8))

    seed_offset = args.local_rank
    seed += seed_offset
    init_key = jax.random.PRNGKey(seed - seed_offset)

    random.seed(seed)
    args.real_seed = random.randint(0, int(1e8))

    key = jax.random.PRNGKey(args.real_seed)
    key, *learner_keys = jax.random.split(key, len(learner_devices) + 1)
    learner_keys = jax.device_put_sharded(learner_keys, devices=learner_devices)
    actor_keys = jax.random.split(key, len(actor_devices) * args.num_actor_threads)

    deck, deck_names = init_duel(args.env_id, "english", args.deck, args.code_list_file, return_deck_names=True,
                                 db_path=args.cards_db)
    args.deck_names = sorted(deck_names)
    if args.cluster_stats_dir:
        if args.world_size != 1:
            raise ValueError("Cluster statistics currently require a single training process")
        from mirrorforce.agent.train.cluster_stats import start_run
        args.cluster_stats_run = start_run(
            args.cluster_stats_dir, args.deck_names,
            Path(args.deck).parent / "manifests/train.jsonl",
            args.checkpoint, args.tb_offset, args.num_actor_threads * len(args.actor_device_ids))
        print(f"CLUSTER_STATS_RUN {args.cluster_stats_run}", flush=True)
    args.deck1 = args.deck1 or deck
    args.deck2 = args.deck2 or deck

    # env setup
    envs = make_env(args, 0, 2, 1)
    obs_space = envs.observation_space
    # the policy reads public observations only: a privileged key (priv:/label:) in its input is refused, once, here
    from mirrorforce.agent.env.privileged import assert_public
    assert_public(obs_space.keys(), "the policy observation")
    action_shape = envs.action_space.shape
    print(f"obs_space={obs_space}, action_shape={action_shape}")
    sample_obs, _ = envs.reset()
    sample_obs = jax.tree.map(lambda x: jnp.asarray(x[:1]), sample_obs)
    envs.close()
    del envs

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    agent = create_agent(args)
    rstate = agent.init_rnn_state(1)
    variables = agent.init(init_key, sample_obs, rstate)
    variables = flax.core.unfreeze(variables)
    if embeddings is not None:
        unknown_embed = embeddings.mean(axis=0)
        embeddings = np.concatenate([unknown_embed[None, :], embeddings], axis=0)
        variables['params']['Encoder_0']['Embed_0']['embedding'] = jax.device_put(embeddings)
        # variables = flax.core.freeze(variables)
    if args.checkpoint:
        with open(args.checkpoint, "rb") as f:
            if args.m1.decision_layers:
                variables, added = restore_decision_checkpoint(
                    variables, f.read(), allow_legacy=args.decision_warm_start
                )
                print(f"decision checkpoint initialized branches: {added}")
            else:
                variables = flax.serialization.from_bytes(variables, f.read())
        print(f"loaded checkpoint from {args.checkpoint}")
    if semantic_tables is not None:
        inject_semantic_constants(variables, semantic_tables)
    run_card_tables = card_tables_from_args(args)
    if run_card_tables is not None:
        card_tables.inject(variables, run_card_tables[0])

    steps_per_update = args.num_minibatches * args.update_epochs

    def ataraxos_lr(count):  # the power schedule over samples consumed (one update = batch_size samples)
        if args.iteration_decisions:  # A0: over the iteration count, as theirs (rl.py)
            c = args.ataraxos
            return ataraxos.power_schedule(c.lr_coef, count // args.iteration_steps, c.lr_decay, c.lr_ceil,
                                           c.lr_floor)
        return ataraxos.learning_rate((count // steps_per_update) * args.batch_size, args.ataraxos)

    def adam_chain():
        if args.recipe == "ataraxos":
            return optax.chain(
                optax.clip_by_global_norm(args.ataraxos.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=ataraxos_lr, eps=args.ataraxos.adam_eps),
            )
        return optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        )

    if args.architecture == "policy_net":
        if args.norm_adv and args.grad_accum > 1:
            raise ValueError("per-minibatch advantage normalization is not exact under gradient accumulation")
        # The belief head has its own clip and Adam state, so its gradients never shrink the policy update through
        # a shared global norm; no skip of non-finite updates (a NaN or inf is fatal on the host).
        tx = optax.multi_transform(
            {"policy": adam_chain(), "belief": adam_chain()},
            lambda params: {name: ("belief" if name == "belief" else "policy") for name in params})
    else:
        tx = optax.MultiSteps(adam_chain(), every_k_schedule=1)
        tx = optax.apply_if_finite(tx, max_consecutive_errors=10)

    if 'batch_stats' not in variables:
        variables['batch_stats'] = {}
    agent_state = TrainState.create(
        apply_fn=None,
        params=variables['params'],
        tx=tx,
        batch_stats=variables['batch_stats'],
        constants=variables.get('constants', {}),
    )
    resume_actors = None
    if args.iteration_decisions:
        args.a0_counter_basis = a0_counters.initial_basis(args.tb_offset)
    ema = None
    if args.critic_advantages and not args.critic:
        raise ValueError("--critic-advantages takes the advantages from the central critic (--critic)")
    args.critic_started, args.advantages_since = (0 if args.critic else None), 0  # resumes carry them on
    args.env_changes, args.advantages_switch, args.declared_changes = [], None, []
    args.value_menu_migrations = []
    if args.resume_add_value_menu and (not args.resume or args.resume_add_critic or args.resume_reset_critic
                                      or any(k in ("model", "critic") for k in args.resume_declare)):
        raise ValueError("--resume-add-value-menu needs a resume without critic reset/add or model declarations")
    if args.resume:
        expected = dict(identity)  # a copy: the declarations below remove keys, the run's identity keeps them
        if args.resume_topology_change:  # the update batch follows the device layout; the iteration does not
            if not args.iteration_decisions:
                raise ValueError("a topology-change resume is declared at A0 iteration boundaries")
            layout = {"batch_size", "minibatch_size", "num_minibatches"}
            expected = {k: v for k, v in identity.items() if k != "recipe"}
            expected.update({f"recipe.{k}": v for k, v in identity["recipe"].items() if k not in layout})
            expected.update({"a0.decisions_per_iteration": args.iteration_updates * args.num_envs * args.num_steps,
                             "a0.iteration_steps": args.iteration_steps})  # the iteration itself is unchanged
        if args.resume_add_critic:  # the checkpoint has neither the critic nor the export; both start here
            if not args.critic:
                raise ValueError("--resume-add-critic adds the central critic (--critic)")
            expected = {k: v for k, v in expected.items() if k != "critic"}
            expected["export_both_seats"] = False
        if args.resume_reset_critic:  # the checkpoint's critic (and its configuration) is replaced by a fresh one
            if not args.critic or args.resume_add_critic:
                raise ValueError("--resume-reset-critic replaces the critic of a run that has one (--critic)")
            expected = {k: v for k, v in expected.items() if k != "critic"}
        if args.resume_env_change:
            expected = {k: v for k, v in expected.items() if k not in ENV_IDENTITY}
        if any(k.startswith("recipe.") for k in args.resume_declare) and "recipe" in expected:
            expected.update({f"recipe.{k}": v for k, v in expected.pop("recipe").items()})
        for key in args.resume_declare:
            if key not in expected:
                raise ValueError(f"--resume-declare {key}: not an identity key this resume compares")
            expected.pop(key)
        if args.world_size > 1 and not args.resume_topology_change:
            raise ValueError("several processes do not checkpoint their environments: a resume restarts them "
                             "(declare --resume-topology-change)")
        restored, keys, counters, resumed_receipt, resume_actors, ema = restore_on_process0(
            args, agent_state, None if args.resume_topology_change else len(learner_devices), expected)
        agent_state = restored
        args.value_menu_migrations = resumed_receipt.get("value_menu_migrations", [])
        if args.resume_add_value_menu:
            from mirrorforce.agent.train.value_menu_migration import LAW
            migration = {"law": LAW, "parent": Path(args.resume).stem, "iteration": counters["learner_update"],
                         "critic": args.critic, "old_state_preserved": True,
                         "new_moments": "zero", "new_ema": "virtual_constant_history"}
            args.value_menu_migrations = args.value_menu_migrations + [migration]
            print("VALUE-MENU-ADDED " + json.dumps(migration), flush=True)
        previous_a0 = resumed_receipt.get("a0", {})
        restart = args.resume_topology_change  # environments and actor memory start afresh
        if args.resume_env_change:
            changed = {k: {"from": resumed_receipt.get(k), "to": identity.get(k)} for k in ENV_IDENTITY
                       if resumed_receipt.get(k) != identity.get(k)}
            if not changed:
                raise ValueError("--resume-env-change: the checkpoint was trained on this environment build")
            args.env_changes = resumed_receipt.get("env_changes", []) + [
                {"iteration": counters["learner_update"], "changed": sorted(changed)}]
            restart = True
            print("ENV-CHANGE " + json.dumps({"from": Path(args.resume).stem, "iteration": counters["learner_update"],
                                              "changed": changed}), flush=True)
        else:
            args.env_changes = resumed_receipt.get("env_changes", [])
        args.declared_changes = resumed_receipt.get("declared_changes", [])
        if args.resume_declare:
            def lookup(tree, key):
                for part in key.split("."):
                    tree = tree.get(part) if isinstance(tree, dict) else None
                return tree
            changed = {k: {"from": lookup(resumed_receipt, k), "to": lookup(identity, k)} for k in args.resume_declare}
            args.declared_changes = args.declared_changes + [{"iteration": counters["learner_update"],
                                                              "changed": changed}]
            print("DECLARED-CHANGE " + json.dumps({"from": Path(args.resume).stem,
                                                   "iteration": counters["learner_update"], "changed": changed}),
                  flush=True)
        if args.resume_add_critic:  # (a checkpoint with a critic has the export: refused by the identity check)
            args.critic_started = counters["learner_update"]
            restart = True  # the actor snapshots lack the other seat's view
            print("CRITIC-ADDED " + json.dumps({"from": Path(args.resume).stem, "iteration": args.critic_started}),
                  flush=True)
        elif args.resume_reset_critic:
            args.critic_started = counters["learner_update"]
            print("CRITIC-RESET " + json.dumps({"from": Path(args.resume).stem, "iteration": args.critic_started,
                                                "previous": resumed_receipt.get("critic"),
                                                "now": identity.get("critic")}), flush=True)
        elif args.critic:
            args.critic_started = previous_a0["critic_started"]
        advantages = "critic" if args.critic_advantages else "public"
        args.advantages_switch = previous_a0.get("advantages_switch")
        if previous_a0.get("advantages", "public") != advantages:
            args.advantages_since = counters["learner_update"]
            if advantages == "critic":
                if not args.advantages_switch_evidence:
                    raise ValueError("switching the advantages to the critic needs --advantages-switch-evidence")
                args.advantages_switch = {"iteration": args.advantages_since,
                                          "evidence": json.loads(Path(args.advantages_switch_evidence).read_text())}
            print("ADVANTAGES-SWITCH " + json.dumps({"from": previous_a0.get("advantages", "public"), "to": advantages,
                                                     "iteration": args.advantages_since}), flush=True)
        else:
            args.advantages_since = previous_a0.get("advantages_since", 0)
        if restart:
            if not args.iteration_decisions:
                raise ValueError("a resume that restarts the environments is declared at A0 iteration boundaries")
            resume_actors = None
            args.actor_start_update = counters["learner_update"] * args.iteration_updates + 1
        if args.resume_topology_change:
            keys = np.asarray(jax.random.split(jnp.asarray(keys[0]), len(learner_devices)))
            print("TOPOLOGY-CHANGE " + json.dumps({"from": Path(args.resume).stem, "iteration": counters["learner_update"],
                                                   "learner_devices": len(learner_devices) * args.world_size,
                                                   "envs": args.num_envs}), flush=True)
        learner_keys = jax.device_put_sharded(list(keys), devices=learner_devices)
        if args.iteration_decisions:
            args.a0_counter_basis = a0_counters.resume_basis(
                Path(args.resume).stem, counters, resume_actors,
                actor_threads=len(args.actor_device_ids) * args.num_actor_threads,
                envs_per_actor=args.local_num_envs,
                expected_update=counters["learner_update"] * args.iteration_updates + 1)
            args.tb_offset = args.a0_counter_basis["offset"]
        else:
            args.tb_offset = 0 if resume_actors is not None else counters["global_step"]
        update_offset = counters["learner_update"]
        parent_sha = Path(args.resume).stem
        print("RESUMED " + json.dumps({"checkpoint": parent_sha, **counters}), flush=True)
    if args.recipe == "ataraxos" and ema is None:  # the EMA starts as a copy of the parameters
        ema = jax.tree.map(jnp.array, agent_state.params)
    agent_state = flax.jax_utils.replicate(agent_state, devices=learner_devices)
    # print(agent.tabulate(agent_key, sample_obs))

    if args.eval_checkpoint:
        eval_agent = create_agent(args, eval=True)
        eval_rstate = eval_agent.init_rnn_state(1)
        eval_variables = eval_agent.init(init_key, sample_obs, eval_rstate)
        with open(args.eval_checkpoint, "rb") as f:
            eval_variables = flax.serialization.from_bytes(eval_variables, f.read())
        if semantic_tables is not None:
            eval_variables = flax.core.unfreeze(eval_variables)
            inject_semantic_constants(eval_variables, semantic_tables)
            if run_card_tables is not None:
                card_tables.inject(eval_variables, run_card_tables[0])
            eval_variables = flax.core.freeze(eval_variables)
        print(f"loaded eval checkpoint from {args.eval_checkpoint}")
    else:
        eval_variables = None

    def compute_advantage(
        new_logits, new_values, next_dones, switch_or_mains,
        actions, logits, rewards, next_v):
        num_envs = jax.tree.leaves(next_v)[0].shape[0]
        num_steps = next_dones.shape[0] // num_envs

        def reshape_time_series(x):
            return jnp.reshape(x, (num_steps, num_envs) + x.shape[1:])

        ratios = distrax.importance_sampling_ratios(distrax.Categorical(
            new_logits), distrax.Categorical(logits), actions)
        ratios = reshape_time_series(ratios)

        new_values_, rewards, next_dones, switch_or_mains = jax.tree.map(
            reshape_time_series, (new_values, rewards, next_dones, switch_or_mains),
        )

        target_values, advantages = advantage_fn(
            args, next_v, new_values_, rewards, next_dones, switch_or_mains, ratios)

        target_values, advantages = jax.tree.map(
            lambda x: jnp.reshape(x, (-1,)), (target_values, advantages))
        return target_values, advantages

    def compute_loss(
        new_logits, new_values, actions, logits, target_values, advantages,
        mask, num_steps=None, denominator=None):
        ratios = distrax.importance_sampling_ratios(distrax.Categorical(
            new_logits), distrax.Categorical(logits), actions)
        logratio = jnp.log(ratios)
        approx_kl = (ratios - 1) - logratio

        if args.norm_adv:
            advantages = masked_normalize(advantages, mask, eps=1e-8)

        # Policy loss
        if args.spo_kld_max is not None:
            pg_loss = simple_policy_loss(
                ratios, logits, new_logits, advantages, args.spo_kld_max)
        elif args.logits_threshold is not None:
            pg_loss = ach_loss(
                actions, logits, new_logits, advantages, args.logits_threshold, args.clip_coef, args.dual_clip_coef)
        elif args.ppo_clip:
            pg_loss = clipped_surrogate_pg_loss(
                ratios, advantages, args.clip_coef, args.dual_clip_coef)
        else:
            pg_advs = jnp.clip(ratios, args.rho_clip_min, args.rho_clip_max) * advantages
            pg_loss = policy_gradient_loss(new_logits, actions, pg_advs)

        v_loss = mse_loss(new_values, target_values)
        if args.vloss_clip is not None:
            v_loss = jnp.minimum(v_loss, args.vloss_clip)

        ent_loss = entropy_loss(new_logits)

        if args.burn_in_steps:
            mask = jax.tree.map(
                lambda x: x.reshape(num_steps, -1), mask)
            burn_in_mask = jnp.arange(num_steps) < args.burn_in_steps
            mask = jnp.where(burn_in_mask[:, None], 0.0, mask)
            mask = jnp.reshape(mask, (-1,))

        n_valids = jnp.sum(mask) if denominator is None else denominator
        pg_loss, v_loss, ent_loss, approx_kl = jax.tree.map(
            lambda x: jnp.sum(x * mask) / n_valids, (pg_loss, v_loss, ent_loss, approx_kl))

        loss = pg_loss - args.ent_coef * ent_loss + v_loss * args.vf_coef
        return loss, pg_loss, v_loss, ent_loss, approx_kl

    def apply_fn(
        variables, obs, init_rstate, dones, next_dones, switch_or_mains, train=True):
        if args.switch:
            dones = dones | next_dones
        mutable = ["batch_stats"] if train else False
        rets = agent.apply(
            variables, obs, init_rstate, dones, switch_or_mains,
            train=train, mutable=mutable,
            return_aux=bool(train and (args.decision_aux_coef or args.belief_coef)))
        if train:
            outputs, state_updates = rets
            (rstate1, rstate2), new_logits, new_values, _ = outputs[:4]
            if args.decision_aux_coef:
                state_updates = dict(state_updates)
                state_updates["decision_prediction"] = outputs[4]
            if args.belief_coef:
                state_updates = dict(state_updates)
                state_updates["belief"] = outputs[4]
        else:
            (rstate1, rstate2), new_logits, new_values, _ = rets
            state_updates = {}
        new_values = jax.tree.map(lambda x: x.squeeze(-1), new_values)
        return ((rstate1, rstate2), new_logits, new_values), state_updates

    def compute_next_value(variables, next_rstate, next_obs, next_main):
        rstate1, rstate2 = next_rstate
        if args.architecture == "policy_net":  # the agent selects the acting seat's memory itself
            next_value = agent.apply(variables, next_obs, (rstate1, rstate2), None, next_main)[2]
        else:
            rstate = select_rows(next_main, rstate1, rstate2)
            next_value = agent.apply(variables, next_obs, rstate)[2]
        next_value = jax.tree.map(lambda x: x.squeeze(-1), next_value)
        next_value = jax.lax.stop_gradient(next_value)
        sign = -1 if args.switch else 1
        next_value = jnp.where(next_main, sign * next_value, -sign * next_value)
        return next_value

    def get_advantage(
        variables, init_rstate, obs, dones, next_dones,
        switch_or_mains, actions, logits, rewards, next_obs, next_main):
        num_steps = dones.shape[0]

        obs, dones, next_dones, switch_or_mains, actions, logits, rewards = \
            jax.tree.map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                (obs, dones, next_dones, switch_or_mains, actions, logits, rewards))

        (next_rstate, new_logits, new_values), state_updates = apply_fn(
            variables, obs, init_rstate, dones, next_dones, switch_or_mains, train=False)

        next_value = compute_next_value(
            variables, next_rstate, next_obs, next_main)

        target_values, advantages = compute_advantage(
            new_logits, new_values, next_dones, switch_or_mains,
            actions, logits, rewards, next_value)

        target_values, advantages = jax.tree.map(
            lambda x: jnp.reshape(x, (num_steps, -1) + x.shape[2:]),
            (target_values, advantages))
        return target_values, advantages

    def get_loss(
        params, batch_stats, constants, init_rstate, obs, dones, next_dones,
        switch_or_mains, actions, logits, target_values, advantages, mask):
        variables = {
            'params': params,
            'batch_stats': batch_stats,
            'constants': constants,
        }
        ((rstate1, rstate2), new_logits, new_values), state_updates = apply_fn(
            variables, obs, init_rstate, dones, next_dones, switch_or_mains)

        loss, pg_loss, v_loss, ent_loss, approx_kl = compute_loss(
            new_logits, new_values, actions, logits, target_values, advantages,
            mask, num_steps=None)

        loss = jnp.where(jnp.isnan(loss) | jnp.isinf(loss), 0.0, loss)
        approx_kl, rstate1, rstate2 = jax.tree.map(
            jax.lax.stop_gradient, (approx_kl, rstate1, rstate2))
        return loss, (state_updates, pg_loss, v_loss, ent_loss, approx_kl, rstate1, rstate2)

    def get_advantage_loss(
        params, batch_stats, constants, init_rstate, obs, dones, next_dones,
        switch_or_mains, actions, logits, rewards, mask, next_data,
        outcome_targets=None, outcome_valid=None, belief_labels=None, denominators=None):
        """``denominators`` (accumulation): the whole minibatch's (valid decisions, belief weight); without it each
        loss term is normalized by this batch's own counts."""
        num_envs = jax.tree.leaves(next_data)[0].shape[0]
        variables = {
            'params': params,
            'batch_stats': batch_stats,
            'constants': constants,
        }
        (next_rstate, new_logits, new_values), state_updates = apply_fn(
            variables, obs, init_rstate, dones, next_dones, switch_or_mains)

        if args.collect_steps == args.num_steps:
            next_obs, next_main = next_data
            variables = {
                'params': params,
                'batch_stats': state_updates['batch_stats'],
                'constants': constants,
            }
            next_v = compute_next_value(
                variables, next_rstate, next_obs, next_main)
        else:
            next_v = next_data

        target_values, advantages = compute_advantage(
            new_logits, new_values, next_dones, switch_or_mains,
            actions, logits, rewards, next_v)

        loss, pg_loss, v_loss, ent_loss, approx_kl = compute_loss(
            new_logits, new_values, actions, logits, target_values, advantages,
            mask, num_steps=dones.shape[0] // num_envs,
            denominator=None if denominators is None else denominators[0])

        if args.decision_aux_coef:
            auxiliary_valid = outcome_valid & mask.astype(jnp.bool_)
            if args.burn_in_steps:
                step = jnp.arange(dones.shape[0]) // num_envs
                auxiliary_valid &= step >= args.burn_in_steps
            auxiliary = outcome_loss(
                state_updates.pop("decision_prediction"), actions,
                outcome_targets, auxiliary_valid,
            )
            loss = loss + args.decision_aux_coef * auxiliary
        if args.belief_coef:
            belief_logits, belief_valid = state_updates.pop("belief")
            targets, _, _ = belief_targets(obs["candidates_"], belief_labels)
            nll = -jnp.take_along_axis(jax.nn.log_softmax(belief_logits.astype(jnp.float32), axis=-1),
                                       targets[..., None], axis=-1)[..., 0]  # [N, A, L]
            weight = (belief_valid & mask.astype(jnp.bool_)[:, None])[..., None].astype(jnp.float32)
            belief_den = weight.sum() * nll.shape[-1] if denominators is None else denominators[1]
            loss = loss + args.belief_coef * (nll * weight).sum() / jnp.maximum(belief_den, 1.0)
        if args.architecture != "policy_net" and not args.m1.decision_layers:  # the fork's own behaviour
            loss = jnp.where(jnp.isnan(loss) | jnp.isinf(loss), 0.0, loss)
        approx_kl = jax.lax.stop_gradient(approx_kl)
        return loss, (state_updates, pg_loss, v_loss, ent_loss, approx_kl)

    def single_device_update(
        agent_state: TrainState,
        sharded_storages: List,
        sharded_init_rstate: List,
        sharded_next_data: List,
        key: jax.random.PRNGKey,
    ):
        storage = jax.tree.map(lambda *x: jnp.hstack(x), *sharded_storages)
        next_data, init_rstate = [
            jax.tree.map(lambda *x: jnp.concatenate(x), *x)
            for x in [sharded_next_data, sharded_init_rstate]
        ]

        # reorder storage of individual players
        # main first, opponent second
        num_steps, num_envs = storage.rewards.shape
        if args.switch:
            T = jnp.arange(num_steps, dtype=jnp.int32)
            B = jnp.arange(num_envs, dtype=jnp.int32)
            mains = storage.mains.astype(jnp.int32)
            indices = jnp.argsort(T[:, None] - mains * num_steps, axis=0)
            switch_steps = jnp.sum(mains, axis=0)
            switch = T[:, None] == (switch_steps[None, :] - 1)
            storage = jax.tree.map(lambda x: x[indices, B[None, :]], storage)

        if args.segment_length is None:
            loss_grad_fn = jax.value_and_grad(get_advantage_loss, has_aux=True)
        else:
            # TODO: fix it
            loss_grad_fn = jax.value_and_grad(get_loss, has_aux=True)

        def accumulated_grads(agent_state, minibatch, num_steps):
            """The minibatch's gradient as the sum over ``grad_accum`` micro-batches of its environments, each loss
            term normalized by the whole minibatch's counts (equal to the one-batch gradient up to summation order)."""
            k = args.grad_accum
            init_rstate, obs, dones, next_dones, mains, actions, logits, rewards, mask, next_data, *extra = minibatch
            envs = jax.tree.leaves(init_rstate)[0].shape[0]
            if envs % k:
                raise ValueError(f"{envs} environments per minibatch do not split into {k} micro-batches")

            def steps_first(x):  # [T * E, ...] (time-major) -> [k, T * E / k, ...]
                x = x.reshape((num_steps, k, envs // k) + x.shape[1:])
                return jnp.moveaxis(x, 1, 0).reshape((k, num_steps * (envs // k)) + x.shape[3:])

            envs_first = lambda x: x.reshape((k, envs // k) + x.shape[1:])
            micro = (jax.tree.map(envs_first, init_rstate), jax.tree.map(steps_first, obs),
                     *[jax.tree.map(steps_first, x) for x in (dones, next_dones, mains, actions, logits, rewards, mask)],
                     jax.tree.map(envs_first, next_data),
                     *[None if x is None else jax.tree.map(steps_first, x) for x in extra])
            valid = jnp.sum(mask)
            belief_weight = jnp.asarray(0.0)
            if args.belief_coef:
                candidates_valid = obs["candidates_"][..., 2] > 0
                belief_weight = (candidates_valid & mask.astype(jnp.bool_)[:, None]).sum() * len(BELIEF_LOCATIONS)
            denominators = (valid, belief_weight)

            def body(total, part):
                (loss, aux), grads = loss_grad_fn(
                    agent_state.params, agent_state.batch_stats, agent_state.constants, *part,
                    denominators=denominators)
                return jax.tree.map(jnp.add, total, ((loss, aux[1:]), grads)), aux[0]

            zero = jax.tree.map(jnp.zeros_like, agent_state.params)
            first = jnp.zeros(())
            (loss, (pg_loss, v_loss, ent_loss, approx_kl)), grads = jax.lax.scan(
                lambda total, part: body(total, part), ((first, (first, first, first, first)), zero), micro)[0]
            state_updates = {"batch_stats": agent_state.batch_stats}
            return (loss, (state_updates, pg_loss, v_loss, ent_loss, approx_kl)), grads

        def update_epoch(carry, _):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def convert_data(x: jnp.ndarray, multi_step=True):
                return reshape_minibatch(
                    x, multi_step, args.num_minibatches, num_steps, args.segment_length, key=subkey)

            b_init_rstate, b_next_data = \
                jax.tree.map(partial(convert_data, multi_step=False),
                             (init_rstate, next_data))
            b_storage = jax.tree.map(convert_data, storage)
            if args.switch:
                switch_or_mains = convert_data(switch)
            else:
                switch_or_mains = b_storage.mains
            b_mask = ~b_storage.dones
            b_rewards = b_storage.rewards

            if args.segment_length is None:
                def update_minibatch(agent_state, minibatch):
                    if args.grad_accum > 1:
                        (loss, (state_updates, pg_loss, v_loss, ent_loss, approx_kl)), grads = \
                            accumulated_grads(agent_state, minibatch, num_steps)
                    else:
                        (loss, (state_updates, pg_loss, v_loss, ent_loss, approx_kl)), grads = \
                            loss_grad_fn(
                                agent_state.params,
                                agent_state.batch_stats,
                                agent_state.constants,
                                *minibatch,
                            )
                    grads = jax.lax.pmean(grads, axis_name="local_devices")
                    agent_state = agent_state.apply_gradients(grads=grads)
                    agent_state = agent_state.replace(batch_stats=state_updates['batch_stats'])
                    return agent_state, (loss, pg_loss, v_loss, ent_loss, approx_kl)
            else:
                def update_minibatch(carry, minibatch):
                    def update_minibatch_t(carry, minibatch_t):
                        agent_state, init_rstate = carry
                        minibatch_t = init_rstate, *minibatch_t
                        (loss, (state_updates, pg_loss, v_loss, ent_loss, approx_kl, next_rstate)), \
                            grads = loss_grad_fn(
                                agent_state.params,
                                agent_state.batch_stats,
                                agent_state.constants,
                                *minibatch_t,
                            )
                        grads = jax.lax.pmean(grads, axis_name="local_devices")
                        agent_state = agent_state.apply_gradients(grads=grads)
                        agent_state = agent_state.replace(batch_stats=state_updates['batch_stats'])
                        return (agent_state, next_rstate), (loss, pg_loss, v_loss, ent_loss, approx_kl)

                    init_rstate, *minibatch_t, mask = minibatch
                    target_values, advantages = get_advantage(
                        get_variables(carry), init_rstate, *minibatch_t)
                    minibatch_t = *minibatch_t[:-2], target_values, advantages, mask

                    (carry, _next_rstate), \
                        (loss, pg_loss, v_loss, ent_loss, approx_kl) = jax.lax.scan(
                        update_minibatch_t, (carry, init_rstate), minibatch_t)
                    return carry, (loss, pg_loss, v_loss, ent_loss, approx_kl)

            minibatches = (
                b_init_rstate,
                b_storage.obs,
                b_storage.dones,
                b_storage.next_dones,
                switch_or_mains,
                b_storage.actions,
                b_storage.logits,
                b_rewards,
                b_mask,
                b_next_data,
            )
            if args.decision_aux_coef:
                minibatches = (
                    *minibatches, b_storage.outcome_targets, b_storage.outcome_valid
                )
            if args.belief_coef:
                minibatches = (*minibatches, None, None, b_storage.belief_labels)
            agent_state, (loss, pg_loss, v_loss, ent_loss, approx_kl) = jax.lax.scan(
                update_minibatch, agent_state, minibatches,
            )
            return (agent_state, key), (loss, pg_loss, v_loss, ent_loss, approx_kl)

        (agent_state, key), (loss, pg_loss, v_loss, ent_loss, approx_kl) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        loss = jax.lax.pmean(loss, axis_name="local_devices").mean()
        pg_loss = jax.lax.pmean(pg_loss, axis_name="local_devices").mean()
        v_loss = jax.lax.pmean(v_loss, axis_name="local_devices").mean()
        ent_loss = jax.lax.pmean(ent_loss, axis_name="local_devices").mean()
        approx_kl = jax.lax.pmean(approx_kl, axis_name="local_devices").mean()
        return agent_state, loss, pg_loss, v_loss, ent_loss, approx_kl, key

    def ataraxos_loss(params, batch_stats, constants, init_rstate, obs, dones, mains, actions, logits, returns,
                      advantages, kept, belief_labels, withdrawn, temperature_now, denominators=None):
        """Arm A's loss on one minibatch (ataraxos.loss_terms on policy_net's segment forward), plus the belief loss."""
        variables = {"params": params, "batch_stats": batch_stats, "constants": constants}
        outputs, _ = agent.apply(variables, obs, init_rstate, dones, mains, train=True, mutable=["batch_stats"],
                                 return_aux=bool(args.belief_coef))
        new_logits, wdl = outputs[1], outputs[3]
        loss, terms = ataraxos.loss_terms(
            new_logits, wdl, logits, actions, advantages, returns, kept, temperature_now, args.ataraxos,
            None if denominators is None else denominators[0],
            groups=obs["action_single_refs_"][..., 0].astype(jnp.int32))  # each option's source card row + 1
        if args.belief_coef:
            belief_logits, belief_valid = outputs[4]
            targets, _, _ = belief_targets(obs["candidates_"], belief_labels)
            nll = -jnp.take_along_axis(jax.nn.log_softmax(belief_logits.astype(jnp.float32), axis=-1),
                                       targets[..., None], axis=-1)[..., 0]
            weight = (belief_valid & ~(dones | withdrawn)[:, None])[..., None].astype(jnp.float32)
            belief_den = weight.sum() * nll.shape[-1] if denominators is None else denominators[1]
            terms["belief"] = (nll * weight).sum() / jnp.maximum(belief_den, 1.0)
            loss = loss + args.belief_coef * terms["belief"]
        return loss, terms

    ataraxos_grad_fn = jax.value_and_grad(ataraxos_loss, has_aux=True)

    def a0_step(agent_state, init_rstate, packed, small, temperature_now):
        """One A0 optimizer step on this learner device's share of the minibatch (segments of num_steps decisions,
        time-major [T * S]): the |A| filter over this device's decisions (theirs is per rank), then micro-batches of
        iteration_micro_segments segments accumulated with the step's counts, pmean over devices, one Adam step."""
        cfg = args.ataraxos
        steps = args.num_steps
        # the policy's input: public keys only (the central critic's priv: keys travel in the same minibatch)
        obs = {k: v for k, v in packed.items() if k not in ("logits_", "labels_") and not k.startswith("priv:")}
        logits, labels = packed["logits_"], packed.get("labels_")
        dones, mains, actions, withdrawn = small["dones"], small["mains"], small["actions"], small["withdrawn"]
        returns, advantages = small["returns"], small["advantages"]
        valid = ~(dones | withdrawn)
        kept, threshold = ataraxos.advantage_mask(advantages, valid, cfg)
        segments = init_rstate[0][1].shape[0]
        if args.iteration_kept_rows:
            return a0_kept_step(agent_state, init_rstate, obs, logits, labels, dones, mains, actions, withdrawn,
                                returns, advantages, valid, kept, threshold, temperature_now)
        m = args.iteration_micro_segments
        k = segments // m

        def micro(x):  # [T * S, ...] -> [k, T * m, ...] (each micro-batch keeps whole segments, time-major)
            x = x.reshape((steps, k, m) + x.shape[1:])
            return jnp.moveaxis(x, 1, 0).reshape((k, steps * m) + x.shape[3:])
        belief_rows = (obs["candidates_"][..., 2] > 0) & ~(dones | withdrawn)[:, None]
        denominators = (kept.sum().astype(jnp.float32),
                        belief_rows.sum().astype(jnp.float32) * len(BELIEF_LOCATIONS))
        parts = (jax.tree.map(lambda y: y.reshape((k, m) + y.shape[1:]), init_rstate), jax.tree.map(micro, obs),
                 micro(dones), micro(mains), micro(actions), micro(logits), micro(returns), micro(advantages),
                 micro(kept), None if labels is None else micro(labels), micro(withdrawn))

        def body(total, part):
            (loss, terms), grads = ataraxos_grad_fn(
                agent_state.params, agent_state.batch_stats, agent_state.constants, *part, temperature_now,
                denominators=denominators)
            return jax.tree.map(jnp.add, total, ((loss, terms), grads)), None

        probe = jax.eval_shape(lambda: ataraxos_grad_fn(
            agent_state.params, agent_state.batch_stats, agent_state.constants,
            *jax.tree.map(lambda y: y[0], parts), temperature_now, denominators=denominators))
        zero = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), probe)
        (loss, terms), grads = jax.lax.scan(body, zero, parts)[0]
        grads = jax.lax.pmean(grads, axis_name="local_devices")
        terms = dict(terms, g_norm=optax.global_norm(grads), kept_fraction=kept.sum() / jnp.maximum(valid.sum(), 1),
                     threshold=threshold, loss=loss)
        terms = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name="local_devices"), terms)
        return agent_state.apply_gradients(grads=grads), terms

    def a0_kept_step(agent_state, init_rstate, obs, logits, labels, dones, mains, actions, withdrawn, returns,
                     advantages, valid, kept, threshold, temperature_now):
        """a0_step with the per-decision passes on the kept rows only (a0_kept); the same loss: every term over the
        kept rows with this device's counts, the belief loss over the kept rows' candidates (belief_rows kept)."""
        from mirrorforce.agent.model.policy_net import belief_targets
        from mirrorforce.agent.train import a0_kept
        cfg = args.ataraxos
        rows = kept.shape[0]
        cap = a0_kept.capacity(rows, 1.0 - cfg.adv_filt_rate, args.iteration_kept_chunk)
        belief_rows = (obs["candidates_"][..., 2] > 0) & kept[:, None]
        den_policy = kept.sum().astype(jnp.float32)
        den_belief = belief_rows.sum().astype(jnp.float32) * len(BELIEF_LOCATIONS)
        small = {"logits": logits, "actions": actions, "returns": returns, "advantages": advantages,
                 "groups": obs["action_single_refs_"][..., 0].astype(jnp.int32), "candidates": obs["candidates_"]}
        if labels is not None:
            small["labels"] = labels

        def loss_rows(outputs, r, weight):
            loss, terms = ataraxos.loss_terms(outputs[0], outputs[2], r["logits"], r["actions"], r["advantages"],
                                              r["returns"], weight, temperature_now, cfg, den_policy,
                                              groups=r["groups"])
            if args.belief_coef:
                belief_logits, belief_valid = outputs[3]
                targets, _, _ = belief_targets(r["candidates"], r["labels"])
                nll = -jnp.take_along_axis(jax.nn.log_softmax(belief_logits.astype(jnp.float32), axis=-1),
                                           targets[..., None], axis=-1)[..., 0]
                rows_b = (belief_valid & (weight > 0)[:, None])[..., None].astype(jnp.float32)
                terms["belief"] = (nll * rows_b).sum() / jnp.maximum(den_belief, 1.0)
                loss = loss + args.belief_coef * terms["belief"]
            return loss, terms

        consts = {"constants": agent_state.constants}
        loss, terms, grads, bad, overflow = a0_kept.kept_value_and_grad(
            agent.model, agent_state.params, consts, obs, small, mains, dones, init_rstate, kept, loss_rows,
            steps=args.num_steps, cap=cap, chunk=args.iteration_kept_chunk, belief=bool(args.belief_coef))
        grads = jax.lax.pmean(grads, axis_name="local_devices")
        terms = dict(terms, g_norm=optax.global_norm(grads), kept_fraction=kept.sum() / jnp.maximum(valid.sum(), 1),
                     threshold=threshold, loss=loss, delivery_bad=bad.astype(jnp.float32),
                     kept_overflow=overflow.astype(jnp.float32))
        terms = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name="local_devices"), terms)
        return agent_state.apply_gradients(grads=grads), terms

    def ataraxos_update(agent_state, sharded_storages, sharded_init_rstate, sharded_next_data, key, temperature_now):
        """Arm A's update on one learner device: targets and the advantage filter once from the actor's values,
        then num_minibatches optimizer steps (each optionally accumulated over grad_accum micro-batches)."""
        cfg = args.ataraxos
        storage = jax.tree.map(lambda *x: jnp.hstack(x), *sharded_storages)
        next_dist, init_rstate = [jax.tree.map(lambda *x: jnp.concatenate(x), *x)
                                  for x in [sharded_next_data, sharded_init_rstate]]
        num_steps, num_envs = storage.rewards.shape
        not_decision = storage.dones | storage.withdrawn  # reset transitions and withdrawn decisions
        returns, advantages = ataraxos.targets_by_seat(
            storage.value_dists, storage.rewards, not_decision, storage.next_dones, storage.mains, next_dist, cfg)
        valid = ~not_decision
        kept, threshold = ataraxos.advantage_mask(advantages, valid, cfg)

        def split_micro(x, k, steps_axis):
            if steps_axis:  # [T * E, ...] time-major -> [k, T * E / k, ...]
                envs = x.shape[0] // num_steps_mb
                x = x.reshape((num_steps_mb, k, envs // k) + x.shape[1:])
                return jnp.moveaxis(x, 1, 0).reshape((k, num_steps_mb * (envs // k)) + x.shape[3:])
            return x.reshape((k, x.shape[0] // k) + x.shape[1:])

        num_steps_mb = num_steps

        def minibatch_grads(agent_state, minibatch):
            init_r, obs, dones, mains, actions, logits, rets, adv, kept_mb, labels, withdrawn = minibatch
            if args.grad_accum == 1:
                return ataraxos_grad_fn(agent_state.params, agent_state.batch_stats, agent_state.constants,
                                        init_r, obs, dones, mains, actions, logits, rets, adv, kept_mb, labels,
                                        withdrawn, temperature_now)
            k = args.grad_accum
            belief_rows = (obs["candidates_"][..., 2] > 0) & ~(dones | withdrawn)[:, None]
            denominators = (kept_mb.sum().astype(jnp.float32),
                            belief_rows.sum().astype(jnp.float32) * len(BELIEF_LOCATIONS))
            steps = lambda x: None if x is None else jax.tree.map(lambda y: split_micro(y, k, True), x)
            micro = (jax.tree.map(lambda y: split_micro(y, k, False), init_r), steps(obs), steps(dones),
                     steps(mains), steps(actions), steps(logits), steps(rets), steps(adv), steps(kept_mb),
                     steps(labels), steps(withdrawn))

            def body(total, part):
                (loss, terms), grads = ataraxos_grad_fn(
                    agent_state.params, agent_state.batch_stats, agent_state.constants, *part, temperature_now,
                    denominators=denominators)
                return jax.tree.map(jnp.add, total, ((loss, terms), grads)), None

            probe = jax.eval_shape(lambda: ataraxos_grad_fn(
                agent_state.params, agent_state.batch_stats, agent_state.constants,
                *jax.tree.map(lambda y: y[0], micro), temperature_now, denominators=denominators))
            zero = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), probe)
            return jax.lax.scan(body, zero, micro)[0]

        def update_epoch(carry, _):
            agent_state, key = carry
            key, subkey = jax.random.split(key)
            convert = lambda x, multi_step=True: reshape_minibatch(
                x, multi_step, args.num_minibatches, num_steps, None, key=subkey)
            b_init = jax.tree.map(partial(convert, multi_step=False), init_rstate)
            per_step = (storage.obs, storage.dones, storage.mains, storage.actions, storage.logits, returns,
                        advantages, kept, storage.belief_labels, storage.withdrawn)
            b = jax.tree.map(convert, per_step)

            def update_minibatch(agent_state, minibatch):
                (loss, terms), grads = minibatch_grads(agent_state, minibatch)
                grads = jax.lax.pmean(grads, axis_name="local_devices")
                terms = dict(terms, g_norm=optax.global_norm(grads))  # before clipping (rl.py logs g_norm)
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, terms)

            agent_state, (loss, terms) = jax.lax.scan(update_minibatch, agent_state, (b_init, *b))
            return (agent_state, key), (loss, terms)

        (agent_state, key), (loss, terms) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs)
        mean = lambda x: jax.lax.pmean(x, axis_name="local_devices").mean()
        first_g_norm = terms["g_norm"].reshape(-1)[0]  # the update's first step: both runs start from equal params
        terms = {name: mean(value) for name, value in terms.items()}
        terms["g_norm_first"] = first_g_norm
        terms["kept_fraction"] = mean(kept.sum() / jnp.maximum(valid.sum(), 1))
        terms["threshold"] = mean(threshold)
        loss = mean(loss)
        return agent_state, loss, terms["policy"], terms["value"], terms["entropy"], terms["kl"], key, terms

    all_reduce_value = jax.pmap(
        lambda x: jax.lax.pmean(x, axis_name="main_devices"),
        axis_name="main_devices",
        devices=global_main_devices,
    )

    multi_device_update = first_call_serialized(jax.pmap(
        ataraxos_update if args.recipe == "ataraxos" else single_device_update,
        axis_name="local_devices",
        devices=global_learner_decices,
    ))

    params_queues = []
    rollout_queues = []
    n_actor_threads_total = len(args.actor_device_ids) * args.num_actor_threads
    if resume_actors is not None and len(resume_actors) != n_actor_threads_total:
        raise ValueError(f"the checkpoint holds {len(resume_actors)} actor threads, this run has {n_actor_threads_total}")

    unreplicated_params = flax.jax_utils.unreplicate(get_variables(agent_state))
    for d_idx, d_id in enumerate(args.actor_device_ids):
        actor_device = local_devices[d_id]
        device_params = jax.device_put(unreplicated_params, actor_device)
        for thread_id in range(args.num_actor_threads):
            params_queues.append(queue.Queue(maxsize=1))
            rollout_queues.append(queue.Queue(maxsize=1))
            if eval_variables:
                params_queues[-1].put(
                    jax.device_put(eval_variables, actor_device))
            actor_thread_id = d_idx * args.num_actor_threads + thread_id             
            threading.Thread(
                target=rollout,
                args=(
                    jax.device_put(actor_keys[actor_thread_id], actor_device),
                    args,
                    rollout_queues[-1],
                    params_queues[-1],
                    writer if d_idx == 0 and thread_id == 0 else dummy_writer,
                    actor_device,
                    learner_devices,
                    actor_thread_id,
                    None if resume_actors is None else resume_actors[actor_thread_id],
                ),
            ).start()
            params_queues[-1].put(device_params)

    if args.iteration_decisions:
        critic = None
        if args.critic:
            critic = make_critic(args, sample_obs, semantic_tables, run_card_tables, learner_devices,
                                 global_learner_decices,
                                 args.resume if args.resume and not (args.resume_add_critic or args.resume_reset_critic)
                                 else None)
        run_iterations(args, agent_state, ema, learner_devices, global_learner_decices, rollout_queues, params_queues,
                       local_devices, n_actor_threads_total, update_offset, receipt_base, parent_sha, learner_keys,
                       writer, a0_step, critic)
        if args.distributed:
            jax.distributed.shutdown()
        writer.close()
        return

    rollout_queue_get_time = deque(maxlen=10)
    learner_policy_version = 0
    while True:
        learner_policy_version += 1
        rollout_queue_get_time_start = time.time()
        sharded_data_list = []
        eval_stat_list = []
        for d_idx, d_id in enumerate(args.actor_device_ids):
            for thread_id in range(args.num_actor_threads):
                (
                    global_step,
                    update,
                    *sharded_data,
                    avg_params_queue_get_time,
                    eval_stats,
                ) = rollout_queues[d_idx * args.num_actor_threads + thread_id].get()
                sharded_data_list.append(sharded_data)
                if eval_stats is not None:
                    eval_stat_list.append(eval_stats)

        tb_global_step = args.tb_offset + global_step
        if args.eval_interval > 0 and update % args.eval_interval == 0:
            eval_stats = np.mean(eval_stat_list, axis=0)
            eval_stats = jax.device_put(eval_stats, local_devices[0])
            eval_stats = np.array(all_reduce_value(eval_stats[None])[0])
            eval_time, eval_return, eval_win_rate = eval_stats
            writer.add_scalar(f"charts/eval_return", eval_return, tb_global_step)
            writer.add_scalar(f"charts/eval_win_rate", eval_win_rate, tb_global_step)
            print(f"eval_time={eval_time:.4f}, eval_return={eval_return:.4f}, eval_win_rate={eval_win_rate:.4f}")

        rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
        training_time_start = time.time()
        if args.recipe == "ataraxos":
            temperature_now = float(ataraxos.temperature(
                (update_offset + learner_policy_version - 1) * args.batch_size, args.ataraxos))
            (agent_state, loss, pg_loss, v_loss, ent_loss, approx_kl, learner_keys, terms) = multi_device_update(
                agent_state, *list(zip(*sharded_data_list)), learner_keys,
                jnp.full((len(learner_devices),), temperature_now, jnp.float32))
        else:
            (agent_state, loss, pg_loss, v_loss, ent_loss, approx_kl, learner_keys) = multi_device_update(
                agent_state,
                *list(zip(*sharded_data_list)),
                learner_keys,
            )
        unreplicated_params = flax.jax_utils.unreplicate(get_variables(agent_state))
        if args.recipe == "ataraxos":  # EMA of the parameters, once per update (exponential_weighted_average.py)
            ema = ataraxos.ema_update(ema, unreplicated_params["params"], args.ataraxos.ema_decay)
        params_queue_put_time = 0
        for d_idx, d_id in enumerate(args.actor_device_ids):
            device_params = jax.device_put(unreplicated_params, local_devices[d_id])
            jax.block_until_ready(device_params)
            params_queue_put_start = time.time()
            for thread_id in range(args.num_actor_threads):
                params_queues[d_idx * args.num_actor_threads + thread_id].put(device_params)
            params_queue_put_time += time.time() - params_queue_put_start

        loss = loss[-1].item()
        if args.recipe == "ataraxos" and learner_policy_version % args.log_frequency == 0:
            report = {name: float(value[-1]) for name, value in terms.items()}
            report.update(temperature=temperature_now, lr=float(ataraxos.learning_rate(
                (update_offset + learner_policy_version - 1) * args.batch_size, args.ataraxos)))
            for name, value in report.items():
                writer.add_scalar(f"ataraxos/{name}", value, args.tb_offset + global_step)
            print("ATARAXOS " + json.dumps({"update": update_offset + learner_policy_version,
                                            **{k: float(f"{v:.9g}") for k, v in report.items()}}), flush=True)
        if learner_policy_version == 1:  # the checkpoint path's slices of the replicated state compile now
            jax.block_until_ready(flax.jax_utils.unreplicate(agent_state))
        if learner_policy_version == 3:  # every declared shape has compiled (actor blocks, first learner update)
            compiles_after_warmup = COMPILES[0]
            print(f"WARMUP-COMPILES {compiles_after_warmup}", flush=True)
        elif learner_policy_version > 3 and COMPILES[0] != compiles_after_warmup:
            raise RuntimeError(f"{COMPILES[0] - compiles_after_warmup} compiles after warmup: a shape outside the "
                               "declared blocks")
        if np.isnan(loss) or np.isinf(loss):
            raise ValueError(f"loss is {loss}")

        # record rewards for plotting purposes
        if learner_policy_version % args.log_frequency == 0:
            writer.add_scalar("stats/rollout_queue_get_time", np.mean(rollout_queue_get_time), tb_global_step)
            writer.add_scalar(
                "stats/rollout_params_queue_get_time_diff",
                np.mean(rollout_queue_get_time) - avg_params_queue_get_time,
                tb_global_step,
            )
            writer.add_scalar("stats/training_time", time.time() - training_time_start, tb_global_step)
            writer.add_scalar("stats/rollout_queue_size", rollout_queues[-1].qsize(), tb_global_step)
            writer.add_scalar("stats/params_queue_size", params_queues[-1].qsize(), tb_global_step)
            print(
                f"{tb_global_step} actor_update={update}, "
                f"train_time={time.time() - training_time_start:.2f}, "
                f"data_time={rollout_queue_get_time[-1]:.2f}, "
                f"put_time={params_queue_put_time:.2f}"
            )
            writer.add_scalar(
                "charts/learning_rate",
                (agent_state.opt_state.inner_states["policy"].inner_state[1].hyperparams["learning_rate"][-1].item()
                 if args.architecture == "policy_net" else
                 agent_state.opt_state[3][2][1].hyperparams["learning_rate"][-1].item()), tb_global_step
            )
            writer.add_scalar("losses/value_loss", v_loss[-1].item(), tb_global_step)
            writer.add_scalar("losses/policy_loss", pg_loss[-1].item(), tb_global_step)
            writer.add_scalar("losses/entropy", ent_loss[-1].item(), tb_global_step)
            writer.add_scalar("losses/approx_kl", approx_kl[-1].item(), tb_global_step)
            writer.add_scalar("losses/loss", loss, tb_global_step)
            if args.decision_aux_coef:
                auxiliary = (
                    loss - pg_loss[-1].item()
                    + args.ent_coef * ent_loss[-1].item()
                    - args.vf_coef * v_loss[-1].item()
                ) / args.decision_aux_coef
                writer.add_scalar("losses/decision_outcome_loss", auxiliary, tb_global_step)
                print(f"decision_loss: step={tb_global_step} total={loss:.6f} "
                      f"outcome={auxiliary:.6f}", flush=True)
            if args.belief_coef:
                belief = (
                    loss - pg_loss[-1].item()
                    + args.ent_coef * ent_loss[-1].item()
                    - args.vf_coef * v_loss[-1].item()
                ) / args.belief_coef
                writer.add_scalar("losses/belief_nll", belief, tb_global_step)
                print(f"belief_nll: step={tb_global_step} nll={belief:.6f}", flush=True)

        stopping = STOP_REQUESTED.is_set()
        if args.local_rank == 0 and not args.debug and (
            (update_offset + learner_policy_version) % args.save_interval == 0  # absolute: the same updates on resume
            or learner_policy_version >= args.num_updates
            or stopping
        ):
            actors = collect_snapshots(update_offset + learner_policy_version + 1, n_actor_threads_total)
            saved = state_io.save(
                args.ckpt_dir, flax.jax_utils.unreplicate(agent_state), learner_keys,
                {"global_step": tb_global_step, "learner_update": update_offset + learner_policy_version},
                {**receipt_base, "parent": parent_sha, "resume_contract": resume_contract(args),
                 "value_menu_migrations": args.value_menu_migrations}, actors=actors,
                ema=ema)
            if args.keep_checkpoints > 0:
                from mirrorforce.agent.train.checkpoint_store import prune
                pruned = prune(args.ckpt_dir, args.keep_checkpoints, args.keep_every)
                if pruned:
                    print("PRUNED " + json.dumps(pruned), flush=True)
            print("CHECKPOINT " + json.dumps({"sha256": saved, "global_step": tb_global_step,
                                              "learner_update": update_offset + learner_policy_version}), flush=True)

        if learner_policy_version >= args.num_updates or stopping:
            if stopping:
                print(f"STOP_REQUESTED: checkpoint written at update {update_offset + learner_policy_version}", flush=True)
            break

    if args.distributed:
        jax.distributed.shutdown()

    writer.close()


class Critic(NamedTuple):
    model: object
    state: object  # replicated TrainState
    forward: object  # pmapped: (params, constants, packed) -> (win, draw, loss) [N, 3]
    step: object  # pmapped: (state, packed, small) -> (state, loss)


def make_critic(args, sample_obs, semantic_tables, run_card_tables, learner_devices, global_devices, resume):
    """A0's central critic: its own parameters, Adam (A0's lr schedule, clip 0.267) and pmapped forward and step."""
    from mirrorforce.agent.model.critic import CriticNet
    if not (args.iteration_decisions and args.export_both_seats):
        raise ValueError("the central critic trains in A0 iterations on the both-seat export (--export-both-seats)")
    config = dataclasses.replace(args.critic_model, hand_limit=args.net.hand_limit, public_status=args.net.public_status)
    model = CriticNet(config, tuple(args.semantic_shape), args.critic_mix_layers, args.critic_zero_init)
    obs = {k: jnp.asarray(v[:1]) for k, v in sample_obs.items() if v is not None}
    obs.update({f"priv:{k}": v for k, v in list(obs.items())})
    variables = flax.core.unfreeze(model.init(jax.random.PRNGKey(args.seed + 7), obs))
    if semantic_tables is not None:
        inject_semantic_constants(variables, semantic_tables)
    if run_card_tables is not None:
        card_tables.inject(variables, run_card_tables[0])
    c = args.ataraxos
    lr = lambda count: ataraxos.power_schedule(c.lr_coef, count // args.iteration_steps, c.lr_decay, c.lr_ceil,
                                               c.lr_floor)
    tx = optax.chain(optax.clip_by_global_norm(c.max_grad_norm),
                     optax.inject_hyperparams(optax.adam)(learning_rate=lr, eps=c.adam_eps))
    state = TrainState.create(apply_fn=None, params=variables["params"], tx=tx, batch_stats={},
                              constants=variables.get("constants", {}))
    if resume:  # a checkpoint of a run with the critic holds its state (a run without it: --resume-add-critic)
        core = state.replace(constants={})
        restored = (state_io.restore_extra(resume, "critic", core,
                                          add_value_menu=getattr(args, "resume_add_value_menu", False))
                    if args.local_rank == 0 else core)
        if restored is None:
            raise ValueError(f"{Path(resume).name} holds no central critic; add one with --resume-add-critic")
        if args.world_size > 1:  # read on process 0 only (restore_on_process0)
            from jax.experimental import multihost_utils
            restored = jax.tree.map(np.asarray, multihost_utils.broadcast_one_to_all(restored))
        state = restored.replace(constants=state.constants)
        print("CRITIC-RESUMED", flush=True)

    chunks = args.critic_chunks
    split = lambda x: x.reshape((chunks, x.shape[0] // chunks) + x.shape[1:])  # rows are independent here

    def forward(params, constants, packed):  # parameters only: the optimizer state's types change after a step
        obs = {k: split(v) for k, v in packed.items() if k not in ("logits_", "labels_")}
        run = lambda o: jax.nn.softmax(model.apply({"params": params, "constants": constants}, o), axis=-1)
        return jax.lax.map(run, obs).reshape((-1, 3))

    def step(st, packed, small):
        from mirrorforce.agent.model.critic import critic_loss
        obs = {k: split(v) for k, v in packed.items() if k not in ("logits_", "labels_")}
        valid = (~(small["dones"] | small["withdrawn"])).astype(jnp.float32)
        count = jax.lax.psum(valid.sum(), axis_name="local_devices")

        def chunk(carry, x):
            o, returns, weight = x
            loss = lambda params: critic_loss(model.apply({"params": params, "constants": st.constants}, o),
                                              returns, weight)[0] / jnp.maximum(count, 1.0)
            value, grads = jax.value_and_grad(loss)(st.params)
            return (carry[0] + value, jax.tree.map(jnp.add, carry[1], grads)), None
        zeros = (jnp.zeros((), jnp.float32), jax.tree.map(jnp.zeros_like, st.params))
        (value, grads), _ = jax.lax.scan(chunk, zeros, (obs, split(small["critic_returns"]), split(valid)))
        grads = jax.lax.psum(grads, axis_name="local_devices")
        return st.apply_gradients(grads=grads), jax.lax.psum(value, axis_name="local_devices")

    return Critic(model, flax.jax_utils.replicate(state, devices=learner_devices),
                  jax.pmap(forward, axis_name="local_devices", devices=global_devices),
                  jax.pmap(step, axis_name="local_devices", devices=global_devices))


CRITIC_TARGETS = jax.jit(ataraxos.targets_by_seat, static_argnums=6)  # one compile per (shape, recipe), not per call


def calibration(p, outcome, known, turns):
    """Brier score and log loss of (win, draw, loss) predictions ``p`` against outcomes over the ``known`` decisions,
    overall and by turn (the calibration monitor's buckets)."""
    from mirrorforce.agent.train.calibration_monitor import TURNS

    def stats(mask):
        if not mask.any():
            return {"n": 0, "brier": None, "logloss": None}
        brier = ((p - outcome) ** 2).sum(-1)[mask].mean()
        logloss = -np.log(np.maximum((p * outcome).sum(-1)[mask], 1e-12)).mean()
        return {"n": int(mask.sum()), "brier": round(float(brier), 5), "logloss": round(float(logloss), 5)}
    out = stats(known)
    out["by_turn"] = {f"{lo}-{hi if hi < 10 ** 9 else ''}": stats(known & (turns >= lo) & (turns <= hi))
                      for lo, hi in TURNS}
    return out


def critic_targets(segments, dists, turns, threads, envs, T, advantages_from_critic, cfg, td_lambda=None, *, timings=None):
    """The central critic's TD(lambda) returns (the recipe's 0.8, or ``td_lambda``: 1.0 gives the outcome of every
    game that ends within the iteration) and GAE(0.5) advantages along each environment's whole iteration, and a
    held-out comparison with the public value head.

    ``segments`` in collection order (per update, per rollout queue, per environment: index (u * threads + q) * envs
    + e), ``dists`` the critic's (win, draw, loss) per segment [T, 3], ``turns`` each decision's turn per segment [T].
    An environment's segments are consecutive in
    time, so its chains cross segment boundaries; after its last segment they bootstrap from the actor's public value
    of the next observation (the critic has no other-seat view there), which reaches earlier decisions with weight
    0.8^k (returns) and 0.5^k (advantages). The comparison: Brier scores of the critic and of the public head (both
    predicted before training on these games) against the outcome of every decision whose game ends within the
    iteration: Brier score and log loss, overall and by turn."""
    began = time.monotonic()
    width = threads * envs
    updates = len(segments) // width
    if updates * width != len(segments):
        raise ValueError(f"{len(segments)} segments are not whole updates of {threads} x {envs} environments")
    grid = lambda get: np.concatenate([np.stack([get(segments[(u * threads + q) * envs + e])
                                                 for q in range(threads) for e in range(envs)], axis=1)
                                       for u in range(updates)], axis=0)  # [U*T, threads*envs, ...]
    critic = np.concatenate([np.stack([dists[(u * threads + q) * envs + e] for q in range(threads)
                                       for e in range(envs)], axis=1) for u in range(updates)], axis=0)
    rewards, dones, nexts, mains = grid(lambda s: s.rewards), grid(lambda s: s.dones | s.withdrawn), \
        grid(lambda s: s.next_dones), grid(lambda s: s.mains)
    public = grid(lambda s: s.value_dists)
    turn = np.concatenate([np.stack([turns[(u * threads + q) * envs + e] for q in range(threads)
                                     for e in range(envs)], axis=1) for u in range(updates)], axis=0)
    last = np.stack([segments[((updates - 1) * threads + q) * envs + e].bootstrap
                     for q in range(threads) for e in range(envs)])
    run = lambda d, lam: [np.asarray(x) for x in CRITIC_TARGETS(
        jnp.asarray(d, jnp.float32), jnp.asarray(rewards, jnp.float32), jnp.asarray(dones), jnp.asarray(nexts),
        jnp.asarray(mains), jnp.asarray(last, jnp.float32), dataclasses.replace(cfg, td_lambda=lam))]
    grid_done = time.monotonic()
    effective_lambda = cfg.td_lambda if td_lambda is None else td_lambda
    returns, advantages = run(critic, effective_lambda)
    # This run has already computed the identical TD(1) outcome when lambda is
    # exactly 1. Keep the distinct calibration pass for every other lambda.
    outcome = returns if effective_lambda == 1.0 else run(critic, 1.0)[0]
    targets_done = time.monotonic()
    ended = (nexts & ~dones)[::-1].cumsum(0)[::-1] > 0  # a game end at or after the decision in this environment
    known = ended & ~dones
    report = {"critic": calibration(critic, outcome, known, turn), "public": calibration(public, outcome, known, turn)}
    calibration_done = time.monotonic()
    for u in range(updates):
        for q in range(threads):
            for e in range(envs):
                seg = segments[(u * threads + q) * envs + e]
                seg.critic_returns = returns[u * T:(u + 1) * T, q * envs + e].copy()
                if advantages_from_critic:
                    seg.advantages = advantages[u * T:(u + 1) * T, q * envs + e].copy()
    if timings is not None:
        timings.update(critic_target_grid_s=grid_done - began,
                       critic_target_compute_readback_s=targets_done - grid_done,
                       critic_calibration_s=calibration_done - targets_done,
                       critic_target_assign_s=time.monotonic() - calibration_done,
                       critic_target_passes=1 if effective_lambda == 1.0 else 2)
    return report


def run_iterations(args, agent_state, ema, learner_devices, global_learner_devices, rollout_queues, params_queues,
                   local_devices, n_actor_threads_total, update_offset, receipt_base, parent_sha, learner_keys, writer,
                   a0_step, critic=None):
    """A0's learner: for each iteration, the actors' iteration_updates updates of segments (one behaviour policy),
    then iteration_steps optimizer steps over a permutation of them (one epoch; minibatches built in a background
    thread), the EMA, a checkpoint at iteration boundaries and the next behaviour policy."""
    cfg, c = args.ataraxos, args.net
    update = jax.pmap(a0_step, axis_name="local_devices", devices=global_learner_devices)
    pool = concurrent.futures.ThreadPoolExecutor(args.iteration_threads)
    builder = concurrent.futures.ThreadPoolExecutor(args.iteration_prefetch)  # minibatches built ahead of the steps
    devices = len(learner_devices)
    put = lambda x: jax.device_put_sharded(list(x), learner_devices)
    decisions_per_iteration = args.iteration_updates * args.num_envs * args.num_steps
    total_iterations = max(1, args.total_timesteps // decisions_per_iteration)
    iteration, compiles_after_warmup, global_step = update_offset, None, 0
    while True:
        iteration += 1
        iteration_started = time.monotonic()
        timings = {name: 0.0 for name in ("buffer_account_s", "critic_assemble_check_s", "critic_put_enqueue_s",
                   "critic_forward_readback_s", "critic_scatter_s", "critic_precompute_s", "stats_sync_log_s",
                   "checkpoint_save_s", "checkpoint_prune_s", "param_publish_s")}
        start = time.time()
        segments, layout = [], None
        for _ in range(args.iteration_updates):
            for q in rollout_queues:
                global_step, actor_update, _, layout, part, _, _ = q.get()
                segments.extend(part)
        counters = a0_counters.iteration_counters(args.a0_counter_basis, global_step, iteration)
        collect_time = time.time() - start
        phase = time.monotonic()
        timings["collect_s"] = phase - iteration_started
        stored = a0_buffer.nbytes(segments)
        timings["buffer_account_s"] = time.monotonic() - phase
        critic_report, critic_losses = None, []
        if critic is not None:  # the critic's values on this iteration's data (held out), then its targets
            critic_started = time.monotonic()
            dists, turns = [None] * len(segments), [None] * len(segments)
            for group in np.array_split(np.arange(len(segments)), args.iteration_steps):
                phase = time.monotonic()
                packed = a0_buffer.assemble_critic([segments[i] for i in group], layout, args.num_steps, devices, pool)
                if not packed["priv:cards_"].any():  # the export must reach the critic (it once was dropped)
                    raise RuntimeError("the central critic's batch holds no other-seat observation (priv:cards_)")
                timings["critic_assemble_check_s"] += time.monotonic() - phase
                phase = time.monotonic()
                device_packed = {k: put(v) for k, v in packed.items()}
                timings["critic_put_enqueue_s"] += time.monotonic() - phase
                phase = time.monotonic()
                out = np.asarray(critic.forward(critic.state.params, critic.state.constants, device_packed))
                del device_packed  # do not keep an extra device minibatch alive during the next assemble/put
                timings["critic_forward_readback_s"] += time.monotonic() - phase
                phase = time.monotonic()
                per = len(group) // devices
                for d in range(devices):
                    block = out[d].reshape(args.num_steps, per, 3)
                    turn = np.asarray(packed["global_"][d]).reshape(args.num_steps, per, -1)[..., 4].astype(np.int64)
                    for j in range(per):
                        dists[group[d * per + j]], turns[group[d * per + j]] = block[:, j], turn[:, j]
                timings["critic_scatter_s"] += time.monotonic() - phase
            critic_report = critic_targets(segments, dists, turns, len(rollout_queues), args.local_num_envs,
                                           args.num_steps, args.critic_advantages, cfg, args.critic_td_lambda,
                                           timings=timings)
            timings["critic_precompute_s"] = time.monotonic() - critic_started

        train_started = time.monotonic()
        start = time.time()
        temperature_now = float(ataraxos.power_schedule(cfg.temperature_coef, iteration - 1, cfg.temperature_decay,
                                                        cfg.temperature_ceil, cfg.temperature_floor))
        lr_now = float(ataraxos.power_schedule(cfg.lr_coef, iteration - 1, cfg.lr_decay, cfg.lr_ceil, cfg.lr_floor))
        groups = np.split(np.random.default_rng([args.seed, iteration, args.local_rank]).permutation(len(segments)),
                          args.iteration_steps)

        def build(group):
            began = time.time()
            packed, small, rstate = a0_buffer.assemble([segments[i] for i in group], layout, args.num_steps, devices,
                                                       c.memory_slots, c.chunk_slots, c.d, pool)
            out = (jax.tree.map(put, rstate), {k: put(v) for k, v in packed.items()},
                   {k: put(v) for k, v in small.items()})
            return out, time.time() - began
        pending = collections.deque(builder.submit(build, g) for g in groups[:args.iteration_prefetch])
        sums, wait, build_s, step_s = None, 0.0, 0.0, 0.0
        temperature = jnp.full((devices,), temperature_now, jnp.float32)
        for j in range(len(groups)):
            waited = time.time()
            (rstate, packed, small), built = pending.popleft().result()
            wait += time.time() - waited
            build_s += built
            if j + args.iteration_prefetch < len(groups):
                pending.append(builder.submit(build, groups[j + args.iteration_prefetch]))
            stepped = time.time()
            agent_state, terms = update(agent_state, rstate, packed, small, temperature)
            if critic is not None:
                critic_state, critic_loss = critic.step(critic.state, packed, small)
                critic = critic._replace(state=critic_state)
                critic_losses.append(critic_loss[0])
            terms = jax.tree.map(lambda x: x[0], terms)
            jax.block_until_ready(terms)  # each step's wall time on this process (collectives included)
            step_s += time.time() - stepped
            sums = terms if sums is None else jax.tree.map(jnp.add, sums, terms)
        report = {name: float(value) / len(groups) for name, value in sums.items()}
        params = flax.jax_utils.unreplicate(get_variables(agent_state))
        ema = ataraxos.ema_update(ema, params["params"], cfg.ema_decay)
        train_time = time.time() - start
        timings["train_host_s"] = time.monotonic() - train_started
        phase = time.monotonic()
        if not np.isfinite(report["loss"]):
            raise ValueError(f"A0 iteration {iteration}: loss {report['loss']}")
        if report.get("kept_overflow", 0) > 0 or report.get("delivery_bad", 0) > 0:  # before this iteration is saved
            raise RuntimeError(f"A0 iteration {iteration}: kept rows beyond the capacity (kept_overflow "
                               f"{report.get('kept_overflow')}) or an inconsistent delivery (delivery_bad "
                               f"{report.get('delivery_bad')})")
        line = {"iteration": iteration, "global_step": counters["global_step"], "actor_global_step": global_step,
                "decisions": len(segments) * args.num_steps, "collect_s": round(collect_time, 1),
                "train_s": round(train_time, 1), "minibatch_wait_s": round(wait, 1), "build_s": round(build_s, 1),
                "step_s": round(step_s, 1),
                "decisions_per_s": round(len(segments) * args.num_steps / (collect_time + train_time)),
                "buffer_gb": round(stored / 1e9, 2), "temperature": temperature_now, "lr": lr_now,
                **{k: float(f"{v:.9g}") for k, v in report.items()}}
        print("A0 " + json.dumps(line), flush=True)
        stats = [d.memory_stats() for d in jax.local_devices()]
        if all(stats):  # GPU: each local device's peak since the start against its allocator limit
            print("MEMORY " + json.dumps({"iteration": iteration,
                                          "peak_gib": [round(m["peak_bytes_in_use"] / 2 ** 30, 2) for m in stats],
                                          "limit_gib": [round(m["bytes_limit"] / 2 ** 30, 2) for m in stats]}),
                  flush=True)
        if critic is not None:
            critic_line = {"iteration": iteration, "advantages": "critic" if args.critic_advantages else "public",
                           "loss": float(np.mean([float(x) for x in critic_losses])), **critic_report}
            print("CRITIC " + json.dumps(critic_line), flush=True)
        for name, value in line.items():
            writer.add_scalar(f"a0/{name}", value, iteration)
        if iteration == update_offset + 1:
            jax.block_until_ready(flax.jax_utils.unreplicate(agent_state))  # the checkpoint's slices compile here
            if critic is not None:
                jax.block_until_ready(flax.jax_utils.unreplicate(critic.state))
            compiles_after_warmup = COMPILES[0]
            print(f"WARMUP-COMPILES {compiles_after_warmup}", flush=True)
        elif COMPILES[0] != compiles_after_warmup:
            raise RuntimeError(f"{COMPILES[0] - compiles_after_warmup} compiles after the first iteration")
        del segments
        timings["stats_sync_log_s"] = time.monotonic() - phase

        stopping = STOP_REQUESTED.is_set()
        done = iteration >= update_offset + total_iterations
        if args.local_rank == 0 and not args.debug and (iteration % args.save_interval == 0 or stopping or done):
            phase = time.monotonic()
            # several processes: only process 0 writes, so other processes' environments are not in the checkpoint
            # and a resume restarts every environment (declared: --resume-topology-change)
            actors = (collect_snapshots(iteration * args.iteration_updates + 1, n_actor_threads_total)
                      if args.world_size == 1 else None)
            saved = state_io.save(
                args.ckpt_dir, flax.jax_utils.unreplicate(agent_state), learner_keys,
                counters,
                {**receipt_base, "parent": parent_sha, "resume_contract": resume_contract(args),
                 "a0_counter_basis": args.a0_counter_basis,
                 "env_changes": args.env_changes, "declared_changes": args.declared_changes,
                 "value_menu_migrations": args.value_menu_migrations,
                 "a0": {"iteration_updates": args.iteration_updates, "iteration_steps": args.iteration_steps,
                        "decisions_per_iteration": decisions_per_iteration,
                        "advantages": "critic" if args.critic_advantages else "public",
                        "advantages_since": args.advantages_since, "advantages_switch": args.advantages_switch,
                        **({"critic_started": args.critic_started} if critic is not None else {})}},
                actors=actors, ema=ema,
                critic=None if critic is None else flax.jax_utils.unreplicate(critic.state))
            timings["checkpoint_save_s"] = time.monotonic() - phase  # includes state_io's existing arrival rehash
            if args.keep_checkpoints > 0:
                phase = time.monotonic()
                from mirrorforce.agent.train.checkpoint_store import prune
                pruned = prune(args.ckpt_dir, args.keep_checkpoints, args.keep_every)
                if pruned:
                    print("PRUNED " + json.dumps(pruned), flush=True)
                timings["checkpoint_prune_s"] = time.monotonic() - phase
            print("CHECKPOINT " + json.dumps({"sha256": saved, **counters}), flush=True)
        def emit_timing():
            # Host spans with ONLY the existing barriers: put is enqueue time;
            # forward includes its existing np.asarray wait/readback. Child
            # critic spans are inside critic_precompute_s, never added twice.
            print("A0-TIMING " + json.dumps({"schema": "mirrorforce_a0_host_phase_timing/v1",
                  "iteration": iteration, "iteration_wall_s": time.monotonic() - iteration_started,
                  "recorded_epoch": time.time(), **timings}), flush=True)

        if done or stopping:
            emit_timing()
            if stopping:
                print(f"STOP_REQUESTED: checkpoint written at iteration {iteration}", flush=True)
            break
        phase = time.monotonic()
        for d_idx, d_id in enumerate(args.actor_device_ids):
            device_params = jax.device_put(params, local_devices[d_id])
            for thread_id in range(args.num_actor_threads):
                params_queues[d_idx * args.num_actor_threads + thread_id].put(device_params)
        timings["param_publish_s"] = time.monotonic() - phase
        emit_timing()


if __name__ == "__main__":
    main()
