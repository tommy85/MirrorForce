"""The policy service: a checkpoint playing network games through its clients' received streams.

Each game is a session holding a client-mode builder (``netduel.agent_client.AgentClientDuel`` over
``duel_native.ClientDuel``: the env's observation, menus, sub-choices and response encoding, rebuilt from one
seat's received messages) and the policy's recurrent state. A client
(``netduel.agent_policy.RemotePolicy``) forwards every game message it receives; at each prompt the service feeds the
messages, lets the builder answer forced prompts itself, and otherwise asks the policy at every decision until the
builder returns the response bytes. Protocol: ``mirrorforce.netduel.agent_wire``.

Selection: ``sample`` (temperature ``--temperature`` over the policy's logits of the shown rows; one seeded generator
per session) or ``greedy`` (the highest logit). ``--weights`` picks the checkpoint's EMA or iterate parameters. The
checkpoint, its receipt and the native module are verified by the trainer's own loader (``agent.train.policy_io``).
Explicit ``--weights debiased_ema`` additionally requires a SHA-pinned
``--debiased-ema-request`` and admits only explicitly identified behavior-diagnostic sessions.

    MF_DUEL_NATIVE=<module> python mf_runtime_policy_service.py --checkpoint <sha>.ckpt --weights ema \
        --selection sample --temperature 1 --cards-db ... --code-list ... --script-root ... --announce-tables ... \
        --socket /path/svc.sock --out <dir>

``--backend uniform`` replaces the network by uniform logits (protocol tests only; its identity says so).
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import socketserver
import threading
import time

import numpy as np

from mirrorforce.netduel import agent_wire as W
from mirrorforce.netduel import agent_public_recipe as PR
from mirrorforce.probes.client_parity import card_view_law
from mirrorforce.agent.public_world_codec import RPC_SCHEMA, RPC_CAPABILITY, encode_world, world_sha256

SCHEMA = "mirrorforce_policy_service_receipt/v1"
STREAM_SCHEMA = "mirrorforce_replayed_client/v1"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation_sha256(obs):
    h = hashlib.sha256()
    for name in sorted(obs):
        array = np.ascontiguousarray(obs[name])
        h.update(name.encode() + b"\0" + str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())
    return h.hexdigest()


def memory_sha256(state: object) -> str:
    """A deterministic digest of recurrent-state trees, including structure, shapes and dtypes."""
    h = hashlib.sha256()

    def visit(value):
        if value is None:
            h.update(b"none;")
        elif isinstance(value, (tuple, list)):
            h.update(f"{type(value).__name__}:{len(value)}[".encode())
            for child in value:
                visit(child)
            h.update(b"]")
        elif isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise ValueError("memory dictionaries require string keys")
            h.update(b"dict{")
            for key in sorted(value):
                h.update(json.dumps(key).encode())
                visit(value[key])
            h.update(b"}")
        else:
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError("object arrays are not recurrent memory")
            header = json.dumps([str(array.dtype), array.shape], separators=(",", ":")).encode()
            h.update(len(header).to_bytes(8, "big") + header + array.tobytes())

    visit(state)
    return h.hexdigest()


def checked_stream(request: dict) -> tuple[list, list]:
    """Only synthetic client packets and recorded response bytes cross the replay boundary.

    This validates framing, not the producer's information provenance. Option A must separately prove that its
    producer uses a public prefix plus a particle, never the live opponent's session or hidden truth.
    """
    required = {"op", "seat", "main", "extra", "seed", "frames", "messages"}
    optional = {"opponent_recipe_mode", "public_opponent_recipe"}
    if not required <= request.keys() or request.keys() - required - optional:
        raise ValueError("open_stream requires only a fresh client's declaration, frames and trailing messages")
    if request["op"] != "open_stream" or type(request["seat"]) is not int or request["seat"] not in (0, 1) \
            or type(request["seed"]) is not int:
        raise ValueError("open_stream needs a seat and an integer seed")
    for field in ("main", "extra"):
        if not isinstance(request[field], list) or any(type(c) is not int or c <= 0 for c in request[field]):
            raise ValueError("a submitted recipe is a list of positive card codes")

    def unhex(value):
        if not isinstance(value, str):
            raise ValueError("stream bytes must be canonical hexadecimal strings")
        raw = bytes.fromhex(value)
        if raw.hex() != value:
            raise ValueError("stream bytes must be canonical hexadecimal strings")
        return raw

    def messages(value):
        if not isinstance(value, list):
            raise ValueError("messages must be a list")
        out = []
        for item in value:
            if not isinstance(item, list) or len(item) != 2 or type(item[0]) is not int or not 0 <= item[0] <= 255:
                raise ValueError("a message is [byte-sized message id, hexadecimal payload]")
            out.append((item[0], unhex(item[1])))
        return out

    if not isinstance(request["frames"], list):
        raise ValueError("frames must be a list")
    frames = []
    for frame in request["frames"]:
        if not isinstance(frame, dict) or set(frame) != {"messages", "response"}:
            raise ValueError("each historical frame contains only messages and its response")
        frames.append((messages(frame["messages"]), unhex(frame["response"])))
    return frames, messages(request["messages"])


#: The magnets the root step implements, as the trainer names them (``agent/train/ataraxos.AtaraxosConfig.magnet``).
MAGNETS = ("uniform_legal", "magnet_card_grouped/v1")


def search_batch_sizes(value: str) -> tuple[int, ...]:
    """Explicit bounded compilation shapes for search; pure-policy services need not enable them."""
    try:
        sizes = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("search batch sizes are comma-separated integers") from exc
    if not sizes or sizes[0] != 1 or tuple(sorted(set(sizes))) != sizes or sizes[-1] > 256:
        raise argparse.ArgumentTypeError("search batch sizes must increase from 1, with a maximum of 256")
    return sizes


def constant_batch_size(value: str) -> int:
    """One explicitly registered batch shape for ALL forwards, including single live decisions."""
    try:
        size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("constant batch size must be an integer from 1 to 256") from exc
    if not 1 <= size <= 256:
        raise argparse.ArgumentTypeError("constant batch size must be an integer from 1 to 256")
    return size


def padded_batches(items: list, sizes: tuple[int, ...]):
    """Yield (real count, padded items), duplicating a valid last row, never making an invalid zero observation.

    Padding changes only network execution shapes. Discard its outputs before touching any session or RNG.
    Larger requests are split at the largest registered shape, in input order.
    """
    if not items:
        return
    if not sizes:
        yield len(items), items
        return
    for begin in range(0, len(items), sizes[-1]):
        part = items[begin:begin + sizes[-1]]
        count = len(part)
        size = next(size for size in sizes if size >= count)
        yield count, part + [part[-1]] * (size - count)


def root_log_magnet(magnet, groups):
    """log rho over one menu under the checkpoint's magnet: uniform over the prompt's legal options, or the trainer's
    card-grouped magnet."""
    if magnet == "uniform_legal":
        return np.full(len(groups), -np.log(len(groups)))
    if magnet == "magnet_card_grouped/v1":
        return magnet_log_probs(groups)
    raise ValueError(f"unknown magnet {magnet!r}")


def magnet_log_probs(groups):
    """The trainer's card-grouped magnet (``agent/train/ataraxos.magnet_log_probs``) over one menu, every row legal."""
    import jax.numpy as jnp
    from mirrorforce.agent.train.ataraxos import magnet_log_probs as trainer_magnet
    groups = np.asarray(groups, np.int32)[None]
    return np.asarray(trainer_magnet(jnp.ones(groups.shape, bool), jnp.asarray(groups)), np.float64)[0]


def compute_device():
    """Where the network runs (JAX's default device): CPU and GPU bf16 round differently, so every game records it."""
    try:
        import jax
    except ImportError:  # the uniform backend's tests run without JAX
        return {"platform": "none"}
    device = jax.devices()[0]
    return {"platform": device.platform, "kind": device.device_kind, "count": len(jax.devices()),
            "jax": jax.__version__}


class UniformBackend:
    """Uniform logits and a zero value (protocol tests)."""

    identity = {"backend": "uniform"}

    def __init__(self, magnet="uniform_legal"):
        self.magnet = magnet

    def initial_state(self):
        return None

    def act(self, obs, state, first, rows):
        return state, np.zeros(rows, np.float64), 0.0, [1 / 3, 1 / 3, 1 / 3]

    def act_batch(self, items):
        return [self.act(*item) for item in items]


class CheckpointBackend:
    """A checkpoint through the trainer's loader (``mirrorforce.agent.train.policy_io``): it verifies the payload, the
    receipt and the env semantics the receipt names against the serving module (semantic file, code list, card
    tables, window, guard and step-limit laws, core), never the module binary; both module digests are recorded."""

    def __init__(self, native, checkpoint, weights, *, semantic_file, code_list, card_tables, announce_tables,
                 dormant_table, batch_sizes=(), constant_size=None, compute_dtype=None, debiased_request=None,
                 inference_geometry=None, replay_belief=False, ar_features=False, ar_head=None,
                 ar_request_clock=None, specialization_request=None):
        if specialization_request is not None and (weights!='iterate' or debiased_request is not None or
                replay_belief or ar_features or ar_head is not None):
            raise ValueError('specialization requires a separate raw behavior-only actor without an old AR head')
        from mirrorforce.netduel.ar_clock import from_head
        ar_request_clock = None if ar_request_clock is None else from_head({"request_clock": ar_request_clock})
        if ar_request_clock is not None and ar_head is None:
            raise ValueError("an explicit AR request clock requires a pinned runtime head")
        if type(ar_features) is not bool or ar_features and not replay_belief:
            raise ValueError("AR feature capture requires explicit same-forward replay belief")
        self.ar_features = ar_features
        if ar_head is not None and (not ar_features or weights != 'iterate'):
            raise ValueError("AR head requires opt-in public features from its raw iterate actor")
        if type(replay_belief) is not bool or replay_belief and inference_geometry is None:
            raise ValueError("replay belief requires the explicit shared public inference geometry")
        self.replay_belief = replay_belief
        self.constant_size = constant_size
        if constant_size is not None:
            if type(constant_size) is not int or not 1 <= constant_size <= 256 or batch_sizes:
                raise ValueError("constant batch size is 1..256 and mutually exclusive with search buckets")
        geometry = None
        if inference_geometry is not None:
            from mirrorforce.agent.inference_geometry import validate_geometry
            geometry = validate_geometry(inference_geometry)
            if constant_size != geometry["batch_size"] or compute_dtype is not None:
                raise ValueError("fixed public geometry requires constant B64 and the checkpoint's BF16 precision")
        if replay_belief:
            self.code_by_id = {i + 1: int(row.split()[0]) for i, row in enumerate(Path(code_list).read_text().splitlines())}
        if (weights == "debiased_ema") != (debiased_request is not None):
            raise ValueError("debiased_ema weights require their explicit request; ordinary weights refuse it")
        from mirrorforce.agent.train import policy_io
        if debiased_request is None:
            agent, variables, receipt = policy_io.load_policy(
                str(checkpoint), weights, semantic_file=str(semantic_file), code_list_file=str(code_list),
                card_tables_file=None if card_tables is None else str(card_tables), native=native,
                compute_dtype=compute_dtype)
        else:
            from mirrorforce.agent.train import debiased_policy
            agent, variables, receipt, debiased_identity = debiased_policy.load_policy(
                str(checkpoint), debiased_request, semantic_file=str(semantic_file), code_list_file=str(code_list),
                card_tables_file=None if card_tables is None else str(card_tables), native=native,
                compute_dtype=compute_dtype)
        specialization_identity=None
        if specialization_request is not None:
            from mirrorforce.agent.train.specialization_policy import load_actor
            variables,specialization_identity=load_actor(checkpoint,specialization_request,variables,
                semantic_file=semantic_file,code_list_file=code_list,card_tables_file=card_tables)
        policy_io.native_setup(native, receipt, announce_tables=str(announce_tables),
                               dormant_table=None if dormant_table is None else str(dormant_table))
        precision = "highest" if compute_dtype == "float32" else None
        self.policy, self.receipt = policy_io.Policy(agent, variables, receipt, matmul_precision=precision,
            **({"inference_geometry": geometry} if geometry is not None else {})), receipt
        if geometry is not None and self.policy.inference_geometry != geometry:
            raise ValueError("the policy did not retain its registered inference geometry")
        self.batch_sizes = tuple(batch_sizes)
        if self.batch_sizes:
            search_batch_sizes(",".join(map(str, self.batch_sizes)))
        # The search's magnet is the checkpoint's own training magnet (AtaraxosConfig.magnet in the receipt).
        self.magnet = receipt["recipe"].get("ataraxos", {}).get("magnet")
        if self.magnet not in MAGNETS:
            raise ValueError(f"the checkpoint's magnet {self.magnet!r} is not one the search implements")
        # The default serves the checkpoint's recorded computation. The explicit diagnostic/serving override is
        # separately identified: changing float32 compute does not rewrite the checkpoint or training precision.
        recorded = receipt["model"]["args"].get("dtype")
        effective = recorded if compute_dtype is None else compute_dtype
        if getattr(agent.config, "dtype", None) != effective:
            raise ValueError(f"the loaded network computes in {getattr(agent.config, 'dtype', None)}, "
                             f"the declared inference compute dtype is {effective}")
        if geometry is not None and (effective != geometry["compute_dtype"]
                                     or self.policy.max_options != geometry["menu_rows"]):
            raise ValueError("the loaded policy differs from the fixed geometry's BF16/full192 contract")
        self.identity = {"backend": "checkpoint", "checkpoint_sha256": file_sha256(checkpoint), "weights": weights,
                         "receipt_sha256": hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest(),
                         "training_native_sha256": receipt.get("identity", receipt).get("native_sha256"),
                         "compute_dtype": effective or "float32", "magnet": self.magnet,
                         "semantic_file_sha256": file_sha256(semantic_file),
                         "card_tables_sha256": None if card_tables is None else file_sha256(card_tables)}
        if specialization_identity is not None:
            self.identity.update(backend='specialization',
                checkpoint_sha256=specialization_identity['checkpoint_sha256'],
                parent_checkpoint_sha256=specialization_identity['parent_checkpoint_sha256'],
                specialization=specialization_identity)
        if compute_dtype is not None:
            self.identity["inference_precision"] = {"law": "explicit-inference-dtype/v1",
                "training_dtype": recorded or "float32", "compute_dtype": effective,
                "matmul_precision": precision, "checkpoint_parameters_unchanged": True}
        if self.batch_sizes:
            self.identity["search_batching"] = {"law": "valid-row-padding/v1", "sizes": list(self.batch_sizes)}
        if self.constant_size is not None:
            self.identity["search_batching"] = {"law": "constant-batch-full-menu/v1",
                "size": self.constant_size, "menu_rows": self.policy.max_options, "applies_to": "all_forwards"}
        if geometry is not None:
            self.identity["inference_geometry"] = dict(geometry)
        if replay_belief:
            from mirrorforce.netduel.replay_filtered_particles import HEAD_LAW
            self.identity["replay_belief"] = {"law": HEAD_LAW, "locations": [1, 2, 3, 4, 6, 7],
                                            "counts": 4, "source": "same-public-forward"}
        if ar_features:
            from mirrorforce.agent.belief_feature_cache import SCHEMA as FEATURE_SCHEMA
            self.identity["ar_features"] = {"schema": FEATURE_SCHEMA, "source": "same-public-forward",
                                             "search_admission": False}
        if ar_head is not None:
            from mirrorforce.agent.belief_ar_inference import ARHeadRuntime, REQUEST_SCHEMA
            if type(ar_head) is not dict or set(ar_head) != {'schema', 'checkpoint', 'contract_sha256',
                                                           'parent_actor_sha256', 'diagnostic_only'} \
                    or ar_head['schema'] != REQUEST_SCHEMA or ar_head['diagnostic_only'] is not True \
                    or ar_head['parent_actor_sha256'] != self.identity['checkpoint_sha256']:
                raise ValueError("AR head request does not bind this exact raw actor")
            self.ar_runtime = ARHeadRuntime(ar_head['checkpoint'], parent_actor_sha256=ar_head['parent_actor_sha256'],
                                            contract_sha256=ar_head['contract_sha256'], code_by_id=self.code_by_id,
                                            **({"request_clock": ar_request_clock} if ar_request_clock is not None else {}))
            self.identity['ar_head'] = self.ar_runtime.identity
            self.identity['ar_head_warmup'] = self.ar_runtime.warmup()
        if debiased_request is not None:
            if compute_dtype is not None:
                self.identity["inference_precision"].update(
                    checkpoint_parameters_unchanged=False, checkpoint_artifact_unchanged=True)
            self.identity.update(debiased_policy.service_identity(debiased_identity, debiased_request))

    def client_config(self):
        """The env's own configuration from the receipt: the trainer's arguments and its room format."""
        from mirrorforce.agent.env import room_format
        args = self.receipt["config"]
        identity = self.receipt.get("identity", self.receipt).get("room_format")
        public = self.receipt["recipe"].get("public_opponent_recipe", False)
        if type(public) is not bool or args.get("public_opponent_recipe", False) != public:
            raise ValueError("the checkpoint's public opponent recipe configuration is inconsistent")
        return {"max_options": int(args["max_options"]), "max_steps": int(args["max_steps"]),
                "n_history_actions": int(args["n_history_actions"]), "public_opponent_recipe": public,
                **room_format.env_config(identity)}

    def initial_state(self):
        return self.policy.initial_state()

    def warm_up(self, native, config, deck):
        """Compile the network for both menu blocks before serving (a first compile takes minutes on a GPU, longer
        than a client waits for a decision): one real observation of a mirror game of ``deck``."""
        import jax
        import jax.numpy as jnp
        from mirrorforce.agent.train.cleanba import MENU_BLOCK, sliced_obs
        from mirrorforce.worldmodel.engine import load_ydk
        from tools.mf_runtime_client_parity import mirror_deal
        duel = native.ScriptedDuel(mirror_deal(load_ydk(deck), 0), dict(config))
        duel.start()
        if duel.prompt() is None:
            raise RuntimeError("the warm-up game has no decision")
        obs = {k[len("obs:"):]: np.asarray(v)[None] for k, v in duel.observation().items() if k.startswith("obs:")}
        started = time.perf_counter()
        constant = getattr(self, "constant_size", None)
        sizes = (constant,) if constant is not None else self.batch_sizes or (1,)
        menus = (self.policy.max_options,) if constant is not None else (MENU_BLOCK, self.policy.max_options)
        for size in sizes:
            batch_obs = {key: np.repeat(value, size, axis=0) for key, value in obs.items()}
            state = jax.tree_util.tree_map(lambda value: jnp.repeat(value, size, axis=0), self.initial_state())
            for menu in menus:
                out = self.policy._apply(self.policy.variables, sliced_obs(batch_obs, menu), state,
                                         jnp.ones((size,), dtype=bool))
                np.asarray(out[1])
        return round(time.perf_counter() - started, 1)

    def act(self, obs, state, first, rows):
        if getattr(self, "constant_size", None) is not None:
            return self.act_batch([(obs, state, first, rows)])[0]
        state, logits, value, wdl = self.policy.act(obs, state, first)
        logits = np.asarray(logits, np.float64).reshape(-1)
        if logits.shape != (rows,):
            raise RuntimeError(f"the policy returned {logits.shape[0]} logits for a menu of {rows}")
        wdl = np.asarray(wdl, np.float64).reshape(-1)[:3]
        p = np.exp(wdl - wdl.max())
        return state, logits, float(np.asarray(value).reshape(-1)[0]), [float(x) for x in p / p.sum()]

    def act_batch(self, items):
        """[(obs, state, first, rows)] -> [(state, logits, value, wdl)]: one network call through
        ``Policy.act_batch`` when the loader has it, else one ``act`` per item (the same outputs, slower)."""
        return self._act_batch(items, belief=False)

    def act_batch_belief(self, items):
        if not getattr(self, "replay_belief", False):
            raise ValueError("count belief capture was not explicitly enabled")
        return self._act_batch(items, belief=True)

    def _act_batch(self, items, *, belief):
        if belief and any(str(key).startswith(("priv", "label")) for item in items for key in item[0]):
            raise ValueError("same-forward belief capture accepts only public inputs")
        constant = getattr(self, "constant_size", None)
        batch = getattr(self.policy, "act_batch", None)
        if batch is None:
            if belief:
                raise RuntimeError("same-forward belief capture requires Policy.act_batch_features")
            if self.batch_sizes or constant is not None:
                raise RuntimeError("fixed search batches require Policy.act_batch")
            return [self.act(*item) for item in items]
        out = []
        sizes = (constant,) if constant is not None else self.batch_sizes
        for count, padded in padded_batches(items, sizes):
            options = {"full_menu": True} if constant is not None else {}
            if belief:
                result = self.policy.act_batch_features([i[0] for i in padded], [i[1] for i in padded],
                                                        [i[2] for i in padded])
            else:
                result = batch([i[0] for i in padded], [i[1] for i in padded], [i[2] for i in padded], **options)
            if len(result) != len(padded):
                raise RuntimeError("the policy returned a different batch size")
            for (obs, state, first, rows), output in zip(padded[:count], result[:count]):
                state2, logits, value, wdl = output[:4]
                logits = np.asarray(logits, np.float64).reshape(-1)
                if logits.shape != (rows,):
                    raise RuntimeError(f"the policy returned {logits.shape[0]} logits for a menu of {rows}")
                wdl = np.asarray(wdl, np.float64).reshape(-1)[:3]
                p = np.exp(wdl - wdl.max())
                row = (state2, logits, float(np.asarray(value).reshape(-1)[0]), [float(x) for x in p / p.sum()])
                if belief:
                    candidates = np.asarray(obs["obs:candidates_"])
                    if len(output) != 7:
                        raise ValueError("same-forward count head output must have exactly seven fields")
                    logits_count, valid = np.asarray(output[5], np.float64), np.asarray(output[6])
                    if candidates.ndim != 2 or candidates.shape[1] != 3 or candidates.dtype.kind not in "iu" \
                            or logits_count.shape != (len(candidates), 6, 4) or not np.isfinite(logits_count).all() \
                            or valid.shape != (len(candidates),) or valid.dtype.kind != "b" \
                            or np.any(candidates[:, :2] < 0) or np.any(candidates[:, :2] > 255) \
                            or not np.array_equal(valid, candidates[:, 2] > 0):
                        raise ValueError("same-forward count head differs from its public candidate rows")
                    ids = candidates[valid, 0].astype(np.int64) * 256 + candidates[valid, 1]
                    if len(set(map(int, ids))) != len(ids) or any(int(i) not in self.code_by_id for i in ids):
                        raise ValueError("belief candidates are not distinct IDs in the pinned public code list")
                    payload = {"codes": [self.code_by_id[int(i)] for i in ids],
                               "locations": [1, 2, 3, 4, 6, 7], "logits": logits_count[valid].tolist()}
                    row += (payload,)
                    if getattr(self, "ar_features", False):
                        from mirrorforce.agent.belief_feature_cache import PendingPublicFeatures
                        row += (PendingPublicFeatures(output[4], candidates,
                                                     obs_sha256=observation_sha256(obs)),)
                out.append(row)
        return out


class Session:
    def __init__(self, client, backend, seed):
        self.client, self.backend = client, backend
        self.state, self.first = backend.initial_state(), True
        self.rng = np.random.default_rng(seed)
        self.decisions = self.forced = self.prompts = 0
        self.act_ms = []
        self.pending = None  # a scored decision not yet stepped (search roots): its row count and outputs
        self.lock = threading.RLock()
        self.owner = None
        self.current_root_lease = None

    def clone(self, seed):
        """An independent copy at this point (the builder's ``clone()``; the recurrent state is immutable arrays)."""
        twin = Session.__new__(Session)
        twin.client, twin.backend = self.client.clone(), self.backend
        twin.state, twin.first, twin.pending = self.state, self.first, self.pending
        twin.rng = np.random.default_rng(seed)
        twin.decisions = twin.forced = twin.prompts = 0
        twin.act_ms = []
        twin.lock = threading.RLock()
        twin.owner = None
        twin.current_root_lease = self.current_root_lease
        return twin


from mirrorforce.netduel.current_root_protocol import (  # shared wire identities, no model/engine imports
    MEMORY_LAW as CURRENT_ROOT_MEMORY_LAW, CAPABILITY as CURRENT_ROOT_CAPABILITY)


class CurrentRootLease:
    """An internal capability tied to the exact pending object, never accepted over the wire."""
    def __init__(self, name, parent, view):
        self.name, self.parent, self.pending = name, parent, parent.pending
        self.root_id, self.root_hash = view["root_id"], view["root_hash"]
        self.hypothesis_hash = view["hypothesis_hash"]


class ConnectionOwner:
    """An in-process capability, never supplied by or serialized to an RPC client."""
    def __init__(self):
        self.closed = False


def session_transaction(method):
    """Serialize complete operations per session, with one global order for batch locks.

    Never hold the registry lock while waiting for a session. Close/re-register
    is rechecked after acquiring every session, so a queued read cannot revive
    a closed object. Direct method calls and socket dispatch use the same guard.
    """
    @wraps(method)
    def run(service, request):
        names = [item["session"] for item in request["items"]] if method.__name__ == "rollout_step" \
            else [request["session"]]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("an operation needs distinct registered session names")
        with service.lock:
            if any(name not in service.sessions for name in names):
                raise ValueError("the requested session is closed or unknown")
            requested = [(name, service.sessions[name]) for name in names]
            leases = [session.current_root_lease for _, session in requested
                      if session.current_root_lease is not None] if method.__name__ != "close" else []
            for lease in leases:
                if service.sessions.get(lease.name) is not lease.parent:
                    raise ValueError("the current-root parent session is closed or replaced")
            sessions = sorted(dict(requested + [(lease.name, lease.parent) for lease in leases]).items())
            owner = getattr(service._dispatch_owner, "current", None)
            if owner is not None and (owner.closed or any(session.owner is not owner for _, session in sessions)):
                raise ValueError("a session does not belong to this client connection")
        with ExitStack() as locks:
            for _, session in sessions:
                locks.enter_context(session.lock)
            with service.lock:
                if any(service.sessions.get(name) is not session for name, session in sessions):
                    raise ValueError("the requested session closed while the operation was queued")
                if any(lease.parent.pending is not lease.pending for lease in leases):
                    raise ValueError("the current-root pending decision has ended")
            return method(service, request)
    return run


class Service:
    def __init__(self, native, backend, config, selection, temperature, identity, client_factory=None):
        from mirrorforce.probes.client_parity import client_duel
        self.behavior_audit = "behavior_finite_audit" in identity
        if self.behavior_audit:
            from mirrorforce.netduel.ema_behavior import FINITE_IDENTITY, FiniteAuditBackend
            if identity["behavior_finite_audit"] != FINITE_IDENTITY:
                raise ValueError("unknown behavior finite-audit law")
            backend = FiniteAuditBackend(backend)
        self.native, self.backend, self.config = native, backend, config
        self.client_factory = client_factory or client_duel
        self.selection, self.temperature, self.identity = selection, temperature, identity
        self.opponent_recipe_mode = PR.mode({**identity, "client_config": config})
        if "current_root" in identity and (identity["current_root"] != CURRENT_ROOT_CAPABILITY
                or self.behavior_audit or self.opponent_recipe_mode != "mirror"):
            raise ValueError("current-root search requires its explicit capability and public mirror recipe")
        from mirrorforce.netduel.ar_clock import from_head
        if "ar_head" in identity and from_head(identity["ar_head"]) is not None:
            if identity.get("current_root") != CURRENT_ROOT_CAPABILITY \
                    or identity["ar_head"] != getattr(getattr(backend, "ar_runtime", None), "identity", None):
                raise ValueError("AR request clock must bind the exact current-root runtime head")
        self.sessions, self.lock, self.counter = {}, threading.Lock(), 0
        self._dispatch_owner = threading.local()

    def _new_session(self, request: dict) -> Session:
        if self.identity.get("weights") == "debiased_ema":
            from mirrorforce.agent.train.debiased_policy import check_session
            check_session(self.identity, request)
        seat, main, extra = request["seat"], request["main"], request["extra"]
        if seat not in (0, 1) or not main:
            raise ValueError("a session needs its seat and submitted deck")
        declared = request.get("public_opponent_recipe")
        public_args = {}
        if request.get("opponent_recipe_mode") != self.opponent_recipe_mode:
            raise ValueError("the requested opponent recipe mode differs from the service")
        expected = PR.for_game({**self.identity, "client_config": self.config}, main, extra)
        if expected is not None:
            declared = PR.checked(declared)
            if declared != expected:
                raise ValueError("the requested public opponent recipe differs from the service declaration")
            public_args = {"opponent_main": declared["main"], "opponent_extra": declared["extra"]}
        elif declared is not None:
            raise ValueError("opponent recipe supplied in closed-decklist mode")
        client = self.client_factory(self.native, int(seat), [int(c) for c in main], [int(c) for c in extra],
                                     dict(self.config), **public_args)
        session = Session(client, self.backend, int(request.get("seed", 0)))
        if self.behavior_audit:
            from mirrorforce.netduel.ema_behavior import finite_memory
            session.behavior_initial_memory = finite_memory(session.state)
            session.behavior_forward_checks = 0
        return session

    def _behavior_forward_proof(self, session, state):
        """Called only after the opt-in wrapper checked all outputs, before state commit."""
        from mirrorforce.netduel.ema_behavior import FINITE_LAW, finite_memory
        checksum = finite_memory(state)
        session.behavior_forward_checks += 1
        return {"law": FINITE_LAW, "passed": True, "memory_sha256": checksum,
                "forward_number": session.behavior_forward_checks}

    def _register(self, session: Session) -> dict:
        with self.lock:
            owner = getattr(self._dispatch_owner, "current", None)
            if owner is not None and owner.closed:
                raise ValueError("the client connection closed before session registration")
            self.counter += 1
            name = f"s{self.counter}"
            session.owner = owner
            self.sessions[name] = session
        return {"session": name}

    def close_owned(self, owner):
        """Remove only this disconnected connection's sessions, including lost replies.

        Detach under the registry lock FIRST, then release it before waiting for
        per-session locks in the same sorted order as rollout. Existing queued
        operations recheck registration and fail; no new session can register
        to a closed owner. Dropped clients are freed only after native work ends.
        """
        if not isinstance(owner, ConnectionOwner):
            raise ValueError("connection cleanup needs its private owner capability")
        with self.lock:
            owner.closed = True
            dropped = sorted((name, session) for name, session in self.sessions.items() if session.owner is owner)
            for name, _ in dropped:
                del self.sessions[name]
        with ExitStack() as locks:
            for _, session in dropped:
                locks.enter_context(session.lock)
        return len(dropped)

    def open(self, request):
        return self._register(self._new_session(request))

    @session_transaction
    def open_root_view(self, request):
        """Seed only the opponent's current view; never replay or borrow live private memory.

        The front-end owns and masks the hypothetical engine. This boundary
        validates the closed view and pins it to this connection's live root;
        it does not claim to authenticate a remote engine's hidden state.
        """
        from mirrorforce.netduel.current_root_view import CurrentRootView
        from mirrorforce.netduel.agent_client import AgentClientDuel
        if set(request) != {"op", "session", "expected_obs_sha256", "view", "seed", "memory_law"} \
                or request["op"] != "open_root_view" \
                or self.identity.get("current_root") != CURRENT_ROOT_CAPABILITY \
                or request["memory_law"] != CURRENT_ROOT_MEMORY_LAW \
                or type(request["seed"]) is not int or not 0 <= request["seed"] < 2**63:
            raise ValueError("current-root sessions require an explicit owned-root request and memory law")
        parent = self.sessions[request["session"]]
        if parent.current_root_lease is not None or parent.pending is None \
                or request["expected_obs_sha256"] != parent.pending["obs_sha256"]:
            raise ValueError("current-root view does not bind a live real-session pending observation")
        view = CurrentRootView.from_dict(request["view"]).to_dict()
        if view["viewer"] != 1 - parent.client.seat \
                or view["main"] != sorted(parent.client.main) or view["extra"] != sorted(parent.client.extra) \
                or view["opponent_main"] != sorted(parent.client.main) \
                or view["opponent_extra"] != sorted(parent.client.extra):
            raise ValueError("current-root opponent seat or public mirror recipes differ from the bound parent")
        client = AgentClientDuel.from_root_view(self.native, view, dict(self.config))
        if client.prompt() is not None:
            raise ValueError("a current-root seed must not consume the real pending prompt")
        session = Session(client, self.backend, request["seed"])
        session.current_root_lease = CurrentRootLease(request["session"], parent, view)
        result = {"schema": CURRENT_ROOT_CAPABILITY["schema"], "memory_law": CURRENT_ROOT_MEMORY_LAW,
                  "parent_session": request["session"], "obs_sha256": parent.pending["obs_sha256"],
                  "root_id": view["root_id"], "root_hash": view["root_hash"],
                  "hypothesis_hash": view["hypothesis_hash"], "viewer": view["viewer"],
                  "first": session.first, "memory_sha256": memory_sha256(session.state),
                  "decisions": 0, "forced": 0, "prompts": 0}
        return {**self._register(session), **result}

    @session_transaction
    def current_public_root_seed(self, request):
        """Read only the current common-public facts; never export a full private tracker."""
        from mirrorforce.netduel.current_root_protocol import PUBLIC_SEED_RPC_SCHEMA
        from mirrorforce.netduel.current_root_export import check_seed_reply
        if set(request) != {"op", "session", "expected_obs_sha256"} \
                or request["op"] != "current_public_root_seed" \
                or self.identity.get("current_root") != CURRENT_ROOT_CAPABILITY:
            raise ValueError("current public facts require the explicit owned pending capability")
        session = self.sessions[request["session"]]
        if session.current_root_lease is not None or session.pending is None \
                or request["expected_obs_sha256"] != session.pending["obs_sha256"]:
            raise ValueError("current public facts do not bind a real-session pending observation")
        seed = session.client.current_public_root_seed()
        checksum = hashlib.sha256(json.dumps(seed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        reply = {"schema": PUBLIC_SEED_RPC_SCHEMA, "session": request["session"],
                 "obs_sha256": session.pending["obs_sha256"], "seed": seed, "seed_sha256": checksum}
        check_seed_reply(reply, session=request["session"], obs_sha256=request["expected_obs_sha256"],
                         source_viewer=session.client.seat)
        return reply

    def open_stream(self, request: dict) -> dict:
        """Rebuild a NEW client's memory from synthetic packets and known historical responses.

        Decode each non-forced response on a builder clone, then score and step every sub-decision exactly once.
        No policy sampling, live-session copy, or one-shot step_response that skips memory. Register only on success;
        malformed replay never leaves a partial session or changes existing sessions (including their RNGs).
        """
        stream_request = request
        if self.identity.get("weights") == "debiased_ema":
            from mirrorforce.agent.train.debiased_policy import check_session
            check_session(self.identity, request)
            stream_request = {key: value for key, value in request.items()
                              if key not in ("purpose", "debiased_ema_identity_sha256")}
        frames, trailing = checked_stream(stream_request)
        session = self._new_session(request)
        trace = []

        def feed(messages):
            response = None
            for msg, payload in messages:
                if response is not None or session.client.prompt() is not None:
                    raise ValueError("a replay frame continued after a prompt")
                response = session.client.feed(msg, payload)
            return None if response is None else bytes(response)

        for frame_index, (messages, expected) in enumerate(frames):
            session.prompts += 1
            forced = feed(messages)
            if forced is not None:
                if forced != expected or session.client.prompt() is not None:
                    raise ValueError("historical forced response differs from the builder")
                session.forced += 1
                continue
            if session.client.prompt() is None:
                raise ValueError("a historical frame has no prompt")
            path = session.client.response_path(expected)
            if not path or len(path) > 4096:
                raise ValueError("a response needs a nonempty bounded decision path")
            for subdecision, index in enumerate(path):
                prompt = session.client.prompt()
                if prompt is None or type(index) is not int or not 0 <= index < len(prompt[2]):
                    raise ValueError("historical response path is outside the current menu")
                before = memory_sha256(session.state)
                pending = self._score([session])[0]
                trace.append({"frame": frame_index, "subdecision": subdecision, "row": index,
                              "msg": pending["msg"], "obs_sha256": pending["obs_sha256"],
                              "memory_before_sha256": before, "memory_after_sha256": memory_sha256(session.state)})
                session.pending = None
                session.decisions += 1
                response = session.client.step(index)
                if subdecision + 1 == len(path):
                    if response is None or bytes(response) != expected or session.client.prompt() is not None:
                        raise ValueError("historical response path did not produce the recorded bytes")
                elif response is not None:
                    raise ValueError("historical response path answered before its last row")
        if feed(trailing) is not None or session.client.prompt() is not None:
            raise ValueError("trailing replay messages contain an unanswered prompt")
        digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        result = {"schema": STREAM_SCHEMA, "stream_sha256": digest, "trace": trace,
                  "memory_sha256": memory_sha256(session.state),
                  "decisions": session.decisions, "forced": session.forced, "prompts": session.prompts}
        return {**self._register(session), **result}

    def choose(self, session, logits):
        if self.selection == "greedy":
            return int(np.argmax(logits)), None
        z = logits / self.temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        return int(session.rng.choice(len(p), p=p)), p

    @session_transaction
    def decide(self, request):
        session = self.sessions[request["session"]]
        client, responses, rows_out = session.client, [], []
        for msg, payload in request["messages"]:
            out = client.feed(int(msg), bytes.fromhex(payload))
            if out is not None:
                responses.append(bytes(out))
        session.prompts += 1
        if len(responses) > 1:
            raise RuntimeError(f"the builder answered {len(responses)} prompts in one batch")
        if responses:
            if client.prompt() is not None:
                raise RuntimeError("the builder both answered a prompt and asked for a decision")
            session.forced += 1
            return {"response": responses[0].hex(), "forced": True,
                    "decisions": [{"forced": True, "msg": int(request["messages"][-1][0])}]}
        for _ in range(4096):
            prompt = client.prompt()
            if prompt is None:
                raise RuntimeError("a prompt with neither a decision nor an answer from the builder")
            player, msg, rows = prompt
            obs = client.observation()
            started = time.perf_counter()
            audit = None
            if self.behavior_audit:
                state, logits, value, wdl = self.backend.act(obs, session.state, session.first, len(rows))
                if logits.shape != (len(rows),):
                    raise RuntimeError("the policy returned logits that do not match the menu")
                audit = self._behavior_forward_proof(session, state)
                session.state = state
            else:
                session.state, logits, value, wdl = self.backend.act(obs, session.state, session.first, len(rows))
            act_ms = (time.perf_counter() - started) * 1000
            session.act_ms.append(act_ms)
            session.first = False
            if logits.shape != (len(rows),) or not np.isfinite(logits).all():
                raise RuntimeError("the policy returned logits that do not match the menu")
            index, p = self.choose(session, logits)
            rows_out.append({"forced": False, "msg": int(msg), "rows": len(rows), "chosen": index,
                             "p_chosen": None if p is None else float(p[index]), "value": value, "wdl": wdl,
                             "logits": [round(float(x), 5) for x in logits], "obs_sha256": observation_sha256(obs),
                             "act_ms": round(act_ms, 2),
                             **({"behavior_finite_audit": audit} if audit is not None else {})})
            session.decisions += 1
            out = client.step(index)
            if out is not None:
                return {"response": bytes(out).hex(), "forced": False, "decisions": rows_out}
        raise RuntimeError("the builder never completed the response")

    # -- search operations (netduel/agent_search_policy.py) ------------------------------------------------------

    def _feed(self, session, messages):
        out = [bytes(r) for r in (session.client.feed(int(m), bytes.fromhex(b)) for m, b in messages) if r is not None]
        if len(out) > 1:
            raise RuntimeError(f"the builder answered {len(out)} prompts in one batch")
        if out and session.client.prompt() is not None:
            raise RuntimeError("the builder both answered a prompt and asked for a decision")
        return out[0] if out else None

    def _score(self, items):
        """Score the pending prompt of each session in one network call; returns per session the row count, the
        logits, value, win/draw/loss and the source-card groups of the rows (each option's source card row + 1)."""
        work = []
        for session in items:
            player, msg, rows = session.client.prompt()
            obs = session.client.observation()
            work.append((session, int(msg), len(rows), obs))
        started = time.perf_counter()
        capture_belief = "replay_belief" in self.identity
        method = self.backend.act_batch_belief if capture_belief else self.backend.act_batch
        results = method([(obs, s.state, s.first, n) for s, _, n, obs in work])
        if len(results) != len(work):
            raise RuntimeError("the policy returned a different number of scored sessions")
        ms = (time.perf_counter() - started) * 1000 / max(len(work), 1)
        scored = []
        for (session, msg, n, obs), output in zip(work, results):
            state, logits, value, wdl = output[:4]
            if logits.shape != (n,) or not np.isfinite(logits).all():
                raise RuntimeError("the policy returned logits that do not match the menu")
            obs_sha256 = observation_sha256(obs)
            if "ar_features" in self.identity:
                from mirrorforce.agent.belief_feature_cache import PendingPublicFeatures
                if not capture_belief or len(output) != 6 or type(output[5]) is not PendingPublicFeatures:
                    raise ValueError("AR pending features require the exact opted-in shared forward output")
                output[5].read(obs_sha256=obs_sha256)
            if self.behavior_audit:
                self._behavior_forward_proof(session, state)
            session.state, session.first = state, False
            session.act_ms.append(ms)
            refs = np.asarray(obs.get("obs:action_single_refs_", np.zeros((n, 1))))
            groups = [int(g) for g in refs[:n, 0]] if refs.ndim == 2 and len(refs) >= n else [0] * n
            session.pending = {"msg": msg, "rows": n, "logits": logits, "value": value, "wdl": wdl, "groups": groups,
                               "obs_sha256": obs_sha256}
            if capture_belief:
                session.pending["count_belief"] = copy.deepcopy(output[4])
            if "ar_features" in self.identity:
                session.pending["ar_features"] = output[5]
            scored.append(session.pending)
        return scored

    @staticmethod
    def _public(pending):
        return {"msg": pending["msg"], "rows": pending["rows"], "logits": [float(x) for x in pending["logits"]],
                "value": pending["value"], "wdl": pending["wdl"], "groups": pending["groups"],
                "obs_sha256": pending["obs_sha256"],
                **({"count_belief_available": True} if "count_belief" in pending else {}),
                **({"ar_features_available": True} if "ar_features" in pending else {})}

    @session_transaction
    def prompt(self, request):
        """Feed a session's new messages; a forced prompt comes back answered, otherwise the decision is scored and
        left pending (``commit`` steps it), so a search can clone the session at its root first."""
        session = self.sessions[request["session"]]
        if session.pending is not None:
            raise RuntimeError("the session's last decision was not committed")
        session.prompts += 1
        response = self._feed(session, request["messages"])
        if response is not None:
            session.forced += 1
            return {"response": response.hex(), "forced": True}
        if session.client.prompt() is None:
            raise RuntimeError("a prompt with neither a decision nor an answer from the builder")
        return {"pending": self._public(self._score([session])[0])}

    @session_transaction
    def commit(self, request):
        """Step the pending decision with ``index``; the response bytes, or the next sub-decision scored and pending."""
        session = self.sessions[request["session"]]
        pending, index = session.pending, int(request["index"])
        if pending is None or not 0 <= index < pending["rows"]:
            raise RuntimeError("no pending decision, or a row outside its menu")
        session.pending = None
        session.decisions += 1
        out = session.client.step(index)
        if out is not None:
            return {"response": bytes(out).hex()}
        return {"pending": self._public(self._score([session])[0])}

    @session_transaction
    def clone(self, request):
        """A copy of a session at its current point (a search branch), with its own sampling seed."""
        session = self.sessions[request["session"]]
        twin = session.clone(int(request["seed"]))
        return self._register(twin)

    @session_transaction
    def public_world(self, request):
        """Read only this session's own client World at its exactly scored pending input.

        Both observation and World are read on an unregistered client clone.
        No model call, history response, RNG draw, prompt feed or live mutation
        occurs. Private target layouts, viewer overrides and host objects are
        not part of this closed request or response protocol.
        """
        if not isinstance(request, dict) or set(request) != {"op", "session", "expected_obs_sha256"} \
                or request["op"] != "public_world":
            raise ValueError("public_world accepts only a session and expected pending observation SHA")
        expected = request["expected_obs_sha256"]
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("public_world needs a lowercase expected observation SHA-256")
        if self.identity.get("client_public_world") != RPC_CAPABILITY \
                or not hasattr(getattr(self.native, "ClientDuel", None), "public_world"):
            raise ValueError("this service/native has no registered public_world capability")
        if self.opponent_recipe_mode not in ("mirror", "known") or self.config.get("public_opponent_recipe") is not True:
            raise ValueError("public_world needs an explicitly declared public opponent recipe")
        session = self.sessions[request["session"]]
        if session.pending is None:
            raise ValueError("public_world requires a scored pending decision")
        if session.pending["obs_sha256"] != expected:
            raise ValueError("public_world expected observation is stale")
        scratch = session.client.clone()
        if not scratch.started or scratch.prompt() is None or scratch.prompt()[0] != scratch.seat:
            raise ValueError("public_world requires this client's own started pending prompt")
        declared = PR.for_game({**self.identity, "client_config": self.config}, scratch.main, scratch.extra)
        if scratch.public_opponent_recipe != declared:
            raise ValueError("public_world client declaration differs from this service")
        if observation_sha256(scratch.observation()) != expected:
            raise ValueError("public_world cloned input differs from the scored pending observation")
        world = encode_world(scratch.public_world(), complete=True)
        if observation_sha256(scratch.observation()) != expected:
            raise ValueError("public_world changed its cloned pending observation")
        return {"schema": RPC_SCHEMA, "session": request["session"], "obs_sha256": expected,
                "world_sha256": world_sha256(world), "world": world}

    @session_transaction
    def public_belief(self, request):
        from mirrorforce.netduel.replay_filtered_particles import HEAD_SCHEMA, HEAD_LAW
        if not isinstance(request, dict) or set(request) != {"op", "session", "expected_obs_sha256"} \
                or request["op"] != "public_belief" or "replay_belief" not in self.identity:
            raise ValueError("public belief requires its explicit owned pending capability")
        expected = request["expected_obs_sha256"]
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("public belief needs a lowercase expected observation SHA-256")
        session = self.sessions[request["session"]]
        pending = session.pending
        if pending is None or pending["obs_sha256"] != expected or "count_belief" not in pending:
            raise ValueError("count belief does not belong to this exact owned pending observation")
        if observation_sha256(session.client.clone().observation()) != expected:
            raise ValueError("count belief input differs from the scored pending observation")
        return {"schema": HEAD_SCHEMA, "law": HEAD_LAW, "session": request["session"],
                "obs_sha256": pending["obs_sha256"], "belief": copy.deepcopy(pending["count_belief"])}

    @session_transaction
    def ar_proposals(self, request):
        from mirrorforce.agent.belief_distribution_eval import DistributionBudgetExceeded
        from mirrorforce.agent.search.belief_current_law import CurrentLawBudgetExceeded
        from mirrorforce.agent.belief_ar_inference import SCHEMA as AR_SCHEMA
        from mirrorforce.agent.search.belief_current_law import SCOPED_SCHEMA
        from mirrorforce.agent.search.belief_hand_scope import PublicHandScope, from_world
        from mirrorforce.netduel.ar_clock import validate
        if type(request) is not dict \
                or request.get('op') != 'ar_proposals' or 'ar_head' not in self.identity \
                or self.identity.get('current_root') != CURRENT_ROOT_CAPABILITY:
            raise ValueError('AR proposals require the explicit diagnostic current-root capability')
        deadline = validate(request, self.identity['ar_head'], now=time.monotonic())
        expected = request['expected_obs_sha256']
        if type(expected) is not str or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected) \
                or type(request['count']) is not int or not 1 <= request['count'] <= 128 \
                or type(request['seed']) is not int or not 0 <= request['seed'] < 2 ** 64:
            raise ValueError('AR proposals need an exact pending SHA and the original bounded same-host search clock')
        session = self.sessions[request['session']]
        pending = session.pending
        if pending is None or pending['obs_sha256'] != expected or 'ar_features' not in pending:
            raise ValueError('AR features do not belong to this exact owned pending observation')
        if observation_sha256(session.client.clone().observation()) != expected:
            raise ValueError('AR feature input differs from the scored pending observation')
        specification = request['specification']
        if type(specification) is not dict or specification.get('schema') != SCOPED_SCHEMA:
            raise ValueError('AR proposals require the explicit v2 same-pending hand scope')
        scope = PublicHandScope(specification.get('hand_scope'))
        scoped = scope.to_dict()
        if scoped['obs_sha256'] != expected:
            raise ValueError('AR public hand scope belongs to another pending observation')
        if time.monotonic() >= deadline:
            return {'schema': AR_SCHEMA + '#reply', 'session': request['session'], 'obs_sha256': expected,
                    'status': 'original_deadline_expired', 'bank': None}
        # Reuse the registered read-only own-client boundary. This copies no
        # private layout and calls no actor, AR forward or recurrent update.
        # Additional anchors remain explicitly caller-ledger facts, not
        # falsely attributed to the native observer.
        public = self.public_world({'op': 'public_world', 'session': request['session'],
                                    'expected_obs_sha256': expected})
        actual = from_world(public['world'], obs_sha256=expected, world_sha256=public['world_sha256'],
                            reference_anchors=scoped['reference_anchors'])
        if actual != scoped:
            raise ValueError('AR hand scope changed the actual same-pending public World/group/counts')
        if time.monotonic() >= deadline:
            return {'schema': AR_SCHEMA + '#reply', 'session': request['session'], 'obs_sha256': expected,
                    'status': 'original_deadline_expired', 'bank': None}
        try:
            bank = self.backend.ar_runtime.draw(pending['ar_features'], specification,
                obs_sha256=expected, count=request['count'], seed=request['seed'], deadline=deadline)
        except (DistributionBudgetExceeded, CurrentLawBudgetExceeded):
            if time.monotonic() < deadline:
                raise  # A node-limit/unknown support is not a measured clock expiry.
            return {'schema': AR_SCHEMA + '#reply', 'session': request['session'], 'obs_sha256': expected,
                    'status': 'original_deadline_expired', 'bank': None}
        if time.monotonic() >= deadline:
            return {'schema': AR_SCHEMA + '#reply', 'session': request['session'], 'obs_sha256': expected,
                    'status': 'original_deadline_expired', 'bank': None}
        return {'schema': AR_SCHEMA + '#reply', 'session': request['session'], 'obs_sha256': expected,
                'status': 'complete', 'bank': bank}

    @session_transaction
    def rollout_step(self, request):
        """One step of many rollout sessions: each item feeds its new messages and, unless the builder answers, picks
        rows by sampling its policy (temperature 1) until the response is complete, every network call batched across
        the items. An item may start with a pending decision (a branch's root action is ``index``). ``stop_after``
        (optional, >= 1) cuts the item at the decision that makes that many scored decisions in this call: it is
        scored, not stepped, and the item returns ``cut`` instead of a response (a continuation's leaf). Returns per
        item the response bytes (or ``cut``) and the win/draw/loss and value at each decision it scored."""
        sessions = [self.sessions[item["session"]] for item in request["items"]]
        limits = [item.get("stop_after") for item in request["items"]]
        if any(limit is not None and (type(limit) is not int or limit < 1) for limit in limits):
            raise ValueError("stop_after is a positive decision count")
        out = [{"decisions": []} for _ in sessions]
        active = []
        for k, (session, item) in enumerate(zip(sessions, request["items"])):
            if "index" in item:
                if session.pending is None:
                    raise RuntimeError("a root action for a session without a pending decision")
                active.append((k, int(item["index"])))
                continue
            if session.pending is not None:
                raise RuntimeError("a rollout session has an uncommitted decision")
            response = self._feed(session, item["messages"])
            if response is not None:
                out[k].update(response=response.hex(), forced=True)
            elif session.client.prompt() is None:
                raise RuntimeError("a prompt with neither a decision nor an answer from the builder")
            else:
                active.append((k, None))
        to_score = [k for k, index in active if index is None]
        for _ in range(4096):
            if to_score:
                for k, pending in zip(to_score, self._score([sessions[k] for k in to_score])):
                    out[k]["decisions"].append({"msg": pending["msg"], "rows": pending["rows"],
                                                "value": pending["value"], "wdl": pending["wdl"]})
            to_score, still = [], []
            for k, index in active:
                session = sessions[k]
                pending = session.pending
                if index is None and limits[k] is not None and len(out[k]["decisions"]) >= limits[k]:
                    out[k]["cut"] = True  # the leaf: scored, left pending, never stepped
                    continue
                if index is None:
                    z = pending["logits"] - pending["logits"].max()
                    p = np.exp(z) / np.exp(z).sum()
                    index = int(session.rng.choice(len(p), p=p))
                session.pending = None
                session.decisions += 1
                response = session.client.step(index)
                if response is not None:
                    out[k].update(response=bytes(response).hex(), forced=False)
                else:
                    to_score.append(k)
                    still.append((k, None))
            active = still
            if not active:
                return {"items": out}
        raise RuntimeError("a rollout response never completed")

    def root_step(self, request):
        """The root's mirror-descent step (``agent/search/update.search_policy``) under the checkpoint's own training
        magnet (``uniform_legal``, or the trainer's ``magnet_log_probs``): stepsize 1/beta, temperature alpha. Rows without a value (``q``
        null: not rolled out) keep their prior probabilities; the step runs over the sampled rows and their total
        prior mass is kept."""
        from mirrorforce.agent.search.update import search_policy
        logits = np.asarray(request["logits"], np.float64)
        groups = np.asarray(request["groups"], np.int32)
        q = request["q"]
        alpha, beta = float(request["alpha"]), float(request["beta"])
        if len(q) != len(logits) or len(groups) != len(logits) or not alpha > 0 or not beta > 0:
            raise ValueError("one value, logit and group per row, positive alpha and beta")
        sampled = np.asarray([v is not None for v in q])
        if not sampled.any():
            raise ValueError("a root step needs a rolled-out row")
        magnet = self.backend.magnet
        if request.get("magnet", magnet) != magnet:
            raise ValueError(f"the search asks for magnet {request['magnet']!r}; the checkpoint trained with {magnet!r}")
        log_magnet = root_log_magnet(magnet, groups)
        z = logits - logits.max()
        prior = np.exp(z) / np.exp(z).sum()
        values = np.asarray([v if v is not None else 0.0 for v in q], np.float64)
        p = prior.copy()
        sub = search_policy(values[sampled], logits[sampled], 1.0 / beta, alpha, log_magnet=log_magnet[sampled])
        p[sampled] = prior[sampled].sum() * sub
        return {"policy": [float(x) for x in p], "prior": [float(x) for x in prior], "magnet": magnet, "log_magnet":
                [float(x) for x in log_magnet], "sampled": int(sampled.sum()), "rows": len(logits)}

    @session_transaction
    def close(self, request):
        """End a game: the messages after its last prompt (the end of the duel) are fed too; none may need an answer."""
        with self.lock:
            session = self.sessions.pop(request["session"])
        for msg, payload in request.get("messages", ()):
            if session.client.feed(int(msg), bytes.fromhex(payload)) is not None or session.client.prompt() is not None:
                raise RuntimeError("a message after the game's last prompt asked for an answer")
        ms = np.asarray(session.act_ms or [0.0])
        audit = None
        if self.behavior_audit:
            from mirrorforce.netduel.ema_behavior import FINITE_LAW, finite_memory
            audit = {"law": FINITE_LAW, "passed": True, "initial_state_checks": 1,
                     "initial_memory_sha256": session.behavior_initial_memory,
                     "forward_count": session.behavior_forward_checks,
                     "memory_sha256": finite_memory(session.state)}
        return {"decisions": session.decisions, "forced": session.forced, "prompts": session.prompts,
                "builder_forced": int(session.client.forced_count()),
                "act_ms": {"p50": round(float(np.percentile(ms, 50)), 2), "p90": round(float(np.percentile(ms, 90)), 2),
                           "max": round(float(ms.max()), 2), "first": round(float(ms[0]), 2)},
                **({"behavior_finite_audit": audit} if audit is not None else {})}

    def dispatch(self, request, *, owner=None):
        previous = getattr(self._dispatch_owner, "current", None)
        self._dispatch_owner.current = owner
        try:
            op = request.get("op")
            if self.behavior_audit and op not in ("identity", "open", "open_stream", "decide", "close"):
                raise ValueError("the behavior audit service admits only fresh/rebuilt public-client games, not search")
            if op == "identity":
                return self.identity
            if op in ("open", "open_stream", "decide", "close", "prompt", "commit", "clone", "rollout_step", "root_step",
                      "public_world", "public_belief", "ar_proposals", "open_root_view", "current_public_root_seed"):
                return getattr(self, op)(request)
            raise ValueError(f"unknown op {op!r}")
        finally:
            self._dispatch_owner.current = previous


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        owner = ConnectionOwner()
        try:
            while True:
                try:
                    request = W.recv(self.connection)
                except (W.WireError, OSError):
                    return
                try:
                    reply = self.server.service.dispatch(request, owner=owner)
                except Exception as exc:  # noqa: BLE001 - the client raises on the reported error
                    reply = {"error": f"{type(exc).__name__}: {exc}"}
                try:
                    W.send(self.connection, reply)
                except (W.WireError, OSError):
                    return
        finally:
            dropped = self.server.service.close_owned(owner)
            if dropped:
                print("CONNECTION-CLEANUP " + json.dumps({"sessions": dropped, "reason": "transport-ended"}), flush=True)


class UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def load_native(args):
    from mirrorforce.agent.env.duel import native, native_build
    here = os.getcwd()
    os.chdir(args.script_root)  # init_module reads ./script
    try:
        native.init_module(str(args.cards_db), str(args.code_list), {})
    finally:
        os.chdir(here)
    if args.backend == "uniform":  # the checkpoint backend registers the receipt's laws (policy_io.native_setup)
        from mirrorforce.agent.env.announce_law import register
        register(native, args.announce_tables, args.announce_cap)
    return native, native_build


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument('--specialization-request',type=Path,
                        help='explicit validated human-acceptance actor branch; not an A0 resume checkpoint')
    parser.add_argument('--specialization-request-sha256')
    parser.add_argument('--specialization-search-request',type=Path,
                        help='independent SHA-pinned diagnostic public-uniform search binding for a specialization')
    parser.add_argument('--specialization-search-request-sha256')
    parser.add_argument("--weights", choices=("ema", "iterate", "debiased_ema"), default="ema")
    parser.add_argument("--debiased-ema-request", type=Path,
                        help="explicit behavior-only debiasing request; requires --weights debiased_ema")
    parser.add_argument("--debiased-ema-request-sha256", help="SHA-256 of the explicit debiasing request")
    parser.add_argument("--compute-dtype", choices=("float32",),
                        help="explicit inference-only float32/highest-matmul override; checkpoint/train unchanged")
    parser.add_argument("--inference-geometry", choices=("fixed-public-b64",),
                        help="explicit shared public-feature BF16 B64/full-menu/full-history inference; default unchanged")
    parser.add_argument("--replay-belief", action="store_true",
                        help="cache the same public forward's count head per owned pending session for replay proposals")
    parser.add_argument("--ar-features", action="store_true",
                        help="opt-in immutable pending public features for AR; requires --replay-belief; no admission")
    parser.add_argument("--ar-head-request", type=Path, help="SHA-pinned diagnostic AR head request; requires --ar-features")
    parser.add_argument("--ar-head-request-sha256", help="SHA-256 of the explicit diagnostic AR head request")
    parser.add_argument("--ar-request-max-seconds", type=float, choices=(9.,),
                        help="opt-in registered AR request upper limit of nine seconds; legacy default is five")
    parser.add_argument("--current-root-search", action="store_true",
                        help="opt-in opponent current-root views with empty past memory, bound to a live owned root")
    parser.add_argument("--backend", choices=("checkpoint", "uniform"), default="checkpoint")
    parser.add_argument("--selection", choices=("sample", "greedy"), required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cards-db", type=Path, required=True)
    parser.add_argument("--code-list", type=Path, required=True)
    parser.add_argument("--script-root", type=Path, required=True)
    parser.add_argument("--announce-tables", type=Path, required=True)
    parser.add_argument("--announce-cap", type=int, default=192, help="uniform backend only")
    parser.add_argument("--semantic-file", type=Path)
    parser.add_argument("--card-tables", type=Path)
    parser.add_argument("--dormant-table", type=Path)
    parser.add_argument("--config", help="the ClientDuel configuration as JSON (uniform backend only; a checkpoint's "
                        "comes from its receipt)")
    parser.add_argument("--warmup-deck", type=Path, help="compile both menu blocks on a game of this deck first")
    batching = parser.add_mutually_exclusive_group()
    batching.add_argument("--search-batch-sizes", type=search_batch_sizes, default=(),
                         help="opt-in fixed search batches, e.g. 1,4,16,64,256; requires --warmup-deck")
    batching.add_argument("--constant-batch-size", type=constant_batch_size,
                         help="opt-in one batch size and full menu for ALL forwards, including live/open_stream; "
                              "requires --warmup-deck and a matching pure-policy comparison service")
    parser.add_argument("--opponent-mode", choices=("mirror", "known"),
                        help="required if the checkpoint read a public opponent deck list; no automatic inference")
    parser.add_argument("--opponent-deck", type=Path, help="known mode only: the declared public opponent deck")
    parser.add_argument("--opponent-deck-sha256", help="known mode only: expected SHA-256 of the deck file")
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--behavior-finite-audit", action="store_true",
                        help="explicit behavior-only initial-memory and all-forward finiteness audit; default off")
    args = parser.parse_args(argv)
    checkpoint = args.backend == "checkpoint"
    specialization_request=None
    specialization_search=None
    search_requested=args.specialization_search_request is not None or args.specialization_search_request_sha256 is not None
    if search_requested and (args.specialization_request is None or args.specialization_request_sha256 is None
            or args.specialization_search_request is None or args.specialization_search_request_sha256 is None
            or not args.current_root_search or args.inference_geometry!='fixed-public-b64'
            or args.constant_batch_size!=64 or args.compute_dtype is not None or args.selection!='greedy'
            or args.opponent_mode!='mirror' or args.behavior_finite_audit
            or getattr(args,'tactical_response_path',False) or getattr(args,'forced_chain_frames',False)):
        raise ValueError('specialization search needs its complete explicit diagnostic uniform/B64 request')
    if args.specialization_request is not None or args.specialization_request_sha256 is not None:
        if (args.specialization_request is None or args.specialization_request_sha256 is None or
                not checkpoint or args.checkpoint is None or args.weights!='iterate' or
                args.current_root_search and not search_requested or args.replay_belief or args.ar_features
                or args.ar_head_request is not None):
            raise ValueError('specialization needs a complete behavior-only request; search admission is separate')
        from mirrorforce.agent.train.specialization_policy import read_request
        specialization_request={'path':str(args.specialization_request),'sha256':args.specialization_request_sha256}
        read_request(args.checkpoint,specialization_request)
    if search_requested:
        from mirrorforce.netduel.agent_specialization_search import read_request as read_search_request
        specialization_search=read_search_request(args.checkpoint,
            {'path':str(args.specialization_search_request),'sha256':args.specialization_search_request_sha256},
            behavior_reference=specialization_request)
    if args.ar_features and not args.replay_belief:
        raise ValueError("AR feature capture requires explicit same-forward replay belief")
    ar_head = None
    if args.ar_head_request is not None or args.ar_head_request_sha256 is not None:
        if args.ar_head_request is None or args.ar_head_request_sha256 is None or not args.ar_features \
                or not args.current_root_search or args.weights != 'iterate':
            raise ValueError('AR head requires a complete pinned request and current-root raw public features')
        from mirrorforce.agent.belief_ar_inference import read_request
        ar_head = read_request(args.ar_head_request, args.ar_head_request_sha256)
    ar_request_clock = None
    if args.ar_request_max_seconds is not None:
        from mirrorforce.netduel.ar_clock import contract
        if ar_head is None:
            raise ValueError("an explicit AR request clock requires a pinned runtime head")
        ar_request_clock = contract(args.ar_request_max_seconds)
    if args.current_root_search and (not checkpoint or args.weights != "iterate"
                                     or args.opponent_mode != "mirror" or args.behavior_finite_audit):
        raise ValueError("current-root search requires an iterate checkpoint and explicit public mirror mode")
    if args.replay_belief and (not checkpoint or args.inference_geometry is None):
        raise ValueError("replay belief requires the explicit shared public checkpoint forward")
    if args.behavior_finite_audit and (not checkpoint or args.selection != "greedy"
                                      or args.weights not in ("iterate", "debiased_ema")):
        raise ValueError("behavior finite audit requires a greedy iterate/debiased checkpoint service")
    debiased_request = None
    if args.weights == "debiased_ema":
        if (not checkpoint or args.checkpoint is None or args.debiased_ema_request is None
                or args.debiased_ema_request_sha256 is None):
            raise ValueError("debiased_ema needs a checkpoint backend and a SHA-pinned behavior request")
        debiased_request = {"path": str(args.debiased_ema_request), "sha256": args.debiased_ema_request_sha256}
        from mirrorforce.agent.train.debiased_policy import read_request
        read_request(args.checkpoint, debiased_request)
    elif args.debiased_ema_request is not None or args.debiased_ema_request_sha256 is not None:
        raise ValueError("ordinary EMA/iterate weights do not accept a debiased behavior request")
    if args.compute_dtype is not None and not checkpoint:
        raise ValueError("a compute dtype override requires the checkpoint backend")
    geometry = None
    if args.inference_geometry is not None:
        from mirrorforce.agent.inference_geometry import FIXED_PUBLIC_B64, validate_geometry
        geometry = validate_geometry(FIXED_PUBLIC_B64)
        if not checkpoint or args.constant_batch_size != geometry["batch_size"] or args.compute_dtype is not None:
            raise ValueError("fixed public geometry requires a checkpoint, constant B64 and recorded BF16 precision")
    if (args.search_batch_sizes or args.constant_batch_size is not None) and (not checkpoint or args.warmup_deck is None):
        raise ValueError("fixed search batches require the checkpoint backend and a warmup deck")
    if args.socket.exists() or not args.temperature > 0 or checkpoint != (args.checkpoint is not None) \
            or checkpoint != (args.semantic_file is not None) or checkpoint == (args.config is not None):
        raise ValueError("a new socket, a positive temperature; the checkpoint backend takes a checkpoint and its "
                         "semantic file and reads its env configuration from the receipt, the uniform one takes --config")
    started = time.monotonic()
    native, build = load_native(args)
    if not hasattr(native, "ClientDuel"):
        raise SystemExit("this duel_native build has no ClientDuel")
    if args.current_root_search and any(not hasattr(native.ClientDuel, method)
                                       for method in ("initialize_current_root", "current_public_root_seed")):
        raise ValueError("this native build has no current-root observation initializer")
    if args.current_root_search:
        from mirrorforce.netduel.current_root_protocol import check_native_import
        check_native_import(native)
    if checkpoint:
        backend = CheckpointBackend(native, args.checkpoint, args.weights, semantic_file=args.semantic_file,
                                    code_list=args.code_list, card_tables=args.card_tables,
                                    announce_tables=args.announce_tables, dormant_table=args.dormant_table,
                                    batch_sizes=args.search_batch_sizes, constant_size=args.constant_batch_size,
                                    compute_dtype=args.compute_dtype,
                                    replay_belief=args.replay_belief,
                                    **({"ar_features": True} if args.ar_features else {}),
                                    **({"ar_head": ar_head} if ar_head is not None else {}),
                                    **({"ar_request_clock": ar_request_clock} if ar_request_clock is not None else {}),
                                    **({"specialization_request": specialization_request}
                                       if specialization_request is not None else {}),
                                    **({"inference_geometry": geometry} if geometry is not None else {}),
                                    **({"debiased_request": debiased_request} if debiased_request is not None else {}))
        config = backend.client_config()
    else:
        backend, config = UniformBackend(), json.loads(args.config)
    declared = None
    if args.opponent_mode == "known":
        if args.opponent_deck is None or args.opponent_deck_sha256 is None \
                or file_sha256(args.opponent_deck) != args.opponent_deck_sha256:
            raise ValueError("known mode needs an opponent deck file matching its declared SHA-256")
        from mirrorforce.netduel.cards import load_ydk
        main, extra, side = load_ydk(args.opponent_deck)
        if side:
            raise ValueError("the public opponent declaration is a no-side deck list")
        declared = PR.declare(main, extra)
    elif args.opponent_deck is not None or args.opponent_deck_sha256 is not None:
        raise ValueError("an opponent deck file is only accepted in explicit known mode")
    recipe_identity = {"opponent_recipe_mode": args.opponent_mode, "public_opponent_recipe": declared,
                       "opponent_deck_sha256": args.opponent_deck_sha256}
    PR.mode({"client_config": config, **recipe_identity})
    if checkpoint and args.warmup_deck is not None:
        print("SERVICE-WARMUP " + json.dumps({"seconds": backend.warm_up(native, config, args.warmup_deck)}),
              flush=True)
    identity = {"protocol": W.PROTOCOL, **backend.identity, "device": compute_device(), "selection": args.selection,
                "temperature": args.temperature if args.selection == "sample" else None,
                "serving_native_sha256": file_sha256(os.environ["MF_DUEL_NATIVE"]), "native_build": build,
                "cards_db_sha256": file_sha256(args.cards_db), "code_list_sha256": file_sha256(args.code_list),
                "announce_tables_sha256": file_sha256(args.announce_tables), "client_config": config,
                "card_view_law": card_view_law(native), **recipe_identity,
                "client_public_world": dict(RPC_CAPABILITY) if hasattr(native.ClientDuel, "public_world") else None}
    if args.behavior_finite_audit:
        from mirrorforce.netduel.ema_behavior import FINITE_IDENTITY
        identity["behavior_finite_audit"] = dict(FINITE_IDENTITY)
    if args.current_root_search:
        identity["current_root"] = dict(CURRENT_ROOT_CAPABILITY)
    if specialization_search is not None:
        from mirrorforce.netduel.agent_specialization_search import bind_ready
        identity['specialization_search']=bind_ready(specialization_search,identity,
            request_sha256=args.specialization_search_request_sha256)
    service = Service(native, backend, config, args.selection, args.temperature, identity)
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": SCHEMA, "identity": identity, "socket": str(args.socket),
               "load_seconds": round(time.monotonic() - started, 2), "training_eligible": False}
    body = json.dumps(receipt, sort_keys=True, indent=1).encode()
    (args.out / f"service-{hashlib.sha256(body).hexdigest()}.json").write_bytes(body)
    with UnixServer(str(args.socket), Handler) as server:
        server.service = service
        print("SERVICE-READY " + json.dumps({"socket": str(args.socket), "identity_sha256":
              hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}),
              flush=True)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
