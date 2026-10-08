"""A trained policy for play (the evaluation bridge, internal Elo): a checkpoint's network with its constants.

``load_policy`` verifies the payload against its content-addressed name and receipt, checks that the files and the
native module the caller runs with are the ones the checkpoint was trained with (semantic file and code list digests,
card tables identity, the env's window law, guard laws and step-limit law), and returns the agent, its variables
(the EMA or the iterate, with the frozen semantic and card-table constants injected) and the receipt.

    agent, variables, receipt = load_policy(ckpt, "ema", semantic_file=..., code_list_file=..., card_tables_file=...,
                                            native=module)
    policy = Policy(agent, variables, receipt)
    rstate = policy.initial_state()
    rstate, logits, value, wdl = policy.act(observation, rstate, first)   # logits over the real menu rows

``native_setup`` registers on the env module the laws a checkpoint was trained under (announce law, dormant table)
and refuses files that differ from the receipt; the env construction parameters (max_options, max_steps, room
format, history law) are in ``receipt["config"]`` (the trainer's arguments) and ``receipt["window_law"]``.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import flax
import jax.numpy as jnp
import numpy as np

from mirrorforce.agent.env import card_tables
from mirrorforce.agent.model.reference_agent import ReferenceAgent
from mirrorforce.agent.model.policy_net import PolicyNetConfig
from mirrorforce.agent.train import checkpoint_store, state_io


def _config(args: dict) -> PolicyNetConfig:
    fields = {f.name: f for f in dataclasses.fields(PolicyNetConfig)}
    unknown = set(args) - set(fields)
    if unknown:
        raise ValueError(f"the checkpoint's model has fields this code does not know: {sorted(unknown)}")
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in args.items()}
    return PolicyNetConfig(**values)


def inference_config(args: dict, compute_dtype=None) -> PolicyNetConfig:
    """An explicit inference-only float32 override; the receipt and parameter arrays remain unchanged."""
    original = _config(args)
    if compute_dtype is None:
        return original
    if compute_dtype != "float32":
        raise ValueError("the only inference compute override is explicit float32")
    return dataclasses.replace(original, dtype="float32")


def load_policy(ckpt_path, which: str = "ema", *, semantic_file, code_list_file, card_tables_file, native=None,
                compute_dtype=None):
    if which not in ("ema", "iterate"):
        raise ValueError("which is 'ema' or 'iterate'")
    path = Path(ckpt_path)
    receipt, _ = checkpoint_store.check_checkpoint(path.parent, path.stem)
    model = receipt.get("model", {})
    if model.get("architecture") != "policy_net":
        raise ValueError(f"{path.name}: not a policy_net checkpoint")
    tables = receipt["tables"]
    if state_io.file_digest(semantic_file) != tables["semantic_file_sha256"]:
        raise ValueError("the semantic file differs from the checkpoint's")
    if state_io.file_digest(code_list_file) != tables["code_list_sha256"]:
        raise ValueError("the code list differs from the checkpoint's")
    card_arrays, card_metadata, card_sha = card_tables.load(card_tables_file, code_list_file)
    if receipt.get("card_tables") != card_tables.identity(card_metadata, card_sha):
        raise ValueError("the card tables differ from the checkpoint's")
    if native is not None:
        law = {"window_rows": int(native.history_window_rows), "chunk_rows": int(native.history_chunk_rows),
               "chunk_cap": int(native.history_chunk_cap), "chunk_slots": int(native.history_chunk_slots)}
        if receipt.get("window_law") != law:
            raise ValueError(f"the env's window law {law} differs from the checkpoint's {receipt.get('window_law')}")
        if list(native.guard_laws) != receipt["recipe"]["guard_laws"] or \
                native.step_limit_law != receipt["recipe"]["step_limit_law"]:
            raise ValueError("the env's guard or step-limit laws differ from the checkpoint's")
        if getattr(native, "card_view_law", "fresh_query") != receipt.get("card_view_law", "fresh_query") or \
                list(getattr(native, "observation_laws", ())) != receipt.get("observation_laws", []):
            raise ValueError("the env's observation laws (card view, pending source) differ from the checkpoint's")
    raw = flax.serialization.msgpack_restore(path.read_bytes())
    if which == "ema" and "ema" not in raw:
        raise ValueError(f"{path.name} holds no EMA")
    params = raw["ema"] if which == "ema" else raw["state"]["params"]

    from mirrorforce.agent.train.cleanba import load_structured_semantics
    semantics, semantic_shape, _ = load_structured_semantics(str(semantic_file), str(code_list_file))
    if list(semantic_shape) != list(model["semantic_shape"]):
        raise ValueError("the semantic tables' shape differs from the checkpoint's model")
    constants = {"semantics": {**{k: jnp.asarray(v) for k, v in semantics.items()},
                               **{k: jnp.asarray(v) for k, v in card_arrays.items()}}}
    agent = ReferenceAgent(inference_config(model["args"], compute_dtype), tuple(semantic_shape))
    variables = {"params": params, "constants": constants}
    return agent, variables, json.loads(json.dumps(receipt))


class Policy:
    """One seat's play with a loaded policy, as the trainer's actor runs it: the env's observation for one decision
    (``obs:``-prefixed keys, as ScriptedDuel/ClientDuel.observation() return them) gets the batch axis, the menu is
    cut to the actor's block (32 rows, or max_options when more are offered), the seat's memory sits in slot 0
    (``main`` true), ``first`` clears it at a game's first decision; one compiled function per menu block."""

    def __init__(self, agent, variables, receipt, *, matmul_precision=None, inference_geometry=None):
        import jax
        self.agent, self.variables = agent, variables
        self.max_options = int(receipt["config"]["max_options"])
        if matmul_precision not in (None, "highest"):
            raise ValueError("inference matmul precision must be default or highest")
        self.inference_geometry = None
        if inference_geometry is not None:
            from mirrorforce.agent.inference_geometry import validate_geometry
            from mirrorforce.agent.belief_features import forward
            from mirrorforce.agent.model import policy_net as V
            self.inference_geometry = validate_geometry(inference_geometry)
            profile = self.inference_geometry
            if agent.config.dtype != profile["compute_dtype"] or matmul_precision is not None \
                    or self.max_options != profile["menu_rows"] \
                    or agent.config.memory_slots != profile["memory_slots"] \
                    or agent.config.chunk_slots != profile["chunk_slots"]:
                raise ValueError("loaded policy dimensions/dtype/precision differ from its inference geometry")
            self.batch_size = profile["batch_size"]

            def public_apply(v, o, r, first):
                if first.shape != (profile["batch_size"],) or o["action_ir_"].shape[:2] != (
                        profile["batch_size"], profile["menu_rows"]):
                    raise ValueError("direct public apply differs from its fixed batch/menu geometry")
                for key, axis, expected in (("turn_events_", 1, profile["event_rows"]),
                        ("closed_turns_", 2, profile["event_rows"]),
                        ("turn_chunks_", 2, profile["chunk_event_rows"])):
                    if o[key].shape[0] != profile["batch_size"] or o[key].shape[axis] != expected:
                        raise ValueError("direct public apply differs from its fixed event/summary geometry")
                memory = V.Memory(*(jnp.stack([a, b], axis=1) for a, b in zip(*r)))
                if memory.buffer.shape != (profile["batch_size"], 2, profile["memory_slots"], agent.config.d) \
                        or memory.chunks.shape != (profile["batch_size"], 2, profile["chunk_slots"], agent.config.d) \
                        or any(x.shape != (profile["batch_size"], 2) for x in
                               (memory.count, memory.last_turn, memory.chunk_count, memory.chunk_turn)):
                    raise ValueError("public recurrent memory differs from fixed inference geometry")
                logits, value, wdl, after, bad, features, old_logits, old_valid = forward(
                    agent.model, v, o, jnp.zeros(first.shape, jnp.int32), memory, first, full_shapes=True)
                states = (tuple(x[:, 0] for x in after), tuple(x[:, 1] for x in after))
                return states, logits, value[:, None], wdl, bad, features, old_logits, old_valid

            # One compiled program, including all feature outputs, for BOTH
            # ordinary serving and collection. Select outputs only outside JIT.
            self._public_apply = jax.jit(public_apply)
            self._apply = lambda *args: self._public_apply(*args)[:5]
            return

        def apply(v, o, r, first):
            if matmul_precision is None:
                return agent.apply(v, o, r, first, jnp.ones(first.shape, bool), return_bad=True)
            with jax.default_matmul_precision(matmul_precision):
                return agent.apply(v, o, r, first, jnp.ones(first.shape, bool), return_bad=True)

        self._apply = jax.jit(apply)

    def initial_state(self):
        return (self.agent.init_rnn_state(1), self.agent.init_rnn_state(1))

    def act(self, observation, rstate, first: bool):
        if self.inference_geometry is not None:
            count = self.inference_geometry["batch_size"]
            return self.act_batch([observation] * count, [rstate] * count, [first] * count, full_menu=True)[0]
        from mirrorforce.agent.train.cleanba import MENU_BLOCK, sliced_obs
        if any(str(k).startswith("priv") for k in observation):
            raise ValueError("the policy refuses private (other seat's) inputs")
        obs = {k[len("obs:"):]: np.asarray(v)[None] for k, v in observation.items() if k.startswith("obs:")}
        options = int((obs["action_ir_"][0, :, 0] > 0).sum())
        menu = MENU_BLOCK if options <= MENU_BLOCK else self.max_options
        rstate, logits, value, wdl, bad = self._apply(self.variables, sliced_obs(obs, menu), rstate,
                                                      jnp.asarray([bool(first)]))
        if bool(np.asarray(bad).any()):
            raise RuntimeError("policy_net: inconsistent chunk or closed-turn delivery in the observation")
        return rstate, np.asarray(logits)[0, :options], float(np.asarray(value).reshape(-1)[0]), np.asarray(wdl)[0]


    def _batch_inputs(self, observations, rstates, firsts, *, full_menu):
        """One public packing path shared by policy-only and feature consumers."""
        import jax
        from mirrorforce.agent.train.cleanba import MENU_BLOCK, sliced_obs
        if type(full_menu) is not bool or not len(observations) == len(rstates) == len(firsts):
            raise ValueError("batch observations, memories and first flags must have equal lengths and explicit menu mode")
        if not observations:
            raise ValueError("a packed public batch must contain at least one real row")
        if any(str(k).startswith("priv") for o in observations for k in o):
            raise ValueError("the policy refuses private (other seat's) inputs")
        profile = self.inference_geometry
        if profile is not None and (len(observations) != profile["batch_size"] or full_menu is not True
                or any(str(k).startswith("label") for o in observations for k in o)):
            raise ValueError("fixed public inference requires its exact registered full-menu public rows, never labels")
        keys = [k for k in observations[0] if k.startswith("obs:")]
        if any({k for k in o if k.startswith("obs:")} != set(keys) for o in observations):
            raise ValueError("batch public observation keys differ")
        obs = {k[len("obs:"):]: np.stack([np.asarray(o[k]) for o in observations]) for k in keys}
        options = (obs["action_ir_"][:, :, 0] > 0).sum(-1)
        if profile is not None:
            for key, axis, expected in (("turn_events_", 1, profile["event_rows"]),
                    ("closed_turns_", 2, profile["event_rows"]),
                    ("turn_chunks_", 2, profile["chunk_event_rows"])):
                if key not in obs or obs[key].shape[axis] != expected:
                    raise ValueError("public event/summary windows differ from fixed inference geometry")
            if obs["action_ir_"].shape[1] != profile["menu_rows"]:
                raise ValueError("public menu differs from fixed inference geometry")
        menu = MENU_BLOCK if not full_menu and int(options.max()) <= MENU_BLOCK else self.max_options
        stacked = jax.tree_util.tree_map(lambda *x: np.concatenate([np.asarray(v) for v in x]), *rstates)
        return sliced_obs(obs, menu), stacked, jnp.asarray(np.asarray(firsts, bool)), options

    def act_batch(self, observations, rstates, firsts, *, full_menu=False):
        """Several decisions in one call (search rollouts in lockstep): per row (rstate, logits over that row's real
        menu rows, value, win/draw/loss logits). The menu block is the batch's (32 when every menu fits); one compiled
        function per batch size and block. ``full_menu`` fixes the menu at max_options for an explicitly registered
        constant-shape inference service; it does not change the training policy or default actor path."""
        import jax
        if not observations:
            if rstates or firsts or type(full_menu) is not bool or self.inference_geometry is not None:
                raise ValueError("batch observations, memories and first flags must have equal lengths")
            return []
        obs, stacked, first, options = self._batch_inputs(observations, rstates, firsts, full_menu=full_menu)
        rstate, logits, value, wdl, bad = self._apply(self.variables, obs, stacked, first)
        if bool(np.asarray(bad).any()):
            raise RuntimeError("policy_net: inconsistent chunk or closed-turn delivery in an observation")
        logits, value, wdl = np.asarray(logits), np.asarray(value).reshape(-1), np.asarray(wdl)
        rows = jax.tree_util.tree_map(np.asarray, rstate)
        return [(jax.tree_util.tree_map(lambda x, i=i: x[i:i + 1], rows), logits[i, :options[i]], float(value[i]),
                 wdl[i]) for i in range(len(observations))]

    def act_batch_features(self, observations, rstates, firsts):
        """Public features from the identical opted-in serving executable and memory update."""
        import jax
        if self.inference_geometry is None:
            raise ValueError("public features require the explicitly registered inference geometry")
        obs, stacked, first, options = self._batch_inputs(observations, rstates, firsts, full_menu=True)
        states, logits, value, wdl, bad, features, old_logits, old_valid = jax.tree_util.tree_map(
            np.asarray, self._public_apply(self.variables, obs, stacked, first))
        if bad.any():
            raise RuntimeError("policy_net: inconsistent chunk or closed-turn delivery in the observation")
        return [(jax.tree_util.tree_map(lambda x, i=i: x[i:i + 1], states), logits[i, :options[i]],
                 float(value[i, 0]), wdl[i], {key: x[i] for key, x in features.items()},
                 old_logits[i], old_valid[i]) for i in range(len(observations))]


def native_setup(native, receipt, *, announce_tables, dormant_table=None):
    """Register the checkpoint's announce law (and dormant table) with the env module; refuses tables whose identity
    differs from the receipt's."""
    from mirrorforce.agent.env import announce_law, dormant_law
    config = receipt["config"]
    identity = announce_law.register(native, announce_tables, int(config["announce_cap"]),
                                     config.get("announce_room_format"))
    if identity != receipt["announce"]:
        raise ValueError("the announce tables differ from the checkpoint's")
    if receipt.get("dormant") is not None or dormant_table is not None:
        if dormant_table is None or dormant_law.register(native, dormant_table) != receipt.get("dormant"):
            raise ValueError("the dormant table differs from the checkpoint's")


__all__ = ["load_policy", "Policy", "native_setup"]
