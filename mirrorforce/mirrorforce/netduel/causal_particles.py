"""Pending-bound complete root banks for the natural Option-A producer."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

from . import causal_replay
from .causal_history import LAW as HISTORY_LAW
from .agent_causal_plan import _sha, root_layout
from .causal_sampler import CapacitySampler, LAW, SamplingBudgetExceeded, SamplingError
from ..common.sidecar_io import require_sha256
from ..agent.public_world_codec import RPC_CAPABILITY, RPC_SCHEMA, decode_world, world_sha256

INFORMATION_SET_SEARCH = True


@dataclass(frozen=True)
class PublicCausalParticles:
    assignments: tuple  # complete immutable root layouts, NOT follower UID assignments
    weights: tuple
    law: str
    world_sha256: str
    obs_sha256: str
    history_sha256: str
    source_sha256: str
    sampler_proof_json: str = "{}"

    def __post_init__(self):
        if self.law != LAW or type(self.assignments) is not tuple or not self.assignments \
                or type(self.weights) is not tuple or len(self.assignments) != len(self.weights) \
                or any(type(w) not in (float, int) or not math.isfinite(w) or w <= 0 for w in self.weights):
            raise ValueError("a causal bank needs its exact declared law and immutable positive-weight roots")
        for assignment in self.assignments:
            if type(assignment) is not tuple or not assignment or any(type(row) is not tuple or len(row) != 4
                    or any(type(v) is not int for v in row) or row[0] not in (0, 1)
                    or row[1] not in (1, 2, 4, 8, 16, 32, 64) or row[2] < 0 or not 0 < row[3] < 0x80000000
                    or row[2] >= (7 if row[1] == 4 else 8 if row[1] == 8 else 255)
                    or 999000001 <= row[3] <= 999000004 for row in assignment) \
                    or tuple(sorted(assignment)) != assignment \
                    or len({row[:3] for row in assignment}) != len(assignment):
                raise ValueError("a causal assignment must bind distinct canonical public root positions")
        for value in (self.world_sha256, self.obs_sha256, self.history_sha256, self.source_sha256):
            require_sha256(value, "causal particle binding", error=ValueError)
        if not isinstance(self.sampler_proof_json, str) or not isinstance(json.loads(self.sampler_proof_json), dict):
            raise ValueError("causal sampler evidence must be immutable JSON")

    def open_roots(self, root, public_recipe, *, seed, deadline):
        return causal_replay.natural_roots(root, self.assignments, public_recipe, seed=seed, deadline=deadline,
            source_sha256=self.source_sha256, expected_history_sha256=self.history_sha256)


class PublicCapacityParticles:
    identity = {"law": LAW, "provider": "own-pending-public-capacity/v1", "world_rpc": RPC_CAPABILITY,
                "history_law": HISTORY_LAW, "own_deck": "causal-opening-and-shuffle-plan; root-exact/v1",
                "counting_states": 100000, "historical_oracle_nodes": 1000000,
                "root_distribution": "distinct-root-layouts; not historical witnesses/v1"}

    def propose(self, root, client, *, count, seed, context, extra_origins, call, session, pending, deadline=None):
        root._check()
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError("causal sampling deadline must be finite")
        limit = root.deadline if deadline is None else min(root.deadline, deadline)
        if not callable(call) or not isinstance(pending, dict) or not isinstance(session, str):
            raise ValueError("public causal proposals need this client's pending connection binding")
        reply = call({"op": "public_world", "session": session, "expected_obs_sha256": pending["obs_sha256"]})
        if not isinstance(reply, dict) or set(reply) != {"schema", "session", "obs_sha256", "world_sha256", "world"} \
                or reply["schema"] != RPC_SCHEMA or reply["session"] != session \
                or reply["obs_sha256"] != pending["obs_sha256"] or world_sha256(reply["world"]) != reply["world_sha256"]:
            raise ValueError("causal proposals received a foreign or malformed public World")
        world = decode_world(reply["world"], complete=True)
        from .agent_public_recipe import declare
        public = declare(client.main, client.extra)
        history = causal_replay.public_history(root, public)
        source = hashlib.sha256(Path(causal_replay.__file__).read_bytes()).hexdigest()
        sampler, particles, layouts = None, [], ()
        try:
            sampler = CapacitySampler(history, world, deadline=limit,
                                      max_states=self.identity["counting_states"],
                                      max_oracle_nodes=self.identity["historical_oracle_nodes"])
            particles = sampler.sample(seed, count)
            layouts = tuple(root_layout(history, particle) for particle in particles)
            if time.monotonic() >= limit:
                raise SamplingBudgetExceeded("causal bank finalization exceeded its nonrenewing sampling deadline")
        except SamplingError as exc:
            partial = getattr(exc, "partial_particles", particles)
            exc.proposal_evidence = {"world_sha256": reply["world_sha256"],
                "obs_sha256": pending["obs_sha256"], "history_sha256": _sha(history.record()),
                "source_sha256": source, "sampler": sampler.proof() if sampler is not None else None,
                "partial_particles": partial,
                "partial_layouts": [list(map(list, root_layout(history, p))) for p in partial],
                "requested_proposals": count, "generated_proposals": len(partial), "proposal_seed": seed}
            raise
        return PublicCausalParticles(layouts, (1.0,) * count, LAW, reply["world_sha256"], pending["obs_sha256"],
                                     _sha(history.record()), source, json.dumps(sampler.proof(), sort_keys=True))
