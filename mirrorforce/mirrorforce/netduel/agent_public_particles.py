"""Read-only mapping of ClientDuel public worlds to owned follower UIDs.

The exact sampler supplies the whole hidden layout, including an independent
own-deck order. The latter is returned explicitly but is NOT silently installed
by the current bounded opponent-history producer. Callers must report that
coverage gap; this helper is not the final full-world provider admission.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import replace
import hashlib
import json

from .agent_search_policy import PublicParticles, SearchPolicyError
from ..common.client_entity_map import capture_entities
from ..common.client_shadow import card_code, BLANK_CODES
from ..agent.search.particles import Sampler
from ..agent.public_world_codec import RPC_SCHEMA, RPC_CAPABILITY, encode_world, decode_world, world_sha256
from . import constants as C

INFORMATION_SET_SEARCH = True
LAW = "public-client-world-uniform-uid-diagnostic/v1"


def public_world_bytes(world):
    """Closed wire representation; tuple/int dictionary keys are explicit lists."""
    try:
        out = encode_world(world, complete=True)
    except (ValueError, KeyError, TypeError) as exc:
        raise SearchPolicyError("public world must contain exactly its valid observer-only fields: " + str(exc)) from exc
    return json.dumps(out, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class PublicWorldDiagnosticParticles:
    """Uniform proposals from this client's own pending World, never another session or host.

    This wires the public native sampler online, but does not pretend the
    existing bounded producer already applies its independent own-deck order.
    """
    identity = {"law": LAW, "provider": "own-pending-public-world-rpc/v1", "world_rpc": RPC_CAPABILITY,
                "own_deck": "sampled-but-not-applied; causal-history integration pending"}

    def propose(self, root, client, *, count, seed, context, extra_origins, call, session, pending):
        if not callable(call) or not isinstance(session, str) or not isinstance(pending, dict):
            raise SearchPolicyError("the public-world provider requires this client's pending service binding")
        expected = pending["obs_sha256"]
        reply = call({"op": "public_world", "session": session, "expected_obs_sha256": expected})
        if not isinstance(reply, dict) or set(reply) != {"schema", "session", "obs_sha256", "world_sha256", "world"} \
                or reply["schema"] != RPC_SCHEMA or reply["session"] != session or reply["obs_sha256"] != expected:
            raise SearchPolicyError("public-world reply differs from this client's registered pending input")
        if world_sha256(reply["world"]) != reply["world_sha256"]:
            raise SearchPolicyError("public-world payload SHA differs")
        world = decode_world(reply["world"], complete=True)
        bank, _mapping = map_public_particles(root, world, count=count, seed=seed)
        return replace(bank, world_sha256=reply["world_sha256"], obs_sha256=expected)


def map_public_particles(root, world, *, count, seed):
    """Sample only this client's World, then bind original client slots to local UIDs.

    Any known-UID mismatch, unsupported face-up-extra ordering or lost opening
    object is refused. There is no repair/resampling based on hidden truth or a
    future action's value. Every supplied candidate is kept or the bank fails.
    """
    root._check()
    before = public_world_bytes(world)
    if type(count) is not int or count < 1 or type(seed) is not int:
        raise ValueError("positive count and independent integer client seed are required")
    sampler = Sampler(copy.deepcopy(world))
    particles = sampler.sample(seed, count)
    rows = capture_entities(root).entities
    by_uid = {row.uid: row for row in rows}
    own = root.owner.follower.viewer
    opponent = 1 - own
    origin = root.host["sync"]["receipt_state"].replay_events[0].after.entities
    wanted = {row.uid for row in origin if row.owner == opponent}
    if not wanted <= by_uid.keys():
        raise SearchPolicyError("an initial opponent UID no longer exists at the captured root")
    known = {row.uid: card_code(root.owner.core, root.duel, row.controller, row.location, row.sequence)
             for row in rows if row.controller in (0, 1) and row.location in (1, 2, 4, 8, 16, 32, 64)
             and not row.placeholder}

    def zone(location):
        return sorted((r for r in rows if r.controller == opponent and r.location == location), key=lambda r: r.sequence)

    assignments, own_orders = [], []
    for particle in particles:
        target = {uid: code for uid, code in known.items() if uid in wanted}
        for name, location in (("hand", C.LOCATION_HAND), ("deck", C.LOCATION_DECK), ("extra", C.LOCATION_EXTRA)):
            placed = zone(location)
            if len(placed) != len(world[name]) or [r.sequence for r in placed] != list(range(len(placed))):
                raise SearchPolicyError("public slot layout differs from the follower (including unsupported face-up extra)")
            for row, code in zip(placed, particle[name]):
                if row.uid in known and known[row.uid] != code:
                    raise SearchPolicyError("sampled public slot conflicts with a persistent known follower UID")
                target[row.uid] = int(code)
        for location, sequence, code in particle["facedown"]:
            found = [r for r in zone(location) if r.sequence == sequence]
            if len(found) != 1:
                raise SearchPolicyError("a public face-down slot has no unique follower UID")
            row = found[0]
            if row.uid in known and known[row.uid] != code:
                raise SearchPolicyError("particle changed a known face-down identity")
            target[row.uid] = int(code)
        if not wanted <= target.keys() or any(target[uid] in BLANK_CODES or target[uid] <= 0 for uid in wanted):
            raise SearchPolicyError("particle does not bind every original opponent object")
        own_codes = [known[r.uid] for r in rows if r.controller == own and r.location == C.LOCATION_DECK]
        if Counter(own_codes) != Counter(world["own_deck"]) or Counter(particle["own_deck"]) != Counter(own_codes):
            raise SearchPolicyError("sampled own-deck multiset differs from the client's known inventory")
        assignments.append(tuple(sorted((uid, target[uid]) for uid in wanted)))
        own_orders.append(tuple(particle["own_deck"]))
    if before != public_world_bytes(world):
        raise SearchPolicyError("public sampler changed its read-only input")
    root.snapshot.verify()
    return PublicParticles(tuple(assignments), tuple([1.0] * count), LAW), {
        "schema": "mirrorforce_public_particle_mapping/v1", "world_sha256": hashlib.sha256(before).hexdigest(),
        "own_deck_orders": tuple(own_orders), "own_deck_applied": False,
        "scope": "opponent opening-history coverage only; own-deck continuation integration remains pending"}
