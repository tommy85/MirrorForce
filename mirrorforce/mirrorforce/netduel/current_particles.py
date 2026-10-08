"""the owned in-place particle starts for continuations, without opening replay.

The public sampler/admission and native writer remain the existing ones.
The observer builder is explicit: it must mask the already realized owned
branch and retain its current public facts. This module neither supplies a
dummy observer nor claims that a wire-valid view proves engine equivalence.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import time

from .current_root_view import CurrentRootView
from .current_root_protocol import (AR_BANK_LAW, AR_PARTICLE_IDENTITY, AR_LAW,
    AR_PUBLIC_LAW, AR_HAND_ASSIGNMENT, AR_NATIVE_PROOF, COUNT_BANK_LAW, COUNT_LAW, COUNT_PARTICLE_IDENTITY,
    LAW, MEMORY_LAW, PARTICLE_IDENTITY, ROOT_RESPONSE_LAW)

INFORMATION_SET_SEARCH = True
SCHEMA = "mirrorforce_current_root_realization/v2"


class CurrentRootBudgetExpired(TimeoutError):
    """Only an observed local deadline, after all owned branch resources have unwound."""
    def __init__(self, phase):
        self.phase = phase
        super().__init__("current-root budget expired: " + phase)


class CurrentRootProvider:
    """Current public proposals, with an explicit only observer exporter.

    ``count_head`` resamples the uniform proposals under the policy's own count-belief head at the same pending
    observation (``count_head_bank``); ``ar_head_identity`` asks the service's diagnostic AR head instead."""
    def __init__(self, view_builder=None, *, ar_head_identity=None, count_head=False):
        if view_builder is None:
            from .current_root_export import build_current_view
            view_builder = build_current_view
        if not callable(view_builder):
            raise ValueError("a current-root provider needs its explicit public observer exporter")
        self.view_builder = view_builder
        if ar_head_identity is not None and (type(ar_head_identity) is not dict
                or ar_head_identity.get("schema") != "mirrorforce_current_public_ar_inference/v1"
                or ar_head_identity.get("search_admission") is not False
                or ar_head_identity.get("diagnostic_only") is not True
                or ar_head_identity.get("public_law") != AR_PUBLIC_LAW
                or ar_head_identity.get("hand_physical_law") != AR_HAND_ASSIGNMENT):
            raise ValueError("AR current-root sampling requires its pinned diagnostic-only runtime identity")
        if type(count_head) is not bool or count_head and ar_head_identity is not None:
            raise ValueError("a current-root bank takes one explicit proposal law")
        self.count_head = count_head
        self.ar_head_identity = None if ar_head_identity is None else json.loads(
            json.dumps(ar_head_identity, allow_nan=False))
        if self.ar_head_identity is not None:
            from .ar_clock import from_head
            from_head(self.ar_head_identity)

    @property
    def identity(self):
        if self.count_head:
            return dict(COUNT_PARTICLE_IDENTITY)
        return dict(PARTICLE_IDENTITY if self.ar_head_identity is None else {
            **AR_PARTICLE_IDENTITY, "ar_head": self.ar_head_identity})

    def propose(self, root, client, *, count, seed, context, extra_origins, call, session, pending, deadline=None):
        from ..common.client_midgame import admit_root
        from ..common.stage_a_joint_belief import JointRecipe
        from .current_root_export import check_seed_reply
        if deadline is not None and time.monotonic() >= deadline:
            raise CurrentRootBudgetExpired("before_proposals")
        reply = call({"op": "current_public_root_seed", "session": session,
                      "expected_obs_sha256": pending["obs_sha256"]})
        public_seed = check_seed_reply(reply, session=session, obs_sha256=pending["obs_sha256"],
                                       source_viewer=client.result.our_player)
        if deadline is not None and time.monotonic() >= deadline:
            raise CurrentRootBudgetExpired("before_proposals")
        recipe = JointRecipe.from_decks(client.main, client.extra)
        direct_ar = None
        adapter_options = {"own_deck_order": True, "history_domain": "all_turns"}
        if self.ar_head_identity is not None:
            from ..agent.public_world_codec import RPC_SCHEMA, world_sha256
            from ..agent.search.belief_hand_scope import from_world
            reply = call({"op": "public_world", "session": session,
                          "expected_obs_sha256": pending["obs_sha256"]})
            if type(reply) is not dict or set(reply) != {"schema", "session", "obs_sha256", "world_sha256", "world"} \
                    or reply["schema"] != RPC_SCHEMA or reply["session"] != session \
                    or reply["obs_sha256"] != pending["obs_sha256"] \
                    or world_sha256(reply["world"]) != reply["world_sha256"]:
                raise ValueError("current AR hand scope differs from the same-pending public World")
            if deadline is None or time.monotonic() >= deadline:
                raise CurrentRootBudgetExpired("before_proposals")
            viewer = client.result.our_player
            anchors = sorted(client.board.disclosure.known_slots(viewer, 1 - viewer, 2).items())
            hand_scope = from_world(reply["world"], obs_sha256=pending["obs_sha256"],
                                    world_sha256=reply["world_sha256"], reference_anchors=anchors)
            adapter_options.update(public_hand_scope=hand_scope, hand_deadline=deadline)
            direct_ar = lambda public, prior, viewer, categories: self._ar_bank(
                public, prior, viewer, categories, count=count, seed=seed, extra_origins=extra_origins,
                call=call, session=session, pending=pending, deadline=deadline, hand_scope=hand_scope)
        if self.count_head:
            direct_ar = self._count_head_proposal(call=call, session=session, pending=pending, count=count, seed=seed,
                                                  extra_origins=extra_origins, deadline=deadline)
        from ..agent.search.belief_hand_scope import HandScopeBudgetExceeded
        try:
            adapter = admit_root(root, client, recipe, proposals=count,
                                 seed=seed, context=context, extra_origins=extra_origins,
                                 adapter_options=adapter_options, direct_ar_proposal=direct_ar)
        except HandScopeBudgetExceeded:
            if deadline is None or time.monotonic() < deadline:
                raise  # Node-cap unknown is not a measured clock expiry.
            root._check()
            root.snapshot.verify()
            raise CurrentRootBudgetExpired("public_hand_assignment") from None
        def build(branch, **binding):
            return self.view_builder(branch, public_root_seed=public_seed, **binding)
        return CurrentRootParticles(adapter, own_session=session, obs_sha256=pending["obs_sha256"], view_builder=build)

    def _count_head_proposal(self, *, call, session, pending, count, seed, extra_origins, deadline):
        """The same pending observation's count belief, and the bank builder that resamples under it."""
        from .count_head_bank import bank, check_belief
        from .replay_filtered_particles import HEAD_SCHEMA, HEAD_LAW
        if pending.get("count_belief_available") is not True:
            raise ValueError("count-head particles need the service's same-forward count belief")
        head = call({"op": "public_belief", "session": session, "expected_obs_sha256": pending["obs_sha256"]})
        if not isinstance(head, dict) or set(head) != {"schema", "law", "session", "obs_sha256", "belief"} \
                or head["schema"] != HEAD_SCHEMA or head["law"] != HEAD_LAW or head["session"] != session \
                or head["obs_sha256"] != pending["obs_sha256"]:
            raise ValueError("count belief belongs to another pending public input")
        check_belief(head["belief"])
        if deadline is not None and time.monotonic() >= deadline:
            raise CurrentRootBudgetExpired("before_proposals")
        return lambda public, recipe, viewer, categories: bank(public, recipe, viewer=viewer, count=count, seed=seed,
            belief=head["belief"], obs_sha256=pending["obs_sha256"], extra_origin_slots=extra_origins,
            categories=categories)

    def _ar_bank(self, public, recipe, viewer, categories, *, count, seed, extra_origins,
                 call, session, pending, deadline, hand_scope):
        """Ask the existing service for one direct AR bank under its original search deadline."""
        from ..common.stage_a_joint_belief_runtime import JointDraw, JointParticleBank, ZoneClaim
        from ..common.stage_a_joint_proposals import evidence_for_public_snapshot
        from ..agent.search.belief_current_evidence import public_specification
        from ..agent.search.belief_current_law import CurrentPublicLayoutLaw
        from ..agent.search.belief_hand_scope import PublicHandScope, MAX_NODES
        if deadline is None or time.monotonic() >= deadline:
            raise CurrentRootBudgetExpired("before_proposals")
        obs_sha = pending.get("obs_sha256")
        scope = PublicHandScope(hand_scope)
        if scope.to_dict()["obs_sha256"] != obs_sha:
            raise ValueError("current physical hand belongs to another pending observation")
        specification = public_specification(public, recipe, viewer=viewer,
            extra_origin_slots=extra_origins, categories=categories, hand_scope=scope.to_dict())
        from .ar_clock import bind
        request = bind({"op": "ar_proposals", "session": session, "expected_obs_sha256": obs_sha,
                        "specification": specification, "count": count, "seed": seed,
                        "deadline_monotonic": deadline}, self.ar_head_identity)
        reply = call(request)
        if time.monotonic() >= deadline:
            raise CurrentRootBudgetExpired("before_materialization")
        if type(reply) is not dict or set(reply) != {"schema", "session", "obs_sha256", "status", "bank"} \
                or reply.get("schema") != "mirrorforce_current_public_ar_inference/v1#reply" \
                or reply.get("session") != session or reply.get("obs_sha256") != obs_sha:
            raise ValueError("AR RPC reply differs from the owned pending root")
        if reply["status"] == "original_deadline_expired" and reply["bank"] is None:
            raise CurrentRootBudgetExpired("before_proposals")
        bank_record = reply.get("bank")
        if reply["status"] != "complete" or type(bank_record) is not dict:
            raise ValueError("AR service did not return its whole original proposal bank")
        raw = dict(bank_record)
        received_sha = raw.pop("sha256", None)
        if received_sha != _sha(raw) or bank_record.get("head") != self.ar_head_identity \
                or bank_record.get("obs_sha256") != obs_sha \
                or bank_record.get("public_specification_sha256") != _sha(specification) \
                or bank_record.get("teacher_channels") is not False \
                or bank_record.get("search_admission") is not False \
                or bank_record.get("max_nodes") != MAX_NODES \
                or type(bank_record.get("feasibility_nodes")) is not int \
                or not 0 <= bank_record["feasibility_nodes"] < MAX_NODES:
            raise ValueError("AR bank digest, head, public root or diagnostic contract differs")
        feature_cache = bank_record.get("feature_cache")
        if type(feature_cache) is not dict or feature_cache.get("obs_sha256") != obs_sha \
                or feature_cache.get("source") != "same-public-forward" \
                or feature_cache.get("teacher_channels") is not False \
                or type(feature_cache.get("public_feature_sha256")) is not str:
            raise ValueError("AR bank does not bind the exact pending public-forward features")
        draws = bank_record.get("draws")
        layouts = bank_record.get("layouts")
        if type(draws) is not dict or set(draws) != {"schema", "execution_law", "rng_law", "geometry", "seed",
                "count", "sequences", "sequence_log_probabilities", "monte_carlo_weights", "decoder_identity",
                "decoder_calls", "unique_public_prefixes", "support_mask_sha256", "duplicates_retained",
                "partial_bank", "teacher_channels", "search_admission", "sha256"} \
                or draws.get("schema") != "mirrorforce_direct_public_ar_draws/v1" \
                or draws.get("execution_law") != "same-root-fixed-batch-distinct-generated-prefixes/v1" \
                or draws.get("rng_law") != "PCG64-SeedSequence-spawn-per-original-lane/v1" \
                or draws.get("count") != count or draws.get("seed") != seed \
                or draws.get("partial_bank") is not False or draws.get("search_admission") is not False \
                or draws.get("duplicates_retained") is not True or draws.get("teacher_channels") is not False \
                or len(draws.get("sequences", ())) != count or len(layouts) != count \
                or len(draws.get("monte_carlo_weights", ())) != count \
                or any(weight != 1 / count for weight in draws.get("monte_carlo_weights", [])) \
                or len(draws.get("sequence_log_probabilities", ())) != count \
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in draws.get("sequence_log_probabilities", [])) \
                or type(draws.get("decoder_calls")) is not int or draws["decoder_calls"] != (
                    len(specification["field_targets"]) + len(specification["hand_targets"])) \
                or not isinstance(draws.get("decoder_identity"), dict) \
                or any(type(draws.get(key)) is not int or draws[key] < 0 for key in ("unique_public_prefixes",)) \
                or any(type(draws.get(key)) is not str or len(draws[key]) != 64 for key in ("support_mask_sha256",)):
            raise ValueError("AR sample count, law or Monte Carlo weights differ from registration")
        raw_draws = dict(draws)
        draws_sha = raw_draws.pop("sha256")
        if draws_sha != _sha(raw_draws):
            raise ValueError("AR draw bank content digest differs")
        law = CurrentPublicLayoutLaw(specification, deadline=deadline, require_feasible=False)
        joint_draws = []
        for index, (sequence, layout) in enumerate(zip(draws["sequences"], layouts)):
            if time.monotonic() >= deadline:
                raise CurrentRootBudgetExpired("materialization_or_view")
            if type(sequence) is not list or any(type(code) is not int for code in sequence):
                raise ValueError("AR target token sequence is not canonical integer data")
            law.check_complete_layout(layout, tuple(sequence))
            joint_draws.append(JointDraw(tuple(layout["hand"]), tuple(layout["deck"]),
                tuple(tuple(row) for row in layout["facedown"]), tuple(layout["extra"])))
        evidence = evidence_for_public_snapshot(public, viewer, recipe, extra_origin_slots=extra_origins,
            categories=tuple(item for item in categories if not isinstance(item, ZoneClaim)))
        keys = tuple(sorted((location, sequence) for location, sequence in evidence.facedown_slot_keys
                            if location in (4, 8)))
        sizes = (evidence.hand_size, evidence.deck_size, len(keys), evidence.extra_facedown_size)
        probabilities = tuple(1 / count for _ in range(count))
        logp = tuple(math.log(value) for value in probabilities)
        bank = JointParticleBank(viewer, recipe, tuple(joint_draws), probabilities, logp, keys, sizes,
            seed, 0., None, proposal_law=AR_BANK_LAW)
        proof = {"schema": AR_NATIVE_PROOF, "bank_sha256": received_sha,
            "feature_sha256": bank_record["feature_cache"]["public_feature_sha256"],
            "head": self.ar_head_identity, "obs_sha256": obs_sha, "seed": seed, "count": count,
            "specification_sha256": _sha(specification), "search_admission": False, "teacher_channels": False,
            "hand_scope": scope.to_dict(), "hand_physical_law": AR_HAND_ASSIGNMENT,
            "max_nodes": MAX_NODES, "proposal_nodes": bank_record["feasibility_nodes"]}
        return bank, proof


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class CurrentRootParticle:
    """A retained start on ONE shared branch, not an independently live duel.

    Activate before observing or snapshotting its driver. Later snapshots
    belong to this same duel, and must be freed before leaving open_roots.
    """
    owner: object
    index: int
    request_json: str
    proof_json: str

    @property
    def driver(self):
        self.owner.check()
        return self.owner.starts.driver

    @property
    def prompt(self):
        self.owner.check()
        return self.owner.starts.branch.root.message

    @property
    def stream(self):
        # Common planner transport field; this is open_root_view, NEVER open_stream.
        self.owner.check()
        return json.loads(self.request_json)

    @property
    def proof(self):
        return json.loads(self.proof_json)

    def activate(self):
        self.owner.check()
        self.owner.starts.restore(self.owner.starts.starts[self.index])

    def map_root_response(self, response):
        self.owner.check()
        return self.owner.starts.starts[self.index].binding.local_response(response)


class CurrentRootRealizations:
    def __init__(self, starts):
        self.starts, self.live, self.particles = starts, True, ()

    def check(self):
        if not self.live:
            raise ValueError("the current-root realization lease has ended")
        self.starts.branch._check()

    def __iter__(self):
        self.check()
        return iter(self.particles)

    def __len__(self):
        self.check()
        return len(self.particles)

    def complete(self, index):
        self.check()
        if type(index) is not int or not 0 <= index < len(self.particles):
            raise ValueError("completed stripe is outside the original current-root bank")
        self.starts.complete(index)


class CurrentRootParticles:
    """An explicit uniform baseline over an already admitted public bank.

    Uniform and diagnostic direct-AR banks carry different immutable proposal
    laws. Root/observer construction never sees labels, a server duel, the
    original hidden deck order, or a guessed past.
    """
    law = LAW
    world_sha256 = None  # public snapshot digest below is NOT a World digest.
    _FROZEN = frozenset(("adapter", "own_session", "obs_sha256", "view_builder", "weights", "assignments",
                         "root_hash", "root_id", "seed", "proposal_law", "direct_public_proof", "world_sha256"))

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False) and name in self._FROZEN:
            raise ValueError("current-root bank binding is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, adapter, *, own_session, obs_sha256, view_builder):
        from ..common.client_particle_adapter import ClientParticleAdapter
        from ..common.stage_a_joint_proposals import PROPOSAL_LAW
        ar_mode = type(adapter) is ClientParticleAdapter and adapter.bank.proposal_law == AR_BANK_LAW
        count_mode = type(adapter) is ClientParticleAdapter and adapter.bank.proposal_law == COUNT_BANK_LAW
        if type(adapter) is not ClientParticleAdapter or not adapter.own_deck_order \
                or adapter.history_domain != "all_turns" \
                or adapter.bank.proposal_law not in (PROPOSAL_LAW, AR_BANK_LAW, COUNT_BANK_LAW) \
                or adapter.profile != "fixed-sky-all-turns/v1" or adapter.bank.power != 0 \
                or adapter.bank.head_sha256 is not None \
                or any(not math.isclose(p, 1 / len(adapter.bank.draws), rel_tol=1e-12, abs_tol=1e-14)
                       for p in adapter.bank.probabilities):
            raise ValueError("current-root mode needs its registered public bank and realized own deck order")
        if (ar_mode or count_mode) != (adapter.direct_public_proof is not None):
            raise ValueError("current-root AR identity and native bank proof differ")
        if not isinstance(own_session, str) or not own_session or not callable(view_builder) \
                or not isinstance(obs_sha256, str) or len(obs_sha256) != 64 \
                or any(c not in "0123456789abcdef" for c in obs_sha256):
            raise ValueError("current-root bank requires its own pending session and explicit observer builder")
        adapter._check()
        self.adapter, self.own_session, self.obs_sha256 = adapter, own_session, obs_sha256
        self.world_sha256 = None if not ar_mode else adapter.public_hand_scope["world_sha256"]
        self.proposal_law = AR_LAW if ar_mode else COUNT_LAW if count_mode else LAW
        self.law = self.proposal_law
        self.direct_public_proof = (None if adapter.direct_public_proof is None else
                                    json.loads(json.dumps(adapter.direct_public_proof)))
        self.view_builder = view_builder
        self.weights = tuple(adapter.bank.probabilities)
        self.assignments = tuple(tuple(sorted(adapter._assignment(i).items())) for i in range(len(self.weights)))
        self.root_hash = adapter.root.snapshot.digest
        self.root_id = "current-root-" + str(adapter.root.epoch) + "-" + self.root_hash
        self.seed = adapter.bank.seed
        self._used = False
        self._sealed = True

    @contextmanager
    def open_roots(self, root, public_recipe, *, seed, deadline):
        from ..common.client_root import RootTimeout
        if type(deadline) not in (float, int) or not math.isfinite(deadline):
            raise ValueError("current-root bank requires a finite original deadline")
        try:
            with self._open_roots(root, public_recipe, seed=seed, deadline=deadline) as roots:
                yield roots
        except RootTimeout:
            # Do not turn an expired REAL root lease or unrelated invariant
            # error into a prior fallback. Cleanup runs before either check.
            root._check()
            root.snapshot.verify()
            if time.monotonic() < deadline:
                raise
            raise CurrentRootBudgetExpired("materialization_or_view") from None

    @contextmanager
    def _open_roots(self, root, public_recipe, *, seed, deadline):
        from .agent_public_recipe import checked
        if root is not self.adapter.root:
            raise ValueError("current-root bank changed its owned root")
        declared = checked(public_recipe)
        recipe = self.adapter.bank.recipe
        expected = {name: sorted(code for code, copies, extra in zip(recipe.codes, recipe.copies, recipe.extra)
                                 for _ in range(copies) if extra == is_extra)
                    for name, is_extra in (("main", False), ("extra", True))}
        own_deck = root.host["sync"]["decks"][self.adapter.bank.viewer]
        if seed != self.seed or self._used \
                or any(declared[name] != expected[name] for name in expected) \
                or sorted(own_deck.main) != declared["main"] or sorted(own_deck.extra) != declared["extra"]:
            raise ValueError("current-root bank changed root, public recipe, independent seed or was reused")
        if deadline <= time.monotonic():
            raise CurrentRootBudgetExpired("before_materialization")
        self.adapter._check()
        self._used = True
        with self.adapter.particles(range(len(self.weights)), max_seconds=deadline - time.monotonic()) as starts:
            owner = CurrentRootRealizations(starts)
            try:
                particles = []
                for start in starts.starts:
                    if time.monotonic() >= deadline:
                        raise CurrentRootBudgetExpired("materialization_or_view")
                    starts.restore(start)
                    target, own_order, _ = self.adapter._writer(start.index)
                    hypothesis_hash = _sha({"assignments": sorted(target.items()), "own_deck_uids": own_order,
                                            "seed": seed, "index": start.index})
                    binding = {"root_id": self.root_id, "root_hash": self.root_hash,
                               "hypothesis_hash": hypothesis_hash, "viewer": 1 - self.adapter.bank.viewer}
                    value = self.view_builder(starts.branch, public_recipe=declared, **binding)
                    view = CurrentRootView.from_dict(value.to_dict() if isinstance(value, CurrentRootView) else value)
                    view.require_binding(**binding)
                    # Discard view queries' cache effects: continuation starts at the exact retained native root.
                    starts.restore(start)
                    request = {"op": "open_root_view", "session": self.own_session,
                               "expected_obs_sha256": self.obs_sha256, "view": view.to_dict(),
                               "seed": (seed * 1_000_003 + start.index + 71) & 0x7fffffff, "memory_law": MEMORY_LAW}
                    proof = {"schema": SCHEMA, **binding, "index": start.index, "proposal_seed": seed,
                             "proposal_law": self.proposal_law, "native_adapter": self.adapter.profile,
                             "adapter_binding": self.adapter.binding_digest, "view_sha256": _sha(view.to_dict()),
                             "root_response_law": ROOT_RESPONSE_LAW,
                             "root_menu_binding_sha256": start.binding.binding_sha256,
                             "memory_law": MEMORY_LAW, "opening_replayed": False,
                             "own_deck_order_realized": True, "training_eligible": False}
                    if self.proposal_law == COUNT_LAW:
                        proof["count_head"] = self.direct_public_proof
                    elif self.direct_public_proof is not None:
                        proof["direct_ar"] = self.direct_public_proof
                        proof["hand_assignment"] = self.adapter.hand_assignment_proof(start.index)
                    particles.append(CurrentRootParticle(owner, start.index, _json(request), _json(proof)))
                owner.particles = tuple(particles)
                yield owner
            finally:
                owner.live = False
                for start in starts.starts:
                    starts.api.duel_snapshot_free(start.snap)
                    start.snap = None
