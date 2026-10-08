"""Natural complete-hypothesis replay from one observer's received history.

Opening identities and every shuffle come from a public causal plan. No blank
follower hydration, entity reorder, forced activation or old raw-menu UID
binding is copied into this duel. Hidden historical responses are local,
bounded witnesses: only a naturally accepted path reproducing the *entire*
received wire and unchanged proposed root can be admitted.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import ctypes
import math
import time

from . import constants as C
from .causal_history import PublicPositionHistory
from .agent_causal_plan import solve, verify, _sha, PlanBudgetExceeded, PlanIncompatible
from .opponent_stream import OpponentStreamError, SyntheticRoot
from .causal_proof import PROOF_SCHEMA, LAW, ACTIVATION_LEGALITY, RESPONSE_LAW, WITNESS_LAW, verify_natural_root_proof
from .agent_public_recipe import checked, declare
from .agent_shuffle_plan import ShufflePlanAPI, ShufflePlanReplayError, steps_from_causal_plan
from .wire_projection import InitialRefreshCore, project_message
from ..puzzle.messages import Message, split_messages
from ..puzzle.single import RESPONSE_REQUIRED
from ..common.client_replay_journal import at_root
from ..common.client_root import RootEnvelope
from ..common.sidecar_io import require_sha256
from ..worldmodel.engine import DeckList, DuelConfig, DuelDriver, PROCESSOR_BUFFER_LEN

INFORMATION_SET_SEARCH = True


class ReplayBudgetExceeded(OpponentStreamError):
    pass


class HistoricalWitnessExhausted(OpponentStreamError):
    """A valid hypothetical search exhausted mismatches, not native failures."""
    pass


class _Mismatch(Exception):
    pass


class ReplayCandidateDropped(Exception):
    """One explicitly biased proposal filter verdict, never a whole-root impossibility proof."""
    def __init__(self, reason, message, *, evidence=None):
        super().__init__(message)
        self.reason, self.evidence = reason, evidence


def _ordinary_shuffle_mismatch(state, steps, *, before):
    """Recognize only a different ordinary event, not an ABI/internal fault.

    EVENT_SHAPE also covers internal sequence/flavor invariants. Equal public
    event fields therefore stay hard errors. Field/set-group shuffles and all
    other native fault codes are deliberately outside this narrow classifier.
    SET_GROUP can set EVENT_SHAPE without refreshing the last-event fields.
    Require one newly observed event at the same plan cursor; a process call
    that already consumed another event cannot establish that provenance.
    """
    if any(type(row) is not tuple or len(row) != 9 or any(type(v) is not int or v < 0 for v in row)
           for row in (before, state)):
        return None
    if before[:4] != state[:4] or before[4] != 0 or before[5:] == state[5:]:
        return None
    version, mode, count, cursor, fault, message, player, location, participants = state
    if (version, mode, count, fault) != (1, 1, len(steps), 2) or not 0 <= cursor < len(steps):
        return None
    ordinary = {C.MSG_SHUFFLE_DECK: C.LOCATION_DECK, C.MSG_SHUFFLE_HAND: C.LOCATION_HAND,
                C.MSG_SHUFFLE_EXTRA: C.LOCATION_EXTRA}
    expected = steps[cursor]
    if message not in ordinary or location != ordinary[message] or player not in (0, 1) \
            or not int(message != C.MSG_SHUFFLE_EXTRA) <= participants <= 255 \
            or expected.message not in ordinary or expected.location != ordinary[expected.message]:
        return None
    expected.words()  # retain the exact installed plan's structural contract
    actual = (message, player, location, participants)
    wanted = (expected.message, expected.player, expected.location, len(expected.sequences))
    if actual == wanted:
        return None
    return {"state_before": list(before), "state": list(state),
            "expected_event": list(wanted), "actual_event": list(actual),
            "scope": "one-historical-response-branch-only"}


def public_history(root, public_recipe):
    """Only the root owner's received packets, sent responses and public recipes."""
    if type(root) is not RootEnvelope:
        raise OpponentStreamError("causal history requires this client's owned root")
    root._check()
    at_root(root)  # verify the root metadata and accepted local boundary, not its invented hidden history
    public = checked(public_recipe)
    follower = root.owner.follower
    own = follower.decks[follower.viewer]
    if public != declare(own.main, own.extra):
        raise OpponentStreamError("causal replay currently requires explicitly registered mirror recipes")
    types = {int(code): int(card.type) for code, card in root.owner.core.card_pool().cards.items()}
    history = PublicPositionHistory(follower.viewer, own.main, own.extra, public["main"], public["extra"], card_types=types)
    answers = iter(root.host["sync"]["own"])
    pending = None
    packets = tuple(root.host["sync"]["receipt_wire_packets"])
    for index, raw in enumerate(packets):
        history.feed(bytes(raw))
        if history.pending is not None and index + 1 < len(packets):
            response = next(answers, None)
            if response is None:
                raise OpponentStreamError("a received later packet lacks its actual preceding own response")
            history.respond(bytes(response))
        pending = history.pending
    if pending != root.server_prompt or next(answers, None) is not None:
        raise OpponentStreamError("public causal history does not stop at exactly the owned pending prompt")
    return history


def _player(prompt):
    return prompt.payload[1] if prompt.msg == C.MSG_SELECT_SUM else prompt.payload[0]


def _config(root, plan, seed):
    opening = {(p, z): codes for p, z, _, codes in plan.openings}
    decks = tuple(DeckList("complete-causal-hypothesis", opening[p, C.LOCATION_DECK],
                           tuple(reversed(opening[p, C.LOCATION_EXTRA]))) for p in (0, 1))
    for p in (0, 1):
        original = root.owner.follower.decks[root.owner.follower.viewer]
        if Counter(decks[p].main) != Counter(original.main) or Counter(decks[p].extra) != Counter(original.extra):
            raise OpponentStreamError("causal opening does not preserve the registered public recipe")
    # Full menu enumeration is a witness-search budget, never a hidden action
    # truncation. ANNOUNCE_CARD sees the entire public card catalogue.
    return DuelConfig(decks, seed=seed, forced_deck_orders=tuple(d.main for d in decks),
                      full_phase_menu=True, full_card_sort_menu=True, auto_end_phase_discard=False,
                      max_options=len(root.owner.core.card_pool().cards), **dict(root.owner.follower.rules))


@dataclass
class _Choice:
    snapshot: object
    pystate: dict
    cursor: int
    answered: int
    frames: list
    outbox: tuple
    prompt: Message
    candidates: object
    snapshot_bytes: int


class _Witness:
    def __init__(self, root, history, plan, public, *, seed, deadline, source_sha256,
                 max_branches, max_snapshot_bytes, events, runtime_profile=None, first_mismatch=False):
        self.root, self.history, self.plan, self.public = root, history, plan, public
        self.viewer, self.opponent = history.viewer, 1 - history.viewer
        self.deadline, self.max_branches = deadline, max_branches
        self.max_snapshot_bytes = max_snapshot_bytes
        self.expected = tuple(root.host["sync"]["receipt_wire_packets"])
        self.own = tuple(bytes.fromhex(raw) for _, raw in history.record()["own_responses"])
        self.driver = DuelDriver(_config(root, plan, seed), InitialRefreshCore(root.owner.core))
        self.api = root.api
        self.shuffle = ShufflePlanAPI(root.owner.core)
        self.cursor = self.answered = self.branches = self.response_nodes = 0
        self.submitted_total = 0
        self.native_before_response, self.script_errors = [], []
        self.native_trace, self.submitted_trace = [], []
        self.last_counterexample = None
        self.shuffle_mismatch_count, self.last_shuffle_mismatch = 0, None
        self.snapshot_bytes = self.peak_snapshot_bytes = 0
        self.frames, self.outbox, self.choices = [], [[], []], []
        self.hints = self._hints(events)
        self.last_failure = "no historical response witness"
        self.source_sha256, self.seed = source_sha256, seed
        self.started = time.monotonic()
        self.runtime_profile = runtime_profile
        self.first_mismatch = first_mismatch

    def _check(self):
        self.root._check()
        if time.monotonic() >= self.deadline:
            raise ReplayBudgetExceeded("complete natural history witness deadline exhausted")

    def _hints(self, events):
        """Old local answers only prioritize attempts; they confer NO semantic or legal proof."""
        hints = {}
        indices = self.root.host["sync"]["receipt_packet_raw_indices"]
        for event in events:
            if event.kind != "response":
                continue
            prompt, response = event.payload
            player = prompt[2] if prompt[0] == C.MSG_SELECT_SUM else prompt[1]
            if player != self.opponent:
                continue
            cursor = indices[event.matched_cursor - 1] + 1 if event.matched_cursor else 0
            hints.setdefault((cursor, prompt[0], event.own_answered), []).append(bytes(response))
        return hints

    def _append(self, projected, native_message=None):
        for raw in projected[self.viewer]:
            if self.cursor >= len(self.expected) or raw != self.expected[self.cursor]:
                expected = self.expected[self.cursor].hex() if self.cursor < len(self.expected) else None
                native = (bytes([native_message.msg]) + native_message.payload).hex() if native_message else None
                self.last_counterexample = {"packet_index": self.cursor, "received_packet_hex": expected,
                    "synthetic_packet_hex": raw.hex(), "native_message_hex": native,
                    "own_answered": self.answered, "native_submitted_responses": self.submitted_total,
                    "branches": self.branches, "choices": len(self.choices)}
                raise _Mismatch(f"received packet {self.cursor}: expected={expected}, hypothetical={raw.hex()}, "
                                f"native_message={native}")
            self.cursor += 1
        # The real observer already owns its complete received stream. Keeping
        # a second growing own outbox at every response snapshot would copy an
        # unused prefix quadratically; only the synthetic opponent needs frames.
        self.outbox[self.opponent].extend(projected[self.opponent])

    def _answers(self, prompt):
        """Lazy complete-response enumeration; no physical UID lookup of virtual deck indices."""
        seen = set()
        for response in self.hints.get((self.cursor, prompt.msg, self.answered), ()):
            if response not in seen:
                seen.add(response)
                yield response  # a bad hint must meet natural RETRY and rollback, never bypass validation
        saved = self.driver.save_pystate()
        stack = [()]
        while stack:
            self._check()
            self.response_nodes += 1
            if self.response_nodes > self.max_branches:
                raise ReplayBudgetExceeded("natural response-enumeration budget exhausted")
            prefix = stack.pop()
            if len(prefix) > self.driver.config.max_sub_rounds:
                raise ReplayBudgetExceeded("natural multi-selection witness is beyond its declared depth")
            try:
                self.driver.restore_pystate(saved)
                self.driver._ctx.our_player = _player(prompt)
                first = self.driver.parse_prompt(prompt.msg, prompt.payload)
                response = first.auto_response
                selector = first.selector
                if response is None:
                    for index in prefix:
                        response = selector.choose(index)
                    if response is None:
                        options = selector.options()
                        if len(options) + len(stack) > self.max_branches:
                            raise ReplayBudgetExceeded("natural response frontier exceeds its declared budget")
                        stack.extend(prefix + (index,) for index in reversed(range(len(options))))
                        continue
                response = bytes(response)
            finally:
                self.driver.restore_pystate(saved)
            if response not in seen:
                seen.add(response)
                yield response

    def _send(self, prompt, response):
        self._check()
        if _player(prompt) == self.opponent:
            self.frames.append({"messages": [[p[0], p[1:].hex()] for p in self.outbox[self.opponent]],
                                "response": bytes(response).hex()})
            self.outbox[self.opponent] = []
        self.driver._answering_msg, self.driver._answering_payload = prompt.msg, prompt.payload
        self.driver._respond(response)
        self.submitted_total += 1
        self.submitted_trace.append({"native_index": len(self.native_trace) - 1,
            "prompt_hex": (bytes([prompt.msg]) + prompt.payload).hex(), "response_hex": bytes(response).hex(),
            "player": _player(prompt), "received_cursor": self.cursor})

    def _save_choice(self, prompt):
        self._check()
        size = int(self.api.duel_arena_extent(self.driver.pduel))
        if self.snapshot_bytes + size > self.max_snapshot_bytes:
            raise ReplayBudgetExceeded("historical response snapshots exceed the declared memory budget")
        snapshot = self.api.duel_snapshot(self.driver.pduel)
        if not snapshot:
            raise OpponentStreamError("cannot retain a natural response witness boundary")
        self.snapshot_bytes += size
        self.peak_snapshot_bytes = max(self.peak_snapshot_bytes, self.snapshot_bytes)
        choice = _Choice(snapshot, self.driver.save_pystate(), self.cursor, self.answered, list(self.frames),
                         tuple(list(box) for box in self.outbox), prompt, None, size)
        # The generator starts only AFTER its own snapshot/metadata is retained.
        choice.candidates = self._answers(prompt)
        self.choices.append(choice)

    def _next(self):
        while self.choices:
            self._check()
            choice = self.choices[-1]
            if self.api.duel_rollback(self.driver.pduel, choice.snapshot) != 0:
                raise OpponentStreamError("cannot restore the natural response witness boundary")
            self.driver.restore_pystate(choice.pystate)
            self.cursor, self.answered = choice.cursor, choice.answered
            self.frames = list(choice.frames)
            self.outbox = [list(box) for box in choice.outbox]
            try:
                response = next(choice.candidates)
            except StopIteration:
                self.api.duel_snapshot_free(choice.snapshot)
                self.snapshot_bytes -= choice.snapshot_bytes
                self.choices.pop()
                continue
            self.branches += 1
            if self.branches > self.max_branches:
                raise ReplayBudgetExceeded("natural response-branch budget exhausted")
            self._send(choice.prompt, response)
            return
        raise HistoricalWitnessExhausted("unchanged causal layout has no accepted natural witness: " + self.last_failure)

    def _root_layout(self):
        """Query only this owned hypothetical duel and roll back query-cache writes."""
        from .board import parse_query_segments
        snapshot = self.api.duel_snapshot(self.driver.pduel)
        if not snapshot:
            raise OpponentStreamError("cannot isolate the hypothetical root layout audit")
        found = []
        buffer = ctypes.create_string_buffer(0x10000)
        try:
            for player in (0, 1):
                for location in (C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE,
                                 C.LOCATION_GRAVE, C.LOCATION_REMOVED, C.LOCATION_EXTRA):
                    self._check()
                    length = self.driver.core.query_field_card(self.driver.pduel, player, location,
                                                               C.QUERY_CODE | C.QUERY_POSITION, buffer)
                    for sequence, row in enumerate(parse_query_segments(buffer.raw[:length])):
                        if row and row.get("code"):
                            found.append((player, location, sequence, int(row["code"])))
        finally:
            restored = self.api.duel_rollback(self.driver.pduel, snapshot)
            self.api.duel_snapshot_free(snapshot)
            if restored != 0:
                raise OpponentStreamError("hypothetical layout audit changed the live continuation")
        return tuple(sorted(found))

    def _process(self, steps):
        before = self.shuffle.state(self.driver.pduel)
        try:
            return self.shuffle.process(self.driver.pduel)
        except ShufflePlanReplayError as exc:
            # Never hide simultaneous native/Lua faults, stale state or a
            # generic ABI failure behind a plausible received-history mismatch.
            if type(exc) is not ShufflePlanReplayError or self.driver.core.log \
                    or self.shuffle.state(self.driver.pduel) != exc.native_state:
                raise
            detail = _ordinary_shuffle_mismatch(exc.native_state, steps, before=before)
            if detail is None:
                raise
            self.shuffle_mismatch_count += 1
            self.last_shuffle_mismatch = detail
            self.last_counterexample = None  # no packet/whole-layout impossibility certificate
            raise _Mismatch("historical response produced a different ordinary shuffle event: " + str(detail)) from exc

    def run(self):
        self._check()
        if self.runtime_profile is None:
            self.driver.build()
        else:
            from .causal_must_emit_runtime import build_fresh
            build_fresh(self, self.runtime_profile)
        steps, graph_proof = steps_from_causal_plan(self.history,
            tuple((p, z, s, code) for p, z, s, _, code in self.plan.root), self.plan)
        registration = self.shuffle.install(self.driver.pduel, steps, source_sha256=self.source_sha256,
                                             history_sha256=self.plan.history_sha256, root_sha256=self.plan.root_sha256)
        initial = {seat: (self.expected[0][:1] + bytes([seat]) + self.expected[0][2:],
                           *self.driver.core.initial_packets[seat]) for seat in (0, 1)}
        self._append(initial)
        while True:
            self._check()
            prompt = None
            try:
                raw = self._process(steps)
                self.driver.steps += 1
                if self.driver.core.log:
                    self.script_errors.extend(self.driver.core.log)
                    raise OpponentStreamError("natural history emitted a native script error: " + str(self.driver.core.log[:1]))
                if raw & PROCESSOR_BUFFER_LEN:
                    n = self.driver.core.get_message(self.driver.pduel, self.driver._msgbuf)
                    for message in split_messages(self.driver._msgbuf.raw[:n]):
                        self.native_trace.append((bytes([message.msg]) + message.payload).hex())
                        if self.submitted_total == 0:
                            self.native_before_response.append((bytes([message.msg]) + message.payload).hex())
                        if message.msg == C.MSG_RETRY:
                            raise _Mismatch("a proposed historical response was not naturally legal")
                        self.driver._observe(message)
                        self._append(project_message(self.driver.core, self.driver.pduel, message.msg, message.payload), message)
                        if message.msg in RESPONSE_REQUIRED:
                            if prompt is not None:
                                raise OpponentStreamError("one native batch contains multiple historical prompts")
                            prompt = message
                if self.driver.core.log:
                    raise OpponentStreamError("natural history emitted a native script error: " + str(self.driver.core.log[:1]))
                if prompt is None:
                    if self.driver.finished:
                        raise _Mismatch("hypothesis terminated before the actual owned root")
                    continue
                if _player(prompt) == self.viewer:
                    if self.answered < len(self.own):
                        self._send(prompt, self.own[self.answered])
                        self.answered += 1
                        continue
                    if self.cursor != len(self.expected) or bytes([prompt.msg]) + prompt.payload != self.history.pending:
                        raise _Mismatch("hypothesis reached a different own root or incomplete received prefix")
                    wanted = tuple((p, z, s, code) for p, z, s, _, code in self.plan.root)
                    if self._root_layout() != tuple(sorted(wanted)):
                        raise _Mismatch("natural root layout differs from the unchanged complete proposal")
                    self._check()
                    receipt = self.shuffle.finish(self.driver.pduel)
                    self.driver._answering_msg, self.driver._answering_payload = prompt.msg, prompt.payload
                    stream = {"op": "open_stream", "seat": self.opponent, "main": self.public["main"],
                              "extra": self.public["extra"], "seed": self.seed, "opponent_recipe_mode": "mirror",
                              "public_opponent_recipe": self.public, "frames": self.frames,
                              "messages": [[p[0], p[1:].hex()] for p in self.outbox[self.opponent]]}
                    proof = {"schema": PROOF_SCHEMA, "training_eligible": False,
                             "initialization": LAW, "activation_legality": ACTIVATION_LEGALITY,
                             "response_witness_law": RESPONSE_LAW, "witness_selection": WITNESS_LAW,
                             "public_stream_equal": True, "root_menu_equal": True, "root_layout_equal": True,
                             "own_deck_applied": True, "hypothesis_sha256": self.plan.root_sha256,
                             "history_sha256": self.plan.history_sha256, "causal_plan": graph_proof,
                             "causal_plan_record": self.plan.record(),
                             "causal_cuts": [asdict(cut) for cut in self.history.permutations],
                             "causal_dead_tokens": list(self.history.dead),
                             "shuffle_words": [1, len(steps), *(word for step in steps for word in step.words())],
                             "shuffle_registration": registration, "shuffle_receipts": list(receipt),
                             "stream_sha256": _sha(stream), "public_prefix_sha256": _sha([p.hex() for p in self.expected]),
                             "response_branches": self.branches, "response_nodes": self.response_nodes,
                             "shuffle_mismatch_branches": self.shuffle_mismatch_count,
                             "peak_snapshot_bytes": self.peak_snapshot_bytes,
                             "seconds": time.monotonic() - self.started}
                    verify_natural_root_proof(proof, hypothesis_sha256=self.plan.root_sha256,
                        history_sha256=self.plan.history_sha256, source_sha256=self.source_sha256)
                    return SyntheticRoot(self.driver, prompt, stream, proof)
                self._save_choice(prompt)
                self._next()
            except _Mismatch as exc:
                self.last_failure = str(exc)
                if self.first_mismatch:
                    raise ReplayCandidateDropped("packet_mismatch", str(exc), evidence=self.last_counterexample) from exc
                self._next()

    def close_snapshots(self):
        for choice in reversed(self.choices):
            self.api.duel_snapshot_free(choice.snapshot)
        self.choices.clear()
        self.snapshot_bytes = 0


def trial_replay_filtered_candidate(root, history, layout, public_recipe, *, events, seed, deadline,
                                    source_sha256, entropy_profile=None):
    """One positive construction and one natural replay, sharing the caller's <=2s cap.

    No feasibility classification, Farkas certificate, alternative opening or
    mismatch backtracking is performed. Native/source/cleanup failures remain hard.
    """
    if type(root) is not RootEnvelope or root.branch_session is None:
        raise OpponentStreamError("replay filtering requires the owned producer branch lease")
    witness, accepted = None, False
    started = time.monotonic()
    construction_seconds = 0.
    constructed = False
    try:
        try:
            plan = solve(history, layout, seed=seed, max_nodes=None, deadline=deadline, mode="positive-replay")
            construction_seconds = time.monotonic() - started
            constructed = True
            witness = _Witness(root, history, plan, checked(public_recipe), seed=seed, deadline=deadline,
                source_sha256=source_sha256, max_branches=65536, max_snapshot_bytes=1024 ** 3,
                events=events, runtime_profile=entropy_profile, first_mismatch=True)
            particle = witness.run()
            particle.proof["replay_filter"] = {"law": "first-mismatch-or-two-second-discard/v1",
                "construction_seconds": construction_seconds, "replay_seconds": time.monotonic() - witness.started,
                "historical_existence_node_budget": None, "negative_certificate": False}
            accepted = True
            return particle
        except PlanIncompatible as exc:
            raise ReplayCandidateDropped("construction_mismatch", str(exc)) from exc
        except PlanBudgetExceeded as exc:
            raise ReplayCandidateDropped("construction_timeout", str(exc)) from exc
        except ReplayBudgetExceeded as exc:
            raise ReplayCandidateDropped("replay_timeout", str(exc)) from exc
        except (_Mismatch, HistoricalWitnessExhausted) as exc:
            raise ReplayCandidateDropped("packet_mismatch", str(exc),
                evidence=witness.last_counterexample if witness is not None else None) from exc
    except ReplayCandidateDropped as exc:
        exc.construction_seconds = construction_seconds if constructed else time.monotonic() - started
        exc.replay_seconds = time.monotonic() - witness.started if witness is not None else 0.
        raise
    finally:
        if witness is not None:
            cleaned = False
            try:
                if entropy_profile is not None:
                    from .causal_must_emit_runtime import discard_build
                    discard_build(witness.driver)
                witness.close_snapshots()
                cleaned = True
            finally:
                if not accepted or not cleaned:
                    witness.driver.close()
                    if getattr(witness.driver, "pduel", None) is not None:
                        raise OpponentStreamError("filtered replay retained a native driver after cleanup")


def trial_natural_candidate(root, history, layout, public_recipe, *, events, seed, deadline,
                            source_sha256, entropy_profile=None, max_witnesses=16, max_branches=65536,
                            max_snapshot_bytes=1024 ** 3):
    """One fixed root, under the caller's existing lease: accepted engine or typed unknown/negative.

    Exhausting a finite list of historical witnesses is NEVER a whole-root
    negative proof. Only the separately audited no-choice deterministic case
    can issue such a certificate; all other failures remain unknown.
    """
    from .conditioned_particles import CertifiedNaturalInfeasible, NaturalFeasibilityUnknown
    from .bank_fallback import UnknownReason
    from . import causal_profile as profile_bundle
    if type(root) is not RootEnvelope or root.branch_session is None:
        raise OpponentStreamError("candidate trials require the producer's existing owned branch lease")
    public = checked(public_recipe)
    bundled = profile_bundle.is_bundle(entropy_profile)
    legacy_profile = profile_bundle.legacy_profile(entropy_profile)
    failures = []
    for attempt in range(max_witnesses):
        witness, accepted = None, False
        certificate_phase = False
        try:
            try:
                plan = solve(history, layout, seed=seed + attempt, deadline=deadline)
                witness = _Witness(root, history, plan, public, seed=seed + attempt, deadline=deadline,
                    source_sha256=source_sha256, max_branches=max_branches,
                    max_snapshot_bytes=max_snapshot_bytes, events=events,
                    **({"runtime_profile": entropy_profile} if bundled else {}))
                particle = witness.run()
                particle.proof["witness_attempt"] = attempt
                particle.proof["rejected_witnesses"] = failures
                accepted = True
                return particle
            except HistoricalWitnessExhausted as exc:
                certificate_phase = True
                detail = {"attempt": attempt, "error": str(exc), "counterexample": witness.last_counterexample if witness else None}
                if witness is not None and getattr(witness, "last_shuffle_mismatch", None) is not None:
                    detail["last_shuffle_mismatch"] = witness.last_shuffle_mismatch
                failures.append(detail)
                if witness is not None and entropy_profile is not None:
                    from .causal_entropy import certify_initial_counterexample
                    certificate = certify_initial_counterexample(root, history, layout, plan, witness,
                                                                  entropy_profile=legacy_profile, source_sha256=source_sha256)
                    if certificate is None:
                        from .causal_entropy import certify_own_pass_counterexample
                        certificate = certify_own_pass_counterexample(root, history, layout, plan, witness,
                            entropy_profile=legacy_profile, source_sha256=source_sha256)
                    if bundled:
                        from .causal_must_emit_runtime import certify_counterexample
                        certificate = certify_counterexample(witness, history, layout,
                            profile=entropy_profile, legacy_certificate=certificate)
                    if certificate is not None:
                        message = "the complete root contradicts received wire under its explicitly registered lemma" if bundled else \
                            "the unique opening contradicts received wire under its registered deterministic-prefix lemma"
                        raise CertifiedNaturalInfeasible(message, certificate) from exc
        except (ReplayBudgetExceeded, PlanBudgetExceeded) as exc:
            # A typed budget may expire inside the certificate path's live
            # attestation too. Sibling except clauses did not catch that path.
            # Keep legacy certificate failures fatal; only the registered 
            # producer can later issue a live capability after bank postchecks.
            if certificate_phase and not bundled:
                raise
            raise NaturalFeasibilityUnknown("historical existence is unknown after its explicit resource budget: " + str(exc),
                evidence={"witness_attempts": failures, "budget": str(exc)},
                reason=UnknownReason.HISTORY_BUDGET if isinstance(exc, PlanBudgetExceeded)
                else UnknownReason.RESPONSE_BUDGET) from exc
        finally:
            if witness is not None:
                if bundled:
                    from .causal_must_emit_runtime import discard_build
                    cleaned = False
                    try:
                        try:
                            discard_build(witness.driver)
                        finally:
                            witness.close_snapshots()
                        cleaned = True
                    finally:
                        if not accepted or not cleaned:
                            witness.driver.close()
                            if getattr(witness.driver, "pduel", None) is not None:
                                raise OpponentStreamError("failed natural witness retained its native handle after close")
                else:
                    witness.close_snapshots()
                    if not accepted:
                        witness.driver.close()
                        if getattr(witness.driver, "pduel", None) is not None:
                            raise OpponentStreamError("failed natural witness retained its native handle after close")
    raise NaturalFeasibilityUnknown("no historical witness found; finite witness attempts do not prove the root impossible",
        evidence={"witness_attempts": failures, "maximum_witnesses": max_witnesses},
        reason=UnknownReason.WITNESSES_EXHAUSTED)


@contextmanager
def natural_roots(root, layouts, public_recipe, *, seed, deadline, source_sha256,
                  expected_history_sha256=None, max_witnesses=16, max_branches=65536,
                  max_snapshot_bytes=1024 ** 3):
    """Own all detached engines under one root lease, admitting the whole fixed bank."""
    if type(root) is not RootEnvelope or not isinstance(layouts, tuple) or not layouts \
            or type(seed) is not int or type(max_witnesses) is not int or max_witnesses < 1 \
            or type(max_branches) is not int or max_branches < 1 \
            or type(max_snapshot_bytes) is not int or max_snapshot_bytes < 1 \
            or type(deadline) not in (float, int) or not math.isfinite(deadline):
        raise OpponentStreamError("natural replay requires an owned root and explicit positive witness budgets")
    require_sha256(source_sha256, "causal producer source", error=OpponentStreamError)
    history = public_history(root, public_recipe)
    if expected_history_sha256 is not None and _sha(history.record()) != expected_history_sha256:
        raise OpponentStreamError("the proposed bank was sampled against another received history")
    public, admitted = checked(public_recipe), []
    events = at_root(root)  # export while the real root owns its arena, before any branch lease
    with root.branch():
        try:
            for index, layout in enumerate(layouts):
                failures = []
                for attempt in range(max_witnesses):
                    witness = None
                    try:
                        plan = solve(history, layout, seed=seed + 1009 * index + attempt, deadline=deadline)
                        verify(history, layout, plan)
                        witness = _Witness(root, history, plan, public, seed=seed + 1009 * index + attempt,
                                           deadline=deadline, source_sha256=source_sha256, max_branches=max_branches,
                                           max_snapshot_bytes=max_snapshot_bytes, events=events)
                        particle = witness.run()
                        particle.proof["witness_attempt"] = attempt
                        particle.proof["rejected_witnesses"] = failures
                        admitted.append(particle)
                        break
                    except (ReplayBudgetExceeded, PlanBudgetExceeded):
                        raise
                    except HistoricalWitnessExhausted as exc:
                        failures.append(str(exc))
                        if attempt + 1 == max_witnesses:
                            raise OpponentStreamError("fixed root proposal exhausted its natural historical witnesses: "
                                                      + str(exc), particle_index=index) from exc
                    finally:
                        if witness is not None:
                            witness.close_snapshots()
                            if not admitted or admitted[-1].driver is not witness.driver:
                                witness.driver.close()
            yield tuple(admitted)
        finally:
            for particle in reversed(admitted):
                particle.driver.close()
