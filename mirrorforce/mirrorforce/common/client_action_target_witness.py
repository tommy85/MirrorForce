"""Deferred public-action witnesses whose legality needs a hidden deck card.

The permanent follower never keeps a speculative candidate.  Candidates are
materialized one at a time under the already-owned choice origin, used only to
prove that the received public prompt is reachable, and rolled back.  The real
native history is replayed only after the selected target becomes public.

This is deliberately a tiny v1: one Engage/Foolish-style source and one later
public deck target.  It is history following, not search admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
import weakref

from ..netduel import constants as C
from ..netduel import host_view
from ..puzzle.messages import Message
from ..puzzle.single import RESPONSE_REQUIRED
from .client_origin_receipt import PublicMutation, PublicSource, boundary, history_attribution
from .client_shadow import BLANK_MAIN, hydrate
from .client_sync import SyncError, _Divergence, _Frontier, _prompt_player

INFORMATION_SET_SEARCH = True
SCHEMA = "client-public-action-target-witness/v1"
MARKERS = ("opaque_prefix_replay_unvalidated", "public_action_menu_rebind_unvalidated",
           "public_action_target_witness_unvalidated")
ENGAGE, FOOLISH = 63166095, 35726888
_SUPPORTED = frozenset((ENGAGE, FOOLISH))
_SECRETS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class TargetWitnessError(SyncError):
    pass


class TargetWitnessDeferred(Exception):
    def __init__(self, state):
        super().__init__("public target is not received yet; native history is deferred")
        self.state = state


@dataclass(frozen=True)
class DeferredTargetAction:
    choice_index: int
    begin: int
    source: tuple[int, int, int, int]
    description: int
    source_move: PublicSource
    source_chain: PublicSource
    prompt_index: int
    server_prompt: bytes
    own_index: int
    candidate_set_sha256: str
    candidate_count: int
    schema: str = SCHEMA
    search_ready: bool = False
    native_ready: bool = False

    def message(self):
        return Message(self.server_prompt[0], self.server_prompt[1:])


@dataclass(frozen=True)
class TargetProgram:
    deferred: DeferredTargetAction
    target: tuple[int, int, int, int]
    target_uid: int
    sources: tuple[PublicSource, ...]
    available_after: int

    @property
    def fixes(self):
        return self.deferred.source, self.target

    def receipt(self):
        return PublicMutation("public_action_target_program",
            (SCHEMA, self.deferred.source, self.target, self.deferred.prompt_index,
             self.deferred.own_index, self.available_after), self.target[:3], self.target_uid,
            self.target[3], False, self.sources, MARKERS)


def clear(follower):
    _SECRETS.pop(follower, None)


def _source_program(follower, choice, reveal, description):
    player, location, sequence, code = reveal
    if code not in _SUPPORTED or location != C.LOCATION_HAND or player != 1 - follower.viewer:
        return None
    move = chain = None
    for index in range(choice.begin + 1, len(follower.packets)):
        packet = follower.packets[index]
        if packet[0] in (C.MSG_WAITING, C.MSG_HINT):
            continue
        if move is None:
            if (len(packet) != 17 or packet[0] != C.MSG_MOVE
                    or struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff != code
                    or tuple(packet[5:8]) != reveal[:3] or packet[9] != player
                    or packet[10] != C.LOCATION_SZONE or not packet[12] & C.POS_FACEUP):
                return None
            move = (index, packet)
            continue
        if chain is None:
            if (len(packet) != 17 or packet[0] != C.MSG_CHAINING
                    or struct.unpack_from("<I", packet, 1)[0] & 0x7fffffff != code
                    or packet[5:9] != move[1][9:13] or packet[9:12] != move[1][9:12]
                    or struct.unpack_from("<I", packet, 12)[0] != description):
                return None
            chain = (index, packet)
            continue
        if packet[0] == C.MSG_CHAINED:
            continue
        if packet[0] in RESPONSE_REQUIRED and _prompt_player(Message(packet[0], packet[1:])) == follower.viewer:
            source = lambda row, role: PublicSource(row[0], follower.receipt_packet_raw_indices[row[0]], row[1], role)
            return source(move, "program_source_move"), source(chain, "program_source_chain"), index, packet
        return None
    return None


def _candidate_order(codes):
    """A testable scheduling detail; set identity never depends on this order."""
    return tuple(sorted(set(codes)))


def _next_prompt(follower):
    while not follower.local.finished:
        saved = follower._save()
        try:
            prompt = follower._fixed_batch(saved)
        finally:
            follower._free(saved)
        if prompt is not None:
            return prompt
    return None


def _walk_to_received_prompt(follower, prompt, prompt_index, server_prompt, depth=0):
    if depth > 4 or prompt is None:
        return False
    if _prompt_player(prompt) == follower.viewer:
        raw = host_view.deliver(prompt.msg, prompt.payload).payloads.get(follower.viewer)
        return follower.cursor - 1 == prompt_index and raw == server_prompt
    for data in follower._answers(prompt):
        saved = follower._save()
        try:
            follower._respond(prompt, data)
            if _walk_to_received_prompt(follower, _next_prompt(follower), prompt_index, server_prompt, depth + 1):
                return True
        except (_Divergence, _Frontier, SyncError):
            pass
        finally:
            follower._load(saved)
            follower._free(saved)
    return False


def _candidate_reaches_prompt(follower, choice, reveal, description, candidate, deck_sequence,
                              prompt_index, server_prompt):
    follower._load(choice.origin)
    # Ephemeral proof only: bypass receipt/audit mutation recording.  The
    # enclosing origin snapshot is restored after every candidate.
    hydrate(follower.core, follower.local.pduel, *reveal)
    hydrate(follower.core, follower.local.pduel, reveal[0], C.LOCATION_DECK, deck_sequence, candidate)
    fixed = follower._save()
    try:
        prompt = follower._fixed_batch(fixed, receipt_origin=choice.origin)
    finally:
        follower._free(fixed)
    if prompt is None or follower.cursor - 1 != choice.begin or _prompt_player(prompt) == follower.viewer:
        return False
    response = follower._public_activation_answer(prompt, reveal, description)
    if response not in follower._answers(prompt):
        return False
    follower._respond(prompt, response)
    return _walk_to_received_prompt(follower, _next_prompt(follower), prompt_index, server_prompt)


def begin(follower, choice, reveal, description):
    if follower.deferred_action is not None:
        raise TargetWitnessError("a second target witness cannot overlap the deferred prefix")
    program = _source_program(follower, choice, reveal, description)
    if program is None:
        return None
    source_move, source_chain, prompt_index, server_prompt = program
    origin = follower.receipt_saved.get(choice.origin[0]) if choice.origin else None
    deck_rows = (() if origin is None else tuple(r for r in origin.entities
        if r.controller == reveal[0] and r.location == C.LOCATION_DECK and r.placeholder == 1))
    if not deck_rows:
        raise TargetWitnessError("target witness origin has no blank main-deck entity")
    deck_sequence = min(r.sequence for r in deck_rows)
    candidates = _candidate_order(follower.public_action_witness_main)
    if not candidates or any(type(code) is not int or code <= 0 or code == BLANK_MAIN for code in candidates):
        raise TargetWitnessError("target witness recipe is empty or invalid")
    baseline = follower._save()
    owner = boundary(follower).owner
    audit_length = len(owner.audit)
    successful = []
    try:
        for candidate in candidates:
            try:
                if _candidate_reaches_prompt(follower, choice, reveal, description, candidate, deck_sequence,
                                             prompt_index, server_prompt):
                    successful.append(candidate)
            except (SyncError, _Divergence, _Frontier):
                pass
            finally:
                follower._load(baseline)
                del owner.audit[audit_length:]
    finally:
        follower._load(baseline)
        follower._free(baseline)
        del owner.audit[audit_length:]
    successful = tuple(sorted(set(successful)))
    if not successful:
        raise TargetWitnessError("no public-recipe candidate can reproduce the received target-dependent prompt")
    candidate_digest = hashlib.sha256((SCHEMA + ":" + ",".join(map(str, successful))).encode()).hexdigest()
    state = DeferredTargetAction(len(follower.choices) - 1, choice.begin, tuple(reveal), int(description),
        source_move, source_chain, prompt_index, bytes(server_prompt), len(follower.own),
        candidate_digest, len(successful))
    _SECRETS[follower] = frozenset(successful)
    owner.audit.append({"kind": "target-witness-set", "sha256": candidate_digest, "count": len(successful)})
    return state


def _confirm_records(packet):
    if len(packet) < 4 or packet[0] != C.MSG_CONFIRM_CARDS:
        return ()
    count, out = packet[3], []
    if len(packet) != 4 + count * 7:
        return ()
    for index in range(count):
        offset = 4 + index * 7
        code = struct.unpack_from("<I", packet, offset)[0] & 0x7fffffff
        out.append((packet[offset + 4], packet[offset + 5], packet[offset + 6], code))
    return tuple(out)


def complete(follower):
    state = follower.deferred_action
    if type(state) is not DeferredTargetAction or len(follower.own) <= state.own_index:
        raise TargetWitnessError("deferred target action has no committed public response")
    player, _location, _sequence, source_code = state.source
    start, end = state.prompt_index + 1, None
    solving = solved = False
    target_moves, confirms, sources = [], [], [state.source_move, state.source_chain]
    for index in range(start, len(follower.packets)):
        packet = follower.packets[index]
        source = PublicSource(index, follower.receipt_packet_raw_indices[index], packet, "program_public_prefix")
        if packet[0] in (getattr(C, "MSG_CHAIN_NEGATED", -1), getattr(C, "MSG_CHAIN_DISABLED", -1)):
            raise TargetWitnessError("negated target-dependent actions are unsupported")
        if packet[0] == C.MSG_CHAIN_SOLVING:
            solving = True
        elif packet[0] == C.MSG_MOVE and len(packet) == 17 and packet[5] == player \
                and packet[6] == C.LOCATION_DECK:
            target_moves.append((index, packet, source))
        elif packet[0] == C.MSG_CONFIRM_CARDS:
            confirms.append((index, packet, source))
        elif packet[0] == C.MSG_CHAIN_SOLVED:
            solved = True
        elif packet[0] == C.MSG_CHAIN_END:
            end = index
            sources.append(source)
            break
        elif packet[0] in RESPONSE_REQUIRED and _prompt_player(Message(packet[0], packet[1:])) == follower.viewer:
            # The already-recorded prompt is allowed; another own decision
            # before resolution means this v1 prefix is not closed.
            raise TargetWitnessError("target-dependent opaque prefix reached another own prompt before its public result")
    if end is None:
        raise TargetWitnessError("target-dependent opaque prefix has no complete public resolution")
    if not solving or not solved or len(target_moves) != 1:
        raise TargetWitnessError("target-dependent action has no unique solved deck transition")
    move_index, move, move_source = target_moves[0]
    target_code = struct.unpack_from("<I", move, 1)[0] & 0x7fffffff
    destination = tuple(move[9:12])
    target_sources = [move_source]
    available_after = move_index + 1
    if source_code == ENGAGE:
        if (target_code or move[10] != C.LOCATION_HAND or move[9] != player
                or any(packet[0] == C.MSG_SHUFFLE_DECK for packet in follower.packets[start:move_index])):
            raise TargetWitnessError("Engage target transition differs from the strict public program")
        matches = []
        for index, packet, source in confirms:
            for controller, location, sequence, code in _confirm_records(packet):
                if (controller, location, sequence) == destination and code:
                    matches.append((index, code, source))
        if len(matches) != 1:
            raise TargetWitnessError("Engage target has no unique public confirmation")
        confirm_index, target_code, confirm_source = matches[0]
        if confirm_index <= move_index:
            raise TargetWitnessError("Engage target confirmation precedes its move")
        target_sources.append(confirm_source)
        available_after = confirm_index + 1
    elif source_code == FOOLISH:
        if not target_code or move[9] != player or move[10] != C.LOCATION_GRAVE:
            raise TargetWitnessError("Foolish target transition differs from the strict public program")
    else:
        raise TargetWitnessError("unsupported target-dependent source")
    candidates = _SECRETS.get(follower)
    if not candidates or target_code not in candidates:
        raise TargetWitnessError("public target is outside the prompt-compatible witness set")
    target = (player, C.LOCATION_DECK, move[7], target_code)
    choice = follower.choices[state.choice_index] if 0 <= state.choice_index < len(follower.choices) else None
    origin = follower.receipt_saved.get(choice.origin[0]) if choice is not None and choice.origin else None
    rows = (() if origin is None else tuple(r for r in origin.entities
        if (r.controller, r.location, r.sequence) == target[:3]))
    if len(rows) != 1 or rows[0].placeholder != 1:
        raise TargetWitnessError("public target does not name one blank origin deck entity")
    sources.extend(target_sources)
    return TargetProgram(state, target, rows[0].uid, tuple(sources), available_after)


def resume(follower):
    state = follower.deferred_action
    program = complete(follower)
    choice = follower.choices[state.choice_index]
    previous = (choice.saved, choice.prompt, choice.answers, choice.next, choice.fixes)
    try:
        choice.fixes = [program.fixes]
        with history_attribution(follower, MARKERS[0], proofs=(program.receipt(),)), \
                history_attribution(follower, MARKERS[1]), history_attribution(follower, MARKERS[2]):
            if not follower._rewrite(choice):
                raise TargetWitnessError("public target program could not regenerate its source IDLE menu")
            response = follower._public_activation_answer(choice.prompt, state.source, state.description)
            if response not in choice.answers:
                raise TargetWitnessError("public target source is absent from regenerated native answers")
            choice.answers = [response]
        follower.deferred_action = None
        clear(follower)
        follower._try(choice)
    except BaseException:
        # The controller's catch-up transaction restores the full graph.  This
        # local assignment only avoids publishing a half-cleared secret if the
        # exception is inspected before that restoration.
        choice.saved, choice.prompt, choice.answers, choice.next, choice.fixes = previous
        raise
