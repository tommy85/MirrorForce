"""Model-independent, in-place materialization at an already admitted client root.

This is the writer, shared with current-root search adapters. It does not
sample or admit a hypothesis, reconstruct history, advance the engine, or
answer the pending prompt. The caller owns the root transaction and checks
its public constraints and menu binding before/after this mutation.
"""

from .client_entity_map import capture_entities
from .client_shadow import DeckOrderError
from ..netduel import constants as C
from ..puzzle.messages import Message

INFORMATION_SET_SEARCH = True


def _effectyn_reference(message, root_entities, viewer, invariant_error):
    # playerop.cpp::select_effect_yes_no emits player:u8, code:u32,
    # card::get_info_location():u32, description:u32. Its response is a
    # boolean, not a selectable-card index. Resolve the original handler UID
    # before declaring the unchanged wire independent of an old-hand shuffle.
    raw = message.payload
    if len(raw) != 13:
        raise invariant_error("HAND_EFFECTYN_PROMPT_LENGTH_MISMATCH")
    player, controller, location, sequence, position = raw[0], raw[5], raw[6], raw[7], raw[8]
    if player != viewer or controller not in (0, 1):
        raise invariant_error("HAND_EFFECTYN_PROMPT_OBSERVER_MISMATCH")
    base = location & ~C.LOCATION_OVERLAY
    if base not in (1, 2, 4, 8, 16, 32, 64):
        raise invariant_error("HAND_EFFECTYN_REFERENCE_LOCATION_UNSUPPORTED")
    matches = [row for row in root_entities
               if (row.controller, row.location, row.sequence) == (controller, base, sequence)]
    if len(matches) != 1:
        raise invariant_error("HAND_EFFECTYN_REFERENCE_UID_AMBIGUOUS_OR_ABSENT")
    reference = matches[0]
    if location & C.LOCATION_OVERLAY:
        matches = [row for row in root_entities if row.location == C.LOCATION_OVERLAY
                   and row.overlay_parent == reference.uid and row.overlay_ordinal == position]
        if len(matches) != 1:
            raise invariant_error("HAND_EFFECTYN_OVERLAY_UID_AMBIGUOUS_OR_ABSENT")
        reference = matches[0]
    return reference


def hand_prompt_projection(message, root_entities, hand_order, *, viewer, scope, invariant_error):
    """Project current coordinates of unchanged pending-vector UIDs.

    The core reorder API does NOT rewrite pending vectors. Their index order
    stays unchanged; this message view reports each referenced UID's new slot
    so RootMenuBinding can certify a complete response-index permutation.
    Old-group UIDs cannot cross into fresh slots and actual public anchors
    keep their UIDs. Unsupported formats refuse before any native mutation.
    """
    opponent = 1 - viewer
    hand = sorted((row for row in root_entities if row.controller == opponent and row.location == C.LOCATION_HAND),
                  key=lambda row: row.sequence)
    old = tuple(row.uid for row in hand)
    order = tuple(hand_order)
    if len(old) != scope.size or [row.sequence for row in hand] != list(range(scope.size)) \
            or len(set(order)) != len(old) or set(order) != set(old):
        raise invariant_error("HAND_UID_ORDER_GEOMETRY_MISMATCH")
    if any(order[i] != old[i] for i in range(scope.size) if i not in scope.group or i in scope.anchors) \
            or {order[i] for i in scope.group} != {old[i] for i in scope.group}:
        raise invariant_error("HAND_UID_ORDER_CROSSED_PUBLIC_SCOPE")
    if message.msg == C.MSG_SELECT_EFFECTYN:
        reference = _effectyn_reference(message, root_entities, viewer, invariant_error)
        coordinate_uid = reference.overlay_parent or reference.uid
        if coordinate_uid in old and order.index(coordinate_uid) != old.index(coordinate_uid):
            # Moving the handler (or its overlay host) changes its public slot;
            # a boolean wire alone cannot certify the same offered effect.
            raise invariant_error("HAND_REORDER_EFFECTYN_REFERENCE_MOVED")
        return message  # exact validated handler UID and complete wire are unchanged
    if order == old:
        return message
    sequence = {i: order.index(uid) for i, uid in enumerate(old)}
    raw = bytearray(bytes([message.msg]) + message.payload)
    msg = message.msg

    def records(offset, count, width, controller=4):
        if offset + count * width > len(raw):
            raise invariant_error("HAND_PENDING_PROMPT_TRUNCATED")
        for index in range(count):
            at = offset + index * width + controller
            if raw[at] == opponent and raw[at + 1] == C.LOCATION_HAND:
                previous = raw[at + 2]
                if previous not in sequence:
                    raise invariant_error("HAND_PENDING_PROMPT_SLOT_ABSENT")
                raw[at + 2] = sequence[previous]
        return offset + count * width

    try:
        if msg == C.MSG_SELECT_CARD:
            end = records(6, raw[5], 8)
        elif msg == C.MSG_SELECT_CHAIN:
            end = records(12, raw[2], 14, controller=6)
        elif msg in (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD):
            offset = 2
            widths = (7, 7, 7, 7, 7, 11) if msg == C.MSG_SELECT_IDLECMD else (11, 8)
            for width in widths:
                count = raw[offset]
                offset = records(offset + 1, count, width)
            end = offset + (3 if msg == C.MSG_SELECT_IDLECMD else 2)
        elif msg == C.MSG_SELECT_UNSELECT_CARD:
            end = records(7, raw[6], 8)
            end = records(end + 1, raw[end], 8)
        elif msg in (C.MSG_SELECT_YESNO, C.MSG_SELECT_OPTION, C.MSG_SELECT_PLACE, C.MSG_SELECT_POSITION,
                     C.MSG_SELECT_DISFIELD, C.MSG_ANNOUNCE_ATTRIB, C.MSG_ANNOUNCE_RACE,
                     C.MSG_ANNOUNCE_CARD, C.MSG_ANNOUNCE_NUMBER):
            return message  # these exact wire schemas contain no card-coordinate vector
        else:
            raise invariant_error("HAND_REORDER_PENDING_FORMAT_UNSUPPORTED", str(msg))
    except (IndexError, KeyError) as exc:
        raise invariant_error("HAND_PENDING_PROMPT_TRUNCATED") from exc
    if end != len(raw):
        raise invariant_error("HAND_PENDING_PROMPT_LENGTH_MISMATCH")
    return Message(raw[0], bytes(raw[1:]))


def materializer(root_entities, assignments, *, viewer, own_deck_order, invariant_error,
                 hand_order=None, hand_scope=None, hand_prefix=None):
    """Return the existing root mutation, retaining the caller's failure type.

    Only placeholder objects receive identities. Existing objects, their
    instance state and the already emitted prompt stay in the same arena;
    the enclosing RootEnvelope branch provides rollback on success or error.
    """
    def write(branch):
        if capture_entities(branch).entities != root_entities:
            raise invariant_error("ROOT_ENTITY_DRIFT")
        projected = None if hand_order is None else branch.message
        if hand_order is not None:
            if hand_scope is None:
                raise invariant_error("HAND_ORDER_WITHOUT_PUBLIC_SCOPE")
            projected = hand_prompt_projection(branch.message, root_entities, hand_order,
                viewer=viewer, scope=hand_scope, invariant_error=invariant_error)
        for row in root_entities:
            if row.placeholder and row.uid in assignments:
                branch.hydrate(row.controller, row.location, row.sequence, assignments[row.uid])
        if own_deck_order is not None:
            try:
                branch.reorder_deck(viewer, own_deck_order)
            except DeckOrderError as exc:
                raise invariant_error("OWN_DECK_ORDER_REJECTED", str(exc)) from exc
        if hand_order is not None:
            old = tuple(row.uid for row in sorted((row for row in root_entities
                if row.controller == 1 - viewer and row.location == C.LOCATION_HAND), key=lambda row: row.sequence))
            if tuple(hand_order) != old:
                try:
                    branch.reorder_hand(1 - viewer, hand_order)
                except DeckOrderError as exc:
                    raise invariant_error("PUBLIC_HAND_ORDER_REJECTED", str(exc)) from exc
            branch.message = projected
            branch.path = tuple(hand_prefix or ())
    return write
