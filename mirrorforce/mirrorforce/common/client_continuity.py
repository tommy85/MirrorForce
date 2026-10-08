"""Search continuity: the worlds a search's lines reached, kept for the seat's later roots (design 10.13).

A searched root's lines play the chosen candidate on and on through the seat's later prompts. When the real duel then
reaches one of those prompts along exactly the path a line took, that line's particle world is a draw of the hidden
cards consistent with everything public since the root (the opponent's answers, and the windows it was offered or
not), and its rollout from there is a rollout of the action it took there. The later root starts from those worlds
instead of fresh draws only.

The path is what the seat saw: every message's bytes as the host sends them to the seat (``host_view.deliver``) and a
WAITING whenever the opponent was asked, from the root on. Two paths the seat cannot tell apart are the same path;
paths that reach the same board otherwise are different ones (what was used this turn is not on the board).

A world is kept as the opponent's cards by position (``JointDraw``: the hand as a multiset, the deck and the extra
deck by sequence, every field and banished card by slot), read from our own hypothetical engine at the line's prompt:
our guesses carried along, never the real hidden cards. The later root writes it as it writes any draw, and every
check a written particle passes holds for it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib

from ..netduel import constants as C
from ..netduel import host_view
from ..netduel.wire_projection import _WAITING_PROMPTS
from .client_entity_map import capture_entities
from .client_shadow import card_code
from .stage_a_joint_belief_runtime import JointDraw

CONTINUED_LAW = "mirrorforce_continued_worlds/v1"


def public_tokens(messages, viewer: int) -> list[bytes]:
    """``messages`` (``(msg, payload)`` records of a local duel) as the seat ``viewer`` receives them from the host,
    one token per message: a WAITING mark when the opponent answers the message's prompt, then the message's bytes
    for the seat as many times as it receives them (nothing for a message the seat does not receive)."""
    tokens = []
    for message in messages:
        msg, payload = int(message.msg), bytes(message.payload)
        delivery = host_view.deliver(msg, payload)
        token = b""
        if msg in _WAITING_PROMPTS and (payload[1] if msg == C.MSG_SELECT_SUM else payload[0]) != viewer:
            token = b"W"
        raw = delivery.payloads.get(viewer)
        if raw is not None:
            token += (len(raw).to_bytes(4, "little") + raw) * delivery.counts.get(viewer, 0)
        tokens.append(token)
    return tokens


def path_digest(tokens) -> str:
    """The digest of a path of tokens (``public_tokens``)."""
    hasher = hashlib.sha256()
    for token in tokens:
        hasher.update(len(token).to_bytes(4, "little"))
        hasher.update(token)
    return hasher.hexdigest()


_FIELD = (C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_REMOVED)


def capture_world(branch, opponent: int) -> JointDraw:
    """The opponent's cards in the branch's current (hypothetical) duel by position, as a ``JointDraw``."""
    core, pduel = branch.driver.core, branch.driver.pduel
    rows = [row for row in capture_entities(branch).entities if row.controller == opponent and not row.overlay_parent]

    def codes(location):
        return tuple(card_code(core, pduel, opponent, location, row.sequence)
                     for row in sorted((row for row in rows if row.location == location), key=lambda row: row.sequence))

    slots = tuple((row.location, row.sequence, card_code(core, pduel, opponent, row.location, row.sequence))
                  for row in sorted((row for row in rows if row.location in _FIELD),
                                    key=lambda row: (row.location, row.sequence)))
    return JointDraw(hand=codes(C.LOCATION_HAND), deck=codes(C.LOCATION_DECK), facedown=slots,
                     extra=codes(C.LOCATION_EXTRA))


@dataclass
class Waypoint:
    """A line at one of the seat's later action menus: its tokens up to the menu's own message (``at``), the world
    there, and the local row the line took (None until it was scored)."""

    at: int
    world: JointDraw
    choice: int | None = None


@dataclass
class LineTrace:
    """What a line keeps for later roots: the seat's path from the line's root (``public_tokens``), its waypoints,
    and its rollout value when it ended as a fresh line from any of them would (not at a decision cap)."""

    tokens: list = field(default_factory=list)
    waypoints: list = field(default_factory=list)
    value: float | None = None


@dataclass(frozen=True)
class Kept:
    """One kept world: waypoint ``waypoint`` of ``trace``, its path measured from token ``start`` (the tokens the
    root it was kept for had seen), and the particle weight it carries there."""

    trace: LineTrace
    start: int
    waypoint: int
    weight: float

    @property
    def point(self) -> Waypoint:
        return self.trace.waypoints[self.waypoint]

    @property
    def digest(self) -> str:
        return path_digest(self.trace.tokens[self.start:self.point.at])

    @property
    def sample(self):
        """``(local row, value)``: the line's rollout from this world, or None when it has none."""
        point = self.point
        return None if point.choice is None or self.trace.value is None else (point.choice, self.trace.value)

    def later(self, weight):
        """The line's later waypoints, measured from this one, carrying ``weight``: kept for the root after this
        one when its row is the one this world's sample took."""
        return tuple(Kept(self.trace, self.point.at, number, weight)
                     for number in range(self.waypoint + 1, len(self.trace.waypoints)))


@dataclass(frozen=True)
class Continuation:
    """The worlds a searched root kept: the root's prompt (``search_rows`` index), where its path starts in the
    local message log, its turn and the kept worlds."""

    prompt: int
    cursor: int
    turn: int
    kept: tuple[Kept, ...]

    def matching(self, messages, viewer: int, turn: int) -> tuple[Kept, ...]:
        """The kept worlds whose path is the real one from the root to this prompt (``messages``, the local log)."""
        if turn != self.turn or len(messages) < self.cursor:
            return ()
        digest = path_digest(public_tokens(messages[self.cursor:], viewer))
        return tuple(item for item in self.kept if item.digest == digest)




def _matched(slots, cards) -> bool:
    """Whether every slot (a set of codes, or ``(zone, codes)`` with cards ``(zone, code)``) gets a distinct card."""
    owner = {}

    def fits(slot, card):
        return card[1] in slot[1] and card[0] == slot[0] if isinstance(slot, tuple) else card in slot

    def augment(number, seen):
        for index, card in enumerate(cards):
            if index in seen or not fits(slots[number], card):
                continue
            seen.add(index)
            if index not in owner or augment(owner[index], seen):
                owner[index] = number
                return True
        return False

    return all(augment(number, set()) for number in range(len(slots)))


def kept_world_breach(evidence, zone_claims, draw: JointDraw) -> str | None:
    """The public claim at a later root that a kept world breaks, or None. A world whose path matched the real one
    honors them all (the path shows everything they come from); a breach is a defect, never a world to drop."""
    facedown = {(location, sequence): code for location, sequence, code in draw.facedown}
    for key, codes in evidence.facedown_categories:
        if facedown.get(tuple(key)) not in codes:
            return "facedown_category"
    unknown = Counter(draw.hand)
    unknown.subtract(evidence.disclosed_hand)
    if any(count < 0 for count in unknown.values()):
        return "disclosed_hand"
    if not _matched(list(evidence.hand_categories), list(unknown.elements())):
        return "hand_category"
    if any(tuple(key) not in facedown for key in evidence.facedown_sampling_keys):
        return "facedown_slot"
    sampled = [(int(key[0]), facedown[tuple(key)]) for key in evidence.facedown_sampling_keys]
    fresh = {tuple(key) for key in evidence.fresh_facedown_keys}
    # The identities known in a zone sit in its slots that are not fresh.
    zones = Counter((int(key[0]), facedown[tuple(key)]) for key in evidence.facedown_sampling_keys
                    if tuple(key) not in fresh)
    if any(zones[(int(location), int(code))] < least
           for (location, code), least in Counter(evidence.unpositioned_facedown).items()):
        return "unpositioned_identity"
    witnesses = sampled + [(int(location), int(code)) for location, code in evidence.fixed_field_identities]
    if not _matched([(int(location), frozenset(codes)) for location, codes in evidence.unpositioned_facedown_categories],
                    witnesses):
        return "unpositioned_category"
    layout = draw.layout()
    if not all(claim.honored_by(layout) for claim in zone_claims):
        return "zone_claim"
    return None


__all__ = ["CONTINUED_LAW", "Continuation", "Kept", "LineTrace", "Waypoint", "capture_world", "kept_world_breach",
           "path_digest", "public_tokens"]
