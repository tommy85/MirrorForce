"""Opponent answers read from the public stream.

The opponent's response to a prompt never reaches the seat, but what it did
does: the packets the seat receives next name the card the answer chose (its
code and place), the effect (its description), the zone and the position. For
an opponent prompt of the local duel this module reads, from the packets
received after it, which option the real opponent took -- a lookup in the
local prompt's own records, not a replay of candidate answers.

A reading gives the evidenced answer bytes, or the identities the batch start
must hold first (the evidenced card is a placeholder in the local duel, so
its option is missing or differs), or a reason no answer could be read (the
consequence is script specific, or comes only after the seat's next prompt);
the follower's ordinary answer search then takes over and is counted.

Only received packets are read; nothing here touches the local core.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

from ..netduel import constants as C
from ..puzzle.single import RESPONSE_REQUIRED

INFORMATION_SET_SEARCH = True
#: Packets that act on no card: prompts of others, hints, and card hints (a summon's materials are followed by one)
_SKIPPED = frozenset({C.MSG_WAITING, C.MSG_HINT, C.MSG_CARD_HINT})
#: Zones whose cards are listed in order: a card leaving shifts the later ones down.
_LIST_ZONES = frozenset({C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_GRAVE, C.LOCATION_REMOVED,
                         C.LOCATION_EXTRA})
_PLACEHOLDERS = frozenset((999000001, 999000002, 999000003, 999000004))
#: The set proxies (``client_shadow.BLANK_SSET``/``BLANK_FIELD``): a hidden hand card set in the spell/trap zones
#: shows only where it went, and the proxy is what could be set there.
_SET_PROXY, _FIELD_PROXY = 999000003, 999000004
_PHASE_BATTLE_START, _PHASE_MAIN2, _PHASE_END = 0x08, 0x100, 0x200
#: ``HINT_EVENT`` values the core writes to the other player when the turn player closes a Main Phase (idle
#: command answers 6 and 7) or the Battle Step (battle command answers 2 and 3), before that player's chain window
_MAIN_END_HINT, _BATTLE_END_HINT = 23, 29
_HINT_OPSELECTED = 4


@dataclass(frozen=True)
class Reading:
    """What the received packets say about one opponent prompt."""

    answers: tuple = ()
    placements: tuple = ()
    reason: str = ""
    #: the answers are the classes the packets up to here cannot tell apart; their own consequences, played on
    #: to the seat's next prompt, must
    by_consequence: bool = False
    #: those answers are different decisions (not the same one on other hidden objects): every one whose consequences
    #: fit stays an open decision, which a later packet tells apart
    open: bool = False


def _int(value: int) -> bytes:
    return struct.pack("<i", value)


def _code(raw: bytes, offset: int) -> int:
    return struct.unpack_from("<I", raw, offset)[0] & 0x7FFFFFFF


def _own_prompt(packet: bytes, viewer: int) -> bool:
    if not packet or packet[0] not in RESPONSE_REQUIRED or len(packet) < 3:
        return False
    return packet[2 if packet[0] == C.MSG_SELECT_SUM else 1] == viewer


def window(packets, begin: int, viewer: int):
    """The received packets from ``packets[begin]``, the first an opponent prompt can explain, up to the seat's next
    prompt, and whether that prompt was reached.

    The WAITINGs among them show only that the opponent was asked something, not what: the server asks it prompts
    the local duel never asks (only its hidden cards open them), so which WAITING belongs to which local prompt is
    not known in general (plan 5.10). Their count bounds it: a chain prompt whose window shows a single WAITING
    before the link it offers was answered by that link (``_chain``)."""
    out = []
    for packet in packets[begin:]:
        if _own_prompt(packet, viewer):
            return out, True
        out.append(packet)
    return out, False


class _Arrivals:
    """Where cards moved within the window: a card acting at a place came from the source of its move there."""

    def __init__(self):
        self.source = {}

    def see(self, packet):
        if packet[0] == C.MSG_MOVE and len(packet) >= 13:
            self.source[tuple(packet[9:12])] = (tuple(packet[5:8]), _code(packet, 1))

    def origin(self, place, code):
        found = self.source.get(tuple(place))
        return (found[0], found[1] or code) if found else (tuple(place), code)


def _flips_to_activate(events, index) -> bool:
    """A face-down card turned face up by its own activation: the chaining at its place follows."""
    place = tuple(events[index][5:8])
    for packet in events[index + 1:]:
        if packet[0] in _SKIPPED:
            continue
        return packet[0] == C.MSG_CHAINING and len(packet) >= 16 and tuple(packet[5:8]) == place
    return False


def _action(events):
    """The first card action of an idle or battle command: ``(kind, origin place, code, description)``."""
    arrivals = _Arrivals()
    for index, packet in enumerate(events):
        msg = packet[0]
        if msg in _SKIPPED:
            continue
        if msg == C.MSG_MOVE:
            arrivals.see(packet)
            continue
        if msg == C.MSG_POS_CHANGE and _flips_to_activate(events, index):
            continue
        if msg in (C.MSG_SUMMONING, C.MSG_SPSUMMONING) and len(packet) >= 9:
            origin, code = arrivals.origin(packet[5:8], _code(packet, 1))
            return ("summon" if msg == C.MSG_SUMMONING else "spsummon"), origin, code, None
        if msg == C.MSG_FLIPSUMMONING and len(packet) >= 9:
            return "repos", tuple(packet[5:8]), _code(packet, 1), None
        if msg == C.MSG_POS_CHANGE and len(packet) >= 9:
            return "repos", tuple(packet[5:8]), _code(packet, 1), None
        if msg == C.MSG_SET and len(packet) >= 9:
            place = tuple(packet[5:8])
            origin, code = arrivals.origin(place, _code(packet, 1))
            return ("mset" if place[1] == C.LOCATION_MZONE else "sset"), origin, code, None
        if msg == C.MSG_CHAINING and len(packet) >= 16:
            origin, code = arrivals.origin(packet[5:8], _code(packet, 1))
            return "activate", origin, code, struct.unpack_from("<I", packet, 12)[0]
        if msg == C.MSG_ATTACK and len(packet) >= 5:
            return "attack", tuple(packet[1:4]), 0, None
        if msg == C.MSG_NEW_PHASE and len(packet) >= 3:
            return "phase", None, struct.unpack_from("<H", packet, 1)[0], None
        if msg == C.MSG_SHUFFLE_HAND:
            return "shuffle", None, 0, None
        return "other", None, msg, None
    return None


def _records(payload: bytes, offset: int, count: int, width: int):
    out = []
    for i in range(count):
        raw = payload[offset + i * width:offset + (i + 1) * width]
        out.append((_code(raw, 0), tuple(raw[4:7]), struct.unpack_from("<I", raw, 7)[0] if width >= 11 else None))
    return out, offset + count * width


def _pick(records, origin, code, description=None):
    """The record acting from ``origin`` (and with ``description``), or the identity it needs there first."""
    at = [i for i, (record_code, place, desc) in enumerate(records)
          if place == origin and (description is None or desc == description)]
    known = [i for i in at if not code or records[i][0] == code]
    if known:
        return Reading(answers=(known[0],))
    if code:
        # The local card at that place is a placeholder or another object (a shuffled hand holds its known cards
        # anywhere): the evidenced identity goes there first, and the menu is written again.
        return Reading(placements=((*origin, code),))
    return Reading(reason="the evidenced card has no option at its place")


def _event_hint(packet: bytes, value: int) -> bool:
    return (len(packet) >= 7 and packet[0] == C.MSG_HINT and packet[1] == C.HINT_EVENT
            and struct.unpack_from("<I", packet, 3)[0] == value)


def _asked_aside(events):
    """The events without the WAITINGs: which prompts the opponent was asked shows nothing a reading may use."""
    return [packet for packet in events if packet[0] != C.MSG_WAITING]


def _new_phase(packet: bytes):
    return struct.unpack_from("<H", packet, 1)[0] if len(packet) >= 3 and packet[0] == C.MSG_NEW_PHASE else None


def _main_exit(events, bp: int, ep: int) -> Reading:
    """The idle command closed the Main Phase (``events`` open with its end hint): 6, the Battle Phase, or 7, the
    End Phase. Both write the same window; then ``process_turn`` step 9 writes the Battle Phase's
    ``MSG_NEW_PHASE`` for 6 and goes to the End Phase for 7. A chain in the window gives the turn player a new idle
    command instead, which makes the two answers lead to the same duel."""
    phases = [answer for answer, allowed in ((6, bp), (7, ep)) if allowed]
    for packet in events[1:]:
        if packet[0] == C.MSG_CHAINING:
            return Reading(answers=(_int(phases[0]),))
        phase = _new_phase(packet)
        if phase is not None or packet[0] == C.MSG_NEW_TURN:
            answer = 6 if phase == _PHASE_BATTLE_START else 7
            return Reading(answers=(_int(answer),)) if answer in phases else Reading(reason="unexpected phase")
    # The phase entered shows only after the seat's chain window: each answer stays open until then.
    return Reading(answers=tuple(_int(answer) for answer in phases))


def _battle_exit(events, m2: int, ep: int) -> Reading:
    """The battle command closed the Battle Step (``events`` open with its end hint, WAITINGs aside): 2, Main Phase
    2, or 3, the End Phase. Both end the Battle Phase through the same packets and enter Main Phase 2
    (``process_turn`` steps 12 and 14); 3 only sets ``skip_m2``, which the Main Phase 2 idle command reads at once:
    unset, the turn player is asked; set, the phase goes straight to its end hint. So the Main Phase's end hint
    right after it is 3, or 2 with the idle command closing the phase at once, the same duel: 3 when offered. Any
    other packet there is the turn player acting in Main Phase 2: 2. A second Battle Phase or a skipped Main Phase 2
    makes the two answers lead to the same duel."""
    phases = [answer for answer, allowed in ((2, m2), (3, ep)) if allowed]
    for index in range(1, len(events)):
        phase = _new_phase(events[index])
        if phase is None:
            continue
        if phase != _PHASE_MAIN2:
            return Reading(answers=(_int(phases[0]),))
        if index + 1 == len(events):
            break
        closed = _event_hint(events[index + 1], _MAIN_END_HINT)
        answer = (3 if ep else 2) if closed else 2
        return Reading(answers=(_int(answer),)) if answer in phases else Reading(reason="unexpected Battle Phase exit")
    # Main Phase 2 shows which only after the seat's chain window: each answer stays open until then.
    return Reading(answers=tuple(_int(answer) for answer in phases))


def _idle(payload: bytes, events, reached: bool) -> Reading:
    offset, groups = 1, []
    for group in range(6):
        records, offset = _records(payload, offset + 1, payload[offset], 11 if group == 5 else 7)
        groups.append(records)
    bp, ep, shuffle = payload[offset:offset + 3]
    events = _asked_aside(events)
    if events and _event_hint(events[0], _MAIN_END_HINT) and (bp or ep):
        return _main_exit(events, bp, ep)
    found = _action(events)
    if found is None:
        return Reading(reason="no consequence of the command before the seat's next prompt")
    kind, origin, code, description = found
    if kind == "phase":
        return Reading(reason="a phase change without the Main Phase's end window")
    if kind == "shuffle":
        return Reading(answers=(_int(8),)) if shuffle else Reading(reason="hand shuffle not offered")
    group = {"summon": 0, "spsummon": 1, "repos": 2, "mset": 3, "sset": 4, "activate": 5}.get(kind)
    if group is None:
        return Reading(reason="the command's first consequence is not an action (%s)" % kind)
    reading = _pick(groups[group], origin, code, description if group == 5 else None)
    if kind in ("mset", "sset") and not code and not reading.answers:
        # A face-down set from a shuffled hand: which hidden object left is not public; any settable blank did.
        blanks = [i for i, (record_code, place, _desc) in enumerate(groups[group])
                  if place[:2] == origin[:2] and record_code in _PLACEHOLDERS]
        if blanks:
            reading = Reading(answers=(blanks[0],))
        elif kind == "sset" and origin[1] == C.LOCATION_HAND:
            # A hand blank is a monster placeholder, which cannot be set there: the set proxy of the zone the card
            # went to stands in for it (its printed identity stays open).
            destination = next((packet for packet in events if packet[0] == C.MSG_SET and len(packet) >= 9), None)
            proxy = _FIELD_PROXY if destination is not None and destination[7] == 5 else _SET_PROXY
            return Reading(placements=((*origin, proxy),))
    if reading.answers:
        return Reading(answers=(_int(reading.answers[0] << 16 | group),))
    return reading


def _battle(payload: bytes, events, reached: bool) -> Reading:
    activations, offset = _records(payload, 2, payload[1], 11)
    attacks, offset = _records(payload, offset + 1, payload[offset], 8)
    m2, ep = payload[offset:offset + 2]
    events = _asked_aside(events)
    if events and _event_hint(events[0], _BATTLE_END_HINT) and (m2 or ep):
        return _battle_exit(events, m2, ep)
    # An attack declared is the command's (an attack given up at its target selection leaves nothing behind, and
    # asks the command again: the one declared is what the turn player did from here).
    found = _action(events)
    if found is None:
        return Reading(reason="no consequence of the command before the seat's next prompt")
    kind, origin, code, description = found
    if kind == "phase":
        return Reading(reason="a phase change without the Battle Step's end window")
    if kind == "attack":
        at = [i for i, (_code_, place, _desc) in enumerate(attacks) if place == origin]
        return Reading(answers=(_int(at[0] << 16 | 1),)) if at else Reading(reason="attacker not offered")
    if kind == "activate":
        reading = _pick(activations, origin, code, description)
        return Reading(answers=(_int(reading.answers[0] << 16),)) if reading.answers else reading
    return Reading(reason="the command's first consequence is not an action (%s)" % kind)


def _leads_to_activation(lead, place) -> bool:
    """Whether the packets before a MSG_CHAINING at ``place`` are its own activation's: nothing, the activated card
    moving to where it is activated, or a set card turning face up (hints and prompts aside). Anything else
    (another card's move, say) came from elsewhere."""
    lead = [packet for packet in lead if packet[0] not in (C.MSG_HINT, C.MSG_WAITING)]
    arrives = lambda packet: packet[0] == C.MSG_MOVE and len(packet) >= 13 and tuple(packet[9:12]) == place
    if not lead:
        return True
    return len(lead) == 1 and (arrives(lead[0]) or lead[0][0] == C.MSG_POS_CHANGE and tuple(lead[0][5:8]) == place)


#: ``SELECT_CHAIN`` entry kinds whose choice chains nothing (``EDESC_OPERATION``: a field-only effect's operation;
#: ``EDESC_RESET``: an effect's reset, such as control lent until the End Phase)
_UNCHAINED = frozenset((1, 2))


def _chain(payload: bytes, events, viewer: int) -> Reading:
    count, offset = payload[1], 11
    records, kinds = [], []
    for i in range(count):
        raw = payload[offset + 14 * i:offset + 14 * (i + 1)]
        records.append((_code(raw, 2), tuple(raw[6:9]), struct.unpack_from("<I", raw, 10)[0], raw[1]))
        kinds.append(raw[0])
    arrivals = _Arrivals()
    for index, packet in enumerate(events):
        if packet[0] in _SKIPPED:
            continue
        if packet[0] == C.MSG_MOVE and len(packet) >= 13 and packet[9] == 1 - viewer:
            arrivals.see(packet)
            continue
        if packet[0] == C.MSG_POS_CHANGE and len(packet) >= 9 and _flips_to_activate(events, index):
            continue
        if packet[0] == C.MSG_CHAINING and len(packet) >= 16:
            if not _leads_to_activation(events[:index], tuple(packet[5:8])):
                break  # the chain link came from a later prompt: this one was passed
            origin, code = arrivals.origin(packet[5:8], _code(packet, 1))
            description = struct.unpack_from("<I", packet, 12)[0]
            reading = _pick([(record_code, place, desc) for record_code, place, desc, _forced in records],
                            origin, code, description)
            if reading.answers:
                if any(forced for *_rest, forced in records):
                    return Reading(answers=(_int(reading.answers[0]),))
                if sum(item[0] == C.MSG_WAITING for item in events[:index]) == 1:
                    # The server sends one WAITING per prompt it asks the opponent, and the link's MSG_CHAINING
                    # comes before its cost and targets are asked. It asks every prompt the local duel asks (hidden
                    # cards only add windows), so one WAITING before the link is this prompt's own: chained here.
                    return Reading(answers=(_int(reading.answers[0]),))
                # More prompts were asked before the link: chained here after a window only a hidden card opened,
                # or passed here and chained in a later window of the same timing (the Battle Step's end, then the
                # Battle Phase's), which decides where the duel goes after the chain (the battle command again, or
                # Main Phase 2). Both stay open.
                return Reading(answers=(_int(reading.answers[0]), _int(-1)), by_consequence=True, open=True)
            if any(forced for *_rest, forced in records):
                return reading
            # Not offered here: a later prompt chained it (the server asks prompts the local duel does not, and
            # which WAITING is whose is not known), so this one was passed, as its consequences must show; or its
            # legality needs a hidden card (the identities to write, or a construction point).
            return Reading(answers=(_int(-1),), placements=reading.placements, reason=reading.reason,
                           by_consequence=True)
        break
    if any(forced for *_rest, forced in records):
        # A forced entry may not be passed. With no chain link after the prompt, the one taken is an entry that
        # chains nothing: a field-only operation or an effect's reset (control lent until the End Phase returns).
        quiet = [index for index, kind in enumerate(kinds) if kind in _UNCHAINED]
        if not quiet:
            return Reading(reason="a forced chain link was not activated")
        return Reading(answers=tuple(_int(index) for index in quiet), by_consequence=len(quiet) > 1)
    return Reading(answers=(_int(-1),))


#: ``SELECT_EFFECTYN`` descriptions the core asks with when a player's only chain candidate is optional: 0, a quick
#: effect in a chain window (``process_quick_effect``); 221, a trigger (``process_point_event``). Taken, it is
#: chained at once.
_OPTIONAL_ACTIVATIONS = frozenset((0, 221))
#: ``SELECT_YESNO`` 31: the core's question whether an attacker that may attack directly does so (battle step 4);
#: 30: whether an attack whose target changed goes on (battle step 11)
_DIRECT_ATTACK, _REPLAY_ATTACK = 31, 30


def _effect_yes_no(payload: bytes, events, reached: bool) -> Reading:
    code, place, description = _code(payload, 1), tuple(payload[5:8]), struct.unpack_from("<I", payload, 9)[0]
    for index, packet in enumerate(events):
        if packet[0] in _SKIPPED:
            continue
        if packet[0] == C.MSG_POS_CHANGE and _flips_to_activate(events, index):
            continue
        if packet[0] == C.MSG_CHAINING and len(packet) >= 16 and _code(packet, 1) == code \
                and tuple(packet[5:8]) == place:
            return Reading(answers=(_int(1),))
        break
    else:
        if not reached:
            return Reading(reason="the packets after the question are not received yet")
    if description in _OPTIONAL_ACTIVATIONS:
        # Declined: the chain link would have been the first packet.
        return Reading(answers=(_int(0),))
    return Reading(reason="an optional effect's use shows only in its script's operation")


def _replay(events, attacker) -> Reading:
    """Yes: the attack goes on, declared again by the same attacker (at once, or after its new target is chosen).
    No: the battle command is asked again, and what it does next is the Battle Step's end or another monster's
    attack (the attacker has attacked)."""
    for packet in events:
        if packet[0] == C.MSG_WAITING or packet[0] == C.MSG_HINT and not _event_hint(packet, _BATTLE_END_HINT):
            continue
        if packet[0] == C.MSG_ATTACK and len(packet) >= 9 and attacker is not None:
            return Reading(answers=(_int(1 if tuple(packet[1:4]) == attacker else 0),))
        if _event_hint(packet, _BATTLE_END_HINT):
            return Reading(answers=(_int(0),))
        break
    return Reading(reason="the attack's replay shows neither the same attack nor the battle command")


def _yes_no(payload: bytes, events, packets, begin: int) -> Reading:
    description = struct.unpack_from("<I", payload, 1)[0]
    if description == _REPLAY_ATTACK:
        last = next((packet for packet in reversed(packets[:begin]) if packet[0] == C.MSG_ATTACK and len(packet) >= 9),
                    None)
        return _replay(events, tuple(last[1:4]) if last is not None else None)
    if description == _DIRECT_ATTACK:
        # Yes: the attack is declared with no target. No: with the target then asked.
        for packet in events:
            if packet[0] in (C.MSG_HINT, C.MSG_WAITING):
                continue
            if packet[0] == C.MSG_ATTACK and len(packet) >= 9:
                return Reading(answers=(_int(1 if packet[5:9] == bytes(4) else 0),))
            break
    return Reading(reason="no reading for the yes/no question %d" % description)


def _option(payload: bytes, events) -> Reading:
    options = [struct.unpack_from("<I", payload, 2 + 4 * i)[0] for i in range(payload[1])]
    for packet in events:
        if packet[0] == C.MSG_HINT and len(packet) >= 7 and packet[1] == _HINT_OPSELECTED:
            chosen = struct.unpack_from("<I", packet, 3)[0]
            if chosen in options:
                return Reading(answers=(_int(options.index(chosen)),))
    return Reading(reason="no selected-option hint")


def _acted_places(events, viewer: int):
    """The cards the first public operation acts on: [(place, code)] in order (list-zone places as they were
    before the operation's own moves), or None. A hand the operation shuffled first (a card chosen, the hand
    shuffled, then the card moved) names no chosen slot: its places carry sequence -1."""
    shuffled = set()
    for index, packet in enumerate(events):
        msg = packet[0]
        if msg in _SKIPPED:
            continue
        if msg == C.MSG_SHUFFLE_HAND and len(packet) >= 2:
            shuffled.add((packet[1], C.LOCATION_HAND))
            continue
        if msg == C.MSG_BECOME_TARGET and len(packet) >= 2:
            out, end = [], index
            for position in range(index, len(events)):
                later = events[position]
                if later[0] == C.MSG_WAITING:
                    break  # the targets of one selection are written together; another prompt picks the next
                if later[0] in _SKIPPED:
                    continue
                if later[0] != C.MSG_BECOME_TARGET or len(later) != 2 + 4 * later[1]:
                    break
                out.extend((tuple(later[2 + 4 * i:5 + 4 * i]), 0) for i in range(later[1]))
                end = position
            after = next((position for position in range(end + 1, len(events))
                          if events[position][0] not in (C.MSG_HINT, C.MSG_CARD_HINT)), None)
            if after is not None and events[after][0] in (C.MSG_CONFIRM_CARDS, C.MSG_MOVE):
                # Shown while an effect resolves (a selection hint, not a chain link's targets): the operation that
                # follows acts on every chosen card, those the hint shows and those it cannot (a hand card, say).
                shown = {place for place, _code in out}
                out.extend(item for item in (_acted_places(events[after:], viewer) or []) if item[0] not in shown)
            return out
        if msg == C.MSG_CONFIRM_CARDS and len(packet) >= 4 and len(packet) == 4 + 7 * packet[3]:
            return [(tuple(packet[offset + 4:offset + 7]), _code(packet, offset)) for offset in range(4, len(packet), 7)]
        if msg == C.MSG_MOVE and len(packet) >= 13:
            out, removed, landed = [], {}, []
            for later in events[index:]:
                # A card set face down writes its MSG_SET right after its move; the operation goes on.
                if later[0] in _SKIPPED or later[0] == C.MSG_SET:
                    continue
                if later[0] != C.MSG_MOVE or len(later) < 13:
                    break
                controller, location, sequence = later[5:8]
                zone = (controller, location)
                if location in _LIST_ZONES:
                    for gone in sorted(removed.get(zone, ())):
                        if gone <= sequence:
                            sequence += 1
                    removed.setdefault(zone, []).append(sequence)
                if (controller, location) in shuffled:
                    sequence = -1
                out.append(((controller, location, sequence), _code(later, 1)))
                landed.append(tuple(later[9:12]))
            # A card moved face down (to a hand, say) is named by the confirmation of where it landed.
            shown = _confirmed(events[index:])
            return [(place, code or shown.get(spot, 0)) for (place, code), spot in zip(out, landed)]
        if msg in (C.MSG_POS_CHANGE, C.MSG_SPSUMMONING, C.MSG_SUMMONING, C.MSG_FLIPSUMMONING) and len(packet) >= 9:
            return [(tuple(packet[5:8]), _code(packet, 1))]
        if msg == C.MSG_ATTACK and len(packet) >= 9:
            # An attack target chosen from the listed monsters (the attacker's own command named the attacker).
            return [(tuple(packet[5:8]), 0)]
        return None
    return None


def _confirmed(events, limit: int = 16):
    """``{place: code}`` of the cards the next confirmation shows."""
    for packet in events[:limit]:
        if packet[0] == C.MSG_CONFIRM_CARDS and len(packet) >= 4 and len(packet) == 4 + 7 * packet[3]:
            return {tuple(packet[offset + 4:offset + 7]): _code(packet, offset) for offset in range(4, len(packet), 7)}
    return {}


def _cards(payload: bytes, count: int, offset: int, width: int):
    """(code, place) of each listed card; a deck card's place is a running number, not its deck slot."""
    return [(_code(payload, offset + width * i), tuple(payload[offset + width * i + 4:offset + width * i + 7]))
            for i in range(count)]


def _chosen(candidates, acted, lowest: int, highest: int):
    """Indices of the listed cards the operation acted on, or a Reading when they cannot be named."""
    chosen, placements = [], []
    for place, code in acted:
        if place[1] == C.LOCATION_DECK or place[2] < 0:
            # Deck cards are listed by a running number, and a shuffled hand's slots name no card: a public identity
            # names one; with none, the zone's cards are interchangeable to the seat (placeholders first, so no
            # identity is taken where none was shown).
            zone = [i for i, (listed, spot) in enumerate(candidates) if spot[:2] == place[:2] and i not in chosen]
            matches = [i for i in zone if candidates[i][0] == code] if code \
                else [i for i in zone if candidates[i][0] in _PLACEHOLDERS] + [i for i in zone
                                                                                 if candidates[i][0] not in _PLACEHOLDERS]
            if not matches and code:
                placements.append((*place, code))
                continue
        else:
            matches = [i for i, (listed, spot) in enumerate(candidates) if spot == place and i not in chosen]
            if matches and code and candidates[matches[0]][0] != code:
                placements.append((*place, code))
                continue
        if not matches:
            break  # the operation went on to cards the prompt did not list
        chosen.append(matches[0])
    if placements:
        return Reading(placements=tuple(placements))
    if not lowest <= len(chosen) <= highest:
        return Reading(reason="the first operation acts on %d listed cards, the prompt takes %d to %d"
                       % (len(chosen), lowest, highest))
    return chosen


def _select_cards(msg: int, payload: bytes, events, viewer: int) -> Reading:
    lowest, highest, count = payload[2], payload[3], payload[4]
    candidates = _cards(payload, count, 5, 8)
    acted = _acted_places(events, viewer)
    chosen = _chosen(candidates, acted, lowest, highest) if acted is not None \
        else Reading(reason="no operation on listed cards before the seat's next prompt")
    if isinstance(chosen, Reading):
        if chosen.placements:
            return chosen
        if acted is not None and highest == 1:
            # One card per prompt, then one operation on all of them: which prompt took which card the operation's
            # order (the core's object order) does not show, and either way the same cards are taken.
            every = _chosen(candidates, acted, 0, 255)
            if not isinstance(every, Reading) and len(every) > 1:
                return Reading(answers=tuple(bytes([1, index]) for index in every), by_consequence=True)
        if payload[1]:
            # Nothing acts on the listed cards as the prompt allows: a cancelable selection may have been given up
            # and the prompt before it asked again (an attack target, a tribute set for a summon). Only its
            # consequences can tell.
            return Reading(answers=(_int(-1),), by_consequence=True)
        return chosen
    answers = [bytes([len(chosen), *chosen])]
    for position, (place, code) in enumerate(acted[:len(chosen)]):
        if code or not (place[1] == C.LOCATION_DECK or place[2] < 0):
            continue
        # A pick the packets never show: placeholders are interchangeable, but a known card is not -- whether it
        # stayed shows later (the hand's shuffle at the chain's end, a later reveal).
        known = {}
        for index, (listed, spot) in enumerate(candidates):
            if spot[:2] == place[:2] and listed not in _PLACEHOLDERS and index not in chosen:
                known.setdefault(listed, index)
        for index in known.values():
            alternative = list(chosen)
            alternative[position] = index
            answers.append(bytes([len(alternative), *alternative]))
    # A selection given up and asked again for the same cards is the same duel: the cards acted on are the answer.
    return Reading(answers=tuple(answers), by_consequence=len(answers) > 1)


def _unselect(payload: bytes, events, viewer: int) -> Reading:
    finishable, lowest, highest = payload[1], payload[3], payload[4]
    selectable_count = payload[5]
    selectable = _cards(payload, selectable_count, 6, 8)
    offset = 6 + 8 * selectable_count
    selected = _cards(payload, payload[offset], offset + 1, 8)
    acted = _acted_places(events, viewer)
    if acted is None:
        return Reading(reason="no operation on listed cards before the seat's next prompt")
    everyone = selectable + selected
    chosen = _chosen(everyone, acted, 0, 255)
    if isinstance(chosen, Reading):
        return chosen
    pending = [i for i in chosen if i < len(selectable)]
    if pending:
        return Reading(answers=(bytes([1, pending[0]]),))
    if finishable and lowest <= len(selected) <= max(highest, len(selected)):
        return Reading(answers=(_int(-1),))
    return Reading(reason="the acted-on cards are selected but the prompt cannot finish")


def _place(payload: bytes, events) -> Reading:
    count, flag = payload[1], struct.unpack_from("<I", payload, 2)[0]
    player = payload[0]
    places = []
    for packet in events:
        # Several cards placed one after another are asked their zones first, then moved in that order, each set
        # card followed by its MSG_SET; a zone already chosen is not offered again.
        if packet[0] in _SKIPPED or packet[0] == C.MSG_SET:
            continue
        if packet[0] != C.MSG_MOVE or len(packet) < 13:
            break
        controller, location, sequence = packet[9:12]
        if location in (C.LOCATION_MZONE, C.LOCATION_SZONE):
            bit = 1 << (sequence + (0 if controller == player else 16) + (0 if location == C.LOCATION_MZONE else 8))
            if not bit & flag:
                places.append(bytes([controller, location, sequence]))
        if len(places) == max(1, count):
            return Reading(answers=(b"".join(places),))
    return Reading(reason="no move to an offered zone")


def _position(payload: bytes, events) -> Reading:
    code, positions = _code(payload, 1), payload[5]
    for packet in events:
        if packet[0] in _SKIPPED:
            continue
        if packet[0] == C.MSG_MOVE and len(packet) >= 13 and _code(packet, 1) in (code, 0):
            position = packet[12]
        elif packet[0] in (C.MSG_SPSUMMONING, C.MSG_SUMMONING) and len(packet) >= 9:
            position = packet[8]
        else:
            continue
        if position in (1, 2, 4, 8) and position & positions:
            return Reading(answers=(_int(position),))
        break
    return Reading(reason="no placement of the card in an offered position")


def settles(msg: int, packets, begin: int, index: int, real: bytes, local: bytes) -> bool:
    """Whether a divergence at ``packets[index]`` is the packet that tells apart the open phase exit of the ``msg``
    command read from ``packets[begin]`` on: the first phase change after it (the Main Phase's end window led to
    the Battle or the End Phase), or the first packet of the Main Phase 2 that followed it."""
    changes = [position for position in range(begin, index)
               if packets[position][:1] in (bytes([C.MSG_NEW_PHASE]), bytes([C.MSG_NEW_TURN]))]
    if msg == C.MSG_SELECT_IDLECMD:
        return not changes and any(packet[:1] in (bytes([C.MSG_NEW_PHASE]), bytes([C.MSG_NEW_TURN]))
                                   for packet in (real, local))
    if msg == C.MSG_SELECT_BATTLECMD:
        # Only WAITINGs may stand between (the follower skips them).
        return len(changes) == 1 and _new_phase(packets[changes[0]]) == _PHASE_MAIN2 and all(
            packets[position][:1] == bytes([C.MSG_WAITING]) for position in range(changes[0] + 1, index))
    return False


def read(msg: int, payload: bytes, packets, begin: int, viewer: int) -> Reading:
    """The opponent's answer to the local prompt ``msg``/``payload``, read from ``packets[begin]`` on (the first
    received packet the prompt's answer can explain, :func:`window`)."""
    events, reached = window(packets, begin, viewer)
    try:
        if msg == C.MSG_SELECT_IDLECMD:
            return _idle(payload, events, reached)
        if msg == C.MSG_SELECT_BATTLECMD:
            return _battle(payload, events, reached)
        if msg == C.MSG_SELECT_CHAIN:
            return _chain(payload, events, viewer)
        if msg == C.MSG_SELECT_EFFECTYN:
            return _effect_yes_no(payload, events, reached)
        if msg == C.MSG_SELECT_YESNO:
            return _yes_no(payload, events, packets, begin)
        if msg == C.MSG_SELECT_OPTION:
            return _option(payload, events)
        if msg in (C.MSG_SELECT_CARD, C.MSG_SELECT_TRIBUTE):
            return _select_cards(msg, payload, events, viewer)
        if msg == C.MSG_SELECT_UNSELECT_CARD:
            return _unselect(payload, events, viewer)
        if msg == C.MSG_SELECT_PLACE:
            return _place(payload, events)
        if msg == C.MSG_SELECT_POSITION:
            return _position(payload, events)
    except (IndexError, struct.error) as exc:
        return Reading(reason="malformed prompt or packet: %s" % exc)
    return Reading(reason="no reading for this prompt type")
