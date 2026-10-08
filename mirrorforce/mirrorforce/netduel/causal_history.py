"""Public position tokens and explicit anonymous permutation cuts.

These are NOT native card UIDs. A received move/draw carries a token; a public
shuffle makes new position tokens joined by a bijection constraint, not by an
assumed same-index identity. Deck-selection menu numbers are kept opaque.
The complete-hypothesis compiler will solve this graph before native replay.
No native duel, server deal, private target or follower hydration is an input.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import struct

from . import constants as C
from .board import ShadowBoard
from ..puzzle.single import RESPONSE_REQUIRED

INFORMATION_SET_SEARCH = True
SCHEMA = "mirrorforce_public_position_history/v1"
LAW = "public-position-tokens-and-anonymous-permutation-cuts/v1"
LISTS = (C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_EXTRA, C.LOCATION_GRAVE, C.LOCATION_REMOVED)
FIELDS = {C.LOCATION_MZONE: 7, C.LOCATION_SZONE: 8}
TOKEN_TYPE = 0x4000
BLANK_CODES = frozenset((999000001, 999000002, 999000003, 999000004))
PASSIVE = frozenset(getattr(C, name) for name in (
    "MSG_START", "MSG_HINT", "MSG_WAITING", "MSG_WIN", "MSG_UPDATE_DATA", "MSG_UPDATE_CARD",
    "MSG_SET", "MSG_REFRESH_DECK", "MSG_NEW_TURN", "MSG_NEW_PHASE", "MSG_SUMMONED", "MSG_SPSUMMONED",
    "MSG_FLIPSUMMONED", "MSG_CHAINED", "MSG_CHAIN_SOLVING", "MSG_CHAIN_SOLVED", "MSG_CHAIN_END",
    "MSG_CHAIN_NEGATED", "MSG_CHAIN_DISABLED", "MSG_FIELD_DISABLED", "MSG_BECOME_TARGET", "MSG_DAMAGE",
    "MSG_RECOVER", "MSG_EQUIP", "MSG_LPUPDATE", "MSG_UNEQUIP", "MSG_CARD_TARGET", "MSG_CANCEL_TARGET",
    "MSG_PAY_LPCOST", "MSG_ADD_COUNTER", "MSG_REMOVE_COUNTER", "MSG_ATTACK", "MSG_BATTLE", "MSG_ATTACK_DISABLED",
    "MSG_DAMAGE_STEP_START", "MSG_DAMAGE_STEP_END", "MSG_TOSS_COIN", "MSG_TOSS_DICE", "MSG_HAND_RES",
    "MSG_CARD_HINT", "MSG_PLAYER_HINT", "MSG_MISSED_EFFECT"))


class HistoryError(ValueError):
    pass


class HistoryUnsupported(HistoryError):
    pass


@dataclass(frozen=True)
class PositionToken:
    token: int
    created_at: int
    pools: tuple  # possible opening or observed-birth groups; never a guessed identity


@dataclass(frozen=True)
class IdentityFact:
    token: int
    code: int
    packet: int
    source: str


@dataclass(frozen=True)
class PermutationCut:
    packet: int
    msg: int
    before: tuple
    after: tuple
    positions: tuple
    player: int
    location: int


class PublicPositionHistory:
    """One observer's received stream. All mutable state is local to this compiler.

    Main decks begin as public multisets, not the blank follower's arbitrary
    initial order. Native refreshes can reflect a whole C batch's final state;
    their identities are therefore bound only at a stable own prompt, never
    naively attached to an earlier MOVE inside that batch.
    """

    def __init__(self, viewer, own_main, own_extra, opponent_main, opponent_extra, *, card_types):
        if type(viewer) is not int or viewer not in (0, 1):
            raise HistoryError("history needs its one registered observer")
        decks = {viewer: (tuple(own_main), tuple(own_extra)), 1 - viewer: (tuple(opponent_main), tuple(opponent_extra))}
        if any(type(code) is not int or code <= 0 or code in BLANK_CODES
               for main, extra in decks.values() for code in (*main, *extra)):
            raise HistoryError("history recipes contain only real public card codes")
        self.viewer, self.decks, self.card_types = viewer, decks, dict(card_types)
        if any(type(code) is not int or type(kind) is not int or code < 0 or kind < 0
               for code, kind in self.card_types.items()):
            raise HistoryError("card types must be the registered public integer catalogue")
        self.catalog_sha256 = hashlib.sha256(json.dumps(sorted(self.card_types.items())).encode()).hexdigest()
        self.tokens, self.facts, self.permutations, self.pools = [], [], [], []
        self.zones = {(p, z): [] for p in (0, 1) for z in LISTS}
        self.zones.update({(p, z): [None] * size for p in (0, 1) for z, size in FIELDS.items()})
        self.dead, self.known, self.responses, self.packets = [], {}, [], []
        self.extra_faceup_tokens = {0: set(), 1: set()}
        self.started, self.pending = False, None
        self.failure = None
        self.board = ShadowBoard()
        self.board.start(viewer, own_main, own_extra)
        for player in (0, 1):
            for location, codes in zip((C.LOCATION_DECK, C.LOCATION_EXTRA), decks[player]):
                pool = (player, location)
                nodes = tuple(self._new(-1, (pool,)) for _ in codes)
                self.zones[pool] = list(nodes)
                self.pools.append({"pool": pool, "tokens": nodes, "counts": tuple(sorted(Counter(codes).items()))})

    def _new(self, packet, pools):
        token = len(self.tokens) + 1
        self.tokens.append(PositionToken(token, packet, tuple(sorted(set(pools)))))
        return token

    def _zone(self, player, location):
        if player not in (0, 1) or (player, location) not in self.zones:
            raise HistoryUnsupported("unsupported public card zone")
        return self.zones[player, location]

    def at(self, player, location, sequence):
        zone = self._zone(player, location)
        if type(sequence) is not int or not 0 <= sequence < len(zone) or zone[sequence] is None:
            raise HistoryError(f"no public position token at {(player, location, sequence)}")
        return zone[sequence]

    def _fact(self, token, code, packet, source):
        code = int(code) & 0x7fffffff
        if not code:
            return
        if code in BLANK_CODES:
            raise HistoryError("a public identity cannot be a follower placeholder")
        if token in self.known and self.known[token] != code:
            raise HistoryUnsupported("an unshuffled public token has conflicting identities or an unsupported transform")
        if token in self.known:
            return
        self.known[token] = code
        self.facts.append(IdentityFact(token, code, packet, source))

    def _take(self, player, location, sequence):
        token = self.at(player, location, sequence)
        if location in FIELDS:
            self.zones[player, location][sequence] = None
        else:
            self.zones[player, location].pop(sequence)
        return token

    def _put(self, player, location, sequence, token):
        if location == 0:
            if not self.card_types.get(self.known.get(token), 0) & TOKEN_TYPE:
                raise HistoryUnsupported("a non-token disappeared from the public position graph")
            self.dead.append(token)
            return
        zone = self._zone(player, location)
        if location in FIELDS:
            if not 0 <= sequence < len(zone) or zone[sequence] is not None:
                raise HistoryError("public field placement overwrites another token")
            zone[sequence] = token
        else:
            if not 0 <= sequence <= len(zone):
                raise HistoryError("public list placement has an invalid sequence")
            zone.insert(sequence, token)

    def _shuffle(self, packet, msg, positions, *, player=None, location=None):
        zones = {place[:2] for place in positions}
        if positions:
            if len(zones) != 1:
                raise HistoryError("one public shuffle cannot mix controllers or zones")
            actual_player, actual_location = zones.pop()
            if player is not None and (player, location) != (actual_player, actual_location):
                raise HistoryError("shuffle position and explicit event coordinates differ")
            player, location = actual_player, actual_location
        if player not in (0, 1) or location not in (1, 2, 4, 8, 64) \
                or not positions and (msg != C.MSG_SHUFFLE_EXTRA or location != C.LOCATION_EXTRA):
            raise HistoryError("an empty shuffle must explicitly name its player's Extra event")
        before = tuple(self.at(*place) for place in positions)
        if len(set(before)) != len(before):
            raise HistoryError("a shuffle repeats a participating token")
        pools = {pool for token in before for pool in self.tokens[token - 1].pools}
        after = tuple(self._new(packet, pools) for _ in positions)
        for place, token in zip(positions, after):
            self.zones[place[:2]][place[2]] = token
        self.permutations.append(PermutationCut(packet, msg, before, after, tuple(positions), player, location))
        return after

    def _stable_view(self, packet):
        """Only this observer's proved slots/current visible cards; never omniscient ledger.counts."""
        for (player, location), zone in self.zones.items():
            if location == C.LOCATION_DECK:
                count = self.board.deck_count[player]
            elif location == C.LOCATION_EXTRA:
                count = self.board.extra_count[player]
            elif location in FIELDS:
                count = len(zone)
            else:
                count = len(self.board.zone(player, location))
            if count != len(zone):
                raise HistoryUnsupported("stable public counts differ from the physical message trace")
            for sequence, token in enumerate(zone):
                if token is None:
                    continue
                code = self.board.disclosure.known_code_at(self.viewer, player, location, sequence)
                cards = self.board.zone(player, location)
                card = cards[sequence] if sequence < len(cards) else None
                if card is not None and (player == self.viewer and location != C.LOCATION_DECK
                                         or location == C.LOCATION_GRAVE or card.position & C.POS_FACEUP):
                    code = card.code or code
                self._fact(token, code, packet, "stable-own-prompt-public-view")

    def feed(self, raw):
        if self.failure is not None:
            raise HistoryError("public position history is permanently invalid: " + self.failure)
        try:
            return self._feed(raw)
        except Exception as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            raise

    def _feed(self, raw):
        if not isinstance(raw, bytes) or not raw:
            raise HistoryError("history takes exact received packet bytes")
        if self.pending is not None:
            raise HistoryError("the own prompt must receive its actual response before later packets")
        index, msg, body = len(self.packets), raw[0], raw[1:]
        if not self.started:
            if msg != C.MSG_START or len(body) != 18 or body[0] != self.viewer:
                raise HistoryError("history must start with this observer's complete MSG_START")
            counts = struct.unpack_from("<HHHH", body, 10)
            if counts != tuple(len(cards) for player in (0, 1) for cards in self.decks[player]):
                raise HistoryError("public starting counts differ from the declared recipes")
            self.started = True
        elif msg == C.MSG_START:
            raise HistoryError("history cannot start twice")
        self.packets.append(raw)
        self.board.apply(msg, body)
        if msg == C.MSG_RETRY:
            raise HistoryError("a rejected actual response cannot enter an accepted history")
        if msg == C.MSG_MOVE:
            if len(body) != 16:
                raise HistoryError("malformed public MOVE")
            code = struct.unpack_from("<I", body)[0]
            source, target = tuple(body[4:7]), tuple(body[8:11])
            if source[1] & C.LOCATION_OVERLAY or target[1] & C.LOCATION_OVERLAY:
                raise HistoryUnsupported("overlay movement requires a parent/material token relation")
            if source[1] == 0:
                if not self.card_types.get(code & 0x7fffffff, 0) & TOKEN_TYPE:
                    raise HistoryUnsupported("public birth is not a proved token")
                birth = (-1, len(self.tokens) + 1)
                token = self._new(index, (birth,))
                self.pools.append({"pool": birth, "tokens": (token,), "counts": ((code & 0x7fffffff, 1),)})
            else:
                token = self._take(*source)
                if source[1] == C.LOCATION_EXTRA:
                    self.extra_faceup_tokens[source[0]].discard(token)
            self._fact(token, code, index, "received-MOVE")
            self._put(*target, token)
            if target[1] == C.LOCATION_EXTRA and body[11] & C.POS_FACEUP:
                self.extra_faceup_tokens[target[0]].add(token)
        elif msg == C.MSG_DRAW:
            if len(body) < 2 or len(body) != 2 + 4 * body[1] or body[0] not in (0, 1):
                raise HistoryError("malformed public DRAW")
            player, count = body[:2]
            for n in range(count):
                token = self._take(player, C.LOCATION_DECK, len(self.zones[player, C.LOCATION_DECK]) - 1)
                self.zones[player, C.LOCATION_HAND].append(token)
                self._fact(token, struct.unpack_from("<I", body, 2 + 4 * n)[0], index, "received-DRAW")
        elif msg in (C.MSG_SHUFFLE_HAND, C.MSG_SHUFFLE_DECK, C.MSG_SHUFFLE_EXTRA):
            if not body or body[0] not in (0, 1):
                raise HistoryError("malformed public shuffle")
            player = body[0]
            location = {C.MSG_SHUFFLE_HAND: C.LOCATION_HAND, C.MSG_SHUFFLE_DECK: C.LOCATION_DECK,
                        C.MSG_SHUFFLE_EXTRA: C.LOCATION_EXTRA}[msg]
            size = len(self.zones[player, location])
            if msg == C.MSG_SHUFFLE_DECK and len(body) != 1 or msg != C.MSG_SHUFFLE_DECK and (
                    len(body) != 2 + 4 * size or body[1] != size):
                raise HistoryError("shuffle count differs from the public zone")
            hidden = size
            if location == C.LOCATION_EXTRA:
                faceup = self.extra_faceup_tokens[player]
                hidden -= len(faceup)
                if hidden < 0 or set(self.zones[player, location][hidden:]) != faceup:
                    raise HistoryUnsupported("public Extra face-up suffix differs from the received position trace")
            self._shuffle(index, msg, [(player, location, s) for s in range(hidden)],
                          player=player, location=location)
            if msg != C.MSG_SHUFFLE_DECK:
                # The owner's vector also names the preserved face-up suffix.
                for n, token in enumerate(self.zones[player, location]):
                    self._fact(token, struct.unpack_from("<I", body, 2 + 4 * n)[0], index, "received-shuffle-vector")
        elif msg == C.MSG_SHUFFLE_SET_CARD:
            if len(body) < 2 or body[0] not in FIELDS or len(body) != 2 + 8 * body[1]:
                raise HistoryError("malformed field shuffle")
            places = [tuple(body[2 + 4 * n:5 + 4 * n]) for n in range(body[1])]
            if any(place[1] != body[0] for place in places):
                raise HistoryError("field shuffle participants differ from its zone")
            # The second half does NOT certify the hidden identity permutation.
            self._shuffle(index, msg, places)
        elif msg == C.MSG_REVERSE_DECK:
            if body:
                raise HistoryError("malformed deck reversal")
            for player in (0, 1):
                self.zones[player, C.LOCATION_DECK].reverse()
        elif msg in (C.MSG_SWAP_GRAVE_DECK, C.MSG_SWAP):
            raise HistoryUnsupported("this multi-zone exchange needs a separately defined token operation")
        elif msg in (C.MSG_CONFIRM_CARDS, C.MSG_CONFIRM_DECKTOP, C.MSG_CONFIRM_EXTRATOP):
            offset = 3 if msg == C.MSG_CONFIRM_CARDS else 2
            if len(body) < offset or len(body) != offset + 7 * body[offset - 1]:
                raise HistoryError("malformed public confirmation")
            for at in range(offset, len(body), 7):
                code = struct.unpack_from("<I", body, at)[0]
                self._fact(self.at(*body[at + 4:at + 7]), code, index, "received-confirmation")
        elif msg == C.MSG_DECK_TOP:
            if len(body) != 6:
                raise HistoryError("malformed public deck-top observation")
            player, depth = body[:2]
            token = self.at(player, C.LOCATION_DECK, len(self.zones[player, C.LOCATION_DECK]) - 1 - depth)
            self._fact(token, struct.unpack_from("<I", body, 2)[0], index, "received-DECK_TOP")
        elif msg in (C.MSG_CHAINING, C.MSG_SUMMONING, C.MSG_SPSUMMONING, C.MSG_FLIPSUMMONING, C.MSG_POS_CHANGE):
            if len(body) < 8:
                raise HistoryError("malformed public card identity event")
            self._fact(self.at(*body[4:7]), struct.unpack_from("<I", body)[0], index, "received-card-event")
        elif msg in RESPONSE_REQUIRED:
            player = body[1] if msg == C.MSG_SELECT_SUM and len(body) >= 2 else body[0] if body else None
            if player != self.viewer:
                raise HistoryError("another player's private prompt is not an observer input")
            self._stable_view(index)
            # In particular, SELECT_CARD/UNSELECT deck numbers are NOT looked
            # up in a physical deck vector. The complete menu stays opaque.
            self.pending = raw
        elif msg not in PASSIVE:
            raise HistoryUnsupported("public message has no declared position semantics: " + str(msg))

    def respond(self, raw):
        if self.failure is not None:
            raise HistoryError("public position history is permanently invalid: " + self.failure)
        try:
            if self.pending is None or not isinstance(raw, bytes) or not raw or len(raw) > 255:
                raise HistoryError("an own pending prompt needs its exact actually submitted response")
            self.board.observe_response(raw)
            self.responses.append((len(self.packets) - 1, raw))
            self.pending = None
        except Exception as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            raise

    def record(self):
        if self.failure is not None or not self.started:
            raise HistoryError("cannot export a rejected public position history")
        cuts = []
        for cut in self.permutations:
            row = asdict(cut)
            if cut.positions:
                # Preserve the existing nonempty-graph wire/hash. Coordinates
                # there are derivable; an empty Extra event needs them explicit.
                row.pop("player")
                row.pop("location")
            cuts.append(row)
        return {"schema": SCHEMA, "law": LAW, "viewer": self.viewer,
                "catalog_sha256": self.catalog_sha256,
                "tokens": [asdict(token) for token in self.tokens], "pools": [dict(pool) for pool in self.pools],
                "permutations": cuts,
                "facts": [asdict(fact) for fact in self.facts],
                "root_slots": [(p, z, s, token) for (p, z), zone in sorted(self.zones.items())
                               for s, token in enumerate(zone) if token is not None],
                "dead": list(self.dead), "pending": self.pending.hex() if self.pending else None,
                "public_prefix_sha256": hashlib.sha256(b"".join(struct.pack("<I", len(p)) + p for p in self.packets)).hexdigest(),
                "own_responses": [(index, raw.hex()) for index, raw in self.responses]}
