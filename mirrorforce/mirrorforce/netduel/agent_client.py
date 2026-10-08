"""One seat's observation in a network room, from the messages that seat receives (client mode).

The env builds an observation from its engine; a client has no engine. ``AgentClientDuel`` keeps the client's card
view with the netduel shadow board (``board.ShadowBoard``: the server's MSG_UPDATE_DATA / MSG_UPDATE_CARD refreshes
merged by slot, moves, counters, materials and the own deck's remaining multiset) and hands it, as card records, to
the native ``ClientDuel`` of the module, which runs the env's own message handlers, history and observation writer
on the same stream. The env's observation laws (``card_view_law`` refreshed_view/v1, ``pending_source_law``
own_decisions/v1) make the env show what a client can build, so a policy trained on the env reads the same arrays
here (tests/test_client_parity.py, tests/test_client_duel.py). Zones the server does not refresh (graveyard,
banished, Extra Deck) show the card database's off-field values, as the engine does.

Usage: ``client = AgentClientDuel(native, seat, main, extra, config)``; ``client.feed(msg, payload)`` for every
STOC_GAME_MSG in order returns the response bytes when the builder answers a prompt itself (one legal row), else
None; ``client.prompt()`` is the pending decision (as ``ScriptedDuel.prompt()``) or None; ``client.observation()``
the arrays; ``client.step(index)`` returns the response bytes, or None while a multi-card selection still collects
its sub-choices; ``client.step_response(bytes)`` answers by response bytes instead (a recorded game's player).
``forced_count()`` counts the prompts answered inside; ``clone()`` copies the whole state.
"""
from __future__ import annotations

import copy
from collections import Counter

from . import constants as C
from .board import ShadowBoard, ShadowCard

#: the zones the native store holds, in the env's order
ZONES = (C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_GRAVE, C.LOCATION_REMOVED,
         C.LOCATION_EXTRA)
#: messages only the server sends (no engine writes them): they update the card view, the native side never sees them
SERVER_ONLY = frozenset({C.MSG_START, C.MSG_WAITING, C.MSG_UPDATE_DATA, C.MSG_UPDATE_CARD})


def _packed(location: tuple[int, int, int] | None) -> int:
    if location is None:
        return 0
    controller, place, sequence = location
    return controller | (place << 8) | (sequence << 16)


class AgentClientDuel:
    def __init__(self, native, seat: int, main, extra, config: dict, *, opponent_main=None, opponent_extra=None):
        if seat not in (0, 1):
            raise ValueError("a seat is 0 or 1")
        self.seat = seat
        self.main = [int(code) for code in main]
        self.extra = [int(code) for code in extra]
        public = config.get("public_opponent_recipe", False)
        self.public_opponent_recipe = None
        kwargs = {}
        if public:
            from .agent_public_recipe import declare
            if opponent_main is None or opponent_extra is None:
                raise ValueError("the client's public opponent recipe must be explicitly declared")
            self.public_opponent_recipe = declare(opponent_main, opponent_extra)
            kwargs = {"opponent_main": self.public_opponent_recipe["main"],
                      "opponent_extra": self.public_opponent_recipe["extra"]}
        elif opponent_main is not None or opponent_extra is not None:
            raise ValueError("opponent recipe supplied in closed-decklist mode")
        self.duel = native.ClientDuel(seat, sorted(self.main), list(self.extra), dict(config), **kwargs)
        self.board = ShadowBoard()
        self.started = False
        self.forced = 0

    @classmethod
    def from_root_view(cls, native, view, config: dict) -> "AgentClientDuel":
        """Start a new opponent observer at a current hypothetical root, with no past events or neural Memory.

        The caller owns/masks the engine and authenticates the root binding. The root's already-issued prompt is
        not fed here. Subsequent host-projected messages go through the ordinary feed/_push path. The policy
        service supplies fresh Memory and ``first=True`` independently of this card/history observer.
        """
        from .current_root_view import CurrentRootView
        checked = CurrentRootView.from_dict(view.to_dict() if isinstance(view, CurrentRootView) else view)
        data = checked.to_dict()
        if not config.get("public_opponent_recipe", False):
            raise ValueError("current-root observation requires an explicit public opponent recipe")
        out = cls(native, data["viewer"], data["main"], data["extra"], config,
                  opponent_main=data["opponent_main"], opponent_extra=data["opponent_extra"])
        # The full root view preserves field-set constraints for ownership and
        # audit. Keep the existing model input projection: anonymous field rows,
        # never a guessed slot identity or a field count mislabelled as Extra.
        types = out.duel.initialize_current_root(checked.to_native_dict())
        board = out.board
        board.our_player = out.seat
        board.turn_player, board.phase = data["turn_player"], data["phase"]
        board.our_deck_start = Counter(out.main)
        remaining = Counter(row[0] for row in data["cards"][out.seat][0])
        board.seen_out_of_deck = board.our_deck_start - remaining
        board.our_extra_start = tuple(out.extra)
        owners = {tuple(row[:4]): row[5] for row in data["material_owners"]}
        seed = data["public_seed"]
        shown = {tuple(row[:3]): row[3] for row in seed["positioned"] + seed["reveals"]}
        for player in (0, 1):
            board.deck_count[player] = len(data["cards"][player][0])
            board.extra_count[player] = len(data["cards"][player][-1])
            for location, rows in zip(ZONES, data["cards"][player]):
                if location == C.LOCATION_DECK:
                    continue  # a multiset is never installed as network deck slots
                zone = []
                for seq, row in enumerate(rows):
                    if row is None:
                        zone.append(None)
                        continue
                    code, _, _, _, position, level, rank, atk, defense, equip, materials, counters, owner, status, \
                        lscale, rscale, link, markers, known = row
                    key = (player, location, seq)
                    code = code or shown.get(key, 0)
                    card = ShadowCard(code=code, type=types.get(code, 0), controller=player, location=location,
                                      sequence=seq, position=position, level=level, rank=rank, attack=atk,
                                      defense=defense, owner=owner, status=status, lscale=lscale, rscale=rscale,
                                      link=link, link_marker=markers, queried=known, hidden=not bool(code),
                                      info_location=player | (location << 8) | (seq << 16) | (position << 24))
                    zone.append(card)
                    board.positions[key] = position
                    if counters:
                        board.counters[key] = dict(counters)
                    if materials:
                        board.materials[key] = [ShadowCard(code=material, type=types.get(material, 0),
                            controller=player, location=location | C.LOCATION_OVERLAY, sequence=seq, position=i,
                            owner=owners[(*key, i)]) for i, material in enumerate(materials)]
                    if equip:
                        board.equip_targets[key] = (equip & 255, (equip >> 8) & 255, (equip >> 16) & 255)
                    if code:
                        board.disclosure.disclose(player, location, code, sequence=seq, audience=1 << out.seat)
                board.zones[(player, location)] = zone
        # This is an explicit current-state seed, not an invented sequence of confirms, moves or turns.
        for player, location, seq, code in seed["positioned"] + seed["reveals"]:
            board.disclosure.disclose(player, location, code, sequence=seq, audience=1 << out.seat)
        for source, target in seed["card_targets"]:
            board.card_targets.setdefault(tuple(source), set()).add(tuple(target))
        board.disclosure._turn = data["turn"]
        context = seed["chain_context"]
        board.resolution.count = context["solving"]
        board.resolution.current = context["resolution"]
        board.disclosure._chain_stack = [0] * context["links"]
        for chain in seed["chain"]:
            board.disclosure._chain_stack[chain["link"] - 1] = chain["code"]
        active = context["chain_link"]
        board.disclosure._last_chaining = board.disclosure._chain_stack[active - 1] if active else 0
        out.started = True
        out._root_view = checked
        return out

    # -- the card view -----------------------------------------------------

    def _record(self, card, player: int, location: int, sequence: int) -> tuple:
        """One card as the native store reads it (a query segment's fields)."""
        key = (player, location, sequence)
        # Xyz materials: of a monster, and of an Xyz Monster still in the Extra Deck while its summon attaches them
        materials = [int(m.code) for m in self.board.materials.get(key, ())]
        counters = sorted(self.board.counters_of(player, location, sequence).items())
        owner = card.owner if card.owner in (0, 1) else player
        return (int(card.code), player, location, sequence, int(card.position) & 0xFF, int(card.level),
                int(card.rank), int(card.attack), int(card.defense), self.board.equip_target_of(player, location, sequence),
                materials, [(int(t), int(n)) for t, n in counters], int(owner), int(card.status), int(card.lscale),
                int(card.rscale), int(card.link), int(card.link_marker), bool(card.queried))

    def _zone_records(self, player: int, location: int) -> list:
        def unknown(code, i):
            return (code, player, location, i, C.POS_FACEDOWN_DEFENSE, 0, 0, 0, 0, 0, [], [], player, 0, 0, 0, 0, 0,
                    False)

        if location == C.LOCATION_DECK:
            count = int(self.board.deck_count[player])
            if player != self.seat:
                return [unknown(0, i) for i in range(count)]
            remaining = sorted(self.board.our_remaining_deck().elements())
            if len(remaining) > count:
                raise RuntimeError(f"own deck: {len(remaining)} known cards for {count} places")
            codes = remaining + [0] * (count - len(remaining))
            return [unknown(code, i) for i, code in enumerate(codes)]
        if location == C.LOCATION_EXTRA and player != self.seat:
            # the opponent's Extra Deck is count-only for a client: its face-up cards (pendulums) are not followed,
            # so a game that puts one there is refused (no pendulum card is in A0's pool)
            if int(self.board.extra_faceup[player]):
                raise RuntimeError("the opponent's Extra Deck holds a face-up card (pendulum): not followed")
            rows = [unknown(0, i) for i in range(int(self.board.extra_count[player]))]
            if hasattr(self, "_root_view"):
                # Public materials can already be attached while their monster is still in the Extra Deck.
                # The hidden host stays anonymous; its public material slots keep their actual order.
                for i, row in enumerate(rows):
                    materials = self.board.materials.get((player, location, i), ())
                    if materials:
                        rows[i] = row[:10] + ([int(card.code) for card in materials],) + row[11:]
            return rows
        rows = []
        for sequence, card in enumerate(self.board.zone(player, location)):
            if card is None:
                rows.append(None)
                continue
            record = self._record(card, player, location, sequence)
            if location in (C.LOCATION_GRAVE, C.LOCATION_REMOVED, C.LOCATION_EXTRA):
                # the server refreshes these zones only on a graveyard swap or an Extra Deck shuffle: a card's stats
                # and status are what it carried from its last refreshed zone, so the engine's off-field values (the
                # card database's, no disabled status) stand instead
                record = record[:13] + (0,) + record[14:18] + (False,)
            rows.append(record)
        return rows

    def _push(self) -> None:
        for player in (0, 1):
            for location in ZONES:
                self.duel.set_cards(player, location, self._zone_records(player, location))

    # -- the stream ----------------------------------------------------------

    def feed(self, msg: int, payload: bytes):
        if msg == C.MSG_START:
            self.board.start(self.seat, self.main, self.extra)
            self.started = True
        if not self.started:
            raise RuntimeError("the stream must begin with MSG_START")
        self.board.apply(msg, bytes(payload))
        if msg in SERVER_ONLY:
            return None
        self._push()
        try:
            out = self.duel.feed(msg, bytes(payload))
        except RuntimeError as error:
            raise RuntimeError(f"message {msg} ({bytes(payload).hex()}): {error}") from error
        if out is not None:
            self.forced += 1
        return out

    def prompt(self):
        return self.duel.prompt()

    def observation(self) -> dict:
        self._push()
        return self.duel.observation()

    def public_world(self) -> dict:
        """Read-only own-observer belief context from this public client stream, never an engine/target layout.

        Only a started, explicitly public-decklist client at a pending decision is supported. Locations/indices are
        the client's public zone slots (not hand identities sorted by the AR decoder); own_deck is a sorted multiset
        and own_deck_fixed contains only publicly revealed positions. Unknown material ownership is refused.
        """
        if not self.started or not self.duel.pending:
            raise RuntimeError("client public_world requires a started client with its own pending decision")
        if self.public_opponent_recipe is None:
            raise RuntimeError("client public_world requires an explicitly declared public opponent decklist")
        materials = []
        for (player, location, sequence), cards in sorted(self.board.materials.items()):
            for index, card in enumerate(cards):
                if type(card.owner) is not int or card.owner not in (0, 1) or not card.code:
                    raise RuntimeError("client public_world has an unknown public material owner/identity")
                materials.append([player, location, sequence, index, int(card.code), card.owner])
        scratch = self.duel.clone()
        for player in (0, 1):
            for location in ZONES:
                scratch.set_cards(player, location, self._zone_records(player, location))
        return scratch.public_world(materials)

    def current_public_root_seed(self) -> dict:
        """Read current common-public effects/chain/field relations for another seat's root initialization.

        The native getter is const and requires a pending decision. It never pushes the board, observes a menu,
        consumes history, clones/rebinds the client, or exports this client's private tracker/choices/Memory.
        Coordinates and player owners are absolute; private source references are None. A relevant chain
        participant without a safe public anchor raises native.CurrentPublicRootSeedError with a reason code.
        """
        return self.duel.current_public_root_seed()

    def step(self, index: int):
        self._push()
        return self.duel.step(int(index))

    def step_response(self, response: bytes) -> bytes:
        """The pending decision answered by its response bytes (a recorded game's player): the one menu path (a row,
        or a multi-card selection's sub-choices in the order the bytes list them) whose response they are, applied as
        step would, so the seat's own records match; returns the bytes. No path, or more than one, is an error."""
        self._push()
        return self.duel.step_response(bytes(response))

    def response_path(self, response: bytes) -> list[int]:
        """Read the unique row path without advancing this client or its history delivery.

        A replaying policy must score and step each returned row, not skip the
        intermediate memory updates by calling ``step_response`` directly.
        """
        return list(self.duel.response_path(bytes(response)))

    def forced_count(self) -> int:
        return self.forced

    def clone(self) -> "AgentClientDuel":
        """An independent copy at this point of the stream (a search branches per rollout): the card view and the
        native state; feeding the copy and the original the same continuation gives the same arrays and bytes."""
        out = object.__new__(AgentClientDuel)
        out.seat, out.main, out.extra = self.seat, list(self.main), list(self.extra)
        out.public_opponent_recipe = copy.deepcopy(self.public_opponent_recipe)
        out.duel = self.duel.clone()
        out.board = copy.deepcopy(self.board)
        out.started, out.forced = self.started, self.forced
        if hasattr(self, "_root_view"):
            out._root_view = self._root_view  # immutable; live board/history remain independent
        return out
