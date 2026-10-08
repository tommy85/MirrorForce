"""Current-root wire boundary and Python shadow board; no engine/build is needed here.

Native observation/continuation assertions live in test_current_root_native.py.
These tests do not claim to authenticate an engine owner or complete a search game.
"""
from __future__ import annotations

import copy
import struct
from types import SimpleNamespace

import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.netduel.agent_client import AgentClientDuel
from mirrorforce.netduel.current_root_view import CurrentRootView, SCHEMA, ZONES


def card(code, p, loc, seq, position=C.POS_FACEDOWN_DEFENSE, *, owner=None, materials=(), counters=(), equip=0):
    return [code, p, loc, seq, position, 0, 0, 0, 0, equip, list(materials), list(counters),
            p if owner is None else owner, 0, 0, 0, 0, 0, False]


def root_view(viewer=0):
    cards = [[[] for _ in ZONES] for _ in range(2)]
    other = 1 - viewer
    cards[viewer][0] = [card(10, viewer, C.LOCATION_DECK, 0), card(20, viewer, C.LOCATION_DECK, 1)]
    cards[viewer][1] = [card(code, viewer, C.LOCATION_HAND, i) for i, code in enumerate([30, 10, 30])]
    cards[viewer][2] = [None, None, card(40, viewer, C.LOCATION_MZONE, 2, C.POS_FACEUP_ATTACK,
                                       owner=other, materials=[50, 60], counters=[(1, 3)])]
    cards[viewer][3] = [card(70, viewer, C.LOCATION_SZONE, 0, C.POS_FACEUP_ATTACK,
                            equip=viewer | C.LOCATION_MZONE << 8 | 2 << 16)]
    cards[viewer][6] = [card(80, viewer, C.LOCATION_EXTRA, 0)]
    cards[other][0] = [card(0, other, C.LOCATION_DECK, 0)]
    cards[other][1] = [card(0, other, C.LOCATION_HAND, i) for i in range(2)]
    cards[other][2] = [None, card(0, other, C.LOCATION_MZONE, 1)]
    return dict(schema=SCHEMA, root_id="owned-root-7", root_hash="a" * 64, hypothesis_hash="b" * 64,
                viewer=viewer, lp=[1300, 6100], turn=7, turn_player=other, phase=C.PHASE_MAIN1, cards=cards,
                main=[10, 10, 20, 30, 30, 50, 60, 70], extra=[80], opponent_main=[40, 90, 100], opponent_extra=[],
                material_owners=[[viewer, C.LOCATION_MZONE, 2, 0, 50, viewer],
                                 [viewer, C.LOCATION_MZONE, 2, 1, 60, other]],
                public_seed={"unpositioned": [[C.LOCATION_HAND, 90, 1]], "hand_group": [0, 1]})


class FakeNativeClient:
    def __init__(self, seat, main, extra, config, **kwargs):
        self.seat, self.main, self.extra, self.config = seat, main, extra, config
        self.initial = None
        self.fed = []
        self.zones = {}
        self.pending = False

    def initialize_current_root(self, data):
        assert not self.fed and self.initial is None
        self.initial = copy.deepcopy(data)
        return {row[0]: 1 for player in data["cards"] for zone in player for row in zone if row and row[0]}

    def feed(self, msg, body):
        self.fed.append((msg, body))
        return None

    def set_cards(self, p, loc, cards):
        self.zones[p, loc] = copy.deepcopy(cards)

    def clone(self):
        return copy.deepcopy(self)


NATIVE = SimpleNamespace(ClientDuel=FakeNativeClient)
CONFIG = {"public_opponent_recipe": True}


@pytest.mark.parametrize("viewer", [0, 1])
def test_explicit_root_keeps_order_holes_relations_and_never_feeds_an_opening(viewer):
    payload = root_view(viewer)
    client = AgentClientDuel.from_root_view(NATIVE, payload, CONFIG)
    assert client.started and not client.duel.fed
    assert client.duel.initial["lp"] == [1300, 6100] and client.duel.initial["turn"] == 7
    assert client.board.phase == C.PHASE_MAIN1 and client.board.turn_player == 1 - viewer
    assert [c.code for c in client.board.zone(viewer, C.LOCATION_HAND)] == [30, 10, 30]
    assert client.board.zone(viewer, C.LOCATION_MZONE)[:2] == [None, None]
    assert client.board.zone(viewer, C.LOCATION_MZONE)[2].owner == 1 - viewer
    assert [c.owner for c in client.board.materials[viewer, C.LOCATION_MZONE, 2]] == [viewer, 1 - viewer]
    client._push()
    row = client.duel.zones[viewer, C.LOCATION_MZONE][2]
    assert row[10] == [50, 60] and row[11] == [(1, 3)]
    assert client.duel.zones[viewer, C.LOCATION_SZONE][0][9] == viewer | C.LOCATION_MZONE << 8 | 2 << 16
    assert list(client.board.our_remaining_deck().elements()) == [10, 20]
    assert client.board.zone(viewer, C.LOCATION_DECK) == []


def test_followup_draw_and_clone_are_independent_and_keep_network_hand_slots():
    live = AgentClientDuel.from_root_view(NATIVE, root_view(), CONFIG)
    branch = live.clone()
    branch.feed(C.MSG_DRAW, bytes([0, 1]) + struct.pack("<I", 20))
    assert [r[0] for r in branch.duel.zones[0, C.LOCATION_HAND]] == [30, 10, 30, 20]
    assert [r[0] for r in branch.duel.zones[0, C.LOCATION_DECK]] == [10]
    assert not live.duel.fed and len(live.board.zone(0, C.LOCATION_HAND)) == 3
    branch.board.materials[0, C.LOCATION_MZONE, 2][0].owner = 1
    assert live.board.materials[0, C.LOCATION_MZONE, 2][0].owner == 0


def test_public_materials_of_an_anonymous_extra_host_survive_push():
    payload = root_view()
    payload["opponent_extra"] = [80]
    payload["cards"][1][6] = [card(0, 1, C.LOCATION_EXTRA, 0, materials=[50])]
    payload["material_owners"].append([1, C.LOCATION_EXTRA, 0, 0, 50, 0])
    client = AgentClientDuel.from_root_view(NATIVE, payload, CONFIG)
    client._push()
    assert client.duel.zones[1, C.LOCATION_EXTRA][0][0] == 0
    assert client.duel.zones[1, C.LOCATION_EXTRA][0][10] == [50]
    assert client.board.materials[1, C.LOCATION_EXTRA, 0][0].owner == 0


def test_current_chain_resolution_and_known_deck_positions_are_separate_from_multiset():
    data = root_view()
    data["public_seed"].update(
        reveals=[[0, C.LOCATION_DECK, 0, 20]],
        chain=[dict(link=1, player=0, code=70, desc=70 * 16, origin=[0, C.LOCATION_SZONE, 0],
                    source=[0, C.LOCATION_SZONE, 0])],
        chain_context=dict(links=1, solving=12, resolution=12, chain_link=1, settlement_link=1))
    client = AgentClientDuel.from_root_view(NATIVE, data, CONFIG)
    assert client.board.chain_resolution() == 12
    assert client.board.disclosure.known_code_at(0, 0, C.LOCATION_DECK, 0) == 20
    assert [r[0] for r in client._zone_records(0, C.LOCATION_DECK)] == [10, 20]
    client.feed(C.MSG_CHAIN_SOLVED, b"\x01")
    assert client.board.chain_resolution() == 0
    client.feed(C.MSG_CHAIN_SOLVING, b"\x01")
    assert client.board.chain_resolution() == 13


def test_view_is_immutable_and_binding_is_explicit():
    payload = root_view()
    view = CurrentRootView.from_dict(payload)
    payload["lp"][0] = 2
    exported = view.to_dict()
    exported["lp"][0] = 3
    assert view.to_dict()["lp"][0] == 1300
    view.require_binding(root_id="owned-root-7", root_hash="a" * 64, hypothesis_hash="b" * 64, viewer=0)
    for field, wrong in (("viewer", 1), ("root_id", "old-root"), ("root_hash", "c" * 64),
                         ("hypothesis_hash", "d" * 64)):
        args = {k: view.to_dict()[k] for k in ("root_id", "root_hash", "hypothesis_hash", "viewer")}
        args[field] = wrong
        with pytest.raises(ValueError, match="binding mismatch"):
            view.require_binding(**args)


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(server_hidden=[123]), "unknown fields"),
    (lambda d: d.update(viewer=2), "integer"),
    (lambda d: d.update(root_hash="old"), "root_hash"),
    (lambda d: d["cards"][0][1][0].__setitem__(3, 1), "slot"),
    (lambda d: d["cards"][1][1][0].__setitem__(0, 90), "concealed identity"),
    (lambda d: d["cards"][1][1][0].__setitem__(7, 2200), "private stats"),
    (lambda d: d["cards"][0][0][0].__setitem__(0, 30), "sorted remaining multiset"),
    (lambda d: d["material_owners"].pop(), "missing material owners"),
    (lambda d: d["material_owners"][0].__setitem__(4, 99), "material owner"),
    (lambda d: d["public_seed"].update(memory=[0]), "unknown fields"),
    (lambda d: d["public_seed"].update(hand_group=[9]), "hand group"),
    (lambda d: d["public_seed"].update(unpositioned=[[C.LOCATION_HAND, 90, 3]]), "exceed"),
    (lambda d: d["public_seed"].update(card_status=[dict(place=[0, 4, 0], bits=8, turn=7)]), "absent card"),
    (lambda d: d["public_seed"].update(chain_context=dict(solving=1, resolution=2)), "chain context"),
    (lambda d: d["public_seed"].update(reveals=[[0, C.LOCATION_DECK, 0, 99]]), "remaining multiset"),
    (lambda d: d.update(phase=3), "unknown phase"),
])
def test_bad_root_or_hidden_payload_is_rejected(mutate, match):
    data = root_view()
    mutate(data)
    with pytest.raises(ValueError, match=match):
        CurrentRootView.from_dict(data)


def test_explicit_public_known_identity_is_accepted_but_extra_truth_fields_are_not():
    data = root_view()
    data["cards"][1][1][0][0] = 90
    data["public_seed"].update(positioned=[[1, C.LOCATION_HAND, 0, 90]], hand_group=[1])
    view = CurrentRootView.from_dict(data)
    assert view.to_dict()["cards"][1][1][0][0] == 90
    with pytest.raises(ValueError, match="public opponent recipe"):
        AgentClientDuel.from_root_view(NATIVE, view, {})


def test_faceup_position_bit_in_opponent_hand_is_not_an_identity_proof():
    data = root_view()
    data["cards"][1][1][0][4] = C.POS_FACEUP_ATTACK
    CurrentRootView.from_dict(data)  # the anonymous network slot is still legal
    data["cards"][1][1][0][0] = 90
    with pytest.raises(ValueError, match="concealed identity"):
        CurrentRootView.from_dict(data)


def test_default_client_still_needs_start_and_has_no_root_seed():
    client = AgentClientDuel(NATIVE, 0, [10], [], {})
    assert not client.started and client.duel.initial is None
    with pytest.raises(RuntimeError, match="begin with MSG_START"):
        client.feed(C.MSG_NEW_TURN, b"\x00")


def test_current_public_getter_does_not_push_observe_or_feed_and_keeps_typed_errors():
    client = AgentClientDuel.from_root_view(NATIVE, root_view(), CONFIG)
    dto = {"schema": "mirrorforce_current_public_root_seed/v1", "public_seed": {"public_effects": []}}
    before = copy.deepcopy(vars(client.board))
    def forbidden(*args, **kwargs):
        raise AssertionError("the read-only public getter touched the observer")
    client.duel.set_cards = client.duel.feed = client.observation = forbidden
    client.duel.current_public_root_seed = lambda: copy.deepcopy(dto)
    assert client.current_public_root_seed() == dto
    assert client.board.zones == before["zones"]
    assert not client.duel.fed
    class TypedRejection(RuntimeError):
        pass
    def reject():
        raise TypedRejection("current_public_root_seed/private_chain_move: link 1 has no public anchor")
    client.duel.current_public_root_seed = reject
    with pytest.raises(TypedRejection, match="private_chain_move"):
        client.current_public_root_seed()


def test_common_equip_consistency_and_moved_destination_are_explicit():
    data = root_view()
    data["public_seed"]["equips"] = [[[0, C.LOCATION_SZONE, 0], [0, C.LOCATION_MZONE, 2]]]
    data["public_seed"]["chain"] = [dict(link=1, player=0, code=70, desc=70 * 16,
        origin=[0, C.LOCATION_SZONE, 0], source=[0, C.LOCATION_SZONE, 0],
        moved=[dict(place=[0, C.LOCATION_MZONE, 2], from_location=C.LOCATION_GRAVE, control=False,
                    destination=[0, C.LOCATION_SZONE, 1])])]
    result = CurrentRootView.from_dict(data).to_dict()
    assert result["public_seed"]["chain"][0]["moved"][0]["destination"] == [0, C.LOCATION_SZONE, 1]
    # Destination need not be currently occupied. Place is the current public entity anchor.
    data["public_seed"]["equips"][0][1] = [0, C.LOCATION_SZONE, 0]
    with pytest.raises(ValueError, match="equip seed contradicts"):
        CurrentRootView.from_dict(data)
