"""Actual ClientDuel binding/encoder tests, run with the newly built MF_DUEL_NATIVE.

These are constructed current positions and projected continuations, not a 600 s
room, full-game rollout, or authentication of the separate hypothetical-engine builder.
"""
import struct
import copy
import json

import numpy as np
import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.netduel.agent_client import AgentClientDuel
from mirrorforce.netduel.current_root_view import CurrentRootView
import test_client_duel as client_tests
from test_current_root_view import root_view


CODES = {10: 63288573, 20: 63166095, 30: 14558127, 40: 8491308, 50: 97268402,
         60: 52340444, 70: 24010609, 80: 90673288, 90: 98338152, 100: 51227866}
CONFIG = {"max_options": 192, "public_opponent_recipe": True}
native = client_tests.native  # reuse the existing isolated native/announce fixture


def view(seat=0):
    data = root_view(seat)
    for name in ("main", "extra", "opponent_main", "opponent_extra"):
        data[name] = [CODES[c] for c in data[name]]
    for player in data["cards"]:
        for zone in player:
            for row in zone:
                if row:
                    row[0] = CODES.get(row[0], 0)
                    row[10] = [CODES[c] for c in row[10]]
    own_deck = data["cards"][seat][0]
    own_deck.sort(key=lambda row: row[0])
    for seq, row in enumerate(own_deck):
        row[3] = seq
    for row in data["material_owners"]:
        row[4] = CODES[row[4]]
    for row in data["public_seed"]["unpositioned"]:
        row[1] = CODES[row[1]]
    return data


def ask(client):
    client.feed(C.MSG_SELECT_YESNO, bytes([client.seat]) + struct.pack("<I", CODES[70] * 16))
    assert client.prompt() is not None
    return client.observation()


def eq(a, b):
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.parametrize("seat", [0, 1])
def test_midgame_scalars_hand_slots_and_cold_temporal_inputs(native, seat):
    client = AgentClientDuel.from_root_view(native, view(seat), CONFIG)
    assert client.prompt() is None
    assert [r[0] for r in client._zone_records(seat, C.LOCATION_HAND)] == [CODES[30], CODES[10], CODES[30]]
    obs = ask(client)
    expected = [1300, 6100] if seat == 0 else [6100, 1300]
    assert list(obs["obs:global_"][:4]) == [x for lp in expected for x in (lp >> 8, lp & 255)]
    assert obs["obs:global_"][4] == 7 and obs["obs:global_"][5] == 2  # phase2id MAIN1
    assert obs["obs:global_"][7] == 0
    assert list(obs["info:turn_rows"]) == [7, 0]
    for key in ("obs:turn_events_", "obs:turn_chunks_", "obs:closed_turns_", "obs:closed_turn_meta_",
                "obs:turn_chunk_meta_", "obs:h_actions_"):
        assert not np.any(obs[key]), key
    assert obs["obs:card_turn_"][:, 4].sum() == 3
    assert np.any(obs["obs:card_status_"][:, :3])  # current equip relation, not historical events
    eq(obs, client.clone().observation())


def test_native_fresh_guard_and_recipe_binding(native):
    data = CurrentRootView.from_dict(view()).to_dict()
    raw = native.ClientDuel(0, data["main"], data["extra"], CONFIG,
                           opponent_main=data["opponent_main"], opponent_extra=data["opponent_extra"])
    bad = dict(data, viewer=1)
    with pytest.raises((RuntimeError, ValueError), match="viewer|own current|concealed"):
        raw.initialize_current_root(bad)
    bad = dict(data, main=data["main"] + [CODES[10]])
    with pytest.raises(RuntimeError, match="recipes"):
        raw.initialize_current_root(bad)
    raw.initialize_current_root(data)
    with pytest.raises(RuntimeError, match="fresh"):
        raw.initialize_current_root(data)
    twin = raw.clone()
    with pytest.raises(RuntimeError, match="fresh"):
        twin.initialize_current_root(data)
    fed = native.ClientDuel(0, data["main"], data["extra"], CONFIG,
                           opponent_main=data["opponent_main"], opponent_extra=data["opponent_extra"])
    fed.feed(C.MSG_NEW_TURN, b"\x00")
    with pytest.raises(RuntimeError, match="fresh"):
        fed.initialize_current_root(data)


def test_seeded_chain_and_status_continue_and_expire_without_old_events(native):
    data = view()
    data["public_seed"].update(
        chain=[dict(link=1, player=0, code=CODES[70], desc=CODES[70] * 16,
                    origin=[0, C.LOCATION_SZONE, 0], source=[0, C.LOCATION_SZONE, 0])],
        chain_context=dict(links=1, solving=9, resolution=0, chain_link=1),
        card_status=[dict(place=[0, C.LOCATION_MZONE, 2], source=[0, C.LOCATION_SZONE, 0], bits=8, turn=7)],
        public_effects=[[23434538, 0, 1, 1, 7]], turn_counts=[[2, 0, 0, 0, 0, 0], [0] * 6])
    client = AgentClientDuel.from_root_view(native, data, CONFIG)
    first = ask(client)
    assert first["obs:chain_"][0, 0] == 1 and first["obs:chain_"][0, 7] == 0
    assert first["obs:turn_ledger_"][0, 0] == 2
    assert first["obs:public_effects_"][0, 0] == 1
    assert first["obs:card_status_"][:, 3].sum() == 8
    branch = client.clone()
    client.step(0)
    client.feed(C.MSG_CHAIN_SOLVING, b"\x01")
    current = ask(client)
    assert current["obs:chain_"][0, 7] == 1
    client.step(0)
    client.feed(C.MSG_CHAIN_SOLVED, b"\x01")
    client.feed(C.MSG_CHAIN_END, b"")
    client.feed(C.MSG_NEW_TURN, b"\x00")
    client.feed(C.MSG_NEW_PHASE, struct.pack("<H", C.PHASE_DRAW))
    later = ask(client)
    assert later["obs:global_"][4] == 8 and later["info:turn_rows"][0] == 8
    assert later["info:closed_turn_rows"][0, 0] == 7
    assert not np.any(later["obs:chain_"]) and not np.any(later["obs:public_effects_"])
    assert later["obs:card_status_"][:, 3].sum() == 0
    assert later["obs:turn_ledger_"][0, 0] == 0
    eq(first, branch.observation())


def test_root_clone_continues_with_same_draw_but_does_not_mutate_real_client(native):
    client = AgentClientDuel.from_root_view(native, view(), CONFIG)
    root = ask(client)
    first, second = client.clone(), client.clone()
    for branch in (first, second):
        branch.step(0)
        branch.feed(C.MSG_DRAW, b"\x00\x01" + struct.pack("<I", CODES[20]))
    eq(ask(first), ask(second))
    eq(root, client.observation())
    assert len(first.board.zone(0, C.LOCATION_HAND)) == 4
    assert len(client.board.zone(0, C.LOCATION_HAND)) == 3


def public_state(data):
    data["public_seed"].update(
        card_status=[dict(place=[0, C.LOCATION_MZONE, 2], source=[0, C.LOCATION_HAND, 0], bits=8, turn=7)],
        public_effects=[[23434538, 23434538 * 16, 1, 1, 7], [CODES[70], CODES[70] * 16, 0, 2, 7]],
        field_origins=[[0, C.LOCATION_MZONE, 2, 1], [0, C.LOCATION_SZONE, 0, 0]])
    return data


def test_const_public_export_ignores_private_hand_identity_and_preserves_both_player_effects(native):
    payloads = [public_state(view()), public_state(view())]
    payloads[1]["cards"][0][1][0][0], payloads[1]["cards"][0][1][1][0] = CODES[10], CODES[30]
    projected = []
    for data in payloads:
        client = AgentClientDuel.from_root_view(native, data, CONFIG)
        before = ask(client)
        branch = client.clone()
        result = client.current_public_root_seed()
        assert result["schema"] == "mirrorforce_current_public_root_seed/v1"
        assert result["source_viewer"] == 0 and result["turn"] == 7 and result["turn_player"] == 1
        assert set(result["public_seed"]) == {"chain", "chain_context", "card_status", "public_effects", "equips", "field_origins"}
        assert result["public_seed"]["card_status"][0]["source"] is None
        assert result["public_seed"]["card_status"][0]["bits"] == 8
        assert [r[3] for r in result["public_seed"]["public_effects"]] == [1, 2]  # Maxx C and no-response-to-spells
        assert str(CODES[10]) not in json.dumps(result) and str(CODES[30]) not in json.dumps(result)
        assert result == client.current_public_root_seed()
        eq(before, client.observation())
        client.step(0)
        branch.step(0)
        eq(ask(client), ask(branch))
        projected.append(result)
    assert projected[0] == projected[1]


def test_public_export_is_absolute_and_can_seed_opposite_viewer_without_private_tracker(native):
    data = public_state(view())
    own = AgentClientDuel.from_root_view(native, data, CONFIG)
    ask(own)
    public = own.current_public_root_seed()
    other = copy.deepcopy(data)
    other["viewer"] = 1
    other["main"], other["opponent_main"] = other["opponent_main"], other["main"]
    other["extra"], other["opponent_extra"] = other["opponent_extra"], other["extra"]
    for zone in (0, 1, 6):
        for row in other["cards"][0][zone]:
            row[0] = 0
    other["cards"][1][0][0][0] = CODES[40]
    other["cards"][1][1][0][0] = CODES[90]
    other["cards"][1][1][1][0] = CODES[100]
    other["cards"][1][2][1][0] = CODES[90]  # its own assumed face-down field card, never in the public export
    other["public_seed"] = public["public_seed"]
    opponent = AgentClientDuel.from_root_view(native, other, CONFIG)
    result = ask(opponent)
    assert result["obs:card_status_"][:, 3].sum() == 8
    assert list(result["obs:public_effects_"][:2, 4]) == [1, 2]  # absolute owner1 becomes own, owner0 opponent
    assert opponent.current_public_root_seed()["public_seed"] == public["public_seed"]


def test_relevant_private_chain_participant_has_typed_rejection_without_mutation(native):
    data = public_state(view())
    data["public_seed"].update(
        chain=[dict(link=1, player=0, code=CODES[70], desc=CODES[70] * 16 + 1,
                    origin=[0, C.LOCATION_SZONE, 0], source=[0, C.LOCATION_SZONE, 0], state=1,
                    moved=[dict(place=[0, C.LOCATION_HAND, 0], from_location=C.LOCATION_GRAVE,
                                control=False, destination=[0, C.LOCATION_SZONE, 1])])],
        chain_context=dict(links=1, solving=12, resolution=12, chain_link=1, settlement_link=1))
    client = AgentClientDuel.from_root_view(native, data, CONFIG)
    before = ask(client)
    with pytest.raises(native.CurrentPublicRootSeedError, match="current_public_root_seed/private_chain_move:"):
        client.current_public_root_seed()
    eq(before, client.observation())
    data["public_seed"]["chain"][0]["moved"][0].update(from_location=C.LOCATION_DECK,
                                                       destination=[0, C.LOCATION_HAND, 0])
    irrelevant = AgentClientDuel.from_root_view(native, data, CONFIG)
    with pytest.raises(native.CurrentPublicRootSeedError, match="not_pending"):
        irrelevant.current_public_root_seed()
    ask(irrelevant)
    assert irrelevant.current_public_root_seed()["public_seed"]["chain"][0]["moved"] == []
