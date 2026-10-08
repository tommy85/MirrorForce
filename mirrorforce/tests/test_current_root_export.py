"""The encoder masks hypothetical states; these fixtures are not server engine access or deployment proof."""
import copy
from types import SimpleNamespace

import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.netduel import current_root_export as E
from mirrorforce.netduel.current_root_protocol import PUBLIC_SEED_RPC_SCHEMA, PUBLIC_SEED_SCHEMA
from mirrorforce.worldmodel.state import CardState, StateSnapshot
from mirrorforce.common.client_entity_map import LocalEntity
from mirrorforce.netduel.disclosure import DisclosureLedger


def seed():
    return {"chain": [], "chain_context": {"links": 0, "solving": 4, "resolution": 0,
            "chain_link": 0, "settlement_link": 0, "in_damage_step": False},
            "card_status": [{"place": [0, 4, 3], "source": None, "bits": 8, "turn": 7}],
            "public_effects": [[23434538, 23434538*16, 1, 1, 7]],
            "equips": [[[0, 8, 1], [0, 4, 3]]], "field_origins": [[0, 4, 3, 0]]}


def raw_state(*, other_hand=30, other_deck=(20, 10), uid_offset=0):
    cards = []
    for player in (0, 1):
        for loc, codes in ((C.LOCATION_DECK, other_deck if player == 0 else (20, 10)),
                           (C.LOCATION_HAND, (other_hand,) if player == 0 else (30,)), (C.LOCATION_EXTRA, (80,))):
            for seq, code in enumerate(codes):
                cards.append(CardState(player, loc, seq, code=code, owner=player, position=C.POS_FACEDOWN_DEFENSE,
                                       attack=9999, level=4, status=3))
    cards.extend([CardState(0, C.LOCATION_MZONE, 3, code=40, owner=0, position=1, attack=2500, defense=2000),
                  CardState(0, C.LOCATION_SZONE, 1, code=50, owner=0, position=1, equip_card=4 << 8 | 3 << 16)])
    counts = {(p, loc): sum(c.controller == p and c.location == loc for c in cards) for p in (0, 1) for loc in E.ZONES}
    entities = tuple(LocalEntity(i+1+uid_offset, 0, c.owner, c.controller, c.location, 0, c.sequence, 0xffffffff)
                     for i, c in enumerate(cards))
    return StateSnapshot(7, 0, C.PHASE_MAIN1, (4000, 7200), cards, counts), entities


def encode(raw, entities, public_seed=None, disclosure=None):
    return E._view_from_current(raw, entities, {"main": [10, 20, 30, 40, 50], "extra": [80]},
                               seed() if public_seed is None else public_seed,
                               root_id="real-root", root_hash="a"*64, hypothesis_hash="b"*64, viewer=1,
                               disclosure=DisclosureLedger() if disclosure is None else disclosure)


def test_two_hidden_layouts_and_entity_allocations_have_identical_opponent_view():
    first, entities = raw_state()
    second, other_entities = raw_state(other_hand=20, other_deck=(30, 10), uid_offset=500)
    original = copy.deepcopy(first)
    a, b = encode(first, entities).to_dict(), encode(second, other_entities).to_dict()
    assert a == b
    assert first == original
    assert [r[0] for r in a["cards"][1][0]] == [10, 20]
    assert [r[0] for r in a["cards"][1][1]] == [30]
    for zone in (0, 1, 6):
        assert all(r[0] == 0 and not any(r[k] for k in (5, 6, 7, 8, 13, 14, 15, 16, 17, 18))
                   for r in a["cards"][0][zone])
    assert a["cards"][0][2][:3] == [None] * 3
    assert a["cards"][0][2][3][7] == 2500
    assert a["cards"][0][3][1][9] == 4 << 8 | 3 << 16
    assert a["public_seed"]["public_effects"] == seed()["public_effects"]
    assert a["public_seed"]["card_status"][0]["source"] is None


@pytest.mark.parametrize('position',[1,2,4,8])
def test_native_equip_target_position_byte_is_not_a_different_public_relation(position):
    raw,entities=raw_state()
    target=next(c for c in raw.cards if (c.controller,c.location,c.sequence)==(0,4,3))
    equip=next(c for c in raw.cards if (c.controller,c.location,c.sequence)==(0,8,1))
    target.position=position
    equip.equip_card=(position<<24)|(4<<8)|(3<<16)
    value=encode(raw,entities).to_dict()
    assert value['cards'][0][3][1][9]==(4<<8)|(3<<16)
    assert value['cards'][0][2][3][4]==position
    wrong=seed();wrong['equips'][0][1]=[0,4,2]
    with pytest.raises(ValueError,match='equip relation'):
        encode(raw,entities,wrong)


@pytest.mark.parametrize("corruption", ["placeholder", "missing_entity", "count", "gap", "faceup_extra", "equip", "private_field"])
def test_export_does_not_repair_bad_or_private_state(corruption):
    raw, entities = raw_state()
    public = seed()
    if corruption == "placeholder":
        entities = (SimpleNamespace(**{**vars(entities[0]), "placeholder": 1}), *entities[1:])
    elif corruption == "missing_entity":
        entities = entities[1:]
    elif corruption == "count":
        raw.counts[0, 1] += 1
    elif corruption == "gap":
        raw.cards[0].sequence = 9
    elif corruption == "faceup_extra":
        next(c for c in raw.cards if c.controller == 0 and c.location == 64).position = 1
    elif corruption == "equip":
        public["equips"][0][1] = [0, 4, 2]
    else:
        public["private_choices"] = [1]
    with pytest.raises(ValueError):
        encode(raw, entities, public)


def reply():
    dto = {"schema": PUBLIC_SEED_SCHEMA, "source_viewer": 0, "turn": 7, "turn_player": 0,
           "phase": C.PHASE_MAIN1, "public_seed": seed()}
    return {"schema": PUBLIC_SEED_RPC_SCHEMA, "session": "real", "obs_sha256": "a"*64,
            "seed": dto, "seed_sha256": E._sha(dto)}


@pytest.mark.parametrize("key,value", [("session", "another"), ("obs_sha256", "b"*64),
                                      ("seed_sha256", "c"*64), ("server_hidden", [])])
def test_public_seed_rpc_rejects_wrong_binding_or_undeclared_truth(key, value):
    payload = reply()
    payload[key] = value
    with pytest.raises(ValueError):
        E.check_seed_reply(payload, session="real", obs_sha256="a"*64, source_viewer=0)


def test_public_seed_is_copied_without_accepting_private_dto_fields():
    payload = reply()
    seed_copy = E.check_seed_reply(payload, session="real", obs_sha256="a"*64, source_viewer=0)
    seed_copy["public_seed"]["card_status"].clear()
    assert payload["seed"]["public_seed"]["card_status"]
    payload["seed"]["hand"] = [30]
    payload["seed_sha256"] = E._sha(payload["seed"])
    with pytest.raises(ValueError, match="private fields"):
        E.check_seed_reply(payload, session="real", obs_sha256="a"*64, source_viewer=0)


def test_a_common_public_projection_cannot_smuggle_private_source_coordinates():
    payload = reply()
    payload["seed"]["public_seed"]["card_status"][0]["source"] = [0, C.LOCATION_HAND, 0]
    payload["seed_sha256"] = E._sha(payload["seed"])
    with pytest.raises(ValueError, match="private or invalid anchor"):
        E.check_seed_reply(payload, session="real", obs_sha256="a"*64, source_viewer=0)


def test_public_builder_never_accepts_a_raw_server_duel_handle():
    with pytest.raises(ValueError, match="owned particle branch"):
        E.build_current_view(SimpleNamespace(pduel=17), public_recipe={}, public_root_seed={}, root_id="r",
                             root_hash="a"*64, hypothesis_hash="b"*64, viewer=1)


@pytest.mark.parametrize("position", [C.POS_FACEUP_ATTACK, C.POS_FACEDOWN_DEFENSE])
def test_unknown_opponent_hand_never_uses_its_engine_position_as_disclosure(position):
    raw, entities = raw_state()
    next(c for c in raw.cards if c.controller == 0 and c.location == C.LOCATION_HAND).position = position
    data = encode(raw, entities).to_dict()
    assert data["cards"][0][1][0][0] == 0
    assert data["public_seed"]["positioned"] == []


def test_disclosure_retains_exact_identity_without_concealed_dynamic_stats():
    raw, entities = raw_state()
    ledger = DisclosureLedger()
    ledger.disclose(0, C.LOCATION_HAND, 30, sequence=0, audience=2)
    ledger.disclose(0, C.LOCATION_EXTRA, 80, sequence=0, audience=2)
    before = copy.deepcopy(ledger)
    data = encode(raw, entities, disclosure=ledger).to_dict()
    assert data["public_seed"]["positioned"] == [[0, C.LOCATION_HAND, 0, 30]]
    assert data["public_seed"]["reveals"] == [[0, C.LOCATION_EXTRA, 0, 80]]
    assert data["cards"][0][1][0][0] == 30
    assert not any(data["cards"][0][1][0][i] for i in (5, 6, 7, 8, 13, 14, 15, 16, 17, 18))
    assert data["cards"][0][6][0][0] == 0
    assert data["public_seed"]["hand_group"] == []
    assert ledger.known_counts(1) == before.known_counts(1)
    assert ledger.known_slots(1, 0, C.LOCATION_HAND) == before.known_slots(1, 0, C.LOCATION_HAND)


def test_owner_only_disclosure_is_not_laundered_into_the_other_seat():
    raw, entities = raw_state()
    ledger = DisclosureLedger()
    ledger.disclose(0, C.LOCATION_HAND, 30, sequence=0, audience=1)
    assert encode(raw, entities, disclosure=ledger).to_dict() == encode(raw, entities).to_dict()


def test_shuffle_keeps_names_but_does_not_reinvent_positions_from_hypothesis():
    raw, entities = raw_state()
    ledger = DisclosureLedger()
    ledger.disclose(0, C.LOCATION_HAND, 30, sequence=0, audience=2)
    ledger.observe_shuffle(C.MSG_SHUFFLE_HAND, bytes([0, 1]) + b"\x00" * 4)
    data = encode(raw, entities, disclosure=ledger).to_dict()
    assert data["cards"][0][1][0][0] == 0
    assert data["public_seed"]["positioned"] == []
    assert data["public_seed"]["unpositioned"] == [[C.LOCATION_HAND, 30, 1]]
    assert data["public_seed"]["hand_group"] == [0]


def test_visible_identity_is_not_counted_twice_as_unpositioned():
    raw, entities = raw_state()
    ledger = DisclosureLedger()
    ledger.disclose(0, C.LOCATION_MZONE, 40, audience=2)
    assert encode(raw, entities, disclosure=ledger).to_dict()["public_seed"]["unpositioned"] == []
