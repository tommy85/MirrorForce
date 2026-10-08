"""Field knowledge is retained in the root proof, never guessed into old policy slots.

Includes pure boundaries and opt-in actual native tests, not GPU/full-game acceptance.
"""
import copy
from types import SimpleNamespace

import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.netduel import current_root_export as E
from mirrorforce.netduel.disclosure import DisclosureLedger
from mirrorforce.netduel.agent_client import AgentClientDuel
from mirrorforce.netduel.current_root_view import CurrentRootView
from mirrorforce.common.client_entity_map import LocalEntity
from mirrorforce.worldmodel.state import CardState
from test_current_root_export import raw_state, encode
from test_current_root_view import root_view, card, FakeNativeClient, CONFIG
import test_client_duel as native_tests

native = native_tests.native


def test_production_current_root_mode_requires_the_native_import_guard():
    from mirrorforce.netduel.current_root_protocol import check_native_import, NATIVE_IMPORT_LAW
    for library in (SimpleNamespace(), SimpleNamespace(current_root_unpositioned_import_law="unverified")):
        with pytest.raises(ValueError, match="guarded three-zone"):
            check_native_import(library)
    check_native_import(SimpleNamespace(current_root_unpositioned_import_law=NATIVE_IMPORT_LAW))


def field_root():
    data = root_view()
    data["cards"][1][3] = [None, card(0, 1, C.LOCATION_SZONE, 1), None,
                              card(0, 1, C.LOCATION_SZONE, 3)]
    data["cards"][1][6] = [card(0, 1, C.LOCATION_EXTRA, 0)]
    data["public_seed"].update(
        unpositioned=[[C.LOCATION_HAND, 90, 1], [C.LOCATION_DECK, 100, 1],
                      [C.LOCATION_EXTRA, 80, 1], [C.LOCATION_MZONE, 40, 1],
                      [C.LOCATION_SZONE, 90, 1], [C.LOCATION_SZONE, 100, 1]],
        field_groups=[[C.LOCATION_MZONE, 1], [C.LOCATION_SZONE, 1, 3]])
    return data


def test_real_disclosure_shuffle_and_two_hidden_layouts_have_same_projection():
    raw, _ = raw_state()
    for seq, code in ((3, 90), (5, 100)):
        raw.cards.append(CardState(0, C.LOCATION_SZONE, seq, code=code, owner=0,
                                  position=C.POS_FACEDOWN_DEFENSE, attack=9000 + code))
    raw.counts[0, C.LOCATION_SZONE] += 2
    entities = tuple(LocalEntity(i + 1, 0, c.owner, c.controller, c.location, 0, c.sequence, 0xffffffff)
                     for i, c in enumerate(raw.cards))
    ledger = DisclosureLedger()
    for seq, code in ((3, 90), (5, 100)):
        ledger.disclose(0, C.LOCATION_SZONE, code, sequence=seq, audience=2)
    payload = bytes([C.LOCATION_SZONE, 2]) + b"".join(
        bytes([0, C.LOCATION_SZONE, seq, C.POS_FACEDOWN_DEFENSE]) for seq in (3, 5)) + b"\x00" * 8
    ledger.observe_shuffle_set_card(payload)
    other = copy.deepcopy(raw)
    a, b = other.cards[-2:]
    a.code, b.code = b.code, a.code
    first, second = encode(raw, entities, disclosure=ledger), encode(other, entities, disclosure=ledger)
    assert first.to_dict() == second.to_dict()
    assert first.to_native_dict() == second.to_native_dict()
    seed = first.to_dict()["public_seed"]
    assert seed["unpositioned"] == [[C.LOCATION_SZONE, 90, 1], [C.LOCATION_SZONE, 100, 1]]
    assert seed["field_groups"] == [[C.LOCATION_SZONE, 3, 5]]
    for seq in (3, 5):
        row = first.to_native_dict()["cards"][0][3][seq]
        assert row[0] == 0 and row[7] == 0  # neither identity nor private stats leak
    assert first.to_native_dict()["public_seed"]["unpositioned"] == []


def test_projection_changes_only_declared_abi_and_field_knowledge_fields():
    checked = CurrentRootView.from_dict(field_root())
    full = checked.to_dict()
    expected = copy.deepcopy(full)
    expected["schema"] = "mirrorforce_current_root_view/v1"
    expected["history_law"] = "current_root_cold/v1"
    expected["public_seed"]["unpositioned"] = [r for r in full["public_seed"]["unpositioned"]
                                                 if r[0] not in (C.LOCATION_MZONE, C.LOCATION_SZONE)]
    del expected["public_seed"]["field_groups"]
    projected = checked.to_native_dict()
    assert projected == expected
    assert projected["public_seed"]["unpositioned"] == [[C.LOCATION_HAND, 90, 1],
        [C.LOCATION_DECK, 100, 1], [C.LOCATION_EXTRA, 80, 1]]
    assert checked.to_dict() == full
    projected["cards"][1][3][1][0] = 999
    assert checked.to_dict() == full  # projection is not a mutable alias


def test_field_constraints_remain_hash_bound_even_when_policy_input_is_identical():
    first = CurrentRootView.from_dict(field_root())
    changed = field_root()
    changed["public_seed"]["unpositioned"][-1][1] = 101
    second = CurrentRootView.from_dict(changed)
    assert first.to_native_dict() == second.to_native_dict()
    assert E._sha(first.to_dict()) != E._sha(second.to_dict())
    changed = field_root()
    changed["cards"][1][3].append(card(0, 1, C.LOCATION_SZONE, 4))
    changed["public_seed"]["field_groups"][-1].append(4)
    larger = CurrentRootView.from_dict(changed)
    assert E._sha(first.to_dict()) != E._sha(larger.to_dict())


def test_client_native_boundary_never_routes_field_counts_to_legacy_extra():
    class LegacyImporter(FakeNativeClient):
        def initialize_current_root(self, data):
            assert data["schema"] == "mirrorforce_current_root_view/v1"
            assert data["history_law"] == "current_root_cold/v1"
            assert "field_groups" not in data["public_seed"]
            self.legacy_extra = [row for row in data["public_seed"]["unpositioned"]
                                 if row[0] not in (C.LOCATION_HAND, C.LOCATION_DECK)]
            assert self.legacy_extra == [[C.LOCATION_EXTRA, 80, 1]]
            return super().initialize_current_root(data)

    checked = CurrentRootView.from_dict(field_root())
    client = AgentClientDuel.from_root_view(SimpleNamespace(ClientDuel=LegacyImporter), checked, CONFIG)
    assert client.duel.initial == checked.to_native_dict()
    assert client._root_view.to_dict() == checked.to_dict()
    assert client.clone()._root_view.to_dict() == checked.to_dict()
    assert client.board.zone(1, C.LOCATION_SZONE)[1].code == 0
    assert client.board.zone(1, C.LOCATION_SZONE)[3].code == 0


@pytest.mark.parametrize("bad", ["known", "positioned", "absent", "duplicate_slot", "duplicate_zone",
                                 "capacity", "missing_group", "unsupported_zone"])
def test_bad_field_constraints_cannot_be_hidden_by_projection(bad):
    data = field_root()
    group = data["public_seed"]["field_groups"][-1]
    if bad == "known":
        data["cards"][1][3][1][0] = 90
    elif bad == "positioned":
        data["public_seed"]["positioned"] = [[1, C.LOCATION_SZONE, 1, 90]]
    elif bad == "absent":
        group[1] = 2
    elif bad == "duplicate_slot":
        group.append(1)
    elif bad == "duplicate_zone":
        data["public_seed"]["field_groups"].append(group.copy())
    elif bad == "capacity":
        data["public_seed"]["unpositioned"][-1][2] = 2
    elif bad == "missing_group":
        data["public_seed"]["field_groups"].pop()
    else:
        data["public_seed"]["unpositioned"][-1][0] = C.LOCATION_GRAVE
    with pytest.raises(ValueError):
        CurrentRootView.from_dict(data).to_native_dict()


def test_actual_legacy_native_field_projection_matches_all_anonymous_inputs(native):
    from test_current_root_native import view, CODES, ask, eq

    data = view()
    data["opponent_extra"] = [CODES[80]]
    data["cards"][1][3] = [None, card(0, 1, C.LOCATION_SZONE, 1), None,
                             card(0, 1, C.LOCATION_SZONE, 3)]
    data["cards"][1][6] = [card(0, 1, C.LOCATION_EXTRA, 0)]
    data["public_seed"].update(
        unpositioned=[[C.LOCATION_HAND, CODES[90], 1], [C.LOCATION_DECK, CODES[100], 1],
                      [C.LOCATION_EXTRA, CODES[80], 1], [C.LOCATION_MZONE, CODES[40], 1],
                      [C.LOCATION_SZONE, CODES[90], 1], [C.LOCATION_SZONE, CODES[100], 1]],
        field_groups=[[C.LOCATION_MZONE, 1], [C.LOCATION_SZONE, 1, 3]])
    full = CurrentRootView.from_dict(data)
    anonymous = copy.deepcopy(data)
    anonymous["public_seed"]["unpositioned"] = anonymous["public_seed"]["unpositioned"][:3]
    anonymous["public_seed"].pop("field_groups")
    baseline = CurrentRootView.from_dict(anonymous)
    assert full.to_native_dict() == baseline.to_native_dict()
    native_config = dict(CONFIG, max_options=192)
    live = AgentClientDuel.from_root_view(native, full, native_config)
    old_inputs = AgentClientDuel.from_root_view(native, baseline, native_config)
    obs, expected = ask(live), ask(old_inputs)
    assert len([key for key in obs if key.startswith("obs:")]) == 33
    eq(obs, expected)
    eq(obs, live.clone().observation())
    assert live._root_view.to_dict() == full.to_dict()
    assert live.clone()._root_view.to_dict() == full.to_dict()
    # All three original zone rows survive; no field identity became an Extra row.
    assert sum(bool(row[0] or row[1]) for row in obs["obs:unpositioned_"]) == 3
    assert all(live.board.zone(1, C.LOCATION_SZONE)[seq].code == 0 for seq in (1, 3))


def test_actual_legacy_native_two_exported_hidden_permutations_are_identical(native):
    from test_current_root_export import seed
    from test_current_root_native import CODES, ask, eq

    raw, _ = raw_state()
    for seq, code in ((3, 90), (5, 100)):
        raw.cards.append(CardState(0, C.LOCATION_SZONE, seq, code=code, owner=0,
                                  position=C.POS_FACEDOWN_DEFENSE))
    raw.counts[0, C.LOCATION_SZONE] += 2
    for c in raw.cards:
        c.code = CODES[c.code]
    entities = tuple(LocalEntity(i + 1, 0, c.owner, c.controller, c.location, 0, c.sequence, 0xffffffff)
                     for i, c in enumerate(raw.cards))
    ledger = DisclosureLedger()
    for seq, code in ((3, 90), (5, 100)):
        ledger.disclose(0, C.LOCATION_SZONE, CODES[code], sequence=seq, audience=2)
    ledger.observe_shuffle_set_card(bytes([8, 2, 0, 8, 3, 8, 0, 8, 5, 8]) + b"\x00" * 8)
    twin = copy.deepcopy(raw)
    twin.cards[-2].code, twin.cards[-1].code = twin.cards[-1].code, twin.cards[-2].code
    views, clients, observations = [], [], []
    for state in (raw, twin):
        checked = E._view_from_current(state, entities,
            {"main": [CODES[c] for c in (10, 20, 30, 40, 50)], "extra": [CODES[80]]}, seed(),
            root_id="field-permutation", root_hash="a" * 64, hypothesis_hash="b" * 64,
            viewer=1, disclosure=ledger)
        client = AgentClientDuel.from_root_view(native, checked, dict(CONFIG, max_options=192))
        views.append(checked)
        clients.append(client)
        observations.append(ask(client))
    assert views[0].to_dict() == views[1].to_dict()
    assert views[0].to_native_dict() == views[1].to_native_dict()
    assert E._sha(views[0].to_dict()) == E._sha(views[1].to_dict())
    eq(*observations)
    assert len([key for key in observations[0] if key.startswith("obs:")]) == 33
    for client, checked in zip(clients, views):
        assert client._root_view.to_dict() == checked.to_dict()
        assert checked.to_dict()["public_seed"]["field_groups"] == [[8, 3, 5]]
        assert len(checked.to_dict()["public_seed"]["unpositioned"]) == 2
        assert not checked.to_native_dict()["public_seed"]["unpositioned"]
        assert all(client.board.zone(0, C.LOCATION_SZONE)[seq].code == 0 for seq in (3, 5))


def test_legacy_projection_validation_normalizes_without_changing_native_payload():
    full = CurrentRootView.from_dict(field_root())
    projected = full.to_native_dict()
    normalized = CurrentRootView.from_dict(projected)
    assert normalized.to_dict()["schema"] == full.to_dict()["schema"]
    assert normalized.to_native_dict() == projected
    assert not normalized.to_dict()["public_seed"]["field_groups"]
    assert projected["schema"] == "mirrorforce_current_root_view/v1"
    # Empty optional field_groups does not grant access to any field knowledge.
    projected["public_seed"]["field_groups"] = []
    assert CurrentRootView.from_dict(projected).to_dict() == normalized.to_dict()


@pytest.mark.parametrize("corruption", ["field_count", "field_group", "new_history", "unknown_schema"])
def test_legacy_projection_validation_refuses_field_smuggling(corruption):
    projected = CurrentRootView.from_dict(field_root()).to_native_dict()
    if corruption == "field_count":
        projected["public_seed"]["unpositioned"].append([C.LOCATION_SZONE, 90, 1])
    elif corruption == "field_group":
        projected["public_seed"]["field_groups"] = [[C.LOCATION_SZONE, 1, 3]]
    elif corruption == "new_history":
        projected["history_law"] = CurrentRootView.from_dict(field_root()).to_dict()["history_law"]
    else:
        projected["schema"] = "mirrorforce_current_root_view/v0"
    with pytest.raises(ValueError):
        CurrentRootView.from_dict(projected)


def test_candidate_native_importer_rejects_unprojected_field_knowledge(native):
    if getattr(native, "current_root_unpositioned_import_law", None) != "three-zone-projection-only/v1":
        pytest.skip("requires the separate reviewed importer-guard candidate build; old 47ff is not guarded")
    from test_current_root_native import view, CODES

    data = view()
    data["public_seed"]["unpositioned"].append([C.LOCATION_MZONE, CODES[40], 1])
    data["public_seed"]["field_groups"] = [[C.LOCATION_MZONE, 1]]
    checked = CurrentRootView.from_dict(data)
    full = checked.to_dict()
    raw = native.ClientDuel(full["viewer"], full["main"], full["extra"], dict(CONFIG, max_options=192),
                           opponent_main=full["opponent_main"], opponent_extra=full["opponent_extra"])
    with pytest.raises(RuntimeError, match="field knowledge|unpositioned location"):
        raw.initialize_current_root(full)
