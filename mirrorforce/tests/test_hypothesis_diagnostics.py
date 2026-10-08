"""Rejection evidence is read-only client/particle data, not a new acceptance path."""
from types import SimpleNamespace

import pytest

from mirrorforce.netduel.opponent_stream import OpponentStreamError, _opening_config
from mirrorforce.netduel.agent_public_recipe import declare
from mirrorforce.netduel import constants as C
from mirrorforce.common.client_entity_map import LocalEntity
from mirrorforce.common.client_origin_receipt import PublicMutation
from mirrorforce.common.client_replay_journal import NativeBoundary, ReplayEvent


def test_identity_conflict_records_the_actual_event_and_does_not_modify_the_particle():
    entities = (LocalEntity(7, 0, 1, 1, C.LOCATION_DECK, 1, 0, 0xffffffff),
                LocalEntity(8, 0, 1, 1, C.LOCATION_DECK, 1, 1, 0xffffffff))
    state = NativeBoundary(1, entities)
    decks = (((11, 22), ()), ((11, 22), ()))
    opening = ReplayEvent(0, "opening", state, state, (0, 17, (), decks, ((11, 22), (11, 22)), "test"), 0, 0)
    mutation = PublicMutation("hydrate", (), None, 7, 22, True, (), ())
    reveal = ReplayEvent(19, "mutation", state, state, (mutation, ()), 4, 2)
    assignment = {7: 11, 8: 22}
    with pytest.raises(OpponentStreamError, match="previously accepted") as caught:
        _opening_config((opening, reveal), assignment, declare([11, 22], []))
    diagnostic = caught.value.diagnostic
    assert diagnostic["context"] == {"event_ordinal": 19, "event_kind": "mutation", "matched_cursor": 4,
        "own_answered": 2, "mutation_kind": "hydrate", "uid": 7, "received_code": 22, "sampled_code": 11}
    assert diagnostic["training_eligible"] is False
    assert assignment == {7: 11, 8: 22}


def test_diagnostic_context_is_owned_json_and_deepest_event_survives_wrapping():
    original = OpponentStreamError("public prefix differs", event_ordinal=19, packet_index=4,
                                   received_packet_hex="0b01", synthetic_packet_hex="0b02")
    wrapped = original.with_context(particle_index=1, stream_seed=77, event_ordinal=80)
    assert str(wrapped) == str(original)
    assert wrapped.diagnostic["context"]["event_ordinal"] == 19
    assert wrapped.diagnostic["context"]["particle_index"] == 1
    first = wrapped.diagnostic
    first["context"]["received_packet_hex"] = "corrupt copy"
    assert wrapped.diagnostic["context"]["received_packet_hex"] == "0b01"
    assert "particle_index" not in original.diagnostic["context"]


@pytest.mark.parametrize("data", [{"host": SimpleNamespace()}, {"server_truth": [11, 22]},
                                  {"received_code": [11]}, {"uid": object()}])
def test_diagnostics_reject_unregistered_or_non_scalar_sources(data):
    with pytest.raises(ValueError, match="declared client/synthetic"):
        OpponentStreamError("invalid diagnostic", **data)
