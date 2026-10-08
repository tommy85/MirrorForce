"""Real ClientDuel RPC reads at every decision, including multi-choice and nonzero memory."""
import copy
import json

import pytest

from mirrorforce.netduel.agent_public_recipe import declare
from mirrorforce.puzzle.single import RESPONSE_REQUIRED
from mirrorforce.agent.public_world_codec import RPC_CAPABILITY, encode_world
from tools import mf_runtime_policy_service as S
from test_client_public_world import native as native_fixture, game as game_fixture, CONFIG
from test_open_stream import MemoryBackend

PRIVILEGED_TARGET = True


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    return native_fixture.__wrapped__(tmp_path_factory)


@pytest.fixture(scope="module")
def game(native, tmp_path_factory):
    return game_fixture.__wrapped__(native, tmp_path_factory)


def wire(value):
    return json.loads(json.dumps(value))


def test_whole_native_game_rpc_worlds_match_public_oracle_without_advancing_live_sessions(native, game):
    played, expected_worlds, streams = game
    service = S.Service(native, MemoryBackend(), CONFIG, "sample", 1,
                        {"opponent_recipe_mode": "mirror", "client_public_world": dict(RPC_CAPABILITY)})
    total, multiple, memories = 0, 0, []
    for seat in (0, 1):
        main, extra = sorted(played.deal["deck_orders"][seat]), list(played.deal["extra"][seat])
        name = service.open({"seat": seat, "main": main, "extra": extra, "seed": 319 + seat,
                             "opponent_recipe_mode": "mirror", "public_opponent_recipe": declare(main, extra)})["session"]
        session = service.sessions[name]
        responses = [r for r, player in zip(played.responses, played.owners) if player == seat]
        messages, di, ri = [], 0, 0
        for msg, payload in streams[seat]:
            messages.append([msg, payload.hex()])
            if msg not in RESPONSE_REQUIRED:
                continue
            reply = service.prompt({"session": name, "messages": messages})
            messages = []
            subdecisions = 0
            while "pending" in reply:
                pending = reply["pending"]
                request = {"op": "public_world", "session": name, "expected_obs_sha256": pending["obs_sha256"]}
                before = (S.memory_sha256(session.state), copy.deepcopy(session.rng.bit_generator.state),
                          session.decisions, session.forced, session.prompts, list(session.act_ms), service.counter)
                untouched = session.client.clone()
                one, two = wire(service.dispatch(request)), wire(service.dispatch(request))
                assert one == two and one["world"] == wire(encode_world(expected_worlds[seat][di], complete=True))
                assert before == (S.memory_sha256(session.state), session.rng.bit_generator.state,
                                  session.decisions, session.forced, session.prompts, session.act_ms, service.counter)
                assert S.observation_sha256(untouched.observation()) == pending["obs_sha256"]
                assert session.client.response_path(responses[ri]) == untouched.response_path(responses[ri])
                index = played.decisions[seat][di].index
                expected_response = untouched.step(index)
                reply = service.commit({"session": name, "index": index})
                assert (reply.get("response") is None) == (expected_response is None)
                if expected_response is not None:
                    assert reply["response"] == bytes(expected_response).hex()
                di += 1
                total += 1
                subdecisions += 1
            multiple += int(subdecisions > 1)
            assert reply["response"] == responses[ri].hex()
            ri += 1
        memories.append(S.memory_sha256(session.state) != S.memory_sha256(service.backend.initial_state()))
        assert ri == len(responses) and di == len(played.decisions[seat])
        service.close({"session": name, "messages": messages})
        with pytest.raises(ValueError, match="closed"):
            service.dispatch(request)
    assert total > 30 and all(memories) and multiple > 0 and not service.sessions
