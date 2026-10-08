"""Original native model inputs remain exact; contradictory hypothetical moves still fail."""
import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from mirrorforce.netduel.cards import load_ydk
from mirrorforce.netduel.client import SELECT_MESSAGES
from mirrorforce.netduel.agent_client import AgentClientDuel
from mirrorforce.agent.env.announce_law import register
from mirrorforce.agent.search.particles import Sampler
from test_fresh_draw_wire_prefix import fixture


def sha(obs):
    result = hashlib.sha256()
    for name in sorted(obs):
        array = np.ascontiguousarray(obs[name])
        result.update(name.encode() + b'\0' + str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())
    return result.hexdigest()


def test_original_292_responses_177_model_inputs_and_fresh_group_contradiction():
    paths = [os.environ.get(key) for key in ('MF_DUEL_NATIVE', 'MF_TEST_ASSETS', 'MF_TEST_ANNOUNCE')]
    if not all(paths):
        pytest.skip('requires original pinned native/assets/announce in an isolated process')
    native, assets, tables = map(Path, paths)
    data = fixture()
    for path, expected in ((native, data['native_sha256']), (assets / 'cards.cdb', data['cards_db_sha256']),
                           (assets / 'code_list.txt', data['code_list_sha256']), (tables, data['announce_sha256'])):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location('duel_native', native)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    here = os.getcwd()
    try:
        os.chdir(assets)
        module.init_module(str(assets / 'cards.cdb'), str(assets / 'code_list.txt'), {})
    finally:
        os.chdir(here)
    register(module, tables, 192)
    main, extra, _ = load_ydk(Path(__file__).parents[1] / 'decks/stage-a/SkyStriker.ydk')
    client = AgentClientDuel(module, 0, main, extra, data['client_config'], opponent_main=main, opponent_extra=extra)
    rows = {index: [r for r in data['reports'] if r['response_index'] == index and not r['forced']]
            for index in range(len(data['responses']))}
    sent, checked = [], 0
    for index, (msg, body) in enumerate(data['public_messages']):
        response = client.feed(msg, bytes.fromhex(body))
        if msg not in SELECT_MESSAGES:
            assert response is None
            continue
        ordinal = len(sent)
        if ordinal == len(data['responses']):
            assert index == 2805 and ordinal == 292 and checked == 177
            assert response is None and sha(client.observation()) == data['pending_obs_sha256']
            world = client.public_world()
            assert world['hand'] == [0] and world['hand_group'] == [0]
            assert world['unpositioned'][2, 51227866] == 1
            assert client.board.disclosure.known_counts(0)[1, 2, 51227866] == 1
            assert all(p['hand'] == [51227866] for p in Sampler(world).sample(1, 8))
            before = sha(client.observation())
            bad, good = client.clone(), client.clone()
            for branch in (bad, good):
                branch.step(2)  # original pending menu's legal pass; no model call
            suffix = bytes.fromhex('0102000a0108040540000000')
            with pytest.raises(RuntimeError, match='1 copies over 0 unidentified'):
                bad.feed(50, (35726888).to_bytes(4, 'little') + suffix)
            good.feed(50, (51227866).to_bytes(4, 'little') + suffix)
            assert sha(client.observation()) == before
            break
        for row in rows[ordinal]:
            assert response is None and sha(client.observation()) == row['obs_sha256']
            checked += 1
            response = client.step(row['chosen'])
        assert bytes(response).hex() == data['responses'][ordinal]
        sent.append(bytes(response).hex())
    else:
        raise AssertionError('the original pending root was not reached')
