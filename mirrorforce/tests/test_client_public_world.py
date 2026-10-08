"""Engine-free public worlds: whole-field parity, read-only delivery and two hidden truths.

One short complete native game supplies both observers' recorded streams; no model or GPU is used. Requires a
development or committed module with ClientDuel.public_world. A distinct same-byte host core projects packets.
"""
from __future__ import annotations

import copy
import ctypes
import importlib.util
import os
from pathlib import Path
import pickle
import random
import shutil
import sqlite3

import numpy as np
import pytest

from mirrorforce.netduel.agent_client import AgentClientDuel
from mirrorforce.probes import client_parity as V

RUN = Path(os.environ.get("MF_SCRIPTED_RUN", "/path/to/workspace/ygopro-client-run"))
TABLES = next((Path(__file__).parent / "fixtures/announce").glob("announce-tables-*.json"))
CONFIG = {"max_options": 192, "public_opponent_recipe": True}


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    path = os.environ.get("MF_DUEL_NATIVE")
    if not path or not (RUN / "cards.cdb").is_file():
        pytest.skip("set MF_DUEL_NATIVE to the public-context module and provide scripted assets")
    spec = importlib.util.spec_from_file_location("duel_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module.ClientDuel, "public_world"):
        pytest.skip("module predates the explicit client public-world API")
    with sqlite3.connect(f"file:{RUN / 'cards.cdb'}?mode=ro", uri=True) as conn:
        codes = sorted(row[0] for row in conn.execute("SELECT id FROM datas"))
    code_list = tmp_path_factory.mktemp("client-world") / "code_list.txt"
    code_list.write_text("".join(f"{code} {int((RUN / 'script' / f'c{code}.lua').is_file())}\n" for code in codes))
    here = os.getcwd()
    os.chdir(RUN)
    try:
        module.init_module(str(RUN / "cards.cdb"), str(code_list), {})
    finally:
        os.chdir(here)
    from mirrorforce.agent.env.announce_law import register
    register(module, TABLES, 192)
    return module


@pytest.fixture(scope="module")
def game(native, tmp_path_factory):
    from mirrorforce.worldmodel.engine import load_ydk
    from mirrorforce.puzzle.core import get_core
    deck = load_ydk(Path(__file__).resolve().parents[1] / "decks/stage-a/SkyStriker.ydk")
    rng = random.Random(51217)
    orders = [list(deck.main), list(deck.main)]
    for order in orders:
        rng.shuffle(order)
    deal = {"seed_words": [rng.getrandbits(32) for _ in range(8)], "deck_orders": orders,
            "extra": [list(deck.extra), list(deck.extra)], "start_lp": 8000, "start_hand": 5, "draw_count": 1,
            "duel_options": 5 << 16}
    duel = native.SearchDuel(deal, CONFIG, keep_history=True)
    duel.start()
    decisions, worlds = ([], []), ([], [])
    choose = V.active_rule(53)
    for _ in range(6000):
        prompt = duel.prompt()
        if prompt is None:
            break
        seat, msg, rows = prompt
        obs = {key: value.copy() for key, value in duel.observation().items()}
        worlds[seat].append(copy.deepcopy(duel.public_world(seat)))
        chosen = choose(seat, msg, rows)
        decisions[seat].append(V.Decision(msg, rows, obs, chosen))
        duel.step(chosen)
    else:
        pytest.fail("the one-game fixture did not end")
    played = V.Game(deal, duel.winner, decisions, [bytes(r) for r in duel.responses()])
    source = Path(os.environ["MF_DUEL_NATIVE"]).parent / "libmfcore.so"
    host_path = tmp_path_factory.mktemp("client-world-host") / "core-host.so"
    shutil.copyfile(source, host_path)
    core = get_core(lib_path=host_path, db_path=RUN / "cards.cdb", script_dirs=[RUN / "script"], mode=ctypes.RTLD_LOCAL)
    streams = V.seat_streams(core, played)
    return played, worlds, streams


def client_for(native, game, seat, *, public=True, **config_extra):
    d = game.deal
    kwargs = {"opponent_main": d["deck_orders"][1 - seat], "opponent_extra": d["extra"][1 - seat]} if public else {}
    return AgentClientDuel(native, seat, d["deck_orders"][seat], d["extra"][seat],
                        dict(CONFIG, public_opponent_recipe=public, **config_extra), **kwargs)


def first_pending(client, stream):
    for msg, payload in stream:
        out = client.feed(msg, payload)
        if client.duel.pending:
            assert out is None
            return
    pytest.fail("fixture observer never had a decision")


def arrays_equal(a, b):
    assert set(a) == set(b)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_every_client_world_matches_native_and_reads_do_not_consume_history(native, game):
    played, worlds, streams = game
    total = cross_turn = 0
    for seat in (0, 1):
        client = client_for(native, played, seat)
        decisions = played.decisions[seat]
        response_list = [r for r, owner in zip(played.responses, played.owners) if owner == seat]
        di = ri = 0
        for msg, payload in streams[seat]:
            response = client.feed(msg, payload)
            while client.duel.pending:
                # Clone BEFORE the new read, before either original or clone Observe consumes delivery.
                untouched = client.clone()
                board_before, forced_before = pickle.dumps(vars(client.board)), client.forced_count()
                actual = client.public_world()
                assert actual == worlds[seat][di], (seat, di, {k for k in actual if actual[k] != worlds[seat][di][k]})
                assert pickle.dumps(vars(client.board)) == board_before
                assert client.forced_count() == forced_before
                arrays_equal(client.observation(), untouched.observation())
                # Only obs: arrays are shared across engine and client modes; info:step_limit includes actual
                # engine process counts that an engine-free client cannot have. Within-client read-only checks
                # above/below still compare EVERY key, and the complete public-world dict is compared separately.
                arrays_equal({k: v for k, v in client.observation().items() if k.startswith("obs:")},
                             {k: v for k, v in decisions[di].arrays.items() if k.startswith("obs:")})
                assert client.response_path(response_list[ri]) == untouched.response_path(response_list[ri])
                assert client.clone().public_world() == actual
                cross_turn += int(np.asarray(decisions[di].arrays["obs:closed_turn_meta_"])[..., 0].any())
                response = client.step(decisions[di].index)
                other_response = untouched.step(decisions[di].index)
                assert response == other_response
                assert client.duel.pending == untouched.duel.pending
                if client.duel.pending:
                    arrays_equal(client.observation(), untouched.observation())
                    assert client.public_world() == untouched.public_world()
                di += 1
                total += 1
            if response is not None:
                assert bytes(response) == response_list[ri]
                ri += 1
        assert di == len(decisions) and ri == len(response_list)
        with pytest.raises(RuntimeError, match="pending"):
            client.public_world()
    assert total > 30 and cross_turn > 0


def test_no_declaration_no_pending_or_hidden_arguments_are_refused(native, game):
    played, _, streams = game
    client = client_for(native, played, 0)
    with pytest.raises(RuntimeError, match="pending"):
        client.public_world()
    first_pending(client, streams[0])
    with pytest.raises(TypeError):
        client.public_world(viewer=1)
    with pytest.raises(TypeError):
        client.public_world(labels=[[123]])
    with pytest.raises(TypeError):
        client.duel.public_world([], truth={"hand": [123]})
    closed = client_for(native, played, 0, public=False)
    first_pending(closed, streams[0])
    with pytest.raises(RuntimeError, match="declared"):
        closed.public_world()
    with pytest.raises(RuntimeError, match="declared"):
        closed.duel.public_world([])


def test_two_real_hidden_truths_give_identical_complete_client_world(native, game):
    from mirrorforce.agent.search.particles import Sampler, realize
    played, _, streams = game
    duel = native.SearchDuel(played.deal, dict(CONFIG, belief_labels=1), keep_history=True)
    duel.start()
    seat = duel.prompt()[0]
    client = client_for(native, played, seat)
    first_pending(client, streams[seat])
    original = client.public_world()
    assert original == duel.public_world(seat)
    before = duel.labels()["label:hidden_"].copy()
    public = {k: v.copy() for k, v in duel.observation().items()}
    changed = False
    for particle in Sampler(original).sample(582, 3):
        realize(duel, seat, particle)
        changed |= not np.array_equal(before, duel.labels()["label:hidden_"])
        arrays_equal(public, duel.observation())
        assert duel.public_world(seat) == original == client.public_world()
    assert changed


@pytest.mark.parametrize("owners", [[[0, 4, 0, 0, 11, True]], [[0, 4, 0, 0.5, 11, 1]],
                                     [[0, 4, 0, 0, 11, -1]], [[0, 4, 0, 0, 11, 1]]])
def test_native_material_owner_metadata_is_strict_and_never_guessed(native, game, owners):
    played, _, streams = game
    client = client_for(native, played, 0)
    first_pending(client, streams[0])
    before = client.observation()
    with pytest.raises((RuntimeError, TypeError)):
        client.duel.public_world(owners)
    arrays_equal(before, client.observation())


def test_public_material_owner_not_host_controller_drives_recipe_subtraction(native, game):
    """Isolated public-card-view fixture: a shown material belongs to the opponent under our monster slot.

    This checks owner accounting, not that this synthetic position came from the recorded Sky game.
    """
    from mirrorforce.netduel import constants as C
    from mirrorforce.netduel.board import ShadowCard
    played, _, streams = game
    client = client_for(native, played, 0, allow_unreviewed_public_effects=True)
    first_pending(client, streams[0])
    before = client.public_world()
    code = next(code for code, n in before["pool_main"].items() if n)
    with sqlite3.connect(f"file:{RUN / 'cards.cdb'}?mode=ro", uri=True) as conn:
        host = conn.execute("SELECT id FROM datas WHERE type & 8388608 ORDER BY id LIMIT 1").fetchone()[0]
    assert client.board.zone(1, C.LOCATION_HAND)
    client.board.zone(1, C.LOCATION_HAND).pop()  # one anonymous opponent card is now the public material
    client.board.zones[0, C.LOCATION_MZONE] = [ShadowCard(code=host, controller=0, location=C.LOCATION_MZONE,
                                                     sequence=0, position=C.POS_FACEUP_ATTACK, owner=0)]
    material = ShadowCard(code=code, owner=1)
    client.board.materials[0, C.LOCATION_MZONE, 0] = [material]
    world = client.public_world()
    assert (code, False) in world["public_owned"]
    assert world["pool_main"].get(code, 0) == before["pool_main"][code] - 1
    material.owner = -1
    with pytest.raises(RuntimeError, match="unknown public material owner"):
        client.public_world()
    material.owner = 0
    with pytest.raises(RuntimeError, match="pool"):
        client.public_world()  # incorrectly attributing it to our monster's controller must not pass
