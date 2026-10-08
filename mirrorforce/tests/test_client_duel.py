"""Client mode of the env (cxx/duelpool/duel/client_driver.h ClientDuel, mirrorforce/netduel/agent_client.py).

- The module names the observation laws that make env and client agree by construction: ``card_view_law``
  (refreshed_view/v1: monsters' level, ATK, DEF and status as the server last refreshed them) and
  ``pending_source_law`` (own_decisions/v1: a player's candidate-selection source from its own decisions).
- A seat's client builder, fed the seat's received stream through the in-process host, shows every decision's menu
  and arrays as the env does and sends the env's responses (probes/client_parity.py; more games and decks in
  tests/test_client_parity.py and tools/mf_runtime_client_parity.py).
- The server's own messages go to the card view, never to the native side; a refused response (MSG_RETRY) and a
  message while a decision is pending are errors.

Runs in the venv with ``MF_DUEL_NATIVE``; run it in its own pytest process (two copies of the core).
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import random
import shutil
import sqlite3

import pytest

RUN = Path(os.environ.get("MF_SCRIPTED_RUN", "/path/to/workspace/ygopro-client-run"))
TABLES = next((Path(__file__).resolve().parent / "fixtures" / "announce").glob("announce-tables-*.json"))
PINNED = Path(__file__).resolve().parents[1] / "decks/stage-a/SkyStriker.ydk"
CONFIG = {"max_options": 192}


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    path = os.environ.get("MF_DUEL_NATIVE")
    if not path or not (RUN / "cards.cdb").is_file():
        pytest.skip("set MF_DUEL_NATIVE to a built duel_native module (and have the card database and scripts)")
    spec = importlib.util.spec_from_file_location("duel_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with sqlite3.connect(f"file:{RUN / 'cards.cdb'}?mode=ro", uri=True) as conn:
        codes = sorted(row[0] for row in conn.execute("SELECT id FROM datas"))
    code_list = tmp_path_factory.mktemp("client") / "code_list.txt"
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
def host_core(native, tmp_path_factory):
    from mirrorforce.puzzle.core import get_core
    module_core = Path(os.environ["MF_DUEL_NATIVE"]).resolve().parent / "libmfcore.so"
    path = tmp_path_factory.mktemp("client-host") / "libmfcore-host.so"
    shutil.copyfile(module_core, path)
    return get_core(lib_path=path, db_path=RUN / "cards.cdb", script_dirs=[RUN / "script"], mode=ctypes.RTLD_LOCAL)


def deal(seed):
    from mirrorforce.worldmodel.engine import load_ydk
    recipe = load_ydk(PINNED)
    rng = random.Random(seed)
    orders = [list(recipe.main), list(recipe.main)]
    for order in orders:
        rng.shuffle(order)
    return {"seed_words": [rng.getrandbits(32) for _ in range(8)], "deck_orders": orders,
            "extra": [list(recipe.extra)] * 2, "start_lp": 8000, "start_hand": 5, "draw_count": 1,
            "duel_options": 5 << 16}


def test_the_module_names_its_observation_laws(native):
    assert native.card_view_law == "refreshed_view/v1"
    assert native.pending_source_law == "own_decisions/v1"
    assert native.unpositioned_law == "reveal_returns/v1"
    assert native.public_effects_law == "public_effects/v1"
    assert native.placement_ref_law == "received_placement_card_ref/v1"
    assert tuple(native.observation_laws) == (native.card_view_law, native.pending_source_law, native.unpositioned_law,
                                              native.public_effects_law, native.placement_ref_law)


@pytest.mark.parametrize("seed", [21, 22])
def test_each_seat_builds_the_envs_observations_from_its_stream(native, host_core, seed):
    from mirrorforce.probes import client_parity as V
    game = V.play_game(native, deal(seed), CONFIG, V.active_rule(seed))
    streams = V.seat_streams(host_core, game)
    for seat in (0, 1):
        report = V.check_seat(native, game, seat, streams[seat], CONFIG, law=V.REFRESHED_VIEW)
        assert report["equal"], report["mismatches"]
        assert report["checked_decisions"] == len(game.decisions[seat]) and report["forced"] > 0


def test_server_messages_and_refusals(native, host_core):
    from mirrorforce.netduel import constants as C
    from mirrorforce.netduel.agent_client import SERVER_ONLY, AgentClientDuel
    from mirrorforce.probes import client_parity as V
    d = deal(23)
    game = V.play_game(native, d, CONFIG, V.random_rule(23))
    stream = V.seat_streams(host_core, game)[0]
    raw = native.ClientDuel(0, sorted(d["deck_orders"][0]), list(d["extra"][0]), dict(CONFIG))
    for msg in (C.MSG_START, C.MSG_WAITING, C.MSG_UPDATE_DATA, C.MSG_UPDATE_CARD):
        with pytest.raises(RuntimeError, match="the server's"):
            raw.feed(msg, b"\x00")
    with pytest.raises(RuntimeError, match="MSG_RETRY"):
        raw.feed(C.MSG_RETRY, b"")
    client = AgentClientDuel(native, 0, d["deck_orders"][0], d["extra"][0], CONFIG)
    at = None
    for at, (msg, payload) in enumerate(stream):
        client.feed(msg, payload)
        if client.prompt() is not None:
            break
    later = next((m, p) for m, p in stream[at + 1:] if m not in SERVER_ONLY)
    with pytest.raises(RuntimeError, match="decision is pending"):
        client.feed(*later)


def test_a_deck_with_a_pendulum_card_is_refused(native):
    """The client's card view does not follow face-up Extra Deck cards: a pendulum deck is refused, not approximated."""
    from mirrorforce.worldmodel.engine import load_ydk
    recipe = load_ydk(PINNED)
    odd_eyes = 16178681  # Odd-Eyes Pendulum Dragon
    with pytest.raises(RuntimeError, match="pendulum"):
        native.ClientDuel(0, sorted(list(recipe.main[1:]) + [odd_eyes]), list(recipe.extra), dict(CONFIG))


def test_public_opponent_recipe_is_required_only_in_declared_mode(native):
    from mirrorforce.netduel.agent_client import AgentClientDuel
    d = deal(31)
    config = {**CONFIG, "public_opponent_recipe": True}
    args = (0, d["deck_orders"][0], d["extra"][0])
    with pytest.raises(RuntimeError, match="explicitly declared"):
        native.ClientDuel(*args, config)
    with pytest.raises(ValueError, match="explicitly declared"):
        AgentClientDuel(native, *args, config)
    with pytest.raises(RuntimeError, match="closed-decklist"):
        native.ClientDuel(*args, CONFIG, opponent_main=d["deck_orders"][1], opponent_extra=d["extra"][1])


@pytest.mark.parametrize("seat", [0, 1])
def test_public_recipe_client_matches_native_and_clone_preserves_it(native, host_core, seat):
    import numpy as np
    from mirrorforce.netduel.agent_client import AgentClientDuel
    from mirrorforce.netduel.agent_public_recipe import declare
    from mirrorforce.probes import client_parity as V
    d = deal(32)
    config = {**CONFIG, "public_opponent_recipe": True}
    # The fixture declares the deck CONTENT before play. No current hand or order
    # is obtained from a running engine by the client constructor.
    main, extra = sorted(d["deck_orders"][1 - seat]), sorted(d["extra"][1 - seat])
    game = V.play_game(native, d, config, V.random_rule(32))
    streams = V.seat_streams(host_core, game)

    def factory(n, s, m, e, cfg):
        return AgentClientDuel(n, s, m, e, cfg, opponent_main=main, opponent_extra=extra)

    report = V.check_seat(native, game, seat, streams[seat], config, opponent_recipe=declare(main, extra),
                          law=V.REFRESHED_VIEW)
    assert report["equal"], report["mismatches"]
    client = factory(native, seat, d["deck_orders"][seat], d["extra"][seat], config)
    for msg, payload in streams[seat]:
        client.feed(msg, payload)
        if client.prompt() is not None:
            before = client.observation()
            assert before["obs:opponent_recipe_"][:, 3].sum() == len(main) + len(extra)
            twin = client.clone()
            assert twin.public_opponent_recipe == client.public_opponent_recipe
            after = twin.observation()
            for key in before:
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)
            break
    else:
        raise AssertionError("no client decision")


def test_public_recipe_input_is_equal_behind_two_different_hidden_hands(native, host_core):
    import copy
    from collections import Counter
    import numpy as np
    from mirrorforce.netduel.agent_client import AgentClientDuel
    from mirrorforce.probes import client_parity as V
    first = deal(33)
    second = copy.deepcopy(first)
    second["deck_orders"][1].reverse()
    assert Counter(first["deck_orders"][1][:5]) != Counter(second["deck_orders"][1][:5])
    config = {**CONFIG, "public_opponent_recipe": True}
    declared_main, declared_extra = sorted(first["deck_orders"][1]), sorted(first["extra"][1])
    roots = []
    for d in (first, second):
        game = V.play_game(native, d, config, V.random_rule(33))
        stream = V.seat_streams(host_core, game)[0]
        client = AgentClientDuel(native, 0, first["deck_orders"][0], first["extra"][0], config,
                              opponent_main=declared_main, opponent_extra=declared_extra)
        prefix = []
        for msg, payload in stream:
            prefix.append((msg, payload))
            client.feed(msg, payload)
            if client.prompt() is not None:
                roots.append((prefix, client.prompt(), client.observation()))
                break
    assert len(roots) == 2 and roots[0][:2] == roots[1][:2]
    for key, value in roots[0][2].items():
        np.testing.assert_array_equal(value, roots[1][2][key], err_msg=key)


def test_respond_replays_one_players_raw_responses(native):
    """ScriptedDuel.respond(bytes): a recorded game replays with one player answering by its raw response bytes (as a
    real game's opponent is known), the other by its menu rows; the game ends the same and every observation of the
    menu-stepped player is the original's (obs: keys; info:step_limit counts the responder's sub-decisions too).
    """
    import numpy as np
    d = deal(24)
    duel = native.ScriptedDuel(d, CONFIG)
    duel.start()
    rng = random.Random(24)
    record = []  # (player, row, the response the step produced or None, player 0's observation)
    while (prompt := duel.prompt()) is not None:
        obs = {k: v.copy() for k, v in duel.observation().items()} if prompt[0] == 0 else None
        before = len(duel.responses())
        row = rng.randrange(len(prompt[2]))
        duel.step(row)
        produced = bytes(duel.responses()[before]) if len(duel.responses()) > before and prompt[0] == 1 else None
        record.append((prompt[0], row, produced, obs))
    replay = native.ScriptedDuel(d, CONFIG)
    replay.start()
    k = compared = responded = 0
    while (prompt := replay.prompt()) is not None:
        player, row, produced, obs = record[k]
        assert prompt[0] == player
        if player == 0:
            again = replay.observation()
            for key in obs:
                if key.startswith("obs:"):  # info:step_limit counts the responding player's decisions too
                    np.testing.assert_array_equal(obs[key], again[key], err_msg=f"{key} at decision {k}")
            compared += 1
            replay.step(row)
            k += 1
            continue
        while record[k][2] is None:  # a selection's sub-choices: the raw response answers the whole prompt
            k += 1
        replay.respond(record[k][2])
        responded += 1
        k += 1
    assert k == len(record) and replay.winner == duel.winner and compared > 20 and responded > 20


def test_a_clone_fed_the_same_continuation_equals_the_original(native, host_core):
    """AgentClientDuel.clone() (native ClientDuel.Clone plus the board): at several points of each seat's stream, a clone
    and the original fed the same remaining messages and stepped by the same rows give byte-identical arrays, menus
    and responses; a clone made at a pending prompt, also inside a multi-card selection, rebuilds its menu."""
    import numpy as np
    from mirrorforce.netduel.agent_client import AgentClientDuel
    from mirrorforce.probes import client_parity as V

    def same(a, b, where):
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=f"{key} {where}")

    totals = {"clones": 0, "selection": 0, "checked": 0}
    for seed in (25, 26):
        d = deal(seed)
        game = V.play_game(native, d, CONFIG, V.random_rule(seed))
        streams = V.seat_streams(host_core, game)
        for seat in (0, 1):
            records = game.decisions[seat]
            cuts = {5, len(records) // 3, len(records) // 2}
            client = AgentClientDuel(native, seat, d["deck_orders"][seat], d["extra"][seat], CONFIG)
            clones, k, inside_selection, selection_cut = [], 0, False, False
            for msg, payload in streams[seat]:
                out = client.feed(msg, payload)
                for c in clones:
                    assert c.feed(msg, payload) == out
                while (prompt := client.prompt()) is not None:
                    if k in cuts or (inside_selection and not selection_cut):
                        clones.append(client.clone())
                        cuts.discard(k)
                        if inside_selection:
                            selection_cut = True
                    obs = client.observation()
                    for c in clones:
                        assert c.prompt() == prompt
                        same(obs, c.observation(), f"seed {seed} seat {seat} decision {k}")
                        totals["checked"] += 1
                    out = client.step(records[k].index)
                    for c in clones:
                        assert c.step(records[k].index) == out
                    inside_selection = out is None and client.prompt() is not None  # a sub-choice follows
                    k += 1
            assert k == len(records)
            totals["clones"] += len(clones)
            totals["selection"] += int(selection_cut)
    assert totals["clones"] >= 12 and totals["selection"] >= 1 and totals["checked"] > 400, totals


def test_step_response_follows_the_recorded_bytes(native, host_core):
    """AgentClientDuel.step_response(bytes): each seat's client answered by the response bytes its decisions produced
    (a multi-card selection's whole response at its first sub-choice) gives, at every decision, the menus and arrays
    of a client stepped by menu rows, and sends the same bytes; bytes no path gives are refused."""
    import numpy as np
    from mirrorforce.netduel.agent_client import AgentClientDuel
    from mirrorforce.probes import client_parity as V
    selections = 0
    for seed in (27, 28):
        d = deal(seed)
        game = V.play_game(native, d, CONFIG, V.random_rule(seed))
        streams = V.seat_streams(host_core, game)
        for seat in (0, 1):
            records = game.decisions[seat]
            by_rows = AgentClientDuel(native, seat, d["deck_orders"][seat], d["extra"][seat], CONFIG)
            # the response each decision ended with (None for a selection's sub-choices before its last)
            responses, k = [], 0
            for msg, payload in streams[seat]:
                by_rows.feed(msg, payload)
                while by_rows.prompt() is not None:
                    responses.append(by_rows.step(records[k].index))
                    k += 1
            assert k == len(records)
            by_bytes = AgentClientDuel(native, seat, d["deck_orders"][seat], d["extra"][seat], CONFIG)
            again = AgentClientDuel(native, seat, d["deck_orders"][seat], d["extra"][seat], CONFIG)
            by_path = AgentClientDuel(native, seat, d["deck_orders"][seat], d["extra"][seat], CONFIG)
            k = 0
            for msg, payload in streams[seat]:
                assert by_bytes.feed(msg, payload) == again.feed(msg, payload) == by_path.feed(msg, payload)
                while by_bytes.prompt() is not None:
                    assert by_bytes.prompt() == again.prompt()
                    a, b = by_bytes.observation(), again.observation()
                    for key in a:
                        np.testing.assert_array_equal(a[key], b[key], err_msg=f"{key} seed {seed} seat {seat} at {k}")
                    end = k
                    while responses[end] is None:
                        end += 1
                    selections += int(end > k)
                    if end == k and k % 7 == 0:
                        with pytest.raises(RuntimeError, match="no menu path gives the response"):
                            by_bytes.step_response(b"\xfe\xfe\xfe\xfe\xfe\xfe\xfe")
                        with pytest.raises(RuntimeError, match="no menu path gives the response"):
                            by_path.response_path(b"\xfe\xfe\xfe\xfe\xfe\xfe\xfe")
                    path = by_path.response_path(responses[end])
                    assert path == by_path.response_path(responses[end])
                    assert path == [records[j].index for j in range(k, end + 1)]
                    # Decoding, including a rejected response, never consumes observations.
                    after_decode = by_path.observation()
                    for key in a:
                        np.testing.assert_array_equal(a[key], after_decode[key], err_msg=f"read-only {key}")
                    assert by_bytes.step_response(responses[end]) == responses[end]
                    for j in range(k, end + 1):
                        path_obs = by_path.observation()
                        row_obs = again.observation()
                        for key in row_obs:
                            np.testing.assert_array_equal(path_obs[key], row_obs[key], err_msg=f"path {key}")
                        assert by_path.step(path[j - k]) == responses[j]
                        assert again.step(records[j].index) == responses[j]
                    assert by_bytes.prompt() == again.prompt() == by_path.prompt()
                    k = end + 1
            assert k == len(records)
    assert selections > 0


def test_open_stream_rebuilds_every_real_client_subdecision_and_memory(native, host_core):
    """The new replay primitive matches live row-stepping on both seats' native packet streams.

    A deterministic observation-sensitive test backend proves forward count/order/memory, not JAX arithmetic or
    Option A's yet-to-be-built particle-to-packet producer. All recipe input is explicitly declared before play.
    """
    from test_open_stream import MemoryBackend
    from mirrorforce.netduel.agent_public_recipe import declare
    from mirrorforce.probes import client_parity as V
    from tools import mf_runtime_policy_service as S
    selections = total = 0
    config = {**CONFIG, "public_opponent_recipe": True}
    for seed in (27, 28):
        d = deal(seed)
        game = V.play_game(native, d, config, V.random_rule(seed))
        streams = V.seat_streams(host_core, game)
        for seat in (0, 1):
            svc = S.Service(native, MemoryBackend(), config, "sample", 1,
                            {"opponent_recipe_mode": "mirror"})
            opening = {"seat": seat, "main": sorted(d["deck_orders"][seat]), "extra": sorted(d["extra"][seat]),
                       "seed": seed, "opponent_recipe_mode": "mirror",
                       "public_opponent_recipe": declare(d["deck_orders"][seat], d["extra"][seat])}
            live = svc.sessions[svc.open(opening)["session"]]
            frames, pending, expected = [], [], []
            k = 0
            for msg, payload in streams[seat]:
                pending.append([msg, payload.hex()])
                response = live.client.feed(msg, payload)
                subdecision = 0
                while (prompt := live.client.prompt()) is not None:
                    obs = live.client.observation()
                    before = S.memory_sha256(live.state)
                    live.state, _, _, _ = svc.backend.act(obs, live.state, live.first, len(prompt[2]))
                    live.first = False
                    row = game.decisions[seat][k].index
                    expected.append({"frame": len(frames), "subdecision": subdecision, "row": row,
                                     "msg": prompt[1], "obs_sha256": S.observation_sha256(obs),
                                     "memory_before_sha256": before,
                                     "memory_after_sha256": S.memory_sha256(live.state)})
                    response = live.client.step(row)
                    k += 1
                    subdecision += 1
                selections += int(subdecision > 1)
                if response is not None:
                    frames.append({"messages": pending, "response": bytes(response).hex()})
                    pending = []
            assert k == len(game.decisions[seat])
            result = svc.dispatch({"op": "open_stream", **opening, "frames": frames, "messages": pending})
            assert result["decisions"] == k
            assert result["trace"] == expected
            assert result["memory_sha256"] == S.memory_sha256(live.state)
            rebuilt = svc.sessions[result["session"]]
            assert rebuilt.client.prompt() == live.client.prompt() is None
            assert rebuilt.client.forced_count() == live.client.forced_count()
            total += k
    assert selections > 0 and total > 200
