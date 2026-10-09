"""Recorded online games replay through the search follower to their end, and a damaged record does not.

The fixtures are the viewer's own received stream and sent answers of two competition games on a YGOPro server
whose engine build and card scripts differ from ours (``tests/fixtures/follower``): no hidden information, nothing the client did not receive. They cover the server
differences the follower had to learn: select-message hint strings, a hint only one side sends, the upper bound of
select-unselect prompts, and a chain prompt offering fewer optional activations than the local engine.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import mf_runtime_follower_replay_audit as audit

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "follower"
GAMES = sorted(FIXTURES.glob("*.json"))
ROOM = {"start_lp": 8000, "start_hand": 5, "draw_count": 1, "duel_rule": 5}
DECK = Path(__file__).resolve().parents[1] / "decks" / "stage-a" / "SkyStriker.ydk"


@pytest.fixture(scope="module")
def engine():
    from mirrorforce.effectinfo import EffectInfoError, get_effectinfo_core
    from mirrorforce.netduel.cards import load_ydk
    from mirrorforce.worldmodel.engine import DeckList
    try:
        core = get_effectinfo_core()
    except (EffectInfoError, OSError) as exc:
        pytest.skip(f"needs the effectinfo core, card database and scripts: {exc}")
    main, extra, _ = load_ydk(DECK)
    return core, DeckList("recorded-own", tuple(main), tuple(extra))


def test_the_fixtures_are_present():
    assert [p.name for p in GAMES] == ["human-final-surrender.json", "league-game-natural-end.json"]


@pytest.mark.parametrize("path", GAMES, ids=lambda p: p.stem)
def test_a_recorded_game_is_followed_to_its_end(engine, path):
    core, deck = engine
    game = audit.load_game(path)
    assert audit.replay(core, game, deck, ROOM, 600.) is None


def test_a_record_missing_its_later_answers_is_reported_incomplete(tmp_path):
    path = FIXTURES / "league-game-natural-end.json"
    record = json.loads(path.read_text())
    record["policy_report"]["responses"] = record["policy_report"]["responses"][:100]
    copy_path = tmp_path / path.name
    copy_path.write_text(json.dumps(record))
    failure = audit.replay(None, audit.load_game(copy_path), None, ROOM, 600.)
    assert failure == {"kind": "incomplete", "prompts": 338, "answers": 100}


@pytest.mark.parametrize("path", GAMES, ids=lambda p: p.stem)
def test_a_changed_own_answer_is_a_divergence(engine, path, tmp_path):
    """Another answer in the middle of the game sends the local duel down another path than the recorded one."""
    core, deck = engine
    record = json.loads(path.read_text())
    responses = record["policy_report"]["responses"]
    index = len(responses) // 2
    raw = bytearray.fromhex(responses[index])
    raw[0] ^= 1
    responses[index] = raw.hex()
    copy_path = tmp_path / path.name
    copy_path.write_text(json.dumps(record))
    assert audit.replay(core, audit.load_game(copy_path), deck, ROOM, 600.) is not None


def test_a_changed_server_message_is_a_divergence(engine, tmp_path):
    """The first own draw the server reports with another card code: the follower must not accept it."""
    from mirrorforce.netduel import constants as C
    core, deck = engine
    path = FIXTURES / "league-game-natural-end.json"
    record = json.loads(path.read_text())
    messages = record["policy_report"]["public_messages"]
    for index, (msg, payload) in enumerate(messages):
        body = bytearray.fromhex(payload)
        if msg == C.MSG_DRAW and body[0] == record["result"]["our_player"] and body[1] >= 1:
            code = int.from_bytes(body[2:6], "little")
            body[2:6] = ((code & 0x80000000) | 89631139).to_bytes(4, "little")  # a card outside the deck
            messages[index] = [msg, body.hex()]
            break
    else:
        pytest.fail("no own draw in the fixture")
    copy_path = tmp_path / path.name
    copy_path.write_text(json.dumps(record))
    assert audit.replay(core, audit.load_game(copy_path), deck, ROOM, 600.) is not None
