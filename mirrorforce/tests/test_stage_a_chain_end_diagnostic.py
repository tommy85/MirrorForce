"""Regress the post-negation CHAIN_END attribution gap across public parsers.

The protocol supplies CHAIN_END even when activation negation skips SOLVED.
The original four strict expected failures are now normal assertions. Normal
SOLVED handling and within-resolution attribution remain positive controls.
"""
import struct

import pytest

from mirrorforce.netduel import constants as C
from mirrorforce.puzzle.messages import Message
from mirrorforce.worldmodel.state import (
    CHAIN_NEGATED, CHAIN_SOLVED, CHAIN_SOLVING, chain_link_after_message,
    parse_chain, parse_moves, parse_targets, parse_lp_events, parse_public_events,
)


def messages_and_reader(kind):
    if kind == "move":
        event = Message(C.MSG_MOVE, struct.pack("<IIII", 100, C.LOCATION_HAND << 8, C.LOCATION_SZONE << 8, 0))
        reader = lambda messages: parse_moves(messages)[0]
    elif kind == "draw":
        event = Message(C.MSG_DRAW, bytes([0, 1]) + struct.pack("<I", 100))
        reader = lambda messages: parse_moves(messages)[2]
    elif kind == "position":
        event = Message(C.MSG_POS_CHANGE, struct.pack("<I", 100) + bytes([0, C.LOCATION_MZONE, 0, 1, 4]))
        reader = lambda messages: parse_moves(messages)[1]
    elif kind == "target":
        event = Message(C.MSG_BECOME_TARGET, b"\x01" + struct.pack("<I", C.LOCATION_MZONE << 8))
        reader = parse_targets
    elif kind == "lp":
        event = Message(C.MSG_DAMAGE, b"\x00" + struct.pack("<I", 500))
        reader = parse_lp_events
    else:
        event = Message(C.MSG_TOSS_DICE, bytes([0, 1, 6]))
        reader = parse_public_events
    return event, reader


@pytest.mark.parametrize("kind", ["move", "draw", "position", "target", "lp", "public"])
def test_existing_parser_keeps_real_resolution_and_normal_completion_controls(kind):
    event, reader = messages_and_reader(kind)
    beginning = [Message(C.MSG_CHAIN_SOLVING, b"\x01")]
    assert reader(beginning + [event])[0].link == 1
    assert reader(beginning + [Message(C.MSG_CHAIN_SOLVED, b"\x01"), event])[0].link == 0


@pytest.mark.parametrize("kind", ["move", "draw", "position", "target", "lp", "public"])
def test_event_after_explicit_chain_end_has_no_previous_resolution_link(kind):
    event, reader = messages_and_reader(kind)
    messages = [Message(C.MSG_CHAIN_NEGATED, b"\x01"), Message(C.MSG_CHAIN_SOLVING, b"\x01"),
        Message(C.MSG_CHAIN_END, b""), event]
    assert reader(messages)[0].link == 0


@pytest.mark.parametrize("kind", ["move", "draw", "position", "target", "lp", "public"])
def test_negation_of_lower_link_does_not_close_the_effect_currently_resolving(kind):
    event, reader = messages_and_reader(kind)
    messages = [Message(C.MSG_CHAIN_SOLVING, b"\x02"), Message(C.MSG_CHAIN_NEGATED, b"\x01"), event,
        Message(C.MSG_CHAIN_SOLVED, b"\x02"), Message(C.MSG_CHAIN_SOLVING, b"\x01"), event,
        Message(C.MSG_CHAIN_END, b""), event, Message(C.MSG_CHAIN_SOLVING, b"\x03"), event]
    assert [row.link for row in reader(messages)] == [2, 1, 0, 3]


def test_negated_link_never_gets_a_fabricated_solved_record():
    messages = [Message(C.MSG_CHAIN_NEGATED, b"\x01"), Message(C.MSG_CHAIN_SOLVING, b"\x01"),
        Message(C.MSG_CHAIN_END, b"")]
    records = parse_chain(messages)
    assert [(row.kind, row.link, row.trace_index) for row in records] == [
        (CHAIN_NEGATED, 1, 0), (CHAIN_SOLVING, 1, 1)]
    assert all(row.kind != CHAIN_SOLVED for row in records)


def test_attribution_keeps_activation_and_resolution_contexts_distinct():
    body = bytes(15) + b"\x02"
    assert chain_link_after_message(0, C.MSG_CHAINING, body) == 2
    assert chain_link_after_message(0, C.MSG_CHAINING, body, include_chaining=False) == 0
    assert chain_link_after_message(0, C.MSG_CHAINING, b"\xff") == 0
    for include_chaining in (True, False):
        assert chain_link_after_message(2, C.MSG_CHAIN_END, b"", include_chaining=include_chaining) == 0
