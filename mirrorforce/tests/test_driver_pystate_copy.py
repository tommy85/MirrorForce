"""Snapshot copying shares frozen message leaves, never mutable rollback state."""

from dataclasses import FrozenInstanceError, dataclass, field
from types import SimpleNamespace

import pytest

from mirrorforce.puzzle.messages import Message
from mirrorforce.worldmodel.engine import DeckList, DuelConfig, DuelDriver, DuelError


def driver():
    deck = DeckList("fixture", (1,), (2,))
    return DuelDriver(DuelConfig((deck, deck), seed=17), core=SimpleNamespace(card_pool=lambda: object()))


def test_saved_and_restored_message_containers_are_independent():
    duel = driver()
    first, second = Message(1, b"first"), Message(2, b"second")
    duel.messages = [first, second]
    saved = duel.save_pystate()
    assert saved["messages"] == duel.messages
    assert saved["messages"] is not duel.messages
    assert saved["messages"][0] is first
    with pytest.raises(FrozenInstanceError):
        first.payload = b"changed"

    duel.messages.append(Message(3, b"branch"))
    duel.messages.pop(0)
    assert saved["messages"] == [first, second]
    duel.restore_pystate(saved)
    assert duel.messages == [first, second]
    assert duel.messages is not saved["messages"]
    duel.messages.clear()
    assert saved["messages"] == [first, second]
    duel.restore_pystate(saved)
    saved["messages"].append(Message(4, b"another branch"))
    assert duel.messages == [first, second]


@pytest.mark.parametrize("name", ("disclosure", "banish_origin", "responses", "_replay"))
def test_other_mutable_fields_still_use_recursive_copies(name):
    original = {"nested": [[1], {"events": [2]}]}
    saved = DuelDriver._copy_field(name, original)
    saved["nested"][0].append(3)
    saved["nested"][1]["events"].append(4)
    assert original == {"nested": [[1], {"events": [2]}]}


def test_context_rng_and_mutable_codes_remain_isolated():
    duel = driver()
    duel._ctx.known_codes = [123]
    expected_state = duel._ctx.rng.getstate()
    saved = duel.save_pystate()
    assert saved["_ctx"] is not duel._ctx
    assert saved["_ctx"].card_pool is duel._ctx.card_pool
    assert saved["_ctx"].rng is not duel._ctx.rng
    duel._ctx.known_codes.append(456)
    duel._ctx.rng.random()
    duel._rng.random()
    assert saved["_ctx"].known_codes == [123]
    assert saved["_ctx"].rng.getstate() == expected_state
    assert saved["_rng"].getstate() == expected_state
    duel.restore_pystate(saved)
    assert duel._ctx.rng.getstate() == expected_state
    assert duel._rng.getstate() == expected_state
    assert duel._ctx.known_codes == [123]
    duel._ctx.known_codes.append(789)
    assert saved["_ctx"].known_codes == [123]


def test_mutable_message_payloads_keep_deep_copy_semantics():
    original = [Message(1, bytearray(b"mutable"))]
    saved = DuelDriver._copy_field("messages", original)
    assert saved == original and saved[0] is not original[0]
    original[0].payload[0] = 0
    assert saved[0].payload == bytearray(b"mutable")


def test_message_subclasses_and_extra_attributes_keep_deep_copy_semantics():
    @dataclass(frozen=True)
    class ExtendedMessage(Message):
        metadata: list = field(default_factory=list)

    extra = Message(2, b"extra")
    object.__setattr__(extra, "metadata", [2])
    original = [ExtendedMessage(1, b"subclass", [1]), extra]
    saved = DuelDriver._copy_field("messages", original)
    original[0].metadata.append(3)
    original[1].metadata.append(4)
    assert saved[0].metadata == [1]
    assert saved[1].metadata == [2]


def test_a_future_message_schema_is_not_implicitly_shared(monkeypatch):
    from mirrorforce.worldmodel import engine

    @dataclass(frozen=True)
    class FutureMessage:
        msg: int
        payload: bytes
        metadata: list = field(default_factory=list)

    monkeypatch.setattr(engine, "Message", FutureMessage)
    original = [FutureMessage(1, b"future", [1])]
    saved = DuelDriver._copy_field("messages", original)
    original[0].metadata.append(2)
    assert saved[0].metadata == [1]


def test_nonstandard_message_list_keeps_its_type_and_mutable_attributes():
    class MessageList(list):
        pass

    original = MessageList([Message(1, b"message")])
    original.metadata = [1]
    saved = DuelDriver._copy_field("messages", original)
    assert type(saved) is MessageList and saved is not original
    original.metadata.append(2)
    assert saved.metadata == [1]


def test_unfreezing_the_message_schema_restores_deep_copy(monkeypatch):
    from mirrorforce.worldmodel import engine

    @dataclass
    class MutableMessage:
        msg: int
        payload: bytes

    monkeypatch.setattr(engine, "Message", MutableMessage)
    original = [MutableMessage(1, b"before")]
    saved = DuelDriver._copy_field("messages", original)
    original[0].payload = b"after"
    assert saved[0].payload == b"before"


def test_unclassified_driver_fields_still_fail_closed():
    duel = driver()
    duel.new_mutable_state = []
    with pytest.raises(DuelError, match="new_mutable_state"):
        duel.save_pystate()
