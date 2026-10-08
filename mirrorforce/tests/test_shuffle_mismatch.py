"""Narrow native replay classification: a wrong witness is not a corrupt ABI."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mirrorforce.netduel import constants as C, causal_replay as R
from mirrorforce.netduel.agent_shuffle_plan import ShufflePlanError, ShufflePlanReplayError, ShuffleStep

DECK = ShuffleStep(C.MSG_SHUFFLE_DECK, 1, C.LOCATION_DECK, tuple(range(30)),
                   tuple(range(30)), (63166095,) * 30)
STEPS = (DECK,) * 18
STATE = (1, 1, 18, 14, 2, C.MSG_SHUFFLE_HAND, 1, C.LOCATION_HAND, 3)
BEFORE = (1, 1, 18, 14, 0, C.MSG_SHUFFLE_DECK, 1, C.LOCATION_DECK, 31)


def test_real_gui6_state_is_only_a_different_historical_event():
    detail = R._ordinary_shuffle_mismatch(STATE, STEPS, before=BEFORE)
    assert detail["actual_event"] == [33, 1, 2, 3]
    assert detail["expected_event"] == [32, 1, 1, 30]
    assert detail["scope"] == "one-historical-response-branch-only"


@pytest.mark.parametrize("index,value", [(0, 2), (1, 0), (1, 2), (2, 17), (3, 18),
    (4, 1), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
    (5, C.MSG_SHUFFLE_SET_CARD), (6, 2), (7, C.LOCATION_DECK), (8, 0), (8, 256), (3, True)])
def test_other_faults_or_invalid_native_metadata_remain_hard(index, value):
    state = list(STATE)
    state[index] = value
    assert R._ordinary_shuffle_mismatch(tuple(state), STEPS, before=BEFORE) is None


def test_same_event_internal_shape_fault_and_field_plan_stay_hard():
    same = (1, 1, 18, 14, 2, 32, 1, 1, 30)
    assert R._ordinary_shuffle_mismatch(same, STEPS, before=BEFORE) is None
    field = ShuffleStep(C.MSG_SHUFFLE_SET_CARD, 1, C.LOCATION_SZONE, (0, 1), (1, 0), (63166095,) * 2)
    assert R._ordinary_shuffle_mismatch(STATE, (field,) * 18, before=BEFORE) is None
    with pytest.raises(ShufflePlanError):
        R._ordinary_shuffle_mismatch(STATE, (replace(DECK, destination_from_source=(0,) * 30),) * 18,
                                     before=BEFORE)


def witness(error, *, log=(), state=STATE, before=BEFORE):
    obj = object.__new__(R._Witness)
    current = before
    def process(_):
        nonlocal current
        current = state
        raise error
    obj.shuffle = SimpleNamespace(process=process, state=lambda _: current)
    obj.driver = SimpleNamespace(pduel=7, core=SimpleNamespace(log=list(log)))
    obj.shuffle_mismatch_count, obj.last_shuffle_mismatch = 0, None
    obj.last_counterexample = {"must_not_reuse": "old packet mismatch"}
    return obj


def test_typed_fault_discards_stale_packet_certificate_without_touching_native_state():
    obj = witness(ShufflePlanReplayError(STATE))
    with pytest.raises(R._Mismatch):
        obj._process(STEPS)
    assert obj.shuffle_mismatch_count == 1 and obj.last_counterexample is None
    assert obj.last_shuffle_mismatch == R._ordinary_shuffle_mismatch(STATE, STEPS, before=BEFORE)
    assert obj.shuffle.state(7) == STATE  # only the owned snapshot rollback may clear the fault


@pytest.mark.parametrize("kind", ["generic", "script", "stale", "subclass", "same_event"])
def test_no_generic_error_message_matching_or_simultaneous_failure_downgrade(kind):
    class Foreign(ShufflePlanReplayError):
        pass
    error = ShufflePlanError(str(ShufflePlanReplayError(STATE))) if kind == "generic" else \
        Foreign(STATE) if kind == "subclass" else ShufflePlanReplayError(STATE)
    obj = witness(error, log=["native Lua error"] if kind == "script" else (),
                  state=(1, 1, 18, 14, 0, 33, 1, 2, 3) if kind == "stale" else STATE)
    if kind == "same_event":
        obj = witness(ShufflePlanReplayError((1, 1, 18, 14, 2, 32, 1, 1, 30)),
                      state=(1, 1, 18, 14, 2, 32, 1, 1, 30))
    with pytest.raises(ShufflePlanError):
        obj._process(STEPS)
    assert obj.shuffle_mismatch_count == 0 and obj.last_counterexample is not None


def test_set_group_fault_with_stale_ordinary_event_fields_is_hard():
    # operations.cpp can fail(EVENT_SHAPE) before consume updates last_*.
    before = STATE[:4] + (0,) + STATE[5:]
    obj = witness(ShufflePlanReplayError(STATE), before=before)
    with pytest.raises(ShufflePlanReplayError):
        obj._process(STEPS)
    assert obj.shuffle_mismatch_count == 0 and obj.last_counterexample is not None


def test_process_that_already_advanced_plan_cursor_cannot_downgrade_a_later_fault():
    # A successful ordinary event then SET_GROUP failure may change last_*,
    # but those fields still need not describe the actual failing event.
    before = BEFORE[:3] + (13,) + BEFORE[4:]
    obj = witness(ShufflePlanReplayError(STATE), before=before)
    with pytest.raises(ShufflePlanReplayError):
        obj._process(STEPS)
    assert obj.shuffle_mismatch_count == 0 and obj.last_counterexample is not None
