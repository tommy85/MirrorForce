from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace
import copy
import ctypes
import time

import numpy as np
import pytest

from mirrorforce.netduel import replay_filtered_particles as P
from mirrorforce.netduel import causal_replay as R
from mirrorforce.netduel.agent_causal_plan import PlanBudgetExceeded


def world():
    return {"decklist_public": True, "hand": [0, 0], "hand_group": [0], "deck": [0, 0], "extra": [0],
            "facedown": [[4, 0, 0, False], [8, 5, 0, False]],
            "pool_main": {11: 3, 22: 2, 33: 1}, "pool_extra": {44: 1},
            "unpositioned": {(2, 11): 1, (1, 22): 1}, "own_deck": [11, 22, 33], "own_deck_fixed": {0: 22},
            "types": {11: 1, 22: 2, 33: 0x80002, 44: 0x41}, "public_owned": []}


def test_sequential_proposals_preserve_all_public_counts_types_lower_and_own_fixed():
    public = world()
    proposer = P.SequentialProposal(public)
    for ordinal in range(50):
        layout, missing = proposer.sample(99, ordinal)
        assert Counter(layout['hand'] + layout['deck'] + [c for _, _, c in layout['facedown']]) == Counter(public['pool_main'])
        assert layout['hand'][0] == 11 and 22 in layout['deck']
        assert layout['facedown'] == [[4, 0, 11], [8, 5, 33]]
        assert layout['extra'] == [44] and Counter(layout['own_deck']) == Counter(public['own_deck'])
        assert layout['own_deck'][0] == 22 and missing == []
        assert proposer.sample(99, ordinal) == (layout, missing)
    assert public == world()  # the input never mutates


def test_no_private_truth_can_enter_proposals_and_known_slots_are_preserved():
    public = world()
    public['hand'][1] = 22
    public['pool_main'][22] -= 1
    left, right = P.SequentialProposal(public), P.SequentialProposal(public)
    # True layouts are deliberately unrelated objects outside the interface.
    hidden_truths = [{'hand': [11, 22]}, {'hand': [33, 11]}]
    assert hidden_truths[0] != hidden_truths[1]
    for i in range(20):
        assert left.sample(88, i) == right.sample(88, i)
        assert left.sample(88, i)[0]['hand'][1] == 22
    with pytest.raises(ValueError):
        P.SequentialProposal({**public, 'label:hidden_': [1]})


def test_actual_count_factor_increment_and_missing_candidates_are_declared():
    logits = np.zeros((1, 6, 4))
    logits[0, 1] = [0, 20, 40, 60]
    weighted = P.SequentialProposal(world(), {'codes': [11], 'locations': list(P.LOCATIONS), 'logits': logits.tolist()})
    uniform = P.SequentialProposal(world())
    hands = [weighted.sample(171, i)[0]['hand'].count(11) for i in range(200)]
    base = [uniform.sample(171, i)[0]['hand'].count(11) for i in range(200)]
    assert sum(hands) > sum(base)
    assert weighted.mode == 'count_head' and weighted.uniform_reason is None
    assert 22 in weighted.sample(171, 0)[1]
    no_candidates = P.SequentialProposal(world(), {'codes': [], 'locations': list(P.LOCATIONS), 'logits': []})
    assert no_candidates.mode == 'uniform' and no_candidates.uniform_reason == 'no_public_candidates'
    for bad in (np.full((1, 6, 4), np.nan), np.zeros((1, 5, 4))):
        with pytest.raises(ValueError):
            P.SequentialProposal(world(), {'codes': [11], 'locations': list(P.LOCATIONS), 'logits': bad.tolist()})


class FakeRoot:
    def __init__(self):
        self.snapshot = SimpleNamespace(verify=lambda: None)

    def _check(self):
        pass

    @contextmanager
    def branch(self):
        yield


class Driver:
    def __init__(self):
        self.pduel = object()

    def close(self):
        self.pduel = None


def bank():
    return P.ReplayFilteredBank(P.SequentialProposal(world()), object(), 2, 7, 'a'*64, 'b'*64, 'c'*64, 'd'*64, 0.)


@pytest.mark.parametrize('dropped', ['packet_mismatch', 'construction_timeout', 'replay_timeout'])
def test_one_dropped_candidate_does_not_destroy_accepted_prefix(monkeypatch, dropped):
    monkeypatch.setattr(P, 'at_root', lambda root: [])
    monkeypatch.setattr(P, 'root_layout', lambda history, layout: ((0, 1, 0, 11),))
    outcomes, drivers = [], []
    def trial(*args, **kwargs):
        outcomes.append(kwargs['deadline'])
        assert 0 < kwargs['deadline'] - time.monotonic() <= 2.
        if len(outcomes) == 2:
            raise R.ReplayCandidateDropped(dropped, 'registered test drop')
        driver = Driver()
        drivers.append(driver)
        return SimpleNamespace(driver=driver, proof={'replay_filter': {'construction_seconds': 0., 'replay_seconds': 0.}})
    monkeypatch.setattr(P, 'trial_replay_filtered_candidate', trial)
    with bank().open_roots(FakeRoot(), {}, seed=7, deadline=time.monotonic()+10) as roots:
        assert len(roots) == 2
        assert roots.admission['accepted_ordinals'] == [0, 2]
        assert roots.admission['discard_counts'] == {dropped: 1}
        assert roots.admission['attempted'] == 3 and roots.admission['exact_conditional_posterior'] is False
        assert all(driver.pduel is not None for driver in drivers)
    assert all(driver.pduel is None for driver in drivers)


def test_empty_and_partial_banks_are_valid_explicit_outcomes(monkeypatch):
    monkeypatch.setattr(P, 'at_root', lambda root: [])
    monkeypatch.setattr(P, 'root_layout', lambda history, layout: ())
    monkeypatch.setattr(P, 'PROPOSAL_LIMIT', 2)
    calls = []
    def trial(*args, **kw):
        calls.append(1)
        if len(calls) == 1:
            return SimpleNamespace(driver=Driver(), proof={'replay_filter': {'construction_seconds': 0., 'replay_seconds': 0.}})
        raise R.ReplayCandidateDropped('packet_mismatch', 'test')
    monkeypatch.setattr(P, 'trial_replay_filtered_candidate', trial)
    with bank().open_roots(FakeRoot(), {}, seed=7, deadline=time.monotonic()+10) as roots:
        assert len(roots) == 1 and roots.admission['status'] == 'partial'
    with bank().open_roots(FakeRoot(), {}, seed=7, deadline=time.monotonic()+10) as roots:
        assert len(roots) == 0 and roots.admission['status'] == 'empty'


def test_rule_and_cleanup_failures_are_hard_even_after_a_success(monkeypatch):
    monkeypatch.setattr(P, 'at_root', lambda root: [])
    monkeypatch.setattr(P, 'root_layout', lambda history, layout: ())
    calls, driver = [], Driver()
    def trial(*args, **kw):
        calls.append(1)
        if len(calls) == 1:
            return SimpleNamespace(driver=driver, proof={'replay_filter': {'construction_seconds': 0., 'replay_seconds': 0.}})
        raise RuntimeError('native Lua rule fault')
    monkeypatch.setattr(P, 'trial_replay_filtered_candidate', trial)
    with pytest.raises(RuntimeError, match='native Lua rule fault'):
        with bank().open_roots(FakeRoot(), {}, seed=7, deadline=time.monotonic()+10):
            pytest.fail('hard rule fault was yielded as a valid bank')
    assert driver.pduel is None
    monkeypatch.setattr(P, 'trial_replay_filtered_candidate', lambda *args, **kw: SimpleNamespace(
        driver=SimpleNamespace(pduel=object(), close=lambda: None),
        proof={'replay_filter': {'construction_seconds': 0., 'replay_seconds': 0.}}))
    with pytest.raises(RuntimeError, match='clean all'):
        with bank().open_roots(FakeRoot(), {}, seed=7, deadline=time.monotonic()+10):
            pass


def test_returned_driver_is_owned_before_a_hard_proof_copy_failure(monkeypatch):
    monkeypatch.setattr(P, 'at_root', lambda root: [])
    monkeypatch.setattr(P, 'root_layout', lambda history, layout: ())
    driver = Driver()
    class BrokenProof:
        def __deepcopy__(self, memo):
            raise RuntimeError('malformed returned proof')
    monkeypatch.setattr(P, 'trial_replay_filtered_candidate', lambda *args, **kw:
                        SimpleNamespace(driver=driver, proof=BrokenProof()))
    with pytest.raises(RuntimeError, match='malformed returned proof'):
        with bank().open_roots(FakeRoot(), {}, seed=7, deadline=time.monotonic()+10):
            pytest.fail('bad proof became a filtered drop')
    assert driver.pduel is None


def timed_audit(monkeypatch, *, late=False):
    from mirrorforce.netduel import causal_proof as proof
    monkeypatch.setattr(P, 'at_root', lambda root: [])
    monkeypatch.setattr(P, 'root_layout', lambda history, layout: ())
    monkeypatch.setattr(proof, 'verify_natural_root_proof', lambda *args, **kwargs: None)
    now, drivers = [0.], []
    monkeypatch.setattr(P.time, 'monotonic', lambda: now[0])
    def trial(*args, **kw):
        driver = Driver()
        drivers.append(driver)
        seconds = 4. if late and len(drivers) == 2 else .5
        now[0] += seconds
        return SimpleNamespace(driver=driver, proof={'replay_filter': {
            'law': 'first-mismatch-or-two-second-discard/v1', 'construction_seconds': seconds / 2,
            'replay_seconds': seconds / 2, 'historical_existence_node_budget': None, 'negative_certificate': False}})
    monkeypatch.setattr(P, 'trial_replay_filtered_candidate', trial)
    with bank().open_roots(FakeRoot(), {}, seed=7, deadline=10.) as roots:
        result = copy.deepcopy(roots.admission)
    assert all(driver.pduel is None for driver in drivers)
    return result


def validate(audit, **extra):
    return P.check_filtered_admission(audit, requested_count=2, proposal_seed=7,
                                     world_sha256='a'*64, obs_sha256='b'*64, **extra)


def test_late_positive_result_keeps_real_stage_times_and_only_discards_that_candidate(monkeypatch):
    audit = timed_audit(monkeypatch, late=True)
    validate(audit, expected_source_bundle=audit['source_bundle'])
    assert audit['status'] == 'partial' and audit['accepted_ordinals'] == [0]
    late = audit['attempts'][1]
    assert late['status'] == 'replay_timeout' and late['construction_seconds'] == late['replay_seconds'] == 2.
    assert late['seconds'] == 4. and late['deadline_remaining_seconds'] == -2.
    assert late['late_proof']['replay_filter']['construction_seconds'] == 2.


@pytest.mark.parametrize('tamper', ['accepted_200s', 'allocation', 'bank_total', 'proof_phase', 'row_deadline', 'missing_dependency', 'source_proof'])
def test_independent_admission_rejects_expired_or_tampered_time_and_source(monkeypatch, tamper):
    audit = timed_audit(monkeypatch)
    validate(audit)
    row = audit['attempts'][0]
    if tamper == 'accepted_200s':
        row.update(seconds=200., construction_seconds=100., replay_seconds=100., overhead_seconds=0.,
                   finished_seconds=200., deadline_remaining_seconds=-198.)
        audit.update(allocated_admission_seconds=.1, admission_seconds=200.)
    elif tamper == 'allocation':
        audit['allocated_admission_seconds'] = .1
    elif tamper == 'bank_total':
        audit['admission_seconds'] = 200.
    elif tamper == 'proof_phase':
        row['proof']['replay_filter']['construction_seconds'] = 0.
    elif tamper == 'row_deadline':
        row['deadline_seconds'] = 200.
    elif tamper == 'missing_dependency':
        del audit['source_bundle']['files']['agent/public_world_codec.py']
        body = {k: v for k, v in audit['source_bundle'].items() if k != 'sha256'}
        audit['source_bundle']['sha256'] = P._sha(body)
    else:
        row['proof']['replay_filter']['source_bundle_sha256'] = 'e'*64
    with pytest.raises(ValueError):
        validate(audit)


def test_source_bundle_covers_constructor_codec_and_real_replay_helpers_even_with_profile(monkeypatch):
    from mirrorforce.netduel import causal_profile as profile
    monkeypatch.setattr(profile, 'verify_bundle', lambda value: None)
    provider = P.PublicReplayFilteredParticles({'producer_source_sha256': 'e'*64})
    bundle = provider.identity['source_bundle']
    required = {'netduel/replay_filtered_particles.py', 'netduel/causal_replay.py',
                'netduel/agent_causal_plan.py', 'netduel/causal_linear.py', 'agent/public_world_codec.py',
                'agent/search/particles.py', 'common/client_replay_journal.py', 'common/client_root.py',
                'worldmodel/engine.py', 'netduel/agent_shuffle_plan.py', 'netduel/causal_proof.py'}
    assert required <= bundle['files'].keys() and len(bundle['files']) > 11
    assert provider.identity['producer_source_sha256'] == 'e'*64
    P.check_source_bundle(bundle, live=True)


def test_dependency_changes_during_candidate_remain_hard_and_close_returned_driver(monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(P, 'at_root', lambda root: [])
    monkeypatch.setattr(P, 'root_layout', lambda history, layout: ())
    selected_bank = bank()
    original, changed, driver = Path.read_bytes, [False], Driver()
    def read(path):
        data = original(path)
        return data + b'\n' if changed[0] and str(path).endswith('/agent/public_world_codec.py') else data
    monkeypatch.setattr(Path, 'read_bytes', read)
    def trial(*args, **kwargs):
        changed[0] = True
        return SimpleNamespace(driver=driver, proof={'replay_filter': {'construction_seconds': 0., 'replay_seconds': 0.}})
    monkeypatch.setattr(P, 'trial_replay_filtered_candidate', trial)
    with pytest.raises(ValueError, match='source changed'):
        with selected_bank.open_roots(FakeRoot(), {}, seed=7, deadline=time.monotonic()+10):
            pytest.fail('modified helper was admitted')
    assert driver.pduel is None


def test_cold_source_registration_rejects_a_missing_required_helper(monkeypatch):
    from pathlib import Path
    original = Path.is_file
    monkeypatch.setattr(P, '_SOURCE_PATHS', None)
    monkeypatch.setattr(Path, 'is_file', lambda path: False if str(path).endswith('/agent/public_world_codec.py') else original(path))
    with pytest.raises(ValueError, match='source closure lost'):
        P.source_bundle()


def test_filtered_trial_uses_positive_constructor_and_drops_timeout_without_certificates(monkeypatch):
    root = object.__new__(R.RootEnvelope)
    root.branch_session = object()
    def solve(*args, **kw):
        assert kw['mode'] == 'positive-replay' and kw['max_nodes'] is None and kw['deadline'] == 17.
        raise PlanBudgetExceeded('test two second constructor cap')
    monkeypatch.setattr(R, 'solve', solve)
    with pytest.raises(R.ReplayCandidateDropped) as caught:
        R.trial_replay_filtered_candidate(root, None, (), {}, events=[], seed=1, deadline=17., source_sha256='a'*64)
    assert caught.value.reason == 'construction_timeout'


def test_first_packet_divergence_never_backtracks_to_another_hidden_response(monkeypatch):
    from mirrorforce.netduel import constants as C
    witness = object.__new__(R._Witness)
    witness.root = SimpleNamespace(_check=lambda: None)
    witness.deadline, witness.viewer, witness.opponent = time.monotonic()+2., 0, 1
    witness.runtime_profile, witness.first_mismatch = None, True
    witness.history, witness.plan = object(), SimpleNamespace(root=[], history_sha256='a'*64, root_sha256='b'*64)
    witness.source_sha256 = 'c'*64
    witness.expected = (bytes([C.MSG_START, 0]), bytes([C.MSG_NEW_PHASE, 0, 1]))
    witness.cursor = witness.answered = witness.branches = witness.submitted_total = 0
    witness.choices, witness.native_trace, witness.native_before_response = [], [], []
    witness.outbox, witness.script_errors = [[], []], []
    witness.driver = SimpleNamespace(build=lambda: None, steps=0, finished=False, pduel=object(),
        _msgbuf=ctypes.create_string_buffer(b'xx'), _observe=lambda message: None,
        core=SimpleNamespace(log=[], initial_packets={0: (), 1: ()}, get_message=lambda *args: 2))
    witness.shuffle = SimpleNamespace(install=lambda *args, **kw: {})
    witness._process = lambda steps: R.PROCESSOR_BUFFER_LEN
    def forbidden():
        pytest.fail('first-mismatch proposal tried a second hidden response branch')
    witness._next = forbidden
    monkeypatch.setattr(R, 'steps_from_causal_plan', lambda *args: ([], {}))
    monkeypatch.setattr(R, 'split_messages', lambda raw: [SimpleNamespace(msg=C.MSG_NEW_TURN, payload=b'\0')])
    monkeypatch.setattr(R, 'project_message', lambda *args: {0: [bytes([C.MSG_NEW_TURN, 0])], 1: []})
    with pytest.raises(R.ReplayCandidateDropped) as caught:
        witness.run()
    assert caught.value.reason == 'packet_mismatch' and witness.cursor == 1
    assert caught.value.evidence['packet_index'] == 1
