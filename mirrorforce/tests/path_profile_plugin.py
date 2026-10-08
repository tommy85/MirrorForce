"""Read-only test-process timing wrappers; never installed in a running job.

Run only the fixed native public-prefix regression with this pytest plugin.
Inclusive durations overlap. Exclusive durations subtract DIRECT instrumented
children and can be added within the same packet context. Original RPCs in
that regression are recorded-output fixtures, so these are CPU lifecycle
measurements, not GPU/queue/network latency or production speedup estimates.
"""
from collections import defaultdict
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import statistics
import threading
import time

_local = threading.local()
_rows = defaultdict(list)
_ledgers = []
_wrapped = []
_sources = {}
_started = time.perf_counter()


def _wrap(owner, name, label):
    original = getattr(owner, name)
    @wraps(original)
    def measured(*args, **kwargs):
        stack = getattr(_local, 'stack', None)
        if stack is None:
            stack = _local.stack = []
        phase = label(args, kwargs) if callable(label) else label
        context = phase if phase.startswith('wire.') else stack[-1]['context'] if stack else 'outside_packets'
        row = {'phase': phase, 'context': context, 'children': 0., 'started': time.perf_counter_ns()}
        stack.append(row)
        error = False
        try:
            value = original(*args, **kwargs)
            if phase == 'gate.finish' and context.startswith('wire.'):
                _ledgers.append(dict(value))
            return value
        except BaseException:
            error = True
            raise
        finally:
            elapsed = (time.perf_counter_ns() - row['started']) / 1e9
            assert stack.pop() is row
            if stack:
                stack[-1]['children'] += elapsed
            _rows[(phase, context)].append((elapsed, max(0., elapsed-row['children']), error))
    setattr(owner, name, measured)
    _wrapped.append({'owner': getattr(owner, '__name__', repr(owner)), 'name': name})


def pytest_sessionstart(session):
    from mirrorforce.netduel import client, board, actions, disclosure, agent_search_policy as P, agent_search_gate as G
    from mirrorforce.common import client_root as R, client_shadow as S, client_root_menu as M
    from mirrorforce.worldmodel import engine as E
    from mirrorforce.search import banish_origin as B
    modules = (client, board, actions, disclosure, P, G, R, S, M, E, B)
    for module in modules:
        path = Path(module.__file__)
        _sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    _wrap(client.NetDuelClient, '_game_msg', lambda args, _:
          'wire.prompt' if args[1] and args[1][0] in client.SELECT_MESSAGES else 'wire.non_prompt')
    for owner, names, prefix in (
        (client.NetDuelClient, ('_track', '_send_response'), 'client'),
        (board.ShadowBoard, ('apply', 'observe_response', 'cross_check'), 'public_board'),
        (P.SearchPolicy, ('observe_game_message', '_parse', '_commit_sent', '_response_sent'), 'policy'),
        (R.ClientRootController, ('receive', '_drain', 'advance', 'capture_root', 'commit_real_response',
                                  '_capture_host', '_restore_host'), 'owner'),
        (R.RootEnvelope, ('__init__', '__enter__', '__exit__', '_restore', '_check'), 'root'),
        (R._Snapshot, ('__init__', 'verify', 'restore', 'close'), 'snapshot'),
        (R._ReaderState, ('__init__', 'restore'), 'reader'),
        (S.BlankClientSync, ('receive', 'advance'), 'follower'),
        (S.BlankClientSync, ('_save', '_load', '_batch', '_fixed_batch', '_settled', '_align_public_identities',
                            '_recorded_query', '_match', '_equivalent', '_run', '_run_deferring', '_backtrack',
                            '_try', '_respond', '_prepare_choice', '_choice_answers', '_consequence_answers',
                            '_leads_on', '_read_with_identities', '_batch_evidence', '_traced', '_traced_sizes',
                            '_before_batch'), 'follow_detail'),
        (E.DuelDriver, ('save_pystate', 'restore_pystate', '_observe', '_respond'), 'driver'),
        (B.BanishOriginTracker, ('observe',), 'banish'),
        (G.TurnSearchGate, ('begin_prompt', 'decide', 'finish_prompt'), 'gate'),
    ):
        for name in names:
            _wrap(owner, name, 'gate.finish' if prefix == 'gate' and name == 'finish_prompt' else prefix+'.'+name)
    _wrap(P.SearchPolicy, '_call', lambda args, _: 'policy.rpc.'+args[1]['op'])
    for name in ('_digest', '_pool_digest', '_clone', '_copy_card_pool'):
        _wrap(R, name, 'state.'+name)
    _wrap(actions, 'parse_select', 'parse_select')
    for name in vars(disclosure.DisclosureLedger):
        if name.startswith(('observe_', 'disclose', 'forget', 'known_')) or name in ('resolve', 'category_constraints'):
            if callable(getattr(disclosure.DisclosureLedger, name)):
                _wrap(disclosure.DisclosureLedger, name, 'ledger.'+name)


def _stats(values):
    ordered = sorted(values)
    at = .95*(len(ordered)-1)
    low = int(at)
    p95 = ordered[low] + (at-low)*(ordered[min(low+1, len(ordered)-1)]-ordered[low])
    return {'sum': sum(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'p95': p95, 'max': max(values)}


def pytest_sessionfinish(session, exitstatus):
    destination = os.environ.get('MF_PATH_PROFILE_OUT')
    if not destination:
        raise RuntimeError('set an exclusive MF_PATH_PROFILE_OUT for this diagnostic')
    functions = [{'phase': phase, 'context': context, 'calls': len(values),
                  'errors': sum(v[2] for v in values), 'inclusive_seconds': _stats([v[0] for v in values]),
                  'exclusive_seconds': _stats([v[1] for v in values])}
                 for (phase, context), values in sorted(_rows.items())]
    packet = {key: sum(v[0] for v in _rows.get((key, key), [])) for key in ('wire.prompt', 'wire.non_prompt')}
    accounted = sum(row['prompt_seconds'] for row in _ledgers)
    changed = [path for path, sha in _sources.items() if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha]
    output = {'schema': 'mirrorforce_fixed_prefix_cpu_path_profile/v1', 'actual_exit': int(exitstatus),
        'profiler_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'test_seconds': time.perf_counter()-_started, 'wrapped': _wrapped, 'source_sha256': _sources,
        'source_changed': changed, 'functions': functions, 'packet_inclusive_seconds': packet,
        'gate_prompt_ledger_seconds': accounted, 'actual_prompt_ledgers': len(_ledgers),
        'packet_work_outside_gate_ledger_seconds': sum(packet.values())-accounted,
        'current_prompt_preobserver_and_postcallback_seconds': packet['wire.prompt']-accounted,
        'gate_ledger_rows': _ledgers,
        'scope': 'one workstation CPU fixed-prefix run; original policy RPC returns recorded outputs',
        'gpu_network_queue_measured': False, 'production_speedup_claimed': False,
        'inclusive_is_not_additive': True, 'exclusive_subtracts_direct_wrapped_children': True}
    with Path(destination).open('x') as stream:
        json.dump(output, stream, sort_keys=True, indent=1)
        stream.write('\n')
    assert not changed, changed
