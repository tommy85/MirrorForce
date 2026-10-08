"""Real CPU subprocess fixture for the coordinator protocol, NOT a native/model/data-validity proof."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def checksum(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def cpus(value):
    result = set()
    for part in value.split(","):
        values = list(map(int, part.split("-")))
        result.update(range(values[0], values[-1] + 1))
    return result


def cpu_text(values):
    ranges = []
    for value in sorted(values):
        if ranges and value == ranges[-1][-1] + 1:
            ranges[-1].append(value)
        else:
            ranges.append([value])
    return ",".join(str(row[0]) if len(row) == 1 else f"{row[0]}-{row[-1]}" for row in ranges)


def publish(path, value):
    body = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {"path": str(path), "sha256": hashlib.sha256(body).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    job = json.loads(Path(parser.parse_args().job).read_text())
    os.environ.update(job["env"])
    argv = job["argv"]
    config_path = Path(argv[argv.index("--config") + 1])
    config = json.loads(config_path.read_text())
    worker, workers = int(argv[argv.index("--worker") + 1]), int(argv[argv.index("--workers") + 1])
    upgrade, upgrade_ref = None, None
    if "--source-upgrade" in argv:
        upgrade_ref = {"path": argv[argv.index("--source-upgrade") + 1],
                       "sha256": argv[argv.index("--source-upgrade-sha256") + 1]}
        raw = Path(upgrade_ref["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == upgrade_ref["sha256"]
        upgrade = json.loads(raw)
        assert upgrade["producer_config"]["path"] == str(config_path)
        assert upgrade["producer_config"]["sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
        assert upgrade["seconds"] == int(argv[argv.index("--verify-seconds") + 1])
    root = Path(config["out"] if upgrade is None else upgrade["out"]) / "verification/workers"
    allowed = cpus(job["env"]["MF_RUNTIME_EXPECTED_CPUS"])
    assert set(os.sched_getaffinity(0)) == allowed
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and os.environ["NVIDIA_VISIBLE_DEVICES"] == "void"
    physical = sorted({tuple(sorted(cpus(Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip())))
                       for cpu in allowed})
    assert {cpu for pair in physical for cpu in pair} == allowed
    metadata = {}
    node = config["cpu_node"]
    if upgrade is not None and upgrade['schema'].endswith('/v2'):
        placed = upgrade['verifier_placement']
        assert placed['hostname'] == socket.gethostname()
        assert upgrade['producer_config_semantic_sha256'] == checksum(config)
        assert placed['host_memory_mib'] == config['host_memory_mib']
        node = placed['cpu_node']
        all_allowed = cpus(placed['cpu_cpus'])
        cores = sorted({tuple(sorted(cpus(Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip())))
                        for cpu in all_allowed})
        assert {cpu for core in cores for cpu in core} == all_allowed
        effective = [{'worker': i, 'workers': workers, 'node': node,
                      'cpus': cpu_text({cpu for core in cores[i::workers] for cpu in core}),
                      'physical_cores': [list(core) for core in cores[i::workers]]} for i in range(workers)]
        assert allowed == cpus(effective[worker]['cpus'])
        for cpu in allowed:
            assert (Path(f'/sys/devices/system/cpu/cpu{cpu}') / f'node{node}').exists()
        for key in ('TMPDIR', 'XDG_CACHE_HOME'):
            assert Path(job['env'][key]).is_relative_to(Path(upgrade['out']))
        sys.path.insert(0, str(Path(__file__).parent / 'tools'))
        import affinity_plan
        actual_audit = affinity_plan.audit(affinity_plan.discover(), os.getpid(),
                                          expected_node=node, expected_cpus=allowed)
        assert actual_audit['passed'], actual_audit
        metadata = {'verification_placement': {'schema': 'mirrorforce_belief_verifier_placement/v2',
            'producer': {key: config[key] for key in ('cpu_node', 'cpu_cpus', 'host_memory_mib')},
            'verifier': placed, 'effective': effective,
            'memory_accounting': 'per-process RSS only; aggregate cgroup limit/peak requires independent launch admission',
            'aggregate_memory_verified': False},
            'resources': {'affinity': actual_audit, 'protocol_only_not_aggregate_RAM_or_native_admission': True}}
        pointer = Path(argv[argv.index('--launch-request-pointer') + 1])
        request_ref = json.loads(pointer.read_text())
        request_raw = Path(request_ref['path']).read_bytes()
        assert hashlib.sha256(request_raw).hexdigest() == request_ref['sha256']
        request = json.loads(request_raw)
        gate_ref = {'path': argv[argv.index('--launch-gate') + 1],
                    'sha256': argv[argv.index('--launch-gate-sha256') + 1]}
        gate_raw = Path(gate_ref['path']).read_bytes()
        assert hashlib.sha256(gate_raw).hexdigest() == gate_ref['sha256']
        gate = json.loads(gate_raw)
        assert request['placement_upgrade'] == upgrade_ref and request['nonce'] == argv[argv.index('--launch-nonce') + 1]
        assert gate['request'] == request_ref and gate['nonce'] == request['nonce']
        stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
        membership = [line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                      if line.startswith('0::')][0]
        metadata.update(launch_binding={'request': request_ref, 'gate': gate_ref},
                        process_identity={'pid': os.getpid(), 'start_ticks': int(stat[19]),
                                          'uid': list(map(int, next(line.split(':', 1)[1] for line in
                                              Path('/proc/self/status').read_text().splitlines() if line.startswith('Uid:')).split())),
                                          'cgroup': membership})
    publish(root / f"actual-process-{worker:03d}.json", {"pid": os.getpid(), "cpus": sorted(allowed), "worker": worker})
    time.sleep(config.get("_protocol_fixture_sleep", 0))
    indices = list(range(worker, config["games"], workers))
    proofs = [{"index": index, **publish(root / f"real-game-protocol-{index:03d}.json",
                                        {"index": index, "pid": os.getpid(), "cpus": sorted(allowed), "protocol_only": True})}
              for index in indices]
    part = {"schema": "mirrorforce_belief_new_collection/v1#worker-verified", "complete": True,
            "placement": {"worker": worker, "workers": workers, "node": node,
                          "cpus": job["env"]["MF_RUNTIME_EXPECTED_CPUS"], "physical_cores": [list(pair) for pair in physical]},
            "operator": {"source_commit": config["source_commit"] if upgrade is None else upgrade["verifier_source_commit"]},
            "config_sha256": checksum(config),
            "collection": ({"path": "/protocol-fixture/collection", "sha256": argv[argv.index("--collection-sha256") + 1]}
                           if upgrade is None else upgrade["collection"]),
            "training_admitted": False, "proofs": proofs,
            "decisions": {"train": 0, "heldout": 0, "smoke": 2 * len(indices)}, "assets": {"protocol_only": True},
            "producer_operator": {"source_commit": config["source_commit"]}, "verification_upgrade": upgrade_ref,
            **metadata}
    corrupt = config.get('_protocol_fixture_corrupt')
    if corrupt and worker == 1:
        if corrupt == 'placement':
            part['placement']['cpus'] = config['cpu_cpus']
        elif corrupt == 'audit':
            part['resources']['affinity']['allowed_cpus'] = config['cpu_cpus']
        elif corrupt == 'memory_policy':
            part['resources']['affinity']['bind_local_validation']['passed'] = False
        elif corrupt == 'metadata':
            part['verification_placement']['verifier']['cpu_node'] += 1
        elif corrupt == 'launch_gate':
            part['launch_binding']['gate']['sha256'] = 'f' * 64
        elif corrupt == 'process_identity':
            part['process_identity']['start_ticks'] += 1
    result = publish(root / f"real-process-result-{worker:03d}.json", part)
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
