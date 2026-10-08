"""checkpoint files: a payload named by its SHA-256, a canonical receipt and per-host arrival receipts.

One implementation shared by the trainer (writing, local arrival, resume checks) and the replication tool
(``tools/mf_runtime_checkpoint_replicate.py``); design: the design notes. Standard library
only, so the JAX runtime imports it without torch.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
from zoneinfo import ZoneInfo

ARRIVAL_SCHEMA = "mirrorforce_checkpoint_arrival/v1"
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def paths(directory, sha, host=None):
    directory = Path(directory)
    names = {"payload": directory / f"{sha}.ckpt", "receipt": directory / f"{sha}.receipt.json"}
    if host is not None:
        names["arrival"] = directory / f"{sha}.arrival-{host}.json"
    return names


def check_checkpoint(directory, sha):
    """The receipt of checkpoint ``sha`` after recomputing the payload's digest and size; raises on any mismatch."""
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("a checkpoint is named by a lowercase SHA-256")
    files = paths(directory, sha)
    raw = files["receipt"].read_bytes()
    receipt = json.loads(raw)
    if raw != canonical(receipt):
        raise ValueError("the checkpoint receipt is not canonical JSON")
    if receipt.get("payload_sha256") != sha or file_sha256(files["payload"]) != sha:
        raise ValueError("the checkpoint payload differs from its name or its receipt")
    if receipt.get("payload_bytes") != files["payload"].stat().st_size:
        raise ValueError("the checkpoint payload size differs from its receipt")
    return receipt, hashlib.sha256(raw).hexdigest()


def arrive(directory, sha, *, source_host, route, host=None):
    """Check checkpoint ``sha`` where it now lies and write this host's arrival receipt (refusing a mismatch)."""
    host = host or socket.gethostname()
    receipt, receipt_sha = check_checkpoint(directory, sha)
    record = {"schema": ARRIVAL_SCHEMA, "payload_sha256": sha, "payload_bytes": receipt["payload_bytes"],
              "receipt_sha256": receipt_sha, "host": host, "source_host": source_host, "route": route,
              "arrived": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S CST")}
    target = paths(directory, sha, host)["arrival"]
    temporary = target.with_suffix(".json.part")
    temporary.write_bytes(canonical(record))
    os.replace(temporary, target)
    return record


def verify_resume(directory, sha, *, expected, host=None):
    """The receipt of a checkpoint this host may resume from, or a refusal.

    ``expected`` maps receipt keys (dotted for nested ones) to the values the run's configuration requires; the
    payload must match its name and receipt, and this host's arrival receipt must name the same payload and
    receipt digests."""
    host = host or socket.gethostname()
    receipt, receipt_sha = check_checkpoint(directory, sha)
    arrival_path = paths(directory, sha, host)["arrival"]
    if not arrival_path.exists():
        raise ValueError("no arrival receipt on this host: the checkpoint was not replicated and checked here")
    arrival = json.loads(arrival_path.read_bytes())
    if arrival.get("schema") != ARRIVAL_SCHEMA or arrival.get("payload_sha256") != sha \
            or arrival.get("receipt_sha256") != receipt_sha or arrival.get("host") != host:
        raise ValueError("the arrival receipt names another payload, receipt or host")
    for key, value in expected.items():
        node = receipt
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise ValueError(f"the checkpoint receipt lacks {key}")
            node = node[part]
        if node != value:
            raise ValueError(f"the checkpoint receipt's {key} differs from the run configuration")
    return receipt


def write_checkpoint(directory, payload, receipt_fields, *, host=None):
    """Write ``payload`` as ``<sha256>.ckpt`` with its canonical receipt, then this host's arrival receipt.

    ``receipt_fields`` must not name the payload digest or size; they are added here. Files are written to a
    temporary name and renamed, so a reader never sees a partial checkpoint. Returns the payload SHA-256."""
    if {"payload_sha256", "payload_bytes"} & set(receipt_fields):
        raise ValueError("the payload digest and size are computed here, not supplied")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(payload).hexdigest()
    files = paths(directory, sha)
    receipt = {**receipt_fields, "payload_sha256": sha, "payload_bytes": len(payload)}
    for target, data in ((files["payload"], payload), (files["receipt"], canonical(receipt))):
        temporary = target.with_name(target.name + ".part")
        temporary.write_bytes(data)
        os.replace(temporary, target)
    arrive(directory, sha, source_host=host or socket.gethostname(), route="local-write", host=host)
    return sha


def prune(directory, keep_last: int, keep_every: int):
    """Delete checkpoint payloads outside the retention: the ``keep_last`` most recent updates and every update that
    is a multiple of ``keep_every``. Receipts and arrival receipts stay (they are small and record what existed);
    a pruned payload is listed in ``pruned.jsonl``. Returns the deleted payload digests."""
    directory = Path(directory)
    entries = []
    for receipt in directory.glob("*.receipt.json"):
        record = json.loads(receipt.read_text())
        sha = receipt.name[: -len(".receipt.json")]
        payload = directory / f"{sha}.ckpt"
        if payload.exists():
            entries.append((int(record["counters"]["learner_update"]), sha, payload))
    entries.sort()
    recent = {sha for _, sha, _ in entries[-keep_last:]} if keep_last > 0 else set()
    deleted = []
    for update, sha, payload in entries:
        if sha in recent or (keep_every > 0 and update % keep_every == 0):
            continue
        payload.unlink()
        deleted.append(sha)
        with open(directory / "pruned.jsonl", "a") as log:
            log.write(json.dumps({"sha256": sha, "learner_update": update}) + "\n")
    return deleted


def main(argv=None):
    """``arrive --dir D --sha S``: check a checkpoint copied to this host by any means and record its arrival."""
    import argparse
    parser = argparse.ArgumentParser(description="Check a checkpoint copied to this host and write its arrival receipt.")
    commands = parser.add_subparsers(dest="command", required=True)
    here = commands.add_parser("arrive")
    here.add_argument("--dir", required=True, help="the directory holding <sha256>.ckpt and its receipt")
    here.add_argument("--sha", required=True, help="the checkpoint's SHA-256")
    here.add_argument("--source", default="copied", help="where the files came from, recorded in the receipt")
    args = parser.parse_args(argv)
    print(json.dumps(arrive(args.dir, args.sha, source_host=args.source, route="manual-copy"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
