"""The checkpoint store's command line: record the arrival of a checkpoint copied to this host."""
import json
import socket

import pytest

from mirrorforce.agent.train import checkpoint_store as C


def test_command_line_arrival_of_a_copied_checkpoint(tmp_path, capsys):
    sha = C.write_checkpoint(tmp_path / "written", b"payload", {"config": {"x": 1}}, host="writer")
    copied = tmp_path / "copied"
    copied.mkdir()
    for suffix in (".ckpt", ".receipt.json"):
        (copied / (sha + suffix)).write_bytes((tmp_path / "written" / (sha + suffix)).read_bytes())
    with pytest.raises(ValueError, match="no arrival receipt"):
        C.verify_resume(copied, sha, expected={}, host="reader")
    assert C.main(["arrive", "--dir", str(copied), "--sha", sha, "--source", "release-bundle"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["host"] == socket.gethostname() and record["source_host"] == "release-bundle"
    assert C.verify_resume(copied, sha, expected={"config.x": 1})["payload_sha256"] == sha
    (copied / (sha + ".ckpt")).write_bytes(b"changed")
    with pytest.raises(ValueError):
        C.main(["arrive", "--dir", str(copied), "--sha", sha])
