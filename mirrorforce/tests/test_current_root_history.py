"""Compile only the small engine-free history/tracker unit, not the native duel/core extension."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("fixture", ["current_root_history.cc", "current_public_seed.cc"])
def test_actual_cpp_current_root_history_and_followup(tmp_path, fixture):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("the engine-free C++ history unit needs g++")
    root = Path(__file__).resolve().parents[1]
    cxx = root / "cxx"
    sources = [cxx / "mfenv" / name for name in
               ("public_tracker.cc", "history.cc", "messages.cc", "semantics.cc", "encoding.cc", "sha256.cc")]
    output = tmp_path / "current-root-history"
    built = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(cxx), "-I", str(cxx / "duelpool"),
                            str(root / "tests" / "fixtures" / fixture),
                            *map(str, sources), "-o", str(output)], capture_output=True, text=True, timeout=90)
    assert built.returncode == 0, built.stdout + built.stderr
    checked = subprocess.run([str(output)], capture_output=True, text=True, timeout=10)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "checks passed" in checked.stdout
