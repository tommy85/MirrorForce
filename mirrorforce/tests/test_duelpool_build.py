import importlib.util
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1] / "cxx" / "duelpool"


def load_build():
    spec = importlib.util.spec_from_file_location("duelpool_build", HERE / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dependency_manifest_pins_every_snapshot():
    manifest = json.loads((HERE / "deps.json").read_text())
    assert manifest["schema"] == "mirrorforce_duelpool_build_deps/v1"
    assert all(len(digest) == 64 and int(digest, 16) >= 0 for digest in manifest["deps"].values())
    assert {"python3.11.13-include", "lua-5.3-lib", "pybind11-2.13.6/pybind11/include"} <= set(manifest["deps"])


def test_tree_digest_covers_paths_and_contents(tmp_path):
    build = load_build()
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.h").write_text("1")
    first = build.tree_digest(tmp_path)
    (tmp_path / "a" / "x.h").write_text("2")
    assert build.tree_digest(tmp_path) != first
    (tmp_path / "a" / "x.h").write_text("1")
    (tmp_path / "a" / "x.h").rename(tmp_path / "a" / "y.h")
    assert build.tree_digest(tmp_path) != first


def test_build_refuses_an_existing_output_directory(tmp_path):
    build = load_build()
    with pytest.raises(ValueError, match="new"):
        build.build(tmp_path, None, tmp_path, "HEAD", tmp_path, tmp_path, 1)


def prebuilt_fixture(tmp_path, build, deps=None):
    directory = tmp_path / "prebuilt"
    directory.mkdir()
    files = {build.CORE_LIB: b"fixture core, not a real native library", build.LUA_LIB: b"fixture Lua"}
    for name, body in files.items():
        (directory / name).write_bytes(body)
        (directory / name).chmod(0o444)
    record = {"schema": build.SCHEMA, "core_commit": "b"*40, "core_tree": "c"*40,
              "deps": {} if deps is None else deps, "source_commit": "f"*40,
              "uncommitted_worktree_build": False, "compiler": "fixture", "flags": list(build.COMMON),
              "outputs": {name: hashlib.sha256(body).hexdigest() for name, body in files.items()}}
    (directory / "BUILD.json").write_text(json.dumps(record))
    return directory, record


def checked(build, directory, record, **kwargs):
    args = {"directory": directory, "build_sha256": build.sha256(directory / "BUILD.json"),
            "core_commit": "b"*40, "core_tree": "c"*40, "deps": record["deps"],
            "lua_sha256": record["outputs"][build.LUA_LIB]}
    args.update(kwargs)
    return build.checked_prebuilt_core(**args)


def test_prebuilt_core_consumes_only_exact_committed_manifest_and_regular_files(tmp_path):
    build = load_build()
    directory, record = prebuilt_fixture(tmp_path, build)
    proof, files = checked(build, directory, record)
    assert proof["core_commit"] == "b"*40 and proof["input_build"] == record
    assert files[build.CORE_LIB] == ((directory/build.CORE_LIB).read_bytes(), 0o444)
    assert proof["outputs"] == record["outputs"]


@pytest.mark.parametrize("field,value", [("schema", "unbound"), ("core_commit", "d"*40),
    ("core_tree", "d"*40), ("deps", {"unexpected": "a"*64}), ("source_commit", None),
    ("uncommitted_worktree_build", True), ("compiler", ""), ("flags", None), ("outputs", {})])
def test_prebuilt_metadata_cannot_change_selected_source_or_dependencies(tmp_path, field, value):
    build = load_build()
    directory, record = prebuilt_fixture(tmp_path, build)
    changed = {**record, field: value}
    (directory / "BUILD.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        checked(build, directory, record)


@pytest.mark.parametrize("failure", ["manifest_sha", "missing_manifest", "core_bytes", "lua_bytes", "lua_dependency",
                                     "core_symlink", "manifest_symlink", "directory_symlink", "parent_symlink"])
def test_prebuilt_pin_and_non_symlink_boundary(tmp_path, failure):
    build = load_build()
    directory, record = prebuilt_fixture(tmp_path, build)
    kwargs = {}
    if failure == "manifest_sha":
        kwargs["build_sha256"] = "0"*64
    elif failure == "missing_manifest":
        (directory/"BUILD.json").unlink()
        kwargs["build_sha256"] = "0"*64
    elif failure in ("core_bytes", "lua_bytes"):
        changed = directory / (build.CORE_LIB if failure == "core_bytes" else build.LUA_LIB)
        changed.chmod(0o644)
        changed.write_bytes(b"changed")
    elif failure == "lua_dependency":
        kwargs["lua_sha256"] = "0"*64
    elif failure in ("core_symlink", "manifest_symlink"):
        name = build.CORE_LIB if failure == "core_symlink" else "BUILD.json"
        p = directory/name
        p.rename(directory/(name+".original"))
        p.symlink_to(directory/(name+".original"))
    else:
        link = tmp_path/"link"
        link.symlink_to(directory if failure == "directory_symlink" else tmp_path, target_is_directory=True)
        kwargs["directory"] = link if failure == "directory_symlink" else link/"prebuilt"
    digest = kwargs.pop("build_sha256", build.sha256(directory/"BUILD.json") if (directory/"BUILD.json").exists() else "0"*64)
    with pytest.raises(ValueError):
        build.checked_prebuilt_core(kwargs.pop("directory", directory), digest, "b"*40, "c"*40,
                                    record["deps"], kwargs.pop("lua_sha256", record["outputs"][build.LUA_LIB]))


@pytest.mark.parametrize("which", ["directory", "sha"])
def test_half_prebuilt_arguments_fail_before_output_or_compilation(tmp_path, monkeypatch, which):
    build = load_build()
    out = tmp_path/"new"
    monkeypatch.setattr(build, "archive", lambda *_: pytest.fail("partial prebuilt args reached source archive"))
    kwargs = {"prebuilt_core_dir": tmp_path} if which == "directory" else {"prebuilt_core_build_sha256": "a"*64}
    with pytest.raises(ValueError, match="together"):
        build.build(tmp_path, "HEAD", tmp_path, "HEAD", tmp_path, out, 1, **kwargs)
    assert not out.exists()


@pytest.mark.parametrize("reuse,drift", [(False, False), (True, False), (True, True)])
def test_build_control_flow_compiles_only_module_when_core_is_explicitly_reused(tmp_path, monkeypatch, reuse, drift):
    """Compiler/ELF commands are stubs: this is a dispatch/provenance test, not native acceptance."""
    build = load_build()
    deps = tmp_path/"deps"
    (deps/"lua-5.3-lib").mkdir(parents=True)
    (deps/"lua-5.3-lib"/build.LUA_LIB).write_bytes(b"fixture Lua")
    declared = {"lua-5.3-lib": build.tree_digest(deps/"lua-5.3-lib")}
    directory, record = prebuilt_fixture(tmp_path, build, declared)
    out, commands = tmp_path/"out", []

    def archive(repo, rev, paths, dest):
        if paths:
            src = dest/build.SOURCE_PATH
            src.mkdir(parents=True)
            (src/"deps.json").write_text(json.dumps({"deps": declared}))
        else:
            dest.mkdir(parents=True)
            (dest/"interpreter.h").write_text("#include <lua.h>\n#include <lauxlib.h>\n#include <lualib.h>\n")
            (dest/"fixture.cpp").write_text("fixture")
        return "a"*40 if paths else "b"*40

    def run(command, cwd=None):
        commands.append(command)
        if "-o" in command:
            output = Path(command[command.index("-o")+1])
            output.write_bytes(b"stub compiler output")
            if drift and output.name == "duel_env.o":
                (directory/build.CORE_LIB).chmod(0o644)
                (directory/build.CORE_LIB).write_bytes(b"changed input during compilation")
        result = ""
        if command[0] == "readelf":
            lib = build.LUA_LIB if Path(command[-1]).name == build.CORE_LIB else build.CORE_LIB
            result = f"0 (NEEDED) Shared library: [{lib}]\n"
        elif command[0] == "nm" and "--defined-only" in command:
            result = "core_symbol" if Path(command[-1]).name == build.CORE_LIB else "PyInit_"+build.MODULE
        elif command == ["g++", "--version"]:
            result = "fixture compiler, not native proof\n"
        return SimpleNamespace(stdout=result)

    monkeypatch.setattr(build, "archive", archive)
    monkeypatch.setattr(build, "run", run)
    monkeypatch.setattr(build.subprocess, "check_output", lambda args, **_: "c"*40+"\n")
    options = {"prebuilt_core_dir": directory, "prebuilt_core_build_sha256": build.sha256(directory/"BUILD.json")} if reuse else {}
    if drift:
        with pytest.raises(ValueError, match="prebuilt core input differs"):
            build.build(tmp_path, "a"*40, tmp_path, "b"*40, deps, out, 1, **options)
        assert not (out/"BUILD.json").exists()
        return
    manifest = build.build(tmp_path, "a"*40, tmp_path, "b"*40, deps, out, 1, **options)
    compiles = [cmd for cmd in commands if "-c" in cmd]
    assert any("duel_env.cc" in " ".join(cmd) for cmd in compiles)
    assert any("core_fixture.o" in " ".join(cmd) for cmd in compiles) is (not reuse)
    assert any("core_mfsnap_dlmalloc.o" in " ".join(cmd) for cmd in compiles) is (not reuse)
    assert any("-Wl,-Bsymbolic" in cmd for cmd in commands) is (not reuse)
    assert manifest["source_commit"] == "a"*40 and manifest["core_commit"] == "b"*40
    if reuse:
        assert manifest["compiler_scope"] == "environment_and_shared_module"
        assert manifest["prebuilt_core"]["input_build"] == record
        assert (out/build.CORE_LIB).read_bytes() == (directory/build.CORE_LIB).read_bytes()
        assert manifest["outputs"][build.CORE_LIB] == record["outputs"][build.CORE_LIB]
    else:
        assert "prebuilt_core" not in manifest and "compiler_scope" not in manifest
