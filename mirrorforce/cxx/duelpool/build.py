"""Build the environment extension from committed revisions.

The environment sources come from ``git archive`` of ``mirrorforce/cxx/duelpool``
at ``--rev`` and the engine core from ``git archive`` of the maintained engine-core
repository (``--core``) at ``--core-rev``, so uncommitted edits never reach a build. The core
is its own shared library, ``libmfcore.so``, linked with ``-Bsymbolic`` and a version script
that keeps its snapshot-arena ``operator new``/``delete`` local: statically linked, that
allocator would serve every allocation in the module, and memory crossing to libstdc++
would be freed by the wrong allocator (glibc "munmap_chunk(): invalid pointer" at import).
The module and the core find ``libmfcore.so`` and ``liblua5.3.so.0`` next to themselves
through ``$ORIGIN``. Every dependency snapshot is checked against
``deps.json`` first. The output directory receives the module, ``libmfcore.so``,
``liblua5.3.so.0`` and ``BUILD.json`` with all input and output digests.

An explicit ``--prebuilt-core-dir`` plus ``--prebuilt-core-build-sha256`` reuses
only a pinned core/Lua pair. Its BUILD.json, files, core commit/tree and dependency
snapshots must agree with the selected Git core and this build's dependencies.
The selected core is still archived for exact headers; no core object or core
library is compiled in this mode. The new record retains the new environment
source identity and records the prebuilt input separately.

    python3 mirrorforce/cxx/duelpool/build.py --rev <sha> --core <repo> --core-rev <sha> \
        --deps <snapshots> --out <new dir>
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
SOURCE_PATH = "mirrorforce/cxx/duelpool"
#: mfenv's public-history sources, compiled into the module as its own objects (never into the core library): the
#: message splitter, the per-viewer interval records, the public tracker and the card-row table they read. They reach
#: no core internals. Archived from the same revision as the environment.
SHARED_PATH = "mirrorforce/cxx/mfenv"
SHARED_SOURCES = ("messages.cc", "history.cc", "public_tracker.cc", "semantics.cc", "encoding.cc", "sha256.cc")
MODULE = "duel_native"
SUFFIX = ".cpython-311-x86_64-linux-gnu.so"
SCHEMA = "mirrorforce_duelpool_build/v1"
CORE_LIB = "libmfcore.so"
LUA_LIB = "liblua5.3.so.0"
COMMON = ["-O2", "-fPIC", "-fvisibility=hidden", "-march=x86-64-v3", "-DNDEBUG"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(path.relative_to(root).as_posix().encode() + b"\0" + sha256(path).encode() + b"\n")
    return h.hexdigest()


def archive(repo: Path, rev: str, paths: list[str], dest: Path) -> str:
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", rev + "^{commit}"], text=True).strip()
    data = subprocess.check_output(["git", "-C", str(repo), "archive", "--format=tar", commit, *paths])
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest)
    return commit


def run(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("command failed: " + " ".join(map(str, command)) + "\n" + result.stderr[:6000])
    return result


def _digest(value, length=64) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _regular_bytes(path: Path) -> tuple[bytes, int]:
    """Read a required regular file without following a replaced final-component symlink."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError(f"prebuilt core input is missing or not a regular non-symlink file: {path}") from error
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"prebuilt core input is not a regular file: {path}")
        return stream.read(), stat.S_IMODE(info.st_mode) & 0o777


def checked_prebuilt_core(directory: Path, build_sha256: str, core_commit: str, core_tree: str,
                          deps: dict, lua_sha256: str) -> tuple[dict, dict[str, tuple[bytes, int]]]:
    """Validate and retain the exact input bytes, before any compilation or copying to the fresh output."""
    if not _digest(build_sha256):
        raise ValueError("prebuilt core requires an explicit lowercase BUILD.json SHA-256")
    directory = directory.absolute()
    if any(path.is_symlink() for path in (directory, *directory.parents)) or not directory.is_dir():
        raise ValueError("prebuilt core directory must exist without symlink components")
    directory = directory.resolve(strict=True)
    raw, _ = _regular_bytes(directory / "BUILD.json")
    if hashlib.sha256(raw).hexdigest() != build_sha256:
        raise ValueError("prebuilt core BUILD.json differs from its pinned SHA-256")
    record = json.loads(raw)
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("prebuilt core BUILD.json has an unsupported schema")
    if record.get("core_commit") != core_commit or record.get("core_tree") != core_tree:
        raise ValueError("prebuilt core commit/tree differs from the explicitly selected Git core")
    if record.get("deps") != deps:
        raise ValueError("prebuilt core dependencies differ from the verified dependency snapshots")
    if record.get("uncommitted_worktree_build") is not False or not _digest(record.get("source_commit"), 40):
        raise ValueError("prebuilt core BUILD.json must reference a committed input build")
    if not isinstance(record.get("compiler"), str) or not record["compiler"] \
            or not isinstance(record.get("flags"), list) or not all(isinstance(x, str) for x in record["flags"]):
        raise ValueError("prebuilt core BUILD.json is missing compiler/flags provenance")
    outputs = record.get("outputs")
    if not isinstance(outputs, dict) or any(not _digest(outputs.get(name)) for name in (CORE_LIB, LUA_LIB)):
        raise ValueError("prebuilt core BUILD.json is missing valid core/Lua output references")
    files = {}
    for name in (CORE_LIB, LUA_LIB):
        data, mode = _regular_bytes(directory / name)
        if hashlib.sha256(data).hexdigest() != outputs[name]:
            raise ValueError(f"prebuilt core input differs from BUILD.json: {name}")
        files[name] = data, mode
    if outputs[LUA_LIB] != lua_sha256:
        raise ValueError("prebuilt Lua differs from the verified Lua dependency snapshot")
    provenance = {"mode": "sha256_pinned_prebuilt_core/v1", "input_directory": str(directory),
                  "build_json_sha256": build_sha256, "core_commit": core_commit, "core_tree": core_tree,
                  "outputs": {name: outputs[name] for name in (CORE_LIB, LUA_LIB)}, "input_build": record}
    return provenance, files


def build(repo: Path, rev: str | None, core: Path, core_rev: str, deps_root: Path, out: Path, jobs: int, *,
          prebuilt_core_dir: Path | None = None, prebuilt_core_build_sha256: str | None = None) -> dict:
    if (prebuilt_core_dir is None) != (prebuilt_core_build_sha256 is None):
        raise ValueError("prebuilt core directory and BUILD.json SHA-256 must be supplied together")
    if out.exists():
        raise ValueError("the output directory must be new")
    work = out / ".work"
    work.mkdir(parents=True)
    if rev is None:  # development only: the manifest records that no commit was built
        shutil.copytree(repo / SOURCE_PATH, work / "src" / SOURCE_PATH)
        shutil.copytree(repo / SHARED_PATH, work / "src" / SHARED_PATH)
        commit = None
    else:
        commit = archive(repo, rev, [SOURCE_PATH, SHARED_PATH], work / "src")
    src = work / "src" / SOURCE_PATH
    shared = work / "src" / SHARED_PATH
    declared = json.loads((src / "deps.json").read_text())["deps"]
    for name, digest in declared.items():
        if tree_digest(deps_root / name) != digest:
            raise ValueError("dependency snapshot differs from deps.json: " + name)
    core_commit = archive(core, core_rev, [], work / "core")
    core_tree = subprocess.check_output(["git", "-C", str(core), "rev-parse", core_commit + "^{tree}"], text=True).strip()
    prebuilt, prebuilt_files = None, {}
    if prebuilt_core_dir is not None:
        prebuilt, prebuilt_files = checked_prebuilt_core(prebuilt_core_dir, prebuilt_core_build_sha256,
            core_commit, core_tree, declared, sha256(deps_root / "lua-5.3-lib" / LUA_LIB))
    interpreter = work / "core" / "interpreter.h"
    text = interpreter.read_text()
    anchor = "#include <lua.h>\n#include <lauxlib.h>\n#include <lualib.h>\n"
    if anchor not in text:
        raise ValueError("core interpreter.h lacks the Lua include anchor")
    interpreter.write_text(text.replace(anchor, 'extern "C" {\n' + anchor + "}\n", 1))
    include = work / "inc" / "ocgcore"
    include.mkdir(parents=True)
    for header in (work / "core").glob("*.h"):
        shutil.copy2(header, include / header.name)
    d = deps_root
    lua_inc = d / "lua-5.3-include"
    objects = work / "obj"
    objects.mkdir()
    tasks = []
    if prebuilt is None:
        for cpp in sorted((work / "core").glob("*.cpp")):
            tasks.append(["g++", "-std=c++17", *COMMON, "-DOCGCORE_EXPORT_FUNCTIONS", "-I", str(work / "core"), "-I", str(lua_inc),
                          "-c", str(cpp), "-o", str(objects / ("core_" + cpp.stem + ".o"))])
        tasks.append(["gcc", *COMMON, "-DONLY_MSPACES=1", "-DHAVE_MMAP=0", "-DHAVE_MORECORE=0", "-DUSE_LOCKS=0",
                      "-DNO_MALLOC_STATS=1", "-c", str(work / "core" / "mfsnap_dlmalloc.c"),
                      "-o", str(objects / "core_mfsnap_dlmalloc.o")])
    scpp = d / "SQLiteCpp-3.3.3"
    for cpp in sorted((scpp / "src").glob("*.cpp")):
        tasks.append(["g++", "-std=c++17", *COMMON, "-I", str(scpp / "include"), "-I",
                      str(d / "sqlite-amalgamation-3.45.1"), "-c", str(cpp), "-o", str(objects / ("scpp_" + cpp.stem + ".o"))])
    tasks.append(["gcc", *COMMON, "-DSQLITE_ENABLE_COLUMN_METADATA", "-c",
                  str(d / "sqlite-amalgamation-3.45.1" / "sqlite3.c"), "-o", str(objects / "sqlite3.o")])
    module_flags = ["-std=c++17", *COMMON, "-DFMT_HEADER_ONLY=1", "-I", str(src / "compat"), "-I", str(src),
                    "-I", str(work / "src" / "mirrorforce" / "cxx"),
                    "-I", str(work / "inc"),
                    "-I", str(d / "fmt-10.2.1-include"), "-I", str(d / "unordered_dense-include"),
                    "-I", str(scpp / "include"), "-I", str(d / "sqlite-amalgamation-3.45.1"),
                    "-I", str(d / "concurrentqueue-include"), "-I", str(lua_inc),
                    "-I", str(d / "pybind11-2.13.6/pybind11/include"), "-I", str(d / "python3.11.13-include")]
    tasks.append(["g++", *module_flags, "-c", str(src / "duel" / "duel_env.cc"), "-o", str(objects / "duel_env.o")])
    for name in SHARED_SOURCES:
        tasks.append(["g++", "-std=c++17", *COMMON, "-I", str(shared), "-c", str(shared / name),
                      "-o", str(objects / ("mfenv_" + Path(name).stem + ".o"))])
    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(run, tasks))
    module = out / (MODULE + SUFFIX)
    core_lib = out / CORE_LIB
    if prebuilt is None:
        shutil.copy2(d / "lua-5.3-lib" / "liblua5.3.so.0", out / "liblua5.3.so.0")
        (work / "core.ver").write_text("{ local: _Znw*; _Zna*; _Zdl*; _Zda*; };\n")
        run(["g++", "-shared", *COMMON, "-Wl,-Bsymbolic", "-Wl,--version-script=" + str(work / "core.ver"),
             "-o", str(core_lib), *sorted(map(str, objects.glob("core_*.o"))), "-L", str(out), "-l:liblua5.3.so.0",
             "-Wl,-rpath,$ORIGIN", "-lpthread", "-ldl", "-lm"])
    else:
        # Copy the verified bytes retained above, not a path that could now name different bytes or a symlink.
        for name, (data, mode) in prebuilt_files.items():
            with (out / name).open("xb") as stream:
                stream.write(data)
            (out / name).chmod(mode)
            if sha256(out / name) != prebuilt["outputs"][name]:
                raise ValueError(f"copied prebuilt core output differs: {name}")
    run(["g++", "-shared", *COMMON, "-o", str(module),
         *sorted(str(path) for path in objects.glob("*.o") if not path.name.startswith("core_")),
         "-L", str(out), "-l:" + CORE_LIB, "-Wl,-rpath,$ORIGIN", "-lpthread", "-ldl", "-lm"])
    system = {"libstdc++.so.6", "libm.so.6", "libgcc_s.so.1", "libc.so.6", "ld-linux-x86-64.so.2"}
    for library, own in ((core_lib, {"liblua5.3.so.0"}), (module, {CORE_LIB})):
        needed = [line.split("[")[1].rstrip("]") for line in run(["readelf", "-d", str(library)]).stdout.splitlines()
                  if "(NEEDED)" in line]
        if set(needed) - system - own:
            raise ValueError(f"{library.name}: unexpected shared dependencies: "
                             + ", ".join(sorted(set(needed) - system - own)))
    core_exports = set(run(["nm", "-D", "--defined-only", str(core_lib)]).stdout.split())
    if any(name.startswith(("_Znw", "_Zna", "_Zdl", "_Zda")) for name in core_exports):
        raise ValueError("the core exports its interposed allocator")
    # Python symbols resolve from the interpreter at import; anything else must
    # come from a declared library (versioned libc/libstdc++ symbols carry "@").
    undefined = [line.split()[-1] for line in run(["nm", "-D", "--undefined-only", str(module)]).stdout.splitlines()]
    stray = [name for name in undefined if "@" not in name and not name.startswith(("Py", "_Py"))
             and name not in core_exports
             and name not in ("__gmon_start__", "_ITM_deregisterTMCloneTable", "_ITM_registerTMCloneTable")]
    if stray:
        raise ValueError("unresolved symbols outside Python and Lua: " + ", ".join(sorted(stray)[:20]))
    symbols = run(["nm", "-D", "--defined-only", str(module)]).stdout.split()
    if "PyInit_" + MODULE not in symbols:
        raise ValueError("module init symbol missing")
    compiler = run(["g++", "--version"]).stdout.splitlines()[0]
    manifest = {"schema": SCHEMA, "source_commit": commit,
        "source_tree": None if commit is None else subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", commit + ":" + SOURCE_PATH], text=True).strip(),
        "shared_tree": None if commit is None else subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", commit + ":" + SHARED_PATH], text=True).strip(),
        "shared_sources": list(SHARED_SOURCES),
        "uncommitted_worktree_build": commit is None,
        "core_commit": core_commit,
        "core_tree": core_tree,
        "deps": declared, "compiler": compiler, "flags": COMMON,
        "outputs": {module.name: sha256(module), CORE_LIB: sha256(core_lib),
                    "liblua5.3.so.0": sha256(out / "liblua5.3.so.0")}}
    if prebuilt is not None:
        after, after_files = checked_prebuilt_core(prebuilt_core_dir, prebuilt_core_build_sha256,
            core_commit, core_tree, declared, sha256(deps_root / "lua-5.3-lib" / LUA_LIB))
        if after != prebuilt or after_files != prebuilt_files:
            raise ValueError("prebuilt core inputs changed while compiling the environment module")
        manifest["prebuilt_core"] = prebuilt
        manifest["compiler_scope"] = "environment_and_shared_module"
    shutil.rmtree(work)
    (out / "BUILD.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=HERE.parents[2])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rev", help="commit to build (deployments use only commit builds)")
    source.add_argument("--worktree", action="store_true", help="development build of uncommitted sources")
    parser.add_argument("--core", type=Path, required=True, help="the maintained engine-core Git repository")
    parser.add_argument("--core-rev", required=True)
    parser.add_argument("--deps", type=Path, required=True, help="directory holding the deps.json snapshots")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--prebuilt-core-dir", type=Path, help="reuse only this verified core/Lua pair in the new output")
    parser.add_argument("--prebuilt-core-build-sha256", help="required SHA-256 of the prebuilt directory's BUILD.json")
    args = parser.parse_args(argv)
    if (args.prebuilt_core_dir is None) != (args.prebuilt_core_build_sha256 is None):
        parser.error("--prebuilt-core-dir and --prebuilt-core-build-sha256 must be supplied together")
    manifest = build(args.repo.resolve(), None if args.worktree else args.rev, args.core.resolve(), args.core_rev, args.deps.resolve(),
                     args.out.absolute(), args.jobs, prebuilt_core_dir=args.prebuilt_core_dir,
                     prebuilt_core_build_sha256=args.prebuilt_core_build_sha256)
    print(json.dumps({"outputs": manifest["outputs"], "source_commit": manifest["source_commit"],
                      "core_commit": manifest["core_commit"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
