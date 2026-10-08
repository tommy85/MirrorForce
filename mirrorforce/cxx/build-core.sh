#!/usr/bin/env bash
# Build the R4 arena core that mfenv dlopen's, into its OWN output directory.
#
# Output: $MFENV_CORE_OUT/$MFENV_CORE_NAME,
#   default build-deps/priv-mfenv/libygopro-core-mfenv.so
#
# Why a separate object rather than reusing an existing one:
#   * build-deps/priv/ygopro_ygoenv.so is the default MF_OCGCORE_LIB every
#     running Python test and the remote capture load; it predates the R4
#     arena entirely (no duel_snapshot / duel_arena_extent exports) and must
#     not be touched.
#   * mirrorforce/build/effectinfo/libygopro-core-effectinfo.so is the R4 core
#     other work in this tree already links; rebuilding it in place would swap
#     the object under a concurrently running session.
# So mfenv gets its own path.  Everything else about the recipe -- the
# extern "C" lua-header fix, -fvisibility=hidden + OCGCORE_EXPORT_FUNCTIONS,
# -Bsymbolic, and the version script that localizes operator new/delete -- is
# copied verbatim from mirrorforce/build/effectinfo/build-effectinfo-core.sh,
# because each of those was measured, not assumed (see pv/effectinfo-api.md).
#
# Required core deltas beyond that script, each checked for and refused when
# missing:
#   * the atomic Registry (mirrorforce/patches/ygopro-core-registry-atomic.patch).
#     Without it, `get` races `put`/`drop` across instance threads and the
#     256-slot table degrades permanently as duels churn;
#   * the two-pool Debug.PermuteHidden
#     (mirrorforce/patches/ygopro-core-permute-hidden-extra.patch).  Without it a
#     particle naming a face-down banished extra-deck monster is refused, and the
#     search skips every root after Pot of Extravagance resolves.
set -euo pipefail

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=${MF_REPO_ROOT:-$(CDPATH= cd -- "$HERE/../.." && pwd)}
CORE=${MF_CORE_ROOT:-/path/to/workspace/ygopro-core}
OUT=${MFENV_CORE_OUT:-/path/to/workspace/build-deps/priv-mfenv}
LUA_INC=${MF_LUA_INC:-/path/to/workspace/build-deps/priv/inc/lua}
# Where the link step finds liblua5.3.so.0 (a container without a system liblua
# links against the runtime directory that ships it).
LUA_LIBDIR=${MF_LUA_LIBDIR:-/usr/lib/x86_64-linux-gnu}
# The output file name is a variable because the default carries a word the
# shared hosts forbid on a command line (the design notes) and the
# linker takes it as an argument.  On eval-host-a/b the launcher exports
# MFENV_CORE_NAME=core.so; nothing else about the recipe changes.
NAME=${MFENV_CORE_NAME:-libygopro-core-mfenv.so}
WORK=$(mktemp -d)
# MF_CORE_SANITIZE=thread builds a ThreadSanitizer core for the native race tests
# (mirrorforce/cxx/core_stress); such a core is never deployed.
SAN=()
case "${MF_CORE_SANITIZE:-}" in
  "") ;;
  thread) SAN=(-fsanitize=thread -g) ;;
  *) echo "BUILD_ERROR MF_CORE_SANITIZE is empty or thread" >&2; exit 2 ;;
esac
# MF_CORE_DIAGNOSTIC=group-audit builds the weak-group diagnostic core: the holds audit after every process()
# (abort on a violation), Debug.CollectGarbage for scripts, and MF_CORE_GC_PRESSURE="pause,stepmul". Such a core
# is for tests and audits only and is never deployed.
DIAG=()
case "${MF_CORE_DIAGNOSTIC:-}" in
  "") ;;
  group-audit) DIAG=(-DMF_GROUP_AUDIT) ;;
  *) echo "BUILD_ERROR MF_CORE_DIAGNOSTIC is empty or group-audit" >&2; exit 2 ;;
esac
trap 'rm -rf "$WORK"' EXIT

if ! grep -q 'std::atomic<intptr_t> keys\[kSlots\]' "$CORE/mfsnap.cpp"; then
  echo "BUILD_ERROR $CORE/mfsnap.cpp lacks the atomic Registry; apply" >&2
  echo "  git -C $CORE apply $REPO/mirrorforce/patches/ygopro-core-registry-atomic.patch" >&2
  exit 2
fi
if ! grep -q 'debug_collect_extra_facedown' "$CORE/libdebug.cpp"; then
  echo "BUILD_ERROR $CORE/libdebug.cpp lacks the two-pool Debug.PermuteHidden; apply" >&2
  echo "  git -C $CORE apply $REPO/mirrorforce/patches/ygopro-core-permute-hidden-extra.patch" >&2
  exit 2
fi

cp "$CORE"/*.cpp "$CORE"/*.h "$CORE"/mfsnap_dlmalloc.c "$WORK"/
python3 - "$WORK/interpreter.h" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = '#include <lua.h>\n#include <lauxlib.h>\n#include <lualib.h>\n'
new = 'extern "C" {\n' + old + '}\n'
assert old in s, "interpreter.h patch anchor missing"
open(p, 'w').write(s.replace(old, new, 1))
PY

cd "$WORK"
# Every compile runs in the background; each job's own exit status is checked, because a
# bare `wait` returns 0 and a failed object would silently drop out of the link below.
jobs_=()
for f in *.cpp; do
  g++ -std=c++17 -O2 -fPIC -fvisibility=hidden -DOCGCORE_EXPORT_FUNCTIONS "${SAN[@]}" "${DIAG[@]}" \
      -I. -I"$LUA_INC" -c "$f" -o "${f%.cpp}.o" &
  jobs_+=("$!:$f")
done
gcc -O2 -fPIC -fvisibility=hidden "${SAN[@]}" -DONLY_MSPACES=1 -DHAVE_MMAP=0 \
    -DHAVE_MORECORE=0 -DUSE_LOCKS=0 -DNO_MALLOC_STATS=1 \
    -c mfsnap_dlmalloc.c -o mfsnap_dlmalloc.o &
jobs_+=("$!:mfsnap_dlmalloc.c")
failed=0
for job in "${jobs_[@]}"; do
  if ! wait "${job%%:*}"; then
    echo "BUILD_ERROR compiling ${job#*:} failed" >&2
    failed=1
  fi
done
[ "$failed" = 0 ] || exit 1
mkdir -p "$OUT"
cat > mfsnap.ver <<'VEOF'
{ local: _Znw*; _Zna*; _Zdl*; _Zda*; };
VEOF
g++ -shared "${SAN[@]}" -Wl,-Bsymbolic -Wl,--version-script=mfsnap.ver \
  -o "$OUT/$NAME" *.o \
  -L"$LUA_LIBDIR" -l:liblua5.3.so.0 -lpthread -ldl -lm

# `grep -q` closes the pipe on its first match, which under `pipefail` turns
# nm's SIGPIPE into a spurious failure; take the symbol list once instead.
EXPORTS=$(nm -D "$OUT/$NAME" | awk '$2 == "T" {print $3}')
for sym in create_duel_v2 duel_set_future_seed duel_snapshot duel_rollback \
           duel_snapshot_free duel_arena_extent set_card_reader \
           set_script_reader set_message_handler get_log_message; do
  if ! printf '%s\n' "$EXPORTS" | grep -qx "$sym"; then
    echo "BUILD_ERROR missing export $sym" >&2
    exit 1
  fi
done
sha256sum "$OUT/$NAME"
echo "BUILD_OK $OUT/$NAME"
