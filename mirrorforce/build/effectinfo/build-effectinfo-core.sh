#!/usr/bin/env bash
# Build the maintained MirrorForce ygopro-core as a standalone ocgapi object.
#
# Deliberately NOT the ygoenv extension: mirrorforce.puzzle.core loads the core
# through plain ctypes and needs only the extern "C" ocgapi symbols, so a bare
# core .so drops the pybind11 / sqlite / glog dependency chain entirely and --
# more importantly -- lands on its own path, leaving every existing .so alone.
#
# Output: mirrorforce/build/effectinfo/libygopro-core-effectinfo.so
#
# Required core delta: nested ygopro-core commit
# 1c2069669178fce3a8804a661b6d3b87e28b054d.  On a checkout that does not
# carry that commit, apply the portable top-level copy first:
#   git -C "$MF_CORE_ROOT" apply "$MF_REPO_ROOT/mirrorforce/patches/ygopro-core-future-rng-particle.patch"
set -e
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=${MF_REPO_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)}
CORE=${MF_CORE_ROOT:-$REPO_ROOT/ygopro-core}
BUILD_ROOT=${MF_BUILD_ROOT:-$REPO_ROOT/mirrorforce/build}
OUT=$BUILD_ROOT/effectinfo
LUA_INC=${MF_LUA_INC:-$REPO_ROOT/build-deps/priv/inc/lua} # Lua 5.3 headers
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# The only build-time compatibility rewrite is for the C Lua library: its headers
# are included without extern "C", which mismatches liblua5.3.so.0
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
# -fvisibility=hidden + OCGCORE_EXPORT_FUNCTIONS exports exactly the ocgapi
# surface and hides everything else; -Bsymbolic binds the rest internally.
# Without both, loading this next to another core built from the same sources
# lets the earlier RTLD_GLOBAL object interpose read_script / read_card /
# handle_message -- the duel then runs on somebody else's card reader and fails
# silently rather than loudly.  Measured, not assumed: see pv/effectinfo-api.md.
for f in *.cpp; do
  g++ -std=c++17 -O2 -fPIC -fvisibility=hidden -DOCGCORE_EXPORT_FUNCTIONS \
      -I. -I"$LUA_INC" -c "$f" -o "${f%.cpp}.o" &
done
# the per-duel heap (dlmalloc mspaces, C): the region handed over at creation
# is the whole heap -- no mmap, no morecore, no locks -- so exhaustion fails
# the allocation instead of escaping the snapshot's reach
gcc -O2 -fPIC -fvisibility=hidden -DONLY_MSPACES=1 -DHAVE_MMAP=0 \
    -DHAVE_MORECORE=0 -DUSE_LOCKS=0 -DNO_MALLOC_STATS=1 \
    -c mfsnap_dlmalloc.c -o mfsnap_dlmalloc.o &
wait
mkdir -p "$OUT"
# R4: the interposed operator new/delete must never leave this object.  GCC
# force-exports replaceable allocation functions even under hidden
# visibility, and an exported delete gets bound by whatever the host loads
# later (torch's lazily-dlopened BLAS backend, first matmul) -- which then
# feeds us pointers our header rewind was never behind.  A version script
# localizing the mangled allocator names is the reliable lever.
cat > mfsnap.ver <<'VEOF'
{ local: _Znw*; _Zna*; _Zdl*; _Zda*; };
VEOF
g++ -shared -Wl,-Bsymbolic -Wl,--version-script=mfsnap.ver \
  -o "$OUT/libygopro-core-effectinfo.so" *.o \
  -L/usr/lib/x86_64-linux-gnu -l:liblua5.3.so.0 -lpthread -ldl -lm
if ! nm -D "$OUT/libygopro-core-effectinfo.so" | grep -q ' duel_set_future_seed$'; then
  echo "BUILD_ERROR missing duel_set_future_seed; apply mirrorforce/patches/ygopro-core-future-rng-particle.patch" >&2
  exit 1
fi
echo "BUILD_OK $OUT/libygopro-core-effectinfo.so"
