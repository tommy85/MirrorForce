#!/usr/bin/env bash
# Build the unmodified upstream ygopro-core, the one a real server runs, for the host side of the
# follower's random-duel sweeps: the server of those duels must be the original core, while the
# follower's own local core carries the MirrorForce patches.
#
# Source: the upstream commit the maintained branch is based on (Fluorohydride/ygopro-core), read
# from the core repository with `git archive`, so no working tree is touched. Same compiler flags and
# Lua binding as build/effectinfo/build-effectinfo-core.sh, without any MirrorForce file.
#
# Output: mirrorforce/build/upstream/libygopro-core-upstream-<commit>.so and its .sha256
set -euo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=${MF_REPO_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)}
CORE=${MF_CORE_ROOT:-$REPO_ROOT/ygopro-core}
COMMIT=${MF_UPSTREAM_COMMIT:-8046f3f}
LUA_INC=${MF_LUA_INC:-$REPO_ROOT/build-deps/priv/inc/lua}
FULL=$(git -C "$CORE" rev-parse --verify "$COMMIT^{commit}")
git -C "$CORE" merge-base --is-ancestor "$FULL" upstream/master \
  || { echo "BUILD_ERROR $FULL is not on upstream/master" >&2; exit 1; }
OUT=$SCRIPT_DIR/libygopro-core-upstream-${FULL:0:8}.so
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
git -C "$CORE" archive "$FULL" | tar -x -C "$WORK"
if ls "$WORK"/mfsnap* >/dev/null 2>&1; then echo "BUILD_ERROR MirrorForce files in the upstream tree" >&2; exit 1; fi
python3 - "$WORK/interpreter.h" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = '#include <lua.h>\n#include <lauxlib.h>\n#include <lualib.h>\n'
assert old in s, "interpreter.h patch anchor missing"
open(p, 'w').write(s.replace(old, 'extern "C" {\n' + old + '}\n', 1))
PY
cd "$WORK"
for f in *.cpp; do
  g++ -std=c++17 -O2 -fPIC -fvisibility=hidden -DOCGCORE_EXPORT_FUNCTIONS -I. -I"$LUA_INC" -c "$f" -o "${f%.cpp}.o" &
done
wait
g++ -shared -Wl,-Bsymbolic -o "$OUT" *.o -L/usr/lib/x86_64-linux-gnu -l:liblua5.3.so.0 -lpthread -ldl -lm
if nm -D "$OUT" | grep -qE ' (duel_hydrate_card|duel_order_cards|duel_snapshot)$'; then
  echo "BUILD_ERROR a MirrorForce symbol in the upstream core" >&2; exit 1
fi
sha256sum "$OUT" | cut -d' ' -f1 > "$OUT.sha256"
echo "BUILD_OK $OUT $FULL $(cat "$OUT.sha256")"
