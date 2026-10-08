#!/usr/bin/env bash
# Full local build of ygoenv against ygopro-core master (8046f3f).
# Deps already fetched under $SC/deps by the earlier PV-2 build; see setup-notes.md.
set -e
SC=/tmp/scratch
CORE=/path/to/workspace/ygopro-core          # checked out at origin/master
AGENT=/path/to/workspace/ygo-agent

# 1. stage core sources + the ONE patch still needed (lua headers need C linkage
#    because we link the C build, liblua5.3.so.0).  The old pin also needed
#    `#include <cstring>` in field.h; master no longer does.
rm -rf $SC/build/core-new && mkdir -p $SC/build/core-new
cp $CORE/*.cpp $CORE/*.h $SC/build/core-new/
python3 - "$SC/build/core-new/interpreter.h" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = '#include <lua.h>\n#include <lauxlib.h>\n#include <lualib.h>\n'
new = 'extern "C" {\n' + old + '}\n'
assert old in s
open(p, 'w').write(s.replace(old, new, 1))
PY

# 2. core static lib (16 TUs; mem.cpp was deleted upstream)
cd $SC/build/core-new
for f in *.cpp; do g++ -std=c++17 -O2 -fPIC -I. -I$SC/inc/lua -c "$f" -o "${f%.cpp}.o" & done; wait
ar rcs $SC/build/libygopro-core-new.a *.o

# 3. headers ygoenv includes as "ygopro-core/*.h"
rm -rf $SC/inc/ygopro-core-new && mkdir -p $SC/inc/ygopro-core-new
cp $SC/build/core-new/*.h $SC/inc/ygopro-core-new/
mkdir -p $SC/inc-new
ln -sfn $SC/inc/ygopro-core-new $SC/inc-new/ygopro-core
ln -sfn $SC/inc/concurrentqueue $SC/inc-new/concurrentqueue
ln -sfn $SC/inc/lua            $SC/inc-new/lua

# 4. link the python extension (sqlite/SQLiteCpp objects reused from $SC/build/sq)
g++ -std=c++17 -O2 -fPIC -shared -DFMT_HEADER_ONLY=1 \
  -I$AGENT/ygoenv -I$SC/inc-new -I$SC/deps/fmt/include -I$SC/deps/ud/include \
  -I$SC/deps/scpp/include -I$SC/deps/inc/sqlite-amalgamation-3450100 \
  -I$(python3 -c "import pybind11;print(pybind11.get_include())") -I/usr/include/python3.10 \
  $AGENT/ygoenv/ygoenv/ygopro/ygopro.cpp \
  $SC/build/sq/*.o $SC/build/libygopro-core-new.a \
  -L/usr/lib/x86_64-linux-gnu -l:liblua5.3.so.0 -lglog -lpthread -ldl -lm \
  -o $SC/build/ygopro_ygoenv-new.so
cp $SC/build/ygopro_ygoenv-new.so $SC/pypath-new/ygoenv/ygopro/ygopro_ygoenv.so
cp $AGENT/ygoenv/ygoenv/ygopro/__init__.py $SC/pypath-new/ygoenv/ygopro/
echo BUILD_OK
# NOTE: run from a directory containing `script/` -> ygopro-scripts, otherwise
# every card script silently fails to load and all cards lose their effects.
