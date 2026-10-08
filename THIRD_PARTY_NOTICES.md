# Third-party notices

The code in this repository is MIT licensed (see `LICENSE`), except as listed below.

## Code derived from other projects

| Path | Origin | License |
|---|---|---|
| `mirrorforce/cxx/duelpool/`, `mirrorforce/mirrorforce/agent/{model,rl,train,semantics}/`, `mirrorforce/mirrorforce/agent/{constants,utils}.py` | Started from a fork of [sbl1996/ygo-agent](https://github.com/sbl1996/ygo-agent), imported with the fork author's consent; see `mirrorforce/mirrorforce/agent/NOTICE` | MIT, Copyright (c) 2024 Hastur (`mirrorforce/mirrorforce/agent/LICENSE.upstream`) |
| `mirrorforce/cxx/duelpool/envpool/` | [EnvPool](https://github.com/sail-sg/envpool) | Apache License 2.0, Copyright 2021 Garena Online Private Limited; the headers keep their notices |

## Scripts distributed under GPL-2.0

These files modify, or are derived from, GPL-2.0 projects and are distributed under the
[GNU General Public License, version 2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html):

| Path | Project |
|---|---|
| `mirrorforce/script-overrides/*.lua` | [Fluorohydride/ygopro-scripts](https://github.com/Fluorohydride/ygopro-scripts) |

## The rules engine

`third_party/ygopro-core` is a git submodule: our fork of
[Fluorohydride/ygopro-core](https://github.com/Fluorohydride/ygopro-core), MIT licensed, Copyright (c) 2015
Fluorohydride.

## Not included

The card database (`cards.cdb`), the Lua card scripts, training data and WinBot are not part of this repository.
The champion's weights are a release asset, bundled with the card database and the scripts they were trained with;
the scripts in that bundle are GPL-2.0 (Fluorohydride/ygopro-scripts). Yu-Gi-Oh! is a trademark of its owners; this project is not affiliated with or endorsed by them.
