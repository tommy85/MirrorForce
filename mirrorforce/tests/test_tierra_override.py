"""Regression for Tierra's exponential 10-distinct-name summon check.

**This duel is pinned by content, not by index.** It used to take
``_manifest_matchups(pool, 8, 26961102, "eval")[4]``, an **index** into a pool of 1,104 decks, 1,000 of which
(the perturb + synth layers) are not committed and are
regenerated locally from seeds. Tierra (91588074) exists in the whole manifest
**only in one deck, `synth/synth-00260.ydk`**; once the pool was regenerated, the index chose another duel,
the target card never appeared, and ``assertGreater(rows, 0)`` got 0: the failure looked like broken functionality,
while actually the fixture had drifted.
That deck is now committed as a fixture in ``tests/fixtures/`` (its source and generation seed are in that directory's
README), so it no longer drifts when the synthetic layer is regenerated.

**What this fixture honestly guards now**: the duel is reproducible and the four menu / resolution checks remain. But
**it no longer reproduces the pathological enumeration**: rerun without the override, this duel also finishes in 0.3 seconds. The pathological board
(a hand of 10 different names) is extremely rare in random duels; a scan of 6 duels with Tierra
hit it in none; the original one came out of a 14,000-game corpus. So the 20-second time-limit assertion is currently
**empty**, and the real guard is :func:`_override_in_effect`. Giving the time limit its teeth back
needs a **constructed board** (`mirrorforce/mirrorforce/search/rebuild.py` can lay out any board with the Debug API)
instead of a random duel, which is a redesign of its own.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import unittest
from pathlib import Path

PKG = Path(os.environ.get(
    "MF_PKG_ROOT", Path(__file__).resolve().parents[1]
))
ROOT = PKG.parent
TRAIN = PKG / "archive" / "worldmodel-train"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(TRAIN))

WM_ROOT = Path(os.environ.get("WM_ROOT", ROOT))
DECK_MANIFEST = WM_ROOT / "data/decks-v1/deck-manifest.json"
if not DECK_MANIFEST.is_file():
    DECK_MANIFEST = PKG / "data/decks-v1/deck-manifest.json"
OVERRIDE = Path(os.environ.get(
    "MF_SCRIPT_OVERRIDES", PKG / "script-overrides"
)) / "c91588074.lua"
ZH_DB = Path(os.environ.get(
    "WM_ZH_DB", ROOT / "ygo-agent/assets/locale/zh/cards.cdb"
))
EN_DB = Path(os.environ.get(
    "WM_EN_DB", ROOT / "ygo-agent/assets/locale/en/cards.cdb"
))

TIERRA = 91588074
FIXTURES = Path(__file__).resolve().parent / "fixtures"
DECKS = PKG / "data/decks-v1"


def _override_in_effect(name: str) -> bool:
    """Does this override really come before the stock script?

    This is **the only assertion with teeth** left in this file: the pathological enumeration only blows up on specific boards,
    and such boards are extremely rare in random duels. So "this duel runs fast" cannot prove the override is still there;
    what proves it is "when this card's script is resolved, the override directory wins the search order".
    """
    from mirrorforce.effectinfo import DEFAULT_SCRIPT_OVERRIDES, get_effectinfo_core

    resolved = get_effectinfo_core()._resolve_script(f"./script/{name}")
    return resolved is not None and str(DEFAULT_SCRIPT_OVERRIDES) in str(resolved)


def _deck_has(path: Path, code: int) -> bool:
    return f"\n{code}\n" in "\n" + path.read_text(errors="replace")


#: Pinned by content. The left deck is the committed fixture (the synthetic layer is not committed; see fixtures/README.md),
#: the right one from the committed real layer.
TIERRA_MATCHUP = (
    FIXTURES / "synth-00260.ydk",
    DECKS / "real/Floowandereeze.ydk",
    20663519990378,
)


def _pathological_game(queue):
    from gen_corpus import _manifest_run_one, _manifest_worker_init

    left, right, seed = TIERRA_MATCHUP
    matchup = (str(left), str(right), seed, ("mixed", "mixed"))
    _manifest_worker_init(
        str(ZH_DB), [str(EN_DB)], str(DECK_MANIFEST), 5
    )
    result = _manifest_run_one((
        *matchup,
        {
            "branch_stride": 1,
            "branches": 0,
            "branch_chosen": False,
            "max_branch_index": 100000,
            "include_opponent_candidates": True,
            "effectinfo": True,
        },
        True,
    ))
    # This counts "Tierra was put among the candidates", not "the random policy happened to choose it". It used to
    # take the conjunction with ``sample.labels``, requiring the random policy to **choose** it: policy noise,
    # unrelated to the override; its appearing among the candidates already proves the summon condition was evaluated (the
    # exponential enumeration blew up exactly during evaluation), which is the real coupling of this regression. The Eater test always
    # counted appearances only, so both now use the same calibration.
    tierra_rows = sum(
        any(candidate.code == TIERRA for candidate in sample.candidates)
        for sample in result.menu
    )
    queue.put((
        result.ok, result.error, len(result.menu), len(result.settle), tierra_rows
    ))


@unittest.skipUnless(
    DECK_MANIFEST.is_file() and OVERRIDE.is_file()
    and ZH_DB.is_file() and EN_DB.is_file(),
    "needs deck-v1, databases and the Tierra override",
)
class TierraOverrideTest(unittest.TestCase):
    def test_the_override_wins_the_script_search_order(self):
        self.assertTrue(
            _override_in_effect("c91588074.lua"),
            "the stock script comes before the override: Tierra's summon condition would fall back to the exponential upstream version, "
            "and this duel running fast proves nothing, since pathological boards are rare anyway",
        )

    def test_pathological_large_hand_finishes(self):
        left, right, _seed = TIERRA_MATCHUP
        for deck in (left, right):
            if not deck.is_file():
                self.skipTest(f"fixture deck missing: {deck}")
        if not (_deck_has(left, TIERRA) or _deck_has(right, TIERRA)):
            self.skipTest(
                f"neither deck of this duel contains Tierra ({TIERRA}): the fixture drifted, the functionality is not broken. "
                "The perturb/synth layers of the pool are not committed and are regenerated from seeds, so never pin an index "
                "into the pool",
            )
        ctx = mp.get_context("fork")
        queue = ctx.Queue()
        process = ctx.Process(target=_pathological_game, args=(queue,))
        process.start()
        process.join(20)
        if process.is_alive():
            process.terminate()
            process.join(5)
            self.fail("Tierra summon condition still exceeds 20 seconds")
        self.assertEqual(process.exitcode, 0)
        ok, error, menu, settle, tierra_rows = queue.get(timeout=1)
        self.assertTrue(ok)
        self.assertEqual(error, "")
        # this duel measured 68 rows; the lower bound only catches "it never ran", it is not a performance metric
        self.assertGreater(menu, 40)
        self.assertEqual(menu, settle)
        if tierra_rows == 0:
            self.skipTest(
                "this duel finished but Tierra never entered the candidates: the fixture drifted, the functionality is not broken; "
                "never pin an index into the regenerated pool",
            )


if __name__ == "__main__":
    unittest.main()
