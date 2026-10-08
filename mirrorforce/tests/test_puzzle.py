"""Smoke test for the headless single-mode (puzzle) harness.

Two halves.  The message splitter is exercised on hand-built buffers and needs
nothing but Python -- it is the piece most likely to drift, because its length
rules are a port of ``single_mode.cpp``.  The rest needs the pinned core, the
pinned card database and the archived puzzle packs, and is skipped when any of
them is missing (a checkout without ``build-deps`` still runs the first half).

The engine half is the PV-12 gate in executable form: the three puzzles load
with no card missing from the database and no message from the core, and each
written solution reaches ``MSG_WIN`` with the player as the winner.
"""

from __future__ import annotations

import json
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirrorforce.netduel import constants as C  # noqa: E402
from mirrorforce.puzzle.core import (  # noqa: E402
    DEFAULT_DB,
    DEFAULT_LIB,
    DEFAULT_PUZZLE_DIR,
    DEFAULT_SCRIPTS,
)
from mirrorforce.puzzle.messages import UnknownMessage, split_messages  # noqa: E402

SOLUTIONS_FILE = DEFAULT_PUZZLE_DIR / "verified-solutions.json"

_HAVE_ENGINE = (
    Path(DEFAULT_LIB).is_file()
    and Path(DEFAULT_DB).is_file()
    and Path(DEFAULT_SCRIPTS).is_dir()
    and SOLUTIONS_FILE.is_file()
)
_SKIP_REASON = (
    f"needs the pinned engine ({DEFAULT_LIB}), database ({DEFAULT_DB}), "
    f"card scripts ({DEFAULT_SCRIPTS}) and puzzle archive ({SOLUTIONS_FILE})"
)


class SplitMessagesTest(unittest.TestCase):
    """The length rules ported from ``SingleMode::SinglePlayAnalyze``."""

    def test_fixed_length_messages_are_split(self):
        buf = (
            bytes([C.MSG_NEW_TURN, 0])
            + bytes([C.MSG_NEW_PHASE]) + struct.pack("<H", C.PHASE_MAIN1)
            + bytes([C.MSG_WIN, 0, 1])
        )
        got = [(m.msg, m.payload) for m in split_messages(buf)]
        self.assertEqual(
            got,
            [
                (C.MSG_NEW_TURN, b"\x00"),
                (C.MSG_NEW_PHASE, struct.pack("<H", C.PHASE_MAIN1)),
                (C.MSG_WIN, b"\x00\x01"),
            ],
        )

    def test_counted_lists(self):
        # MSG_SELECT_IDLECMD: player, five 7-byte lists, one 11-byte list, 3 tail
        payload = bytes([0])
        payload += bytes([1]) + b"\x01" * 7          # summonable
        payload += bytes([0]) * 4                     # spsummonable/repos/msetable/sset
        payload += bytes([0])                         # activatable count
        payload += bytes([1, 0, 0])                   # to_bp, to_ep, shuffle
        buf = bytes([C.MSG_SELECT_IDLECMD]) + payload
        (message,) = split_messages(buf)
        self.assertEqual(message.msg, C.MSG_SELECT_IDLECMD)
        self.assertEqual(message.payload, payload)

    def test_length_prefixed_strings(self):
        name = "Rätsel ¿".encode("utf8")
        buf = bytes([C.MSG_AI_NAME]) + struct.pack("<H", len(name)) + name + b"\x00"
        (message,) = split_messages(buf)
        self.assertEqual(message.msg, C.MSG_AI_NAME)
        self.assertEqual(message.payload[2 : 2 + len(name)], name)

    def test_unknown_message_id_is_reported(self):
        with self.assertRaises(UnknownMessage):
            split_messages(bytes([254, 0, 0]))

    def test_truncated_buffer_is_reported(self):
        with self.assertRaises(UnknownMessage):
            split_messages(bytes([C.MSG_WIN, 0]))


@unittest.skipUnless(_HAVE_ENGINE, _SKIP_REASON)
class PuzzleEngineTest(unittest.TestCase):
    """Load and play the puzzles PV-12 signed off on."""

    @classmethod
    def setUpClass(cls):
        from mirrorforce.puzzle import get_core

        cls.core = get_core()
        cls.solutions = json.loads(SOLUTIONS_FILE.read_text(encoding="utf8"))[
            "solutions"
        ]

    def _puzzle(self, entry):
        from mirrorforce.puzzle import SinglePuzzle

        return SinglePuzzle(
            DEFAULT_PUZZLE_DIR / entry["puzzle"],
            edopro_shim=entry.get("edopro_shim", False),
        )

    def test_board_matches_the_script(self):
        """A puzzle's board is what its ``Debug.AddCard`` calls asked for."""
        from mirrorforce.puzzle import SinglePuzzle

        path = (
            DEFAULT_PUZZLE_DIR
            / "projectignis-custom-puzzles/single/Tutorial_Xyz.lua"
        )
        with SinglePuzzle(path) as puzzle:
            result = puzzle.run_result
            self.assertTrue(result.loaded)
            self.assertEqual(result.missing_codes, [])
            self.assertEqual(result.core_log, [])
            self.assertEqual(result.ai_name, "TutorialBot")
            self.assertEqual(len(result.hints), 3)

            info = puzzle.field_info()
            self.assertEqual(info.duel_rule, 3)  # the script asks for Master Rule 3
            self.assertEqual(info.lp, (4000, 2300))

            ours = puzzle.zone(0, C.LOCATION_MZONE)
            self.assertEqual(
                [card.code if card else None for card in ours[:4]],
                [None, 67120578, 26082117, 12423762],
            )
            self.assertTrue(all(card.position == C.POS_FACEUP_ATTACK for card in ours[1:4]))

            # three monsters were added to the same occupied zone, so the two
            # later ones became Xyz materials (libdebug.cpp debug_add_card)
            theirs = puzzle.zone(1, C.LOCATION_MZONE)[2]
            self.assertEqual(theirs.code, 88120966)
            self.assertEqual(theirs.overlay, [92418590, 34620088])

    def test_every_verified_solution_wins(self):
        """Every archived solution still reaches ``MSG_WIN`` with player 0 winning.

        Three per-entry knobs, all optional so the PV-12 phase-1 entries stay as
        they were written: ``decline_chains`` (default true) matches
        ``solution_policy``'s own default, and is set false by a solution that
        chains on purpose; ``win_reason`` (default ``WIN_REASON_LP``) covers the
        puzzles that win by an alternate condition such as Exodia;
        ``edopro_shim`` (default false) is set only by a puzzle that will not
        load without it, and only where the shim is the ``Card.Type`` alias,
        which leaves the rules unchanged.  An entry that needed a duel-mode
        flag would be rule-unfaithful and does not belong in this file at all,
        which the ``rules_unfaithful`` assertion below enforces.
        """
        from mirrorforce.puzzle import solution_policy

        self.assertGreaterEqual(len(self.solutions), 3)
        for entry in self.solutions:
            with self.subTest(puzzle=entry["puzzle"]):
                puzzle = self._puzzle(entry)
                try:
                    loaded = puzzle.load()
                    self.assertTrue(loaded.loaded)
                    self.assertEqual(loaded.missing_codes, [])
                    self.assertEqual(loaded.core_log, [])
                    # a solution is only meaningful under the puzzle's own rules
                    self.assertFalse(loaded.rules_unfaithful)
                    self.assertEqual(list(puzzle.field_info().lp), entry["life"])
                    self.assertEqual(puzzle.field_info().duel_rule, entry["duel_rule"])
                    result = puzzle.play(
                        solution_policy(
                            entry["steps"],
                            decline_chains=entry.get("decline_chains", True),
                        )
                    )
                finally:
                    puzzle.close()
                self.assertEqual(result.winner, 0, result.error)
                self.assertEqual(
                    result.win_reason, entry.get("win_reason", C.WIN_REASON_LP)
                )
                self.assertEqual(result.core_log, [])
                self.assertEqual(result.missing_codes, [])

    def test_edopro_shim_is_off_by_default(self):
        """A puzzle that needs an EDOPro-only global fails loudly, not silently."""
        path = (
            DEFAULT_PUZZLE_DIR
            / "projectignis-puzzles/Duel Links/Naim_DuelLinks_1_1.lua"
        )
        puzzle = self._puzzle({"puzzle": path.relative_to(DEFAULT_PUZZLE_DIR)})
        try:
            result = puzzle.load()
        finally:
            puzzle.close()
        self.assertFalse(result.loaded)
        self.assertTrue(any("DUEL_MODE_SPEED" in line for line in result.core_log))
        self.assertEqual(result.shim_symbols, [])
        self.assertFalse(result.rules_unfaithful)

    def test_edopro_shim_marks_a_mode_flag_as_rules_unfaithful(self):
        """Defining DUEL_MODE_SPEED as 0 loads the board under *our* rules."""
        from mirrorforce.puzzle import SinglePuzzle

        path = (
            DEFAULT_PUZZLE_DIR
            / "projectignis-puzzles/Duel Links/Naim_DuelLinks_1_1.lua"
        )
        with SinglePuzzle(path, edopro_shim=True) as puzzle:
            result = puzzle.run_result
            self.assertTrue(result.loaded)
            self.assertEqual(result.missing_codes, [])
            self.assertEqual(result.core_log, [])
            self.assertEqual(result.shim_symbols, ["DUEL_MODE_SPEED"])
            self.assertTrue(result.rules_unfaithful)

    def test_edopro_shim_card_type_alias_stays_faithful(self):
        """``Card.Type`` is EDOPro's spelling of ``Card.GetType``: same rules."""
        from mirrorforce.puzzle import SinglePuzzle, first_policy

        path = (
            DEFAULT_PUZZLE_DIR
            / "projectignis-puzzles/Miscellaneous/Eroldin_10_Ladies_Night.lua"
        )
        puzzle = SinglePuzzle(path, edopro_shim=True)
        try:
            result = puzzle.load()
            self.assertTrue(result.loaded)
            self.assertEqual(result.core_log, [])
            self.assertEqual(result.shim_symbols, ["Card.Type"])
            self.assertFalse(result.rules_unfaithful)
            run = puzzle.play(first_policy, max_steps=3000)
        finally:
            puzzle.close()
        self.assertIsNotNone(run.winner)
        self.assertEqual(run.core_log, [])

    def test_edopro_shim_does_not_disturb_a_puzzle_that_does_not_need_it(self):
        """The shim only defines names; a puzzle that ignores them is unchanged."""
        from mirrorforce.puzzle import SinglePuzzle, solution_policy

        entry = next(
            e for e in self.solutions if e["puzzle"].endswith("Frog Family.lua")
        )
        puzzle = SinglePuzzle(DEFAULT_PUZZLE_DIR / entry["puzzle"], edopro_shim=True)
        try:
            self.assertTrue(puzzle.load().loaded)
            self.assertEqual(puzzle.run_result.shim_symbols, [])
            result = puzzle.play(solution_policy(entry["steps"]))
        finally:
            puzzle.close()
        self.assertEqual(result.winner, 0)
        self.assertEqual(result.core_log, [])

    def test_select_sum_prompt_no_longer_stalls_the_harness(self):
        """Regression for the MSG_SELECT_SUM enumeration fix.

        Two must-select cards already reach the target, so the only answer is
        "nothing more"; the harness used to raise ``offers no legal action``
        here and could not drive the puzzle at all.
        """
        from mirrorforce.puzzle import SinglePuzzle, first_policy

        path = (
            DEFAULT_PUZZLE_DIR
            / "projectignis-puzzles/Miscellaneous/Furtie_Hubo_02_DDD_Remastered.lua"
        )
        puzzle = SinglePuzzle(path)
        try:
            self.assertTrue(puzzle.load().loaded)
            result = puzzle.play(first_policy, max_steps=3000)
        finally:
            puzzle.close()
        self.assertIsNotNone(result.winner)
        self.assertEqual(result.core_log, [])

    def test_missing_card_is_detected(self):
        """A puzzle built on codes the 2026 database lacks is reported, not hidden."""
        path = (
            DEFAULT_PUZZLE_DIR
            / "projectignis-custom-puzzles/single/[WCS2006]34_fuurinkazan.lua"
        )
        puzzle = self._puzzle({"puzzle": path.relative_to(DEFAULT_PUZZLE_DIR)})
        try:
            result = puzzle.load()
        finally:
            puzzle.close()
        self.assertTrue(result.loaded)
        self.assertIn(511000228, result.missing_codes)
        self.assertTrue(result.core_log)


if __name__ == "__main__":
    unittest.main()
