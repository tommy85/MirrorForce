"""Tests for ``query_effect_info``, the core's "already applied" dump.

Four scenarios, each a single-mode puzzle built with the ``Debug`` library so
the board is exact rather than whatever a random walk reached:

* **Maxx "C"** (23434538) -- after it resolves, three ``EFFECT_TYPE_CONTINUOUS``
  effects are registered on the activating player, their source card is the
  copy now in the *graveyard*, and they are gone once the turn ends.  This is
  the case the dump exists for: from outside the core the graveyard cannot tell
  a discarded copy from a resolved one.
* **Droll & Lock Bird** (94145021) -- two restriction auras
  (``EFFECT_CANNOT_TO_HAND``, ``EFFECT_CANNOT_DRAW``), registered by one player
  and pointed at *both*, likewise gone at the end phase.
* **Ash Blossom** (14558127) -- the control: it registers nothing, so the only
  residue is one row in the once-per-turn table.
* **once-per-turn accounting** -- the second copy of Ash Blossom is never
  offered while the first copy's row stands, and the row is cleared when the
  next turn begins.

Every assertion is against the dump *and* against what the core actually did,
so a dump that agrees with itself but not with the duel fails here.

The engine half is skipped when the patched core has not been built; the format
parser is pure Python and always runs.
"""

from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirrorforce import effectinfo  # noqa: E402
from mirrorforce.effectinfo import (  # noqa: E402
    DEFAULT_EFFECTINFO_LIB,
    EffectInfo,
    EffectInfoError,
    get_effectinfo_core,
    parse,
    query_effect_info,
    script_override_provenance,
)
from mirrorforce.netduel import constants as C  # noqa: E402
from mirrorforce.puzzle.core import DEFAULT_DB, DEFAULT_SCRIPTS  # noqa: E402

MAXX_C = 23434538
DROLL = 94145021
ASH = 14558127
IMPERMANENCE = 10045474
CALLED_BY = 24224830
POT_OF_GREED = 55144522
FLAMVELL_GUARD = 21615956  # a vanilla monster: no effects, no noise

# effect.h
EFFECT_CANNOT_DRAW = 25
EFFECT_CANNOT_TO_HAND = 65
EFFECT_DISABLE = 2
EFFECT_DISABLE_EFFECT = 8
EVENT_CHAIN_SOLVED = 1022
EVENT_SPSUMMON_SUCCESS = 1102
EFFECT_TYPE_CONTINUOUS = 0x0800
EFFECT_TYPE_FIELD = 0x0002
RESET_PHASE = 0x40000000
PHASE_END = 0x200

MR5_OPTIONS = 5 << 16

ENGINE_READY = (
    Path(DEFAULT_EFFECTINFO_LIB).is_file()
    and Path(DEFAULT_DB).is_file()
    and Path(DEFAULT_SCRIPTS).is_dir()
)

_PREAMBLE = """
Debug.SetAIName("effectinfo")
Debug.SetPlayerInfo(0,8000,0,0)
Debug.SetPlayerInfo(1,8000,0,0)
"""

_DECKS = """
for i=1,10 do
  Debug.AddCard(21615956,0,0,LOCATION_DECK,0,POS_FACEDOWN_DEFENSE)
  Debug.AddCard(21615956,1,1,LOCATION_DECK,0,POS_FACEDOWN_DEFENSE)
end
Debug.ReloadFieldEnd()
"""


def _puzzle(tmpdir: Path, name: str, hands: dict[int, list[int]]) -> Path:
    """Write a puzzle that deals ``hands`` and stocks both decks."""
    lines = [_PREAMBLE]
    for player, codes in sorted(hands.items()):
        for sequence, code in enumerate(codes):
            lines.append(
                f"Debug.AddCard({code},{player},{player},"
                f"LOCATION_HAND,{sequence},POS_FACEDOWN)\n"
            )
    lines.append(_DECKS)
    path = tmpdir / f"{name}.lua"
    path.write_text("".join(lines), encoding="utf8")
    return path


class Walk:
    """One puzzle played to a fixed depth, dumping at every decision.

    ``frames`` is the dump at each decision in order, ``menus`` the options the
    core offered there; asserting on both together is what makes "the second
    copy was never offered" a statement about the duel and not just about the
    dump.
    """

    def __init__(self, path: Path, max_steps: int = 600):
        from mirrorforce.puzzle.single import SinglePuzzle

        self.core = get_effectinfo_core()
        self.frames: list[EffectInfo] = []
        self.menus: list[list[str]] = []
        self.prompts: list[int] = []
        self.chosen: list[str] = []
        self.puzzle = SinglePuzzle(path, core=self.core, options=MR5_OPTIONS)
        self.load_result = self.puzzle.load()
        self.run_result = None
        self.max_steps = max_steps

    def play(self, policy=None) -> "Walk":
        if policy is None:
            policy = _prefer_activate
        try:
            self.run_result = self.puzzle.play(
                policy=self._record(policy), max_steps=self.max_steps
            )
        finally:
            self.puzzle.close()
        return self

    def _record(self, policy):
        def choose(selector, actions, puzzle):
            self.frames.append(query_effect_info(puzzle.core, puzzle.pduel))
            self.menus.append([a.describe() for a in actions])
            self.prompts.append(selector.msg)
            index = policy(selector, actions, puzzle)
            self.chosen.append(actions[index].describe())
            return index

        return choose

    def frame_at_turn(self, turn: int) -> EffectInfo:
        """The first dump taken during ``turn``."""
        for info in self.frames:
            if info.turn.turn_id == turn:
                return info
        raise AssertionError(f"the walk never reached turn {turn}")


def _prefer_activate(selector, actions, puzzle) -> int:
    """Take the first activation on offer, else the first option.

    Both hand traps here are the *only* activation their holder has, so this
    plays them at the first window that offers them -- which is precisely the
    condition each test wants to observe.
    """
    for index, action in enumerate(actions):
        if "activate" in action.describe():
            return index
    return 0


class FormatTests(unittest.TestCase):
    def test_runtime_script_override_has_frozen_provenance(self):
        self.assertEqual(script_override_provenance(), [
            {
                "name": "c63845230.lua",
                "sha256": (
                    "d3ab8925ecaed25dfbfb753250680961f91e7a2588bfcc621fcc79793e32857a"
                ),
            },
            {
                "name": "c91588074.lua",
                "sha256": (
                    "18f56fd0646bd2cd10d044602a942a7a8e835caa9deea75c8d68fd60cb56e90e"
                ),
            },
        ])

    """The wire format, without the core."""

    def test_header_round_trip(self):
        raw = effectinfo.HEADER.pack(effectinfo.VERSION, 8, 0, 13)
        raw += effectinfo.SECTION.pack(effectinfo.SECTION_END, 0)
        info = parse(raw)
        self.assertEqual(info.version, effectinfo.VERSION)
        self.assertEqual(info.effects, ())
        self.assertIsNone(info.turn)

    def test_unknown_section_is_kept_not_dropped(self):
        """A newer core may add sections; an older reader must not lose them."""
        payload = b"\xde\xad\xbe\xef"
        body = effectinfo.SECTION.pack(99, len(payload)) + payload
        body += effectinfo.SECTION.pack(effectinfo.SECTION_END, 0)
        raw = effectinfo.HEADER.pack(effectinfo.VERSION, 8, 0, 8 + len(body)) + body
        info = parse(raw)
        self.assertEqual(info.unknown_sections, {99: payload})

    def test_version_mismatch_is_refused(self):
        raw = effectinfo.HEADER.pack(effectinfo.VERSION + 1, 8, 0, 8)
        with self.assertRaises(EffectInfoError):
            parse(raw)

    def test_truncated_dump_is_refused(self):
        raw = effectinfo.HEADER.pack(effectinfo.VERSION, 8, 0, 64)
        with self.assertRaises(EffectInfoError):
            parse(raw)

    def test_record_sizes_match_the_core(self):
        """The struct strings and the C writer must agree byte for byte."""
        self.assertEqual(effectinfo._EFFECT.size, 72)
        self.assertEqual(effectinfo._EFFECT_EXT.size, 92)
        self.assertEqual(effectinfo._COUNT.size, 10)
        self.assertEqual(effectinfo._SPONCE.size, 9)
        self.assertEqual(effectinfo._ACTIVITY.size, 9)
        self.assertEqual(effectinfo._TURN_HEAD.size, 14)
        self.assertEqual(effectinfo._TURN_PLAYER.size, 12)
        self.assertEqual(effectinfo._PLAYER.size, 20)
        self.assertEqual(effectinfo._CARD_STATE.size, 60)
        self.assertEqual(effectinfo._RELATION.size, 28)
        self.assertEqual(effectinfo._SETCODE.size, 12)
        self.assertEqual(effectinfo.HEADER.size, 8)
        self.assertEqual(effectinfo.SECTION.size, 5)

    def test_extended_state_sections_round_trip(self):
        """The core-only state additions are parsed, not retained as opaque bytes."""
        sections = []
        payload = struct.pack("<HH5I", 1, effectinfo._PLAYER.size,
                              1, 0x1234, 0x40, 2, 6)
        sections.append(effectinfo.SECTION.pack(effectinfo.SECTION_PLAYER_STATE,
                                                len(payload)) + payload)
        card_values = (0x03040201, 123, 0, 1, 0x49000000, 0,
                       1, 0, 2, 0, 0, 0, 2, 1, 0)
        payload = struct.pack("<HH15I", 1, effectinfo._CARD_STATE.size,
                              *card_values)
        sections.append(effectinfo.SECTION.pack(effectinfo.SECTION_CARD_STATE,
                                                len(payload)) + payload)
        relation = struct.pack("<HH7I", 1, effectinfo._RELATION.size,
                               1, 1, 123, 2, 456, 0x400, 0)
        sections.append(effectinfo.SECTION.pack(effectinfo.SECTION_RELATIONS,
                                                len(relation)) + relation)
        setcode = struct.pack("<HH3I", 1, effectinfo._SETCODE.size,
                              0x03040201, 123, 0x1234)
        sections.append(effectinfo.SECTION.pack(effectinfo.SECTION_SETCODE,
                                                len(setcode)) + setcode)
        body = b"".join(sections) + effectinfo.SECTION.pack(effectinfo.SECTION_END, 0)
        raw = effectinfo.HEADER.pack(effectinfo.VERSION, 8, 0, 8 + len(body)) + body
        info = parse(raw)
        self.assertEqual(info.players[0].disabled_location, 0x40)
        self.assertEqual(info.cards[0].summon_info, 0x49000000)
        self.assertEqual(info.relations[0].kind, 1)
        self.assertEqual(info.setcodes[0].setcode, 0x1234)

    def test_longer_records_from_a_newer_core_are_skipped_by_length(self):
        """Records carry their own width, so a wider one still parses."""
        wide = effectinfo._EFFECT.size + 4
        payload = struct.pack("<HH", 1, wide)
        payload += effectinfo._EFFECT.pack(
            7, EFFECT_CANNOT_DRAW, EFFECT_TYPE_FIELD, DROLL, 0x10, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 1, 1, 0b11, 0, 0, 0, 0,
        )
        payload += b"\x00\x00\x00\x00"
        body = effectinfo.SECTION.pack(effectinfo.SECTION_EFFECTS, len(payload))
        body += payload + effectinfo.SECTION.pack(effectinfo.SECTION_END, 0)
        raw = effectinfo.HEADER.pack(effectinfo.VERSION, 8, 0, 8 + len(body)) + body
        info = parse(raw)
        self.assertEqual(len(info.effects), 1)
        self.assertEqual(info.effects[0].code, EFFECT_CANNOT_DRAW)
        self.assertTrue(info.effects[0].affects(0))
        self.assertTrue(info.effects[0].affects(1))

    def test_effect_extension_keeps_handler_and_card_label_object(self):
        owner_at = 1 | (C.LOCATION_GRAVE << 8) | (2 << 16) | (1 << 24)
        handler_at = 0 | (C.LOCATION_MZONE << 8) | (3 << 16) | (1 << 24)
        label_at = 1 | (C.LOCATION_GRAVE << 8) | (4 << 16) | (1 << 24)
        payload = struct.pack("<HH", 1, effectinfo._EFFECT_EXT.size)
        payload += effectinfo._EFFECT_EXT.pack(
            7, EFFECT_CANNOT_DRAW, EFFECT_TYPE_FIELD, DROLL, owner_at,
            0, RESET_PHASE | PHASE_END, 1, 0, 0,
            0, 0, 1, 1, 0, 0, 9, 1, 0b11, 0, 0, 0, 0,
            handler_at, 123, 4, label_at, 456,
        )
        body = effectinfo.SECTION.pack(effectinfo.SECTION_EFFECTS, len(payload))
        body += payload + effectinfo.SECTION.pack(effectinfo.SECTION_END, 0)
        raw = effectinfo.HEADER.pack(effectinfo.VERSION, 8, 0, 8 + len(body)) + body
        row = parse(raw).effects[0]
        self.assertEqual(row.container_name, "card_single")
        self.assertEqual(row.owner_info_location, owner_at)
        self.assertEqual(row.handler_info_location, handler_at)
        self.assertEqual(row.handler_code, 123)
        self.assertEqual(row.label_object_info_location, label_at)
        self.assertEqual(row.label_object_code, 456)


@unittest.skipUnless(
    ENGINE_READY,
    f"patched core not built ({DEFAULT_EFFECTINFO_LIB}); run "
    "mirrorforce/build/effectinfo/build-effectinfo-core.sh",
)
class EngineTests(unittest.TestCase):
    """The dump against a live duel."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(__file__).resolve().parent / "_effectinfo_tmp"
        cls.tmp.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        for path in cls.tmp.glob("*.lua"):
            path.unlink()
        cls.tmp.rmdir()

    # -- 1. Maxx "C" -------------------------------------------------------

    def test_maxx_c_registers_three_effects_owned_by_a_card_in_the_graveyard(self):
        path = _puzzle(
            self.tmp, "maxxc", {0: [FLAMVELL_GUARD], 1: [MAXX_C]}
        )
        walk = Walk(path).play()
        self.assertEqual(walk.load_result.missing_codes, [])
        self.assertEqual(walk.run_result.core_log, [])

        before = walk.frames[0]
        self.assertEqual(before.effects_from(MAXX_C), ())
        self.assertEqual(before.count_used(1, MAXX_C), 0)

        applied = [f for f in walk.frames if f.effects_from(MAXX_C)]
        self.assertTrue(applied, 'Maxx "C" never resolved in this walk')
        after = applied[0]
        self.assertEqual(after.turn.turn_id, 1)

        effects = after.effects_from(MAXX_C)
        self.assertEqual(len(effects), 3, [e.code for e in effects])
        self.assertEqual(
            sorted(e.code for e in effects),
            [EVENT_CHAIN_SOLVED, EVENT_SPSUMMON_SUCCESS, EVENT_SPSUMMON_SUCCESS],
        )
        for effect in effects:
            with self.subTest(effect=effect.id):
                # registered on the activating player, and only on them
                self.assertEqual(effect.effect_owner, 1)
                self.assertEqual(effect.container_name, "continuous")
                self.assertTrue(effect.type & EFFECT_TYPE_CONTINUOUS)
                self.assertTrue(effect.type & EFFECT_TYPE_FIELD)
                # the source card is the copy sent to the graveyard as cost --
                # the thing no QUERY_* flag can point at
                self.assertEqual(effect.owner_code, MAXX_C)
                self.assertEqual(effect.owner_location, C.LOCATION_GRAVE)
                self.assertEqual(effect.owner_controler, 1)
                # SetReset(RESET_PHASE+PHASE_END), one use
                self.assertTrue(effect.expires_this_turn)
                self.assertEqual(effect.reset_count, 1)
                self.assertTrue(effect.reset_flag & RESET_PHASE)
                self.assertTrue(effect.reset_flag & PHASE_END)
        self.assertEqual(after.count_used(1, MAXX_C), 1)
        self.assertEqual(after.count_used(0, MAXX_C), 0)

    def test_maxx_c_effects_are_gone_after_the_turn_they_resolved_in(self):
        path = _puzzle(self.tmp, "maxxc2", {0: [FLAMVELL_GUARD], 1: [MAXX_C]})
        walk = Walk(path).play()
        resolved = [f for f in walk.frames if f.effects_from(MAXX_C)]
        self.assertTrue(resolved, "Maxx \"C\" never resolved in this walk")
        turn = resolved[0].turn.turn_id

        later = [f for f in walk.frames if f.turn.turn_id > turn]
        self.assertTrue(later, "the walk never left the turn it resolved on")
        for info in later:
            with self.subTest(turn=info.turn.turn_id):
                # the RESET_PHASE+PHASE_END reset fired
                self.assertEqual(info.effects_from(MAXX_C), ())
                # and the turn count table was cleared at the turn boundary
                self.assertEqual(info.count_used(1, MAXX_C), 0)

    # -- 2. Droll & Lock Bird ---------------------------------------------

    def test_droll_registers_two_restriction_auras_aimed_at_both_players(self):
        path = _puzzle(self.tmp, "droll", {0: [POT_OF_GREED], 1: [DROLL]})
        walk = Walk(path).play()
        self.assertEqual(walk.load_result.missing_codes, [])
        self.assertEqual(walk.run_result.core_log, [])

        # the two auras only; the script also registers a duel-long EVENT_TO_HAND
        # watcher at initial_effect, which is not part of what resolving does
        applied = [
            f for f in walk.frames
            if any(
                e.code in (EFFECT_CANNOT_DRAW, EFFECT_CANNOT_TO_HAND)
                for e in f.effects_from(DROLL)
            )
        ]
        self.assertTrue(applied, "Droll & Lock Bird never resolved in this walk")
        info = applied[0]
        auras = [
            e for e in info.effects_from(DROLL)
            if e.code in (EFFECT_CANNOT_DRAW, EFFECT_CANNOT_TO_HAND)
        ]
        self.assertEqual(
            sorted(e.code for e in auras),
            [EFFECT_CANNOT_DRAW, EFFECT_CANNOT_TO_HAND],
        )
        for effect in auras:
            with self.subTest(code=effect.code):
                self.assertEqual(effect.container_name, "aura")
                self.assertEqual(effect.effect_owner, 1)
                self.assertEqual(effect.owner_code, DROLL)
                self.assertEqual(effect.owner_location, C.LOCATION_GRAVE)
                # SetTargetRange(1,1): both players, which is the whole reason
                # the effect_owner alone is not enough to encode the state
                self.assertTrue(effect.affects(0))
                self.assertTrue(effect.affects(1))
                self.assertEqual(effect.s_range, 1)
                self.assertEqual(effect.o_range, 1)
                self.assertTrue(effect.expires_this_turn)

        turn = info.turn.turn_id
        later = [f for f in walk.frames if f.turn.turn_id > turn]
        self.assertTrue(later, "the walk never left the turn Droll resolved on")
        for frame in later:
            with self.subTest(turn=frame.turn.turn_id):
                self.assertEqual(
                    [
                        e.code for e in frame.effects_from(DROLL)
                        if e.code in (EFFECT_CANNOT_DRAW, EFFECT_CANNOT_TO_HAND)
                    ],
                    [],
                )

    def test_droll_global_watcher_is_reported_and_never_expires(self):
        """The script's ``initial_effect`` watcher is a field-only effect too.

        It is registered when the card is created, belongs to player 0 whoever
        holds the card, and has no reset -- so a reader that treats every
        field-only effect as "somebody resolved something this turn" would be
        wrong.  The dump carries enough to tell them apart.
        """
        path = _puzzle(self.tmp, "droll2", {0: [POT_OF_GREED], 1: [DROLL]})
        walk = Walk(path).play()
        first = walk.frames[0]
        watchers = [
            e for e in first.effects_from(DROLL)
            if e.code not in (EFFECT_CANNOT_DRAW, EFFECT_CANNOT_TO_HAND)
        ]
        self.assertEqual(len(watchers), 1, [e.code for e in watchers])
        watcher = watchers[0]
        self.assertEqual(watcher.reset_flag, 0)
        self.assertEqual(watcher.count_code, 0)
        self.assertFalse(watcher.expires_this_turn)
        # Duel.RegisterEffect(ge1,0) in initial_effect: player 0 regardless of
        # who is holding the card, which is player 1 here
        self.assertEqual(watcher.effect_owner, 0)

    # -- 3. Ash Blossom, the control --------------------------------------

    def test_ash_blossom_leaves_only_a_once_per_turn_row(self):
        path = _puzzle(self.tmp, "ash", {0: [POT_OF_GREED], 1: [ASH]})
        walk = Walk(path).play()
        self.assertEqual(walk.load_result.missing_codes, [])
        self.assertEqual(walk.run_result.core_log, [])

        spent = [f for f in walk.frames if f.count_used(1, ASH)]
        self.assertTrue(spent, "Ash Blossom never resolved in this walk")
        info = spent[0]
        self.assertEqual(info.count_used(1, ASH), 1)
        # nothing registered: the whole point of the control case
        self.assertEqual(info.effects_from(ASH), ())
        self.assertEqual(info.effects, ())

        later = [f for f in walk.frames if f.turn.turn_id > info.turn.turn_id]
        self.assertTrue(later)
        self.assertEqual(later[0].count_used(1, ASH), 0)

    def test_impermanence_exports_dynamic_single_effects_on_its_target(self):
        lines = [
            _PREAMBLE,
            (f"Debug.AddCard({IMPERMANENCE},0,0,LOCATION_HAND,0,"
             "POS_FACEDOWN)\n"),
            (f"Debug.AddCard({ASH},1,1,LOCATION_MZONE,0,"
             "POS_FACEUP_ATTACK)\n"),
            _DECKS,
        ]
        path = self.tmp / "imperm.lua"
        path.write_text("".join(lines), encoding="utf8")
        walk = Walk(path).play()
        applied = [
            frame for frame in walk.frames
            if any(e.container_name == "card_single"
                   and e.owner_code == IMPERMANENCE
                   for e in frame.effects)
        ]
        self.assertTrue(applied, "Infinite Impermanence never left a card effect")
        effects = [
            e for e in applied[0].effects
            if e.container_name == "card_single"
            and e.owner_code == IMPERMANENCE
        ]
        self.assertEqual(sorted(e.code for e in effects),
                         [EFFECT_DISABLE, EFFECT_DISABLE_EFFECT])
        for effect in effects:
            with self.subTest(code=effect.code):
                self.assertEqual(effect.handler_code, ASH)
                self.assertEqual(effect.handler_location, C.LOCATION_MZONE)
                self.assertTrue(effect.expires_this_turn)

    def test_called_by_exports_the_public_card_held_in_label_object(self):
        lines = [
            _PREAMBLE,
            (f"Debug.AddCard({CALLED_BY},0,0,LOCATION_HAND,0,"
             "POS_FACEDOWN)\n"),
            (f"Debug.AddCard({ASH},1,1,LOCATION_GRAVE,0,"
             "POS_FACEUP_ATTACK)\n"),
            _DECKS,
        ]
        path = self.tmp / "calledby.lua"
        path.write_text("".join(lines), encoding="utf8")
        walk = Walk(path).play()
        applied = [
            frame for frame in walk.frames
            if any(e.owner_code == CALLED_BY and e.label_object_code == ASH
                   for e in frame.effects)
        ]
        self.assertTrue(applied, "Called by the Grave lost its selected card")
        effects = [
            e for e in applied[0].effects
            if e.owner_code == CALLED_BY and e.label_object_code == ASH
        ]
        self.assertEqual(len(effects), 2)
        for effect in effects:
            with self.subTest(code=effect.code):
                self.assertEqual(effect.label_object_type, 4)
                self.assertEqual(effect.label_object_location, C.LOCATION_REMOVED)
                self.assertEqual(effect.reset_count, 2)

    # -- 4. once-per-turn accounting --------------------------------------

    def test_second_copy_is_barred_while_the_first_copy_holds_the_count(self):
        path = _puzzle(
            self.tmp, "ash2",
            {0: [POT_OF_GREED, POT_OF_GREED], 1: [ASH, ASH]},
        )
        walk = Walk(path).play()
        self.assertEqual(walk.run_result.core_log, [])

        turn1 = [
            i for i, info in enumerate(walk.frames) if info.turn.turn_id == 1
        ]
        # player 0 really did activate both copies of Pot of Greed, so the
        # opportunity to chain the second Ash existed twice over
        activations = [
            i for i in turn1
            if walk.prompts[i] == C.MSG_SELECT_IDLECMD
            and "activate" in walk.chosen[i]
        ]
        self.assertEqual(len(activations), 2, walk.chosen)

        chains = [i for i in turn1 if walk.prompts[i] == C.MSG_SELECT_CHAIN]
        # exactly one window: the second activation opens none, because with the
        # row spent the core has nothing to offer player 1 at all
        self.assertEqual(len(chains), 1, [walk.menus[i] for i in chains])

        first = chains[0]
        self.assertEqual(walk.frames[first].count_used(1, ASH), 0)
        self.assertEqual(
            sum("activate" in entry for entry in walk.menus[first]),
            2,
            walk.menus[first],
        )

        after = [f for f in walk.frames[first + 1:] if f.count_used(1, ASH)]
        self.assertTrue(after, "the first copy never resolved")
        self.assertEqual(after[0].count_used(1, ASH), 1)
        self.assertEqual(after[0].effects, ())

        counts = {walk.frames[i].count_used(1, ASH) for i in turn1}
        self.assertEqual(counts, {0, 1}, "the row should never exceed one")

        # and the whole table is cleared at the next turn boundary
        turn2 = [f for f in walk.frames if f.turn.turn_id == 2]
        self.assertTrue(turn2)
        self.assertEqual(turn2[0].count_used(1, ASH), 0)

    # -- shape of the rest of the dump ------------------------------------

    def test_turn_and_activity_sections_track_the_duel(self):
        path = _puzzle(self.tmp, "turns", {0: [FLAMVELL_GUARD], 1: [MAXX_C]})
        walk = Walk(path).play()
        first = walk.frames[0]
        self.assertIsNotNone(first.turn)
        self.assertEqual(first.turn.turn_id, 1)
        self.assertEqual(first.turn.duel_rule, 5)
        self.assertEqual(len(first.turn.summon_count), 2)

        summoned = [
            f for f in walk.frames if f.turn.normalsummon_state_count[0]
        ]
        self.assertTrue(summoned, "player 0 never normal summoned")
        info = summoned[0]
        self.assertEqual(info.turn.turn_player, 0)
        self.assertEqual(info.turn.summon_count[0], 1)
        self.assertEqual(info.turn.summon_count[1], 0)

        # the tally is per turn, like the count table
        later = [f for f in walk.frames if f.turn.turn_id > info.turn.turn_id]
        self.assertTrue(later)
        self.assertEqual(later[0].turn.summon_count[0], 0)

    def test_dump_is_reproducible_for_the_same_duel(self):
        """Two runs of one puzzle produce byte-identical dumps.

        The tables behind this are ``unordered_map``s, so the writer sorts; if
        it stopped, this is where heap-address-dependent output would show up.
        """
        path = _puzzle(self.tmp, "repro", {0: [POT_OF_GREED], 1: [DROLL]})
        first = Walk(path).play()
        second = Walk(path).play()
        self.assertEqual(len(first.frames), len(second.frames))
        for a, b in zip(first.frames, second.frames):
            self.assertEqual(a.as_dict(), b.as_dict())

    def test_dump_does_not_disturb_the_duel(self):
        """Taking the dump at every decision must not change how a duel goes.

        A read-only API that is not actually read-only would show up as a
        different move log, not as an error.
        """
        path = _puzzle(self.tmp, "readonly", {0: [POT_OF_GREED], 1: [DROLL]})
        dumped = Walk(path).play()

        from mirrorforce.puzzle.single import SinglePuzzle

        quiet = SinglePuzzle(path, core=get_effectinfo_core(), options=MR5_OPTIONS)
        quiet.load()
        menus: list[list[str]] = []

        def policy(selector, actions, puzzle):
            menus.append([a.describe() for a in actions])
            return _prefer_activate(selector, actions, puzzle)

        try:
            quiet.play(policy=policy, max_steps=600)
        finally:
            quiet.close()
        self.assertEqual(dumped.menus, menus)

    def test_declared_record_widths_match_the_reader(self):
        """The widths the core writes are the widths the struct strings expect.

        The core measures each width from what it actually wrote, so this is
        the check that catches a field added on one side only.
        """
        import ctypes

        path = _puzzle(self.tmp, "widths", {0: [POT_OF_GREED], 1: [DROLL]})
        walk = Walk(path)
        core = walk.core
        walk.puzzle.start()
        buf = ctypes.create_string_buffer(effectinfo.QUERY_BUFFER_SIZE)
        try:
            core._lib.query_effect_info(walk.puzzle.pduel, buf, len(buf))
            raw = buf.raw[: struct.unpack_from("<I", buf.raw, 4)[0]]
        finally:
            walk.puzzle.close()

        expected = {
            effectinfo.SECTION_EFFECTS: effectinfo._EFFECT_EXT.size,
            effectinfo.SECTION_COUNT_CODE: effectinfo._COUNT.size,
            effectinfo.SECTION_SPSUMMON_ONCE: effectinfo._SPONCE.size,
            effectinfo.SECTION_ACTIVITY: effectinfo._ACTIVITY.size,
            effectinfo.SECTION_PLAYER_STATE: effectinfo._PLAYER.size,
            effectinfo.SECTION_CARD_STATE: effectinfo._CARD_STATE.size,
            effectinfo.SECTION_RELATIONS: effectinfo._RELATION.size,
            effectinfo.SECTION_SETCODE: effectinfo._SETCODE.size,
        }
        pos = raw[1]
        seen = {}
        while pos + effectinfo.SECTION.size <= len(raw):
            section_id, length = effectinfo.SECTION.unpack_from(raw, pos)
            pos += effectinfo.SECTION.size
            if section_id == effectinfo.SECTION_END:
                break
            if section_id in expected:
                count, width = struct.unpack_from("<HH", raw, pos)
                seen[section_id] = (count, width)
            pos += length
        self.assertEqual(sorted(seen), sorted(expected))
        for section_id, (count, width) in seen.items():
            with self.subTest(section=section_id):
                self.assertEqual(width, expected[section_id])
        # the effects section is non-empty here, so its width was measured
        # rather than taken from the header constant
        self.assertGreater(seen[effectinfo.SECTION_EFFECTS][0], 0)
        self.assertEqual(seen[effectinfo.SECTION_PLAYER_STATE][0], 2)
        self.assertGreater(seen[effectinfo.SECTION_CARD_STATE][0], 0)

    def test_buffer_too_small_is_reported_not_overrun(self):
        path = _puzzle(self.tmp, "small", {0: [POT_OF_GREED], 1: [DROLL]})
        from mirrorforce.puzzle.single import SinglePuzzle
        import ctypes

        core = get_effectinfo_core()
        puzzle = SinglePuzzle(path, core=core, options=MR5_OPTIONS)
        puzzle.load()
        puzzle.start()
        try:
            tiny = ctypes.create_string_buffer(4)
            with self.assertRaises(EffectInfoError):
                query_effect_info(core, puzzle.pduel, buf=tiny)
        finally:
            puzzle.close()


if __name__ == "__main__":
    unittest.main()
