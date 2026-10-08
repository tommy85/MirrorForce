"""Viewer isolation and wire tests for ``query_duel_state`` v2."""

from __future__ import annotations

import ctypes
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirrorforce import duelstate  # noqa: E402
from mirrorforce.effectinfo import (  # noqa: E402
    DEFAULT_EFFECTINFO_LIB,
    get_effectinfo_core,
)
from mirrorforce.netduel import constants as C  # noqa: E402
from mirrorforce.puzzle.core import DEFAULT_DB, DEFAULT_SCRIPTS  # noqa: E402


ASH = 14558127
DROLL = 94145021
MAXX_C = 23434538
IMPERMANENCE = 10045474
XYZ_HOST = 88120966
XYZ_MATERIAL_1 = 92418590
XYZ_MATERIAL_2 = 34620088
SOFT_OPT_CARD = 96501677

ENGINE_READY = (
    Path(DEFAULT_EFFECTINFO_LIB).is_file()
    and Path(DEFAULT_DB).is_file()
    and Path(DEFAULT_SCRIPTS).is_dir()
)


class WireFormatTest(unittest.TestCase):
    def test_record_widths(self):
        self.assertEqual(duelstate._META.size, 64)
        self.assertEqual(duelstate._PLAYER.size, 64)
        self.assertEqual(duelstate._CARD.size, 124)
        self.assertEqual(duelstate._RELATION.size, 32)
        self.assertEqual(duelstate._EFFECT.size, 128)
        self.assertEqual(duelstate._EFFECT_OBJECT.size, 16)

    def test_unknown_section_is_retained(self):
        payload = b"future"
        body = duelstate.SECTION.pack(99, len(payload)) + payload
        body += duelstate.SECTION.pack(duelstate.SECTION_END, 0)
        raw = duelstate.HEADER.pack(
            duelstate.VERSION, duelstate.HEADER.size, 0,
            duelstate.HEADER.size + len(body),
        ) + body
        self.assertEqual(duelstate.parse(raw).unknown_sections, {99: payload})

    def test_short_record_is_rejected(self):
        payload = struct.pack("<HH", 1, duelstate._CARD.size - 4)
        payload += bytes(duelstate._CARD.size - 4)
        body = duelstate.SECTION.pack(duelstate.SECTION_CARD, len(payload))
        body += payload + duelstate.SECTION.pack(duelstate.SECTION_END, 0)
        raw = duelstate.HEADER.pack(
            duelstate.VERSION, duelstate.HEADER.size, 0,
            duelstate.HEADER.size + len(body),
        ) + body
        with self.assertRaises(duelstate.DuelStateError):
            duelstate.parse(raw)


@unittest.skipUnless(ENGINE_READY, "needs the stateapi-enabled core build")
class ViewerStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(__file__).resolve().parent / "_duelstate_tmp"
        cls.tmp.mkdir(exist_ok=True)
        cls.path = cls.tmp / "viewer.lua"
        cls.path.write_text(
            f'''\
Debug.SetAIName("duelstate")
Debug.SetPlayerInfo(0,7000,0,0)
Debug.SetPlayerInfo(1,6500,0,0)
Debug.AddCard({DROLL},0,0,LOCATION_DECK,0,POS_FACEDOWN_DEFENSE)
Debug.AddCard({ASH},0,0,LOCATION_DECK,1,POS_FACEDOWN_DEFENSE)
Debug.AddCard({MAXX_C},0,0,LOCATION_HAND,0,POS_FACEDOWN)
Debug.AddCard({IMPERMANENCE},1,1,LOCATION_HAND,0,POS_FACEDOWN)
Debug.AddCard({ASH},1,1,LOCATION_GRAVE,0,POS_FACEUP_ATTACK)
Debug.AddCard({DROLL},1,1,LOCATION_SZONE,0,POS_FACEDOWN)
Debug.AddCard({XYZ_HOST},1,1,LOCATION_MZONE,1,POS_FACEUP_ATTACK)
Debug.AddCard({XYZ_MATERIAL_1},1,1,LOCATION_MZONE,1,POS_FACEUP_ATTACK)
Debug.AddCard({XYZ_MATERIAL_2},1,1,LOCATION_MZONE,1,POS_FACEUP_ATTACK)
Debug.AddCard({MAXX_C},1,1,LOCATION_DECK,0,POS_FACEDOWN_DEFENSE)
Debug.ReloadFieldEnd()
''',
            encoding="utf8",
        )

    @classmethod
    def tearDownClass(cls):
        for path in cls.tmp.glob("*.lua"):
            path.unlink()
        cls.tmp.rmdir()

    def setUp(self):
        from mirrorforce.puzzle.single import SinglePuzzle

        self.puzzle = SinglePuzzle(self.path, core=get_effectinfo_core(), options=5 << 16)
        result = self.puzzle.load()
        self.assertTrue(result.loaded)
        self.assertEqual(result.missing_codes, [])
        self.assertEqual(result.core_log, [])

    def tearDown(self):
        self.puzzle.close()

    def query(self, view):
        return duelstate.query_duel_state(
            self.puzzle.core, self.puzzle.pduel, view,
        )

    def test_player_views_do_not_receive_opponent_hidden_zones(self):
        p0 = self.query(duelstate.VIEW_PLAYER_0)
        p1 = self.query(duelstate.VIEW_PLAYER_1)
        p0_codes = {card.printed_code for card in p0.cards if card.printed_code}
        p1_codes = {card.printed_code for card in p1.cards if card.printed_code}
        self.assertIn(MAXX_C, p0_codes, "player 0 must see their own hand")
        self.assertNotIn(IMPERMANENCE, p0_codes,
                         "player 0 must not see player 1's hidden hand")
        self.assertIn(IMPERMANENCE, p1_codes)
        self.assertNotIn(MAXX_C, {
            card.printed_code for card in p1.cards
            if card.location == C.LOCATION_HAND
        })

    def test_face_down_field_is_positional_but_anonymous(self):
        p0 = self.query(duelstate.VIEW_PLAYER_0)
        rows = [
            card for card in p0.cards
            if card.controller == 1 and card.location == C.LOCATION_SZONE
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].printed_code, 0)
        self.assertTrue(rows[0].anonymous)
        self.assertEqual(rows[0].sequence, 0)

    def test_own_deck_is_a_canonical_multiset_not_engine_order(self):
        p0 = self.query(duelstate.VIEW_PLAYER_0)
        deck = [
            card for card in p0.cards
            if card.controller == 0 and card.location == C.LOCATION_DECK
        ]
        self.assertEqual([card.printed_code for card in deck], sorted([ASH, DROLL]))
        self.assertEqual([card.sequence for card in deck], [0, 1])
        self.assertTrue(all(card.identity_visible for card in deck))
        self.assertTrue(all(not (card.flags & duelstate.CARD_SEQUENCE_VISIBLE)
                            for card in deck))

    def test_revealed_cross_owner_card_in_deck_has_a_strict_sort_order(self):
        """The deck comparator must not mix a-side code and b-side sequence."""
        from mirrorforce.puzzle.single import SinglePuzzle

        path = self.tmp / "mixed-owner-deck.lua"
        path.write_text(
            f'''\
Debug.SetAIName("mixed-owner-deck")
Debug.SetPlayerInfo(0,8000,0,0)
Debug.SetPlayerInfo(1,8000,0,0)
Debug.AddCard({ASH},0,0,LOCATION_DECK,1,POS_FACEDOWN_DEFENSE)
Debug.AddCard({DROLL},1,0,LOCATION_DECK,0,POS_FACEUP_ATTACK)
Debug.AddCard({MAXX_C},1,1,LOCATION_DECK,0,POS_FACEDOWN_DEFENSE)
Debug.ReloadFieldEnd()
''',
            encoding="utf8",
        )
        puzzle = SinglePuzzle(path, core=get_effectinfo_core(), options=5 << 16)
        try:
            self.assertTrue(puzzle.load().loaded)
            for _ in range(32):
                frame = duelstate.query_duel_state(
                    puzzle.core, puzzle.pduel, duelstate.VIEW_PLAYER_0,
                )
            deck = [
                card for card in frame.cards
                if card.controller == 0 and card.location == C.LOCATION_DECK
            ]
            self.assertEqual(
                [(card.owner, card.printed_code) for card in deck],
                [(0, ASH), (1, DROLL)],
            )
        finally:
            puzzle.close()

    def test_xyz_materials_are_public_and_point_to_the_host(self):
        shared = self.query(duelstate.VIEW_SHARED_PUBLIC)
        host = next(card for card in shared.cards if card.printed_code == XYZ_HOST)
        materials = [card for card in shared.cards if card.overlay]
        self.assertEqual(
            [card.printed_code for card in materials],
            [XYZ_MATERIAL_1, XYZ_MATERIAL_2],
        )
        self.assertTrue(all(card.host_ref == host.ref for card in materials))
        overlay_edges = [
            edge for edge in shared.relations
            if edge.kind == 2 and edge.source_ref == host.ref
        ]
        self.assertEqual(
            [edge.target_ref for edge in overlay_edges],
            [card.ref for card in materials],
        )
        targets = duelstate.as_training_targets(shared)
        translated = [
            edge for edge in targets["relations"]
            if edge["kind"] == 2 and edge["source_ref"] == host.ref
        ]
        self.assertEqual(
            [edge["target_ref"] for edge in translated],
            [card.ref for card in materials],
        )
        self.assertTrue(all(edge["source_code"] == XYZ_HOST for edge in translated))

    def test_hidden_card_initial_effect_does_not_leak_through_field_watchers(self):
        shared = self.query(duelstate.VIEW_SHARED_PUBLIC)
        visible = {card.ref: card for card in shared.cards if card.identity_visible}
        self.assertTrue(all(effect.source_ref in visible for effect in shared.effects))
        self.assertNotIn(
            DROLL,
            {visible[effect.source_ref].printed_code for effect in shared.effects},
            "a hidden Deck card must not leak through its global initial watcher",
        )
        full = self.query(duelstate.VIEW_OMNISCIENT_LABEL)
        full_deck_droll = next(
            card for card in full.cards
            if card.controller == 0 and card.location == C.LOCATION_DECK
            and card.printed_code == DROLL
        )
        self.assertIn(
            full_deck_droll.ref,
            {effect.source_ref for effect in full.effects},
            "the omniscient label view retains the watcher for diagnostics",
        )

    def test_full_view_is_a_strict_superset_and_query_is_reproducible(self):
        public = self.query(duelstate.VIEW_SHARED_PUBLIC)
        full = self.query(duelstate.VIEW_OMNISCIENT_LABEL)
        self.assertGreater(len(full.cards), len(public.cards))
        self.assertEqual(full, self.query(duelstate.VIEW_OMNISCIENT_LABEL))
        self.assertEqual(tuple(p.lp for p in full.players), (7000, 6500))

    def test_stable_entity_ids_are_target_only_and_opt_in(self):
        public = self.query(duelstate.VIEW_PLAYER_0)
        labelled = duelstate.query_duel_state(
            self.puzzle.core, self.puzzle.pduel,
            duelstate.VIEW_PLAYER_0,
            flags=duelstate.FLAG_TARGET_ENTITY_IDS,
        )
        self.assertTrue(all(card.entity_id == 0 for card in public.cards))
        self.assertTrue(all(card.entity_id > 0 for card in labelled.cards))
        self.assertEqual(
            len({card.entity_id for card in labelled.cards}), len(labelled.cards)
        )
        self.assertTrue(all(effect.entity_id == 0 for effect in public.effects))
        self.assertTrue(all(effect.entity_id > 0 for effect in labelled.effects))
        targets = duelstate.as_training_targets(labelled)
        self.assertTrue(all(row["entity_id"] for row in targets["cards"]))
        self.assertTrue(all(
            row["source_entity_id"] and row["target_entity_id"]
            for row in targets["relations"]
        ))

    def test_invalid_view_and_short_buffer_fail_loudly(self):
        with self.assertRaises(ValueError):
            self.query(99)
        self.assertTrue(duelstate.declare(self.puzzle.core._lib))
        tiny = ctypes.create_string_buffer(8)
        length = self.puzzle.core._lib.query_duel_state(
            self.puzzle.pduel, duelstate.VIEW_PLAYER_0, 0, tiny, len(tiny)
        )
        self.assertEqual(length, 0)


@unittest.skipUnless(ENGINE_READY, "needs the stateapi-enabled core build")
class RuntimeEffectStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(__file__).resolve().parent / "_duelstate_effect_tmp"
        cls.tmp.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        for path in cls.tmp.glob("*.lua"):
            path.unlink()
        cls.tmp.rmdir()

    def _path(self, name: str, setup: str) -> Path:
        path = self.tmp / f"{name}.lua"
        path.write_text(
            f'''\
Debug.SetAIName("duelstate-effect")
Debug.SetPlayerInfo(0,8000,0,0)
Debug.SetPlayerInfo(1,8000,0,0)
{setup}
for i=1,10 do
  Debug.AddCard(21615956,0,0,LOCATION_DECK,0,POS_FACEDOWN_DEFENSE)
  Debug.AddCard(21615956,1,1,LOCATION_DECK,0,POS_FACEDOWN_DEFENSE)
end
Debug.ReloadFieldEnd()
''',
            encoding="utf8",
        )
        return path

    def _frames(self, path: Path):
        from mirrorforce.puzzle.single import SinglePuzzle

        puzzle = SinglePuzzle(path, core=get_effectinfo_core(), options=5 << 16)
        self.assertTrue(puzzle.load().loaded)
        frames = []

        def policy(selector, actions, live):
            frames.append(duelstate.query_duel_state(
                live.core, live.pduel, duelstate.VIEW_PLAYER_0,
            ))
            for index, action in enumerate(actions):
                if "activate" in action.describe():
                    return index
            return 0

        try:
            puzzle.play(policy=policy, max_steps=600)
        finally:
            puzzle.close()
        return frames

    def test_impermanence_effects_point_to_the_target_handler(self):
        path = self._path(
            "imperm",
            f'''Debug.AddCard({IMPERMANENCE},0,0,LOCATION_HAND,0,POS_FACEDOWN)
Debug.AddCard({ASH},1,1,LOCATION_MZONE,0,POS_FACEUP_ATTACK)''',
        )
        frames = self._frames(path)
        hits = []
        for frame in frames:
            cards = {card.ref: card for card in frame.cards}
            for effect in frame.effects:
                source = cards.get(effect.source_ref)
                if (source and source.printed_code == IMPERMANENCE
                        and effect.code in (2, 8)):
                    hits.append((frame, effect))
        self.assertTrue(hits)
        frame = hits[0][0]
        cards = {card.ref: card for card in frame.cards}
        effects = [effect for seen, effect in hits if seen is frame]
        self.assertEqual(sorted(effect.code for effect in effects), [2, 8])
        self.assertTrue(all(cards[effect.handler_ref].printed_code == ASH
                            for effect in effects))
        self.assertTrue(all(effect.reset_count == 1 for effect in effects))

    def test_called_by_keeps_its_public_card_object_and_two_turn_reset(self):
        called_by = 24224830
        path = self._path(
            "called-by",
            f'''Debug.AddCard({called_by},0,0,LOCATION_HAND,0,POS_FACEDOWN)
Debug.AddCard({ASH},1,1,LOCATION_GRAVE,0,POS_FACEUP_ATTACK)''',
        )
        frames = self._frames(path)
        for frame in frames:
            cards = {card.ref: card for card in frame.cards}
            source_effects = [
                effect for effect in frame.effects
                if effect.source_ref in cards
                and cards[effect.source_ref].printed_code == called_by
                and effect.reset_count == 2
            ]
            if not source_effects:
                continue
            objects = {
                obj.effect_ref: cards[obj.card_ref]
                for obj in frame.effect_objects if obj.card_ref in cards
            }
            self.assertEqual(len(source_effects), 2)
            self.assertTrue(all(effect.reset_count == 2
                                for effect in source_effects))
            self.assertTrue(all(objects[effect.ref].printed_code == ASH
                                for effect in source_effects))
            return
        self.fail("Called by the Grave never appeared in the effect state")

    def test_consumed_initial_effect_count_is_exported_without_static_flood(self):
        path = self._path(
            "initial-count",
            f'''Debug.AddCard({SOFT_OPT_CARD},0,0,LOCATION_MZONE,0,POS_FACEUP_ATTACK)
Debug.AddCard({ASH},1,1,LOCATION_MZONE,0,POS_FACEUP_ATTACK)''',
        )
        frames = self._frames(path)
        for frame in frames:
            cards = {card.ref: card for card in frame.cards}
            rows = [
                effect for effect in frame.effects
                if effect.source_ref in cards
                and cards[effect.source_ref].printed_code == SOFT_OPT_CARD
                and effect.flag & 0x1  # EFFECT_FLAG_INITIAL
                and effect.count_limit_max == 1
                and effect.count_limit == 0
            ]
            if rows:
                self.assertLessEqual(len(rows), 1)
                return
        self.fail("consumed INITIAL count_limit never entered the effect state")


if __name__ == "__main__":
    unittest.main()
