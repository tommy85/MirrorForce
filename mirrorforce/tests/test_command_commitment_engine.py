"""Committed command selections on a real core.

A constructed battle: our 3000-ATK monster may attack the opponent's 2500-ATK
monster.  Historically the attack target selection offers a cancel that puts
the core back at the identical battle command; opted in, the selection offers
the targets only and choosing one declares the attack.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirrorforce.effectinfo import DEFAULT_EFFECTINFO_LIB, get_effectinfo_core  # noqa: E402
from mirrorforce.netduel import constants as C  # noqa: E402
from mirrorforce.netduel.actions import ActionAct, ActionPhase  # noqa: E402
from mirrorforce.puzzle.core import DEFAULT_DB, DEFAULT_SCRIPTS  # noqa: E402
from mirrorforce.search.rebuild import DUEL_ATTACK_FIRST_TURN, BoardSpec, Placement, rebuild  # noqa: E402
from mirrorforce.worldmodel.engine import DeckList, DuelConfig, StopDuel  # noqa: E402

ENGINE_READY = Path(DEFAULT_EFFECTINFO_LIB).is_file() and Path(DEFAULT_DB).is_file() and Path(DEFAULT_SCRIPTS).is_dir()
BLUE_EYES, DARK_MAGICIAN = 89631139, 46986414


def _battle(commit):
    placements = [
        Placement(code=BLUE_EYES, owner=0, controller=0, location=C.LOCATION_MZONE, sequence=0,
                  position=C.POS_FACEUP_ATTACK, proc=True),
        Placement(code=DARK_MAGICIAN, owner=1, controller=1, location=C.LOCATION_MZONE, sequence=0,
                  position=C.POS_FACEUP_ATTACK, proc=True),
    ]
    placements += [Placement(code=DARK_MAGICIAN, owner=player, controller=player, location=C.LOCATION_DECK,
                             sequence=i, position=C.POS_FACEDOWN_DEFENSE) for player in (0, 1) for i in range(5)]
    spec = BoardSpec(lp=(8000, 8000), duel_rule=5, duel_options=DUEL_ATTACK_FIRST_TURN,
                     placements=tuple(placements), target_phase=C.PHASE_MAIN1)
    empty = DeckList(name="battle", main=(), extra=())
    config = DuelConfig(decks=(empty, empty), seed=20260917, start_lp=8000, start_hand=0, draw_count=1,
                        duel_options=(5 << 16) | DUEL_ATTACK_FIRST_TURN, full_phase_menu=True,
                        auto_end_phase_discard=False, commit_command_selections=commit)
    return rebuild(spec, seed=20260917, config=config, core=get_effectinfo_core())


def _walk(commit):
    """Enter battle, attack, cancel the target once if offered, then attack for real."""
    prompts, cancelled = [], False

    def responder(prompt, driver):
        nonlocal cancelled
        prompts.append((prompt.msg, [action.describe() for action in prompt.actions]))
        acts = [action.act for action in prompt.actions]
        if prompt.player != 0:
            raise StopDuel
        if prompt.msg == C.MSG_SELECT_IDLECMD:
            return [action.phase for action in prompt.actions].index(ActionPhase.BATTLE)
        if prompt.msg == C.MSG_SELECT_BATTLECMD and ActionAct.ATTACK in acts:
            return acts.index(ActionAct.ATTACK)
        if prompt.msg == C.MSG_SELECT_CARD:
            if ActionAct.CANCEL in acts and not cancelled:
                cancelled = True
                return acts.index(ActionAct.CANCEL)
            return acts.index(ActionAct.NONE)
        raise StopDuel

    duel = _battle(commit)
    duel.record_messages = True
    try:
        duel.run(responder, max_steps=20000, max_seconds=30.0)
    except StopDuel:
        pass
    finally:
        declared = [message for message in duel.messages if message.msg == C.MSG_ATTACK]
        duel.close()
    return prompts, declared


@unittest.skipUnless(ENGINE_READY, "needs the effectinfo core, card database and scripts")
class CommandCommitmentEngineTest(unittest.TestCase):
    def test_historical_cancel_returns_to_the_identical_battle_command(self):
        prompts, declared = _walk(commit=False)
        battles = [actions for msg, actions in prompts if msg == C.MSG_SELECT_BATTLECMD]
        targets = [actions for msg, actions in prompts if msg == C.MSG_SELECT_CARD]
        self.assertEqual(targets[0][-1], "cancel")
        self.assertEqual(battles[0], battles[1])
        self.assertEqual(len(targets), 2)
        self.assertEqual(len(declared), 1)

    def test_committed_target_selection_declares_the_attack(self):
        prompts, declared = _walk(commit=True)
        targets = [actions for msg, actions in prompts if msg == C.MSG_SELECT_CARD]
        self.assertEqual(len(targets), 1)
        self.assertNotIn("cancel", targets[0])
        self.assertEqual(len(declared), 1)


if __name__ == "__main__":
    unittest.main()
