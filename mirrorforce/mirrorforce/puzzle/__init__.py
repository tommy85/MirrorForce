"""Headless ygopro single-mode (puzzle) support.

``mirrorforce.puzzle`` loads a ``Debug``-API board script into the pinned
ygopro-core and plays it without a client, a host or a deck.  It exists to
answer one question first -- does a community puzzle still load against our
2026 snapshot? -- and to be the shape the ygoenv board-setup entry point takes
afterwards.
"""

from .core import DEFAULT_PUZZLE_DIR, Core, CoreError, get_core
from .messages import Message, UnknownMessage, split_messages
from .single import (
    Decision,
    FieldInfo,
    PuzzleError,
    PuzzleRun,
    SinglePuzzle,
    ZoneCard,
    first_policy,
    random_policy,
    scripted_policy,
    solution_policy,
)

__all__ = [
    "Core",
    "CoreError",
    "DEFAULT_PUZZLE_DIR",
    "Decision",
    "FieldInfo",
    "Message",
    "PuzzleError",
    "PuzzleRun",
    "SinglePuzzle",
    "UnknownMessage",
    "ZoneCard",
    "first_policy",
    "get_core",
    "random_policy",
    "scripted_policy",
    "solution_policy",
    "split_messages",
]
