"""Turning duels into labelled samples.

One duel produces three kinds of row.

**Menu rows.**  At every decision point -- an idle, battle or chain prompt --
the state is masked for the player being asked, the enumerator names every
candidate, and each candidate is labelled against the engine's menu.  This is
the free part: the engine tells the truth for a few hundred candidates at once,
and it costs one board query.

**Settlement rows, on trajectory.**  What the action that was actually taken did
to the board, read out of the message stream between this decision point and
the next one.

**Settlement rows, off trajectory.**  The same thing for actions that were *not*
taken.  A verified world model that only ever saw on-policy transitions still
plays badly, because search leaves the trajectory almost immediately and the
model degrades silently out there (arXiv 2607.14169).  So the generator forks
the duel at the decision point, plays a different legal action, and records
that settlement too.  A fork replays the recorded response prefix into a fresh
duel, which is exact rather than approximate -- see :mod:`.engine`.

Every forked settlement is taken with the opponent **forced to pass**, which is
what makes the next menu a function of our state and our action alone; whether
the opponent even got a response window is kept as its own label, because that
is a fact about the rules the model should learn rather than a nuisance.

Value supervision comes along for free: a decision point on the trajectory
knows how the duel ended, how many decisions were left, and which policy each
seat was playing.  Forked decision points have no ending -- they are a
counterfactual -- so their outcome column is null.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass, field

from ..netduel import constants as C
from ..netduel.actions import place_to_ls, spec_to_ls
from .candidates import (
    ActionKind,
    Candidate,
    engine_key,
    enumerate_candidates,
    match_menu,
)
from .cardtext import CardTextIndex
from .engine import (
    DuelConfig,
    DuelDeadlineExceeded,
    DuelDriver,
    DuelError,
    Prompt,
    StopDuel,
    SubSelectionUnresolved,
    pass_responder,
)
from .state import (
    DET_HIDDEN,
    StateSnapshot,
    capture,
    diff_snapshots,
    leaks,
    CardState,
    identity_visible,
    DisclosureLedger,
    mask_for,
    supervision_leaks,
    roundtrip_occupancy,
)

__all__ = [
    "GameResult",
    "GenConfig",
    "MenuSample",
    "NEG_CLASSES",
    "ReplayChoice",
    "SettleSample",
    "SubCandidate",
    "SubPrompt",
    "generate_game",
]

#: why an enumerated candidate the engine refused is *interesting*.  These are
#: proxies read off the game, not semantic judgements: the engine never says
#: why an action is illegal, so each class is defined by what else was true.
NEG_CLASSES = (
    "easy",         # never legal anywhere in this game for this card
    "opt_used",     # this player used (code, slot) earlier in this same turn
    "speed",        # legal at an idle/battle prompt this turn, refused in a chain window
    "timing",       # legal this game, but only in another phase
    "resource",     # legal this game, and this player has strictly fewer resources now
    "restriction",  # legal this turn at the same prompt kind and phase, resources intact
    "other_hard",   # legal somewhere in this game, none of the above
)
_NEG_INDEX = {name: i for i, name in enumerate(NEG_CLASSES)}


@dataclass
class GenConfig:
    """How much branching to buy."""

    #: fork at every ``branch_stride``-th decision point (1 = every one)
    branch_stride: int = 1
    #: how many *unchosen* actions to fork per branched decision point
    branches: int = 3
    #: also fork the action that was actually taken, so the on-trajectory
    #: settlement is available under the same forced-pass condition
    branch_chosen: bool = True
    #: skip branching once a duel has run this long (forks replay the prefix,
    #: so their cost grows with the decision index)
    max_branch_index: int = 100000
    max_steps: int = 40000
    #: Per-duel wall clock (seconds). None = no limit. Checked between steps; a timeout counts as a dropped duel.
    #: **It cannot stop an infinite loop inside a single `core.process()`**: then the interpreter never gets
    #: control, and only a process-level wall clock helps (a child-process timeout or an external watchdog).
    #:
    max_seconds: float | None = None
    #: cap on prompts a fork walks before giving up on reaching the next menu
    branch_steps: int = 4000
    #: enumerate activations of cards the *opponent* controls as well.  Rare
    #: but real: a Field Spell in the opponent's field zone can offer its effect
    #: to both players, and 0.13% of engine menu entries at corpus scale name an
    #: opponent-controlled card.  Leaving this off costs 38% fewer candidates
    #: and 100% coverage, which is not a trade the menu head can afford.
    include_opponent_candidates: bool = True
    #: use the standalone core with query_effect_info sections.  Formal v1
    #: generation enables this; old callers retain the cheap stock core.
    effectinfo: bool = False


@dataclass
class MenuSample:
    """One decision point and its labelled candidate set."""

    game_id: str
    decision_id: str
    parent_id: str | None
    source: str            # "trajectory" | "branch"
    forced_pass: bool
    player: int          #: the seat being asked, 0 or 1
    first_player: int    #: the seat that took turn 1
    on_play: bool        #: this decision point's player went first
    msg: int
    turn: int
    turn_player: int
    phase: int
    response_index: int
    decision_ordinal: int
    state: StateSnapshot
    candidates: list[Candidate]
    labels: list[bool]
    neg_class: list[int]
    #: per candidate: is this card code outside the PV-1 training deck pool?
    unseen: list[bool]
    #: the engine's full desc for each candidate (0 on a miss); used to look up effect identity in desc_map
    desc: list[int]
    chosen_index: int      # index into candidates, -1 when nothing was taken
    menu_size: int
    n_missed: int
    n_duplicate: int
    n_system_desc: int
    truncated: bool
    # value supervision; null on a fork
    outcome: int | None = None       # 1 win, 0 draw, -1 loss, from `player`
    steps_to_end: int | None = None
    policy_self: str = ""
    policy_opp: str = ""
    deck_self: str = ""
    deck_opp: str = ""
    banlist_id: int = 0
    banlist: tuple[tuple[int, int], ...] = ()


@dataclass
class SettleSample:
    """One action and what it did."""

    game_id: str
    decision_id: str
    settle_id: str
    source: str            # "trajectory" | "branch"
    on_trajectory: bool
    forced_pass: bool
    player: int
    on_play: bool
    action_key: tuple
    action_menu_index: int
    #: the engine's full desc of the chosen action
    action_desc: int
    #: code and absolute location of the card taking the action. The other seat masks this row by publicity when consuming it.
    action_code: int
    action_at: int
    #: sub-prompts within the action. The raw prompts belong only to the answering side; the other seat consumes only the
    #: PublicEvents the core broadcasts afterwards, not these private candidates directly.
    sub_prompts: list["SubPrompt"]
    #: sub-prompts within the action: (prompt type, number of options, **chosen index**, description of the chosen option).
    #: The chosen option is part of the resolution: the same "activate effect" with different targets resolves to different diffs,
    #: and without this column the resolution is not a function of (state, action).
    diff: object
    #: per move: whether its code is **visible from this sample's viewpoint**.
    #: `diff` is computed on the raw snapshot, and `mv_code` used to be written as is: a card the opponent searched from the deck to the hand
    #: without disclosing it got its identity into our viewpoint's supervision that way.
    #: This judges each move, and the writer writes 0 for invisible codes. **`diff` itself is not changed**,
    #: because the round-trip consistency check still needs it.
    mv_code_visible: list[bool] | None
    #: the legal action keys at this player's next decision point
    next_menu: list[tuple] | None
    next_msg: int
    next_player: int
    #: the opponent was offered a non-empty chain window inside the interval
    opponent_window: bool
    roundtrip_ok: bool
    reached_next: bool
    ended: bool
    #: chain probes: branches taken at the same chain prompt share this id
    lifo_group: str = ""


@dataclass(frozen=True)
class SubCandidate:
    """One option in a private sub-selection prompt.

    The answering player sees this list. It is stored so the world model can
    predict the list, never so another seat can consume the raw core prompt.
    The parent prompt's ``msg`` disambiguates ``value`` semantics.
    """

    code: int = 0
    at: int = 0
    value: int = 0
    desc: int = 0
    finish: bool = False


@dataclass(frozen=True)
class SubPrompt:
    """One selected option inside an action.

    ``stage`` separates activation-time cost/target choices from choices made
    while a chain link is resolving.  Both used to be emitted immediately
    after ``ACT``; that leaked a later link's resolution choice into every
    earlier link in the chain.  ``link`` is the active chain count, or zero for
    a non-chain procedure.
    """

    player: int
    msg: int
    n: int
    choice: int
    desc: str
    code: int = 0
    at: int = 0
    value: int = 0
    stage: int = 0  # 0 activation/procedure, 1 resolution
    link: int = 0
    #: Ordinal in the same interval message stream as PublicEvent.trace_index.
    trace_index: int = 0
    candidates: tuple[SubCandidate, ...] = ()
    truncated: bool = False


@dataclass(frozen=True)
class ReplayChoice:
    """One policy-visible choice round inside a response."""

    response_index: int
    round_index: int
    msg: int
    player: int
    choice: int
    action: str


@dataclass
class GameResult:
    game_id: str
    ok: bool
    error: str = ""
    winner: int | None = None
    first_player: int | None = None
    turns: int = 0
    menu: list[MenuSample] = field(default_factory=list)
    settle: list[SettleSample] = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)
    leak_problems: list[str] = field(default_factory=list)
    #: Lossless in-process replay capsule.  A standard .yrp is a client file;
    #: the label factory instead needs the exact DuelConfig plus raw response
    #: bytes consumed by DuelDriver.
    config: DuelConfig | None = None
    policies: tuple[str, str] = ("", "")
    responses: tuple[bytes, ...] = ()
    choices: tuple[ReplayChoice, ...] = ()


def _game_id(config: DuelConfig, policies: tuple[str, str]) -> str:
    payload = (
        f"{config.decks[0].name}|{config.decks[1].name}|{config.seed}|"
        f"{policies[0]}|{policies[1]}|{config.duel_options}"
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


#: Key prefix of the dropped-duel ledger. Dropped duels are a normal path, but the **distribution** matters far more than the total:
#: dropping one in ten thousand uniformly does not matter; dropping 100% of a scene of one card or one deck is a coverage hole dug out
#: silently: that position never enters the training set while the total drop rate still looks small.
DROP = "duel_dropped"


def _record_drop(stats, config, reason: str, msg: int | None = None, codes=()) -> None:
    """Book one dropped duel on four axes: (reason, deck, sub-prompt type, triggering card).

    `reason` is the kill_reason axis:
      * ``subselect``: sub-choices exceeded `max_sub_rounds` (the Python layer, a defensive upper bound);
      * ``deadline``: exceeded the per-duel wall clock `max_seconds` (checked between steps);
      * ``watchdog``: killed by the process-level wall clock, filled in by an external tool, not through here.

    Duels killed for an infinite loop in the core belong to ``watchdog``: then the interpreter never gets control and
    no Python code has any chance to run; only a watchdog outside the process can book it.
    """
    stats[DROP] = stats.get(DROP, 0) + 1
    stats[f"{DROP}:reason:{reason}"] = stats.get(f"{DROP}:reason:{reason}", 0) + 1
    for deck in config.decks:
        key = f"{DROP}:deck:{deck.name}"
        stats[key] = stats.get(key, 0) + 1
    if msg is not None:
        key = f"{DROP}:msg{msg}"
        stats[key] = stats.get(key, 0) + 1
    # Deduplicate before truncating: the ledger means "how many **duels** were dropped because of this card", not "how many times this card appeared".
    # Two copies of one code in the same prompt (two of the same card in hand) must not double this card's count.
    for code in sorted(set(codes))[:16]:  # truncated to keep keys from exploding
        key = f"{DROP}:card:{code}"
        stats[key] = stats.get(key, 0) + 1


def generate_game(
    config: DuelConfig,
    responders: tuple,
    policies: tuple[str, str],
    texts: CardTextIndex,
    gen: GenConfig | None = None,
    unseen_codes: frozenset[int] = frozenset(),
    check_leaks: bool = False,
) -> GameResult:
    """Play one duel and label everything it passes through."""
    gen = gen or GenConfig()
    game_id = _game_id(config, policies)
    result = GameResult(
        game_id=game_id, ok=False, config=config, policies=policies,
    )
    stats = result.stats

    # -- pass 1: the trajectory ------------------------------------------
    points: list[dict] = []
    effect_core = None
    if gen.effectinfo:
        from ..effectinfo import get_effectinfo_core
        effect_core = get_effectinfo_core()
    driver = DuelDriver(config, core=effect_core)
    driver.record_messages = True
    replay_choices: list[ReplayChoice] = []
    response_rounds: Counter = Counter()
    try:
        driver.build()

        def responder(prompt: Prompt, dd: DuelDriver) -> int:
            if prompt.is_decision:
                _open_point(points, prompt, dd, texts, gen, check_leaks, result)
            choice = int(responders[prompt.player](prompt, dd))
            round_index = int(response_rounds[prompt.index])
            response_rounds[prompt.index] += 1
            selected = prompt.actions[choice]
            describe = getattr(selected, "describe", None)
            replay_choices.append(ReplayChoice(
                response_index=int(prompt.index),
                round_index=round_index,
                msg=int(prompt.msg),
                player=int(prompt.player),
                choice=choice,
                action=describe() if callable(describe) else repr(selected),
            ))
            if points:
                last = points[-1]
                if prompt.is_decision and last["open"]:
                    last["chosen_menu"] = choice
                    last["chosen_key"] = engine_key(prompt.actions[choice])
                    last["chosen_desc"] = int(
                        getattr(prompt.actions[choice], "desc", 0) or 0
                    )
                    last["open"] = False
                else:
                    last["sub_prompts"].append(_sub_prompt(
                        prompt, choice, dd.messages[last["msg_start"]:],
                    ))
            return choice

        driver.run(responder, max_steps=gen.max_steps,
                   max_seconds=gen.max_seconds)
    except SubSelectionUnresolved as exc:
        result.error = str(exc)
        _record_drop(stats, config, "subselect", msg=exc.msg, codes=exc.codes)
    except DuelDeadlineExceeded as exc:
        result.error = str(exc)
        _record_drop(stats, config, "deadline")
    except DuelError as exc:
        result.error = str(exc)
    except StopDuel:
        result.error = "stopped"
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        result.error = f"{type(exc).__name__}: {exc}"

    final_snapshot = None
    try:
        if driver.pduel is not None:
            final_snapshot = _capture(driver)
    except Exception:  # noqa: BLE001 - a dead duel simply has no final board
        final_snapshot = None

    result.winner = driver.winner
    result.first_player = driver.first_player
    result.turns = driver.turn
    stats["responses"] = len(driver.responses)
    stats["fallbacks"] = driver.fallbacks
    stats["truncations"] = driver.truncations

    _close_intervals(points, driver, final_snapshot)

    n_points = len(points)
    for i, point in enumerate(points):
        point["ordinal"] = i
        point["steps_to_end"] = n_points - 1 - i

    # -- pass 2: hard-negative classes ------------------------------------
    _classify_negatives(points)

    # -- emit ---------------------------------------------------------------
    for point in points:
        result.menu.append(
            _menu_sample(
                point, game_id, config, policies, driver, unseen_codes,
                source="trajectory",
            )
        )
        settle = _settle_from_interval(
            point,
            game_id,
            source="trajectory",
            on_trajectory=True,
            forced_pass=False,
            settle_id=f"{game_id}:{point['ordinal']}:t",
            leak_sink=result.leak_problems if check_leaks else None,
            first_player=0 if driver.first_player is None else driver.first_player,
        )
        if settle is not None:
            # nobody had a window to pass on, so this transition is already the
            # one the forced-pass condition would have produced
            settle.forced_pass = not settle.opponent_window
            point["opponent_window"] = settle.opponent_window
            result.settle.append(settle)

    if result.error and not driver.finished:
        stats["incomplete"] = 1

    # -- pass 3: off-trajectory forks --------------------------------------
    try:
        _branch(points, config, driver, texts, gen, game_id, result, unseen_codes,
                policies, check_leaks)
    except Exception as exc:  # noqa: BLE001
        stats["branch_error"] = 1
        if not result.error:
            result.error = f"branch: {type(exc).__name__}: {exc}"
    finally:
        result.responses = tuple(bytes(response) for response in driver.responses)
        result.choices = tuple(replay_choices)
        driver.close()

    result.ok = not result.error or bool(result.menu)
    return result


def _pack_action_at(action, player: int) -> int:
    spec = getattr(action, "spec", "")
    place = int(getattr(action, "place", 0) or 0)
    if spec:
        opponent, location, sequence, _ = spec_to_ls(spec)
        controller = 1 - player if opponent else player
    elif place:
        controller, location, sequence = place_to_ls(place, player)
    else:
        return 0
    position = int(getattr(action, "position", 0) or 0)
    return (
        (controller & 0xFF)
        | ((location & 0xFF) << 8)
        | ((sequence & 0xFF) << 16)
        | ((position & 0xFF) << 24)
    )


def _sub_context(messages) -> tuple[int, int]:
    """Return ``(stage, link)`` at a sub-prompt from the public core trace.

    ``MSG_CHAINING`` is written before cost/target callbacks, whereas
    ``MSG_CHAIN_SOLVING`` is written immediately before the operation callback.
    The distinction is therefore engine truth, not a guess based on whether a
    script happened to call ``SelectTarget`` or ``SelectMatchingCard``.  In
    particular, an ordinary Fusion Summon performed by a Fusion effect belongs
    to that effect's resolving link; only a procedure such as contact Fusion
    has no resolving link and stays in the action/procedure stage.
    """
    building = 0
    solving = 0
    saw_solved = False
    for message in messages:
        body = message.payload
        if message.msg == C.MSG_CHAINING and len(body) >= 16:
            building = int(body[15])
            solving = 0
            saw_solved = False
        elif message.msg == C.MSG_CHAIN_SOLVING and body:
            solving = int(body[0])
        elif message.msg == C.MSG_CHAIN_SOLVED:
            saw_solved = True
            solving = 0
        elif message.msg == C.MSG_CHAIN_END:
            building = 0
            solving = 0
    if solving:
        return 1, solving
    if building:
        return 0, building
    # A delayed continuous operation can prompt after the normal chain was
    # solved. It has no chain number, but is still a resolution-time choice.
    if saw_solved:
        return 1, 0
    return 0, 0


def _sub_value(action) -> int:
    return (
        int(getattr(action, "position", 0) or 0)
        or int(getattr(action, "place", 0) or 0)
        or int(getattr(action, "number", 0) or 0)
        or int(getattr(action, "attribute", 0) or 0)
        or int(getattr(action, "race", 0) or 0)
        or int(getattr(action, "response", 0) or 0)
    )


def _sub_candidate(action, player: int) -> SubCandidate:
    return SubCandidate(
        code=int(getattr(action, "code", 0)
                 or getattr(action, "response", 0) or 0),
        at=_pack_action_at(action, player),
        value=_sub_value(action),
        desc=int(getattr(action, "desc", 0) or 0),
        finish=bool(getattr(action, "finish", False)),
    )


def _sub_prompt(prompt: Prompt, choice: int, messages=()) -> SubPrompt:
    action = prompt.actions[choice]
    value = _sub_value(action)
    code = int(getattr(action, "code", 0) or getattr(action, "response", 0) or 0)
    stage, link = _sub_context(messages)
    return SubPrompt(
        player=int(prompt.player),
        msg=int(prompt.msg),
        n=len(prompt.actions),
        choice=int(choice),
        desc=action.describe(),
        code=code,
        at=_pack_action_at(action, prompt.player),
        value=value,
        stage=stage,
        link=link,
        trace_index=max(len(messages) - 1, 0),
        candidates=tuple(_sub_candidate(a, prompt.player) for a in prompt.actions),
        truncated=bool(prompt.truncated),
    )


# -- trajectory bookkeeping ------------------------------------------------


def _miss_tag(key, view=None, player=0, action=None) -> str:
    """A short, groupable name for an engine entry no candidate covered.

    ``kind:spec-shape:slot:code`` -- enough to say *which corner* of the action
    space is missing, and which card, without keeping a per-card histogram
    alive for the millions of entries that are covered.
    """
    from ..netduel.actions import spec_to_ls

    kind, spec, slot = key[:3]
    shape = "".join(ch for ch in spec if not ch.isdigit()) or "-"
    source_code = 0
    if view is not None and spec:
        opponent, location, sequence, _ = spec_to_ls(spec)
        owner = (1 - player) if opponent else player
        for card in view.cards:
            if (
                card.controller == owner
                and card.location == location
                and card.sequence == sequence
            ):
                source_code = card.code
                break
    action_code = int(getattr(action, "code", 0) or 0)
    desc = int(getattr(action, "desc", 0) or 0)
    return f"{kind}:{shape}:{slot}:{source_code}:{action_code}:{desc}"


def _miss_action(key, actions):
    """Return the prompt entry behind a missed key for diagnostics."""
    from .candidates import engine_key

    for action in actions:
        if engine_key(action) == key:
            return action
    return None


def _disclosure_of(driver) -> DisclosureLedger:
    """The driver's public identity ledger; an old object without this attribute counts as "nothing disclosed"."""
    ledger = getattr(driver, "disclosure", None)
    return ledger if ledger is not None else DisclosureLedger()


def _open_point(points, prompt, driver, texts, gen, check_leaks, result) -> None:
    snapshot = _capture(driver, prompt.player)
    # Disclosed cards (activated, revealed) have identities visible to both sides; see mask_for.
    # The multiset ledger must be resolved into instance coordinates against the **current** board: sequences are
    # shifted silently by `reset_sequence`, and comparing with coordinates recorded at activation would anonymize disclosed cards again.
    revealed = _disclosure_of(driver).resolve(snapshot, prompt.player)
    view = mask_for(
        snapshot, prompt.player, revealed,
        unpositioned=_disclosure_of(driver).unanchored_identities(
            prompt.player, revealed))
    if check_leaks:
        problems = leaks(view)
        if problems:
            result.leak_problems.extend(problems)
    candidates = enumerate_candidates(
        view, prompt.player, texts, include_opponent=gen.include_opponent_candidates
    )
    match = match_menu(candidates, prompt.actions)
    points.append(
        {
            "open": True,
            "prompt_msg": prompt.msg,
            "player": prompt.player,
            "turn": prompt.turn,
            "turn_player": prompt.turn_player,
            "phase": prompt.phase,
            "response_index": prompt.index,
            "truncated": prompt.truncated,
            "snapshot": snapshot,
            "view": view,
            "candidates": candidates,
            "match": match,
            "actions": list(prompt.actions),
            "msg_start": len(driver.messages),
            "chosen_menu": -1,
            "chosen_key": None,
            "chosen_desc": 0,
            "sub_prompts": [],
        }
    )
    for key, desc in match.missed:
        action = _miss_action(key, prompt.actions)
        result.stats[
            f"missed:{_miss_tag(key, view, prompt.player, action)}"
        ] += 1
    result.stats["missed"] += len(match.missed)
    result.stats["menu_entries"] += len(prompt.actions)
    result.stats["candidates"] += len(candidates)


def _capture(driver, viewer: int | None = None) -> StateSnapshot:
    """Capture the board plus exact core state when the loaded core supports it."""
    core_info = None
    from ..effectinfo import query_effect_info, supports
    if supports(driver.core):
        legacy = query_effect_info(driver.core, driver.pduel)
        core_info = legacy
        if viewer in (0, 1):
            from ..duelstate import (
                FLAG_TARGET_ENTITY_IDS,
                as_training_targets,
                query_duel_state,
                supports as supports_duel_state,
            )
            if supports_duel_state(driver.core):
                scoped = query_duel_state(
                    driver.core, driver.pduel, int(viewer),
                    flags=FLAG_TARGET_ENTITY_IDS,
                )
                core_info = as_training_targets(scoped, legacy.as_dict())
    return capture(driver, core_info=core_info)


def _close_intervals(points, driver, final_snapshot) -> None:
    """Attach the message interval and the after-snapshot of each action."""
    messages = driver.messages
    for i, point in enumerate(points):
        start = point["msg_start"]
        end = points[i + 1]["msg_start"] if i + 1 < len(points) else len(messages)
        point["messages"] = messages[start:end]
        point["next_point"] = points[i + 1] if i + 1 < len(points) else None
        if point["next_point"] is None:
            point["final_snapshot"] = final_snapshot


def _classify_negatives(points) -> None:
    """Label each negative candidate with why it is (or is not) interesting.

    The engine never explains a refusal, so the classes are read off the rest of
    the game: an action this card never got to make anywhere is easy, one it
    made under other circumstances is hard, and the circumstance that differs
    names the class.
    """
    # where each (kind, code, slot, location) was legal, per player
    legal_at: dict[tuple, list[int]] = {}
    for i, point in enumerate(points):
        player = point["player"]
        for candidate, label in zip(point["candidates"], point["match"].labels):
            if label and candidate.kind not in _GLOBALS:
                legal_at.setdefault(
                    _hard_key(player, candidate), []
                ).append(i)

    # (player, turn, code, slot) actually activated
    used: set[tuple] = set()
    for point in points:
        key = point.get("chosen_key")
        if key and key[0] == ActionKind.ACTIVATE_EFFECT.value:
            index = point["match"].menu_index
            for candidate, mi in zip(point["candidates"], index):
                if mi == point["chosen_menu"]:
                    used.add(
                        (point["player"], point["turn"], candidate.code, candidate.eff_slot)
                    )
                    break

    for i, point in enumerate(points):
        player = point["player"]
        resources = _resources(point["snapshot"], player)
        classes = []
        for candidate, label in zip(point["candidates"], point["match"].labels):
            if label or candidate.kind in _GLOBALS:
                classes.append(_NEG_INDEX["easy"])
                continue
            where = legal_at.get(_hard_key(player, candidate))
            if not where:
                classes.append(_NEG_INDEX["easy"])
                continue
            classes.append(
                _NEG_INDEX[_hard_class(i, point, where, points, candidate, used, resources)]
            )
        point["neg_class"] = classes


_GLOBALS = frozenset(
    {
        ActionKind.TO_BATTLE,
        ActionKind.TO_MAIN2,
        ActionKind.TO_END,
        ActionKind.CHAIN_PASS,
    }
)


def _hard_key(player: int, candidate: Candidate) -> tuple:
    return (player, candidate.kind.value, candidate.code, candidate.eff_slot,
            candidate.location)


def _resources(snapshot: StateSnapshot, player: int) -> tuple:
    counts = snapshot.counts
    return (
        counts.get((player, C.LOCATION_HAND), 0),
        counts.get((player, C.LOCATION_MZONE), 0),
        counts.get((player, C.LOCATION_SZONE), 0),
        counts.get((player, C.LOCATION_GRAVE), 0),
        snapshot.lp[player],
    )


def _hard_class(i, point, where, points, candidate, used, resources) -> str:
    player, turn, phase, msg = (
        point["player"], point["turn"], point["phase"], point["prompt_msg"],
    )
    if (
        candidate.kind is ActionKind.ACTIVATE_EFFECT
        and (player, turn, candidate.code, candidate.eff_slot) in used
    ):
        return "opt_used"
    same_turn = [j for j in where if points[j]["turn"] == turn]
    if msg == C.MSG_SELECT_CHAIN and any(
        points[j]["prompt_msg"] != C.MSG_SELECT_CHAIN for j in same_turn
    ):
        return "speed"
    if same_turn and any(
        points[j]["prompt_msg"] == msg and points[j]["phase"] == phase
        for j in same_turn
    ):
        richer = any(
            _ge(_resources(points[j]["snapshot"], player), resources)
            and _resources(points[j]["snapshot"], player) != resources
            for j in same_turn
        )
        return "resource" if richer else "restriction"
    if any(points[j]["phase"] != phase for j in where):
        return "timing"
    if any(
        _ge(_resources(points[j]["snapshot"], player), resources)
        and _resources(points[j]["snapshot"], player) != resources
        for j in where
    ):
        return "resource"
    return "other_hard"


def _ge(a: tuple, b: tuple) -> bool:
    return all(x >= y for x, y in zip(a, b))


def _menu_sample(
    point, game_id, config, policies, driver, unseen_codes, source,
    parent_id=None, forced_pass=False, decision_id=None,
) -> MenuSample:
    player = point["player"]
    first_player = 0 if driver.first_player is None else driver.first_player
    match = point["match"]
    chosen_index = -1
    if point["chosen_menu"] >= 0:
        for idx, mi in enumerate(match.menu_index):
            if mi == point["chosen_menu"]:
                chosen_index = idx
                break
    outcome = None
    steps_to_end = None
    if source == "trajectory" and driver.winner is not None:
        outcome = 1 if driver.winner == player else (-1 if driver.winner in (0, 1) else 0)
        steps_to_end = point.get("steps_to_end")
    did = decision_id or f"{game_id}:{point['ordinal']}"
    return MenuSample(
        game_id=game_id,
        decision_id=did,
        parent_id=parent_id,
        source=source,
        forced_pass=forced_pass,
        player=player,
        first_player=first_player,
        on_play=player == first_player,
        msg=point["prompt_msg"],
        turn=point["turn"],
        turn_player=point["turn_player"],
        phase=point["phase"],
        response_index=point["response_index"],
        decision_ordinal=point.get("ordinal", -1),
        state=point["view"],
        candidates=point["candidates"],
        labels=match.labels,
        neg_class=point.get("neg_class") or [0] * len(point["candidates"]),
        unseen=[c.code in unseen_codes for c in point["candidates"]],
        desc=list(getattr(match, "desc", []) or [0] * len(point["candidates"])),
        chosen_index=chosen_index,
        menu_size=len(point["actions"]),
        n_missed=len(match.missed),
        n_duplicate=match.duplicate,
        n_system_desc=match.system_desc,
        truncated=point["truncated"],
        outcome=outcome,
        steps_to_end=steps_to_end,
        policy_self=policies[player],
        policy_opp=policies[1 - player],
        deck_self=config.decks[player].name,
        deck_opp=config.decks[1 - player].name,
        banlist_id=int(getattr(config, "banlist_id", 0)),
        banlist=tuple(
            sorted((int(code), int(limit)) for code, limit in
                   getattr(config, "banlist", ()) if 0 <= int(limit) < 3)
        ),
    )


# -- settlements -----------------------------------------------------------


def _move_code_visible(mv, player: int, revealed) -> bool:
    """Whether the code of this move should appear in `player`'s viewpoint supervision.

    Visible at **either** end is enough: played from the hand to the field, the identity is public at the destination.
    Invisible at both ends (an undisclosed search from the opponent's deck to the hand) would be a leak.
    Shares `identity_visible` with `mask_for`, so there is only one copy of the visibility rules.
    """
    if not mv.code:
        return True
    def _at(ctrl, loc, seq, pos=0):
        return identity_visible(
            CardState(controller=ctrl, location=loc, sequence=seq,
                      code=mv.code, position=pos), player, revealed)
    return (_at(mv.from_controller, mv.from_location, mv.from_sequence)
            or _at(mv.to_controller, mv.to_location, mv.to_sequence,
                   mv.to_position))


def _settle_from_interval(point, game_id, source, on_trajectory, forced_pass,
                          settle_id, first_player=0, lifo_group="",
                          leak_sink=None) -> SettleSample | None:
    after_point = point.get("next_point")
    snapshot_before = point["snapshot"]
    if after_point is not None:
        snapshot_after = after_point["snapshot"]
    else:
        snapshot_after = point.get("final_snapshot")
    if snapshot_after is None:
        return None
    messages = point.get("messages") or []
    diff = diff_snapshots(snapshot_before, snapshot_after, messages)
    # A prompt answered by the other seat is hidden at this action player's
    # pre-disclosure information set.  The responder's own view will receive
    # SUB and become conditioned; the action-player view remains DET_HIDDEN
    # until a public confirmation/result token arrives.
    if any(
            int(prompt.player) != int(point["player"])
            for prompt in (point.get("sub_prompts") or ())):
        diff.determinism = max(int(diff.determinism), DET_HIDDEN)
    # The loss mask and the observation mask share one source: resolution deltas are computed on the **raw snapshot**, observations are masked.
    # The same predicate as `mask_for` rechecks the supervision targets here, and anything out of bounds goes into `leak_problems`:
    # a leak into the loss is as much a leak as one into the observation, and they share the same existing gate (that count must be 0).
    if leak_sink is not None:
        leak_sink.extend(
            f"settle {settle_id}: {p}"
            # Use the public set **at the end of the interval**: the "disclosure" of a search effect happens **within** the interval,
            # and exempting with the set from before would always miss it and report legal public information as a leak.
            # Concretely, "disclosure is input" here means: judge visibility by the state after the disclosure.
            for p in supervision_leaks(
                diff, point["player"],
                snapshot_after.revealed | snapshot_before.revealed,
            )
        )
    seen = snapshot_after.revealed | snapshot_before.revealed
    mv_visible = [
        _move_code_visible(mv, point["player"], seen) for mv in diff.moves
    ]
    predicted = roundtrip_occupancy(snapshot_before, diff)
    roundtrip_ok = all(
        predicted.get(key, 0) == snapshot_after.counts.get(key, 0)
        for key in set(predicted) | set(snapshot_after.counts)
    )
    next_menu = None
    next_msg = 0
    next_player = -1
    if after_point is not None:
        next_player = after_point["player"]
        if next_player == point["player"]:
            next_menu = [engine_key(a) for a in after_point["actions"]]
            next_msg = after_point["prompt_msg"]
    chosen_menu = int(point.get("chosen_menu", -1))
    chosen_action = (
        point["actions"][chosen_menu]
        if 0 <= chosen_menu < len(point.get("actions") or ())
        else None
    )
    return SettleSample(
        mv_code_visible=mv_visible,
        game_id=game_id,
        decision_id=f"{game_id}:{point.get('ordinal', -1)}",
        settle_id=settle_id,
        source=source,
        on_trajectory=on_trajectory,
        forced_pass=forced_pass,
        player=point["player"],
        on_play=point["player"] == first_player,
        action_key=point.get("chosen_key") or (),
        action_menu_index=point.get("chosen_menu", -1),
        action_desc=int(point.get("chosen_desc") or 0),
        action_code=int(getattr(chosen_action, "code", 0) or 0),
        action_at=(
            _pack_action_at(chosen_action, point["player"])
            if chosen_action is not None else 0
        ),
        sub_prompts=point.get("sub_prompts") or [],
        diff=diff,
        next_menu=next_menu,
        next_msg=next_msg,
        next_player=next_player,
        opponent_window=_opponent_window(messages, point["player"]),
        roundtrip_ok=roundtrip_ok,
        reached_next=after_point is not None,
        ended=after_point is None,
        lifo_group=lifo_group,
    )


def _opponent_window(messages, player: int) -> bool:
    """Did the other side get a chain window with something in it?

    Read straight off the message stream: a ``MSG_SELECT_CHAIN`` addressed to
    the opponent with at least one entry means the rules opened a response
    timing there.  Whether the opponent *used* it is a different question and
    belongs to the belief model, not here.
    """
    for message in messages:
        if message.msg == C.MSG_SELECT_CHAIN and len(message.payload) >= 3:
            if message.payload[0] != player and message.payload[1] > 0:
                return True
    return False


# -- branching -------------------------------------------------------------


def _branch(points, config, driver, texts, gen, game_id, result, unseen_codes,
            policies, check_leaks) -> None:
    """Fork each sampled decision point onto actions the trajectory skipped."""
    rng = random.Random(("branch", config.seed).__repr__())
    for point in points:
        ordinal = point["ordinal"]
        if ordinal % gen.branch_stride:
            continue
        if point["response_index"] > gen.max_branch_index:
            break
        actions = point["actions"]
        if len(actions) < 2 and not gen.branch_chosen:
            continue
        alternatives = [i for i in range(len(actions)) if i != point["chosen_menu"]]
        picked = _stratified_sample(actions, alternatives, gen.branches, rng)
        lifo_group = (
            f"{game_id}:{ordinal}"
            if point["prompt_msg"] == C.MSG_SELECT_CHAIN
            and sum(1 for a in actions if a.act.name != "CANCEL") >= 2
            else ""
        )
        # Re-taking the chosen action is only worth a fork when the trajectory
        # let the opponent respond (so the forced-pass version is a different
        # transition) or when it is the reference arm of a chain-order probe.
        if (
            gen.branch_chosen
            and point["chosen_menu"] >= 0
            and (point.get("opponent_window") or lifo_group)
        ):
            picked = [point["chosen_menu"]] + picked
        for choice in picked:
            try:
                sample = _run_branch(
                    point, choice, config, driver, texts, gen, game_id, result,
                    unseen_codes, policies, check_leaks, lifo_group, rng,
                )
            except Exception:  # noqa: BLE001 - one bad fork must not cost the game
                result.stats["branch_dropped"] += 1
                continue
            if sample is not None:
                result.stats["branches"] += 1


def _stratified_sample(actions, alternatives, k, rng) -> list[int]:
    """Pick ``k`` alternatives, spreading them over distinct action types.

    A menu is usually several instances of the same kind plus one or two
    others; sampling uniformly would spend the whole budget on repositions.
    """
    if not alternatives or k <= 0:
        return []
    buckets: dict[str, list[int]] = {}
    for i in alternatives:
        buckets.setdefault(engine_key(actions[i])[0], []).append(i)
    for group in buckets.values():
        rng.shuffle(group)
    order = sorted(buckets)
    rng.shuffle(order)
    out: list[int] = []
    while len(out) < k and any(buckets[key] for key in order):
        for key in order:
            if buckets[key]:
                out.append(buckets[key].pop())
                if len(out) >= k:
                    break
    return out


def _run_branch(point, choice, config, driver, texts, gen, game_id, result,
                unseen_codes, policies, check_leaks, lifo_group, rng):
    """Replay to the decision point, take ``choice``, read the consequence."""
    player = point["player"]
    branch = driver.fork(point["response_index"])
    branch.record_messages = True
    try:
        return _branch_body(
            point, choice, config, driver, branch, texts, gen, game_id, result,
            unseen_codes, policies, check_leaks, lifo_group, player,
        )
    finally:
        branch.close()


def _branch_body(point, choice, config, driver, branch, texts, gen, game_id,
                 result, unseen_codes, policies, check_leaks, lifo_group, player):
    """The fork itself; ``_run_branch`` owns closing it."""
    state = {
        "answered": False,
        "sub": [],
        "after": None,
        "next_prompt": None,
        "msg_start": 0,
    }
    tail = random.Random(("tail", config.seed, point["ordinal"], choice).__repr__())

    def inner(prompt: Prompt, dd: DuelDriver) -> int:
        if not state["answered"]:
            if not prompt.is_decision or prompt.player != player:
                # the fork landed on a different prompt than the trajectory did;
                # the prefix replay is exact, so this cannot happen -- guard
                # anyway rather than silently mislabel
                raise DuelError("fork diverged before the decision point")
            state["answered"] = True
            state["msg_start"] = len(dd.messages)
            return choice
        # The interval ends at the next decision point, whichever side it
        # belongs to.  ``pass_responder`` has already absorbed the opponent's
        # optional chain windows, so anything reaching here from the opponent
        # is either a forced chain (answered, not a handover) or a real
        # handover of control.
        if prompt.is_decision and (
            prompt.player == player or prompt.msg != C.MSG_SELECT_CHAIN
        ):
            state["next_prompt"] = prompt
            state["after"] = _capture(dd, prompt.player)
            raise StopDuel
        # Choose first, then record: the old version recorded ``actions[0]`` but returned a random index, so the record did not match the real answer
        pick = tail.randrange(len(prompt.actions))
        state["sub"].append(_sub_prompt(
            prompt, pick, dd.messages[state["msg_start"]:],
        ))
        return pick

    responder = pass_responder(1 - player, inner)
    ended = False
    try:
        branch.run(responder, max_steps=gen.branch_steps)
        ended = True
        if state["after"] is None and state["answered"]:
            state["after"] = _capture(branch, player)
    except StopDuel:
        pass
    except DuelError:
        result.stats["branch_dropped"] += 1
        return None
    if not state["answered"] or state["after"] is None:
        result.stats["branch_dropped"] += 1
        return None

    messages = branch.messages[state["msg_start"]:]
    next_prompt = state["next_prompt"]
    fake_next = None
    if next_prompt is not None:
        fake_next = {
            "snapshot": state["after"],
            "player": next_prompt.player,
            "actions": list(next_prompt.actions),
            "prompt_msg": next_prompt.msg,
        }
    branch_point = dict(point)
    branch_point["messages"] = messages
    branch_point["next_point"] = fake_next
    branch_point["final_snapshot"] = state["after"]
    branch_point["chosen_menu"] = choice
    branch_point["chosen_key"] = engine_key(point["actions"][choice])
    branch_point["chosen_desc"] = int(
        getattr(point["actions"][choice], "desc", 0) or 0
    )
    branch_point["sub_prompts"] = state["sub"]

    settle = _settle_from_interval(
        branch_point,
        game_id,
        source="branch",
        on_trajectory=(choice == point["chosen_menu"]),
        forced_pass=True,
        settle_id=f"{game_id}:{point['ordinal']}:b{choice}",
        first_player=0 if driver.first_player is None else driver.first_player,
        lifo_group=lifo_group,
        leak_sink=result.leak_problems if check_leaks else None,
    )
    if settle is not None:
        settle.ended = next_prompt is None and ended
        result.settle.append(settle)

    # the state the fork landed in is itself a decision point worth labelling,
    # and it is by construction off the trajectory
    if next_prompt is not None:
        revealed = _disclosure_of(driver).resolve(
            state["after"], next_prompt.player
        )
        view = mask_for(state["after"], next_prompt.player, revealed)
        if check_leaks:
            result.leak_problems.extend(leaks(view))
        candidates = enumerate_candidates(
            view, next_prompt.player, texts,
            include_opponent=gen.include_opponent_candidates,
        )
        match = match_menu(candidates, next_prompt.actions)
        result.stats["missed"] += len(match.missed)
        result.stats["menu_entries"] += len(next_prompt.actions)
        result.stats["candidates"] += len(candidates)
        for key, desc in match.missed:
            action = _miss_action(key, next_prompt.actions)
            result.stats[
                f"missed:{_miss_tag(key, view, next_prompt.player, action)}"
            ] += 1
        child = {
            "player": next_prompt.player,
            "prompt_msg": next_prompt.msg,
            "turn": next_prompt.turn,
            "turn_player": next_prompt.turn_player,
            "phase": next_prompt.phase,
            "response_index": next_prompt.index,
            "truncated": next_prompt.truncated,
            "snapshot": state["after"],
            "view": view,
            "candidates": candidates,
            "match": match,
            "actions": list(next_prompt.actions),
            "chosen_menu": -1,
            "ordinal": point["ordinal"],
            "neg_class": [0] * len(candidates),
        }
        result.menu.append(
            _menu_sample(
                child, game_id, config, policies, driver, unseen_codes,
                source="branch",
                parent_id=f"{game_id}:{point['ordinal']}",
                forced_pass=True,
                decision_id=f"{game_id}:{point['ordinal']}:b{choice}",
            )
        )
    return settle
