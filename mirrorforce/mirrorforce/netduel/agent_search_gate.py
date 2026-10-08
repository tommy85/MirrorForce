"""Opt-in search admission from an unchanged legal policy and public hints.

SearchPolicy installs this only through an explicit opt-in. It neither runs a model nor
changes candidates, particles, rules, or responses. Public lethal hints are
deliberately overinclusive triggers, never a proof that a win is available.

The caller must begin a prompt before preparation/the original policy RPC and
finish it only after response delivery and owned-root cleanup. Every prompt,
including skips/failures, consumes the turn's wall-time allowance. Peer wait
between prompts is separate from this client-work ledger; complete turn wall
time must still be audited from public NEW_TURN/WIN wire events.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics

LAW = "mirrorforce_on_demand_search_gate/v1"
NO_TURN_CAP_LAW = "mirrorforce_on_demand_search_gate/v2-no-fixed-turn-cap"
TURN_WORK450_LAW = "mirrorforce_on_demand_search_gate/v3-turn-work450"
ROOT_SCHEMA = "mirrorforce_on_demand_prior_fallback/v1"
PUBLIC_HINT_LAW = "public-combat-potential-trigger-not-proof/v1"
_COMBAT_ACTIONS = frozenset({"attack", "battle", "summon", "special_summon", "activate"})
PUBLIC_TACTICAL_CANDIDATES = frozenset({
    "borrelsword-double-attack-defense-body", "bomber-burn-or-clear",
})


def _number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or positive and value == 0:
        raise ValueError(f"invalid {name}")
    return float(value)


@dataclass(frozen=True)
class GateConfig:
    """Provisional limits, not a production registration or measured latency.

    The existing three-second finalization reserve is preserved. These eight/
    twelve/ thirty-second caps await a new runtime measurement; none is a promise
    that a complete fair stripe fits after preparation and cleanup.
    """
    enabled: bool = False
    entropy_threshold: float = .6
    margin_threshold: float = .2
    uncertain_seconds: float = 8.
    lethal_seconds: float = 12.
    turn_seconds: float | None = 30.
    finalize_seconds: float = 3.
    low_lp_threshold: int = 3000

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be explicit bool")
        for name in ("entropy_threshold", "margin_threshold"):
            if not 0 <= _number(getattr(self, name), name) <= 1:
                raise ValueError("confidence thresholds must lie in [0, 1]")
        for name in ("uncertain_seconds", "lethal_seconds", "finalize_seconds"):
            _number(getattr(self, name), name, positive=True)
        if self.turn_seconds is not None:
            _number(self.turn_seconds, 'turn_seconds', positive=True)
        if self.lethal_seconds < self.uncertain_seconds or (
                self.turn_seconds is not None and self.finalize_seconds >= self.turn_seconds):
            raise ValueError("invalid trigger caps or finalization reserve")
        if type(self.low_lp_threshold) is not int or self.low_lp_threshold <= 0:
            raise ValueError("invalid public low-LP threshold")


@dataclass(frozen=True)
class Confidence:
    rows: int
    raw_greedy_index: int
    normalized_entropy: float
    top_probability: float
    top_two_margin: float


def policy_confidence(logits, *, rows):
    """Read EVERY original legal logit; probabilities are not win estimates."""
    if type(rows) is not int or rows < 1 or not isinstance(logits, (list, tuple)) or len(logits) != rows \
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in logits):
        raise ValueError("confidence requires the complete finite legal menu")
    best = max(range(rows), key=logits.__getitem__)
    if rows == 1:
        return Confidence(1, best, 0., 1., 1.)
    exponentials = [math.exp(x - logits[best]) for x in logits]
    total = math.fsum(exponentials)
    probabilities = [x / total for x in exponentials]
    entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0) / math.log(rows)
    first, second = sorted(probabilities, reverse=True)[:2]
    return Confidence(rows, best, min(1., max(0., entropy)), first, first - second)


@dataclass(frozen=True)
class PublicCombatHint:
    """Caller extracts only its public board, own legal menu, and public LP.

    Attack potential ignores blockers, attack counts, immunity and effects on
    purpose. It is a reason to inspect a position, never executable damage.
    Hidden cards, private core fields, and hypothetical identities have no
    fields in this DTO. The trusted caller still owns provenance validation.
    """
    our_turn: bool
    battle_window: bool
    opponent_lp: int
    visible_own_attacks: tuple[int, ...]
    legal_action_kinds: frozenset[str]
    tactical_candidates: frozenset[str] = frozenset()

    def __post_init__(self):
        if type(self.our_turn) is not bool or type(self.battle_window) is not bool \
                or type(self.opponent_lp) is not int \
                or type(self.visible_own_attacks) is not tuple \
                or any(type(x) is not int or x < 0 for x in self.visible_own_attacks) \
                or type(self.legal_action_kinds) is not frozenset \
                or not self.legal_action_kinds <= _COMBAT_ACTIONS \
                or type(self.tactical_candidates) is not frozenset \
                or not self.tactical_candidates <= PUBLIC_TACTICAL_CANDIDATES:
            raise ValueError("invalid public combat hint")


def lethal_candidate(hint, *, low_lp_threshold):
    if type(hint) is not PublicCombatHint or type(low_lp_threshold) is not int or low_lp_threshold <= 0:
        raise ValueError("a declared public combat hint is required")
    if hint.opponent_lp <= 0:
        return False
    # These flags must be produced by a separately reviewed public rule
    # detector. No detector/teacher/tactical-search implementation is imported
    # here. Bomber burn/clear need not be in a battle window or our own turn.
    if hint.tactical_candidates:
        return True
    if not hint.our_turn or not hint.battle_window or not hint.legal_action_kinds:
        return False
    return sum(hint.visible_own_attacks) >= hint.opponent_lp or hint.opponent_lp <= low_lp_threshold


@dataclass(frozen=True)
class GateDecision:
    reason: str
    search: bool
    confidence: Confidence
    lethal_candidate: bool
    search_deadline: float
    response_deadline: float
    spent_this_turn: float
    trigger: str


class TurnSearchGate:
    """Serial prompt ledger with non-extending absolute deadlines.

    This cannot preempt work or make transport failures safe. Its caller must
    obey the returned deadlines and retain the existing exact root cleanup,
    native error, response, and full-fair-stripe checks. An overrun is recorded
    as an overrun, including on a failed prompt; it is never a successful win.
    """
    def __init__(self, config=GateConfig()):
        if type(config) is not GateConfig:
            raise ValueError("an explicit gate config is required")
        self.config = config
        self.turn = None
        self.spent = 0.
        self.last_stamp = 0.
        self.prompt = None
        self.failed = False
        self.search_stopped = False

    def _clock(self, stamp):
        stamp = _number(stamp, "monotonic timestamp")
        if stamp < self.last_stamp:
            raise ValueError("gate clock moved backwards")
        return stamp

    def begin_prompt(self, *, turn, started, search_deadline, response_deadline, optional_response_deadline=None):
        started = self._clock(started)
        search_deadline = _number(search_deadline, "public search deadline")
        response_deadline = _number(response_deadline, "public response deadline")
        if type(turn) is not int or turn < 1 or self.prompt is not None or self.failed \
                or self.turn is not None and turn < self.turn \
                or search_deadline > response_deadline or response_deadline <= started:
            raise ValueError("invalid prompt lifecycle or public deadlines")
        if turn != self.turn:
            self.turn, self.spent = turn, 0.
            self.search_stopped = False
        self.prompt = {"started": started, "search": search_deadline, "response": response_deadline,
                       "trigger_cap": None, "admitted_response": response_deadline}
        if optional_response_deadline is not None:
            optional_response_deadline = _number(optional_response_deadline, 'optional response deadline')
            if optional_response_deadline > response_deadline:
                raise ValueError('optional work cannot extend the ordinary response deadline')
            self.prompt['optional_response'] = optional_response_deadline
        self.last_stamp = started

    def decide(self, *, logits, rows, hint, now, opponent_clear=None, opponent_known=None):
        now = self._clock(now)
        if self.prompt is None:
            raise ValueError("gate decision without an original prompt")
        confidence = policy_confidence(logits, rows=rows)
        blocked = None
        if opponent_clear is not None:
            from . import agent_opponent_clear as OC
            OC.check(opponent_clear)
            if not opponent_clear['allow_search']:
                blocked = opponent_clear['reason']
        if opponent_known is not None:
            from . import agent_opponent_known as OK
            if opponent_clear is not None:
                raise ValueError('opponent public-information identities are mutually exclusive')
            OK.check(opponent_known)
            if not opponent_known['allow_search']:
                blocked = opponent_known['reason']
        lethal = False if hint is None and blocked is not None else \
            lethal_candidate(hint, low_lp_threshold=self.config.low_lp_threshold)
        elapsed = self.spent + now - self.prompt["started"]
        uncertain = confidence.normalized_entropy >= self.config.entropy_threshold \
            or confidence.top_two_margin <= self.config.margin_threshold
        trigger = "lethal_candidate" if lethal else "uncertain" if uncertain else "none"
        reason = "disabled" if not self.config.enabled else "singleton" if rows == 1 else \
            blocked if blocked is not None else "previous_overrun" if self.search_stopped else \
            trigger if trigger != "none" else "confident"
        search = reason in ("lethal_candidate", "uncertain")
        hard, soft = self.prompt["response"], self.prompt["search"]
        if self.config.enabled:
            # A skipped original RPC still consumes this turn. Exhaustion
            # disables optional search; it does not authorize dropping a legal
            # response or shortening its already registered transport reserve.
            turn_end = None if self.config.turn_seconds is None else \
                self.prompt["started"] + max(0., self.config.turn_seconds - self.spent)
            if search:
                cap = self.config.lethal_seconds if lethal else self.config.uncertain_seconds
                previous = self.prompt["trigger_cap"]
                self.prompt["trigger_cap"] = cap if previous is None else min(previous, cap)
                hard = min(hard, self.prompt["started"] + self.prompt["trigger_cap"],
                           self.prompt.get('optional_response', hard))
                if turn_end is not None:
                    hard = min(hard, turn_end)
                soft = min(soft, hard - self.config.finalize_seconds)
                self.prompt["search"] = soft
                if soft <= now:
                    search, reason = False, "no_search_budget"
                    # Use the original safe response deadline for fallback.
                    hard = self.prompt["response"]
                else:
                    self.prompt["admitted_response"] = min(self.prompt["admitted_response"], hard)
        self.last_stamp = now
        return GateDecision(reason, search, confidence, lethal, now if not search else soft,
                            hard, elapsed, trigger)

    def finish_prompt(self, *, now, delivered, cleanup_complete):
        now = self._clock(now)
        if self.prompt is None or type(delivered) is not bool or type(cleanup_complete) is not bool:
            raise ValueError("invalid prompt finish")
        elapsed = now - self.prompt["started"]
        self.spent += elapsed
        result = {"law": gate_law(self.config), "turn": self.turn, "prompt_seconds": elapsed,
                  "started": self.prompt["started"], "finished": now,
                  "turn_work_seconds": self.spent, "delivered": delivered, "cleanup_complete": cleanup_complete,
                  "over_turn_cap": self.config.turn_seconds is not None and self.spent > self.config.turn_seconds,
                  "over_response_deadline": now > self.prompt["response"],
                  "over_admitted_prompt_cap": now > self.prompt["admitted_response"]}
        self.failed = not delivered or not cleanup_complete
        self.search_stopped |= result['over_turn_cap'] or result['over_response_deadline'] \
            or result['over_admitted_prompt_cap']
        self.prompt = None
        self.last_stamp = now
        return result


def public_combat_witness(client, parsed):
    """Extract only wire-maintained public field cards and the received menu.

    No core, World, hypothesis, hand, deck order, teacher or learned value is
    accepted. Hidden field cards are skipped BEFORE reading their identity or
    stats. Visible face-up cards on either side may supply a defense body.
    """
    from .board import ShadowBoard
    from . import constants as C
    if type(client.board) is not ShadowBoard:
        raise ValueError("search gate requires the wire ShadowBoard")
    board, seat = client.board, client.result.our_player
    if type(seat) is not int or seat not in (0, 1) or board.our_player != seat:
        raise ValueError("search gate public viewer differs from the live client")
    cards = []
    for player in (0, 1):
        for card in board.zone(player, C.LOCATION_MZONE):
            if card is None or card.hidden or not card.position & C.POS_FACEUP:
                continue
            if card.code <= 0:
                continue
            cards.append([player, int(card.sequence), int(card.code), int(card.type),
                          max(0, int(card.attack)), int(card.position)])
    actions = [] if parsed.selector is None else [
        [int(action.act), int(action.phase), int(action.code)] for action in parsed.selector.options()]
    result = {"law": PUBLIC_HINT_LAW, "viewer": seat, "turn_player": board.turn_player,
              "phase": board.phase, "opponent_lp": client._lp[1 - seat],
              "visible_monsters": sorted(cards), "legal_actions": actions}
    hint_from_witness(result)
    return result


def hint_from_witness(witness):
    """Recompute conservative triggers; identities/stats are NOT damage proof.

    Coverage includes an already visible/legally summonable Borrelsword plus
    a visible attack-position non-Link body, and an already visible/legally
    summonable Bomber with an activation/summon/attack opportunity. It misses
    deeper combos whose setup is not represented by these public conditions.
    """
    from . import constants as C
    from .actions import ActionAct as A, ActionPhase as P
    if not isinstance(witness, dict) or set(witness) != {
            "law", "viewer", "turn_player", "phase", "opponent_lp", "visible_monsters", "legal_actions"} \
            or witness['law'] != PUBLIC_HINT_LAW:
        raise ValueError("unknown/private public combat witness fields")
    for name in ('viewer', 'turn_player'):
        if type(witness[name]) is not int or witness[name] not in (0, 1):
            raise ValueError("invalid public combat player")
    if type(witness['phase']) is not int or type(witness['opponent_lp']) is not int \
            or not isinstance(witness['visible_monsters'], list) or not isinstance(witness['legal_actions'], list):
        raise ValueError("invalid public combat clock or rows")
    cards, actions, seat = witness['visible_monsters'], witness['legal_actions'], witness['viewer']
    places = set()
    for row in cards:
        if not isinstance(row, list) or len(row) != 6 or any(type(x) is not int for x in row) \
                or row[0] not in (0, 1) or not 0 <= row[1] < 7 or row[2] <= 0 or row[3] < 0 or row[4] < 0 \
                or row[5] not in (C.POS_FACEUP_ATTACK, C.POS_FACEUP_DEFENSE) or tuple(row[:2]) in places:
            raise ValueError("combat witness contains a concealed/ambiguous monster")
        places.add(tuple(row[:2]))
    kinds = set()
    for row in actions:
        if not isinstance(row, list) or len(row) != 3 or any(type(x) is not int for x in row) \
                or row[0] not in set(A) or row[1] not in set(P) or row[2] < 0:
            raise ValueError("invalid public legal action hint")
        kind = {A.ATTACK: 'attack', A.DIRECT_ATTACK: 'attack', A.SUMMON: 'summon',
                A.SPSUMMON: 'special_summon', A.ACTIVATE: 'activate'}.get(row[0])
        if kind:
            kinds.add(kind)
        if row[1] == P.BATTLE:
            kinds.add('battle')
    own_codes = {r[2] for r in cards if r[0] == seat}
    summonable = {r[2] for r in actions if r[0] == A.SPSUMMON}
    candidates = set()
    if 85289965 in own_codes | summonable and any(
            r[3] > 0 and not r[3] & C.TYPE_LINK and r[5] == C.POS_FACEUP_ATTACK for r in cards):
        candidates.add('borrelsword-double-attack-defense-body')
    if 5821478 in own_codes | summonable and kinds & {'activate', 'special_summon', 'attack'}:
        candidates.add('bomber-burn-or-clear')
    battle = witness['phase'] in (C.PHASE_MAIN1, C.PHASE_BATTLE_START, C.PHASE_BATTLE_STEP,
                                  C.PHASE_DAMAGE, C.PHASE_DAMAGE_CAL, C.PHASE_BATTLE)
    return PublicCombatHint(witness['turn_player'] == seat, battle, witness['opponent_lp'],
        tuple(r[4] for r in cards if r[0] == seat and r[5] == C.POS_FACEUP_ATTACK),
        frozenset(kinds), frozenset(candidates))


def decision_wire(decision):
    if type(decision) is not GateDecision:
        raise ValueError("expected computed gate decision")
    return asdict(decision)


def gate_law(config):
    if config.turn_seconds is None:
        return NO_TURN_CAP_LAW
    return TURN_WORK450_LAW if config.turn_seconds == 450. else LAW


def summary(roots, *, config=None):
    """All original prompts, admitted searches, explicit skips and failures."""
    reasons, by_turn = {}, {}
    counts = {name: 0 for name in ('non_singleton_decisions', 'heuristic_trigger_decisions',
        'admitted_search_decisions', 'positive_search_decisions', 'triggered_zero_stripe_decisions',
        'missing_triggered_search_reports', 'explicit_skip_decisions', 'missing_gate_traces',
        'unfinished_prompt_ledgers', 'prompt_cap_overruns')}
    times, optional, non_search, empty_optional = [], 0., 0., 0.
    for root in roots:
        trace = root.get('on_demand')
        if not isinstance(trace, dict):
            counts['missing_gate_traces'] += 1
            continue
        optional += trace.get('optional_seconds', 0.)
        finish = trace.get('finish')
        if finish is None:
            counts['unfinished_prompt_ledgers'] += 1
        else:
            times.append(finish['prompt_seconds'])
            non_search += trace['non_search_seconds']
            key = (root['turn'], root['viewer'])
            by_turn[key] = max(by_turn.get(key, 0.), finish['turn_work_seconds'])
            counts['prompt_cap_overruns'] += finish['over_admitted_prompt_cap'] or finish['over_response_deadline']
        searches = root.get('searches', [])
        for index, event in enumerate(trace.get('decisions', [])):
            gate = event['gate']
            if gate['confidence']['rows'] <= 1:
                continue
            counts['non_singleton_decisions'] += 1
            counts['heuristic_trigger_decisions'] += gate['trigger'] != 'none'
            reasons[gate['reason']] = reasons.get(gate['reason'], 0) + 1
            if gate['search']:
                counts['admitted_search_decisions'] += 1
                search = searches[index] if index < len(searches) else None
                positive = search is not None and search.get('anytime', {}).get('completed_stripes', 0) > 0
                counts['positive_search_decisions'] += positive
                counts['triggered_zero_stripe_decisions'] += not positive
                counts['missing_triggered_search_reports'] += search is None
                if not positive:
                    empty_optional += event.get('optional_seconds', 0.)
            else:
                counts['explicit_skip_decisions'] += 1
    total, admitted, positive = (counts[k] for k in (
        'non_singleton_decisions', 'admitted_search_decisions', 'positive_search_decisions'))
    return {'law': LAW if config is None else gate_law(config), 'prompts': len(roots), **counts, 'reasons': reasons,
        'admission_fraction_all_non_singleton': admitted / total if total else None,
        'positive_fraction_all_non_singleton': positive / total if total else None,
        'positive_fraction_admitted': positive / admitted if admitted else None,
        'zero_stripe_fraction_admitted': (admitted - positive) / admitted if admitted else None,
        'optional_work_seconds': optional, 'zero_stripe_optional_seconds': empty_optional,
        'non_search_work_seconds': non_search,
        'prompt_wall_seconds': {'mean': statistics.mean(times), 'median': statistics.median(times),
            'max': max(times), 'sum': math.fsum(times)} if times else None,
        'turn_accounted_work_seconds': [{'turn': t, 'viewer': v, 'seconds': seconds}
            for (t, v), seconds in sorted(by_turn.items())],
        'budget_scope': ('server-floor-admission-includes-base-work;no-fixed-turn-work-cap'
                         if config is not None and config.turn_seconds is None else
                         'conservative-admission-includes-base-work;not-hard-turn-wall-cap'),
        'clock_scope': 'original-prompt-through-successful-send-and-owned-cleanup',
        'peer_wait_included': False, 'complete_wire_turn_wall_required_separately': True,
        'lethal_candidate_is_win_proof': False, 'strength_evaluation': False}


def check_traces(roots, decisions, config, *, room_clocks=None, opponent_clear_profile=None,
                 opponent_known_profile=None):
    """Recompute every admission and serial clock transition in a clean game."""
    if room_clocks is not None:
        from . import agent_final_room_budget as B
        B.check_clocks(room_clocks, player=roots[0]['viewer'] if roots else room_clocks.get('player'))
    if opponent_clear_profile is not None:
        from . import agent_opponent_clear as OC
        OC.validate_profile(opponent_clear_profile)
    if opponent_known_profile is not None:
        from . import agent_opponent_known as OK
        OK.validate_profile(opponent_known_profile)
        if opponent_clear_profile is not None:
            raise ValueError('opponent public-information identities are mutually exclusive')
    ledger = TurnSearchGate(config)
    for index, root in enumerate(roots):
        trace = root.get('on_demand')
        if not isinstance(trace, dict) or set(trace) != {'started', 'public_search_deadline',
                'public_response_deadline', 'public', 'decisions', 'optional_seconds', 'non_search_seconds',
                'successful_send_ns', 'finish'}:
            raise ValueError("on-demand prompt lost its complete public/clock trace")
        begin = trace['started']
        if room_clocks is not None:
            from . import agent_final_room_budget as B
            previous = room_clocks['bound_at'] if index == 0 else roots[index-1]['on_demand']['successful_send_ns']/1e9
            allocation = B.check_trace(root.get('final_room_budget'), room_clocks,
                                      prompt_index=index, started=begin, previous_response_at=previous)
            if trace['public_search_deadline'] != allocation.search_deadline \
                    or trace['public_response_deadline'] != allocation.original_response_deadline:
                raise ValueError('final-room trace changed its original prompt deadline')
            public = None
        else:
            public = root['public_budget']
        # A deferred prompt starts at wire arrival, while public allocation
        # is computed only after its actual TIME_LIMIT arrives. The remaining
        # allocation is relative to that later time; the cap is NOT reset.
        if public is not None:
            elapsed = _number(public.get('prompt_elapsed', 0.), 'original public prompt wait seconds')
            if abs(trace['public_search_deadline'] - begin - elapsed - public['allocated_seconds']) > 1e-8 \
                    or abs(trace['public_response_deadline'] - begin - elapsed - public['response_seconds']) > 1e-8:
                raise ValueError("on-demand trace altered the original public clock")
        ledger.begin_prompt(turn=root['turn'], started=begin, search_deadline=trace['public_search_deadline'],
                            response_deadline=trace['public_response_deadline'],
                            **({'optional_response_deadline': allocation.response_deadline}
                               if room_clocks is not None and allocation.allow_optional else {}))
        choices = [d for d in decisions if d['response_index'] == index and not d['forced']]
        if len(trace['decisions']) != len(choices) or len(root['searches']) != len(choices):
            raise ValueError("on-demand trace omitted a real subdecision")
        clear = None
        if opponent_clear_profile is not None:
            clear = OC.check(root.get('opponent_clear'))
            if clear['viewer'] != root['viewer'] or (trace['public'] is None) != (not clear['allow_search']):
                raise ValueError('opponent-clear witness changed its observer or bypassed public gating')
        elif 'opponent_clear' in root:
            raise ValueError('legacy gate cannot acquire an unregistered opponent-clear witness')
        known = None
        if opponent_known_profile is not None:
            known = OK.check(root.get('opponent_known'))
            if known['public_shape']['viewer'] != root['viewer'] or (trace['public'] is None) != (not known['allow_search']):
                raise ValueError('opponent-known witness changed its observer or bypassed public gating')
        elif 'opponent_known' in root:
            raise ValueError('legacy gate cannot acquire an unregistered opponent-known witness')
        hint = None if trace['public'] is None and (clear is not None or known is not None) else hint_from_witness(trace['public'])
        if trace['public'] is not None and trace['public']['viewer'] != root['viewer']:
            raise ValueError("public gate witness changed its viewer")
        for event, choice, search in zip(trace['decisions'], choices, root['searches']):
            if set(event) != {'at', 'obs_sha256', 'gate', 'optional_seconds'} or event['obs_sha256'] != choice['obs_sha256']:
                raise ValueError("gate event belongs to another pending observation")
            expected = decision_wire(ledger.decide(logits=choice['logits'], rows=choice['rows'],
                                                   hint=hint, now=event['at'],
                                                   **({'opponent_clear':clear} if clear is not None else {}),
                                                   **({'opponent_known':known} if known is not None else {})))
            if event['gate'] != expected or search.get('gate_decision') != expected \
                    or choice['rows'] > 1 and (search.get('gate_skipped') is True) == expected['search']:
                raise ValueError("gate admission differs from full original logits/public facts/clock")
            optional = _number(event['optional_seconds'], 'decision optional work seconds')
            if not expected['search'] and optional != 0:
                raise ValueError("an explicit gate skip cannot contain hidden search work")
        finish = trace['finish']
        if not isinstance(finish, dict) or finish.get('delivered') is not True or finish.get('cleanup_complete') is not True:
            raise ValueError("a clean game needs actual successful-send and owned-cleanup evidence")
        expected = ledger.finish_prompt(now=finish['finished'], delivered=True, cleanup_complete=True)
        if finish != expected:
            raise ValueError("gate turn accounting changed or omitted an overrun")
        stamp = trace['successful_send_ns']
        if type(stamp) is not int or not begin <= stamp / 1e9 <= finish['finished']:
            raise ValueError("gate cleanup is not after the successful network send")
        optional = _number(trace['optional_seconds'], 'optional work seconds')
        non_search = _number(trace['non_search_seconds'], 'non-search work seconds')
        if optional > finish['prompt_seconds'] or abs(optional + non_search - finish['prompt_seconds']) > 1e-8:
            raise ValueError("gate wall clock dropped or duplicated preparation/RPC/cleanup")
        if abs(optional - math.fsum(row['optional_seconds'] for row in trace['decisions'])) > 1e-8:
            raise ValueError("gate empty-search accounting lost a subdecision's work")
