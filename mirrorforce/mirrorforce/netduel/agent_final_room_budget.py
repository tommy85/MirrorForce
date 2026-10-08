"""Prepared final-human room budget, not yet wired to a live search client.

Only genuine TIME_LIMIT observations may enter this allocator. Its output
controls optional work; it neither authorizes a protocol response before
TIME_CONFIRM nor handles native/follower/transport failures. The gate separately
accounts every prompt's preparation, policy, search, send and cleanup against
450 seconds of work per turn. Neither limit grants a new prompt allowance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

LAW = 'final-human-room600-search10-floor150-turn450/v2'
ROOM_SECONDS = 600
SEARCH_WINDOW_SECONDS = 10.
FINALIZE_SECONDS = 3.
REMAINING_FLOOR_SECONDS = 150.
TURN_WORK_SECONDS = 450.
TURN_WORK_SCOPE = 'all-prompts-including-policy-preparation-search-send-cleanup/v1'
PROMPT_SECONDS = SEARCH_WINDOW_SECONDS + FINALIZE_SECONDS
DELIVERIES = ('before_prompt', 'prompt_then_clock')
PROFILE = {'law': LAW, 'room_seconds': ROOM_SECONDS, 'search_window_seconds': SEARCH_WINDOW_SECONDS,
           'finalize_seconds': FINALIZE_SECONDS, 'remaining_floor_seconds': REMAINING_FLOOR_SECONDS,
           'turn_work_cap_seconds': TURN_WORK_SECONDS, 'turn_work_scope': TURN_WORK_SCOPE}
TRACE_SCHEMA = LAW + '#prompt-clock/v1'
CLOCK_SCHEMA = LAW + '#received-clocks/v1'
CONFIRM_SCHEMA = LAW + '#successful-time-confirm/v1'


def validate_profile(value):
    if type(value) is not dict or value != PROFILE:
        raise ValueError('the final600/10+3/150/turn450 clock profile must be exact and explicit')
    return dict(value)


def settings(search='off'):
    from .final_search import settings as legacy_settings
    result = legacy_settings(search)
    if result is None:
        return None
    result.update(seconds=SEARCH_WINDOW_SECONDS, total_seconds=PROMPT_SECONDS,
                  finalize_seconds=FINALIZE_SECONDS, response_margin=FINALIZE_SECONDS,
                  clock_reserve=REMAINING_FLOOR_SECONDS, clock_share=1., final_room_profile=dict(PROFILE))
    result['on_demand'].update(uncertain_seconds=PROMPT_SECONDS, lethal_seconds=PROMPT_SECONDS,
                               finalize_seconds=FINALIZE_SECONDS, turn_seconds=TURN_WORK_SECONDS)
    return result


def _stamp(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid monotonic ' + name)
    return float(value)


def _binding(game_id, prompt_index, player):
    if type(game_id) is not str or not game_id or len(game_id) > 128 \
            or type(prompt_index) is not int or prompt_index < 0 \
            or type(player) is not int or player not in (0, 1):
        raise ValueError('clock needs an explicit game, prompt and actual local seat')


@dataclass(frozen=True)
class ReceivedClock:
    game_id: str
    prompt_index: int
    player: int
    seconds: int
    received_at: float
    room_seconds: int
    payload_hex: str

    def __post_init__(self):
        _binding(self.game_id, self.prompt_index, self.player)
        _stamp(self.received_at, 'clock receipt')
        if type(self.room_seconds) is not int or self.room_seconds not in (450, 600) \
                or type(self.seconds) is not int or not 0 <= self.seconds <= self.room_seconds:
            raise ValueError('received clock differs from the verified room bounds')
        try:
            raw = bytes.fromhex(self.payload_hex)
        except (TypeError, ValueError) as exc:
            raise ValueError('clock needs its original TIME_LIMIT payload') from exc
        if len(raw) != 4 or raw[0] != self.player or int.from_bytes(raw[2:4], 'little') != self.seconds:
            raise ValueError('decoded clock differs from its original TIME_LIMIT payload')


def receive_time_limit(payload, *, game_id, prompt_index, received_at, room_seconds):
    """Called only for an actual TIME_LIMIT after the room packet is verified.

    The caller binds the current held prompt for prompt_then_clock, or the
    next prompt after the last real send for before_prompt. Values are never
    inferred from LP, a model value or elapsed game time.
    """
    if type(payload) is not bytes or len(payload) != 4:
        raise ValueError('malformed actual TIME_LIMIT packet')
    return ReceivedClock(game_id, prompt_index, payload[0], int.from_bytes(payload[2:4], 'little'),
                         _stamp(received_at, 'clock receipt'), room_seconds, payload.hex())


@dataclass(frozen=True)
class Allocation:
    law: str
    game_id: str
    prompt_index: int
    player: int
    prompt_started: float
    evaluated_at: float
    enabled: bool
    clock_status: str
    remaining_seconds: float | None
    allow_optional: bool
    search_deadline: float
    response_deadline: float
    original_response_deadline: float
    floor_enforced_for_optional: bool


def allocate(*, game_id, prompt_index, player, prompt_started, now, clock,
             clock_delivery, previous_response_at, room_verified, room_seconds,
             enabled=False, previous=None):
    """Keep the original prompt and floor deadlines; never grant a fresh 10s.

    A missing/stale clock stops optional search. The ordinary response remains
    bounded by its original prompt/valid room budget and protocol guards; this
    function must not be used to send before the server's handshake permits it.
    For a valid searched prompt, even cleanup/send finishes above the 150s floor.
    """
    _binding(game_id, prompt_index, player)
    started, now = _stamp(prompt_started, 'prompt start'), _stamp(now, 'current time')
    boundary = _stamp(previous_response_at, 'preceding actual response or game start')
    if not boundary <= started <= now or type(enabled) is not bool \
            or room_verified is not True or type(room_seconds) is not int or room_seconds != ROOM_SECONDS \
            or clock_delivery not in DELIVERIES:
        raise ValueError('final search requires its verified600-second room and original prompt lifecycle')
    if clock is not None and type(clock) is not ReceivedClock:
        raise ValueError('only a decoded actual TIME_LIMIT observation may supply remaining time')
    status, remaining, expiry = 'missing', None, None
    if clock is not None:
        if clock.game_id != game_id:
            status = 'other_game'
        elif clock.player != player:
            status = 'other_player'
        elif clock.prompt_index != prompt_index:
            status = 'stale_prompt'
        elif clock.room_seconds != ROOM_SECONDS:
            status = 'other_room_clock'
        elif clock.received_at > now:
            status = 'future'
        elif clock.received_at < boundary:
            status = 'before_previous_response'
        elif clock_delivery == 'prompt_then_clock' and clock.received_at < started \
                or clock_delivery == 'before_prompt' and clock.received_at > started:
            status = 'wrong_delivery_order'
        else:
            expiry = clock.received_at + clock.seconds
            remaining = expiry - now
            status = 'expired' if remaining <= 0 else 'valid'
    original_hard = started + PROMPT_SECONDS
    if expiry is not None:
        original_hard = min(original_hard, expiry - FINALIZE_SECONDS)
    optional_hard = original_hard if expiry is None else min(original_hard, expiry - REMAINING_FLOOR_SECONDS)
    soft = min(started + SEARCH_WINDOW_SECONDS, optional_hard - FINALIZE_SECONDS)
    allowed = enabled and status == 'valid' and remaining > REMAINING_FLOOR_SECONDS and soft > now
    response = optional_hard if allowed else original_hard
    if previous is not None:
        if type(previous) is not Allocation or previous.law != LAW \
                or (previous.game_id, previous.prompt_index, previous.player, previous.prompt_started) \
                != (game_id, prompt_index, player, started) or previous.evaluated_at > now:
            raise ValueError('a deadline cannot be reused across games/prompts or moved backwards')
        original_hard = min(original_hard, previous.original_response_deadline)
        response = min(response, previous.response_deadline)
        soft = min(soft, previous.search_deadline, response - FINALIZE_SECONDS)
        allowed = allowed and previous.allow_optional and soft > now
    if not allowed:
        soft = min(soft, now, response)
    return Allocation(LAW, game_id, prompt_index, player, started, now, enabled, status, remaining,
                      allowed, soft, response, original_hard, allowed)


def prepared_profile(search='off'):
    if search not in ('off', 'on'):
        raise ValueError('explicit search=off|on required')
    return {'law': LAW, 'search': search, 'room_seconds': ROOM_SECONDS,
            'search_window_seconds_including_preparation_and_policy': SEARCH_WINDOW_SECONDS,
            'cleanup_and_send_reserve_seconds': FINALIZE_SECONDS, 'whole_prompt_seconds': PROMPT_SECONDS,
            'remaining_floor_seconds_including_cleanup': REMAINING_FLOOR_SECONDS,
            'turn_work_cap_seconds': TURN_WORK_SECONDS, 'turn_work_scope': TURN_WORK_SCOPE,
            'triggers': ['public_lethal_candidate', 'policy_uncertain'],
            'entropy_threshold': .6, 'top_two_probability_margin_threshold': .2,
            'model_value_alone_triggers_search': False, 'guaranteed_lethal': False,
            'runtime_integrated': False, 'admitted': False, 'training_eligible': False}


def trace(allocation, *, clock_sequence, clock_events_seen, clock, clock_delivery, previous_response_at):
    if type(allocation) is not Allocation:
        raise ValueError('record an actual final-room allocation')
    return {'schema': TRACE_SCHEMA, 'clock_sequence': clock_sequence, 'clock_events_seen': clock_events_seen,
            'clock': None if clock is None else asdict(clock), 'clock_delivery': clock_delivery,
            'previous_response_at': previous_response_at, 'allocation': asdict(allocation)}


def check_clocks(value, *, player):
    fields = {'schema', 'profile', 'game_id', 'player', 'bound_at', 'clock_delivery', 'receipts'}
    if type(value) is not dict or set(value) != fields or value['schema'] != CLOCK_SCHEMA \
            or type(value['player']) is not int or value['player'] != player or value['clock_delivery'] not in DELIVERIES \
            or type(value['receipts']) is not list:
        raise ValueError('final search lost its actual game/seat clock ledger')
    validate_profile(value['profile']);_binding(value['game_id'], 0, player)
    prior = _stamp(value['bound_at'], 'policy session binding')
    prior_prompt = 0
    for sequence, row in enumerate(value['receipts']):
        if type(row) is not dict or set(row) != {'sequence', 'sample'} \
                or type(row['sequence']) is not int or row['sequence'] != sequence:
            raise ValueError('clock receipt sequence was omitted or reordered')
        sample = ReceivedClock(**row['sample'])
        if sample.game_id != value['game_id'] or sample.room_seconds != ROOM_SECONDS \
                or sample.received_at < prior or sample.prompt_index < prior_prompt:
            raise ValueError('actual clock receipts cross sessions or run backwards')
        prior, prior_prompt = sample.received_at, sample.prompt_index
    return value


def check_trace(value, clocks, *, prompt_index, started, previous_response_at):
    fields = {'schema', 'clock_sequence', 'clock_events_seen', 'clock', 'clock_delivery',
              'previous_response_at', 'allocation'}
    if type(value) is not dict or set(value) != fields or value['schema'] != TRACE_SCHEMA \
            or value['clock_delivery'] != clocks['clock_delivery'] \
            or value['previous_response_at'] != previous_response_at:
        raise ValueError('prompt clock evidence changed its exact lifecycle')
    if type(value['allocation']) is not dict or set(value['allocation']) != set(Allocation.__dataclass_fields__) \
            or any(type(value['allocation'][key]) is not bool for key in ('enabled', 'allow_optional', 'floor_enforced_for_optional')) \
            or any(type(value['allocation'][key]) is not int for key in ('player', 'prompt_index')) \
            or value['clock_sequence'] is not None and type(value['clock_sequence']) is not int:
        raise ValueError('final-room allocation must keep its exact typed fields')
    supplied = Allocation(**value['allocation'])
    seen = value['clock_events_seen'];receipts = clocks['receipts']
    if type(seen) is not int or not 0 <= seen <= len(receipts):
        raise ValueError('prompt must retain its actual clock receipt count')
    own = [r for r in receipts[:seen] if r['sample']['player'] == clocks['player']]
    latest = own[-1] if own else None
    if value['clock_sequence'] != (None if latest is None else latest['sequence']) \
            or value['clock'] != (None if latest is None else latest['sample']):
        raise ValueError('prompt did not use its latest actual own-player clock')
    if any(r['sample']['received_at'] < supplied.evaluated_at for r in receipts[seen:]):
        raise ValueError('prompt omitted a clock already received before allocation')
    clock = None if value['clock'] is None else ReceivedClock(**value['clock'])
    expected = allocate(game_id=clocks['game_id'], prompt_index=prompt_index, player=clocks['player'],
        prompt_started=started, now=supplied.evaluated_at, clock=clock, clock_delivery=clocks['clock_delivery'],
        previous_response_at=previous_response_at, room_verified=True, room_seconds=ROOM_SECONDS, enabled=True)
    if supplied != expected or supplied.enabled is not True:
        raise ValueError('final-room deadlines or remaining time were relabeled')
    return expected


def check_confirmation(root, clocks):
    value = root.get('final_time_confirmation')
    fields = {'schema', 'session', 'response_index', 'player', 'payload_hex', 'received_at', 'confirmed_at_ns'}
    if type(value) is not dict or set(value) != fields or value['schema'] != CONFIRM_SCHEMA \
            or type(value['response_index']) is not int or value['response_index'] != root['prompt'] \
            or type(value['player']) is not int or value['player'] != clocks['player'] \
            or type(value['confirmed_at_ns']) is not int or value['confirmed_at_ns'] <= 0:
        raise ValueError('final response lacks its actual same-prompt TIME_CONFIRM')
    raw = bytes.fromhex(value['payload_hex'])
    sample = receive_time_limit(raw, game_id=clocks['game_id'], prompt_index=root['prompt'],
                                received_at=value['received_at'], room_seconds=ROOM_SECONDS)
    if sample.player != clocks['player'] or sample.payload_hex != value['payload_hex'] \
            or not sample.received_at <= value['confirmed_at_ns']/1e9 \
                   <= root['on_demand']['successful_send_ns']/1e9:
        raise ValueError('TIME_CONFIRM did not precede this original response send')
    if clocks['clock_delivery'] == 'before_prompt' and sample.received_at > root['on_demand']['started'] \
            or clocks['clock_delivery'] == 'prompt_then_clock' and sample.received_at < root['on_demand']['started']:
        raise ValueError('TIME_CONFIRM changed its registered prompt ordering')
    if value['session'] is None:
        if root['prompt'] != 0 or value['confirmed_at_ns']/1e9 > clocks['bound_at']:
            raise ValueError('an unbound confirmation cannot authorize later responses')
    elif value['session'] != clocks['game_id'] or not any(row['sample'] == asdict(sample) for row in clocks['receipts']):
        raise ValueError('TIME_CONFIRM belongs to another policy session or unreceived clock')
    return value
