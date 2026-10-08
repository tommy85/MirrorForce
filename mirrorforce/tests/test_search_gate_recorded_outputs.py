"""Offline predicate counts on the original failed r4 trajectory, not new play."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mirrorforce.netduel import agent_search_gate as G
from mirrorforce.netduel.actions import SelectResult, parse_select
from mirrorforce.netduel.cards import CardPool, load_ydk
from mirrorforce.netduel.client import NetDuelClient, SELECT_MESSAGES

ATTEMPT_SHA = '744c4d7e76a090a2362396f1715a1409d8bf60666224f417b3273d1743f2438e'


def test_original_r4_public_predicate_counts_do_not_claim_new_trajectory(record_property):
    source, database = os.environ.get('MF_GATE_ORIGINAL_ATTEMPT'), os.environ.get('MF_GATE_CARDS_DB')
    if not source or not database:
        pytest.skip('requires the original r4 attempt and public cards database')
    raw = Path(source).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ATTEMPT_SHA
    assert hashlib.sha256(Path(database).read_bytes()).hexdigest() == '24bc1c9bf75e3a3c5990976a128e51a48819f1bb7ab21495d264017ab6795afa'
    attempt = json.loads(raw)
    report = attempt['policy_report']
    deck = Path(__file__).parents[1] / 'decks/stage-a/SkyStriker.ydk'
    assert hashlib.sha256(deck.read_bytes()).hexdigest() == '30b8821a8023b6c7c7c7a293331a476e36574c0ec353eca7b2e47009b45a8069'
    main, extra, _ = load_ydk(deck)
    client = NetDuelClient(host='offline-public', port=1, name='offline-public', main=main, extra=extra,
                           policy=SimpleNamespace(), card_pool=CardPool(database), max_options=192)
    sent, observations = [], []
    records = {i: [r for r in report['reports'] if r['response_index'] == i and not r['forced'] and r['rows'] > 1]
               for i in range(len(report['responses']))}
    config = G.GateConfig(enabled=True)
    def send(_op, data):
        assert bytes(data).hex() == report['responses'][len(sent)]
        sent.append(bytes(data))
    client.stream = SimpleNamespace(send=send)
    def parsed(msg, body, context):
        ordinary = parse_select(msg, body, context)
        witness = G.public_combat_witness(client, ordinary)
        hint = G.hint_from_witness(witness)
        for row in records[len(sent)]:
            confidence = G.policy_confidence(row['logits'], rows=row['rows'])
            uncertain = confidence.normalized_entropy >= config.entropy_threshold \
                or confidence.top_two_margin <= config.margin_threshold
            lethal = G.lethal_candidate(hint, low_lp_threshold=config.low_lp_threshold)
            observations.append({'response_index': len(sent), 'subdecision': row['subdecision'], 'turn': row['turn'],
                'rows': row['rows'], 'obs_sha256': row['obs_sha256'], 'uncertain': uncertain, 'lethal_candidate': lethal,
                'trigger': 'lethal_candidate' if lethal else 'uncertain' if uncertain else 'confident',
                'confidence': asdict(confidence), 'public': witness, 'tactical_candidates': sorted(hint.tactical_candidates)})
        return SelectResult(msg, ordinary.player, auto_response=bytes.fromhex(report['responses'][len(sent)]),
                            cards=ordinary.cards)
    client.selection_parser = parsed
    for message, body in report['public_messages']:
        if message in SELECT_MESSAGES and len(sent) == len(report['responses']):
            break
        client._game_msg(bytes([message])+bytes.fromhex(body))
    assert len(sent) == 506 and len(observations) == 304
    assert not client.board_problems and client.board.mismatches == 0
    counts = dict(Counter(row['trigger'] for row in observations))
    flags = dict(Counter(flag for row in observations for flag in row['tactical_candidates']))
    assert counts.get('uncertain', 0) > 0
    output = {'schema': 'mirrorforce_on_demand_original_trajectory_predicates/v1',
        'source_attempt_sha256': ATTEMPT_SHA, 'gate': asdict(config), 'original_wire_responses': len(sent),
        'original_non_singleton_decisions': len(observations), 'predicate_counts': counts,
        'tactical_candidate_counts': flags, 'candidate_fraction': 1-counts.get('confident', 0)/len(observations),
        'candidate_wire_roots': len({r['response_index'] for r in observations if r['trigger'] != 'confident'}),
        'rows': observations, 'public_only_reconstruction': True, 'new_game_or_admission': False,
        'counterfactual_timing_predicted': False, 'strength_evaluation': False, 'training_eligible': False,
        'scope': 'predicates on the frozen original trajectory; changed actions/clocks can change all later roots'}
    for key, value in counts.items():
        record_property('original_trajectory_' + key, value)
    destination = os.environ.get('MF_GATE_PREDICATE_EXPORT')
    if destination:
        with Path(destination).open('x') as stream:
            json.dump(output, stream, sort_keys=True, indent=1)
            stream.write('\n')
