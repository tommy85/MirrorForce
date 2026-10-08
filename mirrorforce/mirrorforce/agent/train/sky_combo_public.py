"""Verify/extract imitation decisions from a curriculum's public-only tape.

The caller supplies a registered native ClientDuel module. This entry has no
host seed, forced deck order, engine snapshot, truth labels or teacher state.
"""
import hashlib
import numpy as np

from ...netduel.agent_client import AgentClientDuel
from ...netduel import constants as C
from ..env.privileged import assert_public

SCHEMA = 'mirrorforce_sky_halq_bulb_borrelsword_curriculum/v1#public-tape'


def array_sha(observation):
    h = hashlib.sha256()
    for name, value in sorted(observation.items()):
        value = np.ascontiguousarray(value)
        h.update(name.encode()+b'\0'+str(value.dtype).encode()+str(value.shape).encode()+value.tobytes())
    return h.hexdigest()


def extract(native, tape, *, on_decision=None, on_pending=None):
    fields = {'schema','viewer','own_recipe','opponent_recipe_mode','messages','responses',
              'terminal','winner','private_layouts'}
    if (type(tape) is not dict or set(tape) != fields or tape['schema'] != SCHEMA
            or tape['private_layouts'] is not False or tape['opponent_recipe_mode'] != 'mirror'
            or type(tape['viewer']) is not int or tape['viewer'] not in (0,1)
            or type(tape['terminal']) is not bool or set(tape['own_recipe']) != {'main','extra'}
            or type(tape['messages']) is not list or not tape['messages']
            or type(tape['responses']) is not list or not tape['responses']):
        raise ValueError('curriculum replay accepts only its closed public tape, never hidden setup fields')
    if tape['messages'][0][0] != C.MSG_START:
        raise ValueError('curriculum public stream must begin at the actual start')
    wins = [entry for entry in tape['messages'] if entry[0] == C.MSG_WIN]
    if tape['terminal']:
        if len(wins)!=1 or wins[0]!=tape['messages'][-1] or type(tape['winner']) is not int \
                or bytes.fromhex(wins[0][1]) != bytes([tape['winner'],C.WIN_REASON_LP]):
            raise ValueError('curriculum terminal needs its real LP-win event, not a completed flag')
    elif wins or tape['winner'] is not None:
        raise ValueError('a nonterminal curriculum prefix cannot claim a winner')
    recipe = tape['own_recipe']
    client = AgentClientDuel(native,tape['viewer'],recipe['main'],recipe['extra'],
        {'max_options':192,'max_steps':1000,'n_history_actions':32,'public_opponent_recipe':True,
         'room_era':0,'room_format':0},opponent_main=recipe['main'],opponent_extra=recipe['extra'])
    answers = [bytes.fromhex(raw) for raw in tape['responses']]
    if any(not raw or len(raw)>255 or raw.hex()!=text for raw,text in zip(answers,tape['responses'])):
        raise ValueError('curriculum response tape is not canonical')
    cursor, forced = 0, 0
    decisions = []
    pending = False
    for packet, entry in enumerate(tape['messages']):
        if type(entry) is not list or len(entry)!=2 or type(entry[0]) is not int or not 0<=entry[0]<=255:
            raise ValueError('curriculum needs complete typed public packets')
        msg, text = entry
        body = bytes.fromhex(text)
        if body.hex()!=text: raise ValueError('public packet hex changed')
        response = client.feed(msg,body)
        if response is None and client.prompt() is not None:
            if cursor==len(answers):
                if packet != len(tape['messages'])-1 or tape['terminal']:
                    raise ValueError('an unanswered curriculum prompt is not the exact registered prefix end')
                pending = True
                if on_pending is not None:
                    prompt=client.prompt();observation=client.observation()
                    assert_public(observation,'public-prefix pending diagnostic')
                    on_pending({'packet':packet,'msg':int(prompt[1]),'viewer':tape['viewer'],
                        'rows':len(prompt[2]),'legal_rows':[dict(a) for a in prompt[2]],
                        'obs_sha256':array_sha(observation)},
                        {k:np.array(v,copy=True) for k,v in observation.items()})
                break
            path = client.response_path(answers[cursor])
            for subdecision,index in enumerate(path):
                prompt = client.prompt()
                if prompt is None or not 0<=index<len(prompt[2]):
                    raise ValueError('teacher response has no current native legal continuation')
                if len(prompt[2])==1:
                    forced += 1
                else:
                    observation = client.observation()
                    assert_public(observation, 'curriculum public export')
                    row = {'decision':len(decisions),'response':cursor,'subdecision':subdecision,
                           'packet':packet,'msg':int(prompt[1]),'viewer':tape['viewer'],
                           'rows':len(prompt[2]),'chosen':int(index),'obs_sha256':array_sha(observation),
                           'legal_rows':[dict(action) for action in prompt[2]]}
                    if on_decision is not None:
                        on_decision(dict(row), {k:np.array(v,copy=True) for k,v in observation.items()})
                    decisions.append(row)
                response = client.step(index)
        if response is not None:
            if cursor>=len(answers) or bytes(response)!=answers[cursor]:
                expected=answers[cursor].hex() if cursor<len(answers) else '<end>'
                raise ValueError('public-only replay changed an actual curriculum response '
                                 f'at packet {packet}, msg {msg}, response {cursor}: '
                                 f'{bytes(response).hex()} != {expected}')
            cursor += 1
    if cursor!=len(answers) or (not tape['terminal'] and not pending) \
            or (tape['terminal'] and client.prompt() is not None):
        raise ValueError('public curriculum replay did not consume its exact complete/prefix boundary')
    return {'schema':SCHEMA+'#verified','responses':cursor,'forced_subdecisions':forced,
            'decisions':decisions,'public_inputs_only':True,'teacher_forcing':True,
            'network_forward_calls':0,'model_updates':0,'terminal':tape['terminal'],
            'training_eligible':False}
