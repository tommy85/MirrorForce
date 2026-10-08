"""Public-only autonomous scene validation, separate from the teacher.

The policy endpoint receives only projected wire packets and its own recipe.
Native host setup, seeds and outcome scoring never enter that endpoint.
"""
import numpy as np

from ...netduel.agent_client import AgentClientDuel
from ..env.privileged import assert_public


class PublicPlayer:
    def __init__(self, native, policy, recipe, viewer=1):
        if set(recipe) != {'main', 'extra'} or viewer not in (0, 1):
            raise ValueError('only own recipe and viewer initialize a public player')
        self.client = AgentClientDuel(native, viewer, recipe['main'], recipe['extra'],
            {'max_options':192, 'max_steps':1000, 'n_history_actions':32,
             'public_opponent_recipe':True, 'room_era':0, 'room_format':0},
            opponent_main=recipe['main'], opponent_extra=recipe['extra'])
        self.policy = policy
        self.memory = policy.initial_state()
        self.decisions = 0

    def answer(self, packets):
        responses, decisions = [], []
        for msg, body in packets:
            response = self.client.feed(msg, bytes.fromhex(body))
            for _ in range(256):
                prompt = self.client.prompt()
                if response is not None or prompt is None:
                    break
                rows = prompt[2]
                if not rows:
                    raise ValueError('empty native public menu')
                index = 0
                if len(rows) > 1:
                    observation = self.client.observation()
                    assert_public(observation, 'autonomous curriculum decision')
                    self.memory, logits, _, _ = self.policy.act(
                        observation, self.memory, self.decisions == 0)
                    logits = np.asarray(logits)
                    if logits.shape != (len(rows),) or not np.isfinite(logits).all():
                        raise ValueError('invalid autonomous logits')
                    index = int(np.argmax(logits))
                    decisions.append({'decision':self.decisions, 'msg':int(prompt[1]),
                                      'index':index, 'action':dict(rows[index])})
                    self.decisions += 1
                response = self.client.step(index)
            else:
                raise ValueError('public selection exceeded its finite subdecision budget')
            if response is not None:
                responses.append(bytes(response).hex())
        if len(responses) != 1:
            raise ValueError('a host prompt must produce exactly one actual response')
        return {'response':responses[0], 'decisions':decisions,
                'public_inputs_only':True, 'teacher_forcing':False}


def validate_request(request):
    """Closed protocol: no seed, layout, teacher choice or host state field."""
    if type(request) is not dict or set(request) != {'packets'}:
        raise ValueError('only public packets may cross the policy boundary')
    packets = request['packets']
    if type(packets) is not list or not packets:
        raise ValueError('nonempty public packet batch required')
    for entry in packets:
        if type(entry) is not list or len(entry) != 2 or type(entry[0]) is not int \
                or not 0 <= entry[0] <= 255 or type(entry[1]) is not str:
            raise ValueError('invalid public packet')
        if bytes.fromhex(entry[1]).hex() != entry[1]:
            raise ValueError('noncanonical public packet')
    return packets
