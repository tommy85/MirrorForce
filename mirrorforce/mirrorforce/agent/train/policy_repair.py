"""Compose an explicitly new A0 state with repaired actor parameters only.

This primitive does not publish a checkpoint or admit a training run. Its
caller must bind the immutable parent, repair branch and actual acceptance
evidence. The branch's separate supervised Adam is never used as A0 Adam.
"""
from __future__ import annotations

import hashlib
import re
import numpy as np
from flax.serialization import msgpack_serialize


def digest(value):
    return hashlib.sha256(msgpack_serialize(value)).hexdigest()


def _check_params(parent, candidate):
    if isinstance(parent, dict):
        if not isinstance(candidate, dict) or set(parent)!=set(candidate):
            raise ValueError('repair actor parameter keys differ')
        for key in parent:_check_params(parent[key],candidate[key])
    else:
        if not isinstance(parent,np.ndarray) or not isinstance(candidate,np.ndarray):
            raise ValueError('repair parameters must be the original numeric array tree')
        if parent.shape!=candidate.shape or parent.dtype!=candidate.dtype:
            raise ValueError('repair actor shape/dtype differs')
        if not np.isfinite(parent).all() or not np.isfinite(candidate).all():
            raise ValueError('nonfinite actor parameters')


def compose(parent, branch, *, parent_sha256):
    """Return a fresh outer payload and verifiable preservation fingerprints."""
    if not isinstance(parent_sha256,str) or not re.fullmatch('[0-9a-f]{64}',parent_sha256):
        raise ValueError('name the actual immutable A0 parent')
    required={'state','critic','ema','learner_keys','counters'}
    if set(parent)!=required:
        raise ValueError('this repair requires the complete A0 state without live actor/environment snapshots')
    state=parent['state']
    if not isinstance(state,dict) or not {'params','opt_state','step'}<=set(state):
        raise ValueError('the A0 actor optimizer and step must exist')
    if not isinstance(parent['critic'],dict) or not {'params','opt_state','step'}<=set(parent['critic']):
        raise ValueError('the original central critic/optimizer must be retained')
    if branch.get('parent_checkpoint_sha256')!=parent_sha256 or type(branch.get('specialization_updates')) is not int \
            or branch['specialization_updates']<1:
        raise ValueError('repair branch does not belong to the immutable parent')
    if set(parent['counters'])!={'global_step','learner_update'} or any(type(x) is not int or x<0 for x in parent['counters'].values()):
        raise ValueError('original training counters are malformed')
    keys=parent['learner_keys']
    if not isinstance(keys,np.ndarray) or keys.dtype!=np.uint32 or keys.ndim!=2 or keys.shape[1]!=2:
        raise ValueError('original learner keys are malformed')
    _check_params(state['params'],branch['params'])
    if digest(state['params'])==digest(branch['params']):raise ValueError('no actual actor repair')
    before=digest(parent)
    result={**parent,'state':{**state,'params':branch['params']}}
    unchanged=lambda v:{**v,'state':{k:x for k,x in v['state'].items() if k!='params'}}
    assert digest(unchanged(parent))==digest(unchanged(result)) and digest(parent)==before
    proof={'schema':'mirrorforce_actor_parameter_repair/v1','parent_checkpoint_sha256':parent_sha256,
       'actor_parameters_before_sha256':digest(state['params']),'actor_parameters_after_sha256':digest(branch['params']),
       'unchanged_A0_state_sha256':digest(unchanged(parent)),
       'original_actor_optimizer_sha256':digest(state['opt_state']),'original_critic_sha256':digest(parent['critic']),
       'original_ema_sha256':digest(parent['ema']),'learner_keys_sha256':digest(keys),
       'counters':dict(parent['counters']),'supervised_updates_not_counted_as_A0_updates':branch['specialization_updates'],
       'supervised_optimizer_imported':False,'actor_environment_memory_imported':False,
       'environment_resume_requires_original_declared_restart_contract':True,
       'checkpoint_published':False,'training_resume_admitted':False}
    return result,proof
