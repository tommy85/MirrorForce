import copy
import numpy as np
import pytest
from mirrorforce.agent.train.policy_repair import compose,digest

def fixture():
    state={'params':{'w':np.ones((2,3),np.float32)},'opt_state':{'moment':np.full((2,3),7,np.float32)},
           'step':np.asarray(200000,np.int32),'constants':{},'batch_stats':{}}
    parent={'state':state,'critic':copy.deepcopy(state),'ema':{'w':np.full((2,3),.7,np.float32)},
            'learner_keys':np.arange(8,dtype=np.uint32).reshape(4,2),
            'counters':{'global_step':2840985600,'learner_update':1000}}
    branch={'params':{'w':np.full((2,3),1.01,np.float32)},'parent_checkpoint_sha256':'a'*64,
            'specialization_updates':448,'specialization_optimizer':{'do_not_import':True}}
    return parent,branch

def test_only_actor_params_change_original_Adam_critic_ema_keys_counters_are_exact():
    parent,branch=fixture();before=digest(parent);result,proof=compose(parent,branch,parent_sha256='a'*64)
    assert digest(parent)==before and digest(result)!=before
    np.testing.assert_array_equal(result['state']['params']['w'],branch['params']['w'])
    assert result['state']['opt_state'] is parent['state']['opt_state']
    assert result['state']['step']==200000 and result['counters']==parent['counters']
    assert all(digest(result[k])==digest(parent[k]) for k in ('critic','ema','learner_keys','counters'))
    assert proof['supervised_optimizer_imported'] is False and proof['training_resume_admitted'] is False
    assert 'specialization_optimizer' not in result

@pytest.mark.parametrize('fault',['live_actors','missing_critic','missing_Adam','other_parent','shape','dtype','nan','same','keys'])
def test_unsafe_or_unbound_repairs_are_rejected(fault):
    parent,branch=fixture()
    if fault=='live_actors':parent['actors']={}
    elif fault=='missing_critic':del parent['critic']
    elif fault=='missing_Adam':del parent['state']['opt_state']
    elif fault=='other_parent':branch['parent_checkpoint_sha256']='b'*64
    elif fault=='shape':branch['params']['w']=np.ones((3,2),np.float32)
    elif fault=='dtype':branch['params']['w']=branch['params']['w'].astype(np.float64)
    elif fault=='nan':branch['params']['w'][0,0]=np.nan
    elif fault=='same':branch['params']=parent['state']['params']
    else:parent['learner_keys']=np.arange(8)
    with pytest.raises(ValueError):compose(parent,branch,parent_sha256='a'*64)
