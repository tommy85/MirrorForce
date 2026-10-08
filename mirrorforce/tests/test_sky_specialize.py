import copy
import numpy as np
import pytest
from mirrorforce.agent.train import sky_specialize as S


def manifest():
    return {'schema':S.SCHEMA,'human_or_evaluation_replays':False,'episodes':[
        {'seed':i,'sha256':str(i),'family':family,'split':split}
        for i,(split,family) in enumerate((s,f) for s in ('train','heldout') for f in ('sword','bomber','main2_set'))]}


def test_dataset_split_is_whole_unique_episodes_and_all_families():
    value=manifest();assert len(S.check_dataset(value))==6
    for key in ('seed','sha256'):
        bad=copy.deepcopy(value);bad['episodes'][-1][key]=bad['episodes'][0][key]
        with pytest.raises(ValueError):S.check_dataset(bad)
    value['human_or_evaluation_replays']=True
    with pytest.raises(ValueError):S.check_dataset(value)


@pytest.mark.parametrize('missing_split',[None,'train','heldout'])
def test_command_dataset_requires_every_variant_in_both_splits(missing_split):
    from mirrorforce.agent.train.sky_command_timing import VARIANTS
    value=manifest()
    for split in ('train','heldout'):
        for variant in VARIANTS:
            if split==missing_split and variant=='own_turn_g_chain':continue
            i=len(value['episodes'])
            value['episodes'].append(dict(seed=i,sha256=str(i),family='command_timing',split=split,variant=variant))
    if missing_split:
        with pytest.raises(ValueError,match='all six'):S.check_dataset(value)
    else:assert len(S.check_dataset(value))==18


def test_episode_loader_keeps_labels_out_of_inputs_and_rejects_illegal(tmp_path):
    p=tmp_path/'public.npz'
    obs=np.ones((3,4,2),np.int32)
    np.savez_compressed(p,**{'obs:action_ir_':obs,'target':np.array([0,1,3],np.int32)})
    ref={'path':str(p),'sha256':S.file_sha256(p),'decisions':3}
    inputs,labels=S.load_episode(ref)
    assert set(inputs)=={'action_ir_'} and labels.tolist()==[0,1,3]
    obs[1,1,0]=0
    np.savez_compressed(p,**{'obs:action_ir_':obs,'target':np.array([0,1,3],np.int32)})
    ref['sha256']=S.file_sha256(p)
    with pytest.raises(ValueError,match='legal'):S.load_episode(ref)


def test_export_and_real_reader_agree_with_client_diagnostic_fields(tmp_path,monkeypatch):
    obs={'obs:action_ir_':np.ones((4,2),np.uint8),'info:to_play':np.array(1,np.int32),
         'reward':np.array([0.],np.float32),'done':np.array(False)}
    def replay(native,tape,*,on_decision):
        for i in range(3):on_decision({'chosen':i,'obs_sha256':S.array_sha(obs)},obs)
        return {'responses':3,'terminal':False}
    monkeypatch.setattr(S,'extract',replay)
    ref=S.export_episode(None,{},tmp_path/'episode.npz')
    inputs,labels=S.load_episode(ref)
    assert set(inputs)=={'action_ir_'} and labels.tolist()==[0,1,2]


def test_loss_reduces_teacher_ce_without_training_anchor():
    import jax
    import jax.numpy as jnp
    logits=jnp.array([[0.,1.,-1e9],[1.,0.,-1e9]])
    labels=jnp.array([0,1])
    grad=jax.grad(lambda x:S.loss_from_logits(x,logits,labels,.25)[0])(logits)
    before=S.loss_from_logits(logits,logits,labels,.25)[0]
    after=S.loss_from_logits(logits-.1*grad,logits,labels,.25)[0]
    assert float(after)<float(before) and np.all(np.asarray(grad)[:,2]==0)


def test_command_importance_changes_only_objective_not_reported_unweighted_ce():
    import jax
    import jax.numpy as jnp
    logits=jnp.zeros((2,2));labels=jnp.array([0,0])
    plain=jax.grad(lambda x:S.loss_from_logits(x,logits,labels,0.)[0])(logits)
    weighted=jax.grad(lambda x:S.loss_from_logits(x,logits,labels,0.,decision_weights=jnp.array([4.,1.]))[0])(logits)
    np.testing.assert_allclose(weighted[0],4*plain[0]);np.testing.assert_allclose(weighted[1],plain[1])
    _,a=S.loss_from_logits(logits,logits,labels,0.)
    _,b=S.loss_from_logits(logits,logits,labels,0.,decision_weights=jnp.array([4.,1.]))
    assert float(a['teacher_ce'])==float(b['teacher_ce'])
    assert float(b['weighted_teacher_ce'])==pytest.approx(2.5*float(a['teacher_ce']))


def test_command_weights_are_separate_from_policy_inputs_and_padding():
    row={'decisions':3,'decision_rows':[{'decision':i,'msg':msg} for i,msg in enumerate([11,18,10])]}
    assert S.command_weights(row,5,4).tolist()==[4.,1.,4.,1.,1.]
    with pytest.raises(ValueError):S.command_weights(row,5,float('nan'))


def test_padding_resets_only_synthetic_rows_and_counts_no_new_decisions():
    obs={'x':np.array([[1,2],[3,4],[5,6]])};labels=np.array([1,0,1],np.int32)
    padded,target,valid,first=S.pad_episode(obs,labels,5)
    assert padded['x'].tolist()==[[1,2],[3,4],[5,6],[1,2],[1,2]]
    assert valid.tolist()==[True,True,True,False,False]
    assert first.tolist()==[True,False,False,True,True]
    import jax
    import jax.numpy as jnp
    logits=jnp.zeros((5,2));target=jnp.asarray(target)
    grad=jax.grad(lambda x:S.loss_from_logits(x,logits,target,.25,valid)[0])(logits)
    assert not np.asarray(grad)[3:].any() and np.asarray(grad)[:3].any()


def test_specialization_resume_preserves_adam_and_rejects_recipe_drift(tmp_path):
    import json
    import jax
    import jax.numpy as jnp
    import optax
    import flax.serialization
    params={'w':jnp.array([1.,2.])}
    tx=optax.chain(optax.clip_by_global_norm(.1),optax.adam(1e-5,eps=1e-8))
    opt=tx.init(params)
    for _ in range(3):
        updates,opt=tx.update({'w':jnp.array([.3,.7])},opt,params)
        params=optax.apply_updates(params,updates)
    cfg={'learning_rate':1e-5,'kl_coef':.25,'assets':{'checkpoint':{'sha256':'a'*64}}}
    branch={'params':jax.device_get(params),'specialization_optimizer':flax.serialization.to_state_dict(jax.device_get(opt)),
            'specialization_updates':3,'parent_checkpoint_sha256':'a'*64}
    cp=tmp_path/'branch';cp.write_bytes(flax.serialization.msgpack_serialize(branch))
    old=tmp_path/'config.json';old.write_text(json.dumps(cfg))
    result=tmp_path/'result.json';result.write_text(json.dumps({'complete':True,'checkpoint_sha256':S.file_sha256(cp),
        'candidate_parameter_sha256':S.parameter_digest(params)}))
    binding={key:{'path':str(p),'sha256':S.file_sha256(p)} for key,p in
             [('checkpoint',cp),('training_config',old),('completed_result',result)]}
    actual,state,count=S.load_specialization_resume(binding,cfg,params,tx.init(params))
    assert count==3 and S.parameter_digest(actual)==S.parameter_digest(params)
    assert all(np.array_equal(a,b) for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(opt)))
    with pytest.raises(ValueError,match='recipe'):
        S.load_specialization_resume(binding,{**cfg,'learning_rate':2e-5},params,tx.init(params))
