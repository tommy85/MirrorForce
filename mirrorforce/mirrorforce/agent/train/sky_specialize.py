"""Independent public-only imitation pilot; never modifies the A0 parent state.

Each optimization example is a complete own-seat public decision sequence.
Memory is reconstructed with the current parameters from its actual first
decision. Teacher choices are loss targets only. A frozen-parent KL is a local
retention regularizer, not a claim about global playing strength.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..env.privileged import assert_public
from .sky_combo_public import extract, array_sha
from .checkpoint_store import file_sha256

SCHEMA='mirrorforce_public_tactical_specialization_dataset/v1'


def training_schedule(rows, updates, law='episode-cycle/v1'):
    """Predeclare whole-sequence sampling; never admit heldout examples.

    The opt-in schedule gives the new clear variant half the updates and
    interleaves the three original families in the other half. This changes
    training exposure, not legal actions, inputs, loss, or optimizer state.
    """
    if type(updates) is not int or not 1 <= updates <= 1000:
        raise ValueError('register a finite specialization schedule')
    train=[i for i,row in enumerate(rows) if row['split']=='train']
    if not train or len({rows[i]['seed'] for i in train})!=len(train):
        raise ValueError('training schedule requires unique registered episodes')
    if law=='episode-cycle/v1':return [train[i%len(train)] for i in range(updates)]
    if law=='command-balanced/v1':
        from .sky_command_timing import VARIANTS
        new=[[i for i in train if rows[i]['family']=='command_timing' and rows[i].get('variant')==v] for v in VARIANTS]
        old=[[i for i in train if rows[i]['family']==family and rows[i].get('variant')==variant]
             for family,variant in (('sword',None),('bomber',None),('main2_set',None),
                                    ('bomber','linked_clear'),('backrow_timing',None))]
        if any(not g for g in new+old) or sum(map(len,new+old))!=len(train):
            raise ValueError('command balancing requires all six variants and five retained routes')
        schedule=[]
        for i in range(updates):
            groups=new if i%2==0 else old;turn=i//2;group=groups[turn%len(groups)]
            schedule.append(group[(turn//len(groups))%len(group)])
        return schedule
    if law=='backrow-balanced/v1':
        target=[i for i in train if rows[i]['family']=='backrow_timing' and rows[i].get('variant') is None]
        groups=[[i for i in train if rows[i]['family']==family and rows[i].get('variant')==variant]
                for family,variant in (('sword',None),('bomber',None),('main2_set',None),('bomber','linked_clear'))]
        if not target or any(not g for g in groups) or len(target)+sum(map(len,groups))!=len(train):
            raise ValueError('backrow balancing requires all four retained routes and timing examples')
        return [target[(i//2)%len(target)] if i%2 else
                groups[(i//2)%4][((i//2)//4)%len(groups[(i//2)%4])] for i in range(updates)]
    if law!='clear-balanced/v1':raise ValueError('unknown specialization sampling law')
    clear=[i for i in train if rows[i].get('variant')=='linked_clear' and rows[i]['family']=='bomber']
    old=[[i for i in train if rows[i].get('variant') is None and rows[i]['family']==family]
         for family in ('sword','bomber','main2_set')]
    if not clear or any(not group for group in old) or len(clear)+sum(map(len,old))!=len(train):
        raise ValueError('balanced sampling needs the clear variant and all three original families')
    schedule=[]
    for i in range(updates):
        turn=i//2
        if i%2:schedule.append(clear[turn%len(clear)])
        else:
            group=old[turn%3]
            schedule.append(group[(turn//3)%len(group)])
    return schedule


def export_episode(native,tape,path):
    observations=[];labels=[];rows=[]
    def accept(row,obs):
        assert_public(obs,'specialization export')
        if array_sha(obs)!=row['obs_sha256']:raise ValueError('public observation changed')
        observations.append(obs);labels.append(row['chosen']);rows.append(row)
    verified=extract(native,tape,on_decision=accept)
    if not observations or len(observations)>256:raise ValueError('invalid curriculum sequence length')
    keys={k for k in observations[0] if k.startswith('obs:')}
    if not keys or any({k for k in o if k.startswith('obs:')}!=keys for o in observations):
        raise ValueError('public sequence keys differ')
    # ClientDuel also returns diagnostic info:/reward/step fields. The normal
    # policy filters these out; exported learner inputs must do the same.
    arrays={k:np.stack([o[k] for o in observations]) for k in sorted(keys)}
    arrays['target']=np.asarray(labels,np.int32)
    with Path(path).open('xb') as f:np.savez_compressed(f,**arrays)
    return {'path':str(path),'sha256':file_sha256(path),'decisions':len(labels),
            'responses':verified['responses'],'terminal':verified['terminal'],
            'public_inputs_only':True,'decision_rows':rows}


def load_episode(ref):
    if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('specialization episode changed')
    with np.load(ref['path'],allow_pickle=False) as raw:arrays={k:raw[k] for k in raw.files}
    labels=arrays.pop('target')
    assert_public(arrays,'specialization learner')
    if any(not k.startswith('obs:') for k in arrays):raise ValueError('only declared public observations are inputs')
    n=ref['decisions']
    if labels.shape!=(n,) or labels.dtype!=np.int32 or n<2 or any(v.shape[0]!=n for v in arrays.values()):
        raise ValueError('incomplete public sequence')
    legal=arrays['obs:action_ir_'][:,:,0]>0
    if (labels<0).any() or (labels>=legal.shape[1]).any() or not legal[np.arange(n),labels].all():
        raise ValueError('teacher label is not a native legal row')
    return {k[4:]:v for k,v in arrays.items()},labels


def check_dataset(manifest):
    if manifest.get('schema')!=SCHEMA or manifest.get('human_or_evaluation_replays') is not False:
        raise ValueError('only independently generated synthetic training is admitted')
    rows=manifest.get('episodes',[])
    if not rows or any(r.get('split') not in ('train','heldout') for r in rows):raise ValueError('explicit whole-episode split required')
    if len({r['seed'] for r in rows})!=len(rows):raise ValueError('train/heldout seeds overlap')
    if len({r['sha256'] for r in rows})!=len(rows):raise ValueError('train/heldout episode bytes overlap')
    families=[{r['family'] for r in rows if r['split']==split} for split in ('train','heldout')]
    base={'sword','bomber','main2_set'}
    if families[0]!=families[1] or not base<=families[0]<=base|{'backrow_timing','command_timing'}:
        raise ValueError('each split needs all registered families')
    if 'command_timing' in families[0]:
        from .sky_command_timing import VARIANTS
        if any({r.get('variant') for r in rows if r['split']==split and r['family']=='command_timing'}!=set(VARIANTS)
               for split in ('train','heldout')):
            raise ValueError('both splits need all six command-timing variants')
    return rows


def loss_from_logits(logits,anchor,labels,kl_coef,valid=None,decision_weights=None):
    import jax
    import jax.numpy as jnp
    legal=anchor>-1e8
    lp=jax.nn.log_softmax(jnp.where(legal,logits,-1e9),axis=-1)
    base=jax.nn.log_softmax(jnp.where(legal,anchor,-1e9),axis=-1)
    weight=jnp.ones(labels.shape,jnp.float32) if valid is None else jnp.asarray(valid,jnp.float32)
    mean=lambda x:(x*weight).sum()/jnp.maximum(weight.sum(),1)
    ce=mean(-jnp.take_along_axis(lp,labels[:,None],axis=-1)[:,0])
    importance=jnp.ones(labels.shape,jnp.float32) if decision_weights is None else jnp.asarray(decision_weights,jnp.float32)
    weighted_ce=mean(-jnp.take_along_axis(lp,labels[:,None],axis=-1)[:,0]*importance)
    kl=mean(jnp.where(legal,jnp.exp(lp)*(lp-base),0).sum(-1))
    return weighted_ce+kl_coef*kl,{'teacher_ce':ce,'weighted_teacher_ce':weighted_ce,'parent_kl':kl,
        'teacher_top1':mean(jnp.argmax(logits,axis=-1)==labels)}


def pad_episode(obs,labels,steps=64):
    """Pad with independent first observations, never fake recurrent history.

    Every padded row resets memory and has zero loss/metrics weight. No extra
    native responses or training decision counters are manufactured.
    """
    n=len(labels)
    if not 1<n<=steps:raise ValueError('sequence exceeds registered whole-game training geometry')
    padded={k:np.concatenate([v,np.repeat(v[:1],steps-n,axis=0)]) for k,v in obs.items()}
    target=np.concatenate([labels,np.full(steps-n,labels[0],np.int32)])
    valid=np.arange(steps)<n
    first=(np.arange(steps)==0)|~valid
    return padded,target,valid,first


def command_weights(row,steps,weight):
    """Training-only importance for public command/battle choices, not an action mask."""
    from ...netduel import constants as C
    if type(weight) not in (int,float) or not np.isfinite(weight) or not 1<=weight<=16:
        raise ValueError('register a finite command-loss weight in [1,16]')
    decisions=row['decision_rows']
    if len(decisions)!=row['decisions'] or len(decisions)>steps:
        raise ValueError('command-loss metadata does not match the public sequence')
    weights=np.ones(steps,np.float32)
    for i,item in enumerate(decisions):
        if item['decision']!=i:raise ValueError('decision ordering changed')
        if item['msg'] in (C.MSG_SELECT_IDLECMD,C.MSG_SELECT_BATTLECMD):weights[i]=weight
    return weights


def make_learner(agent,variables,*,learning_rate=1e-5,kl_coef=.25):
    import jax
    import jax.numpy as jnp
    import optax
    if not 0<learning_rate<=1e-4 or not 0<=kl_coef<=10:raise ValueError('invalid specialization recipe')
    constants=variables['constants']
    tx=optax.chain(optax.clip_by_global_norm(.1),optax.adam(learning_rate,eps=1e-8))
    def forward(params,obs,first):
        n=next(iter(obs.values())).shape[0]
        memory=(agent.init_rnn_state(1),agent.init_rnn_state(1))
        return agent.apply({'params':params,'constants':constants},obs,memory,first,jnp.ones(n,bool))[1]
    def objective(params,obs,labels,anchor,valid,first,decision_weights):
        return loss_from_logits(forward(params,obs,first),jax.lax.stop_gradient(anchor),labels,kl_coef,valid,decision_weights)
    @jax.jit
    def update(params,opt_state,obs,labels,anchor,valid,first,decision_weights=None):
        (loss,metrics),grads=jax.value_and_grad(objective,has_aux=True)(params,obs,labels,anchor,valid,first,decision_weights)
        updates,next_state=tx.update(grads,opt_state,params)
        candidate=optax.apply_updates(params,updates)
        finite=jnp.isfinite(loss)&jnp.isfinite(optax.global_norm(grads))
        return candidate,next_state,{**metrics,'loss':loss,'gradient_norm':optax.global_norm(grads),'finite':finite}
    return tx,jax.jit(forward),update


def parameter_digest(params):
    import jax
    h=hashlib.sha256()
    for path,leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        value=np.ascontiguousarray(leaf)
        h.update(str(path).encode()+str(value.shape).encode()+str(value.dtype).encode()+value.tobytes())
    return h.hexdigest()


def load_specialization_resume(binding, config, params, optimizer):
    """Resume this separate branch's Adam, never reconstruct an A0 checkpoint."""
    import json
    import flax.serialization
    import jax
    if type(binding) is not dict or set(binding)!={'checkpoint','training_config','completed_result'}:
        raise ValueError('bind the completed specialization and its original recipe')
    for item in binding.values():
        if set(item)!={'path','sha256'} or file_sha256(item['path'])!=item['sha256']:
            raise ValueError('specialization resume input changed')
    previous=json.loads(Path(binding['training_config']['path']).read_bytes())
    result=json.loads(Path(binding['completed_result']['path']).read_bytes())
    if (previous['learning_rate']!=config['learning_rate'] or previous['kl_coef']!=config['kl_coef'] or
            {k:v['sha256'] for k,v in previous['assets'].items()}!=
            {k:v['sha256'] for k,v in config['assets'].items()}):
        raise ValueError('specialization optimizer recipe or immutable parent changed')
    if result.get('complete') is not True or result.get('checkpoint_sha256')!=binding['checkpoint']['sha256']:
        raise ValueError('resume needs a complete original result')
    branch=flax.serialization.msgpack_restore(Path(binding['checkpoint']['path']).read_bytes())
    if (branch['parent_checkpoint_sha256']!=config['assets']['checkpoint']['sha256'] or
            type(branch['specialization_updates']) is not int or branch['specialization_updates']<1):
        raise ValueError('specialization parent or optimizer counter differs')
    candidate=branch['params']
    if jax.tree.structure(params)!=jax.tree.structure(candidate) or any(
            a.shape!=b.shape or a.dtype!=b.dtype for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(candidate))):
        raise ValueError('resumed actor architecture differs')
    if parameter_digest(candidate)!=result['candidate_parameter_sha256']:
        raise ValueError('resumed parameters differ from the completed result')
    restored=flax.serialization.from_state_dict(optimizer,branch['specialization_optimizer'])
    if int(np.asarray(restored[1][0].count))!=branch['specialization_updates']:
        raise ValueError('specialization Adam count differs from saved updates')
    return jax.tree.map(jax.numpy.asarray,candidate),restored,branch['specialization_updates']
