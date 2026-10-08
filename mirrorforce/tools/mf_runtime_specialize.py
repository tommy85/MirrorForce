#!/usr/bin/env python3
"""Export verified synthetic public tapes or fit a bounded independent actor branch."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['export','fit'])
    p.add_argument('--config',required=True,type=Path);p.add_argument('--config-sha256',required=True)
    p.add_argument('--out',required=True,type=Path);args=p.parse_args()
    from mirrorforce.agent.train.checkpoint_store import file_sha256
    if file_sha256(args.config)!=args.config_sha256:raise ValueError('configuration changed')
    cfg=json.loads(args.config.read_bytes())
    if args.out.exists():raise ValueError('preserve the original attempt')
    args.out.mkdir(parents=True)
    def save(name,v):
        with (args.out/name).open('x') as f:json.dump(v,f,sort_keys=True,indent=2)
    assets=cfg['assets']
    required={'checkpoint','native','core','semantic','code_list','card_tables','cards_db','announce'}
    if set(assets)!=required:raise ValueError('pin all native/model assets')
    for row in assets.values():
        if file_sha256(row['path'])!=row['sha256']:raise ValueError('input asset changed')
    spec=importlib.util.spec_from_file_location('duel_native',assets['native']['path'])
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    mapped={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
            if line.split()[-1].endswith('/libmfcore.so')}
    if mapped!={str(Path(assets['core']['path']).resolve())}:raise ValueError('wrong mapped core')
    native.init_module(assets['cards_db']['path'],assets['code_list']['path'],{})
    from mirrorforce.agent.train import sky_specialize as S
    from mirrorforce.agent.env.announce_law import register
    register(native,Path(assets['announce']['path']),192)
    if args.mode=='export':
        episodes=[]
        for i,row in enumerate(cfg['tapes']):
            if file_sha256(row['path'])!=row['sha256']:raise ValueError('public tape changed')
            result=S.export_episode(native,json.loads(Path(row['path']).read_bytes()),args.out/f'episode-{i:03d}.npz')
            metadata={k:row[k] for k in ('seed','family','split','start_lp','variant','finish_turn',
                       'trigger_resources','defenses','spares','spare','first_draw','threat') if k in row}
            episodes.append({**result,**metadata,'public_tape':row})
        manifest={'schema':S.SCHEMA,'episodes':episodes,'human_or_evaluation_replays':False,
            'config_sha256':args.config_sha256,'assets':assets,'labels':'one native-legal demonstrated route, not unique optimal actions',
            'training_eligible':True,'playing_strength_evidence':False}
        S.check_dataset(manifest);save('dataset.json',manifest)
        print(json.dumps({'exported':len(episodes),'decisions':sum(r['decisions'] for r in episodes)}),flush=True)
        return 0
    import flax.serialization
    import jax
    import numpy as np
    from mirrorforce.agent.train.policy_io import load_policy
    if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':raise ValueError('fit requires one explicitly isolated GPU')
    dataset=cfg['dataset']
    if file_sha256(dataset['path'])!=dataset['sha256']:raise ValueError('dataset changed')
    manifest=json.loads(Path(dataset['path']).read_bytes());rows=S.check_dataset(manifest)
    if cfg['updates'] not in range(1,1001):raise ValueError('pilot needs a finite update budget')
    sampling_law=cfg.get('sampling_law','episode-cycle/v1')
    schedule=S.training_schedule(rows,cfg['updates'],sampling_law)
    schedule_seeds=[rows[i]['seed'] for i in schedule]
    import hashlib
    schedule_sha256=hashlib.sha256(json.dumps(schedule_seeds,separators=(',',':')).encode()).hexdigest()
    save('sampling-schedule.json',{'law':sampling_law,'seeds':schedule_seeds,'sha256':schedule_sha256,
                                 'heldout_sampled':False})
    agent,variables,receipt=load_policy(assets['checkpoint']['path'],'iterate',
        semantic_file=assets['semantic']['path'],code_list_file=assets['code_list']['path'],
        card_tables_file=assets['card_tables']['path'],native=native)
    params=jax.tree.map(jax.numpy.asarray,variables['params']);initial=S.parameter_digest(params);parent_initial=initial
    tx,forward,update=S.make_learner(agent,variables,learning_rate=cfg['learning_rate'],kl_coef=cfg['kl_coef'])
    sequence_steps=cfg.get('sequence_steps',64)
    if type(sequence_steps) is not int or sequence_steps not in (64,128,256):
        raise ValueError('register a supported whole-sequence training geometry')
    data=[]
    for row in rows:
        obs,labels=S.load_episode(row)
        obs,labels,valid,first=S.pad_episode(obs,labels,sequence_steps)
        importance=S.command_weights(row,sequence_steps,cfg.get('command_weight',1.))
        anchor=np.asarray(forward(params,obs,first))
        if not np.isfinite(anchor).all():raise ValueError('parent logits nonfinite')
        data.append((row,obs,labels,anchor,valid,first,importance))
    def evaluate(current):
        report=[]
        for row,obs,labels,anchor,valid,first,importance in data:
            logits=forward(current,obs,first)
            _,metrics=S.loss_from_logits(logits,anchor,labels,cfg['kl_coef'],valid,importance)
            prediction=np.asarray(logits).argmax(-1)
            command=valid&(S.command_weights(row,sequence_steps,2)>1)
            wrong=np.flatnonzero(valid&(prediction!=labels))
            wrong_command=np.flatnonzero(command&(prediction!=labels))
            report.append({'family':row['family'],'split':row['split'],'seed':row['seed'],
                'decisions':int(valid.sum()),**{k:float(v) for k,v in metrics.items()},
                'command_decisions':int(command.sum()),
                'command_top1':float((prediction[command]==labels[command]).mean()),
                'first_teacher_mismatch':int(wrong[0]) if len(wrong) else None,
                'first_command_mismatch':int(wrong_command[0]) if len(wrong_command) else None})
        return report
    opt=tx.init(params);prior_updates=0
    if cfg.get('resume') is not None:
        params,opt,prior_updates=S.load_specialization_resume(cfg['resume'],cfg,params,opt)
        initial=S.parameter_digest(params)
    before=evaluate(params);save('before.json',before)
    train=[x for x in data if x[0]['split']=='train'];started=time.monotonic();updates=[]
    def snapshot(current,current_opt,steps):
        import hashlib
        payload=flax.serialization.msgpack_serialize({'params':jax.device_get(current),
            'specialization_optimizer':flax.serialization.to_state_dict(jax.device_get(current_opt)),
            'specialization_updates':prior_updates+steps,'parent_checkpoint_sha256':assets['checkpoint']['sha256']})
        digest=hashlib.sha256(payload).hexdigest();path=args.out/(digest+'.specialization')
        if path.exists():
            if file_sha256(path)!=digest:raise ValueError('existing branch checkpoint changed')
        else:
            with path.open('xb') as f:f.write(payload)
        save('checkpoint-'+str(steps)+'.json',{'steps':steps,'optimizer_updates':prior_updates+steps,'path':str(path),'sha256':digest,
            'parent_checkpoint_sha256':assets['checkpoint']['sha256'],'A0_resume_admitted':False})
        return digest
    print('SPECIALIZATION-READY '+json.dumps({'parent':assets['checkpoint']['sha256'],'parent_counters':receipt['counters'],
        'train_episodes':len(train),'heldout_episodes':len(data)-len(train),'new_optimizer':not bool(prior_updates),
        'parent_optimizer_untouched':True,'all_actor_parameters_trainable':True,
        'specialization_optimizer_resumed':bool(prior_updates),'prior_specialization_updates':prior_updates,
        'command_weight':cfg.get('command_weight',1.),'sequence_steps':sequence_steps,
        'sampling_law':sampling_law,'schedule_sha256':schedule_sha256}),flush=True)
    for step in range(cfg['updates']):
        row,obs,labels,anchor,valid,first,importance=data[schedule[step]]
        candidate,next_opt,metrics=update(params,opt,obs,labels,anchor,valid,first,importance)
        metrics={k:float(v) for k,v in jax.device_get(metrics).items()}
        if not metrics['finite'] or any(not np.isfinite(v) for v in metrics.values()):raise ValueError('nonfinite update rejected')
        params,opt=candidate,next_opt
        record={'update':step+1,'optimizer_updates':prior_updates+step+1,'family':row['family'],'seed':row['seed'],**metrics}
        updates.append(record)
        print('SPECIALIZATION-UPDATE '+json.dumps(record),flush=True)
        if step==0 or (step+1)%len(train)==0:
            checkpoint=snapshot(params,opt,step+1)
            print('SPECIALIZATION-CHECKPOINT '+json.dumps({'update':step+1,'sha256':checkpoint}),flush=True)
        if time.monotonic()-started>cfg['max_train_seconds']:raise TimeoutError('original training budget exhausted')
    final=S.parameter_digest(params)
    if initial==final:raise ValueError('no actual parameter update')
    after=evaluate(params)
    last=args.out/('checkpoint-'+str(cfg['updates'])+'.json')
    digest=json.loads(last.read_bytes())['sha256'] if last.exists() else snapshot(params,opt,cfg['updates'])
    for row in assets.values():
        if file_sha256(row['path'])!=row['sha256']:raise ValueError('asset changed during fit')
    result={'schema':'mirrorforce_tactical_specialization_pilot/v1','complete':True,
        'parent':assets['checkpoint'],'parent_counters_unchanged':receipt['counters'],
        'parent_parameter_sha256':parent_initial,'starting_parameter_sha256':initial,
        'candidate_parameter_sha256':final,'updates':updates,
        'before':before,'after':after,'checkpoint_sha256':digest,'dataset':dataset,
        'on_policy_success':None,'playing_strength_evidence':False,'A0_resume_admitted':False,
        'seconds':time.monotonic()-started,'resume':cfg.get('resume'),
        'prior_specialization_updates':prior_updates,'total_specialization_updates':prior_updates+len(updates),
        'sampling_law':sampling_law,'schedule_sha256':schedule_sha256}
    save('result.json',result)
    print('SPECIALIZATION-COMPLETE '+json.dumps({'updates':len(updates),'checkpoint_sha256':digest}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
