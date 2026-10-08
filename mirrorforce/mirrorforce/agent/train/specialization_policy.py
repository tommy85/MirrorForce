"""Explicit behavior-only loader for a separately trained tactical actor.

It does not manufacture an A0 training checkpoint or replace the parent's
optimizer/EMA. Serving identities name the actual branch and its immutable
parent separately. Search/AR admission is a separate contract.
"""
import json
from pathlib import Path

from .checkpoint_store import file_sha256

SCHEMA='mirrorforce_specialization_behavior_request/v1'
PREVIEW_SCHEMA='mirrorforce_specialization_behavior_request/v2'


def read_ref(ref, *, limit=16*1024*1024):
    if type(ref) is not dict or set(ref)!={'path','sha256'}:
        raise ValueError('specialization requires an exact content reference')
    path=Path(ref['path'])
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size>limit:
        raise ValueError('invalid specialization file')
    if file_sha256(path)!=ref['sha256']:
        raise ValueError('specialization file changed')
    return path


def read_request(parent,reference):
    path=read_ref(reference)
    value=json.loads(path.read_bytes())
    preview=type(value) is dict and value.get('schema')==PREVIEW_SCHEMA
    fields={'schema','parent_checkpoint_sha256','branch','completed_result','training_config',
            'validation','deployment_kind','training_resume_admitted'}|({'preview_acknowledged'} if preview else set())
    if type(value) is not dict or set(value)!=fields:
        raise ValueError('closed specialization behavior schema required')
    mode='human_preview_with_known_failures' if preview else 'human_acceptance'
    if (value['schema'] not in (SCHEMA,PREVIEW_SCHEMA) or value['deployment_kind']!=mode or
            preview and value['preview_acknowledged'] is not True or
            value['training_resume_admitted'] is not False or file_sha256(parent)!=value['parent_checkpoint_sha256']):
        raise ValueError('specialization is behavior-only and must name its original parent')
    read_ref(value['branch'],limit=512*1024*1024)
    result=json.loads(read_ref(value['completed_result']).read_bytes())
    config=json.loads(read_ref(value['training_config']).read_bytes())
    validation=json.loads(read_ref(value['validation']).read_bytes())
    if (result.get('complete') is not True or result.get('checkpoint_sha256')!=value['branch']['sha256'] or
            config.get('assets',{}).get('checkpoint',{}).get('sha256')!=value['parent_checkpoint_sha256'] or
            result.get('A0_resume_admitted') is not False or result.get('dataset')!=config.get('dataset')):
        raise ValueError('specialization does not bind a completed independent training result')
    if (validation.get('schema')!='mirrorforce_specialization_human_acceptance_gate/v1' or
            validation.get('candidate_checkpoint_sha256')!=value['branch']['sha256'] or
            validation.get('complete') is not True or validation.get('known_issue_gate_passed') is not (not preview) or
            validation.get('teacher_forcing') is not False or validation.get('general_strength_proven') is not False):
        raise ValueError('specialization needs its autonomous known-issue acceptance evidence')
    if set(validation.get('checks',{}))!={'registered_lethals','no_premature_target_set',
            'no_unnecessary_set_then_activate','unused_field_stays_in_hand'} or any(
                type(flag) is not bool for flag in validation['checks'].values()) or (not preview and not all(validation['checks'].values())):
        raise ValueError('one or more explicitly requested behavior fixes remain unverified')
    import sqlite3
    db=read_ref(config['assets']['cards_db'])
    dataset=json.loads(read_ref(config['dataset']).read_bytes())
    from .sky_specialize import SCHEMA as DATASET_SCHEMA
    if dataset.get('schema')!=DATASET_SCHEMA:raise ValueError('registered training dataset schema differs')
    fields=('seed','family','split','start_lp','variant','finish_turn','trigger_resources',
            'defenses','spares','spare','first_draw','threat')
    expected=[{key:row[key] for key in fields if key in row} for row in dataset['episodes'] if row['split']=='heldout']
    with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as connection:
        card_types=dict(connection.execute('select id,type from datas'))
    assessed=verify_scene_evidence(validation.get('scene_reports'),request_actor_sha=value['branch']['sha256'],
                          card_types=card_types,expected_scenes=expected,allow_known_failures=preview)
    if preview and (not assessed['remaining_issues'] or
                    validation.get('remaining_issues')!=assessed['remaining_issues']):
        raise ValueError('preview must disclose the exact remaining issues from all original cases')
    return value,result,config


def verify_scene_evidence(references,*,request_actor_sha,card_types,expected_scenes=None,allow_known_failures=False):
    """Recompute this pilot's narrow gate from its SHA-bound autonomous originals."""
    from ...netduel import constants as C
    from ...netduel.actions import ActionAct as A
    if type(allow_known_failures) is not bool:raise ValueError('preview mode must be explicit')
    if type(references) is not list or len(references)<6:
        raise ValueError('autonomous scene originals are required, not just success flags')
    counts={'sword':0,'bomber':0,'main2_set':0};seeds=set();command_variants=set()
    issues=[]
    def issue(seed,reason):
        if not allow_known_failures:raise ValueError(reason)
        row={'seed':seed,'reason':reason}
        if row not in issues:issues.append(row)
    registered=None if expected_scenes is None else {row['seed']:row for row in expected_scenes}
    if registered is not None and (len(registered)!=len(expected_scenes) or len(references)!=len(registered)):
        raise ValueError('all predeclared heldout cases are required; no selection of successful cases')
    for ref in references:
        row=json.loads(read_ref(ref).read_bytes())
        if row.get('actor_checkpoint_sha256')!=request_actor_sha:
            raise ValueError('autonomous scene belongs to another actor')
        scene=row['scene'];family=scene['family'];seed=scene['seed']
        if registered is not None and (seed not in registered or
                any(scene.get(k)!=v for k,v in registered[seed].items())):
            raise ValueError('autonomous scenario differs from its predeclared heldout construction')
        if family=='backrow_timing':counts.setdefault(family,0)
        if family=='command_timing':counts.setdefault(family,0)
        if family not in counts or scene['split']!='heldout' or seed in seeds or row.get('teacher_forcing') is not False:
            raise ValueError('scene is not an independent heldout autonomous original')
        seeds.add(seed);counts[family]+=1
        if scene.get('variant')=='linked_clear' and (row.get('success') is not True or row.get('opponent_monsters_remaining')!=0):
            issue(seed,'registered clear was not completed')
        elif family not in ('main2_set','backrow_timing','command_timing') and scene.get('variant')!='linked_clear' and (row.get('terminal') is not True or row.get('winner')!=1):
            issue(seed,'registered lethal was not an actual win')
        if family=='main2_set' and scene['start_lp']-row['lp'][0]<7500:
            issue(seed,'registered nonlethal attack sequence was incomplete')
        if family=='command_timing':
            from .sky_command_timing import assess_command_scene
            command_variants.add(scene.get('variant'))
            for reason in assess_command_scene(scene,row['decisions'],row.get('draws',[]),row.get('own_hand',[]),row['lp']):
                issue(seed,reason)
        if family=='backrow_timing' and row.get('winner')!=1:
            from collections import Counter
            target=Counter(scene['defenses'])
            kept=Counter(x['code'] for x in row.get('own_backrow',[]))
            sets=Counter(x['action'].get('code') for x in row['decisions']
                         if x['action'].get('act')==int(A.SET) and x['phase']==C.PHASE_MAIN2)
            first_set=next((i for i,x in enumerate(row['decisions']) if x['action'].get('act')==int(A.SET)),0)
            attacked=any(x['action'].get('act') in (int(A.ATTACK),int(A.DIRECT_ATTACK)) for x in row['decisions'][:first_set])
            if not attacked or scene['start_lp']-row['lp'][0]<1500 or not target<=kept or not target<=sets:
                issue(seed,'ordinary attack then retained defensive main2 sets was incomplete')
        target_sets=[]
        for index,decision in enumerate(row['decisions']):
            action=decision['action']
            if action.get('act')==int(A.SET):
                # Defensive traps/quick spells remain valid after battle. This
                # gate targets the registered unneeded normal/field setting;
                # it is not an online action mask or a universal SET ban.
                card_type=card_types.get(action.get('code'),0)
                defensive=bool(card_type&0x4 or card_type&0x2 and card_type&0x10000)
                if not defensive or decision['phase']!=C.PHASE_MAIN2:
                    issue(seed,'unnecessary setting remains in the registered scene')
                if action.get('code')==98338152:target_sets.append(index)
        if family=='main2_set' and (not target_sets or not any(
                d['action'].get('act') in (int(A.ATTACK),int(A.DIRECT_ATTACK))
                for d in row['decisions'][:target_sets[0]])):
            issue(seed,'target quick spell was not set once after battle')
    if any(n<2 for n in counts.values()):
        raise ValueError('each registered behavior family needs two heldout cases')
    if counts.get('command_timing'):
        from .sky_command_timing import VARIANTS
        if command_variants!=set(VARIANTS):
            raise ValueError('command validation omitted a positive or negative variant')
    return {'families':counts,'remaining_issues':issues} if allow_known_failures else counts


def load_actor(parent,reference,variables,*,semantic_file,code_list_file,card_tables_file):
    import flax.serialization
    import jax
    import numpy as np
    from .sky_specialize import parameter_digest
    request,result,config=read_request(parent,reference)
    for key,path in [('semantic',semantic_file),('code_list',code_list_file),('card_tables',card_tables_file)]:
        if path is None or file_sha256(path)!=config['assets'][key]['sha256']:
            raise ValueError('specialization serving tables differ from training')
    branch=flax.serialization.msgpack_restore(read_ref(request['branch'],limit=512*1024*1024).read_bytes())
    if branch.get('parent_checkpoint_sha256')!=request['parent_checkpoint_sha256']:
        raise ValueError('specialization payload parent differs')
    before,after=variables['params'],branch['params']
    if jax.tree.structure(before)!=jax.tree.structure(after) or any(a.shape!=b.shape or a.dtype!=b.dtype
            for a,b in zip(jax.tree.leaves(before),jax.tree.leaves(after))):
        raise ValueError('specialization actor architecture changed')
    digest=parameter_digest(after)
    if digest!=result['candidate_parameter_sha256']:
        raise ValueError('specialization actor does not match completed training')
    if any(not np.isfinite(leaf).all() for leaf in jax.tree.leaves(after)):
        raise ValueError('specialization actor contains nonfinite parameters')
    preview=request['schema']==PREVIEW_SCHEMA
    validation=json.loads(read_ref(request['validation']).read_bytes())
    identity={'schema':request['schema']+'#identity','request_sha256':reference['sha256'],
        'checkpoint_sha256':request['branch']['sha256'],'parent_checkpoint_sha256':request['parent_checkpoint_sha256'],
        'parameter_sha256':digest,'specialization_updates':branch['specialization_updates'],
        'validation_sha256':request['validation']['sha256'],'training_resume_admitted':False,
        'deployment_kind':request['deployment_kind'],'known_issue_gate_passed':not preview,
        'remaining_issue_count':len(validation.get('remaining_issues',[])),
        'general_strength_proven':False}
    return {**variables,'params':after},identity
