import json
from pathlib import Path
import pytest
from mirrorforce.agent.train import specialization_policy as S
from mirrorforce.netduel.actions import ActionAct as A

TYPES={98338152:0x10002,50005218:0x80002,41420027:4}


@pytest.mark.parametrize('fault',[None,'omit_positive','omit_negative','own_free_g'])
def test_command_gate_reads_every_registered_positive_and_negative(tmp_path,fault):
    from test_command_timing_assessment import assessment,row
    from mirrorforce.agent.train.sky_command_timing import VARIANTS,MAXX_C
    reports=scenes(tmp_path)
    for i,variant in enumerate(VARIANTS):
        if fault=='omit_positive' and variant=='own_turn_g_chain':continue
        if fault=='omit_negative' and variant=='idle_g_hold':continue
        scene,decisions,draws,hand,lp=assessment(variant)
        if fault=='own_free_g' and variant=='own_turn_g_chain':decisions.append(row(MAXX_C,A.ACTIVATE))
        scene.update(family='command_timing',split='heldout',seed=300+i)
        reports.append(ref(tmp_path/f'command-{i}.json',dict(actor_checkpoint_sha256='a'*64,
            scene=scene,teacher_forcing=False,terminal=False,winner=None,lp=lp,
            decisions=decisions,draws=draws,own_hand=hand)))
    if fault:
        with pytest.raises(ValueError):S.verify_scene_evidence(reports,request_actor_sha='a'*64,card_types=TYPES)
    else:assert S.verify_scene_evidence(reports,request_actor_sha='a'*64,card_types=TYPES)['command_timing']==6


def ref(path,value):
    path.write_text(json.dumps(value))
    return {'path':str(path),'sha256':S.file_sha256(path)}


def scenes(tmp_path,actor='a'*64):
    result=[]
    for i,family in enumerate(('sword','sword','bomber','bomber','main2_set','main2_set')):
        row={'actor_checkpoint_sha256':actor,'scene':{'family':family,'split':'heldout','seed':100+i,'start_lp':8000},
            'teacher_forcing':False,'finisher_selected':True,'terminal':family!='main2_set',
            'winner':1 if family!='main2_set' else None,'lp':[500,8000],
            'decisions':[{'phase':8,'action':{'act':int(A.DIRECT_ATTACK),'code':85289965}}]}
        if family=='main2_set':row['decisions'].append({'phase':256,'action':{'act':int(A.SET),'code':98338152}})
        result.append(ref(tmp_path/f'scene-{i}.json',row))
    return result


def test_known_issue_gate_rechecks_originals_not_only_summary_flags(tmp_path):
    reports=scenes(tmp_path)
    assert S.verify_scene_evidence(reports,request_actor_sha='a'*64,card_types=TYPES)=={'sword':2,'bomber':2,'main2_set':2}
    path=tmp_path/'scene-4.json';value=json.loads(path.read_bytes())
    value['decisions'].append({'phase':256,'action':{'act':int(A.SET),'code':50005218}})
    reports[4]=ref(path,value)
    with pytest.raises(ValueError,match='unnecessary setting'):S.verify_scene_evidence(reports,request_actor_sha='a'*64,card_types=TYPES)


@pytest.mark.parametrize('fault',['teacher','no_win','no_battle','early_set','missing_family','duplicate'])
def test_known_issue_gate_rejects_incomplete_or_mismatched_cases(tmp_path,fault):
    reports=scenes(tmp_path);path=tmp_path/'scene-4.json';value=json.loads(path.read_bytes())
    if fault=='teacher':value['teacher_forcing']=True
    elif fault=='no_win':
        path=tmp_path/'scene-0.json';value=json.loads(path.read_bytes());value['winner']=None
    elif fault=='no_battle':value['decisions']=value['decisions'][1:]
    elif fault=='early_set':value['decisions'][-1]['phase']=4
    elif fault=='missing_family':reports=reports[:4]
    else:reports[-1]=reports[-2]
    if fault in ('teacher','no_win','no_battle','early_set'):reports[0 if fault=='no_win' else 4]=ref(path,value)
    with pytest.raises(ValueError):S.verify_scene_evidence(reports,request_actor_sha='a'*64,card_types=TYPES)


def test_specialization_cli_rejects_search_before_loading_any_model():
    from tools.mf_runtime_policy_service import main
    with pytest.raises(ValueError,match='search admission is separate'):
        main(['--checkpoint','missing','--weights','iterate','--specialization-request','missing',
          '--specialization-request-sha256','0'*64,'--current-root-search','--selection','greedy',
          '--cards-db','missing','--code-list','missing','--script-root','missing',
          '--announce-tables','missing','--socket','missing','--out','missing'])


def test_behavior_actor_is_explicit_and_never_rewrites_parent_variables(tmp_path):
    import numpy as np
    import flax.serialization
    import sqlite3
    from mirrorforce.agent.train.sky_specialize import parameter_digest
    parent=tmp_path/'parent';parent.write_bytes(b'original immutable parent')
    parent_sha=S.file_sha256(parent)
    candidate={'w':np.array([3.,4.],np.float32)}
    branch=tmp_path/'branch';branch.write_bytes(flax.serialization.msgpack_serialize({
        'params':candidate,'parent_checkpoint_sha256':parent_sha,'specialization_updates':174}))
    branch_ref={'path':str(branch),'sha256':S.file_sha256(branch)}
    assets={'checkpoint':{'sha256':parent_sha}}
    for key in ('semantic','code_list','card_tables'):
        p=tmp_path/key;p.write_bytes(key.encode());assets[key]={'path':str(p),'sha256':S.file_sha256(p)}
    db=tmp_path/'cards.cdb'
    with sqlite3.connect(db) as connection:
        connection.execute('create table datas (id integer, type integer)')
        connection.executemany('insert into datas values (?,?)',TYPES.items())
    assets['cards_db']={'path':str(db),'sha256':S.file_sha256(db)}
    reports=scenes(tmp_path,branch_ref['sha256'])
    from mirrorforce.agent.train.sky_specialize import SCHEMA as DATASET_SCHEMA
    dataset=ref(tmp_path/'dataset.json',{'schema':DATASET_SCHEMA,'episodes':[
        json.loads(Path(r['path']).read_bytes())['scene'] for r in reports]})
    config=ref(tmp_path/'config.json',{'assets':assets,'dataset':dataset})
    result=ref(tmp_path/'result.json',{'complete':True,'checkpoint_sha256':branch_ref['sha256'],
        'A0_resume_admitted':False,'candidate_parameter_sha256':parameter_digest(candidate),'dataset':dataset})
    validation=ref(tmp_path/'validation.json',{'schema':'mirrorforce_specialization_human_acceptance_gate/v1',
        'candidate_checkpoint_sha256':branch_ref['sha256'],'complete':True,'known_issue_gate_passed':True,
        'teacher_forcing':False,'general_strength_proven':False,'scene_reports':reports,
        'checks':dict.fromkeys(('registered_lethals','no_premature_target_set',
            'no_unnecessary_set_then_activate','unused_field_stays_in_hand'),True)})
    request=ref(tmp_path/'request.json',{'schema':S.SCHEMA,'parent_checkpoint_sha256':parent_sha,
        'branch':branch_ref,'completed_result':result,'training_config':config,'validation':validation,
        'deployment_kind':'human_acceptance','training_resume_admitted':False})
    original={'params':{'w':np.array([1.,2.],np.float32)},'constants':{'frozen':3}}
    loaded,identity=S.load_actor(parent,request,original,semantic_file=assets['semantic']['path'],
        code_list_file=assets['code_list']['path'],card_tables_file=assets['card_tables']['path'])
    np.testing.assert_array_equal(loaded['params']['w'],candidate['w'])
    np.testing.assert_array_equal(original['params']['w'],[1.,2.])
    assert loaded['constants'] is original['constants']
    assert identity['checkpoint_sha256']==branch_ref['sha256'] and identity['parent_checkpoint_sha256']==parent_sha
    assert identity['training_resume_admitted'] is False


def test_defensive_traps_after_battle_are_not_mistaken_for_normal_spell_misplay(tmp_path):
    reports=scenes(tmp_path);path=tmp_path/'scene-4.json';value=json.loads(path.read_bytes())
    value['decisions'].append({'phase':256,'action':{'act':int(A.SET),'code':41420027}})
    reports[4]=ref(path,value)
    assert S.verify_scene_evidence(reports,request_actor_sha='a'*64,card_types=TYPES)['main2_set']==2


def test_an_alternative_native_lethal_is_not_rejected_for_using_another_finisher(tmp_path):
    reports=scenes(tmp_path);path=tmp_path/'scene-2.json';value=json.loads(path.read_bytes())
    value['finisher_selected']=False
    reports[2]=ref(path,value)
    assert S.verify_scene_evidence(reports,request_actor_sha='a'*64,card_types=TYPES)['bomber']==2


def test_gate_cannot_drop_failed_registered_scenes_or_change_initial_lp(tmp_path):
    reports=scenes(tmp_path);expected=[json.loads(Path(r['path']).read_bytes())['scene'] for r in reports]
    S.verify_scene_evidence(reports,request_actor_sha='a'*64,card_types=TYPES,expected_scenes=expected)
    with pytest.raises(ValueError,match='all predeclared'):
        S.verify_scene_evidence(reports,request_actor_sha='a'*64,card_types=TYPES,
                               expected_scenes=expected+[{**expected[0],'seed':999}])
    expected[0]={**expected[0],'start_lp':1000}
    with pytest.raises(ValueError,match='construction'):
        S.verify_scene_evidence(reports,request_actor_sha='a'*64,card_types=TYPES,expected_scenes=expected)


@pytest.mark.parametrize('fault',[None,'old_schema','false_ack','omitted_issues','wrong_actor','missing_case'])
def test_known_failure_preview_requires_explicit_ack_and_complete_evidence(tmp_path,fault):
    test_behavior_actor_is_explicit_and_never_rewrites_parent_variables(tmp_path)
    request=json.loads((tmp_path/'request.json').read_bytes())
    validation=json.loads((tmp_path/'validation.json').read_bytes())
    scene=json.loads((tmp_path/'scene-0.json').read_bytes())
    scene['winner']=None
    validation['scene_reports'][0]=ref(tmp_path/'scene-0.json',scene)
    validation['known_issue_gate_passed']=False
    validation['checks']['registered_lethals']=False
    validation['remaining_issues']=[{'seed':100,'reason':'registered lethal was not an actual win'}]
    request.update(schema=S.PREVIEW_SCHEMA,deployment_kind='human_preview_with_known_failures',preview_acknowledged=True)
    if fault=='old_schema':
        request.update(schema=S.SCHEMA,deployment_kind='human_acceptance');del request['preview_acknowledged']
    elif fault=='false_ack':request['preview_acknowledged']=False
    elif fault=='omitted_issues':validation['remaining_issues']=[]
    elif fault=='wrong_actor':
        scene['actor_checkpoint_sha256']='f'*64
        validation['scene_reports'][0]=ref(tmp_path/'scene-0.json',scene)
    elif fault=='missing_case':validation['scene_reports'].pop()
    request['validation']=ref(tmp_path/'validation.json',validation)
    reference=ref(tmp_path/'request.json',request)
    if fault:
        with pytest.raises(ValueError):S.read_request(tmp_path/'parent',reference)
    else:
        assert S.read_request(tmp_path/'parent',reference)[0]['training_resume_admitted'] is False


@pytest.mark.parametrize('fault',[None,'early','destroyed','no_attack','no_recorded_attack'])
def test_ordinary_backrow_gate_requires_damage_and_retained_post_battle_sets(tmp_path,fault):
    reports=scenes(tmp_path)
    for i in range(2):
        row={'actor_checkpoint_sha256':'a'*64,'scene':{'family':'backrow_timing','split':'heldout',
              'seed':200+i,'start_lp':8000,'defenses':[98338152,41420027]},
             'teacher_forcing':False,'winner':None,'terminal':False,'lp':[6500,8000],
             'own_backrow':[{'code':98338152},{'code':41420027}],
             'decisions':[{'phase':8,'action':{'act':int(A.DIRECT_ATTACK),'code':8491308}},
                          {'phase':256,'action':{'act':int(A.SET),'code':98338152}},
                          {'phase':256,'action':{'act':int(A.SET),'code':41420027}}]}
        if i==0:
            if fault=='early':row['decisions'][1]['phase']=4
            elif fault=='destroyed':row['own_backrow']=[]
            elif fault=='no_attack':row['lp'][0]=8000
            elif fault=='no_recorded_attack':row['decisions'].pop(0)
        reports.append(ref(tmp_path/f'timing-{i}.json',row))
    if fault:
        with pytest.raises(ValueError):S.verify_scene_evidence(reports,request_actor_sha='a'*64,card_types=TYPES)
    else:assert S.verify_scene_evidence(reports,request_actor_sha='a'*64,card_types=TYPES)['backrow_timing']==2
