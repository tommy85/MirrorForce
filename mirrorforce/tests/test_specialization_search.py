"""Pure opt-in contract tests, not actor/GPU/native/current-game admission."""
import copy

import pytest

from mirrorforce.netduel import agent_specialization_search as S


def request():
    return {"schema":S.SCHEMA,"law":S.LAW,"behavior_request":{"path":"/branch/request.json","sha256":"a"*64},
            "parent_checkpoint_sha256":"b"*64,"branch_checkpoint_sha256":"c"*64,"parameter_sha256":"d"*64,
            "particles":copy.deepcopy(S.PARTICLE_IDENTITY),"inference_geometry":dict(S.FIXED_PUBLIC_B64),
            "budget":copy.deepcopy(S.BUDGET),"diagnostic_only":True,"production_admitted":False,
            "training_resume_admitted":False}


def identity():
    value=request()
    return {"backend":"specialization","weights":"iterate","checkpoint_sha256":"c"*64,
        "parent_checkpoint_sha256":"b"*64,"inference_geometry":dict(S.FIXED_PUBLIC_B64),
        "specialization":{"checkpoint_sha256":"c"*64,"parent_checkpoint_sha256":"b"*64,
                          "parameter_sha256":"d"*64,"request_sha256":"a"*64,"training_resume_admitted":False}}


def test_actual_branch_binding_is_explicit_without_relabeling_parent_or_admitting_production():
    ready=S.bind_ready(request(),identity(),request_sha256="e"*64)
    assert ready["branch_checkpoint_sha256"]=="c"*64 and ready["parent_checkpoint_sha256"]=="b"*64
    assert ready["production_admitted"] is False and ready["training_resume_admitted"] is False
    actor=identity();actor['specialization_search']=ready
    assert S.registered_service(actor)==ready


@pytest.mark.parametrize("fault",["ar_head","ar_features","replay_belief","wrong_actor","wrong_parent",
                                 "wrong_params","wrong_request","checkpoint_relabel","wrong_geometry"])
def test_old_or_relabelled_actor_head_and_geometry_are_rejected(fault):
    ready=identity()
    if fault in ("ar_head","ar_features","replay_belief"):ready[fault]={}
    elif fault=="wrong_actor":ready["checkpoint_sha256"]="f"*64
    elif fault=="wrong_parent":ready["parent_checkpoint_sha256"]="f"*64
    elif fault=="wrong_params":ready["specialization"]["parameter_sha256"]="f"*64
    elif fault=="wrong_request":ready["specialization"]["request_sha256"]="f"*64
    elif fault=="checkpoint_relabel":ready["backend"]="checkpoint"
    else:ready["inference_geometry"]={}
    with pytest.raises(ValueError):S.bind_ready(request(),ready,request_sha256="e"*64)


@pytest.mark.parametrize("fault",["extra_field","extra_trigger","longer_turn","less_reserve","ar_provider",
                                 "resume","production","parent_as_branch","bad_sha","relative_request"])
def test_no_budget_scope_or_authority_expansion(fault):
    value=request()
    if fault=="extra_field":value["ar_head"]={}
    elif fault=="extra_trigger":value["budget"]["triggers"].append("all")
    elif fault=="longer_turn":value["budget"]["hard_turn_seconds"]=601
    elif fault=="less_reserve":value["budget"]["reserve_seconds"]=2
    elif fault=="ar_provider":value["particles"]["provider"]="old-ar"
    elif fault=="resume":value["training_resume_admitted"]=True
    elif fault=="production":value["production_admitted"]=True
    elif fault=="parent_as_branch":value["branch_checkpoint_sha256"]=value["parent_checkpoint_sha256"]
    elif fault=="bad_sha":value["parameter_sha256"]="z"*64
    else:value["behavior_request"]["path"]="request.json"
    with pytest.raises(ValueError):S.checked(value)


def test_unpaired_new_cli_flag_does_not_load_any_model_or_unlock_default():
    from tools.mf_runtime_policy_service import main
    with pytest.raises(ValueError,match="complete explicit diagnostic"):
        main(['--checkpoint','missing','--weights','iterate','--specialization-search-request','missing',
              '--selection','greedy','--cards-db','missing','--code-list','missing','--script-root','missing',
              '--announce-tables','missing','--socket','missing','--out','missing'])


def test_ready_contract_itself_is_closed_and_cannot_disappear():
    actor=identity()
    with pytest.raises(ValueError,match='no explicit'):S.registered_service(actor)
    actor['specialization_search']=S.bind_ready(request(),actor,request_sha256='e'*64)
    actor['specialization_search']['extra']='unregistered'
    with pytest.raises(ValueError,match='undeclared'):S.registered_service(actor)


def test_registration_keeps_original_gate_and_full_candidate_geometry():
    from mirrorforce.netduel.agent_search_policy import SearchConfig,search_settings
    from mirrorforce.netduel.agent_search_gate import GateConfig
    from mirrorforce.netduel.agent_stripe_batching import LAW
    actor=identity();actor['specialization_search']=S.bind_ready(request(),actor,request_sha256='e'*64)
    config=SearchConfig(seconds=9.,total_seconds=12.,particles=8,selection='greedy',budget_law=LAW,
        candidate_law='all-legal-common-bank/v1',on_demand=GateConfig(enabled=True))
    settings=search_settings(config)
    assert S.registration(actor,S.PARTICLE_IDENTITY,settings)==actor['specialization_search']
    for key,value in [('particles',1),('rollouts',5),('seconds',12.)]:
        changed=copy.deepcopy(settings);changed[key]=value
        with pytest.raises(ValueError,match='unchanged'):S.registration(actor,S.PARTICLE_IDENTITY,changed)


def test_final_human_client_budget_keeps_original_service_receipt_and_request_immutable():
    from mirrorforce.netduel.final_search import settings
    actor=identity();actor['specialization_search']=S.bind_ready(request(),actor,request_sha256='e'*64)
    before=copy.deepcopy(actor)
    final=settings('on')
    assert S.registration(actor,S.PARTICLE_IDENTITY,final)==actor['specialization_search']
    assert actor==before and actor['specialization_search']['budget']==S.BUDGET
    assert final['seconds']==10. and final['total_seconds']==13.
    forged=request();forged['budget'].update(uncertain_seconds=13.,lethal_seconds=13.,soft_cap_seconds=10.)
    with pytest.raises(ValueError):S.checked(forged)
    actor['specialization']['parameter_sha256']='f'*64
    with pytest.raises(ValueError):S.registration(actor,S.PARTICLE_IDENTITY,final)


@pytest.mark.parametrize('fault',['soft_cap','prompt_cap','mixed_trigger','reduced_cleanup','longer_turn',
    'threshold','alpha','depth','particles','tactical','recipe','provider','extra'])
def test_final_human_profile_is_exact_and_never_accepts_nearby_settings(fault):
    from mirrorforce.netduel.final_search import settings
    actor=identity();actor['specialization_search']=S.bind_ready(request(),actor,request_sha256='e'*64)
    cfg=settings('on');particles=copy.deepcopy(S.PARTICLE_IDENTITY)
    if fault=='soft_cap':cfg['seconds']=11.
    elif fault=='prompt_cap':cfg['total_seconds']=14.
    elif fault=='mixed_trigger':cfg['on_demand']['uncertain_seconds']=8.
    elif fault=='reduced_cleanup':cfg['finalize_seconds']=2.
    elif fault=='longer_turn':cfg['on_demand']['turn_seconds']=31.
    elif fault=='threshold':cfg['on_demand']['entropy_threshold']=0.
    elif fault=='alpha':cfg['alpha']=.003
    elif fault=='depth':cfg['depth']=6
    elif fault=='particles':cfg['particles']=16
    elif fault=='tactical':cfg['tactical']={'enabled':True}
    elif fault=='recipe':cfg['follower_recipe_law']='unregistered'
    elif fault=='provider':particles['provider']='old-ar'
    else:cfg['waive_clock']=True
    with pytest.raises(ValueError):S.registration(actor,particles,cfg)
