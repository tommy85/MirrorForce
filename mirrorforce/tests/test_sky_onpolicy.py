import pytest
from mirrorforce.agent.train.sky_onpolicy import validate_request


def test_only_public_wire_packets_cross_boundary():
    assert validate_request({'packets':[[5,'0102']]}) == [[5,'0102']]
    for key in ('seed', 'layout', 'target', 'teacher', 'snapshot'):
        with pytest.raises(ValueError):
            validate_request({'packets':[[5,'0102']], key:0})


@pytest.mark.parametrize('packets', [[], [[True,'00']], [[256,'00']], [[5,'AA']], [[5,'0']]])
def test_reject_malformed_packets(packets):
    with pytest.raises(ValueError):
        validate_request({'packets':packets})


@pytest.mark.parametrize('family,lp', [('sword',7400), ('bomber',1400)])
def test_autonomous_host_wire_boundary_matches_independent_teacher_fixture(family,lp):
    from mirrorforce.effectinfo import get_effectinfo_core
    from mirrorforce.agent.train.sky_combo_curriculum import demonstrate, BOMBER
    from mirrorforce.agent.train.sky_scene_host import run_scene
    from test_sky_combo_curriculum import deck
    scene = {'family':family, 'seed':2026112001, 'start_lp':lp}
    kwargs = {'finisher':BOMBER, 'opponent':'set_raye'} if family == 'bomber' else {}
    record = demonstrate(get_effectinfo_core(), deck(), seed=scene['seed'],
                         start_lp=lp, finish_lethal=True, **kwargs)
    tape = record['public_tape']
    responses = iter(tape['responses'])
    cursor = 0
    def answer(request):
        nonlocal cursor
        packets = validate_request(request)
        assert packets == tape['messages'][cursor:cursor+len(packets)]
        cursor += len(packets)
        return {'response':next(responses), 'decisions':[],
                'public_inputs_only':True, 'teacher_forcing':False}
    result = run_scene(get_effectinfo_core(), deck(), scene, answer)
    assert result['winner'] == 1 and result['terminal']
    assert result['responses'] == tape['responses']
    assert next(responses,None) is None


def test_autonomous_clear_host_replays_same_public_setup_without_using_own_teacher():
    from mirrorforce.effectinfo import get_effectinfo_core
    from mirrorforce.agent.train.sky_bomber_clear import demonstrate_clear
    from mirrorforce.agent.train.sky_scene_host import run_scene
    from test_sky_combo_curriculum import deck
    scene={'family':'bomber','variant':'linked_clear','seed':2026118001,'start_lp':2800,'trigger_resources':True}
    tape=demonstrate_clear(get_effectinfo_core(),deck(),seed=scene['seed'],start_lp=scene['start_lp'])['public_tape']
    answers=iter(tape['responses']);cursor=0
    def answer(request):
        nonlocal cursor
        packets=validate_request(request)
        assert packets==tape['messages'][cursor:cursor+len(packets)]
        cursor+=len(packets)
        return {'response':next(answers),'decisions':[],'public_inputs_only':True,'teacher_forcing':False}
    result=run_scene(get_effectinfo_core(),deck(),scene,answer)
    assert result['winner']==1 and result['success'] and result['opponent_monsters_remaining']==0
    assert not result['finisher_selected']  # no fake neural selection records in this protocol-only test
    assert result['goal_law'].endswith('/v2') and next(answers,None) is None


def test_autonomous_backrow_host_uses_same_ordinary_setup_and_counts_retained_defenses():
    from mirrorforce.effectinfo import get_effectinfo_core
    from mirrorforce.agent.train.sky_backrow_timing import demonstrate_timing
    from mirrorforce.agent.train.sky_scene_host import run_scene
    from test_sky_combo_curriculum import deck
    from test_sky_backrow_timing import scene
    s=scene();tape=demonstrate_timing(get_effectinfo_core(),deck(),s)['public_tape']
    answers=iter(tape['responses']);cursor=0
    def answer(request):
        nonlocal cursor
        packets=validate_request(request)
        assert packets==tape['messages'][cursor:cursor+len(packets)]
        cursor+=len(packets)
        return {'response':next(answers),'decisions':[],'public_inputs_only':True,'teacher_forcing':False}
    result=run_scene(get_effectinfo_core(),deck(),s,answer)
    assert result['lp'][0]==6500 and len(result['own_backrow'])==2
    assert not result['success']  # no invented neural SET records in this protocol-only fixture
    assert next(answers,None) is None


@pytest.mark.parametrize('variant',['idle_g_hold','own_turn_g_chain','opponent_turn_g_chain',
                                  'main1_upstart','engage_direct','main2_engage_value'])
def test_command_host_keeps_public_wire_boundary_and_does_not_invent_neural_success(variant):
    from mirrorforce.effectinfo import get_effectinfo_core
    from mirrorforce.agent.train.sky_command_timing import demonstrate_command_timing
    from mirrorforce.agent.train.sky_scene_host import run_scene
    from test_sky_combo_curriculum import deck
    s={'family':'command_timing','variant':variant,'seed':2026125001,'start_lp':8000,
       'defenses':[98338152,84749824],'spare':14558127,'first_draw':97268402}
    tape=demonstrate_command_timing(get_effectinfo_core(),deck(),s)['public_tape']
    answers=iter(tape['responses']);cursor=0
    def answer(request):
        nonlocal cursor
        packets=validate_request(request)
        assert packets==tape['messages'][cursor:cursor+len(packets)]
        cursor+=len(packets)
        return {'response':next(answers),'decisions':[],'public_inputs_only':True,'teacher_forcing':False}
    result=run_scene(get_effectinfo_core(),deck(),s,answer)
    assert result['turn']==4 and result['teacher_forcing'] is False
    assert result['command_issues'] and not result['success']  # no invented network decision records
    assert result['responses']==tape['responses'] and next(answers,None) is None
    if variant in ('own_turn_g_chain','opponent_turn_g_chain'):
        turn=2 if variant=='own_turn_g_chain' else 3
        draws=[x for x in result['draws'] if x['player']==1 and x['turn']==turn and x['phase']>=4]
        assert any({'code':26077387,'controller':0} in x['chain_sources'] and
                   {'code':23434538,'controller':1} in x['chain_sources'] for x in draws)
