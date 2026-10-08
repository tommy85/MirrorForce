from mirrorforce.effectinfo import get_effectinfo_core
from mirrorforce.agent.train.sky_backrow_timing import demonstrate_timing
from test_sky_combo_curriculum import deck

def scene(threat=True):
    return {'family':'backrow_timing','seed':2026119001,'start_lp':8000,'defenses':[98338152,84749824],
            'spares':[14558127,23434538],'first_draw':97268402,'threat':threat}

def test_delayed_sets_preserve_both_defenses_after_attacking():
    r=demonstrate_timing(get_effectinfo_core(),deck(),scene())
    sets=[x for x in r['choices'] if x['player']==1 and x['action']['act']==1]
    assert len(sets)==2 and all(x['phase']==256 for x in sets)
    assert r['lp'][0]==6500 and len(r['own_backrow'])==2 and r['turn']==3

def test_premature_sets_have_a_real_twin_twisters_counterexample():
    r=demonstrate_timing(get_effectinfo_core(),deck(),scene(),premature=True)
    assert len(r['own_backrow'])==0
    assert any(x['player']==0 and x['action']['code']==43898403 and x['action']['act']==8 for x in r['choices'])

def test_without_removal_early_sets_do_not_magically_disappear():
    r=demonstrate_timing(get_effectinfo_core(),deck(),scene(False),premature=True)
    assert len(r['own_backrow'])==2
