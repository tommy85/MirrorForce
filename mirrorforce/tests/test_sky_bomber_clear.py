import pytest
from mirrorforce.effectinfo import get_effectinfo_core
from mirrorforce.agent.train.sky_bomber_clear import demonstrate_clear
from test_sky_combo_curriculum import deck


@pytest.mark.parametrize('lp',[2500,3000,3500])
def test_bomber_extra_zone_survives_linked_summon_board_wipe_and_direct_attacks(lp):
    r=demonstrate_clear(get_effectinfo_core(),deck(),seed=2026116001,start_lp=lp)
    before=r['before_trigger'];after=r['before_battle']
    assert len(before['opponent_field'])==2 and not before['opponent_backrow']
    assert all(c['position'] in (4,8) and c['sequence']<5 for c in before['opponent_field'])
    assert not after['opponent_field']
    assert len(after['own_field'])==1 and after['own_field'][0]['code']==5821478
    assert after['own_field'][0]['sequence'] in (5,6)
    assert r['lp'][0]==lp-3000 and (r['winner']==1)==(lp<=3000)


def test_no_known_trigger_resource_does_not_get_a_free_board_wipe():
    r=demonstrate_clear(get_effectinfo_core(),deck(),seed=2026116001,start_lp=3000,trigger_resources=False)
    assert r['before_trigger'] is None
    assert len(r['before_battle']['opponent_field'])==2
    assert r['winner'] is None and r['lp'][0]>0
