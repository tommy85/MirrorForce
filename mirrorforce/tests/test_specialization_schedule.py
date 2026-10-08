import collections
import pytest
from mirrorforce.agent.train.sky_specialize import training_schedule


def rows():
    result=[{'seed':0,'split':'heldout','family':'bomber','variant':'linked_clear'}]
    for family in ('sword','bomber','main2_set'):
        result += [dict(seed=len(result)+j,split='train',family=family) for j in range(12)]
    result += [dict(seed=len(result)+j,split='train',family='bomber',variant='linked_clear') for j in range(8)]
    return result


def test_default_is_exact_old_whole_episode_order():
    assert training_schedule(rows(),88)==list(range(1,45))*2


def test_opt_in_is_reproducible_balanced_and_never_uses_heldout():
    r=rows();order=training_schedule(r,88,'clear-balanced/v1')
    assert order==training_schedule(r,88,'clear-balanced/v1') and len(order)==88
    assert 0 not in order and set(order)==set(range(1,45))
    assert all(r[i].get('variant')=='linked_clear' for i in order[1::2])
    assert [r[i]['family'] for i in order[::2]][:6]==['sword','bomber','main2_set']*2
    assert sum('variant' in r[i] for i in order)==44
    assert max(collections.Counter(order[1::2]).values())-min(collections.Counter(order[1::2]).values())<=1


@pytest.mark.parametrize('fault',['empty','duplicate','no_clear','missing_original','unknown_variant','unknown_law','bool_budget'])
def test_bad_schedule_is_rejected(fault):
    r=rows();law='clear-balanced/v1';updates=88
    if fault=='empty':r=r[:1]
    elif fault=='duplicate':r[2]['seed']=r[1]['seed']
    elif fault=='no_clear':r=r[:-8]
    elif fault=='missing_original':r=[x for x in r if x['family']!='sword']
    elif fault=='unknown_variant':r[1]['variant']='new_unregistered_variant'
    elif fault=='unknown_law':law='automatic_balance'
    else:updates=True
    with pytest.raises(ValueError):training_schedule(r,updates,law)


def test_backrow_schedule_retains_old_four_routes_and_never_samples_holdout():
    r=rows()+[{'seed':100+i,'family':'backrow_timing','split':'train'} for i in range(18)]
    r += [{'seed':1000,'family':'backrow_timing','split':'heldout'}]
    order=training_schedule(r,186,'backrow-balanced/v1')
    assert sum(r[i]['family']=='backrow_timing' for i in order)==93
    assert all(r[i]['split']=='train' for i in order)
    assert set(order)=={i for i,x in enumerate(r) if x['split']=='train'}


def command_rows():
    from mirrorforce.agent.train.sky_command_timing import VARIANTS
    r=rows()+[{'seed':100+i,'family':'backrow_timing','split':'train'} for i in range(6)]
    for vi,variant in enumerate(VARIANTS):
        r += [dict(seed=200+vi*10+i,family='command_timing',variant=variant,split='train') for i in range(3)]
        r.append(dict(seed=1000+vi,family='command_timing',variant=variant,split='heldout'))
    return r


def test_command_schedule_preserves_all_routes_and_excludes_holdout():
    from mirrorforce.agent.train.sky_command_timing import VARIANTS
    r=command_rows();order=training_schedule(r,360,'command-balanced/v1')
    assert order==training_schedule(r,360,'command-balanced/v1')
    assert all(r[i]['split']=='train' for i in order)
    assert all(r[i]['family']=='command_timing' for i in order[::2])
    assert all(r[i]['family']!='command_timing' for i in order[1::2])
    assert collections.Counter(r[i]['variant'] for i in order[::2])==dict.fromkeys(VARIANTS,30)
    assert set(order)=={i for i,x in enumerate(r) if x['split']=='train'}


@pytest.mark.parametrize('fault',['missing_positive','missing_negative','missing_old','unknown_variant'])
def test_command_schedule_rejects_incomplete_or_unregistered_routes(fault):
    r=command_rows()
    if fault=='missing_positive':r=[x for x in r if x.get('variant')!='own_turn_g_chain']
    elif fault=='missing_negative':r=[x for x in r if x.get('variant')!='idle_g_hold']
    elif fault=='missing_old':r=[x for x in r if x['family']!='backrow_timing']
    else:r.append(dict(seed=9999,family='command_timing',variant='unregistered',split='train'))
    with pytest.raises(ValueError,match='all six variants'):
        training_schedule(r,360,'command-balanced/v1')
