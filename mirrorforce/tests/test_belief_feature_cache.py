import copy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from mirrorforce.agent.belief_feature_cache import BLOCKS, PendingPublicFeatures


def arrays():
    values = {}
    for i, block in enumerate(BLOCKS):
        n = 2 if block == 'candidates' else i + 3
        values[block] = np.full((n, 8), i + 1., np.float32)
        values['candidate_valid' if block == 'candidates' else block + '_valid'] = np.ones(n, bool)
    return values, np.array([[0, 1, 1], [0, 2, 1]], np.uint8)


def test_full_features_are_owned_read_only_and_cannot_be_unsealed_by_shared_clones():
    values, candidates = arrays()
    cache = PendingPublicFeatures(values, candidates, obs_sha256='a' * 64)
    identity = cache.identity
    actual, rows = cache.read(obs_sha256='a' * 64)
    assert identity['nbytes'] == sum(value.nbytes for value in values.values()) + candidates.nbytes
    for key in values:
        np.testing.assert_array_equal(values[key], actual[key])
        values[key].fill(0)
        assert actual[key].any()
        with pytest.raises(ValueError): actual[key].setflags(write=True)
        with pytest.raises(ValueError): actual[key].flat[0] = 0
    candidates[:] = 0
    assert rows[0, 1] == 1
    with pytest.raises(ValueError): rows.setflags(write=True)
    actual.pop('memory')
    assert 'memory' in cache.read(obs_sha256='a' * 64)[0]
    identity['shapes']['state'][0] = 1000
    assert cache.identity['shapes']['state'][0] == 3
    assert copy.deepcopy(cache) is cache
    with pytest.raises(FrozenInstanceError): cache._obs_sha256 = 'b' * 64
    with pytest.raises(ValueError, match='another pending'):
        cache.read(obs_sha256='b' * 64)


@pytest.mark.parametrize('fault', ['nan', 'int-features', 'mask-shape', 'mask-int', 'truth', 'missing',
                                 'candidate-mask', 'candidate-byte', 'candidate-int', 'bad-sha'])
def test_corrupt_or_private_features_rejected(fault):
    values, candidates = arrays()
    sha = 'a' * 64
    if fault == 'nan': values['memory'][0, 0] = np.nan
    elif fault == 'int-features': values['state'] = values['state'].astype(np.int32)
    elif fault == 'mask-shape': values['state_valid'] = values['state_valid'][:-1]
    elif fault == 'mask-int': values['state_valid'] = values['state_valid'].astype(np.int8)
    elif fault == 'truth': values['targets'] = np.zeros(4)
    elif fault == 'missing': del values['turn']
    elif fault == 'candidate-mask': candidates[0, 2] = 0
    elif fault == 'candidate-byte': candidates = candidates.astype(np.int32); candidates[0, 0] = 256
    elif fault == 'candidate-int': candidates = candidates.astype(np.float32)
    else: sha = 'A' * 64
    with pytest.raises(ValueError): PendingPublicFeatures(values, candidates, obs_sha256=sha)


def test_feature_digest_changes_with_any_data_or_geometry_but_not_dict_order():
    values, candidates = arrays()
    cache = PendingPublicFeatures(values, candidates, obs_sha256='a' * 64)
    same = PendingPublicFeatures(dict(reversed(list(values.items()))), candidates, obs_sha256='a' * 64)
    assert cache.identity == same.identity
    values['chunks'][0, 0] += 1
    changed = PendingPublicFeatures(values, candidates, obs_sha256='a' * 64)
    assert cache.identity['public_feature_sha256'] != changed.identity['public_feature_sha256']
