"""SHA-pinned, frozen AR head and deadline-bounded current-public proposal bridge.

An explicit diagnostic runtime, not a promotion of the historical training law
to search admission. No optimizer, actor parameters or teacher rows are applied.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import zlib

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from .belief_distribution_model import PublicDecoderScorer, PublicDecoderSnapshot
from .belief_feature_cache import PendingPublicFeatures
from .belief_features import public_for_layout
from .belief_online_draw import draw_bank
from .model.belief_ar import ARConfig, ARDecoder
from .search.belief_current_law import (CurrentPublicLayoutLaw, SCOPED_SCHEMA as PUBLIC_LAW,
                                       SCOPED_RESIDUAL_LAW as RESIDUAL_LAW)
from .search.belief_hand_scope import ASSIGNMENT_LAW, MAX_NODES

SCHEMA = 'mirrorforce_current_public_ar_inference/v1'
REQUEST_SCHEMA = SCHEMA + '#request'
MAGIC = b'MFBARCK1'
CAPACITY_KEYS = frozenset(('state', 'turn', 'chunks', 'memory', 'candidates', 'slots'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('AR inference requires lowercase SHA-256 bindings')
    return value


def read_request(path, sha256):
    from mirrorforce.common.sidecar_io import read_regular
    _sha(sha256)
    if Path(path).lstat().st_size > 65536:
        raise ValueError('AR inference request exceeds its bounded metadata size')
    payload = read_regular(path, 'AR inference request')
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError('AR inference request checksum differs')
    request = json.loads(payload)
    if type(request) is not dict or set(request) != {'schema', 'checkpoint', 'contract_sha256',
                                                    'parent_actor_sha256', 'diagnostic_only'} \
            or request['schema'] != REQUEST_SCHEMA or request['diagnostic_only'] is not True:
        raise ValueError('AR inference requires its explicit diagnostic-only request')
    _sha(request['contract_sha256']); _sha(request['parent_actor_sha256'])
    return request


def load_head(reference, *, parent_actor_sha256, contract_sha256, max_bytes=128 << 20):
    """Read only named numeric params after independently checking the entire original artifact."""
    from mirrorforce.common.sidecar_io import read_regular
    if type(reference) is not dict or set(reference) != {'path', 'sha256'} \
            or type(max_bytes) is not int or not 0 < max_bytes <= 1 << 30:
        raise ValueError('AR inference needs a closed checkpoint reference and explicit byte bound')
    _sha(reference['sha256']); _sha(parent_actor_sha256); _sha(contract_sha256)
    path = Path(reference['path'])
    if path.lstat().st_size > max_bytes + max_bytes // 512 + 32:
        raise ValueError('AR inference checkpoint exceeds stored-byte bound')
    payload = read_regular(path, 'AR inference checkpoint')
    if hashlib.sha256(payload).hexdigest() != reference['sha256'] or payload[:8] != MAGIC or len(payload) <= 16:
        raise ValueError('AR inference checkpoint checksum or magic differs')
    size = int.from_bytes(payload[8:16], 'big')
    if not 0 < size <= max_bytes:
        raise ValueError('AR inference checkpoint exceeds uncompressed-byte bound')
    decoder = zlib.decompressobj()
    raw = decoder.decompress(payload[16:], size + 1)
    if len(raw) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('AR inference checkpoint compressed stream/length differs')
    data = serialization.msgpack_restore(raw)
    if type(data) is not dict or set(data) != {'meta_json', 'head', 'row_order'}:
        raise ValueError('AR inference checkpoint fields differ')
    meta = json.loads(data['meta_json'])
    if type(meta) is not dict or set(meta) != {'schema', 'contract', 'jax_version', 'numpy_version'} \
            or meta['schema'] != 'mirrorforce_offline_ar_head_checkpoint/v1' \
            or data['meta_json'] != json.dumps(meta, sort_keys=True, separators=(',', ':'), allow_nan=False):
        raise ValueError('AR inference checkpoint metadata is not its canonical training record')
    contract = meta['contract']
    if digest(contract) != contract_sha256 or contract.get('parent_actor_sha256') != parent_actor_sha256:
        raise ValueError('AR head was not trained on the pinned actor/public-feature contract')
    if type(data['head']) is not dict or set(data['head']) != {'params', 'optimizer', 'dropout_key', 'updates', 'batches', 'tokens'}:
        raise ValueError('AR head state fields differ from the original complete checkpoint')
    model = ARDecoder(ARConfig(**contract['model']))
    capacities = contract.get('capacities')
    if type(capacities) is not dict or set(capacities) != CAPACITY_KEYS \
            or any(type(n) is not int or not 1 <= n <= 512 for n in capacities.values()) \
            or capacities['slots'] > model.config.max_targets or capacities['candidates'] > 128:
        raise ValueError('AR inference requires the original complete public padding capacities')
    params = data['head']['params']
    public = {}
    for name, count in capacities.items():
        projection = 'candidate_projection' if name == 'candidates' else 'slot_projection' if name == 'slots' else name + '_projection'
        kernel = params[projection]['kernel']
        if kernel.ndim != 2 or not 1 <= kernel.shape[0] <= 4096:
            raise ValueError('AR input projection width is invalid')
        public[name] = jax.ShapeDtypeStruct((1, count, kernel.shape[0]), jnp.float32)
        mask = 'candidate_valid' if name == 'candidates' else 'slot_valid' if name == 'slots' else name + '_valid'
        public[mask] = jax.ShapeDtypeStruct((1, count), jnp.bool_)
    previous = jax.ShapeDtypeStruct((1, capacities['slots']), jnp.int32)
    template = jax.eval_shape(lambda x, p: model.init(jax.random.PRNGKey(0), x, p, deterministic=True)['params'],
                              public, previous)
    if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(template):
        raise ValueError('AR inference parameter tree differs from the declared decoder')
    for actual, expected in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(template)):
        if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.isfinite(actual).all():
            raise ValueError('AR inference parameter shape/dtype/finiteness differs')
    identity = {'checkpoint_sha256': reference['sha256'], 'training_contract_sha256': contract_sha256,
                'parent_actor_sha256': parent_actor_sha256, 'model': asdict(model.config),
                'capacities': capacities, 'training_runtime': {k: meta[k] for k in ('jax_version', 'numpy_version')},
                'training_updates': int(data['head']['updates']), 'training_batches': int(data['head']['batches']),
                'training_tokens': int(data['head']['tokens']), 'optimizer_applied': False,
                'training_contract_search_admission': contract.get('search_admission', False)}
    return model, params, contract, identity


class ARHeadRuntime:
    def __init__(self, reference, *, parent_actor_sha256, contract_sha256, code_by_id, request_clock=None):
        from ..netduel.ar_clock import from_head
        request_clock = None if request_clock is None else from_head({"request_clock": request_clock})
        model, params, contract, identity = load_head(reference, parent_actor_sha256=parent_actor_sha256,
                                                     contract_sha256=contract_sha256)
        batch = contract.get('micro_batch')
        if type(batch) is not int or not 1 <= batch <= 128:
            raise ValueError('AR inference must explicitly retain its original decoder microbatch size')
        if type(code_by_id) is not dict or any(type(i) is not int or i < 1 or type(code) is not int or code < 1
                                             for i, code in code_by_id.items()):
            raise ValueError('AR runtime needs the pinned actor public code dictionary')
        self.model, self.params, self.batch_size = model, params, batch
        self.capacities, self.code_by_id = dict(contract['capacities']), dict(code_by_id)
        self.snapshot = PublicDecoderSnapshot(model, params)
        self._identity = {**identity, 'schema': SCHEMA, 'inference_batch_size': batch, 'public_law': PUBLIC_LAW,
                          'residual_law': RESIDUAL_LAW, 'compute_dtype': 'float32',
                          'hand_physical_law': ASSIGNMENT_LAW,
                          'actual_parameter_tree_sha256': self.snapshot.identity['actual_parameter_tree_sha256'],
                          'diagnostic_only': True, 'teacher_channels': False, 'search_admission': False}
        if request_clock is not None:
            self._identity['request_clock'] = request_clock

    @property
    def identity(self):
        return json.loads(json.dumps(self._identity, allow_nan=False))

    def warmup(self):
        """Compile the exact full AR geometry on synthetic zeros before accepting timed requests."""
        public = {}
        for name, capacity in self.capacities.items():
            projection = 'candidate_projection' if name == 'candidates' else 'slot_projection' if name == 'slots' else name + '_projection'
            public[name] = np.zeros((capacity, self.params[projection]['kernel'].shape[0]), np.float32)
            mask = 'candidate_valid' if name == 'candidates' else 'slot_valid' if name == 'slots' else name + '_valid'
            public[mask] = np.zeros(capacity, bool)
        public['candidate_valid'][0] = public['slot_valid'][0] = True
        scorer = PublicDecoderScorer(self.model, self.params, public, codes=(1,), target_length=1,
                                    batch_size=self.batch_size, snapshot=self.snapshot)
        scorer.prefix_batch(((),))
        return {'synthetic_zero_features': True, 'training_steps': 0, 'search_admission': False,
                'fixed_batch_size': self.batch_size, 'capacities': dict(self.capacities)}

    def scorer(self, cache, law, *, obs_sha256):
        if type(cache) is not PendingPublicFeatures:
            raise ValueError('AR inference reads only an owned pending-public-feature snapshot')
        features, candidates = cache.read(obs_sha256=obs_sha256)
        original = public_for_layout(features, candidates, self.code_by_id, law)
        public = {}
        for name, capacity in self.capacities.items():
            mask = 'candidate_valid' if name == 'candidates' else 'slot_valid' if name == 'slots' else name + '_valid'
            values, valid = original[name][0], original[mask][0]
            if values.shape[0] > capacity:
                raise ValueError('current public features exceed the original AR capacity; no truncation')
            public[name] = np.zeros((capacity, values.shape[1]), np.float32)
            public[mask] = np.zeros(capacity, bool)
            public[name][:len(values)], public[mask][:len(valid)] = values, valid
        return PublicDecoderScorer(self.model, self.params, public, codes=law.codes, target_length=law.length,
                                   batch_size=self.batch_size, snapshot=self.snapshot)

    def draw(self, cache, specification, *, obs_sha256, count, seed, deadline, max_nodes=MAX_NODES):
        if type(specification) is not dict or specification.get('schema') != PUBLIC_LAW \
                or specification.get('hand_scope', {}).get('obs_sha256') != obs_sha256:
            raise ValueError('AR proposals require the exact same-pending scoped hand law')
        law = CurrentPublicLayoutLaw(specification, deadline=deadline, max_nodes=max_nodes)
        scorer = self.scorer(cache, law, obs_sha256=obs_sha256)
        bank = draw_bank(law, scorer, count=count, seed=seed, deadline=deadline)
        residual_rngs = [np.random.default_rng(child) for child in np.random.SeedSequence([seed, 0x41525245]).spawn(count)]
        layouts = [law.complete(tuple(sequence), rng) for sequence, rng in zip(bank['sequences'], residual_rngs)]
        result = {'schema': SCHEMA + '#bank', 'obs_sha256': obs_sha256, 'public_specification_sha256': digest(specification),
                  'feature_cache': cache.identity, 'head': self.identity, 'draws': bank, 'layouts': layouts,
                  'feasibility_nodes': law.nodes, 'max_nodes': max_nodes,
                  'teacher_channels': False, 'search_admission': False}
        result['sha256'] = digest(result)
        law._check()
        return result
