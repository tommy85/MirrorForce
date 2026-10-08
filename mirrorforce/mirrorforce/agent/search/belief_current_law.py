"""Current-public slot constraints for direct AR proposals; no history reconstruction.

Exact feasibility uses public inventory flow plus bounded existential witnesses
for position-free facts. Unknown/expired work raises, never broadens support.
This is a distinct law from the offline teacher history law and the uniform
sampler. It grants neither checkpoint nor native realization admission.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import time

import numpy as np

from .belief_ar_law import transport_feasible

SCHEMA = "mirrorforce_current_public_slot_law/v1"
SCOPED_SCHEMA = "mirrorforce_current_public_slot_law/v2"
RESIDUAL_LAW = "sequential-public-feasible-copy-weighted-residual/v1"
SCOPED_RESIDUAL_LAW = "canonical-hand-multiset;sequential-public-feasible-copy-weighted-other-zones/v2"


class CurrentLawBudgetExceeded(TimeoutError):
    pass


@dataclass(frozen=True)
class Slot:
    location: int
    sequence: int


def _int(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("current public law requires non-boolean bounded integers")
    return value


class CurrentPublicLayoutLaw:
    """Canonical unknown field identities, then sorted undisclosed hand cards.

    Public hand identities without positions occupy canonical free hand slots,
    as in the existing writer. They are not additional unknown AR targets.
    ``lower`` facts may share witnesses with ``categories``; categories need
    distinct physical cards. ``any`` clauses may share witnesses with each
    other. A fresh field slot is excluded from old identity facts by its scope.
    """
    def __init__(self, specification, *, deadline, max_nodes=50000, clock=time.monotonic,
                 require_feasible=True):
        if type(deadline) not in (int, float) or not math.isfinite(deadline) \
                or type(max_nodes) is not int or not 1 <= max_nodes <= 1000000 or not callable(clock) \
                or type(require_feasible) is not bool:
            raise ValueError("current public law requires finite deadline and bounded feasibility work")
        self.deadline, self.max_nodes, self.clock, self.nodes = deadline, max_nodes, clock, 0
        self._check()
        spec = deepcopy(specification)
        fields = {'schema', 'pools', 'slots', 'lower', 'categories', 'any', 'field_targets', 'hand_targets'}
        scoped = type(spec) is dict and spec.get('schema') == SCOPED_SCHEMA
        if type(spec) is not dict or set(spec) != fields | ({'hand_scope'} if scoped else set()) \
                or spec['schema'] not in (SCHEMA, SCOPED_SCHEMA):
            raise ValueError("current public law accepts only its closed public constraint specification")
        declared_spec = deepcopy(spec)
        self.hand_scope = None
        if scoped:
            from .belief_hand_scope import PublicHandScope
            self.hand_scope = PublicHandScope(spec['hand_scope'])
        if type(spec['pools']) is not list or len(spec['pools']) != 2:
            raise ValueError("current public law requires separate main and extra inventories")
        pools = []
        for pool in spec['pools']:
            if type(pool) is not list or any(type(row) is not list or len(row) != 2 for row in pool):
                raise ValueError("public inventory rows must be code/count pairs")
            if pool != sorted(pool) or len({row[0] for row in pool}) != len(pool):
                raise ValueError("public inventory must have unique sorted card codes")
            pools.append({_int(code, 1): _int(count, 1) for code, count in pool})
        self.codes = tuple(sorted(set(pools[0]) | set(pools[1])))
        if len(self.codes) > 128 or set(pools[0]) & set(pools[1]):
            raise ValueError("public inventory has too many codes or ambiguous pool membership")
        self.index = {code: i for i, code in enumerate(self.codes)}
        self.counts = tuple(sum(pool.get(code, 0) for pool in pools) for code in self.codes)
        slots = spec['slots']
        if type(slots) is not list or len(slots) > 160 or len(slots) != sum(self.counts):
            raise ValueError("public slots must exactly fill bounded public inventories")
        self.domains, self.positions = [], []
        for row in slots:
            if type(row) is not list or len(row) != 4:
                raise ValueError("public slots are location, sequence, pool, allowed-code rows")
            location, sequence, pool, allowed = row
            _int(location); _int(sequence); _int(pool)
            if location not in (1, 2, 4, 8, 32, 64) or pool not in (0, 1) \
                    or type(allowed) is not list or allowed != sorted(set(allowed)) or not allowed \
                    or any(type(code) is not int or code not in pools[pool] for code in allowed):
                raise ValueError("public slot domain differs from its explicit inventory")
            if location == 4 and sequence >= 7 or location == 8 and sequence >= 8:
                raise ValueError("public field slot outside board geometry")
            self.domains.append(frozenset(self.index[code] for code in allowed))
            self.positions.append((location, sequence))
        if len(set(self.positions)) != len(slots):
            raise ValueError("public slot positions must be unique")
        self.domains = tuple(self.domains)
        self.positions = tuple(self.positions)
        if scoped:
            hand_slots = [i for i, position in enumerate(self.positions) if position[0] == 2]
            if sorted(self.positions[i][1] for i in hand_slots) != list(range(self.hand_scope.size)):
                raise ValueError("public hand scope differs from the complete current hand geometry")
            if any(code not in pools[0] for code in self.hand_scope.minimum):
                raise ValueError("public hand scope identity is absent from the current main inventory")
            if type(spec['lower']) is not list or len(spec['lower']) > 128:
                raise ValueError("declared public witness facts must be bounded lists")
            # Hand AR tokens remain a sorted multiset, NOT physical locations.
            # For one anonymous old group, fixed anchors plus its residual
            # lower bounds reduce EXACTLY to these total-hand requirements.
            # Physical arrangements are sampled/checked separately by UID.
            # Do not insert a partial physical scope or bypass exchangeability.
            spec['lower'] = spec['lower'] + [[hand_slots, code, count]
                for code, count in sorted(self.hand_scope.minimum.items())]

        def scope(values):
            if type(values) is not list or values != sorted(set(values)) \
                    or any(type(i) is not int or not 0 <= i < len(slots) for i in values):
                raise ValueError("public fact scopes must be canonical slot indices")
            return frozenset(values)

        def codes(values):
            if type(values) is not list or values != sorted(set(values)) or not values \
                    or any(type(code) is not int or code not in self.index for code in values):
                raise ValueError("public fact codes must belong to the explicit inventory")
            return frozenset(self.index[code] for code in values)

        def code_index(code):
            if _int(code, 1) not in self.index:
                raise ValueError("public identity lower bound names a code outside the inventory")
            return self.index[code]

        for key in ('lower', 'categories', 'any'):
            limit = 208 if scoped and key == 'lower' else 128  # <=128 input facts plus <=80 derived hand codes
            if type(spec[key]) is not list or len(spec[key]) > limit:
                raise ValueError("public witness facts must be bounded lists")
        self.lower = tuple((scope(row[0]), code_index(row[1]), _int(row[2], 1))
                           for row in spec['lower'] if self._row(row, 3))
        self.categories = tuple((scope(row[0]), codes(row[1])) for row in spec['categories'] if self._row(row, 2))
        self.any = tuple((scope(row[0]), codes(row[1])) for row in spec['any'] if self._row(row, 2))
        if type(spec['field_targets']) is not list or type(spec['hand_targets']) is not list:
            raise ValueError("current public AR targets require canonical index lists")
        fields, hands = tuple(spec['field_targets']), tuple(spec['hand_targets'])
        scope(list(fields)); scope(list(hands))
        if set(fields) & set(hands) or len(fields) + len(hands) > 80 \
                or any(self.positions[i][0] not in (4, 8) for i in fields) \
                or any(self.positions[i][0] != 2 for i in hands) \
                or [self.positions[i] for i in fields] != sorted(self.positions[i] for i in fields) \
                or [self.positions[i] for i in hands] != sorted(self.positions[i] for i in hands):
            raise ValueError("AR targets must be all canonical field/hand targets")
        self.field_indices, self.hand_indices = fields, hands
        hand_set = frozenset(hands)
        scopes = [scope for scope, _, _ in self.lower] + [scope for scope, _ in self.categories + self.any]
        if hands and (any(self.domains[i] != self.domains[hands[0]] for i in hands)
                      or any(scope & hand_set not in (frozenset(), hand_set) for scope in scopes)):
            raise ValueError("canonical unknown hand slots must be publicly exchangeable")
        self.fields = tuple(Slot(*self.positions[i]) for i in fields)
        self.hand_size, self.length = len(hands), len(fields) + len(hands)
        self._spec_json = json.dumps(declared_spec, sort_keys=True, separators=(',', ':'), allow_nan=False)
        self._mask_cache = {}
        if require_feasible and not self.feasible(()):
            raise ValueError("current public constraints have no complete assignment")

    @staticmethod
    def _row(row, length):
        if type(row) is not list or len(row) != length:
            raise ValueError("malformed current public fact")
        return True

    def to_dict(self):
        return json.loads(self._spec_json)

    def _check(self):
        if self.clock() >= self.deadline or self.nodes >= self.max_nodes:
            raise CurrentLawBudgetExceeded("current public feasibility budget exhausted; support remains unknown")

    def _prefix_domains(self, prefix):
        if type(prefix) is not tuple or len(prefix) > self.length \
                or any(type(code) is not int or code not in self.index for code in prefix):
            raise ValueError("current AR prefix must contain canonical public vocabulary codes")
        hand = prefix[len(self.fields):]
        if tuple(sorted(hand)) != hand:
            raise ValueError("current AR hand prefix must be sorted")
        domains = list(self.domains)
        for slot, code in zip((*self.field_indices, *self.hand_indices), prefix):
            domains[slot] &= {self.index[code]}
        if hand:
            allowed = {i for i, code in enumerate(self.codes) if code >= hand[-1]}
            for slot in self.hand_indices[len(hand):]:
                domains[slot] &= allowed
        return tuple(frozenset(domain) for domain in domains)

    def _possible(self, domains):
        memo = {}

        def solve(domains):
            self._check()
            self.nodes += 1
            if domains in memo:
                return memo[domains]
            if any(not domain for domain in domains):
                return False
            fixed = {i: next(iter(domain)) for i, domain in enumerate(domains) if len(domain) == 1}
            remaining = np.asarray(self.counts, np.int64).copy()
            for code in fixed.values(): remaining[code] -= 1
            if (remaining < 0).any(): return False
            free = [i for i in range(len(domains)) if i not in fixed]
            groups = Counter(domains[i] for i in free)
            upper = np.asarray([[min(int(remaining[ci]), n) if ci in domain else 0
                                 for domain, n in groups.items()] for ci in range(len(self.codes))], np.int64)
            upper = upper.reshape(len(self.codes), len(groups))
            if not transport_feasible(remaining, np.asarray(list(groups.values()), np.int64),
                                      np.zeros_like(upper), upper):
                memo[domains] = False
                return False
            choices = []
            for scope, code, need in self.lower:
                missing = need - sum(fixed.get(i) == code for i in scope)
                if missing > 0:
                    available = [i for i in free if i in scope and code in domains[i]]
                    if len(available) < missing or remaining[code] < missing: return False
                    choices.append([(i, code) for i in available])
            for scope, codes in self.any:
                if not any(fixed.get(i) in codes for i in scope):
                    choices.append([(i, code) for i in free if i in scope
                                    for code in sorted(domains[i] & codes) if remaining[code]])
            # Exact augmenting-path matching: anonymous category facts need
            # different physical cards, but alternative matchings must not be
            # enumerated (nor counted as different possible hidden layouts).
            matched = {}
            def augment(category, seen):
                scope, codes = self.categories[category]
                for i, code in fixed.items():
                    if i not in scope or code not in codes or i in seen: continue
                    seen.add(i)
                    if i not in matched or augment(matched[i], seen):
                        matched[i] = category
                        return True
                return False
            complete_categories = all(augment(j, set()) for j in range(len(self.categories)))
            if not complete_categories:
                candidates = []
                for i in free:
                    for code in sorted(domains[i]):
                        if not remaining[code]: continue
                        if any(i in scope and code in codes for scope, codes in self.categories):
                            candidates.append((i, code))
                choices.append(candidates)
            if not choices:
                memo[domains] = True
                return True
            candidates = min(choices, key=len)
            seen = set()
            for i, code in candidates:
                # Exchangeable physical slots need only one existential branch.
                signature = (domains[i], code, tuple(i in scope for scope, _, _ in self.lower),
                             tuple(i in scope for scope, _ in self.categories),
                             tuple(i in scope for scope, _ in self.any))
                if signature in seen: continue
                seen.add(signature)
                child = list(domains)
                child[i] = frozenset((code,))
                if solve(tuple(child)):
                    memo[domains] = True
                    return True
            memo[domains] = False
            return False

        result = solve(domains)
        self._check()
        return result

    def feasible(self, prefix):
        return self._possible(self._prefix_domains(prefix))

    def mask(self, prefix):
        self._check()
        self._prefix_domains(prefix)
        if len(prefix) >= self.length:
            raise ValueError("a completed AR layout has no next token")
        if prefix not in self._mask_cache:
            hand = prefix[len(self.fields):]
            self._mask_cache[prefix] = tuple(False if hand and code < hand[-1]
                                            else self.feasible((*prefix, code)) for code in self.codes)
        return np.asarray(self._mask_cache[prefix], bool)

    def layout(self, sequence):
        if len(sequence) != self.length or not self.feasible(sequence):
            raise ValueError("AR layout lacks a complete public-consistent residual")
        return {'schema': SCHEMA, 'facedown': [[slot.location, slot.sequence, code]
                                               for slot, code in zip(self.fields, sequence)],
                'hidden_hand': list(sequence[len(self.fields):]), 'residual_assignment_exists': True}

    def check_complete_layout(self, layout, sequence):
        """Verify the serialized complete residual against this exact public law.

        The autoregressive targets cover only hidden field identities and the
        exchangeable unknown hand. Native realization also needs every other
        hidden card. Validate that residual here rather than trusting a bank
        shape, a histogram, or the producer's own assertion.
        """
        if type(layout) is not dict or set(layout) != {'hand', 'deck', 'extra', 'facedown'}:
            raise ValueError("complete AR layout fields differ")
        positions = {}
        for location, key in ((2, 'hand'), (1, 'deck'), (64, 'extra')):
            values = layout[key]
            if type(values) is not list:
                raise ValueError("complete AR zones must be lists")
            for sequence_index, code in enumerate(values):
                _int(code, 1)
                positions[(location, sequence_index)] = code
        if type(layout['facedown']) is not list:
            raise ValueError("complete AR field assignments must be a list")
        for row in layout['facedown']:
            if type(row) is not list or len(row) != 3:
                raise ValueError("complete AR field rows need location/sequence/code")
            location, sequence_index, code = row
            _int(location); _int(sequence_index); _int(code, 1)
            if location not in (4, 8, 32) or (location, sequence_index) in positions:
                raise ValueError("complete AR field assignment is invalid or duplicated")
            positions[(location, sequence_index)] = code
        if set(positions) != set(self.positions):
            raise ValueError("complete AR layout does not fill the registered public inventory slots")
        values = tuple(positions[position] for position in self.positions)
        if Counter(values) != Counter({code: count for code, count in zip(self.codes, self.counts)}):
            raise ValueError("complete AR layout changed the public main/extra multiset")
        for code, domain in zip(values, self.domains):
            if self.index[code] not in domain:
                raise ValueError("complete AR residual violates a public slot domain")
        for scope, code, count in self.lower:
            if sum(values[index] == self.codes[code] for index in scope) < count:
                raise ValueError("complete AR residual violates a public identity lower bound")
        for scope, allowed in self.any:
            if not any(values[index] in {self.codes[code] for code in allowed} for index in scope):
                raise ValueError("complete AR residual violates a public existence fact")
        matched = {}

        def augment(category, seen):
            self._check()
            scope, allowed = self.categories[category]
            for index in sorted(scope):
                if index in seen or self.index[values[index]] not in allowed:
                    continue
                seen.add(index)
                if index not in matched or augment(matched[index], seen):
                    matched[index] = category
                    return True
            return False

        if not all(augment(category, set()) for category in range(len(self.categories))):
            raise ValueError("complete AR residual lacks distinct public category witnesses")
        expected = tuple(values[index] for index in (*self.field_indices, *self.hand_indices))
        if type(sequence) is not tuple or sequence != expected:
            raise ValueError("complete AR layout differs from its sampled target sequence")
        if self.hand_scope is not None and not self.hand_scope.feasible_multiset(layout['hand']):
            raise ValueError("canonical AR hand has no public-scope physical realization")
        return True

    def complete(self, sequence, rng):
        """Draw the remaining physical slots under a separately declared conditional residual law."""
        self.layout(sequence)
        domains = self._prefix_domains(sequence)
        for i in range(len(domains)):
            self._check()
            if len(domains[i]) == 1: continue
            candidates, weights = [], []
            fixed = Counter(next(iter(domain)) for domain in domains if len(domain) == 1)
            for code in sorted(domains[i]):
                weight = self.counts[code] - fixed[code]
                if weight <= 0: continue
                trial = list(domains)
                trial[i] = frozenset((code,))
                if self._possible(tuple(trial)):
                    candidates.append(code)
                    weights.append(weight)
            if not candidates:
                raise ValueError("public residual lost a previously proven completion")
            picked = candidates[int(rng.choice(len(candidates), p=np.asarray(weights, np.float64) / sum(weights)))]
            trial = list(domains)
            trial[i] = frozenset((picked,))
            domains = tuple(trial)
        if not self._possible(domains):
            raise ValueError("sampled residual violates the original public constraints")
        assigned = {position: self.codes[next(iter(domain))] for position, domain in zip(self.positions, domains)}
        return {'hand': [assigned[(2, i)] for i in range(sum(p[0] == 2 for p in self.positions))],
                'deck': [assigned[(1, i)] for i in range(sum(p[0] == 1 for p in self.positions))],
                'extra': [assigned[(64, i)] for i in range(sum(p[0] == 64 for p in self.positions))],
                'facedown': [[loc, seq, code] for (loc, seq), code in sorted(assigned.items()) if loc in (4, 8, 32)]}
