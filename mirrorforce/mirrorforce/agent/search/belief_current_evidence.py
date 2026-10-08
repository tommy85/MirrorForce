"""Translate the existing leak-audited public evidence, without engine/truth reads.

The current AR proposal uses the same public constraints as root admission,
but has its own direct-draw law. It never discards a constraint to fit the old
offline AR world codec. The resulting closed JSON can cross the policy RPC.
"""
from __future__ import annotations

from collections import Counter

from .belief_current_law import SCHEMA


def public_specification(public, recipe, *, viewer, extra_origin_slots=(), categories=(), hand_scope=None):
    from ...common.stage_a_joint_proposals import evidence_for_public_snapshot
    from ...common.stage_a_joint_belief_runtime import ZoneClaim
    claims = tuple(claim for claim in categories if isinstance(claim, ZoneClaim))
    evidence = evidence_for_public_snapshot(public, viewer, recipe, extra_origin_slots=extra_origin_slots,
        categories=tuple(claim for claim in categories if not isinstance(claim, ZoneClaim)))
    pinned = {}
    for controller, location, sequence, code in public.revealed:
        if controller == 1 - viewer and code:
            pinned[(int(location), int(sequence))] = int(code)
    # Public field rows retain physical coordinates, unlike canonical hidden
    # hand/extra rows. A visible identity there is already entitled evidence.
    for card in public.cards:
        key = (card.location, card.sequence)
        if card.controller == 1 - viewer and key in evidence.facedown_slot_keys and card.code:
            if key in pinned and pinned[key] != card.code:
                raise ValueError("public field row and disclosed slot disagree")
            pinned[key] = int(card.code)
    result = _from_evidence(evidence, pinned=pinned, zone_claims=claims)
    if hand_scope is not None:
        from .belief_hand_scope import PublicHandScope
        from .belief_current_law import SCOPED_SCHEMA
        scope = PublicHandScope(hand_scope)
        if scope.size != evidence.hand_size:
            raise ValueError("same-pending native/v4 hand counts differ")
        result.update(schema=SCOPED_SCHEMA, hand_scope=scope.to_dict())
    return result


def _from_evidence(evidence, *, pinned, zone_claims=()):
    """Internal bridge; pinned identities must come from the already audited public snapshot."""
    evidence.check()
    if evidence.other_hidden_slots or evidence.unattributed_extra_slots() \
            or evidence.unexpressed_field_knowledge:
        raise ValueError("current AR cannot express every public hidden inventory/field constraint")
    pools = [evidence.unknown_pool(), evidence.unknown_extra_pool()]
    for counter in (evidence.disclosed_hand, evidence.disclosed_deck, evidence.disclosed_facedown):
        pools[0].update(counter)
    pools[0].update(evidence.deck_top)
    for counter in (evidence.disclosed_extra_slots, evidence.disclosed_extra_deck):
        pools[1].update(counter)
    pools = [+pool for pool in pools]
    extra_keys = set(evidence.extra_slot_keys)
    positions = [(1, i, 0) for i in range(evidence.deck_size)]
    positions += [(2, i, 0) for i in range(evidence.hand_size)]
    positions += [(loc, seq, int((loc, seq) in extra_keys)) for loc, seq in evidence.facedown_slot_keys]
    positions += [(64, i, 1) for i in range(evidence.extra_facedown_size)]
    positions.sort()
    if len({(loc, seq) for loc, seq, _ in positions}) != len(positions):
        raise ValueError("current public evidence repeats a physical hidden slot")
    indices = {(loc, seq): i for i, (loc, seq, _) in enumerate(positions)}
    domains = [set(pools[pool]) for _, _, pool in positions]
    pinned = {tuple(key): int(code) for key, code in pinned.items() if tuple(key) in indices}
    field_pins = {key: code for key, code in pinned.items() if key[0] in (4, 8, 32)}
    known_main = Counter(code for key, code in field_pins.items() if key not in extra_keys)
    known_extra = Counter(code for key, code in field_pins.items() if key in extra_keys)
    grouped_keys = set()
    for location, sequences, codes in evidence.shuffled_facedown:
        grouped_keys.update((location, sequence) for sequence in sequences)
        known_main.update(codes)
    if known_main != evidence.disclosed_facedown or known_extra != evidence.disclosed_extra_slots:
        raise ValueError("public fixed/shuffled field identities do not account for the disclosed inventory")
    if grouped_keys & set(pinned):
        raise ValueError("a shuffled public field identity cannot also have a claimed known position")

    hand_pins = Counter(code for (loc, _), code in pinned.items() if loc == 2)
    if any(n > evidence.disclosed_hand[code] for code, n in hand_pins.items()):
        raise ValueError("public hand anchors exceed the disclosed multiset")
    # The existing sampler's canonical placement of unpositioned known hand
    # cards; the native writer later preserves every genuinely known UID.
    spare = iter(sorted((evidence.disclosed_hand - hand_pins).elements()))
    for sequence in range(evidence.hand_size):
        key = (2, sequence)
        if key not in pinned:
            code = next(spare, None)
            if code is not None: pinned[key] = code
    if next(spare, None) is not None:
        raise ValueError("public disclosed hand cannot fit its canonical slots")
    for sequence, code in enumerate(evidence.deck_top, start=evidence.deck_size - len(evidence.deck_top)):
        key = (1, sequence)
        if key not in indices or key in pinned and pinned[key] != code:
            raise ValueError("public deck-top identities contradict the declared slots")
        pinned[key] = code
    for key, code in pinned.items():
        domains[indices[key]] &= {code}
    lower, categories, any_clauses = [], [], []

    def scope(keys):
        keys = tuple(keys)
        if any(key not in indices for key in keys):
            raise ValueError("public constraint names a missing hidden slot")
        return sorted(indices[key] for key in keys)

    for location, sequences, codes in evidence.shuffled_facedown:
        keys = [(location, sequence) for sequence in sequences]
        if len(keys) != len(codes): raise ValueError("public shuffled field group has unequal slots and identities")
        for key in keys: domains[indices[key]] &= set(codes)
        for code, count in sorted(Counter(codes).items()): lower.append([scope(keys), code, count])
    fresh = set(evidence.fresh_facedown_keys)
    sampling_keys = set(evidence.facedown_sampling_keys)
    if sampling_keys != {key for key in indices if key[0] in (4, 8, 32)
                         and key not in pinned and key not in grouped_keys and key not in extra_keys}:
        raise ValueError("public field sampling keys omit or add a hidden slot")
    for (location, code), count in sorted(Counter(evidence.unpositioned_facedown).items()):
        keys = [key for key in sampling_keys - fresh if key[0] == location]
        lower.append([scope(keys), code, count])
    for key, codes in evidence.facedown_categories:
        if key not in sampling_keys:
            raise ValueError("public field category is outside the current sampled slots")
        domains[indices[key]] &= set(codes)
    hand_targets = scope((2, sequence) for sequence in range(evidence.hand_size) if (2, sequence) not in pinned)
    vocabulary = set(pools[0]) | set(pools[1])
    for codes in evidence.hand_categories:
        categories.append([hand_targets, sorted(set(codes) & vocabulary)])
    for location, codes in evidence.unpositioned_facedown_categories:
        categories.append([scope(key for key in indices if key[0] == location), sorted(set(codes) & vocabulary)])
    # Known identities in unordered residual zones remain lower bounds, not
    # predictions or artificially anchored physical objects.
    for location, known in ((1, evidence.disclosed_deck + Counter(evidence.deck_top)),
                            (64, evidence.disclosed_extra_deck)):
        for code, count in sorted(known.items()):
            lower.append([scope(key for key in indices if key[0] == location), code, count])
    for claim in zone_claims:
        if any(zone not in (1, 2, 64) for zone in claim.zones):
            raise ValueError("unsupported public zone-existence claim")
        any_clauses.append([scope(key for key in indices if key[0] in claim.zones),
                            sorted(set(claim.codes) & vocabulary)])
    return {'schema': SCHEMA, 'pools': [[[code, count] for code, count in sorted(pool.items())] for pool in pools],
            'slots': [[loc, seq, pool, sorted(domain)] for (loc, seq, pool), domain in zip(positions, domains)],
            'lower': lower, 'categories': categories, 'any': any_clauses,
            'field_targets': scope(key for key in indices if key[0] in (4, 8) and key not in pinned),
            'hand_targets': hand_targets}
