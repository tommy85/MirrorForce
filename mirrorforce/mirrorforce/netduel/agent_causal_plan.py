"""Bounded public-history capacity solver, separate from native replay admission.

Every anonymous shuffle preserves a multiset, not an index-to-identity claim.
This solver colors public position tokens from the opening recipes through
those cuts to one *unchanged* complete root proposal. It then supplies a
same-code bijection at each cut. A solution is a causal layout witness only:
natural script legality, the full received wire and private memory still need
to be reproduced by the complete hypothetical engine.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import json
import math
import time

from . import constants as C
from .causal_history import BLANK_CODES, LAW as HISTORY_LAW, PublicPositionHistory

INFORMATION_SET_SEARCH = True
SCHEMA = "mirrorforce_causal_opening_plan/v1"
LAW = "bounded-public-capacity-witness; unchanged-root-proposal/v1"


class PlanError(ValueError):
    pass


class PlanIncompatible(PlanError):
    """The unchanged proposal has no witness within the declared public graph."""


class PlanBudgetExceeded(PlanError):
    """No accepted witness; the original particle must not be silently replaced."""


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _code(code):
    if type(code) is not int or not 0 < code < 2 ** 31 or code in BLANK_CODES:
        raise PlanError("a causal layout contains only real exact integer card codes")
    return code


def root_layout(history, particle):
    """Bind a whole public sampler proposal without reading any follower UID.

    Known current cards come from the observer's proved position history. The
    five particle fields supply all hidden opponent slots and the entire own
    deck order. Missing, extra or contradictory slots fail; no repair occurs.
    The caller remains responsible for binding this particle to its pending
    public World and the public sampler law.
    """
    if not isinstance(history, PublicPositionHistory):
        raise PlanError("a received public position history is required")
    record = history.record()
    if not isinstance(particle, dict) or set(particle) != {"hand", "deck", "extra", "facedown", "own_deck"}:
        raise PlanError("the complete root proposal must have exactly its five declared fields")
    places = {(p, z, s): token for p, z, s, token in record["root_slots"]}
    layout = {place: history.known[token] for place, token in places.items() if token in history.known}
    own, opponent = history.viewer, 1 - history.viewer

    def bind(place, code):
        _code(code)
        if place not in places:
            raise PlanError("the root proposal names an unoccupied public slot")
        if place in layout and layout[place] != code:
            raise PlanIncompatible("root proposal contradicts an unshuffled known identity")
        layout[place] = code

    for name, player, location in (("hand", opponent, C.LOCATION_HAND), ("deck", opponent, C.LOCATION_DECK),
                                   ("extra", opponent, C.LOCATION_EXTRA), ("own_deck", own, C.LOCATION_DECK)):
        codes = particle[name]
        if not isinstance(codes, (tuple, list)) or len(codes) != len(history.zones[player, location]):
            raise PlanError("root proposal count differs from the received public positions: " + name)
        for sequence, code in enumerate(codes):
            bind((player, location, sequence), code)
    facedown = particle["facedown"]
    if not isinstance(facedown, (list, tuple)):
        raise PlanError("face-down proposals must be explicit rows")
    seen = set()
    for row in facedown:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise PlanError("a face-down proposal needs location, sequence and code")
        location, sequence, code = row
        place = (opponent, location, sequence)
        if type(location) is not int or location not in (C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_REMOVED) \
                or type(sequence) is not int or sequence < 0 or place in seen:
            raise PlanError("invalid or duplicate face-down proposal coordinate")
        seen.add(place)
        bind(place, code)
    if layout.keys() != places.keys():
        raise PlanError("the proposal plus public history does not identify every root card")
    return tuple((*place, layout[place]) for place in sorted(layout))


@dataclass(frozen=True)
class OpeningPlan:
    history_sha256: str
    root_sha256: str
    seed: int
    token_codes: tuple  # code at token ID minus one
    openings: tuple    # (player, location, ordered opening tokens, ordered codes)
    births: tuple      # (birth token, code)
    cuts: tuple        # (packet, msg, after tokens, corresponding before tokens)
    root: tuple        # (player, location, sequence, token, code)
    search_nodes: int

    def record(self):
        return {"schema": SCHEMA, "law": LAW, "history_law": HISTORY_LAW,
                "history_sha256": self.history_sha256, "root_sha256": self.root_sha256,
                "seed": self.seed, "token_codes": self.token_codes, "openings": self.openings,
                "births": self.births, "cuts": self.cuts, "root": self.root,
                "search_nodes": self.search_nodes, "own_deck_applied": False,
                "native_replay_admitted": False}


def _graph_problem(history):
    if not isinstance(history, PublicPositionHistory):
        raise PlanError("a received public position history is required")
    record = history.record()
    pools = {tuple(pool["pool"]): Counter(dict(pool["counts"])) for pool in record["pools"]}
    domains = [set().union(*(pools[tuple(pool)] for pool in token["pools"])) for token in record["tokens"]]
    constraints = []
    for pool in record["pools"]:
        constraints.append((tuple(t - 1 for t in pool["tokens"]), (), Counter(dict(pool["counts"]))))
    for cut in record["permutations"]:
        constraints.append((tuple(t - 1 for t in cut["before"]), tuple(t - 1 for t in cut["after"]), None))
    for fact in record["facts"]:
        token, code = fact["token"], fact["code"]
        if code not in domains[token - 1]:
            raise PlanIncompatible("a public identity contradicts the declared recipe capacity")
        domains[token - 1] = {code}
    return record, domains, constraints


def _problem(history, layout):
    record, domains, constraints = _graph_problem(history)
    if not isinstance(layout, (tuple, list)):
        raise PlanError("root layout must be an explicit complete coordinate sequence")
    placed = {}
    for row in layout:
        if not isinstance(row, (tuple, list)) or len(row) != 4:
            raise PlanError("root layout rows are player, location, sequence, code")
        p, z, s, code = row
        if any(type(value) is not int for value in (p, z, s)) or (p, z, s) in placed:
            raise PlanError("root layout repeats a slot or has a noninteger coordinate")
        placed[p, z, s] = _code(code)
    root_slots = {(p, z, s): token for p, z, s, token in record["root_slots"]}
    if placed.keys() != root_slots.keys():
        raise PlanError("root layout must bind exactly every occupied public position")
    for place, code in placed.items():
        token = root_slots[place]
        if code not in domains[token - 1]:
            raise PlanIncompatible("root proposal conflicts with a public identity or recipe capacity")
        domains[token - 1] = {code}
    canonical_layout = tuple((*place, placed[place]) for place in sorted(placed))
    return record, canonical_layout, domains, constraints


@dataclass(frozen=True)
class _RowBound:
    rows: tuple       # (original constraint index, integer multiplier)
    code: int
    terms: tuple      # (token index, signed integer coefficient)
    target: int       # fixed initial-domain terms have been subtracted
    groups: tuple = field(init=False, repr=False)  # coefficient -> token bitset

    def __post_init__(self):
        groups = {}
        for token, coefficient in self.terms:
            groups[coefficient] = groups.get(coefficient, 0) | (1 << token)
        object.__setattr__(self, "groups", tuple(sorted(groups.items())))


def _compile_row_bounds(domains, constraints, check, *, max_combinations=64, max_work=100000):
    """Optional exact necessary bounds, never a change to a token's domain.

    Each original row is sum(left indicators) - sum(right indicators) =
    fixed[code] (or zero). Two-row integer combinations can expose a shared
    token contradiction before enumerating its many impossible descendants.
    Folding only initial singleton/absent terms cannot change any later DFS
    domain, MRV choice, SHA preference or first feasible coloring.

    Optional work limits discard only further acceleration; the caller's
    absolute deadline is always checked and is never renewed or swallowed.
    """
    check()
    work, next_check = 0, 64

    class OptionalLimit(Exception):
        pass

    def spend(amount=1):
        nonlocal work, next_check
        work += amount
        if work >= next_check or work > max_work:
            check()
            next_check = work + 64
        if work > max_work:
            raise OptionalLimit

    rows, bounds, seen = [], [], set()
    combinations = 0
    try:
        for left, right, fixed in constraints:
            spend(len(left) + len(right) + 1)
            coefficients = Counter(left)
            coefficients.subtract(right)  # preserve negative and repeated terms
            rows.append(({t: c for t, c in coefficients.items() if c}, fixed))
        for i, (left, left_fixed) in enumerate(rows):
            for j in range(i + 1, len(rows)):
                right, right_fixed = rows[j]
                spend(min(len(left), len(right)) + 1)
                if left.keys().isdisjoint(right):
                    continue
                for sign in (-1, 1):
                    if combinations >= max_combinations:
                        raise OptionalLimit
                    combinations += 1
                    spend(len(left) + len(right))
                    combined = dict(left)
                    for token, coefficient in right.items():
                        combined[token] = combined.get(token, 0) + sign * coefficient
                    combined = tuple(sorted((t, c) for t, c in combined.items() if c))
                    spend(len(left_fixed or ()) + len(right_fixed or ()))
                    codes = set(left_fixed or ()) | set(right_fixed or ())
                    for token, _ in combined:
                        spend(len(domains[token]))
                        codes.update(domains[token])
                    for code in sorted(codes):
                        target = (left_fixed[code] if left_fixed is not None else 0) + sign * (
                            right_fixed[code] if right_fixed is not None else 0)
                        terms = []
                        for token, coefficient in combined:
                            spend()
                            domain = domains[token]
                            if code in domain:
                                if len(domain) == 1:
                                    target -= coefficient
                                else:
                                    terms.append((token, coefficient))
                        terms = tuple(terms)
                        key = (code, terms, target)
                        if (terms or target) and key not in seen:
                            seen.add(key)
                            bounds.append(_RowBound(((i, 1), (j, sign)), code, terms, target))
    except OptionalLimit:
        pass
    check()  # even optional exhaustion must not hide an expired deadline
    return tuple(bounds)


def _row_bound_index(bounds, check):
    check()
    tokens, codes, work = set(), set(), 0
    for bound in bounds:
        codes.add(bound.code)
        for token, _ in bound.terms:
            tokens.add(token)
            work += 1
            if work % 64 == 0:
                check()
    check()
    return tuple(sorted(tokens)), frozenset(codes)


def _reject_row_bounds(bounds, domains, check, *, index=None):
    """Reject only a proven impossible branch; never narrow a surviving one."""
    check()
    tokens, codes = _row_bound_index(bounds, check) if index is None else index
    possible, forced = dict.fromkeys(codes, 0), dict.fromkeys(codes, 0)
    work = 0
    # Compute each code's indicators once per branch, not once per expression.
    # Integer bitsets count original token positions; duplicate coefficients
    # remain exact integer multipliers in each independently derived row.
    for token in tokens:
        domain, bit = domains[token], 1 << token
        singleton = len(domain) == 1
        for code in domain:
            if code in possible:
                possible[code] |= bit
                if singleton:
                    forced[code] |= bit
            work += 1
            if work % 64 == 0:
                check()
    for bound in bounds:
        work += 1
        if work % 64 == 0:
            check()
        low = high = 0
        allowed, certain = possible[bound.code], forced[bound.code]
        for coefficient, mask in bound.groups:
            may = (mask & allowed).bit_count()
            must = (mask & certain).bit_count()
            low += coefficient * (must if coefficient > 0 else may)
            high += coefficient * (may if coefficient > 0 else must)
            work += 1
            if work % 64 == 0:
                check()
        if not low <= bound.target <= high:
            check()
            raise PlanIncompatible(f"public capacity row combination {bound.rows} cannot balance "
                                   f"code {bound.code}: [{low},{high}] target {bound.target}")
    check()


class _Solver:
    def __init__(self, constraints, count, seed, max_nodes, deadline, clock):
        self.constraints, self.seed, self.max_nodes = constraints, seed, max_nodes
        self.deadline, self.clock, self.nodes, self.operations = deadline, clock, 0, 0
        self.linear_witness_report = None
        self.linear_infeasibility_report = None
        self.linear_infeasibility_certificate = None
        self.adjacent = [[] for _ in range(count)]
        for ci, (left, right, _) in enumerate(constraints):
            for token in (*left, *right):
                self.adjacent[token].append(ci)

    def check(self):
        if self.clock() >= self.deadline:
            raise PlanBudgetExceeded("public-history witness deadline exhausted")

    def preference(self, token, code):
        # Explicit version-independent tie law, no engine RNG consumption.
        return hashlib.sha256(f"{LAW}:{self.seed}:{token + 1}:{code}".encode()).digest()

    def propagate(self, domains):
        queue, queued = deque(range(len(self.constraints))), set(range(len(self.constraints)))

        def narrow(token, domain):
            if not domain:
                raise PlanIncompatible("no identity satisfies all public capacities")
            if domain != domains[token]:
                domains[token] = domain
                for ci in self.adjacent[token]:
                    if ci not in queued:
                        queue.append(ci)
                        queued.add(ci)

        while queue:
            self.check()
            ci = queue.popleft()
            queued.remove(ci)
            left, right, fixed = self.constraints[ci]
            codes = set().union(*(domains[t] for t in (*left, *right)))
            if fixed is not None:
                codes.update(fixed)
            for code in sorted(codes):
                self.operations += 1
                if self.operations % 64 == 0:
                    self.check()
                lp = [t for t in left if code in domains[t]]
                lf = [t for t in lp if len(domains[t]) == 1]
                rp = [t for t in right if code in domains[t]]
                rf = [t for t in rp if len(domains[t]) == 1]
                lo, hi = (len(rf), len(rp)) if fixed is None else (fixed[code], fixed[code])
                if len(lf) > hi or len(lp) < lo:
                    raise PlanIncompatible(f"public capacity {ci} cannot balance code {code}: "
                                           f"left[{len(lf)},{len(lp)}] right[{lo},{hi}]")
                if len(lf) == hi:
                    for t in lp:
                        if len(domains[t]) > 1:
                            narrow(t, domains[t] - {code})
                if len(lp) == lo:
                    for t in lp:
                        narrow(t, {code})
                if fixed is None:
                    if len(rf) == len(lp):
                        for t in rp:
                            if len(domains[t]) > 1:
                                narrow(t, domains[t] - {code})
                    if len(rp) == len(lf):
                        for t in rp:
                            narrow(t, {code})

    def search(self, domains, *, reject=None):
        # Lazy explicit stack: neither Python recursion depth nor an eager
        # copy of every sibling limits a long real public trace.
        def children(parent, token, options):
            for code in options:
                child = parent.copy()  # propagation replaces sets, never mutates them
                child[token] = {code}
                yield child

        stack = [iter((domains,))]
        last_failure = None
        while stack:
            self.check()
            try:
                current = next(stack[-1])
            except StopIteration:
                stack.pop()
                continue
            if self.nodes >= self.max_nodes:
                raise PlanBudgetExceeded("public-history witness search-node budget exhausted")
            self.nodes += 1
            try:
                self.propagate(current)
                if reject is not None:
                    reject(current)
            except PlanIncompatible as exc:
                last_failure = exc
                continue
            unknown = [t for t, domain in enumerate(current) if len(domain) > 1]
            if not unknown:
                return tuple(next(iter(domain)) for domain in current)
            token = min(unknown, key=lambda t: (len(current[t]), -len(self.adjacent[t]), self.preference(t, 0)))
            options = sorted(current[token], key=lambda code: self.preference(token, code))
            stack.append(children(current, token, options))
        raise PlanIncompatible("the unchanged root proposal has no public-history witness: " + str(last_failure))

    def feasible_witness(self, domains):
        """Existence only: exact integer positive/negative proofs, otherwise original search.

        The seeded natural replay path deliberately keeps using search(). The
        accelerator's negative/timeout/status is NEVER treated as infeasible.
        A negative requires a separately checked original-token Farkas proof.
        """
        self.check()
        if len(domains) >= 64 and self.max_nodes - self.nodes >= 2:
            from .causal_linear import try_witness
            colors, self.linear_witness_report = try_witness(domains, self.constraints,
                deadline=self.deadline, clock=self.clock, max_nodes=self.max_nodes - self.nodes)
            self.nodes += self.linear_witness_report["nodes"]
            self.check()
            if colors is not None:
                return colors
            if self.max_nodes - self.nodes >= 1:
                from .causal_farkas import try_certificate, exact_certificate
                certificate, self.linear_infeasibility_report = try_certificate(domains, self.constraints,
                    deadline=self.deadline, clock=self.clock)
                self.nodes += self.linear_infeasibility_report["nodes"]
                self.check()
                if certificate is not None and exact_certificate(certificate, domains, self.constraints):
                    self.linear_infeasibility_certificate = certificate
                    raise PlanIncompatible("public capacity has an exact original-token integer Farkas certificate")
        return self.search(domains)


def solve(history, layout, *, seed, max_nodes=10000, deadline, clock=time.monotonic, mode="seeded-search"):
    """Find one witness under an absolute, nonrenewing deadline and node limit.

    A seed chooses between otherwise valid historical witnesses; it does not
    resample or reweight the supplied root layout. Callers must account for
    this work in the shared prompt budget and fail the full bank on any error.
    """
    if mode not in ("seeded-search", "positive-replay"):
        raise PlanError("unknown explicit causal construction mode")
    if type(seed) is not int or (mode == "seeded-search" and (type(max_nodes) is not int or max_nodes < 1)) \
            or mode == "positive-replay" and max_nodes is not None \
            or type(deadline) not in (int, float) or not math.isfinite(deadline) or not callable(clock):
        raise PlanError("an integer seed, positive node limit and finite absolute deadline are required")
    if clock() >= deadline:
        raise PlanBudgetExceeded("public-history witness deadline exhausted before initialization")
    record, canonical_layout, domains, constraints = _problem(history, layout)
    solver = _Solver(constraints, len(domains), seed, max_nodes, deadline, clock)
    # Only the seeded natural-plan path opts in. Boolean existence/counting
    # callers retain search()'s original default and feasible_witness() path.
    bounds = () if mode == "positive-replay" else _compile_row_bounds(domains, constraints, solver.check)
    if mode == "positive-replay":
        from .causal_linear import try_witness, exact_witness
        colors, report = try_witness(domains, constraints, deadline=deadline, clock=clock,
                                    max_nodes=None, max_seconds=max(1e-9, deadline - clock()))
        solver.check()
        if colors is None:
            raise PlanBudgetExceeded("positive replay construction produced no exact integer witness")
        if not exact_witness(colors, domains, constraints):
            raise PlanError("positive replay constructor returned an invalid integer coloring")
        solver.nodes = report["nodes"]  # accounting only; this mode has no existence-node cap
    elif bounds:
        bound_index = _row_bound_index(bounds, solver.check)
        def reject(current):
            _reject_row_bounds(bounds, current, solver.check, index=bound_index)
        colors = solver.search(domains, reject=reject)
    else:
        colors = solver.search(domains)
    cuts = []
    for cut in record["permutations"]:
        solver.check()
        by_code = defaultdict(list)
        for token in cut["before"]:
            by_code[colors[token - 1]].append(token)
        for code, tokens in by_code.items():
            tokens.sort(key=lambda t: solver.preference(t - 1, code))
        matched = tuple(by_code[colors[token - 1]].pop() for token in cut["after"])
        cuts.append((cut["packet"], cut["msg"], tuple(cut["after"]), matched))
    openings, births = [], []
    for pool in record["pools"]:
        player, location = pool["pool"]
        if player in (0, 1):
            tokens = tuple(pool["tokens"])
            openings.append((player, location, tokens, tuple(colors[t - 1] for t in tokens)))
        else:
            births.extend((t, colors[t - 1]) for t in pool["tokens"])
    plan = OpeningPlan(_sha(record), _sha(canonical_layout), seed, colors, tuple(openings), tuple(births),
                       tuple(cuts), tuple((*row, colors[row[-1] - 1]) for row in record["root_slots"]), solver.nodes)
    verify(history, canonical_layout, plan)
    solver.check()
    return plan


def verify(history, layout, plan):
    """Independent linear certificate check; does not rerun the search."""
    if not isinstance(plan, OpeningPlan):
        raise PlanError("an immutable causal opening plan is required")
    record, canonical_layout, domains, constraints = _problem(history, layout)
    colors = plan.token_codes
    if plan.history_sha256 != _sha(record) or plan.root_sha256 != _sha(canonical_layout) \
            or len(colors) != len(domains) or any(_code(c) not in domain for c, domain in zip(colors, domains)):
        raise PlanError("causal plan identity or per-token public facts differ")
    for left, right, fixed in constraints:
        if Counter(colors[t] for t in left) != (Counter(colors[t] for t in right) if fixed is None else fixed):
            raise PlanError("causal plan violates an opening or shuffle capacity")
    expected_openings, expected_births = [], []
    for pool in record["pools"]:
        p, z = pool["pool"]
        if p in (0, 1):
            tokens = tuple(pool["tokens"])
            expected_openings.append((p, z, tokens, tuple(colors[t - 1] for t in tokens)))
        else:
            expected_births.extend((t, colors[t - 1]) for t in pool["tokens"])
    if plan.openings != tuple(expected_openings) or plan.births != tuple(expected_births):
        raise PlanError("causal plan opening or birth identity differs")
    if len(plan.cuts) != len(record["permutations"]):
        raise PlanError("causal plan must certify every anonymous cut")
    aliases = {t: t for pool in record["pools"] for t in pool["tokens"]}
    for cut, proof in zip(record["permutations"], plan.cuts):
        packet, msg, after, before = proof
        if (packet, msg, after) != (cut["packet"], cut["msg"], tuple(cut["after"])) \
                or len(before) != len(set(before)) or set(before) != set(cut["before"]):
            raise PlanError("causal cut is not a full certified bijection")
        for a, b in zip(after, before):
            if b not in aliases or a in aliases or colors[a - 1] != colors[b - 1]:
                raise PlanError("causal cut changes a card identity or its continuity")
            aliases[a] = aliases[b]
        for b in before:
            del aliases[b]
    root = tuple((*row, colors[row[-1] - 1]) for row in record["root_slots"])
    if plan.root != root or set(aliases) != {row[3] for row in root} | set(record["dead"]) \
            or len(set(aliases.values())) != len(aliases):
        raise PlanError("causal root does not preserve every opening/born card exactly once")
    return {"schema": SCHEMA, "law": LAW, "history_sha256": plan.history_sha256,
            "root_sha256": plan.root_sha256, "plan_sha256": _sha(plan.record()),
            "tokens": len(colors), "cuts": len(plan.cuts), "root_cards": len(root),
            "native_replay_admitted": False}
