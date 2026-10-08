"""Current-root planner routing/coverage uses real RootLines, with fake engines and service forwards."""
from contextlib import contextmanager
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from mirrorforce.netduel import agent_anytime as A, current_particles as B, current_root_protocol as C
from mirrorforce.netduel import agent_search_policy as P
from tools import mf_runtime_policy_service as S


def identity_for(config):
    """Synthetic registration for planner-only tests, not actual numeric admission."""
    identity = {"schema": P.current_identity_schema(config), "law": "current-root-in-place/v1",
                "settings": P.search_settings(config), "particles": dict(C.PARTICLE_IDENTITY),
                "memory_law": C.MEMORY_LAW, "rollout_resources": P.current_resources(config)}
    if config.budget_law == P.MULTI.LAW:
        from mirrorforce.netduel.agent_stripe_contract import SCHEMA
        from mirrorforce.agent.inference_geometry import FIXED_PUBLIC_B64
        identity["stripe_contract"] = {"schema":SCHEMA,"law":P.MULTI.LAW,"resources":dict(P.MULTI.RESOURCES),
            "passed_exact":True,**{k:"a"*64 for k in ("report_sha256","worker_sha256","config_sha256",
                "parameters_sha256","trace_sha256")},"candidate_rows":2,"trace_forwards":2,
            "depth":config.depth,"particles":config.particles,"td_lambda":1.,
            "inference_geometry":dict(FIXED_PUBLIC_B64),
            "scope":"registered finite original public current-root full-bank continuations only",
            "full_search_budget_accepted":False}
    return identity


@pytest.mark.parametrize("rows", [2, 17])
@pytest.mark.parametrize("exhaustion", [False, True, "late_clone"])
@pytest.mark.parametrize("law", [A.LAW, P.MULTI.LAW])
def test_shared_current_roots_are_activated_before_snapshots_and_never_replay(monkeypatch, rows, exhaustion, law):
    now, active, serial = [0.], [-1], [0]
    monkeypatch.setattr(P.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(P, "synthetic_roots", lambda *_a, **_k: pytest.fail("current-root mode replayed an opening"))
    saved, freed, closed, completed, open_bases, requests = [], [], [], [], set(), []
    driver = SimpleNamespace(pduel=7, save_pystate=lambda: {"particle": active[0]},
                             restore_pystate=lambda state: active.__setitem__(0, state["particle"]))

    def snapshot(duel):
        assert duel == 7
        saved.append(active[0])
        return len(saved)

    root = SimpleNamespace(deadline=100., _check=lambda: None,
        owner=SimpleNamespace(follower=SimpleNamespace(viewer=0)),
        api=SimpleNamespace(duel_snapshot=snapshot, duel_snapshot_free=lambda snap: freed.append(snap),
                            duel_rollback=lambda *_: 0))
    particles = []
    for index in range(2):
        proof = {"schema": B.SCHEMA, "root_id": "fixture-root", "root_hash": "d"*64, "hypothesis_hash": str(index)*64,
                 "viewer": 1, "index": index, "proposal_seed": 31, "proposal_law": C.LAW,
                 "native_adapter": "fixed-sky-all-turns/v1", "adapter_binding": "e"*64, "view_sha256": "f"*64,
                 "memory_law": C.MEMORY_LAW, "opening_replayed": False, "own_deck_order_realized": True,
                 "root_response_law": C.ROOT_RESPONSE_LAW, "root_menu_binding_sha256": "7"*64,
                 "training_eligible": False}
        particles.append(SimpleNamespace(driver=driver, prompt=None, proof=proof,
            map_root_response=lambda value: value,
            stream={"op": "open_root_view", "fixture": index}, activate=lambda i=index: active.__setitem__(0, i)))

    class Realizations(list):
        def complete(self, index):
            completed.append(index)

    @contextmanager
    def opened(self, owned, recipe, *, seed, deadline):
        assert owned is root and seed == 31 and deadline == 3.
        if exhaustion is True:
            now[0] = 3.
        yield Realizations(particles)
        assert not open_bases

    monkeypatch.setattr(B.CurrentRootParticles, "open_roots", opened)
    bank = object.__new__(B.CurrentRootParticles)
    bank.own_session, bank.obs_sha256 = "real", "a"*64
    bank.weights, bank.assignments = (.5, .5), ((), ())
    bank.root_hash, bank.root_id = "d"*64, "fixture-root"
    service = S.Service(None, S.UniformBackend(), {}, "greedy", 1., {})

    def call(request):
        requests.append(copy.deepcopy(request))
        op = request["op"]
        if op == "open_root_view":
            assert len(open_bases) < (2 if law == P.MULTI.LAW else 1)
            name = "base" + str(request["fixture"])
            open_bases.add(name)
            return {"session": name}
        if op == "clone":
            serial[0] += 1
            if exhaustion == "late_clone" and serial[0] == 5:
                now[0] = 3.
            return {"session": "line"+str(serial[0])}
        if op == "rollout_step":
            return {"items": [{"cut": True, "decisions": [{"wdl": [.6, .2, .2]}]}] * len(request["items"])}
        if op == "close":
            closed.append(request["session"])
            open_bases.discard(request["session"])
            return {}
        if op == "root_step":
            assert not open_bases
            return service.root_step(request)
        pytest.fail("unexpected request " + op)

    config = P.SearchConfig(rollouts=2*rows, depth=1, seconds=3., selection="greedy", budget_law=law,
                            candidate_law=P.ALL_CANDIDATES, total_seconds=40., particles=2)
    pending = {"rows": rows, "logits": list(np.linspace(0, 1, rows)), "groups": [0]*rows, "obs_sha256": "a"*64}
    chosen, record = P.plan_root(root, pending, "real", bank, {}, call=call, config=config, seed=31,
                                 rng=np.random.default_rng(5), deadline=3.)
    assert saved == [0, 1] and sorted(freed) == [1, 2]  # not [last, last] on a shared driver
    assert record["schema"] == P.current_root_schema(config) and record["opening_replayed"] is False
    expected = [0] if exhaustion == "late_clone" and rows == 2 and law == A.LAW else [] if exhaustion else [0, 1]
    assert sorted(completed) == expected
    assert record["anytime"]["completed_stripes"] == len(expected)
    assert record["candidate_rows"] == list(range(rows)) and record["particle_weights"] == [.5, .5]
    assert not open_bases and "real" not in closed
    assert len(closed) == serial[0] + requests.count({"op":"open_root_view","fixture":0}) \
        + requests.count({"op":"open_root_view","fixture":1})
    assert record["rollout_resources"]["peak_line_sessions"] <= (32 if law == P.MULTI.LAW else 16)
    assert record["rollout_resources"]["peak_base_sessions"] <= (2 if law == P.MULTI.LAW else 1)
    assert all(r["op"] != "open_stream" for r in requests)
    decision = {**pending, "viewer": 0, "chosen": chosen, "p_chosen": record["policy"][chosen]}
    identity = identity_for(config)
    P.check_root_record(record, decision, identity, native_snapshot_sha256="d"*64)
    for change in (lambda r: r.update(opening_replayed=True), lambda r: r.update(memory_law="copied-real-opponent"),
                   lambda r: r.update(particle_weights=[.4, .6]),
                   lambda r: r["particle_proofs"][0].update(root_hash="b"*64),
                   lambda r: r["particle_proofs"][0].pop("root_menu_binding_sha256"),
                   lambda r: r["particle_proofs"][0].update(root_response_law="unmapped-response"),
                   lambda r: r["particle_proofs"][0].update(index=1)):
        broken = copy.deepcopy(record)
        change(broken)
        with pytest.raises(ValueError, match="current-root"):
            P.check_root_record(broken, decision, identity, native_snapshot_sha256="d"*64)
    if law == P.MULTI.LAW:
        for field, value in (("schema","mirrorforce_current_root_identity/v1"),
                             ("rollout_resources",P.REPLAY_FILTERED_RESOURCES),("stripe_contract",None)):
            with pytest.raises(ValueError,match="current-root"):
                P.check_root_record(record,decision,{**identity,field:value},native_snapshot_sha256="d"*64)


@pytest.mark.parametrize("law", [A.LAW, P.MULTI.LAW])
def test_clock_only_no_proposal_result_keeps_prior_and_requires_exact_live_root(monkeypatch, law):
    monkeypatch.setattr(P.time, "monotonic", lambda: 10.)
    checked = []
    root = SimpleNamespace(epoch=3, _check=lambda: checked.append("root"),
                           snapshot=SimpleNamespace(digest="d"*64, verify=lambda: checked.append("snapshot")))
    config = P.SearchConfig(rollouts=4, selection="greedy", budget_law=law, total_seconds=40.,
                            candidate_law=P.ALL_CANDIDATES, particles=2)
    pending = {"rows": 2, "logits": [0., 1.], "groups": [0, 0], "obs_sha256": "a"*64}
    with pytest.raises(P.SearchPolicyError, match="actually exhausted"):
        P.current_budget_result(root, pending, config=config, seed=31, deadline=11., phase="before_proposals")
    chosen, record = P.current_budget_result(root, pending, config=config, seed=31, deadline=9., phase="before_proposals")
    assert checked == ["root", "snapshot"] and chosen == 1
    assert record["q"] == [None, None] and record["particle_proofs"] == [] and not record["own_deck_order_realized"]
    assert record["policy"] == record["prior"] and record["anytime"]["planned_stripes"] == 0
    identity = identity_for(config)
    P.check_root_record(record, {**pending, "viewer": 0, "chosen": chosen, "p_chosen": record["policy"][chosen]},
                        identity, native_snapshot_sha256="d"*64)
    root.snapshot.verify = lambda: (_ for _ in ()).throw(RuntimeError("damaged real root"))
    with pytest.raises(RuntimeError, match="damaged real root"):
        P.current_budget_result(root, pending, config=config, seed=31, deadline=9., phase="before_proposals")


def test_current_root_full_denominator_keeps_failed_prompt_and_budget_zero():
    roots = [{"complete": True, "non_singleton_attempts": [1],
              "searches": [{"anytime": {"completed_stripes": 2}, "particle_proofs": [1, 2]}]},
             {"complete": True, "non_singleton_attempts": [1], "searches": [{"current_budget_empty": True}]},
             {"complete": False, "non_singleton_attempts": [1], "searches": []},
             {"complete": False, "searches": []}]
    summary = P.current_root_summary(roots)
    assert summary["wire_prompt_attempts"] == 4 and summary["failed_wire_prompts"] == 2
    assert summary["model_attempts"] == 3 and summary["positive_search_fraction"] == 1/3
    assert summary["zero_proposal_budget_roots"] == 1 and summary["realized_particles"] == 2


def test_current_identity_is_separate_and_requires_explicit_service_memory_capability():
    from mirrorforce.netduel import search_rules as R
    service = {"backend": "checkpoint", "weights": "iterate", "magnet": "uniform_legal",
               "opponent_recipe_mode": "mirror", "selection": "greedy", "compute_dtype": "bfloat16",
               "native_build": {"core_commit": "abc", "outputs": {"libmfcore.so": "a"*64}},
               "search_batching": {"law": "valid-row-padding/v1", "sizes": [1]},
               "current_root": dict(C.CAPABILITY)}
    rules = {"schema": R.SCHEMA, "core_sha256": "a"*64, "core_commit": "abc", "training_eligible": False}
    config = P.SearchConfig(rollouts=4, selection="greedy", budget_law=A.LAW, total_seconds=40.,
                            candidate_law=P.ALL_CANDIDATES, particles=2)
    identity = P.create_search_identity(service, rules, dict(C.PARTICLE_IDENTITY), config)
    assert identity["law"] == "current-root-in-place/v1" and identity["schema"] == "mirrorforce_current_root_identity/v1"
    assert identity["memory_law"] == C.MEMORY_LAW and "producer" not in identity
    for bad in ({**service, "current_root": None}, {**service, "current_root": {**C.CAPABILITY, "memory_law": "borrowed"}}):
        with pytest.raises(P.SearchPolicyError, match="explicit observer"):
            P.create_search_identity(bad, rules, dict(C.PARTICLE_IDENTITY), config)
    with pytest.raises(P.SearchPolicyError, match="explicit observer"):
        P.create_search_identity(service, rules, {**C.PARTICLE_IDENTITY, "belief": "AR"}, config)
