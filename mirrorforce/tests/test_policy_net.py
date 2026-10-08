"""The structure in JAX (mirrorforce/agent/model/policy_net.py): the decision-by-decision runner and the learner's segment
runner give the same outputs and the same memory, and the production parameter count is pinned.

Runs in the venv (JAX on CPU). The parity checks use a small width and synthetic observations of the key layout
produced by a model of the environment's delivery protocol (``Protocol``): turns grow by random rows with bursts, rows
beyond a 10-row window leave in chunks of 4, each chunk and each completed turn goes to each observer once, in order,
at most 3 chunks and 2 completed turns per decision (the rest queued); seats act irregularly, so a seat may receive
two completed turns, the chunks of the next turn and a queue backlog at one decision. The carried memory starts full
so the window drops its oldest summaries, and the summary row blocks switch between 4, 6 and 10 rows.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mirrorforce.agent.model import policy_net
from mirrorforce.agent.model.policy_net import Memory, Prefix, PolicyNet, PolicyNetConfig, parameter_count

act = jax.jit(policy_net.act, static_argnums=0)  # compiled as in production (the model is static)
segment = jax.jit(policy_net.segment, static_argnums=0, static_argnames=("belief",))

SMALL = PolicyNetConfig(d=32, heads=4, ff=64, state_layers=2, turn_layers=2, readout_layers=1, semantic_dim=16,
                    event_hidden=48, action_hidden=48, readout_hidden=40, value_queries=2, memory_slots=4,
                    chunk_slots=5, row_blocks=(4, 6))
WINDOW, CHUNK_ROWS, CHUNK_DELIVERY = 10, 4, 3
SMALL_SEMANTICS = (50, 6, 3, 5)
PRODUCTION_SEMANTICS = (14969, 78, 16, 64)


def shapes(cards=12, events=WINDOW, options=6, recipe=10, activations=8, hints=4, chunk_rows=CHUNK_ROWS,
           chunk_delivery=CHUNK_DELIVERY, event_width=24):
    return {
        "cards_": (cards, 41), "global_": (25,), "selection_": (12,), "card_turn_": (cards, 8),
        "own_recipe_": (recipe, 6),
        "turn_ledger_": (2, 16), "player_hints_": (hints, 5), "chain_": (8, 8), "unpositioned_": (32, 4),
        "opponent_recipe_": (recipe, 4), "turn_activations_": (activations, 12),
        "turn_events_": (events, event_width), "turn_event_refs_": (events, 4),
        "closed_turns_": (2, events, event_width), "closed_turn_refs_": (2, events, 4), "closed_turn_meta_": (2, 5),
        "turn_chunks_": (chunk_delivery, chunk_rows, event_width), "turn_chunk_refs_": (chunk_delivery, chunk_rows, 4),
        "turn_chunk_meta_": (chunk_delivery, 4),
        "action_ir_": (options, 24), "action_single_refs_": (options, 4),
        "action_group_refs_": (options, 5, 8), "action_group_mask_": (options, 5, 8), "candidates_": (options, 3),
    }


def random_obs(rng: np.random.Generator, lead: tuple, layout: dict, num_cards: int):
    obs = {k: rng.integers(0, 256, lead + s, dtype=np.uint8) for k, s in layout.items()}
    cards = layout["cards_"][0]
    for key, id_cols in (("cards_", (0, 1)), ("chain_", (2, 3)), ("unpositioned_", (0, 1)),
                         ("opponent_recipe_", (0, 1)), ("own_recipe_", (0, 1)), ("turn_activations_", (1, 2))):
        obs[key][..., id_cols[0]] = 0
        obs[key][..., id_cols[1]] = rng.integers(0, num_cards, obs[key].shape[:-1])
    obs["cards_"][..., 2] *= rng.random(obs["cards_"].shape[:-1]) < 0.7  # some padding rows
    for key in ("turn_events_", "closed_turns_", "turn_chunks_"):
        obs[key][..., 0] = 1
        obs[key][..., 6] = 0
        obs[key][..., 7] = rng.integers(0, num_cards, obs[key].shape[:-1])
    for key in ("closed_turn_meta_", "turn_chunk_meta_"):
        obs[key] = np.zeros(obs[key].shape, np.int32)
    for key in ("turn_event_refs_", "closed_turn_refs_", "turn_chunk_refs_"):
        obs[key][..., 0] = rng.integers(0, cards + 1, obs[key].shape[:-1])
        obs[key][..., 2] = rng.integers(0, cards + 1, obs[key].shape[:-1])
    obs["action_ir_"][..., 0] *= rng.random(obs["action_ir_"].shape[:-1]) < 0.8
    obs["action_ir_"][..., 0, 0] = 1
    obs["action_ir_"][..., 15] = 0
    obs["action_ir_"][..., 16] = rng.integers(0, num_cards, obs["action_ir_"].shape[:-1])
    obs["action_single_refs_"] = rng.integers(0, cards + 1, obs["action_single_refs_"].shape).astype(np.uint8)
    obs["action_group_refs_"] = rng.integers(0, cards + 1, obs["action_group_refs_"].shape).astype(np.uint8)
    obs["action_group_mask_"] = (rng.random(obs["action_group_mask_"].shape) < 0.5).astype(np.uint8)
    obs["chain_"][..., 0] = rng.random(obs["chain_"].shape[:-1]) < 0.3
    obs["candidates_"][..., 0] = 0
    obs["candidates_"][..., 1] = rng.integers(0, num_cards, obs["candidates_"].shape[:-1])
    obs["candidates_"][..., 2] = rng.integers(0, 6, obs["candidates_"].shape[:-1])
    obs["unpositioned_"][..., 3] *= rng.random(obs["unpositioned_"].shape[:-1]) < 0.2
    obs["opponent_recipe_"][..., 3] %= 4
    obs["turn_activations_"][..., 0] = rng.random(obs["turn_activations_"].shape[:-1]) < 0.5
    return obs


def left_aligned(array, rows):
    """Row validity (column 0) of windows [..., K, 24]: the first ``rows`` [...] rows are valid."""
    array[..., 0] = np.arange(array.shape[-2]) < np.asarray(rows)[..., None]


class Protocol:
    """One environment's delivery protocol (the model of history_obs.h the runners rely on): the current turn's rows
    [CHUNK_ROWS * c, n) form the window, a chunk leaves whenever the window would exceed WINDOW rows, a completed
    turn's closed window is its last [CHUNK_ROWS * c, n); every chunk and completed turn is queued for both observers
    and delivered in order at their decisions (at most CHUNK_DELIVERY chunks and 2 completed turns per decision)."""

    def __init__(self, rng, first_turn, cap):
        self.rng, self.cap = rng, cap
        self.turn, self.rows, self.chunks = first_turn, 0, 0
        self.queues = ([], [])

    def new_episode(self):
        self.turn, self.rows, self.chunks = 1, 0, 0
        self.queues = ([], [])

    def end_turn(self):
        for q in self.queues:
            q.append(("closed", self.turn, self.rows - CHUNK_ROWS * self.chunks, self.chunks))
        self.turn, self.rows, self.chunks = self.turn + 1, 0, 0

    def add_rows(self, count):
        for _ in range(count):
            if self.rows - CHUNK_ROWS * self.chunks == WINDOW:
                if self.chunks == self.cap:
                    self.end_turn()  # the generator respects the environment's cap
                else:
                    for q in self.queues:
                        q.append(("chunk", self.turn, self.chunks, CHUNK_ROWS))
                    self.chunks += 1
            self.rows += 1

    def deliver(self, seat):
        """(chunk meta [CHUNK_DELIVERY, 4], closed meta [2, 5] oldest first, window rows, backlog)."""
        chunk_meta, closed = np.zeros((CHUNK_DELIVERY, 4), np.int32), []
        queue, used = self.queues[seat], 0
        while queue:
            kind, turn, a, b = queue[0]
            if kind == "chunk":
                if used == CHUNK_DELIVERY:
                    break
                chunk_meta[used] = (1, turn, a, b)
                used += 1
            else:
                if len(closed) == 2:
                    break
                closed.append((1, turn, a, b))
            queue.pop(0)
        closed_meta = np.zeros((2, 5), np.int32)
        for slot, entry in enumerate(closed):  # oldest first, as the environment
            closed_meta[slot, :4] = entry
        return chunk_meta, closed_meta, self.rows - CHUNK_ROWS * self.chunks, len(queue)


def episode(rng: np.random.Generator, steps: int, batch: int, layout: dict, num_cards: int, first_turn: int,
            resets: float = 0.0, cap: int = SMALL.chunk_slots):
    """Observations [T, B] from ``Protocol``; with ``resets`` > 0 new episodes start at random steps (``first``)."""
    obs = random_obs(rng, (steps, batch), layout, num_cards)
    seat = rng.integers(0, 2, (steps, batch)).astype(np.int32)
    first = np.zeros((steps, batch), bool)
    envs = [Protocol(rng, first_turn, cap) for _ in range(batch)]
    for t in range(steps):
        for b, env in enumerate(envs):
            if t > 0 and rng.random() < resets:
                first[t, b] = True
                env.new_episode()
            elif rng.random() < 0.3:
                env.end_turn()
            env.add_rows(int(rng.integers(0, 4)) + (int(rng.integers(8, 30)) if rng.random() < 0.25 else 0))
            chunk_meta, closed_meta, window, _ = env.deliver(seat[t, b])
            obs["turn_chunk_meta_"][t, b], obs["closed_turn_meta_"][t, b] = chunk_meta, closed_meta
            left_aligned(obs["turn_events_"][t, b], window)
            left_aligned(obs["closed_turns_"][t, b], closed_meta[:, 2])
            left_aligned(obs["turn_chunks_"][t, b], chunk_meta[:, 3])
    return (obs, seat, first) if resets else (obs, seat)


def initial_prefix(batch, config):
    positions = lambda n: jnp.broadcast_to(jnp.arange(n), (batch, n))
    return Prefix(jnp.zeros((batch, config.memory_slots, config.d)), jnp.ones((batch, config.memory_slots), bool),
                  positions(config.memory_slots), jnp.zeros((batch, config.chunk_slots, config.d)),
                  jnp.ones((batch, config.chunk_slots), bool), positions(config.chunk_slots))


def make(config, semantics, layout, batch, rng):
    model = PolicyNet(config, semantics)
    obs = jax.tree_util.tree_map(jnp.asarray, random_obs(rng, (batch,), layout, semantics[0]))
    variables = model.init(jax.random.PRNGKey(0), obs, initial_prefix(batch, config), method=PolicyNet.init_all)
    constants = {"semantics": {
        "cdb_exact": jnp.asarray(rng.normal(size=semantics[:2]), jnp.float32),
        "lua_effects": jnp.asarray(rng.normal(size=(semantics[0], semantics[2], semantics[3])), jnp.float16),
        "lua_mask": jnp.asarray(rng.random((semantics[0], semantics[2])) < 0.6, jnp.uint8),
        "setcode_bag": jnp.asarray(rng.random((semantics[0], 32)) < 0.1, jnp.float32),
        "link_arrows": jnp.asarray(rng.random((semantics[0], 8)) < 0.2, jnp.float32),
        "genericity": jnp.asarray(rng.random((semantics[0], 1)), jnp.float32),
        "effect_unit_bits": jnp.asarray(rng.integers(0, 1 << semantics[2], (semantics[0], 14)), jnp.uint16)}}
    variables = dict(variables, constants=constants)
    return model, variables


def carried(rng, batch, config):
    zeros = Memory.zeros(batch, config)
    return zeros._replace(buffer=jnp.asarray(rng.normal(size=(batch, 2, config.memory_slots, config.d)), jnp.float32),
                          count=jnp.asarray([[3, 7], [0, 1], [5, 0]][:batch], jnp.int32),
                          last_turn=jnp.asarray([[0, 1], [0, 0], [1, 0]][:batch], jnp.int32))


def assert_same_memory(a: Memory, b: Memory):
    for name in ("count", "last_turn", "chunk_count", "chunk_turn"):
        np.testing.assert_array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)), err_msg=name)
    for name in ("buffer", "chunks"):
        np.testing.assert_allclose(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)), atol=1e-5, err_msg=name)


@pytest.fixture(scope="module")
def small():
    rng = np.random.default_rng(11)
    layout = shapes()
    model, variables = make(SMALL, SMALL_SEMANTICS, layout, 3, rng)
    return model, variables, layout


def run_both(model, variables, layout, memory, rng, steps, first_turn):
    obs, seat = episode(rng, steps, memory.count.shape[0], layout, SMALL_SEMANTICS[0], first_turn)
    obs = jax.tree_util.tree_map(jnp.asarray, obs)
    seat = jnp.asarray(seat)
    loop_memory, per_step = memory, []
    for t in range(steps):
        logits, value, wdl, loop_memory, bad = act(model, variables, jax.tree_util.tree_map(lambda x: x[t], obs),
                                                   seat[t], loop_memory)
        assert not bool(bad.any())
        per_step.append((logits, value, wdl))
    logits, value, wdl, seg_memory, plan = segment(model, variables, obs, seat, memory)
    return obs, per_step, (logits, value, wdl), loop_memory, seg_memory, plan


def test_the_segment_runner_matches_the_decision_runner(small):
    model, variables, layout = small
    rng = np.random.default_rng(5)
    memory = carried(rng, 3, SMALL)
    obs, per_step, (logits, value, wdl), loop_memory, seg_memory, plan = run_both(
        model, variables, layout, memory, rng, steps=24, first_turn=2)
    assert not bool(plan.bad.any())
    turns = int((plan.valid & (plan.kind == 0)).sum())
    chunks = int((plan.valid & (plan.kind == 1)).sum())
    assert turns >= 8 and chunks >= 8, (turns, chunks)  # completed turns and chunks inside the segment
    valid = np.asarray(obs["action_ir_"][..., 0] > 0)
    for t, (l, v, w) in enumerate(per_step):
        np.testing.assert_allclose(np.asarray(logits[t])[valid[t]], np.asarray(l)[valid[t]], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(value[t]), np.asarray(v), atol=1e-5)
        np.testing.assert_allclose(np.asarray(wdl[t]), np.asarray(w), atol=1e-5)
    assert_same_memory(seg_memory, loop_memory)


def test_memory_carries_across_segments(small):
    """Two consecutive segments through the learner's runner equal one decision-by-decision pass."""
    model, variables, layout = small
    rng = np.random.default_rng(9)
    memory = carried(rng, 3, SMALL)
    obs, seat = episode(rng, 20, 3, layout, SMALL_SEMANTICS[0], first_turn=1)
    obs = jax.tree_util.tree_map(jnp.asarray, obs)
    seat = jnp.asarray(seat)
    loop_memory, loop_values, middle_loop = memory, [], None
    for t in range(20):
        _, value, _, loop_memory, _ = act(model, variables, jax.tree_util.tree_map(lambda x: x[t], obs), seat[t],
                                          loop_memory)
        loop_values.append(np.asarray(value))
        if t == 9:
            middle_loop = loop_memory
    first = jax.tree_util.tree_map(lambda x: x[:10], obs)
    second = jax.tree_util.tree_map(lambda x: x[10:], obs)
    _, value_a, _, middle, _ = segment(model, variables, first, seat[:10], memory)
    _, value_b, _, end, _ = segment(model, variables, second, seat[10:], middle)
    assert int(np.asarray(middle.chunk_count).sum()) > 0  # a chunk carry crosses the boundary
    assert_same_memory(middle, middle_loop)
    np.testing.assert_allclose(np.concatenate([np.asarray(value_a), np.asarray(value_b)]), np.stack(loop_values),
                               atol=1e-5)
    assert_same_memory(end, loop_memory)


def test_an_inconsistent_delivery_is_reported_by_both_runners(small):
    """A skipped chunk index, a completed turn whose chunk count disagrees with the carry, and a chunk beyond the
    cap each raise on the host (the runners return the flag)."""
    model, variables, layout = small
    rng = np.random.default_rng(3)
    obs, seat = episode(rng, 24, 3, layout, SMALL_SEMANTICS[0], first_turn=2)
    plan = segment(model, variables, jax.tree_util.tree_map(jnp.asarray, obs), jnp.asarray(seat),
                   Memory.zeros(3, SMALL))[4]
    assert not bool(plan.bad.any())
    for corrupt in ("index", "count"):
        broken = {k: v.copy() for k, v in obs.items()}
        if corrupt == "index":
            t, b, j = map(int, np.argwhere(broken["turn_chunk_meta_"][..., 0] > 0)[0])
            broken["turn_chunk_meta_"][t, b, j, 2] += 1
        else:
            t, b, k = map(int, np.argwhere((broken["closed_turn_meta_"][..., 0] > 0)
                                           & (broken["closed_turn_meta_"][..., 3] > 0))[0])
            broken["closed_turn_meta_"][t, b, k, 3] -= 1
        broken = jax.tree_util.tree_map(jnp.asarray, broken)
        plan = segment(model, variables, broken, jnp.asarray(seat), Memory.zeros(3, SMALL))[4]
        assert bool(plan.bad[b]), corrupt
        memory, flagged = Memory.zeros(3, SMALL), False
        for step in range(t + 1):
            memory, bad = act(model, variables, jax.tree_util.tree_map(lambda x: x[step], broken), jnp.asarray(seat[step]),
                              memory)[3:]
            flagged |= bool(bad[b])
        assert flagged, corrupt
    assert int(obs["turn_chunk_meta_"][..., 2].max()) >= 1  # a turn with two chunks ...
    tight = dataclasses.replace(SMALL, chunk_slots=1)
    tight_variables = make(tight, SMALL_SEMANTICS, layout, 3, np.random.default_rng(1))[1]
    plan = segment(PolicyNet(tight, SMALL_SEMANTICS), tight_variables, jax.tree_util.tree_map(jnp.asarray, obs),
                   jnp.asarray(seat), Memory.zeros(3, tight))[4]
    assert bool(plan.bad.any())  # ... is more than chunk_slots = 1


def test_a_repeated_observation_leaves_the_state_as_if_dropped(small):
    """illegal_activation_withdrawal/v1 re-presents a withdrawn decision's observation exactly, deliveries included:
    processing both (the withdrawn decision and its re-presentation) gives every later output and the carry of
    processing it once, in both runners."""
    model, variables, layout = small
    rng = np.random.default_rng(51)
    obs, seat = episode(rng, 20, 3, layout, SMALL_SEMANTICS[0], first_turn=2)
    delivered = (obs["turn_chunk_meta_"][..., 0].sum(-1) > 0) & (obs["closed_turn_meta_"][..., 0].sum(-1) > 0)
    t = int(np.argwhere(delivered.any(-1))[0][0])  # a step delivering chunks and a completed turn
    repeat = lambda x: np.concatenate([x[:t + 1], x[t:t + 1], x[t + 1:]])
    twice = {k: repeat(v) for k, v in obs.items()}
    once_out = segment(model, variables, jax.tree_util.tree_map(jnp.asarray, obs), jnp.asarray(seat),
                       Memory.zeros(3, SMALL))
    twice_out = segment(model, variables, jax.tree_util.tree_map(jnp.asarray, twice), jnp.asarray(repeat(seat)),
                        Memory.zeros(3, SMALL))
    assert not bool(once_out[4].bad.any()) and not bool(twice_out[4].bad.any())
    keep = np.array([i for i in range(len(seat) + 1) if i != t + 1])
    np.testing.assert_allclose(np.asarray(twice_out[1])[keep], np.asarray(once_out[1]), atol=1e-5)
    assert_same_memory(twice_out[3], once_out[3])
    memory = Memory.zeros(3, SMALL)
    for step in range(len(seat) + 1):
        memory, bad = act(model, variables, jax.tree_util.tree_map(lambda x: jnp.asarray(x[step]), twice),
                          jnp.asarray(repeat(seat)[step]), memory)[3:]
        assert not bool(bad.any())
    assert_same_memory(memory, once_out[3])


def test_the_turn_pass_shapes_do_not_change_the_outputs(small):
    model, variables, layout = small
    rng = np.random.default_rng(31)
    obs, seat = episode(rng, 16, 3, layout, SMALL_SEMANTICS[0], first_turn=2)
    obs, seat = jax.tree_util.tree_map(jnp.asarray, obs), jnp.asarray(seat)
    blocked = segment(model, variables, obs, seat, Memory.zeros(3, SMALL))
    whole = segment(PolicyNet(dataclasses.replace(SMALL, row_blocks=()), SMALL_SEMANTICS), variables, obs, seat,
                    Memory.zeros(3, SMALL))
    np.testing.assert_allclose(np.asarray(blocked[1]), np.asarray(whole[1]), atol=1e-5)
    assert_same_memory(blocked[3], whole[3])


def test_chunk_rows_reach_decisions_only_through_their_summaries(small):
    """Changing a chunk's rows changes nothing before its delivery and changes later decisions of that seat."""
    model, variables, layout = small
    rng = np.random.default_rng(41)
    obs, seat = episode(rng, 20, 3, layout, SMALL_SEMANTICS[0], first_turn=2)
    t, b, j = map(int, np.argwhere(obs["turn_chunk_meta_"][..., 0] > 0)[0])
    other = {k: v.copy() for k, v in obs.items()}
    other["turn_chunks_"][t, b, j, :, 7] = (other["turn_chunks_"][t, b, j, :, 7] + 1) % SMALL_SEMANTICS[0]
    a = segment(model, variables, jax.tree_util.tree_map(jnp.asarray, obs), jnp.asarray(seat), Memory.zeros(3, SMALL))
    c = segment(model, variables, jax.tree_util.tree_map(jnp.asarray, other), jnp.asarray(seat),
                Memory.zeros(3, SMALL))
    np.testing.assert_allclose(np.asarray(a[1])[:t, b], np.asarray(c[1])[:t, b], atol=1e-6)
    later = seat[t:, b] == seat[t, b]
    assert not np.allclose(np.asarray(a[1])[t:, b][later], np.asarray(c[1])[t:, b][later], atol=1e-6)


def test_the_memory_reads_completed_turns_only_through_their_summaries(small):
    """Changing a completed turn's window changes later decisions only once that turn is summarized."""
    model, variables, layout = small
    rng = np.random.default_rng(21)
    obs, seat = episode(rng, 6, 3, layout, SMALL_SEMANTICS[0], first_turn=2)
    obs["closed_turn_meta_"][:] = 0  # no completed turn yet ...
    obs["turn_chunk_meta_"][:] = 0
    obs["closed_turn_meta_"][4:, :, 0, :3] = (1, 3, WINDOW)  # ... until step 4
    other = {k: v.copy() for k, v in obs.items()}
    other["closed_turns_"][:4] = 255 - other["closed_turns_"][:4]  # windows never summarized
    a = segment(model, variables, jax.tree_util.tree_map(jnp.asarray, obs), jnp.asarray(seat), Memory.zeros(3, SMALL))
    b = segment(model, variables, jax.tree_util.tree_map(jnp.asarray, other), jnp.asarray(seat),
                Memory.zeros(3, SMALL))
    np.testing.assert_allclose(np.asarray(a[1]), np.asarray(b[1]), atol=1e-6)


def test_the_segment_summaries_backward_equals_autodiff(small, monkeypatch):
    """The custom VJP of the segment's summaries (forward skips, backward recomputes used entries from the final
    arrays) gives the gradients of plain autodiff through the same forward, for every parameter and the carry."""
    model, variables, layout = small
    rng = np.random.default_rng(13)
    memory = carried(rng, 3, SMALL)
    obs, seat = episode(rng, 16, 3, layout, SMALL_SEMANTICS[0], first_turn=2)
    obs, seat = jax.tree_util.tree_map(jnp.asarray, obs), jnp.asarray(seat)
    consts = {k: v for k, v in variables.items() if k != "params"}

    def loss(params, buffer):
        logits, value, wdl, after, _ = policy_net.segment(model, {"params": params, **consts}, obs, seat,
                                                     memory._replace(buffer=buffer))
        return (value ** 2).sum() + jnp.where(logits > -1e8, logits, 0).sum() * 0.01 + (after.buffer ** 2).sum() \
            + (after.chunks ** 2).sum()
    grad = jax.jit(jax.grad(loss, argnums=(0, 1)))
    custom = grad(variables["params"], memory.buffer)
    monkeypatch.setattr(policy_net, "_segment_summaries",
                        lambda m, *args: policy_net._segment_summaries_forward(m, *args))
    plain = jax.jit(jax.grad(loss, argnums=(0, 1)))(variables["params"], memory.buffer)
    norm = lambda tree: float(jnp.sqrt(sum((x.astype(jnp.float64) ** 2).sum() for x in jax.tree_util.tree_leaves(tree))))
    difference = jax.tree_util.tree_map(lambda a, b: a - b, custom, plain)
    assert norm(difference) < 1e-5 * norm(plain), (norm(difference), norm(plain))  # float32 summation order only
    tangent = jax.tree_util.tree_map(lambda x: jnp.asarray(rng.normal(size=x.shape), x.dtype), variables["params"])
    _, directional = jax.jit(lambda p, t: jax.jvp(lambda q: loss(q, memory.buffer), (p,), (t,)))(variables["params"],
                                                                                                tangent)
    along = sum(float((g * t).sum()) for g, t in zip(jax.tree_util.tree_leaves(custom[0]),
                                                      jax.tree_util.tree_leaves(tangent)))
    assert along == pytest.approx(float(directional), rel=1e-4)  # forward mode agrees
    assert any(float(jnp.abs(g).max()) > 0 for g in jax.tree_util.tree_leaves(custom[0]["summary_projection"]))


def test_the_hand_limit_inputs_enter_zero_initialized():
    """--net.hand-limit: obs:hand_limit_ into the state token and obs:action_discard_ into the action rows, both zero
    at initialization (outputs equal for any values) and live once trained (a changed weight moves the outputs)."""
    rng = np.random.default_rng(23)
    layout = dict(shapes(), hand_limit_=(2, 4), action_discard_=(6,))
    config = dataclasses.replace(SMALL, hand_limit=True)
    model, variables = make(config, SMALL_SEMANTICS, layout, 3, rng)
    obs = random_obs(rng, (3,), layout, SMALL_SEMANTICS[0])
    other = dict(obs, hand_limit_=255 - obs["hand_limit_"], action_discard_=(obs["action_discard_"] + 3) % 16)
    run = lambda v, o: act(model, v, jax.tree_util.tree_map(jnp.asarray, o), jnp.zeros(3, jnp.int32),
                           Memory.zeros(3, config))
    a, b = run(variables, obs), run(variables, other)
    np.testing.assert_allclose(np.asarray(a[1]), np.asarray(b[1]), atol=1e-6)
    params = jax.tree_util.tree_map(lambda x: x, variables["params"])
    params["inputs"]["hand_limit"]["kernel"] = jnp.asarray(  # (a constant column would vanish in the token's norm)
        rng.normal(size=params["inputs"]["hand_limit"]["kernel"].shape), jnp.float32)
    params["inputs"]["action_discard"]["embedding"] = jnp.asarray(
        rng.normal(size=params["inputs"]["action_discard"]["embedding"].shape), jnp.float32)
    trained = dict(variables, params=params)
    c, d = run(trained, obs), run(trained, other)
    assert not np.allclose(np.asarray(c[1]), np.asarray(d[1]), atol=1e-6)
    valid = obs["action_ir_"][..., 0] > 0
    assert not np.allclose(np.asarray(c[0])[valid], np.asarray(d[0])[valid], atol=1e-6)


def test_permuting_the_menu_permutes_the_logits(small):
    """Candidates are scored by content, never by menu position (action_ir_ column 22 is the position and is not
    read): reordering a menu's rows (with their references) reorders the logits and leaves the value unchanged."""
    model, variables, layout = small
    rng = np.random.default_rng(29)
    obs = random_obs(rng, (3,), layout, SMALL_SEMANTICS[0])
    obs["action_ir_"][..., 0] = 1  # every row offered
    perm = np.array([3, 0, 5, 1, 4, 2])
    moved = dict(obs)
    for key in ("action_ir_", "action_single_refs_", "action_group_refs_", "action_group_mask_", "candidates_"):
        moved[key] = obs[key][:, perm]
    moved["action_ir_"][..., 22] = obs["action_ir_"][..., 22]  # the position column stays positional
    run = lambda o: act(model, variables, jax.tree_util.tree_map(jnp.asarray, o), jnp.zeros(3, jnp.int32),
                        Memory.zeros(3, SMALL))
    a, b = run(obs), run(moved)
    np.testing.assert_allclose(np.asarray(b[0]), np.asarray(a[0])[:, perm], atol=1e-5)
    np.testing.assert_allclose(np.asarray(b[1]), np.asarray(a[1]), atol=1e-5)


def test_the_public_status_inputs():
    """--net.public-status: card statuses on the card tokens, lingering effects as state tokens, 27-wide event rows;
    both runners agree, and each new input moves the outputs."""
    rng = np.random.default_rng(37)
    config = dataclasses.replace(SMALL, public_status=True)
    layout = dict(shapes(event_width=27), card_status_=(12, 6), public_effects_=(16, 8))
    model, variables = make(config, SMALL_SEMANTICS, layout, 3, rng)
    obs, seat = episode(rng, 12, 3, layout, SMALL_SEMANTICS[0], first_turn=2)
    obs["public_effects_"][..., 0] = rng.random(obs["public_effects_"].shape[:-1]) < 0.5
    obs["card_status_"][..., 0] %= 13
    obs["card_status_"][..., 4] %= 13
    jobs, jseat = jax.tree_util.tree_map(jnp.asarray, obs), jnp.asarray(seat)
    out = segment(model, variables, jobs, jseat, Memory.zeros(3, config))
    memory, values = Memory.zeros(3, config), []
    for t in range(12):
        _, value, _, memory, bad = act(model, variables, jax.tree_util.tree_map(lambda x: x[t], jobs), jseat[t], memory)
        values.append(np.asarray(value))
    np.testing.assert_allclose(np.asarray(out[1]), np.stack(values), atol=1e-5)
    for key, change in (("card_status_", lambda a: (a + 1) % 7), ("public_effects_", lambda a: 255 - a),
                        ("turn_events_", lambda a: np.concatenate([a[..., :24], 255 - a[..., 24:]], axis=-1))):
        other = dict(obs, **{key: change(obs[key])})
        moved = segment(model, variables, jax.tree_util.tree_map(jnp.asarray, other), jseat, Memory.zeros(3, config))
        assert not np.allclose(np.asarray(moved[1]), np.asarray(out[1]), atol=1e-6), key


def test_the_policy_refuses_the_other_seats_private_export(small):
    """priv: keys (the env's both-seat export for a central critic) never enter the policy."""
    from mirrorforce.agent.model.reference_agent import ReferenceAgent
    model, variables, layout = small
    agent = ReferenceAgent(SMALL, SMALL_SEMANTICS)
    obs = jax.tree_util.tree_map(jnp.asarray, random_obs(np.random.default_rng(1), (2,), layout, SMALL_SEMANTICS[0]))
    rstate = agent.init_rnn_state(2)
    agent.apply(variables, obs, (rstate, rstate))  # accepted
    with pytest.raises(ValueError, match="private"):
        agent.apply(variables, dict(obs, **{"priv:cards_": obs["cards_"]}), (rstate, rstate))


def test_the_production_parameter_count():
    config = PolicyNetConfig()
    layout = shapes(cards=160, events=512, options=192, recipe=80, activations=64, hints=16, chunk_rows=64,
                    chunk_delivery=4)
    model = PolicyNet(config, PRODUCTION_SEMANTICS)

    def init():
        obs = {k: jnp.zeros((1,) + s, jnp.uint8) for k, s in layout.items()}
        return model.init(jax.random.PRNGKey(0), obs, initial_prefix(1, config), method=PolicyNet.init_all)

    variables = jax.eval_shape(init)
    count = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(variables["params"]))
    assert count == PRODUCTION_PARAMETERS, count
    assert parameter_count.__name__  # the helper counts the same tree


PRODUCTION_PARAMETERS = 112_022_562  # MT input set, belief head (1.98M), chunk positions (48 x 768), no menu position


def test_episodes_starting_inside_a_segment_clear_the_memory(small):
    model, variables, layout = small
    rng = np.random.default_rng(17)
    memory = carried(rng, 3, SMALL)
    obs, seat, first = episode(rng, 20, 3, layout, SMALL_SEMANTICS[0], first_turn=3, resets=0.15)
    assert first.any()
    obs = jax.tree_util.tree_map(jnp.asarray, obs)
    seat, first = jnp.asarray(seat), jnp.asarray(first)
    loop_memory, loop_values = memory, []
    for t in range(20):
        _, value, _, loop_memory, _ = act(model, variables, jax.tree_util.tree_map(lambda x: x[t], obs), seat[t],
                                          loop_memory, first[t])
        loop_values.append(np.asarray(value))
    _, value, _, after, plan = segment(model, variables, obs, seat, memory, first)
    assert not bool(plan.bad.any())
    np.testing.assert_allclose(np.asarray(value), np.stack(loop_values), atol=1e-5)
    assert_same_memory(after, loop_memory)


def test_bfloat16_keeps_the_activations_in_bfloat16(small):
    """With dtype bfloat16 the state block and turn pass run in bfloat16 (no silent promotion to float32 from a
    float32 lookup or embedding), and a decision is finite."""
    _, variables, layout = small
    model = PolicyNet(dataclasses.replace(SMALL, dtype="bfloat16"), SMALL_SEMANTICS)
    rng = np.random.default_rng(2)
    obs = jax.tree_util.tree_map(jnp.asarray, random_obs(rng, (3,), layout, SMALL_SEMANTICS[0]))
    table = model.apply(variables, method=PolicyNet.semantic_table)
    enc = model.apply(variables, obs, table, method=PolicyNet.encode)
    events, _ = model.apply(variables, obs["turn_events_"], obs["turn_event_refs_"], enc.card_inputs, table,
                            method=lambda m, *a: m.event_encoder(*a))
    assert enc.state.dtype == jnp.bfloat16 and events.dtype == jnp.bfloat16
    assert enc.card_inputs.dtype == jnp.bfloat16 and enc.action_base.dtype == jnp.bfloat16
    logits, value, wdl, _, _ = act(model, variables, obs, jnp.zeros(3, jnp.int32), Memory.zeros(3, SMALL))
    assert np.isfinite(np.asarray(value)).all() and np.isfinite(np.asarray(wdl)).all()


def test_belief_targets_count_hidden_copies_per_candidate_and_location():
    from mirrorforce.agent.model.policy_net import BELIEF_LOCATIONS, belief_targets
    # candidates: card 300 (tier 1), card 7 (tier 3), padding
    candidates = np.zeros((1, 4, 3), np.uint8)
    candidates[0, 0] = (1, 44, 1)  # id 300
    candidates[0, 1] = (0, 7, 3)
    # labels: 2 copies of 300 in the deck, 1 in the hand, 5 of card 7 face-down (clipped to 3), 1 of card 9 (no
    # candidate names it)
    labels = np.zeros((1, 6, 4), np.uint8)
    labels[0, 0] = (1, 44, 1, 2)
    labels[0, 1] = (1, 44, 2, 1)
    labels[0, 2] = (0, 7, 4, 5)
    labels[0, 3] = (0, 9, 1, 1)
    targets, uncovered, total = belief_targets(jnp.asarray(candidates), jnp.asarray(labels))
    deck, hand, spell = BELIEF_LOCATIONS.index(1), BELIEF_LOCATIONS.index(2), BELIEF_LOCATIONS.index(4)
    t = np.asarray(targets)[0]
    assert t[0, deck] == 2 and t[0, hand] == 1 and t[0].sum() == 3
    assert t[1, spell] == 3 and t[1].sum() == 3
    assert t[2:].sum() == 0
    assert int(uncovered[0]) == 1 and int(total[0]) == 9


def test_the_belief_head_reads_the_state_through_stop_gradient(small):
    """The policy's gradient never reaches the belief head and the belief loss never reaches the trunk."""
    model, variables, layout = small
    rng = np.random.default_rng(8)
    obs, seat = episode(rng, 4, 3, layout, SMALL_SEMANTICS[0], first_turn=2)
    obs = jax.tree_util.tree_map(jnp.asarray, obs)
    consts = {k: v for k, v in variables.items() if k != "params"}

    def belief_loss(params):
        out = segment(model, {"params": params, **consts}, obs, jnp.asarray(seat), Memory.zeros(3, SMALL),
                      belief=True)
        return out[5][0].astype(jnp.float32).mean()

    grads = jax.grad(belief_loss)(variables["params"])
    nonzero = {k for k, g in grads.items() if any(float(jnp.abs(x).max()) > 0 for x in jax.tree_util.tree_leaves(g))}
    assert nonzero <= {"belief"}, nonzero
