"""T10: additive legal-menu value path, bitwise zero-start and strict state migration (CPU)."""
import copy
import dataclasses

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mirrorforce.agent.model.critic import CriticNet
from mirrorforce.agent.model.utils import TrainState
from mirrorforce.agent.model.policy_net import Readout, ValueMenuAttention, PolicyNet, act
from mirrorforce.agent.train import state_io, value_menu_migration as M
from test_critic import both_seats
from test_policy_net import SMALL, SMALL_SEMANTICS, carried, initial_prefix, make, random_obs, shapes


def exact(a, b):
    aa, bb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(aa) == len(bb)
    for x, y in zip(aa, bb):
        x, y = np.asarray(x), np.asarray(y)
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


@pytest.mark.parametrize("dtype", [None, "bfloat16"])
def test_public_readout_zero_start_bitwise_policy_value_wdl_and_learnable(dtype):
    c = dataclasses.replace(SMALL, dtype=dtype)
    rng = np.random.default_rng(13)
    obs = random_obs(rng, (2,), shapes(cards=4, options=3), SMALL_SEMANTICS[0])
    dt = c.compute_dtype
    args = (jnp.asarray(rng.normal(size=(2, 5, c.d)), dt), jnp.ones((2, 5), bool),
            jnp.asarray(rng.normal(size=(2, c.d)), dt), 4,
            jnp.asarray(rng.normal(size=(2, 3, c.d)), dt), jnp.array([[True, True, False], [False]*3]), obs)
    old, new = Readout(c), Readout(dataclasses.replace(c, value_menu=True))
    p = old.init(jax.random.PRNGKey(2), *args)
    q = new.init(jax.random.PRNGKey(2), *args)
    stripped = copy.deepcopy(q)
    del stripped["params"]["value_menu"]
    exact(p, stripped)  # factoring action tokens has not renamed/reinitialized any old parameter
    run_old, run_new = jax.jit(old.apply, static_argnums=4), jax.jit(new.apply, static_argnums=4)
    exact(run_old(p, *args), run_new(q, *args))
    gradient = jax.grad(lambda params: new.apply({"params": params}, *args)[2][0, 0])(q["params"])
    assert np.abs(np.asarray(gradient["value_menu"]["out"]["kernel"])).max() > 0
    trained = copy.deepcopy(q)
    trained["params"]["value_menu"]["out"]["kernel"] -= .1 * gradient["value_menu"]["out"]["kernel"]
    result = run_new(trained, *args)
    exact(result[0], run_old(p, *args)[0])  # this path never directly changes policy logits
    assert not np.array_equal(np.asarray(result[2][0]), np.asarray(run_old(p, *args)[2][0]))
    exact(result[2][1], run_old(p, *args)[2][1])  # the empty menu stays zero even after learning


def test_menu_mask_padding_empty_and_learning_sensitivity():
    m = ValueMenuAttention(8, 2)
    query = jnp.arange(16, dtype=jnp.float32).reshape(2, 1, 8) / 9
    action = jnp.arange(48, dtype=jnp.float32).reshape(2, 3, 8) / 11
    mask = jnp.array([[True, False, False], [False, False, False]])
    variables = m.init(jax.random.PRNGKey(4), query, action, mask)
    variables["params"]["out"]["kernel"] = jnp.ones((2, 4, 8)) / 8
    run = jax.jit(m.apply)
    a = run(variables, query, action, mask)
    b = run(variables, query, jnp.where(mask[..., None], action, jnp.nan), mask)
    exact(a, b)
    assert np.isfinite(a).all() and not np.asarray(a[1]).any()
    c = run(variables, query, action.at[0, 0].add(2), mask)
    assert not np.array_equal(np.asarray(a[0]), np.asarray(c[0]))


@pytest.mark.parametrize("dtype", [None, "bfloat16"])
def test_central_critic_zero_start_and_legal_menu_gradient(dtype):
    c = dataclasses.replace(SMALL, dtype=dtype, state_layers=1)
    rng = np.random.default_rng(4)
    obs = jax.tree.map(jnp.asarray, both_seats(rng, 2, shapes(cards=4, options=3)))
    old, new = CriticNet(c, SMALL_SEMANTICS, 1), CriticNet(dataclasses.replace(c, value_menu=True), SMALL_SEMANTICS, 1)
    p, q = (m.init(jax.random.PRNGKey(1), obs) for m in (old, new))
    stripped = copy.deepcopy(q)
    for key in ("menu_tokens", "value_menu"):
        stripped["params"].pop(key)
    exact(p, stripped)
    exact(jax.jit(old.apply)(p, obs), jax.jit(new.apply)(q, obs))
    if dtype == "bfloat16" and jax.default_backend() == "cpu":
        # JAX 0.5.3 CPU cannot lower the singleton-query BF16 attention's full reverse dot_general.
        # Test the actually trainable first-step projection here, not a precision-changed production model.
        # GPU executes the full backward branch below; FP32 CPU also checks the full parameter gradient.
        def loss(kernel):
            params = copy.deepcopy(q["params"])
            params["value_menu"]["out"]["kernel"] = kernel
            return new.apply(dict(q, params=params), obs)[0, 0]
        projection_grad = jax.jit(jax.grad(loss))(q["params"]["value_menu"]["out"]["kernel"])
    else:
        grad = jax.jit(jax.grad(lambda params: new.apply(dict(q, params=params), obs)[0, 0]))(q["params"])
        projection_grad = grad["value_menu"]["out"]["kernel"]
    assert np.abs(np.asarray(projection_grad)).max() > 0


@pytest.mark.parametrize("dtype", [None, "bfloat16"])
def test_full_public_actor_preserves_logits_wdl_value_and_recurrent_carry(dtype):
    c = dataclasses.replace(SMALL, dtype=dtype, d=16, heads=2, state_layers=1, turn_layers=1)
    rng = np.random.default_rng(8)
    layout = shapes(cards=4, options=3)
    old, p = make(c, SMALL_SEMANTICS, layout, 1, rng)
    obs = jax.tree.map(jnp.asarray, random_obs(rng, (1,), layout, SMALL_SEMANTICS[0]))
    new = PolicyNet(dataclasses.replace(c, value_menu=True), SMALL_SEMANTICS)
    q = new.init(jax.random.PRNGKey(0), obs, initial_prefix(1, c), method=PolicyNet.init_all)
    q["constants"] = p["constants"]
    # Restore old weights rather than rely on the fresh-init random-number sequence.
    q["params"] = M.extend_tree(p["params"], q["params"], M.ACTOR_ROOTS)
    memory = carried(rng, 1, c)
    run = jax.jit(act, static_argnums=0)
    exact(run(old, p, obs, jnp.zeros(1, jnp.int32), memory),
          run(new, q, obs, jnp.zeros(1, jnp.int32), memory))


def test_legacy_loader_and_new_training_defaults():
    from mirrorforce.agent.train.cleanba import Args, model_config_identity
    from mirrorforce.agent.train.policy_io import _config
    assert not _config({"d": 16}).value_menu
    assert "value_menu" not in model_config_identity(SMALL)
    assert Args().net.value_menu and Args().critic_model.value_menu


def train_state(params, *, multi=False):
    adam = lambda: optax.chain(optax.clip_by_global_norm(.267),
                               optax.inject_hyperparams(optax.adam)(learning_rate=1e-3))
    tx = (optax.multi_transform({"policy": adam(), "belief": adam()},
          lambda ps: {k: "belief" if k == "belief" else "policy" for k in ps}) if multi else adam())
    return TrainState.create(apply_fn=None, params=params, tx=tx, batch_stats={}, constants={})


def states(critic=False):
    roots = M.CRITIC_ROOTS if critic else M.ACTOR_ROOTS
    old = {"value_out": {"kernel": jnp.ones((2, 3))}}
    if not critic:
        old = {"readout": old, "belief": {"kernel": jnp.ones((2, 3))}}
    params = copy.deepcopy(old)
    for root in roots:
        owner = params
        for key in root[:-1]:
            owner = owner[key]
        owner[root[-1]] = {"out": {"kernel": jnp.zeros((2, 3))}, "key": {"kernel": jnp.ones((2, 3))}}
    source = train_state(old, multi=not critic)
    for _ in range(3):
        source = source.apply_gradients(grads=jax.tree.map(jnp.ones_like, old))
    return source, train_state(params, multi=not critic)


@pytest.mark.parametrize("critic", [False, True])
@pytest.mark.parametrize("jit_saved", [False, True])
def test_migration_preserves_every_old_adam_leaf_count_and_ema(critic, jit_saved):
    old, fresh = states(critic)
    if jit_saved:
        old = jax.jit(lambda state: state)(old)  # real training saves a typed int32 step, not Python int
        assert np.asarray(old.step).dtype == np.dtype("int32")
    raw = flax.serialization.to_state_dict(old)
    expanded = M.extend_state(raw, fresh, critic=critic)
    def check(saved, restored):
        if isinstance(saved, dict):
            for key in saved:
                check(saved[key], restored[key])
        else:
            exact(saved, restored)
    check(raw, expanded)
    restored = flax.serialization.from_state_dict(fresh, expanded)
    assert int(restored.step) == 3
    restored.apply_gradients(grads=jax.tree.map(jnp.ones_like, restored.params))  # real next Adam step works
    if not critic:
        ema = jax.tree.map(lambda x: x * .75, old.params)
        expanded_ema = M.extend_ema(flax.serialization.to_state_dict(ema), fresh.params)
        check(flax.serialization.to_state_dict(ema), expanded_ema)
        exact(expanded_ema["readout"]["value_menu"], fresh.params["readout"]["value_menu"])


def identities():
    old = {"model": {"architecture": "policy_net", "args": {"d": 8}, "semantic_shape": [2, 3, 4, 5]},
           "critic": {"model": {"d": 4}, "mix_layers": 2}, "recipe": {"batch_size": 8192}}
    new = copy.deepcopy(old)
    new["model"]["args"]["value_menu"] = new["critic"]["model"]["value_menu"] = True
    return old, new


def test_checkpoint_migration_is_explicit_and_keeps_keys_counters_critic(tmp_path):
    old, new = identities()
    saved, fresh = states()
    saved_critic, fresh_critic = states(True)
    keys = np.array([[41, 73]], np.uint32)
    sha = state_io.save(tmp_path, saved, keys, {"global_step": 1234, "learner_update": 3}, old,
                        ema=saved.params, critic=saved_critic)
    path = tmp_path / f"{sha}.ckpt"
    with pytest.raises(ValueError, match="model"):
        state_io.restore(path, fresh, 1, new)
    st, gotkeys, counts, _, _, ema = state_io.restore(path, fresh, 1, new, add_value_menu=True)
    exact(gotkeys, keys)
    assert counts == {"global_step": 1234, "learner_update": 3} and int(st.step) == 3
    assert int(state_io.restore_extra(path, "critic", fresh_critic, add_value_menu=True).step) == 3
    exact(ema["belief"], saved.params["belief"])


@pytest.mark.parametrize("fault", ["missing_critic", "partial_ema"])
def test_checkpoint_rejects_inconsistent_extra_state_before_returning_policy(tmp_path, fault):
    from mirrorforce.agent.train.checkpoint_store import write_checkpoint
    old, new = identities()
    saved, fresh = states()
    saved_critic, _ = states(True)
    raw = {"state": flax.serialization.to_state_dict(saved), "learner_keys": np.array([[1, 2]], np.uint32),
           "counters": {"global_step": 1234, "learner_update": 3},
           "ema": flax.serialization.to_state_dict(saved.params),
           "critic": flax.serialization.to_state_dict(saved_critic)}
    if fault == "missing_critic": raw.pop("critic")
    if fault == "partial_ema": raw["ema"]["readout"]["value_menu"] = {}
    sha = write_checkpoint(tmp_path, flax.serialization.msgpack_serialize(raw),
                           {**old, "schema": state_io.SCHEMA, "counters": raw["counters"]})
    with pytest.raises(ValueError):
        state_io.restore(tmp_path / f"{sha}.ckpt", fresh, 1, new, add_value_menu=True)


@pytest.mark.parametrize("fault", ["already", "architecture", "width", "critic_width", "no_critic"])
def test_migration_rejects_other_identity_changes(fault):
    old, new = identities()
    if fault == "already": old["model"]["args"]["value_menu"] = True
    if fault == "architecture": new["model"]["architecture"] = "other"
    if fault == "width": new["model"]["args"]["d"] = 16
    if fault == "critic_width": new["critic"]["model"]["d"] = 16
    if fault == "no_critic": new.pop("critic")
    with pytest.raises(ValueError): M.check_identity(old, new)


@pytest.mark.parametrize("fault", ["missing_old", "extra_old", "shape", "dtype", "nonzero", "partial_new"])
def test_migration_rejects_broken_trees(fault):
    old, fresh = states()
    raw = flax.serialization.to_state_dict(old)
    if fault == "missing_old": raw["params"]["readout"].pop("value_out")
    if fault == "extra_old": raw["params"]["unknown"] = {}
    if fault == "shape": raw["params"]["readout"]["value_out"]["kernel"] = np.ones((1, 3), np.float32)
    if fault == "dtype": raw["params"]["readout"]["value_out"]["kernel"] = np.ones((2, 3), np.float64)
    if fault == "nonzero": fresh.params["readout"]["value_menu"]["out"]["kernel"] = jnp.ones((2, 3))
    if fault == "partial_new": raw["params"]["readout"]["value_menu"] = {}
    with pytest.raises(ValueError): M.extend_state(raw, fresh)


@pytest.mark.parametrize("step", [np.array([3], np.int32), np.float32(3), np.int32(-1), np.uint32(3), True])
def test_step_placeholder_compatibility_does_not_accept_wrong_counter_schema(step):
    old, fresh = states()
    raw = flax.serialization.to_state_dict(old)
    raw["step"] = step
    with pytest.raises(ValueError): M.extend_state(raw, fresh)
