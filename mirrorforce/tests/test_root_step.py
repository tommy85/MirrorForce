"""The policy service's root step (``Service.root_step``): Ataraxos' mirror-descent step with the trainer's
card-grouped magnet, over every row or, when not every row was rolled out, over the sampled rows with their total
prior mass kept and the other rows unchanged. Runs where JAX is installed (the venv)."""
import numpy as np
import pytest

pytest.importorskip("jax")

from mirrorforce.agent.search.update import search_policy  # noqa: E402
from tools import mf_runtime_policy_service as S  # noqa: E402

ALPHA, BETA = 0.002, 0.02
LOGITS = np.array([1.0, 0.2, -0.5, 0.0, -2.0])
GROUPS = [3, 3, 7, 0, 7]  # rows 0-1 from card row 2, rows 2 and 4 from card row 6, row 3 without a source card


def step(q, magnet="magnet_card_grouped/v1", asked=None):
    service = S.Service(None, S.UniformBackend(magnet), {}, "sample", 1.0, {})
    request = {"logits": list(LOGITS), "groups": GROUPS, "q": q, "alpha": ALPHA, "beta": BETA}
    if asked is not None:
        request["magnet"] = asked
    return service.root_step(request)


def test_the_magnet_is_uniform_over_groups_then_within():
    log_magnet = S.magnet_log_probs(GROUPS)
    assert np.allclose(np.exp(log_magnet), [1 / 6, 1 / 6, 1 / 6, 1 / 3, 1 / 6])


def test_every_row_rolled_out_is_the_closed_form():
    q = [0.1, -0.2, 0.3, 0.0, -0.4]
    out = step(q)
    expected = search_policy(np.array(q), LOGITS, 1 / BETA, ALPHA, log_magnet=S.magnet_log_probs(GROUPS))
    assert np.allclose(out["policy"], expected)
    # the coordinator's form: p(a) ~ exp((q + alpha log rho + beta log pi) / (alpha + beta))
    log_pi = LOGITS - np.log(np.exp(LOGITS).sum())
    z = (np.array(q) + ALPHA * S.magnet_log_probs(GROUPS) + BETA * log_pi) / (ALPHA + BETA)
    assert np.allclose(out["policy"], np.exp(z - z.max()) / np.exp(z - z.max()).sum())


def test_rows_not_rolled_out_keep_their_prior():
    out = step([0.1, None, 0.3, None, -0.4])
    prior, policy = np.array(out["prior"]), np.array(out["policy"])
    assert np.isclose(policy.sum(), 1) and out["sampled"] == 3
    assert np.allclose(policy[[1, 3]], prior[[1, 3]])
    assert np.isclose(policy[[0, 2, 4]].sum(), prior[[0, 2, 4]].sum())
    with pytest.raises(ValueError):
        step([None] * 5)


def test_the_uniform_legal_magnet_and_a_mismatch_is_refused():
    q = [0.1, -0.2, 0.3, 0.0, -0.4]
    out = step(q, magnet="uniform_legal")
    assert out["magnet"] == "uniform_legal" and np.allclose(np.exp(out["log_magnet"]), 0.2)
    assert np.allclose(out["policy"], search_policy(np.array(q), LOGITS, 1 / BETA, ALPHA))
    with pytest.raises(ValueError, match="trained with"):
        step(q, magnet="uniform_legal", asked="magnet_card_grouped/v1")
