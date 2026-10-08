"""Public-only joint energies, conservation and frozen artifact binding."""

from dataclasses import replace
import io

import pytest
import torch

from mirrorforce.search.particles import HiddenLayout
from mirrorforce.common import stage_a_joint_belief as B
from mirrorforce.common.sidecar_io import digest
from mirrorforce.common.world_model_contract import WorldLatent


def inputs(batch=2):
    torch.manual_seed(91)
    head = B.OpponentJointBelief(B.JointBeliefConfig(d_model=16, card_dim=8, heads=2, hidden_dim=24))
    latent = WorldLatent(torch.randn(batch, 4, 16), torch.tensor([[False, False, True, True]]*batch),
                         torch.tensor([i % 2 for i in range(batch)]))
    slots = B.PublicFieldSlots(torch.tensor([[8, 0]]*batch), torch.tensor([[2, 0]]*batch),
                               torch.tensor([[1, 0]]*batch), torch.tensor([[True, False]]*batch))
    args = (latent, torch.randn(3, 8), torch.tensor([2, 1, 1]), torch.tensor([False, False, True]),
            torch.tensor([[2, 1, 1, 1]]*batch), slots)
    return head, args


def test_zero_readouts_preserve_joint_prior_and_exclude_impossible_counts():
    head, args = inputs()
    out = head(*args)
    assert out.count_adjustments.shape == (2, 3, 4, 4)
    assert out.slot_adjustments.shape == (2, 2, 3)
    assert torch.equal(out.slot_adjustments, torch.zeros_like(out.slot_adjustments))
    for card, n in enumerate(args[2]):
        assert torch.equal(out.count_adjustments[:, card, :, :n+1], torch.zeros_like(out.count_adjustments[:, card, :, :n+1]))
        assert bool((out.count_adjustments[:, card, :, n+1:] < -100).all())


def test_public_padding_is_inert_after_both_outputs_have_learned():
    head, args = inputs()
    torch.nn.init.normal_(head.count[-1].weight)
    torch.nn.init.normal_(head.slot_query[-1].weight)
    first = head(*args)
    tokens = args[0].tokens.clone()
    tokens[:, 2:] += 300
    second = head(replace(args[0], tokens=tokens), *args[1:])
    torch.testing.assert_close(first.count_adjustments, second.count_adjustments)
    torch.testing.assert_close(first.slot_adjustments, second.slot_adjustments)
    assert bool((first.slot_adjustments[:, 1] == 0).all())


def test_deck_and_hand_and_field_readouts_all_receive_gradients_without_policy_gradients():
    head, args = inputs()
    tokens = args[0].tokens.requires_grad_()
    latent = replace(args[0], tokens=tokens).detached()
    out = head(latent, *args[1:])
    # Actual cross-entropy, with different target counts for the four zones.
    target = torch.tensor([[[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]*2)
    loss = torch.nn.functional.cross_entropy(out.count_adjustments.reshape(-1, 4), target.flatten())
    loss = loss + torch.nn.functional.cross_entropy(out.slot_adjustments[:, 0], torch.tensor([1, 1]))
    loss.backward()
    assert tokens.grad is None
    grads = head.count[-1].weight.grad.reshape(4, 4, -1)
    assert all(float(grads[zone].abs().sum()) > 0 for zone in range(4))
    assert float(head.slot_query[-1].weight.grad.abs().sum()) > 0


def test_zero_field_slots_supported_and_duplicate_or_hidden_identity_columns_refused():
    head, args = inputs()
    empty = B.PublicFieldSlots(*(torch.zeros(2, 0, dtype=torch.long) for _ in range(3)), torch.zeros(2, 0, dtype=torch.bool))
    sizes = args[4].clone()
    sizes[:, 2] = 0
    assert head(*args[:4], sizes, empty).slot_adjustments.shape == (2, 0, 3)
    with pytest.raises(ValueError, match="field slot"):
        head(*args[:4], sizes, args[5])
    duplicate = B.PublicFieldSlots(torch.tensor([[8, 8]]*2), torch.tensor([[2, 2]]*2),
                                   torch.tensor([[1, 1]]*2), torch.ones(2, 2, dtype=torch.bool))
    sizes[:, 2] = 2
    with pytest.raises(ValueError, match="duplicate"):
        head(*args[:4], sizes, duplicate)
    with pytest.raises(TypeError):
        B.PublicFieldSlots(**{**vars(args[5]), "true_codes": torch.ones(2, 2)})


def test_hard_zero_prior_remains_zero_even_under_overwhelming_logits():
    adjustments = torch.tensor([[0., 1e6, 0., 0.]], requires_grad=True)
    prior = torch.tensor([[.7, 0., .3, 0.]])
    logp = B.supported_log_probabilities(adjustments, prior)
    torch.testing.assert_close(logp.exp(), prior)
    (-logp[0, 0]).backward()
    assert adjustments.grad[0, 1] == 0
    with pytest.raises(ValueError):
        B.supported_log_probabilities(adjustments, torch.zeros_like(prior))


def proposals():
    recipe = B.JointRecipe.from_decks([10, 10, 20], [30])
    first = HiddenLayout(hand=(10,), deck=(10,), facedown=((8, 2, 20),), extra=(30,))
    second = HiddenLayout(hand=(20,), deck=(10,), facedown=((8, 2, 10),), extra=(30,))
    output = B.JointBeliefOutput(torch.zeros(1, 3, 4, 4), torch.zeros(1, 1, 3))
    return recipe, [first, second], output


def test_joint_scoring_uses_remaining_deck_and_slot_identity_not_just_hand():
    recipe, layouts, out = proposals()
    assert B.layout_counts(layouts[0], recipe) == [[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert B.layout_log_weights(out, layouts, recipe, ((8, 2),)).tolist() == [0., 0.]
    out.slot_adjustments[0, 0, 1] = 2
    assert B.layout_log_weights(out, layouts, recipe, ((8, 2),)).tolist() == [2., 0.]
    out.count_adjustments[0, 0, 0, 1] = 3
    assert B.layout_log_weights(out, layouts, recipe, ((8, 2),)).tolist() == [5., 0.]
    assert B.layout_log_weights(out, layouts, recipe, ((8, 2),), power=0.).tolist() == [0., 0.]
    moved = HiddenLayout(hand=(10,), deck=(20,), facedown=((8, 2, 10),), extra=(30,))
    out.count_adjustments[0, 1, 1, 1] = 7
    assert B.layout_log_weights(out, [moved], recipe, ((8, 2),)).tolist() == [10.]


@pytest.mark.parametrize("change", [
    {"hand": (10, 20)},  # duplicates the copy in the field
    {"deck": (99,)},
    {"facedown": ((8, 2, 20), (8, 2, 20))},
    {"facedown": ((8, 3, 20),)},
    {"extra": None},
])
def test_invalid_joint_proposals_are_not_silently_scored(change):
    recipe, layouts, out = proposals()
    with pytest.raises(ValueError):
        B.layout_log_weights(out, [replace(layouts[0], **change)], recipe, ((8, 2),))


def test_banished_copy_consumes_inventory_without_becoming_a_field_slot():
    recipe, layouts, out = proposals()
    layout = replace(layouts[0], deck=(), facedown=((8, 2, 20), (32, 0, 10)))
    assert B.layout_counts(layout, recipe)[0] == [1, 0, 0, 0]
    assert B.layout_log_weights(out, [layout], recipe, ((8, 2),)).tolist() == [0.]
    with pytest.raises(ValueError):
        B.layout_counts(replace(layout, hand=(10, 10)), recipe)


def test_joint_sidecar_roundtrip_refuses_different_backbone_or_semantics(tmp_path):
    head, args = inputs()
    recipe, _, _ = proposals()
    fitted = {"checkpoint_sha256": "a"*64, "training_provenance_sha256": "b"*64, "artifact_fingerprint": "c"*64}
    payload = B.head_payload(head, recipe, fitted)
    stream = io.BytesIO()
    torch.save(payload, stream)
    raw = stream.getvalue()
    path = tmp_path / "head.pt"
    path.write_bytes(raw)
    loaded, read_recipe, read_fit = B.load_head(path, digest(raw), checkpoint_sha256="a"*64, artifact_fingerprint="c"*64)
    assert recipe == read_recipe and fitted == read_fit
    torch.testing.assert_close(head(*args).count_adjustments, loaded(*args).count_adjustments)
    for checkpoint, artifact in (("d"*64, "c"*64), ("a"*64, "d"*64)):
        with pytest.raises(ValueError, match="differs"):
            B.load_head(path, digest(raw), checkpoint_sha256=checkpoint, artifact_fingerprint=artifact)
    with pytest.raises(ValueError, match="checksum"):
        B.load_head(path, "0"*64, checkpoint_sha256="a"*64, artifact_fingerprint="c"*64)
