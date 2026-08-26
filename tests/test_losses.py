import torch

from losses import chamfer_distance, compute_class_weights, segmentation_loss


def test_chamfer_distance_zero_for_identical_clouds():
    cloud = torch.randn(2, 100, 3)
    d = chamfer_distance(cloud, cloud, target_subsample=None)
    # atol=1e-3, not tighter: torch.cdist has known floating-point noise on
    # near-identical inputs (catastrophic cancellation in its ||a||^2+||b||^2
    # -2a.b formulation) -- this is documented cdist behavior, not a bug in
    # chamfer_distance. Negligible against real range values of meters.
    assert torch.isclose(d, torch.tensor(0.0), atol=1e-3)


def test_chamfer_distance_positive_for_different_clouds():
    a = torch.randn(2, 100, 3)
    b = a + 10.0  # shifted far away
    d = chamfer_distance(a, b, target_subsample=None)
    assert d.item() > 1.0


def test_chamfer_distance_handles_mismatched_point_counts():
    # The real use case: 2,048 reconstructed points vs. a 32,768-point input
    # (dataset.py's N_POINTS). Pred and target are never the same size.
    pred = torch.randn(3, 2048, 3)
    target = torch.randn(3, 32768, 3)
    d = chamfer_distance(pred, target, target_subsample=4096)
    assert d.dim() == 0
    assert torch.isfinite(d)


def test_chamfer_distance_target_subsampling_is_close_to_full():
    # Subsampling the target is meant to be a cheap, unbiased-ish estimate,
    # not a wildly different number -- sanity check it's in the right
    # ballpark against the exact full computation, not just "doesn't crash".
    torch.manual_seed(0)
    pred = torch.randn(2, 200, 3)
    target = torch.randn(2, 5000, 3)
    d_full = chamfer_distance(pred, target, target_subsample=None)
    d_sub = chamfer_distance(pred, target, target_subsample=1000)
    assert abs(d_full.item() - d_sub.item()) < 0.5 * d_full.item()


def test_compute_class_weights_handles_zero_count_class():
    # human/car/bus (indices 1,2,3) have ZERO examples in the real scenario
    # -- this is not a hypothetical edge case, it's the actual data (see
    # README). A naive inverse-frequency weight would divide by zero.
    counts = {0: 1_000_000, 4: 100, 5: 50, 6: 80, 7: 150}  # 1,2,3 absent
    weights = compute_class_weights(counts, num_classes=8, cap=100.0)
    assert torch.isfinite(weights).all()
    assert weights[1] == 100.0
    assert weights[2] == 100.0
    assert weights[3] == 100.0


def test_compute_class_weights_caps_extreme_values():
    counts = {0: 1_000_000, 4: 1}  # class 4 is vanishingly rare
    weights = compute_class_weights(counts, num_classes=5, cap=50.0)
    assert weights[4] == 50.0
    assert weights.max() <= 50.0


def test_compute_class_weights_gives_low_weight_to_dominant_class():
    counts = {0: 1_000_000, 4: 100}
    weights = compute_class_weights(counts, num_classes=5, cap=1000.0)
    assert weights[0] < weights[4]


def test_segmentation_loss_runs_and_is_finite():
    logits = torch.randn(2, 100, 8, requires_grad=True)
    target = torch.randint(0, 8, (2, 100))
    weights = torch.ones(8)
    loss = segmentation_loss(logits, target, weights)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
