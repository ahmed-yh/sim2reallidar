"""Chamfer distance (reconstruction) + class-weighted cross-entropy
(segmentation) for the PointNet++ candidate, plus masked-grid equivalents
for SalsaNext (models/salsanext.py) -- same class-weighting scheme, but MSE
and cross-entropy over a (B,C,H,W) range-image grid instead of Chamfer
distance over a point set, and with a per-pixel validity mask (no-return
LiDAR cells carry no signal and must not contribute to either loss).
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor,
                      target_subsample: int | None = 4096) -> torch.Tensor:
    """pred: (B,M,3) reconstructed points, target: (B,N,3) input points.

    Symmetric nearest-neighbor distance: for every predicted point, distance
    to its closest target point, and vice versa, both averaged.

    target_subsample: the target cloud here is 32,768 points (see dataset.py)
    against a much smaller reconstruction (2,048 by default, see
    models/pointnet2.py); the full cdist is (B, M, 32768), which gets
    expensive in memory during backward on top of everything else training
    needs. Chamfer distance against a random subsample of the target is an
    unbiased estimate of the same quantity (just with more variance run to
    run), so subsampling the target side only -- never pred, never the
    model's actual input -- is a standard, safe way to bound memory. Set to
    None to use the full target cloud.
    """
    if target_subsample is not None and target.shape[1] > target_subsample:
        idx = torch.randint(0, target.shape[1], (target.shape[0], target_subsample),
                             device=target.device)
        target = torch.gather(target, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

    dists = torch.cdist(pred, target)  # (B, M, N)
    pred_to_target = dists.min(dim=2).values.mean(dim=1)  # (B,)
    target_to_pred = dists.min(dim=1).values.mean(dim=1)  # (B,)
    return (pred_to_target + target_to_pred).mean()


def compute_class_weights(class_counts: dict[int, int], num_classes: int,
                           cap: float = 100.0) -> torch.Tensor:
    """Inverse-frequency class weights from raw per-class point counts.

    Capped at `cap` in both directions this matters:
    - A class with very few examples (cylinder, the rarest class present in
      this scenario) would otherwise get an enormous weight that can
      destabilize training -- one rare-class pixel outweighing thousands of
      ordinary ones in a single batch.
    - A class with ZERO examples (human/car/bus -- confirmed absent from this
      recording, see README) would otherwise divide by zero and produce inf,
      which is silently catastrophic the moment anything touches the full
      weight vector (normalizing it, logging it, etc.), even though no
      individual training step ever selects that class as a target.
    """
    total = sum(class_counts.values())
    if total == 0:
        raise ValueError("class_counts sum to zero")

    weights = torch.empty(num_classes)
    for c in range(num_classes):
        count = class_counts.get(c, 0)
        if count == 0:
            weights[c] = cap
        else:
            weights[c] = min(total / (num_classes * count), cap)
    return weights


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor,
                       class_weights: torch.Tensor) -> torch.Tensor:
    """logits: (B,N,C), target: (B,N) int64 class indices."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        weight=class_weights.to(logits.device),
    )


def make_pointnet2_loss_fn(chamfer_target_subsample: int | None = 4096):
    """Builds the (model, batch, class_weights, lambda_class) -> (total,
    recon, class) callable train_utils.fit() expects, for the PointNet++
    candidate. batch = (xyz, cls). Same computation this repo always used
    for PointNet++ -- pulled out as its own factory only so train_utils'
    training loop could be shared with a second, differently-shaped
    candidate (SalsaNext) instead of duplicated."""
    def loss_fn(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor],
                class_weights: torch.Tensor, lambda_class: float):
        xyz, cls = batch
        out = model(xyz)
        recon_loss = chamfer_distance(out["recon_points"], xyz, target_subsample=chamfer_target_subsample)
        class_loss = segmentation_loss(out["class_logits"], cls, class_weights)
        total = recon_loss + lambda_class * class_loss
        return total, recon_loss, class_loss
    return loss_fn


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """pred, target: (B,H,W) or (B,1,H,W). valid_mask: (B,H,W) bool, True
    where the real recording actually returned a range (no-return LiDAR
    cells carry no ground truth and must not be scored)."""
    pred = pred.reshape(target.shape)
    diff2 = (pred - target) ** 2
    diff2 = diff2 * valid_mask
    return diff2.sum() / valid_mask.sum().clamp(min=1)


def masked_segmentation_loss(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor,
                              class_weights: torch.Tensor) -> torch.Tensor:
    """logits: (B,C,H,W), target: (B,H,W) int64, valid_mask: (B,H,W) bool.
    No-return cells are excluded from the mean the same way masked_mse_loss
    excludes them -- everywhere in this repo, class 0 ("environment") is a
    real observed label (a beam that hit open space), not a stand-in for a
    missing reading, so it must stay in the loss like any other class."""
    per_pixel = F.cross_entropy(
        logits, target, weight=class_weights.to(logits.device), reduction="none",
    )
    per_pixel = per_pixel * valid_mask
    return per_pixel.sum() / valid_mask.sum().clamp(min=1)


def make_salsanext_loss_fn():
    """Builds the (model, batch, class_weights, lambda_class) -> (total,
    recon, class) callable train_utils.fit() expects, for the SalsaNext
    candidate. batch = (x, range_target, cls_target, valid_mask). Unlike
    PointNet++, both heads score directly against the same (64,1024) grid
    the input came from -- no reprojection needed, since neither output
    changes the grid's own indexing (see models/salsanext.py)."""
    def loss_fn(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                class_weights: torch.Tensor, lambda_class: float):
        x, range_target, cls_target, valid_mask = batch
        out = model(x)
        recon_loss = masked_mse_loss(out["recon_range"], range_target, valid_mask)
        class_loss = masked_segmentation_loss(out["class_logits"], cls_target, valid_mask, class_weights)
        total = recon_loss + lambda_class * class_loss
        return total, recon_loss, class_loss
    return loss_fn
