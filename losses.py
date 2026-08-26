"""Chamfer distance (reconstruction) + class-weighted cross-entropy
(segmentation) for the PointNet++ candidate.
"""
from __future__ import annotations

import torch
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
