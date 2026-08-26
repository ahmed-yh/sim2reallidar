"""FPS / ball-query / feature-propagation building blocks.

Pure PyTorch, no custom CUDA ops (deliberate — see benchmarks/pointnet2_jetson_bench.py
and project notes: compiling custom extensions against Jetson's aarch64 + JetPack-pinned
CUDA/PyTorch build is a real, avoidable source of pain). These are the same ops already
timed on the Orin; this file is the cleaned-up, importable version of that benchmark code.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """points: (B,N,C), idx: (B,S) or (B,S,K) -> (B,S,C) or (B,S,K,C)."""
    B = points.shape[0]
    batch_idx = torch.arange(B, device=points.device).view(B, *([1] * (idx.dim() - 1)))
    return points[batch_idx, idx]


def farthest_point_sample(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    """xyz: (B, N, 3) -> centroid indices (B, n_samples).

    Fixed start point (index 0), not random. A random start would make the
    ENCODER non-deterministic across calls on identical input -- caught by
    tests/test_pointnet2_model.py::test_encode_matches_forward_latent, which
    failed before this fix even though the model was in eval() mode (eval()
    only affects Dropout/BatchNorm, not an explicit torch.randint call
    elsewhere in the forward pass). For a deployed perception system this
    matters concretely: the robot seeing a different latent vector for the
    exact same static scan between two consecutive frames would be a real
    debugging nightmare, not just a test inconvenience.
    """
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, n_samples, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.zeros(B, dtype=torch.long, device=xyz.device)
    batch_idx = torch.arange(B, device=xyz.device)
    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_idx, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1).indices
    return centroids


def ball_query(radius: float, k: int, xyz: torch.Tensor, centroid_xyz: torch.Tensor) -> torch.Tensor:
    """xyz: (B,N,3) all points, centroid_xyz: (B,S,3) -> group idx (B,S,k).

    Two edge cases, both caught by tests/test_pointnet2_utils.py and both
    real -- not hypothetical:

    1. A centroid can have ZERO points within `radius` (confirmed: happens
       on ordinary random test data, not just contrived inputs). The
       previous version's fallback ("if the first sorted slot is itself the
       out-of-range sentinel, fall back to... the sentinel") let the
       sentinel value N leak straight into the returned indices -- an
       out-of-bounds index that would crash or silently corrupt the next
       indexing operation downstream. Fixed here with a fallback to each
       centroid's true nearest neighbor (radius-independent, always a valid
       index), which is also what real PointNet++ implementations do.
    2. N < k (fewer total points than requested neighbors) previously
       crashed outright: slicing sorted_idx[:, :, :k] with k > N silently
       returns only N columns, not k, breaking every downstream shape
       assumption. Doesn't happen on this project's real data (N=32,768,
       k=32) but is worth being correct about rather than merely lucky
       about, so it's handled explicitly by padding with the nearest-
       neighbor fallback rather than left to crash.
    """
    B, N, _ = xyz.shape
    S = centroid_xyz.shape[1]
    sqrdists = torch.cdist(centroid_xyz, xyz) ** 2  # (B,S,N)

    nearest_idx = sqrdists.argmin(dim=-1)  # (B,S), always valid regardless of radius

    in_range = sqrdists <= radius ** 2
    idx_grid = torch.arange(N, device=xyz.device).view(1, 1, N).expand(B, S, N)
    sentinel = N
    candidate_idx = torch.where(in_range, idx_grid, torch.full_like(idx_grid, sentinel))
    sorted_idx, _ = candidate_idx.sort(dim=-1)  # in-range indices first, sentinels last

    k_eff = min(k, N)
    group_idx = sorted_idx[:, :, :k_eff]
    if k_eff < k:
        pad = nearest_idx.unsqueeze(-1).expand(-1, -1, k - k_eff)
        group_idx = torch.cat([group_idx, pad], dim=-1)

    fallback = nearest_idx.unsqueeze(-1).expand(-1, -1, k)
    mask = group_idx == sentinel
    return torch.where(mask, fallback, group_idx)


class SetAbstraction(nn.Module):
    """One level of the PointNet++ encoder: sample centroids, group local
    neighborhoods, run a mini-PointNet (shared MLP + max-pool) per group."""

    def __init__(self, n_samples: int, radius: float, k: int, in_ch: int, mlp_channels: list[int]):
        super().__init__()
        self.n_samples, self.radius, self.k = n_samples, radius, k
        layers = []
        last = in_ch + 3
        for ch in mlp_channels:
            layers += [nn.Conv2d(last, ch, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
            last = ch
        self.mlp = nn.Sequential(*layers)
        self.out_ch = mlp_channels[-1]

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        B = xyz.shape[0]
        centroid_idx = farthest_point_sample(xyz, self.n_samples)
        centroid_xyz = index_points(xyz, centroid_idx)
        group_idx = ball_query(self.radius, self.k, xyz, centroid_xyz)
        grouped_xyz = index_points(xyz, group_idx) - centroid_xyz.unsqueeze(2)
        if points is not None:
            grouped_points = index_points(points, group_idx)
            grouped = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped = grouped_xyz
        grouped = grouped.permute(0, 3, 2, 1)  # (B, C, k, S)
        feat = self.mlp(grouped)
        new_points = torch.max(feat, dim=2).values.permute(0, 2, 1)  # (B, S, C')
        return centroid_xyz, new_points


class FeaturePropagation(nn.Module):
    """Interpolate features from a sparser point set back onto a denser one
    (inverse-distance-weighted 3-NN), concatenate with that level's saved
    encoder features (the skip connection), then a shared MLP. Standard
    PointNet++ segmentation decoder block."""

    def __init__(self, in_ch: int, mlp_channels: list[int]):
        super().__init__()
        layers = []
        last = in_ch
        for ch in mlp_channels:
            layers += [nn.Conv1d(last, ch, 1), nn.BatchNorm1d(ch), nn.ReLU(inplace=True)]
            last = ch
        self.mlp = nn.Sequential(*layers)
        self.out_ch = mlp_channels[-1]

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                points1: torch.Tensor | None, points2: torch.Tensor) -> torch.Tensor:
        """xyz1/points1: denser target level (points1 may be None). xyz2/points2:
        sparser source level whose features get propagated onto xyz1."""
        B, N, _ = xyz1.shape
        S = xyz2.shape[1]
        if S == 1:
            interpolated = points2.repeat(1, N, 1)
        else:
            dists = torch.cdist(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            weight = dist_recip / torch.sum(dist_recip, dim=-1, keepdim=True)
            interpolated = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2)
        new_points = torch.cat([points1, interpolated], dim=-1) if points1 is not None else interpolated
        new_points = new_points.permute(0, 2, 1)
        new_points = self.mlp(new_points)
        return new_points.permute(0, 2, 1)
