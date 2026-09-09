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


def _farthest_point_sample_compute(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    """The actual algorithm, unchanged. Split out from farthest_point_sample
    so the CUDA-graph wrapper below has a plain function to capture -- the
    graph records exactly this sequence of ops, nothing about the math
    differs from the original single-function version."""
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


# (device, B, N, n_samples) -> (static_input_buffer, static_output_buffer, captured_graph).
# Training calls this with the same handful of shapes thousands of times
# (one per SA level x whatever batch sizes the DataLoader produces -- usually
# one, occasionally a smaller final batch); Jetson inference calls it with
# exactly one shape forever (batch_size=1, see benchmarks/pointnet2_jetson_bench.py).
# Both are exactly the access pattern CUDA graphs are for: capture once,
# replay every subsequent call at that shape.
_fps_graph_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, "torch.cuda.CUDAGraph"]] = {}
_fps_graph_disabled = False  # set True permanently on first capture failure -- see farthest_point_sample


def _farthest_point_sample_graph(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    key = (xyz.device, xyz.shape[0], xyz.shape[1], n_samples)
    cached = _fps_graph_cache.get(key)
    if cached is None:
        static_in = torch.empty_like(xyz)
        static_in.copy_(xyz)
        # Required before capture: run a few iterations on a side stream so
        # the capture doesn't record one-time allocator/cuBLAS-handle setup
        # as part of the replayed graph (standard torch.cuda.graph caveat).
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                _farthest_point_sample_compute(static_in, n_samples)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = _farthest_point_sample_compute(static_in, n_samples)
        cached = (static_in, static_out, graph)
        _fps_graph_cache[key] = cached

    static_in, static_out, graph = cached
    static_in.copy_(xyz)
    graph.replay()
    return static_out.clone()  # clone: caller must not alias the reused static buffer


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

    On CUDA, this is CUDA-graph-accelerated (~8.7x faster on the 32768->4096
    SA1 call, measured on real recordings, verified bit-identical to the
    unoptimized version first -- see scratch_benchmark_fps.py): the naive
    Python `for i in range(n_samples)` loop above is ~5,376 sequential,
    data-dependent iterations per encoder forward pass (4096+1024+256 across
    the three SA levels), each paying Python-interpreter + CUDA-kernel-launch
    overhead for a few microseconds of actual math -- confirmed the dominant
    per-step cost during real training (profiled: SA1 alone was 77% of the
    encoder's forward time, 93% of which was this loop, not the neighborhood
    grouping or the conv layers). A captured graph replays that same fixed
    sequence of ops with none of the per-iteration Python/launch overhead;
    the algorithm and its output are unchanged, only how it's executed.
    On CPU (or if CUDA graph capture fails for any reason -- e.g. a CUDA
    version too old to support it) this falls straight back to the plain
    loop, so nothing about correctness depends on graph capture succeeding.
    """
    global _fps_graph_disabled
    if not xyz.is_cuda or _fps_graph_disabled:
        return _farthest_point_sample_compute(xyz, n_samples)
    try:
        return _farthest_point_sample_graph(xyz, n_samples)
    except RuntimeError:
        # Capture failed (e.g. a CUDA/driver version too old to support
        # graphs). Disable permanently rather than retry every call: a
        # failed capture attempt is exactly the kind of thing that can leave
        # CUDA's capture-mode stream state inconsistent for whatever runs
        # next, so don't keep re-entering it call after call.
        _fps_graph_disabled = True
        return _farthest_point_sample_compute(xyz, n_samples)


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
            layers += [nn.Conv2d(last, ch, 1), nn.GroupNorm(8, ch), nn.ReLU(inplace=True)]
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
            layers += [nn.Conv1d(last, ch, 1), nn.GroupNorm(8, ch), nn.ReLU(inplace=True)]
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
