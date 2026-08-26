import torch

from models.pointnet2_utils import (
    FeaturePropagation,
    SetAbstraction,
    ball_query,
    farthest_point_sample,
    index_points,
)


def test_index_points_2d_idx():
    points = torch.arange(2 * 5 * 3).reshape(2, 5, 3).float()
    idx = torch.tensor([[0, 2], [4, 1]])  # (B=2, S=2)
    out = index_points(points, idx)
    assert out.shape == (2, 2, 3)
    assert torch.equal(out[0, 0], points[0, 0])
    assert torch.equal(out[1, 1], points[1, 1])


def test_index_points_3d_idx():
    points = torch.arange(2 * 5 * 3).reshape(2, 5, 3).float()
    idx = torch.zeros(2, 4, 6, dtype=torch.long)  # (B,S,K) all pointing at point 0
    out = index_points(points, idx)
    assert out.shape == (2, 4, 6, 3)
    assert torch.equal(out[0, 0, 0], points[0, 0])
    assert torch.equal(out[1, 3, 5], points[1, 0])


def test_farthest_point_sample_shape_and_range():
    xyz = torch.randn(3, 100, 3)
    idx = farthest_point_sample(xyz, n_samples=20)
    assert idx.shape == (3, 20)
    assert idx.min() >= 0 and idx.max() < 100


def test_farthest_point_sample_picks_distinct_points_when_well_separated():
    # 10 points far apart from each other -- FPS should never need to repeat one.
    xyz = (torch.arange(10).float().view(1, 10, 1) * 100.0).expand(1, 10, 3).contiguous()
    idx = farthest_point_sample(xyz, n_samples=10)
    assert len(torch.unique(idx[0])) == 10


def test_ball_query_shape():
    xyz = torch.randn(2, 50, 3)
    centroids = torch.randn(2, 8, 3)
    idx = ball_query(radius=1.0, k=16, xyz=xyz, centroid_xyz=centroids)
    assert idx.shape == (2, 8, 16)
    assert idx.min() >= 0 and idx.max() < 50


def test_ball_query_handles_fewer_than_k_neighbors_without_crashing():
    # A single, isolated point far from every centroid: each centroid's
    # in-radius neighbor count is 0 or 1, well below k=16. This is the edge
    # case noted while building the Jetson benchmark script -- must not
    # crash or return an out-of-range sentinel index.
    xyz = torch.tensor([[[0.0, 0.0, 0.0]]])  # (B=1, N=1, 3)
    centroids = torch.tensor([[[0.0, 0.0, 0.0]]])  # (B=1, S=1, 3)
    idx = ball_query(radius=1.0, k=16, xyz=xyz, centroid_xyz=centroids)
    assert idx.shape == (1, 1, 16)
    assert (idx == 0).all()  # only point 0 exists; every slot must fall back to it


def test_set_abstraction_output_shape():
    sa = SetAbstraction(n_samples=16, radius=0.5, k=8, in_ch=0, mlp_channels=[8, 16])
    xyz = torch.randn(2, 64, 3)
    new_xyz, new_points = sa(xyz, None)
    assert new_xyz.shape == (2, 16, 3)
    assert new_points.shape == (2, 16, 16)


def test_feature_propagation_output_shape():
    fp = FeaturePropagation(in_ch=16 + 8, mlp_channels=[16, 16])
    xyz1 = torch.randn(2, 64, 3)   # denser target level
    xyz2 = torch.randn(2, 16, 3)   # sparser source level
    points1 = torch.randn(2, 64, 8)
    points2 = torch.randn(2, 16, 16)
    out = fp(xyz1, xyz2, points1, points2)
    assert out.shape == (2, 64, 16)


def test_feature_propagation_global_broadcast_case():
    # S == 1 (the deepest SA level, a single global feature) must broadcast
    # to every point rather than doing distance-weighted interpolation.
    fp = FeaturePropagation(in_ch=32, mlp_channels=[16])
    xyz1 = torch.randn(1, 10, 3)
    xyz2 = torch.randn(1, 1, 3)
    points2 = torch.randn(1, 1, 32)
    out = fp(xyz1, xyz2, None, points2)
    assert out.shape == (1, 10, 16)
