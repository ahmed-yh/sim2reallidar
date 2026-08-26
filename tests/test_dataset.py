import numpy as np
import torch

from dataset import N_POINTS, PointCloudDataset, compute_class_counts
from tests.conftest import MARKER_CLASS_ROUNDED, MARKER_COL


def test_output_shape(synthetic_recording_dir, synthetic_sample_ids):
    ds = PointCloudDataset(synthetic_recording_dir, synthetic_sample_ids)
    xyz, cls = ds[0]
    assert xyz.shape == (N_POINTS, 3)
    assert cls.shape == (N_POINTS,)
    assert xyz.dtype == torch.float32
    assert cls.dtype == torch.int64


def test_no_return_points_never_leak_through(synthetic_recording_dir, synthetic_sample_ids):
    # The synthetic fixture plants exactly one inf point (NO_RETURN_RING,
    # NO_RETURN_COL). After decimation + filtering, the dataset's output
    # must contain zero non-finite values -- an unfiltered inf here would
    # poison every downstream distance computation (FPS, ball query,
    # Chamfer distance) with NaN.
    ds = PointCloudDataset(synthetic_recording_dir, synthetic_sample_ids)
    xyz, _ = ds[0]
    assert torch.isfinite(xyz).all()


def test_marker_survives_decimation_when_column_is_even(synthetic_recording_dir, synthetic_sample_ids):
    # Decimation keeps every ring (RING_STRIDE=1) and every 2nd column
    # (COL_STRIDE=2). The fixture's marker sits at MARKER_COL=500, an even
    # column, so it must survive with its class label intact.
    assert MARKER_COL % 2 == 0, "test assumes an even marker column"
    ds = PointCloudDataset(synthetic_recording_dir, synthetic_sample_ids)
    xyz, cls = ds[0]
    assert (cls == MARKER_CLASS_ROUNDED).any(), "marker class not found after decimation"


def test_rounding_gotcha_handled(synthetic_recording_dir, synthetic_sample_ids):
    # The marker's raw intensity is 6.98863 (off-integer GPU-retro noise,
    # matching the real data documented in the scenario README). It must
    # round to 7, not truncate to 6 or fail an exact-equality check.
    ds = PointCloudDataset(synthetic_recording_dir, synthetic_sample_ids)
    xyz, cls = ds[0]
    assert MARKER_CLASS_ROUNDED == 7
    assert (cls == 7).any()
    assert not (cls == 6).any() or MARKER_CLASS_ROUNDED == 6  # no stray truncation to 6


def test_class_values_in_valid_range(synthetic_recording_dir, synthetic_sample_ids):
    ds = PointCloudDataset(synthetic_recording_dir, synthetic_sample_ids)
    for i in range(len(ds)):
        _, cls = ds[i]
        assert cls.min() >= 0
        assert cls.max() <= 7


def test_padding_is_deterministic_across_repeated_loads(synthetic_recording_dir, synthetic_sample_ids):
    # Regression test for the zlib.crc32-seeded padding: loading the same
    # file twice must produce byte-identical output. (Using the builtin
    # hash() for the seed, tried earlier, would NOT satisfy this --
    # str hashing is randomized per-process unless PYTHONHASHSEED is fixed.)
    ds = PointCloudDataset(synthetic_recording_dir, synthetic_sample_ids)
    xyz_1, cls_1 = ds[0]
    xyz_2, cls_2 = ds[0]
    assert torch.equal(xyz_1, xyz_2)
    assert torch.equal(cls_1, cls_2)


def test_compute_class_counts_matches_manual_count(synthetic_recording_dir, synthetic_sample_ids):
    counts = compute_class_counts(synthetic_recording_dir, synthetic_sample_ids)
    ds = PointCloudDataset(synthetic_recording_dir, synthetic_sample_ids)
    manual = {}
    for i in range(len(ds)):
        _, cls = ds[i]
        u, c = np.unique(cls.numpy(), return_counts=True)
        for uu, cc in zip(u.tolist(), c.tolist()):
            manual[uu] = manual.get(uu, 0) + cc
    assert counts == manual
