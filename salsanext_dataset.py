"""RangeImageDataset for the SalsaNext candidate.

Unlike dataset.py's PointCloudDataset (which decimates, drops no-return
points, and pads back to a fixed point count for PointNet++'s point-set
input), SalsaNext operates natively on the full (64,1024) grid -- no
decimation, no padding, just the raw recording reshaped to its native
resolution plus a per-cell validity mask so no-return cells can be excluded
from the loss instead of faked with a padded duplicate.

Reads the SAME raw recordings dataset.py does (x, y, z, intensity), not
data/sim/scans/*.npz -- for the same reason dataset.py gives: only the raw
recording has real xyz, needed here for the (range,x,y,z) input channels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import resolve_recording_path
from sensor import N_COLS, N_RINGS


class RangeImageDataset(Dataset):
    def __init__(self, recordings_dir: str | Path, sample_ids: Sequence[str], max_range: float,
                 augment: bool = False):
        """augment: inject synthetic real-sensor noise (real_noise_augment.py)
        into every sample -- ONLY meant for the training split. See
        dataset.py's PointCloudDataset docstring for why this is
        deliberately unseeded (fresh noise every epoch, unlike padding
        elsewhere which needs to be reproducible)."""
        self.recordings_dir = Path(recordings_dir)
        self.sample_ids = list(sample_ids)
        self.max_range = max_range
        self.augment = augment
        if len(self.sample_ids) == 0:
            raise ValueError("sample_ids is empty")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]
        npz_path = resolve_recording_path(self.recordings_dir, sample_id)
        d = np.load(npz_path)
        x, y, z, intensity = d["x"], d["y"], d["z"], d["intensity"]
        x, y, z = x.reshape(N_RINGS, N_COLS), y.reshape(N_RINGS, N_COLS), z.reshape(N_RINGS, N_COLS)
        cls = np.rint(intensity).astype(np.int64).reshape(N_RINGS, N_COLS)

        if self.augment:
            from real_noise_augment import inject_real_sensor_noise
            xyz = np.stack([x, y, z], axis=-1)
            xyz = inject_real_sensor_noise(xyz, np.random.default_rng(), cls_grid=cls)
            x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        range_ = np.zeros((N_RINGS, N_COLS), dtype=np.float32)
        range_[valid] = np.sqrt(x[valid] ** 2 + y[valid] ** 2 + z[valid] ** 2)

        # Normalize everything by the SAME shared max_range used across all
        # three candidates' fair comparison (see kevin_pipeline/) -- so
        # SalsaNext's recon_range head predicts the identical normalized
        # quantity Kevin's CNN and the reprojected PointNet++ output are
        # already scored in, with no extra reprojection needed.
        range_n = np.clip(range_, 0, self.max_range) / self.max_range
        x_n = np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) / self.max_range, -1, 1)
        y_n = np.clip(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0) / self.max_range, -1, 1)
        z_n = np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0) / self.max_range, -1, 1)

        inp = np.stack([range_n, x_n, y_n, z_n], axis=0).astype(np.float32)
        inp[:, ~valid] = 0.0  # no-return cells carry no real signal in any channel

        return (
            torch.from_numpy(inp),
            torch.from_numpy(range_n.astype(np.float32)),
            torch.from_numpy(cls),
            torch.from_numpy(valid),
        )


def compute_class_counts(recordings_dir: str | Path, sample_ids: Sequence[str]) -> dict[int, int]:
    """Per-class cell counts over the given split, restricted to valid
    (returned) cells -- mirrors dataset.py's compute_class_counts, same
    purpose (loss weighting via losses.compute_class_weights)."""
    recordings_dir = Path(recordings_dir)
    counts: dict[int, int] = {}
    for sample_id in sample_ids:
        npz_path = resolve_recording_path(recordings_dir, sample_id)
        d = np.load(npz_path)
        x, y, z, intensity = d["x"], d["y"], d["z"], d["intensity"]
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        cls = np.rint(intensity).astype(np.int64).reshape(N_RINGS, N_COLS)
        u, c = np.unique(cls[valid.reshape(N_RINGS, N_COLS)], return_counts=True)
        for uu, cc in zip(u.tolist(), c.tolist()):
            counts[uu] = counts.get(uu, 0) + cc
    return counts
