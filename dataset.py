"""PointCloudDataset for the PointNet++ candidate.

Reads directly from the ORIGINAL raw recordings (x, y, z, intensity, ring --
real 3D coordinates), not from data/sim/scans/*.npz (which only stores a
scalar range value, computed during Phase 1 parsing for the CNN/VAE
candidates, and therefore cannot be inverted back into xyz without knowing
this simulated sensor's exact per-ring beam angles). Keyed by the same
sample_id used everywhere else in this project, so the existing spatial-block
train/val/test split (data/sim/{train,val,test}_ids.npy) applies unchanged.

Decimation: full ring resolution (all 64 rings) + every 2nd azimuth column ->
64 x 512 = 32,768 points. This specific axis choice (decimate azimuth, never
rings) is not arbitrary -- see README "Why decimate azimuth, not rings" for
the empirical class-coverage comparison that justifies it.
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

RING_STRIDE = 1
COL_STRIDE = 2
N_RINGS = 64
N_COLS = 1024
N_POINTS = (N_RINGS // RING_STRIDE) * (N_COLS // COL_STRIDE)  # 32,768

# Raw laser_retro values that are actually possible per the scenario README:
# 2.0 ("mercedes") is a dead duplicate of "car" (placeable: False, never
# spawns) -- rounding still needs to handle it landing on 2, since that's a
# legitimate value class-index-wise (identical to "car"), just never
# observed with a *different* meaning than car in this data.
NUM_CLASSES = 8  # raw indices 0-7


def resolve_recording_path(recordings_dir: Path, sample_id: str) -> Path:
    """sample_id is either a bare timestamp (single-scenario layout) or
    "{scenario_id}__{timestamp}" (multi-scenario layout, see
    data_prep/parse_recordings.py's module docstring) -- the latter lives
    one directory level deeper, under its own scenario_id subdirectory."""
    if "__" in sample_id:
        scenario_id, timestamp = sample_id.split("__", 1)
        return recordings_dir / scenario_id / f"{timestamp}.npz"
    return recordings_dir / f"{sample_id}.npz"


def _load_raw(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """One recording -> decimated (xyz, class) point arrays, no-return
    points dropped, padded back to exactly N_POINTS by resampling existing
    valid points (keeps every sample's tensor shape identical for batching)."""
    d = np.load(npz_path)
    x, y, z, intensity = d["x"], d["y"], d["z"], d["intensity"]
    if x.size != N_RINGS * N_COLS:
        raise ValueError(f"{npz_path}: expected {N_RINGS * N_COLS} points, got {x.size}")

    xyz = np.stack([x, y, z], axis=-1).reshape(N_RINGS, N_COLS, 3)
    cls = np.rint(intensity).astype(np.int64).reshape(N_RINGS, N_COLS)

    xyz = xyz[::RING_STRIDE, ::COL_STRIDE].reshape(-1, 3)
    cls = cls[::RING_STRIDE, ::COL_STRIDE].reshape(-1)
    assert xyz.shape[0] == N_POINTS, f"decimation produced {xyz.shape[0]}, expected {N_POINTS}"

    finite = np.isfinite(xyz).all(axis=-1)
    xyz, cls = xyz[finite], cls[finite]

    n_missing = N_POINTS - xyz.shape[0]
    if n_missing > 0:
        # zlib.crc32, not the builtin hash() -- str hashing is randomized
        # per-process (PYTHONHASHSEED) unless disabled, which would make
        # this padding non-reproducible run to run.
        seed = zlib.crc32(npz_path.name.encode("utf-8"))
        pad_idx = np.random.default_rng(seed).choice(
            xyz.shape[0], size=n_missing, replace=True
        )
        xyz = np.concatenate([xyz, xyz[pad_idx]], axis=0)
        cls = np.concatenate([cls, cls[pad_idx]], axis=0)

    return xyz.astype(np.float32), cls


class PointCloudDataset(Dataset):
    def __init__(self, recordings_dir: str | Path, sample_ids: Sequence[str]):
        self.recordings_dir = Path(recordings_dir)
        self.sample_ids = list(sample_ids)
        if len(self.sample_ids) == 0:
            raise ValueError("sample_ids is empty")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.sample_ids[idx]
        npz_path = resolve_recording_path(self.recordings_dir, sample_id)
        xyz, cls = _load_raw(npz_path)
        return torch.from_numpy(xyz), torch.from_numpy(cls)


def compute_class_counts(recordings_dir: str | Path, sample_ids: Sequence[str]) -> dict[int, int]:
    """Per-class point counts over the GIVEN split, after the same decimation
    the model actually trains on -- used for loss weighting (losses.py). Must
    be computed on the decimated distribution, not the full-resolution one:
    decimation itself changes per-class visibility (see README), so weights
    calibrated on full-res data would be calibrated for a distribution the
    model never actually sees."""
    recordings_dir = Path(recordings_dir)
    counts: dict[int, int] = {}
    for sample_id in sample_ids:
        _, cls = _load_raw(resolve_recording_path(recordings_dir, sample_id))
        u, c = np.unique(cls, return_counts=True)
        for uu, cc in zip(u.tolist(), c.tolist()):
            counts[uu] = counts.get(uu, 0) + cc
    return counts
