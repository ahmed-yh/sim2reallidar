"""Synthetic real-sensor-noise augmentation, applied ONLY to the training
split (never val/test -- those stay clean simulated data, so validation
loss still honestly measures whether the model is learning, not whether
this file's noise model happens to be easy or hard).

Motivation: this project's simulated LiDAR is noise-free -- no dropout
beyond genuine geometric occlusion. Real Ouster OS1 bags show meaningfully
higher no-return dropout than any simulated scan, and (2026-09-16 finding)
that dropout is strongly RING-DEPENDENT, not flat: measured directly from
11 real bag3 frames spread across the whole drive, it ranges from ~2% in
the mid rings up to 34% at ring 63 (the most ground-facing beam -- a beam
grazing the floor at a steep angle scatters away from the sensor instead
of reflecting straight back, so it misses far more often). The model had
never seen this pattern, and produced obvious false-positive object
classifications on real scans as a result.

This is domain randomization, not real-data fine-tuning: it doesn't need
or use any real bag at training time, since real data has no ground-truth
labels to supervise against anyway -- it just teaches the model, using
ONLY the labels we already trust (simulated), to not be thrown by the
dropout pattern actually observed on real hardware.

An earlier version of this file also injected synthetic short-range
"phantom" points, modeled on a visual impression of bright near-range
speckle in real bag playbacks. Checked directly against real data
(2026-09-16) and removed: genuine isolated near-range spikes (a cell
whose range is >1m shorter than both neighbors -- what a real spurious
return would look like, as opposed to a real surface) occur in only
~0.06% of real returns, not the 5% that was being injected. The "bright"
speckle that motivated it was a colormap misreading: bright/yellow in
this project's viridis rendering means FAR range or no-return, not near
-- the actual phenomenon being seen was the ring-dependent dropout above,
not phantom close points. Don't re-add false-point injection without a
fresh measurement backing a specific rate.

Injected on the RAW (ring, col, 3) grid, before decimation/padding --
downstream code (dataset.py's _load_raw, salsanext_dataset.py) is
unchanged and unaware augmentation happened; it just sees a slightly
noisier xyz array.
"""
from __future__ import annotations

import numpy as np

N_RINGS = 64

# Measured no-return (dropout) rate per ring, averaged over 11 real Ouster
# OS1-64 frames spread across the full bag3 drive (real_bags/bag3_20260910_152806,
# 2026-09-16). Ring 0 is the highest/most upward beam (~+21 deg elevation,
# see ros2_inference/_shared_inference.py's orientation note) and also runs
# a bit elevated (~10%), plausibly scanning past the corridor into open
# space more often; the climb from ring ~30 onward is the floor-grazing
# effect described above.
REAL_RING_DROPOUT_RATE = np.array([
    0.1005, 0.0663, 0.0467, 0.0312, 0.0377, 0.0256, 0.0217, 0.0270,
    0.0234, 0.0265, 0.0199, 0.0216, 0.0243, 0.0248, 0.0258, 0.0283,
    0.0277, 0.0239, 0.0279, 0.0274, 0.0246, 0.0216, 0.0222, 0.0199,
    0.0215, 0.0256, 0.0178, 0.0205, 0.0222, 0.0260, 0.0313, 0.0328,
    0.0354, 0.0431, 0.0424, 0.0485, 0.0583, 0.0471, 0.0390, 0.0465,
    0.0463, 0.0471, 0.0563, 0.0577, 0.0566, 0.0527, 0.0629, 0.0692,
    0.0964, 0.0991, 0.1199, 0.1372, 0.1470, 0.1575, 0.1460, 0.1569,
    0.1650, 0.1574, 0.1416, 0.1795, 0.1871, 0.2120, 0.1982, 0.3441,
], dtype=np.float32)
assert REAL_RING_DROPOUT_RATE.shape == (N_RINGS,)


def inject_real_sensor_noise(
    xyz_grid: np.ndarray,
    rng: np.random.Generator,
    cls_grid: np.ndarray | None = None,
    ring_dropout_rate: np.ndarray = REAL_RING_DROPOUT_RATE,
) -> np.ndarray:
    """xyz_grid: (N_RINGS, N_COLS, 3), NaN (or any non-finite) where the
    real recording had no return. Returns a NEW array (does not modify
    xyz_grid in place).

    cls_grid: (N_RINGS, N_COLS) class labels, if given. Dropout is
    restricted to cells NOT labeled as one of the rare real object classes
    (1-7) -- corrupting an object cell's xyz while leaving its label
    untouched would teach the model to predict that object FROM missing
    data, exactly backwards, and object classes are already too rare
    (well under 1% of points) to risk damaging further. Environment-
    labeled cells are fair game; that's the actual noise this is modeling.

    ring_dropout_rate: (N_RINGS,) per-ring extra-dropout probability,
    stacked on top of whatever dropout the simulated scan already has
    (near zero, since sim dropout is geometric-occlusion-only). Defaults
    to the measured real-hardware profile above; override for experiments.
    """
    h, w, _ = xyz_grid.shape
    flat = xyz_grid.reshape(h * w, 3).copy()
    valid = np.isfinite(flat).all(axis=-1)

    if cls_grid is not None:
        eligible = (cls_grid.reshape(h * w) == 0)
    else:
        eligible = np.ones(h * w, dtype=bool)

    per_cell_rate = np.repeat(ring_dropout_rate, w)  # ring-major, matches xyz_grid's own row order
    drop = valid & eligible & (rng.random(h * w) < per_cell_rate)
    flat[drop] = np.nan

    return flat.reshape(h, w, 3)
