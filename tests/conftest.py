"""Shared test fixtures.

Deliberately synthetic, not the real ~400MB scenario recordings -- tests
must be runnable by anyone who clones this repo, without that external data
mounted. The synthetic recording matches the REAL file format exactly (same
fields, same ring-major point order, same off-integer intensity noise, same
inf-for-no-return convention) so passing these tests is meaningful, not just
shape-checking against made-up data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

N_RINGS = 64
N_COLS = 1024

# Known marker used by several dataset tests: one single point, given a
# specific (ring, col), assigned a specific class. Tests assert this exact
# point survives (or is correctly dropped by) decimation.
MARKER_RING = 10
MARKER_COL = 500
MARKER_CLASS_RAW = 6.98863  # off-integer, matching the real GPU-retro noise
MARKER_CLASS_ROUNDED = 7

NO_RETURN_RING = 0
NO_RETURN_COL = 0


def _make_recording_arrays(rng: np.random.Generator) -> dict[str, np.ndarray]:
    ring = np.repeat(np.arange(N_RINGS, dtype=np.uint16), N_COLS)

    # Simple synthetic "tunnel wall": points on a circle of varying radius,
    # finite and range-realistic (a few meters), never zero, never inf,
    # except the one deliberately-placed no-return marker below.
    azimuth = np.tile(np.linspace(0, 2 * np.pi, N_COLS, endpoint=False), N_RINGS)
    radius = 2.0 + 0.1 * rng.standard_normal(N_RINGS * N_COLS)
    x = (radius * np.cos(azimuth)).astype(np.float32)
    y = (radius * np.sin(azimuth)).astype(np.float32)
    z = rng.uniform(-0.5, 0.5, N_RINGS * N_COLS).astype(np.float32)

    intensity = np.zeros(N_RINGS * N_COLS, dtype=np.float32)

    idx = lambda r, c: r * N_COLS + c  # matches the real ring-major layout

    intensity[idx(MARKER_RING, MARKER_COL)] = MARKER_CLASS_RAW

    x[idx(NO_RETURN_RING, NO_RETURN_COL)] = np.inf
    y[idx(NO_RETURN_RING, NO_RETURN_COL)] = np.inf
    z[idx(NO_RETURN_RING, NO_RETURN_COL)] = np.inf

    return {"x": x, "y": y, "z": z, "intensity": intensity, "ring": ring}


@pytest.fixture
def synthetic_recording_dir(tmp_path: Path) -> Path:
    """A directory of two synthetic recordings, named like the real ones."""
    rng = np.random.default_rng(42)
    for name in ["2026_01_01_00_00_00_000", "2026_01_01_00_00_01_000"]:
        arrays = _make_recording_arrays(rng)
        np.savez(tmp_path / f"{name}.npz", **arrays, allow_pickle=False)
    return tmp_path


@pytest.fixture
def synthetic_sample_ids() -> list[str]:
    return ["2026_01_01_00_00_00_000", "2026_01_01_00_00_01_000"]
