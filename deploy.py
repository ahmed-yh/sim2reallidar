"""Deploy the trained PointNet++ encoder: one Ouster scan in, one 256-d observation out.

Needs only numpy and torch (no ROS2, matplotlib or PIL), so this is the module to import on the
robot. Run from the repo root, or put the repo root on PYTHONPATH.

    from deploy import ScanEncoder
    encoder = ScanEncoder()        # weights/pointnet2_epoch39_encoder.pt, on CUDA if available
    latent = encoder(xyz_grid)     # (64, 1024, 3) float32 as the Ouster driver delivers it -> (256,)

    python deploy.py scan.npy      # the same from the command line; writes scan_latent.npy

real_scan_to_points is also what ros2_inference/ uses, so the robot, the bag runner and the live
node all prepare real scans the same way. See docs/deployment.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from dataset import COL_STRIDE, RING_STRIDE, prepare_points
from models.pointnet2 import PointNet2Encoder
from sensor import N_COLS, N_RINGS, SIM_RANGE_M, check_sim_order, ouster_to_sim_order

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "pointnet2_epoch39_encoder.pt"


class SkipScan(Exception):
    """A scan the PointNet++ path declines to encode, with the reason. Callers count or log these
    instead of dying on them."""


# SA1 samples 4,096 centres; a scan with fewer real points than that would be
# mostly padding duplicates -- nothing like what the model was trained on.
MIN_VALID_POINTS = 4096

# For each cell of the DECIMATED simulator-order grid (what prepare_points
# flattens), the flat index of the Ouster-grid cell it came from.
_SIM_DEC_TO_OUSTER = ouster_to_sim_order(
    np.arange(N_RINGS * N_COLS).reshape(N_RINGS, N_COLS))[::RING_STRIDE, ::COL_STRIDE].reshape(-1)


def real_scan_to_points(xyz_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N_RINGS, N_COLS, 3) scan as the Ouster driver delivers it -> exactly
    what training fed the model: (N_POINTS, 3) float32 in simulator order,
    padded the same way (dataset.prepare_points). Also returns ouster_idx:
    ouster_idx[i] is the flat index in xyz_grid of real point i; the points
    after len(ouster_idx) are padding.

    Returns outside the range the simulated sensor ever produced
    (sensor.SIM_RANGE_M) count as no-returns -- that includes (0,0,0)."""
    with np.errstate(invalid="ignore"):
        r = np.linalg.norm(xyz_grid, axis=-1)
        inside = (r >= SIM_RANGE_M[0]) & (r <= SIM_RANGE_M[1])
    n_valid = int(inside.reshape(-1)[_SIM_DEC_TO_OUSTER].sum())
    if n_valid < MIN_VALID_POINTS:
        raise SkipScan(f"only {n_valid} valid returns in the classified columns "
                       f"(need {MIN_VALID_POINTS})")
    sim = ouster_to_sim_order(np.where(inside[..., None], xyz_grid, np.nan))
    pts, _, kept = prepare_points(sim, pad_seed=0)
    return pts, _SIM_DEC_TO_OUSTER[kept]


class ScanEncoder:
    """Loads the encoder once; each call encodes one scan. The first scan's point order is
    checked (sensor.check_sim_order), so a differently configured driver is refused instead of
    silently giving a wrong latent. Raises SkipScan for a scan with too few valid returns."""

    def __init__(self, weights: Path | str = DEFAULT_WEIGHTS, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = PointNet2Encoder()
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()
        self._order_checked = False

    @torch.no_grad()
    def __call__(self, xyz_grid: np.ndarray) -> np.ndarray:
        if not self._order_checked:
            check_sim_order(ouster_to_sim_order(xyz_grid))
            self._order_checked = True
        points, _ = real_scan_to_points(xyz_grid)
        latent, _ = self.model(torch.from_numpy(points).unsqueeze(0).to(self.device))
        return latent[0].cpu().numpy()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python deploy.py scan.npy   "
                         "(one (64, 1024, 3) float32 scan in Ouster order, NaN for no-return)")
    scan_path = Path(sys.argv[1])
    encoder = ScanEncoder()
    latent = encoder(np.load(scan_path).astype(np.float32))
    out = scan_path.with_name(scan_path.stem + "_latent.npy")
    np.save(out, latent)
    print(f"wrote {out}: latent {latent.shape} on {encoder.device}")
