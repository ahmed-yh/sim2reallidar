"""Same visual check as the CNN candidate's "decode and compare as an
image" test, adapted for PointNet++'s free-form point output.

The CNN decoder outputs directly in the (64, 1024) range-image grid, so
comparing two heatmaps pixel-for-pixel is natural there. PointNet++'s
ReconstructionDecoder outputs 2,048 unordered XYZ points with no ring/azimuth
identity (deliberate -- see README, this is what makes it a real test of the
latent rather than a grid-shaped shortcut). To get a directly comparable
heatmap, each real ray's true 3D position is nearest-neighbor matched against
the reconstructed point cloud, and that neighbor's range is plotted in the
real ray's grid cell -- i.e. "what does the reconstruction show near this ray."

    python3 visualize_range_heatmap.py

Picks TEST-split samples (never seen in training or in early-stopping
decisions) that each contain a lot of one present object class, so the
object itself -- not just corridor wall -- is visible in the comparison.
That selection step is dataset.pick_samples_per_class, shared with
visualize_reconstruction.py; only the plotting below differs between them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import CLASS_NAMES, PointCloudDataset, pick_samples_per_class, resolve_recording_path
from models.pointnet2 import PointNet2MultiTask
from sensor import N_COLS, N_RINGS


def load_full_range_grid(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path)
    xyz = np.stack([d["x"], d["y"], d["z"]], axis=-1).reshape(N_RINGS, N_COLS, 3)
    rng = np.linalg.norm(xyz, axis=-1)
    rng[~np.isfinite(rng)] = np.nan
    return xyz, rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings-dir", type=Path,
                     default=Path(__file__).parent / "scenario_data" / "mbut_76scenarios" / "o_robot_nav")
    ap.add_argument("--splits-dir", type=Path, default=Path(__file__).parent / "data" / "sim")
    ap.add_argument("--checkpoint", type=Path,
                     default=Path(__file__).parent / "outputs" / "pointnet2" / "best_multitask_model.pt")
    ap.add_argument("--out-dir", type=Path,
                     default=Path(__file__).parent / "outputs" / "pointnet2" / "range_heatmaps")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    test_ids = np.load(args.splits_dir / "test_ids.npy").tolist()
    print("Scanning test split for the best example of each object class...")
    chosen = pick_samples_per_class(args.recordings_dir, test_ids)
    print(f"  chosen: {chosen}")

    model = PointNet2MultiTask().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    ds = PointCloudDataset(args.recordings_dir, list(chosen.values()))

    with torch.no_grad():
        for (c, sid), (xyz_in, _cls_in) in zip(chosen.items(), ds):
            out = model(xyz_in.unsqueeze(0).to(device))
            recon = out["recon_points"][0]  # (2048, 3), on device

            npz_path = resolve_recording_path(args.recordings_dir, sid)
            xyz_grid, range_grid = load_full_range_grid(npz_path)
            valid = np.isfinite(range_grid)
            xyz_valid = torch.from_numpy(xyz_grid[valid]).float().to(device)

            dists = torch.cdist(xyz_valid, recon)  # (M, 2048)
            nn_range = recon[dists.argmin(dim=1)].norm(dim=-1).cpu().numpy()

            recon_range_grid = np.full_like(range_grid, np.nan)
            recon_range_grid[valid] = nn_range

            vmax = float(np.nanpercentile(range_grid, 99))
            fig, axes = plt.subplots(2, 1, figsize=(14, 6.8), sharex=True)
            im = None
            for ax, grid, title in [
                (axes[0], range_grid, "Original (real recording)"),
                (axes[1], recon_range_grid, "Reconstruction (latent-only, reprojected onto real rays)"),
            ]:
                im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=vmax, aspect="auto", origin="lower")
                ax.set_title(title)
                ax.set_ylabel("Ring (0 = lowest beam, ground-facing)")
            axes[1].set_xlabel("Azimuth column")
            fig.colorbar(im, ax=list(axes), label="range (m)", fraction=0.025, pad=0.01)
            fig.suptitle(f"{sid}  --  highlighting {CLASS_NAMES[c]}")

            out_path = args.out_dir / f"{sid.replace('__', '_')}.png"
            fig.savefig(out_path, dpi=130)
            plt.close(fig)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
