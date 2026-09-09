"""Visual sanity check: does the reconstruction decoder's output actually
resemble the real input scene, not just score well on Chamfer distance?

Same idea as the CNN/VAE candidates' "decode the bottleneck and compare to
the original recording" check, adapted to point clouds -- there's no 2D
image for this candidate to reconstruct (see README "Why raw points, not
the range grid"), so instead this plots the real input point cloud and the
latent-only reconstruction (ReconstructionDecoder(z), no skip connections --
see README "Why two decoders") from a top-down and side view.

    python3 visualize_reconstruction.py

Picks TEST-split samples (never seen in training or in early-stopping
decisions) that each contain a lot of one present object class, so the
object itself -- not just corridor wall -- is actually visible in the
comparison. One figure per sample, saved as PNG.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import PointCloudDataset
from models.pointnet2 import PointNet2MultiTask

CLASS_NAMES = {0: "environment", 4: "sphere", 5: "cylinder", 6: "box2", 7: "box1"}
CLASS_COLORS = {0: "#c8c8c8", 4: "#1f77b4", 5: "#2ca02c", 6: "#ff7f0e", 7: "#9467bd"}
RECON_COLOR = "#d62728"


def pick_samples_per_class(recordings_dir: Path, test_ids: list[str]) -> dict[int, str]:
    """For each present object class (4,5,6,7), the test sample containing
    the most points of that class -- so each plot actually shows the object,
    not an empty stretch of corridor."""
    best: dict[int, tuple[str, int]] = {}
    ds = PointCloudDataset(recordings_dir, test_ids)
    for i, sample_id in enumerate(test_ids):
        _, cls = ds[i]
        cls = cls.numpy()
        for c in (4, 5, 6, 7):
            count = int((cls == c).sum())
            if count > 0 and (c not in best or count > best[c][1]):
                best[c] = (sample_id, count)
    return {c: sid for c, (sid, _) in best.items()}


def robust_limits(values: np.ndarray, pad_frac: float = 0.2) -> tuple[float, float]:
    """1st-99th percentile window, not min/max -- a handful of long-range
    rays grazing far down the 300m corridor would otherwise force the axis
    out to +-80m and squash the actual ~2m-wide corridor + objects into an
    unreadable blob at the center."""
    lo, hi = np.percentile(values, [1, 99])
    center, half = (lo + hi) / 2, (hi - lo) / 2 * (1 + pad_frac)
    return center - half, center + half


def plot_sample(sample_id: str, xyz: np.ndarray, cls: np.ndarray, recon: np.ndarray,
                 highlight_class: int, out_path: Path) -> None:
    axis_names = ["X", "Y", "Z"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (i1, i2, title) in zip(
        axes, [(0, 1, "top-down (X-Y)"), (0, 2, "side (X-Z)")]
    ):
        for c in sorted(CLASS_NAMES):
            m = cls == c
            if not m.any():
                continue
            size = 2 if c == 0 else 14
            alpha = 0.25 if c == 0 else 0.9
            ax.scatter(xyz[m, i1], xyz[m, i2], s=size, alpha=alpha,
                       c=CLASS_COLORS[c], label=CLASS_NAMES[c], linewidths=0)
        ax.scatter(recon[:, i1], recon[:, i2], s=10, alpha=0.6, c=RECON_COLOR,
                   marker="x", label="reconstruction (z only)", linewidths=0.8)
        ax.set_xlim(*robust_limits(xyz[:, i1]))
        ax.set_ylim(*robust_limits(xyz[:, i2]))
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(axis_names[i1])
        ax.set_ylabel(axis_names[i2])
    axes[0].legend(loc="upper right", fontsize=8, markerscale=2)
    fig.suptitle(f"{sample_id}  --  ground truth vs. latent-only reconstruction  "
                 f"(highlighting: {CLASS_NAMES[highlight_class]})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings-dir", type=Path,
                     default=Path(__file__).parent / "scenario_data" / "mbut_76scenarios" / "o_robot_nav")
    ap.add_argument("--splits-dir", type=Path, default=Path(__file__).parent / "data" / "sim")
    ap.add_argument("--checkpoint", type=Path,
                     default=Path(__file__).parent / "outputs" / "pointnet2" / "best_multitask_model.pt")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "outputs" / "pointnet2" / "recon_viz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ids = np.load(args.splits_dir / "test_ids.npy").tolist()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning test split for the best example of each object class...")
    chosen = pick_samples_per_class(args.recordings_dir, test_ids)
    print(f"  chosen: { {CLASS_NAMES[c]: sid for c, sid in chosen.items()} }")

    model = PointNet2MultiTask().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    ds = PointCloudDataset(args.recordings_dir, list(chosen.values()))
    with torch.no_grad():
        for (c, sample_id), (xyz, cls) in zip(chosen.items(), ds):
            out = model(xyz.unsqueeze(0).to(device))
            recon = out["recon_points"][0].cpu().numpy()
            out_path = args.out_dir / f"{sample_id}_{CLASS_NAMES[c]}.png"
            plot_sample(sample_id, xyz.numpy(), cls.numpy(), recon, c, out_path)
            print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
