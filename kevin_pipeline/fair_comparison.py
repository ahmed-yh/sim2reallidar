"""Cross-model comparison: PointNet++ vs. Kevin's CNN autoencoder vs.
SalsaNext, on the SAME held-out test set, in the SAME units where that's
honestly possible -- and clearly flagged where it isn't.

The three candidates don't share a native output format:
  PointNet++:  32,768 raw xyz points in -> 2,048 free-floating xyz points out.
               Native metric: symmetric Chamfer distance, in meters.
  Kevin's CNN: (64,1024) normalized range image in -> (64,1024) range image out.
               Native metric: MSE / SSIM, in NORMALIZED [0,1] range units.
  SalsaNext:   (64,1024) 4-channel grid in -> (64,1024) range image out, natively
               -- same grid shape as Kevin's, no reprojection needed either.

Reconstruction is the one thing all three candidates are actually trying to
do, so it's the one thing worth forcing onto a shared scale: PointNet++'s
free-form output is reprojected onto the real ray grid (nearest reconstructed
point per real ray -- same technique as visualize_range_heatmap.py, here run
over the WHOLE test set for an aggregate number instead of a few example
frames), then scored with Kevin's own MSE/SSIM formula using Kevin's own
max_range. SalsaNext's own dataset loader already normalizes by that same
shared max_range (see salsanext_dataset.py), so its output plots into the
identical scale with no extra math at all.

Classification is NOT forced onto a shared scale between all three. Kevin's
candidate has no classification output at all -- it was never designed to
have one (see masterarbeit_kevinfischer/lidar_preprocessing/
autoencoder_training: the factory function only ever builds an
encoder+decoder pair for reconstruction, nothing else). Bolting a classifier
onto his frozen features for this comparison would be evaluating a model
that doesn't exist, not the model he actually built -- so this script
reports PointNet++'s and SalsaNext's classification numbers side by side
(both are genuinely, natively per-point/per-pixel classifiers, so THAT
comparison is fair) and says plainly that Kevin's candidate has nothing to
compare against, rather than inventing a number for it.

    python3 kevin_pipeline/fair_comparison.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset import PointCloudDataset, resolve_recording_path  # noqa: E402
from models.pointnet2 import PointNet2MultiTask  # noqa: E402
from models.salsanext import SalsaNextMultiTask  # noqa: E402
from salsanext_dataset import RangeImageDataset  # noqa: E402
from sensor import N_COLS, N_RINGS  # noqa: E402


def load_full_range_grid(npz_path: Path) -> np.ndarray:
    d = np.load(npz_path)
    xyz = np.stack([d["x"], d["y"], d["z"]], axis=-1).reshape(N_RINGS, N_COLS, 3)
    return xyz


def pointnet2_reprojected_metrics(recordings_dir: Path, test_ids: list[str],
                                   checkpoint: Path, max_range: float,
                                   device: torch.device) -> dict:
    """For every test sample: run PointNet++, reproject its free-form
    reconstruction onto the real ray grid, normalize by Kevin's max_range,
    and score with his own MSE/SSIM formula. Returns the mean over the set."""
    from pytorch_msssim import ssim as ssim_fn

    model = PointNet2MultiTask().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ds = PointCloudDataset(recordings_dir, test_ids)

    mse_total, ssim_total, correct, total_pts, n = 0.0, 0.0, 0, 0, 0
    with torch.no_grad():
        for sid, (xyz_in, cls_in) in zip(test_ids, ds):
            out = model(xyz_in.unsqueeze(0).to(device))
            recon = out["recon_points"][0]  # (2048, 3)
            pred_cls = out["class_logits"][0].argmax(dim=-1).cpu()
            correct += (pred_cls == cls_in).sum().item()
            total_pts += cls_in.numel()

            npz_path = resolve_recording_path(recordings_dir, sid)
            xyz_grid = load_full_range_grid(npz_path)
            range_grid = np.linalg.norm(xyz_grid, axis=-1)
            valid = np.isfinite(range_grid)

            xyz_valid = torch.from_numpy(xyz_grid[valid]).float().to(device)
            dists = torch.cdist(xyz_valid, recon)
            nn_range = recon[dists.argmin(dim=1)].norm(dim=-1)

            real_norm = torch.from_numpy(np.clip(range_grid[valid], 0, max_range) / max_range).float().to(device)
            pred_norm = torch.clamp(nn_range / max_range, 0, 1)

            mse_total += F.mse_loss(pred_norm, real_norm).item()

            # SSIM needs the full (1,1,64,1024) grid, not just valid pixels --
            # fill invalid (no-return) cells with 0 in both real and
            # reprojected grids, matching how Kevin's dataset.py's own
            # np.clip(...)/max_range treats them (a no-return has range 0
            # after his pipeline's own normalization, since inf clips to
            # max_range... no: RangeImageDataset clips the ALREADY-computed
            # range array, and a no-return's range is never stored as inf
            # there in the first place -- his data comes from a different
            # export path than ours. Zero-filling here is the closest honest
            # match: "no information" contributes nothing to the structural
            # comparison in either grid, rather than an arbitrary value only
            # one side has.
            real_full = torch.zeros(N_RINGS, N_COLS, device=device)
            pred_full = torch.zeros(N_RINGS, N_COLS, device=device)
            real_full[torch.from_numpy(valid)] = real_norm
            pred_full[torch.from_numpy(valid)] = pred_norm
            ssim_total += ssim_fn(real_full.view(1, 1, N_RINGS, N_COLS),
                                   pred_full.view(1, 1, N_RINGS, N_COLS),
                                   data_range=1.0).item()
            n += 1

    return {"mse": mse_total / n, "ssim": ssim_total / n, "acc": correct / max(total_pts, 1), "n": n}


def salsanext_metrics(recordings_dir: Path, test_ids: list[str], checkpoint: Path,
                       max_range: float, device: torch.device) -> dict:
    """SalsaNext's recon_range head already outputs the same (64,1024) grid,
    normalized by the SAME shared max_range its own dataset loader uses
    (salsanext_dataset.py) -- no reprojection, no renormalization, scored
    directly against Kevin's own MSE/SSIM formula. Classification accuracy
    is also genuinely native here (unlike Kevin's candidate), so it's
    reported alongside PointNet++'s for a real per-point comparison."""
    from pytorch_msssim import ssim as ssim_fn

    model = SalsaNextMultiTask().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ds = RangeImageDataset(recordings_dir, test_ids, max_range)

    mse_total, ssim_total, correct, total_px, n = 0.0, 0.0, 0, 0, 0
    with torch.no_grad():
        for inp, range_n, cls, valid in ds:
            inp, range_n, cls, valid = (t.unsqueeze(0).to(device) for t in (inp, range_n, cls, valid))
            out = model(inp)
            recon_n = out["recon_range"]
            pred_cls = out["class_logits"].argmax(dim=1)

            diff2 = (recon_n - range_n) ** 2 * valid
            mse_total += (diff2.sum() / valid.sum().clamp(min=1)).item()
            ssim_total += ssim_fn(range_n.unsqueeze(1), recon_n.clamp(0, 1).unsqueeze(1), data_range=1.0).item()

            correct += ((pred_cls == cls) & valid.bool()).sum().item()
            total_px += valid.sum().item()
            n += 1

    return {"mse": mse_total / n, "ssim": ssim_total / n, "acc": correct / max(total_px, 1), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointnet2-checkpoint", type=Path,
                     default=REPO_ROOT / "outputs" / "pointnet2_checkpoints" / "epoch39_multitask_model.pt")
    ap.add_argument("--salsanext-checkpoint", type=Path,
                     default=REPO_ROOT / "outputs" / "salsanext" / "best_multitask_model.pt")
    ap.add_argument("--recordings-dir", type=Path,
                     default=REPO_ROOT / "scenario_data" / "mbut_76scenarios" / "o_robot_nav")
    ap.add_argument("--splits-dir", type=Path, default=REPO_ROOT / "data" / "sim")
    ap.add_argument("--kevin-outputs-dir", type=Path, default=REPO_ROOT / "outputs" / "kevin_cnn")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ids = np.load(args.splits_dir / "test_ids.npy").tolist()

    with open(args.kevin_outputs_dir / "normalization_stats.json") as f:
        max_range = json.load(f)["max_range"]
    print(f"Shared normalization: max_range = {max_range:.3f}m (Kevin's formula, computed on OUR train split)")
    print(f"Shared test set: {len(test_ids)} samples (identical for all three candidates -- our spatial-block split)")

    print("\n=== PointNet++ (reprojected onto range grid, Kevin's own MSE/SSIM formula) ===")
    pn2 = pointnet2_reprojected_metrics(args.recordings_dir, test_ids, args.pointnet2_checkpoint, max_range, device)
    print(f"  mse:  {pn2['mse']:.6f}")
    print(f"  ssim: {pn2['ssim']:.4f}")
    print(f"  per-point class acc: {pn2['acc']*100:.2f}%")

    print("\n=== SalsaNext (native grid output, same max_range, no reprojection needed) ===")
    salsa = salsanext_metrics(args.recordings_dir, test_ids, args.salsanext_checkpoint, max_range, device)
    print(f"  mse:  {salsa['mse']:.6f}")
    print(f"  ssim: {salsa['ssim']:.4f}")
    print(f"  per-pixel class acc: {salsa['acc']*100:.2f}%")

    kevin_metrics_path = args.kevin_outputs_dir / "test_metrics.json"
    kevin = None
    if kevin_metrics_path.exists():
        with open(kevin_metrics_path) as f:
            kevin = json.load(f)
        print("\n=== Kevin's CNN (native, same test set, same max_range) ===")
        print(f"  mse:  {kevin['mse']:.6f}")
        print(f"  ssim: {kevin['ssim']:.4f}")
    else:
        print(f"\nKevin's test_metrics.json not found at {kevin_metrics_path}.")

    print("\n=== summary: reconstruction (shared scale, all three) ===")
    print(f"  {'':12} {'mse':>10} {'ssim':>8}")
    print(f"  {'PointNet++':12} {pn2['mse']:>10.6f} {pn2['ssim']:>8.4f}")
    print(f"  {'SalsaNext':12} {salsa['mse']:>10.6f} {salsa['ssim']:>8.4f}")
    if kevin is not None:
        print(f"  {'Kevin CNN':12} {kevin['mse']:>10.6f} {kevin['ssim']:>8.4f}")

    print("\n=== summary: classification (only PointNet++ and SalsaNext have one) ===")
    print(f"  {'':12} {'accuracy':>10}")
    print(f"  {'PointNet++':12} {pn2['acc']*100:>9.2f}%")
    print(f"  {'SalsaNext':12} {salsa['acc']*100:>9.2f}%")
    print("  Kevin's CNN has no classification output at all -- it was never built with one, so")
    print("  it's excluded here rather than given an invented number.")


if __name__ == "__main__":
    main()
