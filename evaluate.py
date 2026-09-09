"""Evaluate a trained PointNet++ checkpoint on the held-out TEST split.

Unlike train.py's validation split (used for early-stopping decisions during
training, so not a fully unbiased estimate), the test split is never touched
until this script runs -- the only honest read of real performance.

    python3 evaluate.py

Reports:
- Chamfer reconstruction distance (same metric train.py logs for val, so
  directly comparable to the val numbers printed during training)
- Per-point classification: overall accuracy (inflated by ~99% environment
  points, reported for context only) and per-class recall/precision, which
  is the metric that actually matters given the extreme class imbalance
  (see README "The data").
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PointCloudDataset
from losses import chamfer_distance
from models.pointnet2 import NUM_CLASSES, PointNet2MultiTask

CLASS_NAMES = {0: "environment", 1: "human", 2: "car", 3: "bus",
               4: "sphere", 5: "cylinder", 6: "box2", 7: "box1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings-dir", type=Path,
                     default=Path(__file__).parent / "scenario_data" / "mbut_76scenarios" / "o_robot_nav")
    ap.add_argument("--splits-dir", type=Path, default=Path(__file__).parent / "data" / "sim")
    ap.add_argument("--checkpoint", type=Path,
                     default=Path(__file__).parent / "outputs" / "pointnet2" / "best_multitask_model.pt")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ids = np.load(args.splits_dir / "test_ids.npy")
    print(f"test: {len(test_ids)} samples, device: {device}")
    print(f"checkpoint: {args.checkpoint}")

    loader = DataLoader(
        PointCloudDataset(args.recordings_dir, test_ids),
        batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    model = PointNet2MultiTask().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    confusion = torch.zeros(NUM_CLASSES * NUM_CLASSES, dtype=torch.int64)
    recon_total, n_samples = 0.0, 0

    with torch.no_grad():
        for xyz, cls in loader:
            xyz, cls = xyz.to(device), cls.to(device)
            out = model(xyz)

            recon_total += chamfer_distance(out["recon_points"], xyz).item() * xyz.size(0)
            n_samples += xyz.size(0)

            pred = out["class_logits"].argmax(dim=-1)
            idx = (cls.reshape(-1) * NUM_CLASSES + pred.reshape(-1)).cpu()
            confusion += torch.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES)

    confusion = confusion.reshape(NUM_CLASSES, NUM_CLASSES)  # [true, pred]

    print(f"\nTest Chamfer reconstruction distance: {recon_total / n_samples:.5f}  "
          f"(val was ~0.924 at the restored checkpoint)")

    print(f"\n{'class':<12} {'true points':>12} {'recall':>8} {'precision':>10}")
    total_points = confusion.sum().item()
    correct = confusion.diag().sum().item()
    for c in range(NUM_CLASSES):
        true_count = confusion[c].sum().item()
        pred_count = confusion[:, c].sum().item()
        if true_count == 0 and pred_count == 0:
            continue
        recall = confusion[c, c].item() / true_count if true_count > 0 else float("nan")
        precision = confusion[c, c].item() / pred_count if pred_count > 0 else float("nan")
        print(f"{CLASS_NAMES[c]:<12} {true_count:>12} {recall:>8.1%} {precision:>10.1%}")

    print(f"\nOverall point accuracy: {correct / total_points:.2%} "
          f"(inflated by ~99% environment points -- per-class recall above is the real signal)")


if __name__ == "__main__":
    main()
