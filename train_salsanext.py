"""Train the SalsaNext candidate.

Mirrors train.py's structure and conventions closely (same CLI shape, same
on_new_best checkpoint-every-improvement discipline -- see
project_training_crash_recovery memory for why that matters) so the two
candidates are easy to compare operationally, not just numerically.

    python3 train_salsanext.py --recordings-dir /path/to/recordings
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from losses import compute_class_weights, make_salsanext_loss_fn
from models.salsanext import LATENT_DIM, NUM_CLASSES, SalsaNextMultiTask
from salsanext_dataset import RangeImageDataset, compute_class_counts
from train_utils import fit, validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings-dir", type=Path,
                     default=Path(__file__).parent / "scenario_data" / "mbut_76scenarios" / "o_robot_nav")
    ap.add_argument("--splits-dir", type=Path, default=Path(__file__).parent / "data" / "sim")
    ap.add_argument("--outputs-dir", type=Path, default=Path(__file__).parent / "outputs" / "salsanext")
    ap.add_argument("--kevin-outputs-dir", type=Path, default=Path(__file__).parent / "outputs" / "kevin_cnn",
                     help="Source of the shared max_range normalization -- see salsanext_dataset.py: "
                          "the recon_range head is scored in the SAME normalized units Kevin's CNN and "
                          "the reprojected PointNet++ output already use, for a direct 3-way comparison.")

    ap.add_argument("--batch-size", type=int, default=16,
                     help="Not benchmarked on the actual Quadro yet. Higher than PointNet++'s default "
                          "(2): this is a compact 2D CNN on a 64x1024 grid, not a point-set model with "
                          "large per-sample distance matrices -- closer in memory footprint to Kevin's "
                          "CNN (his best config used batch_size=32) than to PointNet++.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lambda-class", type=float, default=1.0,
                     help="Weight on the classification loss relative to reconstruction. Both losses "
                          "are already in comparable normalized/masked-mean units (see losses.py), "
                          "unlike PointNet++'s Chamfer-distance-in-meters vs cross-entropy mismatch, "
                          "so 1.0 is a more defensible starting point here than it was there.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--weight-cap", type=float, default=100.0)
    ap.add_argument("--init-from", type=Path, default=None,
                     help="Warm-start model weights from a previous best_multitask_model.pt. "
                          "NOT a true resume -- see train.py's --init-from docstring, same caveats apply.")
    ap.add_argument("--augment-noise", action="store_true",
                     help="Inject synthetic real-sensor noise (real_noise_augment.py) into "
                          "TRAINING samples only -- see train.py's --augment-noise docstring, "
                          "same domain-randomization motivation applies here.")
    args = ap.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    with open(args.kevin_outputs_dir / "normalization_stats.json") as f:
        max_range = json.load(f)["max_range"]
    print(f"shared max_range: {max_range:.3f}m (see --kevin-outputs-dir)")

    train_ids = np.load(args.splits_dir / "train_ids.npy")
    val_ids = np.load(args.splits_dir / "val_ids.npy")
    print(f"train: {len(train_ids)} samples, val: {len(val_ids)} samples")

    print("Computing class weights from the train split...")
    t0 = time.time()
    class_counts = compute_class_counts(args.recordings_dir, train_ids)
    class_weights = compute_class_weights(class_counts, num_classes=NUM_CLASSES, cap=args.weight_cap)
    print(f"  done in {time.time() - t0:.1f}s -- counts: {class_counts}")
    print(f"  weights: {class_weights.tolist()}")

    if args.augment_noise:
        print("Synthetic real-sensor noise augmentation ENABLED for training samples "
              "(see real_noise_augment.py)")
    train_loader = DataLoader(
        RangeImageDataset(args.recordings_dir, train_ids, max_range, augment=args.augment_noise),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        RangeImageDataset(args.recordings_dir, val_ids, max_range),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = SalsaNextMultiTask(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)
    if args.init_from is not None:
        model.load_state_dict(torch.load(args.init_from, map_location=device))
        print(f"Warm-started weights from {args.init_from} (fresh optimizer, epoch count restarts at 1)")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    class_weights = class_weights.to(device)
    loss_fn = make_salsanext_loss_fn()

    def log(epoch, train_recon, train_class, val_recon, val_class):
        print(f"epoch {epoch:3d} | train recon {train_recon:.5f} class {train_class:.5f} "
              f"| val recon {val_recon:.5f} class {val_class:.5f}")

    def save_checkpoint(epoch, model):
        torch.save(model.encoder.state_dict(), args.outputs_dir / "best_encoder.pt")
        torch.save(model.state_dict(), args.outputs_dir / "best_multitask_model.pt")
        print(f"  (epoch {epoch}: new best, checkpoint saved to {args.outputs_dir})", flush=True)

    print(f"\nStarting training: 64x1024 grid input, batch_size={args.batch_size}, "
          f"lambda_class={args.lambda_class}")
    try:
        es = fit(
            model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
            class_weights=class_weights, lambda_class=args.lambda_class, device=device,
            max_epochs=args.max_epochs, patience=args.patience, loss_fn=loss_fn, log_fn=log,
            on_new_best=save_checkpoint,
        )
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(
            f"CUDA out of memory at batch_size={args.batch_size}. Not benchmarked on real training "
            f"hardware yet -- retry with a smaller --batch-size."
        )

    print(f"\nBest val loss: {es.best_val_loss:.5f}")
    held_out_recon, held_out_class = validate(
        model, val_loader, class_weights, args.lambda_class, device, loss_fn
    )
    print(f"Held-out validation: recon {held_out_recon:.5f}, class {held_out_class:.5f}")

    torch.save(model.encoder.state_dict(), args.outputs_dir / "best_encoder.pt")
    torch.save(model.state_dict(), args.outputs_dir / "best_multitask_model.pt")

    metrics = {
        "held_out_val_recon": held_out_recon,
        "held_out_val_class": held_out_class,
        "best_val_loss": es.best_val_loss,
        "class_counts": class_counts,
        "class_weights": class_weights.cpu().tolist(),
        "latent_dim": LATENT_DIM,
        "max_range": max_range,
        "args": vars(args) | {
            "recordings_dir": str(args.recordings_dir),
            "splits_dir": str(args.splits_dir),
            "outputs_dir": str(args.outputs_dir),
            "kevin_outputs_dir": str(args.kevin_outputs_dir),
        },
    }
    with open(args.outputs_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"Saved best_encoder.pt, best_multitask_model.pt, train_metrics.json to {args.outputs_dir}")


if __name__ == "__main__":
    main()
