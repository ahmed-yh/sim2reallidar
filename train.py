"""Train the PointNet++ candidate.

Run on the training machine (Quadro RTX 5000 or equivalent) -- NOT on the
Jetson (inference-only deployment target, see README) and NOT expected to
run a real training job on a CPU-only dev machine (correctness can be
smoke-tested there; actual training cannot).

    python3 train.py --recordings-dir /path/to/twc_scenario_.../recordings

Only one candidate exists in this repo so far, so there's no --model flag
yet (unlike the shared 3-candidate harness this is meant to eventually fit
into) -- adding one is a small extension once the CNN/VAE candidates exist
alongside this one, not a rewrite.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import N_POINTS, PointCloudDataset, compute_class_counts
from losses import compute_class_weights, make_pointnet2_loss_fn
from models.pointnet2 import LATENT_DIM, NUM_CLASSES, PointNet2MultiTask
from train_utils import fit, validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings-dir", type=Path,
                     default=Path(__file__).parent / "scenario_data" / "mbut_76scenarios" / "o_robot_nav")
    ap.add_argument("--splits-dir", type=Path, default=Path(__file__).parent / "data" / "sim")
    ap.add_argument("--outputs-dir", type=Path, default=Path(__file__).parent / "outputs" / "pointnet2")

    ap.add_argument("--batch-size", type=int, default=2,
                     help="Conservative default: at 32,768 points/sample, the FPS/ball-query "
                          "distance matrices are large (SA1 alone is ~537MB per sample just for "
                          "one cdist call). Not benchmarked on the actual Quadro yet -- raise this "
                          "if VRAM allows once you can observe real usage.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lambda-class", type=float, default=1.0,
                     help="Weight on the classification loss relative to reconstruction. "
                          "Start at 1.0 and retune once real loss magnitudes are visible "
                          "(recon is a Chamfer distance in meters, class is a cross-entropy -- "
                          "they are not naturally on the same scale).")
    ap.add_argument("--chamfer-target-subsample", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--weight-cap", type=float, default=100.0,
                     help="Cap on class weights (both for the rarest present class and for "
                          "absent classes, which would otherwise be inf -- see losses.py).")
    ap.add_argument("--init-from", type=Path, default=None,
                     help="Warm-start model weights from a previous best_multitask_model.pt "
                          "instead of random init. NOT a true resume: the optimizer state "
                          "(Adam's momentum/variance) is not saved anywhere by this script, so "
                          "it restarts fresh here, and epoch numbering / early-stopping restart "
                          "from 1 too. Practically fine for continuing to improve a checkpoint, "
                          "just not byte-identical to an uninterrupted run.")
    ap.add_argument("--augment-noise", action="store_true",
                     help="Inject synthetic real-sensor noise (real_noise_augment.py) into "
                          "TRAINING samples only -- domain randomization to close part of the "
                          "sim2real gap (real Ouster bags show false-positive classifications "
                          "the clean simulated data never trained against). Val stays clean, "
                          "so val loss still honestly measures learning, not noise-model luck.")
    args = ap.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ids = np.load(args.splits_dir / "train_ids.npy")
    val_ids = np.load(args.splits_dir / "val_ids.npy")
    print(f"train: {len(train_ids)} samples, val: {len(val_ids)} samples")

    print("Computing class weights from the decimated train split...")
    t0 = time.time()
    class_counts = compute_class_counts(args.recordings_dir, train_ids)
    class_weights = compute_class_weights(class_counts, num_classes=NUM_CLASSES, cap=args.weight_cap)
    print(f"  done in {time.time() - t0:.1f}s -- counts: {class_counts}")
    print(f"  weights: {class_weights.tolist()}")

    if args.augment_noise:
        print("Synthetic real-sensor noise augmentation ENABLED for training samples "
              "(see real_noise_augment.py)")
    train_loader = DataLoader(
        PointCloudDataset(args.recordings_dir, train_ids, augment=args.augment_noise),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        PointCloudDataset(args.recordings_dir, val_ids),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = PointNet2MultiTask(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)
    if args.init_from is not None:
        model.load_state_dict(torch.load(args.init_from, map_location=device))
        print(f"Warm-started weights from {args.init_from} (fresh optimizer, epoch count restarts at 1)")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    class_weights = class_weights.to(device)

    def log(epoch, train_recon, train_class, val_recon, val_class):
        print(f"epoch {epoch:3d} | train recon {train_recon:.5f} class {train_class:.5f} "
              f"| val recon {val_recon:.5f} class {val_class:.5f}")

    def save_checkpoint(epoch, model):
        # Fires on every new best -- see fit()'s docstring for why this
        # can't just wait until the end. Same two files main() writes on
        # completion; a run that crashes after this point still has its
        # best-so-far on disk, not just in a process's RAM.
        torch.save(model.encoder.state_dict(), args.outputs_dir / "best_encoder.pt")
        torch.save(model.state_dict(), args.outputs_dir / "best_multitask_model.pt")
        print(f"  (epoch {epoch}: new best, checkpoint saved to {args.outputs_dir})", flush=True)

    loss_fn = make_pointnet2_loss_fn(chamfer_target_subsample=args.chamfer_target_subsample)

    print(f"\nStarting training: {N_POINTS} points/sample, batch_size={args.batch_size}, "
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
            f"CUDA out of memory at batch_size={args.batch_size}. This has not been benchmarked "
            f"on real training hardware yet -- retry with --batch-size 1, and if that still OOMs, "
            f"reduce --chamfer-target-subsample first (cheaper than reducing point count, which "
            f"would break the train/inference decimation symmetry documented in README)."
        )

    print(f"\nBest val loss: {es.best_val_loss:.5f}")
    held_out_recon, held_out_class = validate(
        model, val_loader, class_weights, args.lambda_class, device, loss_fn
    )
    print(f"Held-out validation: recon {held_out_recon:.5f}, class {held_out_class:.5f}")

    # Only the encoder ever deploys (see README "What actually deploys") --
    # saved separately from the full model, which is kept only for
    # inspection/resuming, matching the existing CNN baseline's convention.
    torch.save(model.encoder.state_dict(), args.outputs_dir / "best_encoder.pt")
    torch.save(model.state_dict(), args.outputs_dir / "best_multitask_model.pt")

    metrics = {
        "held_out_val_recon": held_out_recon,
        "held_out_val_class": held_out_class,
        "best_val_loss": es.best_val_loss,
        "class_counts": class_counts,
        "class_weights": class_weights.cpu().tolist(),
        "n_points": N_POINTS,
        "latent_dim": LATENT_DIM,
        "args": vars(args) | {
            "recordings_dir": str(args.recordings_dir),
            "splits_dir": str(args.splits_dir),
            "outputs_dir": str(args.outputs_dir),
        },
    }
    with open(args.outputs_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"Saved best_encoder.pt, best_multitask_model.pt, train_metrics.json to {args.outputs_dir}")


if __name__ == "__main__":
    main()
