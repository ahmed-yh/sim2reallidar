"""Prepare our own parsed range-grid data for Kevin's CNN autoencoder pipeline
(masterarbeit_kevinfischer/lidar_preprocessing/autoencoder_training), and --
critically -- make it use OUR shared spatial-block split instead of his
own script's random 80/10/10 split, so the comparison to PointNet++ is on
identical held-out data.

Kevin's train_ae.py needs, under --data-dir / --outputs-dir:
  data-dir/<id>.npy               range grid only, (64,1024) float32
  outputs-dir/train_indices.npy   POSITIONAL indices into sorted(data-dir/*.npy)
  outputs-dir/val_indices.npy     (his dataset.py sorts data-dir.glob("*.npy")
  outputs-dir/test_indices.npy     once and indexes into that fixed order)
  outputs-dir/normalization_stats.json   {"max_range": ...}

Normally his normalization_stats.py produces the indices + max_range together
by picking its own random split. We skip that script entirely: reuse our own
data/sim/{train,val,test}_ids.npy for the split, and replicate his own
max-range formula (99th percentile of nonzero range values, TRAIN split only)
so the number is computed the same way, just over our train set instead of
his.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = REPO_ROOT / "data" / "sim"
OUT_DATA_DIR = REPO_ROOT / "data" / "kevin_range_npy"
OUT_META_DIR = REPO_ROOT / "outputs" / "kevin_cnn"
PERCENTILE = 99.0  # matches Kevin's normalization_stats.py default


def convert_scans() -> dict[str, list[str]]:
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    scans_dir = SIM_DIR / "scans"

    splits = {}
    for split in ["train", "val", "test"]:
        ids = np.load(SIM_DIR / f"{split}_ids.npy").tolist()
        splits[split] = ids
        for sample_id in ids:
            out_path = OUT_DATA_DIR / f"{sample_id}.npy"
            if out_path.exists():
                continue
            d = np.load(scans_dir / f"{sample_id}.npz")
            np.save(out_path, d["range"].astype(np.float32))
        print(f"  {split}: {len(ids)} samples converted to .npy")
    return splits


def compute_max_range(train_ids: list[str]) -> float:
    """Same formula as Kevin's normalization_stats.py::compute_max_range,
    over our train split only: pool nonzero range values across all train
    scans, take the 99th percentile. Deliberately not "fixed" to exclude
    inf (no-return) points even though they'd technically pass `arr > 0` --
    replicating his exact method is the point, not improving on it, since
    this needs to be the same methodology as whatever his own dataset used."""
    samples = []
    for sample_id in train_ids:
        arr = np.load(OUT_DATA_DIR / f"{sample_id}.npy").ravel()
        nonzero = arr[arr > 0]
        finite = nonzero[np.isfinite(nonzero)]  # inf would blow up the percentile call itself
        if finite.size > 0:
            samples.append(finite)
    pooled = np.concatenate(samples)
    return float(np.percentile(pooled, PERCENTILE))


def write_indices_and_stats(splits: dict[str, list[str]]) -> None:
    OUT_META_DIR.mkdir(parents=True, exist_ok=True)

    # Must match dataset.py's own file discovery exactly: sorted(data_dir.glob("*.npy"))
    all_files = sorted(OUT_DATA_DIR.glob("*.npy"))
    pos_of_id = {p.stem: i for i, p in enumerate(all_files)}

    for split in ["train", "val", "test"]:
        idx = np.array([pos_of_id[sid] for sid in splits[split]], dtype=np.int64)
        np.save(OUT_META_DIR / f"{split}_indices.npy", idx)
        print(f"  {split}_indices.npy: {len(idx)} positional indices")

    max_range = compute_max_range(splits["train"])
    stats = {
        "max_range": max_range,
        "percentile": PERCENTILE,
        "n_total": len(all_files),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "seed": None,  # not applicable -- split reused from our own spatial-block split, not random
        "source": "reused sim2reallidar/data/sim/{train,val,test}_ids.npy, not Kevin's own random split",
    }
    with open(OUT_META_DIR / "normalization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  max_range: {max_range:.3f}  (computed over train split only, matching his formula)")


def main():
    print("Converting our scans to Kevin's expected .npy format...")
    splits = convert_scans()
    print("\nBuilding positional indices + normalization stats (reusing OUR split)...")
    write_indices_and_stats(splits)
    print(f"\nDone. --data-dir {OUT_DATA_DIR}")
    print(f"      --outputs-dir {OUT_META_DIR}  (train_ae.py also writes checkpoints here)")


if __name__ == "__main__":
    main()
