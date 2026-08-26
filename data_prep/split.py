"""Spatial-block train/val/test split for the single MBUT recording.

600 samples, ~2 Hz, one continuous ~121 m drive through the corridor (see
poses.csv). A random frame-level split would leak: consecutive samples are
~0.2 m apart, essentially duplicates. Instead, chop the route into
contiguous blocks and round-robin the blocks across train/val/test, so each
split sees scattered stretches of the corridor rather than one end of it,
while no block itself is split (no leakage across a block boundary within
15s of driving).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BLOCK_SIZE = 20  # ~10s of driving, ~2-4m of travel per block
PATTERN = ["train"] * 14 + ["val"] * 3 + ["test"] * 3  # 70/15/15 per 20-block cycle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", type=Path,
                     default=Path(__file__).resolve().parents[1] / "data" / "sim")
    args = ap.parse_args()

    poses = pd.read_csv(args.sim_dir / "poses.csv").sort_values("time_ns").reset_index(drop=True)
    n = len(poses)
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    split_of_sample = {}
    for block_i in range(n_blocks):
        split = PATTERN[block_i % len(PATTERN)]
        start = block_i * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, n)
        for sample_id in poses["sample_id"].iloc[start:end]:
            split_of_sample[sample_id] = split

    for split in ["train", "val", "test"]:
        ids = [sid for sid, s in split_of_sample.items() if s == split]
        np.save(args.sim_dir / f"{split}_ids.npy", np.array(ids))
        print(f"{split}: {len(ids)} samples")

    # sanity: verify each of the 4 present object classes survives in every split
    scans_dir = args.sim_dir / "scans"
    print("\nPer-split class-presence check (frames containing each class):")
    for split in ["train", "val", "test"]:
        ids = np.load(args.sim_dir / f"{split}_ids.npy")
        class_frame_counts = {c: 0 for c in [4, 5, 6, 7]}
        for sid in ids:
            d = np.load(scans_dir / f"{sid}.npz")
            present = set(np.unique(d["class_"]).tolist())
            for c in class_frame_counts:
                if c in present:
                    class_frame_counts[c] += 1
        print(f"  {split} (n={len(ids)}): {class_frame_counts}")


if __name__ == "__main__":
    main()
