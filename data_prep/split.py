"""Spatial-block train/val/test split, per scenario.

Each scenario is its own continuous drive (see poses.csv, ~2 Hz). A random
frame-level split would leak: consecutive samples are ~0.2 m apart,
essentially duplicates. Instead, chop each scenario's own route into
contiguous blocks and round-robin the blocks across train/val/test, so each
split sees scattered stretches of every scenario's corridor rather than one
end of it, while no block itself is split (no leakage across a block
boundary within 15s of driving).

Done per scenario, not once across the whole combined pose table: with
multiple scenarios, sorting all samples by a shared time_ns and chopping
globally would mix unrelated drives into the same "contiguous" block and
route entire scenarios to a single split by chance. poses.csv's
scenario_id column (added by parse_recordings.py; "" for the original
single-scenario layout, which is just one group of one) is the group key.

The 20-block PATTERN cycle (14 train / 3 val / 3 test) index is NOT reset
per scenario -- it's a running counter across the whole loop. Confirmed the
hard way: with --max-scans-per-scenario capping each scenario to ~100
samples (~5 blocks of 20), a per-scenario-reset index never gets past
block 4, so every block lands in PATTERN's first 5 (all "train") entries
and val/test end up empty. A shared, continuing counter still keeps every
individual block's contents within one scenario (no leakage), it just
doesn't require any single scenario to have a full 20-block cycle's worth
of samples on its own for val/test to appear at all.
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

    poses = pd.read_csv(args.sim_dir / "poses.csv", dtype={"scenario_id": str}).fillna({"scenario_id": ""})

    split_of_sample = {}
    pattern_i = 0  # continues across scenarios -- see module docstring
    for scenario_id, group in poses.groupby("scenario_id", sort=True):
        group = group.sort_values("time_ns").reset_index(drop=True)
        n = len(group)
        n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
        for block_i in range(n_blocks):
            split = PATTERN[pattern_i % len(PATTERN)]
            pattern_i += 1
            start = block_i * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, n)
            for sample_id in group["sample_id"].iloc[start:end]:
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
