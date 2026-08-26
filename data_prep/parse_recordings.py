"""Parse a TWC scenario recording into (64, 1024) range/class grids.

Input: a scenario's recordings/ dir — pairs of <timestamp>.npz (65536 flat
points: x, y, z, intensity, ring) and <timestamp>.json (pose sidecar).

Output, under --out-dir:
    scans/<timestamp>.npz   'range' (64,1024 float32, inf where no return)
                             'class_' (64,1024 uint8, np.rint(intensity))
                             — 'class_' not 'class': the latter is a Python keyword
                             and can't be used as a savez kwarg name.
    poses.csv                sample_id, time_ns, x, y, z (robot translation)

Class index is the raw rounded laser_retro value (0-7), not remapped, so a
future scenario that includes human/car/bus needs no code change here.

Ring order in the source files is already ring-major (1024 contiguous points
per ring, ring 0 first) — reshape(64, 1024) is a direct, valid rasterization,
confirmed against this scenario's data before writing this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_one(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path)
    x, y, z, intensity = d["x"], d["y"], d["z"], d["intensity"]
    if x.size != 64 * 1024:
        raise ValueError(f"{npz_path}: expected 65536 points, got {x.size}")

    range_img = np.sqrt(x**2 + y**2 + z**2).astype(np.float32).reshape(64, 1024)
    class_img = np.rint(intensity).astype(np.uint8).reshape(64, 1024)
    return range_img, class_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings-dir", type=Path,
                     default=Path("/home/ahmed/twc_scenario_6633e5c2/recordings"))
    ap.add_argument("--out-dir", type=Path,
                     default=Path(__file__).resolve().parents[1] / "data" / "sim")
    args = ap.parse_args()

    scans_dir = args.out_dir / "scans"
    scans_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(args.recordings_dir.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No .npz files found in {args.recordings_dir}")
    print(f"Found {len(npz_files)} recordings in {args.recordings_dir}")

    rows = []
    class_totals: dict[int, int] = {}
    for npz_path in npz_files:
        sample_id = npz_path.stem
        json_path = npz_path.with_suffix(".json")
        with open(json_path) as f:
            pose = json.load(f)

        range_img, class_img = parse_one(npz_path)
        np.savez_compressed(scans_dir / f"{sample_id}.npz",
                             range=range_img, class_=class_img)

        u, c = np.unique(class_img, return_counts=True)
        for uu, cc in zip(u.tolist(), c.tolist()):
            class_totals[uu] = class_totals.get(uu, 0) + cc

        rows.append({
            "sample_id": sample_id,
            "time_ns": pose["time_ns"],
            "x": pose["robot_translation_x"],
            "y": pose["robot_translation_y"],
            "z": pose["robot_translation_z"],
        })

    poses = pd.DataFrame(rows).sort_values("time_ns").reset_index(drop=True)
    poses.to_csv(args.out_dir / "poses.csv", index=False)
    print(f"Wrote {len(poses)} scans to {scans_dir}")
    print(f"Wrote pose table to {args.out_dir / 'poses.csv'}")

    total = sum(class_totals.values())
    print("\nClass totals (sanity check):")
    for k in sorted(class_totals):
        print(f"  class {k}: {class_totals[k]:>9}  ({100*class_totals[k]/total:.5f}%)")


if __name__ == "__main__":
    main()
