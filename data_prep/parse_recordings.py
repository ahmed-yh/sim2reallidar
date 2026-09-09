"""Parse TWC scenario recording(s) into (64, 1024) range/class grids.

Two input layouts, auto-detected from --recordings-dir:

  single-scenario: --recordings-dir itself holds <timestamp>.npz/.json pairs
                    (the original twc_scenario_6633e5c2 layout).
                    sample_id = timestamp, e.g. "2026_08_25_17_13_30_968".

  multi-scenario:   --recordings-dir holds one subdirectory per scenario
                     (uuid-named), each holding its own <timestamp>.npz/.json
                     pairs (the o_robot_nav/<scenario_id>/ layout, see
                     scenario_data/mbut_76scenarios/README.md). Real
                     wall-clock timestamps collide across different scenario
                     runs often enough to matter (19 collisions confirmed
                     across 47,958 files in the 76-scenario batch) — a flat
                     sample_id would silently drop scans on write. So here
                     sample_id = "{scenario_id}__{timestamp}", and every
                     downstream file (dataset.py, train.py, split.py) that
                     resolves a sample_id back to a raw file must split on
                     "__" to recover (scenario_id, timestamp).

Output, under --out-dir:
    scans/<sample_id>.npz    'range' (64,1024 float32, inf where no return)
                              'class_' (64,1024 uint8, np.rint(intensity))
                              — 'class_' not 'class': the latter is a Python
                              keyword and can't be used as a savez kwarg name.
    poses.csv                 sample_id, scenario_id, time_ns, x, y, z
                               (robot translation; scenario_id is "" for the
                               single-scenario layout)

Class index is the raw rounded laser_retro value (0-7), not remapped.

Ring order in the source files is already ring-major (1024 contiguous points
per ring, ring 0 first) — reshape(64, 1024) is a direct, valid rasterization,
confirmed against real data before writing this script.
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


def discover_files(recordings_dir: Path) -> list[tuple[str, str, Path]]:
    """-> list of (scenario_id, sample_id, npz_path). scenario_id is ""
    for the single-scenario layout."""
    direct = sorted(recordings_dir.glob("*.npz"))
    if direct:
        return [("", p.stem, p) for p in direct]

    out = []
    for scenario_dir in sorted(p for p in recordings_dir.iterdir() if p.is_dir()):
        for npz_path in sorted(scenario_dir.glob("*.npz")):
            out.append((scenario_dir.name, f"{scenario_dir.name}__{npz_path.stem}", npz_path))
    return out


def subsample_per_scenario(files: list[tuple[str, str, Path]], max_per_scenario: int | None
                            ) -> list[tuple[str, str, Path]]:
    """Even (not just first-N) subsample within each scenario, by filename
    (== timestamp) order, so the kept scans still span the whole drive
    rather than clustering at its start."""
    if max_per_scenario is None:
        return files
    by_scenario: dict[str, list[tuple[str, str, Path]]] = {}
    for row in files:
        by_scenario.setdefault(row[0], []).append(row)

    kept = []
    for scenario_id, rows in by_scenario.items():
        rows.sort(key=lambda r: r[2].name)
        if len(rows) <= max_per_scenario:
            kept.extend(rows)
        else:
            idx = np.linspace(0, len(rows) - 1, max_per_scenario).round().astype(int)
            kept.extend(rows[i] for i in sorted(set(idx.tolist())))
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings-dir", type=Path,
                     default=Path(__file__).resolve().parents[1] / "scenario_data" / "mbut_76scenarios" / "o_robot_nav")
    ap.add_argument("--out-dir", type=Path,
                     default=Path(__file__).resolve().parents[1] / "data" / "sim")
    ap.add_argument("--max-scans-per-scenario", type=int, default=None,
                     help="Cap scans kept per scenario (evenly subsampled across the drive), "
                          "to bound total dataset size / epoch time. Default: keep everything.")
    args = ap.parse_args()

    scans_dir = args.out_dir / "scans"
    if scans_dir.exists():
        # A rerun with a different --max-scans-per-scenario (or a different
        # --recordings-dir) doesn't overwrite its way to a clean state:
        # sample_ids not in THIS run's selection just sit there as stale
        # leftovers from whatever the last run wrote (confirmed the hard
        # way -- a 47,958-scan run followed by a 100/scenario=7,600-scan
        # rerun left 48,558 files, not 7,600). poses.csv/split.py only ever
        # look up ids this run actually selected, so it's not a correctness
        # bug, just silent disk clutter that looks like a bug later.
        import shutil
        shutil.rmtree(scans_dir)
    scans_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(args.recordings_dir)
    if not files:
        raise RuntimeError(f"No .npz files found under {args.recordings_dir}")
    n_scenarios = len({s for s, _, _ in files}) or 1
    print(f"Found {len(files)} recordings across {n_scenarios} scenario(s) in {args.recordings_dir}")

    files = subsample_per_scenario(files, args.max_scans_per_scenario)
    if args.max_scans_per_scenario is not None:
        print(f"Subsampled to {len(files)} recordings (max {args.max_scans_per_scenario}/scenario)")

    rows = []
    class_totals: dict[int, int] = {}
    for scenario_id, sample_id, npz_path in files:
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
            "scenario_id": scenario_id,
            "time_ns": pose["time_ns"],
            "x": pose["robot_translation_x"],
            "y": pose["robot_translation_y"],
            "z": pose["robot_translation_z"],
        })

    poses = pd.DataFrame(rows).sort_values(["scenario_id", "time_ns"]).reset_index(drop=True)
    poses.to_csv(args.out_dir / "poses.csv", index=False)
    print(f"Wrote {len(poses)} scans to {scans_dir}")
    print(f"Wrote pose table to {args.out_dir / 'poses.csv'}")

    total = sum(class_totals.values())
    print("\nClass totals (sanity check):")
    for k in sorted(class_totals):
        print(f"  class {k}: {class_totals[k]:>9}  ({100*class_totals[k]/total:.5f}%)")


if __name__ == "__main__":
    main()
