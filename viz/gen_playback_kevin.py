"""Generate viz/kevin_playback.html: run Kevin's trained CNN autoencoder on
every frame of one real recorded scenario, in time order -- the same
treatment gen_playback_pointnet2.py gives PointNet++, for a direct,
apples-to-apples side-by-side.

Only 2 model-output rows here (original range, reconstructed range), not 4:
Kevin's candidate is a plain autoencoder with no classification head at all
(see kevin_pipeline/fair_comparison.py's docstring) -- there's no "predicted
class" row to draw because that output doesn't exist. A third row (original
class, ground truth) stays for visual context only, same source as always.

No reprojection needed here, unlike PointNet++: Kevin's decoder outputs
directly onto the same (64,1024) grid the input came from, already
sigmoid-scaled to [0,1] -- the same normalized units his own
normalization_stats.json max_range defines, so the reconstruction plots
directly with zero extra math.

    python3 viz/gen_playback_kevin.py

Same default scenario as the PointNet++ playback, using every raw scan for
it (~600, not the ~100/scenario subsample data/sim/ was split from -- see
playback_common.list_scenario_frames) so the two are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# masterarbeit_kevinfischer is a SIBLING checkout of sim2reallidar (both
# live under the same parent directory), not nested inside it -- unlike
# every other cross-reference to Kevin's project in this repo
# (kevin_pipeline/), which only ever touches his DATA/OUTPUT artifacts and
# never needed to locate his actual model code before now.
KEVIN_DEPLOY_DIR = (REPO_ROOT.parent / "masterarbeit_kevinfischer" / "lidar_preprocessing"
                     / "autoencoder_training" / "denkwelt_deploy")
sys.path.insert(0, str(KEVIN_DEPLOY_DIR))

from models.autoencoder import build_autoencoder  # noqa: E402
from playback_common import (class_rgb, encode_frame, even_subsample, label_row,  # noqa: E402
                              list_scenario_frames, load_native_grids, orient, patch_template, range_rgb)

N_RINGS, N_COLS = 64, 1024


def render_kevin_frame(model, device, npz_path, max_range) -> dict:
    orig_range_grid, orig_class_grid, _ = load_native_grids(npz_path)

    valid = np.isfinite(orig_range_grid)
    range_n = np.zeros((N_RINGS, N_COLS), dtype=np.float32)
    range_n[valid] = np.clip(orig_range_grid[valid], 0, max_range) / max_range

    x = torch.from_numpy(range_n).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,64,1024)
    with torch.no_grad():
        x_hat, _z = model(x)
    recon_n = x_hat[0, 0].cpu().numpy()  # already in [0,1], same normalized units

    # Both panels plotted in the SAME normalized [0,1] scale (not re-expanded
    # to meters) -- Kevin's own decoder only ever promises to reconstruct
    # this normalized quantity, not real-world range directly.
    recon_display = np.where(valid, recon_n * max_range, np.nan)
    orig_display = np.where(valid, range_n * max_range, np.nan)

    gap = np.full((3, N_COLS, 3), 10, dtype=np.uint8)
    combined = np.concatenate([
        label_row(range_rgb(orient(orig_display)), "ORIGINAL RANGE"), gap,
        label_row(range_rgb(orient(recon_display)), "RECONSTRUCTED RANGE (native, no reprojection)"), gap,
        label_row(class_rgb(orient(orig_class_grid)), "ORIGINAL CLASS (no classifier in this candidate)"),
    ], axis=0)

    mse = float(np.mean(((range_n - recon_n) ** 2)[valid])) if valid.any() else 0.0

    u, cnt = np.unique(orig_class_grid, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(u.tolist(), cnt.tolist())}

    return {"image": combined, "mse": mse, "class_counts": class_counts}


def build_replacements(scenario_id: str, n_frames: int, duration_s: float,
                        distance_m: float, checkpoint_name: str, frames_json: str) -> dict[str, str]:
    short_id = scenario_id[:8]
    return {
        "<title>MBUT Corridor Playback</title>": "<title>Kevin's CNN Live Inference</title>",
        "LiDAR playback &middot; scenario 6633e5c2":
            f"Kevin's CNN live inference &middot; scenario {short_id}",
        "MBUT corridor &mdash; what the encoder sees":
            "Kevin's CNN autoencoder on a real drive &mdash; reconstruction only",
        "600 scans &middot; ~318s &middot; ~121m corridor":
            f"{n_frames} scans &middot; ~{duration_s:.0f}s &middot; ~{distance_m:.0f}m corridor "
            f"&middot; {checkpoint_name}",
        "Range &amp; class channels, as fed to the model":
            "Original vs. reconstructed range, plus ground-truth class for context",
        "Classes in this frame":
            "Objects in this frame (ground truth -- not predicted, this candidate has no classifier)",
        'Scenario 6633e5c2 &middot; sim2real-lidar &middot; class colors: '
        'env dark, sphere amber, cylinder teal, box2 yellow, box1 coral':
            f"Scenario {short_id} &middot; sim2real-lidar &middot; Kevin's CNN candidate &middot; "
            f"class colors: env dark, sphere amber, cylinder teal, box2 yellow, box1 coral",
        '<span class="channel-tag range">range</span>': "",
        '<span class="channel-tag class">class</span>': "",
        '<div class="channel-divider"></div>': "",
        '<div class="telemetry-row"><span class="k">Distance</span>'
        '<span class="v mono" id="tDist">0.0m</span></div>':
            '<div class="telemetry-row"><span class="k">Distance</span>'
            '<span class="v mono" id="tDist">0.0m</span></div>\n'
            '        <div class="telemetry-row"><span class="k">Split (train/val/test/unused)</span>'
            '<span class="v mono" id="tSplit">?</span></div>\n'
            '        <div class="telemetry-row"><span class="k">Recon MSE (norm.)</span>'
            '<span class="v mono" id="tMse">0.0</span></div>',
        "tDist.textContent = m.dist.toFixed(1) + 'm';":
            "tDist.textContent = m.dist.toFixed(1) + 'm';\n"
            "    document.getElementById('tSplit').textContent = m.split;\n"
            "    document.getElementById('tMse').textContent = m.mse.toFixed(4);",
        "__FRAMES_JSON__": frames_json,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-id", default="122e86f0-8b3f-46f5-8eaa-9f6b1232c7df")
    ap.add_argument("--kevin-outputs-dir", type=Path, default=REPO_ROOT / "outputs" / "kevin_cnn")
    ap.add_argument("--sim-dir", type=Path, default=REPO_ROOT / "data" / "sim")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "kevin_playback.html")
    ap.add_argument("--max-frames", type=int, default=160,
                     help="Evenly subsample down to this many frames if the scenario has more -- "
                          "the Artifact hosting cap (16MB) limits how many upscaled frames fit in "
                          "one self-contained HTML page. Pass 0 to keep every raw scan.")
    ap.add_argument("--upscale", type=int, default=2,
                     help="Nearest-neighbor upscale factor for legibility (see playback_common."
                          "encode_frame). Trades off against --max-frames under the same size cap.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    with open(args.kevin_outputs_dir / "normalization_stats.json") as f:
        max_range = json.load(f)["max_range"]
    print(f"max_range: {max_range:.3f}m")

    with open(KEVIN_DEPLOY_DIR / "outputs" / "best_params.json") as f:
        best = json.load(f)
    params = best["params"] if "params" in best else best
    print(f"model config: {params}")

    model = build_autoencoder(params).to(device)
    model.load_state_dict(torch.load(args.kevin_outputs_dir / "best_autoencoder.pt", map_location=device))
    model.eval()
    print(f"loaded checkpoint: {args.kevin_outputs_dir / 'best_autoencoder.pt'}")

    recordings_dir = REPO_ROOT / "scenario_data" / "mbut_76scenarios" / "o_robot_nav"
    all_frames = list_scenario_frames(recordings_dir, args.scenario_id)
    if len(all_frames) == 0:
        raise SystemExit(f"no raw scans found for scenario_id={args.scenario_id} under {recordings_dir}")
    frames = even_subsample(all_frames, args.max_frames or None)
    print(f"scenario {args.scenario_id}: {len(all_frames)} raw scans available, using {len(frames)} "
          f"(the full recording has far more than the ~100/scenario subsample used for training/eval)")

    train_ids = set(np.load(args.sim_dir / "train_ids.npy").tolist())
    val_ids = set(np.load(args.sim_dir / "val_ids.npy").tolist())
    test_ids = set(np.load(args.sim_dir / "test_ids.npy").tolist())

    def split_of(sid: str) -> str:
        if sid in test_ids:
            return "test"
        if sid in val_ids:
            return "val"
        if sid in train_ids:
            return "train"
        return "unused"  # a real raw scan that simply wasn't in the ~100/scenario subsample

    frames_b64, meta = [], []
    t0 = frames[0]["time_ns"]
    prev_x, prev_y = frames[0]["x"], frames[0]["y"]
    cum_dist = 0.0
    split_counts = {"train": 0, "val": 0, "test": 0, "unused": 0}

    for idx, fr in enumerate(frames):
        sid = fr["sample_id"]
        r = render_kevin_frame(model, device, fr["npz_path"], max_range)
        frames_b64.append(encode_frame(r["image"], upscale=args.upscale))

        cum_dist += float(np.hypot(fr["x"] - prev_x, fr["y"] - prev_y))
        prev_x, prev_y = fr["x"], fr["y"]
        split = split_of(sid)
        split_counts[split] += 1

        meta.append({
            "t": (fr["time_ns"] - t0) / 1e9,
            "x": float(fr["x"]), "y": float(fr["y"]),
            "dist": cum_dist,
            "class_counts": r["class_counts"],
            "split": split,
            "mse": r["mse"],
        })
        if (idx + 1) % 50 == 0:
            print(f"  rendered {idx + 1}/{len(frames)}  (mse={r['mse']:.4f})")

    duration_s = (frames[-1]["time_ns"] - t0) / 1e9
    print(f"split mix in this playback: {split_counts}")
    print(f"mean recon mse: {np.mean([m['mse'] for m in meta]):.4f}")

    frames_json = json.dumps({"frames": frames_b64, "meta": meta})
    template = (Path(__file__).parent / "player_template.html").read_text()
    replacements = build_replacements(
        args.scenario_id, len(frames), duration_s, cum_dist, "best_autoencoder.pt", frames_json,
    )
    html = patch_template(template, replacements)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({len(html) / 1e6:.2f} MB, {len(frames_b64)} frames)")


if __name__ == "__main__":
    main()
