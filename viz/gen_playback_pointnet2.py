"""Generate viz/pointnet2_playback.html: run the trained PointNet++ model on
every frame of one real recorded scenario, in time order, so reconstruction
and classification quality can be watched over a continuous drive instead of
a handful of hand-picked static frames (see visualize_range_heatmap.py /
visualize_reconstruction.py for those).

Two different techniques recover a comparable image per frame, because the
two model outputs relate to the input differently:
  - Reconstruction (recon_points) is a free-form 2,048-point cloud with no
    grid identity of its own -- nearest-neighbor reprojection onto the real
    ray grid, same method as visualize_range_heatmap.py.
  - Classification (class_logits) IS tied 1:1 to the model's own input
    points -- so instead of approximating with another nearest-neighbor
    search, this tracks each input point's origin cell through dataset.py's
    exact decimation/finite-filter/pad pipeline and scatters predictions
    back into a grid directly. Exact, not approximate.

All the reprojection/scatter math below stays in the RAW (ring 0 = row 0)
orientation throughout -- viz/playback_common.orient() is applied only at
the very end, once per grid, right before it becomes a displayed image, so
there's no risk of the display-only flip ever leaking into the numeric
pipeline (the NN search and the grid-index scatter both depend on
consistent indexing between xyz_full/valid/recon and grid_idx/pred_cls).

    python3 viz/gen_playback_pointnet2.py

Defaults to scenario 122e86f0-8b3f-46f5-8eaa-9f6b1232c7df, using EVERY raw
scan for that scenario (~600), not just the ~100/scenario subsample
data/sim/{train,val,test}_ids.npy was built from -- see
playback_common.list_scenario_frames for why that's safe to do (none of the
extra frames were ever part of training or eval, so there's nothing to
contaminate) and gives a much denser, smoother real-time result. Each
frame's telemetry panel reports whether it happened to fall in train/val/
test/unused. Same scenario already used for the static cylinder example in
outputs/pointnet2/{range_heatmaps,recon_viz}/.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import COL_STRIDE, N_COLS, N_POINTS, N_RINGS, RING_STRIDE  # noqa: E402
from models.pointnet2 import PointNet2MultiTask  # noqa: E402
from playback_common import (class_rgb, encode_frame, even_subsample, label_row,  # noqa: E402
                              list_scenario_frames, load_native_grids, orient, patch_template, range_rgb)

DEC_COLS = N_COLS // COL_STRIDE  # 512


def load_raw_with_grid_idx(npz_path: Path):
    """Same decimation as dataset.py's _load_raw, but also returns which
    decimated grid cell (flat index into 64*512) each of the N_POINTS output
    points came from, plus the full native (64,1024,3) xyz grid."""
    d = np.load(npz_path)
    x, y, z, intensity = d["x"], d["y"], d["z"], d["intensity"]
    xyz_full = np.stack([x, y, z], axis=-1).reshape(N_RINGS, N_COLS, 3)
    cls_full = np.rint(intensity).astype(np.int64).reshape(N_RINGS, N_COLS)

    grid_idx_full = np.arange(N_RINGS * DEC_COLS).reshape(N_RINGS, DEC_COLS)

    xyz = xyz_full[::RING_STRIDE, ::COL_STRIDE].reshape(-1, 3)
    cls = cls_full[::RING_STRIDE, ::COL_STRIDE].reshape(-1)
    grid_idx = grid_idx_full.reshape(-1)

    finite = np.isfinite(xyz).all(axis=-1)
    xyz, cls, grid_idx = xyz[finite], cls[finite], grid_idx[finite]

    n_missing = N_POINTS - xyz.shape[0]
    if n_missing > 0:
        seed = zlib.crc32(npz_path.name.encode("utf-8"))
        pad_idx = np.random.default_rng(seed).choice(xyz.shape[0], size=n_missing, replace=True)
        xyz = np.concatenate([xyz, xyz[pad_idx]], axis=0)
        cls = np.concatenate([cls, cls[pad_idx]], axis=0)
        grid_idx = np.concatenate([grid_idx, grid_idx[pad_idx]], axis=0)

    return xyz.astype(np.float32), cls, grid_idx, xyz_full


def render_pointnet2_frame(model, device, npz_path, max_range) -> dict:
    orig_range_grid, orig_class_grid, _ = load_native_grids(npz_path)
    xyz_in, cls_in, grid_idx, xyz_full = load_raw_with_grid_idx(npz_path)

    xyz_in_t = torch.from_numpy(xyz_in).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(xyz_in_t)
    recon = out["recon_points"][0]
    pred_cls = out["class_logits"][0].argmax(dim=-1).cpu().numpy()

    range_native = np.linalg.norm(xyz_full, axis=-1)
    valid = np.isfinite(range_native)
    xyz_valid = torch.from_numpy(xyz_full[valid]).float().to(device)
    dists = torch.cdist(xyz_valid, recon)
    nn_range = recon[dists.argmin(dim=1)].norm(dim=-1).cpu().numpy()
    recon_range_grid = np.full_like(range_native, np.nan)
    recon_range_grid[valid] = nn_range

    pred_grid_dec = np.zeros(N_RINGS * DEC_COLS, dtype=np.uint8)
    pred_grid_dec[grid_idx] = pred_cls
    pred_grid_dec = pred_grid_dec.reshape(N_RINGS, DEC_COLS)
    pred_grid_native = np.repeat(pred_grid_dec, COL_STRIDE, axis=1)

    gap = np.full((3, N_COLS, 3), 10, dtype=np.uint8)
    combined = np.concatenate([
        label_row(range_rgb(orient(orig_range_grid)), "ORIGINAL RANGE"), gap,
        label_row(range_rgb(orient(recon_range_grid)), "RECONSTRUCTED RANGE (reprojected)"), gap,
        label_row(class_rgb(orient(orig_class_grid)), "ORIGINAL CLASS"), gap,
        label_row(class_rgb(orient(pred_grid_native)), "PREDICTED CLASS"),
    ], axis=0)

    real_norm = np.clip(range_native[valid], 0, max_range) / max_range
    pred_norm = np.clip(nn_range, 0, max_range) / max_range
    mse = float(np.mean((real_norm - pred_norm) ** 2))
    acc = float((pred_cls == cls_in).mean())

    u, cnt = np.unique(orig_class_grid, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(u.tolist(), cnt.tolist())}

    return {"image": combined, "mse": mse, "acc": acc, "class_counts": class_counts}


def build_replacements(scenario_id: str, n_frames: int, duration_s: float,
                        distance_m: float, checkpoint_name: str, frames_json: str) -> dict[str, str]:
    short_id = scenario_id[:8]
    return {
        "<title>MBUT Corridor Playback</title>": "<title>PointNet++ Live Inference</title>",
        "LiDAR playback &middot; scenario 6633e5c2":
            f"PointNet++ live inference &middot; scenario {short_id}",
        "MBUT corridor &mdash; what the encoder sees":
            "PointNet++ on a real drive &mdash; reconstruction &amp; classification",
        "600 scans &middot; ~318s &middot; ~121m corridor":
            f"{n_frames} scans &middot; ~{duration_s:.0f}s &middot; ~{distance_m:.0f}m corridor "
            f"&middot; {checkpoint_name}",
        "Range &amp; class channels, as fed to the model":
            "Original vs. reconstructed range &amp; class (top to bottom)",
        "Classes in this frame":
            "Objects in this frame (ground truth)",
        'Scenario 6633e5c2 &middot; sim2real-lidar &middot; class colors: '
        'env dark, sphere amber, cylinder teal, box2 yellow, box1 coral':
            f"Scenario {short_id} &middot; sim2real-lidar &middot; PointNet++ candidate &middot; "
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
            '<span class="v mono" id="tMse">0.0</span></div>\n'
            '        <div class="telemetry-row"><span class="k">Per-point class acc.</span>'
            '<span class="v mono" id="tAcc">0.0%</span></div>',
        "tDist.textContent = m.dist.toFixed(1) + 'm';":
            "tDist.textContent = m.dist.toFixed(1) + 'm';\n"
            "    document.getElementById('tSplit').textContent = m.split;\n"
            "    document.getElementById('tMse').textContent = m.mse.toFixed(4);\n"
            "    document.getElementById('tAcc').textContent = (m.acc * 100).toFixed(1) + '%';",
        "__FRAMES_JSON__": frames_json,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-id", default="122e86f0-8b3f-46f5-8eaa-9f6b1232c7df")
    ap.add_argument("--checkpoint", type=Path,
                     default=REPO_ROOT / "outputs" / "pointnet2_checkpoints" / "epoch39_multitask_model.pt")
    ap.add_argument("--recordings-dir", type=Path,
                     default=REPO_ROOT / "scenario_data" / "mbut_76scenarios" / "o_robot_nav")
    ap.add_argument("--sim-dir", type=Path, default=REPO_ROOT / "data" / "sim")
    ap.add_argument("--kevin-outputs-dir", type=Path, default=REPO_ROOT / "outputs" / "kevin_cnn")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "pointnet2_playback.html")
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

    max_range = 20.0
    stats_path = args.kevin_outputs_dir / "normalization_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            max_range = json.load(f)["max_range"]
    print(f"max_range for quality metric: {max_range:.3f}m")

    model = PointNet2MultiTask().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"loaded checkpoint: {args.checkpoint}")

    all_frames = list_scenario_frames(args.recordings_dir, args.scenario_id)
    if len(all_frames) == 0:
        raise SystemExit(f"no raw scans found for scenario_id={args.scenario_id} under {args.recordings_dir}")
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
        r = render_pointnet2_frame(model, device, fr["npz_path"], max_range)
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
            "acc": r["acc"],
        })
        if (idx + 1) % 50 == 0:
            print(f"  rendered {idx + 1}/{len(frames)}  (mse={r['mse']:.4f} acc={r['acc']*100:.1f}%)")

    duration_s = (frames[-1]["time_ns"] - t0) / 1e9
    print(f"split mix in this playback: {split_counts}")
    print(f"mean recon mse: {np.mean([m['mse'] for m in meta]):.4f}, "
          f"mean class acc: {np.mean([m['acc'] for m in meta]) * 100:.1f}%")

    frames_json = json.dumps({"frames": frames_b64, "meta": meta})
    template = (Path(__file__).parent / "player_template.html").read_text()
    replacements = build_replacements(
        args.scenario_id, len(frames), duration_s, cum_dist, args.checkpoint.name, frames_json,
    )
    html = patch_template(template, replacements)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({len(html) / 1e6:.2f} MB, {len(frames_b64)} frames)")


if __name__ == "__main__":
    main()
