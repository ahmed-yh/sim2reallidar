"""Generate viz/mbut_playback.html from the parsed scan data.

Reads data/sim/ (must exist first -- run data_prep/parse_recordings.py +
split.py, see README), renders each of the 600 scans as a small combined
range+class image (downsampled to 512px wide, 32-color adaptive palette --
cuts the base64 payload roughly in half vs. truecolor PNG with no visible
quality loss for this purpose), and injects the resulting frame data into
player_template.html (the static HTML/CSS/JS player shell) to produce the
final self-contained mbut_playback.html.

    python3 viz/gen_playback.py

Re-run this whenever data/sim/ changes (e.g. a new/different scenario).
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import cm
from PIL import Image

HERE = Path(__file__).resolve().parent
SIM_DIR = HERE.parent / "data" / "sim"

OUT_W = 512  # downsampled from the native 1024 columns
RANGE_CLIP_M = 20.0  # clip for visual contrast on near-field content

CLASS_COLORS = {
    0: (17, 21, 26),      # environment / no-return
    1: (230, 57, 100),    # human (unused in this scenario, kept for completeness)
    2: (67, 99, 216),     # car
    3: (200, 90, 220),    # bus
    4: (242, 166, 90),    # sphere
    5: (79, 209, 197),    # cylinder
    6: (240, 214, 90),    # box2
    7: (235, 110, 90),    # box1
}


def render_frame(range_img: np.ndarray, class_img: np.ndarray) -> str:
    r_disp = np.where(np.isfinite(range_img), range_img, RANGE_CLIP_M)
    r_norm = np.clip(r_disp, 0, RANGE_CLIP_M) / RANGE_CLIP_M
    range_rgb = (cm.viridis(r_norm)[:, :, :3] * 255).astype(np.uint8)

    class_rgb = np.zeros((*class_img.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        class_rgb[class_img == cls] = color

    gap = np.full((3, class_img.shape[1], 3), 10, dtype=np.uint8)
    combined = np.concatenate([range_rgb, gap, class_rgb], axis=0)

    img = Image.fromarray(combined).resize((OUT_W, combined.shape[0]), Image.BILINEAR)
    img = img.convert("P", palette=Image.ADAPTIVE, colors=32)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    poses = pd.read_csv(SIM_DIR / "poses.csv").sort_values("time_ns").reset_index(drop=True)

    frames_b64 = []
    meta = []
    t0 = poses["time_ns"].iloc[0]
    prev_x, prev_y = poses["x"].iloc[0], poses["y"].iloc[0]
    cum_dist = 0.0

    for idx, row in poses.iterrows():
        d = np.load(SIM_DIR / "scans" / f"{row['sample_id']}.npz")
        frames_b64.append(render_frame(d["range"], d["class_"]))

        cum_dist += float(np.hypot(row["x"] - prev_x, row["y"] - prev_y))
        prev_x, prev_y = row["x"], row["y"]

        u, cnt = np.unique(d["class_"], return_counts=True)
        meta.append({
            "t": (row["time_ns"] - t0) / 1e9,
            "x": row["x"], "y": row["y"],
            "dist": cum_dist,
            "class_counts": {int(k): int(v) for k, v in zip(u.tolist(), cnt.tolist())},
        })
        if (idx + 1) % 100 == 0:
            print(f"  rendered {idx + 1}/{len(poses)}")

    frames_json = json.dumps({"frames": frames_b64, "meta": meta})

    template = (HERE / "player_template.html").read_text()
    final = template.replace("__FRAMES_JSON__", frames_json)
    out_path = HERE / "mbut_playback.html"
    out_path.write_text(final)

    print(f"wrote {out_path} ({len(final)/1e6:.2f} MB, {len(frames_b64)} frames)")


if __name__ == "__main__":
    main()
