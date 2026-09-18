"""Generate viz/mbut_playback.html from the parsed scan data.

Reads data/sim/ (must exist first -- run data_prep/parse_recordings.py +
split.py, see README), renders every scan in poses.csv as a small combined
range+class image (downsampled to 512px wide), and injects the resulting
frame data into player_template.html (the static HTML/CSS/JS player shell)
to produce the final self-contained mbut_playback.html.

Unlike gen_playback_pointnet2.py / gen_playback_kevin.py, this one shows
the raw parsed data only -- no model is run, so there's no reconstruction
or predicted-class row, just what the encoder gets fed.

    python3 viz/gen_playback.py

Re-run this whenever data/sim/ changes (e.g. a new/different scenario).
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from playback_common import class_rgb, range_rgb  # noqa: E402

SIM_DIR = HERE.parent / "data" / "sim"

OUT_W = 512  # downsampled from the native 1024 columns


def render_frame(range_img: np.ndarray, class_img: np.ndarray) -> str:
    """Range + class channels stacked, as one base64 PNG.

    Colour mapping comes from playback_common (shared with the model-
    inference playbacks) rather than a second copy of the palette here.
    NEAREST downsampling and truecolor output, both deliberate: the class
    row is discrete colours, and blending them -- whether by a bilinear
    resize or by PIL's adaptive-palette quantization, which this script
    used to do -- silently invents colours that mean nothing, which is the
    same bug playback_common.encode_frame documents avoiding."""
    gap = np.full((3, class_img.shape[1], 3), 10, dtype=np.uint8)
    combined = np.concatenate([range_rgb(range_img), gap, class_rgb(class_img)], axis=0)

    img = Image.fromarray(combined).resize((OUT_W, combined.shape[0]), Image.NEAREST)
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
