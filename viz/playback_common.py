"""Shared rendering helpers for the model-inference playback generators
(gen_playback_pointnet2.py, gen_playback_kevin.py) -- factored out once a
second consumer needed the exact same range/class rendering, frame
encoding, and template-patching logic, rather than duplicated a second time.

Orientation: ring 0 is the lowest, ground-facing beam (verified against real
Z-height data -- see visualize_range_heatmap.py). A raw (64,1024) grid has
ring 0 at row 0, which np.flipud'd puts it at the BOTTOM of the resulting
image -- matching the "lower" origin convention already used everywhere
else in this project's plots (visualize_range_heatmap.py,
visualize_reconstruction.py). The first version of the pointnet2 playback
missed this (it builds raw pixel arrays directly with PIL, not through
matplotlib's `ax.imshow(..., origin=...)`, so there was no origin flag to
get right or wrong -- the array's own row order IS the image's row order).
`orient()` is the single place this correction happens; call it once per
grid, right after it's built, before doing anything else with it.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from matplotlib import cm
from PIL import Image, ImageDraw

CLASS_COLORS = {
    0: (17, 21, 26), 1: (230, 57, 100), 2: (67, 99, 216), 3: (200, 90, 220),
    4: (242, 166, 90), 5: (79, 209, 197), 6: (240, 214, 90), 7: (235, 110, 90),
}
RANGE_CLIP_M = 20.0
N_RINGS = 64
N_COLS = 1024


def orient(grid: np.ndarray) -> np.ndarray:
    """Ring 0 (bottom row after this) is the lowest, ground-facing beam."""
    return np.flipud(grid)


def list_scenario_frames(recordings_dir: Path, scenario_id: str) -> list[dict]:
    """Every raw scan for one scenario, sorted by time -- NOT the
    deliberately-subsampled ~100/scenario set data_prep/parse_recordings.py
    built data/sim/{train,val,test}_ids.npy from (see
    --max-scans-per-scenario there). A demo/QA playback isn't training on
    anything, so there's no reason to limit it to that subsample: this
    scenario alone has ~6x more raw scans sitting unused, and using them
    gives a far denser, smoother real-time result for free. None of these
    extra frames were ever touched by any split -- they're not "test", they
    were never part of the pipeline being compared at all."""
    scenario_dir = Path(recordings_dir) / scenario_id
    frames = []
    for npz_path in sorted(scenario_dir.glob("*.npz")):
        with open(npz_path.with_suffix(".json")) as f:
            pose = json.load(f)
        frames.append({
            "sample_id": f"{scenario_id}__{npz_path.stem}",
            "npz_path": npz_path,
            "time_ns": pose["time_ns"],
            "x": pose["robot_translation_x"],
            "y": pose["robot_translation_y"],
        })
    frames.sort(key=lambda r: r["time_ns"])
    return frames


def load_native_grids(npz_path: Path):
    """range/class grids straight from one raw recording, same formula as
    data_prep/parse_recordings.py's parse_one() -- the single source now
    used for BOTH the "original" display panels and (by the caller) model
    input, since most of the frames list_scenario_frames() returns were
    never processed into data/sim/scans/ in the first place."""
    d = np.load(npz_path)
    x, y, z, intensity = d["x"], d["y"], d["z"], d["intensity"]
    range_grid = np.sqrt(x ** 2 + y ** 2 + z ** 2).astype(np.float32).reshape(N_RINGS, N_COLS)
    class_grid = np.rint(intensity).astype(np.uint8).reshape(N_RINGS, N_COLS)
    xyz_full = np.stack([x, y, z], axis=-1).reshape(N_RINGS, N_COLS, 3)
    return range_grid, class_grid, xyz_full


def range_rgb(range_grid: np.ndarray) -> np.ndarray:
    r_disp = np.where(np.isfinite(range_grid), range_grid, RANGE_CLIP_M)
    r_norm = np.clip(r_disp, 0, RANGE_CLIP_M) / RANGE_CLIP_M
    return (cm.viridis(r_norm)[:, :, :3] * 255).astype(np.uint8)


def class_rgb(class_grid: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*class_grid.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[class_grid == cls] = color
    return rgb


def label_row(img_rgb: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(img_rgb.copy())
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 6 * len(text) + 8, 13], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return np.array(img)


def encode_frame(combined: np.ndarray, upscale: int = 3) -> str:
    # Truecolor, NOT a shared adaptive palette -- see gen_playback_pointnet2.py's
    # original bugfix note: a palette computed jointly across a continuous
    # viridis gradient and a handful of small discrete class-color patches
    # silently remaps the class colors to serve the gradient instead.
    #
    # Upscaled 3x via NEAREST resampling before encoding -- a UNIFORM scale
    # on both axes, unlike an earlier attempt at this that stretched the
    # displayed <img> taller via CSS (object-fit: fill) without touching the
    # source pixels: that made the object blobs easier to see but visibly
    # distorted the baked-in row labels, since non-uniform stretching warps
    # text glyphs but a real, uniform upscale does not (nearest-neighbor
    # keeps every edge crisp -- image-rendering: pixelated in the CSS
    # matches this, no blur is introduced either).
    img = Image.fromarray(combined)
    if upscale != 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def even_subsample(frames: list[dict], max_frames: int | None) -> list[dict]:
    """Same even (not first-N) subsampling as
    data_prep/parse_recordings.py's subsample_per_scenario -- keeps frames
    spanning the whole drive rather than clustering at its start, needed
    because a full scenario's raw scan count doesn't fit under the
    Artifact platform's 16MB page-size cap once frames are upscaled for
    legibility (see encode_frame)."""
    if max_frames is None or len(frames) <= max_frames:
        return frames
    idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
    return [frames[i] for i in sorted(set(idx.tolist()))]


def patch_template(template: str, replacements: dict[str, str]) -> str:
    """Applies each (old, new) pair via a plain substring replace, erroring
    loudly if a target isn't found (template drifted) rather than silently
    no-op'ing -- these targets are short, unique inner-text/inner-code
    substrings, not whitespace-sensitive whole lines, so a genuine template
    edit is the only thing that should ever break this."""
    for old, new in replacements.items():
        if old not in template:
            raise ValueError(f"template patch target not found (template drifted?): {old[:80]!r}")
        template = template.replace(old, new)
    return template
