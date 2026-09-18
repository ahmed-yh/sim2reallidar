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

from sensor import N_COLS, N_RINGS  # noqa: F401  -- re-exported; viz scripts import both from here

CLASS_COLORS = {
    0: (17, 21, 26), 1: (230, 57, 100), 2: (67, 99, 216), 3: (200, 90, 220),
    4: (242, 166, 90), 5: (79, 209, 197), 6: (240, 214, 90), 7: (235, 110, 90),
}
RANGE_CLIP_M = 20.0


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


# Display metadata for the scalars a render_fn can report, so both the legacy
# HTML player and the demo page describe the same field the same way. Order is
# the order rows appear in the telemetry panel.
TELEMETRY_SPECS = {
    "t":     {"label": "Time",     "format": "time"},
    "pos":   {"label": "Position", "format": "xy", "from": ["x", "y"], "unit": "m"},
    "dist":  {"label": "Distance", "format": "meters", "digits": 1},
    "split": {"label": "Split (train/val/test/unused)", "format": "text"},
    "mse":   {"label": "Recon MSE (norm.)",    "format": "fixed",   "digits": 4},
    "acc":   {"label": "Per-point class acc.", "format": "percent", "digits": 1},
}
TELEMETRY_ORDER = ["t", "pos", "dist", "split", "mse", "acc"]

# Consumed by the legend, not the telemetry rows.
_COUNT_KEYS = ("class_counts", "pred_class_counts")


def derive_telemetry_fields(meta: list[dict]) -> list[dict]:
    """Describe only the telemetry a given set of frames can actually support.

    A field is emitted iff EVERY key it reads is present in EVERY meta entry.
    That is what makes the real_bags/*.html failure mode unrepresentable rather
    than merely fixed: those pages named m.x and m.class_counts while the
    exporter emitted only {"t", "mse"}, so render(0) threw before play() ever
    ran and autoplay, telemetry and the legend were all silently dead.

    Keys with no entry in TELEMETRY_SPECS still get a generic descriptor, so a
    render_fn that starts reporting a new scalar shows up in the panel with no
    frontend change at all.
    """
    if not meta:
        return []
    common = set(meta[0])
    for m in meta[1:]:
        common &= set(m)

    fields = []
    for key in TELEMETRY_ORDER:
        spec = TELEMETRY_SPECS[key]
        needed = spec.get("from", [key])
        if all(k in common for k in needed):
            fields.append({"key": key, **spec})

    described = {k for key in TELEMETRY_ORDER
                 for k in TELEMETRY_SPECS[key].get("from", [key])}
    described |= set(TELEMETRY_ORDER)
    for key in sorted(common - described - set(_COUNT_KEYS)):
        if isinstance(meta[0][key], (int, float, str)):
            fields.append({"key": key, "label": key.replace("_", " ").capitalize(),
                           "format": "auto"})
    return fields


def derive_legend(meta: list[dict], class_ids=(4, 5, 6, 7), caption=None) -> dict | None:
    """Legend descriptor, or None when no frame carries per-class counts --
    in which case the player renders no legend block at all rather than an
    empty one."""
    if not meta:
        return None
    for key in _COUNT_KEYS:
        if all(key in m for m in meta):
            return {"source": key, "classIds": list(class_ids),
                    "caption": caption or "Classes in this frame"}
    return None


def build_playback_payload(*, scenario_id: str, recordings_dir: Path, sim_dir: Path,
                            max_frames: int | None, upscale: int, render_fn,
                            frame_sink=None) -> dict:
    """Frame selection, per-frame render, odometry distance, split attribution
    and metric aggregation -- everything run_playback does EXCEPT producing
    HTML. Returns {"frames": [...], "meta": [...], "stats": {...}}.

    frame_sink: optional callable(index, combined_ndarray) -> str. When given,
    it is called instead of encode_frame() and its return value is collected in
    place of a base64 string -- the single hook that lets the demo write frames
    as individual PNG files without a second copy of this loop. When None the
    behaviour is byte-identical to before this split.

    render_fn(npz_path) -> dict with "image" (the frame to encode) plus any
    scalars to expose as telemetry. Every other key is copied into that frame's
    metadata verbatim, so a candidate with an extra metric (PointNet++ reports
    classification accuracy; Kevin's CNN has no classifier and reports none)
    needs no special case here.
    """
    all_frames = list_scenario_frames(recordings_dir, scenario_id)
    if not all_frames:
        raise SystemExit(f"no raw scans found for scenario_id={scenario_id} under {recordings_dir}")
    frames = even_subsample(all_frames, max_frames)
    print(f"scenario {scenario_id}: {len(all_frames)} raw scans available, using {len(frames)} "
          f"(the full recording has far more than the ~100/scenario subsample used for training/eval)")

    splits = {s: set(np.load(sim_dir / f"{s}_ids.npy").tolist()) for s in ("train", "val", "test")}

    def split_of(sid: str) -> str:
        for name in ("test", "val", "train"):
            if sid in splits[name]:
                return name
        return "unused"  # a real raw scan that simply wasn't in the ~100/scenario subsample

    out_frames, meta = [], []
    t0 = frames[0]["time_ns"]
    prev_x, prev_y = frames[0]["x"], frames[0]["y"]
    cum_dist = 0.0
    split_counts = {"train": 0, "val": 0, "test": 0, "unused": 0}

    for idx, fr in enumerate(frames):
        rendered = render_fn(fr["npz_path"])
        if frame_sink is None:
            out_frames.append(encode_frame(rendered["image"], upscale=upscale))
        else:
            out_frames.append(frame_sink(idx, rendered["image"]))

        cum_dist += float(np.hypot(fr["x"] - prev_x, fr["y"] - prev_y))
        prev_x, prev_y = fr["x"], fr["y"]
        split = split_of(fr["sample_id"])
        split_counts[split] += 1

        entry = {"t": (fr["time_ns"] - t0) / 1e9, "x": float(fr["x"]), "y": float(fr["y"]),
                 "dist": cum_dist, "split": split}
        entry.update({k: v for k, v in rendered.items() if k != "image"})
        meta.append(entry)

        if (idx + 1) % 50 == 0:
            scalars = " ".join(f"{k}={v:.4f}" for k, v in sorted(entry.items())
                                if isinstance(v, float) and k not in ("t", "x", "y", "dist"))
            print(f"  rendered {idx + 1}/{len(frames)}  ({scalars})")

    duration_s = (frames[-1]["time_ns"] - t0) / 1e9
    print(f"split mix in this playback: {split_counts}")
    means = {}
    for key in sorted({k for m in meta for k, v in m.items()
                       if isinstance(v, float) and k not in ("t", "x", "y", "dist")}):
        means[key] = float(np.mean([m[key] for m in meta]))
        print(f"mean {key}: {means[key]:.4f}")

    return {
        "frames": out_frames,
        "meta": meta,
        "stats": {
            "n_frames": len(frames), "n_raw_available": len(all_frames),
            "duration_s": duration_s, "distance_m": cum_dist,
            "split_counts": split_counts, "means": means,
        },
    }


def run_playback(*, scenario_id: str, recordings_dir: Path, sim_dir: Path, out_path: Path,
                  max_frames: int | None, upscale: int, checkpoint_name: str,
                  render_fn, build_replacements) -> None:
    """Build a playback payload and write it into the standalone HTML player.

    Kept for the viz/gen_playback_*.py scripts. The demo (demo/build_demo.py)
    calls build_playback_payload directly and emits JSON instead.
    """
    payload = build_playback_payload(
        scenario_id=scenario_id, recordings_dir=recordings_dir, sim_dir=sim_dir,
        max_frames=max_frames, upscale=upscale, render_fn=render_fn,
    )
    stats = payload["stats"]
    frames_json = json.dumps({"frames": payload["frames"], "meta": payload["meta"]})
    template = (Path(__file__).resolve().parent / "player_template.html").read_text()
    html = patch_template(template, build_replacements(
        scenario_id, stats["n_frames"], stats["duration_s"], stats["distance_m"],
        checkpoint_name, frames_json))
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} ({len(html) / 1e6:.2f} MB, {len(payload['frames'])} frames)")


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
