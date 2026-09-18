"""Export the demo's payloads from the project's own artifacts.

    python demo/build_demo.py --check          # validate, needs nothing but stdlib
    python demo/build_demo.py --all            # export everything available
    python demo/build_demo.py --only sim_player:pointnet2_sim
    python demo/build_demo.py --report         # per-file sizes + budget check

Design rules worth keeping:

* **No top-level torch import.** Every heavy import happens inside the exporter
  that needs it, so --check and --report run on a laptop with nothing but the
  standard library plus Pillow.
* **Missing inputs skip, they don't fail.** Checkpoints, scenario_data/ and the
  real bags are all gitignored, and Kevin's model lives in a separate sibling
  checkout. A fresh clone must still be able to build the parts it has and tell
  you plainly what it couldn't.
* **Render functions are imported, never copied** -- viz/gen_playback_*.py own
  the per-candidate frame rendering, and this file only arranges the output.
"""
from __future__ import annotations

import argparse
import base64
import importlib
import json
import shutil
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
PAYLOADS = DEMO_DIR / "payloads"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "viz"))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class Skip(Exception):
    """Raised by an exporter when a required input is absent. Reported at the
    end as a skip, not a failure -- see the module docstring."""


def load_sources() -> dict:
    with open(DEMO_DIR / "sources.json", encoding="utf-8") as f:
        return json.load(f)


def need(path: Path, what: str) -> Path:
    if not path.exists():
        raise Skip(f"{what} not found: {path}")
    return path


def write_payload(rel_path: str, kind: str, payload_id: str, payload: dict) -> Path:
    """Write a payload as a classic script calling DEMO.register().

    Not plain .json on purpose: the demo has to work when opened as a file://
    folder, where fetch() is blocked by the opaque origin. A <script> tag is
    not, so every payload is JS that registers itself.
    """
    out = PAYLOADS / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    out.write_text(
        f'DEMO.register({json.dumps(kind)}, {json.dumps(payload_id)}, {body});\n',
        encoding="utf-8",
    )
    return out


def resolve_renderer(spec: str):
    """'module:function' -> the callable, imported only when actually needed."""
    mod_name, fn_name = spec.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def png_frame_sink(out_dir: Path, upscale: int):
    """Writes each frame as its own PNG and returns its index.

    Individual files rather than base64 inlined in the payload: base64 costs
    ~33% more bytes, has to be parsed as one enormous string before anything
    renders, and keeps every frame resident. Files stream, cache, and let the
    player preload a rolling window.
    """
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    def sink(index: int, combined) -> int:
        img = Image.fromarray(combined)
        if upscale != 1:
            img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
        img.save(out_dir / f"{index:04d}.png", format="PNG", optimize=True)
        return index

    return sink


# --------------------------------------------------------------------------
# exporters
# --------------------------------------------------------------------------

def export_sim_player(cfg: dict) -> str:
    """One simulated-scenario playback, rendered by the candidate's own
    render_*_frame function from viz/."""
    import torch  # noqa: F401  (imported here, never at module level)
    from playback_common import (build_playback_payload, derive_legend,
                                 derive_telemetry_fields)

    recordings = REPO_ROOT / "scenario_data" / "mbut_76scenarios" / "o_robot_nav"
    sim_dir = REPO_ROOT / "data" / "sim"
    ckpt = REPO_ROOT / cfg["checkpoint"]
    need(recordings, "scenario_data recordings"); need(sim_dir, "data/sim splits")
    need(ckpt, f"checkpoint for {cfg['id']}")

    render_fn = build_sim_render_fn(cfg, ckpt)

    frames_dir = PAYLOADS / "players" / "frames" / cfg["id"]
    sink = png_frame_sink(frames_dir, cfg.get("upscale", 1)) if cfg.get("frameMode") == "files" else None

    built = build_playback_payload(
        scenario_id=cfg["scenarioId"], recordings_dir=recordings, sim_dir=sim_dir,
        max_frames=cfg.get("maxFrames"), upscale=cfg.get("upscale", 1),
        render_fn=render_fn, frame_sink=sink,
    )
    meta, stats = built["meta"], built["stats"]

    payload = {
        "schemaVersion": 1,
        "id": cfg["id"], "kind": "sim",
        "title": cfg.get("title", cfg["id"]),
        "subtitle": cfg.get("subtitle", ""),
        "source": {
            "model": cfg["model"], "checkpoint": Path(cfg["checkpoint"]).name,
            "scenarioId": cfg["scenarioId"], "bag": None,
            "nFrames": stats["n_frames"], "nRawAvailable": stats["n_raw_available"],
            "durationS": round(stats["duration_s"], 1),
            "distanceM": round(stats["distance_m"], 1),
            "splitCounts": stats["split_counts"],
        },
        "rows": cfg.get("rows", []),
        "telemetryFields": derive_telemetry_fields(meta),
        "legend": derive_legend(meta, caption=cfg.get("legendCaption")),
        "frames": frames_descriptor(cfg, built["frames"]),
        "meta": meta,
        "stats": stats["means"],
    }
    out = write_payload(f"players/{cfg['id']}.js", "player", cfg["id"], payload)
    return f"{cfg['id']}: {stats['n_frames']} frames -> {out.relative_to(DEMO_DIR)}"


def build_sim_render_fn(cfg: dict, ckpt: Path):
    """Load the candidate's model and bind it into its render_*_frame."""
    import json as _json
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render_raw = resolve_renderer(cfg["renderer"])

    stats_path = REPO_ROOT / "outputs" / "kevin_cnn" / "normalization_stats.json"
    max_range = 20.0
    if stats_path.exists():
        with open(stats_path) as f:
            max_range = _json.load(f)["max_range"]

    if cfg["model"] == "pointnet2":
        from models.pointnet2 import PointNet2MultiTask
        model = PointNet2MultiTask().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
    elif cfg["model"] == "kevin_cnn":
        # Kevin's model lives in a sibling checkout that may simply not be
        # present; skipping is the right behaviour, not an ImportError.
        kevin_deploy = (REPO_ROOT.parent / "masterarbeit_kevinfischer" / "lidar_preprocessing"
                        / "autoencoder_training" / "denkwelt_deploy")
        need(kevin_deploy, "Kevin's denkwelt_deploy checkout")
        from gen_playback_kevin import build_autoencoder
        with open(kevin_deploy / "outputs" / "best_params.json") as f:
            best = _json.load(f)
        model = build_autoencoder(best.get("params", best)).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        raise Skip(f"no sim renderer wired for model {cfg['model']!r}")

    model.eval()
    return lambda npz_path: render_raw(model, device, npz_path, max_range)


def frames_descriptor(cfg: dict, frames) -> dict:
    if cfg.get("frameMode") == "files":
        return {"mode": "files", "format": "png",
                "base": f"players/frames/{cfg['id']}", "pad": 4,
                "count": len(frames), "preload": 10}
    return {"mode": "inline", "format": "png", "data": frames, "count": len(frames)}


def export_bag_player(cfg: dict) -> str:
    """One real-bag playback. Needs the bag itself (tens of GB, gitignored), so
    this is the exporter most likely to skip on any machine but the one that
    recorded them."""
    import torch  # noqa: F401
    from playback_common import derive_legend, derive_telemetry_fields

    sys.path.insert(0, str(REPO_ROOT / "ros2_inference"))
    from _shared_inference import ModelRunner
    from run_inference_on_bag import bag_frames_payload

    bag = need(REPO_ROOT / cfg["bag"], f"bag for {cfg['id']}")
    ckpt = need(REPO_ROOT / cfg["checkpoint"], f"checkpoint for {cfg['id']}")

    runner = ModelRunner(cfg["model"], ckpt, cfg.get("maxRange", 14.425209045410156),
                         class_conf_threshold=cfg.get("classThreshold", 0.97))

    frames_dir = PAYLOADS / "players" / "frames" / cfg["id"]
    sink = png_frame_sink(frames_dir, cfg.get("upscale", 1)) if cfg.get("frameMode") == "files" else None

    built = bag_frames_payload(
        bag=bag, topic=cfg.get("topic", "/ouster/points"), runner=runner,
        max_frames=cfg.get("maxFrames"), frame_sink=sink, upscale=cfg.get("upscale", 1),
    )
    meta, stats = built["meta"], built["stats"]

    payload = {
        "schemaVersion": 1,
        "id": cfg["id"], "kind": "real",
        "title": cfg.get("title", cfg["id"]),
        "subtitle": cfg.get("subtitle", ""),
        "source": {
            "model": cfg["model"], "checkpoint": Path(cfg["checkpoint"]).name,
            "scenarioId": None, "bag": Path(cfg["bag"]).name,
            "classConfThreshold": cfg.get("classThreshold", 0.97),
            "nFrames": stats["n_frames"], "nRawAvailable": stats["n_raw_available"],
            "durationS": round(stats["duration_s"], 1),
        },
        "rows": cfg.get("rows", []),
        "telemetryFields": derive_telemetry_fields(meta),
        "legend": derive_legend(meta, caption=cfg.get("legendCaption")),
        "frames": frames_descriptor(cfg, built["frames"]),
        "meta": meta,
        "stats": stats["means"],
    }
    out = write_payload(f"players/{cfg['id']}.js", "player", cfg["id"], payload)
    return f"{cfg['id']}: {stats['n_frames']} frames -> {out.relative_to(DEMO_DIR)}"


def export_gallery(cfg: dict) -> str:
    copied = 0
    for src_rel in cfg.get("from", []):
        src = REPO_ROOT / src_rel
        if not src.exists():
            continue
        dest = PAYLOADS / "gallery" / Path(src_rel).name
        dest.mkdir(parents=True, exist_ok=True)
        for png in sorted(src.glob("*.png")):
            shutil.copy2(png, dest / png.name)
            copied += 1
    if not copied:
        raise Skip("no gallery PNGs found under outputs/pointnet2/")
    return f"gallery: copied {copied} figures"


# --------------------------------------------------------------------------
# validation / reporting
# --------------------------------------------------------------------------

def cmd_check() -> int:
    from schema import check_all
    problems = check_all(DEMO_DIR)
    for p in problems:
        print("  FAIL " + p)
    if problems:
        print(f"\n{len(problems)} problem(s).")
        return 1
    print("check: OK")
    return 0


def cmd_report() -> int:
    total = 0
    rows = []
    for path in sorted(DEMO_DIR.rglob("*")):
        if path.is_dir() or ".git" in path.parts:
            continue
        size = path.stat().st_size
        total += size
        rows.append((size, path.relative_to(DEMO_DIR)))
    rows.sort(reverse=True)

    print(f"{'size':>12}  file")
    for size, rel in rows[:25]:
        print(f"{size/1e6:>9.2f} MB  {rel}")
    if len(rows) > 25:
        smaller = sum(s for s, _ in rows[25:])
        print(f"{smaller/1e6:>9.2f} MB  ... and {len(rows)-25} smaller files")
    print(f"\n{'total':>9}: {total/1e6:.2f} MB across {len(rows)} files")

    # GitHub refuses any file at/over 100 MB, and Pages serves LFS pointers as
    # text rather than resolving them, so a big file is a hard stop not a nudge.
    bad = [rel for size, rel in rows if size >= 95e6]
    if bad:
        print("\nFAIL: files at/near GitHub's 100 MB hard limit:")
        for rel in bad:
            print(f"  {rel}")
        return 1
    if total >= 200e6:
        print(f"\nFAIL: total {total/1e6:.0f} MB exceeds the 200 MB budget.")
        return 1
    return 0


# --------------------------------------------------------------------------

EXPORTERS = {
    "sim_player": ("simPlayers", export_sim_player),
    "bag_player": ("bagPlayers", export_bag_player),
    "gallery": ("gallery", export_gallery),
}


def run_exports(sources: dict, only: list[str] | None) -> int:
    done, skipped, failed = [], [], []

    def selected(kind: str, ident: str | None) -> bool:
        if not only:
            return True
        for sel in only:
            if sel == kind:
                return True
            if ":" in sel and sel.split(":", 1) == [kind, ident]:
                return True
        return False

    for kind, (src_key, fn) in EXPORTERS.items():
        cfgs = sources.get(src_key)
        if cfgs is None:
            continue
        items = cfgs if isinstance(cfgs, list) else [cfgs]
        for cfg in items:
            ident = cfg.get("id", src_key) if isinstance(cfg, dict) else src_key
            if not selected(kind, ident):
                continue
            try:
                done.append(fn(cfg))
            except Skip as e:
                skipped.append(f"{kind}:{ident} -- {e}")
            except Exception as e:  # noqa: BLE001 -- report and continue
                failed.append(f"{kind}:{ident} -- {type(e).__name__}: {e}")

    print("\n--- export summary ---")
    for d in done:
        print(f"  ok      {d}")
    for s in skipped:
        print(f"  skipped {s}")
    for f in failed:
        print(f"  FAILED  {f}")
    if skipped and not failed:
        print("\nSkips are expected on a machine without the full dataset; "
              "prior payloads (if any) were left untouched.")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run every exporter")
    ap.add_argument("--only", help="comma-separated kind or kind:id selectors")
    ap.add_argument("--check", action="store_true", help="validate payloads, no torch needed")
    ap.add_argument("--report", action="store_true", help="file sizes and budget check")
    args = ap.parse_args()

    if args.check:
        return cmd_check()
    if args.report:
        return cmd_report()
    if not (args.all or args.only):
        ap.print_help()
        return 0

    sources = load_sources()
    only = args.only.split(",") if args.only else None
    return run_exports(sources, only)


if __name__ == "__main__":
    raise SystemExit(main())
