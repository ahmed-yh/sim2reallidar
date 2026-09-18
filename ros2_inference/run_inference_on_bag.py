"""Run a trained encoder over every /ouster/points scan in a recorded ROS2
bag, offline -- no ROS2 install needed. Uses `rosbags` (pip install rosbags),
a pure-Python CDR deserializer that reads ROS2 bags (.db3/.mcap) directly
from their own self-described message schemas, without rclpy or any ROS2
runtime. See live_inference_node.py's module docstring for the live-ROS2
version of this same model-application logic; this script does the same
per-scan work, just reading scans from a bag file in a loop instead of a
topic subscription.

Reuses the exact same per-model inference + visualization code as
live_inference_node.py (same shape check, same input construction, same
reprojection for PointNet++, same viz/playback_common.py rendering) --
kept as a single shared implementation in _shared_inference.py rather than
duplicated between the live node and this offline script.

    python3 ros2_inference/run_inference_on_bag.py \\
        --bag /path/to/my_bag \\
        --model salsanext --checkpoint outputs/salsanext/best_multitask_model.pt \\
        --topic /ouster/points --out bag_playback.html

NOT tested against a real bag file (none available on this machine) --
verify the topic name and message type against your actual bag before
trusting this; run `python3 -m rosbags.convert --list-topics <bag>`-style
inspection first if unsure what topics/types it actually contains.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "viz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared_inference import N_COLS, N_RINGS, ModelRunner  # noqa: E402
from playback_common import encode_frame, label_row, patch_template  # noqa: E402


def iter_pointcloud2(bag_path: Path, topic: str):
    """Yields (timestamp_ns, height, width, xyz (height,width,3) float32)
    for every message on `topic` in the bag, oldest first."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            available = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"topic {topic!r} not found in bag. Available topics:\n  " +
                              "\n  ".join(available))

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            height, width = msg.height, msg.width

            # Parse the PointCloud2 field layout dynamically rather than
            # assuming byte offsets -- point_step/fields describe the exact
            # binary layout, and it's cheap to look up once per message.
            field_offset = {f.name: f.offset for f in msg.fields}
            data = np.frombuffer(msg.data.tobytes(), dtype=np.uint8)
            data = data.reshape(-1, msg.point_step)

            def read_field(name: str) -> np.ndarray:
                off = field_offset[name]
                return data[:, off:off + 4].copy().view(np.float32).reshape(-1)

            x, y, z = read_field("x"), read_field("y"), read_field("z")
            xyz = np.stack([x, y, z], axis=-1).reshape(height, width, 3).astype(np.float32)
            yield timestamp, height, width, xyz


def rendered_frames(all_frames, runner, t0, progress_every):
    """Yields (index, seconds_since_start, visualization, recon_mse) for every
    scan whose grid shape matches what the models were trained on, skipping
    (and warning once about) any that don't. Shared by both output paths --
    the .mp4 writer and the HTML page differ only in what they do with each
    rendered frame, not in how one gets produced."""
    n_shape_mismatch = 0
    for i, (t_ns, height, width, xyz) in enumerate(all_frames):
        if height != N_RINGS or width != N_COLS:
            n_shape_mismatch += 1
            continue
        combined, mse = runner.run(xyz)
        yield i, (t_ns - t0) / 1e9, combined, mse
        if (i + 1) % progress_every == 0:
            print(f"  processed {i + 1}/{len(all_frames)}")

    if n_shape_mismatch:
        print(f"WARNING: skipped {n_shape_mismatch} scans with shape != "
              f"({N_RINGS},{N_COLS}) -- check your OS1's channel count / "
              f"horizontal resolution setting.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", type=Path, required=True, help="Path to the ROS2 bag directory (contains metadata.yaml)")
    ap.add_argument("--topic", default="/ouster/points")
    ap.add_argument("--model", choices=["pointnet2", "salsanext", "kevin_cnn"], default="salsanext")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--max-range", type=float, default=14.425)
    ap.add_argument("--class-threshold", type=float, default=0.97,
                     help="Confidence threshold for the predicted-class display -- 0.97 maximizes "
                          "macro-F1 on the labeled simulated test set (see "
                          "_shared_inference.ModelRunner's docstring). Pass 0 to see the raw, "
                          "unthresholded argmax instead.")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "bag_playback.html")
    ap.add_argument("--max-frames", type=int, default=200,
                     help="Evenly subsample if the bag has more scans than this (Artifact/HTML size cap). "
                          "Ignored when --out-video is given -- a video file has no such cap, so every "
                          "scan in the bag is used, at the bag's own native scan rate.")
    ap.add_argument("--out-video", type=Path, default=None,
                     help="Write an actual .mp4 instead of the HTML page -- every scan, played back at "
                          "the bag's real scan rate, so you can watch inference run live instead of "
                          "scrubbing static frames. Needs opencv-python-headless (pip install).")
    args = ap.parse_args()

    runner = ModelRunner(args.model, args.checkpoint, args.max_range,
                          class_conf_threshold=args.class_threshold)
    print(f"Loaded {args.model} from {args.checkpoint} on {runner.device}")

    print(f"Reading {args.topic} from {args.bag} ...")
    all_frames = list(iter_pointcloud2(args.bag, args.topic))
    print(f"Found {len(all_frames)} scans")
    if len(all_frames) == 0:
        raise SystemExit("no scans found -- check --topic against the bag's actual topics")

    if args.out_video is None and len(all_frames) > args.max_frames:
        idx = np.linspace(0, len(all_frames) - 1, args.max_frames).round().astype(int)
        all_frames = [all_frames[i] for i in sorted(set(idx.tolist()))]
        print(f"Subsampled to {len(all_frames)} scans")

    t0 = all_frames[0][0]

    if args.out_video is not None:
        import cv2
        total_s = (all_frames[-1][0] - t0) / 1e9
        fps = (len(all_frames) - 1) / total_s if total_s > 0 else 10.0
        print(f"Writing video at {fps:.2f} fps (bag's own scan rate: "
              f"{len(all_frames)} scans over {total_s:.1f}s)")

        writer, mses = None, []
        for i, t_s, combined, mse in rendered_frames(all_frames, runner, t0, progress_every=50):
            mses.append(mse)
            bar = label_row(np.full((18, combined.shape[1], 3), 14, dtype=np.uint8),
                             f"t={t_s:6.1f}s  frame {i + 1}/{len(all_frames)}  recon mse {mse:.4f}")
            frame = np.concatenate([bar, combined], axis=0)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(args.out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if writer is None:
            raise SystemExit(f"no scans matched the expected ({N_RINGS},{N_COLS}) shape -- nothing to render")
        writer.release()
        print(f"mean recon mse (normalized): {np.mean(mses):.4f}")
        print(f"wrote {args.out_video} ({len(mses)} frames @ {fps:.2f}fps)")
        return

    frames_b64, meta = [], []
    for _i, t_s, combined, mse in rendered_frames(all_frames, runner, t0, progress_every=20):
        frames_b64.append(encode_frame(combined, upscale=2))
        meta.append({"t": t_s, "mse": mse})

    if not frames_b64:
        raise SystemExit("no scans matched the expected (64,1024) shape -- nothing to render")

    print(f"mean recon mse (normalized): {np.mean([m['mse'] for m in meta]):.4f}")

    import json
    frames_json = json.dumps({"frames": frames_b64, "meta": meta})
    template = (REPO_ROOT / "viz" / "player_template.html").read_text()
    html = patch_template(template, {
        "<title>MBUT Corridor Playback</title>": f"<title>Real Bag: {args.model}</title>",
        "LiDAR playback &middot; scenario 6633e5c2": f"Real bag inference &middot; {args.model}",
        "MBUT corridor &mdash; what the encoder sees": f"{args.bag.name} on real hardware",
        "600 scans &middot; ~318s &middot; ~121m corridor":
            f"{len(frames_b64)} scans &middot; {args.checkpoint.name}",
        "Range &amp; class channels, as fed to the model": "Original vs. reconstructed (real data, no ground truth)",
        "Classes in this frame": "Predicted objects (no ground truth on real data)",
        'Scenario 6633e5c2 &middot; sim2real-lidar &middot; class colors: '
        'env dark, sphere amber, cylinder teal, box2 yellow, box1 coral':
            f"{args.bag.name} &middot; real Ouster OS1-64 bag &middot; class colors: "
            f"env dark, sphere amber, cylinder teal, box2 yellow, box1 coral",
        '<span class="channel-tag range">range</span>': "",
        '<span class="channel-tag class">class</span>': "",
        '<div class="channel-divider"></div>': "",
        '<div class="telemetry-row"><span class="k">Distance</span>'
        '<span class="v mono" id="tDist">0.0m</span></div>':
            '<div class="telemetry-row"><span class="k">Recon MSE (norm.)</span>'
            '<span class="v mono" id="tMse">0.0</span></div>',
        "tDist.textContent = m.dist.toFixed(1) + 'm';":
            "document.getElementById('tMse').textContent = m.mse.toFixed(4);",
        "__FRAMES_JSON__": frames_json,
    })
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({len(html) / 1e6:.2f} MB, {len(frames_b64)} frames)")


if __name__ == "__main__":
    main()
