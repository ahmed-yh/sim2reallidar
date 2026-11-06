#!/usr/bin/env python3
"""Live inference node: Ouster OS1 (ROS2) -> a trained encoder, in real time.

Subscribes directly to the Ouster ROS2 driver's organized point cloud topic
(default: /ouster/points). "Organized" is the key property this depends on:
ouster-ros publishes a sensor_msgs/PointCloud2 with height = channel count
and width = columns per revolution, not a flat unordered list -- so each
point's ring/azimuth identity survives intact, the same structure the
simulated training data has.

The actual per-model inference + visualization logic lives in
_shared_inference.py, shared with run_inference_on_bag.py (the offline,
no-ROS2-needed equivalent for reading recorded bags) -- this file is just
the ROS2 plumbing around that shared logic.

Publishes a live visualization image (sensor_msgs/Image, rgb8) on
viz_topic so reconstruction/classification quality can be watched in
rqt_image_view or RViz2's Image display while driving. Don't mistake the
predicted-class row for a validated result -- it's the model's own guess,
with nothing to check it against here.

The repo is not an ament package, so start the script directly from a
sourced ROS2 shell (rclpy reads the --ros-args from sys.argv):

    python3 ros2_inference/live_inference_node.py --ros-args \\
        -p model:=pointnet2 -p checkpoint:=/path/to/epoch39_multitask_model.pt

The checkpoint is a FULL multitask model (encoder + both decoders), since
the visualization needs the decoders. To produce only the 256-d observation,
use deploy.ScanEncoder with the shipped encoder weights instead.
See docs/deployment.md for every parameter.

This node itself has NOT run on the robot -- this development machine has no
ROS2 install (`import rclpy` fails here). The per-scan logic it calls has
been run over all three recorded bags (run_inference_on_bag.py), which
settled what used to be open here (checked 2026-09-27 against the bags'
/ouster/metadata and /ouster/points):
  1. Scan shape: the recorded OS1-64 runs lidar_mode 1024x10, i.e. (64, 1024)
     -- what the models expect. This node still checks msg.height/msg.width
     on every scan and refuses a mismatch rather than silently resampling.
  2. No-returns: ouster-ros marks them NaN in these bags. A (0,0,0)
     convention would be harmless for PointNet++ anyway: returns outside the
     simulated sensor's range are dropped (sensor.SIM_RANGE_M).
  3. Point order: the Ouster's ring/column order is the reverse of the
     simulator's; the PointNet++ path reorders every scan, and on the first
     scan refuses to run if the order isn't the one it expects
     (sensor.check_sim_order).
  4. The topic name -- /ouster/points is ouster-ros's documented default,
     but launch-file namespacing can change it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared_inference import N_COLS, N_RINGS, ModelRunner, SkipScan  # noqa: E402


def numpy_to_image_msg(arr: np.ndarray, header) -> RosImage:
    """No cv_bridge dependency -- a plain rgb8 sensor_msgs/Image is simple
    enough to build by hand, and this way there's one less package that
    needs to be installed/working on the robot for this node to run."""
    msg = RosImage()
    msg.header = header
    msg.height, msg.width = arr.shape[0], arr.shape[1]
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = arr.shape[1] * 3
    msg.data = np.ascontiguousarray(arr).tobytes()
    return msg


class LiveInferenceNode(Node):
    def __init__(self):
        super().__init__("lidar_encoder_inference")

        self.declare_parameter("topic", "/ouster/points")
        self.declare_parameter("viz_topic", "/lidar_encoder/visualization")
        self.declare_parameter("model", "salsanext")  # 'pointnet2' | 'salsanext' | 'kevin_cnn'
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("max_range", 14.425)  # shared normalization -- see
        # outputs/kevin_cnn/normalization_stats.json. Calibrated from SIMULATED range
        # statistics; whether it's still the right scale for a real OS1's actual
        # return-range distribution is itself an open sim2real question, not assumed here.
        self.declare_parameter("save_dir", "")  # empty = don't save snapshots to disk
        self.declare_parameter("save_every_sec", 5.0)

        topic = self.get_parameter("topic").value
        viz_topic = self.get_parameter("viz_topic").value
        model_name = self.get_parameter("model").value
        checkpoint = self.get_parameter("checkpoint").value
        max_range = self.get_parameter("max_range").value
        save_dir = self.get_parameter("save_dir").value
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_every_sec = self.get_parameter("save_every_sec").value
        self._last_save_t = 0.0
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        if not checkpoint:
            raise SystemExit("--checkpoint is required (no random-weights fallback here -- "
                              "a live QA tool running on random weights would be actively misleading)")
        self.model_name = model_name
        self.runner = ModelRunner(model_name, Path(checkpoint), max_range)
        self.get_logger().info(f"Loaded {model_name} on {self.runner.device}, "
                                f"listening on {topic}, publishing viz on {viz_topic}")

        self.sub = self.create_subscription(PointCloud2, topic, self.on_cloud, 10)
        self.viz_pub = self.create_publisher(RosImage, viz_topic, 10)
        self._warned_shape = False

    def on_cloud(self, msg: PointCloud2):
        if msg.height != N_RINGS or msg.width != N_COLS:
            if not self._warned_shape:
                self.get_logger().error(
                    f"Scan shape ({msg.height}x{msg.width}) doesn't match the trained models' "
                    f"({N_RINGS}x{N_COLS}). This means your OS1's channel count or "
                    f"horizontal-resolution setting doesn't match the simulated training data -- "
                    f"needs resolving (reconfigure the sensor to match, resample the scan, or "
                    f"retrain for your sensor's real shape) before this produces anything "
                    f"meaningful. Not attempting to reshape/resample silently."
                )
                self._warned_shape = True
            return

        pts = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=False)
        xyz = pts.reshape(N_RINGS, N_COLS, 3).astype(np.float32)

        try:
            combined, mse = self.runner.run(xyz)
        except SkipScan as e:
            self.get_logger().warning(f"skipped a scan: {e}", throttle_duration_sec=5.0)
            return
        self.viz_pub.publish(numpy_to_image_msg(combined, msg.header))
        self.get_logger().debug(f"[{self.model_name}] recon mse={mse:.4f}")

        if self.save_dir is not None:
            now = time.monotonic()
            if now - self._last_save_t >= self.save_every_sec:
                self._last_save_t = now
                out = self.save_dir / f"{self.model_name}_{int(time.time())}.png"
                PILImage.fromarray(combined).save(out)
                self.get_logger().info(f"saved snapshot: {out}")


def main():
    rclpy.init()
    node = LiveInferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
