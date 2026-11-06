# Deployment

Target: NVIDIA Jetson AGX Orin 64GB on the Clearpath Jackal, JetPack 5.1.x, inference only.
The sensor is an Ouster OS1-64 publishing over ROS2.

## Only the encoder deploys

The decoders exist to shape the encoder during training and are dropped afterwards. The robot
loads `encoder.state_dict()` and nothing else, so the decoders cost training time but no
inference time. `tests/test_pointnet2_model.py::test_only_encoder_state_dict_is_needed_for_deployment`
checks that a decoder-less `PointNet2Encoder` loaded from those weights reproduces `encode()`
exactly.

The shipped artifact is `weights/pointnet2_epoch39_encoder.pt` (1.4 MB, 352,288 parameters): the
encoder of the epoch-39 PointNet++ used for every thesis result, byte for byte.

## Using the encoder

`deploy.py` needs only numpy and torch. Import it from the repo root, or with the repo root on
`PYTHONPATH`:

```python
from deploy import ScanEncoder, SkipScan

encoder = ScanEncoder()                 # or ScanEncoder("path/to/encoder.pt", device="cuda")
try:
    latent = encoder(xyz_grid)          # (64, 1024, 3) float32 -> (256,) float32
except SkipScan as reason:
    ...                                 # too few valid returns; log it and wait for the next scan
```

or from the command line, for a quick check on a new machine:

```bash
python deploy.py scan.npy               # writes scan_latent.npy next to it
```

`xyz_grid` is one scan exactly as the Ouster driver delivers it: 64 rows x 1024 columns, x, y, z
in metres, NaN for no return. For each scan `ScanEncoder`:

1. on the first scan only, checks that the driver's point order is the one expected
   (`sensor.check_sim_order`) and raises `ValueError` if not;
2. reorders the scan into the simulator's order (`sensor.ouster_to_sim_order`);
3. treats returns outside the range the simulated sensor ever produced (0.30 to 120 m) as no
   return, which also covers drivers that emit (0, 0, 0);
4. decimates to every second column and pads to exactly 32,768 points with the same function
   training used (`dataset.prepare_points`);
5. raises `SkipScan` if fewer than 4,096 valid points remain (the first sampling level picks
   4,096 centres, so such a scan would be mostly padding).

Two things are easy to get wrong:

- Never feed the raw Ouster grid to the encoder; see
  [Real scans must arrive in simulator order](#real-scans-must-arrive-in-simulator-order).
- The encoder does not classify. Its output is the 256-d observation for a downstream policy. The
  per-point classes in the inference videos come from the segmentation decoder, which only exists
  in the full multitask checkpoint.

## Installing PyTorch on the Jetson

`pip install torch` does not work on Jetson: PyPI wheels are x86_64, and the Jetson is aarch64 with
a CUDA build tied to the exact JetPack version. Check the version first,

```bash
sudo apt-cache show nvidia-jetpack | grep -i version
```

then install the matching wheel from NVIDIA's Jetson PyTorch page, plus a matching `torchvision`
if you need it. The repo has no custom CUDA extensions: `models/pointnet2_utils.py` is plain
PyTorch (`torch.cdist`, `torch.sort`, `torch.where`), so nothing has to be compiled against the
Jetson's CUDA.

## Compute budget on the Jetson

The encoder has not been timed on the Orin. The table below is a theoretical comparison: what the
encoder needs for one 32,768-point scan, set against NVIDIA's published specification of the
device.

| | Jetson AGX Orin 64GB | PointNet++ encoder, one scan |
|---|---|---|
| Compute | 5.3 trillion FP32 operations per second | about 5.5 billion operations (3.0 layers, 1.4 farthest point sampling, 1.1 ball query) |
| Memory | 64 GB | 1.4 MB weights, 3.9 GB intermediate tensors |
| Memory bandwidth | 204.8 GB/s | about 8 GB moved |
| Parallelism | 2,048 GPU cores, 12 CPU cores | 5,376 sequential sampling steps |

On paper the encoder fits. Memory traffic and the sequential farthest point sampling are the
tightest bounds, so raw arithmetic throughput says little about the real speed. Reproduce the
encoder column with

```bash
python benchmarks/encoder_cost_estimate.py
```

To measure the encoder on the device, run on the Orin

```bash
python3 benchmarks/pointnet2_jetson_bench.py
```

which times `PointNet2Encoder` at 2,048 to 32,768 points, batch size 1, and reports latency and
peak GPU memory. For realistic numbers, run it while the robot's other processes (driver, SLAM)
are active.

## Real scans must arrive in simulator order

The Ouster and the simulator share the 64 x 1024 grid shape but not its order. The Ouster's row 0
is its highest beam and its columns run clockwise from 0°; the simulator's row 0 is its lowest
beam and its columns run counter-clockwise from -180° (`sensor.py`). PointNet++'s ball query keeps
the first 32 in-range points by array index, so the order is part of the input. Until 2026-09-28
real scans reached the model in Ouster order, unpadded. Scoring the labelled simulated test split
fed the way the real sensor delivers it (`real_eval/before_after.py sim`, 1,140 scans,
threshold 0.97):

| input | macro-F1 |
|---|---|
| simulator order, training input path | 86.7% |
| Ouster order, original input path | 17.4% |
| Ouster order with the real sensor's missing returns, original input path | 18.3% |
| Ouster order with the real sensor's missing returns, corrected input path | **86.9%** |

The corrected path is `deploy.real_scan_to_points`, used by `ScanEncoder` and by everything in
`ros2_inference/`. Outputs made before the fix (the old `real_bags/*.html` players and videos, and
the demo's bag3 player) show the broken input.

The SalsaNext and CNN paths in `ros2_inference/_shared_inference.py` still receive Ouster order;
the fix covered PointNet++ only.

## Offline: running a model over a recorded bag

No ROS2 installation is needed; bags are read with `rosbags`
(`pip install -r requirements-pipeline.txt`).

```bash
python ros2_inference/run_inference_on_bag.py --bag real_bags/bag3_20260910_152806
```

| option | default |
|---|---|
| `--bag` | required: the bag directory (contains `metadata.yaml`) |
| `--topic` | `/ouster/points` |
| `--odom-topic` | `/j100_0847/platform/odom/filtered` |
| `--model` | `pointnet2` (also `salsanext`, `kevin_cnn`) |
| `--checkpoint` | for PointNet++: `outputs/pointnet2_checkpoints/epoch39_multitask_model.pt` |
| `--max-range` | 14.425 (the shared normalization constant) |
| `--class-threshold` | 0.97 (0 = raw argmax) |
| `--out-dir` | `outputs/real_inference/<bag name>/` |
| `--stride`, `--limit` | every scan, no limit |
| `--no-video`, `--html`, `--max-frames` | video on, no HTML page, 200 frames in the page |

It writes `inference.mp4` (every scan, H.264), `predictions.npz` (per scan: class counts,
reconstruction MSE, every point predicted as an object with its confidence, odometry) and
`summary.json`. The bag is streamed one scan at a time. Unusable scans are skipped and counted by
reason; more than 1% skipped fails the run after everything is written.

Follow-up analyses on a finished run:

```bash
# where the model keeps seeing objects across scans -> objects.json, objects_map.png
python real_eval/detect_persistent_objects.py outputs/real_inference/bag3_20260910_152806

# evidence for the input-order fix: with labels (simulated test split) and label-free (the bags)
python real_eval/before_after.py sim
python real_eval/before_after.py bags
```

The bag locations for `before_after.py bags` are set in `BAGS` at the top of that script; edit
them to match your machine.

## On the robot: the live ROS2 node

`ros2_inference/live_inference_node.py` subscribes to the Ouster driver's organized point cloud,
runs a model on every scan and publishes a visualization image (original range, reconstruction,
predicted classes) for `rqt_image_view` or RViz2, so the model can be watched while the robot
drives. The per-scan code is the same as in the bag runner.

The repo is not an ament package, so start it directly from a sourced ROS2 shell:

```bash
source /opt/ros/<distro>/setup.bash
python3 ros2_inference/live_inference_node.py --ros-args \
    -p model:=pointnet2 \
    -p checkpoint:=outputs/pointnet2_checkpoints/epoch39_multitask_model.pt
```

| parameter | default | meaning |
|---|---|---|
| `topic` | `/ouster/points` | the driver's organized `PointCloud2`; launch-file namespaces can change it |
| `viz_topic` | `/lidar_encoder/visualization` | output `sensor_msgs/Image`, rgb8 |
| `model` | `salsanext` | `pointnet2`, `salsanext` or `kevin_cnn` |
| `checkpoint` | none (required) | a full multitask checkpoint; the node refuses to start without one |
| `max_range` | 14.425 | shared normalization constant |
| `save_dir` | empty | if set, saves a PNG snapshot every `save_every_sec` |
| `save_every_sec` | 5.0 | |

It needs `rclpy`, `sensor_msgs_py` (from ROS2), torch, numpy, matplotlib and Pillow. Every scan
must be 64 x 1024 (OS1-64 in `lidar_mode` 1024x10). A scan of any other shape is skipped and
reported once; the node does not resample it.

The node has not yet run on the robot. Its per-scan logic has been run over all three recorded
bags through the bag runner, which settled the scan shape, the no-return convention (NaN) and
the point order against the bags' own metadata.

A node that publishes the 256-d latent for a policy does not exist yet. It would be
`ScanEncoder` inside a subscriber like this one; see [continuing.md](continuing.md).

## Measured on the real recordings

- Dropout depends strongly on the ring. About 2% of returns are lost on the middle rings, and
  about 34% on the lowest, floor-grazing beam. `real_noise_augment.py` holds the measured
  64-value rate array. Until 2026-09-28 it was applied to simulated grids upside down.
- False points are rare. An early noise model injected spurious near-range points at 5%, but the
  bags show isolated near-range spikes at 0.065%, so the injection was removed.
- The ring order is inverted. According to the sensor's metadata, ring 0 is the highest beam at
  +21.6° and ring 63 sits at -21.2°. For display, real grids are not flipped the way simulated
  ones are; for the model, see the order section above.
- The sensor sits higher on the real robot, about 0.63 m against 0.535 m in simulation.
  Classification uses neighbourhood-relative coordinates and is unaffected
  (`tests/test_real_inference.py`). The reconstruction branch outputs absolute positions, though,
  so part of the real reconstruction error is this offset.

## What real data can and cannot show

Real recordings have no labels, so nothing on real data yields an accuracy, precision or F1. The
reports on real data are the predicted class map of a real drive at a stated threshold, shown next
to the reconstruction, and consistency measures: how stable the predictions are from scan to scan
and where objects persist on the map. Since the input-order fix, predictions on real scans are far
more consistent. A sim-to-real gap remains that input handling cannot close.
