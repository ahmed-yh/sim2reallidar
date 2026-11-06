# sim2real-lidar

LiDAR observation encoders for a robot in an underground tunnel, trained entirely in a simulation
of that tunnel and then run on recordings from the real sensor.

An encoder compresses one Ouster OS1-64 scan into a 256-d vector that a downstream policy can use
as its observation. Three candidates share the contract `encode(scan) -> (B, 256)`, train on the
same simulated dataset and are scored on the same held-out split:

| candidate | input | reconstruction | classification | code |
|---|---|---|---|---|
| PointNet++ | 32,768 raw 3D points | Chamfer, 2,048 free-form points | per point, 8 classes | `models/pointnet2.py` |
| SalsaNext | 64 x 1024 grid (range, x, y, z) | native grid | per pixel, 8 classes | `models/salsanext.py` |
| CNN autoencoder | 64 x 1024 range image | native grid | none | Kevin Fischer's repository (see [Credits](#credits)) |

PointNet++ is the selected encoder. Its encoder alone goes onto the robot, an NVIDIA Jetson AGX
Orin on a Clearpath Jackal; the decoders are only there to shape it during training.

## Results

Simulated held-out test split: 1,140 scans from 19 scenarios that contribute no frame to training.

| | PointNet++ | SalsaNext | CNN autoencoder |
|---|---|---|---|
| Reconstruction MSE (normalized range) | **0.000171** | 0.001288 | 0.000664 |
| Reconstruction SSIM | **0.9716** | 0.9524 | 0.9569 |
| Per-point accuracy | 98.96% | 98.93% | no classifier |
| Macro-F1 over the object classes, threshold 0.97 | 86.7% | **88.6%** | no classifier |

Under this protocol PointNet++ is the best-performing model on reconstruction, and SalsaNext
classifies the object classes slightly better. The ranking holds for this protocol and this
tunnel and says nothing about the three architectures in general.

Real recordings have no ground truth (a real Ouster carries no Gazebo `laser_retro` label), so
nothing on real data gets a score. The real-data results are inspections and consistency measures,
described in [docs/deployment.md](docs/deployment.md#what-real-data-can-and-cannot-show).

## Quick start

```bash
git clone https://github.com/ahmed-yh/sim2reallidar.git
cd sim2reallidar
pip install -r requirements.txt
python -m pytest tests/ -q
```

The tests build synthetic scans in the real format, so they need no data and no GPU.

Encode one scan with the shipped weights:

```python
from deploy import ScanEncoder

encoder = ScanEncoder()        # weights/pointnet2_epoch39_encoder.pt, CUDA if available
latent = encoder(xyz_grid)     # (64, 1024, 3) float32 as the Ouster driver delivers it -> (256,)
```

To look at the results without running anything, open `demo/index.html` in a browser.

## Documentation

| page | covers |
|---|---|
| [docs/data.md](docs/data.md) | where the data lives, regenerating `data/`, the label format and its traps, the split, the point budget |
| [docs/training-and-evaluation.md](docs/training-and-evaluation.md) | training each candidate, the training configuration, comparing the three on one scale, the confidence threshold, the test suite |
| [docs/deployment.md](docs/deployment.md) | encoder-only deployment, the Jetson, its compute budget, feeding real scans, the offline bag runner, the live ROS2 node |
| [docs/demo-and-docker.md](docs/demo-and-docker.md) | the offline demo page and the two Docker services |
| [docs/continuing.md](docs/continuing.md) | open items and where to pick the work up |

## What is not in git

The repository holds code, tests, the demo and one trained artifact (the deployable encoder,
1.4 MB). Everything large lives outside it and is ignored by `.gitignore`:

| path | contents | how to get it |
|---|---|---|
| `scenario_data/` | raw Gazebo recordings, about 496 MB | copy from the group's storage |
| `data/` | parsed grids and the split | regenerate, see [docs/data.md](docs/data.md) |
| `real_bags/` | real ROS2 bags from the robot, tens of GB | copy from the group's storage |
| `outputs/` | checkpoints, metrics, inference runs | retrain, or copy |
| `*.ply` at the root | tunnel geometry for the demo | copy from the group's storage |
| `../masterarbeit_kevinfischer/` | the CNN autoencoder, cloned next to this repo | Kevin Fischer's repository |

## Repository layout

```
sim2reallidar/
├── deploy.py               the deployable encoder: ScanEncoder, real_scan_to_points (numpy + torch only)
├── sensor.py               Ouster <-> simulator scan order, the order check, range limits
├── dataset.py              raw recordings -> decimated (xyz, class)          [PointNet++]
├── salsanext_dataset.py    range-image dataset                               [SalsaNext]
├── losses.py               Chamfer, class-weighted cross-entropy
├── train.py, train_salsanext.py, train_utils.py
├── evaluate.py             PointNet++ test-split metrics, threshold sweep
├── real_noise_augment.py   measured per-ring dropout of the real sensor
├── models/                 pointnet2.py, pointnet2_utils.py, salsanext.py
├── data_prep/              parse_recordings.py, split.py (shared by all candidates)
├── kevin_pipeline/         prepare_data.py, fair_comparison.py (all three on one scale)
├── weights/                pointnet2_epoch39_encoder.pt
├── ros2_inference/         offline bag runner, live ROS2 node, their shared per-scan code
├── real_eval/              persistent-object map, before/after evidence for the input fix
├── benchmarks/             encoder cost estimate, on-device latency benchmark
├── viz/, visualize_*.py    playbacks and figures of simulated scans
├── demo/                   self-contained results page
├── thesis_figures/         script and data behind the thesis charts
├── docker/, docker-compose.yml
├── tests/                  synthetic fixtures, no external data
└── docs/
```

## Credits

This repository is the code of the bachelor thesis *Sim-to-Real Evaluation of Edge-Deployed
Perception Models Using LiDAR-Based Environment Reconstruction* by Ahmed Yassine Hachem, carried
out in Team Automotive at OTH Amberg-Weiden (2026), supervised by Prof. Dr.-Ing. Alexander Prinz
and Jakob Götz.

The CNN autoencoder candidate was developed by Kevin Fischer in his master's thesis repository
(`masterarbeit_kevinfischer`). This repo integrates it for the comparison: `kevin_pipeline/`
prepares data for it in his format and scores it next to the other two, and
`ros2_inference/_shared_inference.py` loads it from his code without copying it.

The tunnel point cloud and mesh behind the simulation come from a separate SLAM-to-mesh
pipeline and are not part of this repository.
