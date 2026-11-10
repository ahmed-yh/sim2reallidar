# sim2real-lidar

LiDAR observation encoders for a robot in an underground tunnel. They are trained in a Gazebo
simulation of the tunnel and then run on recordings from the real Ouster OS1-64 sensor. An encoder
turns one scan (64 x 1024 points) into a 256-d vector that a downstream policy can use as its
observation.

Three candidates are compared: PointNet++ (`models/pointnet2.py`), SalsaNext
(`models/salsanext.py`) and a CNN autoencoder from Kevin Fischer's repository. PointNet++ is the
selected encoder, and only its encoder goes onto the robot's NVIDIA Jetson AGX Orin.

| simulated test split, 1,140 scans | PointNet++ | SalsaNext | CNN autoencoder |
|---|---|---|---|
| Reconstruction MSE (normalized range) | **0.000171** | 0.001288 | 0.000664 |
| Reconstruction SSIM | **0.9716** | 0.9524 | 0.9569 |
| Per-point accuracy | 98.96% | 98.93% | no classifier |
| Macro-F1 over the object classes, threshold 0.97 | 86.7% | **88.6%** | no classifier |

This README is a cookbook: find what you want to do in the table, copy the command, and the
section tells you what the command needs and what files it produces. The pages in
[`docs/`](docs/) explain why things are built the way they are.

## Find the command you need

| I want to | command | section |
|---|---|---|
| set up a new machine | `pip install -r requirements.txt` | [1](#1-set-up-a-new-machine) |
| turn recordings into a training dataset | `python data_prep/parse_recordings.py` and `python data_prep/split.py` | [2](#2-build-the-training-dataset) |
| train PointNet++ | `python train.py` | [3](#3-train-pointnet) |
| train SalsaNext | `python train_salsanext.py` | [4](#4-train-salsanext) |
| train the CNN autoencoder | Kevin Fischer's `train_ae.py` | [5](#5-train-the-cnn-autoencoder) |
| restore the trained thesis checkpoints into `outputs/` | `python checkpoints/join_parts.py` | [1](#1-set-up-a-new-machine) |
| continue from a trained model, or fine-tune with real-sensor noise | `python train.py --init-from ...` | [6](#6-continue-from-a-trained-model) |
| score a trained model on the test split | `python evaluate.py --checkpoint ...` | [7](#7-score-a-model-on-the-test-split) |
| compare the three models on one scale | `python kevin_pipeline/fair_comparison.py` | [8](#8-compare-the-three-models) |
| see what a model reconstructs and predicts | `python visualize_range_heatmap.py` and the `viz/` scripts | [9](#9-look-at-what-a-model-does) |
| run a model on a recorded real bag | `python ros2_inference/run_inference_on_bag.py --bag ...` | [10](#10-run-a-model-on-a-real-bag) |
| get the 256-d vector for one scan | `python deploy.py scan.npy` | [11](#11-get-the-256-d-vector-for-a-scan) |
| make a newly trained encoder the shipped one | copy it into `weights/` | [12](#12-ship-a-new-encoder) |
| install on the Jetson and measure speed | `python3 benchmarks/pointnet2_jetson_bench.py` | [13](#13-run-on-the-jetson) |
| watch a model live on the robot | `python3 ros2_inference/live_inference_node.py --ros-args ...` | [14](#14-watch-a-model-live-on-the-robot) |
| redraw the thesis charts or rebuild the demo page | `python thesis_figures/make_thesis_figures.py`, `python demo/build_demo.py --all` | [15](#15-thesis-charts-demo-page-and-docker) |
| check that nothing is broken | `python -m pytest tests/ -q` | [16](#16-run-the-tests) |

All commands run from the repository root. Paths like `path/to/my_recordings` stand for your own
folders.

## 1. Set up a new machine

Get the code, then place the data the group gives you next to it. Everything below the repo in
this tree is outside git; the scripts find it at these paths by default.

```bash
git clone https://github.com/ahmed-yh/sim2reallidar.git
cd sim2reallidar
```

```
some_folder/
├── sim2reallidar/                  this repository
│   ├── scenario_data/              raw Gazebo recordings (from the group)
│   ├── data/                       made by section 2
│   ├── outputs/                    checkpoints and results (made by training, or restored from checkpoints/)
│   ├── real_bags/                  real ROS2 bags (from the group)
│   └── noslil.ply, 2.ply           tunnel geometry, only for the demo page
└── masterarbeit_kevinfischer/      Kevin Fischer's repository, only for the CNN autoencoder
```

Make a virtual environment, install PyTorch, then the rest:

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows
source .venv/bin/activate                  # Linux

pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU build; pick your CUDA version on pytorch.org
pip install -r requirements.txt            # training, evaluation, tests
pip install -r requirements-pipeline.txt   # comparison script, real bags, videos, demo page
```

Install PyTorch first. If you start with `requirements.txt`, pip installs the CPU-only build,
which runs the tests but is far too slow for training.

Check the setup:

```bash
python -m pytest tests/ -q                                   # expect: 53 passed
python -c "import torch; print(torch.cuda.is_available())"   # expect: True on a GPU machine
```

To reproduce the thesis results without retraining, restore the trained checkpoints that ship in
`checkpoints/`:

```bash
python checkpoints/join_parts.py
```

This puts every PointNet++ and SalsaNext run of the thesis into `outputs/` and checks each file
against `checkpoints/SHA256SUMS`. That includes:
- `outputs/pointnet2_checkpoints/epoch39_multitask_model.pt`, the PointNet++ used for every thesis
  result;
- `outputs/salsanext/best_multitask_model.pt`.

[checkpoints/README.md](checkpoints/README.md) lists what each run is. The CNN autoencoder's
`outputs/kevin_cnn/best_autoencoder.pt` and `test_metrics.json` are not in the repository; get
them from the group.

## 2. Build the training dataset

Every model trains from a parsed dataset in `data/sim/`. You build it from raw recordings, either
the 76 thesis scenarios or new ones you recorded yourself.

### What a recording looks like

```
my_recordings/
├── scenario_A/
│   ├── 2026_08_25_17_13_30_968.npz
│   ├── 2026_08_25_17_13_30_968.json
│   └── ...
└── scenario_B/
    └── ...
```

| file | must contain |
|---|---|
| `<timestamp>.npz` | arrays `x`, `y`, `z`, `intensity`, 65,536 values each (64 rings x 1024 columns, ring 0 first, no-return as `inf`) |
| `<timestamp>.json` | `time_ns`, `robot_translation_x`, `robot_translation_y`, `robot_translation_z` |

In simulation `intensity` carries the class label: 0 environment, 1 human, 2 car, 3 bus,
4 sphere, 5 cylinder, 6 box2, 7 box1. The thesis scenarios contain only 0 and 4 to 7. A folder
of `.npz/.json` pairs without scenario subfolders also works; it is treated as one scenario.

### Build it

For the thesis recordings, which sit at the default path:

```bash
python data_prep/parse_recordings.py --max-scans-per-scenario 100
python data_prep/split.py
python kevin_pipeline/prepare_data.py
```

For your own recordings:

```bash
python data_prep/parse_recordings.py --recordings-dir path/to/my_recordings --max-scans-per-scenario 100
python data_prep/split.py
python kevin_pipeline/prepare_data.py
```

What each command does and writes:

| command | writes |
|---|---|
| `parse_recordings.py` | `data/sim/scans/<id>.npz` (range and class grid per scan) and `data/sim/poses.csv` |
| `split.py` | `data/sim/train_ids.npy`, `val_ids.npy`, `test_ids.npy`, and prints the split sizes and how many frames of each object class every split has |
| `prepare_data.py` | `data/kevin_range_npy/` (input for the CNN), split index files and `normalization_stats.json` (the shared `max_range`) in `outputs/kevin_cnn/` |

With the thesis recordings you should see 7,600 scans and a split of 5,320 / 1,140 / 1,140.

Things to know before you run it:

- `--max-scans-per-scenario 100` keeps 100 evenly spaced scans per scenario. The thesis used it;
  leave it out to keep every scan (47,958 for the thesis recordings).
- `parse_recordings.py` deletes and rewrites `data/sim/scans/`, and `prepare_data.py` overwrites
  the index and normalization files in `outputs/kevin_cnn/`. Rename `data/sim/` and back up
  `outputs/kevin_cnn/` first if you want to keep the thesis versions.
- `split.py` deals out blocks of 20 consecutive scans in a cycle of 14 train, 3 validation and 3
  test blocks. Give it at least 400 scans in total, or validation and test stay empty.
- Run `prepare_data.py` even if you never train the CNN: SalsaNext and the comparison script read
  `max_range` from the file it writes.
- PointNet++ and SalsaNext load their points from the raw recordings, so keep the recordings where
  they are and give training the same `--recordings-dir`.

## 3. Train PointNet++

Needs: section 2 done, and the raw recordings.

```bash
python train.py                                                                    # thesis recordings
python train.py --recordings-dir path/to/my_recordings --outputs-dir outputs/pointnet2_mydata   # your recordings
```

While it runs it prints one line per epoch with the training and validation losses, and a line
"new best, checkpoint saved" whenever the validation loss improves. It stops by itself when the
validation loss has not improved for 10 epochs, or after 100 epochs, and prints the best
validation loss.

It writes to `--outputs-dir` (default `outputs/pointnet2/`):

| file | what it is | written |
|---|---|---|
| `best_encoder.pt` | encoder weights only; this is what deploys | at every new best |
| `best_multitask_model.pt` | encoder and both decoders; use this for evaluation, visualization and `--init-from` | at every new best |
| `train_metrics.json` | final numbers and the arguments used | at the end |

A new run overwrites the files in its output folder, so give every run you want to keep its own
`--outputs-dir`. The losses per epoch are only printed; to keep them, append
`| tee train_pointnet2.log` to the command.

| option | default | change it when |
|---|---|---|
| `--batch-size` | 2 | your GPU runs out of memory (1 works) or has room for more |
| `--lr` | 1e-4 | |
| `--max-epochs` | 100 | |
| `--patience` | 10 | epochs without improvement before stopping |
| `--lambda-class` | 1.0 | to weight classification against reconstruction differently |
| `--chamfer-target-subsample` | 4096 | input points the reconstruction is compared against |
| `--weight-cap` | 100 | cap on the class weights (rare classes get large weights) |
| `--num-workers` | 4 | data loading is slow or uses too much memory |
| `--recordings-dir` | `scenario_data/mbut_76scenarios/o_robot_nav` | you use other recordings |
| `--splits-dir` | `data/sim` | your dataset is somewhere else |
| `--outputs-dir` | `outputs/pointnet2` | always, for a run you want to keep |
| `--init-from`, `--augment-noise` | off | see section 6 |

The thesis model is the epoch-39 checkpoint of a run with these defaults. No random seed is set,
so a retrain gives close but not identical numbers.

## 4. Train SalsaNext

Needs: section 2 done, including `prepare_data.py`.

```bash
python train_salsanext.py
python train_salsanext.py --recordings-dir path/to/my_recordings --outputs-dir outputs/salsanext_mydata
```

It prints and writes the same way as section 3, by default to `outputs/salsanext/`. Its defaults
differ: `--batch-size 16`, `--lr 1e-3`, `--patience 8`. It takes the same `--max-epochs`,
`--lambda-class`, `--weight-cap`, `--num-workers`, `--init-from` and `--augment-noise` options,
plus `--kevin-outputs-dir` (default `outputs/kevin_cnn`) for where `normalization_stats.json` is.

## 5. Train the CNN autoencoder

The CNN is Kevin Fischer's model and is trained with his own script, `train_ae.py`, under
`masterarbeit_kevinfischer/lidar_preprocessing/autoencoder_training/`. After section 2, run it
with these two paths (`prepare_data.py` prints them at the end):

```
--data-dir    <path to this repo>/data/kevin_range_npy
--outputs-dir <path to this repo>/outputs/kevin_cnn
```

The index files in `outputs/kevin_cnn/` make his script use this repo's split. This repo then
expects two files from his training in `outputs/kevin_cnn/`: `best_autoencoder.pt` and
`test_metrics.json` (with `mse` and `ssim`). It also rebuilds his model from
`masterarbeit_kevinfischer/lidar_preprocessing/autoencoder_training/denkwelt_deploy/outputs/best_params.json`,
so that file must exist. For his script's other options, see his repository.

## 6. Continue from a trained model

`--init-from` starts training from the weights of an earlier `best_multitask_model.pt` instead of
from random weights. The optimizer starts fresh and the epoch count restarts at 1.

Fine-tune the thesis model with the dropout pattern measured on the real sensor:

```bash
python train.py --init-from outputs/pointnet2_checkpoints/epoch39_multitask_model.pt \
    --augment-noise --outputs-dir outputs/pointnet2_ringnoise
```

Fine-tune it on your own recordings (after section 2 with them):

```bash
python train.py --init-from outputs/pointnet2_checkpoints/epoch39_multitask_model.pt \
    --recordings-dir path/to/my_recordings --outputs-dir outputs/pointnet2_finetuned_mydata
```

`--augment-noise` drops returns per ring at the rates measured on the real Ouster (see
`real_noise_augment.py`), on training scans only. `train_salsanext.py` takes both options too.

## 7. Score a model on the test split

```bash
python evaluate.py --checkpoint outputs/pointnet2/best_multitask_model.pt \
    --class-conf-threshold 0.97 --json-out outputs/measured/evaluate_pointnet2.json \
    --sweep "0,0.3,0.5,0.7,0.8,0.9,0.95,0.97,0.99,0.995"
```

This scores a PointNet++ checkpoint on the test split: the Chamfer reconstruction distance,
precision, recall and F1 per class, macro-F1 over the object classes, and overall accuracy. It prints the results and writes
`outputs/measured/evaluate_pointnet2.json` and, because of `--sweep`,
`outputs/measured/evaluate_pointnet2_sweep.json` with every threshold in the list. Add
`--recordings-dir path/to/my_recordings` if your dataset came from your own recordings.

- To reproduce the thesis numbers, use `--checkpoint outputs/pointnet2_checkpoints/epoch39_multitask_model.pt`.
- The confidence threshold turns low-confidence object predictions into environment. 0.97 was the
  best value for the thesis model. For a new model, look at the sweep file and take the threshold
  with the highest macro-F1.

## 8. Compare the three models

```bash
python kevin_pipeline/fair_comparison.py \
    --pointnet2-checkpoint outputs/pointnet2/best_multitask_model.pt \
    --salsanext-checkpoint outputs/salsanext/best_multitask_model.pt \
    --json-out outputs/measured/fair_comparison.json
```

This puts all three models on one scale: reconstruction MSE and SSIM on the same test scans and
with the same normalization, per-point accuracy, and a confidence sweep for both classifiers. It
prints a table and writes `outputs/measured/fair_comparison.json`, which the demo page and the
thesis charts read.

- Without the two checkpoint options it uses the thesis checkpoints, so you get the thesis table.
- The CNN's row comes from `outputs/kevin_cnn/test_metrics.json` (section 5); it is not
  recomputed. Without that file the CNN is left out and the other two are still scored.
- Add `--recordings-dir path/to/my_recordings` for your own dataset.

The `train_metrics.json` files of the runs cannot be compared with each other: PointNet++ reports
a Chamfer distance in metres, the others a normalized MSE. Always compare with this script.

## 9. Look at what a model does

| command | writes |
|---|---|
| `python visualize_range_heatmap.py --checkpoint <model>` | range image next to its reconstruction from the latent alone, in `outputs/pointnet2/range_heatmaps/` |
| `python visualize_reconstruction.py --checkpoint <model>` | input point cloud and reconstruction from top and side, in `outputs/pointnet2/recon_viz/` |
| `python viz/gen_playback.py` | `viz/mbut_playback.html`: the parsed scans frame by frame, no model |
| `python viz/gen_playback_pointnet2.py --checkpoint <model>` | `viz/pointnet2_playback.html`: PointNet++ on one simulated drive |
| `python viz/gen_playback_salsanext.py --checkpoint <model>` | `viz/salsanext_playback.html`: the same for SalsaNext |
| `python viz/gen_playback_kevin.py` | `viz/kevin_playback.html`: the same for the CNN (reads `outputs/kevin_cnn/best_autoencoder.pt`) |

`<model>` is a `best_multitask_model.pt`. The two `visualize_*` scripts default to
`outputs/pointnet2/best_multitask_model.pt`, the PointNet++ playback to the thesis checkpoint.
The playbacks show scenario `122e86f0-8b3f-46f5-8eaa-9f6b1232c7df` unless you pass
`--scenario-id`; open the HTML files in a browser.

## 10. Run a model on a real bag

### Record a bag on the robot

```bash
ros2 bag record /ouster/points /ouster/metadata /j100_0847/platform/odom/filtered
```

`/ouster/points` must be the driver's organized point cloud of 64 x 1024 (OS1-64 in `lidar_mode`
1024x10). The odometry is only needed for maps. If your topics have other names, pass them with
`--topic` and `--odom-topic` below.

### Run a model over it

This needs no ROS2 installation, only `requirements-pipeline.txt`.

```bash
python ros2_inference/run_inference_on_bag.py --bag real_bags/bag3_20260910_152806
python ros2_inference/run_inference_on_bag.py --bag path/to/my_bag --checkpoint outputs/pointnet2/best_multitask_model.pt
```

`--bag` is the bag folder, the one that contains `metadata.yaml`. Output goes to
`outputs/real_inference/<bag name>/`:

| file | contents |
|---|---|
| `inference.mp4` | every scan: real range, reconstruction, predicted classes |
| `predictions.npz` | per scan: predicted object points with confidences, reconstruction error, odometry |
| `summary.json` | settings, scans used and skipped (with reasons), class totals, runtime |

| option | default |
|---|---|
| `--model` | `pointnet2` (also `salsanext`, `kevin_cnn`) |
| `--checkpoint` | the thesis PointNet++ checkpoint |
| `--class-threshold` | 0.97 (0 shows the raw prediction) |
| `--stride N` | 1 (use every Nth scan) |
| `--limit N` | none (stop after N scans, for a quick test) |
| `--no-video` | video on |
| `--html page.html` | none (also write a scrollable page with `--max-frames` scans, default 200) |
| `--out-dir` | `outputs/real_inference/<bag name>/` |

Unusable scans are skipped and counted; if more than 1% are skipped the script exits with code 1
after writing everything. Only PointNet++ gets real scans in the simulator's point order; the
SalsaNext and CNN paths have not been fixed yet, so treat their real-data output with care.

### Analyse the run

```bash
python real_eval/detect_persistent_objects.py outputs/real_inference/bag3_20260910_152806
```

Finds places where the model keeps seeing the same object across scans and writes `objects.json`
and `objects_map.png` into the run folder. Needs the odometry topic in the bag.

```bash
python real_eval/before_after.py sim     # -> outputs/measured/preprocessing_ablation.json and .png
python real_eval/before_after.py bags    # -> outputs/real_inference/before_after.json and two charts
```

These measure the effect of the real-scan input fix, with labels on the simulated test split and
without labels on the bags. Set the bag paths in `BAGS` at the top of `real_eval/before_after.py`
first.

Real bags have no labels, so none of this gives an accuracy. Report the predicted class maps and
the consistency numbers, with the threshold you used.

## 11. Get the 256-d vector for a scan

From the command line, with one scan saved as a `.npy` file of shape (64, 1024, 3), x, y, z in
metres, NaN for no return, in the order the Ouster driver delivers it:

```bash
python deploy.py scan.npy        # writes scan_latent.npy, shape (256,)
```

In your own Python code:

```python
from deploy import ScanEncoder, SkipScan

encoder = ScanEncoder()                       # weights/pointnet2_epoch39_encoder.pt, GPU if available
# encoder = ScanEncoder("weights/my_encoder.pt", device="cpu")
try:
    latent = encoder(xyz_grid)                # (64, 1024, 3) float32 -> (256,) float32
except SkipScan:
    pass                                      # fewer than 4,096 valid returns; skip this scan
```

From a ROS2 `PointCloud2` message, `xyz_grid` is
`sensor_msgs_py.point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=False).reshape(64, 1024, 3)`.
`ScanEncoder` does the rest the way training did: it checks the point order on the first scan,
reorders, drops returns outside 0.30 to 120 m, keeps every second column and pads to 32,768
points. Never pass a scan to the encoder model directly.

## 12. Ship a new encoder

1. Copy the encoder of your run into the repo under a new name:
   `outputs/pointnet2_mydata/best_encoder.pt` to `weights/my_encoder.pt`. Apart from
   `checkpoints/`, `weights/` is the only folder where `.pt` files are tracked by git.
2. In `deploy.py`, point `DEFAULT_WEIGHTS` at the new file, and do the same for `WEIGHTS` in
   `tests/test_shipped_weights.py`.
3. Run `python -m pytest tests/ -q`.
4. Update every other place that names a checkpoint: the `checkpoint` fields in
   `demo/sources.json`, and the defaults in `kevin_pipeline/fair_comparison.py` and
   `ros2_inference/run_inference_on_bag.py` if the new model replaces the thesis one.

## 13. Run on the Jetson

The Jetson needs only `deploy.py`, `dataset.py`, `sensor.py`, `models/` and `weights/` from this
repo, plus numpy and PyTorch.

`pip install torch` does not work on Jetson. Find the JetPack version, then install NVIDIA's
matching wheel from their Jetson PyTorch page:

```bash
sudo apt-cache show nvidia-jetpack | grep -i version
```

JetPack 5.1 has Python 3.8; the files `deploy.py` imports use no newer syntax. Then check it with
one scan (section 11) and measure the encoder:

```bash
python3 deploy.py scan.npy
python3 benchmarks/pointnet2_jetson_bench.py
```

The benchmark times the encoder at 2,048 to 32,768 points per scan, one scan at a time, and
prints latency and peak GPU memory. Run it while the driver and SLAM are running, and write down
the JetPack and PyTorch versions with the numbers. Nobody has measured this on the Orin yet.

```bash
python benchmarks/encoder_cost_estimate.py
```

prints the theoretical compute and memory cost of one scan that the thesis compares with the
Orin's specification. It runs on any machine.

## 14. Watch a model live on the robot

The live node runs a full model on every scan and publishes an image (range, reconstruction,
predicted classes) that you can watch in `rqt_image_view` or RViz2. It needs ROS2 (`rclpy`,
`sensor_msgs_py`), torch, numpy, matplotlib and Pillow.

```bash
source /opt/ros/<distro>/setup.bash
python3 ros2_inference/live_inference_node.py --ros-args \
    -p model:=pointnet2 \
    -p checkpoint:=outputs/pointnet2_checkpoints/epoch39_multitask_model.pt
```

| parameter | default | meaning |
|---|---|---|
| `checkpoint` | none, required | a full `best_multitask_model.pt`, not the encoder-only file |
| `model` | `salsanext` | `pointnet2`, `salsanext` or `kevin_cnn` |
| `topic` | `/ouster/points` | input point cloud |
| `viz_topic` | `/lidar_encoder/visualization` | output image |
| `max_range` | 14.425 | shared normalization, leave it |
| `save_dir` | empty | set a folder to save a PNG snapshot every `save_every_sec` seconds |
| `save_every_sec` | 5.0 | |

The node has not been run on the robot yet; its per-scan code has been run over all three
recorded bags. Nothing publishes the 256-d vector for a policy yet; that node would wrap
`ScanEncoder` (section 11) in a subscriber like this one.

## 15. Thesis charts, demo page and Docker

```bash
python thesis_figures/make_thesis_figures.py   # charts from outputs/measured/*.json -> thesis_figures/*.png
python demo/build_demo.py --check              # validate the demo page; needs no data
python demo/build_demo.py --all                # rebuild the demo page from checkpoints and data
python demo/build_demo.py --only sim_player:pointnet2_sim   # rebuild one part
docker compose up -d demo                      # serve the demo page on http://localhost:8080
```

The demo page opens without a server: `demo/index.html` in a browser. Parts whose inputs are
missing on your machine are skipped and keep their committed version. To use new models on the
page, change the `checkpoint` fields in `demo/sources.json` first. The Docker images have not
been built and tested yet. More in [docs/demo-and-docker.md](docs/demo-and-docker.md).

## 16. Run the tests

```bash
python -m pytest tests/ -q
```

They use synthetic scans in the real format, need no data or GPU, and finish in under a minute.
Run them after every change. If you add a feature, add a test that runs on synthetic data, so the
suite keeps working on a fresh clone.

## Troubleshooting

| problem | fix |
|---|---|
| `torch.cuda.is_available()` prints `False` on a GPU machine | the CPU build of PyTorch is installed; reinstall it from the PyTorch index for your CUDA version (section 1) |
| `CUDA out of memory` during training | lower `--batch-size`, down to 1 |
| about 48,000 files in `data/sim/scans/` | `--max-scans-per-scenario 100` was left out (section 2) |
| `split.py` reports 0 validation or test samples | fewer than about 400 scans in total; record more |
| `FileNotFoundError` for `outputs/kevin_cnn/normalization_stats.json` | run `python kevin_pipeline/prepare_data.py` |
| `FileNotFoundError` for a recording during training | `--recordings-dir` points somewhere else than the folder `parse_recordings.py` read |
| `parse_recordings.py`: "expected 65536 points" | the recording is not 64 x 1024 |
| "Kevin's test_metrics.json not found" | the CNN row needs `outputs/kevin_cnn/test_metrics.json` from section 5 |
| errors mentioning `denkwelt_deploy` | Kevin's repository is missing or not next to this one (section 1) |
| `ValueError: scan is not in simulator row order` (or `column order`) | the driver delivers points in an unexpected order; compare its settings with the conventions in `sensor.py` |
| live node: "Scan shape (...) doesn't match" | set the OS1-64 to `lidar_mode` 1024x10 |
| the bag runner exits with code 1 | more than 1% of scans were unusable; `summary.json` lists the reasons |
| the bag runner says "no maps" | the bag has no odometry topic; pass `--odom-topic` with its name |

## Background and open work

| page | covers |
|---|---|
| [docs/data.md](docs/data.md) | the recordings, labels and their traps, the split, the 32,768-point budget |
| [docs/training-and-evaluation.md](docs/training-and-evaluation.md) | training configuration, why the two decoders differ, the confidence threshold |
| [docs/deployment.md](docs/deployment.md) | encoder-only deployment, the compute budget, the real-scan input fix, what real data can show |
| [docs/demo-and-docker.md](docs/demo-and-docker.md) | how the demo page is built, the two Docker services |
| [docs/continuing.md](docs/continuing.md) | open items, in order, for whoever continues |

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
├── data_prep/              parse_recordings.py, split.py
├── kevin_pipeline/         prepare_data.py, fair_comparison.py
├── weights/                pointnet2_epoch39_encoder.pt
├── checkpoints/            every trained thesis run; join_parts.py restores them into outputs/
├── ros2_inference/         offline bag runner, live ROS2 node, their shared per-scan code
├── real_eval/              persistent-object map, before/after evidence for the input fix
├── benchmarks/             encoder cost estimate, on-device benchmark
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

The CNN autoencoder was developed by Kevin Fischer in his master's thesis repository
(`masterarbeit_kevinfischer`). This repo prepares data for it (`kevin_pipeline/`), scores it next
to the other two and loads it from his code without copying it. The tunnel point cloud and mesh
behind the simulation come from a separate SLAM-to-mesh pipeline.
