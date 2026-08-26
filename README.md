# sim2real-lidar — PointNet++ candidate

## What this project is

This is one of three candidate encoders being compared for a LiDAR-based perception
system: compress a LiDAR scan into a compact latent vector that a downstream
reinforcement-learning policy can consume as its observation, while also — during
training only — being pushed to encode *semantically* meaningful information (what
kind of object each point belongs to), not just raw geometry.

The three candidates share one contract: `encode(scan) -> (B, 256)`. The other two
are a CNN autoencoder (baseline, modeled on a known-working architecture used
elsewhere in this group) and a CNN-VAE variant. This repo is the third: a **PointNet++
encoder operating directly on raw 3D points**, rather than on a 2D range-image
projection. All three get trained on identical simulated data and compared on the
same held-out set, and eventually on real data captured from the same physical
tunnel.

Why a point-based candidate at all, alongside two image-based ones: the CNN
candidates rasterize each scan into a fixed `(64, 1024)` grid (one pixel per LiDAR
ray). For this specific sensor that rasterization is actually lossless — see
"Why raw points, not the range grid" below — but it still commits to 2D-image
assumptions (translation-equivariant convolutions, a fixed grid topology) that a
point-based method doesn't share. Comparing the two representations empirically,
rather than assuming one is better, is the point of having three candidates instead
of one.

## The data

A single fully-recorded Gazebo scenario: a Jackal robot driving through the "MBUT"
tunnel corridor (~1.95 m wide, ~300 m of corridor, built from a real-world mesh
reconstruction of that tunnel), with objects placed in it. 600 recordings, sampled
at roughly 1.9 Hz along the drive (**not** the sensor's native rate — see
"Point budget and real-time constraints" below).

Each recording (`x, y, z, intensity, ring` — 65,536 points, a VLP-16-labeled but
effectively 64-ring × 1024-column organized scan) carries a per-point class label in
`intensity`, via Gazebo's `laser_retro` mechanism:

| raw value | class | notes |
|---|---|---|
| 0.0 | environment | tunnel walls/floor, **and** no-return points (see below) |
| 1.0 | human | **zero examples in this recording** |
| 2.0 | car | **zero examples** (this value also covers a dead "mercedes" duplicate label that never spawns) |
| 3.0 | bus | **zero examples** |
| 4.0 | sphere | present |
| 5.0 | cylinder | present, the rarest of the four present classes |
| 6.0 | box2 | present |
| 7.0 | box1 | present |

Measured across the full recording (`data_prep/parse_recordings.py`'s sanity output):
environment is 99.10% of all points; sphere 0.24%, box1 0.40%, box2 0.17%, cylinder
0.09%. Human/car/bus are not rare — they are **absent**. No amount of loss weighting
or clever sampling manufactures training signal that doesn't exist; this repo's
class head is sized for the full 8-value taxonomy (so no architecture change is
needed if a future scenario adds them), but only the four present classes get
evaluated honestly. Treat human/car/bus detection as untested, not as "probably
fine."

**Two data gotchas that will silently break things if missed:**

1. **The GPU ray tracer's class labels are not exact integers.** A real sample holds
   values like `6.98863`, not a clean `7.0`. Always `np.rint()` before comparing —
   `dataset.py`'s `_load_raw` does this; a naive `intensity == 7.0` check would miss
   almost every real object point.
2. **No-return points (`inf` in x/y/z) also carry `intensity == 0.0`** — identical to
   an environment hit. They're rare (0.02% of all points, confirmed empirically) but
   must be filtered by checking `isfinite`, not by class value, or they poison every
   distance computation downstream (FPS, ball query, Chamfer distance) with NaN/inf.

## Why raw points, not the range grid

`data_prep/parse_recordings.py` (built for the CNN/VAE candidates) stores a scalar
**range** value per pixel — `sqrt(x²+y²+z²)` — not the original x/y/z. That's a
one-way transform: recovering x/y/z from range alone needs each ray's exact 3D
direction, which requires knowing this simulated sensor's precise per-ring vertical
beam angle, which isn't available. So `dataset.py` in this repo reads directly from
the **original raw recordings** (real x/y/z), not from the already-parsed range
grids — a completely separate data path from the CNN/VAE candidates, joined only by
sharing the same `sample_id` keys and the same train/val/test split
(`data/sim/{train,val,test}_ids.npy`, computed once by `data_prep/split.py` and
reused by every candidate, spatial-block split so adjacent near-duplicate frames
never land in different splits).

One clarification worth being explicit about, since it came up directly: the
`(64, 1024)` rasterization the CNN candidates use is **not** a lossy operation for
*this* sensor. The recordings' `ring` field is already in clean ring-major order —
1024 contiguous points per ring, ring 0 through ring 63, confirmed empirically before
writing any parsing code — so every one of the 65,536 points lands in exactly one
distinct grid cell, always. It's a lossless reindexing, not a projection that merges
or drops points. What raw points genuinely buy over the grid is different: explicit
3D coordinates handed to the network directly, rather than a scalar range value the
network has to implicitly combine with its position in a fixed grid to recover
geometry — and no commitment to convolution's fixed-topology assumption. What raw
points *cost*, covered next, is that PointNet++'s cost doesn't scale as gracefully
with point count as a 2D convolution does, which is why a point *budget* is a real
constraint for this candidate specifically and not for the other two.

## Point budget and real-time constraints

**Deployment target: NVIDIA Jetson AGX Orin, 64GB, JetPack 5.1.x (Ubuntu 20.04,
Python 3.8.10, CUDA 11.4.315), inference only.** Training happens entirely on a
separate machine (Quadro RTX 5000, 16GB VRAM, 64GB system RAM, Xeon Silver 4110) —
nothing in this repo trains on the Jetson, ever.

The full 65,536 points/scan is too many for PointNet++'s FPS + ball-query hierarchy
to process at any useful rate on Orin (measured, see below). The fix decided on:
**decimate the ray pattern itself** — keep every ring, keep every 2nd azimuth
column — rather than randomly dropping points or using a generic spatial heuristic
like voxel-grid downsampling. This is deliberate, not arbitrary: it's literally
simulating a coarser real sensor, and applying the identical decimation to the
Jetson's live feed at inference means there is no train/inference mismatch on "which
points does the model even see" — a property neither random subsampling nor
voxel-grid downsampling gives you for free.

### Why decimate azimuth, never rings

Empirically checked against the real recording (not assumed) — object-class frame
coverage at a few decimation patterns, against the full-resolution baseline
(sphere 581/600, cylinder 410/600, box2 521/600, box1 600/600):

| decimation | points | sphere | cylinder | box2 | box1 |
|---|---|---|---|---|---|
| every 2nd ring, every 2nd col | 16,384 | 481/600 | 271/600 | 259/600 | 542/600 |
| every ring, every 4th col | 16,384 | 424/600 | 222/600 | 413/600 | 550/600 |
| **every ring, every 2nd col** | **32,768** | **528/600 (91%)** | **316/600 (77%)** | **501/600 (96%)** | **598/600 (100%)** |

Same total point count, very different results depending on *which* axis gets cut.
Keeping full ring (vertical) resolution and only thinning azimuth clearly wins.
Reasoning after the fact: with only 64 rings total to begin with, losing rings costs
far more per class than losing azimuth columns (1024 total, much finer to start) —
small objects only span a handful of rings, so any vertical decimation risks
skipping over them entirely. This is implemented as `RING_STRIDE = 1, COL_STRIDE = 2`
in `dataset.py`, producing exactly `64 × 512 = 32,768` points
(`dataset.N_POINTS`).

**Uniform decimation of any kind still doesn't fix class imbalance** — cylinder
coverage drops from 410/600 to 316/600 even in the best decimation pattern tested.
That's an honest, physical limitation of a single recording with this few object
instances, not a sampling bug to engineer around further. A class-aware sampling
scheme that *guarantees* every object point survives was considered and explicitly
rejected: it would need ground-truth labels to decide what to keep, which don't
exist at real inference time, creating exactly the train/inference mismatch this
whole design is trying to avoid.

### The actual measured cost (encoder only)

`benchmarks/pointnet2_jetson_bench.py`, run on the real Orin (`PointNet2Encoder`,
single-scan batches — matching real deployment: one scan in, one prediction out,
repeatedly):

| points | latency | rate | peak memory |
|---|---|---|---|
| 2,048 | 242 ms | 4.1 Hz | 156 MB |
| 4,096 | 470 ms | 2.1 Hz | 156 MB |
| 8,192 | 1015 ms | 1.0 Hz | 881 MB |
| 16,384 | 2260 ms | 0.4 Hz | 3.5 GB |
| 32,768 | 5378 ms | 0.2 Hz | 14.0 GB |

None of these reach the LiDAR's real native rate (10 Hz, confirmed against the
sensor driver config, not the recording's ~1.9 Hz save cadence, which is a separate
logging choice). That turned out not to matter: the SLAM stack this robot also runs
(KISS-Matcher + spark_fast_lio, config checked directly — `local_reg.num_threads: 8`,
zero CUDA/GPU parameters anywhere in either the KISS-Matcher config or the
spark_fast_lio patch) is CPU-only and needs every scan for accurate odometry
regardless of what the classifier does — the two don't need to run at the same rate,
since SLAM doesn't consume the classifier's output. The actual binding constraint
turned out to be almost nothing: the robot moves slowly through this tunnel
(measured from the recording's own trajectory: ~0.38 m/s median), so even at 32,768
points / 0.2 Hz (5.4s between predictions), that's roughly 2m of travel between
updates against a corridor that's ~1.95m wide and ~300m long — workable, if worth
keeping an eye on.

**Chosen: 32,768 points, full ring resolution, every-2nd azimuth column.** Best
class coverage by a wide margin, and the real-time budget turned out not to force a
smaller choice.

One honest caveat: `ball_query`'s implementation changed after this benchmark was
run (see "Testing" below — two real bugs, fixed) and the benchmark script previously
had a separate, untested inline copy of the FPS/ball-query code rather than
importing the real, now-fixed version. It's been refactored to benchmark
`PointNet2Encoder` directly, so it can't drift out of sync again, but **the numbers
above should be re-confirmed with the current code before being treated as final** —
the fixes shouldn't meaningfully change timing (they replace one masked assignment
with one `torch.where`, not a different algorithm), but "shouldn't" isn't the same
as "confirmed."

## What actually deploys — only the encoder

This is not a new decision specific to this repo — it's carried over from how the
existing CNN baseline is already deployed elsewhere in this project: the decoder
exists purely to shape the encoder during training (via the training losses,
i.e. the Information Bottleneck idea — constrain what the latent captures, but push
it to capture whatever's predictive of the training signal), and gets discarded
entirely before deployment. Only `encoder.state_dict()` is ever saved or loaded for
inference; the wrapper that consumes it downstream never even imports a decoder
class.

This repo verifies that claim rather than just asserting it —
`tests/test_pointnet2_model.py::test_only_encoder_state_dict_is_needed_for_deployment`
loads *only* the encoder's weights into a fresh, decoder-less `PointNet2Encoder` and
checks it reproduces the full model's `encode()` output exactly. If that test ever
fails, something has broken the deployment story, not just a training detail.

One consequence worth stating plainly: because nothing decoder-shaped ever runs on
the Jetson, the decoders' internal structure — how many of them, how they're
built — has **zero effect on the deployment point-budget/latency numbers above**.
It only affects training cost, on the Quadro, which has headroom to spare. That's
also why the two decoders below are free to be structured completely differently
from each other; there's no deployment cost trade-off pushing them toward looking
similar.

### Why two decoders, and why they're structurally different

Both exist purely as training-time losses shaping the encoder — but they're not
doing the same job, and were deliberately built with different structure to match:

**`ReconstructionDecoder`** takes the latent vector `z` **alone** — no connection back
to the raw input points. This is the one that actually tests whether the bottleneck
is informative: if reconstruction from `z` alone is close to the real input, `z`
genuinely captured the geometry; if a decoder had shortcuts back to the raw input
(skip connections), it could reconstruct well regardless of how good or bad the
latent is, making the reconstruction-quality metric meaningless for its actual
purpose. Verified directly:
`test_reconstruction_decoder_depends_only_on_latent_not_raw_input` checks both that
the decoder's forward signature structurally accepts nothing but `z`, and that it's
a deterministic function of `z` (same `z` → same output, different `z` → different
output). Scored with **Chamfer distance** (`losses.py`) against the input cloud,
since point sets don't have a natural pixel-wise loss the way images do.

**`SegmentationDecoder`** is the standard PointNet++ feature-propagation decoder,
*with* skip connections to every encoder level. Its job isn't to test the bottleneck
in isolation — it's to give the encoder a strong, semantically rich training signal
(what class does each point belong to), and skip connections are the right call
there for the same reason they were added to the CNN candidates' segmentation heads:
a bottleneck-only decoder demonstrably smooths away small, sparse detail (observed
directly in the baseline CNN's early reconstructions — small object clusters were
visibly blurred out), and small objects are literally the classification target
here. Scored with **class-weighted cross-entropy** (`losses.py`), weights computed
as capped inverse frequency from the actual decimated training distribution — capped
in both directions, since (a) the rarest present class (cylinder) would otherwise get
a destabilizingly large weight, and (b) the *absent* classes (human/car/bus, zero
count) would otherwise produce a literal `inf` from dividing by zero, which is
silently catastrophic the moment anything touches the full weight vector even though
no individual training step ever selects an absent class as a target.

## Setup

### Training machine (Quadro RTX 5000 or equivalent)

```bash
pip install -r requirements.txt
```

Standard x86_64 CUDA PyTorch install — nothing exotic. Verify with `nvidia-smi` /
`python3 -c "import torch; print(torch.cuda.is_available())"` first.

### Jetson AGX Orin (inference only — this repo never trains here)

**`pip install torch` does not work on Jetson.** PyPI's wheels are built for
x86_64; Jetson is ARM64 (aarch64) with a CUDA build tied to the exact JetPack/L4T
version. Confirm the exact JetPack sub-version first (Ubuntu 20.04 + Python 3.8 +
CUDA 11.4.315 narrows it to the 5.1.x family, but not the exact patch):

```bash
sudo apt-cache show nvidia-jetpack | grep -i version
# or: cat /etc/nv_tegra_release
```

Then get the matching PyTorch wheel from NVIDIA's official Jetson PyTorch page for
that exact JetPack version (a generic pip/PyPI install will silently install an
incompatible x86_64 wheel or simply fail) — and a matching `torchvision` build,
which is not automatically compatible just because the torch version matches.

No custom CUDA extensions are used anywhere in this repo (`pointnet2_utils.py` is
pure PyTorch — `torch.cdist`, `torch.sort`, `torch.where`, standard indexing). This
is deliberate: compiling custom CUDA/C++ extensions against Jetson's aarch64 +
JetPack-pinned CUDA/PyTorch build is a well-known, avoidable source of Jetson setup
pain, and this project's data scale doesn't need the speed a compiled extension
would buy.

## Running the pipeline

```bash
# 1. Parse the raw scenario recordings into (range, class) grids for the CNN/VAE
#    candidates, and compute the shared spatial-block train/val/test split.
python3 data_prep/parse_recordings.py --recordings-dir /path/to/twc_scenario_.../recordings
python3 data_prep/split.py

# 2. Run the test suite (fast, CPU, synthetic data -- no GPU or real scenario
#    data needed to run this).
python3 -m pytest tests/ -v

# 3. (Jetson) confirm the point-budget benchmark still holds after any code change
python3 benchmarks/pointnet2_jetson_bench.py

# 4. (Training machine) train -- see models/pointnet2.py + dataset.py +
#    losses.py for the pieces; a train.py entry point wiring them together
#    is the next thing to build in this repo.
```

`data/` is gitignored (134MB+ of derived arrays, regenerable from the raw
recordings by step 1) — never committed, always regenerated locally.

## PointNet++ background (for the thesis writeup)

PointNet (the non-"++" predecessor) processes a point cloud with a single global
step: a shared per-point MLP, then a symmetric aggregation (max-pool) over all
points, producing one global feature vector. It's simple and permutation-invariant
(order doesn't matter, exactly as a "cloud" of points shouldn't have an order to
begin with), but has no notion of *local* structure — every point contributes to
one global pool with no sense of neighborhood.

PointNet++ (Qi et al., 2017) fixes this with a hierarchy of **set abstraction**
levels, each doing three things (`models/pointnet2_utils.py::SetAbstraction`):

1. **Sampling** — pick a subset of points to act as the next level's centroids.
   This project uses farthest-point sampling (FPS,
   `pointnet2_utils.farthest_point_sample`): iteratively pick the point farthest
   from every point already chosen, giving good spatial coverage rather than a
   biased or clustered subset.
2. **Grouping** — for each centroid, find its local neighborhood: every point
   within a fixed radius (`pointnet2_utils.ball_query`), capped at `k` neighbors.
3. **PointNet layer** — run the same simple PointNet idea (shared MLP + max-pool),
   but *locally*, within each neighborhood instead of globally — producing one
   feature vector per centroid, which becomes that centroid's representation for
   the next, coarser level.

Stacking several of these (`PointNet2Encoder` uses three: 32,768 → 4,096 → 1,024 →
256 points) builds up a genuine local-to-global hierarchy — early levels capture
fine local shape, later levels capture broader context — closer to what a CNN's
stack of convolutions does for images, adapted to unordered point sets.

For per-point tasks (here: classification), you need the reverse direction too:
**feature propagation** (`pointnet2_utils.FeaturePropagation`) interpolates a
coarser level's features back onto a denser one (inverse-distance-weighted 3-nearest-
-neighbor interpolation), concatenates that with the denser level's own saved
features from the encoder pass (the skip connection), and runs another shared MLP.
Chaining these back down through the same hierarchy in reverse
(`SegmentationDecoder`) recovers a prediction at every one of the original 32,768
input points.

## Testing

`tests/` uses a **synthetic** fixture (`tests/conftest.py`), not the real ~400MB
scenario recordings — deliberately, so anyone cloning this repo can run the full
suite without that external data present. The fixture matches the real file format
exactly (same fields, same ring-major point order, the same off-integer class-label
noise, the same inf-for-no-return convention), so passing these tests is meaningful
against the real data's actual quirks, not just against made-up shapes.

Building this test suite caught three real bugs, not test-writing mistakes — worth
recording here as a concrete case for why this level of testing is worth the time,
for the thesis writeup:

1. **`farthest_point_sample` was non-deterministic.** It picked a random starting
   point every call (`torch.randint`), meaning the encoder's output for the exact
   same input scan could differ between two consecutive calls — even in
   `model.eval()` mode, since `eval()` only changes Dropout/BatchNorm behavior, not
   an explicit random call elsewhere in the code. For a deployed perception system,
   that means the robot could report a different "reading" of an unchanged static
   scene from one frame to the next, for no reason connected to the actual scene.
   Fixed by using a fixed starting index. Caught by
   `test_encode_matches_forward_latent` and
   `test_only_encoder_state_dict_is_needed_for_deployment`.
2. **`ball_query` leaked an invalid index when a centroid had zero neighbors within
   radius.** Not a contrived edge case — it happened on ordinary random test data.
   The fallback logic's "if the closest sorted slot is itself the out-of-range
   sentinel, fall back to the sentinel" was a no-op bug; the sentinel value (equal
   to the total point count, therefore out of bounds) could flow straight into the
   returned indices, which would corrupt or crash the very next indexing operation.
   Fixed with a proper nearest-neighbor fallback. Caught by `test_ball_query_shape`.
3. **`ball_query` crashed outright when there were fewer total points than
   requested neighbors (N < k).** Slicing a sorted index array with `[:k]` when the
   array is shorter than `k` silently returns fewer than `k` elements rather than
   erroring immediately, which broke every downstream shape assumption several
   lines later with a confusing error message far from the actual cause. Doesn't
   happen on this project's real data (32,768 points, k=32 — N is never close to
   k), but is now handled explicitly (pad with the nearest-neighbor fallback)
   rather than left to fail by luck. Caught by
   `test_ball_query_handles_fewer_than_k_neighbors_without_crashing`.

Run with:
```bash
python3 -m pytest tests/ -v
```

## Known limitations / open items

- **Human/car/bus have zero training examples** in the only scenario recording
  available so far. The class head is sized to support them, but nothing has ever
  taught the network to recognize them — don't read "the model works" as covering
  these three classes.
- **No ground truth exists for real-world evaluation.** Real LiDAR has no
  `laser_retro` — there's no way to score real-world segmentation accuracy
  numerically yet. Planned mitigation (not yet implemented): qualitative visual
  inspection of predicted class maps on real scans, and/or capturing a small number
  of real scans with a known object at a known logged position to get at least a
  partial labeled real set.
- **Single-scenario data** means the spatial-block train/val/test split
  (`data_prep/split.py`) is the only thing standing between this project and
  training on data that's essentially one long, continuous, correlated recording.
  More scenarios from the same Gazebo simulation pipeline are the natural next step
  if any class turns out to need more signal than this recording can give it.
- **Jetson benchmark numbers predate the `ball_query`/`farthest_point_sample`
  fixes** and should be re-confirmed (see "Point budget" above).
- **`train.py` (wiring `dataset.py` + `losses.py` + `models/pointnet2.py` into an
  actual training loop, matching the other two candidates' shared harness) is not
  yet built** — this repo currently has all the pieces, individually tested and
  verified end-to-end together (dataset → model → both losses → backward, run
  against real recordings, not just synthetic data), but not yet a single training
  entry point.

## Project structure

```
sim2real-lidar/
├── data_prep/
│   ├── parse_recordings.py   # raw recordings -> (range, class) grids [CNN/VAE candidates]
│   └── split.py               # spatial-block train/val/test split, shared by all 3 candidates
├── models/
│   ├── pointnet2_utils.py     # FPS, ball query, SetAbstraction, FeaturePropagation
│   └── pointnet2.py           # PointNet2Encoder + both decoders + PointNet2MultiTask wrapper
├── dataset.py                  # PointCloudDataset: raw recordings -> decimated (xyz, class) pairs
├── losses.py                   # Chamfer distance, class-weighted cross-entropy
├── benchmarks/
│   └── pointnet2_jetson_bench.py  # real Jetson latency/memory measurement
├── tests/                      # synthetic-fixture unit tests, no external data needed
└── data/                        # gitignored -- regenerated locally, never committed
```
