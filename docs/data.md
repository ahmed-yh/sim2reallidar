# Data

## The recordings

76 Gazebo scenarios of a Clearpath Jackal driving the MBUT tunnel corridor (about 1.95 m wide and
125 m long). The simulated world is a mesh reconstructed from a real recording of that corridor,
so simulated and real scans come from the same geometry. The reconstructed cloud spans
124.9 x 34.6 x 6.1 m; the 34.6 m is across side branches. Each scenario places objects (spheres,
cylinders and two box types) differently along the drive.

The raw recordings are not in git. Put them at:

```
scenario_data/mbut_76scenarios/o_robot_nav/<scenario_id>/<timestamp>.npz   # x, y, z, intensity, ring
                                                     .../<timestamp>.json  # robot pose
```

That is 47,958 recordings (about 496 MB). Copy them from the group's storage with rsync or scp;
they are too large for the repository history.

## Regenerating `data/`

`data/` is derived and never synced. Rebuild it from the recordings:

```bash
python data_prep/parse_recordings.py --max-scans-per-scenario 100
python data_prep/split.py
```

Every result in the thesis used 100 scans per scenario, evenly subsampled across each drive,
which gives 7,600 scans. Leave out `--max-scans-per-scenario` and the script keeps all 47,958, so
the split and every number downstream come out different.

This writes:

| file | contents |
|---|---|
| `data/sim/scans/<sample_id>.npz` | `range` (64 x 1024 float32, `inf` for no return) and `class_` (64 x 1024 uint8) |
| `data/sim/poses.csv` | sample_id, scenario_id, time_ns, x, y, z |
| `data/sim/{train,val,test}_ids.npy` | the split, shared by all three candidates |

Both scripts are deterministic (no random numbers), so a rebuild reproduces the same files.
`sample_id` is `"{scenario_id}__{timestamp}"`: wall-clock timestamps collide across scenarios
(19 collisions in the 76-scenario batch), so the timestamp alone would silently drop scans.

For the CNN autoencoder, also run

```bash
python kevin_pipeline/prepare_data.py
```

which writes range grids in Kevin Fischer's format to `data/kevin_range_npy/`, and, under
`outputs/kevin_cnn/`, index files that make his training script use this repo's split instead of
its own random one. It also computes `max_range` (14.425 m, the 99th percentile of the training
split's ranges, with his formula) into `outputs/kevin_cnn/normalization_stats.json`. SalsaNext and
the comparison script normalize with the same value.

### Which candidate reads what

`parse_recordings.py` stores one scalar range per pixel. That cannot be inverted to x, y, z
without each ray's exact beam angle, which is not recorded. So:

- PointNet++ (`dataset.py`) and SalsaNext (`salsanext_dataset.py`) read the raw recordings for
  their coordinates, and take only the sample ids and the split from `data/sim/`.
- The CNN autoencoder reads the range grids in `data/kevin_range_npy/`.

## Scan format and labels

Each scan is 65,536 points (`x, y, z, intensity, ring`): 64 rings x 1024 columns, stored ring by
ring, so `reshape(64, 1024)` is an exact rasterization with one point per cell. The class label
rides in `intensity`, set by Gazebo's `laser_retro`:

| raw value | class | share of all points |
|---|---|---|
| 0.0 | environment | 99.10%, and every no-return point |
| 1.0 / 2.0 / 3.0 | human / car / bus | none at all |
| 4.0 | sphere | 0.24% |
| 5.0 | cylinder | 0.09% (the rarest class present) |
| 6.0 | box2 | 0.17% |
| 7.0 | box1 | 0.40% |

Human, car and bus have no examples. The class heads are sized for all 8 values so a future
scenario needs no architecture change, but nothing has ever been trained to recognize those three.

Models see only x, y, z. Intensity is excluded everywhere because in simulation it is the label.

### Two traps

1. Labels are not exact integers. The GPU ray tracer yields values such as `6.98863`, so a check
   like `intensity == 7.0` misses almost every object point. Round with `np.rint()` first, as
   `dataset.py::_load_raw` does.
2. No-return points carry `intensity == 0.0`, the same value as an environment hit, with `inf` in
   x, y, z. Filter them with `np.isfinite` and never by class. Left in, they put NaN and inf into
   farthest point sampling, the ball query and the Chamfer distance.

## The split

`data_prep/split.py` cuts each scenario's drive into blocks of 20 consecutive scans (about 10 s
of driving) and deals the blocks out in a repeating cycle of 14 train, 3 validation and 3 test
blocks. The cycle counter runs on across scenarios instead of restarting in each one, because a
capped scenario only has about 5 blocks. No block is ever split.

With 100 scans per scenario this gives 5,320 / 1,140 / 1,140 scans, and per scenario:

| scenarios | in |
|---|---|
| 38 | train only |
| 19 | train and validation |
| 19 | validation and test |
| 0 | train and test |

So the test set is 19 scenarios that never appear in training: it measures generalization to
unseen object placements in the same tunnel. Validation and test share those 19 scenarios, with
the validation blocks just before the test blocks. Early stopping therefore picks its checkpoint
on frames that lie next to test frames. The effect is expected to be small but is not zero; a
retraining could hold validation out at the scenario level instead.

## Point budget: why 32,768 points

65,536 points per scan is too costly for PointNet++'s sampling and grouping on the Jetson (see
the [compute budget](deployment.md#compute-budget-on-the-jetson)). The budget comes from
decimating the ray pattern itself: every ring is kept, and every second azimuth column. In effect
this simulates a coarser sensor, and with the same decimation on the live feed, training and
inference agree on which points the model sees. It is `RING_STRIDE = 1,
COL_STRIDE = 2` in `dataset.py`, so 64 x 512 = 32,768 points. Only PointNet++ is decimated; the
range-image candidates read the full grid.

Measured per-class frame coverage on the 600-frame recording used for this check (how many frames
still contain at least one point of the class):

| pattern | points | sphere | cylinder | box2 | box1 |
|---|---|---|---|---|---|
| full resolution (baseline) | 65,536 | 581 | 410 | 521 | 600 |
| every 2nd ring, every 2nd column | 16,384 | 481 | 271 | 259 | 542 |
| every ring, every 4th column | 16,384 | 424 | 222 | 413 | 550 |
| **every ring, every 2nd column** | **32,768** | **528** | **316** | **501** | **598** |

At the same point count, cutting rings loses far more than cutting columns. With only 64 rings a
small object spans a handful of them, while azimuth starts at 1,024 columns. Uniform decimation
still costs the cylinder the most. A class-aware scheme that guarantees object points survive was
rejected: it needs labels to decide what to keep, and there are none at inference time.

For this sensor the 64 x 1024 grid is a lossless reindexing of the points, so both
representations hold the same information. With raw points the network gets the 3D coordinates
directly and does not have to work them out from a range and a grid position. The price is the
point budget, which the convolutional candidates can do without.
