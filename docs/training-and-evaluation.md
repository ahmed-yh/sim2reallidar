# Training and evaluation

Before training, rebuild `data/` as described in [data.md](data.md), including
`kevin_pipeline/prepare_data.py`: SalsaNext and the comparison script read `max_range` from the
`outputs/kevin_cnn/normalization_stats.json` it writes. Training ran on an NVIDIA Quadro RTX 5000.

```bash
pip install -r requirements.txt            # training and tests
pip install -r requirements-pipeline.txt   # adds pytorch_msssim (comparison), rosbags, video, demo rendering
```

## Training each candidate

```bash
python train.py              # PointNet++  -> outputs/pointnet2/
python train_salsanext.py    # SalsaNext   -> outputs/salsanext/
```

Each run writes to its output folder:

| file | contents |
|---|---|
| `best_encoder.pt` | encoder weights only: the artifact that deploys |
| `best_multitask_model.pt` | encoder and both decoders, for evaluation, visualization and warm starts |
| `train_metrics.json` | final scalars (best validation loss, held-out losses, class counts, arguments) |

Both checkpoints are saved as soon as a new best validation loss appears, so a crash keeps the
best model so far. Per-epoch losses are only printed; pipe the run through
`tee` if you want the history (`python train.py | tee train_pointnet2.log`; `*.log` is
gitignored).

The CNN autoencoder trains with Kevin Fischer's own script in his repository
(`masterarbeit_kevinfischer/lidar_preprocessing/autoencoder_training/train_ae.py`). Point it at
the data `prepare_data.py` produced: `--data-dir data/kevin_range_npy --outputs-dir
outputs/kevin_cnn` (paths relative to this repo). The index files there make it use this repo's
split.

### Configuration used for the thesis results

| | PointNet++ | SalsaNext | CNN autoencoder |
|---|---|---|---|
| Training code | `train.py` (shared loop in `train_utils.py`) | `train_salsanext.py` (same loop) | its author's script |
| Input | 32,768 points (x, y, z) | 64 x 1024 grid | 64 x 1024 range grid |
| Reconstruction loss | Chamfer distance to 4,096 sampled input points | range MSE over cells with a return | set in its author's script |
| Classification loss | class-weighted cross-entropy | class-weighted cross-entropy over cells with a return | none (no classifier) |
| Optimizer | Adam | Adam | set in its author's script |
| Learning rate | 1e-4 | 1e-3 | about 1.2e-4 |
| Batch size | 2 | 16 | 32 |
| Maximum epochs | 100 | 100 | 89 |
| Early-stopping patience | 10 | 8 | 8 |
| Selected checkpoint | lowest validation loss, epoch 39 | lowest validation loss | set in its author's script |

The PointNet++ and SalsaNext values are the scripts' defaults. Class weights are inverse class
frequency on the training split, capped at 100 (`--weight-cap`): without the cap the rarest present
class gets a destabilizing weight, and the three absent classes would divide by zero. Early
stopping watches the total validation loss and counts an epoch as an improvement only when the
loss drops by more than 1e-4. No random seed is fixed, so a retrain gives close but not identical
numbers.

The PointNet++ checkpoint behind every thesis result is
`outputs/pointnet2_checkpoints/epoch39_multitask_model.pt` (not in git). Its encoder is the shipped
`weights/pointnet2_epoch39_encoder.pt`.

### Fine-tuning with measured real-sensor noise

```bash
python train.py --init-from outputs/pointnet2_checkpoints/epoch39_multitask_model.pt \
    --augment-noise --outputs-dir outputs/pointnet2_ringnoise_finetuned
```

`--augment-noise` applies the per-ring dropout measured on the real Ouster
(`real_noise_augment.py`) to training samples only. `--init-from` copies the weights but not the
optimizer state, so the run starts warm with a fresh optimizer. The existing fine-tune in
`outputs/pointnet2_ringnoise_finetuned` was trained before the dropout array was flipped into
simulator ring order (2026-09-28) and should be redone; see [continuing.md](continuing.md).

### Shipping a newly trained encoder

Copy the run's `best_encoder.pt` into `weights/` (the only folder where `.pt` files are tracked),
point `deploy.DEFAULT_WEIGHTS` and `tests/test_shipped_weights.py` at the new file, and run the
tests.

## Why the two PointNet++ decoders differ

The decoders only shape the encoder during training and are discarded before deployment.

- `ReconstructionDecoder` sees the latent alone, with no path back to the input. Skip connections
  would let it reconstruct well from a poor latent, so leaving them out turns reconstruction
  quality into a test of the 256-d bottleneck.
  `test_reconstruction_decoder_depends_only_on_latent_not_raw_input` checks this.
- `SegmentationDecoder` is the standard feature-propagation decoder, with skip connections to
  every encoder level, and gives the encoder a rich training signal. A decoder that only sees the
  bottleneck smooths away small sparse objects, and those are the classification target.

## Comparing the three candidates

The per-run `train_metrics.json` files are not comparable with each other: PointNet++ reports a
Chamfer distance in metres, SalsaNext and the CNN a normalized MSE. `kevin_pipeline/fair_comparison.py`
is the script that puts all three on one scale. It reprojects PointNet++'s free-form output onto
the real ray grid and scores every candidate with the same normalized MSE and SSIM, on the same
test ids, with the same `max_range`. It also sweeps ten confidence thresholds for both
classifiers in one forward pass each.

```bash
python kevin_pipeline/fair_comparison.py --json-out outputs/measured/fair_comparison.json
```

Defaults: PointNet++ from `outputs/pointnet2_checkpoints/epoch39_multitask_model.pt`, SalsaNext
from `outputs/salsanext/best_multitask_model.pt`, the CNN from `outputs/kevin_cnn/`. The demo
builds its comparison table from this JSON file.

PointNet++ per-class metrics and threshold sweeps:

```bash
python evaluate.py --checkpoint outputs/pointnet2_checkpoints/epoch39_multitask_model.pt \
    --class-conf-threshold 0.97 --json-out outputs/measured/evaluate_pointnet2.json \
    --sweep "0,0.3,0.5,0.7,0.8,0.9,0.95,0.97,0.99,0.995"
```

### The confidence threshold

A point keeps its predicted object class only when its softmax confidence clears the threshold;
otherwise it counts as environment. The threshold changes what is reported and drawn and leaves
the weights alone. It exists because raw argmax scatters low-confidence object predictions across
the floor of real scans. Real data has no labels, so the value was chosen on the labelled
simulated test split:

| threshold | PointNet++ macro-F1 | SalsaNext macro-F1 |
|---|---|---|
| 0 (argmax) | 68.1% | 67.7% |
| 0.70 | 78.7% | 82.3% |
| 0.90 | 84.4% | 87.5% |
| **0.97** | **86.7%** | **88.6%** |
| 0.99 | 84.6% | 85.6% |

Macro-F1 is over the four object classes; overall accuracy is close to 99% for both models and says
little, because 99.1% of points are environment. `ros2_inference/_shared_inference.py` applies the
same rule at inference (`class_conf_threshold=0.97`).

## Tests

```bash
python -m pytest tests/ -q
```

`tests/conftest.py` builds synthetic scans in the real format, with the recordings' fields,
ring-major order, off-integer labels and `inf` for no return. The suite needs no data and no GPU,
and runs in under a minute on a laptop CPU. It covers the dataset loader, the
losses, the model and its sampling utilities, the Ouster/simulator order conversion, the real-scan
input path, and the shipped encoder weights.

Writing the suite found four bugs in code that already ran without errors:

1. `farthest_point_sample` started from a random index, so the same scan could encode
   differently on two calls, even in `eval()` mode. Fixed with a fixed start index.
2. `ball_query` leaked an out-of-bounds sentinel index when a centroid had no neighbour in its
   radius, which happens on ordinary data.
3. `ball_query` crashed when a scan had fewer points than the neighbour count k.
4. Every norm layer was `BatchNorm`, which fails at batch size 1 in training mode. Replaced with
   `GroupNorm`.
