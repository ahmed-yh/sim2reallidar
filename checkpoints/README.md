# Trained checkpoints

Every trained model of the thesis except the CNN autoencoder baseline, mirrored from `outputs/` (which git ignores). To put them where the scripts expect them:

```bash
python checkpoints/join_parts.py
```

This joins the SalsaNext files, which are stored as 90 MB parts because GitHub refuses files over 100 MB. It then copies everything into `outputs/` and checks each checkpoint against `SHA256SUMS`.

| Folder | What it is | Settings |
|---|---|---|
| `pointnet2_checkpoints/` | **PointNet++ epoch 39, the model behind every thesis result** (simulation benchmark and real-bag inference). `epoch39_encoder.pt` is the same file as `weights/pointnet2_epoch39_encoder.pt`. | 76 scenarios; Adam, lr 1e-4, batch 2, λ_class 1.0, Chamfer target 4,096 points, class-weight cap 100 |
| `salsanext/` | SalsaNext reimplementation, the second candidate | 76 scenarios; lr 1e-3, batch 16, up to 100 epochs, patience 8; best val loss 0.0431 |
| `pointnet2_noise_finetuned/` | Epoch 39 fine-tuned with synthetic real-sensor noise | lr 3e-5, up to 15 epochs, patience 5 |
| `pointnet2_ringnoise_finetuned/` | Epoch 39 fine-tuned with the measured per-ring dropout | Same as above. The dropout profile was applied upside down (fixed in `real_noise_augment.py` on 2026-09-28), so treat this run as inconclusive |
| `salsanext_noise_finetuned/` | SalsaNext fine-tuned with synthetic noise | lr 3e-5, batch 16, up to 40 epochs, patience 10 |
| `pointnet2_tuned/` | Early PointNet++ experiment on the single original scenario | λ_class 0.3, Chamfer target 6,144, up to 25 epochs |
| `kevin_cnn/` | Normalisation constant and split indices of the range-image dataset (`kevin_pipeline/prepare_data.py`), needed by SalsaNext and the CNN path of the inference scripts. No weights. | |

Each training folder keeps its `train_metrics.json`.

## Continuing training instead of starting from zero

Restore the files first, then point the training script at a `*_multitask_model.pt`:

```bash
python checkpoints/join_parts.py
python train.py --init-from outputs/pointnet2_checkpoints/epoch39_multitask_model.pt --outputs-dir outputs/pointnet2_continued
python train_salsanext.py --init-from outputs/salsanext/best_multitask_model.pt --outputs-dir outputs/salsanext_continued
```

What carries over and what doesn't:
- **Carries over:** every trained weight, encoder and both decoders. That is how the noise fine-tunes above were started from epoch 39.
- **Restarts:** the optimizer state (Adam's momentum and variance) was never saved, so it starts fresh. Epoch numbering and early stopping also restart at 1.

In practice this continues improving the checkpoint, but the run is not byte-identical to one that was never interrupted. For a fine-tune, use a lower learning rate than the original run, e.g. `--lr 3e-5` as in the fine-tunes.

In every folder:
- `*_encoder.pt` is the encoder alone, which is all that deploys.
- `*_multitask_model.pt` adds the decoders (reconstruction and segmentation) and is what evaluation and the inference videos load.

Both noise fine-tunes count as inconclusive: they were judged on real scans fed in the wrong point order, and the ring-noise run used the dropout profile upside down. [docs/continuing.md](../docs/continuing.md) has redoing them as an open item. The README's section 6 covers the training options.
