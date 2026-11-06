# Thesis figures

`make_thesis_figures.py` draws the thesis charts from project data:

```bash
python thesis_figures/make_thesis_figures.py
```

A figure whose source is not on the machine is skipped. The copies in the thesis were relabelled
for the document, so some label text differs from these PNGs; the plotted values are the same.

| figure | in the thesis | source |
|---|---|---|
| `fig1_drift_fix_comparison.png` | yes | summary statistics of the SLAM drift diagnosis, typed into `DRIFT` in the script (the per-message logs live with the SLAM pipeline, not here) |
| `fig2_reconstructed_pointcloud.png` | yes | `noslil.ply` at the repo root (not in git), measured at run time |
| `fig3_encoder_comparison.png` | yes | `outputs/measured/fair_comparison.json` and `outputs/measured/evaluate_pointnet2.json` |
| `fig5_decimation_validation.png` | yes | per-class frame coverage at full resolution and for three decimation patterns on the 600-frame recording, typed into `DECIMATION` |
| `fig6_confidence_threshold.png` | as a table | `outputs/measured/evaluate_pointnet2_sweep.json` (`evaluate.py --sweep`) |
| `fig7_classifier_head_to_head.png` | as a table | `outputs/measured/fair_comparison.json` |

The corridor marker in fig1 is the reconstructed corridor length, about 125 m. The figure has no
per-step time series or X-Y trajectory, because those need per-message logs that are not in this
repository.

In fig2, the raw bounding box of `noslil.ply` is 315.9 x 100.5 x 18.7 m, but 99% of the cloud
lies within 124.9 x 34.6 x 6.1 m, so the views are cropped to the 0.1 to 99.9th percentile. The
voxel downsample uses 20 cm because the export is already thinned to about 10 cm spacing.

The SSIM axis in fig3 starts at 0.940, since all three candidates fall between 0.952 and 0.972,
and the CNN autoencoder shows "no classifier" where a zero bar would suggest it got everything
wrong. In fig5 the full-resolution row is the baseline the three decimation patterns are measured
against.

`existing_imagery/` holds copies of images the project produced elsewhere (scenario maps,
playback frames, reconstruction views, the real-world placement plan), renamed after the thesis
section they were collected for. `5.7.3_real_bag_prediction_at_threshold_0.97.png` is a demo
frame from before the input-order fix and shows the broken input; the thesis uses a corrected
frame instead.
