/* Comparison metrics.
 *
 * provenance.source is "manual" because these were transcribed from a recorded
 * run of kevin_pipeline/fair_comparison.py rather than regenerated at build
 * time; the page renders that fact as a footnote. Milestone B5 adds --json-out
 * to that script so this file becomes generated and the footnote flips.
 *
 * IMPORTANT: these MSE values come from fair_comparison.py, which puts all
 * three candidates on one shared scale (PointNet++'s free-form point output is
 * reprojected onto the real ray grid and scored with Kevin's own normalized
 * formula). Do NOT substitute the numbers from the per-run train_metrics.json
 * files under outputs: PointNet++'s recon loss there is a Chamfer distance in
 * metres while SalsaNext's and Kevin's are normalized MSE. Those are not
 * comparable, and tabling them side by side would be wrong by roughly three
 * orders of magnitude.
 */
DEMO.register("metrics", "main", {
  "schemaVersion": 1,

  "provenance": {
    "source": "manual",
    "script": "kevin_pipeline/fair_comparison.py",
    "testSamples": 1140,
    "split": { "train": 5320, "val": 1140, "test": 1140 },
    "scenarios": 76,
    "scans": 7600,
    "grid": [64, 1024],
    "sharedMaxRange": 14.425209045410156,
    "classConfThreshold": 0.97
  },

  "models": [
    {
      "id": "pointnet2",
      "name": "PointNet++",
      "family": "Point-based encoder",
      "winner": true,
      "hasClassifier": true,
      "latentDim": 256,
      "metrics": { "mse": 0.000171, "ssim": 0.9716, "pointAcc": 0.9896 },
      "notes": "Consumes 32,768 raw xyz points and emits 2,048 free-form points. Its reconstruction is reprojected onto the real ray grid before scoring, so this MSE survives an extra approximation step the other two don't need — the gap is, if anything, understated.",
      "playerId": "pointnet2_sim"
    },
    {
      "id": "kevin_cnn",
      "name": "CNN autoencoder",
      "family": "Range-image CNN",
      "hasClassifier": false,
      "latentDim": 256,
      "metrics": { "mse": 0.000664, "ssim": 0.9569, "pointAcc": null },
      "notes": "Reconstruction only — it was never built with a classification head, so it is excluded from the accuracy comparison rather than given an invented number.",
      "playerId": "kevin_sim"
    },
    {
      "id": "salsanext",
      "name": "SalsaNext",
      "family": "Range-image CNN",
      "hasClassifier": true,
      "latentDim": 256,
      "metrics": { "mse": 0.001288, "ssim": 0.9524, "pointAcc": 0.9893 },
      "notes": "Outputs on the native (64,1024) grid, so no reprojection is needed. An early version scored 0.0118 MSE until an AdaptiveAvgPool2d before the latent projection was found to be discarding the positional information only the reconstruction decoder needed — fixing that improved reconstruction 9.2x.",
      "playerId": "salsanext_sim"
    }
  ],

  "comparison": {
    "columns": [
      { "key": "metrics.mse", "label": "Test MSE (normalized)", "format": "fixed6",
        "lowerIsBetter": true, "chart": "bar" },
      // No bar on SSIM deliberately: all three land in 0.952-0.972, so a
      // zero-based bar reads as three identical full bars, and a zoomed axis
      // would dramatise a difference that is genuinely small. The number
      // alone is the honest presentation; MSE is where the real spread is.
      { "key": "metrics.ssim", "label": "SSIM", "format": "fixed4",
        "lowerIsBetter": false },
      { "key": "metrics.pointAcc", "label": "Per-point class accuracy", "format": "percent2",
        "lowerIsBetter": false, "nullText": "no classifier" }
    ]
  },

  "perClass": {
    "model": "pointnet2",
    "threshold": 0.97,
    "macroF1": 0.867,
    "rows": [
      { "classId": 4, "name": "sphere",   "precision": 0.951, "recall": 0.810, "f1": 0.875 },
      { "classId": 5, "name": "cylinder", "precision": 0.781, "recall": 0.808, "f1": 0.794 },
      { "classId": 6, "name": "box2",     "precision": 0.897, "recall": 0.865, "f1": 0.881 },
      { "classId": 7, "name": "box1",     "precision": 0.950, "recall": 0.886, "f1": 0.917 }
    ]
  }
});
