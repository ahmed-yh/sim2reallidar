"""Shared per-model inference logic for the Ouster OS1 pipeline, used by
BOTH live_inference_node.py (ROS2 topic subscription, real-time) and
run_inference_on_bag.py (offline bag reading, no ROS2 needed) -- one
implementation of "given a (64,1024,3) scan, run this model and render a
visualization", not duplicated between the two entry points.

Input is (x, y, z) ONLY, matching every training-side dataset loader in
this repo (dataset.py, salsanext_dataset.py, Kevin's denkwelt_deploy/
dataset.py all exclude intensity/reflectivity specifically to avoid leaking
the simulation-only class label into the model's input).

Real-data visualizations only ever have 2-3 rows, never 4: there's no
ground-truth class on real LiDAR (no laser_retro), so only original range
(real, measured), reconstructed range, and (for pointnet2/salsanext)
PREDICTED class are shown -- never a ground-truth class row to compare
against, unlike the simulated-data playbacks in viz/.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "viz"))

from playback_common import class_rgb, label_row, range_rgb  # noqa: E402

# NOT orient()-ed like the simulated-data playbacks (viz/): that flip was
# verified against the SIMULATED sensor's ring order ("ring 0 = lowest,
# ground-facing beam"), which is a property of the simulation's LiDAR
# plugin, not a law of physics. Checked directly against this project's own
# real Ouster OS1-64 bags (mean elevation angle per ring, see project notes,
# 2026-09-14): ring 0 is the HIGHEST beam (~+21 deg) and ring 63 the LOWEST
# (~-17 deg) -- the OPPOSITE of the simulated convention. A raw (ring, col)
# grid rendered row-0-at-top is therefore ALREADY correct for real data;
# applying the simulated flip on top of it was upside down.

EXPECTED_RINGS = 64
EXPECTED_COLS = 1024


class ModelRunner:
    def __init__(self, model_name: str, checkpoint_path: Path, max_range: float,
                 class_conf_threshold: float = 0.7):
        """class_conf_threshold: the classifier's argmax always picks SOME
        class, even when its softmax confidence is barely above chance --
        on real (domain-shifted) data this shows up as speckled false
        positives across the ~99% of points that are legitimately
        environment, since even a small per-point error rate is a lot of
        pixels at 64x1024. Below this threshold, a point/pixel displays as
        environment instead of its low-confidence argmax class. Purely a
        DISPLAY decision -- does not change the model or its weights. Set
        to 0 to see the raw, unthresholded argmax (what the model would
        literally act on downstream)."""
        if model_name not in ("pointnet2", "salsanext", "kevin_cnn"):
            raise ValueError(f"unknown model {model_name!r} (expected pointnet2 | salsanext | kevin_cnn)")
        self.model_name = model_name
        self.max_range = max_range
        self.class_conf_threshold = class_conf_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_name, checkpoint_path)

    def _load_model(self, name: str, checkpoint_path: Path):
        if name == "pointnet2":
            from models.pointnet2 import PointNet2MultiTask
            model = PointNet2MultiTask()
        elif name == "salsanext":
            from models.salsanext import SalsaNextMultiTask
            model = SalsaNextMultiTask()
        else:
            build_autoencoder, kevin_deploy = self._load_kevin_builder()
            with open(kevin_deploy / "outputs" / "best_params.json") as f:
                best = json.load(f)
            params = best["params"] if "params" in best else best
            model = build_autoencoder(params)

        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        return model.to(self.device).eval()

    @staticmethod
    def _load_kevin_builder():
        """Kevin's deploy dir has its own `models` package (autoencoder.py,
        custom_cnn.py) with the SAME top-level name as this repo's own
        models/ package -- both can't be `import models` in the same
        process without one shadowing the other depending on import order
        (confirmed: whichever gets imported first wins, silently). Loading
        Kevin's under an alias via importlib sidesteps the collision
        entirely rather than relying on sys.path ordering luck."""
        kevin_deploy = (REPO_ROOT.parent / "masterarbeit_kevinfischer" / "lidar_preprocessing"
                         / "autoencoder_training" / "denkwelt_deploy")
        if "kevin_denkwelt_models" not in sys.modules:
            models_dir = kevin_deploy / "models"
            spec = importlib.util.spec_from_file_location(
                "kevin_denkwelt_models", models_dir / "__init__.py",
                submodule_search_locations=[str(models_dir)],
            )
            kevin_models = importlib.util.module_from_spec(spec)
            sys.modules["kevin_denkwelt_models"] = kevin_models
            spec.loader.exec_module(kevin_models)
        autoencoder_mod = importlib.import_module("kevin_denkwelt_models.autoencoder")
        return autoencoder_mod.build_autoencoder, kevin_deploy

    @torch.no_grad()
    def run(self, xyz_grid: np.ndarray) -> tuple[np.ndarray, float]:
        """xyz_grid: (64,1024,3) float32, NaN (or non-finite) for no-return.
        Returns (combined visualization image (H,W,3) uint8, reconstruction
        MSE in the shared normalized units)."""
        if self.model_name == "pointnet2":
            return self._run_pointnet2(xyz_grid)
        if self.model_name == "salsanext":
            return self._run_salsanext(xyz_grid)
        return self._run_kevin(xyz_grid)

    def _run_pointnet2(self, xyz_grid: np.ndarray) -> tuple[np.ndarray, float]:
        # Same azimuth-stride-2 decimation dataset.py's _load_raw uses, but
        # WITHOUT the training-time random padding to a fixed point count --
        # that padding exists only to keep a training BATCH's tensor shapes
        # uniform; a single scan has no such constraint.
        xyz = xyz_grid[:, ::2].reshape(-1, 3)
        finite = np.isfinite(xyz).all(axis=-1)
        xyz_in = xyz[finite]
        x = torch.from_numpy(xyz_in).unsqueeze(0).to(self.device)
        out = self.model(x)
        recon = out["recon_points"][0]  # (2048, 3), free-form -- needs reprojection to plot

        range_native = np.linalg.norm(xyz_grid, axis=-1)
        valid = np.isfinite(range_native)
        xyz_valid = torch.from_numpy(xyz_grid[valid]).float().to(self.device)
        dists = torch.cdist(xyz_valid, recon)
        nn_range = recon[dists.argmin(dim=1)].norm(dim=-1).cpu().numpy()
        recon_range_grid = np.full_like(range_native, np.nan)
        recon_range_grid[valid] = nn_range

        # class_logits is per INPUT point (the decimated, stride-2 set), not
        # per native cell -- scatter back to the columns actually classified
        # via the SAME `finite` mask xyz_in was built from, leaving the odd
        # (never-classified) columns dark rather than fabricating a value.
        probs = torch.softmax(out["class_logits"][0], dim=-1)
        conf, pred_cls_dec = probs.max(dim=-1)
        pred_cls_dec = torch.where(conf >= self.class_conf_threshold, pred_cls_dec,
                                    torch.zeros_like(pred_cls_dec))
        pred_cls_dec = pred_cls_dec.cpu().numpy()
        pred_grid_dec = np.zeros((EXPECTED_RINGS, EXPECTED_COLS // 2), dtype=np.uint8)
        pred_grid_dec.reshape(-1)[finite] = pred_cls_dec
        pred_grid = np.repeat(pred_grid_dec, 2, axis=1)

        real_norm = np.clip(range_native[valid], 0, self.max_range) / self.max_range
        pred_norm = np.clip(nn_range, 0, self.max_range) / self.max_range
        mse = float(np.mean((real_norm - pred_norm) ** 2))

        gap = np.full((3, EXPECTED_COLS, 3), 10, dtype=np.uint8)
        combined = np.concatenate([
            label_row(range_rgb(range_native), "ORIGINAL RANGE (real, measured)"), gap,
            label_row(range_rgb(recon_range_grid), "RECONSTRUCTED RANGE"), gap,
            label_row(class_rgb(pred_grid), "PREDICTED CLASS (no ground truth on real data)"),
        ], axis=0)
        return combined, mse

    def _run_salsanext(self, xyz_grid: np.ndarray) -> tuple[np.ndarray, float]:
        range_ = np.linalg.norm(xyz_grid, axis=-1)
        valid = np.isfinite(range_) & (range_ > 1e-3)  # (0,0,0)-for-no-return safety net, see module note
        range_n = np.where(valid, np.clip(range_, 0, self.max_range) / self.max_range, 0.0)
        x_n = np.where(valid, np.clip(xyz_grid[..., 0], -self.max_range, self.max_range) / self.max_range, 0.0)
        y_n = np.where(valid, np.clip(xyz_grid[..., 1], -self.max_range, self.max_range) / self.max_range, 0.0)
        z_n = np.where(valid, np.clip(xyz_grid[..., 2], -self.max_range, self.max_range) / self.max_range, 0.0)
        inp = np.stack([range_n, x_n, y_n, z_n], axis=0).astype(np.float32)
        x = torch.from_numpy(inp).unsqueeze(0).to(self.device)
        out = self.model(x)
        recon_n = out["recon_range"][0].cpu().numpy()
        probs = torch.softmax(out["class_logits"][0], dim=0)
        conf, pred_cls = probs.max(dim=0)
        pred_cls = torch.where(conf >= self.class_conf_threshold, pred_cls, torch.zeros_like(pred_cls))
        pred_cls = pred_cls.cpu().numpy()

        orig_display = np.where(valid, range_, np.nan)
        recon_display = np.where(valid, recon_n * self.max_range, np.nan)
        mse = float(np.mean(((range_n - recon_n) ** 2)[valid])) if valid.any() else 0.0

        gap = np.full((3, EXPECTED_COLS, 3), 10, dtype=np.uint8)
        combined = np.concatenate([
            label_row(range_rgb(orig_display), "ORIGINAL RANGE (real, measured)"), gap,
            label_row(range_rgb(recon_display), "RECONSTRUCTED RANGE"), gap,
            label_row(class_rgb(pred_cls), "PREDICTED CLASS (no ground truth on real data)"),
        ], axis=0)
        return combined, mse

    def _run_kevin(self, xyz_grid: np.ndarray) -> tuple[np.ndarray, float]:
        range_ = np.linalg.norm(xyz_grid, axis=-1)
        valid = np.isfinite(range_) & (range_ > 1e-3)
        range_n = np.where(valid, np.clip(range_, 0, self.max_range) / self.max_range, 0.0).astype(np.float32)
        x = torch.from_numpy(range_n).unsqueeze(0).unsqueeze(0).to(self.device)
        recon, _z = self.model(x)
        recon_n = recon[0, 0].cpu().numpy()

        orig_display = np.where(valid, range_, np.nan)
        recon_display = np.where(valid, recon_n * self.max_range, np.nan)
        mse = float(np.mean(((range_n - recon_n) ** 2)[valid])) if valid.any() else 0.0

        gap = np.full((3, EXPECTED_COLS, 3), 10, dtype=np.uint8)
        combined = np.concatenate([
            label_row(range_rgb(orig_display), "ORIGINAL RANGE (real, measured)"), gap,
            label_row(range_rgb(recon_display), "RECONSTRUCTED RANGE"),
        ], axis=0)
        return combined, mse
