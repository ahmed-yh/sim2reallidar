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

# The real-scan input path lives in deploy.py (numpy + torch only), so the robot
# can import it without the rendering dependencies below. Re-exported here.
from deploy import MIN_VALID_POINTS, SkipScan, real_scan_to_points  # noqa: E402,F401
from playback_common import class_rgb, label_row, range_rgb  # noqa: E402
from sensor import N_COLS, N_RINGS, SIM_RANGE_M, check_sim_order, ouster_to_sim_order  # noqa: E402

# DISPLAY: not orient()-ed like the simulated-data playbacks (viz/): that
# flip was verified against the SIMULATED sensor's ring order ("ring 0 =
# lowest, ground-facing beam"), a property of the simulation's LiDAR plugin,
# not a law of physics. On the real Ouster OS1-64 ring 0 is the HIGHEST beam
# (+21.6 deg) and ring 63 the LOWEST (-21.2 deg, /ouster/metadata) -- the
# OPPOSITE convention -- so a raw (ring, col) grid rendered row-0-at-top is
# already the right way up for real data.
#
# MODEL INPUT: the same mismatch matters far more there. PointNet++ groups
# neighbourhoods by array index, so real scans are put into simulator order
# (sensor.ouster_to_sim_order) before they reach it -- see
# deploy.real_scan_to_points. The SalsaNext and Kevin CNN paths below still
# get Ouster order; they were out of scope for that fix (2026-09-28).


class ModelRunner:
    def __init__(self, model_name: str, checkpoint_path: Path, max_range: float,
                 class_conf_threshold: float = 0.97):
        """class_conf_threshold: below this softmax confidence, a point
        displays as environment instead of its argmax class. Purely a
        DISPLAY decision -- does not change the model or its weights. Set
        to 0 for the raw argmax (what a downstream consumer would act on).

        0.97 is the macro-F1 peak over the object classes, measured on the
        labeled test split; see this commit's message for the full sweep."""
        if model_name not in ("pointnet2", "salsanext", "kevin_cnn"):
            raise ValueError(f"unknown model {model_name!r} (expected pointnet2 | salsanext | kevin_cnn)")
        self.model_name = model_name
        self.max_range = max_range
        self.class_conf_threshold = class_conf_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_name, checkpoint_path)
        self._order_checked = False

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

    def run(self, xyz_grid: np.ndarray) -> tuple[np.ndarray, float]:
        """xyz_grid: (64,1024,3) float32, NaN (or non-finite) for no-return.
        Returns (combined visualization image (H,W,3) uint8, reconstruction
        MSE in the shared normalized units).

        Kept as a 2-tuple so live_inference_node.py is unaffected; callers that
        want the per-class breakdown use run_detailed()."""
        d = self.run_detailed(xyz_grid)
        return d["image"], d["mse"]

    # no_grad lives HERE, not on run(): run_detailed is the real entry point
    # and the demo exporter calls it directly. On run() alone, a direct
    # run_detailed() call would build a graph for every frame and grow memory
    # across a few thousand of them.
    @torch.no_grad()
    def run_detailed(self, xyz_grid: np.ndarray) -> dict:
        """As run(), plus "pred_class_counts": {class_id: n_points} over the
        THRESHOLDED predictions, or None for a model with no classifier.
        PointNet++ also returns "points" (n, 3) -- the classified points,
        sensor frame -- with their argmax class "pred" and softmax "conf",
        and raises SkipScan for a scan it declines to classify.

        These are predictions, not ground truth -- real LiDAR carries no
        laser_retro, so there is nothing to check them against. Anything
        displaying them has to say so.
        """
        if self.model_name == "pointnet2":
            return self._run_pointnet2(xyz_grid)
        if self.model_name == "salsanext":
            return self._run_salsanext(xyz_grid)
        return self._run_kevin(xyz_grid)

    @staticmethod
    def _class_counts(pred_grid: np.ndarray) -> dict:
        ids, counts = np.unique(pred_grid, return_counts=True)
        return {int(i): int(c) for i, c in zip(ids.tolist(), counts.tolist())}

    def _run_pointnet2(self, xyz_grid: np.ndarray) -> dict:
        if not self._order_checked:
            # Once per stream: a driver configured differently is refused
            # rather than fed to the model silently misordered.
            check_sim_order(ouster_to_sim_order(xyz_grid))
            self._order_checked = True

        pts, ouster_idx = real_scan_to_points(xyz_grid)
        out = self.model(torch.from_numpy(pts).unsqueeze(0).to(self.device))
        recon = out["recon_points"][0]  # (2048, 3), free-form -- needs reprojection to plot

        range_native = np.linalg.norm(xyz_grid, axis=-1)
        with np.errstate(invalid="ignore"):
            valid = (range_native >= SIM_RANGE_M[0]) & (range_native <= SIM_RANGE_M[1])
        xyz_valid = torch.from_numpy(xyz_grid[valid]).float().to(self.device)
        dists = torch.cdist(xyz_valid, recon)
        nn_range = recon[dists.argmin(dim=1)].norm(dim=-1).cpu().numpy()
        recon_range_grid = np.full_like(range_native, np.nan)
        recon_range_grid[valid] = nn_range

        # One prediction per REAL input point: the padding duplicates at the
        # end of the input are dropped, and ouster_idx puts each prediction
        # back in the cell it came from.
        n = len(ouster_idx)
        conf, pred = torch.softmax(out["class_logits"][0, :n], dim=-1).max(dim=-1)
        conf, pred = conf.cpu().numpy(), pred.cpu().numpy()
        shown = np.where(conf >= self.class_conf_threshold, pred, 0).astype(np.uint8)
        pred_grid = np.zeros(N_RINGS * N_COLS, dtype=np.uint8)
        pred_grid[ouster_idx] = shown
        pred_grid = pred_grid.reshape(N_RINGS, N_COLS)
        # Only even columns are classified (every-2nd-column decimation, and
        # the reorder keeps parity); each odd column shows its left
        # neighbour -- for the picture only, never counted.
        pred_grid[:, 1::2] = pred_grid[:, 0::2]

        real_norm = np.clip(range_native[valid], 0, self.max_range) / self.max_range
        pred_norm = np.clip(nn_range, 0, self.max_range) / self.max_range
        mse = float(np.mean((real_norm - pred_norm) ** 2))

        gap = np.full((3, N_COLS, 3), 10, dtype=np.uint8)
        combined = np.concatenate([
            label_row(range_rgb(range_native), "ORIGINAL RANGE (real, measured)"), gap,
            label_row(range_rgb(recon_range_grid), "RECONSTRUCTED RANGE"), gap,
            label_row(class_rgb(pred_grid), "PREDICTED CLASS (no ground truth on real data)"),
        ], axis=0)
        return {"image": combined, "mse": mse, "pred_class_counts": self._class_counts(shown),
                "points": pts[:n], "pred": pred, "conf": conf}

    def _run_salsanext(self, xyz_grid: np.ndarray) -> dict:
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

        gap = np.full((3, N_COLS, 3), 10, dtype=np.uint8)
        combined = np.concatenate([
            label_row(range_rgb(orig_display), "ORIGINAL RANGE (real, measured)"), gap,
            label_row(range_rgb(recon_display), "RECONSTRUCTED RANGE"), gap,
            label_row(class_rgb(pred_cls), "PREDICTED CLASS (no ground truth on real data)"),
        ], axis=0)
        return {"image": combined, "mse": mse,
                "pred_class_counts": self._class_counts(pred_cls)}

    def _run_kevin(self, xyz_grid: np.ndarray) -> dict:
        range_ = np.linalg.norm(xyz_grid, axis=-1)
        valid = np.isfinite(range_) & (range_ > 1e-3)
        range_n = np.where(valid, np.clip(range_, 0, self.max_range) / self.max_range, 0.0).astype(np.float32)
        x = torch.from_numpy(range_n).unsqueeze(0).unsqueeze(0).to(self.device)
        recon, _z = self.model(x)
        recon_n = recon[0, 0].cpu().numpy()

        orig_display = np.where(valid, range_, np.nan)
        recon_display = np.where(valid, recon_n * self.max_range, np.nan)
        mse = float(np.mean(((range_n - recon_n) ** 2)[valid])) if valid.any() else 0.0

        gap = np.full((3, N_COLS, 3), 10, dtype=np.uint8)
        combined = np.concatenate([
            label_row(range_rgb(orig_display), "ORIGINAL RANGE (real, measured)"), gap,
            label_row(range_rgb(recon_display), "RECONSTRUCTED RANGE"),
        ], axis=0)
        # No classifier in this candidate, so nothing to count.
        return {"image": combined, "mse": mse, "pred_class_counts": None}
