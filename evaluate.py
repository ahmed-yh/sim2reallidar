"""Evaluate a trained PointNet++ checkpoint on the held-out TEST split.

Unlike train.py's validation split (used for early-stopping decisions during
training, so not a fully unbiased estimate), the test split is never touched
until this script runs -- the only honest read of real performance.

    python3 evaluate.py

Reports:
- Chamfer reconstruction distance (same metric train.py logs for val, so
  directly comparable to the val numbers printed during training)
- Per-point classification: overall accuracy (inflated by ~99% environment
  points, reported for context only) and per-class recall/precision/F1, which
  is the metric that actually matters given the extreme class imbalance
  (see docs/data.md).

--class-conf-threshold applies the same display rule the ROS2 inference path
uses (_shared_inference.ModelRunner): a point keeps its predicted object class
only if softmax confidence clears the threshold, otherwise it reads as
environment. 0 reproduces plain argmax. This is NOT a change to the weights --
it is a decision about what to show, and the sweep it exists to support is why
the demo states 0.97 rather than a number picked by eye:

    threshold   0.0     0.70    0.97
    macro-F1    0.681   0.787   0.867

--json-out writes the same numbers as machine-readable JSON, which is what the
demo's metrics payload is built from, so the page cannot drift from the script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, PointCloudDataset
from losses import chamfer_distance
from models.pointnet2 import NUM_CLASSES, PointNet2MultiTask


def apply_threshold(pred, conf, t):
    """Only object predictions are suppressed. Demoting a low-confidence
    *environment* prediction to environment would be a no-op, and promoting
    it to an object would invent detections."""
    if t <= 0:
        return pred
    return torch.where((pred > 0) & (conf < t), torch.zeros_like(pred), pred)


def score(conf_flat):
    """confusion [true, pred] -> per-class rows, macro-F1, overall accuracy."""
    cm = conf_flat.reshape(NUM_CLASSES, NUM_CLASSES)
    out_rows = []
    for c in range(NUM_CLASSES):
        true_count = cm[c].sum().item()
        pred_count = cm[:, c].sum().item()
        if true_count == 0 and pred_count == 0:
            continue
        recall = cm[c, c].item() / true_count if true_count > 0 else float("nan")
        precision = cm[c, c].item() / pred_count if pred_count > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)
        out_rows.append({"classId": c, "name": CLASS_NAMES[c], "truePoints": true_count,
                         "recall": recall, "precision": precision, "f1": f1})
    # Macro-F1 over the OBJECT classes only. Including environment would fold
    # a ~0.999 F1 on 99% of the points into the average and hide every change
    # the threshold makes -- which is the whole quantity being optimised.
    objs = [r for r in out_rows if r["classId"] > 0]
    macro = sum(r["f1"] for r in objs) / len(objs) if objs else 0.0
    return out_rows, macro, cm.diag().sum().item() / cm.sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings-dir", type=Path,
                     default=Path(__file__).parent / "scenario_data" / "mbut_76scenarios" / "o_robot_nav")
    ap.add_argument("--splits-dir", type=Path, default=Path(__file__).parent / "data" / "sim")
    ap.add_argument("--checkpoint", type=Path,
                     default=Path(__file__).parent / "outputs" / "pointnet2" / "best_multitask_model.pt")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--class-conf-threshold", type=float, default=0.0,
                     help="object classes below this softmax confidence read as "
                          "environment; 0 = plain argmax (see module docstring)")
    ap.add_argument("--json-out", type=Path,
                     help="also write these numbers as JSON (used by demo/build_demo.py)")
    ap.add_argument("--sweep", type=str,
                     help="comma-separated thresholds to score in ONE forward pass, e.g. "
                          "'0,0.5,0.7,0.9,0.97,0.99'. Scoring is a comparison against a "
                          "confidence the model has already produced, so sweeping costs one "
                          "confusion matrix per threshold, not one pass per threshold. "
                          "Writes <json-out stem>_sweep.json next to --json-out.")
    args = ap.parse_args()

    sweep = ([float(t) for t in args.sweep.split(",")] if args.sweep else None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ids = np.load(args.splits_dir / "test_ids.npy")
    print(f"test: {len(test_ids)} samples, device: {device}")
    print(f"checkpoint: {args.checkpoint}")

    loader = DataLoader(
        PointCloudDataset(args.recordings_dir, test_ids),
        batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    model = PointNet2MultiTask().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    confusion = torch.zeros(NUM_CLASSES * NUM_CLASSES, dtype=torch.int64)
    sweep_confusion = ({t: torch.zeros(NUM_CLASSES * NUM_CLASSES, dtype=torch.int64)
                        for t in sweep} if sweep else {})
    recon_total, n_samples = 0.0, 0

    with torch.no_grad():
        for xyz, cls in loader:
            xyz, cls = xyz.to(device), cls.to(device)
            out = model(xyz)

            recon_total += chamfer_distance(out["recon_points"], xyz).item() * xyz.size(0)
            n_samples += xyz.size(0)

            logits = out["class_logits"]
            pred = logits.argmax(dim=-1)
            conf = logits.softmax(dim=-1).gather(-1, pred.unsqueeze(-1)).squeeze(-1)

            # Every swept threshold reuses this same forward pass: the network
            # is not re-run, only the comparison against `conf` changes.
            for t, acc in sweep_confusion.items():
                p = apply_threshold(pred, conf, t)
                idx = (cls.reshape(-1) * NUM_CLASSES + p.reshape(-1)).cpu()
                acc += torch.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES)

            pred = apply_threshold(pred, conf, args.class_conf_threshold)
            idx = (cls.reshape(-1) * NUM_CLASSES + pred.reshape(-1)).cpu()
            confusion += torch.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES)

    print(f"\nTest Chamfer reconstruction distance: {recon_total / n_samples:.5f}  "
          f"(val was ~0.924 at the restored checkpoint)")

    rows, macro_f1, overall_acc = score(confusion)
    print(f"\nclass confidence threshold: {args.class_conf_threshold:g}"
          f"{'  (plain argmax)' if args.class_conf_threshold <= 0 else ''}")
    print(f"\n{'class':<12} {'true points':>12} {'recall':>8} {'precision':>10} {'f1':>8}")
    for r in rows:
        print(f"{r['name']:<12} {r['truePoints']:>12} {r['recall']:>8.1%} "
              f"{r['precision']:>10.1%} {r['f1']:>8.3f}")
    print(f"\nMacro-F1 over object classes: {macro_f1:.3f}")
    print(f"\nOverall point accuracy: {overall_acc:.2%} "
          f"(inflated by ~99% environment points -- per-class recall above is the real signal)")

    if sweep:
        print(f"\n--- threshold sweep ({len(sweep)} thresholds, one shared forward pass) ---")
        print(f"{'threshold':>10} {'macro-F1':>10} {'overall acc':>12}   per-class F1")
        sweep_out = []
        for t in sorted(sweep_confusion):
            r_t, macro_t, acc_t = score(sweep_confusion[t])
            objs = [r for r in r_t if r["classId"] > 0]
            per = " ".join(f"{r['name'][:4]}={r['f1']:.3f}" for r in objs)
            print(f"{t:>10.3g} {macro_t:>10.3f} {acc_t:>11.2%}   {per}")
            sweep_out.append({"threshold": t, "macroF1": macro_t,
                              "overallPointAcc": acc_t, "rows": r_t})
        if args.json_out:
            sp = args.json_out.with_name(args.json_out.stem + "_sweep.json")
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text(json.dumps({
                "script": "evaluate.py --sweep",
                "checkpoint": str(args.checkpoint),
                "testSamples": int(len(test_ids)),
                "sweep": sweep_out,
            }, indent=2), encoding="utf-8")
            print(f"wrote {sp}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "script": "evaluate.py",
            "checkpoint": str(args.checkpoint),
            "testSamples": int(len(test_ids)),
            "threshold": args.class_conf_threshold,
            "chamfer": recon_total / n_samples,
            "overallPointAcc": overall_acc,
            "macroF1": macro_f1,
            "rows": rows,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
