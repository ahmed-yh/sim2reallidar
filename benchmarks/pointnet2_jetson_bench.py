"""Benchmark the REAL PointNet2Encoder at several input point-count budgets.

Run this ON THE JETSON, after installing the JetPack-matched PyTorch wheel
(plain `pip install torch` does not work on Jetson -- see README "Running on
the Jetson"):

    python3 benchmarks/pointnet2_jetson_bench.py

Earlier versions of this script had their own inline, throwaway copy of
farthest-point-sampling and ball-query, written before models/pointnet2.py
existed. That copy had two real bugs (sentinel-index leakage when a centroid
has zero neighbors in radius; a crash when N < k) later found and fixed by
the test suite, in the shared models/pointnet2_utils.py -- see git history /
README for the story. This script now imports PointNet2Encoder directly, so
it benchmarks the exact class that ships to the Jetson, not an approximation
of it, and can never drift out of sync with a second, untested copy again.

Reports latency + peak GPU memory per point count, batch size 1 (matching
real deployment: one scan arrives, one prediction goes out, repeatedly).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.pointnet2 import PointNet2Encoder  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
if device.type == "cuda":
    print(f"  {torch.cuda.get_device_name(0)}")
else:
    print("  WARNING: no CUDA available -- this will not reflect real Jetson GPU timing")

POINT_COUNTS = [2048, 4096, 8192, 16384, 32768]
N_WARMUP, N_TIMED = 3, 10

print(f"\n{'points':>8}  {'latency':>10}  {'rate':>9}  {'peak_mem':>10}")
for n_points in POINT_COUNTS:
    model = PointNet2Encoder().to(device).eval()
    xyz = torch.randn(1, n_points, 3, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(xyz)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N_TIMED):
            model(xyz)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - t0) / N_TIMED

    mem = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0
    print(f"{n_points:>8}  {elapsed*1000:8.1f} ms  {1/elapsed:7.1f} Hz  {mem:8.1f} MB")
