"""Theoretical compute and memory cost of the PointNet++ encoder for one 32,768-point scan.

This is an estimate, not a timing. It is the source of the compute and memory figures that the
thesis compares against the Jetson AGX Orin's published specification. To measure the encoder on
the device itself, use pointnet2_jetson_bench.py.

Layer FLOPs are counted by PyTorch's FlopCounterMode on a smaller input (they depend only on the
fixed centroid and neighbour counts, not on N). Sampling and grouping costs, which do scale with N,
are counted analytically from the code in models/pointnet2_utils.py.

    python benchmarks/encoder_cost_estimate.py
"""
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.pointnet2 import PointNet2Encoder  # noqa: E402

ORIN_FP32_FLOPS = 5.3e12  # Jetson AGX Orin 64GB, NVIDIA's published FP32 peak

enc = PointNet2Encoder().eval()
params = sum(p.numel() for p in enc.parameters())

with torch.no_grad(), FlopCounterMode(display=False) as fc:
    enc(torch.randn(1, 8192, 3))
# cdist inside ball_query is counted by FlopCounterMode only if it lowers to a matmul; take the layer
# FLOPs from the conv / linear entries only and count the distance work separately below.
by_op = {str(k): v for k, v in fc.get_flop_counts()["Global"].items()}
layer_flops = sum(v for k, v in by_op.items() if "convolution" in k or "addmm" in k or k.endswith("mm.default"))

N = 32768
levels = [(N, 4096), (4096, 1024), (1024, 256)]  # (points in, centroids out)
fps = sum(n_out * n_in * 10 for n_in, n_out in levels)     # per step: diff, square, sum, min, argmax ~ 10 FLOP/point
ballq = sum(n_out * n_in * 8 for n_in, n_out in levels)    # squared distance per centroid-point pair ~ 8 FLOP
sort_cmp = sum(n_out * n_in * 15 for n_in, n_out in levels)  # sort of each centroid's row: ~log2(32768)=15 per element
fps_steps = sum(n_out for _, n_out in levels)

# largest intermediate tensors, first level ball query (4096 x 32768 entries)
E = 4096 * N
peak = {
    "squared distances (float32)": 4 * E,
    "in-range mask (bool)": 1 * E,
    "candidate indices (int64)": 8 * E,
    "sorted values + sort indices (int64)": 16 * E,
}
print(f"encoder parameters: {params:,}  ({params * 4 / 1e6:.2f} MB as float32)")
print("layer FLOPs by op:", {k: f"{v / 1e9:.3f} G" for k, v in by_op.items()})
print(f"layer FLOPs (conv + linear): {layer_flops / 1e9:.2f} GFLOP")
print(f"farthest point sampling: {fps / 1e9:.2f} GFLOP over {fps_steps:,} sequential steps")
print(f"ball query distances: {ballq / 1e9:.2f} GFLOP, sort comparisons: {sort_cmp / 1e9:.2f} G")
total = layer_flops + fps + ballq
print(f"arithmetic total (without sort): {total / 1e9:.2f} GFLOP per scan, at 10 Hz {total * 10 / 1e9:.0f} GFLOP/s")
print(f"time at the Orin's FP32 peak: {total / ORIN_FP32_FLOPS * 1e3:.2f} ms (a lower bound, not a prediction)")
print("first-level ball query tensors:", {k: f"{v / 1e9:.2f} GB" for k, v in peak.items()},
      f"total {sum(peak.values()) / 1e9:.2f} GB")
