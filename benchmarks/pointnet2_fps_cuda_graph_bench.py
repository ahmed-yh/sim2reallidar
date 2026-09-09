"""Benchmark candidate speedups for farthest_point_sample against the current
implementation, on real data (a 32-sample slice, data/sim/bench_ids.npy).

For each candidate: verify it produces IDENTICAL output to the baseline on
real data first (FPS is deterministic -- same input must give the same
indices, not just "similar"), then time it. A fast-but-different result is
not a candidate, it's a bug -- not reported as a speed number at all.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from dataset import PointCloudDataset
from models.pointnet2_utils import farthest_point_sample

device = torch.device("cuda")
RECORDINGS_DIR = "scenario_data/mbut_76scenarios/o_robot_nav"
N_SAMPLES_SA1 = 4096  # the dominant cost, per earlier profiling
N_TRIALS = 5


def load_real_batches():
    ids = np.load("data/sim/bench_ids.npy").tolist()
    ds = PointCloudDataset(RECORDINGS_DIR, ids)
    batches = []
    for i in range(0, len(ids), 2):
        xyz0, _ = ds[i]
        xyz1, _ = ds[i + 1]
        batches.append(torch.stack([xyz0, xyz1]).to(device))
    return batches  # 16 batches of (2, 32768, 3)


def bench(fn, batches, label):
    # warmup
    for b in batches[:3]:
        fn(b, N_SAMPLES_SA1)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_TRIALS):
        for b in batches:
            torch.cuda.synchronize()
            t0 = time.time()
            fn(b, N_SAMPLES_SA1)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
    times = np.array(times)
    print(f"  {label:<28} mean {times.mean()*1000:7.1f} ms  "
          f"median {np.median(times)*1000:7.1f} ms  std {times.std()*1000:6.1f} ms  "
          f"(n={len(times)})")
    return times.mean()


def check_identical(fn, batches, label, reference_fn):
    for b in batches:
        ref = reference_fn(b, N_SAMPLES_SA1)
        out = fn(b, N_SAMPLES_SA1)
        if not torch.equal(ref, out):
            print(f"  {label}: MISMATCH vs baseline -- not a valid candidate, skipping")
            return False
    print(f"  {label}: output identical to baseline on all {len(batches)} real batches -- OK")
    return True


def main():
    print("Loading 16 real batches (32 samples) for benchmarking...")
    batches = load_real_batches()

    print("\n=== correctness ===")
    baseline = farthest_point_sample

    results = {}

    print("\n=== speed ===")
    results["baseline (current)"] = bench(baseline, batches, "baseline (current)")

    # candidate: torch.jit.script
    try:
        scripted = torch.jit.script(farthest_point_sample)
        if check_identical(scripted, batches, "torch.jit.script", baseline):
            results["torch.jit.script"] = bench(scripted, batches, "torch.jit.script")
    except Exception as e:
        print(f"  torch.jit.script: FAILED to compile -- {type(e).__name__}: {e}")

    # candidate: torch.compile
    try:
        compiled = torch.compile(farthest_point_sample)
        if check_identical(compiled, batches, "torch.compile", baseline):
            results["torch.compile"] = bench(compiled, batches, "torch.compile")
    except Exception as e:
        print(f"  torch.compile: FAILED -- {type(e).__name__}: {e}")

    # candidate: CUDA graph capture. All batches are the same shape (2, 32768, 3),
    # so the whole call sequence is static -- a fit for graph replay. Static
    # input buffer copied into before each replay (graphs require fixed
    # memory addresses, not a new tensor per call).
    try:
        B, N, _ = batches[0].shape
        static_in = torch.zeros(B, N, 3, device=device)
        static_in.copy_(batches[0])

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                farthest_point_sample(static_in, N_SAMPLES_SA1)
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = farthest_point_sample(static_in, N_SAMPLES_SA1)

        def cuda_graph_fps(xyz, n_samples):
            static_in.copy_(xyz)
            graph.replay()
            return static_out.clone()

        if check_identical(cuda_graph_fps, batches, "CUDA graph", baseline):
            results["CUDA graph"] = bench(cuda_graph_fps, batches, "CUDA graph")
    except Exception as e:
        print(f"  CUDA graph: FAILED -- {type(e).__name__}: {e}")

    print("\n=== summary (vs baseline) ===")
    base = results.get("baseline (current)")
    for name, t in results.items():
        speedup = base / t if base else float("nan")
        print(f"  {name:<28} {t*1000:7.1f} ms  ({speedup:.2f}x)")


if __name__ == "__main__":
    main()
