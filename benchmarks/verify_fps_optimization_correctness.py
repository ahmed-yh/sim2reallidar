"""End-to-end check: does the CUDA-graph FPS change the full model's output
at all (it must not), and what's the REAL total step-time improvement (not
just the isolated FPS number) once it's wired into the actual forward pass.
"""
from __future__ import annotations

import time

import numpy as np
import torch

import models.pointnet2_utils as ptu
from dataset import PointCloudDataset
from losses import chamfer_distance, segmentation_loss
from models.pointnet2 import LATENT_DIM, NUM_CLASSES, PointNet2MultiTask

device = torch.device("cuda")
RECORDINGS_DIR = "scenario_data/mbut_76scenarios/o_robot_nav"

ids = np.load("data/sim/bench_ids.npy").tolist()[:8]
ds = PointCloudDataset(RECORDINGS_DIR, ids)
xyz = torch.stack([ds[i][0] for i in range(2)]).to(device)  # (2, 32768, 3) -- matches real training batch_size
cls = torch.stack([ds[i][1] for i in range(2)]).to(device)
weights = torch.ones(NUM_CLASSES, device=device)


def run_forward(graph_enabled: bool):
    ptu._fps_graph_disabled = not graph_enabled
    torch.manual_seed(0)
    model = PointNet2MultiTask(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)
    model.eval()
    with torch.no_grad():
        out = model(xyz)
    return out["latent"].clone(), out["recon_points"].clone(), out["class_logits"].clone()


print("=== correctness: full model output, graph FPS on vs off ===")
lat_off, recon_off, logits_off = run_forward(graph_enabled=False)
lat_on, recon_on, logits_on = run_forward(graph_enabled=True)

print(f"  latent identical:      {torch.equal(lat_off, lat_on)}")
print(f"  recon_points identical: {torch.equal(recon_off, recon_on)}")
print(f"  class_logits identical: {torch.equal(logits_off, logits_on)}")
if not (torch.equal(lat_off, lat_on) and torch.equal(recon_off, recon_on) and torch.equal(logits_off, logits_on)):
    print("  !! MISMATCH -- the optimization changed model output, do not use it !!")
    raise SystemExit(1)


def timed_full_step(model, optimizer, graph_enabled: bool, n_trials: int):
    ptu._fps_graph_disabled = not graph_enabled
    # warmup (also captures graphs, if enabled, before timing starts)
    for _ in range(2):
        out = model(xyz)
        loss = (chamfer_distance(out["recon_points"], xyz)
                + segmentation_loss(out["class_logits"], cls, weights))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.time()
        out = model(xyz)
        recon_loss = chamfer_distance(out["recon_points"], xyz)
        class_loss = segmentation_loss(out["class_logits"], cls, weights)
        (recon_loss + class_loss).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    return np.array(times)


print("\n=== real end-to-end step time (forward + backward + optimizer, batch_size=2) ===")
torch.manual_seed(0)
model = PointNet2MultiTask(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
t_off = timed_full_step(model, optimizer, graph_enabled=False, n_trials=8)
print(f"  graph FPS OFF (current)  mean {t_off.mean()*1000:8.1f} ms  median {np.median(t_off)*1000:8.1f} ms")

torch.manual_seed(0)
model = PointNet2MultiTask(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
t_on = timed_full_step(model, optimizer, graph_enabled=True, n_trials=8)
print(f"  graph FPS ON  (new)      mean {t_on.mean()*1000:8.1f} ms  median {np.median(t_on)*1000:8.1f} ms")

speedup = t_off.mean() / t_on.mean()
print(f"\n  real end-to-end speedup: {speedup:.2f}x")
print(f"  at ~26min/epoch before -> ~{26/speedup:.1f} min/epoch estimated now (same data amount)")

ptu._fps_graph_disabled = False
