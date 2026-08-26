"""PointNet++ candidate: encoder -> 256-d latent, matching the other two
candidates' latent_dim so the RL side sees a consistent interface regardless
of which encoder is deployed.

Deployment reality (confirmed against Kevin's own pattern -- his design doc:
"The decoder exists only for training -- only the encoder is kept afterwards",
and his actual code only ever saves/loads encoder.state_dict() for inference):
ONLY PointNet2Encoder ever runs on the Jetson. Both decoders below exist
purely to shape the encoder during training, on the Quadro RTX 5000, and are
discarded before deployment -- their cost is irrelevant to the Jetson budget,
which is why benchmarks/pointnet2_jetson_bench.py only ever measured the
encoder.

Input is (x, y, z) only -- never intensity/class, which would leak the
supervision target directly into the model's input.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pointnet2_utils import FeaturePropagation, SetAbstraction

LATENT_DIM = 256
NUM_CLASSES = 8  # raw laser_retro 0-7 (index 2/"mercedes" duplicate never fires in data)


class PointNet2Encoder(nn.Module):
    """Sized for the 32,768-point input (full ring resolution, every-2nd
    azimuth column -- see project notes on the decimation-coverage check).
    Returns the global latent plus each SA level's (xyz, features), the
    latter needed only by SegmentationDecoder's skip connections."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.sa1 = SetAbstraction(n_samples=4096, radius=0.3, k=32, in_ch=0, mlp_channels=[32, 32, 64])
        self.sa2 = SetAbstraction(n_samples=1024, radius=0.6, k=32, in_ch=64, mlp_channels=[64, 64, 128])
        self.sa3 = SetAbstraction(n_samples=256, radius=1.2, k=32, in_ch=128, mlp_channels=[128, 128, 256])
        self.head = nn.Sequential(
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, xyz: torch.Tensor):
        xyz1, pts1 = self.sa1(xyz, None)
        xyz2, pts2 = self.sa2(xyz1, pts1)
        xyz3, pts3 = self.sa3(xyz2, pts2)
        global_feat = torch.max(pts3, dim=1).values
        latent = self.head(global_feat)
        levels = {
            0: (xyz, None),   # original points, no learned features yet
            1: (xyz1, pts1),
            2: (xyz2, pts2),
            3: (xyz3, pts3),
        }
        return latent, levels


class ReconstructionDecoder(nn.Module):
    """z ALONE -> fixed-size point set. No skip connections, deliberately --
    this is what makes it a real test of the latent's information content
    rather than a shortcut around it. Scored with Chamfer distance (losses.py)
    against the input cloud."""

    def __init__(self, latent_dim: int = LATENT_DIM, n_points_out: int = 2048):
        super().__init__()
        self.n_points_out = n_points_out
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, n_points_out * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        return self.mlp(z).view(B, self.n_points_out, 3)


class SegmentationDecoder(nn.Module):
    """Standard PointNet++ feature-propagation decoder: skip connections to
    every SA level are intentional here (unlike the reconstruction decoder)
    -- this head's job is per-point classification accuracy, not testing the
    bottleneck in isolation. Outputs per-point class logits at all original
    input points."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fp3 = FeaturePropagation(in_ch=256 + 128, mlp_channels=[256, 256])
        self.fp2 = FeaturePropagation(in_ch=256 + 64, mlp_channels=[256, 128])
        self.fp1 = FeaturePropagation(in_ch=128, mlp_channels=[128, 128, 128])
        self.head = nn.Conv1d(128, num_classes, 1)

    def forward(self, levels: dict) -> torch.Tensor:
        xyz0, _ = levels[0]
        xyz1, pts1 = levels[1]
        xyz2, pts2 = levels[2]
        xyz3, pts3 = levels[3]

        pts2 = self.fp3(xyz2, xyz3, pts2, pts3)
        pts1 = self.fp2(xyz1, xyz2, pts1, pts2)
        pts0 = self.fp1(xyz0, xyz1, None, pts1)

        logits = self.head(pts0.permute(0, 2, 1))  # (B, num_classes, N)
        return logits.permute(0, 2, 1)  # (B, N, num_classes)


class PointNet2MultiTask(nn.Module):
    """Training-time wrapper: encoder + both decoders. `encode()` is the only
    method that matters for deployment -- it's the only thing that ships to
    the Jetson (mirrors Kevin's Autoencoder.encode())."""

    def __init__(self, latent_dim: int = LATENT_DIM, num_classes: int = NUM_CLASSES,
                 n_recon_points: int = 2048):
        super().__init__()
        self.encoder = PointNet2Encoder(latent_dim)
        self.recon_decoder = ReconstructionDecoder(latent_dim, n_recon_points)
        self.seg_decoder = SegmentationDecoder(num_classes)
        self.latent_dim = latent_dim

    def forward(self, xyz: torch.Tensor):
        latent, levels = self.encoder(xyz)
        recon_points = self.recon_decoder(latent)
        class_logits = self.seg_decoder(levels)
        return {"latent": latent, "recon_points": recon_points, "class_logits": class_logits}

    def encode(self, xyz: torch.Tensor) -> torch.Tensor:
        latent, _ = self.encoder(xyz)
        return latent
