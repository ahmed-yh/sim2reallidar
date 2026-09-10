"""SalsaNext candidate: a range-image segmentation CNN (Cortinhal et al.,
"SalsaNext: Fast, Uncertainty-aware Semantic Segmentation of LiDAR Point
Clouds", 2020 -- https://arxiv.org/abs/2003.03653), adapted to this
project's encode()->256-d-latent interface the same way models/pointnet2.py
and Kevin's CNN autoencoder both are, so all three candidates are
interchangeable from the RL side.

Architecturally distinct from both existing candidates on purpose (the third
candidate needed to not just be a rerun of either): PointNet++ operates
directly on unordered 3D point sets with no grid at all; Kevin's CNN is a
plain autoencoder trained only to reconstruct; this is a 2D CNN over the
native (64,1024) range-image grid, built around SalsaNext's published
context module (dilated residual blocks) and pixel-shuffle decoder, and its
native task is per-pixel semantic segmentation, not reconstruction.

This is a from-scratch reimplementation of the published architecture's
core ideas (dilated-residual context module, strided-conv encoder, a
pixel-shuffle decoder with skip connections), sized for our 64x1024 input
and 8-class taxonomy -- not a port of the original 20-class SemanticKITTI
config or a vendored copy of the authors' code (their official repo,
github.com/TiagoCortinhal/SalsaNext, was intentionally not used as a
dependency here, matching the project's existing "code-only, no external
training framework" convention). GroupNorm throughout, not BatchNorm, for
the same reason as pointnet2.py: batch_size=1 must work.

Input is (range, x, y, z) only -- 4 channels, NOT the 5-channel
(range, x, y, z, remission) input the original paper uses. In this dataset,
"intensity"/remission IS the class label in disguise (see dataset.py's
_load_raw: `cls = np.rint(intensity)`), so including it as an input channel
would leak the supervision target directly into the model -- the same
leakage this project already avoids in pointnet2.py by excluding it from
PointNet++'s point input.

Two decoders, same split as pointnet2.py and for the same reason:
  - ReconstructionDecoder: latent ALONE -> full range image, no skip
    connections. A real test of the 256-d bottleneck's information content,
    not the decoder's upsampling path.
  - SegmentationDecoder: full skip-connected U-Net decoder, matching
    SalsaNext's actual published design -- this head's job is per-pixel
    accuracy, not testing the bottleneck in isolation.

Not implemented: SalsaNext's test-time Monte-Carlo-dropout uncertainty
estimate. The Dropout2d layers below give the option to add it later; doing
so wasn't needed for this project's fair-comparison deliverable, so it isn't
wired up.
"""
from __future__ import annotations

import torch
import torch.nn as nn

LATENT_DIM = 256
NUM_CLASSES = 8  # matches models/pointnet2.py -- same taxonomy, same data
IN_CHANNELS = 4  # range, x, y, z -- see module docstring for why not 5
N_RINGS = 64
N_COLS = 1024


class ResContextBlock(nn.Module):
    """Dilated-residual context module: a 1x1 shortcut plus a 3x3 conv
    followed by a dilated 3x3 conv (dilation=2), matching SalsaNext's
    published context block -- widens the receptive field before any
    downsampling, without losing resolution."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2)
        self.gn3 = nn.GroupNorm(8, out_ch)
        self.act3 = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.act1(self.conv1(x))
        x = self.act2(self.gn2(self.conv2(shortcut)))
        x = self.act3(self.gn3(self.conv3(x)))
        return x + shortcut


class ResBlock(nn.Module):
    """One encoder stage: residual conv block, then 2x2 average-pool
    downsampling. Returns both the pooled output (fed to the next stage)
    and the pre-pool feature (the decoder's skip connection for this
    resolution) -- mirrors PointNet++'s per-level (xyz, features) return."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.2):
        super().__init__()
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shortcut = self.shortcut(x)
        x = self.act1(self.gn1(self.conv1(x)))
        x = self.act2(self.gn2(self.conv2(x)))
        x = self.dropout(x + shortcut)
        return self.pool(x), x


class SalsaNextEncoder(nn.Module):
    """4 downsampling stages, 64x1024 -> 4x64, each halving both dims --
    sized for this project's native grid, not the original paper's exact
    channel schedule. Projects the deepest feature map into the same 256-d
    latent interface as PointNet2Encoder and Kevin's CNN by flattening it
    whole into a Linear layer -- NOT global-average-pooling it first.

    That distinction mattered in practice: an earlier version pooled first
    (avg over the 4x64 spatial map -> 256-d -> Linear), and reconstruction
    quality plateaued almost immediately (val recon loss barely moved
    across 35 epochs) while classification kept improving throughout on the
    exact same encoder. The reason: classification reads the full-resolution
    skip connections directly (SegmentationDecoder never touches this
    latent), so it never passed through the lossy pooling step -- only
    reconstruction did, and average-pooling irreversibly discards *where*
    things are, which a reconstruction decoder needs and a classifier
    mostly doesn't. Kevin's CNN autoencoder (models/autoencoder.py's
    CustomCNNEncoder) flattens its full bottleneck feature map into its
    Linear projection instead of pooling it, and reconstructs far better --
    matching that pattern here fixed the same asymmetry."""

    def __init__(self, in_ch: int = IN_CHANNELS, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.context = ResContextBlock(in_ch, 32)
        self.stage1 = ResBlock(32, 64)
        self.stage2 = ResBlock(64, 128)
        self.stage3 = ResBlock(128, 256)
        self.stage4 = ResBlock(256, 256)
        self.feat_ch, self.feat_h, self.feat_w = 256, N_RINGS // 16, N_COLS // 16
        flat = self.feat_ch * self.feat_h * self.feat_w
        self.head = nn.Sequential(
            nn.Linear(flat, 512), nn.GroupNorm(8, 512), nn.LeakyReLU(inplace=True),
            nn.Linear(512, latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor):
        x0 = self.context(x)
        x1, skip0 = self.stage1(x0)
        x2, skip1 = self.stage2(x1)
        x3, skip2 = self.stage3(x2)
        x4, skip3 = self.stage4(x3)
        latent = self.head(x4.flatten(1))
        skips = {0: skip0, 1: skip1, 2: skip2, 3: skip3}
        return latent, x4, skips


class UpBlockNoSkip(nn.Module):
    """Nearest-upsample + TWO conv layers, no skip connection -- the
    reconstruction decoder's building block (see module docstring). A
    second conv per stage (plus a 1x1 residual shortcut) gives each
    resolution real processing capacity instead of just one linear filter
    on the way up -- the single-conv version wasn't the primary cause of
    the reconstruction plateau (see SalsaNextEncoder's docstring for that),
    but it was still thin for reconstruction specifically, unlike the
    classification decoder which gets 2 convs per stage already
    (PixelShuffleUpBlock)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        shortcut = self.shortcut(x)
        x = self.act1(self.gn1(self.conv1(x)))
        x = self.act2(self.gn2(self.conv2(x)))
        return x + shortcut


class ReconstructionDecoder(nn.Module):
    """z ALONE -> full (64,1024) range image. No skip connections,
    deliberately -- see module docstring, same reasoning as
    models/pointnet2.py's decoder of the same name."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.seed_ch, self.seed_h, self.seed_w = 256, N_RINGS // 16, N_COLS // 16
        self.fc = nn.Linear(latent_dim, self.seed_ch * self.seed_h * self.seed_w)
        self.up1 = UpBlockNoSkip(256, 256)
        self.up2 = UpBlockNoSkip(256, 128)
        self.up3 = UpBlockNoSkip(128, 64)
        self.up4 = UpBlockNoSkip(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b = latent.shape[0]
        x = self.fc(latent).view(b, self.seed_ch, self.seed_h, self.seed_w)
        x = self.up4(self.up3(self.up2(self.up1(x))))
        return self.head(x).squeeze(1)  # (B, 64, 1024)


class PixelShuffleUpBlock(nn.Module):
    """SalsaNext's actual upsampling mechanism: a conv expands channels by
    4x, PixelShuffle(2) trades that channel factor for a 2x spatial
    upsample (sub-pixel convolution -- avoids the checkerboard artifacts
    plain transposed convolution is prone to), then concatenates the
    matching encoder skip connection before two more conv+norm+act layers."""

    def __init__(self, in_ch: int, up_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.pre = nn.Conv2d(in_ch, up_ch * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.conv1 = nn.Conv2d(up_ch + skip_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.shuffle(self.pre(x))
        x = torch.cat([x, skip], dim=1)
        x = self.act1(self.gn1(self.conv1(x)))
        x = self.act2(self.gn2(self.conv2(x)))
        return x


class SegmentationDecoder(nn.Module):
    """Skip-connected pixel-shuffle decoder, 4x64 -> 64x1024, mirroring
    SalsaNext's actual published design -- this head's job is per-pixel
    classification accuracy, not testing the bottleneck (see module
    docstring). Outputs per-pixel class logits over the full input grid."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.up4 = PixelShuffleUpBlock(in_ch=256, up_ch=256, skip_ch=256, out_ch=256)
        self.up3 = PixelShuffleUpBlock(in_ch=256, up_ch=128, skip_ch=256, out_ch=128)
        self.up2 = PixelShuffleUpBlock(in_ch=128, up_ch=64, skip_ch=128, out_ch=64)
        self.up1 = PixelShuffleUpBlock(in_ch=64, up_ch=32, skip_ch=64, out_ch=32)
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, deepest: torch.Tensor, skips: dict) -> torch.Tensor:
        x = self.up4(deepest, skips[3])
        x = self.up3(x, skips[2])
        x = self.up2(x, skips[1])
        x = self.up1(x, skips[0])
        return self.head(x)  # (B, num_classes, 64, 1024)


class SalsaNextMultiTask(nn.Module):
    """Training-time wrapper: encoder + both decoders. `encode()` is the
    only method that matters for deployment, mirroring
    PointNet2MultiTask/Kevin's Autoencoder.encode() so all three candidates
    are interchangeable from the RL side."""

    def __init__(self, in_ch: int = IN_CHANNELS, latent_dim: int = LATENT_DIM,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = SalsaNextEncoder(in_ch, latent_dim)
        self.recon_decoder = ReconstructionDecoder(latent_dim)
        self.seg_decoder = SegmentationDecoder(num_classes)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> dict:
        latent, deepest, skips = self.encoder(x)
        recon_range = self.recon_decoder(latent)
        class_logits = self.seg_decoder(deepest, skips)
        return {"latent": latent, "recon_range": recon_range, "class_logits": class_logits}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent, _, _ = self.encoder(x)
        return latent
