"""The one trained artifact the repo ships: weights/pointnet2_epoch39_encoder.pt.

Guards the deployment path in docs/deployment.md (deploy.ScanEncoder):
the file loads into a bare PointNet2Encoder with no missing or extra keys, and a
real-format scan, prepared the way the robot prepares it, encodes to one finite
256-d latent -- the same latent on every call."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ros2_inference"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared_inference import real_scan_to_points  # noqa: E402
from dataset import N_POINTS  # noqa: E402
from deploy import ScanEncoder  # noqa: E402
from models.pointnet2 import LATENT_DIM, PointNet2Encoder  # noqa: E402
from test_real_inference import _room_scan  # noqa: E402

WEIGHTS = REPO / "weights" / "pointnet2_epoch39_encoder.pt"


def test_shipped_encoder_loads_strictly():
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    encoder = PointNet2Encoder()
    result = encoder.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys


def test_shipped_encoder_encodes_a_real_format_scan():
    encoder = PointNet2Encoder()
    encoder.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    encoder.eval()

    points, _ = real_scan_to_points(_room_scan())
    assert points.shape == (N_POINTS, 3)
    xyz = torch.from_numpy(points).unsqueeze(0)
    with torch.no_grad():
        first, _ = encoder(xyz)
        second, _ = encoder(xyz)

    assert first.shape == (1, LATENT_DIM)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_scan_encoder_matches_the_manual_path():
    """deploy.ScanEncoder (what the robot imports) gives the same latent as loading the encoder and
    preparing the scan by hand, as the two tests above do."""
    encoder = PointNet2Encoder()
    encoder.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    encoder.eval()
    points, _ = real_scan_to_points(_room_scan())
    with torch.no_grad():
        manual, _ = encoder(torch.from_numpy(points).unsqueeze(0))

    latent = ScanEncoder(WEIGHTS, device="cpu")(_room_scan())
    assert latent.shape == (LATENT_DIM,)
    assert torch.equal(torch.from_numpy(latent), manual[0])
