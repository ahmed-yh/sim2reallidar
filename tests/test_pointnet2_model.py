import copy

import torch

from models.pointnet2 import (
    LATENT_DIM,
    NUM_CLASSES,
    PointNet2Encoder,
    PointNet2MultiTask,
    ReconstructionDecoder,
)

B, N = 2, 2048  # small N for fast tests; the real pipeline uses 32,768 (dataset.py)


def _dummy_batch():
    torch.manual_seed(0)
    return torch.randn(B, N, 3)


def test_encoder_output_shape():
    enc = PointNet2Encoder()
    xyz = _dummy_batch()
    latent, levels = enc(xyz)
    assert latent.shape == (B, LATENT_DIM)
    assert set(levels.keys()) == {0, 1, 2, 3}


def test_encoder_param_count_stays_small():
    # Canary: this is the ONLY thing that ships to the Jetson (see README
    # "What actually deploys"). If someone accidentally grows the encoder
    # by an order of magnitude, this should fail loudly rather than only
    # showing up as a surprise on-device.
    enc = PointNet2Encoder()
    n_params = sum(p.numel() for p in enc.parameters())
    assert n_params < 1_000_000, f"encoder grew to {n_params:,} params"


def test_latent_dim_matches_cross_candidate_convention():
    # All three candidates (baseline CNN, ResNet-VAE, PointNet++) must agree
    # on latent_dim so the RL side sees the same interface regardless of
    # which encoder is deployed.
    assert LATENT_DIM == 256


def test_reconstruction_decoder_shape():
    dec = ReconstructionDecoder(n_points_out=512)
    z = torch.randn(B, LATENT_DIM)
    out = dec(z)
    assert out.shape == (B, 512, 3)


def test_reconstruction_decoder_depends_only_on_latent_not_raw_input():
    # The whole point of NOT giving this decoder skip connections (see
    # README "Why two decoders, and why they're structurally different") is
    # that its output is a pure, deterministic function of z alone -- no
    # side-channel back to the original points. Two properties together
    # establish that: (1) its forward signature structurally accepts only
    # z, nothing else CAN flow in; (2) it's deterministic in eval mode, so
    # the same z always gives the same reconstruction (would fail if e.g.
    # dropout were accidentally left active). Different z must give a
    # different reconstruction too, or the test would trivially pass for a
    # decoder that ignores its input entirely.
    import inspect
    sig = inspect.signature(ReconstructionDecoder.forward)
    assert list(sig.parameters.keys()) == ["self", "z"]

    dec = ReconstructionDecoder()
    dec.eval()
    z_a = torch.randn(1, LATENT_DIM)
    z_b = torch.randn(1, LATENT_DIM)
    with torch.no_grad():
        out_a1 = dec(z_a)
        out_a2 = dec(z_a)
        out_b = dec(z_b)
    assert torch.equal(out_a1, out_a2), "same z gave different output -- decoder isn't deterministic"
    assert not torch.equal(out_a1, out_b), "different z gave identical output -- decoder ignores z"


def test_segmentation_decoder_output_matches_input_point_count():
    model = PointNet2MultiTask()
    xyz = _dummy_batch()
    out = model(xyz)
    assert out["class_logits"].shape == (B, N, NUM_CLASSES)


def test_multitask_forward_shapes():
    model = PointNet2MultiTask()
    xyz = _dummy_batch()
    out = model(xyz)
    assert out["latent"].shape == (B, LATENT_DIM)
    assert out["recon_points"].shape == (B, 2048, 3)
    assert out["class_logits"].shape == (B, N, NUM_CLASSES)


def test_gradient_from_both_losses_reaches_every_encoder_parameter():
    model = PointNet2MultiTask()
    xyz = _dummy_batch()
    out = model(xyz)

    recon_loss = out["recon_points"].pow(2).mean()
    class_loss = torch.nn.functional.cross_entropy(
        out["class_logits"].reshape(-1, NUM_CLASSES),
        torch.randint(0, NUM_CLASSES, (B * N,)),
    )
    (recon_loss + class_loss).backward()

    encoder_params = list(model.encoder.parameters())
    assert len(encoder_params) > 0
    for p in encoder_params:
        assert p.grad is not None, "an encoder parameter got no gradient from either loss"
        assert torch.isfinite(p.grad).all()


def test_encode_matches_forward_latent():
    model = PointNet2MultiTask()
    model.eval()
    xyz = _dummy_batch()
    with torch.no_grad():
        latent_from_encode = model.encode(xyz)
        latent_from_forward = model(xyz)["latent"]
    assert torch.equal(latent_from_encode, latent_from_forward)


def test_only_encoder_state_dict_is_needed_for_deployment():
    # This is the load-bearing claim for the whole Jetson deployment story
    # (matches Kevin's own pattern: only encoder.state_dict() ever gets
    # shipped). Prove it: load ONLY the encoder weights into a fresh,
    # decoder-less PointNet2Encoder, and confirm it reproduces the full
    # model's encode() output exactly.
    model = PointNet2MultiTask()
    model.eval()

    deployed_encoder = PointNet2Encoder()
    deployed_encoder.load_state_dict(copy.deepcopy(model.encoder.state_dict()))
    deployed_encoder.eval()

    xyz = _dummy_batch()
    with torch.no_grad():
        full_model_latent = model.encode(xyz)
        deployed_latent, _ = deployed_encoder(xyz)

    assert torch.equal(full_model_latent, deployed_latent)
