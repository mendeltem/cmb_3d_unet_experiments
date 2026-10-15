import re

import torch

from cmbx.model import AnisoUNet3D, build_model, convert_legacy_keys, count_parameters


def test_forward_shape_on_the_training_patch():
    torch.manual_seed(0)
    m = build_model("aniso_unet3d", 2, 8).eval()
    with torch.no_grad():
        y = m(torch.randn(1, 2, 38, 62, 94))                # odd size -> padded to (40, 64, 96) and cropped back
    assert y.shape == (1, 1, 38, 62, 94)


def test_parameter_count_matches_the_original():
    # meta.json of a03-aniso-e60-gd reports 22 496 065 parameters (2 channels, 32 base channels)
    assert count_parameters(build_model("aniso_unet3d", 2, 32)) == 22_496_065


def test_backward_reaches_every_parameter():
    m = build_model("aniso_unet3d", 1, 4).train()
    m(torch.randn(2, 1, 8, 32, 32)).mean().backward()
    assert all(p.grad is not None for p in m.parameters())


def _to_legacy(key: str) -> str:
    key = re.sub(r"^net\.", "netz.", key)
    key = key.replace(".bottleneck.", ".mitte.").replace(".head.", ".aus.")
    key = re.sub(r"\.enc(\d)\.", r".e\1.", key)
    key = re.sub(r"\.dec(\d)\.", r".d\1.", key)
    key = re.sub(r"\.up(\d)\.", r".u\1.", key)
    return key


def test_legacy_keys_round_trip():
    m = build_model("aniso_unet3d", 2, 4)
    state = m.state_dict()
    legacy = {_to_legacy(k): v for k, v in state.items()}
    assert any(k.startswith("netz.e0.net.0") for k in legacy)
    back = convert_legacy_keys(legacy)
    assert set(back) == set(state)
    m.load_state_dict(back, strict=True)


def test_inner_net_is_aniso():
    m = build_model()
    assert isinstance(m.net, AnisoUNet3D)
    assert m.net.enc0.net[0].kernel_size == (1, 3, 3) and m.net.pool0.kernel_size == (1, 2, 2)
