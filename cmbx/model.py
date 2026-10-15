"""The anisotropic 3D U-Net (``a03-aniso`` in ``code/netze.py`` of cmb-valdo-chain) and its padding wrapper.

Input ``(B, C, D, H, W)`` in ``(z, y, x)`` order -> one logit per voxel ``(B, 1, D, H, W)``.
Level 0 convolves (1, 3, 3) and pools (1, 2, 2) within the slice only: on the 0.5 x 0.5 x 1.0 mm grid the
through-plane direction is mostly interpolated (native slices 0.8-4 mm). From level 1 on the grid is 1 mm isotropic
and the network is an ordinary 3D U-Net. Every block is (Conv3d no bias -> InstanceNorm(affine) -> LeakyReLU 0.01)
twice; channels 32-64-128-256-512; transposed convolutions for upsampling; no attention gates, no deep
supervision. 22.5 M parameters with 2 input channels.

The layer structure and the order of parameter creation are identical to the original, so with the same torch seed
the initial weights are identical and research checkpoints (``modell.pt``, German attribute names) load after a
key rename (``convert_legacy_keys``).
"""
from __future__ import annotations

import re
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_MULTIPLE = (8, 16, 16)          # (z, y, x): four poolings, the first one in-plane only


class ConvBlock(nn.Module):
    """``(Conv3d -> InstanceNorm3d -> LeakyReLU) x 2`` with a selectable (possibly anisotropic) kernel."""

    def __init__(self, in_channels: int, out_channels: int, kernel: Sequence[int] = (3, 3, 3)) -> None:
        super().__init__()
        pad = tuple(k // 2 for k in kernel)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, tuple(kernel), padding=pad, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True), nn.LeakyReLU(0.01, True),
            nn.Conv3d(out_channels, out_channels, tuple(kernel), padding=pad, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True), nn.LeakyReLU(0.01, True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AnisoUNet3D(nn.Module):
    """The tournament winner. Attribute creation order follows ``netze.AnisoUNet3D`` exactly (same init)."""

    def __init__(self, in_channels: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        b = base_channels
        self.enc0 = ConvBlock(in_channels, b, kernel=(1, 3, 3))
        self.enc1, self.enc2, self.enc3 = ConvBlock(b, 2 * b), ConvBlock(2 * b, 4 * b), ConvBlock(4 * b, 8 * b)
        self.bottleneck = ConvBlock(8 * b, 16 * b)
        self.pool0, self.pool = nn.MaxPool3d((1, 2, 2)), nn.MaxPool3d(2)
        self.up3 = nn.ConvTranspose3d(16 * b, 8 * b, 2, stride=2)
        self.up2 = nn.ConvTranspose3d(8 * b, 4 * b, 2, stride=2)
        self.up1 = nn.ConvTranspose3d(4 * b, 2 * b, 2, stride=2)
        self.up0 = nn.ConvTranspose3d(2 * b, b, (1, 2, 2), stride=(1, 2, 2))
        self.dec3, self.dec2, self.dec1 = ConvBlock(16 * b, 8 * b), ConvBlock(8 * b, 4 * b), ConvBlock(4 * b, 2 * b)
        self.dec0 = ConvBlock(2 * b, b, kernel=(1, 3, 3))
        self.head = nn.Conv3d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.enc0(x)
        s1 = self.enc1(self.pool0(s0))
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        h = self.bottleneck(self.pool(s3))
        for up, dec, skip in ((self.up3, self.dec3, s3), (self.up2, self.dec2, s2),
                              (self.up1, self.dec1, s1), (self.up0, self.dec0, s0)):
            h = dec(torch.cat([up(h), skip], 1))
        return self.head(h)


class PadToMultiple(nn.Module):
    """Zero-pads ``(D, H, W)`` at the end to multiples of ``multiple`` and crops the output back (``Polster``)."""

    def __init__(self, net: nn.Module, multiple: Sequence[int] = PAD_MULTIPLE) -> None:
        super().__init__()
        self.net, self.multiple = net, tuple(multiple)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x.shape[-3:]
        target = [-(-g[i] // self.multiple[i]) * self.multiple[i] for i in range(3)]
        pad = [target[i] - g[i] for i in range(3)]
        if any(pad):
            x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
        return self.net(x)[..., :g[0], :g[1], :g[2]]


MODELS = {"aniso_unet3d": lambda in_channels, base_channels: PadToMultiple(AnisoUNet3D(in_channels, base_channels))}


def build_model(name: str = "aniso_unet3d", in_channels: int = 2, base_channels: int = 32) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](in_channels, base_channels)


_LEGACY_RULES = [
    (r"^netz\.", "net."),
    (r"\.mitte\.", ".bottleneck."),
    (r"\.e(\d)\.", r".enc\1."),
    (r"\.d(\d)\.", r".dec\1."),
    (r"\.u(\d)\.", r".up\1."),
    (r"\.aus\.", ".head."),
]


def convert_legacy_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rename research checkpoint keys (``netz.e0.net.0.weight``) to this module (``net.enc0.net.0.weight``)."""
    out = {}
    for key, tensor in state.items():
        new = key
        for pattern, repl in _LEGACY_RULES:
            new = re.sub(pattern, repl, new)
        out[new] = tensor
    return out


def load_state(model: nn.Module, path: str) -> nn.Module:
    """Load a checkpoint written by this package or a research ``modell.pt`` of cmb-valdo-chain."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    if any(k.startswith("netz.") for k in state):
        state = convert_legacy_keys(state)
    model.load_state_dict(state, strict=True)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
