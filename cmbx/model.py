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

import os
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


def read_state(path: str) -> Dict[str, torch.Tensor]:
    """State dict of a checkpoint of this package, a research ``modell.pt`` of cmb-valdo-chain or a released weight
    file (mb_segment layout), in this package's key names."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    if any(k.startswith("netz.") for k in state):
        state = convert_legacy_keys(state)
    if any(re.match(r"^net\.(enc\d|dec\d|bottleneck)\.\d", k) for k in state):    # released weights (mb_segment layout)
        state = {re.sub(r"^net\.(enc\d|dec\d|bottleneck)\.(\d)", r"net.\1.net.\2", k): v for k, v in state.items()}
    return state


def load_state(model: nn.Module, path: str) -> nn.Module:
    """Load a checkpoint (any of the formats of ``read_state``) into ``model``, strictly."""
    model.load_state_dict(read_state(path), strict=True)
    return model


def resolve_pretrained(source: str, channels_hint: Sequence[str] = None):
    """``source`` = a run folder (final/model.pt, else fold0/model.pt; channels from its config.yaml) or a model file
    (channels from the config.yaml of its run folder if there is one, else ``channels_hint``, else [image, frst])."""
    import os
    import yaml
    run = source if os.path.isdir(source) else os.path.dirname(os.path.dirname(os.path.abspath(source)))
    path = source
    if os.path.isdir(source):
        path = next((p for p in (os.path.join(source, "final", "model.pt"), os.path.join(source, "fold0", "model.pt"))
                     if os.path.exists(p)), None)
        if path is None:
            raise FileNotFoundError(f"no final/model.pt or fold0/model.pt in {source}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"pretrained model not found: {path}")
    cfg_path = os.path.join(run, "config.yaml")
    if channels_hint:
        channels = list(channels_hint)
    elif os.path.exists(cfg_path):
        channels = list(yaml.safe_load(open(cfg_path))["train"]["channels"])
    else:
        channels = ["image", "frst"]
    return path, channels


def init_from_pretrained(model: nn.Module, source: str, channels: Sequence[str], source_channels: Sequence[str] = None,
                         log=print) -> Dict:
    """Start of a fine-tuning: every weight from the pretrained model. Input channels are matched BY NAME: a channel
    the pretrained model had keeps its first-layer weights, a NEW channel starts at zero weight (so the network first
    computes exactly what the pretrained one did and learns the new channel during training), a channel left out is
    dropped. The width (``model.base_channels``) must be the same."""
    path, src = resolve_pretrained(source, source_channels)
    state, own = read_state(path), model.state_dict()
    missing = [k for k in own if k not in state]
    if missing:
        raise ValueError(f"{path} does not fit this network (missing {missing[:3]} ...)")
    out = {}
    input_key = next(k for k, w in own.items() if w.ndim == 5)          # the first convolution sees the channels
    for k, w in own.items():
        v = state[k]
        if k == input_key:
            if not (v.ndim == 5 and v.shape[0] == w.shape[0] and v.shape[2:] == w.shape[2:] and v.shape[1] == len(src)):
                raise ValueError(f"{path}: input layer {k} has shape {tuple(v.shape)} for channels {list(src)}, this "
                                 f"network {tuple(w.shape)} (same model.base_channels as the pretrained model needed)")
            new = torch.zeros_like(w)
            for i, ch in enumerate(channels):                        # by name: same, reordered, new (zero) or dropped
                if ch in src:
                    new[:, i] = v[:, src.index(ch)]
            out[k] = new
        elif v.shape == w.shape:
            out[k] = v
        else:
            raise ValueError(f"{path}: {k} has shape {tuple(v.shape)}, this network {tuple(w.shape)} "
                             "(same model.base_channels as the pretrained model needed)")
    model.load_state_dict(out, strict=True)
    info = dict(init_from=os.path.abspath(path), init_channels=list(src), channels=list(channels),
                new_channels=[c for c in channels if c not in src], dropped_channels=[c for c in src if c not in channels])
    log(f"initialised from {path}: channels {list(channels)} <- {list(src)}"
        + (f" (new, zero-initialised: {info['new_channels']})" if info["new_channels"] else ""))
    return info


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
