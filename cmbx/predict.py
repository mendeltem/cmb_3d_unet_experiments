"""Sliding-window prediction on a prepared case (port of ``cv5.proba_fall``).

Windows of the training patch size, half-window stride, last window flush with the end; overlapping probabilities
averaged; windows whose IMAGE channel is entirely 0 (outside the brain) are skipped and stay 0. bfloat16 autocast
on CUDA as in the original. Optional flip test-time augmentation (off by default; the published numbers use none).
"""
from __future__ import annotations

import itertools
from typing import List, Sequence

import numpy as np
import torch


def crop_patch(a: np.ndarray, start, patch) -> np.ndarray:
    """Patch from ``start``; zero-padded at the end if the volume is smaller than the patch."""
    z, y, x = start
    p = a[z:z + patch[0], y:y + patch[1], x:x + patch[2]]
    if p.shape != tuple(patch):
        p = np.pad(p, [(0, patch[i] - p.shape[i]) for i in range(3)])
    return p


def window_starts(shape, patch) -> List[tuple]:
    padded = tuple(max(shape[i], patch[i]) for i in range(3))
    starts = []
    for i in range(3):
        st = list(range(0, padded[i] - patch[i] + 1, patch[i] // 2))
        if st[-1] + patch[i] < padded[i]:
            st.append(padded[i] - patch[i])
        starts.append(st)
    return [(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]]


def flip_sets(enabled: bool) -> List[tuple]:
    """All 8 combinations of flips over the three spatial axes (2, 3, 4 of a 5D tensor), or none."""
    if not enabled:
        return [()]
    return [tuple(a for a, f in zip((2, 3, 4), bits) if f) for bits in itertools.product((0, 1), repeat=3)]


def predict_volume(model: torch.nn.Module, channels: Sequence[np.ndarray], patch, device: str = "cpu",
                   batch_size: int = 8, amp_bf16: bool = True, tta_flip: bool = False):
    """Probability map ``(z, y, x)`` for the channel arrays ``[(z, y, x), ...]`` (channel 0 = image).
    Returns ``(prob float32, number of windows)``."""
    patch = tuple(int(v) for v in patch)
    img = channels[0]
    shp = img.shape
    padded = tuple(max(shp[i], patch[i]) for i in range(3))
    tiles = [s for s in window_starts(shp, patch) if crop_patch(img, s, patch).any()]
    pred = np.zeros(padded, np.float32)
    cnt = np.zeros(padded, np.float32)
    flips = flip_sets(tta_flip)
    use_amp = amp_bf16 and str(device).startswith("cuda")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        for i in range(0, len(tiles), batch_size):
            b = tiles[i:i + batch_size]
            xb = torch.from_numpy(np.stack([np.stack([crop_patch(c, s, patch) for c in channels], 0) for s in b]))
            xb = xb.to(device)
            out = None
            for axes in flips:
                xin = torch.flip(xb, axes) if axes else xb
                y = torch.sigmoid(model(xin)).float()
                y = torch.flip(y, axes) if axes else y
                out = y if out is None else out + y
            out = (out / len(flips)).cpu().numpy()
            for k, (z, y0, x) in enumerate(b):
                pred[z:z + patch[0], y0:y0 + patch[1], x:x + patch[2]] += out[k, 0]
                cnt[z:z + patch[0], y0:y0 + patch[1], x:x + patch[2]] += 1.0
    prob = (pred / np.maximum(cnt, 1))[:shp[0], :shp[1], :shp[2]]
    return prob, len(tiles)
