"""Probability map -> microbleed mask: threshold, minimum size, clump guard, optional CSF rule;
plus the fast grid search over (threshold x minimum size) used for the inner-fold selection.

Port of ``filtert_m`` / ``gitter_zeilen`` / ``waehle`` of cmb-valdo-chain ``code/cv5.py``. The CSF rule follows
``cmb/analysis/operating_point.py``: a component is dropped when the majority SynthSeg label under it (labels > 0;
if it covers only background, widened by 3 voxels) is a CSF label (4, 5, 14, 15, 24, 43, 44).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .metrics import hits, summarize

N26 = np.ones((3, 3, 3), bool)
CSF_LABELS = (4, 5, 14, 15, 24, 43, 44)


def majority_label(mask: np.ndarray, seg: np.ndarray) -> int:
    """Majority label > 0 under ``mask``; if there is none, under the mask dilated by 3 voxels; else 0."""
    for iterations in (0, 3):
        m = ndimage.binary_dilation(mask, structure=N26, iterations=iterations) if iterations else mask
        values = seg[m]
        values = values[values > 0]
        if len(values):
            return int(np.bincount(values).argmax())
    return 0


def _csf_components(labels: np.ndarray, candidates: np.ndarray, tissue: np.ndarray, csf_labels: Sequence[int],
                    origin=(0, 0, 0)) -> np.ndarray:
    """Boolean per component index: True = majority CSF. ``labels`` may be a crop of the full volume starting at
    ``origin``; ``tissue`` is the FULL map, so the 3-voxel widening is never cut by the crop."""
    out = np.zeros(len(candidates), bool)
    csf = set(int(v) for v in csf_labels)
    for c, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None or not candidates[c]:
            continue
        lo = [max(s.start + o - 3, 0) for s, o in zip(box, origin)]
        hi = [min(s.stop + o + 3, d) for s, o, d in zip(box, origin, tissue.shape)]
        local = np.zeros([b - a for a, b in zip(lo, hi)], bool)
        inner = tuple(slice(s.start + o - a, s.stop + o - a) for s, o, a in zip(box, origin, lo))
        local[inner] = labels[box] == c
        out[c] = majority_label(local, tissue[tuple(slice(a, b) for a, b in zip(lo, hi))]) in csf
    return out


def filter_components(p: np.ndarray, threshold: float, min_size: int, max_voxels: int = 2100,
                      counter: Optional[Dict] = None, tissue: Optional[np.ndarray] = None,
                      csf_labels: Sequence[int] = CSF_LABELS) -> np.ndarray:
    """``p > threshold``; keep 26-connected components with ``min_size <= voxels <= max_voxels`` (never the
    background); with ``tissue`` also drop majority-CSF components. ``counter`` collects clump statistics."""
    m = p > threshold
    if not m.any():
        return m
    lab, n = ndimage.label(m, structure=N26)
    sizes = np.bincount(lab.ravel())
    keep = (sizes >= min_size) & (sizes <= max_voxels)
    keep[0] = False
    if counter is not None:
        big = sizes[1:] > max_voxels
        counter["clumps"] = counter.get("clumps", 0) + int(big.sum())
        counter["clump_voxels"] = counter.get("clump_voxels", 0) + int(sizes[1:][big].sum())
    if tissue is not None:
        csf = _csf_components(lab, keep, tissue, csf_labels)
        if counter is not None:
            counter["csf_removed"] = counter.get("csf_removed", 0) + int(csf.sum())
        keep &= ~csf
    return keep[lab]


Cell = Tuple[float, int]


def grid_rows(p: np.ndarray, lab: np.ndarray, thresholds: Sequence[float], min_sizes: Sequence[int],
              max_voxels: int = 2100, tissue: Optional[np.ndarray] = None,
              csf_labels: Sequence[int] = CSF_LABELS) -> Dict[Cell, Tuple[Dict, int]]:
    """``hits(lab, filter_components(p, t, m))`` for every cell at once -> ``{(t, m): (hits row, clumps)}``.

    Identical results to the slow loop (tests/test_postprocess.py checks it), but the reference is labelled once,
    the map once per threshold, hits come from the overlap table, and everything runs inside the bounding box of
    (p > smallest threshold) | reference."""
    m0 = (p > min(thresholds)) | (lab > 0)
    rows: Dict[Cell, Tuple[Dict, int]] = {}
    if not m0.any():
        empty = hits(lab, np.zeros(lab.shape, bool))
        return {(t, m): (dict(empty), 0) for t in thresholds for m in min_sizes}
    sl = ndimage.find_objects(m0.astype(np.uint8))[0]
    origin = tuple(s.start for s in sl)
    pc, lc = p[sl], lab[sl] > 0
    lr, nr = ndimage.label(lc, structure=N26)
    for t in thresholds:
        lp, npred = ndimage.label(pc > t, structure=N26)
        sizes = np.bincount(lp.ravel(), minlength=npred + 1)
        usable = sizes <= max_voxels
        usable[0] = False
        if tissue is not None:
            usable &= ~_csf_components(lp, usable, tissue, csf_labels, origin)
        both = (lp > 0) & (lr > 0)
        pairs = (np.unique(np.stack([lp[both], lr[both]], 1), axis=0) if both.any()
                 else np.zeros((0, 2), np.int64))
        clumps = int((sizes[1:] > max_voxels).sum())
        for m in min_sizes:
            keep = usable & (sizes >= m)
            n_keep = int(keep.sum())
            kp = pairs[keep[pairs[:, 0]]] if len(pairs) else pairs
            tp = int(len(np.unique(kp[:, 1]))) if len(kp) else 0
            fp = n_keep - (int(len(np.unique(kp[:, 0]))) if len(kp) else 0)
            rows[(t, m)] = (dict(n_ref=int(nr), n_pred=n_keep, tp=tp, fp=fp, fn=int(nr) - tp,
                                 case_ref=int(nr > 0), case_pred=int(n_keep > 0),
                                 count_error=abs(n_keep - int(nr))), clumps)
    return rows


def choose_cell(all_rows: List[Dict[Cell, Tuple[Dict, int]]], thresholds: Sequence[float],
                min_sizes: Sequence[int], fallback: Sequence = (0.5, 1)):
    """Best pooled F1 over the cases (first cell wins ties, thresholds outer, sizes inner, as ``cv5.waehle``).
    Returns ``(threshold, min_size, summary, clumps)``; if every cell scores F1 = 0, the fallback cell."""
    best = None
    for t in thresholds:
        for m in min_sizes:
            s = summarize([z[(t, m)][0] for z in all_rows])
            if best is None or s["f1"] > best[2]["f1"] + 1e-12:
                best = (t, m, s, sum(z[(t, m)][1] for z in all_rows))
    if best[2]["f1"] <= 0:
        t, m = fallback[0], int(fallback[1])
        key = (t, m)
        if key not in all_rows[0]:
            raise ValueError(f"fallback cell {key} is not part of the grid")
        best = (t, m, summarize([z[key][0] for z in all_rows]), sum(z[key][1] for z in all_rows))
    return best


def score_grid(maps: Iterable[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]], sel: Dict, csf: bool = False,
               csf_labels: Sequence[int] = CSF_LABELS):
    """Convenience: ``[(prob, label, tissue), ...]`` -> list of grid rows."""
    return [grid_rows(p, l, sel["thresholds"], sel["min_sizes"], sel["max_component_voxels"],
                      t if csf else None, csf_labels) for p, l, t in maps]
