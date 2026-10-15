"""Lesion-level detection metric (port of ``code/metric.py`` of cmb-valdo-chain), 26-connectivity.

Per case:  TP = reference components touched (voxel overlap) by >= 1 predicted component
           FN = reference components not touched
           FP = predicted components touching no reference component
Several predictions on one reference count as 1 TP and 0 FP. Known loophole, kept for comparability with all
published numbers: ONE predicted component touching TWO references counts 2 TP; the clump guard
(``max_component_voxels``) in post-processing limits it.
Overall: pooled sums over cases, F1 = 2TP / (2TP + FP + FN); plus per-case F1 (mean / median over cases with
at least one reference or prediction), false positives per case and case-level detection.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from scipy import ndimage

N26 = np.ones((3, 3, 3), dtype=bool)


def hits(ref, pred) -> Dict[str, int]:
    """Lesion counts for one case; ``ref`` and ``pred`` are arrays of the same shape (non-zero = lesion)."""
    ref = np.asarray(ref) > 0
    pred = np.asarray(pred) > 0
    lr, nr = ndimage.label(ref, structure=N26)
    lp, npred = ndimage.label(pred, structure=N26)
    ref_hit = np.zeros(nr + 1, bool)
    pred_hit = np.zeros(npred + 1, bool)
    both = ref & pred
    if both.any():
        ref_hit[np.unique(lr[both])] = True
        pred_hit[np.unique(lp[both])] = True
    tp = int(ref_hit[1:].sum())
    fp = int((~pred_hit[1:]).sum())
    return dict(n_ref=int(nr), n_pred=int(npred), tp=tp, fp=fp, fn=int(nr) - tp,
                case_ref=int(nr > 0), case_pred=int(npred > 0), count_error=abs(int(npred) - int(nr)))


def case_f1(row: Dict) -> float:
    """F1 of one case; NaN when the case has neither reference nor prediction (nothing to score)."""
    d = 2 * row["tp"] + row["fp"] + row["fn"]
    return float("nan") if d == 0 else 2 * row["tp"] / d


def summarize(rows: Iterable[Dict]) -> Dict:
    """Pooled summary; key names and rounding identical to ``metric.summarize`` of cmb-valdo-chain."""
    rows = list(rows)
    s = {k: int(sum(z[k] for z in rows)) for k in ("n_ref", "n_pred", "tp", "fp", "fn")}
    n = len(rows)
    p = s["tp"] / max(s["tp"] + s["fp"], 1)
    r = s["tp"] / max(s["tp"] + s["fn"], 1)
    case = dict(tp=sum(bool(z["case_ref"] and z["case_pred"]) for z in rows),
                fp=sum(bool((not z["case_ref"]) and z["case_pred"]) for z in rows),
                tn=sum(bool((not z["case_ref"]) and (not z["case_pred"])) for z in rows),
                fn=sum(bool(z["case_ref"] and (not z["case_pred"])) for z in rows))
    return dict(cases=n, **s, precision=round(p, 4), sensitivity=round(r, 4),
                f1=round(2 * s["tp"] / max(2 * s["tp"] + s["fp"] + s["fn"], 1), 4),
                fp_per_case=round(s["fp"] / max(n, 1), 3),
                count_error_mean=round(float(np.mean([z["count_error"] for z in rows])) if rows else 0.0, 3),
                case_level={k: int(v) for k, v in case.items()},
                case_accuracy=round((case["tp"] + case["tn"]) / max(n, 1), 4))


def summarize_extended(rows: List[Dict]) -> Dict:
    """``summarize`` plus per-case F1 statistics (cases with nothing to score are left out of mean/median)."""
    out = summarize(rows)
    f1s = np.array([case_f1(r) for r in rows], dtype=float)
    valid = f1s[~np.isnan(f1s)]
    out["per_case_f1_mean"] = round(float(valid.mean()), 4) if len(valid) else None
    out["per_case_f1_median"] = round(float(np.median(valid)), 4) if len(valid) else None
    out["per_case_f1_n"] = int(len(valid))
    return out
