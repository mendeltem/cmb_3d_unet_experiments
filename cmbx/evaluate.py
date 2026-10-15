"""Scoring of a finished CV run, and re-scoring of its stored probability maps with other post-processing
(no retraining): e.g. the CSF rule, or a cross-fold choice of the operating cell.

Modes
  stored     the binary masks written by the CV (``<id>_pred.nii.gz``) against the prepared labels -- the CV result.
  inner      re-binarise ``<id>_pred_proba.nii.gz`` (uint8 = round(p * 255)) with each fold's inner-fold cell.
  crossfold  for fold k pick the cell with the best pooled F1 over the OTHER folds' cases, apply it to fold k
             (``cmb/analysis/operating_point.py`` of cmb-valdo-chain; this is how the CSF-rule gain was measured).
Note: re-binarising the stored uint8 maps can differ from ``stored`` for voxels within 1/255 of the threshold.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

from .metrics import case_f1, hits, summarize, summarize_extended
from .postprocess import choose_cell, filter_components, grid_rows


def _cases_of_run(run_dir: str):
    out = []
    for fd in sorted(glob.glob(os.path.join(run_dir, "fold[0-9]*"))):
        if not os.path.isdir(fd) or "_incomplete" in fd:
            continue
        mp = os.path.join(fd, "meta.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp))
        for pm in sorted(glob.glob(os.path.join(fd, "*_pred_meta.json"))):
            sid = os.path.basename(pm)[:-len("_pred_meta.json")]
            out.append(dict(id=sid, fold=meta["fold"], dir=fd, threshold=meta["threshold"],
                            min_size=meta["min_size"]))
    return out


def _load(prepared_dir: str, sid: str, name: str, dtype=None):
    p = os.path.join(prepared_dir, sid, f"{sid}_{name}.nii.gz")
    a = np.asarray(nib.load(p).dataobj)
    return a.astype(dtype) if dtype else a


def evaluate_run(run_dir: str, prepared_dir: str, mode: str = "stored", csf_rule: bool = False,
                 selection: Optional[Dict] = None, csf_labels=(4, 5, 14, 15, 24, 43, 44), tag: Optional[str] = None,
                 log=print) -> Dict:
    cases = _cases_of_run(run_dir)
    if not cases:
        raise SystemExit(f"no finished folds in {run_dir}")
    sel = selection or {}
    rows: List[Dict] = []
    if mode == "stored":
        if csf_rule:
            raise SystemExit("mode 'stored' scores the saved masks as they are; use --mode inner/crossfold with --csf-rule")
        for c in cases:
            lab = _load(prepared_dir, c["id"], "label") > 0
            pred = np.asarray(nib.load(os.path.join(c["dir"], f"{c['id']}_pred.nii.gz")).dataobj) > 0
            rows.append(dict(id=c["id"], fold=c["fold"], threshold=c["threshold"], min_size=c["min_size"],
                             **hits(lab, pred)))
    else:
        maxvox = int(sel.get("max_component_voxels", 2100))
        data = {}
        for c in cases:
            p = np.asarray(nib.load(os.path.join(c["dir"], f"{c['id']}_pred_proba.nii.gz")).dataobj).astype(np.float32) / 255.0
            lab = (_load(prepared_dir, c["id"], "label") > 0).astype(np.uint8)
            tis = _load(prepared_dir, c["id"], "tissue", np.int32) if csf_rule else None
            data[c["id"]] = (p, lab, tis)
        if mode == "inner":
            for c in cases:
                p, lab, tis = data[c["id"]]
                pb = filter_components(p, c["threshold"], c["min_size"], maxvox, None, tis, csf_labels)
                rows.append(dict(id=c["id"], fold=c["fold"], threshold=c["threshold"], min_size=c["min_size"],
                                 **hits(lab, pb)))
        elif mode == "crossfold":
            thr, sizes = [float(t) for t in sel["thresholds"]], [int(m) for m in sel["min_sizes"]]
            grids = {c["id"]: grid_rows(*data[c["id"]][:2], thr, sizes, maxvox, data[c["id"]][2], csf_labels)
                     for c in cases}
            for k in sorted({c["fold"] for c in cases}):
                others = [grids[c["id"]] for c in cases if c["fold"] != k]
                t, m, _, _ = choose_cell(others, thr, sizes, sel.get("fallback", (0.5, 1)))
                for c in cases:
                    if c["fold"] == k:
                        rows.append(dict(id=c["id"], fold=k, threshold=t, min_size=m, **grids[c["id"]][(t, m)][0]))
        else:
            raise SystemExit(f"unknown mode {mode!r}")
    summary = summarize_extended(rows)
    summary.update(mode=mode, csf_rule=csf_rule, run=os.path.abspath(run_dir),
                   per_fold_f1={int(k): summarize([r for r in rows if r["fold"] == k])["f1"]
                                for k in sorted({r["fold"] for r in rows})})
    name = tag or (mode + ("-csf" if csf_rule else ""))
    json.dump(summary, open(os.path.join(run_dir, f"evaluation-{name}.json"), "w"), indent=1)
    with open(os.path.join(run_dir, f"per_case-{name}.csv"), "w", newline="") as f:
        keys = list(rows[0]) + ["f1"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["id"]):
            f1 = case_f1(r)
            w.writerow({**r, "f1": "" if np.isnan(f1) else round(f1, 4)})
    log(f"{os.path.basename(os.path.abspath(run_dir))} [{name}]: F1 {summary['f1']}  P {summary['precision']}  "
        f"S {summary['sensitivity']}  FP/case {summary['fp_per_case']}  per-case F1 mean "
        f"{summary['per_case_f1_mean']}  ({summary['cases']} cases, TP {summary['tp']} FP {summary['fp']} "
        f"FN {summary['fn']})")
    return summary


SUMMARY_KEYS = ["name", "mode", "f1", "precision", "sensitivity", "fp_per_case", "per_case_f1_mean",
                "per_case_f1_median", "cases", "n_ref", "tp", "fp", "fn", "train_hours", "max_vram_gib", "config"]


def run_summary(run_dir: str, cfg: Dict, metas: List[Dict], evaluation: Dict) -> Dict:
    s = {k: evaluation.get(k) for k in SUMMARY_KEYS if k in evaluation}
    s.update(name=cfg["name"], train_hours=round(sum(m.get("seconds", 0) for m in metas) / 3600, 3),
             max_vram_gib=max([m.get("vram_gib") or 0 for m in metas] or [0]) or None,
             config=cfg.get("_config_file", ""),
             folds=[{k: m.get(k) for k in ("fold", "test_f1", "inner_f1", "epoch_chosen", "threshold", "min_size",
                                           "seconds", "vram_gib")} for m in metas],
             per_fold_f1=evaluation.get("per_fold_f1"))
    json.dump(s, open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    return s


def compare_runs(root: str, out_csv: Optional[str] = None, log=print) -> List[Dict]:
    """Collect ``<root>/*/summary.json`` and all ``evaluation-*.json`` into one table (``<root>/summary.csv``)."""
    rows = []
    for sp in sorted(glob.glob(os.path.join(root, "*", "summary.json"))):
        s = json.load(open(sp))
        rows.append({k: s.get(k) for k in SUMMARY_KEYS})
        run_dir = os.path.dirname(sp)
        for ep in sorted(glob.glob(os.path.join(run_dir, "evaluation-*.json"))):
            e = json.load(open(ep))
            tag = os.path.basename(ep)[len("evaluation-"):-5]
            if tag == "stored":
                continue
            r = {k: e.get(k) for k in SUMMARY_KEYS}
            r.update(name=f"{s['name']} [{tag}]", mode=tag, config=s.get("config"))
            rows.append(r)
    out_csv = out_csv or os.path.join(root, "summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_KEYS)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        log(f"{r['name']:40s} F1 {r['f1']}  P {r['precision']}  S {r['sensitivity']}  FP/case {r['fp_per_case']}")
    return rows
