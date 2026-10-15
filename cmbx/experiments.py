"""One experiment = one YAML file: prepare (if needed) -> split (if needed) -> 5-fold CV -> evaluation -> summary.

Prepared volumes are shared between experiments that use the same preprocessing: ``<prepared_dir>/preprocess.json``
records the preprocessing settings, and a config with different settings must point to a different
``data.prepared_dir`` (e.g. ``data/prepared_inpaint``) -- the run refuses to mix them.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

from . import config as C
from .data import make_split, prepared_lesion_counts, read_cases
from .evaluate import compare_runs, evaluate_run, run_summary
from .preprocess import prepare_all
from .train import run_cv


def ensure_prepared(cfg: Dict, workers: Optional[int] = None, overwrite: bool = False, log=print) -> None:
    cases = read_cases(cfg["data"]["cases"])
    out = cfg["data"]["prepared_dir"]
    os.makedirs(out, exist_ok=True)
    sig_path = os.path.join(out, "preprocess.json")
    sig = C.preprocess_signature(cfg)
    if os.path.exists(sig_path):
        old = json.load(open(sig_path))
        if old != sig and not overwrite:
            raise SystemExit(f"{out} was prepared with different preprocessing settings:\n  existing {old}\n"
                             f"  this config {sig}\nPoint data.prepared_dir to a new folder for this experiment.")
    json.dump(sig, open(sig_path, "w"), indent=1)
    todo = [c for c in cases if overwrite or not os.path.exists(os.path.join(out, c["id"], "info.json"))]
    if todo:
        log(f"prepare: {len(todo)} of {len(cases)} cases -> {out}")
        res = prepare_all(todo, out, cfg["preprocess"], workers or int(cfg["preprocess"]["workers"]), overwrite, log)
        bad = [r for r in res if "error" in r]
        if bad:
            raise SystemExit(f"{len(bad)} case(s) failed to prepare: {[r['id'] for r in bad]} "
                             f"(see <prepared_dir>/<id>/error.json)")
    else:
        log(f"prepare: all {len(cases)} cases already in {out}")


def ensure_split(cfg: Dict, log=print) -> Dict:
    d = cfg["data"]
    cases = [c for c in read_cases(d["cases"]) if c.get("label")]
    ids = sorted(c["id"] for c in cases)
    if os.path.exists(d["splits"]):
        split = json.load(open(d["splits"]))
        if sorted(x for f in split["folds"] for x in f) != ids or split["n_folds"] != int(d["n_folds"]):
            raise SystemExit(f"{d['splits']} does not match the labelled cases of {d['cases']} "
                             f"(or n_folds); delete it to rebuild")
        return split
    counts = prepared_lesion_counts(d["prepared_dir"], ids)
    split = make_split(cases, counts, int(d["n_folds"]), int(d["split_seed"]))
    os.makedirs(os.path.dirname(os.path.abspath(d["splits"])), exist_ok=True)
    json.dump(split, open(d["splits"], "w"), indent=1)
    log(f"split written to {d['splits']}: cases {split['cases_per_fold']}, lesions {split['lesions_per_fold']}")
    return split


def check_config(cfg: Dict) -> None:
    ch = cfg["train"]["channels"]
    allowed = {"image", "frst"} | set(cfg["preprocess"].get("extra_channels") or {})
    if not ch or ch[0] != "image" or any(c not in allowed for c in ch) or len(set(ch)) != len(ch):
        raise SystemExit("train.channels must start with 'image' and may add 'frst' and channels named in "
                         f"preprocess.extra_channels (available: {sorted(allowed)}); got {ch}")
    if "frst" in ch and not cfg["preprocess"]["frst"]:
        raise SystemExit("train.channels uses 'frst' but preprocess.frst is false")
    if int(cfg["data"]["n_folds"]) < 3:
        raise SystemExit("data.n_folds must be >= 3 (test fold + inner fold + training folds)")


def run_experiment(cfg: Dict, folds=None, workers: Optional[int] = None, log=print, final: bool = True) -> Dict:
    check_config(cfg)
    run_dir = os.path.join(cfg["output_root"], cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    cfg_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(cfg_path):
        import yaml
        old = yaml.safe_load(open(cfg_path))
        new = {k: v for k, v in cfg.items() if not k.startswith("_")}
        if json.dumps(old, sort_keys=True, default=str) != json.dumps(new, sort_keys=True, default=str):
            raise SystemExit(f"{run_dir} exists with a different config; choose another `name` or delete the folder")
    C.dump(cfg, cfg_path)
    t0 = time.time()
    ensure_prepared(cfg, workers, log=log)
    split = ensure_split(cfg, log)
    json.dump(split, open(os.path.join(run_dir, "splits.json"), "w"), indent=1)
    metas = run_cv(cfg, split, run_dir, folds, log)
    if len(metas) < split["n_folds"]:
        log(f"{len(metas)} of {split['n_folds']} folds done; summary covers those only")
    ev = evaluate_run(run_dir, cfg["data"]["prepared_dir"], "stored", log=log)
    s = run_summary(run_dir, cfg, metas, ev)
    log(f"experiment {cfg['name']}: pooled F1 {s['f1']} in {(time.time() - t0) / 3600:.2f} h -> {run_dir}")
    compare_runs(cfg["output_root"], log=lambda *_: None)
    if final and len(metas) == split["n_folds"]:      # the model to keep: trained on ALL cases (cmbx/final.py)
        from .final import train_final
        train_final(cfg, run_dir, log)
    return s
