"""``cmbx selftest``: one-minute end-to-end check on synthetic phantoms, no data needed.

Runs prepare -> split -> 5-fold CV (2 epochs, small network) -> evaluation -> final model on 10 phantoms, on the GPU
if one is visible (``--device auto``), and checks that every step wrote its outputs. Numbers from it mean nothing.
Phantoms: generated into a temporary folder, or ``--data DIR`` with a ``cases.csv`` (e.g. ``examples/phantoms``)."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time

from .config import load_config
from .experiments import run_experiment
from .synthetic import write_phantom_dataset


def selftest(data: str | None = None, device: str = "auto", keep: bool = False, log=print) -> int:
    t0 = time.time()
    work = tempfile.mkdtemp(prefix="cmbx_selftest_")
    try:
        cases = os.path.join(data, "cases.csv") if data else write_phantom_dataset(os.path.join(work, "phantoms"), 10, 0)
        if not os.path.exists(cases):
            log(f"selftest FAILED: no case list {cases}")
            return 1
        cfg = load_config(None, {
            "name": "selftest", "output_root": os.path.join(work, "runs"),
            "data": {"cases": cases, "prepared_dir": os.path.join(work, "prepared"),
                     "splits": os.path.join(work, "splits.json"), "n_folds": 5},
            "preprocess": {"workers": 2},
            "model": {"base_channels": 8},
            "train": {"epochs": 2, "min_epochs": 2, "eval_every": 1, "n_patches": 8, "batch_size": 4,
                      "patch_size": [16, 32, 32], "device": device},
            "postprocess": {"csf_rule": True}})
        s = run_experiment(cfg, log=log)
        run = os.path.join(work, "runs", "selftest")
        metas = [json.load(open(os.path.join(run, f"fold{k}", "meta.json"))) for k in range(5)]
        final = json.load(open(os.path.join(run, "final", "meta.json")))
        problems = [f"fold {m.get('fold')}" for m in metas if not m.get("ok")]
        if not final.get("ok") or not os.path.exists(os.path.join(run, "final", "model.pt")):
            problems.append("final model")
        if problems or not 0 <= s["f1"] <= 1:
            log(f"selftest FAILED: {', '.join(problems) or 'no valid score'} (outputs in {work})")
            keep = True
            return 1
        log(f"selftest PASSED: 5-fold CV + final model on {s['cases']} phantoms, device {metas[0].get('device')}, "
            f"{time.time() - t0:.0f} s" + (f" (outputs kept in {work})" if keep else ""))
        return 0
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
