"""Apply the fold models of a finished run (mean probability of the folds) to new cases.

The new cases are prepared with the run's own preprocessing settings into ``<out>/prepared``; outputs stay on that
grid (``<out>/<id>_prob.nii.gz`` uint8 = round(p * 255), ``<id>_mask.nii.gz``, ``<id>.json`` with the lesion
count). The operating cell defaults to the (threshold, min size) chosen most often by the folds' inner selection.
"""
from __future__ import annotations

import collections
import glob
import json
import os

import nibabel as nib
import numpy as np
import torch
import yaml

from .data import load_prepared, read_cases
from .model import build_model, load_state
from .postprocess import filter_components
from .preprocess import prepare_all
from .train import resolve_device


def predict_run(run_dir, cases_path, out, threshold=None, min_size=None, device="auto", csf_rule=False, log=print):
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
    T, S = cfg["train"], cfg["selection"]
    device = resolve_device(device)
    folds = sorted(glob.glob(os.path.join(run_dir, "fold[0-9]*", "model.pt")))
    folds = [f for f in folds if "_incomplete" not in f]
    if not folds:
        raise SystemExit(f"no fold models in {run_dir}")
    metas = [json.load(open(os.path.join(os.path.dirname(f), "meta.json"))) for f in folds]
    if threshold is None or min_size is None:
        (t, m), _ = collections.Counter((x["threshold"], x["min_size"]) for x in metas).most_common(1)[0]
        threshold = t if threshold is None else threshold
        min_size = m if min_size is None else min_size
    models = []
    for f in folds:
        mdl = build_model(cfg["model"]["name"], len(T["channels"]), int(cfg["model"]["base_channels"]))
        models.append(load_state(mdl, f).to(device).eval())
    prepared = os.path.join(out, "prepared")
    cases = read_cases(cases_path)
    res = prepare_all(cases, prepared, cfg["preprocess"], int(cfg["preprocess"].get("workers", 2)), False, log)
    if any("error" in r for r in res):
        raise SystemExit("some cases failed to prepare")
    from .predict import predict_volume
    for c in cases:
        r = load_prepared(prepared, c["id"], T["channels"], tissue=csf_rule)
        probs = [predict_volume(m, r.channels(T["channels"]), T["patch_size"], device, int(cfg["predict"]["batch_size"]),
                                bool(T["amp_bf16"]), bool(cfg["predict"]["tta_flip"]))[0] for m in models]
        p = np.mean(probs, axis=0)
        counter = {}
        mask = filter_components(p, threshold, min_size, int(S["max_component_voxels"]), counter,
                                 r.tissue if csf_rule else None, cfg["postprocess"]["csf_labels"])
        from scipy import ndimage
        n = int(ndimage.label(mask, structure=np.ones((3, 3, 3), bool))[1])
        back = lambda a: np.ascontiguousarray(a.transpose(2, 1, 0))
        nib.save(nib.Nifti1Image(back(np.round(p * 255).astype(np.uint8)), r.affine), os.path.join(out, f"{c['id']}_prob.nii.gz"))
        nib.save(nib.Nifti1Image(back(mask.astype(np.uint8)), r.affine), os.path.join(out, f"{c['id']}_mask.nii.gz"))
        json.dump(dict(id=c["id"], n_lesions=n, threshold=threshold, min_size=min_size, folds=len(models),
                       csf_rule=csf_rule, **counter), open(os.path.join(out, f"{c['id']}.json"), "w"), indent=1)
        log(f"{c['id']}: {n} microbleeds (threshold {threshold}, min size {min_size}, {len(models)} fold models)")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
