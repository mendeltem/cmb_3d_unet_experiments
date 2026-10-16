"""``cmbx predict``: apply a finished run to new scans -- its final model (default) or its fold models (``--folds``).

Input: a case list (``--cases``, columns ``id``, ``image``, optional ``brain_mask``, ``tissue_map``) or one scan
(``--image`` [``--brain-mask``] [``--tissue-map``] [``--id``]). The image must be bias-corrected like the training data.
The cases are prepared with the run's own preprocessing settings into ``<out>/prepared`` (nothing else is needed).

Outputs in ``<out>/``:
  <id>_mask.nii.gz, <id>_prob.nii.gz   on the INPUT image grid (same matrix and affine as the scan); prob = uint8 p*255
  grid/<id>_mask.nii.gz, ..._prob...   on the model grid (0.5 x 0.5 x 1 mm, cropped) where the model works
  <id>.json                            settings + one entry per microbleed (volume, max probability, centre in mm and
                                       as input-image voxel)
  predictions.csv                      one row per case
Back to the input grid: nearest neighbour in world coordinates; a microbleed smaller than one input voxel that would
vanish keeps its centre voxel, so the lesion count is the same on both grids.
The operating cell (threshold, minimum size) defaults to the one the CV folds chose most often.
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import os

import nibabel as nib
import nibabel.processing
import numpy as np
import torch
import yaml
from scipy import ndimage

from .data import load_prepared, read_cases
from .model import build_model, load_state
from .postprocess import filter_components
from .preprocess import prepare_all
from .train import resolve_device

S26 = np.ones((3, 3, 3), bool)


def _native_target(path: str):
    img = nib.load(path)
    return tuple(int(v) for v in img.shape[:3]), img.affine


def to_native(labels_xyz: np.ndarray, prob_xyz: np.ndarray, grid_affine: np.ndarray, target):
    """Model-grid labels (int) and probability (float) -> the input image grid; every grid component survives."""
    shape, aff = target
    lab_img = nib.Nifti1Image(labels_xyz.astype(np.int32), grid_affine)
    lab = np.asarray(nibabel.processing.resample_from_to(lab_img, target, order=0).dataobj).astype(np.int32)
    prob = np.asarray(nibabel.processing.resample_from_to(nib.Nifti1Image(prob_xyz.astype(np.float32), grid_affine),
                                                          target, order=1).dataobj, dtype=np.float32)
    present = set(np.unique(lab).tolist())
    inv = np.linalg.inv(aff)
    centres = {}
    for k, sl in enumerate(ndimage.find_objects(labels_xyz.astype(np.int32)), start=1):
        if sl is None:
            continue
        idx = np.argwhere(labels_xyz[sl] == k) + [s.start for s in sl]
        world = grid_affine @ np.r_[idx.mean(0), 1.0]
        vox = np.clip(np.round((inv @ world)[:3]).astype(int), 0, np.array(shape) - 1)
        centres[k] = (world[:3], vox)
        if k not in present:
            lab[tuple(vox)] = k
    return lab, prob, centres


def _load_models(run_dir, use_folds, device):
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"{run_dir} is not a finished run (no config.yaml); give the folder that holds final/ and fold0/")
    cfg = yaml.safe_load(open(cfg_path))
    final = os.path.join(run_dir, "final", "model.pt")
    if os.path.exists(final) and not use_folds:        # default: the one model trained on all cases (cmbx/final.py)
        paths = [final]
    else:
        paths = [f for f in sorted(glob.glob(os.path.join(run_dir, "fold[0-9]*", "model.pt"))) if "_incomplete" not in f]
        if not use_folds and paths:
            print(f"note: {run_dir} has no final model (final/model.pt); using its {len(paths)} fold model(s) instead", flush=True)
    if not paths:
        raise SystemExit(f"no final or fold models in {run_dir}")
    T = cfg["train"]
    models = []
    for f in paths:
        mdl = build_model(cfg["model"]["name"], len(T["channels"]), int(cfg["model"]["base_channels"]))
        models.append(load_state(mdl, f).to(device).eval())
    metas = [json.load(open(os.path.join(os.path.dirname(f), "meta.json"))) for f in paths]
    return cfg, models, metas, ("final model" if paths == [final] else f"{len(paths)} fold models")


def predict_run(run_dir, cases_path=None, out="pred", threshold=None, min_size=None, device="auto", csf_rule=False,
                log=print, use_folds=False, image=None, brain_mask=None, tissue_map=None, case_id=None, native=True):
    device = resolve_device(device)
    cfg, models, metas, what = _load_models(run_dir, use_folds, device)
    T, S = cfg["train"], cfg["selection"]
    log(f"using the {what} of {run_dir} on {device}")
    if threshold is None or min_size is None:
        (t, m), _ = collections.Counter((x["threshold"], x["min_size"]) for x in metas).most_common(1)[0]
        threshold = t if threshold is None else threshold
        min_size = m if min_size is None else min_size
    if image:
        sid = case_id or os.path.basename(image).split(".nii")[0]
        cases = [dict(id=sid, image=os.path.abspath(image), brain_mask=os.path.abspath(brain_mask) if brain_mask else None,
                      tissue_map=os.path.abspath(tissue_map) if tissue_map else None)]
    elif cases_path:
        cases = read_cases(cases_path)
    else:
        raise SystemExit("give --cases <case list> or --image <scan>")
    for c in cases:
        for col in ("image", "brain_mask", "tissue_map"):
            if c.get(col) and not os.path.exists(c[col]):
                raise SystemExit(f"{c['id']}: {col} not found: {c[col]}")
        if csf_rule and not c.get("tissue_map"):
            raise SystemExit(f"{c['id']}: --csf-rule needs a tissue map (column tissue_map / --tissue-map, SynthSeg)")
        for name, col in (cfg["preprocess"].get("extra_channels") or {}).items():
            if name in T["channels"] and not c.get(col):
                raise SystemExit(f"{c['id']}: the run uses the extra channel {name!r}: column {col!r} is needed")
    os.makedirs(os.path.join(out, "grid"), exist_ok=True)
    prepared = os.path.join(out, "prepared")
    res = prepare_all(cases, prepared, cfg["preprocess"], int(cfg["preprocess"].get("workers", 2)), False, log)
    bad = [r for r in res if "error" in r]
    if bad:
        raise SystemExit("could not prepare: " + "; ".join(f"{r.get('id')}: {r['error']}" for r in bad))
    from .predict import predict_volume
    back = lambda a: np.ascontiguousarray(a.transpose(2, 1, 0))         # model order (z, y, x) -> NIfTI (x, y, z)
    rows = []
    for c in cases:
        r = load_prepared(prepared, c["id"], T["channels"], tissue=csf_rule)
        probs = [predict_volume(m, r.channels(T["channels"]), T["patch_size"], device, int(cfg["predict"]["batch_size"]),
                                bool(T["amp_bf16"]), bool(cfg["predict"]["tta_flip"]))[0] for m in models]
        p = np.mean(probs, axis=0)
        counter = {}
        mask = filter_components(p, threshold, min_size, int(S["max_component_voxels"]), counter,
                                 r.tissue if csf_rule else None, cfg["postprocess"]["csf_labels"])
        lab, n = ndimage.label(mask, structure=S26)
        lab_x, p_x = back(lab), back(p)
        nib.save(nib.Nifti1Image(np.round(p_x * 255).astype(np.uint8), r.affine), os.path.join(out, "grid", f"{c['id']}_prob.nii.gz"))
        nib.save(nib.Nifti1Image((lab_x > 0).astype(np.uint8), r.affine), os.path.join(out, "grid", f"{c['id']}_mask.nii.gz"))
        vox_mm3 = float(abs(np.linalg.det(r.affine[:3, :3])))
        lesions, centres = [], {}
        if native:
            target = _native_target(c["image"])
            nlab, nprob, centres = to_native(lab_x, p_x, r.affine, target)
            nib.save(nib.Nifti1Image(np.round(np.clip(nprob, 0, 1) * 255).astype(np.uint8), target[1]), os.path.join(out, f"{c['id']}_prob.nii.gz"))
            nib.save(nib.Nifti1Image((nlab > 0).astype(np.uint8), target[1]), os.path.join(out, f"{c['id']}_mask.nii.gz"))
            n_native = int(len(set(np.unique(nlab).tolist()) - {0}))
        for k, sl in enumerate(ndimage.find_objects(lab_x), start=1):
            if sl is None:
                continue
            sel = lab_x[sl] == k
            e = dict(lesion=k, grid_voxels=int(sel.sum()), volume_mm3=round(sel.sum() * vox_mm3, 3),
                     max_prob=round(float(p_x[sl][sel].max()), 3))
            if k in centres:
                e.update(centre_mm=[round(float(v), 2) for v in centres[k][0]], centre_voxel=[int(v) for v in centres[k][1]])
            lesions.append(e)
        info = dict(id=c["id"], n_lesions=int(n), total_volume_mm3=round(sum(e["volume_mm3"] for e in lesions), 3),
                    threshold=threshold, min_size=min_size, model=what, folds=len(models), run=os.path.abspath(run_dir), csf_rule=csf_rule,
                    native_grid=bool(native), **counter, lesions=lesions)
        if native and n_native != n:
            info["warning"] = f"{n} lesions on the model grid but {n_native} on the input grid"
            log(f"WARNING {c['id']}: {info['warning']}")
        json.dump(info, open(os.path.join(out, f"{c['id']}.json"), "w"), indent=1)
        rows.append(dict(id=c["id"], n_lesions=int(n), total_volume_mm3=info["total_volume_mm3"],
                         mask=os.path.join(out, f"{c['id']}_mask.nii.gz") if native else os.path.join(out, "grid", f"{c['id']}_mask.nii.gz")))
        log(f"{c['id']}: {n} microbleeds (threshold {threshold}, min size {min_size}, {what})")
    with open(os.path.join(out, "predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log(f"{len(rows)} cases -> {out}/ (masks on the input grid; model-grid copies in {out}/grid)")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows
