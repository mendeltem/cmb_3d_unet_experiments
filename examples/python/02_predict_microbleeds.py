"""Find cerebral microbleeds in new scans with a trained 3D U-Net, step by step in plain Python.

    python examples/python/02_predict_microbleeds.py -i examples/new/derivatives/preprocessed -m ../my_runs/py_model -o ../my_runs/py_pred

The steps of `cmbx evaluate` (step 1 of the method, without the optional classifier of step 2) written out. Same
masks as `cmbx evaluate -i ... -m ... -o ...` (tests/test_python_examples.py checks it).

-m     a model folder from 01_train_unet.py, 03_fine_tune_unet.py or cmbx train, or `pretrained`: the VALDO T2* model
       (CC BY-NC-SA 4.0, downloaded once into models/valdo_t2)
Input  -i  one folder per case with <name>/<name>_t2star.nii.gz (or _swi) and <name>/<name>_brainmask.nii.gz,
           bias-corrected like the training scans; no microbleed mask needed
Output -o  <name>_cmb.nii.gz       microbleed mask on the scan's own grid (same matrix and affine as the scan)
           <name>_lesions.nii.gz   the same as a label map: microbleed k of microbleeds.csv has value k
           <name>_prob.nii.gz      probability per voxel, on the scan's grid
           microbleeds.csv         one row per microbleed: volume, highest probability, centre in mm and as scan voxel
           predictions.csv         one row per case: number of microbleeds, total volume
           grid/                   the cases on the model grid (0.5 x 0.5 x 1 mm)
"""
import argparse
import csv
import json
import os

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from cmbx.config import load_config
from cmbx.data import load_prepared
from cmbx.model import build_model, load_state
from cmbx.postprocess import filter_components
from cmbx.predict import predict_volume
from cmbx.predict_cli import to_native
from cmbx.preprocess import prepare_all
from cmbx.train import resolve_device
from cmbx.workflow import collect, pretrained_dir

TOUCHING = np.ones((3, 3, 3), bool)     # voxels that share a face, an edge or a corner belong to the same microbleed


def load_model(model_dir, device="cpu"):
    """The network with the weights of <model>/final/model.pt, the settings of <model>/config.yaml and the threshold
    and minimum size of <model>/final/meta.json. Returns (net, settings, threshold, min_size)."""
    weights = os.path.join(model_dir, "final", "model.pt")
    if not os.path.exists(weights):
        raise SystemExit(f"{weights} not found: give a folder written by 01_train_unet.py, 03_fine_tune_unet.py or "
                         "cmbx train, or pretrained")
    cfg = load_config(os.path.join(model_dir, "config.yaml"))
    meta = json.load(open(os.path.join(model_dir, "final", "meta.json")))
    net = build_model(cfg["model"]["name"], len(cfg["train"]["channels"]), int(cfg["model"]["base_channels"]))
    load_state(net, weights)
    return net.to(device).eval(), cfg, float(meta["threshold"]), int(meta["min_size"])


def prepare_grid(cases, folder, settings, workers=4, force=False):
    """Bring every case onto the model grid in `folder`. A case already there is kept, unless the settings changed or
    force is set. Stops with the names of the cases that failed."""
    signature = json.loads(json.dumps({k: v for k, v in settings.items() if k != "workers"}))
    signature_file = os.path.join(folder, "preprocess.json")
    changed = os.path.exists(signature_file) and json.load(open(signature_file)) != signature
    os.makedirs(folder, exist_ok=True)
    results = prepare_all(cases, folder, settings, workers, overwrite=force or changed, log=print)
    failed = [r["id"] for r in results if "error" in r]
    if failed:
        raise SystemExit(f"{len(failed)} case(s) could not be prepared: {failed} (see {folder}/<name>/error.json)")
    json.dump(signature, open(signature_file, "w"), indent=1)


def find_microbleeds(net, case, cfg, threshold, min_size, device="cpu"):
    """Probability map (sliding window of the training patch size, half-window steps, overlaps averaged), then keep the
    connected spots above `threshold` with at least `min_size` and at most max_component_voxels voxels (larger is no
    microbleed). Returns (probability, label map, number of microbleeds), arrays in model order (z, y, x)."""
    T, S, PP = cfg["train"], cfg["selection"], cfg["postprocess"]
    prob = predict_volume(net, case.channels(T["channels"]), tuple(T["patch_size"]), device,
                          int(cfg["predict"]["batch_size"]), bool(T["amp_bf16"]), bool(cfg["predict"]["tta_flip"]))[0]
    mask = filter_components(prob, threshold, min_size, int(S["max_component_voxels"]),
                             tissue=case.tissue if PP["csf_rule"] else None, csf_labels=PP["csf_labels"])
    labels, n = ndimage.label(mask, structure=TOUCHING)
    return prob, labels, n


def write_case(name, scan, labels, prob, grid_affine, out_dir):
    """Write the mask, label map and probability of one case on the grid of `scan` (nearest neighbour; a microbleed
    smaller than one scan voxel keeps its centre voxel, so none is lost) and return one row per microbleed."""
    labels_xyz = np.ascontiguousarray(labels.transpose(2, 1, 0))    # model order (z, y, x) -> NIfTI order (x, y, z)
    prob_xyz = np.ascontiguousarray(prob.transpose(2, 1, 0))
    img = nib.load(scan)
    native, native_prob, centres = to_native(labels_xyz, prob_xyz, grid_affine, (tuple(img.shape[:3]), img.affine))
    nib.save(nib.Nifti1Image((native > 0).astype(np.uint8), img.affine), os.path.join(out_dir, f"{name}_cmb.nii.gz"))
    nib.save(nib.Nifti1Image(native.astype(np.uint16), img.affine), os.path.join(out_dir, f"{name}_lesions.nii.gz"))
    nib.save(nib.Nifti1Image(np.clip(native_prob, 0, 1).astype(np.float32), img.affine),
             os.path.join(out_dir, f"{name}_prob.nii.gz"))
    voxel_mm3 = float(abs(np.linalg.det(grid_affine[:3, :3])))
    rows = []
    for k, box in enumerate(ndimage.find_objects(labels_xyz), start=1):
        if box is None:
            continue
        inside = labels_xyz[box] == k
        (x, y, z), (i, j, kk) = centres[k]
        rows.append(dict(case=name, microbleed=k, volume_mm3=round(float(inside.sum()) * voxel_mm3, 3),
                         max_prob=round(float(prob_xyz[box][inside].max()), 3), x_mm=round(float(x), 2),
                         y_mm=round(float(y), 2), z_mm=round(float(z), 2), i=int(i), j=int(j), k=int(kk)))
    return rows


if __name__ == "__main__":     # the steps; worker processes of step 3 import this file and must not run them again
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("-i", "--inp_dir", required=True, help="folder with one sub-folder per case")
    p.add_argument("-m", "--model_dir", required=True, help="model folder, or pretrained")
    p.add_argument("-o", "--out_dir", required=True, help="output folder")
    p.add_argument("--threshold", type=float, default=None, help="[chosen in training]")
    p.add_argument("--min_size", type=int, default=None, help="minimum size in model-grid voxels [chosen in training]")
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu [auto]")
    p.add_argument("--workers", type=int, default=4, help="cases prepared in parallel in step 3 [4]")
    p.add_argument("--force", default="false", help="true: predict again into a folder that holds predictions [false]")
    a = p.parse_args()
    if os.path.exists(os.path.join(a.out_dir, "predictions.csv")) and a.force != "true":
        raise SystemExit(f"{a.out_dir} holds predictions already; --force true predicts again")

    # 1 the model: network, weights, and the settings it was trained with
    model_dir = pretrained_dir() if a.model_dir in ("pre", "pretrained") else a.model_dir
    device = resolve_device(a.device)
    net, cfg, threshold, min_size = load_model(model_dir, device)
    threshold = threshold if a.threshold is None else a.threshold
    min_size = min_size if a.min_size is None else a.min_size
    T, csf = cfg["train"], bool(cfg["postprocess"]["csf_rule"])
    print(f"model {model_dir}: channels {T['channels']}, threshold {threshold}, minimum size {min_size}", flush=True)
    if os.path.exists(os.path.join(model_dir, "stage2", "classifier.pt")):
        print("note: this model also has a step-2 classifier (stage2/); this example applies step 1 only, "
              "cmbx evaluate applies both")

    # 2 the cases: scan and brain mask of each
    os.makedirs(a.out_dir, exist_ok=True)
    cases = collect(a.inp_dir, need_tissue=csf, need_flair="flair" in T["channels"], work=os.path.join(a.out_dir, "cmbx"))
    print(f"{len(cases)} cases in {a.inp_dir}", flush=True)

    # 3 every case onto the grid the model was trained on (its own preprocessing settings)
    grid = os.path.join(a.out_dir, "grid")
    prepare_grid(cases, grid, cfg["preprocess"], a.workers, a.force == "true")

    # 4 per case: probability, microbleeds, back onto the scan's grid
    all_rows, summary = [], []
    with torch.no_grad():
        for c in cases:
            case = load_prepared(grid, c["id"], T["channels"], tissue=csf)
            prob, labels, n = find_microbleeds(net, case, cfg, threshold, min_size, device)
            rows = write_case(c["id"], c["image"], labels, prob, case.affine, a.out_dir)
            all_rows += rows
            summary.append(dict(case=c["id"], microbleeds=n, total_volume_mm3=round(sum(r["volume_mm3"] for r in rows), 3),
                                mask=os.path.join(a.out_dir, f"{c['id']}_cmb.nii.gz")))
            print(f"{c['id']}: {n} microbleed(s)", flush=True)

    # 5 the two tables
    columns = ["case", "microbleed", "volume_mm3", "max_prob", "x_mm", "y_mm", "z_mm", "i", "j", "k"]
    with open(os.path.join(a.out_dir, "microbleeds.csv"), "w", newline="") as f:
        csv.DictWriter(f, fieldnames=columns).writeheader()
        csv.DictWriter(f, fieldnames=columns).writerows(all_rows)
    with open(os.path.join(a.out_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"{len(cases)} case(s) -> {a.out_dir}/ (masks <name>_cmb.nii.gz, tables microbleeds.csv, predictions.csv)")
