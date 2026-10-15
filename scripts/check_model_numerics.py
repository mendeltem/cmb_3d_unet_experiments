#!/usr/bin/env python3
"""Numerics check: a research checkpoint (``modell.pt`` of cmb-valdo-chain, German key names) loaded into
``cmbx.model`` must reproduce the probability map the old pipeline stored (``<id>_pred_proba.nii.gz``,
uint8 = round(p * 255)) on the same grid case.

  python scripts/check_model_numerics.py --model <run>/fold0/modell.pt --case-dir <grid>/<id> --id <id> \
      --stored <run>/fold0/<id>_pred_proba.nii.gz --device cuda
Prints the max absolute difference in uint8 steps and the share of voxels that differ.
"""
import argparse
import json
import time

import nibabel as nib
import numpy as np
import torch

from cmbx.model import build_model, load_state
from cmbx.predict import predict_volume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--stored", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true", help="float32 instead of the original bfloat16 autocast")
    a = ap.parse_args()
    tr = lambda x: np.ascontiguousarray(np.asarray(x, np.float32).transpose(2, 1, 0))
    img = tr(nib.load(f"{a.case_dir}/{a.id}_image.nii.gz").dataobj)
    frs = tr(nib.load(f"{a.case_dir}/{a.id}_frst.nii.gz").dataobj)
    model = load_state(build_model("aniso_unet3d", 2, 32), a.model).to(a.device)
    t0 = time.time()
    p, n = predict_volume(model, [img, frs], (38, 62, 94), a.device, 8, not a.no_amp, False)
    new = np.round(p.transpose(2, 1, 0) * 255).astype(np.int16)
    old = np.asarray(nib.load(a.stored).dataobj).astype(np.int16)
    d = np.abs(new - old)
    out = dict(windows=n, seconds=round(time.time() - t0, 1), device=a.device, bf16=not a.no_amp,
               max_abs_diff_uint8=int(d.max()), voxels_differing=int((d > 0).sum()), voxels=int(d.size),
               share_differing=float((d > 0).mean()), voxels_diff_gt_1=int((d > 1).sum()),
               above_0_5_new=int((new > 127).sum()), above_0_5_old=int((old > 127).sum()))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
