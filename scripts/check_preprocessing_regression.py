#!/usr/bin/env python3
"""Regression check: `cmbx` preprocessing against the training grid of cmb-valdo-chain (VALDO, ``dev/gitter_d``).

Starts, like every cmbx input, from the BIAS-CORRECTED image: the FSL FAST -B output the original used
(``dev/preproc/<id>/fast_restore.nii.gz``, made by cmb-valdo-chain ``code/preprocess.py``: ``fast -t 2 -n 3 -B -b
--nopve`` on the skull-stripped VALDO T2S), with brain mask = binary_fill_holes(VALDO T2S > 0) and the VALDO CMB
label; compared voxel by voxel with ``<gitter_d>/<id>/<id>_{image,frst,label}.nii.gz``.
Only for the machine that holds those folders; writes into --work (never into the repository).

  python scripts/check_preprocessing_regression.py --cases sub-101 sub-305 sub-228 \
      --valdo ~/data/extern/valdo2021/Task2 --preproc .../valdo-t2s/dev/preproc --grid .../valdo-t2s/dev/gitter_d \
      --work /tmp/cmbx-regression
"""
import argparse
import json
import os
import time

import nibabel as nib
import numpy as np
from scipy import ndimage

from cmbx.config import DEFAULTS
from cmbx.preprocess import prepare_case


def compare(new_dir, old_dir, sid):
    out = {}
    for name in ("image", "frst", "label"):
        a = nib.load(f"{new_dir}/{sid}_{name}.nii.gz"); b = nib.load(f"{old_dir}/{sid}_{name}.nii.gz")
        r = dict(shape_new=list(a.shape), shape_old=list(b.shape),
                 affine_max_abs_diff=float(np.abs(a.affine - b.affine).max()))
        if a.shape == b.shape:
            x = np.asarray(a.dataobj, dtype=np.float64); y = np.asarray(b.dataobj, dtype=np.float64)
            d = np.abs(x - y)
            r.update(max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()), voxels_differing=int((d > 0).sum()),
                     voxels_diff_gt_1e_3=int((d > 1e-3).sum()))
            if name == "label":
                s26 = np.ones((3, 3, 3), bool)
                r.update(lesion_voxels_new=int((x > 0).sum()), lesion_voxels_old=int((y > 0).sum()),
                         components_new=int(ndimage.label(x > 0, s26)[1]),
                         components_old=int(ndimage.label(y > 0, s26)[1]))
            if name == "image":
                r.update(nonzero_new=int((x > 0).sum()), nonzero_old=int((y > 0).sum()))
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--valdo", required=True)
    ap.add_argument("--preproc", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    res = {}
    for sid in a.cases:
        t2s = f"{a.valdo}/{sid}/{sid}_space-T2S_desc-masked_T2S.nii.gz"
        lab = f"{a.valdo}/{sid}/{sid}_space-T2S_CMB.nii.gz"
        pp = json.loads(json.dumps(DEFAULTS["preprocess"]))
        v = nib.load(t2s)
        mpath = f"{a.work}/{sid}_mask.nii.gz"
        nib.save(nib.Nifti1Image(ndimage.binary_fill_holes(np.asarray(v.dataobj) > 0).astype(np.uint8), v.affine), mpath)
        case = dict(id=sid, image=f"{a.preproc}/{sid}/fast_restore.nii.gz", label=lab, brain_mask=mpath)
        t0 = time.time()
        prepare_case(case, a.work, pp, overwrite=True)
        r = compare(f"{a.work}/{sid}", f"{a.grid}/{sid}", sid)
        r["seconds"] = round(time.time() - t0, 1)
        res[sid] = r
        print(sid, json.dumps(r), flush=True)
    json.dump(res, open(f"{a.work}/regression.json", "w"), indent=1)


if __name__ == "__main__":
    main()
