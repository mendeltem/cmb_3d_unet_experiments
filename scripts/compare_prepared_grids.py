#!/usr/bin/env python3
"""Compare `cmbx prepare` output with another grid folder (e.g. one made by an earlier version or by
cmb-valdo-chain, ``<grid_dir>/<prefix><id>/``), voxel by voxel, for image / FRST / label.
Prints only aggregate numbers per case index (no case identifiers in the JSON output unless --with-ids).

  python scripts/compare_prepared_grids.py --new <prepared_dir> --old <grid_dir> --old-prefix c2s- \
      --ids <id> <id> --out /tmp/compare.json
"""
import argparse
import json

import nibabel as nib
import numpy as np
from scipy import ndimage

S26 = np.ones((3, 3, 3), bool)


def compare_case(new_dir, new_id, old_dir, old_id):
    out = {}
    for name in ("image", "frst", "label"):
        a = nib.load(f"{new_dir}/{new_id}_{name}.nii.gz"); b = nib.load(f"{old_dir}/{old_id}_{name}.nii.gz")
        r = dict(same_shape=a.shape == b.shape, affine_max_abs_diff=float(np.abs(a.affine - b.affine).max()))
        if a.shape == b.shape:
            x = np.asarray(a.dataobj, dtype=np.float64); y = np.asarray(b.dataobj, dtype=np.float64)
            d = np.abs(x - y)
            r.update(max_abs_diff=float(d.max()), voxels_differing=int((d > 0).sum()),
                     voxels_diff_gt_1e_3=int((d > 1e-3).sum()))
            if name == "label":
                r.update(lesion_voxels_new=int((x > 0).sum()), lesion_voxels_old=int((y > 0).sum()),
                         components_new=int(ndimage.label(x > 0, S26)[1]),
                         components_old=int(ndimage.label(y > 0, S26)[1]))
        else:
            r.update(shape_new=list(a.shape), shape_old=list(b.shape))
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--old-prefix", default="")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-ids", action="store_true")
    a = ap.parse_args()
    res = {}
    for i, sid in enumerate(a.ids):
        old_id = a.old_prefix + sid
        r = compare_case(f"{a.new}/{sid}", sid, f"{a.old}/{old_id}", old_id)
        res[sid if a.with_ids else f"case{i}"] = r
    agg = {}
    for name in ("image", "frst", "label"):
        rows = [r[name] for r in res.values()]
        agg[name] = dict(cases=len(rows), same_shape=sum(x["same_shape"] for x in rows),
                         affine_max_abs_diff=max(x["affine_max_abs_diff"] for x in rows),
                         max_abs_diff=max((x.get("max_abs_diff", float("nan")) for x in rows), default=None),
                         cases_with_voxels_diff_gt_1e_3=sum(1 for x in rows if x.get("voxels_diff_gt_1e_3", 1) > 0))
        if name == "label":
            agg[name].update(lesion_voxels_equal=sum(1 for x in rows if x.get("lesion_voxels_new") == x.get("lesion_voxels_old")),
                             components_equal=sum(1 for x in rows if x.get("components_new") == x.get("components_old")),
                             components_new_total=sum(x.get("components_new", 0) for x in rows),
                             components_old_total=sum(x.get("components_old", 0) for x in rows))
    json.dump({"aggregate": agg, "cases": res}, open(a.out, "w"), indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
