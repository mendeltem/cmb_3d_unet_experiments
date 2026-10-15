"""From the BIAS-CORRECTED native T2*/SWI volume + brain mask to the training grid (port of cmb-valdo-chain
``code/preproc_roh_bauen.py``, ``code/gitter_c_bauen.py`` and ``cmb/transfer/build_private_grids.py``).

Brain extraction and bias-field correction are deliberately NOT part of this package: they are done once, uniformly,
on the data (in our data: HD-BET or the scanner mask, then FSL ``fast -B`` on the brain-extracted image). The case
list points ``image`` at the bias-corrected volume.

  1. orientation: kept as in the file (``reorient: none``, VALDO grid) or ``as_closest_canonical`` (``ras``, in-house);
  2. brain mask: the given one, else ``image > 0`` with holes filled (skull-stripped input such as VALDO);
     ``brain_mask_mode: mask_and_positive`` additionally drops voxels <= 0 of the image, as
     ``cmb/transfer/build_private_grids.py`` did for the in-house cohorts;
  3. INPAINT HOOK (experiment entry point, off by default): ``inpaint(image, mask) -> image`` on the native grid,
     before the inversion -- where MicrobleedNet painted over vessels;
  4. inverted scale ``1 - I / max(I)`` inside the brain, 0 outside (bright microbleeds; 0 is reserved for outside);
  5. resampling to 0.5 x 0.5 x 1.0 mm with the CENTRE of the field of view preserved (source index
     ``i = off + r * j``, ``off = ((N - 1) - r (M - 1)) / 2``), linear, edge slices held (mode nearest); the label by
     nearest neighbour with a rescue of components that nearest sampling would drop;
  6. crop to the bounding box of the non-zero image plus 8 voxels (the affine is shifted, world coordinates exact);
  7. the FRST channel on that grid; the SynthSeg tissue map (if given) by nearest neighbour in world coordinates.

Output per case: ``<out>/<id>/<id>_{image,frst,label[,tissue]}.nii.gz`` + ``info.json`` (axes x, y, z).
"""
from __future__ import annotations

import importlib
import json
import os
import time
from typing import Callable, Dict, Optional

import nibabel as nib
import numpy as np
from scipy import ndimage

from .frst import brain_mask_from_image, frst_volume

S26 = np.ones((3, 3, 3), bool)


# ---------------------------------------------------------------- loading
def _squeeze(a: np.ndarray) -> np.ndarray:
    return a[..., 0] if a.ndim == 4 else a


def load_nifti(path: str, reorient: str = "none") -> nib.Nifti1Image:
    img = nib.load(path)
    if reorient == "ras":
        img = nib.as_closest_canonical(img)
    elif reorient != "none":
        raise ValueError(f"reorient must be 'none' or 'ras', not {reorient!r}")
    return img


def brain_mask(image: np.ndarray, mask_img: Optional[nib.Nifti1Image]) -> np.ndarray:
    """Given mask (> 0), else ``binary_fill_holes(image > 0)`` in 3D (MicrobleedNet ``preprocess_subject``)."""
    if mask_img is not None:
        m = _squeeze(np.asarray(mask_img.dataobj)) > 0
        if m.shape != image.shape:
            raise ValueError(f"brain mask {m.shape} is not on the image grid {image.shape}")
        return m
    return ndimage.binary_fill_holes(image > 0)


# ---------------------------------------------------------------- inpainting (EXPERIMENT ENTRY POINT)
def inpaint_mean(image: np.ndarray, mask: np.ndarray, brain: np.ndarray, ring_voxels: int = 2, **_) -> np.ndarray:
    """Reference implementation: every 26-connected component of ``mask`` is replaced by the mean of the brain
    voxels in a shell of ``ring_voxels`` around it (the 'neighbourhood mean' fill of MicrobleedNet's
    ``inpaint_vessels``, without its Frangi/KMeans vessel detection -- the mask is given)."""
    out = image.astype(np.float32).copy()
    lab, n = ndimage.label(mask & brain, structure=S26)
    for j, box in enumerate(ndimage.find_objects(lab), start=1):
        if box is None:
            continue
        lo = [max(s.start - ring_voxels - 1, 0) for s in box]
        hi = [min(s.stop + ring_voxels + 1, d) for s, d in zip(box, image.shape)]
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        comp = lab[sl] == j
        ring = ndimage.binary_dilation(comp, structure=S26, iterations=ring_voxels) & ~mask[sl] & brain[sl]
        if ring.any():
            out[sl][comp] = float(image[sl][ring].mean())
    return out


def inpaint_biharmonic(image: np.ndarray, mask: np.ndarray, brain: np.ndarray, ring_voxels: int = 2,
                       **_) -> np.ndarray:
    """Reference implementation: biharmonic inpainting (``skimage.restoration.inpaint_biharmonic``) per
    component, inside its bounding box plus a margin, so the cost stays proportional to the masked volume."""
    from skimage.restoration import inpaint_biharmonic
    out = image.astype(np.float32).copy()
    lab, n = ndimage.label(mask & brain, structure=S26)
    margin = max(ring_voxels, 2) + 1
    for j, box in enumerate(ndimage.find_objects(lab), start=1):
        if box is None:
            continue
        lo = [max(s.start - margin, 0) for s in box]
        hi = [min(s.stop + margin, d) for s, d in zip(box, image.shape)]
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        sub_mask = mask[sl] & brain[sl]
        filled = inpaint_biharmonic(image[sl].astype(np.float64), sub_mask)
        comp = lab[sl] == j
        out[sl][comp] = filled[comp].astype(np.float32)
    return out


INPAINTERS: Dict[str, Callable] = {"mean": inpaint_mean, "biharmonic": inpaint_biharmonic}


def resolve_inpainter(method: str) -> Optional[Callable]:
    """``none`` -> None; a built-in name; or ``package.module:function`` with the signature
    ``fn(image, mask, brain, **options) -> image`` (all arrays on the native grid, x y z)."""
    if method in (None, "none"):
        return None
    if method in INPAINTERS:
        return INPAINTERS[method]
    if ":" in method:
        mod, fn = method.split(":", 1)
        return getattr(importlib.import_module(mod), fn)
    raise ValueError(f"unknown inpaint method {method!r}: none | {' | '.join(INPAINTERS)} | module:function")


def inpaint(image: np.ndarray, mask: np.ndarray, brain: np.ndarray, method: str = "mean", dilate_voxels: int = 0,
            **options) -> np.ndarray:
    """THE HOOK: paint over ``mask`` (e.g. a vessel mask, other lesions) in the (bias-corrected) native image.

    Called by ``prepare_case`` when ``preprocess.inpaint.method`` is not ``none``. To test your own method, write
    ``def my_fill(image, mask, brain, **options) -> image`` in an importable module and set
    ``method: mypackage.mymodule:my_fill`` in the config. Warning from cmb-valdo-chain: MicrobleedNet's vessel
    inpainting hit 109 of 236 VALDO microbleeds (27 completely); removing it was the largest single gain (+0.136 F1).
    """
    fn = resolve_inpainter(method)
    if fn is None:
        return image
    mask = np.asarray(mask) > 0
    if dilate_voxels:
        mask = ndimage.binary_dilation(mask, structure=S26, iterations=int(dilate_voxels))
    return fn(image, mask, brain, **options).astype(np.float32)


# ---------------------------------------------------------------- inversion and grid
def invert(image: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """``1 - I / max(I)`` inside the brain mask (max over the brain), 0 outside (``preproc_roh_bauen.py``).

    As in the original, voxels inside the (hole-filled) mask with I = 0 become 1. Brain voxels that come out <= 0
    (the maximum voxel itself) get 1e-6 so that 'non-zero == brain' holds on the grid (the original left them at 0,
    i.e. background; that concerns single voxels)."""
    if not brain.any():
        raise ValueError("empty brain mask")
    scale = float(image[brain].max())
    if scale <= 0:
        raise ValueError("image is not positive inside the brain mask")
    out = np.where(brain, 1.0 - image / scale, 0.0).astype(np.float32)
    out[brain & (out <= 0)] = 1e-6
    return out


def target_grid(affine: np.ndarray, shape, spacing):
    """Target shape M, step r (source voxels per target voxel) and offset off with the field-of-view centre fixed."""
    spacing = np.asarray(spacing, float)
    s = np.sqrt((affine[:3, :3] ** 2).sum(0))
    N = np.array(shape[:3], dtype=float)
    M = np.rint(N * s / spacing).astype(int)
    r = spacing / s
    off = ((N - 1) - r * (M - 1)) / 2.0
    return s, M, r, off


def resample(volume: np.ndarray, r, off, M, order: int = 1) -> np.ndarray:
    return ndimage.affine_transform(volume.astype(np.float32), r, offset=off, output_shape=tuple(int(v) for v in M),
                                    order=order, mode="nearest", output=np.float32)


def label_nearest(lab: np.ndarray, r, off, M):
    """Label onto the target grid by nearest neighbour, with component bookkeeping (``gitter_c_bauen.label_nearest``).

    Nearest sampling when shrinking can skip a 1-voxel-thin component entirely. Such components are rescued: their
    indicator is resampled linearly within its own box and target voxels >= 0.5 x its maximum are set.
    Returns ``(uint8 mask, info)`` with rescued / merged / split counts."""
    lk, n = ndimage.label(lab > 0, structure=S26)
    tk = ndimage.affine_transform(lk.astype(np.int32), r, offset=off, output_shape=tuple(int(v) for v in M),
                                  order=0, mode="nearest", output=np.int32)
    missing = sorted(set(range(1, n + 1)) - set(np.unique(tk).tolist()))
    rescued = []
    if missing:
        boxes = ndimage.find_objects(lk)
        for k in missing:
            ks = boxes[k - 1]
            i0 = np.array([max(q.start - 2, 0) for q in ks])
            i1 = np.array([min(q.stop + 2, lk.shape[d]) for d, q in enumerate(ks)])
            ind = (lk[tuple(slice(a, b) for a, b in zip(i0, i1))] == k).astype(np.float32)
            j0 = np.maximum(np.floor((i0 - off) / r).astype(int), 0)
            j1 = np.minimum(np.ceil((i1 - 1 - off) / r).astype(int) + 1, np.array(M))
            w = ndimage.affine_transform(ind, r, offset=off + r * j0 - i0, output_shape=tuple(j1 - j0),
                                         order=1, mode="constant", cval=0.0, output=np.float32)
            sub = tk[tuple(slice(a, b) for a, b in zip(j0, j1))]
            put = (w >= 0.5 * w.max()) & (sub == 0)
            sub[put] = k
            rescued.append({"source_voxels": int(ind.sum()), "target_voxels": int(put.sum())})
    del lk
    full = (tk > 0).astype(np.uint8)
    tl, tn = ndimage.label(full, structure=S26)
    if full.any():
        pairs = np.unique(np.stack([tl[full > 0], tk[full > 0]]), axis=1)
        merged = int((np.bincount(pairs[0]) > 1).sum())
        split = int((np.bincount(pairs[1]) > 1).sum())
    else:
        merged = split = 0
    return full, {"rescued": rescued, "merged": merged, "split": split}


def n_components(lab: np.ndarray) -> int:
    return int(ndimage.label(lab > 0, structure=S26)[1])


def save_grid(arr: np.ndarray, affine: np.ndarray, ref_header, path: str, dtype, spacing) -> None:
    hdr = ref_header.copy()
    hdr.set_data_dtype(dtype)
    img = nib.Nifti1Image(np.asarray(arr, dtype=dtype), affine, hdr)
    img.set_sform(affine, code=int(ref_header["sform_code"]) or 1)
    img.set_qform(affine, code=int(ref_header["qform_code"]) or 1)
    img.header.set_zooms(tuple(float(z) for z in spacing))
    nib.save(img, path)


def tissue_to_grid(tissue_path: str, shape, affine: np.ndarray) -> np.ndarray:
    """SynthSeg map -> the prepared grid by nearest neighbour in world coordinates (any input grid)."""
    import nibabel.processing
    seg = nib.load(tissue_path)
    if seg.ndim == 4:
        seg = seg.slicer[..., 0]
    if seg.shape[:3] == tuple(shape) and np.allclose(seg.affine, affine, atol=1e-4):
        return np.asarray(seg.dataobj).astype(np.int16)
    out = nibabel.processing.resample_from_to(seg, (tuple(int(v) for v in shape), affine), order=0)
    return np.asarray(out.dataobj).astype(np.int16)


# ---------------------------------------------------------------- one case
def prepare_case(case: Dict, out_dir: str, pp: Dict, overwrite: bool = False) -> Dict:
    """Run steps 1-8 for one case; returns the info dict (also written to ``<out>/<id>/info.json``)."""
    sid = case["id"]
    dst = os.path.join(out_dir, sid)
    info_path = os.path.join(dst, "info.json")
    if os.path.exists(info_path) and not overwrite:
        return json.load(open(info_path))
    os.makedirs(dst, exist_ok=True)
    t0 = time.time()
    spacing = np.asarray(pp["spacing"], float)
    src = load_nifti(case["image"], pp["reorient"])
    image = _squeeze(np.asarray(src.dataobj, dtype=np.float32))
    if image.ndim != 3:
        raise ValueError(f"{sid}: expected a 3D image, got {image.shape}")

    affine_offsets = {}

    def same_grid(path):
        """Masks must have the image matrix; small header differences are tolerated (the image grid is used,
        as in the original scripts) and recorded in info.json."""
        im = load_nifti(path, pp["reorient"])
        if im.shape[:3] != image.shape:
            raise ValueError(f"{sid}: {os.path.basename(path)} {im.shape[:3]} is not on the image matrix {image.shape}")
        affine_offsets[os.path.basename(path)] = round(float(np.abs(im.affine - src.affine).max()), 6)
        if not np.allclose(im.affine[:3, :3], src.affine[:3, :3], atol=1e-2 * float(np.abs(src.affine[:3, :3]).max())):
            raise ValueError(f"{sid}: {os.path.basename(path)} has a different orientation/voxel size than the image")
        return im

    mask = brain_mask(image, same_grid(case["brain_mask"]) if case.get("brain_mask") else None)
    label = (_squeeze(np.asarray(same_grid(case["label"]).dataobj)) > 0).astype(np.uint8) if case.get("label") \
        else np.zeros(image.shape, np.uint8)
    corrected = image                               # the input IS the bias-corrected image (done outside)

    ip = pp.get("inpaint", {}) or {}
    inpaint_info = {"method": ip.get("method", "none")}
    if ip.get("method", "none") != "none":
        col = ip.get("mask_column", "inpaint_mask")
        if not case.get(col):
            raise ValueError(f"{sid}: inpaint method {ip['method']!r} needs the case-list column {col!r}")
        imask = _squeeze(np.asarray(same_grid(case[col]).dataobj)) > 0
        opts = {k: v for k, v in ip.items() if k not in ("method", "mask_column")}
        corrected = inpaint(corrected, imask, mask, method=ip["method"], **opts)
        inpaint_info.update(mask_voxels=int(imask.sum()),
                            label_voxels_under_mask=int((imask & (label > 0)).sum()))

    mode = pp.get("brain_mask_mode", "mask")
    if mode == "mask_and_positive":                 # build_private_grids.py (in-house cohorts): brain &= image > 0
        mask = mask & (corrected > 0)
    elif mode != "mask":
        raise ValueError(f"brain_mask_mode must be 'mask' or 'mask_and_positive', not {mode!r}")
    inverted = invert(corrected, mask)
    del corrected
    s, M, r, off = target_grid(src.affine, image.shape, spacing)
    full = resample(inverted, r, off, M, order=1)
    if int(pp.get("interpolation_order", 1)) != 1:          # higher order: clip, keep the linear brain support
        hi_order = resample(inverted, r, off, M, order=int(pp["interpolation_order"]))
        full = np.where(full > 0, np.clip(hi_order, 1e-6, 1.0), 0).astype(np.float32)
        del hi_order
    lab_full, lab_info = label_nearest(label, r, off, M)
    nz = np.nonzero(full)
    if nz[0].size == 0:
        raise ValueError(f"{sid}: nothing left after resampling (check the brain mask)")
    margin = int(pp["margin"])
    lo = np.maximum(np.array([a.min() for a in nz]) - margin, 0)
    hi = np.minimum(np.array([a.max() for a in nz]) + margin + 1, M)
    del nz
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    grid_img = np.ascontiguousarray(full[sl])
    grid_lab = np.ascontiguousarray(lab_full[sl])
    n_source, n_full, n_crop = n_components(label), n_components(lab_full), n_components(grid_lab)
    vox_full, vox_crop = int(lab_full.sum()), int(grid_lab.sum())
    del full, lab_full
    T = np.eye(4)
    T[:3, :3] = np.diag(r)
    T[:3, 3] = off + r * lo
    grid_aff = src.affine @ T

    brain_grid = brain_mask_from_image(grid_img)
    files = {"image": (grid_img, np.float32), "label": (grid_lab, np.uint8)}
    frst_info = None
    if pp.get("frst", True):
        fr, frst_info = frst_volume(grid_img, brain_grid)
        files["frst"] = (fr, np.float32)
    if case.get("tissue_map"):
        files["tissue"] = (tissue_to_grid(case["tissue_map"], grid_img.shape, grid_aff), np.int16)
    for name, (arr, dt) in files.items():
        save_grid(arr, grid_aff, src.header, os.path.join(dst, f"{sid}_{name}.nii.gz"), dt, spacing)

    info = dict(id=sid, source_shape=list(image.shape), source_spacing=[round(float(v), 5) for v in s],
                full_shape=[int(v) for v in M], shape=list(grid_img.shape),
                crop_lo=[int(v) for v in lo], crop_hi=[int(v) for v in hi], affine=np.round(grid_aff, 6).tolist(),
                n_lesions_source=n_source, n_lesions_full=n_full, n_lesions=n_crop,
                components_equal=bool(n_source == n_full == n_crop), label_resampling=lab_info,
                lesion_voxels=vox_crop, all_lesions_in_crop=bool(vox_full == vox_crop),
                brain_voxels=int(brain_grid.sum()), brain_mean=round(float(grid_img[brain_grid].mean()), 4),
                frst=frst_info, inpaint=inpaint_info, mask_affine_offsets=affine_offsets, preprocess=pp, seconds=round(time.time() - t0, 1))
    json.dump(info, open(info_path, "w"), indent=1)
    return info


def _worker(args):
    case, out_dir, pp, overwrite = args
    try:
        return prepare_case(case, out_dir, pp, overwrite)
    except Exception as e:                                  # one bad case must not stop the others
        import traceback
        os.makedirs(os.path.join(out_dir, case["id"]), exist_ok=True)
        json.dump({"id": case["id"], "error": repr(e), "trace": traceback.format_exc()},
                  open(os.path.join(out_dir, case["id"], "error.json"), "w"), indent=1)
        return {"id": case["id"], "error": repr(e)}


def prepare_all(cases, out_dir: str, pp: Dict, workers: int = 4, overwrite: bool = False, log=print):
    """Prepare every case (resumable: cases with info.json are skipped). One process per case, one thread each."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(c, out_dir, pp, overwrite) for c in cases]
    results = []
    if workers <= 1:
        it = map(_worker, jobs)
        for res in it:
            results.append(res)
            log(_line(res))
    else:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
            for res in pool.imap_unordered(_worker, jobs):
                results.append(res)
                log(_line(res))
    return results


def _line(info: Dict) -> str:
    if "error" in info:
        return f"{info['id']}: ERROR {info['error']}"
    return (f"{info['id']}: {info['source_shape']} @ {info['source_spacing']} mm -> {info['shape']}, "
            f"lesions {info['n_lesions_source']}/{info['n_lesions']}, {info['seconds']} s")
