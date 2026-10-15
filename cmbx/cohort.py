"""Case list for the in-house cohorts from the prepared dataset layout (group drive ALITA/DATA/CHARITE, mirrored
locally). Brain extraction and bias correction are NOT done here: they were done once on the data.

Layout (one source folder per modality, read only):
    <source>/participants.tsv                               the cohort (participant_id, ...)
    <source>/sub-X/<t2|swi>/sub-X_<T2|SWI>.nii.gz            raw image (only used for the quality checks)
    <source>/sub-X/<t2|swi>/sub-X_<T2|SWI>_roi.nii.gz        reference microbleed mask
    <source>/sub-X/<t2|swi>/sub-X_<T2|SWI>_brainmask.nii.gz  brain mask
    <biascorr_dir>/sub-X/anat/sub-X*_desc-biascorr*.nii.gz    bias-corrected image (FSL FAST -B on the brain-extracted
                                                              image) -> becomes the case-list ``image``
    excluded.tsv / duplicates.tsv (participant_id)           cases left out (also any --exclude-tsv file)

Default ``biascorr_dir``: ``<source>/../derivatives/biascorr/<t2star|swi>`` (the ALITA layout). A case without a
source brain mask takes ``<brainmask_dir>/sub-X/anat/*brain_mask*`` if given; a case missing image, reference mask,
brain mask or bias-corrected image is NOT silently dropped: it is listed with the reason (``missing.tsv`` and screen).

Quality checks (report only, they never exclude; from cmb-valdo-chain ``catalina/kohorte_bauen.py``):
  * mask fit: mean intensity in the reference mask / mean in a ring around it (dilation 7x7x3 minus 3x3x1, inside the
    brain mask) > 0.9 -> flagged "mask not darker than surroundings";
  * header offset: reference mask header offset from the image by > 0.25 x the smallest voxel size -> flagged. (The
    preparation uses the mask array on the image grid, which is what rewriting the header did.)
Folds: lesion classes 0 / 1-2 / 3-5 / >5 (26-connected components of the reference), ``numpy.random.default_rng(0)``
shuffle per class, round-robin over 5 folds with the counter running on across classes -- the folds of all earlier
in-house results.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from scipy import ndimage

MODALITIES = {"t2star": ("t2", "T2"), "swi": ("swi", "SWI")}
CLASSES = [(0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10 ** 6, ">5")]


def lesion_class(n: int) -> str:
    for lo, hi, name in CLASSES:
        if lo <= n <= hi:
            return name
    raise ValueError(n)


def corner_offset_mm(a: nib.Nifti1Image, b: nib.Nifti1Image) -> float:
    nx, ny, nz = a.shape[:3]
    c = np.array([[0, 0, 0, 1], [nx, 0, 0, 1], [0, ny, 0, 1], [0, 0, nz, 1], [nx, ny, 0, 1], [nx, 0, nz, 1],
                  [0, ny, nz, 1], [nx, ny, nz, 1]], float).T
    return float(np.abs((a.affine @ c)[:3] - (b.affine @ c)[:3]).max())


def mask_fit_ratio(image: nib.Nifti1Image, roi: nib.Nifti1Image, brain: nib.Nifti1Image) -> Optional[float]:
    """Mean intensity in the reference mask / mean in the ring around it (``kohorte_bauen.ausschluss``).
    None if the mask is empty; inf if there is no ring inside the brain mask."""
    m = np.asarray(roi.dataobj) > 0
    if m.sum() == 0:
        return None
    b = np.asarray(image.dataobj).astype(float)
    h = np.asarray(brain.dataobj) > 0
    ring = ndimage.binary_dilation(m, np.ones((7, 7, 3))) & ~ndimage.binary_dilation(m, np.ones((3, 3, 1))) & h
    if ring.sum() == 0:
        return float("inf")
    return float(b[m].mean() / b[ring].mean())


def assign_folds(rows: List[Dict], n_folds: int = 5, seed: int = 0) -> None:
    """``kohorte_bauen.py``: per class (fixed order) shuffle with default_rng(seed), round-robin with a counter that
    runs on across classes."""
    rng = np.random.default_rng(seed)
    start = 0
    for _, _, k in CLASSES:
        idx = np.array([i for i, r in enumerate(rows) if r["lesion_class"] == k], dtype=int)
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            rows[j]["fold"] = (start + i) % n_folds
        start += len(idx)


def _ids_from_tsv(path: str) -> set:
    with open(path, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if not rows:
        return set()
    col = rows[0].index("participant_id") if "participant_id" in rows[0] else 0
    return {r[col] for r in rows[1:] if len(r) > col and r[col]}


def _first(patterns) -> Optional[str]:
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None


def build_caselist(source: str, modality: str, biascorr_dir: Optional[str] = None,
                   brainmask_dir: Optional[str] = None, exclude_tsv: Optional[List[str]] = None,
                   tissue_pattern: Optional[str] = None, quality_checks: bool = True, report_dir: Optional[str] = None,
                   log=print) -> List[Dict]:
    sub, suffix = MODALITIES[modality]
    src = os.path.abspath(source)
    biascorr_dir = biascorr_dir or os.path.join(os.path.dirname(src), "derivatives", "biascorr", modality)
    with open(os.path.join(src, "participants.tsv"), newline="") as f:
        ids = [r["participant_id"] for r in csv.DictReader(f, delimiter="\t")]
    ex_files = [os.path.join(src, n) for n in ("excluded.tsv", "duplicates.tsv", "ausgeschlossen.tsv")
                if os.path.exists(os.path.join(src, n))] + list(exclude_tsv or [])
    excluded = set().union(*[_ids_from_tsv(p) for p in ex_files]) if ex_files else set()
    log(f"{src}: {len(ids)} in participants.tsv, {len(excluded & set(ids))} of them excluded by "
        f"{[os.path.basename(p) for p in ex_files] or 'no exclusion file'}; bias-corrected images from {biascorr_dir}")
    rows, missing, quality = [], [], []
    for sid in ids:
        if sid in excluded:
            continue
        d = os.path.join(src, sid, sub)
        raw = os.path.join(d, f"{sid}_{suffix}.nii.gz")
        roi = os.path.join(d, f"{sid}_{suffix}_roi.nii.gz")
        brain = os.path.join(d, f"{sid}_{suffix}_brainmask.nii.gz")
        brain_from = "source"
        if not os.path.exists(brain):
            brain = _first([os.path.join(brainmask_dir, sid, "anat", "*brain*mask*.nii*"),
                            os.path.join(brainmask_dir, f"{sid}*brain*mask*.nii*")]) if brainmask_dir else None
            brain_from = "brainmask_dir"
        image = _first([os.path.join(biascorr_dir, sid, "anat", f"{sid}*desc-biascorr*.nii*"),
                        os.path.join(biascorr_dir, f"{sid}*desc-biascorr*.nii*")])
        lack = [n for n, p in (("reference mask", roi), ("brain mask", brain), ("bias-corrected image", image))
                if not p or not os.path.exists(p)]
        if lack:
            missing.append((sid, ", ".join(lack) + " missing"))
            continue
        ir = nib.load(roi)
        m = np.asarray(ir.dataobj) > 0
        n = int(ndimage.label(m, structure=np.ones((3, 3, 3)))[1])
        row = dict(id=sid, image=image, label=roi, brain_mask=brain, tissue_map="", inpaint_mask="", group="",
                   n_lesions=n, lesion_class=lesion_class(n), brain_mask_from=brain_from)
        if tissue_pattern:
            t = tissue_pattern.format(id=sid)
            row["tissue_map"] = t if os.path.exists(t) else ""
        if quality_checks and os.path.exists(raw):
            ib = nib.load(raw)
            q = dict(id=sid, header_offset_mm=round(corner_offset_mm(ib, ir), 3))
            q["header_offset_flag"] = bool(q["header_offset_mm"] > 0.25 * min(ib.header.get_zooms()[:3]))
            ratio = mask_fit_ratio(ib, ir, nib.load(brain))
            q["mask_fit_ratio"] = None if ratio is None else round(ratio, 3)
            q["mask_fit_flag"] = bool(ratio is not None and ratio > 0.9)
            quality.append(q)
        rows.append(row)
    assign_folds(rows)
    report_dir = report_dir or "."
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "missing.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["participant_id", "reason"])
        w.writerows(missing)
    if quality:
        with open(os.path.join(report_dir, "quality.tsv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(quality[0]), delimiter="\t")
            w.writeheader()
            w.writerows(quality)
    with open(os.path.join(report_dir, "cohort.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "n_lesions", "lesion_class", "fold", "brain_mask_from"],
                           delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    report = dict(source=src, modality=modality, participants=len(ids), excluded=len(excluded & set(ids)),
                  missing=len(missing), included=len(rows), lesions=sum(r["n_lesions"] for r in rows),
                  brain_mask_from={k: sum(1 for r in rows if r["brain_mask_from"] == k) for k in ("source", "brainmask_dir")},
                  classes={k: sum(1 for r in rows if r["lesion_class"] == k) for _, _, k in CLASSES},
                  cases_per_fold=[sum(1 for r in rows if r["fold"] == k) for k in range(5)],
                  quality_flags=dict(mask_fit=sum(1 for q in quality if q["mask_fit_flag"]),
                                     header_offset=sum(1 for q in quality if q["header_offset_flag"])))
    json.dump(report, open(os.path.join(report_dir, "cohort_report.json"), "w"), indent=1)
    log(json.dumps({k: v for k, v in report.items() if k != "source"}))
    for sid, why in missing:
        log(f"  NOT included (missing input): {sid}: {why}")
    for q in quality:
        if q["mask_fit_flag"] or q["header_offset_flag"]:
            log(f"  quality flag (report only): {q}")
    return rows
