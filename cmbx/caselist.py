"""``cmbx make-caselist``: a case list from ANY folder layout, described by path patterns (no dataset is built in).

Each pattern is a path with ``{id}`` for the case id and optional glob wildcards (``*``); the first match is used:

    cmbx make-caselist --image "/data/derivatives/biascorr/{id}/anat/{id}*_desc-biascorr*.nii.gz" \\
        --label "/data/{id}/anat/{id}_cmb_mask.nii.gz" --brain-mask "/data/{id}/anat/{id}_brainmask.nii.gz" \\
        [--brain-mask "<fallback pattern>"] [--tissue-map "<SynthSeg pattern>"] [--raw "<raw image pattern>"] \\
        [--ids ids.tsv] [--exclude leave_out.tsv ...] --out ../my_runs/cases.csv

Case ids: from ``--ids`` (a text file with one id per line, or a CSV/TSV with an ``id`` or ``participant_id`` column;
its order is kept) or, without it, every id the ``--image`` pattern matches. ``--exclude`` files (same formats) leave
cases out. Only the files given are read: nothing is looked up by a fixed name.
A case without image, reference mask or brain mask is not silently dropped: it is listed with the reason.

Quality checks (report only, they never exclude) on the ``--raw`` image if given, else on ``--image``:
  * mask fit: mean intensity in the reference mask / mean in a ring around it (dilation 7x7x3 minus 3x3x1, inside the
    brain mask) > 0.9 -> "reference mask not darker than its surroundings";
  * header offset: reference mask header offset from the image by > 0.25 x the smallest voxel size.
Folds: lesion classes 0 / 1-2 / 3-5 / >5 (26-connected components of the reference); per class a
``numpy.random.default_rng(seed)`` shuffle, then round-robin over the folds with the counter running on across
classes (deterministic for the same ids in the same order). ``--no-folds`` leaves them to ``cmbx split``.
Reports next to the case list: ``missing.tsv``, ``quality.tsv``, ``caselist.tsv``, ``caselist_report.json``.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from scipy import ndimage

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
    """Mean intensity in the reference mask / mean in the ring around it. None if the mask is empty; inf if there is
    no ring inside the brain mask."""
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
    """Stratified by lesion class (fixed class order): shuffle per class with default_rng(seed), round-robin with a
    counter that runs on across classes."""
    rng = np.random.default_rng(seed)
    start = 0
    for _, _, k in CLASSES:
        idx = np.array([i for i, r in enumerate(rows) if r["lesion_class"] == k], dtype=int)
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            rows[j]["fold"] = (start + i) % n_folds
        start += len(idx)


def read_ids(path: str) -> List[str]:
    """One id per line, or a CSV/TSV with an ``id`` / ``participant_id`` column (else its first column)."""
    with open(path, newline="") as f:
        text = f.read()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    delim = "\t" if "\t" in lines[0] else ("," if "," in lines[0] else None)
    if delim is None:
        return [ln.strip() for ln in lines]
    rows = list(csv.reader(lines, delimiter=delim))
    head = [h.strip() for h in rows[0]]
    for name in ("id", "participant_id"):
        if name in head:
            col = head.index(name)
            return [r[col].strip() for r in rows[1:] if len(r) > col and r[col].strip()]
    return [r[0].strip() for r in rows if r and r[0].strip()]


def resolve(pattern: str, sid: str) -> Optional[str]:
    """Pattern with {id} (and optional * wildcards) -> the first existing match, else None."""
    hits = sorted(glob.glob(pattern.replace("{id}", glob.escape(sid))))
    return hits[0] if hits else None


def ids_from_pattern(pattern: str) -> List[str]:
    """Every id the pattern matches (sorted)."""
    if "{id}" not in pattern:
        raise ValueError(f"the --image pattern needs {{id}}: {pattern}")
    rx = re.escape(pattern).replace(re.escape("{id}"), "(?P<id>[^/]+?)", 1).replace(re.escape("{id}"), "(?P=id)")
    rx = re.compile("^" + rx.replace(re.escape("*"), "[^/]*") + "$")
    out = set()
    for p in glob.glob(pattern.replace("{id}", "*")):
        m = rx.match(p)
        if m:
            out.add(m.group("id"))
    return sorted(out)


def build_caselist(image: str, label: Optional[str] = None, brain_mask: Optional[List[str]] = None,
                   tissue_map: Optional[str] = None, raw: Optional[str] = None, ids_file: Optional[str] = None,
                   exclude: Optional[List[str]] = None, n_folds: int = 5, fold_seed: int = 0, folds: bool = True,
                   quality_checks: bool = True, report_dir: Optional[str] = None, log=print) -> List[Dict]:
    ids = read_ids(ids_file) if ids_file else ids_from_pattern(image)
    excluded = set().union(*[set(read_ids(p)) for p in exclude]) if exclude else set()
    log(f"{len(ids)} case ids from {ids_file or 'the --image pattern'}; {len(excluded & set(ids))} left out by "
        f"{[os.path.basename(p) for p in exclude or []] or 'no --exclude file'}")
    rows, missing, quality = [], [], []
    for sid in ids:
        if sid in excluded:
            continue
        img = resolve(image, sid)
        roi = resolve(label, sid) if label else None
        brain, brain_from = None, ""
        for k, pat in enumerate(brain_mask or []):
            brain = resolve(pat, sid)
            if brain:
                brain_from = str(k)
                break
        lack = [n for n, p, need in (("image", img, True), ("reference mask", roi, bool(label)),
                                     ("brain mask", brain, bool(brain_mask))) if need and not p]
        if lack:
            missing.append((sid, ", ".join(lack) + " missing"))
            continue
        n = int(ndimage.label(np.asarray(nib.load(roi).dataobj) > 0, structure=np.ones((3, 3, 3)))[1]) if roi else 0
        row = dict(id=sid, image=img, label=roi or "", brain_mask=brain or "", tissue_map="", inpaint_mask="",
                   group="", n_lesions=n, lesion_class=lesion_class(n), brain_mask_from=brain_from)
        if tissue_map:
            row["tissue_map"] = resolve(tissue_map, sid) or ""
        if quality_checks and roi and brain:
            src = resolve(raw, sid) if raw else img
            if src:
                ib, ir = nib.load(src), nib.load(roi)
                q = dict(id=sid, header_offset_mm=round(corner_offset_mm(ib, ir), 3))
                q["header_offset_flag"] = bool(q["header_offset_mm"] > 0.25 * min(ib.header.get_zooms()[:3]))
                ratio = mask_fit_ratio(ib, ir, nib.load(brain))
                q["mask_fit_ratio"] = None if ratio is None else round(ratio, 3)
                q["mask_fit_flag"] = bool(ratio is not None and ratio > 0.9)
                quality.append(q)
        rows.append(row)
    if folds:
        assign_folds(rows, n_folds, fold_seed)
    report_dir = report_dir or "."
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "missing.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "reason"])
        w.writerows(missing)
    if quality:
        with open(os.path.join(report_dir, "quality.tsv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(quality[0]), delimiter="\t")
            w.writeheader()
            w.writerows(quality)
    with open(os.path.join(report_dir, "caselist.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "n_lesions", "lesion_class", "fold", "brain_mask_from"],
                           delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    report = dict(ids=len(ids), excluded=len(excluded & set(ids)), missing=len(missing), included=len(rows),
                  lesions=sum(r["n_lesions"] for r in rows),
                  brain_mask_from={k: sum(1 for r in rows if r["brain_mask_from"] == str(k)) for k in range(len(brain_mask or []))},
                  classes={k: sum(1 for r in rows if r["lesion_class"] == k) for _, _, k in CLASSES},
                  cases_per_fold=[sum(1 for r in rows if r.get("fold") == k) for k in range(n_folds)] if folds else None,
                  quality_flags=dict(mask_fit=sum(1 for q in quality if q["mask_fit_flag"]),
                                     header_offset=sum(1 for q in quality if q["header_offset_flag"])))
    json.dump(report, open(os.path.join(report_dir, "caselist_report.json"), "w"), indent=1)
    log(json.dumps(report))
    for sid, why in missing:
        log(f"  NOT included (missing input): {sid}: {why}")
    flagged = [q for q in quality if q["mask_fit_flag"] or q["header_offset_flag"]]
    for q in flagged:
        log(f"  quality flag (report only): {q}")
    if flagged:
        log(f"WARNING: {len(flagged)} flagged case(s) are INCLUDED: {[q['id'] for q in flagged]}. Check them; to leave "
            "one out, put its id in a file passed with --exclude")
    return rows
