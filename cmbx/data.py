"""Case lists, the stratified fold split and loading of prepared cases.

Case list: CSV (header row) or JSON (list of objects). Columns:
    id           unique case id (used in file names)                                   required
    image        T2*-weighted GRE or SWI magnitude, NIfTI                               required
    label        reference microbleed mask, same grid as the image (non-zero = CMB)     required for CV
    brain_mask   brain mask, same grid (recommended; else image > 0 with holes filled)  optional
    tissue_map   SynthSeg label map (any grid, resampled by world coordinates)          optional (CSF rule)
    inpaint_mask mask for the inpainting experiment, same grid as the image             optional
    group        stratification group, e.g. scanner or cohort                           optional
    fold         fixed fold 0..n-1; if present for every case it is used as the split   optional
Relative paths are resolved against the folder of the case list.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from scipy import ndimage

N26 = np.ones((3, 3, 3), bool)
PATH_COLUMNS = ("image", "label", "brain_mask", "tissue_map", "inpaint_mask")


def read_cases(path: str) -> List[Dict]:
    """Read a case list; empty strings become None; paths become absolute."""
    if path.endswith(".json"):
        rows = json.load(open(path))
    else:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    base = os.path.dirname(os.path.abspath(path))
    out, seen = [], set()
    for r in rows:
        r = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items() if k}
        r = {k: (None if v == "" else v) for k, v in r.items()}
        if not r.get("id") or not r.get("image"):
            raise ValueError(f"{path}: every row needs 'id' and 'image': {r}")
        if r["id"] in seen:
            raise ValueError(f"{path}: duplicate id {r['id']}")
        seen.add(r["id"])
        for k in PATH_COLUMNS:
            if r.get(k) and not os.path.isabs(r[k]):
                r[k] = os.path.normpath(os.path.join(base, r[k]))
        if r.get("fold") is not None:
            r["fold"] = int(r["fold"])
        out.append(r)
    return out


def load_class(n: int) -> int:
    """Lesion load class used for stratification: 0 / 1-2 / >= 3 microbleeds (as in cmb-valdo-chain split.py)."""
    return 0 if n == 0 else (1 if n <= 2 else 2)


def make_split(cases: List[Dict], lesion_counts: Dict[str, int], n_folds: int = 5, seed: int = 0) -> Dict:
    """Deterministic stratified split.

    If every case carries a ``fold`` column, that assignment is used unchanged. Otherwise the cases are shuffled with
    ``random.Random(seed)`` per group, cut into cells (group x load class 0 / 1-2 / >= 3) and, cell by cell, each case
    goes to the fold with (fewest cases of this cell, fewest cases, smallest lesion sum, lowest index) -- the greedy
    rule of cmb-valdo-chain ``code/split.py`` plus the fold size as second key (the original had no such key and
    produced folds of 11-13 cases; it also held out 15 validation cases first).
    """
    ids = [c["id"] for c in cases]
    if all(c.get("fold") is not None for c in cases):
        folds = [[c["id"] for c in cases if c["fold"] == k] for k in range(n_folds)]
        if sum(len(f) for f in folds) != len(cases):
            raise ValueError(f"fold column outside 0..{n_folds - 1}")
        rule = "fixed: fold column of the case list"
    else:
        rng = random.Random(seed)
        groups: Dict[str, List[Dict]] = {}
        for c in cases:
            groups.setdefault(str(c.get("group") or "all"), []).append(c)
        for g in sorted(groups):
            rng.shuffle(groups[g])
        cells: Dict = {}
        for g in sorted(groups):
            for c in groups[g]:
                cells.setdefault((g, load_class(lesion_counts[c["id"]])), []).append(c["id"])
        folds = [[] for _ in range(n_folds)]
        cell_of = {sid: cell for cell, members in cells.items() for sid in members}
        for cell in sorted(cells):
            for sid in cells[cell]:
                def key(i, cell=cell):
                    return (sum(1 for x in folds[i] if cell_of[x] == cell), len(folds[i]),
                            sum(lesion_counts[x] for x in folds[i]), i)
                folds[min(range(n_folds), key=key)].append(sid)
        rule = (f"stratified by group x load class (0 / 1-2 / >=3 CMB), greedy fold-size and lesion-sum balance, "
                f"random.Random({seed}) shuffle per group")
    if sorted(x for f in folds for x in f) != sorted(ids):
        raise AssertionError("split does not cover every case exactly once")
    return {"n_folds": n_folds, "seed": seed, "rule": rule, "folds": folds,
            "lesions_per_fold": [int(sum(lesion_counts.get(x, 0) for x in f)) for f in folds],
            "cases_per_fold": [len(f) for f in folds]}


def fold_roles(split: Dict, k: int):
    """(test ids, inner ids, train ids) for test fold k; inner fold = (k + 1) % n (cv5.py)."""
    n = split["n_folds"]
    if n < 3:
        raise ValueError("at least 3 folds are needed (test fold, inner fold, >= 1 training fold)")
    i = (k + 1) % n
    test = list(split["folds"][k])
    inner = list(split["folds"][i])
    train = [s for j in range(n) if j not in (k, i) for s in split["folds"][j]]
    return test, inner, train


# ---------------------------------------------------------------- prepared cases
@dataclass
class PreparedCase:
    """One prepared case in TRAINING order (z, y, x) -- the NIfTI arrays (x, y, z) transposed."""
    id: str
    image: np.ndarray
    frst: Optional[np.ndarray]
    label: np.ndarray
    affine: np.ndarray
    tissue: Optional[np.ndarray] = None
    components: List[np.ndarray] = field(default_factory=list)

    def channels(self, names) -> List[np.ndarray]:
        out = []
        for n in names:
            a = getattr(self, n, None)
            if a is None:
                raise ValueError(f"{self.id}: channel {n!r} not available (prepared without it?)")
            out.append(a)
        return out


def to_zyx(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a).transpose(2, 1, 0))


def case_files(prepared_dir: str, sid: str) -> Dict[str, str]:
    d = os.path.join(prepared_dir, sid)
    return {k: os.path.join(d, f"{sid}_{k}.nii.gz") for k in ("image", "frst", "label", "tissue")}


def load_prepared(prepared_dir: str, sid: str, channels=("image", "frst"), tissue: bool = False) -> PreparedCase:
    f = case_files(prepared_dir, sid)
    ni = nib.load(f["image"])
    img = to_zyx(np.asarray(ni.dataobj, dtype=np.float32))
    frs = to_zyx(np.asarray(nib.load(f["frst"]).dataobj, dtype=np.float32)) if "frst" in channels else None
    lab = to_zyx((np.asarray(nib.load(f["label"]).dataobj) > 0).astype(np.uint8)) if os.path.exists(f["label"]) \
        else np.zeros(img.shape, np.uint8)
    tis = None
    if tissue:
        if not os.path.exists(f["tissue"]):
            raise FileNotFoundError(f"{sid}: no tissue map in {prepared_dir} (case list column tissue_map)")
        tis = to_zyx(np.asarray(nib.load(f["tissue"]).dataobj).astype(np.int32))
    assert img.shape == lab.shape and (frs is None or frs.shape == img.shape), sid
    ll, n = ndimage.label(lab, structure=N26)
    comps = [np.argwhere(ll == j) for j in range(1, n + 1)]
    return PreparedCase(sid, img, frs, lab, ni.affine, tis, comps)


def prepared_lesion_counts(prepared_dir: str, ids) -> Dict[str, int]:
    out = {}
    for sid in ids:
        info = json.load(open(os.path.join(prepared_dir, sid, "info.json")))
        out[sid] = int(info["n_lesions"])
    return out


# ---------------------------------------------------------------- BIDS case lists (in-house cohorts)
BIDS_STEMS = {"t2star": "_T2starw", "swi": "_acq-swi_T2starw"}


def _read_tsv(path: str) -> List[Dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def caselist_from_bids(root: str, modality: str, tissue_pattern: Optional[str] = None, use_folds: bool = True,
                       log=print) -> List[Dict]:
    """Case list for a BIDS dataset in the layout of the in-house cohorts:

        sub-XXX/anat/sub-XXX<stem>.nii.gz                                         image (stem _T2starw / _acq-swi_T2starw)
        derivatives/biascorr/sub-XXX/anat/sub-XXX<stem>_desc-biascorr_T2starw.nii.gz   FSL FAST -B, computed once
        derivatives/brainmask/sub-XXX/anat/sub-XXX<stem>_desc-brain_mask.nii.gz       HD-BET / scanner mask
        derivatives/manual-cmb/sub-XXX/anat/*cmb*.nii.gz                             reference
        participants.tsv (optional column fold), ausgeschlossen.tsv (excluded cases, honoured)

    The bias-corrected derivative is used as the image whenever it exists (so preprocess.bias must be ``none``;
    it is never recomputed); without it the raw image is used, as ``build_private_grids.py`` did, with a warning.
    ``tissue_pattern``: optional SynthSeg path with ``{id}``. ``use_folds``: take the fold column of
    participants.tsv (the folds of all earlier in-house results)."""
    stem = BIDS_STEMS[modality]
    excluded = set()
    ex_path = os.path.join(root, "ausgeschlossen.tsv")
    if os.path.exists(ex_path) and os.path.getsize(ex_path) > 0:
        excluded = {r["participant_id"] for r in _read_tsv(ex_path) if r.get("participant_id")}
    part = {}
    p_path = os.path.join(root, "participants.tsv")
    if os.path.exists(p_path):
        part = {r["participant_id"]: r for r in _read_tsv(p_path)}
    subjects = sorted(os.path.basename(d) for d in glob.glob(os.path.join(root, "sub-*"))
                      if os.path.isdir(d))
    rows, raw_used, missing = [], 0, []
    for sid in subjects:
        if sid in excluded:
            continue
        raw = os.path.join(root, sid, "anat", f"{sid}{stem}.nii.gz")
        if not os.path.exists(raw):
            missing.append((sid, "image"))
            continue
        bc = os.path.join(root, "derivatives", "biascorr", sid, "anat", f"{sid}{stem}_desc-biascorr_T2starw.nii.gz")
        bm = os.path.join(root, "derivatives", "brainmask", sid, "anat", f"{sid}{stem}_desc-brain_mask.nii.gz")
        labs = sorted(glob.glob(os.path.join(root, "derivatives", "manual-cmb", sid, "anat", "*.nii*")))
        labs = [x for x in labs if "cmb" in os.path.basename(x).lower()]
        if not os.path.exists(bm):
            missing.append((sid, "brain mask"))
            continue
        if len(labs) != 1:
            missing.append((sid, f"{len(labs)} reference masks"))
            continue
        if not os.path.exists(bc):
            raw_used += 1
        row = dict(id=sid, image=bc if os.path.exists(bc) else raw, label=labs[0], brain_mask=bm,
                   tissue_map="", inpaint_mask="", group="", fold="")
        if tissue_pattern:
            t = tissue_pattern.format(id=sid)
            row["tissue_map"] = t if os.path.exists(t) else ""
        if use_folds and sid in part and part[sid].get("fold") not in (None, "", "-1"):
            row["fold"] = int(part[sid]["fold"])
        rows.append(row)
    log(f"{root}: {len(rows)} cases, {len(excluded)} excluded by ausgeschlossen.tsv, "
        f"{len(missing)} skipped (missing files), {raw_used} without bias-corrected derivative (raw image used)")
    for sid, what in missing:
        log(f"  skipped: {what} missing")
    if tissue_pattern:
        log(f"  tissue maps found for {sum(1 for r in rows if r['tissue_map'])} of {len(rows)} cases")
    if use_folds and rows and any(r["fold"] == "" for r in rows):
        log("  fold column incomplete -> folds left empty, `cmbx split` will compute them")
        for r in rows:
            r["fold"] = ""
    return rows


def write_caselist(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = ["id", "image", "label", "brain_mask", "tissue_map", "inpaint_mask", "group", "fold"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
