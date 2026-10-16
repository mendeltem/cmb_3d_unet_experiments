"""In-house dataset layout -> case list (cmbx.cohort): inclusion, exclusion files, reported gaps, quality flags,
folds, read-only source."""
import csv
import os

import nibabel as nib
import numpy as np

from cmbx.cohort import assign_folds, build_caselist, lesion_class
from cmbx.synthetic import phantom


def _layout(tmp_path, n=8):
    src = tmp_path / "CHARITE" / "BIDS_T2_STAR"
    bc = tmp_path / "CHARITE" / "derivatives" / "biascorr" / "t2star"
    rows = []
    for i in range(n):
        sid = f"sub-p{i:03d}"
        ph = phantom(seed=i, n_spots=i % 4, csf_mimic=False)
        d = src / sid / "t2"
        d.mkdir(parents=True)
        nib.save(ph["image"], str(d / f"{sid}_T2.nii.gz"))
        roi = ph["label"]
        if i == 1:                                   # header offset by 5 mm -> flagged
            aff = roi.affine.copy(); aff[0, 3] += 5.0
            roi = nib.Nifti1Image(np.asarray(roi.dataobj), aff)
        if i == 2:                                   # reference on plain tissue -> mask-fit flag
            lab = np.zeros(roi.shape, np.uint8); lab[30:33, 20:23, 10:12] = 1
            roi = nib.Nifti1Image(lab, roi.affine)
        nib.save(roi, str(d / f"{sid}_T2_roi.nii.gz"))
        if i != 3:                                   # case 3: no brain mask anywhere -> reported, not included
            nib.save(ph["brain"], str(d / f"{sid}_T2_brainmask.nii.gz"))
        (bc / sid / "anat").mkdir(parents=True)
        nib.save(ph["image"], str(bc / sid / "anat" / f"{sid}_T2starw_desc-biascorr_T2starw.nii.gz"))
        rows.append(dict(participant_id=sid, source="old", cmb="yes"))
    with open(src / "participants.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    with open(src / "excluded.tsv", "w") as f:
        f.write("participant_id\treason\nsub-p007\tsome reason\n")
    return src


def test_build_caselist(tmp_path):
    src = _layout(tmp_path)
    before = sorted(str(p) for p in src.rglob("*"))
    rep = tmp_path / "out"
    rows = build_caselist(str(src), "t2star", report_dir=str(rep))
    ids = [r["id"] for r in rows]
    assert ids == [f"sub-p{i:03d}" for i in (0, 1, 2, 4, 5, 6)]           # 3 missing mask, 7 excluded
    miss = {r["participant_id"]: r["reason"] for r in csv.DictReader(open(rep / "missing.tsv"), delimiter="\t")}
    assert "brain mask" in miss["sub-p003"]
    q = {r["id"]: r for r in csv.DictReader(open(rep / "quality.tsv"), delimiter="\t")}
    assert q["sub-p001"]["header_offset_flag"] == "True" and q["sub-p002"]["mask_fit_flag"] == "True"
    assert q["sub-p000"]["mask_fit_flag"] == "False"
    assert all("desc-biascorr" in r["image"] for r in rows)
    assert all(r["n_lesions"] == int(r["id"][-1]) % 4 for r in rows if r["id"] != "sub-p002")
    assert sorted(str(p) for p in src.rglob("*")) == before                # source untouched
    # brain mask from a fallback folder
    alt = tmp_path / "masks" / "sub-p003" / "anat"
    alt.mkdir(parents=True)
    nib.save(phantom(seed=3)["brain"], str(alt / "sub-p003_T2starw_desc-brain_mask.nii.gz"))
    rows2 = build_caselist(str(src), "t2star", brainmask_dir=str(tmp_path / "masks"), report_dir=str(rep))
    assert "sub-p003" in [r["id"] for r in rows2]


def test_assign_folds_matches_kohorte_bauen():
    rows = [dict(id=str(i), lesion_class=lesion_class(n)) for i, n in enumerate([0, 0, 0, 1, 2, 4, 9, 9, 0, 3])]
    assign_folds(rows)
    folds = [r["fold"] for r in rows]
    assert sorted(np.bincount(folds, minlength=5).tolist()) == [2, 2, 2, 2, 2]
    rng = np.random.default_rng(0); start = 0; want = [None] * len(rows)
    for k in ("0", "1-2", "3-5", ">5"):                                  # kohorte_bauen.py, inline
        idx = np.array([i for i, r in enumerate(rows) if r["lesion_class"] == k]); rng.shuffle(idx)
        for i, j in enumerate(idx):
            want[j] = (start + i) % 5
        start += len(idx)
    assert folds == want


def test_dataset_level_exclusion_with_modality_column(tmp_path):
    src = _layout(tmp_path)
    (src / "excluded.tsv").unlink()                               # e.g. a copy without the per-modality file
    with open(src.parent / "excluded.tsv", "w") as f:             # CHARITE/excluded.tsv
        f.write("modality\tparticipant_id\treason\nSWI\tsub-p000\tother modality\nT2*\tsub-p005\tbad mask\n")
    ids = [r["id"] for r in build_caselist(str(src), "t2star", report_dir=str(tmp_path / "o"))]
    assert "sub-p005" not in ids and "sub-p000" in ids and "sub-p007" in ids
