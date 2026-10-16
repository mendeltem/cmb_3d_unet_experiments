"""Case list from path patterns (cmbx.caselist): any layout, ids from a file or the pattern, --exclude files,
fallback brain masks, reported gaps, quality flags, deterministic stratified folds, read-only source."""
import csv
import os

import nibabel as nib
import numpy as np
import pytest

from cmbx.caselist import assign_folds, build_caselist, ids_from_pattern, lesion_class, read_ids
from cmbx.synthetic import phantom


def _layout(tmp_path, n=8):
    """An arbitrary layout: images under derived/, masks under raw/<id>/."""
    root = tmp_path / "anydata"
    for i in range(n):
        sid = f"case{i:02d}"
        ph = phantom(seed=i, n_spots=i % 4, csf_mimic=False)
        d = root / "raw" / sid
        d.mkdir(parents=True)
        nib.save(ph["image"], str(d / f"{sid}_scan.nii.gz"))
        roi = ph["label"]
        if i == 1:                                   # header offset by 5 mm -> flagged
            aff = roi.affine.copy(); aff[0, 3] += 5.0
            roi = nib.Nifti1Image(np.asarray(roi.dataobj), aff)
        if i == 2:                                   # reference on plain tissue -> mask-fit flag
            lab = np.zeros(roi.shape, np.uint8); lab[30:33, 20:23, 10:12] = 1
            roi = nib.Nifti1Image(lab, roi.affine)
        nib.save(roi, str(d / f"{sid}_cmb.nii.gz"))
        if i != 3:                                   # case 3: no brain mask -> reported, not included
            nib.save(ph["brain"], str(d / f"{sid}_brain.nii.gz"))
        (root / "derived" / sid).mkdir(parents=True)
        nib.save(ph["image"], str(root / "derived" / sid / f"{sid}_v2_corrected.nii.gz"))
    return root


def _pat(root):
    return dict(image=str(root / "derived" / "{id}" / "{id}_*corrected.nii.gz"),
                label=str(root / "raw" / "{id}" / "{id}_cmb.nii.gz"),
                brain_mask=[str(root / "raw" / "{id}" / "{id}_brain.nii.gz")],
                raw=str(root / "raw" / "{id}" / "{id}_scan.nii.gz"))


def test_build_from_patterns(tmp_path):
    root = _layout(tmp_path)
    before = sorted(str(p) for p in root.rglob("*"))
    (tmp_path / "leave_out.txt").write_text("case07\n")
    rep = tmp_path / "out"
    rows = build_caselist(**_pat(root), exclude=[str(tmp_path / "leave_out.txt")], report_dir=str(rep), log=print)
    assert [r["id"] for r in rows] == [f"case{i:02d}" for i in (0, 1, 2, 4, 5, 6)]      # 3 no brain mask, 7 excluded
    miss = {r["id"]: r["reason"] for r in csv.DictReader(open(rep / "missing.tsv"), delimiter="\t")}
    assert "brain mask" in miss["case03"]
    q = {r["id"]: r for r in csv.DictReader(open(rep / "quality.tsv"), delimiter="\t")}
    assert q["case01"]["header_offset_flag"] == "True" and q["case02"]["mask_fit_flag"] == "True"
    assert q["case00"]["mask_fit_flag"] == "False"
    assert all("corrected" in r["image"] for r in rows)
    assert all(r["n_lesions"] == int(r["id"][-1]) % 4 for r in rows if r["id"] != "case02")
    assert sorted(str(p) for p in root.rglob("*")) == before                # source untouched
    # fallback brain mask pattern, ids file with a column, order kept
    alt = tmp_path / "masks"
    alt.mkdir()
    nib.save(phantom(seed=3)["brain"], str(alt / "case03-brainmask.nii.gz"))
    (tmp_path / "ids.tsv").write_text("participant_id\tsite\ncase05\ta\ncase03\tb\ncase00\ta\n")
    p = _pat(root)
    p["brain_mask"] = p["brain_mask"] + [str(alt / "{id}-brainmask.nii.gz")]
    rows2 = build_caselist(**p, ids_file=str(tmp_path / "ids.tsv"), report_dir=str(rep), log=print)
    assert [r["id"] for r in rows2] == ["case05", "case03", "case00"]
    assert {r["id"]: r["brain_mask_from"] for r in rows2}["case03"] == "1"


def test_ids_from_pattern_and_read_ids(tmp_path):
    root = _layout(tmp_path, 3)
    assert ids_from_pattern(_pat(root)["image"]) == ["case00", "case01", "case02"]
    with pytest.raises(ValueError):
        ids_from_pattern(str(root / "derived" / "x.nii.gz"))
    (tmp_path / "a.csv").write_text("id,other\nx1,1\nx2,2\n")
    (tmp_path / "b.txt").write_text("y1\n\ny2\n")
    assert read_ids(str(tmp_path / "a.csv")) == ["x1", "x2"] and read_ids(str(tmp_path / "b.txt")) == ["y1", "y2"]


def test_prediction_case_list_and_required_brain_mask(tmp_path):
    """For prediction a case list needs images and brain masks, no labels; a brain mask is required."""
    root = _layout(tmp_path, 3)
    p = _pat(root)
    rows = build_caselist(p["image"], brain_mask=[p["brain_mask"][0]], report_dir=str(tmp_path / "o"), log=print)
    assert [r["id"] for r in rows] == ["case00", "case01", "case02"] and all(r["label"] == "" for r in rows)
    with pytest.raises(SystemExit, match="brain mask is required"):
        build_caselist(p["image"], report_dir=str(tmp_path / "o2"), log=print)


def test_assign_folds_deterministic_stratified():
    rows = [dict(id=str(i), lesion_class=lesion_class(n)) for i, n in enumerate([0, 0, 0, 1, 2, 4, 9, 9, 0, 3])]
    assign_folds(rows)
    folds = [r["fold"] for r in rows]
    assert sorted(np.bincount(folds, minlength=5).tolist()) == [2, 2, 2, 2, 2]
    rng = np.random.default_rng(0); start = 0; want = [None] * len(rows)
    for k in ("0", "1-2", "3-5", ">5"):                                  # the documented algorithm, inline
        idx = np.array([i for i, r in enumerate(rows) if r["lesion_class"] == k]); rng.shuffle(idx)
        for i, j in enumerate(idx):
            want[j] = (start + i) % 5
        start += len(idx)
    assert folds == want
