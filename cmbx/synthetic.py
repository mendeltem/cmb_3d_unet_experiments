"""Synthetic phantom cases for tests and for trying the pipeline without data (adapted from
``mb_segment/synthetic.py`` of microbleed_detection_3D_UNET): an ellipsoid 'brain' (optional slow bias field, off by
default: real inputs are bias-corrected beforehand), a few
small dark spheres (microbleed-like, in the label), a bright 'ventricle' with SynthSeg CSF label 4 that holds one
unlabelled dark sphere (a CSF mimic the CSF rule should remove), and a SynthSeg-like tissue map."""
from __future__ import annotations

import csv
import os

import nibabel as nib
import numpy as np


def phantom(shape=(64, 64, 24), spacing=(1.0, 1.0, 2.0), n_spots: int = 3, seed: int = 0, bias: bool = False,
            csf_mimic: bool = True):
    """Returns ``dict(image, label, brain, tissue)`` of NIfTI images."""
    rng = np.random.RandomState(seed)
    x, y, z = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    centre = np.array(shape) / 2.0
    radii = np.array(shape) / 2.0 - np.array([4, 4, 2])
    brain = (((x - centre[0]) / radii[0]) ** 2 + ((y - centre[1]) / radii[1]) ** 2
             + ((z - centre[2]) / radii[2]) ** 2) <= 1.0
    vent = (((x - centre[0]) / 6.0) ** 2 + ((y - centre[1]) / 4.0) ** 2 + ((z - centre[2]) / 3.0) ** 2) <= 1.0
    image = np.where(brain, 100.0 + rng.normal(0, 4, shape), 0.0)
    image[vent] = 160.0 + rng.normal(0, 4, int(vent.sum()))
    if bias:
        image *= np.where(brain, 1.0 + 0.25 * (x - centre[0]) / shape[0], 1.0)
    tissue = np.where(brain, 2, 0).astype(np.int16)                  # 2 = cerebral white matter
    tissue[vent] = 4                                                  # 4 = lateral ventricle (CSF)

    def sphere(c, r_mm):
        return (((x - c[0]) * spacing[0]) ** 2 + ((y - c[1]) * spacing[1]) ** 2
                + ((z - c[2]) * spacing[2]) ** 2) <= r_mm ** 2

    label = np.zeros(shape, bool)
    centres, tries = [], 0
    while len(centres) < n_spots and tries < 2000:
        tries += 1
        c = np.array([rng.randint(8, shape[0] - 8), rng.randint(8, shape[1] - 8), rng.randint(4, shape[2] - 4)])
        if not brain[tuple(c)] or vent[tuple(c)] or any(np.abs(c - o).max() < 10 for o in centres):
            continue
        if np.linalg.norm((c - centre) * np.array(spacing)) < 12:  # keep away from the ventricle
            continue
        s = sphere(c, 2.0)
        image[s] = 15.0
        label |= s
        centres.append(c)
    if csf_mimic:
        image[sphere(centre.astype(int), 1.5)] = 15.0                # dark spot in CSF, not a microbleed
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    affine[:3, 3] = -np.array(shape) * np.array(spacing) / 2.0
    mk = lambda a, dt: nib.Nifti1Image(a.astype(dt), affine)
    return dict(image=mk(image, np.float32), label=mk(label, np.uint8), brain=mk(brain, np.uint8),
                tissue=mk(tissue, np.int16))


def write_phantom_dataset(out_dir: str, n_cases: int = 10, seed: int = 0, **kw) -> str:
    """Write ``n_cases`` phantoms and a case list ``<out_dir>/cases.csv`` (relative paths). Returns its path."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i in range(n_cases):
        sid = f"phantom-{i:02d}"
        ph = phantom(seed=seed + i, n_spots=(i % 4), **kw)
        d = os.path.join(out_dir, sid)
        os.makedirs(d, exist_ok=True)
        for k in ("image", "label", "brain", "tissue"):
            nib.save(ph[k], os.path.join(d, f"{sid}_{k}.nii.gz"))
        rows.append(dict(id=sid, image=f"{sid}/{sid}_image.nii.gz", label=f"{sid}/{sid}_label.nii.gz",
                         brain_mask=f"{sid}/{sid}_brain.nii.gz", tissue_map=f"{sid}/{sid}_tissue.nii.gz",
                         inpaint_mask="", group="phantom"))
    path = os.path.join(out_dir, "cases.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def write_phantom_preprocessed(root: str, n_cases: int = 10, seed: int = 0, labels: bool = True, flair: bool = True,
                               **kw) -> str:
    """Write ``n_cases`` phantoms as a cohort (README "Data"): ``<root>/derivatives/preprocessed/phantom-XX/`` with
    ``phantom-XX_t2star``, ``_brainmask``, ``_synthseg``, with labels=True ``_cmb``, with flair=True ``_flair`` (on
    the scan matrix, other contrast), plus ``<root>/participants.tsv``. Returns the root."""
    prep = os.path.join(root, "derivatives", "preprocessed")
    ids = []
    for i in range(n_cases):
        sid = f"phantom-{i:02d}"
        ph = phantom(seed=seed + i, n_spots=(i % 4), **kw)
        d = os.path.join(prep, sid)
        os.makedirs(d, exist_ok=True)
        for k, suffix in (("image", "t2star"), ("brain", "brainmask"), ("tissue", "synthseg")) + ((("label", "cmb"),) if labels else ()):
            nib.save(ph[k], os.path.join(d, f"{sid}_{suffix}.nii.gz"))
        if flair:
            im = ph["image"]
            b = np.asarray(ph["brain"].dataobj) > 0
            a = np.asarray(im.dataobj, np.float32)
            nib.save(nib.Nifti1Image(((a.max() - a) * b).astype(np.float32), im.affine), os.path.join(d, f"{sid}_flair.nii.gz"))
        ids.append(sid)
    with open(os.path.join(root, "participants.tsv"), "w") as f:
        f.write("participant_id\n" + "".join(f"{s}\n" for s in ids))
    return root


def write_phantom_bids(root: str, n_cases: int = 10, seed: int = 0, labels: bool = True, **kw) -> str:
    """Write ``n_cases`` phantoms in the recommended BIDS layout (README "Data") under ``root``:

        participants.tsv, dataset_description.json
        sub-XXX/anat/sub-XXX_T2starw.nii.gz                                        raw image
        derivatives/biascorr/sub-XXX/anat/sub-XXX_T2starw_desc-biascorr_T2starw.nii.gz   bias-corrected (= the input)
        derivatives/brainmask/sub-XXX/anat/sub-XXX_T2starw_desc-brain_mask.nii.gz
        derivatives/manual-cmb/sub-XXX/anat/sub-XXX_T2starw_desc-cmb_mask.nii.gz    reference (only with labels=True)
        derivatives/synthseg/sub-XXX/anat/sub-XXX_T2starw_dseg.nii.gz               optional tissue map
    ``labels=False`` gives a dataset as it arrives for inference (no reference masks). Returns ``root``."""
    import json
    os.makedirs(root, exist_ok=True)
    ids = []
    for i in range(n_cases):
        sid = f"sub-{i + 1:03d}"
        ph = phantom(seed=seed + i, n_spots=(i % 4), **kw)
        files = {f"{sid}/anat/{sid}_T2starw.nii.gz": ph["image"],
                 f"derivatives/biascorr/{sid}/anat/{sid}_T2starw_desc-biascorr_T2starw.nii.gz": ph["image"],
                 f"derivatives/brainmask/{sid}/anat/{sid}_T2starw_desc-brain_mask.nii.gz": ph["brain"],
                 f"derivatives/synthseg/{sid}/anat/{sid}_T2starw_dseg.nii.gz": ph["tissue"]}
        if labels:
            files[f"derivatives/manual-cmb/{sid}/anat/{sid}_T2starw_desc-cmb_mask.nii.gz"] = ph["label"]
        for rel, img in files.items():
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            nib.save(img, os.path.join(root, rel))
        ids.append(sid)
    with open(os.path.join(root, "participants.tsv"), "w", newline="") as f:
        f.write("participant_id\n" + "".join(f"{s}\n" for s in ids))
    json.dump({"Name": "cmbx synthetic phantoms (no patient data)", "BIDSVersion": "1.9.0", "DatasetType": "raw"},
              open(os.path.join(root, "dataset_description.json"), "w"), indent=1)
    return root

