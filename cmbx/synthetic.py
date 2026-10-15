"""Synthetic phantom cases for tests and for trying the pipeline without data (adapted from
``mb_segment/synthetic.py`` of microbleed_detection_3D_UNET): an ellipsoid 'brain' with a slow bias field, a few
small dark spheres (microbleed-like, in the label), a bright 'ventricle' with SynthSeg CSF label 4 that holds one
unlabelled dark sphere (a CSF mimic the CSF rule should remove), and a SynthSeg-like tissue map."""
from __future__ import annotations

import csv
import os

import nibabel as nib
import numpy as np


def phantom(shape=(64, 64, 24), spacing=(1.0, 1.0, 2.0), n_spots: int = 3, seed: int = 0, bias: bool = True,
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
