import json
import os

import nibabel as nib
import numpy as np
import pytest

from cmbx.config import DEFAULTS
from cmbx.preprocess import (inpaint, invert, label_nearest, prepare_case, resample, target_grid)
from cmbx.synthetic import phantom


def _pp(**kw):
    pp = json.loads(json.dumps(DEFAULTS["preprocess"]))
    pp.update(bias="none", **kw)
    return pp


def _write_case(tmp_path, sid="p0", seed=0, affine=None):
    ph = phantom(seed=seed, n_spots=3)
    d = tmp_path / "raw"
    d.mkdir(exist_ok=True)
    paths = {}
    for k in ("image", "label", "brain", "tissue"):
        img = ph[k]
        if affine is not None:
            img = nib.Nifti1Image(np.asarray(img.dataobj), affine)
        paths[k] = str(d / f"{sid}_{k}.nii.gz")
        nib.save(img, paths[k])
    return dict(id=sid, image=paths["image"], label=paths["label"], brain_mask=paths["brain"],
                tissue_map=paths["tissue"]), ph


def test_target_grid_keeps_field_of_view_centre():
    aff = np.diag([0.449, 0.449, 4.0, 1.0]); aff[:3, 3] = [-100, -110, -60]
    shape = (512, 512, 35)
    s, M, r, off = target_grid(aff, shape, (0.5, 0.5, 1.0))
    assert M.tolist() == [460, 460, 140]
    centre_src = (aff @ np.r_[(np.array(shape) - 1) / 2.0, 1])[:3]
    T = np.eye(4); T[:3, :3] = np.diag(r); T[:3, 3] = off
    centre_tgt = (aff @ T @ np.r_[(M - 1) / 2.0, 1])[:3]
    assert np.allclose(centre_src, centre_tgt, atol=1e-9)
    # every target voxel centre lies within the source field of view (no zero slices at the edge)
    assert off[2] >= -0.5 and off[2] + r[2] * (M[2] - 1) <= shape[2] - 0.5


def test_invert():
    img = np.array([[[0.0, 50.0, 100.0, 25.0]]], np.float32)
    brain = img > 0
    out = invert(img, brain)
    assert out[0, 0, 0] == 0 and np.isclose(out[0, 0, 1], 0.5) and out[0, 0, 2] == 1e-6
    assert np.isclose(out[0, 0, 3], 0.75)


def test_label_nearest_rescues_thin_component():
    lab = np.zeros((40, 40, 10), np.uint8)
    lab[5:8, 9, 5] = 1                         # 1-voxel-thin line in a row nearest sampling skips
    r = np.array([1.12, 1.12, 1.0]); M = np.array([35, 35, 10]); off = ((np.array(lab.shape) - 1) - r * (M - 1)) / 2
    out, info = label_nearest(lab, r, off, M)
    assert out.sum() > 0
    assert len(info["rescued"]) in (0, 1)


def test_prepare_case_shapes_orientation_and_lesions(tmp_path):
    case, ph = _write_case(tmp_path)
    out = tmp_path / "prep"
    info = prepare_case(case, str(out), _pp())
    d = out / "p0"
    img = nib.load(str(d / "p0_image.nii.gz")); frs = nib.load(str(d / "p0_frst.nii.gz"))
    lab = nib.load(str(d / "p0_label.nii.gz")); tis = nib.load(str(d / "p0_tissue.nii.gz"))
    assert img.shape == frs.shape == lab.shape == tis.shape == tuple(info["shape"])
    assert np.allclose(img.affine, lab.affine) and np.allclose(img.affine, frs.affine)
    assert np.allclose(np.sqrt((img.affine[:3, :3] ** 2).sum(0)), [0.5, 0.5, 1.0])
    # 64 x 64 x 24 at 1 x 1 x 2 mm -> uncropped 128 x 128 x 48
    assert info["full_shape"] == [128, 128, 48]
    assert info["n_lesions_source"] == info["n_lesions"] == 3 and info["components_equal"]
    a = np.asarray(img.dataobj)
    assert a.min() >= 0 and a.max() <= 1 and (a > 0).any()
    f = np.asarray(frs.dataobj)
    assert f.min() >= 0 and f.max() <= 1
    # orientation / world coordinates: every lesion centre on the grid maps to a lesion voxel of the source
    src_lab = np.asarray(ph["label"].dataobj) > 0
    from scipy import ndimage
    gl, n = ndimage.label(np.asarray(lab.dataobj) > 0)
    inv_src = np.linalg.inv(ph["label"].affine)
    for c in ndimage.center_of_mass(gl > 0, gl, range(1, n + 1)):
        w = img.affine @ np.r_[c, 1]
        i = np.rint((inv_src @ w)[:3]).astype(int)
        assert src_lab[tuple(i)], (c, i)
    # lesions are bright on the inverted grid image
    assert a[np.asarray(lab.dataobj) > 0].mean() > a[(a > 0) & (np.asarray(lab.dataobj) == 0)].mean()
    # the tissue map lands on the same anatomy (ventricle = 4 in the centre)
    t = np.asarray(tis.dataobj)
    assert (t == 4).any() and (t == 2).any()
    # resumable: a second call returns the stored info
    assert prepare_case(case, str(out), _pp()) == json.load(open(d / "info.json"))


def test_prepare_case_ras_reorientation(tmp_path):
    flip = np.diag([-1.0, 1.0, 2.0, 1.0]); flip[:3, 3] = [32, -32, -24]
    case, _ = _write_case(tmp_path, affine=flip)
    info = prepare_case(case, str(tmp_path / "ras"), _pp(reorient="ras"))
    img = nib.load(str(tmp_path / "ras" / "p0" / "p0_image.nii.gz"))
    assert nib.aff2axcodes(img.affine) == ("R", "A", "S")
    assert info["n_lesions"] == 3


def test_label_must_be_on_image_grid(tmp_path):
    case, _ = _write_case(tmp_path)
    bad = nib.Nifti1Image(np.zeros((10, 10, 10), np.uint8), np.eye(4))
    nib.save(bad, str(tmp_path / "bad.nii.gz"))
    case["label"] = str(tmp_path / "bad.nii.gz")
    with pytest.raises(ValueError):
        prepare_case(case, str(tmp_path / "x"), _pp())


@pytest.mark.parametrize("method", ["mean", "biharmonic"])
def test_inpaint_hook_fills_mask(method):
    img = np.full((30, 30, 10), 100.0, np.float32)
    img[14:17, 14:17, 4:6] = 10.0                    # dark spot
    mask = img < 50
    brain = np.ones(img.shape, bool)
    out = inpaint(img, mask, brain, method=method)
    assert np.allclose(out[mask], 100.0, atol=1.0)
    assert np.array_equal(out[~mask], img[~mask])


def test_inpaint_custom_function_and_none():
    img = np.random.rand(8, 8, 4).astype(np.float32)
    assert inpaint(img, img > 0.5, img > 0, method="none") is img
    out = inpaint(img, img > 0.5, img > 0, method="tests.test_preprocess:_zero_fill")
    assert (out[img > 0.5] == 0).all()


def _zero_fill(image, mask, brain, **_):
    out = image.copy(); out[mask] = 0
    return out


def test_resample_identity():
    v = np.random.rand(10, 12, 8).astype(np.float32)
    out = resample(v, np.ones(3), np.zeros(3), np.array(v.shape))
    assert np.allclose(out, v)
