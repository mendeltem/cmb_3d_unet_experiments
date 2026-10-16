"""Optional second stage (cmbx stage2): train from the out-of-fold maps with a nested CV, fine-tune with --init,
apply in predict (default when present, --no-stage2 to skip, --stage2 for another classifier incl. the release
format), and stage2.enabled inside run-experiment. CPU, phantoms, tiny settings."""
import json
import os

import pytest
import torch

from cmbx.config import load_config
from cmbx.experiments import run_experiment
from cmbx.stage2 import build_classifier, load_bundle, train_stage2
from cmbx.synthetic import write_phantom_dataset

TINY = {"model": {"base_channels": 8}, "preprocess": {"workers": 2},
        "train": {"epochs": 2, "min_epochs": 2, "eval_every": 1, "n_patches": 8, "batch_size": 4,
                  "patch_size": [16, 32, 32], "device": "cpu"},
        "stage2": {"epochs": 2, "members": 1}}
QUIET = lambda *a: None


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("st2")
    cases = write_phantom_dataset(str(tmp / "ph"), 10, 0)
    cfg = load_config(None, {**TINY, "name": "r", "output_root": str(tmp / "runs"),
                             "data": {"cases": cases, "prepared_dir": str(tmp / "prep"), "splits": str(tmp / "s.json")}})
    run_experiment(cfg, log=QUIET)
    _controlled_maps(str(tmp / "runs" / "r"), str(tmp / "prep"))
    return tmp, str(tmp / "runs" / "r"), cases


def _controlled_maps(run_dir, prepared):
    """Replace the out-of-fold maps of the tiny first stage (near-uniform after 2 epochs) by controlled ones: every
    reference microbleed at 0.8 plus two false spots at 0.4 per case -- this tests the second stage, not the first."""
    import glob
    import nibabel as nib
    import numpy as np
    rng = np.random.RandomState(0)
    for f in glob.glob(os.path.join(run_dir, "fold*", "*_pred_proba.nii.gz")):
        sid = os.path.basename(f)[:-len("_pred_proba.nii.gz")]
        img = nib.load(f)
        lab = np.asarray(nib.load(os.path.join(prepared, sid, f"{sid}_label.nii.gz")).dataobj) > 0
        p = np.where(lab, 0.8, 0.0)
        for _ in range(2):
            c = [rng.randint(10, s - 10) for s in p.shape]
            p[c[0] - 1:c[0] + 2, c[1] - 1:c[1] + 2, max(c[2] - 1, 0):c[2] + 2] = 0.4
        nib.save(nib.Nifti1Image(np.round(p * 255).astype(np.uint8), img.affine), f)


def test_train_stage2_nested_cv(run):
    tmp, r, _ = run
    s = train_stage2(r, "cpu", log=print)
    assert os.path.exists(os.path.join(r, "stage2", "classifier.pt"))
    assert {"stage2_cv", "threshold", "candidates", "stage1_cv_f1"} <= set(s) and 0 <= s["stage2_cv"]["f1"] <= 1
    b = load_bundle(r)
    assert len(b["members"]) == 1 and b["crop"] == [16, 32, 32]


def test_predict_uses_stage2_and_can_skip_it(run):
    from cmbx.predict_cli import predict_run
    tmp, r, cases = run
    if not os.path.exists(os.path.join(r, "stage2", "classifier.pt")):
        train_stage2(r, "cpu", log=QUIET)
    img = os.path.join(tmp, "ph", "phantom-00", "phantom-00_image.nii.gz")
    predict_run(r, out=str(tmp / "p2"), image=img, brain_mask=img.replace("_image", "_brain"), case_id="a", device="cpu", log=QUIET)
    j = json.load(open(tmp / "p2" / "a.json"))
    assert j["stage2"] and j["stage2"]["classifier"].endswith("classifier.pt") and "stage2_candidates" in j
    predict_run(r, out=str(tmp / "p1"), image=img, brain_mask=img.replace("_image", "_brain"), case_id="a", device="cpu", log=QUIET, no_stage2=True)
    assert json.load(open(tmp / "p1" / "a.json"))["stage2"] is None


def test_finetune_and_release_format(run):
    from cmbx.predict_cli import predict_run
    tmp, r, _ = run
    if not os.path.exists(os.path.join(r, "stage2", "classifier.pt")):
        train_stage2(r, "cpu", log=QUIET)
    # a bundle in the format of the microbleed_detection_3D_UNET release (other key names, three members)
    rel = tmp / "release" / "classifier.pt"
    rel.parent.mkdir()
    torch.save(dict(members=[build_classifier().state_dict() for _ in range(3)], threshold=0.3, crop=[16, 32, 32],
                    channels=["image", "frst", "probability"], stored_patch=[20, 40, 40], candidate_threshold=0.15,
                    candidate_min_mm3=1.0), rel)
    s = train_stage2(r, "cpu", init=str(rel), overrides=dict(epochs=1), log=QUIET)      # fine-tune from it
    assert s["init"] == str(rel)
    img = os.path.join(tmp, "ph", "phantom-01", "phantom-01_image.nii.gz")
    predict_run(r, out=str(tmp / "p3"), image=img, brain_mask=img.replace("_image", "_brain"), case_id="b", device="cpu", log=QUIET, stage2=str(rel))
    assert json.load(open(tmp / "p3" / "b.json"))["stage2"]["threshold"] == 0.3


def test_stage2_enabled_in_run_experiment(tmp_path):
    cases = write_phantom_dataset(str(tmp_path / "ph"), 10, 0)
    cfg = load_config(None, {**TINY, "name": "e", "output_root": str(tmp_path / "runs"),
                             "data": {"cases": cases, "prepared_dir": str(tmp_path / "prep"), "splits": str(tmp_path / "s.json")}})
    cfg["stage2"]["enabled"] = True
    msgs = []
    run_experiment(cfg, log=msgs.append)                  # never fails because of stage 2: CV + final model are kept
    st2 = tmp_path / "runs" / "e" / "stage2" / "classifier.pt"
    assert st2.exists() or any("stage 2 not trained" in m for m in msgs)    # tiny first stage: maybe no candidates
    assert os.path.exists(tmp_path / "runs" / "e" / "final" / "model.pt")
