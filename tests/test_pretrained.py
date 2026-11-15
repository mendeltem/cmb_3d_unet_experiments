"""Pretrained models (cmbx get-model): only VALDO-trained weights and VALDO-chosen defaults are offered, the old name
cmb_classifier still works, and the downloaded U-Net + classifier load and predict on a synthetic scan.

The download test fetches the release files into <repo>/models (as install.sh does, SHA-256 checked, kept for the next
run); it is skipped when GitHub cannot be reached. CPU only."""
import json
import os
import subprocess
import sys
import urllib.request

import nibabel as nib
import numpy as np
import pytest

from cmbx.pretrained import ALIASES, REGISTRY, get_model
from cmbx.synthetic import write_phantom_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
QUIET = lambda *a: None


def test_registry_offers_only_valdo_trained_models():
    assert set(REGISTRY) == {"valdo_t2", "valdo_classifier"}
    for name, m in REGISTRY.items():
        assert m["trained_on"].startswith("VALDO 2021 Task 2 only"), name
        assert "CC BY-NC-SA 4.0" in m["licence"] and "doi:10.1016/j.media.2023.103029" in m["cite"], name
        assert "in-house" not in m["cell_note"] and "in-house" not in m["trained_on"], name
        assert "/cmb_3d_unet_experiments/releases/download/v0.2.0/" in m["url"], name
        assert all(len(sha) == 64 for _, sha, _ in m["files"]), name
    assert REGISTRY["valdo_t2"]["threshold"] == 0.3 and REGISTRY["valdo_t2"]["min_size"] == 8
    assert ALIASES == {"cmb_classifier": "valdo_classifier"}


def test_unknown_name_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="unknown model"):
        get_model("charite_t2", str(tmp_path), QUIET)


def _online() -> bool:
    try:
        urllib.request.urlopen("https://github.com", timeout=10).close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def models():
    if not _online():
        pytest.skip("GitHub not reachable: release download not tested")
    get_model("valdo_t2", MODELS, QUIET)
    get_model("valdo_classifier", MODELS, QUIET)
    get_model("cmb_classifier", MODELS, QUIET)            # old name -> the same VALDO-only file
    return MODELS


def test_download_checks_and_writes_the_valdo_defaults(models):
    from cmbx.stage2 import load_bundle
    run = os.path.join(models, "valdo_t2")
    for k in ("final", "fold0", "fold1"):
        meta = json.load(open(os.path.join(run, k, "meta.json")))
        assert (meta["threshold"], meta["min_size"]) == (0.3, 8)
    card = open(os.path.join(run, "MODEL_CARD.md")).read()
    assert "VALDO 2021 Task 2 only" in card and "in-house" not in card
    new, old = load_bundle(os.path.join(models, "valdo_classifier")), load_bundle(os.path.join(models, "cmb_classifier"))
    assert len(new["members"]) == 3 and new["crop"] == [16, 32, 32]
    assert new["threshold"] == REGISTRY["valdo_classifier"]["threshold"]
    assert list(new.get("trained_on", ["valdo"])) == ["valdo"]
    assert old["threshold"] == new["threshold"] and old["created"] == new["created"]


def test_predict_with_the_released_unet_and_classifier(models, tmp_path):
    cases = write_phantom_dataset(str(tmp_path / "ph"), 1, 3)
    d = os.path.dirname(cases)
    image, brain = (os.path.join(d, "phantom-00", f"phantom-00_{k}.nii.gz") for k in ("image", "brain"))
    out = str(tmp_path / "pred")
    r = subprocess.run([sys.executable, "-m", "cmbx", "predict", "--run", os.path.join(models, "valdo_t2"),
                        "--stage2", os.path.join(models, "valdo_classifier"), "--image", image, "--brain-mask", brain,
                        "--out", out, "--device", "cpu"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    sid = os.path.basename(image)[:-len(".nii.gz")]
    m = nib.load(os.path.join(out, f"{sid}_mask.nii.gz"))
    assert m.shape == nib.load(image).shape and np.isfinite(np.asarray(m.dataobj)).all()
    info = json.load(open(os.path.join(out, f"{sid}.json")))
    assert info["threshold"] == 0.3 and info["min_size"] == 8
    assert info["stage2"]["threshold"] == REGISTRY["valdo_classifier"]["threshold"] and info["stage2_candidates"] >= 0
