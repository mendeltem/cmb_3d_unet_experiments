"""Training and inference options: losses, config checks, predict outputs on the input grid, single-scan mode,
readable CLI errors. CPU, synthetic phantoms, tiny settings."""
import csv
import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch

from cmbx.config import load_config
from cmbx.experiments import check_config, run_experiment
from cmbx.synthetic import write_phantom_dataset
from cmbx.train import LOSSES

TINY = {"model": {"base_channels": 8}, "preprocess": {"workers": 2},
        "train": {"epochs": 2, "min_epochs": 2, "eval_every": 1, "n_patches": 8, "batch_size": 4,
                  "patch_size": [16, 32, 32], "device": "cpu"}}


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_losses_finite_with_gradient(name):
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
    target = (torch.rand(2, 1, 8, 8, 8) > 0.9).float()
    loss = LOSSES[name](logits, target)
    loss.backward()
    assert loss.ndim == 0 and torch.isfinite(loss) and float(loss) >= 0 and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("override,msg", [({"train": {"loss": "nope"}}, "train.loss"),
                                          ({"model": {"name": "nope"}}, "model.name"),
                                          ({"train": {"patch_size": [16, 32]}}, "patch_size"),
                                          ({"train": {"device": "gpu"}}, "train.device"),
                                          ({"train": {"min_epochs": 5, "epochs": 2}}, "min_epochs"),
                                          ({"train": {"channels": ["frst"]}}, "channels")])
def test_config_check_messages(override, msg):
    with pytest.raises(SystemExit, match=msg):
        check_config(load_config(None, override))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("opt")
    cases = write_phantom_dataset(str(tmp / "ph"), 10, 0)
    cfg = load_config(None, {**TINY, "name": "opt", "output_root": str(tmp / "runs"),
                             "data": {"cases": cases, "prepared_dir": str(tmp / "prep"), "splits": str(tmp / "s.json")}})
    run_experiment(cfg, log=lambda *a: None)
    return tmp, str(tmp / "runs" / "opt"), cases


def test_predict_outputs_on_input_grid(trained):
    from cmbx.predict_cli import predict_run
    tmp, run, cases = trained
    new = write_phantom_dataset(str(tmp / "new"), 2, 7)
    rows = list(csv.DictReader(open(new)))
    lst = tmp / "unlabeled.csv"                       # what a user has for a new scan: no label column
    with open(lst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image"])
        for r in rows:
            w.writerow([r["id"], os.path.join(tmp / "new", r["image"])])
    out = tmp / "pred"
    res = predict_run(run, str(lst), str(out), device="cpu", log=lambda *a: None)
    assert [r["id"] for r in res] == [r["id"] for r in rows]
    for r in rows:
        src = nib.load(os.path.join(tmp / "new", r["image"]))
        m = nib.load(out / f"{r['id']}_mask.nii.gz")
        assert m.shape == src.shape[:3] and np.allclose(m.affine, src.affine)          # input grid
        g = nib.load(out / "grid" / f"{r['id']}_mask.nii.gz")
        assert g.shape != src.shape[:3]                                                # model grid copy
        info = json.load(open(out / f"{r['id']}.json"))
        assert info["n_lesions"] == len(info["lesions"]) and "warning" not in info
        for e in info["lesions"]:
            v = e["centre_voxel"]
            assert all(0 <= v[i] < src.shape[i] for i in range(3)) and e["volume_mm3"] > 0
    assert len(list(csv.DictReader(open(out / "predictions.csv")))) == 2


def test_predict_single_image_and_errors(trained):
    from cmbx.predict_cli import predict_run
    tmp, run, cases = trained
    r = list(csv.DictReader(open(cases)))[0]
    img, bm = (os.path.join(tmp / "ph", r[k]) for k in ("image", "brain_mask"))
    res = predict_run(run, out=str(tmp / "one"), image=img, brain_mask=bm, case_id="scan1", device="cpu", log=lambda *a: None)
    assert res[0]["id"] == "scan1" and os.path.exists(tmp / "one" / "scan1_mask.nii.gz")
    with pytest.raises(SystemExit, match="tissue map"):
        predict_run(run, out=str(tmp / "two"), image=img, csf_rule=True, device="cpu", log=lambda *a: None)
    with pytest.raises(SystemExit, match="not a finished run"):
        predict_run(str(tmp / "nope"), out=str(tmp / "three"), image=img, device="cpu", log=lambda *a: None)


@pytest.mark.parametrize("args,msg", [(["predict", "--run", "x", "--cases", "/nonexistent.csv", "--out", "o"], "not a finished run"),
                                      (["run-experiment", "configs/baseline.yaml", "--set", "train.epochz=3"], "unknown config key")])
def test_cli_errors_are_one_line(args, msg, tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: v for k, v in os.environ.items() if k != "CMBX_DEBUG"}
    p = subprocess.run([sys.executable, "-m", "cmbx", *args], capture_output=True, text=True, cwd=root, env=env)
    assert p.returncode != 0 and "Traceback" not in p.stderr and msg in p.stderr, p.stderr


def test_cli_missing_case_list_is_one_line(trained):
    tmp, run, cases = trained
    env = {k: v for k, v in os.environ.items() if k != "CMBX_DEBUG"}
    p = subprocess.run([sys.executable, "-m", "cmbx", "predict", "--run", run, "--cases", str(tmp / "missing.csv"),
                        "--out", str(tmp / "o")], capture_output=True, text=True, env=env)
    assert p.returncode != 0 and "Traceback" not in p.stderr and "missing.csv" in p.stderr, p.stderr


def test_old_preprocess_signature_is_reused():
    from cmbx.config import preprocess_signature, same_preprocessing
    new = preprocess_signature(load_config(None))
    old = {k: v for k, v in new.items() if k != "extra_channels"}            # written before extra_channels existed
    assert same_preprocessing(old, new)
    assert not same_preprocessing({**old, "spacing": [1.0, 1.0, 1.0]}, new)
    assert not same_preprocessing(old, {**new, "extra_channels": {"v": "v_path"}})


def test_train_final_overwrite(trained):
    from cmbx.final import train_final
    import yaml
    tmp, run, cases = trained
    cfg = yaml.safe_load(open(os.path.join(run, "config.yaml")))
    before = os.path.getmtime(os.path.join(run, "final", "model.pt"))
    msgs = []
    train_final(cfg, run, msgs.append)                                    # exists: skipped, with a hint
    assert any("--overwrite" in m for m in msgs) and os.path.getmtime(os.path.join(run, "final", "model.pt")) == before
    train_final(cfg, run, lambda *a: None, overwrite=True)
    assert os.path.getmtime(os.path.join(run, "final", "model.pt")) > before
