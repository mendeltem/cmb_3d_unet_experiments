"""Own extra input channel (preprocess.extra_channels): prepare puts it on the grid, the CV trains with it; FRST can be left out."""
import csv
import json
import os

import nibabel as nib

from cmbx.config import load_config
from cmbx.experiments import run_experiment
from cmbx.synthetic import write_phantom_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _with_extra_column(cases):
    rows = list(csv.DictReader(open(cases)))
    for r in rows:
        r["extra_path"] = r["image"]                  # any map on the image matrix works; here the image itself
    with open(cases, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return cases


def _run(tmp_path, name, channels, frst):
    cases = _with_extra_column(write_phantom_dataset(str(tmp_path / "phantoms"), n_cases=10, seed=1))
    cfg = load_config(os.path.join(ROOT, "configs", "smoke_cpu.yaml"), {
        "name": name, "output_root": str(tmp_path / "runs"),
        "data": {"cases": cases, "prepared_dir": str(tmp_path / f"prepared_{name}"), "splits": str(tmp_path / "splits.json")},
        "preprocess": {"extra_channels": {"myextra": "extra_path"}, "frst": frst},
        "train": {"channels": channels, "epochs": 1, "min_epochs": 1, "eval_every": 1}})
    s = run_experiment(cfg, log=print)
    meta = json.load(open(tmp_path / "runs" / name / "fold0" / "meta.json"))
    return s, meta, tmp_path / f"prepared_{name}"


def test_three_channels(tmp_path):
    s, meta, prep = _run(tmp_path, "three", ["image", "frst", "myextra"], True)
    assert meta["ok"] and s["cases"] == 10
    img = nib.load(str(prep / "phantom-00" / "phantom-00_image.nii.gz"))
    ex = nib.load(str(prep / "phantom-00" / "phantom-00_myextra.nii.gz"))
    assert img.shape == ex.shape and float(ex.get_fdata().max()) <= 1.0


def test_extra_instead_of_frst(tmp_path):
    s, meta, prep = _run(tmp_path, "nofrst", ["image", "myextra"], False)
    assert meta["ok"] and s["cases"] == 10
    assert not (prep / "phantom-00" / "phantom-00_frst.nii.gz").exists()
