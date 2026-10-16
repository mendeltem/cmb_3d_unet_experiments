"""The recommended BIDS layout end to end, as in the README: training data with references (examples/bids_train
layout) -> make-caselist --bids -> run-experiment -> new data without references (examples/bids_new layout) ->
make-caselist --bids -> predict on the input grid."""
import csv
import glob
import os

import nibabel as nib
import numpy as np

from cmbx.caselist import bids_patterns, build_caselist
from cmbx.config import load_config
from cmbx.data import write_caselist
from cmbx.experiments import run_experiment
from cmbx.synthetic import write_phantom_bids

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = {"model": {"base_channels": 8}, "preprocess": {"workers": 2},
        "train": {"epochs": 2, "min_epochs": 2, "eval_every": 1, "n_patches": 8, "batch_size": 4,
                  "patch_size": [16, 32, 32], "device": "cpu"}}


def _caselist(root, out, **kw):
    b = bids_patterns(root)
    rows = build_caselist(b["image"], b.get("label"), [b["brain_mask"]] if "brain_mask" in b else [], b.get("tissue_map"),
                          b["raw"], b.get("ids"), report_dir=os.path.dirname(out), log=print, **kw)
    write_caselist(rows, out)
    return rows


def test_bids_train_then_predict(tmp_path):
    train = write_phantom_bids(str(tmp_path / "bids_train"), 10, 0)
    new = write_phantom_bids(str(tmp_path / "bids_new"), 2, 7, labels=False)
    rows = _caselist(train, str(tmp_path / "runs" / "cases.csv"))
    assert len(rows) == 10 and all(r["label"] and r["brain_mask"] and r["tissue_map"] for r in rows)
    cfg = load_config(None, {**TINY, "name": "bids", "output_root": str(tmp_path / "runs"),
                             "data": {"cases": str(tmp_path / "runs" / "cases.csv"), "prepared_dir": str(tmp_path / "prep"),
                                      "splits": str(tmp_path / "s.json")}})
    run_experiment(cfg, log=lambda *a: None)
    assert os.path.exists(tmp_path / "runs" / "bids" / "final" / "model.pt")
    b = bids_patterns(new)
    assert "label" not in b and "brain_mask" in b                    # no references in a dataset for inference
    nrows = _caselist(new, str(tmp_path / "pred_in" / "cases.csv"), folds=False)
    assert [r["id"] for r in nrows] == ["sub-001", "sub-002"] and all(r["label"] == "" for r in nrows)
    from cmbx.predict_cli import predict_run
    predict_run(str(tmp_path / "runs" / "bids"), str(tmp_path / "pred_in" / "cases.csv"), str(tmp_path / "pred"),
                device="cpu", log=lambda *a: None)
    src = nib.load(os.path.join(new, "sub-001", "anat", "sub-001_T2starw.nii.gz"))
    m = nib.load(tmp_path / "pred" / "sub-001_mask.nii.gz")
    assert m.shape == src.shape and np.allclose(m.affine, src.affine)


def test_shipped_examples_match_the_layout(tmp_path):
    """examples/train and examples/new in the repository are what make-phantoms --preprocessed writes."""
    import filecmp
    from cmbx.synthetic import write_phantom_preprocessed
    for name, n, seed, labels in (("train", 10, 0, True), ("new", 2, 7, False)):
        ref = write_phantom_preprocessed(str(tmp_path / name), n, seed, labels=labels)
        a = sorted(os.path.relpath(f, ref) for f in glob.glob(os.path.join(ref, "**", "*.*"), recursive=True))
        b = sorted(os.path.relpath(f, os.path.join(ROOT, "examples", name))
                   for f in glob.glob(os.path.join(ROOT, "examples", name, "**", "*.*"), recursive=True))
        assert a == b, name
        assert all(filecmp.cmp(os.path.join(ref, f), os.path.join(ROOT, "examples", name, f), shallow=False) for f in a
                   if f.endswith(".tsv"))
