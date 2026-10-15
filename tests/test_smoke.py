"""End-to-end on synthetic phantoms, CPU: phantoms -> prepare -> split -> 5-fold CV (tiny) -> evaluate -> predict."""
import json
import os

import pytest

from cmbx.config import load_config
from cmbx.evaluate import compare_runs, evaluate_run
from cmbx.experiments import run_experiment
from cmbx.synthetic import write_phantom_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_end_to_end_cv_on_phantoms(tmp_path):
    cases = write_phantom_dataset(str(tmp_path / "phantoms"), n_cases=10, seed=0)
    cfg = load_config(os.path.join(ROOT, "configs", "smoke_cpu.yaml"), {
        "output_root": str(tmp_path / "runs"),
        "data": {"cases": cases, "prepared_dir": str(tmp_path / "prepared"),
                 "splits": str(tmp_path / "splits.json")}})
    s = run_experiment(cfg, log=print)
    run = tmp_path / "runs" / "smoke_cpu"
    for k in range(5):
        meta = json.load(open(run / f"fold{k}" / "meta.json"))
        assert meta["ok"] and meta["epochs_run"] == 2 and (run / f"fold{k}" / "model.pt").exists()
        assert meta["n_train"] + meta["n_inner"] + meta["n_test"] == 10
    assert s["cases"] == 10 and 0 <= s["f1"] <= 1
    split = json.load(open(tmp_path / "splits.json"))
    assert sorted(x for f in split["folds"] for x in f) == [f"phantom-{i:02d}" for i in range(10)]
    # re-scoring the stored maps with other post-processing, no retraining
    e1 = evaluate_run(str(run), str(tmp_path / "prepared"), "inner", True, cfg["selection"])
    e2 = evaluate_run(str(run), str(tmp_path / "prepared"), "crossfold", False, cfg["selection"])
    assert e1["cases"] == e2["cases"] == 10
    rows = compare_runs(str(tmp_path / "runs"), log=print)
    assert any(r["name"] == "smoke_cpu" for r in rows)
    assert os.path.exists(tmp_path / "runs" / "summary.csv")
    # fold-ensemble prediction for "new" cases (here two of the phantoms)
    from cmbx.predict_cli import predict_run
    import csv
    sub = tmp_path / "new.csv"
    with open(cases) as f:
        r = list(csv.DictReader(f))[:2]
    base = os.path.dirname(cases)
    with open(sub, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "image", "brain_mask"])
        w.writeheader()
        for x in r:
            w.writerow(dict(id=x["id"], image=os.path.join(base, x["image"]), brain_mask=os.path.join(base, x["brain_mask"])))
    predict_run(str(run), str(sub), str(tmp_path / "pred"), device="cpu", log=print)
    assert os.path.exists(tmp_path / "pred" / "phantom-00_mask.nii.gz")
