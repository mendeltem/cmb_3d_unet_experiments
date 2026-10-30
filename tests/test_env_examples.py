"""examples/env: settings from a .env file, the choice of cases (folds, nested stratified draws) and run_train.py,
run_fine_tune.py, run_predict.py end to end on phantoms (CPU, tiny network)."""
import csv
import importlib.util
import json
import os
import subprocess
import sys

import pytest
import yaml

from cmbx.synthetic import write_phantom_preprocessed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DIR = os.path.join(ROOT, "examples", "env")
TINY = {"model": {"base_channels": 8}, "preprocess": {"workers": 2},
        "train": {"n_patches": 8, "batch_size": 4, "patch_size": [16, 32, 32], "eval_every": 1}}

spec = importlib.util.spec_from_file_location("cmb_env", os.path.join(ENV_DIR, "cmb_env.py"))
cmb_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmb_env)


def settings_for(tmp_path, data, **extra):
    s = dict(cmb_env.DEFAULTS, CMB_DATA=str(data), CMB_RUNS=str(tmp_path / "runs"), **extra)
    s = cmb_env.resolve(s)
    s["NAME"] = s["NAME"] or os.path.basename(str(data))
    return s


def test_read_env_takes_key_value_lines_and_drops_comments_and_quotes(tmp_path):
    f = tmp_path / ".env"
    f.write_text("# comment\nA=1\nB = 'two'  # trailing\n\nC=\"x y\"\nnot a setting\n")
    assert cmb_env.read_env(str(f)) == {"A": "1", "B": "two", "C": "x y"}


def test_resolve_fills_references_to_other_keys_and_to_the_repository():
    s = cmb_env.resolve({"A": "${REPO}/data", "B": "${A}/sub"})
    assert s["A"] == os.path.join(cmb_env.REPO, "data") and s["B"] == s["A"] + "/sub"


def test_draws_are_nested_and_stratified():
    rows = [dict(id=f"c{i:02d}", n_cmb=i % 6) for i in range(30)]
    d5, d10 = (cmb_env.draw_stratified(rows, n, 3) for n in (5, 10))
    assert [r["id"] for r in d10[:5]] == [r["id"] for r in d5]          # the 5 are the first 5 of the 10
    assert len({cmb_env.stratum(r["n_cmb"]) for r in d5}) == 4          # every count class is in a draw of 5
    assert cmb_env.draw_stratified(rows, 5, 4) != d5                    # another seed, another draw


def test_folds_are_drawn_once_from_the_masks_and_kept(tmp_path):
    data = write_phantom_preprocessed(str(tmp_path / "cohort"), 10, 0)          # participants.tsv without counts/folds
    s = settings_for(tmp_path, data)
    first = cmb_env.cases(s)
    assert {r["fold"] for r in first} == {0, 1, 2, 3, 4} and all(r["n_cmb"] >= 0 for r in first)
    kept = os.path.join(s["CMB_RUNS"], s["NAME"], "_input", "folds.tsv")
    assert os.path.exists(kept)
    assert cmb_env.cases(settings_for(tmp_path, data, FOLD_SEED="9")) == first   # kept folds win over a new seed


def test_cases_for_test_fold_and_n_scans(tmp_path):
    data = write_phantom_preprocessed(str(tmp_path / "cohort"), 10, 0)
    s = settings_for(tmp_path, data, TEST_FOLD="0", N_SCANS="4")
    train, tag = cmb_env.cases_for(s, "train")
    test, ttag = cmb_env.cases_for(s, "predict")
    assert len(train) == 4 and tag == "not_fold0_n4_seed0" and ttag == "fold0"
    assert not set(train) & set(test)
    link_dir = cmb_env.linked_input(s, train, tag)
    assert sorted(os.listdir(link_dir)) == sorted(train)


def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_train_fine_tune_predict_end_to_end(tmp_path):
    data = write_phantom_preprocessed(str(tmp_path / "cohort"), 10, 0)
    tiny = str(tmp_path / "tiny.yaml"); yaml.safe_dump(TINY, open(tiny, "w"))
    env = tmp_path / ".env"
    env.write_text(f"CMB_DATA={data}\nCMB_RUNS={tmp_path}/runs\nTEST_FOLD=0\nN_SCANS=6\nEPOCHS=1\nCONFIG={tiny}\n"
                   "DEVICE=cpu\nWORKERS=2\n")
    common = ["--env", str(env)]
    run(os.path.join(ENV_DIR, "run_train.py"), *common)
    model = os.path.join(tmp_path, "runs", "cohort", "train_not_fold0_n6_seed0")
    assert os.path.exists(os.path.join(model, "final", "model.pt"))
    used = json.load(open(os.path.join(model, "run_settings.json")))["cases"]
    assert len(used) == 6
    run(os.path.join(ENV_DIR, "run_fine_tune.py"), *common, "--set", f"START_MODEL={model}")
    ft = os.path.join(tmp_path, "runs", "cohort", "finetune_train_not_fold0_n6_seed0_not_fold0_n6_seed0")
    assert os.path.exists(os.path.join(ft, "final", "model.pt"))
    out = run(os.path.join(ENV_DIR, "run_predict.py"), *common, "--set", f"MODEL={ft}")
    pred = os.path.join(tmp_path, "runs", "cohort", f"predict_{os.path.basename(ft)}_fold0")
    summary = json.load(open(os.path.join(pred, "comparison_summary.json")))
    rows = list(csv.DictReader(open(os.path.join(pred, "comparison_per_case.csv"))))
    assert summary["cases"] == len(rows) == 2 and not set(r["case"] for r in rows) & set(used)
    assert {"lesion_f1", "precision", "recall", "fp_per_scan", "more_than_4_agree"} <= set(summary)
    assert "comparing only" in run(os.path.join(ENV_DIR, "run_predict.py"), *common, "--set", f"MODEL={ft}")
    with pytest.raises(AssertionError):                       # a finished model is not overwritten without --force true
        run(os.path.join(ENV_DIR, "run_train.py"), *common)
