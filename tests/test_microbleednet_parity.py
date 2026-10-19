"""Parity with MicrobleedNet's command line (Sundaresan 2023, commit 958f1cb).

tests/microbleednet_958f1cb_options.json holds every option of MicrobleedNet's five commands as declared in its
argparse code (extracted from the original file). For each of them this test checks that
  1. cmbx accepts the flag (short and long form) on the same command, so existing MicrobleedNet scripts run;
  2. a non-default value reaches the cmbx config, or the option is listed as handled elsewhere / without effect;
  3. with `--defaults microbleednet` every left-out option resolves to MicrobleedNet's own default;
  4. switches read True/False as written (MicrobleedNet's `type=bool` turns any text, even "False", into True).
"""
import argparse
import json
import os

import pytest

from cmbx.workflow import NO_EFFECT, PRESET_KEYS, _notes, add_parsers, build_config

SPEC = json.load(open(os.path.join(os.path.dirname(__file__), "microbleednet_958f1cb_options.json")))["commands"]

# options consumed outside build_config (data folders, model choice, folds, evaluation output) -> where it is tested
ELSEWHERE = {"inp_dir", "out_dir", "output_dir", "label_dir", "model_dir", "model_name", "image_regex", "label_regex",
             "fsl_preprocessed", "pretrained_model", "intermediate", "cp_load_type", "cv_fold", "resume_from_fold",
             "cand_detection"}
# dest -> (config path, value given on the command line, value expected in the config)
EXPECT = {
    "train_prop": ("train.train_prop", "0.7", 0.7), "batch_factor": ("train.subjects_per_epoch", "3", 3),
    "patch_size": ("train.patch_size", "32", [32, 32, 32]), "data_augmentation": ("train.flip_augment", "False", False),
    "aug_factor": ("train.aug_factor", "3", 3), "save_resume_training": ("train.resume", "True", True),
    "init_learng_rate": ("train.lr", "0.005", 0.005), "lr_sch_mlstone": ("train.lr_milestones", ["4", "8"], [4, 8]),
    "lr_sch_gamma": ("train.lr_gamma", "0.5", 0.5), "optimizer": ("train.optimizer", "sgd", "sgd"),
    "epsilon": ("train.epsilon", "0.001", 0.001), "momentum": ("train.momentum", "0.8", 0.8),
    "batch_size": ("train.batch_size", "2", 2), "num_epochs": ("train.epochs", "7", 7),
    "early_stop_val": ("train.early_stop", "5", 5), "save_full_model": ("train.save_full_model", "True", True),
    "cp_save_type": ("train.checkpoint", "everyN", "everyN"), "cp_everyn_N": ("train.checkpoint_every", "3", 3),
    "multi_gpu": ("train.multi_gpu", "True", True), "ft_layers": ("train.finetune_layers", ["1", "2"], [1, 2]),
    "cand_discrimination": ("stage2.enabled", "False", False),
}
REQ = {"preprocess": ["-i", "IN", "-o", "OUT", "-ir", "_T2S"], "train": ["-i", "IN", "-l", "IN", "-m", "M"],
       "evaluate": ["-i", "IN", "-m", "M", "-o", "OUT"], "fine_tune": ["-i", "IN", "-l", "IN", "-m", "M", "-o", "OUT"],
       "cross_validate": ["-i", "IN", "-l", "IN", "-o", "OUT"]}
SAMPLE = {"int": "3", "float": "0.5", "str": "x", "bool": "True"}
ALL = [(c, o) for c, opts in SPEC.items() for o in opts]


def parser():
    ap = argparse.ArgumentParser()
    add_parsers(ap.add_subparsers(dest="cmd", required=True), lambda *a: None)
    return ap


def value_for(o):
    if o["dest"] in EXPECT:
        v = EXPECT[o["dest"]][1]
        return v if isinstance(v, list) else [v]
    if o["dest"] in ("cp_load_type", "cp_save_type"):
        return ["best"]
    v = SAMPLE.get(o["type"], "x")
    return [v, v] if o["nargs"] in ("+", "*") else [v]


def get(cfg, path):
    for k in path.split("."):
        cfg = cfg[k]
    return cfg


@pytest.mark.parametrize("cmd,opt", ALL, ids=[f"{c}:{o['flags'][0]}" for c, o in ALL])
def test_flag_accepted_short_and_long(cmd, opt):
    for flag in opt["flags"]:
        a = parser().parse_args([cmd] + REQ[cmd] + ([flag] + value_for(opt) if flag not in REQ[cmd] else []))
        assert a.cmd == cmd


@pytest.mark.parametrize("cmd,opt", ALL, ids=[f"{c}:{o['flags'][0]}" for c, o in ALL])
def test_every_option_has_a_meaning(cmd, opt):
    """No MicrobleedNet option may be accepted and silently dropped."""
    d = opt["dest"]
    no_effect = d in NO_EFFECT or (cmd == "fine_tune" and d == "cp_load_type")   # fine_tune: dest cp_load_type_ft
    assert d in EXPECT or d in ELSEWHERE or no_effect, f"{cmd} {opt['flags']} has no tested meaning in cmbx"


@pytest.mark.parametrize("cmd", ["train", "fine_tune", "cross_validate"])
def test_values_reach_the_config(cmd, tmp_path):
    for opt in SPEC[cmd]:
        if opt["dest"] not in EXPECT:
            continue
        path, given, want = EXPECT[opt["dest"]]
        a = parser().parse_args([cmd] + REQ[cmd] + [opt["flags"][0]] + (given if isinstance(given, list) else [given]))
        cfg = build_config(a, str(tmp_path / "r"), str(tmp_path / "r" / "cases.csv"), str(tmp_path / "g"), log=lambda *x: None)
        assert get(cfg, path) == want, f"{cmd} {opt['flags'][0]} {given}: config {path} = {get(cfg, path)!r}, expected {want!r}"


@pytest.mark.parametrize("cmd", ["train", "fine_tune", "cross_validate"])
def test_defaults_microbleednet_equal_958f1cb(cmd, tmp_path):
    a = parser().parse_args([cmd] + REQ[cmd] + ["--defaults", "microbleednet"])
    cfg = build_config(a, str(tmp_path / "r"), str(tmp_path / "r" / "cases.csv"), str(tmp_path / "g"), log=lambda *x: None)
    checked = 0
    for opt in SPEC[cmd]:
        d = opt["dest"]
        if d not in PRESET_KEYS:
            continue
        want = opt["default"]
        if d == "lr_sch_mlstone":
            want = [want] if not isinstance(want, list) else want
        if d == "patch_size":
            want = [want] * 3
        assert get(cfg, PRESET_KEYS[d]) == want, f"{cmd} {opt['flags'][0]}: {get(cfg, PRESET_KEYS[d])!r} != MicrobleedNet {want!r}"
        checked += 1
    assert checked >= 15


def test_switches_read_false_as_false():
    a = parser().parse_args(["train"] + REQ["train"] + ["-da", "False", "-cand_disc", "False", "-sv_mod", "False"])
    assert (a.data_augmentation, a.cand_discrimination, a.save_full_model) == (False, False, False)


def test_options_without_effect_are_reported():
    for cmd in ("train", "fine_tune", "cross_validate", "evaluate", "preprocess"):
        for opt in SPEC[cmd]:
            d = "cp_load_type_ft" if (cmd == "fine_tune" and opt["dest"] == "cp_load_type") else opt["dest"]
            if d not in NO_EFFECT:
                continue
            a = parser().parse_args([cmd] + REQ[cmd] + [opt["flags"][0]] + value_for(opt))
            said = []
            _notes(a, said.append)
            assert said and NO_EFFECT[d] in said[0], f"{cmd} {opt['flags'][0]}: no note"


def test_every_option_at_once_end_to_end(tmp_path):
    """train / evaluate / fine_tune / cross_validate on synthetic phantoms with every MicrobleedNet option of the command
    given at a non-default value at the same time (CPU, tiny settings). Catches options that parse but crash in the run."""
    import glob
    from cmbx.cli import main
    from cmbx.synthetic import write_phantom_preprocessed
    cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "smoke_cpu.yaml")
    T = write_phantom_preprocessed(str(tmp_path / "train"), 10, 0)
    N = write_phantom_preprocessed(str(tmp_path / "new"), 2, 7, labels=False)
    m, out, ft, cv = (str(tmp_path / x) for x in ("m", "pred", "ft", "cv"))
    common = ["-tr_prop", "0.7", "-bfactor", "3", "-da", "False", "-af", "1", "-sv_resume", "False", "-ilr", "0.002",
              "-lrm", "1", "-gamma", "0.5", "-opt", "sgd", "-eps", "0.001", "-mom", "0.8", "-bs", "2", "-es", "5",
              "-sv_mod", "True", "-cp_type", "everyN", "-cp_n", "1", "-v", "True"]
    main(["train", "-i", T, "-l", T, "-m", m, "--config", cfg, "-ep", "2", "-cand_det", "True", "-cand_disc", "False",
          "-mgpu", "False"] + common)
    assert os.path.exists(os.path.join(m, "final", "model_full.pt")) and glob.glob(os.path.join(m, "final", "model_epoch*.pt"))
    main(["evaluate", "-i", N, "-m", m, "-o", out, "-p", "False", "-int", "True", "-cp_type", "everyN", "-cp_n", "1", "-v", "True"])
    assert len(os.listdir(os.path.join(out, "final_predictions"))) == 2
    main(["fine_tune", "-i", T, "-l", T, "-m", m, "-o", ft, "--config", cfg, "-ep", "1", "-p", "False", "-cpld_type", "best",
          "-cpld_n", "1", "-ftlayers", "1", "-cand_det", "True", "-cand_disc", "False"] + common)
    assert os.path.exists(os.path.join(ft, "config.yaml"))
    main(["cross_validate", "-i", T, "-l", T, "-o", cv, "--config", cfg, "-fold", "3", "-resume_fold", "1", "-ep", "1",
          "-int", "True", "-sv", "True"] + common)
    assert os.path.exists(os.path.join(cv, "summary.json")) and len(glob.glob(os.path.join(cv, "fold*", "model.pt"))) == 3
