"""Acceptance test for `--defaults microbleednet` (train, fine_tune, cross_validate).

With the switch, every MicrobleedNet option the user did NOT give takes MicrobleedNet's own default
(console_commands/microbleednet.py at commit 958f1cb); options the user gives still win. Without the
switch (or with `--defaults cmbx`) nothing changes.
"""
import argparse

import pytest

from cmbx.workflow import add_parsers, build_config


def parse(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_parsers(sub, print)
    return ap.parse_args(argv)


def cfg_for(argv, tmp_path):
    a = parse(argv)
    run = tmp_path / "run"
    return a, build_config(a, str(run), str(run / "cases.csv"), str(tmp_path / "prep"))


# MicrobleedNet defaults per command (958f1cb); the patch is cubic.
MBN = {
    "train": dict(lr=1e-3, batch_size=4, epochs=100),
    "fine_tune": dict(lr=1e-4, batch_size=8, epochs=60),
    "cross_validate": dict(lr=1e-3, batch_size=8, epochs=60),
}
COMMON = dict(train_prop=0.8, subjects_per_epoch=10, aug_factor=2, optimizer="adam", epsilon=1e-4, momentum=0.9,
              lr_gamma=0.1, early_stop=20, checkpoint="last", checkpoint_every=10, flip_augment=True)


def base_args(cmd, tmp_path):
    a = ["-i", str(tmp_path / "in"), "-l", str(tmp_path / "in")]
    if cmd == "train":
        return ["train"] + a + ["-m", str(tmp_path / "m")]
    if cmd == "fine_tune":
        return ["fine_tune"] + a + ["-m", "pre", "-o", str(tmp_path / "o")]
    return ["cross_validate"] + a + ["-o", str(tmp_path / "o")]


@pytest.mark.parametrize("cmd", ["train", "fine_tune", "cross_validate"])
def test_microbleednet_defaults_fill_unset_options(cmd, tmp_path):
    _, cfg = cfg_for(base_args(cmd, tmp_path) + ["--defaults", "microbleednet"], tmp_path)
    T = cfg["train"]
    for key, value in {**COMMON, **MBN[cmd]}.items():
        assert T[key] == pytest.approx(value) if isinstance(value, float) else T[key] == value, key
    assert T["patch_size"] == [48, 48, 48]
    assert T["scheduler"] == "multistep" and list(T["lr_milestones"]) == [10]
    assert cfg["stage2"]["enabled"] is True and cfg["stage2"]["method"] == "teacher_student"


@pytest.mark.parametrize("cmd", ["train", "fine_tune", "cross_validate"])
def test_given_options_beat_the_microbleednet_defaults(cmd, tmp_path):
    _, cfg = cfg_for(base_args(cmd, tmp_path) + ["--defaults", "microbleednet", "-ep", "7", "-ilr", "0.005",
                                                 "-opt", "adamw", "-cand_disc", "False", "-psize", "32"], tmp_path)
    T = cfg["train"]
    assert T["epochs"] == 7 and T["lr"] == pytest.approx(0.005) and T["optimizer"] == "adamw"
    assert T["patch_size"] == [32, 32, 32]
    assert cfg["stage2"]["enabled"] is False


@pytest.mark.parametrize("cmd", ["train", "fine_tune", "cross_validate"])
def test_without_the_switch_nothing_changes(cmd, tmp_path):
    _, plain = cfg_for(base_args(cmd, tmp_path), tmp_path)
    _, explicit = cfg_for(base_args(cmd, tmp_path) + ["--defaults", "cmbx"], tmp_path)
    assert plain == explicit
    assert plain["train"]["optimizer"] != "adam" or plain["train"]["epsilon"] != 1e-4   # cmbx recipe, not MicrobleedNet


def test_unknown_preset_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        parse(base_args("train", tmp_path) + ["--defaults", "nnunet"])


def test_help_names_the_switch(capsys):
    with pytest.raises(SystemExit):
        parse(["train", "--help"])
    assert "--defaults" in capsys.readouterr().out


def test_set_beats_the_preset(tmp_path):
    """A key given with --set is not overridden by --defaults microbleednet (the preset only stands in for options nobody gave)."""
    _, cfg = cfg_for(base_args("train", tmp_path) + ["--defaults", "microbleednet", "--set", "train.lr=0.0003",
                                                     "train.epochs=5"], tmp_path)
    assert cfg["train"]["lr"] == pytest.approx(0.0003) and cfg["train"]["epochs"] == 5
    assert cfg["train"]["optimizer"] == "adam"             # the rest still comes from the preset
