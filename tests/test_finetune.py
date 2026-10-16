"""Fine-tuning from a pretrained model (train.init_from): channels matched by name, new channels start at zero weight
(the network first computes exactly what the pretrained one did), dropped channels removed; CV and final model."""
import json
import os

import numpy as np
import pytest
import torch

from cmbx.config import load_config
from cmbx.experiments import run_experiment
from cmbx.model import build_model, init_from_pretrained
from cmbx.synthetic import write_phantom_dataset

TINY = {"model": {"base_channels": 8}, "preprocess": {"workers": 2},
        "train": {"epochs": 2, "min_epochs": 2, "eval_every": 1, "n_patches": 8, "batch_size": 4,
                  "patch_size": [16, 32, 32], "device": "cpu"}}


@pytest.fixture(scope="module")
def pretrained(tmp_path_factory):
    torch.manual_seed(0)
    src = build_model("aniso_unet3d", 2, 8).eval()
    d = tmp_path_factory.mktemp("pre")
    torch.save({"state_dict": src.state_dict()}, d / "model.pt")
    return src, str(d / "model.pt")


def test_new_channel_starts_at_zero_and_reproduces_the_pretrained_output(pretrained):
    src, path = pretrained
    dst = build_model("aniso_unet3d", 3, 8).eval()
    info = init_from_pretrained(dst, path, ["image", "frst", "vessel"], ["image", "frst"], log=print)
    assert info["new_channels"] == ["vessel"]
    x = torch.randn(1, 3, 16, 32, 32)
    with torch.no_grad():
        assert torch.equal(dst(x), src(x[:, :2]))                       # the zero-weight channel changes nothing


def test_dropped_channel_and_reordered_channels(pretrained):
    src, path = pretrained
    one = build_model("aniso_unet3d", 1, 8).eval()
    assert init_from_pretrained(one, path, ["image"], ["image", "frst"], log=print)["dropped_channels"] == ["frst"]
    rev = build_model("aniso_unet3d", 2, 8).eval()
    init_from_pretrained(rev, path, ["frst", "image"], ["image", "frst"], log=print)
    x = torch.randn(1, 2, 16, 32, 32)
    with torch.no_grad():
        assert torch.allclose(rev(x.flip(1)), src(x), atol=1e-5)         # same channels in another order (float sum order)


def test_width_mismatch_is_a_clear_error(pretrained):
    _, path = pretrained
    with pytest.raises(ValueError, match="base_channels"):
        init_from_pretrained(build_model("aniso_unet3d", 2, 16), path, ["image", "frst"], ["image", "frst"], log=print)


def test_finetune_run_from_a_run_folder(tmp_path):
    cases = write_phantom_dataset(str(tmp_path / "ph"), 10, 0)
    base = {**TINY, "output_root": str(tmp_path / "runs"),
            "data": {"cases": cases, "prepared_dir": str(tmp_path / "prep"), "splits": str(tmp_path / "s.json")}}
    run_experiment(load_config(None, {**base, "name": "source"}), log=lambda *a: None)
    ft = load_config(None, {**base, "name": "ft"})
    ft["train"]["init_from"] = str(tmp_path / "runs" / "source")         # a run folder, as models/valdo_t2
    run_experiment(ft, log=lambda *a: None)
    meta = json.load(open(tmp_path / "runs" / "ft" / "fold0" / "meta.json"))
    assert meta["init_from"].endswith(os.path.join("source", "final", "model.pt")) and meta["new_channels"] == []
    assert os.path.exists(tmp_path / "runs" / "ft" / "final" / "model.pt")
