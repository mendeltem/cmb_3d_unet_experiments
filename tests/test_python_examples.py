"""examples/python: 01_train_unet.py, 02_predict_microbleeds.py and 03_fine_tune_unet.py give the same weights and
masks as cmbx train / evaluate / fine_tune, and each of their functions does what it says. CPU, tiny network."""
import csv
import importlib.util
import inspect
import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml
from scipy import ndimage

from cmbx.config import load_config
from cmbx.data import PreparedCase, load_prepared
from cmbx.model import build_model
from cmbx.synthetic import write_phantom_preprocessed
from cmbx.workflow import collect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples", "python")
TINY = {"model": {"base_channels": 8}, "preprocess": {"workers": 2},
        "train": {"n_patches": 8, "batch_size": 4, "patch_size": [16, 32, 32], "eval_every": 1}}
QUIET = lambda *a, **k: None


def script(name):
    """A numbered script as a module (its steps sit behind the __main__ guard, so loading runs nothing)."""
    spec = importlib.util.spec_from_file_location(name[3:-3], os.path.join(EXAMPLES, name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN, PREDICT, FINE_TUNE = script("01_train_unet.py"), script("02_predict_microbleeds.py"), script("03_fine_tune_unet.py")


def run(*args, ok=True):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert (r.returncode == 0) == ok, r.stdout[-3000:] + r.stderr[-3000:]
    return r


def weights(model_dir):
    return torch.load(os.path.join(model_dir, "final", "model.pt"), map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Each script and its cmbx command once on the same phantoms (10 with microbleed masks, 2 new without)."""
    t = tmp_path_factory.mktemp("pyex")
    train = os.path.join(write_phantom_preprocessed(str(t / "train"), 10, 0), "derivatives", "preprocessed")
    new = os.path.join(write_phantom_preprocessed(str(t / "new"), 2, 7, labels=False), "derivatives", "preprocessed")
    tiny = str(t / "tiny.yaml")
    yaml.safe_dump(TINY, open(tiny, "w"))
    d = {k: str(t / k) for k in ("py", "cli", "ft_py", "ft_cli", "pred_py", "pred_cli", "work")}
    common = ["--device", "cpu", "--workers", "2"]
    run(os.path.join(EXAMPLES, "01_train_unet.py"), "-i", train, "-m", d["py"], "--config", tiny, "--epochs", "2", *common)
    run("-m", "cmbx", "train", "-i", train, "-m", d["cli"], "--config", tiny, "-ep", "2", "--frst_from", "cmbx",
        "--work_dir", d["work"], *common)
    run(os.path.join(EXAMPLES, "03_fine_tune_unet.py"), "-i", train, "-m", d["py"], "-o", d["ft_py"], "--epochs", "2", *common)
    run("-m", "cmbx", "fine_tune", "-i", train, "-m", d["py"], "-o", d["ft_cli"], "--config", tiny, "-ep", "2",
        "--frst_from", "cmbx", "--work_dir", d["work"], *common)
    run(os.path.join(EXAMPLES, "02_predict_microbleeds.py"), "-i", new, "-m", d["py"], "-o", d["pred_py"], *common)
    run("-m", "cmbx", "evaluate", "-i", new, "-m", d["py"], "-o", d["pred_cli"], "--device", "cpu")
    return dict(d, train=train, new=new, tiny=tiny, tmp=t)


# ---------------------------------------------------------------- the scripts against the cmbx commands
def test_train_script_gives_the_weights_of_cmbx_train(runs):
    a, b = weights(runs["py"]), weights(runs["cli"])
    assert sorted(a["state_dict"]) == sorted(b["state_dict"])
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k].cpu()) for k in a["state_dict"])
    for key in ("threshold", "min_size", "held_out"):
        assert a["meta"][key] == b["meta"][key], key


def test_fine_tune_script_gives_the_weights_of_cmbx_fine_tune(runs):
    a, b, start = weights(runs["ft_py"]), weights(runs["ft_cli"]), weights(runs["py"])
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k].cpu()) for k in a["state_dict"])
    assert any(not torch.equal(a["state_dict"][k], start["state_dict"][k]) for k in a["state_dict"])   # it did train
    cfg = yaml.safe_load(open(os.path.join(runs["ft_py"], "config.yaml")))
    assert cfg["train"]["lr"] == 1e-4 and cfg["train"]["init_from"] == os.path.abspath(runs["py"])
    assert cfg["model"]["base_channels"] == 8 and cfg["train"]["patch_size"] == [16, 32, 32]   # kept from the start model


def test_predict_script_gives_the_masks_of_cmbx_evaluate(runs):
    names = sorted(r["id"] for r in collect(runs["new"], log=QUIET))
    for n in names:
        a = nib.load(os.path.join(runs["pred_py"], f"{n}_cmb.nii.gz"))
        b = nib.load(os.path.join(runs["pred_cli"], "final_predictions", f"predicted_final_cmbx_{n}.nii.gz"))
        scan = nib.load(os.path.join(runs["new"], n, f"{n}_t2star.nii.gz"))
        assert np.array_equal(np.asarray(a.dataobj), np.asarray(b.dataobj)) and np.allclose(a.affine, b.affine)
        assert a.shape == scan.shape and np.allclose(a.affine, scan.affine)                  # on the scan's own grid
        lesions = np.asarray(nib.load(os.path.join(runs["pred_py"], f"{n}_lesions.nii.gz")).dataobj)
        assert np.array_equal(lesions > 0, np.asarray(a.dataobj) > 0)
    table = list(csv.DictReader(open(os.path.join(runs["pred_py"], "predictions.csv"))))
    listed = list(csv.DictReader(open(os.path.join(runs["pred_py"], "microbleeds.csv"))))
    assert [r["case"] for r in table] == names
    for r in table:
        assert int(r["microbleeds"]) == sum(x["case"] == r["case"] for x in listed)
        assert int(r["microbleeds"]) == json.load(open(os.path.join(runs["pred_cli"], "details", f"{r['case']}.json")))["n_lesions"]


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "models", "valdo_t2", "final", "model.pt")),
                    reason="pretrained model not downloaded (./install.sh fetches it)")
def test_predict_script_with_the_pretrained_model_finds_microbleeds_as_cmbx_evaluate(runs):
    out_py, out_cli = str(runs["tmp"] / "pre_py"), str(runs["tmp"] / "pre_cli")
    run(os.path.join(EXAMPLES, "02_predict_microbleeds.py"), "-i", runs["new"], "-m", "pretrained", "-o", out_py,
        "--device", "cpu", "--workers", "2")
    run("-m", "cmbx", "evaluate", "-i", runs["new"], "-m", "pre", "-o", out_cli, "--device", "cpu")
    found = 0
    for r in csv.DictReader(open(os.path.join(out_py, "predictions.csv"))):
        a = np.asarray(nib.load(os.path.join(out_py, f"{r['case']}_cmb.nii.gz")).dataobj)
        b = np.asarray(nib.load(os.path.join(out_cli, "final_predictions", f"predicted_final_cmbx_{r['case']}.nii.gz")).dataobj)
        assert np.array_equal(a, b)
        found += int(r["microbleeds"])
    assert found > 0                                     # a comparison of two empty masks would prove nothing


def test_scripts_stop_before_overwriting_unless_force(runs):
    r = run(os.path.join(EXAMPLES, "01_train_unet.py"), "-i", runs["train"], "-m", runs["py"], ok=False)
    assert "--force true" in r.stderr
    r = run(os.path.join(EXAMPLES, "02_predict_microbleeds.py"), "-i", runs["new"], "-m", runs["py"], "-o",
            runs["pred_py"], ok=False)
    assert "--force true" in r.stderr
    r = run(os.path.join(EXAMPLES, "03_fine_tune_unet.py"), "-i", runs["train"], "-m", runs["py"], "-o", runs["ft_py"],
            ok=False)
    assert "--force true" in r.stderr


def test_fine_tune_levels_train_only_those_levels(runs):
    out = str(runs["tmp"] / "ft_level1")
    run(os.path.join(EXAMPLES, "03_fine_tune_unet.py"), "-i", runs["train"], "-m", runs["py"], "-o", out, "--epochs", "1",
        "--levels", "1", "--device", "cpu", "--workers", "2")
    a, start = weights(out)["state_dict"], weights(runs["py"])["state_dict"]
    changed = {k.split(".")[1] for k in a if not torch.equal(a[k], start[k])}
    assert changed and changed <= {"head", "dec0", "up0"}, changed


# ---------------------------------------------------------------- the functions of the scripts
def test_the_copied_functions_are_identical():
    for name in ("prepare_grid", "train_one_epoch", "choose_threshold", "save_model"):
        assert inspect.getsource(getattr(TRAIN, name)) == inspect.getsource(getattr(FINE_TUNE, name)), name
    assert inspect.getsource(TRAIN.prepare_grid) == inspect.getsource(PREDICT.prepare_grid)


def test_prepare_grid_keeps_finished_cases_and_redoes_them_when_the_settings_change(tmp_path):
    prep = os.path.join(write_phantom_preprocessed(str(tmp_path / "c"), 2, 0), "derivatives", "preprocessed")
    cases = collect(prep, need_labels=True, log=QUIET)
    settings = load_config(None, TINY)["preprocess"]
    grid = str(tmp_path / "grid")
    TRAIN.prepare_grid(cases, grid, settings, workers=1)
    image = os.path.join(grid, "phantom-00", "phantom-00_image.nii.gz")
    first = os.path.getmtime(image)
    TRAIN.prepare_grid(cases, grid, settings, workers=1)
    assert os.path.getmtime(image) == first                                     # kept
    TRAIN.prepare_grid(cases, grid, dict(settings, spacing=[1.0, 1.0, 2.0]), workers=1)
    assert nib.load(image).header.get_zooms()[:3] == (1.0, 1.0, 2.0)          # settings changed: prepared again
    broken = [dict(cases[0], id="broken", brain_mask=str(tmp_path / "missing.nii.gz"))]
    with pytest.raises(SystemExit, match="broken"):
        TRAIN.prepare_grid(broken, grid, settings, workers=1)


def test_train_one_epoch_returns_the_loss_and_changes_the_weights(runs):
    cfg = load_config(os.path.join(runs["py"], "config.yaml"))
    T = cfg["train"]
    grid = os.path.join(runs["py"], "grid")
    data = {n: load_prepared(grid, n, T["channels"]) for n in ("phantom-01", "phantom-02")}
    torch.manual_seed(0)
    net = build_model("aniso_unet3d", 2, 8)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    loss = TRAIN.train_one_epoch(net, data, list(data), np.random.RandomState(0), T, opt, "cpu")
    assert np.isfinite(loss) and 0 < loss < 2
    assert any(not torch.equal(before[k], v) for k, v in net.state_dict().items())


class Oracle(torch.nn.Module):
    """Logits from the image channel: high where the image is 1 (the microbleeds of `oracle_case`), low elsewhere."""

    def forward(self, x):
        return 40.0 * (x[:, :1] - 0.6)


def oracle_case(name="o"):
    label = np.zeros((40, 64, 64), np.uint8)
    for z, y, x in ((10, 20, 20), (25, 40, 30), (30, 15, 50)):
        label[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2] = 1
    image = np.where(label > 0, 1.0, 0.2).astype(np.float32)
    lab, n = ndimage.label(label, structure=np.ones((3, 3, 3), bool))
    return PreparedCase(name, image, image.copy(), label, np.diag([0.5, 0.5, 1.0, 1.0]),
                        components=[np.argwhere(lab == k) for k in range(1, n + 1)])


def test_choose_threshold_with_a_perfect_network_scores_f1_one():
    cfg = load_config(None, TINY)
    threshold, min_size, scores = TRAIN.choose_threshold(Oracle(), {"o": oracle_case()}, ["o"], cfg, "cpu")
    assert scores["f1"] == 1.0 and scores["tp"] == 3 and scores["fp"] == 0
    assert threshold in cfg["selection"]["thresholds"] and min_size in cfg["selection"]["min_sizes"]


def test_find_microbleeds_with_a_perfect_network_finds_each_one():
    cfg = load_config(None, TINY)
    prob, labels, n = PREDICT.find_microbleeds(Oracle(), oracle_case(), cfg, 0.5, 1)
    assert n == 3 and prob.shape == labels.shape == (40, 64, 64)
    assert np.array_equal(labels > 0, oracle_case().label > 0)
    _, _, n = PREDICT.find_microbleeds(Oracle(), oracle_case(), cfg, 0.5, 28)     # each is 27 voxels: all too small
    assert n == 0


def test_write_case_puts_every_microbleed_back_on_the_scan(runs, tmp_path):
    """The reference mask of a prepared training case, written back as if predicted: the same microbleeds on the
    scan's grid as in the reference mask there."""
    name = "phantom-09"
    case = load_prepared(os.path.join(runs["py"], "grid"), name, ["image", "frst"])
    labels, n = ndimage.label(case.label, structure=np.ones((3, 3, 3), bool))
    rows = PREDICT.write_case(name, os.path.join(runs["train"], name, f"{name}_t2star.nii.gz"), labels,
                              case.label.astype(np.float32), case.affine, str(tmp_path))
    native = np.asarray(nib.load(os.path.join(tmp_path, f"{name}_lesions.nii.gz")).dataobj)
    reference = np.asarray(nib.load(os.path.join(runs["train"], name, f"{name}_cmb.nii.gz")).dataobj) > 0
    assert n > 0 and len(rows) == n == len(set(np.unique(native).tolist()) - {0})
    assert ndimage.label(reference, structure=np.ones((3, 3, 3), bool))[1] == n
    assert (native > 0)[reference].mean() > 0.5                                 # mostly on the reference voxels
    assert all(r["max_prob"] == 1.0 and r["volume_mm3"] > 0 for r in rows)


def test_load_model_reads_what_save_model_wrote(tmp_path):
    cfg = load_config(None, TINY)
    torch.manual_seed(1)
    net = build_model("aniso_unet3d", 2, 8)
    TRAIN.save_model(net, cfg, str(tmp_path / "m"), dict(ok=True, threshold=0.7, min_size=3), [dict(epoch=1, loss=0.5)])
    back, cfg2, threshold, min_size = PREDICT.load_model(str(tmp_path / "m"))
    assert (threshold, min_size) == (0.7, 3) and cfg2["model"]["base_channels"] == 8
    assert all(torch.equal(v, back.state_dict()[k]) for k, v in net.state_dict().items())
    with pytest.raises(SystemExit, match="not found"):
        PREDICT.load_model(str(tmp_path / "nothing"))
