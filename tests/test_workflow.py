"""MicrobleedNet-style commands (cmbx preprocess / train / fine_tune / evaluate / cross_validate): the folder layouts
(cohort with derivatives/preprocessed/<case>/, flat folder, MicrobleedNet preprocess output), MicrobleedNet's case
naming, every option reaching the config, and the commands end to end on the CPU with tiny settings."""
import argparse
import glob
import os

import nibabel as nib
import numpy as np
import pytest

from cmbx.cli import main
from cmbx.synthetic import write_phantom_preprocessed
from cmbx.workflow import build_config, collect, raw_images, str2bool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "configs", "smoke_cpu.yaml")
QUIET = lambda *a: None


def test_str2bool():
    assert str2bool("True") and str2bool("1") and not str2bool("False") and not str2bool("no")
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool("maybe")


def test_collect_cohort_and_preprocessed_folder(tmp_path):
    root = write_phantom_preprocessed(str(tmp_path / "c"), 3, 0)
    a = collect(root, need_labels=True, log=QUIET)
    b = collect(os.path.join(root, "derivatives", "preprocessed"), need_labels=True, log=QUIET)
    assert [r["id"] for r in a] == [r["id"] for r in b] == ["phantom-00", "phantom-01", "phantom-02"]
    assert all(r["image"].endswith("_t2star.nii.gz") and r["brain_mask"] and r["label"] and r["tissue_map"] and r["flair"]
               for r in a)


def test_collect_flat_folder_and_missing_parts(tmp_path):
    root = write_phantom_preprocessed(str(tmp_path / "c"), 2, 0, flair=False)
    flat = tmp_path / "flat"
    flat.mkdir()
    for f in glob.glob(os.path.join(root, "derivatives", "preprocessed", "*", "*.nii.gz")):
        os.rename(f, flat / os.path.basename(f))
    assert len(collect(str(flat), need_labels=True, log=QUIET)) == 2
    with pytest.raises(SystemExit, match="flair"):
        collect(str(flat), need_flair=True, log=QUIET)
    os.remove(flat / "phantom-01_brainmask.nii.gz")
    with pytest.raises(SystemExit, match="brainmask"):
        collect(str(flat), log=QUIET)


def _mbnet_folder(tmp, with_fsl=True):
    """What `microbleednet preprocess` writes: fsl_preprocessed/images (BET+FAST), images (1 - I/max in the brain),
    labels."""
    root = write_phantom_preprocessed(str(tmp / "src"), 2, 0)
    out = tmp / "mb"
    for n in ("phantom-00", "phantom-01"):
        d = os.path.join(root, "derivatives", "preprocessed", n)
        im, br = nib.load(f"{d}/{n}_t2star.nii.gz"), np.asarray(nib.load(f"{d}/{n}_brainmask.nii.gz").dataobj) > 0
        arr = np.asarray(im.dataobj, np.float32) * br
        for sub, data in (("fsl_preprocessed/images", arr), ("images", (1 - arr / arr.max()) * br),
                          ("labels", np.asarray(nib.load(f"{d}/{n}_cmb.nii.gz").dataobj))):
            if sub.startswith("fsl") and not with_fsl:
                continue
            os.makedirs(out / sub, exist_ok=True)
            suffix = "_mask" if sub == "labels" else "_preproc"
            nib.save(nib.Nifti1Image(data.astype(np.float32), im.affine), out / sub / f"{n}{suffix}.nii.gz")
    return str(out), root


def test_collect_microbleednet_folder(tmp_path):
    mb, root = _mbnet_folder(tmp_path)
    rows = collect(mb, need_labels=True, log=QUIET)
    assert [r["id"] for r in rows] == ["phantom-00", "phantom-01"]
    assert "fsl_preprocessed" in rows[0]["image"] and rows[0]["label"].endswith("_mask.nii.gz")
    inp = collect(mb, need_labels=True, inpaint=True, log=QUIET)      # inverted MicrobleedNet image -> I / max again
    a = np.asarray(nib.load(inp[0]["image"]).dataobj)
    f = np.asarray(nib.load(rows[0]["image"]).dataobj)
    assert np.allclose(a, f / f.max(), atol=1e-5)
    mb2, _ = _mbnet_folder(tmp_path / "nofsl", with_fsl=False)
    with pytest.raises(SystemExit, match="-f True"):
        collect(mb2, log=QUIET)


def test_microbleednet_case_names(tmp_path):
    (tmp_path / "raw" / "site1").mkdir(parents=True)
    for n in ("a01", "a02"):
        (tmp_path / "raw" / "site1" / f"{n}_T2S.nii.gz").write_bytes(b"")
    names = [x[0] for x in raw_images(str(tmp_path / "raw"), "_T2S", True)]
    assert names == ["a01", "a02"]                                         # -f True: the name before the ending
    assert [x[0] for x in raw_images(str(tmp_path / "raw"), "_T2S", False)] == ["site1-a01", "site1-a02"]   # as MicrobleedNet


def test_options_reach_the_config(tmp_path):
    a = argparse.Namespace(config=None, set=["train.weight_decay=0.001"], num_epochs=7, init_learng_rate=0.002,
                           batch_size=2, data_augmentation=False, aug_factor=3, batch_factor=5, optimizer="sgd", epsilon=None,
                           momentum=0.7, lr_sch_mlstone=[3, 5], lr_sch_gamma=0.2, train_prop=0.75, early_stop_val=4,
                           save_full_model=True, cp_save_type="best", cp_everyn_N=2, save_resume_training=True, multi_gpu=False,
                           ft_layers=[1, 2], loss="dice", seed=3, device="cpu", patches=16, patch_size=24, frst=False,
                           csf_rule=True, cand_discrimination=True, disc_model="teacher_student", kd_temperature=2.0,
                           kd_alpha=0.5, flair=True)
    cfg = build_config(a, str(tmp_path / "m"), str(tmp_path / "m" / "cases.csv"), str(tmp_path / "g"))
    T = cfg["train"]
    assert (T["epochs"], T["lr"], T["batch_size"], T["flip_augment"], T["aug_factor"], T["subjects_per_epoch"]) == (7, 0.002, 2, False, 3, 5)
    assert (T["optimizer"], T["momentum"], T["scheduler"], T["lr_milestones"], T["lr_gamma"]) == ("sgd", 0.7, "multistep", [3, 5], 0.2)
    assert (T["train_prop"], T["early_stop"], T["save_full_model"], T["checkpoint"], T["checkpoint_every"], T["resume"]) == (0.75, 4, True, "best", 2, True)
    assert (T["finetune_layers"], T["loss"], T["seed"], T["n_patches"], T["patch_size"]) == ([1, 2], "dice", 3, 16, [24, 24, 24])
    assert T["channels"] == ["image", "flair"] and cfg["preprocess"]["extra_channels"] == {"flair": "flair"}
    assert T["weight_decay"] == 0.001 and cfg["postprocess"]["csf_rule"] is True
    assert cfg["stage2"]["enabled"] and cfg["stage2"]["method"] == "teacher_student" and cfg["stage2"]["kd_alpha"] == 0.5


def test_preprocess_already_corrected(tmp_path):
    """cmbx preprocess -f True: flat raw folder with scans, masks, brain masks, FLAIR -> one folder per case + files.xlsx."""
    src = write_phantom_preprocessed(str(tmp_path / "src"), 2, 0)
    raw = tmp_path / "raw"
    raw.mkdir()
    for n in ("phantom-00", "phantom-01"):
        d = os.path.join(src, "derivatives", "preprocessed", n)
        for a, b in (("_t2star", "_T2S"), ("_cmb", "_cmbmask"), ("_brainmask", "_brain"), ("_flair", "_FLAIR")):
            os.link(f"{d}/{n}{a}.nii.gz", raw / f"{n}{b}.nii.gz")
    out = tmp_path / "cohort" / "derivatives" / "preprocessed"
    main(["preprocess", "-i", str(raw), "-ir", "_T2S", "-o", str(out), "-f", "True", "-l", str(raw), "-lr", "_cmbmask.nii.gz",
          "-b", str(raw), "-br", "_brain.nii.gz", "-flr", "_FLAIR", "--flair_registered", "True", "-mod", "swi"])
    for n in ("phantom-00", "phantom-01"):
        assert sorted(os.listdir(out / n)) == [f"{n}_brainmask.nii.gz", f"{n}_cmb.nii.gz", f"{n}_flair.nii.gz", f"{n}_swi.nii.gz"]
    import pandas as pd
    t = pd.read_excel(tmp_path / "cohort" / "files.xlsx")
    assert list(t["case"]) == ["phantom-00", "phantom-01"] and t["cmb"].str.startswith("derivatives/preprocessed/").all()
    assert t["scan"].str.endswith("_swi.nii.gz").all()


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wf")
    return tmp, write_phantom_preprocessed(str(tmp / "train"), 10, 0), write_phantom_preprocessed(str(tmp / "new"), 2, 7, labels=False)


def test_train_evaluate_fine_tune(cohort):
    tmp, train, new = cohort
    m = str(tmp / "m")
    main(["train", "-i", train, "-l", train, "-m", m, "--config", CFG, "-ep", "2", "-cp_type", "everyN", "-cp_n", "1",
          "-sv_mod", "True", "--flair", "True"])
    assert {"model.pt", "model_best.pt", "model_epoch1.pt", "model_full.pt"} <= set(os.listdir(os.path.join(m, "final")))
    assert os.path.exists(os.path.join(train, "derivatives", "cmbx"))          # grid volumes next to preprocessed/
    out = str(tmp / "pred")
    main(["evaluate", "-i", new, "-m", m, "-o", out, "-int", "True", "-cp_type", "best"])
    for d, pre in (("final_predictions", "predicted_final_cmbx_"), ("cdet_predictions", "predicted_cdet_cmbx_"),
                   ("input_images", "input_cmbx_")):
        assert sorted(os.listdir(os.path.join(out, d))) == [f"{pre}phantom-0{i}.nii.gz" for i in (0, 1)]
    ref = nib.load(os.path.join(new, "derivatives", "preprocessed", "phantom-00", "phantom-00_t2star.nii.gz"))
    assert nib.load(os.path.join(out, "final_predictions", "predicted_final_cmbx_phantom-00.nii.gz")).shape == ref.shape
    with pytest.raises(SystemExit, match="other settings"):                  # same folder, other settings
        main(["train", "-i", train, "-l", train, "-m", m, "--config", CFG, "-ep", "3", "--flair", "True"])
    ft = str(tmp / "ft")
    main(["fine_tune", "-i", train, "-l", train, "-m", m, "-o", ft, "--config", CFG, "-ep", "1", "-ftlayers", "1"])
    import yaml
    c = yaml.safe_load(open(os.path.join(ft, "config.yaml")))
    assert c["train"]["lr"] == 0.0001 and c["train"]["init_from"] == m and c["train"]["channels"] == ["image", "frst", "flair"]


def test_cross_validate(cohort):
    tmp, train, _ = cohort
    out = str(tmp / "cv")
    main(["cross_validate", "-i", train, "-l", train, "-o", out, "-fold", "3", "--config", CFG, "-ep", "1", "--frst", "False"])
    assert os.path.exists(os.path.join(out, "summary.json")) and len(glob.glob(os.path.join(out, "fold*", "model.pt"))) == 3


def test_inpaint_from_images_inpainted(tmp_path):
    """--inpaint True on a cohort that carries MicrobleedNet's inpainted input in derivatives/images_inpainted."""
    root = write_phantom_preprocessed(str(tmp_path / "c"), 2, 0, flair=False)
    os.makedirs(os.path.join(root, "derivatives", "images_inpainted"))
    for n in ("phantom-00", "phantom-01"):
        d = os.path.join(root, "derivatives", "preprocessed", n)
        im, br = nib.load(f"{d}/{n}_t2star.nii.gz"), np.asarray(nib.load(f"{d}/{n}_brainmask.nii.gz").dataobj) > 0
        a = np.asarray(im.dataobj, np.float32) * br
        nib.save(nib.Nifti1Image(((1 - a / a.max()) * br).astype(np.float32), im.affine),
                 os.path.join(root, "derivatives", "images_inpainted", f"{n}_preproc.nii.gz"))
    rows = collect(root, need_labels=True, inpaint=True, log=QUIET)
    got = np.asarray(nib.load(rows[0]["image"]).dataobj)
    d = os.path.join(root, "derivatives", "preprocessed", "phantom-00")
    a = np.asarray(nib.load(f"{d}/phantom-00_t2star.nii.gz").dataobj) * (np.asarray(nib.load(f"{d}/phantom-00_brainmask.nii.gz").dataobj) > 0)
    assert np.allclose(got, a / a.max(), atol=1e-5)


def test_microbleednet_style_command(tmp_path):
    """As MicrobleedNet: -i = the folder with images/ + frsts/, -l = the labels; the network gets the given FRST."""
    root = write_phantom_preprocessed(str(tmp_path / "c"), 10, 0)
    der = os.path.join(root, "derivatives")
    m = str(tmp_path / "m")
    main(["train", "-i", der, "-l", os.path.join(der, "preprocessed"), "-m", m, "--config", CFG, "-ep", "1"])
    import yaml
    c = yaml.safe_load(open(os.path.join(m, "config.yaml")))
    assert c["train"]["channels"] == ["image", "frst_mbnet"] and c["preprocess"]["extra_channels"] == {"frst_mbnet": "frst_file"}
    rows = collect(der, os.path.join(der, "preprocessed"), need_labels=True, log=QUIET)
    assert all(r["frst_file"].endswith("_frst.nii.gz") and r["brain_mask"].endswith("_brainmask.nii.gz")
               and r["label"].endswith("_cmb.nii.gz") for r in rows)
    new = write_phantom_preprocessed(str(tmp_path / "n"), 2, 7, labels=False)
    out = str(tmp_path / "p")
    main(["evaluate", "-i", os.path.join(new, "derivatives"), "-m", m, "-o", out])
    assert len(os.listdir(os.path.join(out, "final_predictions"))) == 2
    m2 = str(tmp_path / "m2")                              # the cmbx FRST instead
    main(["train", "-i", der, "-l", os.path.join(der, "preprocessed"), "-m", m2, "--config", CFG, "-ep", "1", "--frst_from", "cmbx"])
    assert yaml.safe_load(open(os.path.join(m2, "config.yaml")))["train"]["channels"] == ["image", "frst"]


def test_split_layout_inpainted(tmp_path):
    """images/ = without inpainting, images_inpainted/ = with: --inpaint picks the folder and its FRSTs."""
    root = write_phantom_preprocessed(str(tmp_path / "c"), 2, 0)
    der = os.path.join(root, "derivatives")
    import shutil
    shutil.copytree(os.path.join(der, "images"), os.path.join(der, "images_inpainted"))
    shutil.copytree(os.path.join(der, "frsts"), os.path.join(der, "frsts_inpainted"))
    a = collect(der, os.path.join(der, "preprocessed"), inpaint=False, log=QUIET)
    b = collect(der, os.path.join(der, "preprocessed"), inpaint=True, log=QUIET)
    assert "/frsts/" in a[0]["frst_file"] and "/frsts_inpainted/" in b[0]["frst_file"]
    assert a[0]["image"].endswith("_images.nii.gz") and b[0]["image"].endswith("_images_inpainted.nii.gz")
