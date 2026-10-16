"""MicrobleedNet-style commands: the sub-commands, option names and folder use of ``microbleednet`` (Sundaresan et al.
2023, doi:10.3389/fninf.2023.1204186; https://github.com/v-sundaresan/microbleed-detection, commit 958f1cb) with the
3D U-Net of this package. A script written for MicrobleedNet runs with ``cmbx`` in place of ``microbleednet``.

    cmbx preprocess     -i <raw scans> -ir <ending> -o <preprocessed> [-l <labels> -lr <ending> ...] [-f True]
    cmbx train          -i <preprocessed> -l <preprocessed> -m <model dir>
    cmbx fine_tune      -i <preprocessed> -l <preprocessed> -m <model dir | pre> -o <new model dir>
    cmbx evaluate       -i <preprocessed> -m <model dir | pre> -o <output dir>
    cmbx cross_validate -i <preprocessed> -l <preprocessed> -o <output dir> [-fold 5]

The data of a cohort (README "Data"); ``-i`` takes the cohort folder or its ``derivatives/preprocessed``:

    <cohort>/derivatives/preprocessed/<name>/<name>_t2star.nii.gz      bias-corrected scan (or _swi; _image also works)
                                             <name>_brainmask.nii.gz   brain mask
                                             <name>_cmb.nii.gz         reference microbleed mask (training; _label too)
                                             <name>_synthseg.nii.gz    optional SynthSeg labels (--csf_rule True)
                                             <name>_flair.nii.gz       optional FLAIR registered to the scan (--flair True)
                                             <name>_inpainted.nii.gz   optional vessel-inpainted scan (--inpaint True)

A folder written by ``microbleednet preprocess`` is read as well: ``fsl_preprocessed/images/<name>_preproc`` (BET +
FAST) is the image, ``images/<name>_preproc`` (vessels inpainted, intensity inverted) is inverted back for
``--inpaint True``, ``labels/<name>_mask`` are the labels, the brain mask is image > 0 with holes filled (the scans are
skull-stripped; MicrobleedNet's own rule). Derived files and the grid volumes of training go to ``cmbx/`` next to the
preprocessed folder (``derivatives/cmbx``; ``--work_dir`` elsewhere), shared by all experiments on the same data.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S26 = np.ones((3, 3, 3), bool)

# MicrobleedNet options without a counterpart here (accepted so that existing scripts run; a note says so)
NO_EFFECT = {"verbose": "-v", "progress_bar": "-pbar", "cp_load_type_ft": "-cpld_type", "cpload_everyn_N": "-cpld_n",
             "num_classes": "-nclass", "save_checkpoint": "-sv"}


def str2bool(v) -> bool:
    """MicrobleedNet declares its switches as ``type=bool`` (any text, even "False", is True); here True/False are read."""
    if isinstance(v, bool):
        return v
    t = str(v).strip().lower()
    if t in ("true", "t", "yes", "y", "1"):
        return True
    if t in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected True or False, got {v!r}")


# ---------------------------------------------------------------- reading a preprocessed folder
def _nii(folder: Optional[str], name: str, suffix: str) -> Optional[str]:
    if not folder:
        return None
    for ext in (".nii.gz", ".nii"):
        p = os.path.join(folder, name + suffix + ext)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def _names(folder: str, suffix: str) -> List[str]:
    rx = re.compile("^(.+)" + re.escape(suffix) + r"\.nii(\.gz)?$")
    return sorted({m.group(1) for m in (rx.match(f) for f in os.listdir(folder)) if m}) if os.path.isdir(folder) else []


def preprocessed_folder(inp: str) -> str:
    """The cohort folder or its derivatives/preprocessed -> the folder that holds the cases."""
    d = os.path.join(inp, "derivatives", "preprocessed")
    return d if os.path.isdir(d) else inp


def default_work(prep: str) -> str:
    prep = os.path.abspath(prep)
    return os.path.join(os.path.dirname(prep), "cmbx") if os.path.basename(prep) == "preprocessed" else os.path.join(prep, "cmbx")


SCAN_SUFFIXES = ("_t2star", "_swi", "_image")             # the scan of a case: named by its sequence, or generic
LABEL_SUFFIXES = ("_cmb", "_label", "_mask", "_mask_preproc")


def _scan(folder: str, name: str) -> Optional[str]:
    for suf in SCAN_SUFFIXES:
        p = _nii(folder, name, suf)
        if p:
            return p
    return None


def case_folders(prep: str) -> Dict[str, str]:
    """name -> folder: one subfolder per case (<name>/<name>_t2star.nii.gz), or all files in the folder itself."""
    out = {n: prep for suf in SCAN_SUFFIXES for n in _names(prep, suf)}
    if os.path.isdir(prep):
        for d in sorted(os.listdir(prep)):
            if _scan(os.path.join(prep, d), d):
                out[d] = os.path.join(prep, d)
    return dict(sorted(out.items()))


def is_microbleednet_folder(folder: str) -> bool:
    return bool(_names(os.path.join(folder, "images"), "_preproc") or
                _names(os.path.join(folder, "fsl_preprocessed", "images"), "_preproc"))


def _label(name: str, inp: str, label_dir: Optional[str]) -> Optional[str]:
    dirs = [label_dir, os.path.join(label_dir, name)] if label_dir else []
    if label_dir:
        lp = preprocessed_folder(label_dir)
        dirs += [lp, os.path.join(lp, name)]
    for d in [x for x in dirs + [inp, os.path.join(inp, "labels")] if x]:
        for suf in LABEL_SUFFIXES:
            p = _nii(d, name, suf)
            if p:
                return p
    return None


def _write_if_stale(path: str, sources: List[str], make) -> str:
    if not os.path.exists(path) or any(os.path.getmtime(s) > os.path.getmtime(path) for s in sources):
        img = make()
        tmp = path + ".part.nii.gz"
        nib.save(img, tmp)
        os.replace(tmp, path)
    return path


def _brain_from(image_path: str, out: str) -> str:
    def make():
        im = nib.load(image_path)
        m = ndimage.binary_fill_holes(np.asarray(im.dataobj) > 0).astype(np.uint8)
        return nib.Nifti1Image(m, im.affine)
    return _write_if_stale(out, [image_path], make)


def collect(inp: str, label_dir: Optional[str] = None, need_labels: bool = False, inpaint: bool = False,
            need_tissue: bool = False, work: Optional[str] = None, log=print, need_flair: bool = False) -> List[Dict]:
    """Case-list rows (id, image, label, brain_mask, tissue_map, flair) from a cohort / preprocessed folder (cmbx or
    MicrobleedNet)."""
    if not os.path.isdir(inp):
        raise SystemExit(f"-i {inp}: not a folder")
    inp = preprocessed_folder(inp)
    folders = case_folders(inp)
    mbnet = is_microbleednet_folder(inp) and not folders
    work = work or default_work(inp)
    rows, missing = [], []
    if mbnet:
        fsl_dir, inv_dir = os.path.join(inp, "fsl_preprocessed", "images"), os.path.join(inp, "images")
        names = sorted(set(_names(fsl_dir, "_preproc")) | set(_names(inv_dir, "_preproc")))
        inputs = os.path.join(work, "inputs")
        os.makedirs(inputs, exist_ok=True)
        log(f"{inp}: MicrobleedNet folder, {len(names)} case(s)")
    else:
        names = list(folders)
        if not names:
            raise SystemExit(f"{inp}: no <name>/<name>_t2star|_swi|_image.nii.gz (cmbx) and no images/<name>_preproc.nii.gz "
                             "(MicrobleedNet folder)")
    for n in names:
        cdir = folders.get(n, inp)
        label = _label(n, cdir, label_dir)
        tissue = _nii(cdir, n, "_synthseg") or _nii(os.path.join(inp, "synthseg"), n, "_synthseg")
        flair = _nii(cdir, n, "_flair")
        if mbnet:
            fsl, inv = _nii(fsl_dir, n, "_preproc"), _nii(inv_dir, n, "_preproc")
            src = inv if inpaint else fsl
            if not src:
                missing.append((n, "images/<name>_preproc (MicrobleedNet's inpainted image)" if inpaint else
                                "fsl_preprocessed/images/<name>_preproc: MicrobleedNet keeps the image before vessel "
                                "inpainting only without -f True; run `cmbx preprocess -i <your scans> -ir <ending> -f True "
                                "-o <new folder>` and use that folder"))
                continue
            brain = _brain_from(fsl or inv, os.path.join(inputs, f"{n}_brainmask.nii.gz"))
            image = src
            if inpaint:                                    # MicrobleedNet stores 1 - I / max(I) in the brain: invert back
                def make(inv=inv, brain=brain):
                    im = nib.load(inv)
                    b = np.asarray(nib.load(brain).dataobj) > 0
                    return nib.Nifti1Image(((1.0 - np.asarray(im.dataobj, np.float32)) * b).astype(np.float32), im.affine)
                image = _write_if_stale(os.path.join(inputs, f"{n}_inpainted.nii.gz"), [inv, brain], make)
        else:
            image = _nii(cdir, n, "_inpainted") if inpaint else _scan(cdir, n)
            brain = _nii(cdir, n, "_brainmask")
            if not image:
                missing.append((n, f"{n}_inpainted.nii.gz (run cmbx preprocess --inpaint True)" if inpaint else f"{n}_t2star"))
                continue
            if not brain:
                missing.append((n, f"{n}_brainmask.nii.gz"))
                continue
        if need_labels and not label:
            missing.append((n, f"microbleed mask ({n}_cmb.nii.gz in -i or -l)"))
            continue
        if need_tissue and not tissue:
            missing.append((n, f"{n}_synthseg.nii.gz (needed by --csf_rule True; cmbx preprocess --synthseg True)"))
            continue
        if need_flair and not flair:
            missing.append((n, f"{n}_flair.nii.gz (needed by --flair True; use a folder with the FLAIR cases only)"))
            continue
        rows.append(dict(id=n, image=image, label=label or "", brain_mask=brain, tissue_map=tissue or "",
                         inpaint_mask="", group="", flair=flair or ""))
    if missing:
        raise SystemExit(f"{len(missing)} case(s) incomplete in {inp}:\n" + "\n".join(f"  {n}: missing {w}" for n, w in missing))
    return rows


def _lesions(rows: List[Dict]) -> None:
    from .caselist import lesion_class
    for r in rows:
        n = int(ndimage.label(np.asarray(nib.load(r["label"]).dataobj) > 0, S26)[1]) if r.get("label") else 0
        r.update(n_lesions=n, lesion_class=lesion_class(n))


# ---------------------------------------------------------------- options -> config
def _given(a, k) -> bool:
    return getattr(a, k, None) is not None


def build_config(a, run_dir: str, cases: str, prepared: str, log=print) -> Dict:
    from .config import load_config, parse_set
    cfg = load_config(getattr(a, "config", None), parse_set(getattr(a, "set", None) or []))
    cfg["name"], cfg["output_root"] = os.path.basename(os.path.normpath(run_dir)), os.path.dirname(os.path.abspath(run_dir))
    cfg["data"].update(cases=os.path.abspath(cases), prepared_dir=os.path.abspath(prepared),
                       splits=os.path.join(os.path.abspath(run_dir), "splits.json"))
    T, st2 = cfg["train"], cfg["stage2"]
    m = dict(num_epochs="epochs", init_learng_rate="lr", batch_size="batch_size", data_augmentation="flip_augment",
             aug_factor="aug_factor", batch_factor="subjects_per_epoch", optimizer="optimizer", epsilon="epsilon",
             momentum="momentum", lr_sch_gamma="lr_gamma", train_prop="train_prop", early_stop_val="early_stop",
             save_full_model="save_full_model", cp_save_type="checkpoint", cp_everyn_N="checkpoint_every",
             save_resume_training="resume", multi_gpu="multi_gpu", ft_layers="finetune_layers", loss="loss",
             seed="seed", device="device", patches="n_patches")
    for opt, key in m.items():
        if _given(a, opt):
            T[key] = getattr(a, opt)
    if _given(a, "num_epochs"):
        T["min_epochs"] = min(int(T["min_epochs"]), int(a.num_epochs))
        T["eval_every"] = min(int(T["eval_every"]), int(a.num_epochs))
    if _given(a, "lr_sch_mlstone") or _given(a, "lr_sch_gamma"):
        T["scheduler"] = "multistep"
        if _given(a, "lr_sch_mlstone"):
            T["lr_milestones"] = list(a.lr_sch_mlstone)
    if _given(a, "patch_size"):                            # MicrobleedNet: cubic patches of -psize voxels
        T["patch_size"] = [int(a.patch_size)] * 3
    if _given(a, "frst"):
        rest = [c for c in T["channels"] if c not in ("image", "frst")]
        T["channels"] = ["image"] + (["frst"] if a.frst else []) + rest
    if getattr(a, "flair", None):                          # FLAIR as an extra input channel (case-list column flair)
        cfg["preprocess"]["extra_channels"] = dict(cfg["preprocess"].get("extra_channels") or {}, flair="flair")
        if "flair" not in T["channels"]:
            T["channels"] = T["channels"] + ["flair"]
    if _given(a, "csf_rule"):
        cfg["postprocess"]["csf_rule"] = bool(a.csf_rule)
    if _given(a, "cand_discrimination"):
        st2["enabled"] = bool(a.cand_discrimination)
        st2["method"] = "teacher_student"                  # MicrobleedNet's step 2 unless --disc_model says otherwise
    for opt, key in (("disc_model", "method"), ("kd_temperature", "kd_temperature"), ("kd_alpha", "kd_alpha")):
        if _given(a, opt):
            st2[key] = getattr(a, opt)
    return cfg


def _notes(a, log) -> None:
    given = [flag for k, flag in NO_EFFECT.items() if _given(a, k)]
    if getattr(a, "cand_detection", None) is False:
        given.append("-cand_det False (step 1 is always trained; use fine_tune or evaluate for an existing model)")
    if given:
        log("note: MicrobleedNet option(s) without effect in cmbx: " + ", ".join(given))


def _prepared_dir(work: str, cfg: Dict, inpaint: bool) -> str:
    from .config import preprocess_signature
    h = hashlib.sha1(json.dumps(preprocess_signature(cfg), sort_keys=True).encode()).hexdigest()[:8]
    return os.path.join(work, f"grid_{'inpainted' if inpaint else 'image'}_{h}")


def _workflow_json(run_dir: str, d: Dict) -> None:
    json.dump(dict(d, written=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(run_dir, "workflow.json"), "w"), indent=1)


def _read_workflow(run_dir: str) -> Dict:
    p = os.path.join(run_dir, "workflow.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def pretrained_dir(log=print) -> str:
    """``-m pre`` / ``-p True``: $CMBX_PRETRAINED_MODEL_PATH, else models/valdo_t2 in the repository (downloaded once)."""
    env = os.environ.get("CMBX_PRETRAINED_MODEL_PATH")
    if env:
        return env
    d = os.path.join(REPO, "models", "valdo_t2")
    if not os.path.exists(os.path.join(d, "config.yaml")):
        from .pretrained import get_model
        get_model("valdo_t2", os.path.join(REPO, "models"), log)
    return d


def _model_arg(name: str, pretrained: Optional[bool], log) -> str:
    if pretrained or name in ("pre", "pretrained"):
        return pretrained_dir(log)
    if os.path.isfile(name) and name.endswith(".pt"):     # .../final/model.pt -> its run folder
        return os.path.dirname(os.path.dirname(os.path.abspath(name)))
    if not os.path.exists(os.path.join(name, "config.yaml")):
        raise SystemExit(f"-m {name}: not a cmbx model folder (no config.yaml); give a folder written by cmbx train, or pre")
    return name


def _same_model(run_dir: str, cfg: Dict) -> Optional[bool]:
    """None: no finished model; True/False: finished with the same / another config."""
    import yaml
    from .config import DEFAULTS, _fill_defaults
    meta = os.path.join(run_dir, "final", "meta.json")
    if not (os.path.exists(meta) and json.load(open(meta)).get("ok") and os.path.exists(os.path.join(run_dir, "config.yaml"))):
        return None
    old = _fill_defaults(DEFAULTS, yaml.safe_load(open(os.path.join(run_dir, "config.yaml"))))
    new = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return json.dumps(old, sort_keys=True, default=str) == json.dumps(new, sort_keys=True, default=str)


# ---------------------------------------------------------------- train / fine_tune
def train(a, log=print, fine_tune: bool = False) -> str:
    from . import config as C
    from .data import write_caselist
    from .experiments import check_config, ensure_prepared
    from .final import train_final
    _notes(a, log)
    init = _model_arg(a.model_name, getattr(a, "pretrained_model", None), log) if fine_tune else None
    run_dir = a.output_dir if fine_tune else a.model_dir
    base = _read_workflow(init) if init else {}
    inpaint = bool(a.inpaint) if _given(a, "inpaint") else bool(base.get("inpaint", False))
    if fine_tune and not _given(a, "flair"):
        a.flair = "flair" in (base.get("channels") or [])
    if fine_tune and not _given(a, "csf_rule"):
        a.csf_rule = bool(base.get("csf_rule", False))
    work = a.work_dir or default_work(preprocessed_folder(a.inp_dir))
    cases = os.path.join(run_dir, "cases.csv")
    cfg = build_config(a, run_dir, cases, os.path.join(work, "grid"), log)
    csf = bool(cfg["postprocess"]["csf_rule"])
    rows = collect(a.inp_dir, a.label_dir, need_labels=True, inpaint=inpaint, need_tissue=csf, work=work, log=log,
                   need_flair="flair" in cfg["train"]["channels"])
    os.makedirs(run_dir, exist_ok=True)
    write_caselist(rows, cases)
    cfg["data"]["prepared_dir"] = _prepared_dir(work, cfg, inpaint)
    T = cfg["train"]
    if not _given(a, "train_prop") and "train_prop" not in str(getattr(a, "set", "")):
        T["train_prop"] = 0.8                              # MicrobleedNet's default: 20 % of the cases held out
    if fine_tune:
        import yaml
        T["init_from"] = init
        old = yaml.safe_load(open(os.path.join(init, "config.yaml")))
        if not _given(a, "frst"):                          # keep the channels of the model being fine-tuned
            T["channels"] = old["train"]["channels"]
        if not _given(a, "init_learng_rate"):
            T["lr"] = 1e-4                                 # MicrobleedNet fine_tune default
    check_config(cfg)
    same = _same_model(run_dir, cfg)
    if same is False and not a.overwrite:
        raise SystemExit(f"{run_dir} holds a model trained with other settings; choose another folder or add --overwrite True")
    C.dump(cfg, os.path.join(run_dir, "config.yaml"))
    _workflow_json(run_dir, dict(command="fine_tune" if fine_tune else "train", inp_dir=os.path.abspath(a.inp_dir),
                                 inpaint=inpaint, csf_rule=csf, init=init, cases=len(rows), channels=T["channels"]))
    ensure_prepared(cfg, a.workers, log=log)
    train_final(cfg, run_dir, log, overwrite=bool(a.overwrite) or same is not True)
    if cfg["stage2"].get("enabled"):
        from .stage2 import train_stage2_final
        train_stage2_final(cfg, run_dir, T["device"], log)
    log(f"done: {run_dir}/final/model.pt -- predict with  cmbx evaluate -i <preprocessed> -m {run_dir} -o <output>")
    return run_dir


# ---------------------------------------------------------------- evaluate
def evaluate(a, log=print) -> str:
    from .data import write_caselist
    from .predict_cli import predict_run
    _notes(a, log)
    run = _model_arg(a.model_name, getattr(a, "pretrained_model", None), log)
    wf = _read_workflow(run)
    inpaint = bool(wf.get("inpaint", False))
    csf = bool(a.csf_rule) if _given(a, "csf_rule") else bool(wf.get("csf_rule", False))
    out = a.output_dir
    os.makedirs(out, exist_ok=True)
    rows = collect(a.inp_dir, None, need_labels=False, inpaint=inpaint, need_tissue=csf,
                   work=a.work_dir or os.path.join(out, "cmbx"), log=log, need_flair="flair" in (wf.get("channels") or []))
    cases = os.path.join(out, "cases.csv")
    write_caselist(rows, cases)
    ck = {"last": None, "best": "model_best.pt"}.get(a.cp_load_type, f"model_epoch{a.cp_everyn_N}.pt")
    if a.cp_load_type == "everyN" and not a.cp_everyn_N:
        raise SystemExit("-cp_type everyN needs -cp_n <epoch>")
    det = os.path.join(out, "details")
    predict_run(run, cases, det, a.threshold, a.min_size, a.device, csf, log, use_folds=bool(a.folds),
                native=True, no_stage2=a.cand_discrimination is False, checkpoint=ck)
    p = a.prefix
    dirs = dict(final_predictions=("_mask", f"predicted_final_{p}_"))
    if a.intermediate:
        dirs.update(cdet_predictions=("_prob", f"predicted_cdet_{p}_"))
    for d, (suf, pre) in dirs.items():
        os.makedirs(os.path.join(out, d), exist_ok=True)
        for r in rows:
            shutil.copyfile(os.path.join(det, f"{r['id']}{suf}.nii.gz"), os.path.join(out, d, f"{pre}{r['id']}.nii.gz"))
    if a.intermediate:
        os.makedirs(os.path.join(out, "input_images"), exist_ok=True)
        for r in rows:
            shutil.copyfile(r["image"], os.path.join(out, "input_images", f"input_{p}_{r['id']}.nii.gz"))
    log(f"{len(rows)} case(s): masks in {out}/final_predictions, lesion lists in {det}")
    return out


# ---------------------------------------------------------------- cross_validate
def cross_validate(a, log=print) -> str:
    from . import config as C
    from .caselist import assign_folds
    from .data import write_caselist
    from .experiments import run_experiment
    _notes(a, log)
    inpaint = bool(a.inpaint) if _given(a, "inpaint") else False
    work = a.work_dir or default_work(preprocessed_folder(a.inp_dir))
    out = a.output_dir
    cases = os.path.join(out, "cases.csv")
    cfg = build_config(a, out, cases, os.path.join(work, "grid"), log)
    csf = bool(cfg["postprocess"]["csf_rule"])
    rows = collect(a.inp_dir, a.label_dir, need_labels=True, inpaint=inpaint, need_tissue=csf, work=work, log=log,
                   need_flair="flair" in cfg["train"]["channels"])
    _lesions(rows)
    assign_folds(rows, int(a.cv_fold), int(a.fold_seed))
    os.makedirs(out, exist_ok=True)
    write_caselist(rows, cases)
    cfg["data"]["prepared_dir"] = _prepared_dir(work, cfg, inpaint)
    cfg["data"]["n_folds"] = int(a.cv_fold)
    if _given(a, "early_stop_val"):                        # CV: early stopping on the inner fold, in evaluations
        T = cfg["train"]
        T["min_epochs"], T["patience"] = 0, max(1, -(-int(a.early_stop_val) // int(T["eval_every"])))
        T["early_stop"] = 0
    if _given(a, "train_prop"):
        log("note: -tr_prop has no effect in cross_validate: the inner fold of every training split chooses the epoch and cell")
        cfg["train"]["train_prop"] = 1.0
    if cfg["train"].get("checkpoint") != "last":
        cfg["train"]["checkpoint"] = "last"                # the CV keeps the inner-fold best state anyway
    _workflow_json(out, dict(command="cross_validate", inp_dir=os.path.abspath(a.inp_dir), inpaint=inpaint, csf_rule=csf,
                             folds=int(a.cv_fold), cases=len(rows), channels=cfg["train"]["channels"]))
    folds = list(range(int(a.resume_from_fold) - 1, int(a.cv_fold))) if _given(a, "resume_from_fold") else None
    run_experiment(cfg, folds, a.workers, log, final=bool(a.final))
    log(f"done: {out}/summary.json (lesion-level F1 out of fold)")
    return out


# ---------------------------------------------------------------- preprocess
def _fsl(cmd: str) -> str:
    d = os.environ.get("FSLDIR")
    p = os.path.join(d, "bin", cmd) if d and os.path.exists(os.path.join(d, "bin", cmd)) else shutil.which(cmd)
    if not p:
        raise SystemExit(f"FSL command {cmd!r} not found: install FSL (or give already preprocessed scans with -f True)")
    return p


def raw_images(inp: str, image_regex: str, fsl_preprocessed: bool):
    """(name, path, basepath) exactly as ``microbleednet preprocess`` names its cases: files ``**/*<regex>.nii*``; name =
    the file name before ``<regex>``, without -f True prefixed by the parent folder (``<folder>-<name>``)."""
    out = []
    for p in sorted(glob.glob(os.path.join(inp, "**", f"*{image_regex}.nii*"), recursive=True)):
        basepath = p.split(image_regex)[0]
        name = basepath.split(os.sep)[-1]
        if not fsl_preprocessed:
            name = basepath.split("/")[-2] + "-" + name
        out.append((name, p, basepath))
    return out


def _find(folder: Optional[str], name: str, endings: List[str], next_to: Optional[str] = None) -> Optional[str]:
    for e in endings:
        if next_to and os.path.isfile(next_to + e):        # MicrobleedNet without -f: the label sits next to the scan
            return next_to + e
        if folder:
            for p in [os.path.join(folder, name + e)] + sorted(glob.glob(os.path.join(folder, "**", name + e), recursive=True)):
                if os.path.isfile(p):
                    return p
    return None


def preprocess(a, log=print) -> str:
    out = a.out_dir
    os.makedirs(out, exist_ok=True)
    items = raw_images(a.inp_dir, a.image_regex, a.fsl_preprocessed)
    if not items:
        raise SystemExit(f"{a.inp_dir}: no file matches **/*{a.image_regex}.nii*")
    if a.label_dir and not a.label_regex:
        raise SystemExit("-l needs -lr <label ending>, e.g. -lr _cmb.nii.gz")
    missing, n_new, sources = [], 0, {}
    for name, path, basepath in items:
        cdir = os.path.join(out, name)                     # one folder per case
        os.makedirs(cdir, exist_ok=True)
        img, brain = os.path.join(cdir, f"{name}_{a.modality}.nii.gz"), os.path.join(cdir, f"{name}_brainmask.nii.gz")
        lab = os.path.join(cdir, f"{name}_cmb.nii.gz")
        if not (os.path.exists(img) and os.path.exists(brain)) or a.overwrite:
            n_new += 1
            if a.fsl_preprocessed:                         # already skull-stripped and bias-corrected
                im = nib.load(path)
                nib.save(im, img)
                given = _find(a.brain_dir, name, [a.brain_regex]) if a.brain_dir else None
                if a.brain_dir and not given:
                    missing.append((name, f"brain mask {name}{a.brain_regex} in {a.brain_dir}"))
                    continue
                if given:
                    nib.save(nib.Nifti1Image((np.asarray(nib.load(given).dataobj) > 0).astype(np.uint8), im.affine), brain)
                else:
                    _brain_from(img, brain)
            else:                                          # MicrobleedNet's script: fslreorient2std, BET, FAST -B --nopve
                with tempfile.TemporaryDirectory(dir=out) as tmp:
                    t = lambda f: os.path.join(tmp, f)
                    subprocess.run([_fsl("fslreorient2std"), path, t("INPUT.nii.gz")], check=True)
                    subprocess.run([_fsl("bet"), t("INPUT.nii.gz"), t("INPUT_brain.nii.gz"), "-m"], check=True)
                    subprocess.run([_fsl("fast"), "-B", "--nopve", t("INPUT_brain.nii.gz")], check=True)
                    shutil.copyfile(t("INPUT_brain_restore.nii.gz"), img)
                    shutil.copyfile(t("INPUT_brain_mask.nii.gz"), brain)
            log(f"{name}: image + brain mask")
        if a.label_dir and (not os.path.exists(lab) or a.overwrite):
            src = _find(a.label_dir, name if a.fsl_preprocessed else os.path.basename(basepath),
                        list(a.label_regex) + ["_mask_preproc.nii.gz"], next_to=None if a.fsl_preprocessed else basepath)
            if not src:
                missing.append((name, f"label <name>{a.label_regex} in {a.label_dir}"))
                continue
            if not a.fsl_preprocessed:                     # same reorientation as the scan (MicrobleedNet copies it as is)
                with tempfile.TemporaryDirectory(dir=out) as tmp:
                    r = os.path.join(tmp, "label.nii.gz")
                    subprocess.run([_fsl("fslreorient2std"), src, r], check=True)
                    src_img = nib.load(r)
                    data = np.asarray(src_img.dataobj) > 0
            else:
                data = np.asarray(nib.load(src).dataobj) > 0
            ref = nib.load(img)
            if data.shape != ref.shape[:3]:
                raise SystemExit(f"{name}: label {src} has shape {data.shape}, the scan {ref.shape[:3]}")
            nib.save(nib.Nifti1Image(data.astype(np.uint8), ref.affine), lab)
        fl = os.path.join(cdir, f"{name}_flair.nii.gz")
        seg = os.path.join(cdir, f"{name}_synthseg.nii.gz")
        flair_src = _find(a.flair_dir or a.inp_dir, name if a.fsl_preprocessed else os.path.basename(basepath),
                          [a.flair_regex + ".nii.gz", a.flair_regex + ".nii"], next_to=basepath) if a.flair_regex else None
        if flair_src and (not os.path.exists(fl) or a.overwrite or (a.synthseg and not os.path.exists(seg))):
            if a.flair_registered:                         # already on the scan matrix: bias field + brain mask only
                ref = nib.load(img)
                f = nib.load(flair_src)
                if f.shape[:3] != ref.shape[:3]:
                    raise SystemExit(f"{name}: --flair_registered True but the FLAIR {f.shape[:3]} is not on the scan {ref.shape[:3]}")
                b = np.asarray(nib.load(brain).dataobj) > 0
                nib.save(nib.Nifti1Image((np.asarray(f.dataobj, np.float32) * b).astype(np.float32), ref.affine), fl)
            else:
                register_flair(flair_src, img, brain, fl, seg if a.synthseg else None, a.threads)
            sources[name] = "flair" if a.synthseg and not a.flair_registered else sources.get(name, "")
            log(f"{name}: FLAIR registered to the scan" + (" + SynthSeg from FLAIR" if sources[name] == "flair" else ""))
        if a.synthseg and (not os.path.exists(seg) or a.overwrite and sources.get(name) != "flair"):
            exe = shutil.which("mri_synthseg")
            if not exe:
                raise SystemExit("--synthseg True needs FreeSurfer 7.3+ (mri_synthseg on the PATH)")
            subprocess.run([exe, "--i", img, "--o", seg, "--robust", "--threads", str(a.threads)], check=True)
            sources[name] = a.modality
        if a.inpaint:
            dst = os.path.join(cdir, f"{name}_inpainted.nii.gz")
            if not os.path.exists(dst) or a.overwrite:
                try:
                    from microbleednet.scripts.data_preparation import inpaint_vessels
                except ImportError:
                    raise SystemExit("--inpaint True uses MicrobleedNet's own inpaint_vessels: install microbleednet "
                                     "(https://github.com/v-sundaresan/microbleed-detection)")
                im = nib.load(img)
                b = np.asarray(nib.load(brain).dataobj) > 0
                arr = inpaint_vessels(np.asarray(im.dataobj, np.float64), b.astype(int))
                nib.save(nib.Nifti1Image((arr * b).astype(np.float32), im.affine), dst)
    if missing:
        raise SystemExit(f"{len(missing)} case(s) incomplete:\n" + "\n".join(f"  {n}: missing {w}" for n, w in missing))
    table = write_file_table(out, sources)
    log(f"{len(items)} case(s) ({n_new} new) -> {out}; file list {table}")
    return out


def register_flair(flair: str, scan: str, brain: str, out: str, synthseg_out: Optional[str] = None, threads: int = 4) -> None:
    """FLAIR of the session -> the scan: fslreorient2std, BET (-R -f 0.4), FAST -B --nopve -t 2, FLIRT 6 dof (normalised
    mutual information) onto the bias-corrected scan, trilinear, masked with the scan's brain mask. With
    ``synthseg_out``: SynthSeg 2.0 --robust on the FLAIR (better tissue contrast and thinner slices than T2* / SWI),
    carried to the scan with the same matrix (nearest neighbour)."""
    import nibabel.processing
    with tempfile.TemporaryDirectory(dir=os.path.dirname(out)) as tmp:
        t = lambda f: os.path.join(tmp, f)
        run = lambda *c: subprocess.run([str(x) for x in c], check=True, stdout=subprocess.DEVNULL)
        run(_fsl("fslreorient2std"), flair, t("F.nii.gz"))
        run(_fsl("bet"), t("F.nii.gz"), t("Fb.nii.gz"), "-R", "-f", "0.4")
        run(_fsl("fast"), "-B", "--nopve", "-t", "2", "-o", t("Fb"), t("Fb.nii.gz"))
        run(_fsl("flirt"), "-in", t("Fb_restore.nii.gz"), "-ref", scan, "-dof", "6", "-cost", "normmi", "-searchrx", "-20", "20",
            "-searchry", "-20", "20", "-searchrz", "-20", "20", "-omat", t("m.mat"), "-out", t("Fs.nii.gz"), "-interp", "trilinear")
        ref, b = nib.load(scan), np.asarray(nib.load(brain).dataobj) > 0
        nib.save(nib.Nifti1Image((np.asarray(nib.load(t("Fs.nii.gz")).dataobj, np.float32) * b).astype(np.float32), ref.affine), out)
        if synthseg_out:
            exe = shutil.which("mri_synthseg")
            if not exe:
                raise SystemExit("--synthseg True needs FreeSurfer 7.3+ (mri_synthseg on the PATH)")
            run(exe, "--i", t("F.nii.gz"), "--o", t("S.nii.gz"), "--robust", "--threads", threads)
            on_f = nibabel.processing.resample_from_to(nib.load(t("S.nii.gz")), nib.load(t("F.nii.gz")), order=0)
            nib.save(nib.Nifti1Image(np.asarray(on_f.dataobj).astype(np.int16), on_f.affine), t("Sf.nii.gz"))
            run(_fsl("flirt"), "-in", t("Sf.nii.gz"), "-ref", scan, "-applyxfm", "-init", t("m.mat"), "-interp",
                "nearestneighbour", "-out", t("Ss.nii.gz"))
            nib.save(nib.Nifti1Image(np.round(np.asarray(nib.load(t("Ss.nii.gz")).dataobj)).astype(np.int16), ref.affine), synthseg_out)


def write_file_table(prep: str, sources: Optional[Dict[str, str]] = None) -> str:
    """files.xlsx (or files.csv without openpyxl) next to the preprocessed folder: one row per case, one column per
    file, paths relative to the table, and what SynthSeg was computed on (flair / the scan) when known."""
    import pandas as pd
    prep = os.path.abspath(prep)
    base = os.path.dirname(os.path.dirname(prep)) if os.path.basename(prep) == "preprocessed" and \
        os.path.basename(os.path.dirname(prep)) == "derivatives" else os.path.dirname(prep)
    rows = []
    for n, d in case_folders(prep).items():
        r = dict(case=n)
        sc = _scan(d, n)
        r["scan"] = os.path.relpath(sc, base) if sc else ""
        for col, suf in (("brainmask", "_brainmask"), ("cmb", "_cmb"), ("synthseg", "_synthseg"), ("flair", "_flair"),
                         ("inpainted", "_inpainted")):
            f = _nii(d, n, suf)
            r[col] = os.path.relpath(f, base) if f else ""
        r["synthseg_source"] = (sources or {}).get(n, "") if r["synthseg"] else ""
        rows.append(r)
    old = os.path.join(base, "files.xlsx")
    if os.path.exists(old):                                # keep sources known from earlier runs
        try:
            import pandas as pd
            prev = dict(pd.read_excel(old).fillna("").set_index("case")["synthseg_source"])
            for r in rows:
                r["synthseg_source"] = r["synthseg_source"] or (prev.get(r["case"], "") if r["synthseg"] else "")
        except (ImportError, KeyError, ValueError):
            pass
    df = pd.DataFrame(rows)
    try:
        import openpyxl  # noqa: F401
        path = os.path.join(base, "files.xlsx")
        df.to_excel(path, sheet_name="files", index=False)
    except ImportError:
        path = os.path.join(base, "files.csv")
        df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- argument parsers
def _common_train(p, fine_tune: bool = False, cv: bool = False) -> None:
    B = str2bool
    g = p.add_argument_group("MicrobleedNet options")
    g.add_argument("-tr_prop", "--train_prop", type=float, default=None,
                   help="share of cases trained on; the rest choose threshold, best epoch, early stop [train: 0.8]")
    g.add_argument("-bfactor", "--batch_factor", type=int, default=None, help="cases drawn per epoch [all]")
    g.add_argument("-psize", "--patch_size", type=int, default=None, help="cubic patches of N voxels [38 x 62 x 94]")
    if not cv:
        g.add_argument("-cand_det", "--cand_detection", type=B, default=None, help="train step 1 [True]")
    g.add_argument("-cand_disc", "--cand_discrimination", type=B, default=None, help="train step 2 [False]")
    g.add_argument("-da", "--data_augmentation", type=B, default=None, help="random flips [True]")
    g.add_argument("-af", "--aug_factor", type=int, default=None, help="patches per epoch x N [1]")
    g.add_argument("-sv_resume", "--save_resume_training", type=B, default=None, help="save and resume [False]")
    g.add_argument("-ilr", "--init_learng_rate", type=float, default=None,
                   help="learning rate [0.0001]" if fine_tune else "learning rate [0.0003]")
    g.add_argument("-lrm", "--lr_sch_mlstone", type=int, nargs="+", default=None, help="step schedule milestones [cosine]")
    g.add_argument("-gamma", "--lr_sch_gamma", type=float, default=None, help="lr factor at each milestone [0.1]")
    g.add_argument("-opt", "--optimizer", choices=["adam", "sgd", "adamw"], default=None, help="[adamw]")
    g.add_argument("-eps", "--epsilon", type=float, default=None, help="Adam epsilon [1e-8]")
    g.add_argument("-mom", "--momentum", type=float, default=None, help="SGD momentum [0.9]")
    g.add_argument("-bs", "--batch_size", type=int, default=None, help="[4]")
    g.add_argument("-ep", "--num_epochs", type=int, default=None, help="[60]")
    g.add_argument("-es", "--early_stop_val", type=int, default=None, help="epochs without held-out gain [off]")
    g.add_argument("-sv_mod", "--save_full_model", type=B, default=None, help="also save the whole module [False]")
    g.add_argument("-cp_type", "--cp_save_type", choices=["last", "best", "everyN"], default=None, help="[last]")
    g.add_argument("-cp_n", "--cp_everyn_N", type=int, default=None, help="N for everyN [10]")
    g.add_argument("-v", "--verbose", type=B, default=None, help="(no effect: cmbx always logs)")
    g.add_argument("-mgpu", "--multi_gpu", type=B, default=None, help="use all visible GPUs [False]")
    if fine_tune:
        g.add_argument("-p", "--pretrained_model", type=B, default=None, help="start from the pretrained model (= -m pre)")
        g.add_argument("-cpld_type", "--cp_load_type", dest="cp_load_type_ft", default=None, help="(no effect)")
        g.add_argument("-cpld_n", "--cpload_everyn_N", type=int, default=None, help="(no effect)")
        g.add_argument("-ftlayers", "--ft_layers", type=int, nargs="+", default=None,
                       help="levels to train, 1 = output ... 9 = input [all]")
    if cv:
        g.add_argument("-fold", "--cv_fold", type=int, default=5)
        g.add_argument("-resume_fold", "--resume_from_fold", type=int, default=None, help="start at fold N (1-based)")
        g.add_argument("-int", "--intermediate", type=B, default=None, help="(no effect: fold maps are always kept)")
        g.add_argument("-sv", "--save_checkpoint", type=B, default=None, help="(no effect: fold models are always kept)")
    c = p.add_argument_group("cmbx options")
    c.add_argument("--inpaint", type=B, default=None, help="train on vessel-inpainted images [False]")
    c.add_argument("--frst", type=B, default=None, help="FRST as second input channel [True]")
    c.add_argument("--csf_rule", type=B, default=None, help="drop detections in CSF, needs <name>_synthseg [False]")
    c.add_argument("--flair", type=B, default=None, help="FLAIR as extra input channel, needs <name>_flair [False]")
    c.add_argument("--disc_model", choices=["teacher_student", "cnn"], default=None, help="step 2 network [teacher_student]")
    c.add_argument("--kd_temperature", type=float, default=None, help="teacher_student temperature [4]")
    c.add_argument("--kd_alpha", type=float, default=None, help="teacher_student distillation weight [0.4]")
    c.add_argument("--loss", choices=["bce_dice", "bce", "dice", "focal_dice"], default=None, help="[bce_dice]")
    c.add_argument("--patches", type=int, default=None, help="patches per epoch [800]")
    c.add_argument("--seed", type=int, default=None, help="[42]")
    c.add_argument("--device", default=None, help="auto | cuda | cuda:N | cpu [auto]")
    c.add_argument("--workers", type=int, default=None, help="parallel cases while preparing [4]")
    c.add_argument("--work_dir", default=None, help="grid volumes and derived files [<-i>/cmbx]")
    c.add_argument("--config", default=None, help="experiment YAML: any other setting (docs/DETAILS.md)")
    c.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config keys, e.g. train.lr=0.0001")
    c.add_argument("--overwrite", type=B, default=False, help="train again into an existing model folder [False]")
    if cv:
        c.add_argument("--fold_seed", type=int, default=0, help="fold assignment seed [0]")
        c.add_argument("--final", type=B, default=False, help="also train one model on all cases after the CV [False]")


def add_parsers(sub, log) -> None:
    B = str2bool
    hf = argparse.RawDescriptionHelpFormatter
    p = sub.add_parser("preprocess", formatter_class=hf, help="raw scans -> derivatives/preprocessed/<case>/ (MicrobleedNet style)")
    r = p.add_argument_group("required")
    r.add_argument("-i", "--inp_dir", required=True, help="folder with the scans (searched recursively)")
    r.add_argument("-ir", "--image_regex", required=True, help="ending of every scan name before .nii, e.g. _T2starw")
    r.add_argument("-o", "--out_dir", required=True, help="the preprocessed folder, e.g. <cohort>/derivatives/preprocessed")
    o = p.add_argument_group("options")
    o.add_argument("-l", "--label_dir", default=None, help="folder with the microbleed masks (written as <name>_cmb)")
    o.add_argument("-lr", "--label_regex", nargs="+", default=None, help="ending(s) of the mask names, e.g. _cmb.nii.gz")
    o.add_argument("-f", "--fsl_preprocessed", type=B, default=False, help="scans already skull-stripped + bias-corrected")
    o.add_argument("-b", "--brain_dir", default=None, help="with -f True: folder with your brain masks [image > 0]")
    o.add_argument("-br", "--brain_regex", default="_brainmask.nii.gz", help="ending of the brain mask names")
    o.add_argument("-flr", "--flair_regex", default=None, help="ending of the FLAIR scans of the same session, e.g. _FLAIR")
    o.add_argument("-fl", "--flair_dir", default=None, help="folder with the FLAIR scans [-i]")
    o.add_argument("--flair_registered", type=B, default=False,
                   help="FLAIR already on the scan matrix; otherwise BET + FAST + FLIRT 6 dof onto the scan (FSL)")
    o.add_argument("-mod", "--modality", choices=["t2star", "swi"], default="t2star", help="name of the scan file [t2star]")
    o.add_argument("--synthseg", type=B, default=False, help="add <name>_synthseg (FreeSurfer mri_synthseg; from the FLAIR if given)")
    o.add_argument("--threads", type=int, default=4, help="threads for SynthSeg")
    o.add_argument("--inpaint", type=B, default=False, help="add <name>_inpainted (MicrobleedNet inpaint_vessels)")
    o.add_argument("--overwrite", type=B, default=False)
    o.add_argument("-v", "--verbose", type=B, default=None, help="(no effect)")
    o.add_argument("-pbar", "--progress_bar", type=B, default=None, help="(no effect)")
    p.set_defaults(fn=lambda a: preprocess(a, log))

    p = sub.add_parser("train", formatter_class=hf, help="train a model on a preprocessed folder (MicrobleedNet style)")
    r = p.add_argument_group("required")
    r.add_argument("-i", "--inp_dir", required=True, help="the preprocessed folder")
    r.add_argument("-l", "--label_dir", default=None, help="folder with <name>_cmb (default: -i)")
    r.add_argument("-m", "--model_dir", required=True, help="where the model goes")
    _common_train(p)
    p.set_defaults(fn=lambda a: train(a, log))

    p = sub.add_parser("fine_tune", aliases=["fine-tune"], formatter_class=hf,
                       help="fine-tune a trained or the pretrained model (MicrobleedNet style)")
    r = p.add_argument_group("required")
    r.add_argument("-i", "--inp_dir", required=True, help="the preprocessed folder")
    r.add_argument("-l", "--label_dir", default=None, help="folder with <name>_cmb (default: -i)")
    r.add_argument("-m", "--model_name", required=True, help="model folder to start from, or pre (pretrained VALDO model)")
    r.add_argument("-o", "--output_dir", required=True, help="where the fine-tuned model goes")
    _common_train(p, fine_tune=True)
    p.set_defaults(fn=lambda a: train(a, log, fine_tune=True))

    p = sub.add_parser("evaluate", formatter_class=hf, help="apply a model to a preprocessed folder (MicrobleedNet style); "
                       "`cmbx evaluate <run>` re-scores a run as before")
    p.add_argument("run", nargs="?", default=None, help=argparse.SUPPRESS)
    r = p.add_argument_group("required")
    r.add_argument("-i", "--inp_dir", default=None, help="the preprocessed folder")
    r.add_argument("-m", "--model_name", default=None, help="model folder, or pre (pretrained VALDO model)")
    r.add_argument("-o", "--output_dir", default=None, help="output folder")
    g = p.add_argument_group("MicrobleedNet options")
    g.add_argument("-p", "--pretrained_model", type=B, default=None, help="use the pretrained model (= -m pre)")
    g.add_argument("-int", "--intermediate", type=B, default=False, help="also step-1 maps and input images [False]")
    g.add_argument("-cp_type", "--cp_load_type", choices=["last", "best", "everyN"], default="last", help="[last]")
    g.add_argument("-cp_n", "--cp_everyn_N", type=int, default=None, help="epoch for everyN")
    g.add_argument("-nclass", "--num_classes", type=int, default=None, help="(no effect)")
    g.add_argument("-v", "--verbose", type=B, default=None, help="(no effect)")
    c = p.add_argument_group("cmbx options")
    c.add_argument("--csf_rule", type=B, default=None, help="drop detections in CSF [as trained]")
    c.add_argument("-cand_disc", "--cand_discrimination", type=B, default=None, help="False: step 1 only [use step 2 if trained]")
    c.add_argument("--threshold", type=float, default=None, help="[chosen in training]")
    c.add_argument("--min_size", type=int, default=None, help="minimum size in grid voxels [chosen in training]")
    c.add_argument("--folds", type=B, default=False, help="mean of the CV fold models instead of the final model")
    c.add_argument("--prefix", default="cmbx", help="file names predicted_final_<prefix>_<name> [cmbx]")
    c.add_argument("--device", default="auto")
    c.add_argument("--work_dir", default=None, help="derived files of a MicrobleedNet folder [<-o>/cmbx]")
    # the former `cmbx evaluate <run>` (re-score a finished run)
    c.add_argument("--mode", choices=["stored", "inner", "crossfold"], default="stored", help=argparse.SUPPRESS)
    c.add_argument("--csf-rule", dest="csf_rule_old", action="store_true", help=argparse.SUPPRESS)
    c.add_argument("--prepared", default=None, help=argparse.SUPPRESS)
    c.add_argument("--tag", default=None, help=argparse.SUPPRESS)

    def ev(a):
        if a.run and not a.inp_dir:
            from .cli import cmd_rescore
            a.csf_rule = a.csf_rule_old
            return cmd_rescore(a)
        if not (a.inp_dir and (a.model_name or a.pretrained_model) and a.output_dir):
            raise SystemExit("cmbx evaluate -i <preprocessed> -m <model folder | pre> -o <output>")
        a.model_name = a.model_name or "pre"
        return evaluate(a, log)
    p.set_defaults(fn=ev)

    p = sub.add_parser("cross_validate", aliases=["cross-validate"], formatter_class=hf,
                       help="k-fold cross-validation on a preprocessed folder (MicrobleedNet style)")
    r = p.add_argument_group("required")
    r.add_argument("-i", "--inp_dir", required=True, help="the preprocessed folder")
    r.add_argument("-l", "--label_dir", default=None, help="folder with <name>_cmb (default: -i)")
    r.add_argument("-o", "--output_dir", required=True, help="output folder (scores in summary.json)")
    _common_train(p, cv=True)
    p.set_defaults(fn=lambda a: cross_validate(a, log))
