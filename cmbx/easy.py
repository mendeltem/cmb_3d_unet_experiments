"""cmbx.easy -- the whole pipeline as a few plain Python functions.

    from cmbx import easy
    easy.preprocess("raw_scans", "my_data")                      # 1. scans -> one folder per case
    easy.train("my_data/train", "my_model")                      # 2. train a model
    easy.predict("my_data/test", "my_model", "my_predictions")   # 3. find microbleeds
    print(easy.score("my_predictions", "my_data/test"))          #    and count hits
    easy.fine_tune("my_data/train", "my_finetuned_model")        # 4. adapt the pretrained model

Each function runs the matching `cmbx` command (`python -m cmbx train ...`), so the results are exactly the same as on
the command line. Any other command-line option can be passed by its long name, e.g. csf_rule=True, patches=500,
device="cpu" (`python -m cmbx train --help` lists them all). The output of the command is shown while it runs; if the
command fails, a RuntimeError says which one.
"""
import glob
import os
import shutil
import subprocess
import sys


def preprocess(scans, out, scan_ending="_T2starw", masks=None, mask_ending="_cmb.nii.gz", brain_extracted=False,
               **options):
    """Bring raw scans into the cmbx layout: one folder per case in `out`.

    scans           folder with the scans, e.g. raw/sub-01_T2starw.nii.gz
    scan_ending     the end of every scan name before .nii.gz, e.g. "_T2starw" or "_swi"
    masks           folder with the microbleed masks (needed for training and scoring; None for new patients)
    mask_ending     the end of every mask name, e.g. "_cmb.nii.gz"
    brain_extracted True if the scans are already skull-stripped and bias-corrected (as the VALDO examples)
    """
    args = {"inp_dir": scans, "image_regex": scan_ending, "out_dir": out, "fsl_preprocessed": brain_extracted}
    if masks:
        args.update(label_dir=masks, label_regex=mask_ending)
    _cmbx("preprocess", args, options)
    return out


def split(data, out, train, test):
    """Copy the case folders of `data` into out/train and out/test. Returns (train_folder, test_folder)."""
    for part, cases in (("train", train), ("test", test)):
        for case in cases:
            target = os.path.join(out, part, case)
            if not os.path.isdir(target):
                shutil.copytree(os.path.join(data, case), target)
    return os.path.join(out, "train"), os.path.join(out, "test")


def train(data, model, epochs=60, **options):
    """Train a new model from scratch on the cases in `data`; the model goes into the folder `model`."""
    _cmbx("train", {"inp_dir": data, "model_dir": model, "num_epochs": epochs}, options)
    return model


def fine_tune(data, model, start_from="pre", epochs=30, **options):
    """Continue training an existing model on the cases in `data`.
    start_from: "pre" (the pretrained VALDO model) or the folder of your own model. The new model goes into `model`."""
    _cmbx("fine_tune", {"inp_dir": data, "model_name": start_from, "output_dir": model, "num_epochs": epochs}, options)
    return model


def predict(data, model, out, **options):
    """Find microbleeds in every case of `data` with `model` ("pre" = the pretrained VALDO model).
    Writes one mask per case to out/final_predictions/ and a lesion list per case."""
    _cmbx("evaluate", {"inp_dir": data, "model_name": model, "output_dir": out}, options)
    return out


def cross_validate(data, out, folds=5, **options):
    """k-fold cross-validation on the cases in `data`; scores in out/summary.json."""
    _cmbx("cross_validate", {"inp_dir": data, "output_dir": out, "cv_fold": folds}, options)
    return out


def score(predictions, data):
    """Compare the predicted masks in `predictions` with the reference masks in `data`.

    A reference microbleed counts as found (TP) when a predicted spot touches it; a predicted spot that touches none
    is a false alarm (FP); a reference microbleed without any predicted spot is missed (FN).
    Returns a dict: one entry per case and "all" for the totals, each with tp, fp, fn, sensitivity, precision, f1.
    """
    import nibabel as nib
    from .metrics import hits

    result, totals = {}, {"tp": 0, "fp": 0, "fn": 0}
    for case in sorted(os.listdir(data)):
        reference_file = os.path.join(data, case, f"{case}_cmb.nii.gz")
        predicted_files = glob.glob(os.path.join(predictions, "final_predictions", f"predicted_final_*_{case}.nii*"))
        if not os.path.exists(reference_file) or not predicted_files:
            continue
        h = hits(nib.load(reference_file).get_fdata(), nib.load(predicted_files[0]).get_fdata())
        result[case] = _scores(h["tp"], h["fp"], h["fn"])
        for key in totals:
            totals[key] += h[key]
    result["all"] = _scores(**totals)
    return result


# ---------------------------------------------------------------------------------------------------------------- helpers

def _scores(tp, fp, fn):
    sensitivity = tp / (tp + fn) if tp + fn else None        # no reference microbleeds -> not defined
    precision = tp / (tp + fp) if tp + fp else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "sensitivity": sensitivity, "precision": precision, "f1": f1}


def _cmbx(command, args, options):
    """Run `python -m cmbx <command>` with the given options (name -> value). True/False become "True"/"False",
    None is left out, lists become several values."""
    words = [sys.executable, "-m", "cmbx", command]
    for name, value in {**args, **options}.items():
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        words += [f"--{name}"] + [str(v) for v in values]
    print("$ cmbx " + " ".join(words[3:]), flush=True)
    if subprocess.run(words).returncode != 0:
        raise RuntimeError(f"cmbx {command} failed (see the messages above)")
