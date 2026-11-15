"""``cmbx get-model``: pretrained models, ready for ``cmbx predict`` right after the install.

    cmbx get-model valdo_t2            # -> models/valdo_t2/ (a run folder: config.yaml, final/, fold0/, fold1/)
    cmbx predict --run models/valdo_t2 --image scan.nii.gz [--brain-mask mask.nii.gz] --out pred/
    cmbx get-model valdo_classifier    # optional second stage -> models/valdo_classifier/stage2/classifier.pt
    cmbx predict --run models/valdo_t2 --stage2 models/valdo_classifier --image scan.nii.gz --out pred/

Only models trained on PUBLIC data (VALDO 2021 Task 2) are listed, the U-Net and the second-stage classifier alike;
operating points are chosen on VALDO as well. The files come from the GitHub release named below and are checked
against their SHA-256; the model weights are not part of this repository (their licence differs from the code's).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.request
from typing import Dict

from .config import dump, load_config

RELEASE = "https://github.com/mendeltem/cmb_3d_unet_experiments/releases/download/v0.2.0/"
VALDO_CITE = ("Sudre et al., Where is VALDO? VAscular Lesions Detection and segmentatiOn challenge at MICCAI 2021, "
              "Medical Image Analysis 2024, doi:10.1016/j.media.2023.103029")
VALDO_LICENCE = "CC BY-NC-SA 4.0 (derived from the VALDO data): non-commercial, share alike, cite the VALDO paper"

REGISTRY: Dict[str, Dict] = {
    "valdo_t2": dict(
        url=RELEASE,
        files=[("valdo_t2_seed42.pt", "789e36098b80bbf87a655720e8b84525365db4708b687700a12c6a07d15c3550", 42),
               ("valdo_t2_seed1.pt", "5a74da0ae50afa49b72bc3247dccea1a4f371bb93e5279f135d01ee160e6a6d8", 1)],
        sequence="T2*-GRE (bias-corrected; trained on FSL FAST-corrected, skull-stripped T2S)",
        trained_on="VALDO 2021 Task 2 only: all 72 public cases (236 microbleeds, three cohorts, 0.8-4 mm slices); "
                   "60 epochs, state after the last epoch, this U-Net (a03-aniso, image + FRST, width 32)",
        licence=VALDO_LICENCE,
        cite=VALDO_CITE,
        threshold=0.3, min_size=8,
        cell_note="threshold 0.3, minimum 8 grid voxels (= 2 mm3): the cell chosen on VALDO only (nested 5-fold "
                  "cross-validation, every fold chose threshold 0.3). Full-data weights give higher probabilities than "
                  "the cross-validation models, so on a new scanner this cell can over-detect: use the VALDO "
                  "classifier (--stage2 models/valdo_classifier) or set --threshold on a few labelled scans of your own",
        caveats="T2*-GRE only (on SWI clearly worse); the input must be bias-corrected like the training data; "
                "research software, not a medical device",
        overrides={"preprocess": {"reorient": "none", "brain_mask_mode": "mask"},
                   "train": {"channels": ["image", "frst"]}, "model": {"name": "aniso_unet3d", "base_channels": 32}}),
    "valdo_classifier": dict(
        kind="stage2",
        url=RELEASE,
        files=[("valdo_classifier.pt", "ef179dca588ab522a27344c94f6001a7c01a4c701b27655e3808216302d6d4ef", None)],
        sequence="second stage for T2*-GRE: re-checks the candidates of any first-stage run",
        trained_on="VALDO 2021 Task 2 only: the out-of-fold candidates of 57 VALDO cases (139 microbleeds; "
                   "278 candidates), 3 members, 40 epochs",
        licence=VALDO_LICENCE,
        cite=VALDO_CITE,
        threshold=0.55, min_size=4,
        cell_note="accepts a candidate at classifier score >= 0.55 (chosen by 5-fold cross-validation by case on the "
                  "VALDO candidates only; pooled out-of-fold lesion F1 0.614); candidates = components above 0.15, "
                  ">= 1 mm3",
        caveats="optional: use with --stage2 models/valdo_classifier, or fine-tune with cmbx stage2 <run> --init "
                "models/valdo_classifier; not downloaded by install.sh",
        overrides={}),
}
# Old names keep working. cmb_classifier (until v0.1.0 a classifier pooled with in-house data, withdrawn) now
# downloads the VALDO-only classifier into models/cmb_classifier.
ALIASES: Dict[str, str] = {"cmb_classifier": "valdo_classifier"}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, target: str, log) -> None:
    tmp = target + ".part"
    log(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        total, done = int(resp.headers.get("Content-Length") or 0), 0
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            out.write(chunk)
            done += len(chunk)
            if total and sys.stderr.isatty():
                sys.stderr.write(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB")
    if sys.stderr.isatty():
        sys.stderr.write("\n")
    os.replace(tmp, target)


def get_model(name: str, out_dir: str = "models", log=print, registry: Dict = None) -> str:
    reg = (registry or REGISTRY)
    key = name if name in reg else ALIASES.get(name, name)
    if key not in reg:
        raise SystemExit(f"unknown model {name!r}; available: {sorted(reg)} (old names: {sorted(ALIASES)})")
    m = reg[key]
    if key != name:
        log(f"{name} is now {key} (VALDO-only weights); installing it as {os.path.join(out_dir, name)}")
    run = os.path.join(out_dir, name)
    if m.get("kind") == "stage2":                      # a second-stage classifier: <dir>/<name>/stage2/classifier.pt
        fname, sha, _ = m["files"][0]
        d = os.path.join(run, "stage2")
        os.makedirs(d, exist_ok=True)
        target = os.path.join(d, "classifier.pt")
        if not (os.path.exists(target) and _sha256(target) == sha):
            _download(m["url"] + fname, target, log)
            if _sha256(target) != sha:
                os.remove(target)
                raise SystemExit(f"{fname}: SHA-256 mismatch; download again")
        with open(os.path.join(run, "MODEL_CARD.md"), "w") as f:
            f.write(f"# {name}\n\n| | |\n|---|---|\n| role | {m['sequence']} |\n| trained on | {m['trained_on']} |\n"
                    f"| operating rule | {m['cell_note']} |\n| use | {m['caveats']} |\n| licence | {m['licence']} |\n"
                    f"| cite | {m['cite']} |\n| file | {fname} (SHA-256 checked), from {m['url']} |\n")
        log(f"{name} ready in {run} (licence: {m['licence'].split(' (')[0]}); use: cmbx predict ... --stage2 {run}  or  "
            f"cmbx stage2 <run> --init {run}")
        return run
    for k, (fname, sha, seed) in enumerate(m["files"]):
        d = os.path.join(run, f"fold{k}")
        os.makedirs(d, exist_ok=True)
        target = os.path.join(d, "model.pt")
        if not (os.path.exists(target) and _sha256(target) == sha):
            _download(m["url"] + fname, target, log)
            got = _sha256(target)
            if got != sha:
                os.remove(target)
                raise SystemExit(f"{fname}: SHA-256 mismatch (got {got[:12]}..., expected {sha[:12]}...); download again")
        json.dump(dict(ok=True, fold=k, threshold=m["threshold"], min_size=m["min_size"], source=fname, sha256=sha,
                       seed=seed, pretrained=name), open(os.path.join(d, "meta.json"), "w"), indent=1)
    fin = os.path.join(run, "final")
    os.makedirs(fin, exist_ok=True)
    if not os.path.exists(os.path.join(fin, "model.pt")):
        shutil.copyfile(os.path.join(run, "fold0", "model.pt"), os.path.join(fin, "model.pt"))
    json.dump(dict(ok=True, threshold=m["threshold"], min_size=m["min_size"], source=m["files"][0][0],
                   pretrained=name, note="the final model is the seed-42 file; --folds averages both seeds"),
              open(os.path.join(fin, "meta.json"), "w"), indent=1)
    cfg = load_config(None, {"name": name, **m["overrides"]})
    dump(cfg, os.path.join(run, "config.yaml"))
    with open(os.path.join(run, "MODEL_CARD.md"), "w") as f:
        f.write(f"# {name}\n\n| | |\n|---|---|\n| sequence | {m['sequence']} |\n| trained on | {m['trained_on']} |\n"
                f"| operating cell | {m['cell_note']} |\n| caveats | {m['caveats']} |\n| licence | {m['licence']} |\n"
                f"| cite | {m['cite']} |\n| files | {', '.join(f for f, _, _ in m['files'])} (SHA-256 checked), from {m['url']} |\n")
    log(f"{name} ready in {run} (licence: {m['licence'].split(' (')[0]}); use: cmbx predict --run {run} --image <scan> --out <dir>")
    return run
