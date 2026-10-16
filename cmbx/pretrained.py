"""``cmbx get-model``: pretrained models, ready for ``cmbx predict`` right after the install.

    cmbx get-model valdo_t2            # -> models/valdo_t2/ (a run folder: config.yaml, final/, fold0/, fold1/)
    cmbx predict --run models/valdo_t2 --image scan.nii.gz [--brain-mask mask.nii.gz] --out pred/

Only models trained on PUBLIC data are listed. The files come from the GitHub release named below and are checked
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

REGISTRY: Dict[str, Dict] = {
    "valdo_t2": dict(
        url="https://github.com/mendeltem/microbleed_detection_3D_UNET/releases/download/v0.1.0/",
        files=[("valdo_t2_seed42.pt", "789e36098b80bbf87a655720e8b84525365db4708b687700a12c6a07d15c3550", 42),
               ("valdo_t2_seed1.pt", "5a74da0ae50afa49b72bc3247dccea1a4f371bb93e5279f135d01ee160e6a6d8", 1)],
        sequence="T2*-GRE (bias-corrected; trained on FSL FAST-corrected, skull-stripped T2S)",
        trained_on="VALDO 2021 Task 2, all 72 public cases (236 microbleeds, three cohorts, 0.8-4 mm slices); "
                   "60 epochs, state after the last epoch, this U-Net (a03-aniso, image + FRST, width 32)",
        licence="CC BY-NC-SA 4.0 (derived from the VALDO data): non-commercial, share alike, cite the VALDO paper",
        cite="Sudre et al., Where is VALDO? VAscular Lesions Detection and segmentatiOn challenge at MICCAI 2021, "
             "Medical Image Analysis 2024, doi:10.1016/j.media.2023.103029",
        threshold=0.6, min_size=8,
        cell_note="threshold 0.6, minimum 8 grid voxels (= 2 mm3): the stage-1 setting measured for these full-data "
                  "weights on an external in-house T2* cohort (pooled lesion F1 about 0.6); the probabilities of "
                  "full-data models run higher than those of cross-validation models, so 0.3 over-detects",
        caveats="T2*-GRE only (on SWI clearly worse); the input must be bias-corrected like the training data; "
                "research software, not a medical device",
        overrides={"preprocess": {"reorient": "none", "brain_mask_mode": "mask"},
                   "train": {"channels": ["image", "frst"]}, "model": {"name": "aniso_unet3d", "base_channels": 32}}),
    "cmb_classifier": dict(
        kind="stage2",
        url="https://github.com/mendeltem/microbleed_detection_3D_UNET/releases/download/v0.1.0/",
        files=[("classifier.pt", "45b1773b27745f932182c74d277a5cbe91c7069f4a7c3b81f04b9868ac3e9cce", None)],
        sequence="T2*-GRE and SWI (second stage: re-checks the candidates of any first-stage run)",
        trained_on="pooled out-of-fold candidates of VALDO and two in-house clinical cohorts (T2* and SWI), "
                   "2230 candidates, 3 members, 40 epochs",
        licence="research use only, non-commercial; derived from clinical data (no patient data in the weights), "
                "released by the author with microbleed_detection_3D_UNET v0.1.0",
        cite="microbleed_detection_3D_UNET v0.1.0 (mendeltem, 2026); VALDO: Sudre et al., Med Image Anal 2024, "
             "doi:10.1016/j.media.2023.103029",
        threshold=0.3, min_size=4,
        cell_note="accepts a candidate at classifier score >= 0.3; candidates = components above 0.15, >= 1 mm3",
        caveats="optional: use with --stage2 models/cmb_classifier, or fine-tune with cmbx stage2 <run> --init "
                "models/cmb_classifier; not downloaded by install.sh",
        overrides={}),
}


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
    if name not in reg:
        raise SystemExit(f"unknown model {name!r}; available: {sorted(reg)}")
    m = reg[name]
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
        log(f"{name} ready in {run} (licence: research use only); use: cmbx predict ... --stage2 {run}  or  "
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
