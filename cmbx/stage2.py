"""``cmbx stage2 <run>``: the optional second stage -- a small 3D CNN that looks at every candidate of the U-Net again.

Why: many microbleeds the U-Net misses still get a raised probability; a low threshold finds them but floods the result
with false alarms. The second stage takes every component above a LOW threshold and answers one question per
candidate: microbleed or not (the design of MicrobleedNet's discriminator; Sundaresan et al. 2023).

    cmbx stage2 <run>                       # train + nested CV on the run's out-of-fold maps -> <run>/stage2/
    cmbx stage2 <run> --init <classifier>   # fine-tune from a trained classifier (a run, a stage2/classifier.pt, or
                                            # the classifier.pt of the microbleed_detection_3D_UNET release)
    cmbx predict --run <run> ...            # uses <run>/stage2/classifier.pt automatically (--no-stage2 to skip)

Recipe (ported from cmb-valdo-chain ``cmb/stage2``, same numbers): candidates = 26-connected components of the
OUT-OF-FOLD probability map above ``low_threshold`` 0.15, >= 4 grid voxels (1 mm3), <= 2100 voxels; per candidate a
block of 20 x 40 x 40 voxels (z, y, x) of image, FRST and probability around its centre; the network sees a 16 x 32 x 32
crop (random offset +-2/4/4 and flips in training, centre crop otherwise); 3 members with different seeds, 40 epochs,
batch 16, AdamW lr 1e-3, weight decay 1e-3, cosine schedule, BCE with positive weight = negatives / positives; scores
averaged over the members. A candidate counts as positive if it touches a reference microbleed.
Evaluation is nested: for every fold k the classifier is trained on the candidates of the other folds and scores the
candidates of fold k; the decision threshold is the one with the best pooled lesion F1 on these out-of-fold scores
(as in the original; reported next to the first-stage CV score). The final classifier is trained on all candidates.
Lesion level as everywhere in cmbx: TP = reference lesions touched by an accepted candidate, FP = accepted candidates
touching none, FN = the remaining reference lesions (also those no candidate reached).
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

S26 = np.ones((3, 3, 3), bool)
CHANNELS = ["image", "frst", "prob"]
DEFAULTS = dict(low_threshold=0.15, min_voxels=4, max_voxels=2100, half=[20, 20, 10], crop=[16, 32, 32],
                jitter=[2, 4, 4], members=3, epochs=40, batch_size=16, lr=1e-3, weight_decay=1e-3)
THRESHOLDS = [round(float(t), 2) for t in np.arange(0.05, 0.96, 0.05)]


def build_classifier() -> nn.Sequential:
    """Identical to the released classifier (microbleed_detection_3D_UNET ``mb_segment/classifier.py``)."""
    def block(c_in, c_out):
        return nn.Sequential(nn.Conv3d(c_in, c_out, 3, padding=1, bias=False), nn.InstanceNorm3d(c_out, affine=True),
                             nn.LeakyReLU(0.01, True), nn.Conv3d(c_out, c_out, 3, padding=1, bias=False),
                             nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, True))
    return nn.Sequential(block(3, 16), nn.MaxPool3d(2), block(16, 32), nn.MaxPool3d(2), block(32, 64),
                         nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))


def crop_block(volume: np.ndarray, centre, half) -> np.ndarray:
    """Block of ``2 * half`` around ``centre`` (x, y, z), zero-padded at the border."""
    out = np.zeros([2 * h for h in half], volume.dtype)
    src, dst = [], []
    for c, h, n in zip(centre, half, volume.shape):
        lo, hi = c - h, c + h
        start, stop = max(lo, 0), min(hi, n)
        if start >= stop:
            return out
        src.append(slice(start, stop))
        dst.append(slice(start - lo, stop - lo))
    out[tuple(dst)] = volume[tuple(src)]
    return out


def candidates(prob: np.ndarray, image: np.ndarray, frst: np.ndarray, label: Optional[np.ndarray], s: Dict):
    """Arrays in NIfTI order (x, y, z). Returns rows (centre, voxels, max prob, touched reference ids), cubes
    (n, 3, z, y, x) float16 and the number of reference lesions."""
    ref, n_ref = ndimage.label(label > 0, S26) if label is not None else (None, 0)
    lab, n = ndimage.label(prob > float(s["low_threshold"]), S26)
    rows, cubes = [], []
    for k, box in enumerate(ndimage.find_objects(lab), 1):
        if box is None:
            continue
        m = lab[box] == k
        v = int(m.sum())
        if v < int(s["min_voxels"]) or v > int(s["max_voxels"]):
            continue
        centre = [int(round(c)) + b.start for c, b in zip(ndimage.center_of_mass(m), box)]
        touched = sorted(int(t) for t in np.unique(ref[box][m]) if t > 0) if ref is not None else []
        block = np.stack([crop_block(a, centre, s["half"]) for a in (image, frst, prob)])         # (3, x, y, z)
        cubes.append(np.transpose(block, (0, 3, 2, 1)).astype(np.float16))                       # -> (3, z, y, x)
        rows.append(dict(component=k, centre_voxel=centre, voxels=v, probability=round(float(prob[box][m].max()), 3),
                         touched=touched))
    shape = (0, 3, 2 * s["half"][2], 2 * s["half"][1], 2 * s["half"][0])
    return rows, (np.stack(cubes) if cubes else np.zeros(shape, np.float16)), int(n_ref), lab


def lesion_scores(rows: List[Dict], accepted: np.ndarray, n_ref: Dict[str, int]) -> Dict:
    tp = fp = 0
    for case in n_ref:
        idx = [i for i, r in enumerate(rows) if r["case"] == case]
        found = set()
        for i in idx:
            if accepted[i]:
                if rows[i]["touched"]:
                    found.update(rows[i]["touched"])
                else:
                    fp += 1
        tp += len(found)
    fn = sum(n_ref.values()) - tp
    return dict(tp=tp, fp=fp, fn=fn, f1=round(2 * tp / max(2 * tp + fp + fn, 1), 4),
                precision=round(tp / max(tp + fp, 1), 4), sensitivity=round(tp / max(tp + fn, 1), 4))


def _crop_center(x: np.ndarray, crop) -> np.ndarray:
    z0, y0, x0 = ((s - c) // 2 for s, c in zip(x.shape[-3:], crop))
    return x[..., z0:z0 + crop[0], y0:y0 + crop[1], x0:x0 + crop[2]]


def fit_members(cubes: np.ndarray, y: np.ndarray, s: Dict, device: str, init_states: Optional[List[Dict]] = None,
                seed: int = 0, log=print) -> List[nn.Module]:
    crop, jit = tuple(s["crop"]), tuple(s["jitter"])
    base = [(a - c) // 2 for a, c in zip(cubes.shape[-3:], crop)]
    pos = max(int((y == 1).sum()), 1)
    nets = []
    for member in range(int(s["members"])):
        torch.manual_seed(seed * 10 + member)
        rng = np.random.RandomState(seed * 10 + member)
        net = build_classifier()
        if init_states:
            net.load_state_dict(init_states[member % len(init_states)], strict=True)
        net = net.to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=float(s["lr"]), weight_decay=float(s["weight_decay"]))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(s["epochs"]))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float((y == 0).sum()) / pos], device=device))
        target = y.astype(np.float32)
        bs = int(s["batch_size"])
        for _ in range(int(s["epochs"])):
            net.train()
            order = rng.permutation(len(cubes))
            for start in range(0, len(order), bs):
                idx = order[start:start + bs]
                batch = []
                for i in idx:                                  # random offset around the centre, random flips
                    z0, y0, x0 = (b - j + rng.randint(0, 2 * j + 1) for b, j in zip(base, jit))
                    p = cubes[i][:, z0:z0 + crop[0], y0:y0 + crop[1], x0:x0 + crop[2]]
                    for axis in (1, 2, 3):
                        if rng.rand() < 0.5:
                            p = np.flip(p, axis)
                    batch.append(np.ascontiguousarray(p, dtype=np.float32))
                xb = torch.from_numpy(np.stack(batch)).to(device)
                yb = torch.from_numpy(target[idx]).to(device)
                opt.zero_grad(set_to_none=True)
                loss_fn(net(xb).squeeze(1), yb).backward()
                opt.step()
            sched.step()
        nets.append(net.eval())
    return nets


def score(nets: List[nn.Module], cubes: np.ndarray, crop, device: str, batch: int = 256) -> np.ndarray:
    if len(cubes) == 0:
        return np.zeros(0, np.float32)
    out = np.zeros(len(cubes), np.float32)
    with torch.no_grad():
        for start in range(0, len(cubes), batch):
            x = torch.from_numpy(np.ascontiguousarray(_crop_center(cubes[start:start + batch], crop), dtype=np.float32)).to(device)
            out[start:start + batch] = np.mean([torch.sigmoid(n(x).squeeze(1)).float().cpu().numpy() for n in nets], axis=0)
    return out


def load_bundle(source: str) -> Dict:
    """A classifier bundle (``{"members": [state, ...], "threshold": t, ...}``): a run folder, its
    stage2/classifier.pt, or the classifier.pt of the microbleed_detection_3D_UNET release (same architecture)."""
    path = source
    if os.path.isdir(source):
        path = os.path.join(source, "stage2", "classifier.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no second-stage classifier at {path} (train one with `cmbx stage2 <run>`)")
    b = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(b, dict) or "members" not in b or "threshold" not in b:
        raise ValueError(f"{path} is not a second-stage classifier bundle")
    if tuple(b.get("crop", DEFAULTS["crop"])) != tuple(DEFAULTS["crop"]):
        raise ValueError(f"{path}: crop {b.get('crop')} differs from {DEFAULTS['crop']}")
    b["path"] = os.path.abspath(path)
    return b


def _settings(cfg: Dict, overrides: Optional[Dict] = None) -> Dict:
    s = dict(DEFAULTS)
    s.update(cfg.get("stage2") or {})
    s.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return s


def run_candidates(run_dir: str, prepared_dir: str, s: Dict, log=print):
    from .evaluate import _cases_of_run
    cases = _cases_of_run(run_dir)
    if not cases:
        raise SystemExit(f"no finished CV folds in {run_dir}: the second stage learns from their out-of-fold maps")
    rows, cubes, n_ref = [], [], {}
    for c in cases:
        load = lambda n: np.asarray(nib.load(os.path.join(prepared_dir, c["id"], f"{c['id']}_{n}.nii.gz")).dataobj)
        if not os.path.exists(os.path.join(prepared_dir, c["id"], f"{c['id']}_frst.nii.gz")):
            raise SystemExit(f"{c['id']}: the second stage needs the FRST channel (preprocess.frst: true)")
        prob = np.asarray(nib.load(os.path.join(c["dir"], f"{c['id']}_pred_proba.nii.gz")).dataobj).astype(np.float32) / 255
        r, cb, nr, _ = candidates(prob, load("image").astype(np.float32), load("frst").astype(np.float32), load("label"), s)
        for x in r:
            x.update(case=c["id"], fold=c["fold"])
        rows += r
        cubes.append(cb)
        n_ref[c["id"]] = nr
    cubes = np.concatenate(cubes) if cubes else np.zeros((0,), np.float16)
    log(f"stage 2: {len(rows)} candidates in {len(n_ref)} cases ({sum(bool(r['touched']) for r in rows)} on a microbleed); "
        f"reachable {sum(len({t for r in rows if r['case'] == c for t in r['touched']}) for c in n_ref)} "
        f"of {sum(n_ref.values())} reference lesions")
    return rows, cubes, n_ref


def train_stage2(run_dir: str, device: str = "auto", init: Optional[str] = None, overrides: Optional[Dict] = None,
                 log=print) -> Dict:
    import yaml
    from .train import resolve_device
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
    s = _settings(cfg, overrides)
    device = resolve_device(device)
    t0 = time.time()
    init = init or s.get("init_from")
    init_states = load_bundle(init)["members"] if init else None
    if init:
        log(f"stage 2: fine-tuning from {init}")
    rows, cubes, n_ref = run_candidates(run_dir, cfg["data"]["prepared_dir"], s, log)
    y = np.array([1 if r["touched"] else 0 for r in rows], np.int64)
    if y.sum() == 0 or (y == 0).sum() == 0:
        raise SystemExit("stage 2 needs candidates of both kinds (on a microbleed and false ones)")
    folds = sorted({r["fold"] for r in rows})
    oof = np.zeros(len(rows), np.float32)
    for k in folds:                                            # nested: fold k is scored by a classifier that never saw it
        tr = np.array([r["fold"] != k for r in rows])
        nets = fit_members(cubes[tr], y[tr], s, device, init_states, seed=k, log=log)
        oof[~tr] = score(nets, cubes[~tr], s["crop"], device)
        log(f"stage 2: fold {k} scored ({int((~tr).sum())} candidates)")
    grid = {t: lesion_scores(rows, oof >= t, n_ref) for t in THRESHOLDS}
    thr = max(THRESHOLDS, key=lambda t: grid[t]["f1"])
    cv = grid[thr]
    stage1 = json.load(open(os.path.join(run_dir, "summary.json"))) if os.path.exists(os.path.join(run_dir, "summary.json")) else {}
    nets = fit_members(cubes, y, s, device, init_states, seed=99, log=log)       # the classifier to keep: all candidates
    out = os.path.join(run_dir, "stage2")
    os.makedirs(out, exist_ok=True)
    bundle = dict(members=[{k: v.cpu() for k, v in n.state_dict().items()} for n in nets], threshold=float(thr),
                  crop=list(s["crop"]), half=list(s["half"]), channels=CHANNELS, low_threshold=float(s["low_threshold"]),
                  min_voxels=int(s["min_voxels"]), max_voxels=int(s["max_voxels"]), init=init)
    torch.save(bundle, os.path.join(out, "classifier.pt"))
    summary = dict(stage1_cv_f1=stage1.get("f1"), stage2_cv=cv, threshold=thr, candidates=len(rows),
                   positives=int(y.sum()), cases=len(n_ref), settings=s, init=init, minutes=round((time.time() - t0) / 60, 1),
                   threshold_grid={str(t): grid[t]["f1"] for t in THRESHOLDS})
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    log(f"stage 2 [nested CV]: F1 {cv['f1']} (P {cv['precision']} S {cv['sensitivity']}, TP {cv['tp']} FP {cv['fp']} "
        f"FN {cv['fn']}) at score >= {thr}; stage 1 alone: F1 {stage1.get('f1')} -> {out}/classifier.pt")
    return summary


def apply_stage2(bundle: Dict, prob_xyz: np.ndarray, image_xyz: np.ndarray, frst_xyz: np.ndarray, device: str):
    """Accepted-candidate mask (x, y, z) and per-candidate rows (score) for one prepared case."""
    s = dict(DEFAULTS, low_threshold=bundle.get("low_threshold", DEFAULTS["low_threshold"]),
             min_voxels=bundle.get("min_voxels", DEFAULTS["min_voxels"]), max_voxels=bundle.get("max_voxels", DEFAULTS["max_voxels"]),
             half=bundle.get("half", DEFAULTS["half"]))
    rows, cubes, _, lab = candidates(prob_xyz, image_xyz, frst_xyz, None, s)
    nets = []
    for st in bundle["members"]:
        n = build_classifier()
        n.load_state_dict(st, strict=True)
        nets.append(n.to(device).eval())
    sc = score(nets, cubes, s["crop"], device)
    keep = [r["component"] for r, v in zip(rows, sc) if v >= float(bundle["threshold"])]
    for r, v in zip(rows, sc):
        r["stage2_score"] = round(float(v), 3)
    return np.isin(lab, keep), rows
