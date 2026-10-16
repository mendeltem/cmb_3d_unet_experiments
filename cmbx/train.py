"""5-fold cross-validation with inner-fold selection (port of cmb-valdo-chain ``code/cv5.py`` as driven by
``code/turnier.py --netz a03-aniso --epochen 60 --gitter dev/gitter_d``).

Per test fold k (nothing below sees fold k before its prediction):
  inner fold i = (k + 1) % n, training on the remaining folds;
  every ``eval_every`` epochs (and at the last epoch): sliding-window maps of the inner-fold cases -> pooled
  lesion F1 for every (threshold x min size) cell, with the clump guard -> best state = (epoch, threshold, min size);
  early stop after ``patience`` evaluations without improvement, but never before ``min_epochs``
  (default 60 = 60: no early stop);
  test fold k: maps with the best state, binarised with the inner choice, scored, saved.
Sampling per epoch: ``n_patches`` patches; ``lesion_fraction`` around a voxel of a reference component (component
drawn uniformly, so large bleeds do not dominate), ``brain_fraction`` around a random brain voxel, the rest
anywhere; the chosen voxel sits at a RANDOM position >= ``edge_offset`` voxels from the patch edge; random flips.
Loss 0.5 BCE + 0.5 soft Dice, AdamW, cosine annealing, bfloat16 autocast on CUDA.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .data import PreparedCase, fold_roles, load_prepared
from .metrics import hits, summarize, summarize_extended
from .model import build_model, count_parameters
from .postprocess import choose_cell, filter_components, grid_rows
from .predict import crop_patch, predict_volume


def resolve_device(name: str) -> str:
    """auto -> cuda if a GPU is visible (NVIDIA, or AMD through ROCm), else cpu. An explicit device is checked."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"device {name!r} requested but no CUDA/ROCm GPU is visible (use --device cpu or auto)")
    if name.startswith("cuda:") and int(name.split(":")[1]) >= torch.cuda.device_count():
        raise SystemExit(f"device {name!r} requested but only {torch.cuda.device_count()} GPU(s) visible (cuda:0 ...)")
    if name == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise SystemExit("device 'mps' requested but Apple MPS is not available")
    return name


# ---------------------------------------------------------------- sampling
def random_start(c, shape, rng: np.random.RandomState, patch, edge: int):
    """Patch start such that voxel ``c`` lies at a random position >= ``edge`` from the patch border (as far as the
    volume allows); 0 if the volume is smaller than the patch."""
    st = []
    for i in range(3):
        o = rng.randint(edge, max(patch[i] - edge, edge + 1))
        st.append(int(min(max(c[i] - o, 0), max(shape[i] - patch[i], 0))))
    return tuple(st)


def sample_patches(recs: Dict[str, PreparedCase], train_ids: List[str], rng: np.random.RandomState, T: Dict):
    """One epoch of patches; the order of random draws is that of ``cv5.stichprobe`` (same seed -> same patches)."""
    patch = tuple(T["patch_size"])
    comps = [(s, j) for s in train_ids for j in range(len(recs[s].components))]
    n = int(T["n_patches"])
    n_l, n_h = int(n * T["lesion_fraction"]), int(n * T["brain_fraction"])
    names = T["channels"]
    xs = np.empty((n, len(names)) + patch, np.float32)
    ys = np.empty((n, 1) + patch, np.float32)
    for j in range(n):
        if j < n_l and comps:
            s, k = comps[rng.randint(len(comps))]
            vox = recs[s].components[k]
            c = vox[rng.randint(len(vox))]
        else:
            s = train_ids[rng.randint(len(train_ids))]
            img = recs[s].image
            for _ in range(50):
                c = tuple(rng.randint(d) for d in img.shape)
                if j >= n_l + n_h or img[c] != 0:
                    break
        r = recs[s]
        st = random_start(c, r.image.shape, rng, patch, int(T["edge_offset"]))
        p = np.stack([crop_patch(a, st, patch) for a in r.channels(names)], 0)
        lab = crop_patch(r.label, st, patch)[None].astype(np.float32)
        if T["flip_augment"]:
            for ax in (1, 2, 3):
                if rng.rand() < 0.5:
                    p, lab = np.flip(p, ax), np.flip(lab, ax)
        xs[j], ys[j] = p, lab
    idx = rng.permutation(n)
    return EpochPatches(xs, ys, idx)


class EpochPatches:
    """The epoch's patches in drawn order plus the shuffle permutation. Batches are gathered through the
    permutation (``batch(b0, bs)`` == ``xs[idx][b0:b0+bs]`` of cv5.stichprobe) without a second full copy of the
    buffer (saves ~2 GB host RAM at 800 patches)."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray, idx: np.ndarray) -> None:
        self.xs, self.ys, self.idx = xs, ys, idx

    def __len__(self) -> int:
        return len(self.idx)

    def batch(self, b0: int, bs: int):
        j = self.idx[b0:b0 + bs]
        return torch.from_numpy(self.xs[j]), torch.from_numpy(self.ys[j])

    def all(self):
        return torch.from_numpy(self.xs[self.idx]), torch.from_numpy(self.ys[self.idx])


# ---------------------------------------------------------------- loss
def loss_bce_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """0.5 * BCE + 0.5 * (1 - soft Dice per patch, smoothing 1), mean over the batch."""
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    pf, tf = p.flatten(1), target.flatten(1)
    dice = 1 - (2 * (pf * tf).sum(1) + 1.0) / (pf.sum(1) + tf.sum(1) + 1.0)
    return 0.5 * bce + 0.5 * dice.mean()


def loss_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain binary cross-entropy (voxel mean)."""
    return F.binary_cross_entropy_with_logits(logits, target)


def loss_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - soft Dice per patch (smoothing 1), mean over the batch."""
    pf, tf = torch.sigmoid(logits).flatten(1), target.flatten(1)
    return (1 - (2 * (pf * tf).sum(1) + 1.0) / (pf.sum(1) + tf.sum(1) + 1.0)).mean()


def loss_focal_dice(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """0.5 * focal BCE (gamma 2, Lin et al. 2017) + 0.5 * soft Dice: down-weights the many easy background voxels."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.exp(-bce)
    return 0.5 * ((1 - pt) ** gamma * bce).mean() + 0.5 * loss_dice(logits, target)


LOSSES = {"bce_dice": loss_bce_dice, "bce": loss_bce, "dice": loss_dice, "focal_dice": loss_focal_dice}


# ---------------------------------------------------------------- one fold
def _check_time(t0: float, limit_s: Optional[float], what: str) -> None:
    if limit_s and time.time() - t0 > limit_s:
        raise RuntimeError(f"time limit {limit_s:.0f} s reached before {what}")


def _zyx_to_xyz(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a.transpose(2, 1, 0))


def train_fold(cfg: Dict, recs: Dict[str, PreparedCase], split: Dict, k: int, outdir: str, log=print) -> Dict:
    T, S, P, PP = cfg["train"], cfg["selection"], cfg["predict"], cfg["postprocess"]
    device = resolve_device(T["device"])
    test_ids, inner_ids, train_ids = fold_roles(split, k)
    assert test_ids and inner_ids and train_ids and not set(test_ids) & set(train_ids + inner_ids)
    seed = int(T["seed"]) + k
    t0 = time.time()
    limit = float(T["fold_time_limit_h"]) * 3600 if T.get("fold_time_limit_h") else None
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    nch = len(T["channels"])
    model = build_model(cfg["model"]["name"], nch, int(cfg["model"]["base_channels"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(T["lr"]), weight_decay=float(T["weight_decay"]))
    max_ep = int(T["epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    loss_fn = LOSSES[T["loss"]]
    use_amp = bool(T["amp_bf16"]) and device.startswith("cuda")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    csf = bool(PP["csf_rule"])
    patch = tuple(T["patch_size"])
    thr_grid, size_grid = [float(t) for t in S["thresholds"]], [int(m) for m in S["min_sizes"]]
    maxvox = int(S["max_component_voxels"])

    def prob(sid):
        return predict_volume(model, recs[sid].channels(T["channels"]), patch, device, int(P["batch_size"]),
                              bool(T["amp_bf16"]), bool(P["tta_flip"]))

    history, best, worse = [], None, 0
    for ep in range(1, max_ep + 1):
        _check_time(t0, limit, f"epoch {ep}")
        model.train()
        t_ep = time.time()
        losses = []
        patches = sample_patches(recs, train_ids, rng, T)
        bs = int(T["batch_size"])
        for b0 in range(0, len(patches), bs):
            x2, y2 = (t.to(device) for t in patches.batch(b0, bs))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(model(x2), y2)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()
        del patches
        row = dict(epoch=ep, loss=round(float(np.mean(losses)), 4), train_s=round(time.time() - t_ep, 1))
        if ep % int(T["eval_every"]) == 0 or ep == max_ep:
            _check_time(t0, limit, f"inner evaluation at epoch {ep}")
            t_k = time.time()
            rows = [grid_rows(prob(s)[0], recs[s].label, thr_grid, size_grid, maxvox,
                              recs[s].tissue if csf else None, PP["csf_labels"]) for s in inner_ids]
            thr, mind, s, clumps = choose_cell(rows, thr_grid, size_grid, S["fallback"])
            row.update(inner_f1=s["f1"], inner_sensitivity=s["sensitivity"], inner_precision=s["precision"],
                       inner_fp_per_case=s["fp_per_case"], threshold=thr, min_size=mind, clumps=clumps,
                       maps_s=round(time.time() - t_k, 1))
            if T.get("use_last_epoch") or best is None or s["f1"] > best["f1"] + 1e-9:
                best = dict(f1=s["f1"], epoch=ep, threshold=thr, min_size=mind, summary=s,
                            state={n: t.detach().cpu().clone() for n, t in model.state_dict().items()})
                worse = 0
            else:
                worse += 1
            log(f"[fold {k}] ep {ep:3d} loss {row['loss']:.3f}  inner F1 {s['f1']:.3f} (P {s['precision']:.2f} "
                f"S {s['sensitivity']:.2f} thr {thr} min {mind} clumps {clumps})  [{time.time() - t0:.0f} s]")
            history.append(row)
            if ep >= int(T["min_epochs"]) and worse >= int(T["patience"]):
                log(f"[fold {k}] early stop after epoch {ep} (best epoch {best['epoch']})")
                break
        else:
            history.append(row)
    json.dump(history, open(os.path.join(outdir, "history.json"), "w"), indent=1)
    model.load_state_dict(best["state"])
    thr, mind = best["threshold"], best["min_size"]

    _check_time(t0, limit, "test fold")
    rows, counter, n_tiles = [], {}, 0
    for sid in test_ids:
        r = recs[sid]
        p, nt = prob(sid)
        n_tiles += nt
        pb = filter_components(p, thr, mind, maxvox, counter, r.tissue if csf else None, PP["csf_labels"])
        row = dict(id=sid, fold=k, threshold=thr, min_size=mind, **hits(r.label, pb))
        rows.append(row)
        nib.save(nib.Nifti1Image(_zyx_to_xyz(pb.astype(np.uint8)), r.affine),
                 os.path.join(outdir, f"{sid}_pred.nii.gz"))
        nib.save(nib.Nifti1Image(_zyx_to_xyz(np.round(p * 255).astype(np.uint8)), r.affine),
                 os.path.join(outdir, f"{sid}_pred_proba.nii.gz"))
        json.dump(row, open(os.path.join(outdir, f"{sid}_pred_meta.json"), "w"), indent=1)
    torch.save({"state_dict": model.state_dict(),
                "meta": dict(model=cfg["model"], channels=T["channels"], patch_size=list(patch), threshold=thr,
                             min_size=mind, max_component_voxels=maxvox, epoch=best["epoch"])},
               os.path.join(outdir, "model.pt"))
    s_test = summarize_extended(rows)
    meta = dict(ok=True, fold=k, inner_fold=(k + 1) % split["n_folds"], seed=seed, device=device,
                epochs_run=history[-1]["epoch"], epoch_chosen=best["epoch"], threshold=float(thr),
                min_size=int(mind), inner_f1=best["f1"], inner_summary=best["summary"],
                test_f1=s_test["f1"], test_summary=s_test, clumps_test=counter,
                n_train=len(train_ids), n_inner=len(inner_ids), n_test=len(test_ids), windows_test=n_tiles,
                parameters=count_parameters(model), seconds=round(time.time() - t0, 1),
                vram_gib=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if device.startswith("cuda") else None)
    json.dump(meta, open(os.path.join(outdir, "meta.json"), "w"), indent=1)
    log(f"[fold {k}] DONE test F1 {s_test['f1']} (P {s_test['precision']} S {s_test['sensitivity']}, thr {thr}, "
        f"min {mind}, epoch {best['epoch']}) {meta['seconds']:.0f} s, VRAM {meta['vram_gib']} GiB")
    return meta


def fold_dir(run_dir: str, k: int):
    """``(folder, done)``. Only a fold with ``meta.json: ok`` counts as done; anything else is moved aside, so maps
    of an old model and metadata of a new one are never mixed."""
    outd = os.path.join(run_dir, f"fold{k}")
    mp = os.path.join(outd, "meta.json")
    if os.path.exists(mp) and json.load(open(mp)).get("ok"):
        return outd, True
    if os.path.isdir(outd) and os.listdir(outd):
        aside = f"{outd}_incomplete-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(outd, aside)
    os.makedirs(outd, exist_ok=True)
    return outd, False


def run_cv(cfg: Dict, split: Dict, run_dir: str, folds=None, log=print) -> List[Dict]:
    """Train and test every requested fold (resumable). Loads all CV cases once (RAM ~ 120 MB per VALDO case)."""
    T = cfg["train"]
    np.random.seed(int(T["seed"]))
    ids = [s for f in split["folds"] for s in f]
    prepared = cfg["data"]["prepared_dir"]
    recs = {s: load_prepared(prepared, s, T["channels"], tissue=bool(cfg["postprocess"]["csf_rule"])) for s in ids}
    n_comp = sum(len(r.components) for r in recs.values())
    log(f"data: {len(recs)} cases, {n_comp} reference components, channels {T['channels']}")
    metas = []
    for k in (folds if folds is not None else range(split["n_folds"])):
        outd, done = fold_dir(run_dir, k)
        if done:
            log(f"[fold {k}] already done -- skipped")
            metas.append(json.load(open(os.path.join(outd, "meta.json"))))
            continue
        metas.append(train_fold(cfg, recs, split, k, outd, log))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return metas
