"""``cmbx train-final``: ONE model trained on ALL cases of the case list -- the model you keep and use.

The cross-validation (``cmbx cv``) gives the honest score but five models, each trained on 3/5 of the data. The final
model sees every case. There is no case left to choose an epoch, so the state after the LAST epoch of the fixed
schedule is kept (as for the released models of microbleed_detection_3D_UNET, cmb-valdo-chain
``cmb/training/full_data.py``; last instead of best inner state measured there: -0.023 F1, within noise). The
operating cell (threshold, minimum size) is the one the CV folds chose most often; without a CV run the defaults
below are used. Everything else is the training recipe of the CV (same sampling, loss, optimizer, schedule).

Output: ``<run>/final/model.pt`` (weights + everything needed to apply it: model, channels, patch, cell,
preprocessing settings) and ``<run>/final/meta.json``. ``cmbx predict --run <run>`` uses it automatically.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import time
from typing import Dict

import numpy as np
import torch

from .data import load_prepared, read_cases
from .model import build_model, count_parameters
from .train import LOSSES, resolve_device, sample_patches

DEFAULT_CELL = (0.5, 2)        # used only when no CV fold of this run exists


def cv_cell(run_dir: str):
    metas = [json.load(open(p)) for p in glob.glob(os.path.join(run_dir, "fold[0-9]", "meta.json"))]
    metas = [m for m in metas if m.get("ok")]
    if not metas:
        return None, 0
    (t, m), n = collections.Counter((x["threshold"], x["min_size"]) for x in metas).most_common(1)[0]
    return (float(t), int(m)), len(metas)


def train_final(cfg: Dict, run_dir: str, log=print, overwrite: bool = False) -> Dict:
    T, S = cfg["train"], cfg["selection"]
    out = os.path.join(run_dir, "final")
    os.makedirs(out, exist_ok=True)
    if not overwrite and os.path.exists(os.path.join(out, "meta.json")) and \
            json.load(open(os.path.join(out, "meta.json"))).get("ok"):
        log(f"final model exists: {out}/model.pt (train it again with `cmbx train-final <config> --overwrite`)")
        return json.load(open(os.path.join(out, "meta.json")))
    ids = [c["id"] for c in read_cases(cfg["data"]["cases"])]
    recs = {s: load_prepared(cfg["data"]["prepared_dir"], s, T["channels"]) for s in ids}
    cell, n_folds = cv_cell(run_dir)
    if cell is None:
        cell = DEFAULT_CELL
        log(f"no CV folds in {run_dir}: operating cell = default {cell} (run the CV first for a data-driven cell)")
    device = resolve_device(T["device"])
    seed = int(T["seed"])
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = build_model(cfg["model"]["name"], len(T["channels"]), int(cfg["model"]["base_channels"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(T["lr"]), weight_decay=float(T["weight_decay"]))
    max_ep = int(T["epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    loss_fn = LOSSES[T["loss"]]
    use_amp = bool(T["amp_bf16"]) and device.startswith("cuda")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0, history = time.time(), []
    log(f"final model: {len(ids)} cases, {max_ep} epochs, cell {cell} ({'from ' + str(n_folds) + ' CV folds' if n_folds else 'default'})")
    for ep in range(1, max_ep + 1):
        model.train()
        patches, losses, bs = sample_patches(recs, ids, rng, T), [], int(T["batch_size"])
        for b0 in range(0, len(patches), bs):
            x, y = (t.to(device) for t in patches.batch(b0, bs))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()
        history.append(dict(epoch=ep, loss=round(float(np.mean(losses)), 4)))
        if ep % int(T["eval_every"]) == 0 or ep == max_ep:
            log(f"[final] ep {ep:3d} loss {history[-1]['loss']:.3f}  [{time.time() - t0:.0f} s]")
    meta = dict(ok=True, cases=len(ids), epochs=max_ep, seed=seed, threshold=cell[0], min_size=cell[1],
                cell_source=f"most frequent cell of {n_folds} CV folds" if n_folds else "default",
                max_component_voxels=int(S["max_component_voxels"]), channels=T["channels"],
                patch_size=list(T["patch_size"]), model=cfg["model"], preprocess=cfg["preprocess"],
                parameters=count_parameters(model), seconds=round(time.time() - t0, 1), device=device,
                vram_gib=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if device.startswith("cuda") else None)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, os.path.join(out, "model.pt"))
    json.dump(history, open(os.path.join(out, "history.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    log(f"final model saved: {out}/model.pt ({meta['seconds'] / 60:.0f} min)")
    return meta
