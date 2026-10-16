"""``cmbx train-final``: ONE model trained on ALL cases of the case list -- the model you keep and use.

The cross-validation (``cmbx cv``) gives the honest score but five models, each trained on 3/5 of the data. The final
model sees every case. There is no case left to choose an epoch, so the state after the LAST epoch of the fixed
schedule is kept (as for the released models of microbleed_detection_3D_UNET, cmb-valdo-chain
``cmb/training/full_data.py``; last instead of best inner state measured there: -0.023 F1, within noise). The
operating cell (threshold, minimum size) is the one the CV folds chose most often; without a CV run the defaults
below are used. Everything else is the training recipe of the CV (same sampling, loss, optimizer, schedule).

With ``train.train_prop`` < 1 (MicrobleedNet ``-tr_prop``; ``cmbx train`` uses 0.8) a random share of the cases is held
out: every ``eval_every`` epochs it is predicted, the operating cell is chosen on it, ``checkpoint: best`` keeps the best
state, ``early_stop`` stops after that many epochs without gain. Without held-out cases the cell is the CV choice or
the default, and the last state is kept.

Output: ``<run>/final/model.pt`` (weights + everything needed to apply it: model, channels, patch, cell,
preprocessing settings) and ``<run>/final/meta.json``; with held-out cases also ``model_best.pt``; ``everyN`` adds
``model_epoch<N>.pt``, ``save_full_model`` adds ``model_full.pt``. ``cmbx predict --run <run>`` uses model.pt.
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
from .postprocess import choose_cell, grid_rows
from .predict import predict_volume
from .train import (LOSSES, apply_finetune_layers, epoch_patches, make_optimizer, make_scheduler,
                    resolve_device, unwrap, wrap_multi_gpu)

DEFAULT_CELL = (0.5, 2)        # used only when no CV fold of this run exists


def cv_cell(run_dir: str):
    metas = [json.load(open(p)) for p in glob.glob(os.path.join(run_dir, "fold[0-9]", "meta.json"))]
    metas = [m for m in metas if m.get("ok")]
    if not metas:
        return None, 0
    (t, m), n = collections.Counter((x["threshold"], x["min_size"]) for x in metas).most_common(1)[0]
    return (float(t), int(m)), len(metas)


def holdout_split(ids, train_prop: float, seed: int, lesions: Dict[str, int] = None):
    """(train ids, held-out ids): ``max(int(n * (1 - train_prop)), 1)`` cases held out (MicrobleedNet's count), drawn
    with a fixed seed; none if train_prop >= 1. With ``lesions`` (reference lesions per case) cases with and without
    microbleeds are held out in proportion, at least one with microbleeds: a held-out set without any reference lesion
    has F1 0 at every cell and cannot choose the threshold."""
    if float(train_prop) >= 1.0:
        return list(ids), []
    n_val = max(int(len(ids) * (1.0 - float(train_prop)) + 1e-9), 1)
    if n_val >= len(ids):
        raise SystemExit(f"train_prop {train_prop}: no case left for training ({len(ids)} cases)")
    rng = np.random.RandomState(seed)
    if lesions:
        pos = [s for s in ids if lesions.get(s, 0) > 0]
        neg = [s for s in ids if lesions.get(s, 0) == 0]
        k = min(max(1, int(round(n_val * len(pos) / len(ids)))), len(pos), n_val) if pos else 0
        k = max(k, n_val - len(neg))
        pick = set(rng.choice(pos, k, replace=False).tolist()) if k else set()
        if n_val - k:
            pick |= set(rng.choice(neg, n_val - k, replace=False).tolist())
    else:
        pick = {ids[i] for i in rng.choice(len(ids), n_val, replace=False)}
    return [s for s in ids if s not in pick], [s for s in ids if s in pick]


def _state(model):
    return {n: t.detach().cpu().clone() for n, t in unwrap(model).state_dict().items()}


def train_final(cfg: Dict, run_dir: str, log=print, overwrite: bool = False) -> Dict:
    T, S, P, PP = cfg["train"], cfg["selection"], cfg["predict"], cfg["postprocess"]
    out = os.path.join(run_dir, "final")
    os.makedirs(out, exist_ok=True)
    if not overwrite and os.path.exists(os.path.join(out, "meta.json")) and \
            json.load(open(os.path.join(out, "meta.json"))).get("ok"):
        log(f"final model exists: {out}/model.pt (train it again with `cmbx train-final <config> --overwrite`)")
        return json.load(open(os.path.join(out, "meta.json")))
    ids = [c["id"] for c in read_cases(cfg["data"]["cases"])]
    seed = int(T["seed"])
    csf = bool(PP["csf_rule"])
    recs = {s: load_prepared(cfg["data"]["prepared_dir"], s, T["channels"], tissue=csf) for s in ids}
    train_ids, val_ids = holdout_split(ids, T.get("train_prop", 1.0), seed, {s: len(recs[s].components) for s in ids})
    ckpt, every, patience = T.get("checkpoint", "last"), int(T.get("checkpoint_every", 10)), int(T.get("early_stop") or 0)
    if ckpt not in ("last", "best", "everyN"):
        raise SystemExit(f"train.checkpoint must be last, best or everyN; got {ckpt!r}")
    if not val_ids and (ckpt == "best" or patience):
        raise SystemExit("checkpoint 'best' and early stopping need held-out cases: set -tr_prop below 1 (e.g. 0.8)")
    cell, n_folds = cv_cell(run_dir)
    if cell is None and not val_ids:
        cell = DEFAULT_CELL
        log(f"no CV folds in {run_dir} and no held-out cases: operating cell = default {cell}")
    device = resolve_device(T["device"])
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = build_model(cfg["model"]["name"], len(T["channels"]), int(cfg["model"]["base_channels"]))
    if T.get("init_from"):                                    # fine-tuning from a pretrained model
        from .model import init_from_pretrained
        init_from_pretrained(model, T["init_from"], T["channels"], T.get("init_channels"), log)
    apply_finetune_layers(model, T.get("finetune_layers"), log)
    model = wrap_multi_gpu(model.to(device), T, device, log)
    opt = make_optimizer([p for p in model.parameters() if p.requires_grad], T)
    max_ep = int(T["epochs"])
    sched = make_scheduler(opt, T, max_ep)
    loss_fn = LOSSES[T["loss"]]
    use_amp = bool(T["amp_bf16"]) and device.startswith("cuda")
    patch = tuple(T["patch_size"])
    thr_grid, size_grid, maxvox = [float(t) for t in S["thresholds"]], [int(m) for m in S["min_sizes"]], int(S["max_component_voxels"])
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0, history, best, last_eval, start = time.time(), [], None, None, 1
    resume_path = os.path.join(out, "resume.pt")
    if T.get("resume") and os.path.exists(resume_path):        # MicrobleedNet -sv_resume: continue an interrupted run
        r = torch.load(resume_path, map_location="cpu", weights_only=False)
        unwrap(model).load_state_dict(r["model"])
        opt.load_state_dict(r["opt"])
        if sched is not None and r.get("sched"):
            sched.load_state_dict(r["sched"])
        rng.set_state(r["rng"])
        torch.set_rng_state(r["torch_rng"])
        history, best, last_eval, start = r["history"], r["best"], r["last_eval"], r["epoch"] + 1
        log(f"resuming from {resume_path} after epoch {r['epoch']}")
    log(f"final model: {len(train_ids)} training cases, {len(val_ids)} held out, {max_ep} epochs, checkpoint {ckpt}"
        + (f", cell {cell} ({'from ' + str(n_folds) + ' CV folds' if n_folds else 'default'})" if not val_ids else ""))
    for ep in range(start, max_ep + 1):
        model.train()
        patches, losses, bs = epoch_patches(recs, train_ids, rng, T), [], int(T["batch_size"])
        for b0 in range(0, len(patches), bs):
            x, y = (t.to(device) for t in patches.batch(b0, bs))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        if sched is not None:
            sched.step()
        del patches
        row = dict(epoch=ep, loss=round(float(np.mean(losses)), 4))
        stop = False
        if val_ids and (ep % int(T["eval_every"]) == 0 or ep == max_ep):
            model.eval()
            rows = [grid_rows(predict_volume(model, recs[s].channels(T["channels"]), patch, device, int(P["batch_size"]),
                                             bool(T["amp_bf16"]), bool(P["tta_flip"]))[0], recs[s].label, thr_grid, size_grid,
                              maxvox, recs[s].tissue if csf else None, PP["csf_labels"]) for s in val_ids]
            thr, mind, sm, _ = choose_cell(rows, thr_grid, size_grid, S["fallback"])
            last_eval = dict(epoch=ep, f1=sm["f1"], threshold=thr, min_size=mind)
            row.update(val_f1=sm["f1"], val_precision=sm["precision"], val_sensitivity=sm["sensitivity"], threshold=thr, min_size=mind)
            if best is None or sm["f1"] > best["f1"] + 1e-9:
                best = dict(last_eval, state=_state(model))
                torch.save({"state_dict": best["state"], "meta": dict(epoch=ep, threshold=thr, min_size=mind)},
                           os.path.join(out, "model_best.pt"))
            log(f"[final] ep {ep:3d} loss {row['loss']:.3f}  held-out F1 {sm['f1']:.3f} (P {sm['precision']:.2f} "
                f"S {sm['sensitivity']:.2f} thr {thr} min {mind}; best {best['f1']:.3f} at ep {best['epoch']})  [{time.time() - t0:.0f} s]")
            stop = bool(patience) and ep - best["epoch"] >= patience
        elif ep % int(T["eval_every"]) == 0 or ep == max_ep:
            log(f"[final] ep {ep:3d} loss {row['loss']:.3f}  [{time.time() - t0:.0f} s]")
        history.append(row)
        if ckpt == "everyN" and ep % every == 0:
            c = last_eval or dict(threshold=cell[0], min_size=cell[1])
            torch.save({"state_dict": _state(model), "meta": dict(epoch=ep, threshold=c["threshold"], min_size=c["min_size"])},
                       os.path.join(out, f"model_epoch{ep}.pt"))
        if T.get("resume"):
            torch.save(dict(model=_state(model), opt=opt.state_dict(), sched=sched.state_dict() if sched is not None else None,
                            rng=rng.get_state(), torch_rng=torch.get_rng_state(), history=history, best=best,
                            last_eval=last_eval, epoch=ep), resume_path)
        if stop:
            log(f"[final] early stop after epoch {ep}: no held-out gain for {patience} epochs (best epoch {best['epoch']})")
            break
    model = unwrap(model)
    if val_ids and not any(len(recs[s].components) for s in val_ids):
        if ckpt == "best":
            model.load_state_dict(best["state"])
        cell, cell_source = cell or DEFAULT_CELL, "default (the held-out cases have no reference lesion)"
        log("WARNING: the held-out cases contain no reference microbleed; operating cell = default")
    elif val_ids:
        chosen = best if ckpt == "best" else last_eval
        if ckpt == "best":
            model.load_state_dict(best["state"])
        cell, cell_source = (chosen["threshold"], chosen["min_size"]), f"held-out cases, epoch {chosen['epoch']} ({ckpt} state)"
    else:
        cell_source = f"most frequent cell of {n_folds} CV folds" if n_folds else "default"
    meta = dict(ok=True, cases=len(ids), train_cases=len(train_ids), held_out=val_ids, epochs=max_ep,
                epochs_run=history[-1]["epoch"], checkpoint=ckpt, seed=seed, threshold=float(cell[0]), min_size=int(cell[1]),
                cell_source=cell_source, held_out_f1=(best if ckpt == "best" else last_eval or {}).get("f1"),
                max_component_voxels=maxvox, channels=T["channels"],
                patch_size=list(T["patch_size"]), model=cfg["model"], preprocess=cfg["preprocess"],
                parameters=count_parameters(model), seconds=round(time.time() - t0, 1), device=device,
                vram_gib=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if device.startswith("cuda") else None)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, os.path.join(out, "model.pt"))
    if T.get("save_full_model"):
        torch.save(model.cpu(), os.path.join(out, "model_full.pt"))
    json.dump(history, open(os.path.join(out, "history.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    if os.path.exists(resume_path):
        os.remove(resume_path)
    log(f"final model saved: {out}/model.pt, cell {cell} from {cell_source} ({meta['seconds'] / 60:.0f} min)")
    return meta
