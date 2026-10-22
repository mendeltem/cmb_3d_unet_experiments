"""Fine-tune a trained 3D U-Net, or the pretrained VALDO model, on your own cases, step by step in plain Python.

    python examples/python/03_fine_tune_unet.py -i examples/train/derivatives/preprocessed -m pretrained -o ../my_runs/py_finetuned

The steps of `cmbx fine_tune --frst_from cmbx`: start from the weights of -m, keep its network, input channels, grid,
patch size and recipe, and train on your cases with the learning rate 0.0001 (MicrobleedNet's fine_tune default, a third
of the rate for training from scratch). With the same settings this script trains the same weights as that command
(tests/test_python_examples.py checks it). Steps 2, 3, 4, 6 and 7 are those of 01_train_unet.py, and so are the
functions (copies; a test keeps them identical). New are step 1 (settings from the start model) and step 5 (start
weights, optionally only some levels trained).

-m      a model folder from 01_train_unet.py, this script or cmbx train, or `pretrained`: the VALDO T2* model
        (CC BY-NC-SA 4.0, downloaded once into models/valdo_t2)
-i, -o  as -i and -m of 01_train_unet.py; -o is a model folder for 02_predict_microbleeds.py and `cmbx evaluate`
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from cmbx.config import dump, load_config
from cmbx.data import load_prepared
from cmbx.final import holdout_split
from cmbx.model import build_model, load_state
from cmbx.postprocess import choose_cell, grid_rows
from cmbx.predict import predict_volume
from cmbx.preprocess import prepare_all
from cmbx.train import LOSSES, apply_finetune_layers, epoch_patches, resolve_device
from cmbx.workflow import collect, pretrained_dir

DEFAULT_CELL = (0.5, 2)        # threshold and minimum size when no held-out case has a microbleed


def prepare_grid(cases, folder, settings, workers=4, force=False):
    """Bring every case onto the model grid in `folder`. A case already there is kept, unless the settings changed or
    force is set. Stops with the names of the cases that failed."""
    signature = json.loads(json.dumps({k: v for k, v in settings.items() if k != "workers"}))
    signature_file = os.path.join(folder, "preprocess.json")
    changed = os.path.exists(signature_file) and json.load(open(signature_file)) != signature
    os.makedirs(folder, exist_ok=True)
    results = prepare_all(cases, folder, settings, workers, overwrite=force or changed, log=print)
    failed = [r["id"] for r in results if "error" in r]
    if failed:
        raise SystemExit(f"{len(failed)} case(s) could not be prepared: {failed} (see {folder}/<name>/error.json)")
    json.dump(signature, open(signature_file, "w"), indent=1)


def train_one_epoch(net, data, train_ids, rng, T, optimizer, device):
    """One epoch: draw T["n_patches"] patches (60 % around a microbleed, 25 % in the brain, the rest anywhere; random
    flips), then one optimizer step per batch. Returns the mean loss."""
    loss_function = LOSSES[T["loss"]]          # bce_dice = 0.5 BCE + 0.5 soft Dice (cmbx/train.py); or your own
    use_bf16 = bool(T["amp_bf16"]) and device.startswith("cuda")
    net.train()
    patches = epoch_patches(data, train_ids, rng, T)
    losses = []
    for start in range(0, len(patches), int(T["batch_size"])):
        x, y = (t.to(device) for t in patches.batch(start, int(T["batch_size"])))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = loss_function(net(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    return float(np.mean(losses))


def choose_threshold(net, data, held_out, cfg, device):
    """Predict the held-out cases and return the threshold and minimum size (voxels) with the best lesion-level F1,
    and the scores at that cell (f1, precision, sensitivity, ...)."""
    T, S, P, PP = cfg["train"], cfg["selection"], cfg["predict"], cfg["postprocess"]
    thresholds, min_sizes = [float(t) for t in S["thresholds"]], [int(m) for m in S["min_sizes"]]
    csf = bool(PP["csf_rule"])
    rows = []
    for name in held_out:
        prob = predict_volume(net, data[name].channels(T["channels"]), tuple(T["patch_size"]), device,
                              int(P["batch_size"]), bool(T["amp_bf16"]), bool(P["tta_flip"]))[0]
        rows.append(grid_rows(prob, data[name].label, thresholds, min_sizes, int(S["max_component_voxels"]),
                              data[name].tissue if csf else None, PP["csf_labels"]))
    threshold, min_size, scores, _ = choose_cell(rows, thresholds, min_sizes, S["fallback"])
    return threshold, min_size, scores


def save_model(net, cfg, model_dir, meta, history):
    """The model folder: config.yaml, final/model.pt (weights + meta), final/meta.json, final/history.json."""
    os.makedirs(os.path.join(model_dir, "final"), exist_ok=True)
    dump(cfg, os.path.join(model_dir, "config.yaml"))
    weights = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save({"state_dict": weights, "meta": meta}, os.path.join(model_dir, "final", "model.pt"))
    json.dump(meta, open(os.path.join(model_dir, "final", "meta.json"), "w"), indent=1)
    json.dump(history, open(os.path.join(model_dir, "final", "history.json"), "w"), indent=1)


if __name__ == "__main__":     # the steps; worker processes of step 3 import this file and must not run them again
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("-i", "--inp_dir", required=True, help="folder with one sub-folder per case")
    p.add_argument("-m", "--model_name", required=True, help="model folder to start from, or pretrained")
    p.add_argument("-o", "--output_dir", required=True, help="where the fine-tuned model goes")
    p.add_argument("--epochs", type=int, default=None, help="[as the start model: 60]")
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate [0.0001]")
    p.add_argument("--levels", type=int, nargs="+", default=None,
                   help="train only these levels, 1 = output ... 5 = bottleneck ... 9 = input [all]")
    p.add_argument("--train_prop", type=float, default=0.8,
                   help="share of the cases trained on; the rest choose the threshold [0.8]")
    p.add_argument("--device", default=None, help="auto | cuda | cuda:N | cpu [auto]")
    p.add_argument("--config", default=None, help="cmbx YAML with settings to change on top of the start model's")
    p.add_argument("--workers", type=int, default=4, help="cases prepared in parallel in step 3 [4]")
    p.add_argument("--force", default="false", help="true: train again into a folder that holds a model [false]")
    a = p.parse_args()
    if os.path.exists(os.path.join(a.output_dir, "final", "model.pt")) and a.force != "true":
        raise SystemExit(f"{a.output_dir} holds a model already; --force true trains it again")

    # 1 settings: those of the start model (network, channels, grid, patch, recipe), then --config, then the options
    start = pretrained_dir() if a.model_name in ("pre", "pretrained") else a.model_name
    if not os.path.exists(os.path.join(start, "final", "model.pt")):
        raise SystemExit(f"{start}/final/model.pt not found: give a model folder or pretrained")
    cfg = load_config(os.path.join(start, "config.yaml"), yaml.safe_load(open(a.config)) if a.config else None)
    T = cfg["train"]
    for key, value in dict(epochs=a.epochs, lr=a.lr, train_prop=a.train_prop, device=a.device).items():
        if value is not None:
            T[key] = value
    T["init_from"], T["finetune_layers"] = os.path.abspath(start), list(a.levels or [])
    grid = os.path.abspath(os.path.join(a.output_dir, "grid"))
    cfg["data"]["prepared_dir"] = grid
    csf = bool(cfg["postprocess"]["csf_rule"])
    print(f"start model {start}: channels {T['channels']}, patch {T['patch_size']}, lr {T['lr']}", flush=True)

    # 2 the cases: scan, brain mask and microbleed mask of each
    cases = collect(a.inp_dir, need_labels=True, need_tissue=csf, need_flair="flair" in T["channels"])
    ids = [c["id"] for c in cases]
    print(f"{len(ids)} cases in {a.inp_dir}", flush=True)

    # 3 every case onto the grid of the start model; kept in <output>/grid
    prepare_grid(cases, grid, cfg["preprocess"], a.workers, a.force == "true")
    data = {name: load_prepared(grid, name, T["channels"], tissue=csf) for name in ids}

    # 4 hold out cases to choose the threshold (with and without microbleeds in proportion), train on the rest
    lesions = {name: len(data[name].components) for name in ids}
    train_ids, held_out = holdout_split(ids, T["train_prop"], int(T["seed"]), lesions)
    print(f"training on {len(train_ids)} cases, {len(held_out)} held out: {held_out}", flush=True)

    # 5 the start model's network and weights; with --levels the other levels stay frozen
    device = resolve_device(T["device"])
    torch.manual_seed(int(T["seed"]))
    rng = np.random.RandomState(int(T["seed"]))
    net = build_model(cfg["model"]["name"], len(T["channels"]), int(cfg["model"]["base_channels"]))
    load_state(net, os.path.join(start, "final", "model.pt"))
    apply_finetune_layers(net, T["finetune_layers"])
    net = net.to(device)
    trainable = [w for w in net.parameters() if w.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(T["lr"]), weight_decay=float(T["weight_decay"]))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(T["epochs"]))

    # 6 train; every eval_every epochs (and after the last) predict the held-out cases and choose the threshold
    t0, history = time.time(), []
    for epoch in range(1, int(T["epochs"]) + 1):
        row = dict(epoch=epoch, loss=round(train_one_epoch(net, data, train_ids, rng, T, optimizer, device), 4))
        schedule.step()
        if held_out and (epoch % int(T["eval_every"]) == 0 or epoch == int(T["epochs"])):
            threshold, min_size, scores = choose_threshold(net, data, held_out, cfg, device)
            row.update(held_out_f1=scores["f1"], threshold=threshold, min_size=min_size)
        history.append(row)
        print(f"epoch {epoch:3d}  loss {row['loss']:.4f}"
              + (f"  held-out F1 {row['held_out_f1']:.3f} at threshold {row['threshold']}, min size {row['min_size']}"
                 if "held_out_f1" in row else "") + f"  [{time.time() - t0:.0f} s]", flush=True)

    # 7 keep the last state with the threshold of the last held-out evaluation, and save the model folder
    if any(lesions[name] for name in held_out):
        threshold, min_size = history[-1]["threshold"], history[-1]["min_size"]
    else:
        threshold, min_size = DEFAULT_CELL
        print(f"no microbleed in the held-out cases: default threshold {threshold}, minimum size {min_size}")
    meta = dict(ok=True, threshold=float(threshold), min_size=int(min_size), cases=len(ids), train_cases=len(train_ids),
                held_out=held_out, held_out_f1=history[-1].get("held_out_f1"), epochs=int(T["epochs"]),
                channels=T["channels"], patch_size=list(T["patch_size"]), device=device, init_from=T["init_from"],
                max_component_voxels=int(cfg["selection"]["max_component_voxels"]),
                seconds=round(time.time() - t0, 1), written_by="examples/python/03_fine_tune_unet.py")
    save_model(net, cfg, a.output_dir, meta, history)
    print(f"model saved: {a.output_dir}/final/model.pt (threshold {threshold}, minimum size {min_size})")
