"""Experiment configuration: defaults = the winner recipe of cmb-valdo-chain, YAML files override single keys.

The defaults reproduce exactly the run ``python code/turnier.py --netz a03-aniso --epochen 60 --gitter dev/gitter_d``
(result folder ``a03-aniso-e60-gd``): cv5.REZEPT + the tournament overrides (InstanceNorm, network a03-aniso,
loss bce_dice_ds == bce_dice for a single-output network) + ``--epochen 60`` (max = min epochs = 60, so early
stopping never triggers; the inner fold is evaluated every 60 // 12 = 5 epochs) + the 0.5 x 0.5 x 1.0 mm grid
WITHOUT vessel inpainting (gitter_d). See README "Provenance of every default".
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

import yaml

DEFAULTS: Dict[str, Any] = {
    "name": "baseline",
    "output_root": "runs",                       # results land in <output_root>/<name>/
    "data": {
        "cases": "data/cases.csv",               # case list (CSV or JSON), see cmbx.data
        "prepared_dir": "data/prepared",         # where `cmbx prepare` writes the grid volumes
        "splits": "data/splits.json",            # fold assignment, written by `cmbx split`
        "n_folds": 5,
        "split_seed": 0,
    },
    "preprocess": {
        "reorient": "none",                      # none = keep the axis order of the file (as in training) | ras
        "brain_mask_mode": "mask",               # mask (VALDO grid) | mask_and_positive (in-house grids)
        "spacing": [0.5, 0.5, 1.0],              # mm, (x, y, z) of the array; z = slice axis
        "margin": 8,                             # voxels around the brain bounding box
        "interpolation_order": 1,                # image resampling order (label: nearest + rescue)
        "frst": True,                            # compute the FRST channel (needed unless channels == [image])
        "extra_channels": {},                    # own extra input channels: {name: case-list column with a NIfTI on the
                                                 # image matrix}; brought onto the grid, scaled to [0, 1] in the brain,
                                                 # then usable as train.channels: [image, frst, <name>]
        "inpaint": {                             # EXPERIMENT ENTRY POINT, see cmbx.preprocess.inpaint
            "method": "none",                    # none | mean | biharmonic | "package.module:function"
            "mask_column": "inpaint_mask",       # case-list column with the mask to paint over
            "ring_voxels": 2,                    # mean fill: shell thickness used for the fill value
            "dilate_voxels": 0,                  # grow the mask before painting
        },
        "workers": 4,                            # parallel cases (one process each, one thread each)
    },
    "model": {"name": "aniso_unet3d", "base_channels": 32},
    "train": {
        "channels": ["image", "frst"],           # ["image"] = the no-FRST control
        "epochs": 60,                            # max epochs (= --epochen 60)
        "min_epochs": 60,                        # early stop only from here on (60 = never)
        "eval_every": 5,                         # inner-fold evaluation interval (60 // 12)
        "patience": 3,                           # evaluations without improvement before early stop
        "n_patches": 800,                        # patches drawn per epoch
        "batch_size": 4,
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,                  # AdamW, cosine annealing over `epochs`
        "loss": "bce_dice",                      # bce_dice (0.5 BCE + 0.5 soft Dice) | bce | dice | focal_dice
        "lesion_fraction": 0.6,                  # patches containing a lesion voxel (random offset)
        "brain_fraction": 0.25,                  # patches centred on a random brain voxel; rest anywhere
        "flip_augment": True,                    # random flips per axis (the only augmentation)
        "patch_size": [38, 62, 94],              # (z, y, x) voxels = 38 x 31 x 47 mm
        "edge_offset": 4,                        # the sampled voxel is >= 4 voxels from the patch edge
        "seed": 42,                              # per fold: seed + fold
        "amp_bf16": True,                        # bfloat16 autocast on CUDA (as in the original)
        "fold_time_limit_h": 6.0,
        "device": "auto",                        # auto | cuda | cpu
        "use_last_epoch": False,                 # True: test with the last state instead of the inner-fold best
    },
    "selection": {                               # chosen on the INNER fold, never on the test fold
        "thresholds": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "min_sizes": [1, 2, 3, 5],               # voxels (0.25 mm3 each)
        "fallback": [0.5, 1],                    # used if every cell has F1 = 0
        "max_component_voxels": 2100,            # clump guard: > 10 mm sphere is not a microbleed
    },
    "predict": {"batch_size": 8, "tta_flip": False},
    "postprocess": {
        "csf_rule": False,                       # drop components whose majority SynthSeg label is CSF
        "csf_labels": [4, 5, 14, 15, 24, 43, 44],
    },
}


FREE_MAPPINGS = {"preprocess.extra_channels"}           # user-chosen keys (channel names), not checked against defaults


def deep_merge(base: Dict, override: Dict, path: str = "") -> Dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k not in out:
            raise KeyError(f"unknown config key {path + k!r} (typo?). Known: {sorted(out)}")
        if isinstance(out[k], dict) and isinstance(v, dict) and (out[k] or path + k not in FREE_MAPPINGS):
            out[k] = deep_merge(out[k], v, path + k + ".")
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | None = None, overrides: Dict | None = None) -> Dict:
    """Defaults <- YAML file <- overrides. Relative data paths stay relative to the working directory."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, user)
        cfg["_config_file"] = os.path.abspath(path)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def parse_set(items) -> Dict:
    """``["train.epochs=2", "predict.tta_flip=true"]`` -> nested dict (values parsed as YAML)."""
    out: Dict = {}
    for item in items or []:
        key, value = item.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def preprocess_signature(cfg: Dict) -> Dict:
    """The part of the config that determines the prepared volumes (written next to them, checked on reuse)."""
    pp = {k: v for k, v in cfg["preprocess"].items() if k != "workers"}
    return json.loads(json.dumps(pp))


def same_preprocessing(old: Dict, new: Dict) -> bool:
    """Signatures written by an older cmbx lack keys added later (e.g. extra_channels); a missing key means its default,
    so a folder prepared before the key existed is still reused when nothing that matters differs."""
    base = {k: v for k, v in DEFAULTS["preprocess"].items() if k != "workers"}
    fill = lambda s: json.loads(json.dumps(_fill_defaults(base, s)))
    return fill(old) == fill(new)


def _fill_defaults(base: Dict, d: Dict) -> Dict:
    out = copy.deepcopy(d)
    for k, v in base.items():
        if k not in out:
            out[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and v and isinstance(out[k], dict):
            out[k] = _fill_defaults(v, out[k])
    return out


def dump(cfg: Dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f, sort_keys=False)
