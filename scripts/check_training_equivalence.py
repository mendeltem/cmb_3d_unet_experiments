#!/usr/bin/env python3
"""Training-path equivalence with the original code (imported read-only from cmb-valdo-chain/code, CPU, nothing
written): same seed -> (1) identical patches and shuffle from ``cv5.stichprobe`` vs ``cmbx.train.sample_patches``,
(2) identical initial weights ``netze.NETZE['a03-aniso']`` vs ``cmbx.model.build_model``, (3) identical loss and
weights after a few AdamW steps on those patches.

  python scripts/check_training_equivalence.py --code /path/to/cmb-valdo-chain/code --prepared <prepared_dir> \
      --ids <id> <id> <id> --patches 16
"""
import argparse
import os
import sys

import numpy as np
import torch

from cmbx.config import DEFAULTS
from cmbx.data import load_prepared
from cmbx.model import build_model, convert_legacy_keys
from cmbx.train import loss_bce_dice, sample_patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--patches", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2)
    a = ap.parse_args()
    os.environ["CV5_DEV"] = "cpu"
    sys.path.insert(0, a.code)
    import cv5
    from netze import NETZE

    recs = {s: load_prepared(a.prepared, s) for s in a.ids}
    cv5.FA = dict(recs={s: dict(img=r.image, frs=r.frst, lab=r.label, komp=r.components) for s, r in recs.items()})
    P = {**cv5.REZEPT, "norm": "instance", "netz": "a03-aniso", "n_patches": a.patches}
    T = dict(DEFAULTS["train"], n_patches=a.patches)
    xo, yo = cv5.stichprobe(list(a.ids), np.random.RandomState(42), P)
    xn, yn = sample_patches(recs, list(a.ids), np.random.RandomState(42), T).all()
    print("patches identical:", torch.equal(xo, xn) and torch.equal(yo, yn),
          f"({a.patches} patches, {int((yo.flatten(1).sum(1) > 0).sum())} with a lesion)")

    torch.manual_seed(42); mo = NETZE["a03-aniso"](2)
    torch.manual_seed(42); mn = build_model("aniso_unet3d", 2, 32)
    so = convert_legacy_keys(mo.state_dict()); sn = mn.state_dict()
    print("initial weights identical:", set(so) == set(sn) and all(torch.equal(so[k], sn[k]) for k in sn))

    oo = torch.optim.AdamW(mo.parameters(), lr=3e-4, weight_decay=1e-4)
    on = torch.optim.AdamW(mn.parameters(), lr=3e-4, weight_decay=1e-4)
    for step in range(a.steps):
        b = slice(4 * step, 4 * step + 4)
        losses = []
        for m, opt, lf in ((mo, oo, cv5.loss_bce_dice), (mn, on, loss_bce_dice)):
            m.train(); opt.zero_grad(set_to_none=True)
            loss = lf(m(xo[b]), yo[b]); loss.backward(); opt.step(); losses.append(float(loss))
        print(f"step {step}: loss original {losses[0]:.8f}  cmbx {losses[1]:.8f}  equal {losses[0] == losses[1]}")
    so = convert_legacy_keys(mo.state_dict()); sn = mn.state_dict()
    print("weights after the steps: max |difference|", max(float((so[k] - sn[k]).abs().max()) for k in sn))


if __name__ == "__main__":
    main()
