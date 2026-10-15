"""``cmbx check-env``: what the installation can do and what is missing (exit code 1 if a REQUIRED part is missing).
Brain extraction and bias correction are not part of cmbx, so FSL / HD-BET are not checked."""
from __future__ import annotations

import importlib
import shutil
import sys


def check_env(log=print) -> int:
    missing = []
    log(f"python {sys.version.split()[0]} ({sys.executable})")
    for mod in ("numpy", "scipy", "nibabel", "skimage", "pandas", "yaml", "torch"):
        try:
            m = importlib.import_module(mod)
            log(f"  ok   {mod} {getattr(m, '__version__', '')}")
        except ImportError:
            log(f"  MISSING {mod} (required)")
            missing.append(mod)
    gpu = False
    try:
        import torch
        if torch.cuda.is_available():
            gpu = True
            free, total = torch.cuda.mem_get_info()
            log(f"  ok   CUDA {torch.version.cuda}: {torch.cuda.get_device_name(0)}, "
                f"{free / 2**30:.1f} of {total / 2**30:.1f} GiB free (one training needs ~1.8 GiB)")
            x = torch.ones(2, device="cuda") * 2
            assert float(x.sum()) == 4.0
        else:
            log("  --   no CUDA GPU visible: training would run on the CPU (only sensible for the tests)")
    except ImportError:
        pass
    synth = shutil.which("mri_synthseg")
    log(f"  {'ok  ' if synth else '--  '} mri_synthseg (optional, tissue maps for the CSF rule): "
        f"{synth or 'not found; precomputed maps work as well'}")
    if missing:
        log(f"REQUIRED parts missing: {missing}")
        return 1
    log("all required parts present" + ("" if gpu else " (no GPU)"))
    return 0
