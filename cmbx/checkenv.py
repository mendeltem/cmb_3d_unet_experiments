"""``cmbx check-env``: what the installation can do and what is missing (exit code 1 if a REQUIRED part is missing).
Brain extraction and bias correction are not part of cmbx, so FSL / HD-BET are not checked."""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys

# Prerequisite: an NVIDIA driver, NOT an installed CUDA toolkit (the PyTorch wheels ship the CUDA libraries).
# Minimum driver per wheel build (NVIDIA minor-version compatibility, Linux): CUDA 12.x >= 525.60.13, CUDA 11.x >= 450.80.02.
MIN_DRIVER = {"12": (525, 60), "11": (450, 80)}


def _driver_version():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return out[0].strip() if out else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


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
    drv = _driver_version()
    if drv is None:
        log("  --   NVIDIA driver: nvidia-smi not found (no NVIDIA GPU or no driver) -> CPU only")
    else:
        dv = tuple(int(x) for x in drv.split(".")[:2])
        ok12, ok11 = dv >= MIN_DRIVER["12"], dv >= MIN_DRIVER["11"]
        log(f"  {'ok  ' if ok12 else '!!  '} NVIDIA driver {drv}: "
            + ("supports the default CUDA 12.4 build (cu124)" if ok12 else
               "too old for the CUDA 12 build; use cu118 in environment.yml (needs >= 450.80)" if ok11 else
               "too old for any supported CUDA build (needs >= 450.80); update the driver or use cpu"))
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
            log(f"  --   no CUDA GPU visible to torch (torch build: {torch.version.cuda or 'cpu'}): training would run on the CPU "
                "(only sensible for the tests); check the driver line above")
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
