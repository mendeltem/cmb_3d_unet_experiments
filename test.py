#!/usr/bin/env python3
"""Does cmbx work on this machine?   python test.py [--quick] [--device auto|cuda|mps|cpu]

Hardware-agnostic: runs on NVIDIA GPUs (CUDA), AMD GPUs (ROCm), Apple Silicon (MPS) and plain CPUs (x86 or ARM).
A missing GPU is reported, not counted as a failure. Model-agnostic: every network registered in cmbx.model.MODELS
is tested, nothing is hard-coded.

  1. environment   required Python packages, PyTorch build
  2. devices       a small computation on every device present (CPU always; CUDA/ROCm; MPS)
  3. networks      every registered network: forward + backward on a random anisotropic patch, on every device
  4. end to end    5-fold CV + evaluation + final model on 10 synthetic phantoms (cmbx selftest, ~1 min), on the
                   best device (skipped with --quick)
Exit code 0 = everything needed for training works here."""
import argparse
import importlib
import platform
import sys
import time

REQUIRED = ("numpy", "scipy", "nibabel", "skimage", "pandas", "yaml", "torch")
rows = []        # (step, item, status, detail); status: PASS | FAIL | INFO


def report(step, item, status, detail=""):
    rows.append((step, item, status, detail))
    print(f"  {status:4s}  {item:28s} {detail}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip the end-to-end run (step 4)")
    ap.add_argument("--device", default="auto", help="device for step 4: auto (cuda > cpu) | cuda | mps | cpu")
    a = ap.parse_args()
    t0 = time.time()

    print(f"1. environment  (python {platform.python_version()}, {platform.system()} {platform.machine()})")
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            report(1, mod, "PASS", getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            report(1, mod, "FAIL", f"{type(e).__name__}: {e}  -> run ./install.sh")
    try:
        import cmbx  # noqa: F401
        report(1, "cmbx", "PASS", "importable")
    except Exception as e:  # noqa: BLE001
        report(1, "cmbx", "FAIL", f"{e}  -> pip install -e .")
    if any(r[2] == "FAIL" for r in rows):
        return finish(t0)

    import torch
    build = f"ROCm {torch.version.hip}" if getattr(torch.version, "hip", None) else \
        f"CUDA {torch.version.cuda}" if torch.version.cuda else "CPU only"
    report(1, "PyTorch build", "INFO", f"{torch.__version__} ({build})")

    print("2. devices")
    devices = []
    candidates = [("cpu", True, platform.processor() or platform.machine())]
    if torch.cuda.is_available():
        candidates.append(("cuda", True, f"{torch.cuda.get_device_name(0)}, "
                           f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB"))
    else:
        report(2, "cuda", "INFO", "no CUDA/ROCm GPU visible (training would run on the CPU)")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        candidates.append(("mps", False, "Apple GPU (optional, not used by default)"))
    for dev, required, desc in candidates:
        try:
            x = torch.arange(6, dtype=torch.float32, device=dev).reshape(2, 3)
            assert float((x @ x.T).sum()) == 83.0   # [[5, 14], [14, 50]]
            devices.append(dev)
            report(2, dev, "PASS", desc)
        except Exception as e:  # noqa: BLE001
            report(2, dev, "FAIL" if required else "INFO", f"{desc}: {type(e).__name__}: {str(e)[:120]}")

    print("3. networks (forward + backward)")
    from cmbx.config import DEFAULTS
    from cmbx.model import MODELS, build_model, count_parameters
    patch = [16, 32, 32]
    nch = len(DEFAULTS["train"].get("channels", ["image", "frst"]))
    for name in sorted(MODELS):
        for dev in devices:
            try:
                torch.manual_seed(0)
                net = build_model(name, nch, 8).to(dev)
                x = torch.randn(2, nch, *patch, device=dev)
                y = net(x)
                assert tuple(y.shape[2:]) == tuple(patch), f"output {tuple(y.shape)} for input {tuple(x.shape)}"
                assert torch.isfinite(y).all(), "non-finite output"
                y.float().mean().backward()
                g = [p.grad for p in net.parameters() if p.grad is not None]
                assert g and all(torch.isfinite(t).all() for t in g), "missing or non-finite gradients"
                report(3, f"{name} @ {dev}", "PASS", f"{count_parameters(net):,} parameters (test width 8), output {tuple(y.shape)}")
            except Exception as e:  # noqa: BLE001
                report(3, f"{name} @ {dev}", "INFO" if dev == "mps" else "FAIL", f"{type(e).__name__}: {str(e)[:150]}")

    if not a.quick:
        dev = a.device if a.device != "auto" else ("cuda" if "cuda" in devices else "cpu")
        print(f"4. end to end on {dev} (5-fold CV + final model on 10 phantoms; numbers mean nothing)")
        from cmbx.selftest import selftest
        lines = []
        t1 = time.time()
        rc = selftest(device=dev, log=lines.append)
        report(4, f"selftest @ {dev}", "PASS" if rc == 0 else "FAIL", f"{time.time() - t1:.0f} s" if rc == 0 else
               (lines[-1] if lines else "no output"))
        if rc != 0:
            print("\n".join("        " + s for s in lines[-15:]))
    return finish(t0)


def finish(t0):
    fails = [r for r in rows if r[2] == "FAIL"]
    print(f"\n{'ALL REQUIRED CHECKS PASSED' if not fails else f'{len(fails)} CHECK(S) FAILED'} ({time.time() - t0:.0f} s)")
    for r in fails:
        print(f"  FAIL  step {r[0]}: {r[1]}  {r[3]}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
