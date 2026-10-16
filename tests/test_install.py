"""install.sh picks the PyTorch build from the machine (dry run with a stub nvidia-smi), and test.py --quick passes."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plan(tmp_path, driver=None, extra=()):
    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    smi = stub / "nvidia-smi"
    smi.write_text(f"#!/bin/sh\necho {driver}\n" if driver else "#!/bin/sh\nexit 1\n")
    smi.chmod(0o755)
    env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}")
    out = subprocess.run(["bash", os.path.join(ROOT, "install.sh"), "--dry-run", *extra], env=env,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.mark.skipif(sys.platform == "darwin" or os.uname().machine in ("aarch64", "arm64"), reason="x86 Linux plan")
@pytest.mark.parametrize("driver,build", [("565.57.01", "cu124"), ("525.60.13", "cu124"), ("525.40", "cu118"),
                                          ("470.82.01", "cu118"), ("440.33", "cpu")])
def test_build_from_driver(tmp_path, driver, build):
    assert f"build {build}" in plan(tmp_path, driver)


@pytest.mark.skipif(sys.platform == "darwin" or os.uname().machine in ("aarch64", "arm64"), reason="x86 Linux plan")
def test_no_gpu_and_overrides(tmp_path):
    if not os.path.exists("/dev/kfd"):
        assert "build cu124" in plan(tmp_path)                      # login node without GPU: CUDA build, runs on CPU
    assert "build cpu" in plan(tmp_path, "565.57.01", ["--torch", "cpu"])
    assert "env:      venv" in plan(tmp_path, "565.57.01", ["--venv"])


def test_test_py_quick():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "test.py"), "--quick"], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "ALL REQUIRED CHECKS PASSED" in out.stdout
