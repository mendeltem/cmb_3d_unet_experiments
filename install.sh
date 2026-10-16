#!/bin/bash
# One-command install of cmbx:  ./install.sh   (then: source env.sh; python test.py)
#
# 1. Python 3.12 environment: conda/mamba if available (only Python is solved by conda -> fast), otherwise a venv in .venv
# 2. PyTorch build for THIS machine (no CUDA toolkit needed, the wheels bring their libraries):
#      Linux + NVIDIA driver >= 525 -> cu124 | driver 450-524 -> cu118 | Linux + AMD GPU (ROCm) -> rocm6.2.4
#      Linux without GPU (e.g. a cluster login node) -> cu124 (also runs on the CPU, ready for the GPU nodes)
#      macOS (Apple Silicon: MPS) and Linux ARM -> the PyPI build
# 3. cmbx itself (pip install -e .), writes env.sh (activation), runs test.py
#
# Options: --name NAME (conda env, default cmbx) | --prefix PATH (conda env at a path) | --venv (force a venv)
#          --torch auto|cu124|cu118|rocm|cpu|pypi | --no-test | --dry-run (print the plan only)
#          --recreate (delete the environment first, also when it is the active one, and build it new)
# Behind a proxy: export http_proxy / https_proxy first.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"; REPO=$(pwd)
NAME=cmbx; PREFIX=""; FORCE_VENV=0; TORCH=auto; TEST=1; DRY=0; RECREATE=0
TORCH_VERSION=2.6.0; PY_VERSION=3.12
while [ $# -gt 0 ]; do case "$1" in
    --name) NAME=$2; shift 2 ;; --prefix) PREFIX=$2; shift 2 ;; --venv) FORCE_VENV=1; shift ;;
    --torch|--cuda) TORCH=$2; shift 2 ;; --no-test) TEST=0; shift ;; --dry-run) DRY=1; shift ;;
    --recreate) RECREATE=1; shift ;;
    -h|--help) sed -n 2,14p "$0"; exit 0 ;; *) echo "unknown option $1 (see --help)"; exit 2 ;; esac; done
say() { echo "[install] $*"; }

# ---- which PyTorch build
OS=$(uname -s); ARCH=$(uname -m); WHY=""
if [ "$TORCH" = auto ]; then
    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)
    if [ "$OS" = Darwin ] || [ "$ARCH" = aarch64 ] || [ "$ARCH" = arm64 ]; then TORCH=pypi; WHY="$OS/$ARCH"
    elif [ -n "$DRV" ]; then
        MAJ=${DRV%%.*}; MIN=$(echo "$DRV" | cut -d. -f2)
        if [ "$MAJ" -gt 525 ] || { [ "$MAJ" -eq 525 ] && [ "${MIN:-0}" -ge 60 ]; }; then TORCH=cu124; WHY="NVIDIA driver $DRV"
        elif [ "$MAJ" -ge 450 ]; then TORCH=cu118; WHY="NVIDIA driver $DRV (< 525.60)"
        else TORCH=cpu; WHY="NVIDIA driver $DRV too old for CUDA builds"; fi
    elif [ -e /dev/kfd ] || command -v rocminfo >/dev/null 2>&1; then TORCH=rocm; WHY="AMD GPU (ROCm)"
    else TORCH=cu124; WHY="no GPU visible here; the CUDA build also runs on the CPU and is ready for GPU nodes (--torch cpu: smaller download)"; fi
fi
[ -z "$WHY" ] && WHY="chosen with --torch"
case "$TORCH" in
    cu124|cu118|cpu) TORCH_INDEX="--index-url https://download.pytorch.org/whl/$TORCH" ;;
    rocm) TORCH_INDEX="--index-url https://download.pytorch.org/whl/rocm6.2.4" ;;
    pypi) TORCH_INDEX="" ;;
    *) echo "unknown --torch $TORCH"; exit 2 ;; esac

# ---- conda or venv
CONDA=""
if [ $FORCE_VENV = 0 ]; then for c in "${CONDA_EXE:-}" "$(command -v mamba 2>/dev/null || true)" "$(command -v conda 2>/dev/null || true)"; do
    [ -n "$c" ] && [ -x "$c" ] && { CONDA=$c; break; }; done; fi
if [ -n "$CONDA" ]; then KIND="conda ($CONDA)"; TARGET=${PREFIX:-$NAME}; else KIND="venv"; TARGET=$REPO/.venv; fi
say "machine:  $OS $ARCH"
say "PyTorch:  $TORCH_VERSION, build $TORCH ($WHY)"
say "env:      $KIND -> $TARGET"
[ $DRY = 1 ] && { say "dry run, nothing installed"; exit 0; }

if [ -n "$CONDA" ]; then
    if [ -n "$PREFIX" ]; then LOC=(-p "$PREFIX"); else LOC=(-n "$NAME"); fi
    if [ $RECREATE = 1 ] && "$CONDA" run "${LOC[@]}" python -V >/dev/null 2>&1; then
        say "--recreate: removing the existing environment"
        env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL "$CONDA" env remove -y "${LOC[@]}"
    fi
    if "$CONDA" run "${LOC[@]}" python -V >/dev/null 2>&1; then say "env exists, reusing it (--recreate builds it new)"
    elif ! "$CONDA" create -y "${LOC[@]}" --override-channels -c conda-forge "python=$PY_VERSION" pip; then
        say "conda create failed (network? see above): falling back to a venv in .venv"
        CONDA=""; KIND="venv"
    fi
  fi
  if [ -n "$CONDA" ]; then
    ENVP=$("$CONDA" run "${LOC[@]}" python -c 'import sys; print(sys.prefix)')
    BASE=$("$CONDA" info --base)
    printf '# written by install.sh: activate the cmbx environment (source env.sh)\nsource "%s/etc/profile.d/conda.sh" && conda activate "%s"\n' "$BASE" "$ENVP" > env.sh
else
    PYS=""; for p in python3.12 python3.11 python3.10 python3; do command -v $p >/dev/null && \
        $p -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null && { PYS=$p; break; }; done
    [ -n "$PYS" ] || { echo "no conda and no Python >= 3.10 found: install Miniforge (https://conda-forge.org/download)"; exit 1; }
    [ $RECREATE = 1 ] && rm -rf .venv
    if [ ! -x .venv/bin/python ] || ! .venv/bin/python -m pip -V >/dev/null 2>&1; then
        rm -rf .venv
        if command -v uv >/dev/null 2>&1; then uv venv --seed --python "$PYS" .venv          # uv: fast, brings pip
        elif "$PYS" -m venv .venv 2>/dev/null; then :
        else   # e.g. Ubuntu without python3-venv (ensurepip missing): venv without pip, then pip via get-pip.py
            rm -rf .venv; "$PYS" -m venv --without-pip .venv
            say "no ensurepip in $PYS: fetching pip with get-pip.py"
            curl -fsSL https://bootstrap.pypa.io/get-pip.py -o .venv/get-pip.py || wget -qO .venv/get-pip.py https://bootstrap.pypa.io/get-pip.py
            .venv/bin/python .venv/get-pip.py --quiet && rm -f .venv/get-pip.py
        fi
    fi
    ENVP=$REPO/.venv
    printf '# written by install.sh: activate the cmbx environment (source env.sh)\nsource "%s/bin/activate"\n' "$ENVP" > env.sh
fi
PY=$ENVP/bin/python
"$PY" -m pip install --upgrade pip
# shellcheck disable=SC2086
"$PY" -m pip install "torch==$TORCH_VERSION" $TORCH_INDEX
"$PY" -m pip install -e ".[test]" "numpy>=1.26,<2.3"
say "installed into $ENVP; activate with:  source env.sh"
if [ $TEST = 1 ]; then "$PY" test.py; fi
