#!/bin/bash
# run_tests.sh -- every test of this repository in one line.
#
#   ./run_tests.sh quick            test.py: every function works (1 min)
#   ./run_tests.sh easy             examples/easy/run_all.py: prepare, train, test, fine-tune (GPU ~5 min, CPU ~30 min)
#   ./run_tests.sh all              test_all_cases.py: every experiment + report (GPU ~10 min, CPU ~2 h)
#   ./run_tests.sh everything       all three, one after another
#   ./run_tests.sh all --slurm      the same as a GPU job on a SLURM cluster (sbatch); check with: squeue --me
#
# The cmbx environment is activated automatically (./env.sh from ./install.sh, else the conda env "cmbx").
# Every run writes a log to test_logs/. Options after the level go to the Python script, e.g.
#   ./run_tests.sh all --only pretrained finetune
# SLURM resources can be changed with SLURM_ARGS, e.g.  SLURM_ARGS="-p gpu --gres=gpu:1 --time=03:00:00" ./run_tests.sh all --slurm
set -uo pipefail
cd "$(dirname "$0")"
LEVEL="${1:-}"; shift || true

case "$LEVEL" in
  quick)      SCRIPTS=("test.py") ;;
  easy)       SCRIPTS=("examples/easy/run_all.py") ;;
  all)        SCRIPTS=("test_all_cases.py") ;;
  everything) SCRIPTS=("test.py" "examples/easy/run_all.py" "test_all_cases.py") ;;
  *)          sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac

SLURM=0; ARGS=()
for a in "$@"; do [ "$a" = "--slurm" ] && SLURM=1 || ARGS+=("$a"); done
mkdir -p test_logs
STAMP=$(date +%Y%m%d_%H%M%S)

# ---- on a SLURM cluster: submit this same script as a job and stop here
if [ "$SLURM" = 1 ]; then
  RES=${SLURM_ARGS:-"-p gpu --gres=shard:1 --cpus-per-task=8 --mem=32G --time=02:00:00"}
  sbatch $RES -J "cmbx_$LEVEL" -o "test_logs/${LEVEL}_${STAMP}_%j.log" --wrap "./run_tests.sh $LEVEL ${ARGS[*]:-}"
  echo "submitted. Follow it with:  squeue --me   and   tail -f test_logs/${LEVEL}_${STAMP}_*.log"
  exit 0
fi

# ---- activate the environment
export PYTHONNOUSERSITE=1          # packages in ~/.local must not leak in
if [ -f ./env.sh ]; then
  source ./env.sh
else
  for c in "${CONDA_EXE:-}" "$(command -v conda 2>/dev/null)" /opt/miniforge/bin/conda "$HOME/miniconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
    [ -n "$c" ] && [ -x "$c" ] && { source "$(dirname "$(dirname "$c")")/etc/profile.d/conda.sh"; break; }
  done
  conda activate cmbx 2>/dev/null || { echo "no cmbx environment found -- run ./install.sh first"; exit 3; }
fi
echo "python: $(command -v python)"

# ---- run
LOG="test_logs/${LEVEL}_${STAMP}.log"
FAILED=()
for s in "${SCRIPTS[@]}"; do
  echo "===== $s $(date +%H:%M:%S)" | tee -a "$LOG"
  start=$(date +%s)
  if [ "$s" = "test.py" ]; then python "$s" 2>&1 | tee -a "$LOG"; else python "$s" "${ARGS[@]}" 2>&1 | tee -a "$LOG"; fi
  rc=${PIPESTATUS[0]}
  echo "===== $s: $([ "$rc" = 0 ] && echo ok || echo "FAILED (exit $rc)") in $(( ($(date +%s) - start) / 60 )) min" | tee -a "$LOG"
  [ "$rc" = 0 ] || FAILED+=("$s")
done

if [ ${#FAILED[@]} -eq 0 ]; then echo "ALL TESTS PASSED ($LEVEL) -- log: $LOG"; exit 0
else echo "FAILED: ${FAILED[*]} -- log: $LOG"; exit 1; fi
