#!/bin/bash
# One experiment on a GPU node: prepare -> split -> 5-fold CV -> evaluation -> final model (resumable: a finished fold
# is skipped when the job is submitted again, e.g. after a time limit).
#   cd cmb_3d_unet_experiments && mkdir -p logs
#   sbatch scripts/slurm_run_experiment.sbatch ../my_runs/my_baseline.yaml [train.epochs=60 ...]
# One training needs ~1.8 GiB GPU memory, so a GPU share (shard) is enough; without shards: --gres=gpu:1.
# Host RAM: all cases are held in memory (~0.2 GB per case at 0.5 x 0.5 x 1 mm, measured 12-17 GB for 60-75 cases).
#SBATCH --job-name=cmbx_run
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cmbx_run_%j.out
set -e
CONFIG=${1:?usage: sbatch scripts/slurm_run_experiment.sbatch <config.yaml> [key=value ...]}; shift
cd "$SLURM_SUBMIT_DIR"
source ./env.sh 2>/dev/null || { eval "$(conda shell.bash hook)" && conda activate cmbx; }
echo "== node $(hostname)  job $SLURM_JOB_ID  $(date '+%F %T')  config $CONFIG $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
if [ $# -gt 0 ]; then cmbx run-experiment "$CONFIG" --set "$@"; else cmbx run-experiment "$CONFIG"; fi
echo "== done $(date '+%F %T')"
