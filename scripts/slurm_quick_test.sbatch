#!/bin/bash
# Quick SLURM test (~1 min of GPU): runs test.py on a GPU node (environment, devices, every network, 5-fold CV on
# synthetic phantoms). No patient data. A quarter GPU (shard) is plenty and usually starts at once.
#   cd cmb_3d_unet_experiments && mkdir -p logs && sbatch scripts/slurm_quick_test.sbatch
# No shards on your cluster: sbatch --gres=gpu:1 scripts/slurm_quick_test.sbatch. Other partition: sbatch -p NAME ...
#SBATCH --job-name=cmbx_quick
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/cmbx_quick_%j.out
cd "$SLURM_SUBMIT_DIR"
source ./env.sh 2>/dev/null || { eval "$(conda shell.bash hook)" && conda activate cmbx; }   # env.sh from install.sh
echo "== node $(hostname)  job $SLURM_JOB_ID  $(date '+%F %T')"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || true
python test.py
