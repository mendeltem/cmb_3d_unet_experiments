#!/bin/bash
# Quick SLURM test (~5 min): GPU visible, environment complete, a tiny 5-fold CV on synthetic phantoms runs on the GPU.
# No patient data involved. Short time limit and small memory so the scheduler starts it fast.
#   cd cmb_3d_unet_experiments && mkdir -p logs && sbatch scripts/slurm_quick_test.sbatch
# Change --partition to the GPU partition of your cluster (sinfo -s lists them).
#SBATCH --job-name=cmbx_quick
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/cmbx_quick_%j.out
set -e
eval "$(conda shell.bash hook)" && conda activate cmbx
cd "$SLURM_SUBMIT_DIR"
echo "== node $(hostname)  job $SLURM_JOB_ID  $(date '+%F %T')"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "== check-env"; cmbx check-env
echo "== phantoms"; cmbx make-phantoms data/phantoms --n 10
echo "== smoke CV on GPU"; time cmbx run-experiment configs/smoke_gpu.yaml
echo "== result"; cat runs/smoke_gpu/summary.json 2>/dev/null | head -20
echo "== done $(date '+%F %T')  (numbers from the smoke run mean nothing; only that every step ran)"
