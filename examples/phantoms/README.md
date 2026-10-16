# Synthetic phantoms (no patient data)

Ten synthetic cases written by `cmbx make-phantoms examples/phantoms --n 10` (default seed): image, reference CMB mask,
brain mask and tissue map per case, paths in `cases.csv` relative to this folder. They exist so that the install and
the SLURM quick test (`scripts/slurm_quick_test.sbatch`, config `configs/smoke_gpu.yaml`) run without any data.
Numbers computed on them mean nothing.
