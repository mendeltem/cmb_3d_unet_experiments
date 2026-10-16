# cmb_3d_unet_experiments

5-fold cross-validation of our best cerebral-microbleed (CMB) detector (anisotropic 3D U-Net `a03-aniso` from
[cmb-valdo-chain](https://github.com/mendeltem/cmb-valdo-chain)) on **your own data**, and a framework to test
**pre- and post-processing variants**, one YAML file per experiment. Package: `cmbx`. Works on any T2*-GRE or SWI
dataset with reference CMB masks (nothing dataset-specific is built in); VALDO 2021 Task 2 is a public example.
Details: [docs/DETAILS.md](docs/DETAILS.md).

> **Patient data never go into this repository** (no images, masks, case lists with paths, predictions, weights,
> per-case tables). Keep case lists, configs for your data and all outputs outside the repo, e.g. in `../my_runs/`.

## Install

```bash
git clone https://github.com/mendeltem/cmb_3d_unet_experiments.git
cd cmb_3d_unet_experiments
./install.sh        # ~5 min: Python env + the PyTorch build for this machine + cmbx, then runs test.py
source env.sh       # activates the environment (in every new shell)
python test.py      # any time: does everything work here? (~1 min)
```

`install.sh` uses conda/mamba if available (otherwise a venv in `.venv`) and picks PyTorch for the hardware:
NVIDIA (driver >= 525: CUDA 12.4, 450-524: CUDA 11.8), AMD (ROCm), Apple Silicon and ARM (PyPI build), or no GPU
(the CUDA build, which also runs on the CPU, e.g. on a cluster login node). No CUDA toolkit is needed, only the GPU
driver. Behind a proxy export `http_proxy` / `https_proxy` first; `./install.sh --help` lists the options
(`--torch cpu|cu118|...`, `--name`, `--prefix`, `--venv`). Manual alternative: `environment.yml`, see docs/DETAILS.md.

`test.py` checks the packages, every device present (CPU, CUDA/ROCm, MPS), every registered network (forward +
backward) and a 5-fold CV + final model on synthetic phantoms. A missing GPU is reported, not a failure; real
training needs a GPU with >= 2 GB memory (one training: ~1.8 GiB) and >= 32 GB RAM. Full test suite: `pytest -q`.

## Data

Brain extraction and bias-field correction are **not** part of this repository: they are done once, uniformly, on
the data. Per case `cmbx` needs a bias-corrected T2*/SWI image, a brain mask and the reference CMB mask (same matrix).
The case list is a CSV (`id`, `image`, `label`, `brain_mask`; optional `tissue_map`, `fold`, own columns). Build it
from any folder layout with path patterns (`{id}` = case id, `*` allowed):

```bash
cmbx make-caselist --image "/data/{id}/anat/{id}_desc-biascorr_T2starw.nii.gz" \
    --label "/data/{id}/anat/{id}_cmb_mask.nii.gz" --brain-mask "/data/{id}/anat/{id}_brainmask.nii.gz" \
    [--ids ids.tsv] [--exclude leave_out.txt] --out ../my_runs/cases.csv
```
It reports cases with missing inputs and suspicious reference masks, and assigns stratified folds (details: DETAILS).

## Run

```bash
cp configs/baseline.yaml ../my_runs/my_baseline.yaml   # set name, data.cases / prepared_dir / splits, output_root
cmbx run-experiment ../my_runs/my_baseline.yaml       # prepare -> split -> 5-fold CV -> evaluation -> final model
cmbx compare ../my_runs/runs                          # all runs side by side -> summary.csv
cmbx predict --run <run folder> --image scan.nii.gz --out pred/     # apply the final model to a new scan
cmbx predict --run <run folder> --cases new.csv --out pred/          # ... or to a case list
```
Per run: `summary.json` = the CV score (pooled lesion-level F1, precision, sensitivity, FP per case, per-case F1) and
**`final/model.pt` = one model trained on all cases**, the model to keep and use (`--no-final` skips it;
`cmbx train-final <config>` makes it later). Time on one GPU: CV ~2.3 h + final model ~0.5 h for 60 epochs.
Validated on two in-house cohorts (T2*-GRE 75 cases, SWI 61 cases): reference pooled F1 **0.665** / **0.630**
(seed noise about +-0.03); this package reproduced 0.644 on the T2* cohort.
`predict` writes the masks on the **input image grid** (`<id>_mask.nii.gz`, `<id>_prob.nii.gz`), a list of the
microbleeds per scan (`<id>.json`: volume, probability, position) and `predictions.csv`; options (`--folds`,
`--threshold`, `--csf-rule`, losses, channels, inpainting, ...): docs/DETAILS.md.

## Experiments = config files

A config lists only what differs from the defaults (= the winning recipe). Copy one, change `name` and the keys
you want to vary, then `cmbx run-experiment your.yaml`.

| config | tests |
|---|---|
| `baseline.yaml` | the winning recipe; copy it for your data (set the `data` paths and `output_root`) |
| `no_frst.yaml` | without the FRST channel (`train.channels: [image]`) |
| `extra_channel_example.yaml` | an **own extra input channel** (e.g. vesselness, FLAIR): `preprocess.extra_channels: {name: column}`, then `train.channels: [image, frst, name]` or `[image, name]` |
| `csf_rule.yaml` | drop candidates in CSF (needs a SynthSeg map per case) |
| `inpaint_example.yaml` | paint over a mask before training (hook for own inpainting methods) |

Post-processing variants without retraining: `cmbx evaluate <run> --mode crossfold --csf-rule`.

## HPC (SLURM)

Install once on the login node (`./install.sh`), then check a GPU node (~1 min, synthetic phantoms, no data):
```bash
mkdir -p logs && sbatch scripts/slurm_quick_test.sbatch && squeue --me      # result in logs/cmbx_quick_<job>.out
```
A real run (one training needs ~1.8 GiB, so a GPU share is enough where the cluster offers `shard`; resubmitting
continues where it stopped):
```bash
sbatch scripts/slurm_run_experiment.sbatch ../my_runs/my_baseline.yaml         # [key=value ...] overrides
```
Use the partition / GRES names of your cluster (`sinfo -o "%P %G"`); without shards use `--gres=gpu:1`.

## License

Code MIT. Data and weights are not part of the repository. Cite cmb-valdo-chain (mendeltem, 2026) and, for the
FRST, Loy & Zelinsky 2003 (doi:10.1109/TPAMI.2003.1217601); VALDO: Sudre et al. 2024 (doi:10.1016/j.media.2024.103112).
