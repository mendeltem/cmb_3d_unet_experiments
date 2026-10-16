# cmb_3d_unet_experiments

5-fold cross-validation of our best cerebral-microbleed (CMB) detector (anisotropic 3D U-Net `a03-aniso` from
[cmb-valdo-chain](https://github.com/mendeltem/cmb-valdo-chain)) on **your own data**, and a framework to test
**pre- and post-processing variants**, one YAML file per experiment. Package: `cmbx`. Primary target: the in-house
Charité T2*-GRE and SWI cohorts; VALDO 2021 Task 2 works as a public example. Details: [docs/DETAILS.md](docs/DETAILS.md).

> **Patient data never go into this repository** (no images, masks, case lists with paths, predictions, weights,
> per-case tables). The in-house configs write everything to `../cmb_3d_unet_experiments_runs/`, outside the repo.

## Prerequisites

| | |
|---|---|
| GPU | NVIDIA GPU with >= 2 GB memory (one training needs ~1.8 GiB) |
| NVIDIA driver | **>= 525** for the default build (`cu124`); driver 450-524: replace `cu124` by `cu118` in `environment.yml` |
| CUDA toolkit | **not needed**: the PyTorch wheels contain CUDA; only the driver must be installed (`nvidia-smi` shows it) |
| RAM / conda | >= 32 GB; Miniconda or Miniforge (behind a proxy: export `http_proxy` / `https_proxy` first) |

## Install

```bash
git clone https://github.com/mendeltem/cmb_3d_unet_experiments.git   # private: needs a GitHub token or SSH key
cd cmb_3d_unet_experiments
conda env create -f environment.yml && conda activate cmbx
pip install -e .
cmbx check-env      # driver, CUDA, GPU memory: says what is missing and which build fits
pytest -q           # 38 tests incl. an end-to-end 5-fold CV on synthetic phantoms (~2 min)
```
CPU-only machine: replace `cu124` by `cpu` in `environment.yml` (tests run; real training is impractically slow).

## Data

Brain extraction and bias-field correction are **not** part of this repository: they are done once, uniformly, on
the data. Per case `cmbx` needs a bias-corrected T2*/SWI image, a brain mask and the reference CMB mask (same matrix).
The prepared in-house data are on the group drive (`ALITA/DATA/CHARITE`, see its `README.txt`):

```bash
cmbx make-caselist --source /path/to/DATA/CHARITE/BIDS_T2_STAR --modality t2star \
    --out ../cmb_3d_unet_experiments_runs/catalina-t2star/cases.csv          # swi: BIDS_SWI, --modality swi
```
This reproduces the training cohorts and folds (T2* 75, SWI 61 cases). Other data: a CSV with the columns `id`,
`image` (bias-corrected), `label`, `brain_mask` (optional: `tissue_map`, `fold`, own columns); see DETAILS.

## Run

```bash
cmbx run-experiment configs/catalina_t2star.yaml     # prepare -> split -> 5-fold CV -> evaluation -> final model
cmbx compare ../cmb_3d_unet_experiments_runs/catalina-t2star/runs   # all runs side by side -> summary.csv
cmbx predict --run <run folder> --cases new.csv --out pred/          # apply the final model to new cases
```
Per run: `summary.json` = the CV score (pooled lesion-level F1, precision, sensitivity, FP per case, per-case F1) and
**`final/model.pt` = one model trained on all cases**, the model to keep and use (`--no-final` skips it;
`cmbx train-final <config>` makes it later). Time on one GPU: CV ~2.3 h + final model ~0.5 h for 60 epochs.
Reference (stage 1, seed 42): catalina T2* **0.665**, SWI **0.630**; seed noise is about +-0.03 F1.

## Experiments = config files

A config lists only what differs from the defaults (= the winning recipe). Copy one, change `name` and the keys
you want to vary, then `cmbx run-experiment your.yaml`.

| config | tests |
|---|---|
| `catalina_t2star.yaml`, `catalina_swi.yaml` | baseline on the in-house cohorts |
| `no_frst.yaml` | without the FRST channel (`train.channels: [image]`) |
| `extra_channel_example.yaml` | an **own extra input channel** (e.g. vesselness, FLAIR): `preprocess.extra_channels: {name: column}`, then `train.channels: [image, frst, name]` or `[image, name]` |
| `csf_rule.yaml` | drop candidates in CSF (needs a SynthSeg map per case) |
| `inpaint_example.yaml` | paint over a mask before training (hook for own inpainting methods) |

Post-processing variants without retraining: `cmbx evaluate <run> --mode crossfold --csf-rule`.

## HPC (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=cmbx --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=04:00:00
source /path/to/miniconda3/etc/profile.d/conda.sh && conda activate /path/to/miniconda3/envs/cmbx
cd /path/to/cmb_3d_unet_experiments && cmbx run-experiment configs/catalina_t2star.yaml
```
Use the partition and account names of your cluster; activate the env by its absolute path.

## License

Code MIT. Data and weights are not part of the repository. Cite cmb-valdo-chain (mendeltem, 2026) and, for the
FRST, Loy & Zelinsky 2003 (doi:10.1109/TPAMI.2003.1217601); VALDO: Sudre et al. 2024 (doi:10.1016/j.media.2024.103112).
