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
(`--torch cpu|cu118|...`, `--name`, `--prefix`, `--venv`, `--recreate` = build the environment new). Manual alternative: `environment.yml`, see docs/DETAILS.md.

`test.py` checks the packages, every device present (CPU, CUDA/ROCm, MPS), every registered network (forward +
backward) and a 5-fold CV + final model on synthetic phantoms. A missing GPU is reported, not a failure; real
training needs a GPU with >= 2 GB memory (one training: ~1.8 GiB) and >= 32 GB RAM. Full test suite: `pytest -q`.

## Data (BIDS layout)

Brain extraction and bias-field correction are **not** part of this repository: they are done once, uniformly, on
the data. Put each dataset into this BIDS layout (synthetic examples in the repository: `examples/bids_train`,
`examples/bids_new`; `cmbx make-phantoms <dir> --bids` writes another):

```
my_dataset/                                      for TRAINING (examples/bids_train)
├── participants.tsv                             participant_id column = the case ids
├── dataset_description.json
├── sub-001/anat/sub-001_T2starw.nii.gz          raw scan (SWI: sub-001_acq-swi_T2starw.nii.gz)
└── derivatives/
    ├── biascorr/sub-001/anat/sub-001_T2starw_desc-biascorr_T2starw.nii.gz   bias-corrected scan  (required)
    ├── brainmask/sub-001/anat/sub-001_T2starw_desc-brain_mask.nii.gz        brain mask           (recommended)
    ├── manual-cmb/sub-001/anat/sub-001_T2starw_desc-cmb_mask.nii.gz         reference CMB mask   (training only)
    └── synthseg/sub-001/anat/sub-001_T2starw_dseg.nii.gz                    SynthSeg labels      (optional, CSF rule)

new_scans/                                       for INFERENCE (examples/bids_new): the same without manual-cmb/
```

| | training (`run-experiment`) | inference (`predict`) |
|---|---|---|
| `derivatives/biascorr` | required | required |
| `derivatives/brainmask` | recommended (else `image > 0`) | recommended (else `image > 0`) |
| `derivatives/manual-cmb` | required | not needed |
| `derivatives/synthseg` | only for the CSF rule | only for `--csf-rule` |

All masks on the matrix of their scan. One line makes the case list (ids from `participants.tsv`; folders that are
missing are skipped; leave cases out with `--exclude ids.txt`):

```bash
cmbx make-caselist --bids examples/bids_train --out ../my_runs/cases.csv               # training
cmbx make-caselist --bids examples/bids_new --no-folds --out ../my_runs/new/cases.csv  # inference
```
Other layouts: path patterns instead of `--bids`, e.g. `--image "/data/{id}/{id}_t2s_corrected.nii.gz" --label ...
--brain-mask ...` (any `{id}` pattern with `*`; docs/DETAILS.md). The tool also reports missing inputs and suspicious
reference masks.

## Run

The whole workflow on the synthetic BIDS example (about 20 s on a GPU, 2.5 min on a CPU; the cross-validation
finds the 13 phantom microbleeds; the numbers say nothing about real data):
```bash
cmbx make-caselist --bids examples/bids_train --out ../my_runs/cases.csv
cmbx run-experiment configs/example.yaml --set output_root=../my_runs/runs \
    data.cases=../my_runs/cases.csv data.prepared_dir=../my_runs/prepared data.splits=../my_runs/splits.json
cmbx make-caselist --bids examples/bids_new --no-folds --out ../my_runs/new/cases.csv
cmbx predict --run ../my_runs/runs/example --cases ../my_runs/new/cases.csv --out ../my_runs/pred
```
Your data, real training:
```bash
cp configs/baseline.yaml ../my_runs/my_baseline.yaml   # set name, output_root and the data paths:
#   data: {cases: ../my_runs/cases.csv, prepared_dir: ../my_runs/prepared, splits: ../my_runs/splits.json}
cmbx run-experiment ../my_runs/my_baseline.yaml       # prepare -> split -> 5-fold CV -> evaluation -> final model
cmbx compare ../my_runs/runs                          # all runs side by side -> summary.csv
cmbx predict --run <run folder> --image scan.nii.gz --out pred/     # apply the final model to a new scan
cmbx predict --run <run folder> --cases ../my_runs/new/cases.csv --out pred/   # ... or to a case list
```
Per run: `summary.json` = the CV score (pooled lesion-level F1, precision, sensitivity, FP per case, per-case F1) and
**`final/model.pt` = one model trained on all cases**, the model to keep and use (`--no-final` skips it;
`cmbx train-final <config>` makes it later). Time on one GPU: CV ~2.3 h + final model ~0.5 h for 60 epochs.
Validated on two in-house cohorts (T2*-GRE 75 cases, SWI 61 cases): reference pooled F1 **0.665** / **0.630**
(seed noise about +-0.03); this package reproduced 0.644 on the T2* cohort.
`predict` writes the masks on the **input image grid** (`<id>_mask.nii.gz`, `<id>_prob.nii.gz`, label map
`<id>_lesions.nii.gz`), a list of the
microbleeds per scan (`<id>.json`: volume, probability, position) and `predictions.csv`; options (`--folds`,
`--threshold`, `--csf-rule`, losses, channels, inpainting, ...): docs/DETAILS.md.

## Experiments = config files

A config lists only what differs from the defaults (= the winning recipe). Copy one, change `name` and the keys
you want to vary, then `cmbx run-experiment your.yaml`.

| config | tests |
|---|---|
| `baseline.yaml` | the winning recipe; copy it for your data (set the `data` paths and `output_root`) |
| `example.yaml` | small and fast, for the synthetic BIDS example (quick start) |
| `no_frst.yaml` | without the FRST channel (`train.channels: [image]`) |
| `extra_channel_example.yaml` | an **own extra input channel** (e.g. vesselness, FLAIR): `preprocess.extra_channels: {name: column}`, then `train.channels: [image, frst, name]` or `[image, name]` |
| `csf_rule.yaml` | drop candidates in CSF (needs a SynthSeg map per case) |
| `inpaint_example.yaml` | paint over a mask before training (hook for own inpainting methods) |

Post-processing variants without retraining: `cmbx evaluate <run> --mode crossfold --csf-rule`.

## HPC (SLURM)

Install once on the login node, inside `tmux` (a dropped SSH session then does not stop it): `./install.sh`, or
`./install.sh --venv` (Python of the active conda base + pip, no conda download; if `conda create` fails the script
falls back to this by itself). Then check a GPU node (~1 min, synthetic phantoms, no data):
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
