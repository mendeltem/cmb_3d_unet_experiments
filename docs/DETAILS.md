# cmb_3d_unet_experiments: details

Everything that is not needed to run the experiments: the model, all options, outputs, validation of the port,
provenance of every default and the deviations from the original code. Short version: [README](../README.md).

## What the model is

| | |
|---|---|
| Input | 2 channels on a 0.5 x 0.5 x 1.0 mm grid, cropped to the brain: **inverted** T2*/SWI (`1 - I/max(I)` in the brain, microbleeds bright, 0 outside) and the **FRST** map (fast radial symmetry transform, Loy & Zelinsky 2003, radii 1-3 mm, per axial slice) |
| Network | anisotropic 3D U-Net: level 0 convolves (1x3x3) and pools (1x2x2) in-plane only, levels 1-4 are isotropic; (Conv-InstanceNorm-LeakyReLU) x 2 per block, channels 32-64-128-256-512, 22.5 M parameters, one logit per voxel |
| Training | patches 38 x 62 x 94 voxels (z, y, x) = 38 x 31 x 47 mm; 800 patches per epoch: 60 % around a reference lesion voxel (random position in the patch), 25 % around a random brain voxel, 15 % anywhere; random flips; loss 0.5 BCE + 0.5 soft Dice (`train.loss`, see Options); AdamW lr 3e-4, weight decay 1e-4, cosine schedule; batch 4; 60 epochs; bfloat16 autocast on the GPU |
| Selection | **inner fold**: for test fold k, fold (k+1) % 5 is the inner fold and the other three are trained on. Every 5 epochs the inner-fold cases are predicted in full and the pooled lesion F1 is computed for every cell of threshold {0.3 ... 0.9} x minimum size {1, 2, 3, 5 voxels}; the best (epoch, threshold, minimum size) is kept. The test fold is touched only once, with that choice |
| Prediction | sliding window of the patch size, half-window stride, overlaps averaged, windows outside the brain skipped |
| Post-processing | threshold, 26-connected components, minimum size, clump guard (components > 2100 voxels = 525 mm3 are not microbleeds); optional CSF rule (component dropped if its majority SynthSeg label is CSF) |
| Metric | lesion level, 26-connectivity: a reference lesion is found if any predicted component touches it; a predicted component touching no reference is a false positive. Pooled F1, precision, sensitivity, FP per case, plus per-case F1 |

Every default is the winning run `python code/turnier.py --netz a03-aniso --epochen 60 --gitter dev/gitter_d` of
cmb-valdo-chain; see [Provenance of every default](#provenance-of-every-default).

## Inputs: prepared outside this repository

**Brain extraction and bias-field correction are deliberately not part of this repository.** They are done once,
uniformly, on the data, and every experiment starts from the same corrected inputs. Per case `cmbx` expects:

| input | how it can be made |
|---|---|
| bias-corrected T2*/SWI image | e.g. FSL FAST `fast -B --nopve -t 2` on the brain-extracted image (VALDO: `fast -t 2 -n 3 -B -b --nopve` on the skull-stripped T2S) |
| brain mask, same matrix | a scanner mask or HD-BET; for skull-stripped data such as VALDO leave `brain_mask` empty (`image > 0`, holes filled) |
| reference CMB mask, same matrix | manual masks (labels are optional for `cmbx predict`) |
| optional SynthSeg map | `mri_synthseg --robust` on the T2*/SWI image (CSF rule only) |

## Prerequisites

| | |
|---|---|
| GPU | GPU with >= 2 GB memory (one training needs ~1.8 GiB): NVIDIA (CUDA), AMD (ROCm build); Apple MPS untested for training; CPU works but is slow |
| NVIDIA driver | **>= 525** for the default CUDA 12.4 build (`cu124`); a driver >= 450 works with the CUDA 11.8 build: replace `cu124` by `cu118` in `environment.yml` |
| CUDA toolkit | **not needed**: the PyTorch wheels ship the CUDA libraries; only the driver must be installed |
| RAM | >= 32 GB (peak ~15-17.5 GB for the T2* cohort) |
| conda | Miniconda / Miniforge; behind a proxy export `http_proxy` / `https_proxy` first |

Check the driver with `nvidia-smi` (version in the top line). `cmbx check-env` reads it and says which build fits.
The minimum driver versions follow NVIDIA's CUDA minor-version compatibility (Linux: CUDA 12.x >= 525.60.13, CUDA 11.x >= 450.80.02).

## Install

Recommended: `./install.sh` (README). It does the steps below automatically and picks the PyTorch build for the
machine; `./install.sh --dry-run` prints what it would do. Manual install with conda:

```bash
conda env create -f environment.yml          # env "cmbx": python 3.12, torch 2.6.0 (CUDA 12.4 wheels), nibabel, ...
conda activate cmbx
pip install -e .                             # the `cmbx` command
cmbx check-env                               # torch / CUDA / GPU memory; what is missing
python test.py                               # environment, devices, every network, CV on phantoms (~1 min)
pytest -q                                    # full test suite, ~2 min on CPU
```
CPU-only machine: replace `cu124` by `cpu` in `environment.yml`. Behind a proxy, export `http_proxy`/`https_proxy`
first. No FSL, HD-BET or FreeSurfer is needed.

## Your data

### Case list from your folder layout (`cmbx make-caselist`)
For the recommended BIDS layout (README "Data") one option is enough: `cmbx make-caselist --bids <root> --out ...`
(patterns of `cmbx.caselist.BIDS`; derivative folders that do not exist, e.g. `manual-cmb` for new scans, are
skipped). Any other layout:
Nothing about a dataset is built in: the layout is given as path patterns with `{id}` (glob `*` allowed; the first
match is used), the ids and exclusions as files you pass.
```bash
cmbx make-caselist --image "<root>/derivatives/biascorr/{id}/anat/{id}*_desc-biascorr*.nii.gz" \
    --label "<root>/{id}/anat/{id}_cmb_mask.nii.gz" \
    --brain-mask "<root>/{id}/anat/{id}_brainmask.nii.gz" --brain-mask "<other masks>/{id}*brain*mask*.nii.gz" \
    [--tissue-map "<synthseg>/{id}_synthseg.nii.gz"] [--raw "<root>/{id}/anat/{id}_T2starw.nii.gz"] \
    [--ids ids.tsv] [--exclude leave_out.txt ...] --out ../my_runs/cases.csv
```
* ids: `--ids` (one id per line, or a CSV/TSV with an `id` / `participant_id` column; its order is kept), otherwise
  every id the `--image` pattern matches; `--exclude` files (same formats) leave cases out; `--brain-mask` can be
  repeated as fallbacks (the column `brain_mask_from` in the report says which one matched).
* A case whose image, reference or brain mask is missing is **not silently dropped**: it is listed with the reason
  (`missing.tsv`, screen).
* Two data-quality checks are **reported only** (`quality.tsv`, a WARNING lists flagged cases that stay in), on
  `--raw` if given, else on `--image`: mask fit (mean intensity in the reference / ring around it > 0.9, i.e. the
  reference is not darker than its surroundings) and a reference-mask header offset > 0.25 voxel.
* Folds: lesion count = 26-connected components of the reference; classes 0 / 1-2 / 3-5 / >5;
  `numpy.random.default_rng(--fold-seed)` shuffle per class, round-robin over `--n-folds` with the counter running on
  across classes (the same ids in the same order always give the same folds). `--no-folds` leaves them to `cmbx split`.
* Reports next to `--out`: `caselist.tsv`, `missing.tsv`, `quality.tsv`, `caselist_report.json`.
* Dataset-specific rules (which file lists the exclusions, which mask comes first) belong in a small script kept
  with the data, outside this repository.

### Any other data: a case list
CSV (or JSON list) with the columns

| column | | |
|---|---|---|
| `id` | required | unique case id |
| `image` | required | **bias-corrected** T2*-GRE or SWI magnitude (NIfTI) |
| `label` | required for CV | reference microbleed mask on the image matrix |
| `brain_mask` | recommended | on the image matrix; if empty: `image > 0` with holes filled (skull-stripped input) |
| `tissue_map` | optional | SynthSeg labels (any grid; resampled by world coordinates) for the CSF rule |
| `inpaint_mask` | optional | mask for the inpainting experiment |
| `group` | optional | stratification group (scanner, site) |
| `fold` | optional | fixed fold 0-4; if every row has one, it is the split |

Relative paths are relative to the case list. VALDO (public example): `image` = the FAST-corrected T2S
(`dev/preproc/<id>/fast_restore.nii.gz` in cmb-valdo-chain), `label` = `*_space-T2S_CMB.nii.gz`, `brain_mask` empty,
`fold` from `dev/split.json`, then `configs/baseline.yaml`. `cmbx make-phantoms data/phantoms` writes ten synthetic
cases with a case list, to try every command without real data (`configs/smoke_cpu.yaml`).

Without a `fold` column, `cmbx split` writes a deterministic stratified split (group x lesion load 0 / 1-2 / >= 3,
`random.Random(seed)`, greedy balance of fold size and lesion sum) to `data.splits`.

## Run a 5-fold cross-validation

```bash
cmbx run-experiment ../my_runs/my_baseline.yaml          # prepare (once) -> split -> 5 folds -> evaluation -> summary
cmbx run-experiment ../my_runs/my_baseline.yaml --folds 0 1   # a subset of folds; rerun later to complete (resumable)
cmbx run-experiment ../my_runs/my_baseline.yaml --set name=quick train.epochs=4 train.min_epochs=4 train.eval_every=2
```
Steps one by one: `cmbx prepare --config X`, `cmbx split --config X`, `cmbx cv --config X`.
Finished folds (`meta.json: ok`) are skipped; an interrupted fold is moved aside and restarted. A run folder refuses a
changed config: give the experiment a new `name`.

### Preprocessing (`cmbx prepare`)
Per case, into `<prepared_dir>/<id>/<id>_{image,frst,label[,tissue]}.nii.gz` + `info.json`:
orientation kept (`reorient: none`, e.g. VALDO) or RAS (`ras`) -> brain mask (`brain_mask_mode: mask`, or
`mask_and_positive` = mask & image > 0) -> **inpaint hook** (off) -> inversion -> resampling to 0.5 x 0.5 x 1.0 mm with the centre of the field of
view fixed (linear, edge slices held; label nearest neighbour with rescue of components nearest sampling would drop)
-> crop to the brain + 8 voxels -> FRST -> tissue map by nearest neighbour. Prepared volumes are shared by all
experiments with the same preprocessing; `<prepared_dir>/preprocess.json` records the settings and a config with
other settings must use another `prepared_dir`.

## Experiments = config files

A config lists only what differs from the defaults (`cmbx/config.py`, = the winner recipe). Shipped:

| config | what it tests |
|---|---|
| `configs/baseline.yaml` | the winning recipe for a case list in `data/` (e.g. VALDO); copy it for your data |
| `configs/no_frst.yaml` | image channel only (`train.channels: [image]`), same prepared data |
| `configs/csf_rule.yaml` | SynthSeg CSF rule inside the CV (inner selection and test fold) |
| `configs/inpaint_example.yaml` | paint over `inpaint_mask` before inversion (mean fill), own `prepared_dir` |
| `configs/smoke_cpu.yaml` | tiny CPU run on phantoms (checks the plumbing; numbers mean nothing) |

To make your own: copy one OUTSIDE the repository (e.g. `../my_runs/`), set `name`, the `data` paths and
`output_root` (plus `data.prepared_dir` if preprocessing changes) and the keys you want to vary, then
`cmbx run-experiment ../my_runs/your.yaml`. Paths in a config are relative to the folder you run cmbx from.

**Pre-processing hook (inpainting).** `cmbx.preprocess.inpaint(image, mask, brain, method=..., **options)` is called on
the (bias-corrected) native image before the inversion. Built in: `mean` (each masked component gets the mean of a
`ring_voxels` shell around it) and `biharmonic` (`skimage.restoration.inpaint_biharmonic` per component). Your own
method: write `def my_fill(image, mask, brain, **options) -> image` in an importable module and set
`preprocess.inpaint.method: mypackage.mymodule:my_fill`. Note: MicrobleedNet's vessel inpainting painted over 109 of
236 VALDO microbleeds; removing it was the largest single gain of cmb-valdo-chain (+0.136 F1). `info.json` reports
how many reference voxels lay under the inpaint mask.

**Post-processing without retraining.** The CV stores every test-fold probability map (`<id>_pred_proba.nii.gz`,
uint8 = round(p x 255)). Re-score them:
```bash
cmbx evaluate <run> --mode crossfold --csf-rule   # cell chosen on the other 4 folds, CSF rule on (how the +0.06 on VALDO was measured)
cmbx evaluate <run> --mode inner --csf-rule       # each fold's inner-fold cell, CSF rule on
cmbx compare <output_root>                        # one table of all runs and re-scorings -> <output_root>/summary.csv
```
(Re-binarising the uint8 maps can differ from the CV masks for voxels within 1/255 of the threshold.)

New post-processing rules go into `cmbx/postprocess.py` (`filter_components` and the fast `grid_rows`; the test
`tests/test_postprocess.py` checks that both agree).

## Own extra input channels
`preprocess.extra_channels: {name: column}` adds any map you have per case (a registered FLAIR, a vesselness or
phase map, ...): the case list gets a column with one NIfTI per case **on the image matrix**; `cmbx prepare` brings it
onto the training grid (same resampling and crop as the image) and scales it to [0, 1] inside the brain (0.5 / 99.5
percentiles), written as `<id>_<name>.nii.gz`. Then list it in `train.channels`: `[image, name]` replaces FRST,
`[image, frst, name]` adds a third channel. The number of network input channels follows the list. Example:
`configs/extra_channel_example.yaml`; test: `tests/test_extra_channel.py`. Prepared folders made without the channel
are completed on the next `prepare` (cases missing the file are prepared again).

## Outputs and metrics
```
<output_root>/<name>/
  config.yaml, splits.json
  fold<k>/  model.pt (weights + chosen cell), meta.json (epoch/threshold/min size chosen, inner and test summary,
            time, VRAM), history.json (loss and inner F1 per evaluation),
            <id>_pred.nii.gz (binary), <id>_pred_proba.nii.gz (uint8), <id>_pred_meta.json   -- on the prepared grid
  evaluation-<mode>.json, per_case-<mode>.csv, summary.json
<output_root>/summary.csv          all runs side by side
```
`summary.json`: pooled F1 / precision / sensitivity, FP per case, per-case F1 mean and median, TP/FP/FN, per-fold F1,
training hours, max VRAM. `per_case-*.csv` holds case ids: keep run folders of patient data outside the repository.

## Inference (`cmbx predict`)

```bash
cmbx predict --run <run> --image scan.nii.gz [--brain-mask mask.nii.gz] --out pred/      # one scan
cmbx predict --run <run> --cases new.csv --out pred/                                     # id,image[,brain_mask,tissue_map]
```
The scans must be bias-corrected like the training data (no label needed); they are prepared with the run's own
settings. Default model: the final model (`--folds`: mean probability of the 5 fold models); default cell: the one the
folds chose most often (`--threshold`, `--min-size` to override); `--csf-rule` needs a SynthSeg map; `--device`
auto | cuda | cpu | mps. Outputs: `<id>_mask.nii.gz` / `<id>_prob.nii.gz` / `<id>_lesions.nii.gz` (label map, value k = microbleed k of the json) on the **input image grid** (same matrix
and affine; nearest neighbour in world coordinates, a microbleed smaller than one input voxel keeps its centre voxel,
so the count matches the model grid; on coarse slices two neighbouring microbleeds may touch in the binary mask, the label map keeps them apart), copies on the model grid in `grid/`, `<id>.json` with one entry per microbleed
(volume mm3, max probability, centre in mm and as input voxel) and `predictions.csv`. Checked on 8 cases of an in-house T2* cohort:
every output on the input matrix/affine, detections on the reference lesions, equal counts on both grids.

## Options (config keys, `--set key=value`)

| key | values |
|---|---|
| `train.loss` | `bce_dice` (default, 0.5 BCE + 0.5 soft Dice), `bce`, `dice`, `focal_dice` (focal BCE gamma 2 + Dice) |
| `train.channels` | `[image, frst]` (default), `[image]`, plus own channels from `preprocess.extra_channels` |
| `train.init_from`, `train.init_channels` | fine-tuning: start from a run folder (e.g. `models/valdo_t2`) or a model.pt; input channels matched by name (new = zero weight, so training starts from exactly the pretrained output; dropped ones removed); same `model.base_channels` needed |
| `train.use_last_epoch` | `false` (inner-fold best state) / `true` (last state) |
| `train.amp_bf16`, `train.device` | bfloat16 autocast on CUDA; `auto` / `cuda` / `cuda:N` / `cpu` / `mps` |
| `preprocess.inpaint.method` | `none`, `mean`, `biharmonic`, `package.module:function` |
| `preprocess.reorient`, `brain_mask_mode`, `interpolation_order` | `none`/`ras`; `mask`/`mask_and_positive`; 1/3 |
| `postprocess.csf_rule`, `predict.tta_flip` | CSF rule in CV and prediction; flip test-time augmentation |

Every option above is run end to end (CV + final model + predict) by the option matrix on synthetic phantoms, and
`tests/test_options.py` covers losses, config checks, predict outputs, the single-scan mode and readable errors.
A wrong key or value stops with one line before anything runs (`CMBX_DEBUG=1` shows the traceback).

## Runtime and memory
One 5-fold run of 60 epochs on one GPU (RTX A5000) took **1.6 h for VALDO** (57 cases, `a03-aniso-e60-gd`) and
**1.7 h / 1.6 h for in-house T2* (75 cases) / SWI (61 cases) cohorts** (cmb-valdo-chain runs); peak VRAM 1.6-1.9 GiB per training (batch 4,
bfloat16; also measured with this package). Host RAM: all CV cases are held in memory (~130 MB per case) plus the
patch buffer (~2 GB at 800 patches): measured peak 11.5 GB for the 61 SWI cases; about 15 GB expected for the 75 T2*
cases (17.5 GB was measured before the duplicate patch buffer was removed). `cmbx prepare`: 5-20 s per case. CPU training is possible but impractically slow (use it
for the tests only).

## Reference numbers (cmb-valdo-chain)
Stage 1 only (this U-Net, inner-fold selection, no second-stage classifier), lesion-level F1, 5-fold CV:

| cohort | this recipe, seed 42 | other |
|---|---|---|
| in-house T2*-GRE (75 cases, 828 lesions) | **0.665** (TP 523, FP 221, FN 305) | + CSF rule 0.675; with the shared second-stage classifier 0.671 |
| in-house SWI (61 cases, 652 lesions) | **0.630** (seed 1: 0.602) | with the shared second-stage classifier 0.672 |
| VALDO Task 2 (57 CV cases) | **0.588** (mean over seeds 0.605 +- 0.034) | two-seed ensemble + CSF rule 0.647; MicrobleedNet re-implementation 0.245 |

Seed noise between runs is about +-0.03-0.08 F1, so compare variants with the same seed and, better, several seeds.
The VALDO numbers come from a different dataset (public, 3 sites) and are not a target for the in-house cohorts.

## Validation of this port (15 Oct 2026, on the machine the original ran on)
* `pytest`: all tests pass in a fresh conda env made from `environment.yml`.
* **Case lists.** `cmbx make-caselist` with path patterns and an exclusion file yields exactly the training cohorts
  of cmb-valdo-chain: 75 T2* and 61 SWI cases, identical case ids, lesion counts, folds and input files (checked row
  by row against the original cohort script and against the earlier, layout-specific builder).
* **Preprocessing.** `cmbx prepare` on these case lists reproduces the original training grids **bit for bit** for
  all 75 T2* and 61 SWI cases (image, FRST, label; 828 / 652 lesion components; affines within 8e-6 mm). VALDO, from
  the FAST-corrected input (3 cases, one per site): reproduced to float precision (max |difference| 6e-7 image,
  1.5e-7 FRST; labels identical). Scripts: `scripts/compare_prepared_grids.py`, `scripts/check_preprocessing_regression.py`.
* **Model numerics.** Research checkpoints (`modell.pt`) load into `cmbx.model` and reproduce the stored test-fold
  probability maps exactly (0 of ~14 M voxels differ, GPU bfloat16) for the fold-0 models of VALDO and of the two
  in-house cohorts (`scripts/check_model_numerics.py`).
* **Training path.** Against the original `cv5.py` / `netze.py` imported read-only: identical sampled patches and
  shuffle for the same seed, identical initial weights, identical loss and weights after AdamW steps
  (`scripts/check_training_equivalence.py`).
* **GPU path.** Short 5-fold CVs (4 epochs, inner evaluation every 2) ran end to end on the GPU for both in-house
  cohorts (the outputs stay outside the repository). These are plumbing checks, not
  results: the reference numbers above need the full 60 epochs.

## Provenance of every default
| setting | value | source in cmb-valdo-chain |
|---|---|---|
| network | `a03-aniso` = `Polster(AnisoUNet3D(nch), (8, 16, 16))` | `code/netze.py` |
| norm / loss | InstanceNorm / BCE+Dice (`bce_dice_ds` = BCE+Dice for one output) | `turnier.py` KONFS override |
| epochs, min epochs, eval interval | 60, 60, 5 | `--epochen 60`: max = min = 60, `eval_jede = 60 // 12` |
| patches, batch, lr, weight decay, shares, flips | 800, 4, 3e-4, 1e-4, 0.6 / 0.25, on | `cv5.REZEPT` |
| patch, edge offset, clump guard, grids, fallback | (38, 62, 94), 4, 2100, 7 x 4 cells, (0.5, 1) | `cv5.py` constants |
| seed | 42 + fold | `cv5.SEED`, `turnier.py --seed` |
| grid | 0.5 x 0.5 x 1.0 mm, no vessel inpainting | the grid builder of the original chain |
| clinical-data grid variant | `reorient: ras`, `brain_mask_mode: mask_and_positive` | the original grids of the in-house cohorts |
| case-list folds | classes 0/1-2/3-5/>5, default_rng(0) round-robin | the original cohort script |

## Deviations from the original code (all deliberate, all small)
* The inversion sets brain voxels that come out exactly 0 (the maximum voxel) to 1e-6 instead of 0 (background);
  this keeps "non-zero = brain" true. On VALDO this touches single voxels (see the regression numbers).
* The README of cmb-valdo-chain mentions "deep supervision on the decoder"; the winner network has none (its loss
  `bce_dice_ds` reduces to plain BCE + Dice). This port has none either, i.e. it matches the code.
* The generic split (`cmbx split`) adds the fold size as a balancing key (the original produced folds of 11-13
  cases); the folds of the original cohort script are reproduced by `cmbx make-caselist`.
* Brain extraction and bias correction were part of the original scripts and are not part of this package (by
  design). The mask-fit exclusion and the mask-header rewrite of the original cohort script became report-only checks;
  exclusions are passed as files (`--exclude`). `cmbx prepare` takes the reference mask array on the image matrix,
  which is what the header rewrite achieved.
* The CSF rule in the CV (`csf_rule: true`) acts on the inner selection and the test fold; in cmb-valdo-chain it was
  applied afterwards to the stored maps with a cross-fold cell choice -- that variant is `cmbx evaluate --mode
  crossfold --csf-rule`.
* Threshold choice when the inner fold holds no reference lesion at all (possible only with tiny or unbalanced
  data): F1 is 0 for every cell, so the cell with the fewest false positives is taken instead of the fallback
  (0.5, 1). With reference lesions in the inner fold the original rule applies unchanged.
* Not ported: the second-stage candidate classifier, ensembles over seeds, synthesis/augmentation variants and the
  other tournament networks.

## License and citation
Code: MIT (`LICENSE`), as cmb-valdo-chain. Data and weights are not part of the repository; VALDO is CC BY-NC-SA 4.0
(cite Sudre et al. 2024, Medical Image Analysis, doi:10.1016/j.media.2023.103029); in-house data never leave the
clinic. If you use this code, cite cmb-valdo-chain (mendeltem, 2026, https://github.com/mendeltem/cmb-valdo-chain) and
for the FRST Loy & Zelinsky 2003 (IEEE TPAMI 25(8):959-973, doi:10.1109/TPAMI.2003.1217601); the starting point of
the chain is MicrobleedNet (Sundaresan et al. 2023).
