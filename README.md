# cmb_3d_unet_experiments
> Find cerebral microbleeds on T2* and SWI MRI with a 3D U-Net. Same commands and folders as MicrobleedNet.

![How it works](docs/architecture.svg)

- Step 1: a 3D U-Net marks microbleed candidates (input: image + FRST).
- Step 2 (optional): a classifier re-checks every candidate; the teacher–student network of MicrobleedNet or a small CNN.
- Scripts written for [MicrobleedNet](https://github.com/v-sundaresan/microbleed-detection) run with `cmbx` in place of
  `microbleednet`. Differences: [below](#differences-to-microbleednet).

## Install

```bash
git clone https://github.com/mendeltem/cmb_3d_unet_experiments.git
```

```bash
cd cmb_3d_unet_experiments
```

```bash
./install.sh
```

```bash
source env.sh
```

- Installs Python, PyTorch for your GPU, `cmbx` and the pretrained model. Ends with `ALL REQUIRED CHECKS PASSED`.
- `source env.sh` in every new shell.

## Data

```
my_cohort/
├── files.xlsx                              list of all files
└── derivatives/
    ├── preprocessed/sub-001/
    │   ├── sub-001_t2star.nii.gz           bias-corrected T2* (SWI: sub-001_swi.nii.gz)
    │   ├── sub-001_brainmask.nii.gz        brain mask
    │   ├── sub-001_cmb.nii.gz              microbleed mask (training only)
    │   ├── sub-001_synthseg.nii.gz         optional: SynthSeg, for --csf_rule True
    │   └── sub-001_flair.nii.gz            optional: FLAIR on the T2* matrix, for --flair True
    ├── images/sub-001_preproc.nii.gz       MicrobleedNet input (inverted T2*)
    ├── frsts/sub-001_frst.nii.gz           MicrobleedNet FRST
    ├── images_inpainted/                   optional: the same with vessel inpainting, for --inpaint True
    └── frsts_inpainted/
```

- Examples: `examples/train` (10 synthetic cases), `examples/new` (2, no labels), `examples/raw` (scans before preprocessing).
- `images/` and `frsts/` are MicrobleedNet's own folder names: both methods read the same inputs.

## Preprocess

```bash
cmbx preprocess -i examples/raw -ir _T2starw -f True -b examples/raw -br _brainmask.nii.gz -l examples/raw -lr _cmb.nii.gz -flr _FLAIR -o ../my_runs/cohort/derivatives/preprocessed
```

- Writes one folder per case and `../my_runs/cohort/files.xlsx`.
- Without `-f True`: skull stripping and bias correction with FSL (BET + FAST), as MicrobleedNet.
- FLAIR (`-flr`, same session): BET, FAST, then onto the T2* by the scanner coordinates (`--flair_registered True` if it is already).
- `--synthseg True` adds the tissue map (FreeSurfer; from the FLAIR when there is one). `-mod swi` names the scan `_swi`.
- `--microbleednet_inputs True` adds `images/` + `frsts/` with MicrobleedNet's own code (installed separately),
  `--inpaint True` also `images_inpainted/` + `frsts_inpainted/`. Without them, `cmbx train -i ../my_runs/cohort`
  computes the FRST itself.

## Train

```bash
cmbx train -i examples/train/derivatives -l examples/train/derivatives/preprocessed -m ../my_runs/my_model
```

- As MicrobleedNet: `-i` holds `images/` and `frsts/`, `-l` the microbleed masks; image and FRST go into the network.
- One model -> `../my_runs/my_model/final/model.pt`; 20 % of the cases choose the threshold (`-tr_prop 0.8`).
- About 25 min on one GPU for 60 cases.

Switch steps on or off, e.g. without FRST:

```bash
cmbx train -i examples/train/derivatives -l examples/train/derivatives/preprocessed -m ../my_runs/no_frst --frst False
```

| option | default | |
|---|---|---|
| `--frst` | `True` | FRST as second input channel |
| `--inpaint` | `False` | use `images_inpainted/` + `frsts_inpainted/` (MicrobleedNet's vessel inpainting) |
| `--frst_from` | `microbleednet` | FRST from `frsts/`, or `cmbx` (computed on the 0.5 mm grid) |
| `--csf_rule` | `False` | drop detections in CSF (needs `_synthseg`) |
| `--flair` | `False` | FLAIR as extra input channel (needs `_flair`) |
| `-cand_disc` | `False` | train step 2, MicrobleedNet's teacher–student (`--disc_model cnn`: small CNN) |
| `-ep`, `-ilr`, `-bs` | `60`, `0.0003`, `4` | epochs, learning rate, batch size |

Every MicrobleedNet option works (`-tr_prop`, `-es`, `-cp_type`, `-lrm`, `-opt`, ...): `cmbx train --help`,
[docs/DETAILS.md](docs/DETAILS.md#microbleednet-options).

Without the switch the options you leave out keep cmbx's own recipe. To start from MicrobleedNet's own values
instead, add `--defaults microbleednet` (also on `fine_tune` and `cross_validate`): every MicrobleedNet option you did
not type on the command line then takes the value MicrobleedNet uses (`-tr_prop 0.8`, `-bfactor 10`, `-psize 48`,
`-af 2`, `-da True`, `-cand_disc True`, `-lrm 10`, `-gamma 0.1`, `-opt adam`, `-eps 1e-4`, `-mom 0.9`, `-es 20`,
`-cp_type last`, `-cp_n 10`, `-bs 4`, `-ep 100`, `-ilr 0.001`; for `fine_tune` `-bs 8`, `-ep 60`, `-ilr 0.0001`,
for `cross_validate` `-bs 8`, `-ep 60`, `-ilr 0.001`). Options you type still win, so
`cmbx train ... --defaults microbleednet -ep 60` runs MicrobleedNet's settings for 60 epochs. `--defaults cmbx` (the
default) changes nothing. `-ftlayers` is not filled: its numbers refer to MicrobleedNet's own decoder blocks ("1 2 -> final two decoder
layers"), not to the cmbx levels (1 = output ... 9 = input), so set it yourself if needed. A key you set with `--set`
(e.g. `--set train.lr=0.0003`) is not overridden by the preset.

## Fine-tune the pretrained model

```bash
cmbx fine_tune -i examples/train/derivatives -l examples/train/derivatives/preprocessed -m pre -o ../my_runs/my_finetuned
```

- `-m pre`: the pretrained VALDO model (trained with the cmbx FRST, which it keeps); or any model folder. Learning rate 0.0001.

## Predict

```bash
cmbx evaluate -i examples/new/derivatives -m ../my_runs/my_model -o ../my_runs/pred
```

- Masks on the input grid: `../my_runs/pred/final_predictions/predicted_final_cmbx_<case>.nii.gz`.
- One entry per microbleed: `../my_runs/pred/details/<case>.json`. `-int True` adds the step-1 maps.

## Cross-validation

```bash
cmbx cross_validate -i examples/train/derivatives -l examples/train/derivatives/preprocessed -o ../my_runs/cv -fold 5
```

- Lesion-level F1 out of fold: `../my_runs/cv/summary.json`. Same folds for every setting you compare.
- Five trainings of 60 epochs: about 1 h on one GPU even for the examples; `-ep 2` for a quick try.

## Differences to MicrobleedNet

| | MicrobleedNet (Sundaresan 2023) | cmbx | effect on VALDO (lesion F1) |
|---|---|---|---|
| vessel inpainting | always | off; `--inpaint True` | without: +0.136 |
| FRST channel | yes | yes; `--frst False` | without: −0.053 |
| step 1 | 3D U-Net on 48³ patches, native grid | anisotropic 3D U-Net, 38 × 62 × 94 patches, 0.5 × 0.5 × 1 mm grid | 24 mm patches: −0.216 |
| step 2 | teacher–student; the teacher is loaded from pretrained files, never trained | optional; teacher trained, then student distilled | |
| CSF rule | no | optional; `--csf_rule True` | +0.06 |
| threshold | fixed | chosen on held-out cases | |

Details and sources: [docs/DETAILS.md](docs/DETAILS.md#differences-to-microbleednet).

## Links

- Details, all options, validation: [docs/DETAILS.md](docs/DETAILS.md)
- MicrobleedNet: https://github.com/v-sundaresan/microbleed-detection
- Issues: https://github.com/mendeltem/cmb_3d_unet_experiments/issues

## Licensing

- Code: MIT. Pretrained VALDO weights: CC BY-NC-SA 4.0 (non-commercial).
- Cite MicrobleedNet: Sundaresan et al., Front Neuroinform 2023, doi:10.3389/fninf.2023.1204186.
- Cite VALDO: Sudre et al., Med Image Anal 2024, doi:10.1016/j.media.2023.103029.
- Patient data never go into this repository.

---

<sub>**Hardware:** NVIDIA or AMD GPU with 2 GB or more (CPU works, slowly); 32 GB RAM for training; only the GPU driver,
no CUDA toolkit. **Software:** Linux or macOS; conda or Python 3.10+ (set up by `install.sh`); FSL for
`cmbx preprocess` without `-f True`; FreeSurfer 7.3+ for `--synthseg True`.</sub>
