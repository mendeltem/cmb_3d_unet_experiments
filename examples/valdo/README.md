# Five real T2* scans from VALDO 2021 (for `test_all_cases.py`)

| Case | Cohort | Scanner | Slices | Microbleeds |
|---|---|---|---|---|
| sub-103 | SABRE | Philips 3 T, 2D T2* GRE | 3 mm at 4 mm spacing | 6 |
| sub-104 | SABRE | Philips 3 T, 2D T2* GRE | 3 mm at 4 mm spacing | 0 |
| sub-106 | SABRE | Philips 3 T, 2D T2* GRE | 3 mm at 4 mm spacing | 3 |
| sub-305 | ALFA | GE Discovery 3 T, 2D T2* GRE | 3 mm | 2 |
| sub-320 | ALFA | GE Discovery 3 T, 2D T2* GRE | 3 mm | 2 |

`raw/<case>_T2starw.nii.gz` is the VALDO file `<case>_space-T2S_desc-masked_T2S.nii.gz` (brain-masked by the challenge, not
bias-corrected), `raw/<case>_cmb.nii.gz` is `<case>_space-T2S_CMB.nii.gz` (the reference microbleed mask). Files are unchanged,
only renamed. They are here so that every step of the repository can be tried on real scans at once; replace them with
your own data for real use.

**Note:** the pretrained VALDO model of this repository was trained on all 72 VALDO training scans, including these five.
Its results here show that the pipeline runs, not how well it generalises.

## Licence and attribution (these five cases only)

The VALDO 2021 training data are licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)**, https://creativecommons.org/licenses/by-nc-sa/4.0/ . This folder and anything derived from these files
stay under that licence: non-commercial use only, with attribution, and shared under the same terms. The MIT licence of the
code in this repository does not apply to these files.

Please cite and acknowledge:

- Dataset: Training dataset for the VALDO 2021 challenge, Zenodo, doi:10.5281/zenodo.4520773.
- Challenge: Sudre CH et al. 2024. Where is VALDO? VAscular Lesions Detection and segmentatiOn challenge at MICCAI 2021.
  Medical Image Analysis 91:103029. doi:10.1016/j.media.2023.103029.
- SABRE: Tillin T et al. 2012. Southall And Brent REvisited: cohort profile of SABRE. International Journal of Epidemiology
  41(1):33-42. doi:10.1093/ije/dyq175. SABRE is funded by the Wellcome Trust and the British Heart Foundation.
- ALFA: data of the ALFA study, Barcelonaβeta Brain Research Center; please name "for the ALFA Study" as corporate author
  together with the contributors listed on the Zenodo record, as the dataset licence asks.
- Further funders and acknowledgements as listed on the Zenodo record above.
