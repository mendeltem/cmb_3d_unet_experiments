"""Predict cerebral microbleeds on your own cohort with a trained (or the pretrained VALDO) 3D U-Net, settings from .env.

    cp examples/env/.env.example examples/env/.env      # once; then set CMB_DATA and the rest (keys: cmb_env.py)
    python examples/env/run_predict.py                                 # MODEL=pretrained: the public VALDO model as it is
    python examples/env/run_predict.py --set MODEL=../my_runs/my_cohort/finetune_valdo_not_fold0 TEST_FOLD=0 --force true
On the HPC: sbatch -p gpu --gres=shard:1 --mem=32G --time=02:00:00 --wrap "source env.sh && python examples/env/run_predict.py"
With TEST_FOLD=k only the cases of fold k are predicted (the ones a model trained with the same TEST_FOLD never saw).

Steps (each prints what it does):
  1 read the settings (.env, then --set overrides)
  2 predict: the repository example examples/python/02_predict_microbleeds.py (step 1 of the method: U-Net, threshold and
    minimum size from training), on the chosen cases of <CMB_DATA>/derivatives/preprocessed
  3 compare with the reference masks <case>_cmb.nii.gz where they exist: lesion-level TP/FP/FN (a reference microbleed
    counts as found if a predicted one touches it, 26-connected), and the number of microbleeds per patient
Outputs in <OUT_DIR> (default <CMB_RUNS>/<NAME>/predict_<MODEL name>_<all|foldk>):
  <case>_cmb.nii.gz, <case>_lesions.nii.gz, <case>_prob.nii.gz   per case, on the scan's own grid
  microbleeds.csv, predictions.csv                               from the repository example
  comparison_per_case.csv                                        predicted vs reference count, TP / FP / FN per case
  comparison_summary.json                                        lesion F1, precision, recall, FP per scan, count agreement
The outputs are derived from your scans: keep them where your data may be stored.
"""
import csv
import json
import os
import subprocess
import sys

from cmb_env import REPO, cases_for, example, linked_input, settings


def components(mask):
    """Connected microbleeds of a binary mask (26-connected). Returns (label map, count)."""
    import numpy as np
    from scipy import ndimage
    return ndimage.label(mask > 0, structure=np.ones((3, 3, 3), bool))


def compare_case(reference_path, predicted_path):
    """Lesion-level counts of one case: TP = reference microbleeds touched by a prediction, FP = predicted ones touching
    no reference microbleed, FN = reference ones not touched. Both masks must be on the same grid."""
    import nibabel as nib
    import numpy as np
    ref = np.asarray(nib.load(reference_path).dataobj)
    pred = np.asarray(nib.load(predicted_path).dataobj)
    if ref.shape != pred.shape:
        raise SystemExit(f"grid differs: {reference_path} {ref.shape} vs {predicted_path} {pred.shape}")
    ref_lab, n_ref = components(ref)
    pred_lab, n_pred = components(pred)
    tp = len(set(np.unique(ref_lab[pred_lab > 0])) - {0})
    fp = n_pred - len(set(np.unique(pred_lab[ref_lab > 0])) - {0})
    return dict(reference=n_ref, predicted=n_pred, tp=tp, fp=fp, fn=n_ref - tp)


def summarise(rows):
    """Lesion F1, precision, recall, false positives per scan; Spearman of the counts; agreement on '> 4 CMB'."""
    from scipy.stats import spearmanr
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    over4 = [(r["reference"] > 4, r["predicted"] > 4) for r in rows]
    return dict(cases=len(rows), reference_microbleeds=tp + fn, tp=tp, fp=fp, fn=fn,
                lesion_f1=round(f1, 4), precision=round(tp / (tp + fp), 4) if tp + fp else None,
                recall=round(tp / (tp + fn), 4) if tp + fn else None, fp_per_scan=round(fp / len(rows), 3),
                count_spearman=round(float(spearmanr([r["reference"] for r in rows], [r["predicted"] for r in rows])[0]), 4)
                if len(rows) > 2 else None,
                more_than_4_agree=sum(a == b for a, b in over4), more_than_4_missed=sum(a and not b for a, b in over4),
                more_than_4_false=sum(b and not a for a, b in over4))


if __name__ == "__main__":     # the steps
    S, force = settings(description=__doc__.split("\n")[0])
    ids, tag = cases_for(S, "predict")
    model_name = "valdo" if S["MODEL"] in ("pre", "pretrained") else os.path.basename(S["MODEL"].rstrip("/"))
    out = S["OUT_DIR"] or os.path.join(S["CMB_RUNS"], S["NAME"], f"predict_{model_name}_{tag}")
    example = example("02_predict_microbleeds.py")
    print(f"  cohort {S['NAME']}, model {S['MODEL']}, {len(ids)} cases ({tag}), device {S['DEVICE']} -> {out}", flush=True)
    inp = linked_input(S, ids, tag)

    # 2 predict (skipped when predictions exist, unless --force true)
    if os.path.exists(os.path.join(out, "predictions.csv")) and not force:
        print(f"predictions exist in {out}; comparing only (--force true predicts again)")
    else:
        cmd = [sys.executable, example, "-i", inp, "-m", S["MODEL"], "-o", out, "--device", S["DEVICE"],
               "--workers", S["WORKERS"], "--force", "true"]
        cmd += ["--threshold", S["THRESHOLD"]] if S["THRESHOLD"] else []
        cmd += ["--min_size", S["MIN_SIZE"]] if S["MIN_SIZE"] else []
        os.makedirs(out, exist_ok=True)
        json.dump(dict(settings=S, command=cmd), open(os.path.join(out, "run_settings.json"), "w"), indent=1)
        print("running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=REPO)

    # 3 compare with the reference masks where they exist
    rows = []
    for r in csv.DictReader(open(os.path.join(out, "predictions.csv"))):
        ref = os.path.join(inp, r["case"], f"{r['case']}_cmb.nii.gz")
        if os.path.exists(ref):
            rows.append(dict(case=r["case"], **compare_case(ref, r["mask"])))
    if not rows:
        print("no reference masks found: predictions only, no comparison")
    else:
        with open(os.path.join(out, "comparison_per_case.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        summary = summarise(rows)
        json.dump(summary, open(os.path.join(out, "comparison_summary.json"), "w"), indent=1)
        print(f"compared {summary['cases']} case(s): lesion F1 {summary['lesion_f1']}, precision {summary['precision']}, "
              f"recall {summary['recall']}, {summary['fp_per_scan']} false positives per scan; count Spearman "
              f"{summary['count_spearman']}; '> 4 CMB' agrees in {summary['more_than_4_agree']} of {summary['cases']}")
        print(f"-> {out}/comparison_summary.json, comparison_per_case.csv")
