#!/usr/bin/env python3
"""Option matrix: every command and option of cmbx end to end on synthetic phantoms (tiny settings, ~6 min on one GPU).

    python scripts/option_matrix.py [--work /tmp/cmbx_matrix] [--device auto|cuda|cpu]

Covers prepare / split / cv (partial + resume) / evaluate (all modes, CSF rule) / train-final / compare / predict
(final, folds, single scan, case list without labels, thresholds, CSF rule, grid only, CPU) and run-experiment for
every config variant (no FRST, CSF rule, own channel, inpainting mean/biharmonic, losses, AMP off, last epoch, RAS,
mask_and_positive, cubic interpolation, TTA), plus user errors that must fail with one readable line.
Writes <work>/matrix_results.json; exit code 1 if any step failed. No patient data involved."""
import argparse
import csv, json, os, subprocess, sys, time, shutil, tempfile
import numpy as np, nibabel as nib
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--work", default=os.path.join(tempfile.gettempdir(), "cmbx_matrix"))
ap.add_argument("--device", default="auto")
A = ap.parse_args()
W, DEV = os.path.abspath(A.work), A.device
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
shutil.rmtree(W, ignore_errors=True); os.makedirs(W); os.chdir(W)
PY = sys.executable
TINY = ["train.epochs=2", "train.min_epochs=2", "train.eval_every=1", "train.n_patches=8", "train.batch_size=4",
        "train.patch_size=[16,32,32]", "model.base_channels=8", "preprocess.workers=2", f"train.device={DEV}"]
results = []
def run(name, args, expect=(), ok_rc=0, stdin=None):
    t = time.time(); p = subprocess.run([PY, "-m", "cmbx", *args], capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip().splitlines()
    missing = [e for e in expect if not os.path.exists(e)]
    good = (p.returncode == ok_rc) and not missing
    results.append(dict(step=name, ok=good, rc=p.returncode, s=round(time.time() - t, 1), missing=missing, tail=out[-6:]))
    print(f"{'PASS' if good else 'FAIL'} {name:45s} rc={p.returncode} {time.time()-t:5.1f}s {('missing ' + str(missing)) if missing else ''}", flush=True)
    if not good: print("     " + "\n     ".join(out[-12:]))
    return p
def data_set(prefix):
    return [f"data.cases={W}/ph/cases.csv", f"data.prepared_dir={W}/{prefix}_prep", f"data.splits={W}/splits.json", f"output_root={W}/runs"]

# data: phantoms + extra columns (own channel, inpaint mask) + 2 new unlabeled cases
run("make-phantoms", ["make-phantoms", "ph", "--n", "10"], ["ph/cases.csv"])
rows = list(csv.DictReader(open("ph/cases.csv")))
for r in rows:
    img = nib.load(os.path.join("ph", r["image"])); a = np.zeros(img.shape, np.uint8); a[20:24, 20:24, 8:10] = 1
    p = os.path.join("ph", r["id"], f"{r['id']}_inpaint.nii.gz"); nib.save(nib.Nifti1Image(a, img.affine), p)
    r["inpaint_mask"] = os.path.relpath(p, "ph"); r["myvessel_path"] = r["tissue_map"]
with open("ph/cases.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
run("make-phantoms (new cases)", ["make-phantoms", "new", "--n", "2", "--seed", "7"], ["new/cases.csv"])
nr = list(csv.DictReader(open("new/cases.csv")))
with open("new/unlabeled.csv", "w", newline="") as f:          # only what a user has for a new scan
    w = csv.writer(f); w.writerow(["id", "image", "brain_mask"]); [w.writerow([r["id"], r["image"], r["brain_mask"]]) for r in nr]
with open("new/image_only.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "image"]); [w.writerow([r["id"], r["image"]]) for r in nr]

C = f"{REPO}/configs"
run("check-env", ["check-env"])
run("prepare", ["prepare", "--config", f"{C}/baseline.yaml", "--set", *data_set("base"), *TINY], ["base_prep"])
run("split", ["split", "--config", f"{C}/baseline.yaml", "--set", *data_set("base"), *TINY], ["splits.json"])
run("cv --folds 0 1", ["cv", "--config", f"{C}/baseline.yaml", "--set", *data_set("base"), *TINY, "--folds", "0", "1"], ["runs/baseline/fold1/model.pt"])
run("cv resume (rest)", ["cv", "--config", f"{C}/baseline.yaml", "--set", *data_set("base"), *TINY], ["runs/baseline/fold4/model.pt"])
for mode in ("stored", "inner", "crossfold"):
    run(f"evaluate {mode}", ["evaluate", "runs/baseline", "--mode", mode])
run("evaluate crossfold --csf-rule --tag", ["evaluate", "runs/baseline", "--mode", "crossfold", "--csf-rule", "--tag", "csf"], ["runs/baseline/evaluation-csf.json"])
run("train-final", ["train-final", f"{C}/baseline.yaml", "--set", *data_set("base"), *TINY], ["runs/baseline/final/model.pt"])
run("compare", ["compare", "runs"], ["runs/summary.csv"])
run("predict final (unlabeled)", ["predict", "--run", "runs/baseline", "--cases", "new/unlabeled.csv", "--out", "pred1"], ["pred1/phantom-00_mask.nii.gz"])
run("predict image only (no mask column)", ["predict", "--run", "runs/baseline", "--cases", "new/image_only.csv", "--out", "pred2"], ["pred2/phantom-00_mask.nii.gz"])
run("predict --folds", ["predict", "--run", "runs/baseline", "--cases", "new/unlabeled.csv", "--out", "pred3", "--folds"], ["pred3/phantom-00_mask.nii.gz"])
run("predict thr/min/csf", ["predict", "--run", "runs/baseline", "--cases", "ph/cases.csv", "--out", "pred4", "--threshold", "0.3", "--min-size", "1", "--csf-rule"], ["pred4/phantom-00_mask.nii.gz"])
run("predict --device cpu", ["predict", "--run", "runs/baseline", "--cases", "new/unlabeled.csv", "--out", "pred5", "--device", "cpu"], ["pred5/phantom-00_mask.nii.gz"])
run("predict tta (via run config)", ["predict", "--run", "runs/baseline", "--cases", "new/unlabeled.csv", "--out", "pred6"], ["pred6/phantom-00_mask.nii.gz"])
for cfg, extra in [("no_frst", []), ("csf_rule", []), ("extra_channel_example", []), ("inpaint_example", ["name=inpaint_mean"]),
                   ("inpaint_example", ["preprocess.inpaint.method=biharmonic", "name=inpaint_biharmonic", f"data.prepared_dir={W}/ib_prep"]),
                   ("baseline", ["name=amp_off", "train.amp_bf16=false"]), ("baseline", ["name=last_epoch", "train.use_last_epoch=true"]),
                   ("baseline", ["name=loss_bce", "train.loss=bce"]), ("baseline", ["name=loss_dice", "train.loss=dice"]), ("baseline", ["name=loss_focal", "train.loss=focal_dice"]), ("baseline", ["name=ras", "preprocess.reorient=ras", f"data.prepared_dir={W}/ras_prep"]),
                   ("baseline", ["name=mask_pos", "preprocess.brain_mask_mode=mask_and_positive", f"data.prepared_dir={W}/mp_prep"]),
                   ("baseline", ["name=order3", "preprocess.interpolation_order=3", f"data.prepared_dir={W}/o3_prep"]),
                   ("baseline", ["name=tta", "predict.tta_flip=true"])]:
    name = next((e.split("=", 1)[1] for e in extra if e.startswith("name=")), cfg)
    prep = next((e.split("=", 1)[1] for e in extra if e.startswith("data.prepared_dir=")), None) or f"{W}/{cfg}_prep"
    sets = [s for s in data_set(cfg) if not s.startswith("data.prepared_dir")] + [f"data.prepared_dir={prep}"] + TINY + extra
    run(f"run-experiment {name}", ["run-experiment", f"{C}/{cfg}.yaml", "--set", *sets], [f"runs/{name}/summary.json", f"runs/{name}/final/model.pt"])
    if name in ("extra_channel_example", "inpaint_mean", "no_frst", "tta"):
        run(f"  predict with {name}", ["predict", "--run", f"runs/{name}", "--cases", "ph/cases.csv" if name in ("extra_channel_example", "inpaint_mean") else "new/unlabeled.csv", "--out", f"pred_{name}"], [f"pred_{name}/phantom-00_mask.nii.gz"])
r0 = rows[0]
run("predict --image single scan", ["predict", "--run", "runs/baseline", "--image", os.path.join(W, "ph", r0["image"]), "--brain-mask", os.path.join(W, "ph", r0["brain_mask"]), "--id", "one", "--out", "pred_one"], ["pred_one/one_mask.nii.gz", "pred_one/predictions.csv"])
run("predict --image + --tissue-map --csf-rule", ["predict", "--run", "runs/baseline", "--image", os.path.join(W, "ph", r0["image"]), "--tissue-map", os.path.join(W, "ph", r0["tissue_map"]), "--csf-rule", "--out", "pred_csf1"], ["pred_csf1/predictions.csv"])
run("predict --grid-only", ["predict", "--run", "runs/baseline", "--cases", "new/unlabeled.csv", "--out", "pred_grid", "--grid-only"], ["pred_grid/grid/phantom-00_mask.nii.gz"])
run("cv --folds 3 (single fold rerun is skipped)", ["cv", "--config", f"{C}/baseline.yaml", "--set", *data_set("base"), *TINY, "--folds", "3"])
# error handling: should fail cleanly (rc != 0, readable message, no traceback)
bm = nib.load(os.path.join(W, "ph", rows[0]["brain_mask"])); nib.save(nib.Nifti1Image(np.asarray(bm.dataobj)[:-4], bm.affine), W + "/new/cropped_mask.nii.gz")
bad = W + "/new/bad.csv"
with open(bad, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "image", "brain_mask"]); w.writerow(["x", os.path.join(W, "new", nr[0]["image"]), W + "/new/cropped_mask.nii.gz"])
for nm, args in [("ERR mask on another matrix", ["predict", "--run", "runs/baseline", "--cases", bad, "--out", "pred_bad"]), ("ERR csf-rule without tissue", ["predict", "--run", "runs/baseline", "--cases", "new/unlabeled.csv", "--out", "x", "--csf-rule"]), ("ERR cuda forced", ["predict", "--run", "runs/baseline", "--cases", "new/unlabeled.csv", "--out", "x", "--device", "cuda:7"]),
                 ("ERR missing file", ["predict", "--run", "runs/baseline", "--cases", "/nonexistent.csv", "--out", "x"]),
                 ("ERR unknown config key", ["run-experiment", f"{C}/baseline.yaml", "--set", *data_set("base"), *TINY, "train.epochz=3"]),
                 ("ERR run folder missing", ["predict", "--run", "runs/nope", "--cases", "new/unlabeled.csv", "--out", "x"])]:
    p = subprocess.run([PY, "-m", "cmbx", *args], capture_output=True, text=True)
    tb = "Traceback" in p.stderr
    print(f"INFO {nm:45s} rc={p.returncode} {'TRACEBACK' if tb else 'clean'}: {(p.stderr or p.stdout).strip().splitlines()[-1][:150] if (p.stderr or p.stdout).strip() else ''}")
    results.append(dict(step=nm, ok=None, rc=p.returncode, traceback=tb, msg=(p.stderr or p.stdout).strip().splitlines()[-1][:200] if (p.stderr or p.stdout).strip() else ""))
json.dump(results, open(f"{W}/matrix_results.json", "w"), indent=1)
n_fail = sum(1 for r in results if r["ok"] is False) + sum(1 for r in results if r.get("traceback"))
print(f"\n{sum(1 for r in results if r['ok'] is True)} PASS, {n_fail} FAIL (a traceback on a user error counts as a failure)")
sys.exit(1 if n_fail else 0)
