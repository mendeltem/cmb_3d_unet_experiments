"""Run every experiment of this repository on five real VALDO T2* scans and write a PDF report.

    python test_all_cases.py                      # all experiments, results in ../my_runs/test_all_cases
    python test_all_cases.py --device cpu         # without a GPU (slower)
    python test_all_cases.py --epochs 2           # quicker (default 4: enough to run, not to learn)
    python test_all_cases.py --only pretrained finetune     # some experiments only (names: --list)

test.py checks quickly that every function works. This script runs whole experiments, one after another, on the
example cases in examples/valdo (five VALDO 2021 scans: four with microbleeds, one without; CC BY-NC-SA 4.0, see
examples/valdo/README.md), with the same `cmbx` commands you use on your own data:

  1. preprocess   examples/valdo/raw -> <out>/derivatives/preprocessed (scans are already brain-masked: -f True);
                  the shipped SynthSeg labels (examples/valdo/synthseg) are copied next to each case for --csf_rule
  2. split        3 cases to train (sub-106, sub-305, sub-320), 2 to test (sub-103 with 6 microbleeds, sub-104 with none)
  3. experiments  the pretrained VALDO model as downloaded; trained from scratch with and without FRST; with the CSF
                  rule; fine-tuned from the pretrained model; with the step-2 classifier (teacher-student and small
                  CNN); vessel inpainting (needs microbleednet); a 3-fold cross-validation
  4. scores       every experiment on the same 2 test cases: TP, FP, FN, sensitivity, precision, lesion F1
                  (a reference microbleed counts as found when a prediction touches it; cmbx.metrics.hits)
  5. report       <out>/report/summary.pdf (all experiments) and <out>/report/<case>.pdf (one per test case),
                  <out>/report/results.csv

Results with 4 epochs on 3 training cases are NOT meaningful; they show that every path runs end to end. The pretrained
model was trained on all 72 VALDO scans, these five included. For your own data: replace examples/valdo by your cases
(same file names) and raise --epochs.
"""
import argparse, csv, glob, json, os, shutil, subprocess, sys, time

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, "examples", "valdo")
TRAIN, TEST = ["sub-106", "sub-305", "sub-320"], ["sub-103", "sub-104"]


def have(module):
    import importlib.util
    return importlib.util.find_spec(module) is not None


def cmbx(*args, log):
    """Run one cmbx command; returns (ok, seconds). The output goes to the log file."""
    cmd = [sys.executable, "-m", "cmbx", *map(str, args)]
    t0 = time.time()
    with open(log, "a") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n"); fh.flush()
        r = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    return r.returncode == 0, time.time() - t0


def experiments(a, pre, train_dir):
    """name -> (description, list of steps, requirement or None). A step: ("train"/"fine_tune"/"evaluate"/..., args)."""
    ep, dev = ["-ep", a.epochs, "--patches", a.patches, "--device", a.device], ["--device", a.device]
    m = lambda n: os.path.join(a.out, "models", n)
    E = {}
    E["pretrained"] = ("pretrained VALDO model, as downloaded", [("evaluate", ["-m", "pre", *dev])], None)
    E["pretrained_csf"] = ("pretrained + CSF rule (drop detections in CSF, SynthSeg)", [("evaluate", ["-m", "pre", "--csf_rule", "True", *dev])], None)
    E["scratch"] = ("trained from scratch, image + FRST", [("train", ["-i", train_dir, "-m", m("scratch"), "--frst_from", "cmbx", *ep]),
                                                          ("evaluate", ["-m", m("scratch"), *dev])], None)
    E["scratch_no_frst"] = ("trained from scratch, image only (no FRST)", [("train", ["-i", train_dir, "-m", m("no_frst"), "--frst", "False", *ep]),
                                                                          ("evaluate", ["-m", m("no_frst"), *dev])], None)
    E["scratch_csf"] = ("trained from scratch + CSF rule", [("evaluate", ["-m", m("scratch"), "--csf_rule", "True", *dev])], "scratch")
    E["finetune"] = ("fine-tuned from the pretrained model", [("fine_tune", ["-i", train_dir, "-m", "pre", "-o", m("finetune"), *ep]),
                                                              ("evaluate", ["-m", m("finetune"), *dev])], None)
    E["finetune_csf"] = ("fine-tuned + CSF rule", [("evaluate", ["-m", m("finetune"), "--csf_rule", "True", *dev])], "finetune")
    E["step2_teacher_student"] = ("scratch + step-2 classifier, MicrobleedNet teacher-student",
                                  [("train", ["-i", train_dir, "-m", m("step2_ts"), "--frst_from", "cmbx", "-cand_disc", "True", *ep]),
                                   ("evaluate", ["-m", m("step2_ts"), "-cand_disc", "True", *dev])], None)
    E["step2_cnn"] = ("scratch + step-2 classifier, small CNN",
                      [("train", ["-i", train_dir, "-m", m("step2_cnn"), "--frst_from", "cmbx", "-cand_disc", "True", "--disc_model", "cnn", *ep]),
                       ("evaluate", ["-m", m("step2_cnn"), "-cand_disc", "True", *dev])], None)
    E["inpaint"] = ("scratch on vessel-inpainted images (MicrobleedNet inpaint_vessels)",
                    [("train", ["-i", train_dir, "-m", m("inpaint"), "--inpaint", "True", "--frst_from", "cmbx", *ep]),
                     ("evaluate", ["-m", m("inpaint"), *dev])], "microbleednet")
    E["cross_validate"] = ("3-fold cross-validation on the 3 training cases (out-of-fold scores)",
                           [("cross_validate", ["-i", train_dir, "-o", os.path.join(a.out, "cv"), "-fold", 3, "--frst_from", "cmbx", *ep])], None)
    return E


def score(pred_dir, pre):
    """Per test case: TP/FP/FN of the predicted mask against the reference, both on the scan grid."""
    import nibabel as nib
    from cmbx.metrics import hits
    out = {}
    for c in TEST:
        f = glob.glob(os.path.join(pred_dir, "final_predictions", f"predicted_final_cmbx_{c}*.nii*"))
        if not f:
            continue
        ref = nib.load(os.path.join(pre, c, f"{c}_cmb.nii.gz")).get_fdata()
        h = hits(ref, nib.load(f[0]).get_fdata())
        out[c] = dict(tp=h["tp"], fp=h["fp"], fn=h["fn"], mask=f[0])
    return out


def metrics(tp, fp, fn):
    sens = tp / (tp + fn) if tp + fn else None
    prec = tp / (tp + fp) if tp + fp else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return sens, prec, f1


def fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


def report(rows, a, pre):
    """summary.pdf (all experiments) and one PDF per test case."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import nibabel as nib, numpy as np
    from scipy import ndimage
    rd = os.path.join(a.out, "report"); os.makedirs(rd, exist_ok=True)
    done = [r for r in rows if r["status"] == "ok" and r.get("cases")]
    with PdfPages(os.path.join(rd, "summary.pdf")) as pdf:
        fig = plt.figure(figsize=(11.7, 8.3)); fig.text(0.04, 0.95, "test_all_cases.py -- all experiments on the VALDO examples",
                                                      fontsize=15, weight="bold")
        fig.text(0.04, 0.91, f"{time.strftime('%Y-%m-%d %H:%M')} | epochs {a.epochs}, patches {a.patches}, device {a.device} | "
                 f"train {', '.join(TRAIN)} | test {', '.join(TEST)} | scores pooled over the test cases", fontsize=9)
        cols = ["experiment", "status", "TP", "FP", "FN", "sensitivity", "precision", "lesion F1", "time"]
        cells = [[r["name"], r["status"], r.get("tp", ""), r.get("fp", ""), r.get("fn", ""), fmt(r.get("sens")), fmt(r.get("prec")),
                  fmt(r.get("f1")), f"{r['seconds'] / 60:.1f} min"] for r in rows]
        ax = fig.add_axes([0.04, 0.32, 0.92, 0.56]); ax.axis("off")
        t = ax.table(cellText=cells, colLabels=cols, loc="upper left", cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(8.5)
        t.scale(1, 1.45)
        for (i, j), cell in t.get_celld().items():
            if i == 0:
                cell.set_text_props(weight="bold"); cell.set_facecolor("#e8e8e8")
            elif j == 1 and cells[i - 1][1] != "ok":
                cell.set_facecolor("#fde2c8" if cells[i - 1][1].startswith("skipped") else "#f5b7b1")
        notes = [f"{r['name']}: {r['description']}" + (f"  [{r['status']}]" if r["status"] != "ok" else "") for r in rows]
        fig.text(0.04, 0.29, "Experiments", weight="bold", fontsize=9)
        fig.text(0.04, 0.03, "\n".join(notes), fontsize=7.5, va="bottom", family="monospace")
        pdf.savefig(fig); plt.close(fig)
        if done:
            fig, ax = plt.subplots(figsize=(11.7, 6))
            names = [r["name"] for r in done]; f1 = [r.get("f1") or 0 for r in done]; fpv = [r.get("fp") or 0 for r in done]
            ax.barh(names, f1, color="#2c7fb8"); ax.set_xlim(0, 1); ax.set_xlabel("lesion F1 (pooled over the test cases)")
            for i, (v, p) in enumerate(zip(f1, fpv)):
                ax.text(min(v + 0.01, 0.9), i, f"F1 {v:.2f}, {p} FP", va="center", fontsize=8)
            ax.invert_yaxis(); ax.set_title("Lesion F1 per experiment (few epochs: shows that it runs, not how good it is)")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    for c in TEST:
        ref_img = nib.load(os.path.join(pre, c, f"{c}_cmb.nii.gz")); ref = ref_img.get_fdata() > 0
        scan = nib.load(glob.glob(os.path.join(pre, c, f"{c}_t2star.nii.gz"))[0]).get_fdata()
        lab, nref = ndimage.label(ref, np.ones((3, 3, 3)))
        z = int(np.argmax(ref.sum(axis=(0, 1)))) if nref else scan.shape[2] // 2
        here = [r for r in done if c in r["cases"]]
        with PdfPages(os.path.join(rd, f"{c}.pdf")) as pdf:
            fig = plt.figure(figsize=(11.7, 8.3))
            fig.text(0.04, 0.95, f"{c}: {nref} reference microbleed(s)", fontsize=15, weight="bold")
            cells = [[r["name"], r["cases"][c]["tp"], r["cases"][c]["fp"], r["cases"][c]["fn"],
                      *map(fmt, metrics(r["cases"][c]["tp"], r["cases"][c]["fp"], r["cases"][c]["fn"]))] for r in here]
            ax = fig.add_axes([0.04, 0.35, 0.92, 0.56]); ax.axis("off")
            if cells:
                t = ax.table(cellText=cells, colLabels=["experiment", "TP", "FP", "FN", "sensitivity", "precision", "lesion F1"],
                             loc="upper left", cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
            fig.text(0.04, 0.05, "TP = reference microbleeds touched by a prediction; FP = predicted spots touching none; FN = reference "
                     "microbleeds missed.\nA case without microbleeds has no sensitivity or F1 (n/a); its FP count is the number of false "
                     "alarms.", fontsize=8)
            pdf.savefig(fig); plt.close(fig)
            per_page = 4
            for k in range(0, len(here), per_page):
                fig, axs = plt.subplots(1, 1 + min(per_page, len(here) - k), figsize=(11.7, 4.2), squeeze=False)
                axs = axs[0]; sl = np.rot90(scan[:, :, z])
                def show(ax, title, mask_ref=None, pred=None):
                    ax.imshow(sl, cmap="gray"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
                    def ring(m2, colour):
                        l2, n2 = ndimage.label(m2)
                        for cy, cx in ndimage.center_of_mass(m2, l2, range(1, n2 + 1)):
                            ax.add_patch(plt.Circle((cx, cy), 8, fill=False, color=colour, lw=1.4))
                    if pred is None:
                        ring(np.rot90(ref[:, :, z]), "#1f77b4"); return
                    pl, _ = ndimage.label(pred, np.ones((3, 3, 3)))
                    found = np.isin(lab, np.unique(lab[pred & ref])) & ref
                    fp = pred & ~np.isin(pl, np.unique(pl[pred & ref]))
                    ring(np.rot90(found[:, :, z]), "#2ca02c"); ring(np.rot90(fp[:, :, z]), "#ff7f0e"); ring(np.rot90((ref & ~found)[:, :, z]), "#1f77b4")
                show(axs[0], f"reference, slice {z + 1}")
                for ax, r in zip(axs[1:], here[k:k + per_page]):
                    pred = nib.load(r["cases"][c]["mask"]).get_fdata() > 0
                    show(ax, r["name"], pred=pred)
                fig.suptitle("green = found, orange = false alarm, blue = missed (this slice)", fontsize=9)
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    with open(os.path.join(rd, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["experiment", "status", "case", "tp", "fp", "fn", "sensitivity", "precision", "f1", "seconds"])
        for r in rows:
            for c, v in (r.get("cases") or {"": {}}).items():
                s = metrics(v["tp"], v["fp"], v["fn"]) if v else (None, None, None)
                w.writerow([r["name"], r["status"], c, v.get("tp", ""), v.get("fp", ""), v.get("fn", ""), *map(fmt, s), round(r["seconds"])])
    return rd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(REPO), "my_runs", "test_all_cases"))
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu [auto]")
    ap.add_argument("--epochs", type=int, default=4, help="epochs per training [4]")
    ap.add_argument("--patches", type=int, default=64, help="patches per epoch [64]")
    ap.add_argument("--only", nargs="*", default=None, help="run only these experiments (see --list)")
    ap.add_argument("--list", action="store_true", help="list the experiments and stop")
    a = ap.parse_args(); a.out = os.path.abspath(a.out)
    pre = os.path.join(a.out, "derivatives", "preprocessed"); train_dir = os.path.join(a.out, "split", "train")
    E = experiments(a, pre, train_dir)
    if a.list:
        for k, (d, _, req) in E.items():
            print(f"{k:24s} {d}" + (f"  (needs {req})" if req else ""))
        return 0
    os.makedirs(a.out, exist_ok=True); log = os.path.join(a.out, "test_all_cases.log")
    print(f"log: {log}")
    # 1. preprocess
    if not os.path.isdir(pre) or len(os.listdir(pre)) < 5:
        ok, s = cmbx("preprocess", "-i", os.path.join(DATA, "raw"), "-ir", "_T2starw", "-f", "True",
                     "-l", os.path.join(DATA, "raw"), "-lr", "_cmb.nii.gz", "-o", pre, log=log)
        print(f"[preprocess] {'ok' if ok else 'FAILED'} ({s:.0f} s)")
        if not ok:
            raise SystemExit(f"preprocessing failed, see {log}")
    for f in glob.glob(os.path.join(DATA, "synthseg", "*_synthseg.nii.gz")):
        c = os.path.basename(f).split("_synthseg")[0]
        if os.path.isdir(os.path.join(pre, c)):
            shutil.copy(f, os.path.join(pre, c, os.path.basename(f)))
    # 2. split: copies of the case folders (small), so every command sees a plain folder
    for name, cases in (("train", TRAIN), ("test", TEST)):
        d = os.path.join(a.out, "split", name); os.makedirs(d, exist_ok=True)
        for c in cases:
            if not os.path.isdir(os.path.join(d, c)):
                shutil.copytree(os.path.join(pre, c), os.path.join(d, c))
    test_dir = os.path.join(a.out, "split", "test")
    # 3.-4. experiments
    rows, models_ok = [], {}
    for name, (desc, steps, req) in E.items():
        if a.only and name not in a.only:
            continue
        row = dict(name=name, description=desc, status="ok", seconds=0.0)
        if req == "microbleednet" and not have("microbleednet"):
            row["status"] = "skipped: needs microbleednet (pip install it to run vessel inpainting)"
        elif req and req in E and not models_ok.get(req):
            row["status"] = f"skipped: needs experiment {req}"
        else:
            for kind, args in steps:
                if kind == "evaluate":
                    args = ["-i", test_dir, "-o", os.path.join(a.out, "predictions", name), *args]
                ok, s = cmbx(kind, *args, log=log); row["seconds"] += s
                if not ok:
                    row["status"] = f"FAILED in {kind} (see log)"; break
            if row["status"] == "ok":
                models_ok[name] = True
                if any(k == "evaluate" for k, _ in steps):
                    row["cases"] = score(os.path.join(a.out, "predictions", name), test_dir)
                    tp, fp, fn = (sum(v[k] for v in row["cases"].values()) for k in ("tp", "fp", "fn"))
                    row.update(tp=tp, fp=fp, fn=fn); row["sens"], row["prec"], row["f1"] = metrics(tp, fp, fn)
                elif kind == "cross_validate":
                    sj = os.path.join(a.out, "cv", "summary.json")
                    if os.path.exists(sj):
                        sm = json.load(open(sj)); row["f1"] = sm.get("f1") or sm.get("lesion_f1")
        rows.append(row)
        print(f"[{name}] {row['status']}" + (f"  TP {row.get('tp')} FP {row.get('fp')} FN {row.get('fn')} F1 {fmt(row.get('f1'))}" if row.get("cases") else "")
              + f"  ({row['seconds'] / 60:.1f} min)", flush=True)
    # 5. report
    rd = report(rows, a, test_dir)
    print(f"report: {rd}/summary.pdf and one PDF per test case ({', '.join(TEST)}); table: {rd}/results.csv")
    failed = [r["name"] for r in rows if r["status"].startswith("FAILED")]
    print("ALL EXPERIMENTS RAN" if not failed else f"FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
