"""
test_all_cases.py -- run every experiment of this repository on five real example scans and write a report (PNG images + CSV).

HOW TO RUN

    python test_all_cases.py                          # all experiments (uses the GPU if there is one)
    python test_all_cases.py --device cpu             # without a GPU (slower)
    python test_all_cases.py --epochs 2               # quicker test
    python test_all_cases.py --only pretrained finetune   # only some experiments
    python test_all_cases.py --list                   # show all experiment names

WHAT IT DOES

    1. Preprocess   the five VALDO 2021 T2* scans in examples/valdo/raw (four with microbleeds, one without)
    2. Split        three cases for training, two for testing (see TRAIN_CASES and TEST_CASES below)
    3. Experiments  each experiment is a short list of `cmbx` commands -- the same commands you would type yourself
    4. Score        every experiment on the same test cases: TP, FP, FN, sensitivity, precision, lesion F1
    5. Report       <out>/report/summary_1.png (table), summary_2.png (F1 chart), <case>_1.png ... per test case,
                    and results.csv

    test.py is the QUICK check that every function works. This script runs whole experiments, start to end.

IMPORTANT

    With 4 epochs on 3 training cases the numbers are NOT meaningful. They only show that every path runs.
    The pretrained model was trained on all 72 VALDO scans, these five included.

HOW TO CHANGE IT

    * Other example cases:  put your scans in examples/valdo/raw (<case>_T2starw.nii.gz and <case>_cmb.nii.gz)
                            and change TRAIN_CASES / TEST_CASES.
    * A new experiment:     add one entry in make_experiments() -- copy an existing one and change the commands.
    * Longer training:      python test_all_cases.py --epochs 50 --patches 500

The example data are VALDO 2021 scans under CC BY-NC-SA 4.0, see examples/valdo/README.md.
"""
import argparse
import csv
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time


# ======================================================================================================================
# SETTINGS -- change these if you use other cases
# ======================================================================================================================

REPO_FOLDER = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DATA = os.path.join(REPO_FOLDER, "examples", "valdo")

TRAIN_CASES = ["sub-106", "sub-305", "sub-320"]    # used to train / fine-tune
TEST_CASES = ["sub-103", "sub-104"]                 # never seen in training; sub-103 has 6 microbleeds, sub-104 none

DEFAULT_OUTPUT = os.path.join(os.path.dirname(REPO_FOLDER), "my_runs", "test_all_cases")


# ======================================================================================================================
# THE EXPERIMENTS
# ======================================================================================================================

class Experiment:
    """One experiment = a name, a sentence that explains it, and the cmbx commands that run it.

    commands: a list of cmbx commands, each written as a list of words, e.g. ["train", "-i", folder, "-m", model].
              "evaluate" commands get the test folder and an output folder added automatically.
    needs:    None, the name of an experiment that must have run first (it reuses that model),
              or "microbleednet" (a Python package that must be installed).
    """

    def __init__(self, name, description, commands, needs=None):
        self.name = name
        self.description = description
        self.commands = commands
        self.needs = needs


def make_experiments(settings, train_folder):
    """All experiments, in the order they run. To add one: copy an entry and change it."""

    # words that every training command gets: epochs, patches per epoch, device
    training = ["-ep", settings.epochs, "--patches", settings.patches, "--device", settings.device]
    # words that every prediction command gets
    predicting = ["--device", settings.device]

    def model_folder(name):
        return os.path.join(settings.out, "models", name)

    return [
        Experiment(
            "pretrained",
            "pretrained VALDO model, as downloaded",
            [["evaluate", "-m", "pre", *predicting]]),

        Experiment(
            "pretrained_csf",
            "pretrained + CSF rule (drop detections in CSF, SynthSeg)",
            [["evaluate", "-m", "pre", "--csf_rule", "True", *predicting]]),

        Experiment(
            "scratch",
            "trained from scratch, image + FRST",
            [["train", "-i", train_folder, "-m", model_folder("scratch"), "--frst_from", "cmbx", *training],
             ["evaluate", "-m", model_folder("scratch"), *predicting]]),

        Experiment(
            "scratch_no_frst",
            "trained from scratch, image only (no FRST)",
            [["train", "-i", train_folder, "-m", model_folder("no_frst"), "--frst", "False", *training],
             ["evaluate", "-m", model_folder("no_frst"), *predicting]]),

        Experiment(
            "scratch_csf",
            "trained from scratch + CSF rule",
            [["evaluate", "-m", model_folder("scratch"), "--csf_rule", "True", *predicting]],
            needs="scratch"),

        Experiment(
            "finetune",
            "fine-tuned from the pretrained model",
            [["fine_tune", "-i", train_folder, "-m", "pre", "-o", model_folder("finetune"), *training],
             ["evaluate", "-m", model_folder("finetune"), *predicting]]),

        Experiment(
            "finetune_csf",
            "fine-tuned + CSF rule",
            [["evaluate", "-m", model_folder("finetune"), "--csf_rule", "True", *predicting]],
            needs="finetune"),

        # Step 2 learns from the candidates of the step-1 model on the training cases, so step 1 must already find
        # some (real ones and false ones). A model trained from scratch for a few epochs finds none; therefore
        # step 2 starts here from the fine-tuned pretrained model.
        Experiment(
            "step2_teacher_student",
            "fine-tuned + step-2 classifier, MicrobleedNet teacher-student",
            [["fine_tune", "-i", train_folder, "-m", "pre", "-o", model_folder("step2_ts"), "-cand_disc", "True", *training],
             ["evaluate", "-m", model_folder("step2_ts"), "-cand_disc", "True", *predicting]]),

        Experiment(
            "step2_cnn",
            "fine-tuned + step-2 classifier, small CNN",
            [["fine_tune", "-i", train_folder, "-m", "pre", "-o", model_folder("step2_cnn"), "-cand_disc", "True",
              "--disc_model", "cnn", *training],
             ["evaluate", "-m", model_folder("step2_cnn"), "-cand_disc", "True", *predicting]]),

        Experiment(
            "inpaint",
            "scratch on vessel-inpainted images (MicrobleedNet inpaint_vessels)",
            [["train", "-i", train_folder, "-m", model_folder("inpaint"), "--inpaint", "True", "--frst_from", "cmbx", *training],
             ["evaluate", "-m", model_folder("inpaint"), *predicting]],
            needs="microbleednet"),

        Experiment(
            "cross_validate",
            "3-fold cross-validation on the 3 training cases (out-of-fold scores)",
            [["cross_validate", "-i", train_folder, "-o", os.path.join(settings.out, "cv"), "-fold", 3,
              "--frst_from", "cmbx", *training]]),
    ]


# ======================================================================================================================
# RUNNING cmbx COMMANDS
# ======================================================================================================================

def run_cmbx(command, log_file):
    """Run one cmbx command (a list of words). All its output goes into the log file.
    Returns (True if it worked, how many seconds it took)."""
    full_command = [sys.executable, "-m", "cmbx"] + [str(word) for word in command]
    start = time.time()
    with open(log_file, "a") as log:
        log.write("\n$ " + " ".join(full_command) + "\n")
        log.flush()
        result = subprocess.run(full_command, cwd=REPO_FOLDER, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode == 0, time.time() - start


def package_installed(name):
    return importlib.util.find_spec(name) is not None


# ======================================================================================================================
# SCORES
# ======================================================================================================================

def count_hits(prediction_folder, case_folder):
    """For each test case: TP, FP, FN of the predicted mask against the reference mask.
    A reference microbleed counts as found (TP) when a predicted spot touches it (cmbx.metrics.hits)."""
    import nibabel as nib
    from cmbx.metrics import hits

    counts = {}
    for case in TEST_CASES:
        found = glob.glob(os.path.join(prediction_folder, "final_predictions", f"predicted_final_cmbx_{case}*.nii*"))
        if not found:
            continue
        predicted_mask_file = found[0]
        reference = nib.load(os.path.join(case_folder, case, f"{case}_cmb.nii.gz")).get_fdata()
        predicted = nib.load(predicted_mask_file).get_fdata()
        h = hits(reference, predicted)
        counts[case] = {"tp": h["tp"], "fp": h["fp"], "fn": h["fn"], "mask": predicted_mask_file}
    return counts


def sensitivity_precision_f1(tp, fp, fn):
    """The three scores; None where they are not defined. A case without microbleeds (tp + fn = 0) has no
    sensitivity and no F1 -- for such a case only the number of false alarms (fp) counts."""
    sensitivity = tp / (tp + fn) if tp + fn > 0 else None
    precision = tp / (tp + fp) if tp + fp > 0 else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fn > 0 else None
    return sensitivity, precision, f1


def as_text(number):
    return "n/a" if number is None else f"{number:.2f}"


# ======================================================================================================================
# THE REPORT (PNG images + CSV; images are always PNG, never PDF)
# ======================================================================================================================

class PngPages:
    """Collects report pages like a PDF would, but writes each page as its own PNG: <prefix>_1.png, <prefix>_2.png ..."""

    def __init__(self, prefix):
        self.prefix, self.pages = prefix, 0

    def __enter__(self):
        return self

    def __exit__(self, *error):
        return False

    def savefig(self, fig):
        self.pages += 1
        fig.savefig(f"{self.prefix}_{self.pages}.png", dpi=150, facecolor="white")


def write_summary_pages(results, settings, prefix):
    """Page 1: a table of all experiments. Page 2: a bar chart of lesion F1."""
    import matplotlib.pyplot as plt

    with PngPages(prefix) as pages:
        # ---- page 1: table
        fig = plt.figure(figsize=(11.7, 8.3))      # A4 landscape
        fig.text(0.04, 0.95, "test_all_cases.py -- all experiments on the VALDO examples", fontsize=15, weight="bold")
        fig.text(0.04, 0.91, f"{time.strftime('%Y-%m-%d %H:%M')} | epochs {settings.epochs}, patches {settings.patches}, "
                             f"device {settings.device} | train {', '.join(TRAIN_CASES)} | test {', '.join(TEST_CASES)} | "
                             f"scores pooled over the test cases", fontsize=9)
        header = ["experiment", "status", "TP", "FP", "FN", "sensitivity", "precision", "lesion F1", "time"]
        rows = []
        for r in results:
            short_status = r["status"].split(" ")[0].rstrip(":")        # "ok", "skipped" or "FAILED"; reason below
            rows.append([r["name"], short_status, r.get("tp", ""), r.get("fp", ""), r.get("fn", ""),
                         as_text(r.get("sensitivity")), as_text(r.get("precision")), as_text(r.get("f1")),
                         f"{r['seconds'] / 60:.1f} min"])
        ax = fig.add_axes([0.04, 0.32, 0.92, 0.56])
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=header, loc="upper left", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.45)
        for (row, column), cell in table.get_celld().items():
            if row == 0:                                          # header row
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#e8e8e8")
            elif column == 1:                                     # status column: colour what did not run
                status = rows[row - 1][1]
                if status == "skipped":
                    cell.set_facecolor("#fde2c8")                 # light orange
                elif status != "ok":
                    cell.set_facecolor("#f5b7b1")                 # light red
        explanations = []
        for r in results:
            line = f"{r['name']}: {r['description']}"
            if r["status"] != "ok":
                line += f"  [{r['status']}]"
            explanations.append(line)
        fig.text(0.04, 0.29, "Experiments", weight="bold", fontsize=9)
        fig.text(0.04, 0.03, "\n".join(explanations), fontsize=7.5, va="bottom", family="monospace")
        pages.savefig(fig)
        plt.close(fig)

        # ---- page 2: bar chart of lesion F1
        scored = [r for r in results if r["status"] == "ok" and r.get("cases")]
        if scored:
            fig, ax = plt.subplots(figsize=(11.7, 6))
            names = [r["name"] for r in scored]
            f1_values = [r.get("f1") or 0 for r in scored]
            false_positives = [r.get("fp") or 0 for r in scored]
            ax.barh(names, f1_values, color="#2c7fb8")
            ax.set_xlim(0, 1.25)                              # room for the labels right of the bars
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xlabel("lesion F1 (pooled over the test cases)")
            for i, (f1, fp) in enumerate(zip(f1_values, false_positives)):
                ax.text(f1 + 0.02, i, f"F1 {f1:.2f}, {fp} FP", va="center", fontsize=8)   # always next to the bar
            ax.invert_yaxis()
            ax.set_title("Lesion F1 per experiment (few epochs: shows that it runs, not how good it is)")
            fig.tight_layout()
            pages.savefig(fig)
            plt.close(fig)


def write_case_pages(case, results, case_folder, prefix):
    """One test case: a table of all experiments, then the slice with the most microbleeds,
    with rings: green = found, orange = false alarm, blue = missed."""
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    from scipy import ndimage

    touching = np.ones((3, 3, 3))                      # 26-neighbourhood: voxels that touch belong to one spot
    reference = nib.load(os.path.join(case_folder, case, f"{case}_cmb.nii.gz")).get_fdata() > 0
    scan = nib.load(os.path.join(case_folder, case, f"{case}_t2star.nii.gz")).get_fdata()
    reference_spots, n_reference = ndimage.label(reference, touching)
    if n_reference > 0:
        z = int(np.argmax(reference.sum(axis=(0, 1))))     # the axial slice with the most microbleed voxels
    else:
        z = scan.shape[2] // 2                             # no microbleeds: the middle slice
    experiments_with_this_case = [r for r in results if r["status"] == "ok" and case in r.get("cases", {})]

    def draw_rings(ax, mask_2d, colour):
        spots, n = ndimage.label(mask_2d)
        for centre_y, centre_x in ndimage.center_of_mass(mask_2d, spots, range(1, n + 1)):
            ax.add_patch(plt.Circle((centre_x, centre_y), 8, fill=False, color=colour, lw=1.4))

    def show_slice(ax, title, predicted=None):
        ax.imshow(np.rot90(scan[:, :, z]), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        if predicted is None:                              # the reference alone
            draw_rings(ax, np.rot90(reference[:, :, z]), "#1f77b4")
            return
        predicted_spots, _ = ndimage.label(predicted, touching)
        found = np.isin(reference_spots, np.unique(reference_spots[predicted & reference])) & reference
        false_alarm = predicted & ~np.isin(predicted_spots, np.unique(predicted_spots[predicted & reference]))
        missed = reference & ~found
        draw_rings(ax, np.rot90(found[:, :, z]), "#2ca02c")          # green
        draw_rings(ax, np.rot90(false_alarm[:, :, z]), "#ff7f0e")    # orange
        draw_rings(ax, np.rot90(missed[:, :, z]), "#1f77b4")         # blue

    with PngPages(prefix) as pages:
        # ---- page 1: table
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(0.04, 0.95, f"{case}: {n_reference} reference microbleed(s)", fontsize=15, weight="bold")
        rows = []
        for r in experiments_with_this_case:
            c = r["cases"][case]
            rows.append([r["name"], c["tp"], c["fp"], c["fn"],
                         *[as_text(x) for x in sensitivity_precision_f1(c["tp"], c["fp"], c["fn"])]])
        ax = fig.add_axes([0.04, 0.35, 0.92, 0.56])
        ax.axis("off")
        if rows:
            table = ax.table(cellText=rows, colLabels=["experiment", "TP", "FP", "FN", "sensitivity", "precision", "lesion F1"],
                             loc="upper left", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
        fig.text(0.04, 0.05, "TP = reference microbleeds touched by a prediction; FP = predicted spots touching none; "
                             "FN = reference microbleeds missed.\nA case without microbleeds has no sensitivity or F1 (n/a); "
                             "its FP count is the number of false alarms.", fontsize=8)
        pages.savefig(fig)
        plt.close(fig)

        # ---- next pages: images, four experiments per page next to the reference
        per_page = 4
        for first in range(0, len(experiments_with_this_case), per_page):
            on_this_page = experiments_with_this_case[first:first + per_page]
            fig, axes = plt.subplots(1, 1 + len(on_this_page), figsize=(11.7, 4.2), squeeze=False)
            axes = axes[0]
            show_slice(axes[0], f"reference, slice {z + 1}")
            for ax, r in zip(axes[1:], on_this_page):
                predicted = nib.load(r["cases"][case]["mask"]).get_fdata() > 0
                show_slice(ax, r["name"], predicted)
            fig.suptitle("green = found, orange = false alarm, blue = missed (this slice)", fontsize=9)
            fig.tight_layout()
            pages.savefig(fig)
            plt.close(fig)


def write_csv(results, csv_file):
    """One row per experiment and test case."""
    with open(csv_file, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["experiment", "status", "case", "tp", "fp", "fn", "sensitivity", "precision", "f1", "seconds"])
        for r in results:
            cases = r.get("cases") or {"": None}             # experiments without scores still get one row
            for case, c in cases.items():
                if c:
                    scores = sensitivity_precision_f1(c["tp"], c["fp"], c["fn"])
                    writer.writerow([r["name"], r["status"], case, c["tp"], c["fp"], c["fn"],
                                     *[as_text(x) for x in scores], round(r["seconds"])])
                else:
                    writer.writerow([r["name"], r["status"], case, "", "", "", "n/a", "n/a", "n/a", round(r["seconds"])])


def write_report(results, settings, case_folder):
    import matplotlib
    matplotlib.use("Agg")                                   # draw into files, no window needed
    report_folder = os.path.join(settings.out, "report")
    os.makedirs(report_folder, exist_ok=True)
    write_summary_pages(results, settings, os.path.join(report_folder, "summary"))
    for case in TEST_CASES:
        write_case_pages(case, results, case_folder, os.path.join(report_folder, case))
    write_csv(results, os.path.join(report_folder, "results.csv"))
    return report_folder


# ======================================================================================================================
# MAIN: the five steps
# ======================================================================================================================

def read_settings():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="output folder [../my_runs/test_all_cases]")
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu [auto]")
    parser.add_argument("--epochs", type=int, default=4, help="epochs per training [4]")
    parser.add_argument("--patches", type=int, default=64, help="patches per epoch [64]")
    parser.add_argument("--only", nargs="*", default=None, help="run only these experiments (names: --list)")
    parser.add_argument("--list", action="store_true", help="list the experiments and stop")
    settings = parser.parse_args()
    settings.out = os.path.abspath(settings.out)
    return settings


def step1_preprocess(preprocessed_folder, log_file):
    """Bring the raw example scans into the cmbx layout (one folder per case). Skipped if already done."""
    already_done = os.path.isdir(preprocessed_folder) and len(os.listdir(preprocessed_folder)) >= 5
    if not already_done:
        raw = os.path.join(EXAMPLE_DATA, "raw")
        ok, seconds = run_cmbx(["preprocess",
                                "-i", raw, "-ir", "_T2starw",          # the scans: <case>_T2starw.nii.gz
                                "-f", "True",                          # VALDO scans are already brain-masked
                                "-l", raw, "-lr", "_cmb.nii.gz",       # the reference masks: <case>_cmb.nii.gz
                                "-o", preprocessed_folder], log_file)
        print(f"[preprocess] {'ok' if ok else 'FAILED'} ({seconds:.0f} s)")
        if not ok:
            raise SystemExit(f"preprocessing failed, see {log_file}")
    # the shipped SynthSeg labels go next to each case (needed by --csf_rule)
    for label_file in glob.glob(os.path.join(EXAMPLE_DATA, "synthseg", "*_synthseg.nii.gz")):
        case = os.path.basename(label_file).split("_synthseg")[0]
        if os.path.isdir(os.path.join(preprocessed_folder, case)):
            shutil.copy(label_file, os.path.join(preprocessed_folder, case, os.path.basename(label_file)))


def step2_split(preprocessed_folder, out):
    """Copy the case folders into split/train and split/test (small files), so every command sees a plain folder."""
    for part, cases in (("train", TRAIN_CASES), ("test", TEST_CASES)):
        part_folder = os.path.join(out, "split", part)
        os.makedirs(part_folder, exist_ok=True)
        for case in cases:
            if not os.path.isdir(os.path.join(part_folder, case)):
                shutil.copytree(os.path.join(preprocessed_folder, case), os.path.join(part_folder, case))
    return os.path.join(out, "split", "train"), os.path.join(out, "split", "test")


def step3_run_one(experiment, finished, settings, test_folder, log_file):
    """Run the commands of one experiment and score it. Returns one result (a dict)."""
    result = {"name": experiment.name, "description": experiment.description, "status": "ok", "seconds": 0.0}

    # is everything there that this experiment needs?
    if experiment.needs == "microbleednet" and not package_installed("microbleednet"):
        result["status"] = "skipped: needs microbleednet (pip install it to run vessel inpainting)"
        return result
    if experiment.needs and experiment.needs != "microbleednet" and experiment.needs not in finished:
        result["status"] = f"skipped: needs experiment {experiment.needs}"
        return result

    prediction_folder = os.path.join(settings.out, "predictions", experiment.name)
    for command in experiment.commands:
        if command[0] == "evaluate":                          # predict the test cases into this experiment's folder
            command = ["evaluate", "-i", test_folder, "-o", prediction_folder] + command[1:]
        ok, seconds = run_cmbx(command, log_file)
        result["seconds"] += seconds
        if not ok:
            result["status"] = f"FAILED in {command[0]} (see log)"
            return result
    finished.add(experiment.name)

    # step 4: scores
    kinds = [command[0] for command in experiment.commands]
    if "evaluate" in kinds:
        result["cases"] = count_hits(prediction_folder, test_folder)
        tp = sum(c["tp"] for c in result["cases"].values())
        fp = sum(c["fp"] for c in result["cases"].values())
        fn = sum(c["fn"] for c in result["cases"].values())
        result.update(tp=tp, fp=fp, fn=fn)
        result["sensitivity"], result["precision"], result["f1"] = sensitivity_precision_f1(tp, fp, fn)
    elif "cross_validate" in kinds:                          # the cross-validation writes its own summary
        summary_file = os.path.join(settings.out, "cv", "summary.json")
        if os.path.exists(summary_file):
            summary = json.load(open(summary_file))
            result["f1"] = summary.get("f1") or summary.get("lesion_f1")
    return result


def check_environment():
    """Stop at once, with a clear message, if the Python that runs this script cannot run cmbx (wrong environment)."""
    missing = [name for name in ("cmbx", "torch", "nibabel", "matplotlib") if not package_installed(name)]
    if missing:
        raise SystemExit(f"This Python ({sys.executable}) is missing: {', '.join(missing)}.\n"
                         "Activate the cmbx environment first:   source env.sh   (or: conda activate cmbx)\n"
                         "If it does not exist yet:              ./install.sh")


def main():
    settings = read_settings()
    if not settings.list:
        check_environment()
    preprocessed_folder = os.path.join(settings.out, "derivatives", "preprocessed")
    train_folder = os.path.join(settings.out, "split", "train")
    experiments = make_experiments(settings, train_folder)

    if settings.list:
        for e in experiments:
            needs = f"  (needs {e.needs})" if e.needs else ""
            print(f"{e.name:24s} {e.description}{needs}")
        return 0

    os.makedirs(settings.out, exist_ok=True)
    log_file = os.path.join(settings.out, "test_all_cases.log")
    print(f"log: {log_file}")

    step1_preprocess(preprocessed_folder, log_file)
    train_folder, test_folder = step2_split(preprocessed_folder, settings.out)

    results, finished = [], set()
    for experiment in experiments:
        if settings.only and experiment.name not in settings.only:
            continue
        result = step3_run_one(experiment, finished, settings, test_folder, log_file)
        results.append(result)
        line = f"[{experiment.name}] {result['status']}"
        if result.get("cases"):
            line += f"  TP {result['tp']} FP {result['fp']} FN {result['fn']} F1 {as_text(result.get('f1'))}"
        print(line + f"  ({result['seconds'] / 60:.1f} min)", flush=True)

    report_folder = write_report(results, settings, test_folder)
    print(f"report: {report_folder}/summary_1.png and _2.png, <case>_N.png per test case ({', '.join(TEST_CASES)}), {report_folder}/results.csv")

    failed = [r["name"] for r in results if r["status"].startswith("FAILED")]
    print("ALL EXPERIMENTS RAN" if not failed else f"FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
