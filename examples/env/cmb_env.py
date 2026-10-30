"""Shared settings of run_train.py, run_fine_tune.py and run_predict.py: the .env file, and which cases each step uses.

Settings come from DEFAULTS, then the .env file (KEY=VALUE, '#' comments, ${HERE}, ${REPO} and ${OTHER_KEY} references),
then --set KEY=VALUE on the command line. ${HERE} is this folder, ${REPO} the repository root.

Data: CMB_DATA is ONE cohort folder as in the README section "Data" (<CMB_DATA>/derivatives/preprocessed/<case>/...).
Cases and folds: <CMB_DATA>/participants.tsv if it has the columns ID_COLUMN, COUNT_COLUMN and FOLD_COLUMN. A missing
count is computed from the case's <case>_cmb.nii.gz (connected components, 26-connectivity); missing folds are drawn once,
stratified by that count (N_FOLDS, FOLD_SEED), and kept in <CMB_RUNS>/<NAME>/_input/folds.tsv so every later run uses
the same folds.
  TEST_FOLD empty  -> train / fine-tune on all cases; predict all cases
  TEST_FOLD = k    -> train / fine-tune on the cases NOT in fold k; predict only fold k
  N_SCANS = n      -> of the training cases take n, stratified by the number of microbleeds (DRAW_SEED fixes the draw);
                      with the same seed a larger draw contains the smaller one (nested)
The chosen cases are linked (symlinks) into <CMB_RUNS>/<NAME>/_input/<tag>/, so the example scripts see one folder.
"""
import argparse
import csv
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULTS = {
    "CMB_DATA": "",                                        # REQUIRED: the cohort folder (with derivatives/preprocessed)
    "NAME": "",                                            # name of the cohort in output paths (empty: folder name)
    "CMB_RUNS": "${REPO}/../my_runs",                      # where models and predictions go
    "ID_COLUMN": "participant_id", "COUNT_COLUMN": "n_cmb", "FOLD_COLUMN": "fold",
    "N_FOLDS": "5", "FOLD_SEED": "0",                      # only used when participants.tsv has no fold column
    "TEST_FOLD": "",                                       # empty | 0 .. N_FOLDS-1
    "N_SCANS": "",                                         # empty = all training cases | a number
    "DRAW_SEED": "0",
    "EPOCHS": "",                                          # empty = 60
    "LR": "",                                              # empty = 0.0003 (train) / 0.0001 (fine-tune)
    "LEVELS": "",                                          # fine-tune only: levels to train, e.g. "1 2" (empty = all)
    "CONFIG": "",                                          # optional cmbx YAML with further settings (docs/DETAILS.md)
    "START_MODEL": "pretrained",                           # fine-tune: pretrained (VALDO T2*) or a model folder
    "MODEL": "pretrained",                                 # predict: pretrained or a model folder
    "OUT_DIR": "",                                         # empty = a name built from the settings
    "DEVICE": "auto",                                      # auto | cuda | cuda:N | cpu
    "THRESHOLD": "",                                       # predict: empty = chosen in training
    "MIN_SIZE": "",
    "WORKERS": "4",
}


def read_env(path):
    """KEY=VALUE lines of a .env file; '#' starts a comment, quotes around the value are removed."""
    values = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("'\"")
    return values


def resolve(settings):
    """Fill ${HERE}, ${REPO} and ${OTHER_KEY} references (a few rounds, so keys may refer to each other); expand ~."""
    out = dict(settings)
    for _ in range(5):
        for k, v in out.items():
            for name, sub in [("HERE", HERE), ("REPO", REPO)] + list(out.items()):
                v = v.replace("${" + name + "}", sub)
            out[k] = os.path.expanduser(v)
    return out


def settings(argv=None, description=""):
    """Parse --env, --set and --force; returns (settings, force). Stops if CMB_DATA is not a cohort folder."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--env", default=os.path.join(HERE, ".env"), help="settings file [examples/env/.env]")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override settings of the .env file")
    p.add_argument("--force", default="false", help="true: compute again although the output exists [false]")
    a = p.parse_args(argv)
    s = dict(DEFAULTS)
    s.update(read_env(a.env))
    s.update(dict(kv.split("=", 1) for kv in a.set))
    s = resolve(s)
    if not s["CMB_DATA"] or not os.path.isdir(os.path.join(s["CMB_DATA"], "derivatives", "preprocessed")):
        raise SystemExit(f"CMB_DATA must be a cohort folder with derivatives/preprocessed/ (README 'Data'), got "
                         f"'{s['CMB_DATA']}': set it in {a.env} or with --set CMB_DATA=<folder>")
    s["NAME"] = s["NAME"] or os.path.basename(os.path.normpath(s["CMB_DATA"]))
    print(f"settings from {a.env if os.path.exists(a.env) else 'defaults (no .env file)'}", flush=True)
    return s, a.force == "true"


def count_microbleeds(mask_path):
    """Number of microbleeds (26-connected components) in a mask file."""
    import nibabel as nib
    import numpy as np
    from scipy import ndimage
    return int(ndimage.label(np.asarray(nib.load(mask_path).dataobj) > 0, structure=np.ones((3, 3, 3), bool))[1])


def stratum(n):
    """Microbleed-count class used for stratified draws: 0, 1, 2-4, >= 5."""
    return 0 if n == 0 else 1 if n == 1 else 2 if n <= 4 else 3


def draw_stratified(rows, n, seed):
    """n of the rows, stratified by microbleed count: a seeded order inside each class, then the classes are taken in
    turn. A larger n with the same seed always contains the smaller draw (nested)."""
    classes = {}
    for r in rows:
        classes.setdefault(stratum(r["n_cmb"]), []).append(r)
    rng = random.Random(seed)
    for c in classes.values():
        c.sort(key=lambda r: r["id"]); rng.shuffle(c)
    order = []
    while any(classes.values()):
        for c in sorted(classes):
            if classes[c]:
                order.append(classes[c].pop(0))
    return order[:n]


def cases(S):
    """Rows (id, n_cmb, fold) of every case folder in <CMB_DATA>/derivatives/preprocessed, from participants.tsv where it
    has the columns; missing counts from the masks, missing folds drawn once and kept in _input/folds.tsv."""
    prep = os.path.join(S["CMB_DATA"], "derivatives", "preprocessed")
    ids = sorted(d for d in os.listdir(prep) if os.path.isdir(os.path.join(prep, d)))
    table = {}
    tsv = os.path.join(S["CMB_DATA"], "participants.tsv")
    if os.path.exists(tsv):
        table = {r[S["ID_COLUMN"]]: r for r in csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t")
                 if S["ID_COLUMN"] in r}
    rows = []
    for i in ids:
        r = table.get(i, {})
        mask = os.path.join(prep, i, f"{i}_cmb.nii.gz")
        n = r.get(S["COUNT_COLUMN"])
        n = int(n) if n not in (None, "") else (count_microbleeds(mask) if os.path.exists(mask) else 0)
        f = r.get(S["FOLD_COLUMN"])
        rows.append(dict(id=i, n_cmb=n, fold=int(f) if f not in (None, "") else None))
    if any(r["fold"] is None for r in rows):
        kept = os.path.join(S["CMB_RUNS"], S["NAME"], "_input", "folds.tsv")
        if os.path.exists(kept):
            old = {r["id"]: int(r["fold"]) for r in csv.DictReader(open(kept), delimiter="\t")}
            for r in rows:
                r["fold"] = old.get(r["id"], r["fold"])
        if any(r["fold"] is None for r in rows):
            order = draw_stratified([dict(r) for r in rows], len(rows), int(S["FOLD_SEED"]))
            fold_of = {r["id"]: k % int(S["N_FOLDS"]) for k, r in enumerate(order)}
            for r in rows:
                r["fold"] = fold_of[r["id"]]
            os.makedirs(os.path.dirname(kept), exist_ok=True)
            with open(kept, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["id", "n_cmb", "fold"], delimiter="\t"); w.writeheader(); w.writerows(rows)
            print(f"folds drawn (stratified, seed {S['FOLD_SEED']}) and kept in {kept}")
    return rows


def cases_for(S, step):
    """(case ids, tag) for step 'train' or 'predict' under TEST_FOLD and N_SCANS."""
    rows, fold = cases(S), S["TEST_FOLD"]
    if step == "predict":
        return [r["id"] for r in rows if fold == "" or r["fold"] == int(fold)], ("all" if fold == "" else f"fold{fold}")
    pool = [r for r in rows if fold == "" or r["fold"] != int(fold)]
    tag = "all" if fold == "" else f"not_fold{fold}"
    if S["N_SCANS"]:
        n = int(S["N_SCANS"])
        if n > len(pool):
            raise SystemExit(f"N_SCANS={n} but only {len(pool)} training cases")
        pool = draw_stratified(pool, n, int(S["DRAW_SEED"]))
        tag += f"_n{n}_seed{S['DRAW_SEED']}"
    return [r["id"] for r in pool], tag


def linked_input(S, ids, tag):
    """A folder with one symlink per chosen case into <CMB_DATA>/derivatives/preprocessed; returns its path."""
    src = os.path.join(S["CMB_DATA"], "derivatives", "preprocessed")
    base = os.path.join(S["CMB_RUNS"], S["NAME"], "_input")
    dst = os.path.join(base, tag)
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(dst):                          # remove links of an earlier, different selection
        if name not in ids and os.path.islink(os.path.join(dst, name)):
            os.remove(os.path.join(dst, name))
    for i in ids:
        link = os.path.join(dst, i)
        if not os.path.lexists(link):
            os.symlink(os.path.join(src, i), link)
    json.dump(dict(cases=ids, source=src), open(os.path.join(base, f"{tag}.json"), "w"), indent=1)
    return dst


def example(name):
    """Path of one of the numbered scripts in examples/python."""
    return os.path.join(REPO, "examples", "python", name)
