"""Command line: ``cmbx <command> ...`` (or ``python -m cmbx ...``).

  cmbx make-phantoms data/phantoms --n 10              synthetic cases + case list, to try everything without data
  cmbx make-caselist --image "<dir>/{id}/..." --label ... --out <csv>    case list from any layout (patterns)
  cmbx check-env                                       torch/CUDA etc.: what is there, what is missing
  cmbx prepare --config configs/baseline.yaml          native images -> 0.5 x 0.5 x 1 mm grid + FRST (+ tissue map)
  cmbx split --config configs/baseline.yaml            stratified fold assignment -> data/splits.json
  cmbx cv --config configs/baseline.yaml               5-fold CV (resumable; --folds 0 1 for a subset)
  cmbx run-experiment configs/no_frst.yaml             prepare + split + CV + evaluation + summary in one go
  cmbx evaluate runs/baseline --mode crossfold --csf-rule   re-score stored maps with other post-processing
  cmbx compare runs                                    one table of all runs -> runs/summary.csv
  cmbx predict --run runs/baseline --cases new.csv     fold-ensemble prediction for prepared new cases
Every command that takes --config also takes --set key.sub=value (e.g. --set train.epochs=2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cfg(args):
    from .config import load_config, parse_set
    return load_config(args.config, parse_set(args.set))


def cmd_make_phantoms(a):
    from .synthetic import write_phantom_dataset
    p = write_phantom_dataset(a.out, a.n, a.seed)
    _log(f"{a.n} phantom cases, case list {p}")


def cmd_make_caselist(a):
    from .caselist import build_caselist
    from .data import write_caselist
    rows = build_caselist(a.image, a.label, a.brain_mask, a.tissue_map, a.raw, a.ids, a.exclude, a.n_folds, a.fold_seed,
                          not a.no_folds, not a.no_quality_checks, os.path.dirname(os.path.abspath(a.out)), _log)
    if a.no_folds:
        for r in rows:
            r["fold"] = ""
    write_caselist(rows, a.out)
    _log(f"case list with {len(rows)} cases -> {a.out}")


def cmd_check_env(a):
    from .checkenv import check_env
    sys.exit(check_env())


def cmd_selftest(a):
    from .selftest import selftest
    sys.exit(selftest(a.data, a.device, a.keep, _log))


def cmd_prepare(a):
    from .experiments import ensure_prepared
    ensure_prepared(_cfg(a), a.workers, a.overwrite, _log)


def cmd_split(a):
    from .experiments import ensure_split
    s = ensure_split(_cfg(a), _log)
    _log(json.dumps({k: s[k] for k in ("n_folds", "cases_per_fold", "lesions_per_fold")}))


def cmd_cv(a):
    from .experiments import run_experiment
    cfg = _cfg(a)
    run_experiment(cfg, a.folds, a.workers, _log)


def cmd_run_experiment(a):
    from .config import load_config, parse_set
    from .experiments import run_experiment
    cfg = load_config(a.config_file, parse_set(a.set))
    run_experiment(cfg, a.folds, a.workers, _log, final=not a.no_final)


def cmd_train_final(a):
    from .config import load_config, parse_set
    from .experiments import check_config, ensure_prepared
    from .final import train_final
    cfg = load_config(a.config_file, parse_set(a.set))
    check_config(cfg)
    ensure_prepared(cfg, None, log=_log)
    train_final(cfg, os.path.join(cfg["output_root"], cfg["name"]), _log)


def cmd_evaluate(a):
    import yaml
    from .evaluate import evaluate_run
    cfg = yaml.safe_load(open(os.path.join(a.run, "config.yaml")))
    prepared = a.prepared or cfg["data"]["prepared_dir"]
    evaluate_run(a.run, prepared, a.mode, a.csf_rule, cfg["selection"], cfg["postprocess"]["csf_labels"], a.tag, _log)


def cmd_compare(a):
    from .evaluate import compare_runs
    compare_runs(a.root, log=_log)


def cmd_predict(a):
    from .predict_cli import predict_run
    predict_run(a.run, a.cases, a.out, a.threshold, a.min_size, a.device, a.csf_rule, _log, use_folds=a.folds,
                image=a.image, brain_mask=a.brain_mask, tissue_map=a.tissue_map, case_id=a.id, native=not a.grid_only)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cmbx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def with_config(p, required=True):
        p.add_argument("--config", required=required, help="experiment YAML (defaults = winner recipe)")
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config keys")

    p = sub.add_parser("make-phantoms", help="write synthetic test cases and a case list")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_make_phantoms)

    p = sub.add_parser("make-caselist", help="case list from any folder layout, given as path patterns with {id}",
                       description="Each pattern is a path with {id} and optional * wildcards; the first match is used.")
    p.add_argument("--image", required=True, help="bias-corrected image pattern, e.g. /data/{id}/anat/{id}*_T2starw.nii.gz")
    p.add_argument("--label", default=None, help="reference CMB mask pattern (needed for training, not for predict)")
    p.add_argument("--brain-mask", action="append", default=[], help="brain mask pattern; repeat for fallbacks (first match wins)")
    p.add_argument("--tissue-map", default=None, help="SynthSeg map pattern (for the CSF rule)")
    p.add_argument("--raw", default=None, help="raw image pattern for the quality checks (default: the --image)")
    p.add_argument("--ids", default=None, help="file with the case ids (one per line, or an id / participant_id column)")
    p.add_argument("--exclude", action="append", default=[], help="file with ids to leave out (same formats); repeatable")
    p.add_argument("--out", required=True, help="case list CSV; reports are written next to it (keep it OUTSIDE the repo)")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=0)
    p.add_argument("--no-quality-checks", action="store_true", help="skip the mask-fit / header-offset report")
    p.add_argument("--no-folds", action="store_true", help="leave folds empty (cmbx split computes them)")
    p.set_defaults(fn=cmd_make_caselist)

    p = sub.add_parser("check-env", help="check torch/CUDA and the Python packages")
    p.set_defaults(fn=cmd_check_env)

    p = sub.add_parser("selftest", help="one-minute end-to-end check on synthetic phantoms (GPU if visible)")
    p.add_argument("--data", default=None, help="folder with cases.csv (default: generate phantoms in a temp folder)")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--keep", action="store_true", help="keep the temporary outputs")
    p.set_defaults(fn=cmd_selftest)

    p = sub.add_parser("prepare", help="preprocess all cases of the case list")
    with_config(p)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("split", help="write the stratified fold split")
    with_config(p)
    p.set_defaults(fn=cmd_split)

    for name, fn in (("cv", cmd_cv),):
        p = sub.add_parser(name, help="run the cross-validation of a config")
        with_config(p)
        p.add_argument("--folds", type=int, nargs="*", default=None)
        p.add_argument("--workers", type=int, default=None)
        p.set_defaults(fn=fn)

    p = sub.add_parser("run-experiment", help="prepare + split + CV + evaluate for one config file")
    p.add_argument("config_file")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--no-final", action="store_true", help="skip the final model on all cases after the CV")
    p.set_defaults(fn=cmd_run_experiment)

    p = sub.add_parser("train-final", help="train ONE model on all cases -> <run>/final/model.pt")
    p.add_argument("config_file")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.set_defaults(fn=cmd_train_final)

    p = sub.add_parser("evaluate", help="(re-)score a finished run")
    p.add_argument("run")
    p.add_argument("--mode", choices=["stored", "inner", "crossfold"], default="stored")
    p.add_argument("--csf-rule", action="store_true", help="apply the SynthSeg CSF rule (needs tissue maps)")
    p.add_argument("--prepared", default=None, help="prepared folder (default: from the run's config.yaml)")
    p.add_argument("--tag", default=None, help="name of the output files evaluation-<tag>.json")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("compare", help="table of all runs under a folder")
    p.add_argument("root", nargs="?", default="runs")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("predict", help="apply a run's final model (or its fold models) to new scans",
                       description="Masks are written on the input image grid (+ model-grid copies in <out>/grid). "
                                   "The scans must be bias-corrected like the training data; they are prepared automatically.")
    p.add_argument("--run", required=True, help="finished run folder (holds final/ and fold0/ ...)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cases", help="case list: id, image [, brain_mask, tissue_map, extra channel columns]")
    src.add_argument("--image", help="one scan (NIfTI) instead of a case list")
    p.add_argument("--brain-mask", default=None, help="with --image: brain mask on the image matrix (default: image > 0)")
    p.add_argument("--tissue-map", default=None, help="with --image: SynthSeg map, needed for --csf-rule")
    p.add_argument("--id", default=None, help="with --image: case id for the output names (default: file name)")
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=None, help="default: the cell chosen most often by the folds")
    p.add_argument("--min-size", type=int, default=None, help="minimum size in model-grid voxels (0.25 mm3 each)")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu | mps")
    p.add_argument("--csf-rule", action="store_true", help="drop detections in CSF (needs tissue maps)")
    p.add_argument("--folds", action="store_true", help="use the 5 fold models (mean probability) instead of the final model")
    p.add_argument("--grid-only", action="store_true", help="skip the outputs on the input image grid")
    p.set_defaults(fn=cmd_predict)

    a = ap.parse_args(argv)
    if os.environ.get("CMBX_DEBUG"):
        return a.fn(a)
    try:                         # user errors as one readable line; CMBX_DEBUG=1 shows the traceback
        a.fn(a)
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        sys.exit(f"cmbx {a.cmd}: error: {e.strerror or e}: {e.filename}" if getattr(e, "filename", None) else f"cmbx {a.cmd}: error: {e}")
    except (KeyError, ValueError) as e:
        sys.exit(f"cmbx {a.cmd}: error: {e.args[0] if e.args else e}  (CMBX_DEBUG=1 for the traceback)")


if __name__ == "__main__":
    sys.exit(main())
