"""Command line: ``cmbx <command> ...`` (or ``python -m cmbx ...``).

  cmbx make-phantoms data/phantoms --n 10              synthetic cases + case list, to try everything without data
  cmbx make-caselist --bids <root> --modality t2star --out <csv>   case list from the in-house BIDS layout
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
    from .data import caselist_from_bids, write_caselist
    rows = caselist_from_bids(a.bids, a.modality, a.tissue_pattern, not a.no_folds, _log)
    write_caselist(rows, a.out)
    _log(f"case list with {len(rows)} cases -> {a.out}")


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
    run_experiment(cfg, a.folds, a.workers, _log)


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
    predict_run(a.run, a.cases, a.out, a.threshold, a.min_size, a.device, a.csf_rule, _log)


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

    p = sub.add_parser("make-caselist", help="case list from a BIDS dataset (in-house layout)")
    p.add_argument("--bids", required=True, help="BIDS root, e.g. /home/uchralt/data/bids/catalina-t2star")
    p.add_argument("--modality", choices=["t2star", "swi"], required=True)
    p.add_argument("--out", required=True, help="case list CSV (keep it OUTSIDE the repository for in-house data)")
    p.add_argument("--tissue-pattern", default=None, help="SynthSeg path with {id}, for the CSF rule")
    p.add_argument("--no-folds", action="store_true", help="ignore the fold column of participants.tsv")
    p.set_defaults(fn=cmd_make_caselist)

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
    p.set_defaults(fn=cmd_run_experiment)

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

    p = sub.add_parser("predict", help="fold-ensemble prediction for new, prepared cases")
    p.add_argument("--run", required=True)
    p.add_argument("--cases", required=True, help="case list; cases must be prepared in the run's prepared_dir "
                                                  "or in --prepared")
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=None, help="default: the cell chosen most often by the folds")
    p.add_argument("--min-size", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--csf-rule", action="store_true")
    p.set_defaults(fn=cmd_predict)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
