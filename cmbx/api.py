"""Python API: the cmbx commands as functions, with the option names of the command line.

    from cmbx import get_model, predict, train, fine_tune

    model = get_model("valdo_t2")                                   # = cmbx get-model valdo_t2
    result = predict("scan.nii.gz", model, brain_mask="mask.nii.gz", out_dir="pred")   # = cmbx predict --image ...
    train("cohort/derivatives", "my_model", num_epochs=60)          # = cmbx train -i cohort/derivatives -m my_model -ep 60
    fine_tune("cohort/derivatives", "my_finetuned", num_epochs=10)  # = cmbx fine_tune -i ... -m pre -o my_finetuned -ep 10

Keyword options are the long option names of ``cmbx train --help`` / ``cmbx fine_tune --help`` without the dashes
(``num_epochs`` or ``epochs``, ``init_learng_rate`` or ``lr``, ``batch_size``, ``frst``, ``csf_rule``, ``device``, ...). True / False become the
``True`` / ``False`` the command line expects; a list becomes several values. Every call runs the command itself
(``python -m cmbx ...`` in a child process), so it gives the same model and masks.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from typing import Dict, List, Optional

__all__ = ["get_model", "predict", "train", "fine_tune"]
ALIASES = {"epochs": "num_epochs", "lr": "init_learng_rate"}   # short names for the most used options


def _subparser(cmd: str) -> argparse.ArgumentParser:
    from .cli import build_parser
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices[cmd]
    raise KeyError(cmd)


def _argv(cmd: str, options: Dict) -> List[str]:
    """``{"num_epochs": 2, "frst": False}`` -> ``["train", "--num_epochs", "2", "--frst", "False"]``."""
    by_dest = {}
    for action in _subparser(cmd)._actions:
        longest = max(action.option_strings, key=len, default=None)
        if longest:
            by_dest[action.dest] = longest
            for o in action.option_strings:
                by_dest.setdefault(o.lstrip("-"), longest)
    argv = [cmd]
    for key, value in options.items():
        key = ALIASES.get(key, key)
        if value is None or value == "":                 # not given (an empty .env entry counts as not given)
            continue
        if key not in by_dest:
            raise TypeError(f"{cmd}() got an unknown option {key!r}; see `cmbx {cmd} --help`")
        values = value if isinstance(value, (list, tuple)) else [value]
        argv += [by_dest[key]] + [str(v) for v in values]
    return argv


def _run(cmd: str, options: Dict) -> None:
    """Run ``python -m cmbx <cmd> ...`` with this interpreter and this cmbx. A separate process, so that the worker
    processes of the preprocessing (spawned) never re-run the calling script, which then needs no
    ``if __name__ == "__main__":`` guard."""
    argv = _argv(cmd, options)
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [package_parent, os.environ.get("PYTHONPATH")])))
    r = subprocess.run([sys.executable, "-m", "cmbx", *argv], env=env)
    if r.returncode != 0:
        raise RuntimeError(f"cmbx {cmd} failed (exit code {r.returncode}); its messages are above. Same as: "
                           f"cmbx {' '.join(argv)}")


def get_model(name: str = "valdo_t2", dir: Optional[str] = None, log=print) -> str:
    """Download a pretrained model (``cmbx get-model``) and return its folder. Default folder: ``models/`` of the
    repository when cmbx runs from a clone, else ``./models``. Files already there are checked, not fetched again."""
    from .pretrained import get_model as _get
    if dir is None:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dir = os.path.join(repo, "models") if os.path.exists(os.path.join(repo, "pyproject.toml")) else "models"
    return _get(name, dir, log)


def predict(image: str, model: str = "valdo_t2", brain_mask: Optional[str] = None, out_dir: str = "pred",
            stage2: Optional[str] = None, threshold: Optional[float] = None, min_size: Optional[int] = None,
            device: str = "auto", log=print) -> Dict:
    """Microbleeds in one scan (``cmbx predict --image``). ``model``: a model folder, or the name of a pretrained model
    (downloaded once). Returns ``{"id", "n_lesions", "total_volume_mm3", "mask"}``; ``mask`` is the microbleed mask on
    the scan's own grid in ``out_dir``. ``stage2``: a classifier folder or name (e.g. ``"valdo_classifier"``)."""
    from .pretrained import ALIASES as MODEL_ALIASES, REGISTRY
    named = lambda m: m if m is None or os.path.exists(m) or m not in set(REGISTRY) | set(MODEL_ALIASES) else get_model(m, log=log)
    _run("predict", dict(run=named(model), image=image, brain_mask=brain_mask, out=out_dir, stage2=named(stage2),
                         threshold=threshold, min_size=min_size, device=device))
    sid = os.path.basename(image).split(".nii")[0]
    row = next(r for r in csv.DictReader(open(os.path.join(out_dir, "predictions.csv"))) if r["id"] == sid)
    return dict(id=sid, n_lesions=int(row["n_lesions"]), total_volume_mm3=float(row["total_volume_mm3"]), mask=row["mask"])


def train(inp_dir: str, model_dir: str, label_dir: Optional[str] = None, **options) -> str:
    """Train a model (``cmbx train -i inp_dir -l label_dir -m model_dir [options]``); returns model_dir."""
    _run("train", dict(inp_dir=inp_dir, label_dir=label_dir, model_dir=model_dir, **options))
    return model_dir


def fine_tune(inp_dir: str, output_dir: str, model_name: str = "pre", label_dir: Optional[str] = None,
              **options) -> str:
    """Fine-tune a model (``cmbx fine_tune -i inp_dir -m model_name -o output_dir``); ``model_name="pre"`` = the
    pretrained VALDO model (downloaded once). Learning rate 0.0001 unless ``init_learng_rate`` is given. Returns
    output_dir."""
    _run("fine_tune", dict(inp_dir=inp_dir, label_dir=label_dir, model_name=model_name, output_dir=output_dir, **options))
    return output_dir
