"""``cmbx.env()``: the settings of a ``.env`` file (KEY=VALUE lines) for the tutorials in examples/python.

Search order when no path is given: the folder of the running script and its parents, then the current folder and
its parents; the first ``.env`` found wins. Relative paths in it are taken relative to the folder that holds the
``.env`` (the repository root), not to the current folder, so a tutorial runs the same from any folder. Values of
``CMB_*_DIR``, ``CMB_SCAN``, ``CMB_*_MASK`` and ``CMB_*_CONFIG`` become absolute paths (an empty value stays ""); whole numbers become int, other numbers float;
everything else stays text. Variables already set in the shell win over the file (``CMB_SCAN=my.nii.gz python ...``).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Union

Value = Union[str, int, float]


def find_env(start: Optional[str] = None) -> str:
    main = getattr(sys.modules.get("__main__"), "__file__", None)
    starts = [start] if start else [os.path.dirname(os.path.abspath(main))] if main else []
    for d in starts + [os.getcwd()]:
        d = os.path.abspath(d)
        while True:
            if os.path.isfile(os.path.join(d, ".env")):
                return os.path.join(d, ".env")
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    raise FileNotFoundError("no .env file found next to the script or in the current folder (or their parents); "
                            "copy .env.example of the repository to .env")


def _is_path(key: str) -> bool:
    return key.endswith(("_DIR", "_MASK", "_CONFIG")) or key == "CMB_SCAN"


def _typed(text: str) -> Value:
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def env(path: Optional[str] = None) -> Dict[str, Value]:
    """Read ``.env`` (found as described above when ``path`` is None) and return its settings as a dict."""
    path = os.path.abspath(path) if path and os.path.isfile(path) else find_env(path)
    root = os.path.dirname(path)
    out: Dict[str, Value] = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (x.strip() for x in line.removeprefix("export ").split("=", 1))
        value = os.environ.get(key, value.split(" #")[0].strip().strip("\"'"))
        if _is_path(key) and value:
            value = os.path.normpath(os.path.join(root, os.path.expanduser(value)))
        out[key] = _typed(value) if not _is_path(key) else value
    return out
