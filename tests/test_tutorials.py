"""examples/python/tutorial_*.py: at most 10 code lines each, shown verbatim in the README, paths only from .env
(relative, inside the repository or ../my_runs, never absolute), and each tutorial runs AS IS on CPU from a copy of
the repository, from the repository root and from another folder. Also the Python API and cmbx.env()."""
import ast
import csv
import io
import os
import shutil
import subprocess
import sys
import tokenize

import pytest

from cmbx import env
from cmbx.api import _argv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUTORIALS = ["tutorial_predict.py", "tutorial_train.py", "tutorial_finetune.py"]


def code_lines(path):
    """Lines that hold code: no blank lines, comments or the module docstring."""
    src = open(path).read()
    doc = ast.get_docstring(ast.parse(src), clean=False)
    lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                        tokenize.ENDMARKER) or (tok.type == tokenize.STRING and doc is not None and doc in tok.string):
            continue
        lines.update(range(tok.start[0], tok.end[0] + 1))
    return len(lines)


@pytest.mark.parametrize("name", TUTORIALS)
def test_tutorial_is_short_has_no_paths_and_is_in_the_readme(name):
    path = os.path.join(ROOT, "examples", "python", name)
    assert code_lines(path) <= 10
    src = open(path).read()
    body = src[src.index('"""', 3) + 3:].strip()
    assert body in open(os.path.join(ROOT, "README.md")).read()
    tree = ast.parse(src)
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)][1:]
    assert not [t for t in strings if "/" in t or t.endswith(".nii.gz")], "paths belong in .env"
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While, ast.comprehension))], "no loops"
    assert "env()" in body


def test_env_file_is_committed_relative_and_private_free():
    for f in (".env", ".env.example"):
        text = open(os.path.join(ROOT, f)).read()
        values = dict(l.split("=", 1) for l in text.splitlines() if l.strip() and not l.startswith("#"))
        assert {"CMB_INPUT_DIR", "CMB_LABEL_DIR", "CMB_MODEL_DIR", "CMB_OUTPUT_DIR", "CMB_SCAN"} <= set(values), f
        for k, v in values.items():
            assert k.startswith("CMB_") and not os.path.isabs(v) and "~" not in v, (f, k, v)
            assert v == "" or v.startswith(("examples/", "configs/", "../my_runs/")), (f, k, v)
    if shutil.which("git") and os.path.isdir(os.path.join(ROOT, ".git")):
        for f in (".env", ".env.example"):
            r = subprocess.run(["git", "check-ignore", "-q", f], cwd=ROOT)
            assert r.returncode == 1, f"{f} is git-ignored"


def test_env_resolves_against_the_file_not_the_current_folder(tmp_path, monkeypatch):
    (tmp_path / "proj" / "sub").mkdir(parents=True)
    (tmp_path / "proj" / ".env").write_text("# c\nCMB_INPUT_DIR=data/in\nCMB_SCAN=s.nii.gz  # note\nCMB_EPOCHS=3\n"
                                            "CMB_NAME='x y'\nexport CMB_OUTPUT_DIR=../out\n")
    monkeypatch.chdir(tmp_path)
    e = env(str(tmp_path / "proj" / "sub"))                       # found in a parent folder
    assert e["CMB_INPUT_DIR"] == str(tmp_path / "proj" / "data" / "in")
    assert e["CMB_SCAN"] == str(tmp_path / "proj" / "s.nii.gz") and e["CMB_OUTPUT_DIR"] == str(tmp_path / "out")
    assert e["CMB_EPOCHS"] == 3 and e["CMB_NAME"] == "x y"
    monkeypatch.setenv("CMB_EPOCHS", "7")                          # the shell wins
    assert env(str(tmp_path / "proj" / ".env"))["CMB_EPOCHS"] == 7
    with pytest.raises(FileNotFoundError):
        env(str(tmp_path / "nowhere"))


def test_api_maps_keywords_to_the_command_line_options():
    assert _argv("train", dict(inp_dir="a", model_dir="b", label_dir=None, epochs=2, frst=False)) == \
        ["train", "--inp_dir", "a", "--model_dir", "b", "--num_epochs", "2", "--frst", "False"]
    assert _argv("predict", dict(run="m", image="i", brain_mask="b", out="o")) == \
        ["predict", "--run", "m", "--image", "i", "--brain-mask", "b", "--out", "o"]
    with pytest.raises(TypeError, match="unknown option"):
        _argv("train", dict(inp_dir="a", model_dir="b", epoch=2))


def test_api_names_survive_submodule_imports():
    import cmbx
    import cmbx.predict  # noqa: F401
    import cmbx.train  # noqa: F401
    from cmbx.train import resolve_device  # noqa: F401  (the module stays importable)
    assert callable(cmbx.train) and callable(cmbx.predict) and cmbx.train.__module__ == "cmbx.api"


@pytest.fixture(scope="module")
def clone(tmp_path_factory):
    """A copy of the repository as a user gets it (tracked files + examples), with models/ shared so the pretrained
    files are downloaded once."""
    t = tmp_path_factory.mktemp("tut")
    repo = t / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git", "models", "__pycache__", "*.egg-info", "derivatives_cmbx"))
    shutil.rmtree(repo / "examples" / "train" / "derivatives" / "cmbx", ignore_errors=True)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    os.symlink(os.path.join(ROOT, "models"), repo / "models")
    (t / "elsewhere").mkdir()
    return t


def run_tutorial(clone, name, cwd):
    env_ = dict(os.environ, PYTHONPATH=str(clone / "repo"), CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, str(clone / "repo" / "examples" / "python" / name)], cwd=cwd, env=env_,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


@pytest.mark.parametrize("where", ["root", "elsewhere"])
def test_tutorials_run_as_is(clone, where):
    cwd = clone / "repo" if where == "root" else clone / "elsewhere"
    try:
        import urllib.request
        urllib.request.urlopen("https://github.com", timeout=10).close()
    except Exception:
        pytest.skip("GitHub not reachable: the pretrained model cannot be downloaded")
    out = run_tutorial(clone, "tutorial_predict.py", cwd)
    assert "microbleeds, mask:" in out
    rows = list(csv.DictReader(open(clone / "my_runs" / "tutorial_pred" / "predictions.csv")))
    assert [r["id"] for r in rows] == ["phantom-00_t2star"] and os.path.exists(rows[0]["mask"])
    assert "model:" in run_tutorial(clone, "tutorial_train.py", cwd)
    assert os.path.exists(clone / "my_runs" / "tutorial_model" / "final" / "model.pt")
    assert "fine-tuned model:" in run_tutorial(clone, "tutorial_finetune.py", cwd)
    assert os.path.exists(clone / "my_runs" / "tutorial_finetuned" / "final" / "model.pt")
    shutil.rmtree(clone / "my_runs")
