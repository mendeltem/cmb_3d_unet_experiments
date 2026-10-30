"""Train the 3D U-Net from scratch on your own cohort, settings from .env (every key: examples/env/cmb_env.py).

    python examples/env/run_train.py                                   # CMB_DATA, TEST_FOLD, N_SCANS ... from examples/env/.env
    python examples/env/run_train.py --set TEST_FOLD=0 N_SCANS=10
On the HPC: sbatch -p gpu --gres=shard:1 --mem=32G --time=04:00:00 --wrap "source env.sh && python examples/env/run_train.py"

Steps:
  1 settings; which cases train (all, or all but TEST_FOLD; N_SCANS of them, stratified by microbleed count)
  2 link those cases into one folder
  3 train: the repository example examples/python/01_train_unet.py (= cmbx train --frst_from cmbx), 60 epochs by default
Output: <CMB_RUNS>/<NAME>/train_<tag>/ (config.yaml, final/model.pt, final/meta.json, final/history.json), a model folder
for run_predict.py (MODEL=<that folder>) and run_fine_tune.py (START_MODEL=<that folder>).
"""
import json
import os
import subprocess
import sys

from cmb_env import REPO, cases_for, example, linked_input, settings

if __name__ == "__main__":
    # 1 settings and cases
    S, force = settings(description=__doc__.split("\n")[0])
    ids, tag = cases_for(S, "train")
    out = S["OUT_DIR"] or os.path.join(S["CMB_RUNS"], S["NAME"], f"train_{tag}")
    print(f"  cohort {S['NAME']}, {len(ids)} training cases ({tag}) -> {out}", flush=True)
    if os.path.exists(os.path.join(out, "final", "model.pt")) and not force:
        raise SystemExit(f"a model exists in {out} (--force true trains again)")

    # 2 one folder with the chosen cases
    inp = linked_input(S, ids, tag)

    # 3 train
    cmd = [sys.executable, example("01_train_unet.py"), "-i", inp, "-m", out, "--device", S["DEVICE"],
           "--workers", S["WORKERS"], "--force", "true"]
    cmd += ["--epochs", S["EPOCHS"]] if S["EPOCHS"] else []
    cmd += ["--lr", S["LR"]] if S["LR"] else []
    cmd += ["--config", S["CONFIG"]] if S["CONFIG"] else []
    os.makedirs(out, exist_ok=True)
    json.dump(dict(settings=S, cases=ids, command=cmd), open(os.path.join(out, "run_settings.json"), "w"), indent=1)
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)
    print(f"model: {out}\nnext: python examples/env/run_predict.py --set MODEL={out}" + (f" TEST_FOLD={S['TEST_FOLD']}" if S["TEST_FOLD"] else ""))
