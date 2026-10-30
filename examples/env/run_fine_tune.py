"""Fine-tune the pretrained VALDO model (or any trained model) on your own cohort, settings from .env (every key: examples/env/cmb_env.py).

    python examples/env/run_fine_tune.py                               # START_MODEL=pretrained, CMB_DATA, TEST_FOLD, N_SCANS ... from examples/env/.env
    python examples/env/run_fine_tune.py --set TEST_FOLD=0 N_SCANS=10
On the HPC: sbatch -p gpu --gres=shard:1 --mem=32G --time=04:00:00 --wrap "source env.sh && python examples/env/run_fine_tune.py"

Steps:
  1 settings; which cases train (as in run_train.py)
  2 link those cases into one folder
  3 fine-tune: the repository example examples/python/03_fine_tune_unet.py (= cmbx fine_tune --frst_from cmbx): start from
    the weights of START_MODEL, keep its network, channels and grid, learning rate 0.0001; LEVELS limits the trained levels
Output: <CMB_RUNS>/<NAME>/finetune_<start model>_<tag>/, a model folder for run_predict.py (MODEL=<that folder>).
Compare with run_train.py at the same TEST_FOLD and N_SCANS: that is the question "from scratch or from the public model?".
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
    start = "valdo" if S["START_MODEL"] in ("pre", "pretrained") else os.path.basename(S["START_MODEL"].rstrip("/"))
    out = S["OUT_DIR"] or os.path.join(S["CMB_RUNS"], S["NAME"], f"finetune_{start}_{tag}")
    print(f"  cohort {S['NAME']}, start {S['START_MODEL']}, {len(ids)} training cases ({tag}) -> {out}", flush=True)
    if os.path.exists(os.path.join(out, "final", "model.pt")) and not force:
        raise SystemExit(f"a model exists in {out} (--force true trains again)")

    # 2 one folder with the chosen cases
    inp = linked_input(S, ids, tag)

    # 3 fine-tune
    cmd = [sys.executable, example("03_fine_tune_unet.py"), "-i", inp, "-m", S["START_MODEL"], "-o", out,
           "--device", S["DEVICE"], "--workers", S["WORKERS"], "--force", "true"]
    cmd += ["--epochs", S["EPOCHS"]] if S["EPOCHS"] else []
    cmd += ["--lr", S["LR"]] if S["LR"] else []
    cmd += ["--config", S["CONFIG"]] if S["CONFIG"] else []
    cmd += ["--levels"] + S["LEVELS"].split() if S["LEVELS"] else []
    os.makedirs(out, exist_ok=True)
    json.dump(dict(settings=S, cases=ids, command=cmd), open(os.path.join(out, "run_settings.json"), "w"), indent=1)
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)
    print(f"model: {out}\nnext: python examples/env/run_predict.py --set MODEL={out}" + (f" TEST_FOLD={S['TEST_FOLD']}" if S["TEST_FOLD"] else ""))
