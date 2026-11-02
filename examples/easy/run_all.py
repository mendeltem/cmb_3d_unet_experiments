"""Run all four easy scripts one after another: prepare -> train -> test -> fine-tune.

    python examples/easy/run_all.py

Stops at the first script that fails. Steps that already have results are not redone (delete ../my_runs/easy to
start fresh). On a GPU everything takes a few minutes, on a CPU about 30 minutes.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("1_prepare_data.py", []),                                # scans -> train / test
    ("2_train.py", []),                                       # train from scratch
    ("3_test.py", []),                                        # test the pretrained model
    ("3_test.py", ["../my_runs/easy/model_scratch"]),         # test the model of step 2
    ("4_fine_tune.py", []),                                   # fine-tune the pretrained model and test it
]

for script, args in STEPS:
    print(f"\n===== {script} {' '.join(args)}", flush=True)
    start = time.time()
    if subprocess.run([sys.executable, os.path.join(HERE, script), *args]).returncode != 0:
        sys.exit(f"STOPPED: {script} failed (see the messages above)")
    print(f"===== {script} done ({(time.time() - start) / 60:.1f} min)", flush=True)

print("\nALL STEPS RAN. Results are in ../my_runs/easy/")
