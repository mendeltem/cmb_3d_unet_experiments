"""Step 3: find microbleeds in the test cases and count how many were found.

    python examples/easy/3_test.py                 # the pretrained VALDO model
    python examples/easy/3_test.py ../my_runs/easy/model_scratch     # or any model folder
"""
import sys
from cmbx import easy

TEST = "../my_runs/easy/data/test"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "pre"      # "pre" = the pretrained model that comes with cmbx
OUT = "../my_runs/easy/predictions"

easy.predict(TEST, MODEL, OUT)

for case, s in easy.score(OUT, TEST).items():
    f1 = "n/a" if s["f1"] is None else f"{s['f1']:.2f}"      # n/a: a case without microbleeds has no F1
    print(f"{case:8s}  found {s['tp']}, false alarms {s['fp']}, missed {s['fn']}, F1 {f1}")
