"""Step 4: adapt the pretrained model to your own scans (fine-tuning), then test it.

    python examples/easy/4_fine_tune.py

Fine-tuning needs far fewer scans than training from scratch: the pretrained model already knows what a microbleed
looks like and only learns your scanner. EPOCHS = 4 is a demo; use 30 for real data.
"""
from cmbx import easy

TRAIN = "../my_runs/easy/data/train"
TEST = "../my_runs/easy/data/test"
MODEL = "../my_runs/easy/model_finetuned"
OUT = "../my_runs/easy/predictions_finetuned"
EPOCHS = 4

easy.fine_tune(TRAIN, MODEL, start_from="pre", epochs=EPOCHS, patches=64)
easy.predict(TEST, MODEL, OUT)

for case, s in easy.score(OUT, TEST).items():
    f1 = "n/a" if s["f1"] is None else f"{s['f1']:.2f}"      # n/a: a case without microbleeds has no F1
    print(f"{case:8s}  found {s['tp']}, false alarms {s['fp']}, missed {s['fn']}, F1 {f1}")
