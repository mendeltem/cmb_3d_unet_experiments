"""Tutorial: train a model on a folder of cases with microbleed masks (paths in .env of the repository root).

    python examples/python/tutorial_train.py
"""
from cmbx import env, train

e = env()  # CMB_INPUT_DIR, CMB_LABEL_DIR, CMB_MODEL_DIR, CMB_TRAIN_CONFIG from .env
model = train(e["CMB_INPUT_DIR"], e["CMB_MODEL_DIR"], label_dir=e["CMB_LABEL_DIR"], config=e["CMB_TRAIN_CONFIG"],
              epochs=2)  # 2 = a quick try on the synthetic examples; for real data leave it out (60)
print("model:", model)
