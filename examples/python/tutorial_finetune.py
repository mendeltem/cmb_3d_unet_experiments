"""Tutorial: fine-tune the pretrained VALDO model on a folder of cases with microbleed masks (paths in .env).

    python examples/python/tutorial_finetune.py
"""
from cmbx import env, fine_tune

e = env()  # CMB_INPUT_DIR, CMB_LABEL_DIR, CMB_FINETUNED_DIR, CMB_FINETUNE_CONFIG from .env
model = fine_tune(e["CMB_INPUT_DIR"], e["CMB_FINETUNED_DIR"], label_dir=e["CMB_LABEL_DIR"], config=e["CMB_FINETUNE_CONFIG"],
                  epochs=2)  # 2 = a quick try on the synthetic examples; for real data leave it out (60)
# Paper protocol (threshold and minimum size chosen on the training scans themselves): add  thr_on="train", epochs=30
print("fine-tuned model:", model)
