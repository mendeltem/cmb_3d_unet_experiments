"""Tutorial: find microbleeds in one scan with the pretrained VALDO model (paths in .env of the repository root).

    python examples/python/tutorial_predict.py
"""
from cmbx import env, get_model, predict

e = env()  # CMB_SCAN, CMB_BRAIN_MASK, CMB_OUTPUT_DIR from .env
model = get_model("valdo_t2")  # the VALDO 2021 U-Net (CC BY-NC-SA 4.0), downloaded once into models/
result = predict(e["CMB_SCAN"], model, brain_mask=e["CMB_BRAIN_MASK"], out_dir=e["CMB_OUTPUT_DIR"])
print(result["n_lesions"], "microbleeds, mask:", result["mask"])
