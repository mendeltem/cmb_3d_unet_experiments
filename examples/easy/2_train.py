"""Step 2: train a new model from scratch on the training cases.

    python examples/easy/2_train.py

EPOCHS = 4 only shows that it runs (a few minutes). For a real model use 60 or more and many more cases.
"""
from cmbx import easy

TRAIN = "../my_runs/easy/data/train"
MODEL = "../my_runs/easy/model_scratch"
EPOCHS = 4

easy.train(TRAIN, MODEL, epochs=EPOCHS, patches=64)   # patches per epoch: 64 for the demo, default 800
