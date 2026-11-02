"""Step 1: bring the five example VALDO scans into the cmbx layout and split them into train and test.

    python examples/easy/1_prepare_data.py

For your own data: change the folders and the case names below.
"""
from cmbx import easy

RAW = "examples/valdo/raw"                  # <case>_T2starw.nii.gz (scan) and <case>_cmb.nii.gz (microbleed mask)
DATA = "../my_runs/easy/data"               # results go outside the repository

easy.preprocess(RAW, DATA + "/all", scan_ending="_T2starw", masks=RAW, mask_ending="_cmb.nii.gz",
                brain_extracted=True)       # the VALDO scans are already skull-stripped

easy.split(DATA + "/all", DATA,
           train=["sub-106", "sub-305", "sub-320"],   # the model learns from these
           test=["sub-103", "sub-104"])               # and is checked on these (never seen in training)
