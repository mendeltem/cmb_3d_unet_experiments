import math

import numpy as np

from cmbx.metrics import case_f1, hits, summarize, summarize_extended


def ball(a, c, r):
    z, y, x = np.ogrid[:a.shape[0], :a.shape[1], :a.shape[2]]
    a[((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) <= r * r] = 1


def test_hits_hand_made():
    ref = np.zeros((40, 40, 40), np.uint8)
    pred = np.zeros_like(ref)
    ball(ref, (10, 10, 10), 3); ball(ref, (25, 25, 25), 2); ball(ref, (30, 8, 30), 2)   # 3 references
    ball(pred, (10, 10, 10), 2); ball(pred, (11, 13, 10), 1)     # two predictions on reference 1 -> 1 TP, 0 FP
    ball(pred, (25, 25, 25), 3)                                  # reference 2 found
    pred[5, 35, 5] = 1; pred[6, 36, 6] = 1                       # diagonal neighbours: ONE component (26-conn.)
    t = hits(ref, pred)
    assert t == dict(n_ref=3, n_pred=3, tp=2, fp=1, fn=1, case_ref=1, case_pred=1, count_error=0)
    assert math.isclose(case_f1(t), 2 * 2 / (2 * 2 + 1 + 1))


def test_summary_pooled_and_case_level():
    ref = np.zeros((20, 20, 20), np.uint8); ball(ref, (10, 10, 10), 2)
    pred = ref.copy(); pred[1, 1, 1] = 1
    a = hits(ref, pred)                                          # 1 TP, 1 FP
    empty = hits(np.zeros((5, 5, 5)), np.zeros((5, 5, 5)))       # true negative case
    fp_only = hits(np.zeros((5, 5, 5)), np.ones((5, 5, 5)))       # 1 FP in a negative case
    s = summarize([a, empty, fp_only])
    assert (s["tp"], s["fp"], s["fn"]) == (1, 2, 0)
    assert s["f1"] == round(2 / 4, 4) and s["precision"] == round(1 / 3, 4) and s["sensitivity"] == 1.0
    assert s["fp_per_case"] == round(2 / 3, 3)
    assert s["case_level"] == dict(tp=1, fp=1, tn=1, fn=0)
    e = summarize_extended([a, empty, fp_only])
    assert e["per_case_f1_n"] == 2                               # the empty case has nothing to score
    assert math.isclose(e["per_case_f1_mean"], round((2 / 3 + 0.0) / 2, 4))


def test_known_loophole_one_blob_two_references():
    ref = np.zeros((30, 30, 30), np.uint8); blob = np.zeros_like(ref)
    ball(ref, (15, 15, 10), 2); ball(ref, (15, 15, 20), 2); ball(blob, (15, 15, 15), 8)
    t = hits(ref, blob)
    assert t["tp"] == 2 and t["fp"] == 0 and t["n_pred"] == 1
