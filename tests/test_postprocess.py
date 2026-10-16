import numpy as np
from scipy import ndimage

from cmbx.metrics import hits
from cmbx.postprocess import choose_cell, filter_components, grid_rows, majority_label

THR = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SIZES = (1, 2, 3, 5)


def _volume(rng, it, shape=(40, 70, 90)):
    lab = np.zeros(shape, np.uint8)
    for _ in range(rng.randint(0, 6)):
        c = [rng.randint(3, s - 3) for s in shape]; r = rng.randint(1, 3)
        lab[c[0] - r:c[0] + r, c[1] - r:c[1] + r, c[2] - r:c[2] + r] = 1
    p = ndimage.gaussian_filter(rng.rand(*shape).astype(np.float32), 1.5)
    p = (p - p.min()) / (p.max() - p.min())
    if it % 3 == 0:
        p[5:35, 10:60, 10:80] = 0.95                                  # clump over several references
    return np.clip(p + 0.5 * ndimage.gaussian_filter(lab.astype(np.float32), 1), 0, 1), lab


def test_grid_rows_equals_slow_loop():
    """The cv5.py self-test: fast grid search == hits(filter(...)) for every cell."""
    rng = np.random.RandomState(0)
    for it in range(8):
        p, lab = _volume(rng, it)
        rows = grid_rows(p, lab, THR, SIZES, 2100)
        for t in THR:
            for m in SIZES:
                assert hits(lab, filter_components(p, t, m, 2100)) == rows[(t, m)][0], (it, t, m)


def test_grid_rows_with_csf_rule_equals_slow_loop():
    rng = np.random.RandomState(1)
    for it in range(4):
        p, lab = _volume(rng, it + 1)
        tissue = np.full(p.shape, 2, np.int32)
        tissue[:, 30:50, :] = 4                                        # a CSF band
        tissue[:, :, :5] = 0                                           # some background
        rows = grid_rows(p, lab, THR, SIZES, 2100, tissue)
        for t in THR:
            for m in SIZES:
                assert hits(lab, filter_components(p, t, m, 2100, tissue=tissue)) == rows[(t, m)][0], (it, t, m)


def test_csf_rule_removes_csf_component_only():
    p = np.zeros((20, 20, 20), np.float32)
    p[3:6, 3:6, 3:6] = 0.9          # in tissue
    p[13:16, 13:16, 13:16] = 0.9    # in CSF
    tissue = np.full(p.shape, 2, np.int32)
    tissue[10:, 10:, 10:] = 24
    counter = {}
    m = filter_components(p, 0.5, 1, 2100, counter, tissue)
    assert m[4, 4, 4] and not m[14, 14, 14] and counter["csf_removed"] == 1


def test_majority_label_widens_on_background():
    seg = np.zeros((10, 10, 10), np.int32); seg[:, :, 7:] = 4
    mask = np.zeros_like(seg, bool); mask[5, 5, 5] = True
    assert majority_label(mask, seg) == 4


def test_clump_guard_and_choose_cell_fallback():
    p = np.zeros((30, 30, 30), np.float32); p[2:28, 2:28, 2:28] = 0.95   # one giant component
    lab = np.zeros(p.shape, np.uint8); lab[10, 10, 10] = 1
    assert not filter_components(p, 0.5, 1, 2100).any()
    rows = [grid_rows(p, lab, THR, SIZES, 2100)]
    t, m, s, clumps = choose_cell(rows, THR, SIZES, (0.5, 1))
    assert (t, m) == (0.5, 1) and s["f1"] == 0 and clumps == 1


def test_choose_cell_without_reference_lesions_takes_fewest_false_positives():
    """An inner fold with no reference lesion (tiny data): F1 is 0 everywhere, so the cell with the fewest false
    positives is chosen instead of the permissive fallback (0.5, 1)."""
    rng = np.random.default_rng(0)
    p = (rng.random((20, 20, 20)) * 0.85).astype(np.float32)                   # noise, all below 0.85
    p[5:7, 5:7, 5:7] = 0.95                                                     # one confident blob (8 voxels)
    lab = np.zeros(p.shape, np.uint8)                                           # no references at all
    rows = [grid_rows(p, lab, THR, SIZES, 2100)]
    t, m, s, _ = choose_cell(rows, THR, SIZES, (0.5, 1))
    assert s["n_ref"] == 0 and s["fp"] == min(r[0]["fp"] for r in rows[0].values()) and (t, m) != (0.5, 1)
