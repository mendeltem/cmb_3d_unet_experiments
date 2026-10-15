import numpy as np
from scipy import ndimage

from cmbx.frst import frst_slice, frst_volume


def _disk(X, Y, cx, cy, r):
    xx, yy = np.mgrid[0:X, 0:Y]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2).astype(float)


def test_frst_peaks_on_blob_not_on_line():
    rng = np.random.default_rng(0)
    img = np.full((200, 200), 0.6)
    img += 0.3 * _disk(200, 200, 50, 50, 4)                 # bright 2 mm disk
    img[119:121, 20:180] += 0.3                             # bright 1 mm line of the same contrast
    img = ndimage.gaussian_filter(img, 0.7) + rng.normal(0, 0.02, img.shape)
    S = frst_slice(img / np.median(img))
    win = S[42:59, 42:59]
    i, j = np.unravel_index(np.argmax(win), win.shape)
    assert max(abs(i - 8), abs(j - 8)) <= 1                 # peak at the disk centre
    assert S[114:127, 40:160].max() < 0.5 * win.max()       # the line answers far less


def test_frst_volume_range_and_background():
    img = np.zeros((60, 60, 4), np.float32)
    xx, yy = np.mgrid[0:60, 0:60]
    brain = ((xx - 30) ** 2 + (yy - 30) ** 2) < 25 ** 2
    for z in range(4):
        img[:, :, z] = np.where(brain, 0.6, 0)
    img[28:32, 28:32, 1] = 0.95
    f, info = frst_volume(img)
    assert f.dtype == np.float32 and f.min() >= 0 and f.max() <= 1
    assert (f[~np.repeat(brain[:, :, None], 4, 2)] == 0).all()
    assert f[29, 29, 1] > 0.5
