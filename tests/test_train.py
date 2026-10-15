import numpy as np
import torch

from cmbx.config import DEFAULTS
from cmbx.data import PreparedCase
from cmbx.predict import predict_volume, window_starts
from cmbx.train import loss_bce_dice, random_start, sample_patches

PATCH = (38, 62, 94)


def test_random_start_keeps_voxel_inside_patch():
    rng = np.random.RandomState(0)
    for _ in range(20000):
        shp = (60, 80, 120)
        c = [rng.randint(s) for s in shp]
        st = random_start(c, shp, rng, PATCH, 4)
        assert all(st[i] <= c[i] < st[i] + PATCH[i] for i in range(3))


def test_loss_bce_dice_values():
    t = torch.zeros(2, 1, 4, 4, 4); t[0, 0, 1:3, 1:3, 1:3] = 1
    perfect = torch.where(t > 0, torch.tensor(20.0), torch.tensor(-20.0))
    assert float(loss_bce_dice(perfect, t)) < 1e-3
    assert float(loss_bce_dice(-perfect, t)) > 0.9


def _case(seed=0, shape=(30, 70, 100)):
    rng = np.random.RandomState(seed)
    img = np.zeros(shape, np.float32); img[3:27, 5:65, 5:95] = rng.rand(24, 60, 90).astype(np.float32) + 1e-3
    lab = np.zeros(shape, np.uint8); lab[10:12, 30:33, 40:43] = 1
    comps = [np.argwhere(lab > 0)]
    return PreparedCase("c", img, img * 0.5, lab, np.eye(4), None, comps)


def test_sample_patches_shapes_and_lesion_share():
    T = dict(DEFAULTS["train"], n_patches=50)
    recs = {"c": _case()}
    x, y = sample_patches(recs, ["c"], np.random.RandomState(1), T)
    assert x.shape == (50, 2, 38, 62, 94) and y.shape == (50, 1, 38, 62, 94)
    assert int((y.flatten(1).sum(1) > 0).sum()) >= int(50 * 0.6)
    T1 = dict(T, channels=["image"])
    x1, _ = sample_patches(recs, ["c"], np.random.RandomState(1), T1)
    assert x1.shape[1] == 1


def test_sliding_window_covers_volume_and_tta_is_consistent():
    starts = window_starts((50, 70, 100), PATCH)
    assert starts[-1] == (12, 8, 6)
    class Const(torch.nn.Module):
        def forward(self, x):
            return x[:, :1] * 0 + 2.0
    r = _case()
    p, n = predict_volume(Const(), [r.image, r.frst], PATCH, "cpu", 8, True, False)
    assert p.shape == r.image.shape and n > 0
    assert np.allclose(p[r.image != 0], 1 / (1 + np.exp(-2.0)), atol=1e-6)
    p2, _ = predict_volume(Const(), [r.image, r.frst], PATCH, "cpu", 8, True, True)
    assert np.allclose(p, p2, atol=1e-6)
