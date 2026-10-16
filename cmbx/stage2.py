"""``cmbx stage2 <run>``: the optional second stage -- a small 3D CNN that looks at every candidate of the U-Net again.

Why: many microbleeds the U-Net misses still get a raised probability; a low threshold finds them but floods the result
with false alarms. The second stage takes every component above a LOW threshold and answers one question per
candidate: microbleed or not (the design of MicrobleedNet's discriminator; Sundaresan et al. 2023).

    cmbx stage2 <run>                       # train + nested CV on the run's out-of-fold maps -> <run>/stage2/
    cmbx stage2 <run> --init <classifier>   # fine-tune from a trained classifier (a run, a stage2/classifier.pt, or
                                            # the classifier.pt of the microbleed_detection_3D_UNET release)
    cmbx predict --run <run> ...            # uses <run>/stage2/classifier.pt automatically (--no-stage2 to skip)

Recipe (ported from cmb-valdo-chain ``cmb/stage2``, same numbers): candidates = 26-connected components of the
OUT-OF-FOLD probability map above ``low_threshold`` 0.15, >= 4 grid voxels (1 mm3), <= 2100 voxels; per candidate a
block of 20 x 40 x 40 voxels (z, y, x) of image, FRST and probability around its centre; the network sees a 16 x 32 x 32
crop (random offset +-2/4/4 and flips in training, centre crop otherwise); 3 members with different seeds, 40 epochs,
batch 16, AdamW lr 1e-3, weight decay 1e-3, cosine schedule, BCE with positive weight = negatives / positives; scores
averaged over the members. A candidate counts as positive if it touches a reference microbleed.
Evaluation is nested: for every fold k the classifier is trained on the candidates of the other folds and scores the
candidates of fold k; the decision threshold is the one with the best pooled lesion F1 on these out-of-fold scores
(as in the original; reported next to the first-stage CV score). The final classifier is trained on all candidates.
Lesion level as everywhere in cmbx: TP = reference lesions touched by an accepted candidate, FP = accepted candidates
touching none, FN = the remaining reference lesions (also those no candidate reached).

``method: teacher_student`` follows MicrobleedNet's step 2 (Sundaresan et al. 2023, Fig. 2): input image + FRST; a
TEACHER with a feature extractor Tf, a voxel-wise segmentor Ts and a patch classifier Tc is trained on the candidates
(cross-entropy on the class + BCE/Dice on the reference voxels of the crop), then a STUDENT (Tf + Tc) learns offline
from it: CE(student, class) + alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * CE(student, class), T = 4,
alpha = 0.4 as in MicrobleedNet's code. Layers follow the original (1x1x1 conv to 3 channels, conv + batch norm +
ReLU blocks, two downsamplings, classifier 1024-128-32-2 with 20 % dropout), sized for the 16 x 32 x 32 crop of the
0.5 x 0.5 x 1 mm grid (first pooling in-plane only). Unlike MicrobleedNet's code, the teacher is trained here (there it
is loaded from pretrained files).
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

from .data import load_prepared, read_cases

S26 = np.ones((3, 3, 3), bool)
CHANNELS = ["image", "frst", "prob"]
DEFAULTS = dict(low_threshold=0.15, min_voxels=4, max_voxels=2100, half=[20, 20, 10], crop=[16, 32, 32],
                jitter=[2, 4, 4], members=3, epochs=40, batch_size=16, lr=1e-3, weight_decay=1e-3,
                method="cnn", kd_temperature=4.0, kd_alpha=0.4)
METHODS = ("cnn", "teacher_student")
THRESHOLDS = [round(float(t), 2) for t in np.arange(0.05, 0.96, 0.05)]


def build_classifier() -> nn.Sequential:
    """Identical to the released classifier (microbleed_detection_3D_UNET ``mb_segment/classifier.py``)."""
    def block(c_in, c_out):
        return nn.Sequential(nn.Conv3d(c_in, c_out, 3, padding=1, bias=False), nn.InstanceNorm3d(c_out, affine=True),
                             nn.LeakyReLU(0.01, True), nn.Conv3d(c_out, c_out, 3, padding=1, bias=False),
                             nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, True))
    return nn.Sequential(block(3, 16), nn.MaxPool3d(2), block(16, 32), nn.MaxPool3d(2), block(32, 64),
                         nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))


def _bn_block(c_in: int, c_out: int, kernel=(3, 3, 3)) -> nn.Sequential:
    pad = tuple(k // 2 for k in kernel)
    return nn.Sequential(nn.Conv3d(c_in, c_out, kernel, padding=pad, bias=False), nn.BatchNorm3d(c_out), nn.ReLU(True),
                         nn.Conv3d(c_out, c_out, kernel, padding=pad, bias=False), nn.BatchNorm3d(c_out), nn.ReLU(True))


class FeatureExtractor(nn.Module):
    """Tf: 1x1x1 conv to 3 channels, then 16 x 32 x 32 -> (1, 2, 2) pool -> 16^3 -> 2 pool -> 8^3 x 4b."""

    def __init__(self, c_in: int = 2, b: int = 32) -> None:
        super().__init__()
        self.inpconv = nn.Conv3d(c_in, 3, 1)
        self.enc0, self.pool0 = _bn_block(3, b, (1, 3, 3)), nn.MaxPool3d((1, 2, 2))
        self.enc1, self.pool1 = _bn_block(b, 2 * b), nn.MaxPool3d(2)
        self.enc2 = _bn_block(2 * b, 4 * b)

    def forward(self, x):
        x0 = self.enc0(self.inpconv(x))
        x1 = self.enc1(self.pool0(x0))
        return x0, x1, self.enc2(self.pool1(x1))


class PatchClassifier(nn.Module):
    """Tc: 1x1x1 conv, two (pool + conv) steps to 2^3 x 128, fully connected 1024-128-32-2 with 20 % dropout."""

    def __init__(self, c_in: int = 128) -> None:
        super().__init__()
        cbr = lambda a, b, k: nn.Sequential(nn.Conv3d(a, b, k, padding=k // 2, bias=False), nn.BatchNorm3d(b), nn.ReLU(True))
        self.conv = cbr(c_in, 64, 1)
        self.down1, self.down2 = nn.Sequential(nn.MaxPool3d(2), cbr(64, 128, 3)), nn.Sequential(nn.MaxPool3d(2), cbr(128, 128, 3))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 8, 128), nn.ReLU(True), nn.Dropout(0.2),
                                nn.Linear(128, 32), nn.ReLU(True), nn.Linear(32, 2))

    def forward(self, f):
        return self.fc(self.down2(self.down1(self.conv(f))))


class TeacherNet(nn.Module):
    """Tf + Ts (voxel-wise segmentor: two upsamplings with skip connections back to 16 x 32 x 32) + Tc."""

    def __init__(self, c_in: int = 2, b: int = 32) -> None:
        super().__init__()
        self.features = FeatureExtractor(c_in, b)
        self.up1, self.dec1 = nn.ConvTranspose3d(4 * b, 2 * b, 2, stride=2), _bn_block(4 * b, 2 * b)
        self.up0, self.dec0 = nn.ConvTranspose3d(2 * b, b, (1, 2, 2), stride=(1, 2, 2)), _bn_block(2 * b, b, (1, 3, 3))
        self.seg = nn.Conv3d(b, 1, 1)
        self.classifier = PatchClassifier(4 * b)

    def forward(self, x):
        x0, x1, f = self.features(x)
        h = self.dec1(torch.cat([self.up1(f), x1], 1))
        h = self.dec0(torch.cat([self.up0(h), x0], 1))
        return self.classifier(f), self.seg(h)


class StudentNet(nn.Module):
    """Tf + Tc: what is applied to new candidates."""

    def __init__(self, c_in: int = 2, b: int = 32) -> None:
        super().__init__()
        self.features, self.classifier = FeatureExtractor(c_in, b), PatchClassifier(4 * b)

    def forward(self, x):
        return self.classifier(self.features(x)[2])


def build_net(method: str) -> nn.Module:
    if method == "teacher_student":
        return StudentNet()
    if method == "cnn":
        return build_classifier()
    raise ValueError(f"stage2.method must be one of {METHODS}; got {method!r}")


def crop_block(volume: np.ndarray, centre, half) -> np.ndarray:
    """Block of ``2 * half`` around ``centre`` (x, y, z), zero-padded at the border."""
    out = np.zeros([2 * h for h in half], volume.dtype)
    src, dst = [], []
    for c, h, n in zip(centre, half, volume.shape):
        lo, hi = c - h, c + h
        start, stop = max(lo, 0), min(hi, n)
        if start >= stop:
            return out
        src.append(slice(start, stop))
        dst.append(slice(start - lo, stop - lo))
    out[tuple(dst)] = volume[tuple(src)]
    return out


def candidates(prob: np.ndarray, image: np.ndarray, frst: np.ndarray, label: Optional[np.ndarray], s: Dict):
    """Arrays in NIfTI order (x, y, z). Returns rows (centre, voxels, max prob, touched reference ids), cubes
    (n, 3, z, y, x) float16, the number of reference lesions, the component labels and, with a reference, its blocks
    (n, z, y, x) uint8 (the teacher's segmentation target)."""
    ref, n_ref = ndimage.label(label > 0, S26) if label is not None else (None, 0)
    lab, n = ndimage.label(prob > float(s["low_threshold"]), S26)
    rows, cubes, lcubes = [], [], []
    for k, box in enumerate(ndimage.find_objects(lab), 1):
        if box is None:
            continue
        m = lab[box] == k
        v = int(m.sum())
        if v < int(s["min_voxels"]) or v > int(s["max_voxels"]):
            continue
        centre = [int(round(c)) + b.start for c, b in zip(ndimage.center_of_mass(m), box)]
        touched = sorted(int(t) for t in np.unique(ref[box][m]) if t > 0) if ref is not None else []
        block = np.stack([crop_block(a, centre, s["half"]) for a in (image, frst, prob)])         # (3, x, y, z)
        cubes.append(np.transpose(block, (0, 3, 2, 1)).astype(np.float16))                       # -> (3, z, y, x)
        if label is not None:
            lcubes.append(np.transpose(crop_block((label > 0).astype(np.uint8), centre, s["half"]), (2, 1, 0)))
        rows.append(dict(component=k, centre_voxel=centre, voxels=v, probability=round(float(prob[box][m].max()), 3),
                         touched=touched))
    shape = (0, 3, 2 * s["half"][2], 2 * s["half"][1], 2 * s["half"][0])
    labels = (np.stack(lcubes) if lcubes else np.zeros((0,) + shape[2:], np.uint8)) if label is not None else None
    return rows, (np.stack(cubes) if cubes else np.zeros(shape, np.float16)), int(n_ref), lab, labels


def lesion_scores(rows: List[Dict], accepted: np.ndarray, n_ref: Dict[str, int]) -> Dict:
    tp = fp = 0
    for case in n_ref:
        idx = [i for i, r in enumerate(rows) if r["case"] == case]
        found = set()
        for i in idx:
            if accepted[i]:
                if rows[i]["touched"]:
                    found.update(rows[i]["touched"])
                else:
                    fp += 1
        tp += len(found)
    fn = sum(n_ref.values()) - tp
    return dict(tp=tp, fp=fp, fn=fn, f1=round(2 * tp / max(2 * tp + fp + fn, 1), 4),
                precision=round(tp / max(tp + fp, 1), 4), sensitivity=round(tp / max(tp + fn, 1), 4))


def _crop_center(x: np.ndarray, crop) -> np.ndarray:
    z0, y0, x0 = ((s - c) // 2 for s, c in zip(x.shape[-3:], crop))
    return x[..., z0:z0 + crop[0], y0:y0 + crop[1], x0:x0 + crop[2]]


def fit_members(cubes: np.ndarray, y: np.ndarray, s: Dict, device: str, init_states: Optional[List[Dict]] = None,
                seed: int = 0, log=print, labels: Optional[np.ndarray] = None, teachers: Optional[List] = None):
    """Trained members (students for teacher_student; their teachers in ``fit_members.teachers``)."""
    if s.get("method", "cnn") == "teacher_student":
        if labels is None:
            raise ValueError("teacher_student needs the reference blocks of the candidates")
        return _fit_teacher_student(cubes, labels, y, s, device, init_states, teachers, seed, log)
    fit_members.teachers = None
    return _fit_cnn(cubes, y, s, device, init_states, seed, log)


def _augmented(cubes, labels, idx, rng, base, jit, crop):
    xs, ls = [], []
    for i in idx:                                          # random offset around the centre, random flips
        z0, y0, x0 = (b - j + rng.randint(0, 2 * j + 1) for b, j in zip(base, jit))
        p = cubes[i][:2, z0:z0 + crop[0], y0:y0 + crop[1], x0:x0 + crop[2]]
        q = labels[i][z0:z0 + crop[0], y0:y0 + crop[1], x0:x0 + crop[2]]
        for axis in (1, 2, 3):
            if rng.rand() < 0.5:
                p, q = np.flip(p, axis), np.flip(q, axis - 1)
        xs.append(np.ascontiguousarray(p, dtype=np.float32))
        ls.append(np.ascontiguousarray(q, dtype=np.float32))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ls))[:, None]


def _fit_teacher_student(cubes, labels, y, s, device, init_states, teacher_states, seed, log):
    import torch.nn.functional as F
    crop, jit = tuple(s["crop"]), tuple(s["jitter"])
    base = [(a - c) // 2 for a, c in zip(cubes.shape[-3:], crop)]
    w = torch.tensor([1.0, float((y == 0).sum()) / max(int((y == 1).sum()), 1)], device=device)   # class balance
    T, alpha, bs, epochs = float(s["kd_temperature"]), float(s["kd_alpha"]), int(s["batch_size"]), int(s["epochs"])
    yt = torch.from_numpy(y.astype(np.int64))
    students, teachers = [], []

    def run(net, step):
        opt = torch.optim.AdamW(net.parameters(), lr=float(s["lr"]), weight_decay=float(s["weight_decay"]))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for _ in range(epochs):
            net.train()
            order = rng.permutation(len(cubes))
            for start in range(0, len(order), bs):
                idx = order[start:start + bs]
                if len(idx) < 2:                           # batch norm needs two samples
                    continue
                xb, lb = (t.to(device) for t in _augmented(cubes, labels, idx, rng, base, jit, crop))
                opt.zero_grad(set_to_none=True)
                step(xb, lb, yt[idx].to(device)).backward()
                opt.step()
            sched.step()
        return net.eval()

    for member in range(int(s["members"])):
        torch.manual_seed(seed * 10 + member)
        rng = np.random.RandomState(seed * 10 + member)
        teacher = TeacherNet()
        if teacher_states:
            teacher.load_state_dict(teacher_states[member % len(teacher_states)], strict=True)
        teacher = teacher.to(device)

        def teacher_step(xb, lb, yb):                      # class CE + voxel-wise BCE + soft Dice on the crop
            logits, seg = teacher(xb)
            p = torch.sigmoid(seg)
            dice = 1 - (2 * (p * lb).sum() + 1) / (p.sum() + lb.sum() + 1)
            return F.cross_entropy(logits, yb, weight=w) + F.binary_cross_entropy_with_logits(seg, lb) + dice
        run(teacher, teacher_step)
        student = StudentNet()
        if init_states:
            student.load_state_dict(init_states[member % len(init_states)], strict=True)
        student = student.to(device)

        def student_step(xb, lb, yb):                      # MicrobleedNet: CE + alpha T^2 KL + (1 - alpha) CE
            with torch.no_grad():
                t_logits = teacher(xb)[0]
            s_logits = student(xb)
            kd = F.kl_div(F.log_softmax(s_logits / T, 1), F.log_softmax(t_logits / T, 1), log_target=True,
                          reduction="batchmean") * T * T
            ce = F.cross_entropy(s_logits, yb)
            return F.cross_entropy(s_logits, yb, weight=w) + alpha * kd + (1 - alpha) * ce
        run(student, student_step)
        students.append(student)
        teachers.append(teacher)
    fit_members.teachers = teachers
    log(f"stage 2 (teacher-student): {len(students)} member(s) trained")
    return students


def _fit_cnn(cubes: np.ndarray, y: np.ndarray, s: Dict, device: str, init_states: Optional[List[Dict]] = None,
             seed: int = 0, log=print) -> List[nn.Module]:
    crop, jit = tuple(s["crop"]), tuple(s["jitter"])
    base = [(a - c) // 2 for a, c in zip(cubes.shape[-3:], crop)]
    pos = max(int((y == 1).sum()), 1)
    nets = []
    for member in range(int(s["members"])):
        torch.manual_seed(seed * 10 + member)
        rng = np.random.RandomState(seed * 10 + member)
        net = build_classifier()
        if init_states:
            net.load_state_dict(init_states[member % len(init_states)], strict=True)
        net = net.to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=float(s["lr"]), weight_decay=float(s["weight_decay"]))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(s["epochs"]))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float((y == 0).sum()) / pos], device=device))
        target = y.astype(np.float32)
        bs = int(s["batch_size"])
        for _ in range(int(s["epochs"])):
            net.train()
            order = rng.permutation(len(cubes))
            for start in range(0, len(order), bs):
                idx = order[start:start + bs]
                batch = []
                for i in idx:                                  # random offset around the centre, random flips
                    z0, y0, x0 = (b - j + rng.randint(0, 2 * j + 1) for b, j in zip(base, jit))
                    p = cubes[i][:, z0:z0 + crop[0], y0:y0 + crop[1], x0:x0 + crop[2]]
                    for axis in (1, 2, 3):
                        if rng.rand() < 0.5:
                            p = np.flip(p, axis)
                    batch.append(np.ascontiguousarray(p, dtype=np.float32))
                xb = torch.from_numpy(np.stack(batch)).to(device)
                yb = torch.from_numpy(target[idx]).to(device)
                opt.zero_grad(set_to_none=True)
                loss_fn(net(xb).squeeze(1), yb).backward()
                opt.step()
            sched.step()
        nets.append(net.eval())
    return nets


def score(nets: List[nn.Module], cubes: np.ndarray, crop, device: str, batch: int = 256, method: str = "cnn") -> np.ndarray:
    if len(cubes) == 0:
        return np.zeros(0, np.float32)
    out = np.zeros(len(cubes), np.float32)
    ts = method == "teacher_student"                       # student: image + FRST, two-class softmax
    prob = (lambda o: torch.softmax(o, 1)[:, 1]) if ts else (lambda o: torch.sigmoid(o.squeeze(1)))
    with torch.no_grad():
        for start in range(0, len(cubes), batch):
            c = cubes[start:start + batch][:, :2] if ts else cubes[start:start + batch]
            x = torch.from_numpy(np.ascontiguousarray(_crop_center(c, crop), dtype=np.float32)).to(device)
            out[start:start + batch] = np.mean([prob(n(x)).float().cpu().numpy() for n in nets], axis=0)
    return out


def load_bundle(source: str) -> Dict:
    """A classifier bundle (``{"members": [state, ...], "threshold": t, ...}``): a run folder, its
    stage2/classifier.pt, or the classifier.pt of the microbleed_detection_3D_UNET release (same architecture)."""
    path = source
    if os.path.isdir(source):
        path = os.path.join(source, "stage2", "classifier.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no second-stage classifier at {path} (train one with `cmbx stage2 <run>`)")
    b = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(b, dict) or "members" not in b or "threshold" not in b:
        raise ValueError(f"{path} is not a second-stage classifier bundle")
    if tuple(b.get("crop", DEFAULTS["crop"])) != tuple(DEFAULTS["crop"]):
        raise ValueError(f"{path}: crop {b.get('crop')} differs from {DEFAULTS['crop']}")
    b["path"] = os.path.abspath(path)
    return b


def _settings(cfg: Dict, overrides: Optional[Dict] = None) -> Dict:
    s = dict(DEFAULTS)
    s.update(cfg.get("stage2") or {})
    s.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return s


def _case_candidates(prepared_dir: str, sid: str, prob: np.ndarray, s: Dict):
    load = lambda n: np.asarray(nib.load(os.path.join(prepared_dir, sid, f"{sid}_{n}.nii.gz")).dataobj)
    if not os.path.exists(os.path.join(prepared_dir, sid, f"{sid}_frst.nii.gz")):
        raise SystemExit(f"{sid}: the second stage needs the FRST channel (preprocess.frst: true)")
    return candidates(prob, load("image").astype(np.float32), load("frst").astype(np.float32), load("label"), s)


def _collect(items, s: Dict, log):
    """items: (case id, fold, prepared dir, probability map (x, y, z)) -> rows, cubes, reference blocks, n_ref."""
    rows, cubes, labels, n_ref = [], [], [], {}
    for sid, fold, prepared_dir, prob in items:
        r, cb, nr, _, lb = _case_candidates(prepared_dir, sid, prob, s)
        for x in r:
            x.update(case=sid, fold=fold)
        rows += r
        cubes.append(cb)
        labels.append(lb)
        n_ref[sid] = nr
    cubes = np.concatenate(cubes) if cubes else np.zeros((0,), np.float16)
    labels = np.concatenate(labels) if labels else np.zeros((0,), np.uint8)
    log(f"stage 2: {len(rows)} candidates in {len(n_ref)} cases ({sum(bool(r['touched']) for r in rows)} on a microbleed); "
        f"reachable {sum(len({t for r in rows if r['case'] == c for t in r['touched']}) for c in n_ref)} "
        f"of {sum(n_ref.values())} reference lesions")
    return rows, cubes, labels, n_ref


def run_candidates(run_dir: str, prepared_dir: str, s: Dict, log=print):
    from .evaluate import _cases_of_run
    cases = _cases_of_run(run_dir)
    if not cases:
        raise SystemExit(f"no finished CV folds in {run_dir}: the second stage learns from their out-of-fold maps")
    items = [(c["id"], c["fold"], prepared_dir,
              np.asarray(nib.load(os.path.join(c["dir"], f"{c['id']}_pred_proba.nii.gz")).dataobj).astype(np.float32) / 255)
             for c in cases]
    return _collect(items, s, log)


def _bundle(nets, teachers, thr: float, s: Dict, init) -> Dict:
    st = lambda n: {k: v.cpu() for k, v in n.state_dict().items()}
    return dict(members=[st(n) for n in nets], teachers=[st(t) for t in teachers] if teachers else None,
                method=s.get("method", "cnn"), threshold=float(thr), crop=list(s["crop"]), half=list(s["half"]),
                channels=CHANNELS[:2] if s.get("method") == "teacher_student" else CHANNELS,
                low_threshold=float(s["low_threshold"]), min_voxels=int(s["min_voxels"]), max_voxels=int(s["max_voxels"]),
                init=init, kd_temperature=float(s["kd_temperature"]), kd_alpha=float(s["kd_alpha"]))


def _init(init, s, log):
    if not init:
        return None, None
    b = load_bundle(init)
    if b.get("method", "cnn") != s.get("method", "cnn"):
        raise SystemExit(f"{init} is a {b.get('method', 'cnn')!r} classifier; stage2.method is {s.get('method')!r}")
    log(f"stage 2: fine-tuning from {init}")
    return b["members"], b.get("teachers")


def train_stage2(run_dir: str, device: str = "auto", init: Optional[str] = None, overrides: Optional[Dict] = None,
                 log=print, out_name: Optional[str] = None) -> Dict:
    import yaml
    from .train import resolve_device
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
    s = _settings(cfg, overrides)
    if s["method"] not in METHODS:
        raise SystemExit(f"stage2.method must be one of {METHODS}; got {s['method']!r}")
    device = resolve_device(device)
    t0 = time.time()
    init = init or s.get("init_from")
    init_states, init_teachers = _init(init, s, log)
    rows, cubes, labels, n_ref = run_candidates(run_dir, cfg["data"]["prepared_dir"], s, log)
    y = np.array([1 if r["touched"] else 0 for r in rows], np.int64)
    if y.sum() == 0 or (y == 0).sum() == 0:
        raise SystemExit("stage 2 needs candidates of both kinds (on a microbleed and false ones)")
    folds = sorted({r["fold"] for r in rows})
    oof = np.zeros(len(rows), np.float32)
    for k in folds:                                            # nested: fold k is scored by a classifier that never saw it
        tr = np.array([r["fold"] != k for r in rows])
        nets = fit_members(cubes[tr], y[tr], s, device, init_states, seed=k, log=log, labels=labels[tr], teachers=init_teachers)
        oof[~tr] = score(nets, cubes[~tr], s["crop"], device, method=s["method"])
        log(f"stage 2: fold {k} scored ({int((~tr).sum())} candidates)")
    grid = {t: lesion_scores(rows, oof >= t, n_ref) for t in THRESHOLDS}
    thr = max(THRESHOLDS, key=lambda t: grid[t]["f1"])
    cv = grid[thr]
    stage1 = json.load(open(os.path.join(run_dir, "summary.json"))) if os.path.exists(os.path.join(run_dir, "summary.json")) else {}
    nets = fit_members(cubes, y, s, device, init_states, seed=99, log=log, labels=labels, teachers=init_teachers)
    out = os.path.join(run_dir, out_name or "stage2")
    os.makedirs(out, exist_ok=True)
    torch.save(_bundle(nets, fit_members.teachers, thr, s, init), os.path.join(out, "classifier.pt"))
    summary = dict(stage1_cv_f1=stage1.get("f1"), stage2_cv=cv, threshold=thr, candidates=len(rows), method=s["method"],
                   positives=int(y.sum()), cases=len(n_ref), settings=s, init=init, minutes=round((time.time() - t0) / 60, 1),
                   threshold_grid={str(t): grid[t]["f1"] for t in THRESHOLDS})
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    log(f"stage 2 [{s['method']}, nested CV]: F1 {cv['f1']} (P {cv['precision']} S {cv['sensitivity']}, TP {cv['tp']} FP {cv['fp']} "
        f"FN {cv['fn']}) at score >= {thr}; stage 1 alone: F1 {stage1.get('f1')} -> {out}/classifier.pt")
    return summary


def train_stage2_final(cfg: Dict, run_dir: str, device: str = "auto", log=print) -> Dict:
    """Step 2 for ``cmbx train -cand_disc True`` (no cross-validation): candidates of the final step-1 model on the
    training cases (as MicrobleedNet trains CDisc on its training subjects); the score threshold is chosen on the
    held-out cases of ``-tr_prop`` (0.5 without held-out cases)."""
    from .predict_cli import _load_models
    from .predict import predict_volume
    from .train import resolve_device
    s = _settings(cfg)
    if s["method"] not in METHODS:
        raise SystemExit(f"stage2.method must be one of {METHODS}; got {s['method']!r}")
    device = resolve_device(device)
    t0 = time.time()
    _, models, metas, _ = _load_models(run_dir, False, device)
    T, P = cfg["train"], cfg["predict"]
    held = set(metas[0].get("held_out") or [])
    prepared = cfg["data"]["prepared_dir"]
    items = []
    for c in read_cases(cfg["data"]["cases"]):
        r = load_prepared(prepared, c["id"], T["channels"])
        p = predict_volume(models[0], r.channels(T["channels"]), T["patch_size"], device, int(P["batch_size"]),
                           bool(T["amp_bf16"]), bool(P["tta_flip"]))[0]
        items.append((c["id"], 1 if c["id"] in held else 0, prepared, np.ascontiguousarray(p.transpose(2, 1, 0))))
    rows, cubes, labels, n_ref = _collect(items, s, log)
    y = np.array([1 if r["touched"] else 0 for r in rows], np.int64)
    tr = np.array([r["fold"] == 0 for r in rows], bool)
    if y[tr].sum() == 0 or (y[tr] == 0).sum() == 0:
        raise SystemExit("step 2 needs candidates of both kinds on the training cases (on a microbleed and false ones)")
    nets = fit_members(cubes[tr], y[tr], s, device, None, seed=0, log=log, labels=labels[tr])
    teachers = fit_members.teachers
    if held:
        sc = score(nets, cubes[~tr], s["crop"], device, method=s["method"])
        sub = [r for r, m in zip(rows, ~tr) if m]
        grid = {t: lesion_scores(sub, sc >= t, {k: v for k, v in n_ref.items() if k in held}) for t in THRESHOLDS}
        thr = max(THRESHOLDS, key=lambda t: grid[t]["f1"])
        held_f1 = grid[thr]
    else:
        thr, held_f1 = 0.5, None
    out = os.path.join(run_dir, "stage2")
    os.makedirs(out, exist_ok=True)
    torch.save(_bundle(nets, teachers, thr, s, None), os.path.join(out, "classifier.pt"))
    summary = dict(method=s["method"], threshold=thr, held_out=held_f1, candidates=int(tr.sum()), positives=int(y[tr].sum()),
                   settings=s, minutes=round((time.time() - t0) / 60, 1))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    log(f"step 2 [{s['method']}]: trained on {int(tr.sum())} candidates, score threshold {thr}"
        + (f" (held-out F1 {held_f1['f1']})" if held_f1 else " (default, no held-out cases)") + f" -> {out}/classifier.pt")
    return summary


def apply_stage2(bundle: Dict, prob_xyz: np.ndarray, image_xyz: np.ndarray, frst_xyz: np.ndarray, device: str):
    """Accepted-candidate mask (x, y, z) and per-candidate rows (score) for one prepared case."""
    s = dict(DEFAULTS, low_threshold=bundle.get("low_threshold", DEFAULTS["low_threshold"]),
             min_voxels=bundle.get("min_voxels", DEFAULTS["min_voxels"]), max_voxels=bundle.get("max_voxels", DEFAULTS["max_voxels"]),
             half=bundle.get("half", DEFAULTS["half"]))
    rows, cubes, _, lab, _ = candidates(prob_xyz, image_xyz, frst_xyz, None, s)
    method = bundle.get("method", "cnn")
    nets = []
    for st in bundle["members"]:
        n = build_net(method)
        n.load_state_dict(st, strict=True)
        nets.append(n.to(device).eval())
    sc = score(nets, cubes, s["crop"], device, method=method)
    keep = [r["component"] for r, v in zip(rows, sc) if v >= float(bundle["threshold"])]
    for r, v in zip(rows, sc):
        r["stage2_score"] = round(float(v), 3)
    return np.isin(lab, keep), rows
