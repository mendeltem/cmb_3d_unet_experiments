"""Writes docs/architecture.svg: the pipeline drawn in the style of MicrobleedNet's Figure 2 (Sundaresan et al. 2023):
step 1 candidate detection (3D U-Net), step 2 candidate discrimination (teacher-student), with the feature-map sizes of
this package (cmbx/model.py, cmbx/stage2.py). Run: python scripts/make_architecture_figure.py"""
import os

W, H = 1180, 1010
out = []
add = out.append
INK, MUTED, LINE = "#0f172a", "#475569", "#94a3b8"
BLOCK, BLOCK2, SEG, CLS = "#334155", "#64748b", "#0e7490", "#7c3aed"


def text(x, y, s, size=12, weight=400, fill=INK, anchor="start", family=None):
    f = f' font-family="{family}"' if family else ""
    add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{f}>{s}</text>')


def cube(x, y, w, h, color=BLOCK, d=None):
    """A block seen slightly from above-right: front face w x h, depth d."""
    d = d if d is not None else max(4, min(10, h * 0.25))
    add(f'<path d="M{x},{y} l{d},{-d} h{w} v{h} l{-d},{d} z" fill="{color}" opacity="0.55"/>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>')


def arrow(x1, y1, x2, y2, color=MUTED, dash=False, width=1.6):
    da = ' stroke-dasharray="5 4"' if dash else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{da} marker-end="url(#ar)"/>')


def tri(x, y, up, color):
    p = f"{x - 5},{y + 4} {x + 5},{y + 4} {x},{y - 5}" if up else f"{x - 5},{y - 4} {x + 5},{y - 4} {x},{y + 5}"
    add(f'<polygon points="{p}" fill="{color}"/>')


def frame(x, y, w, h, title):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="none" stroke="{LINE}" stroke-width="1.4" stroke-dasharray="7 5"/>')
    text(x + w / 2, y - 12, title, 19, 700, anchor="middle")


def volume(x, y, label, sub, grid=True):
    """The input volume with the patch grid, as in MicrobleedNet's figure."""
    add(f'<rect x="{x}" y="{y}" width="92" height="92" rx="3" fill="#1e293b"/>')
    add(f'<ellipse cx="{x + 46}" cy="{y + 46}" rx="34" ry="40" fill="#94a3b8" opacity="0.85"/>')
    add(f'<ellipse cx="{x + 46}" cy="{y + 46}" rx="20" ry="25" fill="#cbd5e1" opacity="0.8"/>')
    if grid:
        for k in range(1, 4):
            add(f'<line x1="{x + 23 * k}" y1="{y}" x2="{x + 23 * k}" y2="{y + 92}" stroke="#fbbf24" stroke-width="0.8" stroke-dasharray="3 2"/>')
            add(f'<line x1="{x}" y1="{y + 23 * k}" x2="{x + 92}" y2="{y + 23 * k}" stroke="#fbbf24" stroke-width="0.8" stroke-dasharray="3 2"/>')
    text(x + 46, y + 110, label, 11.5, 600, anchor="middle")
    text(x + 46, y + 125, sub, 11, fill=MUTED, anchor="middle")


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">')
add('<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

# ---------------- preprocessing strip
frame(20, 40, W - 40, 92, "Preprocessing (cmbx preprocess / train)")
steps = [("scan + brain mask", "bias-corrected T2* or SWI"), ("vessel inpainting", "off; --inpaint True"),
         ("0.5 x 0.5 x 1 mm grid", "cropped to the brain"), ("invert: 1 - I / max", "inside the brain mask"),
         ("FRST", "radii 2, 3, 4, 6 voxels; --frst")]
for i, (a, b) in enumerate(steps):
    x = 40 + i * 226
    dashed = ' stroke-dasharray="5 3"' if "off" in b or "--frst" in b else ""
    add(f'<rect x="{x}" y="62" width="196" height="52" rx="8" fill="#f8fafc" stroke="{LINE}"{dashed}/>')
    text(x + 98, 84, a, 13, 700, anchor="middle")
    text(x + 98, 102, b, 11.5, fill=MUTED, anchor="middle")
    if i:
        arrow(x - 28, 88, x - 3, 88)

# ---------------- step 1
frame(20, 175, W - 40, 330, "Step 1: 3D CMB candidate detection network (anisotropic 3D U-Net)")
volume(40, 245, "patches 38 x 62 x 94", "(z, y, x) of the volume")
volume(40, 372, "the same patches", "of the FRST map")
text(150, 236, "in: 40x64x96 x 2 (image + FRST)", 10.5, family="Menlo, Consolas, monospace")
arrow(140, 300, 196, 330)
arrow(140, 410, 196, 380)
# encoder / decoder blocks: (label, x, y, w, h)
lv = [(40, "40x64x96", 32, 100), (40, "40x32x48", 64, 76), (20, "20x16x24", 128, 56), (10, "10x8x12", 256, 40), (5, "5x4x6", 512, 28)]
xs = [205, 300, 395, 490, 585]
ys = [300, 330, 352, 368, 380]
for i, ((_, size, ch, h), x, y) in enumerate(zip(lv, xs, ys)):
    cube(x, y - h / 2, 14, h)
    cube(x + 22, y - h / 2, 14 + min(ch, 512) / 64, h)
    text(x + 18, y + h / 2 + 18, f"{size} x {ch}", 10.5, anchor="middle", family="Menlo, Consolas, monospace")
    if i < 4:
        tri(x + 66, y + h / 2 - 2 + (ys[i + 1] - y) / 2 + 10, False, "#16a34a")
        arrow(x + 50, y, xs[i + 1] - 6, ys[i + 1], width=1.2)
# decoder (mirrored)
dx = [990, 900, 810, 720]
for i, (x, (_, size, ch, h), y) in enumerate(zip(dx, lv[:4], ys[:4])):
    cube(x, y - h / 2, 14 + ch / 64, h, BLOCK2)
    cube(x + 26, y - h / 2, 14, h)
    text(x + 22, y + h / 2 + 18, f"{size} x {ch}", 10.5, anchor="middle", family="Menlo, Consolas, monospace")
    add(f'<line x1="{xs[i] + 50}" y1="{y - h / 2 - 12}" x2="{x - 6}" y2="{y - h / 2 - 12}" stroke="{LINE}" stroke-width="1.3" stroke-dasharray="4 3" marker-end="url(#ar)"/>')
    tri(x - 14, y - 4, True, "#dc2626")
arrow(640, 380, 714, 368, width=1.2)
arrow(760, 368, 806, 352, width=1.2)
arrow(850, 352, 896, 330, width=1.2)
arrow(940, 330, 986, 300, width=1.2)
cube(1085, 250, 10, 100, SEG)
text(1092, 388, "40x64x96 x 1", 10.5, anchor="middle", family="Menlo, Consolas, monospace")
arrow(1056, 300, 1080, 300)
text(1092, 236, "probability", 11.5, 600, anchor="middle")
# legend step 1
ly = 455
items = [(BLOCK, "(1x)3x3 conv + instance norm + LeakyReLU, twice; level 1 in-plane only"), ("#16a34a", "max pooling (first 1x2x2, then 2x2x2)"),
         ("#dc2626", "transposed convolution (upsampling)"), (LINE, "copy and concatenate (skip connection)")]
for i, (c, s) in enumerate(items):
    x = 205 + (i % 2) * 430
    y = ly + (i // 2) * 20
    add(f'<rect x="{x}" y="{y - 9}" width="12" height="10" fill="{c}"/>')
    text(x + 18, y, s, 11.5, fill=MUTED)
text(1070, 415, "threshold + min size", 11, fill=MUTED, anchor="middle")
text(1070, 430, "chosen on held-out cases;", 11, fill=MUTED, anchor="middle")
text(1070, 445, "CSF rule: --csf_rule", 11, fill=MUTED, anchor="middle")

# ---------------- step 2
frame(20, 550, W - 40, 440, "Step 2 (optional, -cand_disc True): candidate discrimination, teacher-student as in MicrobleedNet")
add(f'<rect x="40" y="572" width="1100" height="190" rx="10" fill="#f8fafc" stroke="{LINE}"/>')
text(56, 594, "TEACHER NETWORK", 14, 700)
volume(56, 610, "16 x 32 x 32 crop", "around each candidate", grid=False)
text(56, 752, "image + FRST", 11, fill=MUTED)
# Tf
add(f'<rect x="185" y="600" width="330" height="150" rx="10" fill="none" stroke="#eab308" stroke-width="1.6"/>')
text(195, 617, "Feature extractor Tf", 12, 700, fill="#a16207")
tf = [("16x32x32 x 3", 6, 70), ("16x32x32 x 32", 14, 70), ("16x16x16 x 64", 16, 52), ("8x8x8 x 128", 20, 36)]
x = 200
for i, (s, w, h) in enumerate(tf):
    cube(x, 680 - h / 2, w, h)
    text(x + w / 2, 680 + h / 2 + 16, s, 9.5, anchor="middle", family="Menlo, Consolas, monospace")
    if i < 3:
        arrow(x + w + 6, 680, x + 70, 680, width=1.2)
    x += 80
arrow(160, 660, 196, 670)
# Ts
add(f'<rect x="530" y="600" width="290" height="150" rx="10" fill="none" stroke="#0e7490" stroke-width="1.6"/>')
text(540, 617, "Voxel-wise segmentor Ts", 12, 700, fill="#0e7490")
for i, (s, w, h) in enumerate([("16x16x16 x 64", 16, 52), ("16x32x32 x 32", 14, 70), ("16x32x32 x 1", 6, 70)]):
    xx = 560 + i * 90
    cube(xx, 680 - h / 2, w, h, SEG if i == 2 else BLOCK2)
    text(xx + w / 2, 680 + h / 2 + 16, s, 9.5, anchor="middle", family="Menlo, Consolas, monospace")
    if i < 2:
        arrow(xx + w + 6, 680, xx + 82, 680, width=1.2)
arrow(500, 668, 556, 672)
text(835, 640, "reference voxels:", 11, fill=MUTED)
text(835, 655, "BCE + soft Dice", 11, fill=MUTED)
# Tc
add(f'<rect x="830" y="672" width="300" height="80" rx="10" fill="none" stroke="#16a34a" stroke-width="1.6"/>')
text(840, 689, "Patch classifier Tc", 12, 700, fill="#15803d")
tc = [("4^3 x 128", 14, 26), ("2^3 x 128", 14, 16)]
for i, (s, w, h) in enumerate(tc):
    cube(862 + i * 52, 720 - h / 2, w, h)
    text(869 + i * 52, 745, s, 9, anchor="middle", family="Menlo, Consolas, monospace")
for i, n in enumerate(["1024", "128", "32", "2"]):
    xx = 965 + i * 38
    add(f'<rect x="{xx}" y="{704 if n != "2" else 712}" width="7" height="{36 if n != "2" else 18}" fill="{CLS}"/>')
    text(xx + 3, 760 if i % 2 == 0 else 700, n, 9.5, anchor="middle")
add(f'<path d="M458,700 C 470,768 780,768 856,728" stroke="{MUTED}" stroke-width="1.3" fill="none" marker-end="url(#ar)"/>')
text(1000, 664, "CE loss; soft labels at T = 4", 11, fill=MUTED, anchor="middle")
# KD
add(f'<rect x="420" y="785" width="340" height="34" rx="6" fill="#f5f3ff" stroke="{CLS}"/>')
text(590, 807, "Knowledge distillation: CE + 0.4 T^2 KL + 0.6 CE", 12.5, 700, fill=CLS, anchor="middle")
add(f'<path d="M1060,752 C 1060,800 900,802 764,802" stroke="{CLS}" stroke-width="2" fill="none" marker-end="url(#ar)"/>')
# student
add(f'<rect x="40" y="835" width="1100" height="140" rx="10" fill="#f8fafc" stroke="{LINE}"/>')
text(56, 857, "STUDENT NETWORK (applied to new candidates)", 14, 700)
for i in range(6):
    add(f'<rect x="{60 + (i % 3) * 30}" y="{870 + (i // 3) * 30}" width="24" height="24" rx="2" fill="#1e293b"/>')
    add(f'<circle cx="{72 + (i % 3) * 30}" cy="{882 + (i // 3) * 30}" r="4" fill="#cbd5e1"/>')
text(105, 948, "crops of the candidates", 11, fill=MUTED, anchor="middle")
x = 200
for i, (s, w, h) in enumerate(tf + tc):
    cube(x, 905 - h / 2, w, h)
    text(x + w / 2, 905 + h / 2 + 15, s, 9.5, anchor="middle", family="Menlo, Consolas, monospace")
    arrow(x + w + 6, 905, x + 74, 905, width=1.2)
    x += 82
for i, n in enumerate(["1024", "128", "32", "2"]):
    xx = x + i * 38
    add(f'<rect x="{xx}" y="{887 if n != "2" else 896}" width="7" height="{36 if n != "2" else 18}" fill="{CLS}"/>')
    text(xx + 3, 937, n, 9.5, anchor="middle")
text(x + 210, 900, "microbleed / not", 12, 700)
text(x + 210, 917, "score threshold chosen out of fold", 11, fill=MUTED)
text(x + 210, 932, "or on held-out cases", 11, fill=MUTED)
add(f'<path d="M590,819 L590,860" stroke="{CLS}" stroke-width="2" fill="none" marker-end="url(#ar)"/>')
text(600, 850, "soft labels, T = 4", 11, fill=CLS)
text(56, 970, "Blocks: 3x3x3 conv + batch norm + ReLU; fully connected layers with 20 % dropout. Alternative: --disc_model cnn (small 3D CNN, image + FRST + probability).", 11, fill=MUTED)
add("</svg>")
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "architecture.svg")
open(path, "w").write("\n".join(out))
print(path)
