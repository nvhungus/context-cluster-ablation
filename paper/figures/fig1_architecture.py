"""Figure 1: the All-Local (2.7M) architecture.

Macro pipeline on top, MetaFormer block detail below. No context clustering and no
LBPConv appear anywhere: the stage-1 token mixer is a learned 3x3 convolution and
stages 2-4 use 7x7 depthwise + 1x1 pointwise.

Geometry follows configs/ablations/hbcc_alllocal_2_68m_conv3x3_dw7.yaml:
    embed_dims [78, 116, 232, 310], depths [1, 1, 2, 1], stem stride 1, mlp ratio 3.

Only matplotlib mathtext is used, so no \\textbf, \\quad or \\, may appear in labels.

Renders at 7.0 x 2.9 in for a two-column \\begin{figure*}.
Run:  python paper/figures/fig1_architecture.py
Out:  paper/figures/fig1_architecture.pdf (+ .png preview)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_HERE = Path(__file__).resolve().parent
# Write straight into the Overleaf upload folder so the PDFs have one home.
OUT = _HERE  # write the PDFs next to this script
OUT.mkdir(parents=True, exist_ok=True)

# Colour-blind safe, prints legibly in greyscale.
C_STEM = "#D9E3F0"
C_S1 = "#F6C9A8"   # learned 3x3 stage
C_S47 = "#A8C8E8"  # 7x7 depthwise stages
C_HEAD = "#DDDDDD"
C_EDGE = "#33415C"
C_ACCENT = "#B5482A"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

W, H = 112.0, 46.0


def box(ax, x, y, w, h, text, fc, fontsize=7.2, weight="normal", lw=0.9):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
        facecolor=fc, edgecolor=C_EDGE, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, zorder=3, linespacing=1.4)


def arrow(ax, x0, y0, x1, y1, lw=1.0, color=C_EDGE):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
        linewidth=lw, color=color, zorder=4, shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(7.0, 2.9))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

ax.text(W / 2, H - 0.8, "All-Local (2.7M): no context clustering, no fixed LBP filters",
        ha="center", va="top", fontsize=8.2, fontweight="bold", color=C_EDGE)

# ---------------------------------------------------------------- macro pipeline
TOP_Y, BH, GAP = 29.0, 9.8, 2.9
stages = [
    (7.8,  "Input\n$32{\\times}32{\\times}3$", C_HEAD, "normal"),
    (10.4, "Stem\n$3{\\times}3$, stride 1\n$32{\\times}32{\\times}78$", C_STEM, "normal"),
    (13.6, "Stage 1 ($\\times$1)\nlearned $3{\\times}3$\n$32{\\times}32{\\times}78$", C_S1, "bold"),
    (13.6, "Stage 2 ($\\times$1)\n$7{\\times}7$ DW, $1{\\times}1$ PW\n$16{\\times}16{\\times}116$", C_S47, "normal"),
    (13.6, "Stage 3 ($\\times$2)\n$7{\\times}7$ DW, $1{\\times}1$ PW\n$8{\\times}8{\\times}232$", C_S47, "normal"),
    (13.6, "Stage 4 ($\\times$1)\n$7{\\times}7$ DW, $1{\\times}1$ PW\n$4{\\times}4{\\times}310$", C_S47, "normal"),
    (8.0,  "GAP\n+ BN", C_HEAD, "normal"),
    (8.4,  "Linear\n$310{\\rightarrow}C$", C_HEAD, "normal"),
]
xs = 1.2
span = []
for i, (w, label, fc, weight) in enumerate(stages):
    box(ax, xs, TOP_Y, w, BH, label, fc, weight=weight)
    span.append((xs, xs + w))
    if i < len(stages) - 1:
        arrow(ax, xs + w + 0.3, TOP_Y + BH / 2, xs + w + GAP - 0.3, TOP_Y + BH / 2)
    xs += w + GAP

# stride-2 downsampling sits on the three inter-stage arrows
for a, b in [(2, 3), (3, 4), (4, 5)]:
    xmid = (span[a][1] + span[b][0]) / 2
    ax.text(xmid, TOP_Y + BH + 0.9, "$\\downarrow$2", ha="center", va="bottom",
            fontsize=7.0, color=C_ACCENT, fontweight="bold")

# ---------------------------------------------------------------- block detail
ax.text(1.2, 25.6, "Every stage is a stack of MetaFormer blocks:",
        ha="left", va="center", fontsize=7.6, fontweight="bold", color=C_EDGE)

BOT_Y, BBH, BGAP = 7.6, 8.8, 2.0
detail = [
    (8.0,  "BN", C_HEAD),
    (16.0, "token mixer\n(stage-dependent)", C_S1),
    (9.6,  "LayerScale\n+ DropPath", C_HEAD),
    (4.6,  "$\\oplus$", "white"),
    (8.0,  "BN", C_HEAD),
    (12.0, "MLP\n(ratio 3)", C_HEAD),
    (9.6,  "LayerScale\n+ DropPath", C_HEAD),
    (4.6,  "$\\oplus$", "white"),
]
bx = 11.0
dsp = []
for i, (w, label, fc) in enumerate(detail):
    box(ax, bx, BOT_Y, w, BBH, label, fc, fontsize=7.0)
    dsp.append((bx, bx + w))
    if i < len(detail) - 1:
        arrow(ax, bx + w + 0.3, BOT_Y + BBH / 2, bx + w + BGAP - 0.3, BOT_Y + BBH / 2, lw=0.9)
    bx += w + BGAP

cy = BOT_Y + BBH / 2
ax.text(6.1, cy, "$x$", ha="right", va="center", fontsize=7.8)
arrow(ax, 6.6, cy, dsp[0][0] - 0.3, cy, lw=0.9)
arrow(ax, dsp[-1][1] + 0.3, cy, dsp[-1][1] + 3.4, cy, lw=0.9)

# residual skips, arcing shallowly above the block row so they clear the heading
for src_x, dst in [(dsp[0][0] - 1.6, 3), (dsp[3][1] + 1.0, 7)]:
    tgt = (dsp[dst][0] + dsp[dst][1]) / 2
    ax.add_patch(FancyArrowPatch(
        (src_x, BOT_Y + BBH - 1.2), (tgt, BOT_Y + BBH + 0.3),
        connectionstyle="arc3,rad=-0.16", arrowstyle="-|>", mutation_scale=7,
        linewidth=0.9, color=C_ACCENT, linestyle=(0, (3, 2)), zorder=1,
        shrinkA=0, shrinkB=0))

# which mixer each stage uses
mixer_cx = (dsp[1][0] + dsp[1][1]) / 2
ax.add_patch(FancyArrowPatch(
    ((span[2][0] + span[2][1]) / 2, TOP_Y - 0.4), (mixer_cx, BOT_Y + BBH + 0.3),
    arrowstyle="-", linewidth=0.8, color="#90A0B4", linestyle=(0, (2, 2)), zorder=0,
    shrinkA=0, shrinkB=0))
ax.text(mixer_cx, BOT_Y - 1.0,
        "learned $3{\\times}3$ at stage 1      •      "
        "$7{\\times}7$ depthwise + $1{\\times}1$ pointwise at stages 2–4",
        ha="center", va="top", fontsize=7.0, color=C_ACCENT)

fig.tight_layout(pad=0.1)
fig.savefig(OUT / "fig1_architecture.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig1_architecture.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT / "fig1_architecture.pdf")
