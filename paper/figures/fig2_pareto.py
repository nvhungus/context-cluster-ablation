"""Figure 2: accuracy against parameters and against MACs on Tiny-ImageNet.

Cost from the M1-v3 benchmark (Tesla T4, fp32, 32x32, 200-class head, single session).
Accuracy is Tiny-ImageNet top-1 validation, mean over three seeds; every student is
trained with the same ResNet-18 teacher, alpha = 0.5, T = 4. ResNet-18 itself is the
CE-trained teacher and is marked separately.

Only matplotlib mathtext is used, so no LaTeX spacing macros may appear in labels.

Renders at 7.0 x 2.7 in for a two-column \\begin{figure*}.
Run:  python paper/figures/fig2_pareto.py
Out:  paper/figures/fig2_pareto.pdf (+ .png preview)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

_HERE = Path(__file__).resolve().parent
# Write straight into the Overleaf upload folder so the PDFs have one home.
OUT = _HERE.parents[1] / 'submission'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.6,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.6,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 7.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.7,
})

# name, params (M), MACs (M), top-1 (%), is_ours, is_teacher,
# label offset for panel (a) and for panel (b), in points
MODELS = [
    ("MobileNetV2",         2.48,  25.8, 53.16, False, False, (7, -4),   (7, -4)),
    ("ShuffleNetV2-x1.0",   1.46,  46.3, 57.74, False, False, (4, -11),  (7, -10)),
    ("ShuffleNetV2-x1.5",   2.68,  96.1, 59.79, False, False, (8, -3),   (8, -3)),
    ("HBCC-Medium",         1.80, 138.9, 58.63, False, False, (-10, 8),  (-6, 8)),
    ("ResNet-18 (teacher)", 11.27, 556.8, 58.71, False, True, (-30, -12), (-30, -12)),
    ("All-Local (ours)",    2.71, 231.3, 61.30, True,  False, (-12, 9),  (-14, 9)),
]

C_BASE = "#4C72B0"
C_TEACH = "#55606E"
C_OURS = "#B5482A"


def pareto(points):
    """Upper-left frontier: no other model is both cheaper and at least as accurate."""
    front = [p for p in points
             if not any(q[0] <= p[0] and q[1] >= p[1] and q != p for q in points)]
    return sorted(front)


def panel(ax, xvals, offs, xlabel, title):
    pts = [(x, m[3]) for x, m in zip(xvals, MODELS)]
    fx, fy = zip(*pareto(pts))
    ax.plot(fx, fy, color=C_OURS, linewidth=0.9, linestyle="--", alpha=0.55,
            zorder=1, label="Pareto frontier")

    for x, m, (dx, dy) in zip(xvals, MODELS, offs):
        name, _, _, acc, ours, teacher = m[0], m[1], m[2], m[3], m[4], m[5]
        if ours:
            ax.scatter([x], [acc], marker="*", s=190, color=C_OURS,
                       edgecolor="white", linewidth=0.7, zorder=5)
        elif teacher:
            ax.scatter([x], [acc], marker="s", s=34, color=C_TEACH,
                       edgecolor="white", linewidth=0.6, zorder=4)
        else:
            ax.scatter([x], [acc], marker="o", s=30, color=C_BASE,
                       edgecolor="white", linewidth=0.6, zorder=4)
        ax.annotate(name, (x, acc), textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.8, color=C_OURS if ours else "#1F2933",
                    fontweight="bold" if ours else "normal", zorder=6)

    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=4)
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.55)
    ax.set_ylim(51.0, 63.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

panel(ax1, [m[1] for m in MODELS], [m[6] for m in MODELS],
      "Parameters (M, log scale)", "(a) Accuracy vs. parameters")
ax1.set_ylabel("Tiny-ImageNet top-1 (%)")
ax1.set_xlim(1.25, 14.5)
ax1.set_xticks([1.5, 2, 3, 5, 8, 12])
ax1.set_xticklabels(["1.5", "2", "3", "5", "8", "12"])

panel(ax2, [m[2] for m in MODELS], [m[7] for m in MODELS],
      "MACs (M, log scale)", "(b) Accuracy vs. MACs")
ax2.set_xlim(20, 800)
ax2.set_xticks([25, 50, 100, 200, 400, 700])
ax2.set_xticklabels(["25", "50", "100", "200", "400", "700"])

# the automatic log-minor labels collide with the fixed major ticks
for ax in (ax1, ax2):
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

handles = [
    plt.Line2D([], [], marker="*", color=C_OURS, linestyle="none", markersize=11,
               markeredgecolor="white", label="All-Local (ours)"),
    plt.Line2D([], [], marker="o", color=C_BASE, linestyle="none", markersize=5,
               markeredgecolor="white", label="Baselines (+KD)"),
    plt.Line2D([], [], marker="s", color=C_TEACH, linestyle="none", markersize=5,
               markeredgecolor="white", label="ResNet-18 teacher (CE)"),
    plt.Line2D([], [], color=C_OURS, linestyle="--", linewidth=0.9, alpha=0.55,
               label="Pareto frontier"),
]
ax2.legend(handles=handles, loc="lower right", frameon=False, handletextpad=0.4,
           borderpad=0.2, labelspacing=0.3)

fig.tight_layout(pad=0.3, w_pad=1.6)
fig.savefig(OUT / "fig2_pareto.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig2_pareto.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT / "fig2_pareto.pdf")
