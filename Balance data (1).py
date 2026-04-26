import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# ── Palette sombre ──────────────────────────────────────────────────────────
BG       = "#0d0d0d"
PANEL_BG = "#1a1a1a"
TEXT     = "#e0e0e0"
GRID     = "#2a2a2a"

CLASSES  = ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]
COLORS   = ["#e05c5c", "#4dbfbf", "#5c9ee0", "#e0b85c", "#7dbd7d"]

np.random.seed(42)

# ── Données simulées ─────────────────────────────────────────────────────────
# Nombre de gènes par cancer (barres)
gene_counts = [100, 300, 300, 309, 100]

# Répartition des classes (%)
class_pct = [20, 20, 20, 20, 20]   # approximativement égal d'après le pie

# Scores d'importance par classe (box + violin)
importance_data = [
    np.random.exponential(0.02, 150) * np.random.choice([-1,1], 150),
    np.random.exponential(0.025, 150) * np.random.choice([-1,1], 150),
    np.random.exponential(0.02, 150) * np.random.choice([-1,1], 150),
    np.random.exponential(0.03, 150) * np.random.choice([-1,1], 150),
    np.random.exponential(0.018, 150) * np.random.choice([-1,1], 150),
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 8), facecolor=BG)
fig.suptitle(
    "⚕  Cancer Biomarker Data Exploration — Pre-Training Balance Check",
    color=TEXT, fontsize=14, fontweight="bold", y=0.98,
    fontfamily="monospace"
)

gs = GridSpec(
    2, 3,
    figure=fig,
    left=0.05, right=0.97,
    top=0.90, bottom=0.10,
    wspace=0.35, hspace=0.55
)

ax_bar   = fig.add_subplot(gs[0, 0:2])   # barres — haut gauche (large)
ax_pie   = fig.add_subplot(gs[0, 2])     # camembert — haut droit
ax_box   = fig.add_subplot(gs[1, 0:2])   # boîtes   — bas gauche (large)
ax_viol  = fig.add_subplot(gs[1, 2])     # violons  — bas droit

for ax in [ax_bar, ax_pie, ax_box, ax_viol]:
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

# ────────────────────────────────────────────────────────────────────────────
# 1. Barres — Nombre de gènes par cancer
# ────────────────────────────────────────────────────────────────────────────
x = np.arange(len(CLASSES))
bars = ax_bar.bar(x, gene_counts, color=COLORS, width=0.55, zorder=3)

# ligne moyenne
mean_val = np.mean(gene_counts)
ax_bar.axhline(mean_val, color="#ffffff", linewidth=1, linestyle="--", zorder=4)
ax_bar.text(
    -0.5, mean_val + 5, f"Moyenne = {int(mean_val)}",
    color="#ffffff", fontsize=8, va="bottom"
)

# étiquettes au sommet des barres
for bar, val in zip(bars, gene_counts):
    ax_bar.text(
        bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
        str(val), ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold"
    )

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(CLASSES, color=TEXT, fontsize=10)
ax_bar.set_ylabel("Nombre de gènes", color=TEXT, fontsize=9)
ax_bar.set_title("Balance des classes — Nombre de gènes par cancer",
                 color=TEXT, fontsize=10, pad=8)
ax_bar.tick_params(colors=TEXT)
ax_bar.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax_bar.set_ylim(0, max(gene_counts) * 1.2)
ax_bar.yaxis.grid(True, color=GRID, zorder=0)
ax_bar.set_axisbelow(True)

# ────────────────────────────────────────────────────────────────────────────
# 2. Camembert — Répartition des classes (%)
# ────────────────────────────────────────────────────────────────────────────
wedges, texts, autotexts = ax_pie.pie(
    class_pct,
    labels=CLASSES,
    colors=COLORS,
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(linewidth=1.5, edgecolor=BG)
)
for t in texts:
    t.set_color(TEXT); t.set_fontsize(8)
for at in autotexts:
    at.set_color(BG); at.set_fontsize(7); at.set_fontweight("bold")

ax_pie.set_title("Répartition des classes (%)", color=TEXT, fontsize=10, pad=8)

# ────────────────────────────────────────────────────────────────────────────
# 3. Box plot — Distribution des scores d'importance par classe
# ────────────────────────────────────────────────────────────────────────────
bp = ax_box.boxplot(
    importance_data,
    patch_artist=True,
    medianprops=dict(color="#ffffff", linewidth=1.5),
    whiskerprops=dict(color=TEXT, linewidth=1),
    capprops=dict(color=TEXT, linewidth=1),
    flierprops=dict(marker="o", markersize=2, alpha=0.4, linestyle="none"),
    zorder=3
)
for patch, col in zip(bp["boxes"], COLORS):
    patch.set_facecolor(col)
    patch.set_alpha(0.75)
for flier, col in zip(bp["fliers"], COLORS):
    flier.set(markerfacecolor=col, markeredgecolor=col)

ax_box.set_xticks(range(1, len(CLASSES)+1))
ax_box.set_xticklabels(CLASSES, color=TEXT, fontsize=10)
ax_box.set_ylabel("Score d'importance", color=TEXT, fontsize=9)
ax_box.set_title("Distribution des scores d'importance par classe",
                 color=TEXT, fontsize=10, pad=8)
ax_box.tick_params(colors=TEXT)
ax_box.yaxis.grid(True, color=GRID, zorder=0)
ax_box.set_axisbelow(True)

# ────────────────────────────────────────────────────────────────────────────
# 4. Violin plot — Densité des importances
# ────────────────────────────────────────────────────────────────────────────
parts = ax_viol.violinplot(
    importance_data,
    positions=range(1, len(CLASSES)+1),
    showmedians=True,
    showextrema=False
)
for body, col in zip(parts["bodies"], COLORS):
    body.set_facecolor(col)
    body.set_edgecolor(col)
    body.set_alpha(0.7)
parts["cmedians"].set_color("#ffffff")
parts["cmedians"].set_linewidth(1.5)

ax_viol.set_xticks(range(1, len(CLASSES)+1))
ax_viol.set_xticklabels(CLASSES, color=TEXT, fontsize=9)
ax_viol.set_ylabel("Score d'importance", color=TEXT, fontsize=9)
ax_viol.set_title("Violin — Densité des importances",
                  color=TEXT, fontsize=10, pad=8)
ax_viol.tick_params(colors=TEXT)
ax_viol.yaxis.grid(True, color=GRID, zorder=0)
ax_viol.set_axisbelow(True)

# ── Export ────────────────────────────────────────────────────────────────────
plt.savefig("cancer_biomarker_dashboard.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("✅ Figure sauvegardée : cancer_biomarker_dashboard.png")