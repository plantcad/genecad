"""Plot gffcompare metrics as a function of intergenic bias (2x5 grid)."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

EXP_DIR = "pipelines/experiments/intergenic_bias_sweep"
TSV_PATH = f"{EXP_DIR}/results/sweep.tsv"
OUTPUT_PATH = f"{EXP_DIR}/results/sweep_plot.png"

df = pd.read_csv(TSV_PATH, sep="\t")
per_bias = df.groupby("intergenic_bias").first().reset_index()

metric_labels = {"sensitivity": "Sensitivity", "precision": "Precision", "f1": "F1"}
metric_colors = {"sensitivity": "#2196F3", "precision": "#F44336", "f1": "#4CAF50"}
metric_styles = {"sensitivity": "--", "precision": ":", "f1": "-"}
metrics = list(metric_labels)

level_panels = ["Base", "Exon", "Intron", "Intron chain", "Transcript"]
count_panels = [
    ("Exons", "missed_exons_pct", "novel_exons_pct"),
    ("Introns", "missed_introns_pct", "novel_introns_pct"),
    ("Loci", "missed_loci_pct", "novel_loci_pct"),
]

group_bg = {"levels": "#EEF2F7", "counts": "#F4EDE4"}
missed_color, novel_color = "#E65100", "#1565C0"

fig = plt.figure(figsize=(22, 9))
# 30-column grid: row 0 = 5x6, row 1 = 3x8 + 1x6 (two legend cells)
gs = fig.add_gridspec(2, 30, hspace=0.35, wspace=0.4)

# Row 0: 5 accuracy panels (6 cols each)
row0_axes = []
for i in range(5):
    ax = fig.add_subplot(gs[0, i * 6 : (i + 1) * 6])
    ax.set_facecolor(group_bg["levels"])
    sub = df[df["level"] == level_panels[i]]
    for metric in metrics:
        ax.plot(
            sub["intergenic_bias"],
            sub[metric],
            marker="o",
            markersize=5,
            linewidth=2,
            label=metric_labels[metric],
            color=metric_colors[metric],
            linestyle=metric_styles[metric],
        )
    ax.set_title(level_panels[i], fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.set_ylabel("Score (%)", fontsize=11)
    else:
        ax.set_yticklabels([])
    row0_axes.append(ax)

# Row 1: 3 missed/novel panels (8 cols each = 24) + legend area (6 cols)
row1_axes = []
for i, (label, missed_col, novel_col) in enumerate(count_panels):
    ax = fig.add_subplot(gs[1, i * 8 : (i + 1) * 8])
    ax.set_facecolor(group_bg["counts"])
    ax.plot(
        per_bias["intergenic_bias"],
        per_bias[missed_col],
        marker="s",
        markersize=5,
        linewidth=2,
        label="Missed %",
        color=missed_color,
    )
    ax.plot(
        per_bias["intergenic_bias"],
        per_bias[novel_col],
        marker="^",
        markersize=5,
        linewidth=2,
        label="Novel %",
        color=novel_color,
    )
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_xlabel("Intergenic Bias", fontsize=11)
    y_max = max(per_bias[[missed_col, novel_col]].max()) * 1.2
    ax.set_ylim(0, y_max)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.set_ylabel("Percentage (%)", fontsize=11)
    else:
        ax.set_yticklabels([])
    row1_axes.append(ax)

# Legend area: right-aligned, filling cols 24-30
legend_ax = fig.add_subplot(gs[1, 24:30])
legend_ax.axis("off")

h1, l1 = row0_axes[0].get_legend_handles_labels()
leg1 = legend_ax.legend(
    h1,
    l1,
    loc="upper center",
    fontsize=11,
    frameon=True,
    title="Accuracy",
    title_fontsize=11,
    bbox_to_anchor=(0.5, 0.95),
    fancybox=True,
    edgecolor="#AAB4C0",
    facecolor=group_bg["levels"],
)

h2, l2 = row1_axes[0].get_legend_handles_labels()
legend_ax.legend(
    h2,
    l2,
    loc="lower center",
    fontsize=11,
    frameon=True,
    title="Missed / Novel",
    title_fontsize=11,
    bbox_to_anchor=(0.5, 0.05),
    fancybox=True,
    edgecolor="#C4B5A5",
    facecolor=group_bg["counts"],
)
legend_ax.add_artist(leg1)

fig.suptitle(
    "Juglans regia chr1 — gffcompare metrics vs. intergenic bias",
    fontsize=16,
    fontweight="bold",
)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT_PATH}")
