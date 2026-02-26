"""Plot gffcompare metrics as a function of intergenic bias for zmays.

Produces two plots — one for REQUIRE_UTRS=yes and one for REQUIRE_UTRS=no.
Reference method baselines (e.g. Helixer) are shown as dotted horizontal lines.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines

EXP_DIR = "pipelines/experiments/intergenic_bias_sweep/zmays"
SWEEP_TSV = f"{EXP_DIR}/results/sweep.tsv"
REF_TSV = f"{EXP_DIR}/results/reference.tsv"

df = pd.read_csv(SWEEP_TSV, sep="\t")

# Load reference baselines if available
ref_df = None
if os.path.exists(REF_TSV):
    ref_df = pd.read_csv(REF_TSV, sep="\t")
    print(f"Loaded {len(ref_df)} reference rows from {REF_TSV}")

metric_labels = {"sensitivity": "Sensitivity", "precision": "Precision", "f1": "F1"}
metric_colors = {"sensitivity": "#2196F3", "precision": "#F44336", "f1": "#4CAF50"}

level_panels = ["Base", "Exon", "Intron", "Intron chain", "Transcript"]
count_panels = [
    ("Exons", "missed_exons_pct", "novel_exons_pct"),
    ("Introns", "missed_introns_pct", "novel_introns_pct"),
    ("Loci", "missed_loci_pct", "novel_loci_pct"),
]

group_bg = {"levels": "#EEF2F7", "counts": "#F4EDE4"}
missed_color, novel_color = "#E65100", "#1565C0"

utrs_titles = {"yes": "UTRs required", "no": "UTRs not required"}

for utrs_val, utrs_title in utrs_titles.items():
    sub_df = df[df["require_utrs"] == utrs_val]
    per_bias = sub_df.groupby("intergenic_bias").first().reset_index()

    fig = plt.figure(figsize=(22, 9))
    gs = fig.add_gridspec(2, 30, hspace=0.35, wspace=0.4)

    # Row 0: 5 accuracy panels (6 cols each)
    row0_axes = []
    for i in range(5):
        ax = fig.add_subplot(gs[0, i * 6 : (i + 1) * 6])
        ax.set_facecolor(group_bg["levels"])
        panel_data = sub_df[sub_df["level"] == level_panels[i]]
        for metric in metric_labels:
            ax.plot(
                panel_data["intergenic_bias"],
                panel_data[metric],
                marker="o",
                markersize=4,
                linewidth=2,
                color=metric_colors[metric],
                alpha=0.9,
            )
            # Reference baselines
            if ref_df is not None:
                ref_level = ref_df[ref_df["level"] == level_panels[i]]
                for _, ref_row in ref_level.iterrows():
                    ax.axhline(
                        ref_row[metric],
                        color=metric_colors[metric],
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.6,
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

    # Row 1: 3 missed/novel panels (8 cols each) + legend area (6 cols)
    row1_axes = []
    for i, (label, missed_col, novel_col) in enumerate(count_panels):
        ax = fig.add_subplot(gs[1, i * 8 : (i + 1) * 8])
        ax.set_facecolor(group_bg["counts"])
        ax.plot(
            per_bias["intergenic_bias"],
            per_bias[missed_col],
            marker="s",
            markersize=4,
            linewidth=2,
            label="Missed %" if i == 0 else None,
            color=missed_color,
        )
        ax.plot(
            per_bias["intergenic_bias"],
            per_bias[novel_col],
            marker="^",
            markersize=4,
            linewidth=2,
            label="Novel %" if i == 0 else None,
            color=novel_color,
        )
        # Reference baselines
        if ref_df is not None:
            ref_first = ref_df.drop_duplicates(subset=["method"])
            for _, ref_row in ref_first.iterrows():
                if missed_col in ref_row and pd.notna(ref_row[missed_col]):
                    ax.axhline(
                        ref_row[missed_col],
                        color=missed_color,
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.6,
                        label=f"Missed % ({ref_row['method']})" if i == 0 else None,
                    )
                if novel_col in ref_row and pd.notna(ref_row[novel_col]):
                    ax.axhline(
                        ref_row[novel_col],
                        color=novel_color,
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.6,
                        label=f"Novel % ({ref_row['method']})" if i == 0 else None,
                    )
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Intergenic Bias", fontsize=11)
        all_vals = per_bias[[missed_col, novel_col]].values.flatten()
        if ref_df is not None:
            ref_first = ref_df.drop_duplicates(subset=["method"])
            for col in [missed_col, novel_col]:
                if col in ref_first.columns:
                    all_vals = list(all_vals) + list(ref_first[col].dropna())
        y_max = max(all_vals) * 1.2 if len(all_vals) > 0 else 100
        ax.set_ylim(0, y_max)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Percentage (%)", fontsize=11)
        else:
            ax.set_yticklabels([])
        row1_axes.append(ax)

    # Legend area
    legend_ax = fig.add_subplot(gs[1, 24:30])
    legend_ax.axis("off")

    accuracy_handles = [
        mlines.Line2D([], [], color=metric_colors[m], linewidth=2, label=label)
        for m, label in metric_labels.items()
    ]
    if ref_df is not None:
        for method in ref_df["method"].unique():
            accuracy_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color="gray",
                    linewidth=1.5,
                    linestyle=":",
                    label=f"{method.capitalize()} baseline",
                )
            )
    leg1 = legend_ax.legend(
        handles=accuracy_handles,
        loc="upper center",
        fontsize=10,
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
        fontsize=9,
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
        f"Zea mays chr1 — gffcompare metrics vs. intergenic bias ({utrs_title})",
        fontsize=16,
        fontweight="bold",
    )
    output_path = f"{EXP_DIR}/results/sweep_plot_utrs_{utrs_val}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}")
