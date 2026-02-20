"""Parse all gffcompare .stats files from a sweep into a single TSV."""

import re
import glob
import pandas as pd

EXP_DIR = "pipelines/experiments/intergenic_bias_sweep"
STATS_DIR = f"{EXP_DIR}/results/stats"
OUTPUT_PATH = f"{EXP_DIR}/results/sweep.tsv"


def parse_stats_file(path: str) -> dict:
    """Extract sensitivity/precision table and missed/novel counts."""
    with open(path) as f:
        text = f.read()

    bias = float(re.search(r"bias_([\d.]+)\.stats", path).group(1))

    rows = []
    for m in re.finditer(
        r"^\s*(.+?)\s+level:\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|",
        text,
        re.MULTILINE,
    ):
        rows.append(
            {
                "intergenic_bias": bias,
                "level": m.group(1).strip(),
                "sensitivity": float(m.group(2)),
                "precision": float(m.group(3)),
            }
        )

    counts = {}
    for m in re.finditer(
        r"(Missed|Novel)\s+(exons|introns|loci):\s+(\d+)/(\d+)\s+\(\s*([\d.]+)%\)",
        text,
    ):
        key = f"{m.group(1).lower()}_{m.group(2)}_pct"
        counts[key] = float(m.group(5))
        counts[f"{m.group(1).lower()}_{m.group(2)}_num"] = int(m.group(3))
        counts[f"{m.group(1).lower()}_{m.group(2)}_total"] = int(m.group(4))

    for row in rows:
        row.update(counts)

    return rows


all_rows = []
for path in sorted(glob.glob(f"{STATS_DIR}/gffcompare_bias_*.stats")):
    all_rows.extend(parse_stats_file(path))

df = pd.DataFrame(all_rows)
df["f1"] = (
    2 * df["sensitivity"] * df["precision"] / (df["sensitivity"] + df["precision"])
)
df.to_csv(OUTPUT_PATH, sep="\t", index=False)
print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
print(df.to_string(index=False))
