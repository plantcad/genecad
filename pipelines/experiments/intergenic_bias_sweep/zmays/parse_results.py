"""Parse gffcompare .stats files into TSVs.

Two entry points:
  - parse_sweep:     Parse bias × UTR sweep stats → results/sweep.tsv
  - parse_reference: Parse reference method stats  → results/reference.tsv

Usage:
  uv run python pipelines/experiments/intergenic_bias_sweep/zmays/parse_results.py sweep
  uv run python pipelines/experiments/intergenic_bias_sweep/zmays/parse_results.py reference
  uv run python pipelines/experiments/intergenic_bias_sweep/zmays/parse_results.py all
"""

import re
import sys
import glob
import pandas as pd

EXP_DIR = "pipelines/experiments/intergenic_bias_sweep/zmays"
STATS_DIR = f"{EXP_DIR}/results/stats"
REF_DIR = f"{EXP_DIR}/results/reference"
SWEEP_TSV = f"{EXP_DIR}/results/sweep.tsv"
REF_TSV = f"{EXP_DIR}/results/reference.tsv"


def parse_stats_text(text: str) -> dict:
    """Parse a gffcompare .stats file into level rows and missed/novel counts.

    Returns {"levels": [{"level": ..., "sensitivity": ..., "precision": ...}, ...],
             "counts": {"missed_exons_pct": ..., ...}}
    """
    levels = []
    for m in re.finditer(
        r"^\s*(.+?)\s+level:\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|",
        text,
        re.MULTILINE,
    ):
        levels.append(
            {
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
        key = f"{m.group(1).lower()}_{m.group(2)}"
        counts[f"{key}_pct"] = float(m.group(5))
        counts[f"{key}_num"] = int(m.group(3))
        counts[f"{key}_total"] = int(m.group(4))

    return {"levels": levels, "counts": counts}


def add_f1(df: pd.DataFrame) -> pd.DataFrame:
    df["f1"] = (
        2 * df["sensitivity"] * df["precision"] / (df["sensitivity"] + df["precision"])
    )
    return df


def parse_sweep():
    all_rows = []
    for path in sorted(glob.glob(f"{STATS_DIR}/gffcompare_bias_*_utrs_*.stats")):
        with open(path) as f:
            text = f.read()
        m = re.search(r"bias_([\d.]+)_utrs_(yes|no)\.stats", path)
        bias = float(m.group(1))
        utrs = m.group(2)
        parsed = parse_stats_text(text)
        for row in parsed["levels"]:
            row["intergenic_bias"] = bias
            row["require_utrs"] = utrs
            row.update(parsed["counts"])
            all_rows.append(row)

    df = add_f1(pd.DataFrame(all_rows))
    df.to_csv(SWEEP_TSV, sep="\t", index=False)
    print(f"Wrote {len(df)} rows to {SWEEP_TSV}")


def parse_reference():
    all_rows = []
    for path in sorted(glob.glob(f"{REF_DIR}/gffcompare_*.stats")):
        with open(path) as f:
            text = f.read()
        if not text.strip():
            print(f"Skipping empty file: {path}")
            continue
        # Extract method name from filename: gffcompare_<method>.stats
        m = re.search(r"gffcompare_(\w+)\.stats", path)
        method = m.group(1)
        parsed = parse_stats_text(text)
        for row in parsed["levels"]:
            row["method"] = method
            row.update(parsed["counts"])
            all_rows.append(row)

    if not all_rows:
        print(f"No reference stats found in {REF_DIR}")
        return

    df = add_f1(pd.DataFrame(all_rows))
    df.to_csv(REF_TSV, sep="\t", index=False)
    print(f"Wrote {len(df)} rows to {REF_TSV}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("sweep", "all"):
        parse_sweep()
    if mode in ("reference", "all"):
        parse_reference()
