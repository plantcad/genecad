#!/bin/bash
set -euo pipefail

# Sweep intergenic-bias values for Juglans regia chr1 and collect gffcompare results.
# Prerequisites:
#   - predictions.zarr must already exist in $OUTPUT_DIR/pipeline/
#   - Reference GFF must exist at $REF_GFF
#   - gffcompare must be on PATH
#
# Usage:
#   OUTPUT_DIR=... REF_GFF=... INPUT_FASTA=... \
#   bash pipelines/experiments/intergenic_bias_sweep/jregia/sweep.sh

OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"
REF_GFF="${REF_GFF:?REF_GFF must be set}"
INPUT_FASTA="${INPUT_FASTA:?INPUT_FASTA must be set (the chr1.fa used for prediction; enables frame-aware decoding)}"
GFFCOMPARE="${GFFCOMPARE:-$WORK/repos/misc/gffcompare/gffcompare}"
REQUIRE_UTRS="${REQUIRE_UTRS:-yes}"

PIPELINE_DIR="$OUTPUT_DIR/pipeline"
PREDICTIONS_ZARR="$PIPELINE_DIR/predictions.zarr"
RESULTS_TSV="$OUTPUT_DIR/intergenic_bias_sweep.tsv"

if [ ! -d "$PREDICTIONS_ZARR" ]; then
    echo "ERROR: predictions.zarr not found at $PREDICTIONS_ZARR"
    echo "Run the GPU prediction step first."
    exit 1
fi

BIAS_VALUES="${BIAS_VALUES:-0.0 0.5 1.0 1.5 2.0 3.0 5.0 8.0}"

echo -e "intergenic_bias\tlevel\tsensitivity\tprecision" > "$RESULTS_TSV"

for BIAS in $BIAS_VALUES; do
    echo "========================================="
    echo "Running with intergenic_bias=$BIAS"
    echo "========================================="

    SWEEP_DIR="$PIPELINE_DIR/sweep_bias_${BIAS}"
    mkdir -p "$SWEEP_DIR"

    # Step 1: Detect intervals (frame-aware, since --input-fasta is given).
    # Incomplete features are removed by default (pass --keep-incomplete-features to keep them).
    python scripts/detect_intervals.py \
        -i "$PREDICTIONS_ZARR" \
        -o "$SWEEP_DIR/intervals.zarr" \
        --domain plant \
        --input-fasta "$INPUT_FASTA" \
        --intergenic-bias "$BIAS"

    # Step 2: Export raw GFF (introns are stripped by default)
    python scripts/export_gff.py \
        -i "$SWEEP_DIR/intervals.zarr" \
        -o "$SWEEP_DIR/predictions__raw.gff" \
        --min-transcript-length 3

    REQUIRE_UTRS_FLAG=()
    [ "$REQUIRE_UTRS" = "yes" ] && REQUIRE_UTRS_FLAG=(--require-utrs)

    # Step 3: Filter small features, short genes, and (optionally) require
    # UTRs -- all three filters run in one pass, in that fixed order.
    python scripts/filter_raw_gff.py \
        -i "$SWEEP_DIR/predictions__raw.gff" \
        -o "$SWEEP_DIR/predictions.gff" \
        --feature-types "five_prime_UTR,three_prime_UTR,CDS" \
        --min-feature-length 2 \
        --min-gene-length 30 \
        "${REQUIRE_UTRS_FLAG[@]}"

    # Step 4: Run gffcompare
    "$GFFCOMPARE" \
        -r "$REF_GFF" \
        -C -o "$SWEEP_DIR/gffcompare" \
        "$SWEEP_DIR/predictions.gff"

    echo "--- gffcompare stats for bias=$BIAS ---"
    cat "$SWEEP_DIR/gffcompare.stats"
    echo ""

    # Parse stats and append to TSV
    python -c "
import sys
sys.path.insert(0, '.')
from src.gff_compare import parse_gffcompare_stats
stats = parse_gffcompare_stats('$SWEEP_DIR/gffcompare.stats')
for _, row in stats.iterrows():
    print(f'$BIAS\t{row[\"level\"]}\t{row[\"sensitivity\"]}\t{row[\"precision\"]}')
" >> "$RESULTS_TSV"

done

echo ""
echo "========================================="
echo "Sweep complete. Results saved to: $RESULTS_TSV"
echo "========================================="
echo ""
column -t -s $'\t' "$RESULTS_TSV"
