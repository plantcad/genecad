#!/bin/bash
set -euo pipefail

# Sweep intergenic-bias values for Juglans regia chr1 and collect gffcompare results.
# Prerequisites:
#   - predictions.zarr must already exist in $OUTPUT_DIR/pipeline/
#   - Reference GFF must exist at $REF_GFF
#   - gffcompare must be on PATH
#
# Usage:
#   bash pipelines/experiments/intergenic_bias_sweep/sweep.sh

OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"
REF_GFF="${REF_GFF:?REF_GFF must be set}"
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

    # Step 1: Detect intervals
    python scripts/predict.py detect_intervals \
        --input-dir "$PREDICTIONS_ZARR" \
        --output "$SWEEP_DIR/intervals.zarr" \
        --decoding-methods "direct,viterbi" \
        --remove-incomplete-features yes \
        --intergenic-bias "$BIAS"

    # Step 2: Export raw GFF
    python scripts/predict.py export_gff \
        --input "$SWEEP_DIR/intervals.zarr" \
        --output "$SWEEP_DIR/predictions__raw.gff" \
        --decoding-method viterbi \
        --min-transcript-length 3 \
        --strip-introns yes

    # Step 3: Filter small features
    python scripts/gff.py filter_to_min_feature_length \
        --input "$SWEEP_DIR/predictions__raw.gff" \
        --output "$SWEEP_DIR/predictions__feat_len_2.gff" \
        --feature-types "five_prime_UTR,three_prime_UTR,CDS" \
        --min-length 2

    # Step 4: Filter short genes
    python scripts/gff.py filter_to_min_gene_length \
        --input "$SWEEP_DIR/predictions__feat_len_2.gff" \
        --output "$SWEEP_DIR/predictions__gene_len_30.gff" \
        --min-length 30

    # Step 5: Filter to valid genes
    python scripts/gff.py filter_to_valid_genes \
        --input "$SWEEP_DIR/predictions__gene_len_30.gff" \
        --output "$SWEEP_DIR/predictions.gff" \
        --require-utrs "$REQUIRE_UTRS"

    # Step 6: Run gffcompare
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
