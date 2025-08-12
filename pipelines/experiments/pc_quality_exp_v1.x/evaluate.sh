#!/bin/bash

#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2:00:00

# PC Quality Filter Experiment - Unified Evaluation Script
# Usage: ./evaluate.sh {1.0|1.1|1.2} {original|pc-filtered}

set -euo pipefail

# Parse arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 {1.0|1.1|1.2} {original|pc-filtered}"
    echo "  1.0          - Use v1.0 model (Athaliana only)"
    echo "  1.1          - Use v1.1 model (Athaliana + Osativa)"
    echo "  1.2          - Use v1.2 model (All 5 species from v1.1 checkpoint)"
    echo "  original     - Use original (unfiltered) ground truth"
    echo "  pc-filtered  - Use PC-filtered ground truth"
    exit 1
fi

MODEL_VERSION="$1"
GROUND_TRUTH_TYPE="$2"

# Validate model version
if [ "$MODEL_VERSION" != "1.0" ] && [ "$MODEL_VERSION" != "1.1" ] && [ "$MODEL_VERSION" != "1.2" ]; then
    echo "ERROR: Invalid model version '$MODEL_VERSION'"
    echo "Must be '1.0', '1.1', or '1.2'"
    exit 1
fi

# Validate ground truth type
if [ "$GROUND_TRUTH_TYPE" != "original" ] && [ "$GROUND_TRUTH_TYPE" != "pc-filtered" ]; then
    echo "ERROR: Invalid ground truth type '$GROUND_TRUTH_TYPE'"
    echo "Must be 'original' or 'pc-filtered'"
    exit 1
fi

# Set model-specific configurations
case "$MODEL_VERSION" in
    "1.0")
        MODEL_DESCRIPTION="v1.0 (Athaliana only)"
        SWEEP_DIR="sweep-v1.0__cfg_013__arch_all__frzn_yes__lr_1e-04"
        ;;
    "1.1")
        MODEL_DESCRIPTION="v1.1 (Athaliana + Osativa)"
        SWEEP_DIR="sweep-v1.1__cfg_013__arch_all__frzn_yes__lr_1e-04"
        ;;
    "1.2")
        MODEL_DESCRIPTION="v1.2 (All 5 species fresh start)"
        SWEEP_DIR="sweep-v1.2__cfg_013__arch_all__frzn_yes__lr_1e-04"
        ;;
esac

# Set version and description based on ground truth type
# Predictions use only model version, results use extended version with ground truth type
PREDICTION_VERSION="v$MODEL_VERSION"
if [ "$GROUND_TRUTH_TYPE" = "original" ]; then
    RUN_VERSION="v$MODEL_VERSION.0"
    GT_DESCRIPTION="Original (unfiltered)"
else
    RUN_VERSION="v$MODEL_VERSION.1"
    GT_DESCRIPTION="PC-filtered (passPlantCADFilter=1 only)"
fi

echo "Starting PC Quality Filter Experiment - Evaluation $RUN_VERSION"
echo "Model: $MODEL_DESCRIPTION"
echo "Ground Truth: $GT_DESCRIPTION"
echo "Species: jregia, pvulgaris, carabica, zmays, ntabacum, nsylvestris"
echo "$(date): Beginning evaluation (assumes predictions already generated)"

# Define paths
RAW_DIR="$DATA_DIR/testing_data"
PREDICT_DIR="$PIPE_DIR/predict"

# Define species to evaluate
SPECIES_LIST="jregia pvulgaris carabica zmays ntabacum nsylvestris"
CHR_ID="chr1"

echo "Prediction directory: $PREDICT_DIR"

# Process each species
for SPECIES in $SPECIES_LIST; do
    echo "$(date): ========================================="
    echo "$(date): Processing species: $SPECIES"
    echo "$(date): ========================================="

    # Convert to proper species ID (capitalize first letter)
    SPECIES_ID=$(echo "$SPECIES" | sed 's/./\u&/')
    # Use PREDICTION_VERSION to find existing predictions, RUN_VERSION for results
    PREDICTION_DIR="$PREDICT_DIR/$SPECIES/runs/$PREDICTION_VERSION/$CHR_ID"
    SPECIES_DIR="$PREDICT_DIR/$SPECIES/runs/$RUN_VERSION/$CHR_ID"

    echo "Species ID: $SPECIES_ID"
    echo "Prediction directory: $PREDICTION_DIR"
    echo "Results directory: $SPECIES_DIR"

    # Check if final results already exist
    if [ -f "$SPECIES_DIR/results/gffcompare.stats.consolidated.csv" ]; then
        echo "$(date): Final results already exist for $SPECIES, skipping processing"
        continue
    fi

    # Create output directories
    mkdir -p "$SPECIES_DIR/gff" "$SPECIES_DIR/results"

    # Verify predictions exist
    if [ ! -d "$PREDICTION_DIR/predictions" ] || [ -z "$(ls -A "$PREDICTION_DIR/predictions" 2>/dev/null)" ]; then
        echo "ERROR: Predictions not found for $SPECIES in $PREDICTION_DIR/predictions"
        echo "Please run predict.sh $MODEL_VERSION first to generate predictions"
        exit 1
    fi
    echo "$(date): Using existing predictions for $SPECIES from $PREDICTION_DIR/predictions"

    # Step 1: Detect intervals and export GFF
    if [ -d "$SPECIES_DIR/intervals.zarr" ]; then
        echo "$(date): Step 1 - Intervals already exist for $SPECIES, skipping detection"
    else
        echo "$(date): Step 1 - Detecting intervals for $SPECIES"
        python scripts/predict.py detect_intervals \
          --input-dir "$PREDICTION_DIR/predictions" \
          --output "$SPECIES_DIR/intervals.zarr" \
          --decoding-methods "direct,viterbi" \
          --remove-incomplete-features yes
    fi

    if [ -f "$SPECIES_DIR/gff/predictions.gff" ]; then
        echo "$(date): Step 2 - GFF predictions already exist for $SPECIES, skipping export"
    else
        echo "$(date): Step 2 - Exporting predictions to GFF for $SPECIES"
        python scripts/predict.py export_gff \
          --input "$SPECIES_DIR/intervals.zarr" \
          --output "$SPECIES_DIR/gff/predictions.gff" \
          --decoding-method viterbi \
          --min-transcript-length 3 \
          --strip-introns yes
    fi

    # Step 3: Process predictions
    echo "$(date): Step 3 - Processing predictions for $SPECIES"
    python scripts/gff.py filter_to_strand \
      --input "$SPECIES_DIR/gff/predictions.gff" \
      --output "$SPECIES_DIR/gff/predictions__strand_both.gff" \
      --strand both

    python scripts/gff.py filter_to_min_feature_length \
      --input "$SPECIES_DIR/gff/predictions__strand_both.gff" \
      --output "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2.gff" \
      --feature-types "five_prime_UTR,three_prime_UTR,CDS" \
      --min-length 2

    python scripts/gff.py filter_to_min_gene_length \
      --input "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2.gff" \
      --output "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2__gene_len_30.gff" \
      --min-length 30

    python scripts/gff.py filter_to_valid_genes \
      --input "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2__gene_len_30.gff" \
      --output "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2__gene_len_30__has_cds.gff" \
      --require-utrs no

    python scripts/gff.py filter_to_valid_genes \
      --input "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2__gene_len_30.gff" \
      --output "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2__gene_len_30__has_all_feats.gff" \
      --require-utrs yes

    python scripts/gff.py remove_exon_utrs \
      --input "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2__gene_len_30__has_cds.gff" \
      --output "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2__gene_len_30__has_cds__no_utrs.gff"

    python scripts/gff.py remove_exon_utrs \
      --input "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2__gene_len_30__has_all_feats.gff" \
      --output "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2__gene_len_30__has_all_feats__no_utrs.gff"

    # Step 4: Process labels (conditional on ground truth type)
    echo "$(date): Step 4 - Processing labels ($GT_DESCRIPTION) for $SPECIES"

    if [ "$GROUND_TRUTH_TYPE" = "original" ]; then
        # Original ground truth - no PC filtering
        python scripts/gff.py resolve \
          --input-dir "$RAW_DIR/gff" \
          --species-id "$SPECIES_ID" \
          --output "$SPECIES_DIR/gff/labels_raw.gff"

        python scripts/gff.py filter_to_chromosome \
          --input "$SPECIES_DIR/gff/labels_raw.gff" \
          --output "$SPECIES_DIR/gff/labels.gff" \
          --chromosome-id "$CHR_ID" \
          --species-id "$SPECIES_ID"
    else
        # PC-filtered ground truth
        python scripts/gff.py resolve \
          --input-dir "$RAW_DIR/gff_tagged" \
          --species-id "$SPECIES_ID" \
          --output "$SPECIES_DIR/gff/labels_raw.gff"

        python scripts/gff.py filter_to_chromosome \
          --input "$SPECIES_DIR/gff/labels_raw.gff" \
          --output "$SPECIES_DIR/gff/labels_chr.gff" \
          --chromosome-id "$CHR_ID" \
          --species-id "$SPECIES_ID"

        echo "$(date): Applying PC quality filter to ground truth for $SPECIES"
        python scripts/gff.py filter_to_pc_quality_score_pass \
          --input "$SPECIES_DIR/gff/labels_chr.gff" \
          --output "$SPECIES_DIR/gff/labels.gff"
    fi

    python scripts/gff.py remove_exon_utrs \
      --input "$SPECIES_DIR/gff/labels.gff" \
      --output "$SPECIES_DIR/gff/labels__no_utrs.gff"

    # Step 5: Run comparisons (gffcompare)
    echo "$(date): Step 5 - Running gffcompare evaluations for $SPECIES"
    python scripts/gff.py compare \
      --reference "$SPECIES_DIR/gff/labels.gff" \
      --input "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2__gene_len_30__has_all_feats.gff" \
      --output "$SPECIES_DIR/results/valid_only__with_utrs" \
      --gffcompare-path /work/10459/eczech/vista/repos/misc/gffcompare/gffcompare

    python scripts/gff.py compare \
      --reference "$SPECIES_DIR/gff/labels__no_utrs.gff" \
      --input "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2__gene_len_30__has_all_feats__no_utrs.gff" \
      --output "$SPECIES_DIR/results/valid_only__no_utrs" \
      --gffcompare-path /work/10459/eczech/vista/repos/misc/gffcompare/gffcompare

    python scripts/gff.py compare \
      --reference "$SPECIES_DIR/gff/labels.gff" \
      --input "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2__gene_len_30__has_cds.gff" \
      --output "$SPECIES_DIR/results/minlen_only__with_utrs" \
      --gffcompare-path /work/10459/eczech/vista/repos/misc/gffcompare/gffcompare

    python scripts/gff.py compare \
      --reference "$SPECIES_DIR/gff/labels__no_utrs.gff" \
      --input "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2__gene_len_30__has_cds__no_utrs.gff" \
      --output "$SPECIES_DIR/results/minlen_only__no_utrs" \
      --gffcompare-path /work/10459/eczech/vista/repos/misc/gffcompare/gffcompare

    # Step 6: Run comparisons (gffeval)
    echo "$(date): Step 6 - Running gffeval evaluations for $SPECIES"
    python scripts/gff.py clean \
      --input "$SPECIES_DIR/gff/labels.gff" \
      --output /tmp/labels_cleaned.gff

    python scripts/gff.py clean \
      --input "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2__gene_len_30__has_cds.gff" \
      --output /tmp/predictions__minlen_only__with_utrs.gff

    python scripts/gff.py clean \
      --input "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2__gene_len_30__has_all_feats.gff" \
      --output /tmp/predictions__valid_only__with_utrs.gff

    python scripts/gff.py evaluate \
      --reference /tmp/labels_cleaned.gff \
      --input /tmp/predictions__minlen_only__with_utrs.gff \
      --output "$SPECIES_DIR/results/minlen_only__with_utrs" \
      --edge-tolerance 0

    python scripts/gff.py evaluate \
      --reference /tmp/labels_cleaned.gff \
      --input /tmp/predictions__valid_only__with_utrs.gff \
      --output "$SPECIES_DIR/results/valid_only__with_utrs" \
      --edge-tolerance 0

    # Step 7: Consolidate results
    echo "$(date): Step 7 - Consolidating results for $SPECIES"
    python scripts/gff.py collect_results \
      --input "$SPECIES_DIR/results" \
      --output "$SPECIES_DIR/results/gffcompare.stats.consolidated.csv"

    # Step 8: Merge labels and predictions
    echo "$(date): Step 8 - Merging labels and predictions for $SPECIES"
    TEMP_DIR=$(mktemp -d)

    python scripts/gff.py set_source --source labels \
      --input "$SPECIES_DIR/gff/labels.gff" \
      --output "$TEMP_DIR/labels.gff"

    python scripts/gff.py set_source --source predictions \
      --input "$SPECIES_DIR/gff/predictions__strand_both__feat_len_2__gene_len_30__has_cds.gff" \
      --output "$TEMP_DIR/predictions.gff"

    python scripts/gff.py merge \
      --input "$TEMP_DIR/labels.gff" "$TEMP_DIR/predictions.gff" \
      --output "$SPECIES_DIR/gff/labeled_predictions__strand_both.gff"

    echo "$(date): Completed evaluation for $SPECIES"
done

echo "$(date): Evaluation $RUN_VERSION completed successfully!"
echo "Using predictions from: $PREDICT_DIR/{jregia,pvulgaris,carabica,zmays,ntabacum,nsylvestris}/runs/$PREDICTION_VERSION/$CHR_ID/predictions/"
echo "Results available in: $PREDICT_DIR/{jregia,pvulgaris,carabica,zmays,ntabacum,nsylvestris}/runs/$RUN_VERSION/$CHR_ID/results/"
