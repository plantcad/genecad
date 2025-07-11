#!/bin/bash

# PC Quality Filter Experiment - Evaluation v1.2
# Model: v1.2 (All 5 species from v1.1 checkpoint)
# Species: jregia, pvulgaris
# Usage: ./eval_v1.2.sh {original|pc-filtered}

set -euo pipefail

# Parse arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 {original|pc-filtered}"
    echo "  original     - Use original (unfiltered) ground truth (v1.2.0)"
    echo "  pc-filtered  - Use PC-filtered ground truth (v1.2.1)"
    exit 1
fi

GROUND_TRUTH_TYPE="$1"

# Validate argument
if [ "$GROUND_TRUTH_TYPE" != "original" ] && [ "$GROUND_TRUTH_TYPE" != "pc-filtered" ]; then
    echo "ERROR: Invalid ground truth type '$GROUND_TRUTH_TYPE'"
    echo "Must be 'original' or 'pc-filtered'"
    exit 1
fi

# Set version and description based on ground truth type
if [ "$GROUND_TRUTH_TYPE" = "original" ]; then
    RUN_VERSION="v1.2.0"
    GT_DESCRIPTION="Original (unfiltered)"
else
    RUN_VERSION="v1.2.1"
    GT_DESCRIPTION="PC-filtered (passPlantCADFilter=1 only)"
fi

echo "Starting PC Quality Filter Experiment - Evaluation $RUN_VERSION"
echo "Model: v1.2 (All 5 species from v1.1 checkpoint)"
echo "Ground Truth: $GT_DESCRIPTION"
echo "Species: jregia, pvulgaris"
echo "$(date): Beginning evaluation"

# Define paths
RAW_DIR="/work/10459/eczech/vista/data/dna/plant_caduceus_genome_annotation_task/data_share_20250326/testing_data"
PIPE_DIR="/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline"
PREDICT_DIR="$PIPE_DIR/predict"
MODEL_PATH="kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000"
MODEL_CHECKPOINT="$PIPE_DIR/sweep/sweep-v1.2__cfg_013__arch_all__frzn_yes__lr_1e-04/pc-genome-annot/v1.2/checkpoints/last.ckpt"

# Define species to evaluate
SPECIES_LIST="jregia pvulgaris"
CHR_ID="chr1"

echo "Model checkpoint: $MODEL_CHECKPOINT"
echo "Output directory: $PREDICT_DIR"

# Verify checkpoint exists
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "ERROR: Model checkpoint not found at $MODEL_CHECKPOINT"
    echo "Please ensure v1.2 training has completed successfully"
    exit 1
fi

# Process each species
for SPECIES in $SPECIES_LIST; do
    echo "$(date): ========================================="
    echo "$(date): Processing species: $SPECIES"
    echo "$(date): ========================================="
    
    # Convert to proper species ID (capitalize first letter)
    SPECIES_ID=$(echo "$SPECIES" | sed 's/./\u&/')
    SPECIES_DIR="$PREDICT_DIR/$SPECIES/runs/$RUN_VERSION/$CHR_ID"
    
    echo "Species ID: $SPECIES_ID"
    echo "Species directory: $SPECIES_DIR"
    
    # Create output directories
    mkdir -p "$SPECIES_DIR/gff" "$SPECIES_DIR/results" "$SPECIES_DIR/predictions"
    
    # Step 1: Extract sequences (if needed)
    echo "$(date): Step 1 - Extracting sequences for $SPECIES"
    if [ ! -d "$PREDICT_DIR/$SPECIES/sequences.zarr" ]; then
        echo "Extracting FASTA sequences for $SPECIES..."
        python scripts/extract.py extract_fasta_sequences \
          --input-dir "$RAW_DIR/fasta" \
          --species-id "$SPECIES_ID" \
          --tokenizer-path "$MODEL_PATH" \
          --output "$PREDICT_DIR/$SPECIES/sequences.zarr"
    else
        echo "Sequences already exist for $SPECIES, skipping extraction"
    fi
    
    # Step 2: Generate predictions
    echo "$(date): Step 2 - Generating predictions for $SPECIES"
    srun bin/tacc \
    python scripts/predict.py create_predictions \
      --input "$PREDICT_DIR/$SPECIES/sequences.zarr" \
      --output-dir "$SPECIES_DIR/predictions" \
      --model-path "$MODEL_PATH" \
      --model-checkpoint "$MODEL_CHECKPOINT" \
      --species-id "$SPECIES_ID" \
      --chromosome-id "$CHR_ID" \
      --batch-size 32
      
    # Step 3: Detect intervals and export GFF
    echo "$(date): Step 3 - Detecting intervals for $SPECIES"
    python scripts/predict.py detect_intervals \
      --input-dir "$SPECIES_DIR/predictions" \
      --output "$SPECIES_DIR/intervals.zarr" \
      --decoding-methods "direct,viterbi" \
      --remove-incomplete-features yes \
      --duration-probs-path local/data/feature_length_distributions.parquet \
      --num-workers 64
      
    echo "$(date): Step 4 - Exporting predictions to GFF for $SPECIES"
    python scripts/predict.py export_gff \
      --input "$SPECIES_DIR/intervals.zarr" \
      --output "$SPECIES_DIR/gff/predictions.gff" \
      --decoding-method viterbi \
      --min-transcript-length 3 \
      --strip-introns yes
      
    # Step 5: Process predictions
    echo "$(date): Step 5 - Processing predictions for $SPECIES"
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
      
    # Step 6: Process labels (conditional on ground truth type)
    echo "$(date): Step 6 - Processing labels ($GT_DESCRIPTION) for $SPECIES"
    python scripts/gff.py resolve \
      --input-dir "$RAW_DIR/gff" \
      --species-id "$SPECIES_ID" \
      --output "$SPECIES_DIR/gff/labels_raw.gff"
      
    if [ "$GROUND_TRUTH_TYPE" = "original" ]; then
        # Original ground truth - no PC filtering
        python scripts/gff.py filter_to_chromosome \
          --input "$SPECIES_DIR/gff/labels_raw.gff" \
          --output "$SPECIES_DIR/gff/labels.gff" \
          --chromosome-id "$CHR_ID" \
          --species-id "$SPECIES_ID"
    else
        # PC-filtered ground truth
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
      
    # Step 7: Run comparisons (gffcompare)
    echo "$(date): Step 7 - Running gffcompare evaluations for $SPECIES"
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
      
    # Step 8: Run comparisons (gffeval)
    echo "$(date): Step 8 - Running gffeval evaluations for $SPECIES"
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
      
    # Step 9: Consolidate results
    echo "$(date): Step 9 - Consolidating results for $SPECIES"
    python scripts/gff.py collect_results \
      --input "$SPECIES_DIR/results" \
      --output "$SPECIES_DIR/results/gffcompare.stats.consolidated.csv"
      
    echo "$(date): Completed evaluation for $SPECIES"
done

echo "$(date): Evaluation $RUN_VERSION completed successfully!"
echo "Results available in: $PREDICT_DIR/{jregia,pvulgaris}/runs/$RUN_VERSION/$CHR_ID/results/" 