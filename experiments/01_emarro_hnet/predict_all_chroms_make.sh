#!/bin/bash
set -e
cd /workdir/zl843/genecad || exit 1

INPUT_FILE="/workdir/zl843/GCA_978657495.1_TAIR12_genomic.fna"
OUTPUT_DIR="/workdir/zl843/genecad_result/Athaliana/prediction_genecad_emarro_hnet_5species"
SPECIES_ID="Athaliana"

BASE_MODEL="emarro/pcad2-200M-cnet-baseline"
HEAD_MODEL="plantcad/GeneCAD-pcad2-200M-cnet-baseline"

# Extract main numbered chromosomes from FASTA or GFF
if [[ "$INPUT_FILE" == *.gff* ]]; then
    CHROM_IDS=$(awk -F'\t' '!/^#/ {print $1}' "$INPUT_FILE" | sort -u | grep -iE '^(chr)?[0-9]+$')
else
    CHROM_IDS=$(grep "^>" "$INPUT_FILE" | sed 's/^>//' | awk '{print $1}' | head -50)

fi
for CHR_ID in $CHROM_IDS; do
    echo "========================================"
    echo "Running Make pipeline for Chr: $CHR_ID"
    echo "========================================"
    
    # Run the Makefile for this specific chromosome
    CUDA_VISIBLE_DEVICES=1 \
    INPUT_FILE="$INPUT_FILE" \
    OUTPUT_DIR="${OUTPUT_DIR}/${CHR_ID}" \
    SPECIES_ID="$SPECIES_ID" \
    CHR_ID="$CHR_ID" \
    BASE_MODEL_PATH="$BASE_MODEL" \
    HEAD_MODEL_PATH="$HEAD_MODEL" \
    PRED_BATCH_SIZE=32 \
    REQUIRE_UTRS="no" \
    LAUNCHER="uv run --extra torch python" \
    make -f pipelines/prediction all
    
    # Process the generated GFF for this chromosome:
    # 1. Skip the standard ##gff-version 3 header
    # 2. Modify ID and Parent attributes to include the chromosome ID
    # 3. Append to the merged sequence file
    awk -v chr="$CHR_ID" '
        /^#/ { next }
        {
            # Add chr prefix to ID=... and Parent=...
            gsub(/ID=/, "ID=" chr "_");
            gsub(/Parent=/, "Parent=" chr "_");
            print $0
        }
    ' "${OUTPUT_DIR}/${CHR_ID}/predictions.gff" >> "${OUTPUT_DIR}/merged_predictions.gff"
done

echo "Adding GFF header to the merged file..."
sed -i '1i ##gff-version 3' "${OUTPUT_DIR}/merged_predictions.gff"

echo "All chromosomes finished and merged to ${OUTPUT_DIR}/merged_predictions.gff!"
