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
# Detect number of GPUs (fallback to 1 if nvidia-smi not available)
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi
echo "Detected $NUM_GPUS GPU(s). Using all for prediction."

# Use the venv's torchrun, NOT system torchrun.
# .venv/bin/torchrun automatically uses .venv/bin/python as the interpreter.
# This is the correct syntax: torchrun [opts] script.py [args]
GPU_LAUNCHER=".venv/bin/torchrun --nproc_per_node=$NUM_GPUS"

# Single-GPU launcher for CPU-only steps (extract, detect_intervals, export_gff)
CPU_LAUNCHER="uv run --extra torch python"

for CHR_ID in $CHROM_IDS; do
    echo "========================================"
    echo "Running Make pipeline for Chr: $CHR_ID"
    echo "========================================"
    
    # Run the Makefile for this specific chromosome
    # CPU-only steps (extract, detect_intervals, export_gff) use LAUNCHER.
    # GPU prediction step uses GPU_LAUNCHER (torchrun with all GPUs).
    INPUT_FILE="$INPUT_FILE" \
    OUTPUT_DIR="${OUTPUT_DIR}/${CHR_ID}" \
    SPECIES_ID="$SPECIES_ID" \
    CHR_ID="$CHR_ID" \
    BASE_MODEL_PATH="$BASE_MODEL" \
    HEAD_MODEL_PATH="$HEAD_MODEL" \
    PRED_BATCH_SIZE=32 \
    REQUIRE_UTRS="no" \
    LAUNCHER="$CPU_LAUNCHER" \
    GPU_LAUNCHER="$GPU_LAUNCHER" \
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
