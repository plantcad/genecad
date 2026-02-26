#!/bin/bash
set -euo pipefail

# Intergenic bias × REQUIRE_UTRS sweep for Zea mays chr1.
#
# Phase 1 and Phase 2 were run as ad-hoc steps (see below).
# Phase 3 is the sweep proper and is the executable part of this script.
#
# ---------------------------------------------------------------------------
# Phase 1: Data preparation (gg node, run ad-hoc)
# ---------------------------------------------------------------------------
# 1. Download maize B73 NAM-5.0 FASTA and GFF from NCBI RefSeq:
#      curl -L -o $SCRATCH/tmp/zmays/genome.fna.gz \
#        https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/902/167/145/GCF_902167145.1_Zm-B73-REFERENCE-NAM-5.0/GCF_902167145.1_Zm-B73-REFERENCE-NAM-5.0_genomic.fna.gz
#      curl -L -o $SCRATCH/tmp/zmays/genome.gff.gz \
#        https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/902/167/145/GCF_902167145.1_Zm-B73-REFERENCE-NAM-5.0/GCF_902167145.1_Zm-B73-REFERENCE-NAM-5.0_genomic.gff.gz
#
# 2. Identify chr1 accession (NC_050096.1), subset FASTA and GFF to chr1,
#    rename contig to "chr1":
#
#    FASTA (via Python script on login node):
#      python3 -c "
#      import gzip, os
#      data_dir = os.environ['SCRATCH'] + '/tmp/zmays'
#      writing = False
#      with gzip.open(f'{data_dir}/genome.fna.gz', 'rt') as fin, \
#           open(f'{data_dir}/chr1.fa', 'w') as fout:
#          for line in fin:
#              if line.startswith('>'):
#                  writing = line.split()[0] == '>NC_050096.1'
#                  if writing:
#                      fout.write('>chr1\n')
#                      continue
#              if writing:
#                  fout.write(line)
#      "
#
#    GFF (via awk):
#      zcat $SCRATCH/tmp/zmays/genome.gff.gz \
#        | awk -F"\t" -v acc="NC_050096.1" \
#          '/^#/ { print; next } $1 == acc { $1 = "chr1"; print }' OFS="\t" \
#        > $SCRATCH/tmp/zmays/chr1.gff3
#
#    Results:
#      -> $SCRATCH/tmp/zmays/chr1.fa   (308,452,471 bp)
#      -> $SCRATCH/tmp/zmays/chr1.gff3 (144,783 features)
#
# 3. Create sequences.zarr via Makefile:
#      make -f pipelines/prediction sequences \
#        INPUT_FILE=$SCRATCH/tmp/zmays/chr1.fa \
#        OUTPUT_DIR=$SCRATCH/tmp/zmays/output \
#        SPECIES_ID=zmays CHR_ID=chr1
#    (Run as sbatch on gg node due to memory requirements)
#
# ---------------------------------------------------------------------------
# Phase 2: GPU predictions (gh-dev, 8 nodes via srun)
# ---------------------------------------------------------------------------
# Submitted as:
#   cd $WORK/repos/genecad
#   PYTHONPATH=$WORK/repos/genecad:$PYTHONPATH \
#   srun -p gh-dev -N 8 -n 8 --tasks-per-node 1 -t 2:00:00 \
#     --output $SCRATCH/tmp/zmays/logs/phase2.log \
#     --error  $SCRATCH/tmp/zmays/logs/phase2.log \
#     bin/tacc python scripts/predict.py create_predictions \
#       --input $SCRATCH/tmp/zmays/output/pipeline/sequences.zarr \
#       --output-dir $SCRATCH/tmp/zmays/output/pipeline/predictions.zarr \
#       --model-path kuleshov-group/PlantCAD2-Small-l24-d0768 \
#       --model-checkpoint plantcad/GeneCAD-l8-d768-PC2-Small \
#       --species-id zmays --chromosome-id chr1 --batch-size 32 --dtype bfloat16
#
# bin/tacc sets RANK=$PMIX_RANK, WORLD_SIZE=$SLURM_NNODES so each node
# processes its shard. Produces predictions.{0..7}.zarr (~18 min total).
#
# ---------------------------------------------------------------------------
# Phase 3: Post-processing sweep (gg node) — this script
# ---------------------------------------------------------------------------
# Sweeps intergenic_bias × require_utrs using direct CLI calls (not the
# Makefile) to avoid the dependency chain rebuilding sequences/predictions.
#
# Prerequisites:
#   - predictions.zarr dir with predictions.{0..7}.zarr shards
#   - Reference GFF at $DATA_DIR/chr1.gff3
#   - gffcompare on PATH or at $GFFCOMPARE
#
# Usage:
#   OUTPUT_DIR=$SCRATCH/tmp/zmays/output \
#   REF_GFF=$SCRATCH/tmp/zmays/chr1.gff3 \
#   bash pipelines/experiments/intergenic_bias_sweep/zmays/sweep.sh

OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"
REF_GFF="${REF_GFF:?REF_GFF must be set}"
GFFCOMPARE="${GFFCOMPARE:-$WORK/repos/misc/gffcompare/gffcompare}"

PIPELINE_DIR="$OUTPUT_DIR/pipeline"
PREDICTIONS_ZARR="$PIPELINE_DIR/predictions.zarr"

if [ ! -d "$PREDICTIONS_ZARR" ]; then
    echo "ERROR: predictions.zarr not found at $PREDICTIONS_ZARR"
    echo "Run the GPU prediction step first."
    exit 1
fi

BIAS_VALUES="${BIAS_VALUES:-0.0 0.5 1.0 1.5 2.0 3.0 5.0 8.0}"
UTRS_VALUES="${UTRS_VALUES:-yes no}"

# Multi-node support: when launched via srun + bin/tacc, RANK and WORLD_SIZE
# are set automatically. Each rank processes a slice of the bias values.
# For single-node execution, defaults to processing all values.
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"

# Split bias values across ranks
ALL_BIAS=($BIAS_VALUES)
MY_BIAS=()
for i in "${!ALL_BIAS[@]}"; do
    if (( i % WORLD_SIZE == RANK )); then
        MY_BIAS+=("${ALL_BIAS[$i]}")
    fi
done
echo "[rank=$RANK/$WORLD_SIZE] Processing bias values: ${MY_BIAS[*]}"

STATS_OUT="$OUTPUT_DIR/stats"
mkdir -p "$STATS_OUT"

for BIAS in "${MY_BIAS[@]}"; do
    BIAS_DIR="$PIPELINE_DIR/sweep_bias_${BIAS}"
    mkdir -p "$BIAS_DIR"

    # Steps 1-2 depend only on INTERGENIC_BIAS, run once per bias value
    echo "========================================="
    echo "Detecting intervals for bias=$BIAS"
    echo "========================================="

    # Step 1: Detect intervals
    python scripts/predict.py detect_intervals \
        --input-dir "$PREDICTIONS_ZARR" \
        --output "$BIAS_DIR/intervals.zarr" \
        --decoding-methods "direct,viterbi" \
        --remove-incomplete-features yes \
        --intergenic-bias "$BIAS"

    # Step 2: Export raw GFF
    python scripts/predict.py export_gff \
        --input "$BIAS_DIR/intervals.zarr" \
        --output "$BIAS_DIR/predictions__raw.gff" \
        --decoding-method viterbi \
        --min-transcript-length 3 \
        --strip-introns yes

    # Step 3: Filter small features (also UTR-independent)
    python scripts/gff.py filter_to_min_feature_length \
        --input "$BIAS_DIR/predictions__raw.gff" \
        --output "$BIAS_DIR/predictions__feat_len_2.gff" \
        --feature-types "five_prime_UTR,three_prime_UTR,CDS" \
        --min-length 2

    # Step 4: Filter short genes (also UTR-independent)
    python scripts/gff.py filter_to_min_gene_length \
        --input "$BIAS_DIR/predictions__feat_len_2.gff" \
        --output "$BIAS_DIR/predictions__gene_len_30.gff" \
        --min-length 30

    # Steps 5-6 depend on REQUIRE_UTRS, loop over UTR settings
    for UTRS in $UTRS_VALUES; do
        echo "--- bias=$BIAS, require_utrs=$UTRS ---"

        UTRS_DIR="$BIAS_DIR/utrs_${UTRS}"
        mkdir -p "$UTRS_DIR"

        # Step 5: Filter to valid genes
        python scripts/gff.py filter_to_valid_genes \
            --input "$BIAS_DIR/predictions__gene_len_30.gff" \
            --output "$UTRS_DIR/predictions.gff" \
            --require-utrs "$UTRS"

        # Step 6: Run gffcompare
        "$GFFCOMPARE" \
            -r "$REF_GFF" \
            -C -o "$UTRS_DIR/gffcompare" \
            "$UTRS_DIR/predictions.gff"

        echo "--- gffcompare stats for bias=$BIAS, utrs=$UTRS ---"
        cat "$UTRS_DIR/gffcompare.stats"
        echo ""

        # Copy stats file with descriptive name
        cp "$UTRS_DIR/gffcompare.stats" \
           "$STATS_OUT/gffcompare_bias_${BIAS}_utrs_${UTRS}.stats"
    done
done

echo ""
echo "========================================="
echo "[rank=$RANK/$WORLD_SIZE] Sweep complete. Stats files in: $STATS_OUT"
echo "========================================="
ls -la "$STATS_OUT"
