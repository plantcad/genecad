#!/bin/bash
set -euo pipefail

# Set input/output locations
export INPUT_DIR=data
export OUTPUT_DIR=results
mkdir -p $INPUT_DIR $OUTPUT_DIR

# Define inputs for Arabidopsis chr 4 (smallest chromosome; ~18Mb)
GFF_URL=https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-61/gff3/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.61.chromosome.4.gff3.gz
GFF_GZ_FILE=$(basename $GFF_URL)
GFF_FILE=$(basename $GFF_URL .gz)

# Download GFF file
echo "Downloading GFF file from $GFF_URL to $INPUT_DIR/$GFF_GZ_FILE"
curl --output-dir $INPUT_DIR -O $GFF_URL
gzip -fd $INPUT_DIR/$GFF_GZ_FILE

# Run comparison
echo "Running evaluation for INPUT_FILE=$INPUT_DIR/$GFF_FILE"
gffcompare -r $INPUT_DIR/$GFF_FILE -C -o $OUTPUT_DIR/gffcompare $OUTPUT_DIR/predictions.gff

# Show results
echo "Evaluation statistics:"
cat $OUTPUT_DIR/gffcompare.stats
