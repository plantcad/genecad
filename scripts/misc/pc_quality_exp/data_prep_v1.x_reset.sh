#!/bin/bash

# PC Quality Filter Experiment - Data Preparation Reset Script
# Moves all v1.x directories to trash before rerunning data preparation
# Run this before executing data_prep_v1.0.sh, data_prep_v1.1.sh, and data_prep_v1.2.sh

set -euo pipefail

echo "PC Quality Filter Experiment - Data Preparation Reset"
echo "$(date): Starting cleanup of v1.x data directories"

# Define paths
PIPE_DIR="/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline"
TRASH_DIR="/scratch/10459/eczech/trash"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create trash directory if it doesn't exist
mkdir -p "$TRASH_DIR"

echo "$(date): Moving v1.x directories to trash..."

# Clean up extract directories
for version in v1.0 v1.1 v1.2; do
    if [ -d "$PIPE_DIR/extract/$version" ]; then
        echo "Moving extract/$version to trash..."
        mv "$PIPE_DIR/extract/$version" "$TRASH_DIR/extract_${version}_${TIMESTAMP}"
    fi
done

# Clean up transform directories  
for version in v1.0 v1.1 v1.2; do
    if [ -d "$PIPE_DIR/transform/$version" ]; then
        echo "Moving transform/$version to trash..."
        mv "$PIPE_DIR/transform/$version" "$TRASH_DIR/transform_${version}_${TIMESTAMP}"
    fi
done

# Clean up prep directories
for version in v1.0 v1.1 v1.2; do
    if [ -d "$PIPE_DIR/prep/$version" ]; then
        echo "Moving prep/$version to trash..."
        mv "$PIPE_DIR/prep/$version" "$TRASH_DIR/prep_${version}_${TIMESTAMP}"
    fi
done

echo "$(date): Cleanup completed successfully!"
echo "All v1.x directories moved to trash with timestamp: $TIMESTAMP"
echo ""
echo "You can now run the data preparation scripts:"
echo "  ./scripts/misc/pc_quality_exp/data_prep_v1.0.sh"
echo "  ./scripts/misc/pc_quality_exp/data_prep_v1.1.sh" 
echo "  ./scripts/misc/pc_quality_exp/data_prep_v1.2.sh" 