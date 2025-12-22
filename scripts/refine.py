#!/usr/bin/env python3
import sys
import argparse
import pathlib
import logging

# Ensure we can import from src
current_dir = pathlib.Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src import reelprotein  # noqa: E402

# Initialize module logger
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="ReelProtein Pipeline: Merge, Embed, Score, and Filter."
    )
    parser.add_argument("--gff", required=True, help="Input GFF file")
    parser.add_argument("--genome", required=True, help="Genome FASTA file")
    parser.add_argument("--out", required=True, help="Final output GFF file")
    parser.add_argument(
        "--model-repo",
        default="plantcad/reelprotein",
        help="Hugging Face Repo ID for models",
    )

    args = parser.parse_args()

    # --- CONFIGURATION ---
    model_source = args.model_repo
    logger.info(f"[Config] Using Hugging Face model repository: {model_source}")

    # --- EXECUTION FLOW ---
    # 1. Parse GFF and Genome
    genes_data = reelprotein.parse_gff3(args.gff)
    protein_candidates = reelprotein.extract_candidate_proteins(genes_data, args.genome)

    if not protein_candidates:
        logger.warning("No protein candidates found. Exiting.")
        return

    # 2. Generate Embeddings
    embeddings_df = reelprotein.generate_embeddings(protein_candidates)

    # 3. Score Proteins
    scored_df = reelprotein.score_proteins(embeddings_df, model_source)

    # 4. Generate Final GFF
    reelprotein.generate_final_gff(scored_df, args.gff, args.out)

    logger.info("Pipeline Finished Successfully.")


if __name__ == "__main__":
    # Configure logging only when run as a script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()
