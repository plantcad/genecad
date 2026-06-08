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


def run_reelprotein(
    input_gff: str,
    input_fasta: str,
    output_gff: str,
    model_source: str,
    filter_unmerged: bool,
    gpus: list[int],
):
    """
    Run ReelProtein refinement pipeline

    Parameters
    ----------
    input_gff
        Species/sample name
    input_fasta
        path to sequences zarr file
    output_gff
        path to directory where intermediate files will be stored
    model_source
        output json file name
    filter_unmerged
        remove unmerged genes that #TODO
    gpus
        list of gpu indices to use for ProtT5 embeddings
    """
    # --- CONFIGURATION ---
    logger.info(f"[Config] Using Hugging Face model repository: {model_source}")

    # --- EXECUTION FLOW ---
    # 1. Parse GFF and Genome
    genes_data = reelprotein.parse_gff3(input_gff)
    protein_candidates = reelprotein.extract_candidate_proteins(genes_data, input_fasta)

    if not protein_candidates:
        logger.warning("No protein candidates found. Exiting.")
        return

    # 2. Generate Embeddings
    embeddings_df = reelprotein.generate_embeddings(protein_candidates, gpus=gpus)

    # 3. Score Proteins
    scored_df = reelprotein.score_proteins(embeddings_df, model_source)

    # 4. Generate Final GFF
    reelprotein.generate_final_gff(
        scored_df, input_gff, output_gff, keep_unmerged=(not filter_unmerged)
    )

    logger.info("Pipeline Finished Successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="ReelProtein Pipeline: Merge, Embed, Score, and Filter."
    )
    parser.add_argument("--input-gff", "-i", required=True, help="Input GFF file")
    parser.add_argument("--input-fasta", "-f", required=True, help="Genome FASTA file")
    parser.add_argument(
        "--output-gff", "-o", required=True, help="Final output GFF file"
    )
    parser.add_argument(
        "--reelprotein-model-path",
        default="plantcad/reelprotein",
        help="Hugging Face Repo ID for models",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU IDs for ProtT5 embedding (e.g. '0,1,2'). Default: '0'.",
    )

    args = parser.parse_args()

    gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]

    run_reelprotein(
        input_gff=args.input_gff,
        input_fasta=args.input_fasta,
        output_gff=args.output_gff,
        model_source=args.reelprotein_model_path,
        filter_unmerged=False,
        gpus=gpus,
    )


if __name__ == "__main__":
    # Configure logging only when run as a script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()
