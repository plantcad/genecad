import sys
import argparse
import pathlib

# Ensure we can import from src
# (This adds the project root to python path if running from scripts/)
current_dir = pathlib.Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src import reelprotein  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="ReelGene Pipeline: Merge, Embed, Score, and Filter."
    )
    parser.add_argument("--gff", required=True, help="Input GFF file")
    parser.add_argument("--genome", required=True, help="Genome FASTA file")
    parser.add_argument("--out", required=True, help="Final output GFF file")

    args = parser.parse_args()

    # Resolve model directory relative to the project root
    # assuming src/reelprotein_models/ exists
    models_dir = project_root / "src" / "reelprotein_models"

    if not models_dir.exists():
        print(f"[Error] Model directory not found at: {models_dir}")
        sys.exit(1)

    print(f"[Config] Model directory resolved to: {models_dir}")

    # --- EXECUTION FLOW ---

    # 1. Parse GFF and Genome
    # (Note: In the future, check if you can use src.gff_parser instead of reelprotein.parse_gff3)
    genes_data = reelprotein.parse_gff3(args.gff)
    protein_candidates = reelprotein.extract_candidate_proteins(genes_data, args.genome)

    if not protein_candidates:
        print("No protein candidates found. Exiting.")
        sys.exit(0)

    # 2. Generate Embeddings
    embeddings_df = reelprotein.generate_embeddings(protein_candidates)

    # 3. Score Proteins
    scored_df = reelprotein.score_proteins(embeddings_df, str(models_dir))

    # 4. Generate Final GFF
    reelprotein.generate_final_gff(scored_df, args.gff, args.out)

    print("Pipeline Finished Successfully.")


if __name__ == "__main__":
    main()
