import argparse
import logging
import os
from pathlib import Path

from zarr.errors import GroupNotFoundError

from src.dataset import open_datatree
import json

logger = logging.getLogger(__name__)


def build_json(
    species_id: str, input_zarr: str, intermediate_dir: str, output_json: str
):
    """
    Build manifest.json pipeline

    Parameters
    ----------
    species_id
        Species/sample name
    input_zarr
        path to sequences zarr file
    intermediate_dir
        path to directory where intermediate files will be stored
    output_json
        output json file name
    """
    try:
        logger.info(f"Opening input sequence datatree from {input_zarr}")
        sequences = open_datatree(input_zarr, consolidated=False)
        logger.info(f"Input sequences:\n{sequences}")
    except GroupNotFoundError as e:
        print(e)
        logger.error(
            f"File {input_zarr} is not formatted as an input sequence datatree. \n"
            f"Check that it is the output of the command extract_train.py extract_fasta_file"
        )

    # Check if species exists
    if species_id not in sequences:
        available_species = list(sequences.keys())
        if available_species == ["sequences", "intervals"]:
            raise ValueError(
                f"{input_zarr} is an intervals file, not a sequence file. \n"
                f"Check that it is the output of the command extract_train.py extract_fasta_file"
            )
        else:
            raise ValueError(
                f"Species '{species_id}' not found in input data. Available species: {available_species}"
            )

    available_chromosomes = list(sequences[species_id].keys())

    manifest = []

    for chromosome_id in available_chromosomes:
        chrom_dict = {}

        os.makedirs(f"{intermediate_dir}/{chromosome_id}", exist_ok=True)

        chrom_dict["chromosome_id"] = chromosome_id
        chrom_dict["sequence_zarr"] = input_zarr
        chrom_dict["predictions_dir"] = (
            f"{intermediate_dir}/{chromosome_id}/predictions_{chromosome_id}"
        )
        chrom_dict["intervals_zarr"] = (
            f"{intermediate_dir}/{chromosome_id}/intervals_{chromosome_id}.zarr"
        )
        chrom_dict["raw_gff"] = (
            f"{intermediate_dir}/{chromosome_id}/predictions_raw_{chromosome_id}.gff"
        )
        chrom_dict["filtered_gff"] = (
            f"{intermediate_dir}/{chromosome_id}/predictions_filtered_{chromosome_id}.gff"
        )

        manifest.append(chrom_dict)

    with open(output_json, "w") as file:
        json.dump(manifest, file)


def main():
    """Create a manifest json file for multi-chromosome pipeline runs."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Suppress noisy HTTP traffic logs from HuggingFace Hub's internal HTTP client
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Creates a manifest json file for use in multi-chromosome pipeline runs."
    )

    parser.add_argument(
        "--input-zarr",
        "-i",
        type=str,
        required=True,
        help="Sequences .zarr file created by extract_train.py",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        type=str,
        required=True,
        help="Path to output manifest json file",
    )
    parser.add_argument(
        "--species-id", type=str, required=True, help="Species/sample name"
    )

    parser.add_argument(
        "--intermediate-dir",
        type=str,
        default=None,
        help="directory in which to store intermediate files. If not set, will be same as --input-zarr parent directory",
    )

    args = parser.parse_args()

    if args.intermediate_dir is None:
        intermediate_dir = str(Path(args.input_zarr).parent)
    else:
        intermediate_dir = args.intermediate_dir

    build_json(
        species_id=args.species_id,
        input_zarr=args.input_zarr,
        intermediate_dir=intermediate_dir,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
