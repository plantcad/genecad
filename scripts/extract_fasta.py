import json
import os
import gzip
import argparse
import logging
import numpy.typing as npt
from typing import Any, Callable

from huggingface_hub import hf_hub_download

import xarray as xr
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature
from src.gff_parser import parse as parse_gff
from src.dataset import (
    DEFAULT_SEQUENCE_CHUNK_SIZE,
    info_str,
    open_datatree,
    set_dimension_chunks,
)
from src.schema import GffFeatureType, SequenceFeature, PositionInfo
from src.config import SpeciesConfig, get_species_configs

logger = logging.getLogger(__name__)

Tokenizer = Callable[[npt.NDArray[np.str_]], npt.NDArray[np.int_]]

# -------------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------------


def _resolve_file_path(path: str) -> str:
    """Resolve file path by falling back to gzip/un-gzipped versions if needed."""
    if os.path.exists(path):
        return path
    if path.endswith(".gz") and os.path.exists(path[:-3]):
        return path[:-3]
    if not path.endswith(".gz") and os.path.exists(path + ".gz"):
        return path + ".gz"
    return path


def get_feature_name(feature: SeqFeature) -> str | None:
    """Extract name from a SeqFeature object."""
    if not hasattr(feature, "qualifiers"):
        return None

    qualifiers = feature.qualifiers
    for key in ["Name", "geneName"]:
        if key in qualifiers and qualifiers[key]:
            return qualifiers[key][0]
    return None


def get_feature_id(feature: SeqFeature) -> str | None:
    """Extract ID from a SeqFeature object."""
    return feature.id if hasattr(feature, "id") else None


def get_position_info(feature: SeqFeature) -> PositionInfo:
    """Extract position information from a SeqFeature"""
    return PositionInfo(
        strand=1 if feature.location.strand > 0 else -1,
        start=int(feature.location.start.real),
        stop=int(feature.location.end.real),
    )


def is_longest_transcript(feature: SeqFeature) -> bool:
    """Check if a transcript is the longest (has longest=1)."""
    if hasattr(feature, "qualifiers") and "longest" in feature.qualifiers:
        return feature.qualifiers["longest"][0] == "1"
    return False


# -------------------------------------------------------------------------------------------------
# Extract GFF features
# -------------------------------------------------------------------------------------------------


def extract_gff_features(
    input_dir: str,
    species_ids: list[str],
    output_path: str,
    skip_exon_features: bool = True,
) -> None:
    """Extract data from GFF file(s) into a structured DataFrame.

    Note that position information is inclusive for start positions and exclusive for stop positions.

    Parameters
    ----------
    input_dir : str
        Directory containing input GFF files
    species_ids : list[str]
        List of species IDs to process
    output_path : str
        Path to output parquet file
    skip_exon_features : bool, optional
        Whether to skip exon features during extraction (default: True).
        Exon features are structural and not used for sequence modeling.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get configs for all specified species
    species_configs = get_species_configs(species_ids)

    # Extract data from all input files and concatenate
    dfs = []
    for config in species_configs:
        input_path = os.path.join(input_dir, config.gff.filename)
        input_path = _resolve_file_path(input_path)
        logger.info(f"Processing file for {config.name} ({config.id}): {input_path}")
        df = _extract_gff_features(input_path, config, skip_exon_features)
        dfs.append(df)
        logger.info(f"Extracted {df.shape[0]} rows from {input_path}")

    # Combine and validate all dataframes
    if dfs:
        df = pd.concat(dfs, ignore_index=True, axis=0)
        del dfs
        logger.info(
            f"Saving combined features with {df.shape[0]} rows to {output_path}"
        )
        with pd.option_context(
            "display.max_info_columns", 1_000, "display.max_info_rows", int(1e8)
        ):
            logger.info(f"Features info:\n{df.pipe(info_str)}")
        df.to_parquet(output_path, index=False)
        del df
        logger.info("Validating extracted features")
        validate_gff_features(pd.read_parquet(output_path))
        logger.info("Validation complete")
    else:
        logger.warning("No data was extracted from the input files")


def _extract_gff_features(
    path: str, species_config: SpeciesConfig, skip_exon_features: bool = True
) -> pd.DataFrame:
    """Extract data from a single GFF file into a structured DataFrame."""
    logger.info(f"Parsing GFF file: {path}")

    # Determine file opening method based on extension of the real file
    real_path = os.path.realpath(path)
    open_func = gzip.open if real_path.endswith(".gz") else open
    mode = "rt" if real_path.endswith(".gz") else "r"

    # Parse GFF file
    with open_func(path, mode) as in_handle:
        records = list(parse_gff(in_handle))

    logger.info(
        f"[species={species_config.id}] Found {len(records)} chromosome records"
    )
    features_data: list[SequenceFeature] = []
    filename = os.path.basename(path)

    # Get chromosome mapping from the species config
    chrom_map = species_config.chromosome_map
    logger.info(
        f"[species={species_config.id}] Valid chromosomes: {list(chrom_map.keys())}"
    )

    # Process each chromosome
    for chrom in records:
        raw_id = chrom.id

        # Skip chromosomes not in the species config
        if raw_id not in chrom_map:
            logger.debug(
                f"[species={species_config.id}] Skipping unmapped chromosome record: {raw_id}"
            )
            continue

        chrom_id = chrom_map[raw_id]
        chrom_name = chrom.name
        chrom_length = len(chrom.seq)
        species_id = species_config.id
        species_name = species_config.name

        logger.info(
            f"[species={species_config.id}] Processing chromosome: {chrom_id} (from {raw_id}) with {len(chrom.features)} features"
        )

        # Process each gene
        for gene in chrom.features:
            if gene.type != GffFeatureType.GENE:
                raise ValueError(
                    f"Found unexpected chromosome feature type: {gene.type}"
                )

            gene_id = get_feature_id(gene)
            gene_info = get_position_info(gene)
            gene_name = get_feature_name(gene)

            # Process each transcript
            for transcript in gene.sub_features:
                if transcript.type != GffFeatureType.MRNA:
                    raise ValueError(
                        f"Found unexpected gene feature type: {transcript.type}"
                    )

                transcript_id = get_feature_id(transcript)
                transcript_info = get_position_info(transcript)
                transcript_name = get_feature_name(transcript)
                # For now, we use the longest transcript as the canonical transcript
                transcript_is_canonical = is_longest_transcript(transcript)

                # Process each feature
                for feature in transcript.sub_features:
                    # Skip exon features if requested (they are structural, not modeling features)
                    if skip_exon_features and feature.type == "exon":
                        continue

                    if feature.type not in [
                        GffFeatureType.FIVE_PRIME_UTR,
                        GffFeatureType.CDS,
                        GffFeatureType.THREE_PRIME_UTR,
                    ]:
                        raise ValueError(
                            f"Found unexpected transcript feature type: {feature.type}"
                        )

                    feature_id = get_feature_id(feature)
                    feature_info = get_position_info(feature)
                    feature_name = get_feature_name(feature)

                    # Create a validated GeneFeatureData object
                    feature_data = SequenceFeature(
                        species_id=species_id,
                        species_name=species_name,
                        chromosome_id=chrom_id,
                        chromosome_name=chrom_name,
                        chromosome_length=chrom_length,
                        gene_id=gene_id,
                        gene_name=gene_name,
                        gene_strand=gene_info.strand,
                        gene_start=gene_info.start,
                        gene_stop=gene_info.stop,
                        transcript_id=transcript_id,
                        transcript_name=transcript_name,
                        transcript_strand=transcript_info.strand,
                        transcript_is_canonical=transcript_is_canonical,
                        transcript_start=transcript_info.start,
                        transcript_stop=transcript_info.stop,
                        feature_id=feature_id,
                        feature_name=feature_name,
                        feature_strand=feature_info.strand,
                        feature_type=feature.type,
                        feature_start=feature_info.start,
                        feature_stop=feature_info.stop,
                        filename=filename,
                    )
                    features_data.append(feature_data)

    logger.info(
        f"[species={species_config.id}] Total features extracted: {len(features_data)}"
    )

    # Convert list of Pydantic models to DataFrame
    df = pd.DataFrame([feature.model_dump() for feature in features_data])
    return df


def validate_gff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the extracted GFF data."""

    # Check for primary key uniqueness
    logger.info("Checking for primary key uniqueness")
    primary_key = [
        "species_id",
        "chromosome_id",
        "gene_id",
        "transcript_id",
        "feature_id",
    ]
    duplicated_keys = df[primary_key].value_counts().pipe(lambda x: x[x > 1])
    if len(duplicated_keys) > 0:
        raise ValueError(
            f"Found {len(duplicated_keys)} duplicate rows for {primary_key=}; "
            f"Examples:\n{duplicated_keys.head()}"
        )

    # Check that at least one transcript is canonical for each species
    logger.info("Checking for existence of canonical transcripts per species")
    if not df.empty:
        # Compute canonical/non-canonical counts by species
        canonical_counts = (
            df.groupby(["species_id"])["transcript_is_canonical"]
            .value_counts()
            .unstack(fill_value=0)
            .rename(columns={True: "canonical", False: "non_canonical"})
        )

        # Check for species with no canonical transcripts
        species_without_canonical = canonical_counts[
            canonical_counts.get("canonical", 0) == 0
        ]

        if len(species_without_canonical) > 0:
            raise ValueError(
                f"Found {len(species_without_canonical)} species with no canonical transcripts. "
                f"Canonical/non-canonical counts by species:\n{canonical_counts}\n"
                f"Species without canonical transcripts: {list(species_without_canonical.index)}"
            )

        logger.info(f"Canonical transcript counts by species:\n{canonical_counts}")

    # Check canonical transcript indicator consistency
    logger.info("Checking for canonical transcript indicator consistency")
    bad_transcripts = (
        df.groupby(["species_id", "chromosome_id", "gene_id", "transcript_id"])[
            "transcript_is_canonical"
        ]
        .nunique()
        .pipe(lambda x: x[x > 1])
    )
    if len(bad_transcripts) > 0:
        raise ValueError(
            f"Found {len(bad_transcripts)} transcripts with multiple canonical status indicators; "
            f"Examples:\n{bad_transcripts.head()}"
        )

    # Check for strand consistency
    logger.info("Checking for strand consistency")
    strand_combinations = set(
        df[["gene_strand", "transcript_strand", "feature_strand"]]
        .drop_duplicates()
        .apply(tuple, axis=1)
        .sort_values()
        .to_list()
    )
    if invalid_combinations := strand_combinations - {(-1, -1, -1), (1, 1, 1)}:
        invalid_examples = df[
            df[["gene_strand", "transcript_strand", "feature_strand"]]
            .apply(tuple, axis=1)
            .isin(invalid_combinations)
        ]
        raise ValueError(
            f"Strands are not consistent across features; found invalid combinations: {invalid_combinations}.\n"
            f"Examples:\n{invalid_examples[['gene_strand', 'transcript_strand', 'feature_strand']].head()}"
        )

    # Check for ids with multiple strand, start, or stop values
    logger.info("Checking for ids with multiple strand, start, or stop values")
    for prefix in ["gene", "transcript", "feature"]:
        for col in ["strand", "start", "stop"]:
            logger.info(f"    Checking {prefix}s with differing {col!r} values")
            inconsistencies = (
                df.groupby(["species_id", "chromosome_id", f"{prefix}_id"])[
                    f"{prefix}_{col}"
                ]
                .unique()
                .pipe(lambda x: x[x.apply(len) > 1])
            )
            if len(inconsistencies) > 0:
                raise ValueError(
                    f"Found {len(inconsistencies)} {prefix}s with differing {col!r} values for the same ID; "
                    f"Examples:\n{inconsistencies.head()}"
                )

    # Check for genes extending beyond chromosome boundaries
    logger.info("Checking for genes extending beyond chromosome boundaries")
    invalid_genes = df[
        (df["gene_start"] < 0) | (df["gene_stop"] > df["chromosome_length"])
    ]
    if len(invalid_genes) > 0:
        invalid_examples = invalid_genes.groupby(["chromosome_id", "gene_id"]).first()
        raise ValueError(
            f"Found {len(invalid_examples)} genes that extend beyond their chromosome boundaries. "
            f"Examples:\n{invalid_examples[['chromosome_length', 'gene_start', 'gene_stop']].head()}"
        )

    # Check for transcripts extending beyond gene boundaries
    logger.info("Checking for transcripts extending beyond gene boundaries")
    invalid_transcripts = df[
        (df["transcript_start"] < df["gene_start"])
        | (df["transcript_stop"] > df["gene_stop"])
    ]
    if len(invalid_transcripts) > 0:
        invalid_examples = invalid_transcripts.groupby(
            ["gene_id", "transcript_id"]
        ).first()
        raise ValueError(
            f"Found {len(invalid_examples)} transcripts that extend beyond their gene boundaries. "
            f"Examples:\n{invalid_examples[['gene_start', 'gene_stop', 'transcript_start', 'transcript_stop']].head()}"
        )

    # Check for features extending beyond transcript boundaries
    logger.info("Checking for features extending beyond transcript boundaries")
    invalid_features = df[
        (df["feature_start"] < df["transcript_start"])
        | (df["feature_stop"] > df["transcript_stop"])
    ]
    if len(invalid_features) > 0:
        invalid_examples = invalid_features.groupby(
            ["transcript_id", "feature_id"]
        ).first()
        raise ValueError(
            f"Found {len(invalid_examples)} features that extend beyond their transcript boundaries. "
            f"Examples:\n{invalid_examples[['transcript_start', 'transcript_stop', 'feature_start', 'feature_stop']].head()}"
        )

    return df


# -------------------------------------------------------------------------------------------------
# Extract FASTA sequences
# -------------------------------------------------------------------------------------------------

# Complete set of IUPAC ambiguity codes (both uppercase and lowercase)
# - A,C,G,T are standard bases; others represent ambiguities
# - Generated by: `from Bio.Data import IUPACData; IUPACData.ambiguous_dna_values`
FASTA_OOV_TOKENS = "MRWSYKVHDBXN"


def extract_fasta_file(
    species_id: str,
    fasta_file: str,
    chrom_map_str: str | None,
    output_path: str,
    chunk_size: int = DEFAULT_SEQUENCE_CHUNK_SIZE,
    tokenizer_path: str | None = None,
) -> None:
    """Extract sequences from a single FASTA file into a structured format.

    Parameters
    ----------
    species_id : str
        Species ID to process
    fasta_file : str
        Path to input FASTA file
    chrom_map_str : str
        Chromosome mapping as 'src1:dst1,src2:dst2,...'
    output_path : str
        Path to output file
    chunk_size : int, optional
        Size of chunks to write to Zarr
    tokenizer_path : str, optional
        Path to tokenizer if tokenizing sequences
    """
    # Parse chromosome map from string
    if chrom_map_str is not None:
        chrom_map = {}
        for mapping in chrom_map_str.split(","):
            src, dst = mapping.split(":")
            chrom_map[src.strip()] = dst.strip()
        logger.info(f"Parsed chromosome mapping: {chrom_map}")
    else:
        logger.info(
            "No chromosome mapping provided. All chromosomes will retain their original names."
        )

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load tokenizer if path provided
    tokenizer = None
    if tokenizer_path:
        tokenizer = _load_tokenizer(tokenizer_path)

    # Extract sequences
    _extract_fasta_sequences(
        species_id=species_id,
        fasta_file=fasta_file,
        chrom_map=chrom_map,
        output_path=output_path,
        chunk_size=chunk_size,
        tokenizer=tokenizer,
    )

    dt = open_datatree(output_path)
    logger.info(f"Final data tree:\n{dt}")
    logger.info("Done")


# TODO: duplicated - move to dedicated utils file in src
def _load_tokenizer(tokenizer_path: str) -> Tokenizer:
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer from {tokenizer_path}")
    token_map = create_token_map(
        AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True),
        specials=list(FASTA_OOV_TOKENS),
    )
    logger.info(f"Vectorizing token map: {token_map}")
    token_map = {k.encode("utf-8"): v for k, v in token_map.items()}
    tokenizer = np.vectorize(lambda x: token_map.get(x, -1))

    return tokenizer


def _extract_fasta_sequences(
    *,
    species_id: str,
    fasta_file: str,
    chrom_map: dict[str, str] | None,
    output_path: str,
    chunk_size: int,
    tokenizer: Tokenizer | None = None,
):
    # Process each species config

    logger.info(f"[species={species_id}] Processing FASTA file: {fasta_file}")

    # Determine file opening method based on extension of the real file
    real_path = os.path.realpath(fasta_file)
    open_func = gzip.open if real_path.endswith(".gz") else open
    mode = "rt" if real_path.endswith(".gz") else "r"

    # Dictionary to collect sequences by species/chromosome
    sequence_records = {}

    # Parse FASTA file
    with open_func(fasta_file, mode) as file:
        for record in SeqIO.parse(file, "fasta"):
            raw_id = record.id

            if chrom_map is not None:
                if raw_id not in chrom_map:
                    logger.debug(
                        f"[species={species_id}] Skipping unmapped chromosome record: {raw_id}"
                    )
                    continue
                chrom_id = chrom_map[raw_id]
            else:
                chrom_id = raw_id
            sequence_records[(species_id, chrom_id)] = record
            logger.info(
                f"[species={species_id}] Added {chrom_id} (from {raw_id}), length: {len(record.seq)}"
            )

    logger.info(
        f"[species={species_id}] Found {len(sequence_records)} total chromosomes"
    )

    # Process each chromosome and save to Zarr
    for (species_id, chrom_id), record in sequence_records.items():
        logger.info(f"[species={species_id}] Processing chromosome: {chrom_id}")

        # Convert sequences to arrays
        seq_str = str(record.seq)
        seq_array = np.array(list(seq_str), dtype="S1")
        seq_mask = np.char.isupper(seq_array)
        rev_comp = str(record.seq.complement())
        rev_array = np.array(list(rev_comp), dtype="S1")
        rev_mask = np.char.isupper(rev_array)
        chrom_length = len(seq_str)

        seq_arrays = np.vstack([seq_array, rev_array])
        assert seq_arrays.shape == (2, chrom_length)
        seq_masks = np.vstack([seq_mask, rev_mask])
        assert seq_masks.shape == (2, chrom_length)

        # Create dataset with base sequences
        logger.info(f"[species={species_id}] Creating dataset...")
        ds = xr.Dataset(
            data_vars={
                "sequence_tokens": (["strand", "sequence"], seq_arrays),
                "sequence_masks": (["strand", "sequence"], seq_masks),
            },
            coords={
                "strand": ["positive", "negative"],
                "sequence": np.arange(chrom_length),
            },
            attrs={"species_id": species_id, "chromosome_id": chrom_id},
        )

        # Add tokenized sequences if tokenizer provided
        if tokenizer:
            # Tokenize sequences for both strands
            input_ids = []

            for i, seq in enumerate([seq_array, rev_array]):
                strand = ["forward", "reverse"][i]
                logger.info(
                    f"[species={species_id}] Tokenizing {strand} strand: {''.join(np.char.decode(seq[:64]))} ..."
                )
                token_ids = tokenizer(seq)
                logger.info(
                    f"[species={species_id}] Token ID frequencies: {pd.Series(token_ids).value_counts().to_dict()}"
                )
                # Fail on presence of any tokens not explicitly defined in the tokenizer
                # or added as a special case by FASTA_OOV_TOKENS
                if np.any(token_ids < 0):
                    bad_tokens = pd.Series(
                        # pyrefly: ignore  # bad-argument-type
                        np.char.decode(token_ids[token_ids < 0])
                    ).value_counts()
                    raise ValueError(
                        f"Found {len(bad_tokens)} unmapped tokens in "
                        f"{strand} strand for {species_id}/{chrom_id}; "
                        f"Frequencies:\n{bad_tokens.head(15)}"
                    )
                assert token_ids.shape == (chrom_length,)
                input_ids.append(token_ids)

            # Add tokenized data to dataset
            input_ids = np.vstack(input_ids)
            assert input_ids.shape == (2, chrom_length)
            ds["sequence_input_ids"] = (["strand", "sequence"], input_ids)

        # Set chunking and save
        ds = set_dimension_chunks(ds, "sequence", chunk_size)
        logger.info(
            f"[species={species_id}] Saving {chrom_id} dataset to {output_path}"
        )
        ds.to_zarr(
            output_path,
            group=f"{species_id}/{chrom_id}",
            zarr_format=2,
            consolidated=True,
            mode="w",
        )

    logger.info(
        f"[species={species_id}] Saved {len(sequence_records)} chromosome sequences to {output_path}"
        if sequence_records
        else f"[species={species_id}] No sequences were extracted"
    )


def create_token_map(
    tokenizer: Any, specials: list[str] | None = None
) -> dict[str, int]:
    # Get all standard tokens
    tokens = [t for t in tokenizer.get_vocab() if t not in tokenizer.all_special_tokens]
    # Create case-insensitive token map for standard tokens
    token_map = {
        t: (
            tokenizer.convert_tokens_to_ids(t)
            if t in tokens
            else tokenizer.convert_tokens_to_ids(token)
        )
        for token in tokens
        for t in [token.lower(), token.upper()]
    }
    # Add special tokens for the resulting mapping that are either not standard
    # or not present in the original tokenizer, which will typically map as UNKs
    if specials:
        for token in specials:
            for t in [token.lower(), token.upper()]:
                if t not in token_map:
                    token_map[t] = tokenizer.convert_tokens_to_ids(token)
    return token_map


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Extract data from fasta files for inference of gene models"
    )

    parser.add_argument("--species-id", required=True, help="Species ID")
    parser.add_argument("--input-fasta", "-i", required=True, help="Path to FASTA file")
    parser.add_argument(
        "--chrom-map",
        help="Rename chromosomes and/or select a subset of chromosomes to process. Format "
        "chromosome mapping as 'src1:dst1,src2:dst2,...'. Optional.",
    )
    parser.add_argument(
        "--output-zarr", "-o", required=True, help="Path to output file"
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to classifier checkpoint. This or --model-path is required.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the base PlantCAD model for tokenizing sequences. This or --model-checkpoint is required.",
    )

    args = parser.parse_args()

    base_model_path = args.model_path

    if base_model_path is None:
        if args.model_checkpoint is None:
            logger.error(
                "Error: must specify either --model-checkpoint or --model-path"
            )
            raise RuntimeError

        config_local = hf_hub_download(args.model_checkpoint, filename="config.json")

        with open(config_local) as file:
            config_dict = json.load(file)

        if "base_encoder_path" in config_dict.keys():
            base_model_path = config_dict["base_encoder_path"]
        else:
            logger.error(
                "Error: could not infer base model from --model-checkpoint. Please specify the base "
                "model path using --model-path instead"
            )
            raise RuntimeError

    extract_fasta_file(
        args.species_id,
        args.input_fasta,
        args.chrom_map,
        args.output_zarr,
        tokenizer_path=base_model_path,
    )


if __name__ == "__main__":
    main()
