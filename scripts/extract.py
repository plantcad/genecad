import os
import gzip
import argparse
import logging
from typing import Any
import xarray as xr
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature
from src.gff_parser import parse as parse_gff
from src.naming import normalize_species_identifier
from src.dataset import DEFAULT_SEQUENCE_CHUNK_SIZE, open_datatree, set_dimension_chunks
from src.schema import FeatureType, SequenceFeature, PositionInfo
from src.config import SPECIES_CONFIGS


# Set up logging
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------------

def get_feature_name(feature: SeqFeature) -> str | None:
    """Extract name from a SeqFeature object."""
    if not hasattr(feature, 'qualifiers'):
        return None
        
    qualifiers = feature.qualifiers
    for key in ['Name', 'geneName']:
        if key in qualifiers and qualifiers[key]:
            return qualifiers[key][0]
    return None

def get_feature_id(feature: SeqFeature) -> str | None:
    """Extract ID from a SeqFeature object."""
    return feature.id if hasattr(feature, 'id') else None

def get_position_info(feature: SeqFeature) -> PositionInfo:
    """Extract position information from a SeqFeature."""
    return PositionInfo(
        strand=1 if feature.location.strand > 0 else -1,
        start=int(feature.location.start.real),
        stop=int(feature.location.end.real)
    )


def is_canonical_transcript(feature: SeqFeature) -> bool:
    """Check if a transcript is canonical (has longest=1)."""
    if hasattr(feature, 'qualifiers') and 'longest' in feature.qualifiers:
        return feature.qualifiers['longest'][0] == '1'
    return False


# -------------------------------------------------------------------------------------------------
# Extract GFF features
# -------------------------------------------------------------------------------------------------

def extract_gff_features(input_paths: list[str], output_path: str) -> None:
    """Extract data from GFF file(s) into a structured DataFrame.
    
    Parameters
    ----------
    input_paths : list[str]
        Path(s) to input GFF file(s)
    output_path : str
        Path to output parquet file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data from all input files and concatenate
    dfs = []
    for input_path in input_paths:
        logger.info(f"Processing file: {input_path}")
        df = _extract_gff_features(input_path)
        dfs.append(df)
        logger.info(f"Extracted {df.shape[0]} rows from {input_path}")
    
    # Combine and validate all dataframes
    if dfs:
        df = pd.concat(dfs, ignore_index=True, axis=0)
        del dfs
        logger.info(f"Saving combined DataFrame with {df.shape[0]} rows to {output_path}; info:")
        df.info()
        df.to_parquet(output_path, index=False)
        del df
        logger.info("Validating extracted features")
        validate_gff_features(pd.read_parquet(output_path))
        logger.info("Validation complete")
    else:
        logger.warning("No data was extracted from the input files")

def _extract_gff_features(path: str) -> pd.DataFrame:
    """Extract data from a single GFF file into a structured DataFrame."""
    logger.info(f"Parsing GFF file: {path}")
    
    # Parse GFF file
    with open(path) as in_handle:
        records = list(parse_gff(in_handle))
    
    logger.info(f"Found {len(records)} chromosome records")
    features_data: list[SequenceFeature] = []
    filename = os.path.basename(path)

    # Process each chromosome
    for chrom in records:
        chrom_id = chrom.id
        chrom_name = chrom.name
        chrom_length = len(chrom.seq)
        species_name = " ".join(chrom.annotations["species"][0])
        species_id = normalize_species_identifier(species_name)
        
        logger.info(f"Processing {species_name} chromosome: {chrom_id} with {len(chrom.features)} features")
        
        # Process each gene
        for gene in chrom.features:
            if gene.type != FeatureType.GENE:
                raise ValueError(f"Found unexpected chromosome feature type: {gene.type}")
                
            gene_id = get_feature_id(gene)
            gene_info = get_position_info(gene)
            gene_name = get_feature_name(gene)
            
            # Process each transcript
            for transcript in gene.sub_features:
                if transcript.type != FeatureType.MRNA:
                    raise ValueError(f"Found unexpected gene feature type: {transcript.type}")
                    
                transcript_id = get_feature_id(transcript)
                transcript_info = get_position_info(transcript)
                transcript_name = get_feature_name(transcript)
                transcript_is_canonical = is_canonical_transcript(transcript)
                
                # Process each feature
                for feature in transcript.sub_features:
                    if feature.type not in [
                        FeatureType.FIVE_PRIME_UTR,
                        FeatureType.CDS,
                        FeatureType.THREE_PRIME_UTR,
                    ]:
                        raise ValueError(f"Found unexpected transcript feature type: {feature.type}")
                        
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
    
    logger.info(f"Total features extracted: {len(features_data)}")
    
    # Convert list of Pydantic models to DataFrame
    df = pd.DataFrame([feature.model_dump() for feature in features_data])
    return df

def validate_gff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the extracted GFF data."""

    # Check for primary key uniqueness
    logger.info("Checking for primary key uniqueness")
    primary_key = ['species_id', 'chromosome_id', 'gene_id', 'transcript_id', 'feature_id']
    duplicated_keys = (
        df[primary_key].value_counts()
        .pipe(lambda x: x[x > 1])
    )
    if len(duplicated_keys) > 0:
        raise ValueError(
            f"Found {len(duplicated_keys)} duplicate rows for {primary_key=}; "
            f"Examples:\n{duplicated_keys.head()}"
        )
    
    # Check canonical transcript indicator consistency
    logger.info("Checking for canonical transcript indicator consistency")
    bad_transcripts = (
        df.groupby(['species_id', 'chromosome_id', 'gene_id', 'transcript_id'])['transcript_is_canonical']
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
        .sort_values().to_list()
    )
    if (invalid_combinations := strand_combinations - {(-1, -1, -1), (1, 1, 1)}):
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
            inconsistencies = (
                df.groupby(['species_id', 'chromosome_id', f'{prefix}_id'])[f'{prefix}_{col}'].unique()
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
        (df["gene_start"] < 0) |
        (df["gene_stop"] > df["chromosome_length"])
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
        (df["transcript_start"] < df["gene_start"]) |
        (df["transcript_stop"] > df["gene_stop"])
    ]
    if len(invalid_transcripts) > 0:
        invalid_examples = invalid_transcripts.groupby(["gene_id", "transcript_id"]).first()
        raise ValueError(
            f"Found {len(invalid_examples)} transcripts that extend beyond their gene boundaries. "
            f"Examples:\n{invalid_examples[['gene_start', 'gene_stop', 'transcript_start', 'transcript_stop']].head()}"
        )
    
    # Check for features extending beyond transcript boundaries
    logger.info("Checking for features extending beyond transcript boundaries")
    invalid_features = df[
        (df["feature_start"] < df["transcript_start"]) |
        (df["feature_stop"] > df["transcript_stop"]) 
    ]
    if len(invalid_features) > 0:
        invalid_examples = invalid_features.groupby(["transcript_id", "feature_id"]).first()
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

def parse_chromosome_map(map_str: str) -> dict[str, str]:
    """Parse and validate a chromosome mapping string.
    
    Parameters
    ----------
    map_str : str
        String in format 'Key1=Value1,Key2=Value2,...'
        
    Returns
    -------
    dict[str, str]
        Mapping from raw chromosome names to standardized names
    """
    # Parse the mapping string
    try:
        pairs = [pair.strip().split('=') for pair in map_str.split(',')]
        chrom_map = {k.strip(): v.strip() for k, v in pairs}
    except Exception as e:
        raise ValueError(f"Invalid chromosome map format. Expected 'Key1=Value1,Key2=Value2,...' but got: {map_str}") from e
    
    # Check for empty mapping
    if not chrom_map:
        raise ValueError("Chromosome map cannot be empty")
    
    return chrom_map

def extract_fasta_sequences(input_paths: list[str], species_ids: list[str], output_path: str, chromosome_map: str | None = None, chunk_size: int = DEFAULT_SEQUENCE_CHUNK_SIZE, tokenizer_path: str | None = None) -> None:
    """Extract sequences from FASTA file(s) into a structured format.
    
    Parameters
    ----------
    input_paths : list[str]
        Path(s) to input FASTA file(s)
    species_ids : list[str]
        List of species IDs corresponding to input files; must have same
        order and length as input_paths
    output_path : str
        Path to output file
    chromosome_map : str, optional
        Case-sensitive mapping from raw chromosome names to standardized names in format
        'Key1=Value1,Key2=Value2,...'. If provided, this mapping will be used for all species.
        If not provided, species-specific mappings from config will be used.
    chunk_size : int, optional
        Size of chunks to write to Zarr
    tokenizer_path : str, optional
        Path to tokenizer if tokenizing sequences
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check that number of species IDs matches number of input paths
    if len(species_ids) != len(input_paths):
        raise ValueError(f"Number of species IDs ({len(species_ids)}) must match number of input paths ({len(input_paths)})")
    
    # Get chromosome mappings - either shared or species-specific
    species_chrom_maps = {}
    if chromosome_map is not None:
        # Use the same mapping for all species
        shared_map = parse_chromosome_map(chromosome_map)
        species_chrom_maps = {species_id: shared_map for species_id in species_ids}
        logger.info(f"Using shared chromosome mapping for all species: {shared_map}")
    else:
        # Get species-specific mappings from config
        for species_id in species_ids:
            if species_id not in SPECIES_CONFIGS:
                raise ValueError(f"No configuration found for species {species_id}. Available species: {list(SPECIES_CONFIGS.keys())}")
            species_chrom_maps[species_id] = SPECIES_CONFIGS[species_id].chromosome_map
            logger.info(f"Using chromosome mapping for {species_id}: {species_chrom_maps[species_id]}")
    
    logger.info(f"Extracting sequences from {len(input_paths)} FASTA file(s) (species: {species_ids})")
    
    # Load tokenizer if path provided
    tokenizer = None
    if tokenizer_path:
        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        token_map = create_token_map(
            AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True),
            specials=list(FASTA_OOV_TOKENS)
        )
        logger.info(f"Vectorizing token map: {token_map}")
        token_map = {k.encode("utf-8"): v for k, v in token_map.items()}
        tokenizer = np.vectorize(lambda x: token_map.get(x, -1))
    
    # Dictionary to collect sequences by species/chromosome
    sequence_records = {}
    
    # Process each FASTA file with its corresponding species_id
    for fasta_file, species_id in zip(input_paths, species_ids):
        # Get the chromosome map for this species
        chrom_map = species_chrom_maps[species_id]
        
        # Determine file opening method based on extension
        open_func = gzip.open if fasta_file.endswith(".gz") else open
        mode = "rt" if fasta_file.endswith(".gz") else "r"
        
        # Parse FASTA file
        with open_func(fasta_file, mode) as file:
            for record in SeqIO.parse(file, "fasta"):
                raw_id = record.id
                if raw_id not in chrom_map:
                    logger.info(f"  Skipping unmapped chromosome record: {species_id}/{raw_id}")
                    continue
                chrom_id = chrom_map[raw_id]
                sequence_records[(species_id, chrom_id)] = record
                logger.info(f"  Added {species_id}/{chrom_id} (from {raw_id}), length: {len(record.seq)}")
    
    logger.info(f"Found {len(sequence_records)} total chromosomes")
    
    # Process each chromosome and save to Zarr
    for (species_id, chrom_id), record in sequence_records.items():
        logger.info(f"  Processing {species_id}/{chrom_id}")

        # Convert sequences to arrays
        seq_str = str(record.seq)
        seq_array = np.array(list(seq_str), dtype='S1')
        seq_mask = np.char.isupper(seq_array)
        rev_comp = str(record.seq.complement())
        rev_array = np.array(list(rev_comp), dtype='S1')
        rev_mask = np.char.isupper(rev_array)
        chrom_length = len(seq_str)

        seq_arrays = np.vstack([seq_array, rev_array])
        assert seq_arrays.shape == (2, chrom_length)
        seq_masks = np.vstack([seq_mask, rev_mask])
        assert seq_masks.shape == (2, chrom_length)

        # Create dataset with base sequences
        logger.info("  Creating dataset...")
        ds = xr.Dataset(
            data_vars={
                "sequence_tokens": (["strand", "sequence"], seq_arrays),
                "sequence_masks": (["strand", "sequence"], seq_masks)
            },
            coords={"strand": ["positive", "negative"], "sequence": np.arange(chrom_length)},
            attrs={"species_id": species_id, "chromosome_id": chrom_id}
        )
        
        # Add tokenized sequences if tokenizer provided
        if tokenizer:
            # Tokenize sequences for both strands
            input_ids = []
            
            for i, seq in enumerate([seq_array, rev_array]):
                strand = ['forward', 'reverse'][i]
                logger.info(f"  Tokenizing {strand} strand: {''.join(np.char.decode(seq[:64]))} ...")                
                token_ids = tokenizer(seq)
                logger.info(f"  Token ID frequencies: {pd.Series(token_ids).value_counts().to_dict()}")
                # Fail on presence of any tokens not explicitly defined in the tokenizer 
                # or added as a special case by FASTA_OOV_TOKENS
                if np.any(token_ids < 0):
                    bad_tokens = (
                        pd.Series(np.char.decode(token_ids[token_ids < 0]))
                        .value_counts()
                    )
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
        logger.info(f"  Saving {species_id}/{chrom_id} dataset to {output_path}")
        ds.to_zarr(
            output_path, 
            group=f"{species_id}/{chrom_id}", 
            zarr_format=2, 
            consolidated=True, 
            mode="w"
        )
    
    logger.info(
        f"Saved {len(sequence_records)} chromosome sequences to {output_path}"
        if sequence_records else 
        "No sequences were extracted"
    )
    dt = open_datatree(output_path)
    logger.info(f"Final data tree:\n{dt}")
    logger.info("Done")


def create_token_map(tokenizer: Any, specials: list[str] | None = None) -> dict[str, int]:
    # Get all standard tokens
    tokens = [
        t for t in tokenizer.get_vocab() 
        if t not in tokenizer.all_special_tokens
    ]
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Extract data from genomic files into structured formats")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")
    
    # Extract GFF features command
    gff_parser = subparsers.add_parser("extract_gff_features", help="Extract data from GFF file(s) into a structured DataFrame")
    gff_parser.add_argument("--input", required=True, nargs='+', help="Path(s) to input GFF file(s)")
    gff_parser.add_argument("--output", required=True, help="Path to output parquet file")
    
    # Extract FASTA sequences command
    fasta_parser = subparsers.add_parser("extract_fasta_sequences", help="Extract sequences from FASTA file(s)")
    fasta_parser.add_argument("--input", required=True, nargs='+', help="Path(s) to input FASTA file(s)")
    fasta_parser.add_argument("--species-id", required=True, nargs='+', help="Species IDs corresponding to input files")
    fasta_parser.add_argument("--output", required=True, help="Path to output file")
    fasta_parser.add_argument("--chromosome-map", help="Mapping from raw to standardized chromosome names (e.g. 'Chrom1=chr1,Chrom2=chr2')")
    fasta_parser.add_argument("--tokenizer-path", help="Path to the tokenizer model for tokenizing sequences")
    
    args = parser.parse_args()
    
    if args.command == "extract_gff_features":
        extract_gff_features(args.input, args.output)
    elif args.command == "extract_fasta_sequences":
        extract_fasta_sequences(
            args.input, args.species_id, args.output,
            args.chromosome_map, tokenizer_path=args.tokenizer_path
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
