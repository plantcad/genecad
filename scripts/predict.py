import argparse
import logging
import os
import tqdm
from typing import Literal, Any
from numpy import typing as npt
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from src.config import WINDOW_SIZE
from src.sequence import convert_entity_labels_to_intervals, create_sequence_windows, viterbi_decode
import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
from argparse import Namespace as Args
from transformers import AutoModel, AutoConfig, AutoTokenizer
from src.dataset import open_datatree, set_dimension_chunks
from src.prediction import merge_prediction_datasets
from src.modeling import GeneClassifier, GeneClassifierConfig, token_transition_probs
from src.decoding import semi_markov_viterbi_segmentation
from src.analysis import load_feature_length_distributions
from src.schema import ModelingFeatureType as MFT
import pandas as pd
from src.dist import process_group

logger = logging.getLogger(__name__)

def batched(input_list: list[Any], batch_size: int) -> list[list[Any]]:
    """Batches a list into sublists of a specified size."""
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]
    
def load_duration_probabilities(feature_labels: list[str]) -> dict[int, np.ndarray]:
    """Load duration probabilities for semi-Markov decoding.
    
    Parameters
    ----------
    feature_labels : list[str]
        List of feature labels in the order they appear in the model
        
    Returns
    -------
    dict[int, np.ndarray]
        Dictionary mapping state index to smoothed duration probability array
    """
    path = 'local/data/feature_length_distributions.parquet'
    sigma = 8.0
    max_length = 8192

    min_lengths = {
        MFT.INTERGENIC: None,
        MFT.INTRON: None,
        MFT.FIVE_PRIME_UTR: None,
        MFT.THREE_PRIME_UTR: None,
        MFT.CDS: None,
    }

    # Load smoothed distributions
    smoothed_distributions = load_feature_length_distributions(
        path=path,
        feature_labels=feature_labels,
        min_length=min_lengths,
        max_length=max_length,
        sigma=sigma
    )
    
    # Convert to the format expected by semi-Markov decoder
    duration_probs = {}
    for i, feature in enumerate(feature_labels):
        duration_probs[i] = smoothed_distributions[feature].values
    
    # Log information
    logger.info(f"Loaded duration probabilities for {len(duration_probs)} features (Gaussian smoothing σ={sigma})")
    for i, feature in enumerate(feature_labels):
        max_length = len(duration_probs[i])
        tail_mass = duration_probs[i][-1]
        # Calculate mean length from probabilities
        durations = np.arange(1, max_length + 1)
        mean_length = np.sum(durations * duration_probs[i])
        logger.info(f"  {feature}: {max_length} duration bins, tail mass: {tail_mass:.3f}, mean length: {mean_length:.1f}")
    
    return duration_probs
    
def load_classifier(args: Args) -> GeneClassifier:
    logger.info(f"Loading model from {args.model_checkpoint}")
    model = GeneClassifier.load_from_checkpoint(args.model_checkpoint, map_location=args.device)
    model = model.eval()
    return model


def load_base_model(args: Args) -> AutoModel:
    logger.info(f"Loading base embedding model from {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        args.model_path, 
        config=config, 
        trust_remote_code=True, 
        dtype=torch.bfloat16
    )
    base_model = base_model.eval().to(args.device)
    return base_model

def load_tokenizer(args: Args) -> AutoTokenizer:
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    return tokenizer

def load_models(args: Args) -> tuple[AutoModel | None, GeneClassifier, AutoTokenizer]:
    """
    Load the base embedding model, the GeneClassifier, and the tokenizer.
        
    Returns
    -------
    tuple
        (base_model, classifier_model, tokenizer) - models and tokenizer loaded and ready for inference
    """
    classifier = load_classifier(args)
    base_model = None
    if classifier.config.use_precomputed_base_encodings:
        base_model = load_base_model(args)
    tokenizer = load_tokenizer(args)
    return base_model, classifier, tokenizer

def load_data(args: Args) -> xr.Dataset:
    """
    Load the input data and select the specific species and chromosome.
        
    Returns
    -------
    xarray.Dataset
        The dataset containing the sequence data for the specified species and chromosome
    """
    logger.info(f"Opening input sequence datatree from {args.input}")
    sequences = open_datatree(args.input, consolidated=False)
    logger.info(f"Input sequences:\n{sequences}")
    
    # Check if species exists
    if args.species_id not in sequences:
        available_species = list(sequences.keys())
        raise ValueError(f"Species '{args.species_id}' not found in input data. Available species: {available_species}")
    
    # Check if chromosome exists for the specified species
    if args.chromosome_id not in sequences[args.species_id]:
        available_chromosomes = list(sequences[args.species_id].keys())
        raise ValueError(f"Chromosome '{args.chromosome_id}' not found for species '{args.species_id}'. Available chromosomes: {available_chromosomes}")
    
    # Select the dataset for the specified species and chromosome
    logger.info(f"Selecting data for species '{args.species_id}' and chromosome '{args.chromosome_id}'")
    ds = sequences[args.species_id][args.chromosome_id].ds
    
    logger.info(f"Loaded dataset with dimensions: {dict(ds.sizes)}")
    return ds

# -------------------------------------------------------------------------------------------------
# Create predictions
# -------------------------------------------------------------------------------------------------

@torch.inference_mode()
def _create_predictions(
    args: Args, 
    ds: xr.Dataset, 
    base_model: AutoModel | None, 
    classifier: GeneClassifier, 
    tokenizer: AutoTokenizer,
) -> xr.DataTree:
    """
    Generate predictions by processing sequences in strided windows.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    ds : xarray.Dataset
        Dataset containing the sequence data
    base_model : AutoModel | None
        The base embedding model, or None if not needed
    classifier : GeneClassifier
        The classifier model
    tokenizer : AutoTokenizer
        The tokenizer for the base model

    Returns
    -------
    xr.DataTree
        A data tree containing the predictions for each strand
    """
    # Get distributed processing info
    rank, world_size = process_group()

    # Construct rank-specific output path
    dataset_path = os.path.join(args.output_dir, f"predictions.{rank}.zarr")

    logger.info(f"Generating predictions with {args.batch_size=}, {args.window_size=}, {args.stride=} ({rank=}, {world_size=})")
    
    # Get padding token ID from tokenizer
    pad_value = tokenizer.unk_token_id
    if pad_value is None:
        raise ValueError("Pad value from tokenizer.unk_token_id cannot be None")
    logger.info(f"Using pad_value={pad_value} (UNK token) for sequence padding")

    # Process data for each strand separately
    token_class_names = classifier.config.token_class_names
    feature_class_names = classifier.config.token_entity_names_with_background()
    num_token_classes = len(token_class_names)
    num_feature_classes = len(feature_class_names)

    strands = ds.strand.values.tolist()
    assert set(strands) == {"positive", "negative"}
    for strand in strands:
        logger.info(f"Processing strand: {strand}")
        negative_strand = strand == "negative"
        
        # Get sequence input ids for this strand
        sequence_input_ids = ds.sel(strand=strand).sequence_input_ids.values
        assert sequence_input_ids.ndim == 1
        sequence_coordinates = ds.sel(strand=strand).sequence.values
        assert sequence_coordinates.ndim == 1
        # While not strictly necessary, ensure that coordinates are autoincrementing,
        # 0-based integers until there is a good reason to support any other coordinates
        assert sequence_coordinates.tolist() == list(range(len(sequence_coordinates)))

        # Flip token ids on negative strand from 3'->5' to 5'->3'
        if negative_strand:
            sequence_input_ids = flip(sequence_input_ids)
            sequence_coordinates = flip(sequence_coordinates)

        # Create windows of input ids to process 
        windows = list(create_sequence_windows(
            sequence_input_ids, window_size=args.window_size,
            stride=args.stride, pad_value=pad_value
        ))

        # Select windows for this rank
        windows = np.array(windows, dtype=object)
        windows = np.array_split(windows, world_size)[rank]

        # Batch windows together
        window_batches = np.array_split(windows, len(windows) // args.batch_size)
        logger.info(f"Processing {len(windows)} windows in {len(window_batches)} batches of size {args.batch_size}")

        # Process batches
        for batch_index, window_batch in enumerate(window_batches):
            logger.info(f"Processing batch {batch_index+1} of {len(window_batches)} [{rank=}, {world_size=}]")
            current_batch_size = len(window_batch)

            # Get equally sized sequence windows to process for batch
            input_ids = np.array([w[0] for w in window_batch])
            input_ids = torch.tensor(input_ids, device=args.device)
            assert input_ids.shape == (current_batch_size, args.window_size)

            # Generate embeddings, if necessary
            inputs_embeds = None
            if classifier.config.use_precomputed_base_encodings:
                inputs_embeds = base_model(input_ids=input_ids).last_hidden_state
                assert inputs_embeds.ndim == 3
                assert inputs_embeds.shape[:2] == (current_batch_size, args.window_size)

            # Get predictions from classifier
            token_logits = classifier(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds
            )
            assert token_logits.shape == (current_batch_size, args.window_size, num_token_classes)

            # Aggregate token logits to entity/feature logits
            feature_logits = classifier.aggregate_logits(token_logits)
            assert feature_logits.shape == (current_batch_size, args.window_size, num_feature_classes)

            token_logits = token_logits.cpu().numpy()
            feature_logits = feature_logits.cpu().numpy()

            # Extract valid regions from the processed windows
            token_logits_arrays, feature_logits_arrays, sequence_coord_arrays = [], [], []
            for i in range(current_batch_size):
                _, local_window, global_window = window_batch[i]
                token_logits_window = token_logits[i, local_window[0]:local_window[1], :]
                feature_logits_window = feature_logits[i, local_window[0]:local_window[1], :]
                sequence_coords_window = sequence_coordinates[global_window[0]:global_window[1]]
                token_logits_arrays.append(token_logits_window)
                feature_logits_arrays.append(feature_logits_window)
                sequence_coord_arrays.append(sequence_coords_window)

            # Concatenate all extracted regions
            token_logits = np.concatenate(token_logits_arrays, axis=0)
            feature_logits = np.concatenate(feature_logits_arrays, axis=0)
            sequence_coords = np.concatenate(sequence_coord_arrays, axis=0)

            # Flip back to 3'->5' if on negative strand
            if negative_strand:
                token_logits = flip(token_logits)
                feature_logits = flip(feature_logits)
                sequence_coords = flip(sequence_coords)

            # Create resulting dataset for batch
            result = xr.Dataset(
                data_vars={
                    "token_logits": (["sequence", "token"], token_logits),
                    "feature_logits": (["sequence", "feature"], feature_logits),
                },
                coords={
                    "sequence": sequence_coords,
                    "token": token_class_names,
                    "feature": feature_class_names,
                },
                attrs={
                    "strand": strand,
                    "species_id": args.species_id,
                    "chromosome_id": args.chromosome_id,
                    "model_checkpoint": args.model_checkpoint,
                    "model_path": args.model_path,
                }
            )

            # Assign predictions as max logits
            result["token_predictions"] = result.token_logits.argmax(dim="token")
            result["feature_predictions"] = result.feature_logits.argmax(dim="feature")

            # Chunk in sequence dim only and save
            result = set_dimension_chunks(result, "sequence", result.sizes["sequence"])
            os.makedirs(args.output_dir, exist_ok=True)
            result.to_zarr(
                dataset_path,
                group=f"/{strand}",
                zarr_format=2,
                **(
                    dict(append_dim="sequence") 
                    if os.path.exists(os.path.join(dataset_path, strand))
                    else {}
                ),
                consolidated=True
            )
    logger.info(f"Loading completed predictions from {dataset_path} ({rank=}, {world_size=})")
    result = open_datatree(dataset_path)
    return result


def flip(sequence: npt.ArrayLike) -> npt.ArrayLike:
    """Reverse a sequence along its first axis."""
    return np.flip(sequence, axis=0)

def create_predictions(args: Args):
    """
    Run the gene prediction inference pipeline to generate logits.
    Each rank writes its results to a separate file.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Set to avoid:
    # UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
    torch.set_float32_matmul_precision("medium") # same setting as training

    # Load the models and tokenizer
    base_model, classifier, tokenizer = load_models(args)
    
    # Load the data
    dataset = load_data(args)

    # Run the windowed inference to generate and save logits per rank
    logger.info(f"Running predictions for {args.species_id}/{args.chromosome_id}")
    predictions = _create_predictions(
        args, 
        dataset, 
        base_model, 
        classifier, 
        tokenizer,
    )
    logger.info(f"Complete predictions dataset:\n{predictions}")

    logger.info("Done")



# -------------------------------------------------------------------------------------------------
# Detect intervals
# -------------------------------------------------------------------------------------------------

def _detect_intervals(
    args: Args,
    predictions: xr.Dataset, 
) -> xr.Dataset:
    """
    Infer region intervals from feature predictions.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    predictions : xr.Dataset
        Dataset containing feature predictions ('feature_predictions' variable)
        
    Returns
    -------
    xr.Dataset
        Dataset containing inferred region intervals
    """
    logger.info("Inferring regions from predicted labels")
    # TODO: Fetch the label properties necessary from attributions stored in the predictions
    # datasets rather than from the configuration files, or from the original model checkpoint.
    config = GeneClassifierConfig()
    region_intervals = []
    strands = predictions.strand.values.tolist()
    assert set(strands) == {"positive", "negative"}
    
    def _decode_intervals_viterbi(logits: npt.ArrayLike) -> np.ndarray:
        transition_probs = token_transition_probs()
        if transition_probs.columns.tolist() != config.token_entity_names_with_background():
            raise ValueError(f"Transition probability classes must match token entity names; expected: {config.token_entity_names_with_background()}, got: {transition_probs.columns.tolist()}")
        emissions = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
        assert emissions.min() >= 0 and emissions.max() <= 1
        assert transition_probs.index.tolist() == transition_probs.columns.tolist()

        # Decoding takes ~90 seconds for 308452471 tokens on Grace CPU
        alpha = args.viterbi_alpha
        logger.info(f"Running viterbi decoding ({alpha=})")
        labels = viterbi_decode(
            emission_probs=emissions, 
            transition_matrix=transition_probs.values,
            alpha=alpha
        )
        
        assert labels.ndim == 1
        assert len(labels) == len(logits)
        return labels

    def _decode_intervals_semi_markov(logits: npt.ArrayLike) -> np.ndarray:
        transition_probs = token_transition_probs()
        if transition_probs.columns.tolist() != config.token_entity_names_with_background():
            raise ValueError(f"Transition probability classes must match token entity names; expected: {config.token_entity_names_with_background()}, got: {transition_probs.columns.tolist()}")
        emissions = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
        assert emissions.min() >= 0 and emissions.max() <= 1
        assert transition_probs.index.tolist() == transition_probs.columns.tolist()
        
        feature_labels = transition_probs.columns.tolist()
        duration_probs = load_duration_probabilities(feature_labels)
        
        logger.info("Running semi-Markov viterbi decoding")
        labels = semi_markov_viterbi_segmentation(
            emission_probs=emissions,
            transition_matrix=transition_probs.values,
            duration_probs=duration_probs,
            background_class=feature_labels.index(MFT.INTERGENIC.value),
            background_class_min_length=32_768,
            background_class_buffer=256,
            num_workers=64
        )
        
        assert labels.ndim == 1
        assert len(labels) == len(logits)
        return labels
        
    for strand in strands:
        # Direct label inference
        labels = predictions.sel(strand=strand).feature_predictions.values 
        logger.info(f"Running direct decoding for {strand!r} strand")
        intervals = convert_entity_labels_to_intervals(labels=labels, class_groups=config.interval_entity_classes)
        region_intervals.append(intervals.assign(strand=strand, decoding="direct"))
        
        # CRF/Semi-Markov label decoding
        logits = predictions.sel(strand=strand).feature_logits.values
        
        # Viterbi decoding
        logger.info(f"Running viterbi decoding for {strand!r} strand")
        if strand == "positive":
            viterbi_labels = _decode_intervals_viterbi(logits=logits)
        else:
            viterbi_labels = flip(_decode_intervals_viterbi(logits=flip(logits).copy()))
        
        intervals = convert_entity_labels_to_intervals(
            labels=viterbi_labels, 
            class_groups=config.interval_entity_classes
        )
        region_intervals.append(intervals.assign(strand=strand, decoding="viterbi"))
        
        # Semi-Markov decoding
        logger.info(f"Running semi-Markov decoding for {strand!r} strand")
        if strand == "positive":
            sm_labels = _decode_intervals_semi_markov(logits=logits)
        else:
            sm_labels = flip(_decode_intervals_semi_markov(logits=flip(logits).copy()))
        
        intervals = convert_entity_labels_to_intervals(
            labels=sm_labels, 
            class_groups=config.interval_entity_classes
        )
        region_intervals.append(intervals.assign(strand=strand, decoding="sm-viterbi"))
        
    region_intervals = pd.concat(region_intervals, ignore_index=True, axis=0)
    region_name_map = {
        i: config.interval_entity_name(i)
        for i in region_intervals["entity"].unique()
    }
    region_intervals = (
        region_intervals
        .rename(columns={"entity": "entity_index"})
        .assign(entity_name=lambda df: df["entity_index"].map(region_name_map))
        .rename_axis("interval", axis="index")
    )
    logger.info(f"Region intervals detected:\n{region_intervals}")
    logger.info("Region interval info:\n")
    region_intervals.info()
    region_intervals = (
        region_intervals
        .to_xarray()
        .assign_attrs(interval_entity_names=config.interval_entity_names)
    )
    return region_intervals


def detect_intervals(args: Args):
    """
    Detect intervals from per-token classifier logits.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments, where `args.input_dir` is the directory
        containing `predictions.*.zarr` files.
    """
    logger.info(f"Detecting intervals from rank files in {args.input_dir} and saving to {args.output}")

    # Merge predictions from all ranks
    sequence_predictions = merge_prediction_datasets(
        args.input_dir,
        drop_variables=["token_predictions", "token_logits"],
    )

    logger.info("Detecting intervals")
    interval_predictions = _detect_intervals(
        args=args,
        predictions=sequence_predictions,
    )
    interval_predictions = interval_predictions.assign_attrs(
        # Copy attributes from sequence predictions, which have
        # been carried along from the original fasta extraction
        **sequence_predictions.attrs
    )

    logger.info("Merging sequence and interval predictions")
    result = xr.DataTree.from_dict({
        "/sequences": sequence_predictions,
        "/intervals": interval_predictions,
    })
    
    logger.info(f"Final results:\n{result}")

    logger.info(f"Saving results to output path {args.output}")
    result.to_zarr(args.output, zarr_format=2, mode="w", consolidated=True)
    
    logger.info("Done")

# -------------------------------------------------------------------------------------------------
# GFF Exports
# -------------------------------------------------------------------------------------------------

class GffRecord(BaseModel):
    """Represents a single record (line) in a GFF3 file using Pydantic."""
    seqid: str = Field(..., description="Sequence identifier (e.g., chromosome name)")
    source: str = Field(..., description="Source of the feature (e.g., program name)")
    type: str = Field(..., description="Type of the feature (e.g., gene, CDS)")
    start: int = Field(..., description="Start position (1-based, inclusive)", gt=0)
    end: int = Field(..., description="End position (1-based, inclusive)", gt=0)
    score: float | None = Field(default=None, description="Score of the feature")
    strand: Literal["+", "-"] = Field(..., description="Strand: '+' or '-'")
    phase: Literal[0, 1, 2] | None = Field(default=None, description="Phase for CDS features (0, 1, 2, or None)")
    attributes: str | None = Field(default=None, description="Attributes in 'key=value;' format")

    @field_validator('end')
    def check_end_ge_start(cls, v: int, info: ValidationInfo) -> int:
        """Validate that end position is greater than or equal to start position."""
        if 'start' in info.data and v < info.data['start']:
            raise ValueError(f"End position {v} must be greater than or equal to start position {info.data['start']}")
        return v

    def to_line(self) -> str:
        """Converts the GffRecord object to a GFF3 formatted string."""
        score_str = f"{self.score}" if self.score is not None else "."
        strand_str = self.strand if self.strand is not None else "."
        phase_str = str(self.phase) if self.phase is not None else "."
        attributes_str = self.attributes if self.attributes is not None else "."
        
        return "\t".join([
            self.seqid,
            self.source,
            self.type,
            str(self.start), 
            str(self.end),
            score_str,
            strand_str,
            phase_str,
            attributes_str
        ])

def _create_gff_attributes(id: str, parent_id: str | None = None) -> str:
    """Creates the GFF attribute string."""
    attrs = f"ID={id}"
    if parent_id:
        attrs += f";Parent={parent_id}"
    return attrs

def process_single_transcript(transcript_row: pd.Series, all_features: pd.DataFrame) -> pd.DataFrame | None:
    # Find features within the transcript bounds on the same strand
    mask = (
        (all_features['strand'] == transcript_row['strand']) &
        (all_features['start'] >= transcript_row['start']) &
        (all_features['stop'] <= transcript_row['stop'])
    )
    matching_features = all_features[mask]

    if not matching_features.empty:
        # Combine transcript and its features, sort by start position
        gene_group = pd.concat([transcript_row.to_frame().T, matching_features]).sort_values('start')
        return gene_group
    else:
        return None
    
def group_intervals_by_transcript(intervals: pd.DataFrame, min_transcript_length: int = 0) -> list[pd.DataFrame]:
    """
    Group feature intervals by their parent transcript.

    Parameters
    ----------
    intervals : pd.DataFrame
        DataFrame containing interval predictions with columns like 
        'start', 'stop', 'strand', 'entity_name'.
    min_transcript_length : int, default 0
        Minimum length of transcript to include. Transcripts shorter than
        this value will be filtered out.

    Returns
    -------
    list[pd.DataFrame]
        A list where each element is a DataFrame representing a transcript 
        and its associated features (CDS, UTRs, etc.), sorted by start position.
        The first row of each DataFrame corresponds to the transcript interval.
    """
    logger.info("Grouping intervals by transcript")
    transcript_intervals = intervals[intervals['entity_name'] == 'transcript'].sort_values('start')
    
    # Filter out transcripts that are too short
    initial_count = len(transcript_intervals)
    transcript_intervals['length'] = transcript_intervals['stop'] - transcript_intervals['start']
    transcript_intervals = transcript_intervals[transcript_intervals['length'] >= min_transcript_length]
    retained_count = len(transcript_intervals)
    filtered_count = initial_count - retained_count
    
    logger.info(f"Filtered out {filtered_count} transcripts below minimum length of {min_transcript_length}bp")
    logger.info(f"Retained {retained_count} transcripts")
    
    feature_intervals = intervals[intervals['entity_name'] != 'transcript']
  
    results = [
        process_single_transcript(row, feature_intervals)
        for _, row in tqdm.tqdm(transcript_intervals.iterrows(), total=len(transcript_intervals))
    ]
    genes = [group for group in results if group is not None]
  
    logger.info(f"Grouped intervals into {len(genes)} transcripts/genes.")
    return genes

def generate_gff(genes: list[pd.DataFrame], chrom_id: str, output_path: str) -> None:
    """
    Generate a GFF3 file from grouped gene intervals.

    Parameters
    ----------
    genes : list[pd.DataFrame]
        A list of DataFrames, each representing a transcript and its features.
    chrom_id : str
        The chromosome ID for the GFF records.
    output_path : str
        Path to write the GFF3 output file.
    """
    logger.info(f"Generating GFF3 output for {len(genes)} genes on {chrom_id}")
    gff_records = []
    gene_counter = 0
    source = "plantCaduceus" # GFF source field

    gff_feature_map = {
        "cds": "CDS",
        "five_prime_utr": "five_prime_UTR",
        "three_prime_utr": "three_prime_UTR",
    }

    for gene_group in genes:
        gene_counter += 1
        gene_id = f"gene_{gene_counter}"
        rna_id = f"{gene_id}.t1"

        gene_start = int(gene_group['start'].min())
        gene_stop = int(gene_group['stop'].max())
        strand_symbol = '+' if gene_group['strand'].iloc[0] == 'positive' else '-'
        
        # Create gene record
        gff_records.append(GffRecord(
            seqid=chrom_id, source=source, type="gene",
            start=gene_start + 1, end=gene_stop + 1, # 1-based
            strand=strand_symbol,
            attributes=_create_gff_attributes(id=gene_id)
        ))

        # Create mRNA record
        gff_records.append(GffRecord(
            seqid=chrom_id, source=source, type="mRNA",
            start=gene_start + 1, end=gene_stop + 1, # 1-based
            strand=strand_symbol,
            attributes=_create_gff_attributes(id=rna_id, parent_id=gene_id)
        ))

        # Create feature records (CDS, UTRs)
        feature_counters = {ftype: 0 for ftype in gff_feature_map.values()}
        for _, interval in gene_group.iterrows():
            entity_name = interval['entity_name']
            if entity_name == 'transcript':
                continue
                
            gff_type = gff_feature_map.get(entity_name)
            if not gff_type:
                continue
            feature_counters[gff_type] += 1
            feature_id = f"{rna_id}.{gff_type}.{feature_counters[gff_type]}"
                
            gff_records.append(GffRecord(
                seqid=chrom_id, source=source, type=gff_type,
                start=int(interval['start']) + 1, # 1-based
                end=int(interval['stop']) + 1,   # 1-based
                strand=strand_symbol, phase=None, # Phase remains None
                attributes=_create_gff_attributes(id=feature_id, parent_id=rna_id)
            ))

    # Convert records to strings and write file
    gff_lines = ["##gff-version 3"] + [rec.to_line() for rec in gff_records]
    logger.info(f"Writing {len(gff_lines)} lines to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(gff_lines) + "\n")

def export_gff(args: Args):
    """
    Export prediction results to GFF format.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    logger.info(f"Loading predictions from {args.input}")
    # Use DataTree to easily access attributes and specific groups
    predictions: xr.DataTree = open_datatree(args.input, consolidated=False) 
    intervals_dataset: xr.Dataset = predictions["/intervals"].ds
    
    # Extract chromosome ID from attributes (assuming it was saved during inference)
    if "chromosome_id" not in intervals_dataset.attrs:
         raise ValueError("Cannot find 'chromosome_id' attribute in /intervals dataset.")
    chrom_id = intervals_dataset.attrs["chromosome_id"]
    logger.info(f"Loaded intervals for chromosome: {chrom_id}")

    logger.info("Converting interval predictions to DataFrame")
    intervals_table = intervals_dataset.to_dataframe().reset_index()
    
    # Add decoding column if not present (for backward compatibility)
    if "decoding" not in intervals_table.columns:
        intervals_table["decoding"] = "direct"
    
    # Filter by decoding method
    intervals_table = intervals_table[intervals_table["decoding"] == args.decoding_method]
    logger.info(f"Filtered intervals to {len(intervals_table)} rows using decoding method: {args.decoding_method}")
    
    # Group intervals by transcript
    genes = group_intervals_by_transcript(intervals_table, args.min_transcript_length)
    
    # Generate and save GFF
    generate_gff(genes, chrom_id, args.output)
    
    logger.info("GFF export complete")

def main():
    """Main entry point for prediction and export functions."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Gene prediction and export tools")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")
    
    # Run inference command
    inference_parser = subparsers.add_parser("create_predictions", help="Generate token and feature logits with predicted classes")
    inference_parser.add_argument("--input", required=True, help="Path to input zarr dataset (from transform.py)")
    inference_parser.add_argument("--output-dir", required=True, help="Directory to save rank-specific output zarr datasets")
    inference_parser.add_argument("--model-checkpoint", required=True, help="Path to classifier checkpoint")
    inference_parser.add_argument("--model-path", required=True, help="Path to base embedding model")
    
    # Selection arguments
    inference_parser.add_argument("--species-id", required=True, help="Species ID to process (e.g., 'Osativa')")
    inference_parser.add_argument("--chromosome-id", required=True, help="Chromosome ID to process (e.g., 'Chr1')")
    
    # Processing parameters
    inference_parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Window size for sequence processing")
    inference_parser.add_argument("--stride", type=int, default=WINDOW_SIZE//2, help="Stride size for overlapping windows")
    inference_parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference")
    inference_parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference (cuda/cpu)")
    
    # Detect intervals command
    detect_parser = subparsers.add_parser("detect_intervals", help="Detect intervals from generated logits")
    detect_parser.add_argument("--input-dir", required=True, help="Path to input zarr dataset from generate_logits")
    detect_parser.add_argument("--output", required=True, help="Path to output zarr dataset for intervals")
    detect_parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Window size for sequence processing")
    detect_parser.add_argument("--viterbi-alpha", type=float, default=None, help="Alpha parameter for viterbi decoding (default: None)")
    
    # Export GFF command
    gff_parser = subparsers.add_parser("export_gff", help="Export predictions to GFF format")
    gff_parser.add_argument("--input", required=True, help="Path to input zarr dataset from run_inference")
    gff_parser.add_argument("--output", required=True, help="Path to output GFF file")
    gff_parser.add_argument("--min-transcript-length", type=int, default=0, help="Minimum transcript length (default: 0, no filtering)")
    gff_parser.add_argument("--decoding-method", type=str, choices=["direct", "viterbi", "sm-viterbi"], default="direct", help="Decoding method to use (default: direct)")
    
    args = parser.parse_args()
    
    if args.command == "create_predictions":
        create_predictions(args)
    elif args.command == "detect_intervals":
        detect_intervals(args)
    elif args.command == "export_gff":
        export_gff(args)
    else:
        parser.print_help()
    
if __name__ == "__main__":
    main()
