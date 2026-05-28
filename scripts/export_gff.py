import argparse
import json
import logging
import multiprocessing as mp
import os
import tqdm
from typing import Literal
from pydantic import BaseModel, Field, field_validator, ValidationInfo
import xarray as xr
from src.dataset import open_datatree
from src.schema import GffFeatureType
import pandas as pd

logger = logging.getLogger(__name__)


class GffRecord(BaseModel):
    """Represents a single record (line) in a GFF3 file using Pydantic."""

    seqid: str = Field(..., description="Sequence identifier (e.g., chromosome name)")
    source: str = Field(..., description="Source of the feature (e.g., program name)")
    type: str = Field(..., description="Type of the feature (e.g., gene, CDS)")
    # pyrefly: ignore  # no-matching-overload
    start: int = Field(..., description="Start position (1-based, inclusive)", gt=0)
    # pyrefly: ignore  # no-matching-overload
    end: int = Field(..., description="End position (1-based, inclusive)", gt=0)
    score: float | None = Field(default=None, description="Score of the feature")
    strand: Literal["+", "-"] = Field(..., description="Strand: '+' or '-'")
    phase: Literal[0, 1, 2] | None = Field(
        default=None, description="Phase for CDS features (0, 1, 2, or None)"
    )
    attributes: str | None = Field(
        default=None, description="Attributes in 'key=value;' format"
    )

    @field_validator("end")
    def check_end_ge_start(cls, v: int, info: ValidationInfo) -> int:
        """Validate that end position is greater than or equal to start position."""
        if "start" in info.data and v < info.data["start"]:
            raise ValueError(
                f"End position {v} must be greater than or equal to start position {info.data['start']}"
            )
        return v

    def to_line(self) -> str:
        """Converts the GffRecord object to a GFF3 formatted string."""
        score_str = f"{self.score}" if self.score is not None else "."
        strand_str = self.strand if self.strand is not None else "."
        phase_str = str(self.phase) if self.phase is not None else "."
        attributes_str = self.attributes if self.attributes is not None else "."

        return "\t".join(
            [
                self.seqid,
                self.source,
                self.type,
                str(self.start),
                str(self.end),
                score_str,
                strand_str,
                phase_str,
                attributes_str,
            ]
        )


def process_single_transcript(
    transcript_row: pd.Series, all_features: pd.DataFrame
) -> pd.DataFrame | None:
    # Find features within the transcript bounds on the same strand
    mask = (
        (all_features["strand"] == transcript_row["strand"])
        & (all_features["start"] >= transcript_row["start"])
        & (all_features["stop"] <= transcript_row["stop"])
    )
    matching_features = all_features[mask]

    if not matching_features.empty:
        # Combine transcript and its features, sort by start position
        gene_group = pd.concat(
            [transcript_row.to_frame().T, matching_features]
        ).sort_values("start")
        return gene_group
    else:
        return None


_TRANSCRIPT_WORKER_FEATURES: pd.DataFrame | None = None


def _init_transcript_worker(all_features: pd.DataFrame) -> None:
    global _TRANSCRIPT_WORKER_FEATURES
    _TRANSCRIPT_WORKER_FEATURES = all_features


def _process_single_transcript_worker(
    transcript_row_dict: dict,
) -> pd.DataFrame | None:
    if _TRANSCRIPT_WORKER_FEATURES is None:
        raise RuntimeError("Transcript worker was not initialized with features")
    transcript_row = pd.Series(transcript_row_dict)
    return process_single_transcript(transcript_row, _TRANSCRIPT_WORKER_FEATURES)


def _create_gff_attributes(id: str, parent_id: str | None = None) -> str:
    """Create a GFF attribute string for a given feature identifier."""
    attrs = f"ID={id}"
    if parent_id:
        attrs += f";Parent={parent_id}"
    return attrs


def group_intervals_by_transcript(
    intervals: pd.DataFrame,
    min_transcript_length: int = 0,
    cpu_workers: int = 1,
    tqdm_position: int | None = None,
    tqdm_desc: str | None = None,
) -> list[pd.DataFrame]:
    """Group feature intervals by transcript with optional length filtering.

    Parameters
    ----------
    intervals : pandas.DataFrame
        DataFrame containing interval predictions with ``start``, ``stop``,
        ``strand``, and ``entity_name`` columns.
    min_transcript_length : int, default 0
        Minimum transcript length in base pairs required to keep a transcript.
    cpu_workers : int, default 1
        Number of CPU worker processes for transcript grouping. A value of 1
        preserves original single-process behavior.
    tqdm_position : int or None, default None
        Optional terminal row index for tqdm when multiple processes are
        writing progress bars concurrently.
    tqdm_desc : str or None, default None
        Optional tqdm description label.

    Returns
    -------
    list[pandas.DataFrame]
        List of DataFrames where each item corresponds to a transcript and its
        associated features sorted by genomic position.
    """
    logger.info("Grouping intervals by transcript")
    transcript_intervals = intervals[
        intervals["entity_name"] == "transcript"
    ].sort_values("start")

    # Filter out transcripts that are too short
    initial_count = len(transcript_intervals)
    transcript_intervals["length"] = (
        transcript_intervals["stop"] - transcript_intervals["start"]
    )
    transcript_intervals = transcript_intervals[
        transcript_intervals["length"] >= min_transcript_length
    ]
    retained_count = len(transcript_intervals)
    filtered_count = initial_count - retained_count

    logger.info(
        f"Filtered out {filtered_count} transcripts below minimum length of {min_transcript_length}bp"
    )
    logger.info(f"Retained {retained_count} transcripts")

    feature_intervals = intervals[intervals["entity_name"] != "transcript"]

    # Preserve deterministic transcript order so generated gene IDs and final
    # GFF record ordering remain unchanged regardless of worker count.
    transcript_rows = [row.to_dict() for _, row in transcript_intervals.iterrows()]

    if cpu_workers <= 1 or len(transcript_rows) == 0:
        results = [
            process_single_transcript(pd.Series(row), feature_intervals)
            for row in tqdm.tqdm(
                transcript_rows,
                total=len(transcript_rows),
                desc=tqdm_desc,
                position=tqdm_position,
                dynamic_ncols=True,
                leave=False,
                mininterval=0.1,
            )
        ]
    else:
        logger.info(
            f"Grouping transcripts with cpu_workers={cpu_workers} (order-preserving map)"
        )
        start_methods = mp.get_all_start_methods()
        ctx_name = "fork" if "fork" in start_methods else start_methods[0]
        ctx = mp.get_context(ctx_name)
        with ctx.Pool(
            processes=cpu_workers,
            initializer=_init_transcript_worker,
            initargs=(feature_intervals,),
        ) as pool:
            results = [
                result
                for result in tqdm.tqdm(
                    pool.imap(_process_single_transcript_worker, transcript_rows),
                    total=len(transcript_rows),
                    desc=tqdm_desc,
                    position=tqdm_position,
                    dynamic_ncols=True,
                    leave=False,
                    mininterval=0.1,
                )
            ]

    genes = [group for group in results if group is not None]

    logger.info(f"Grouped intervals into {len(genes)} transcripts/genes.")
    return genes


def generate_gff(
    genes: list[pd.DataFrame],
    chrom_id: str,
    output_path: str,
    strip_introns: bool = True,
    source: str = "GeneCAD",
    tqdm_position: int | None = None,
    tqdm_desc: str | None = None,
) -> None:
    """Write a GFF3 file from grouped gene intervals.

    Parameters
    ----------
    genes : list[pandas.DataFrame]
        List of transcript DataFrames produced by
        :func:`group_intervals_by_transcript`.
    chrom_id : str
        Chromosome identifier used for all emitted records.
    output_path : str
        Destination file path for the generated GFF3 file.
    strip_introns : bool, default True
        Whether to remove intron records when constructing gene boundaries.
    source : str, default "GeneCAD"
        Value written in the GFF ``source`` column.
    tqdm_position : int or None, default None
        Optional terminal row index for tqdm when multiple processes are
        writing progress bars concurrently.
    tqdm_desc : str or None, default None
        Optional tqdm description label.
    """
    logger.info(f"Generating GFF3 output for {len(genes)} genes on {chrom_id}")
    gff_records = []
    gene_counter = 0

    gff_feature_map = {
        "cds": GffFeatureType.CDS.value,
        "five_prime_utr": GffFeatureType.FIVE_PRIME_UTR.value,
        "three_prime_utr": GffFeatureType.THREE_PRIME_UTR.value,
    }

    # Valid entity names
    valid_entity_names = {
        "exon",
        "cds",
        "intron",
        "five_prime_utr",
        "three_prime_utr",
        "transcript",
    }

    for gene_group in tqdm.tqdm(  # pyrefly: ignore[not-iterable]
        genes,
        desc=tqdm_desc,
        position=tqdm_position,
        dynamic_ncols=True,
        leave=False,
        mininterval=0.1,
    ):
        # Validate entity names
        invalid_entities = set(gene_group["entity_name"].unique()) - valid_entity_names
        if invalid_entities:
            raise ValueError(
                f"Unexpected entity_name values found: {invalid_entities}. "
                f"Valid values are: {valid_entity_names}"
            )

        # Filter out introns if strip_introns is True
        if strip_introns:
            boundary_features = ["five_prime_utr", "three_prime_utr", "cds"]
            gene_group_filtered = gene_group[
                gene_group["entity_name"].isin(boundary_features)
            ].copy()
            if len(gene_group_filtered) == 0:
                continue
        else:
            start_entities = set(
                gene_group[gene_group["start"] == gene_group["start"].min()][
                    "entity_name"
                ]
            )
            stop_entities = set(
                gene_group[gene_group["stop"] == gene_group["stop"].max()][
                    "entity_name"
                ]
            )

            # TODO: if this is an option, then why does it raise an error if not satisfied?
            if "intron" in start_entities or "intron" in stop_entities:
                raise ValueError(
                    "Gene has terminal introns, but strip_introns is False. "
                    f"This would result in incorrect gene boundaries. Gene records:\n{gene_group}"
                )
            gene_group_filtered = gene_group.copy()

        gene_counter += 1
        gene_id = f"gene_{gene_counter}"
        rna_id = f"{gene_id}.t1"

        gene_start = int(gene_group_filtered["start"].min())
        gene_stop = int(gene_group_filtered["stop"].max())
        strand_symbol = (
            "+" if gene_group_filtered["strand"].iloc[0] == "positive" else "-"
        )

        # Create gene record
        gff_records.append(
            GffRecord(
                seqid=chrom_id,
                source=source,
                type=GffFeatureType.GENE.value,
                start=gene_start + 1,
                end=gene_stop + 1,  # 1-based
                strand=strand_symbol,
                attributes=_create_gff_attributes(id=gene_id),
            )
        )

        # Create mRNA record
        gff_records.append(
            GffRecord(
                seqid=chrom_id,
                source=source,
                type=GffFeatureType.MRNA.value,
                start=gene_start + 1,
                end=gene_stop + 1,  # 1-based
                strand=strand_symbol,
                attributes=_create_gff_attributes(id=rna_id, parent_id=gene_id),
            )
        )

        # Create feature records (CDS, UTRs) - use original gene_group to include all features except when filtered
        feature_counters = {ftype: 0 for ftype in gff_feature_map.values()}
        for _, interval in gene_group.iterrows():
            entity_name = interval["entity_name"]
            gff_type = gff_feature_map.get(entity_name)
            if not gff_type:
                continue
            feature_counters[gff_type] += 1
            feature_id = f"{rna_id}.{gff_type}.{feature_counters[gff_type]}"

            gff_records.append(
                GffRecord(
                    seqid=chrom_id,
                    source=source,
                    type=gff_type,
                    start=int(interval["start"]) + 1,  # 1-based
                    end=int(interval["stop"]) + 1,  # 1-based
                    strand=strand_symbol,
                    phase=None,  # Phase remains None
                    attributes=_create_gff_attributes(id=feature_id, parent_id=rna_id),
                )
            )

    # Convert records to strings and write file
    gff_lines = ["##gff-version 3"] + [rec.to_line() for rec in gff_records]
    logger.info(f"Writing {len(gff_lines)} lines to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(gff_lines) + "\n")


def export_gff(
    input_zarr: str,
    output: str,
    tqdm_position: int | None,
    min_transcript_length: int,
    cpu_workers: int,
):
    """Convert interval predictions to a GFF3 file.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments specifying inputs, outputs, and filters.
    """
    logger.info(f"Loading predictions from {input_zarr}")
    # Use DataTree to easily access attributes and specific groups
    predictions: xr.DataTree = open_datatree(input_zarr, consolidated=False)
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

    # Group intervals by transcript
    genes = group_intervals_by_transcript(
        intervals_table,
        min_transcript_length,
        cpu_workers=cpu_workers,
        tqdm_position=tqdm_position,
        tqdm_desc=f"[GFF {chrom_id}]",
    )

    # Generate and save GFF
    generate_gff(
        genes,
        chrom_id,
        output,
        strip_introns=True,
        tqdm_position=tqdm_position,
        tqdm_desc=f"[GFF write {chrom_id}]",
    )

    logger.info("GFF export complete")


def main():
    """Main entry point for prediction and export functions."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Suppress noisy HTTP traffic logs from HuggingFace Hub's internal HTTP client
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Export interval predictions to GFF format"
    )

    parser.add_argument(
        "--input-zarr",
        "-i",
        type=str,
        default=None,
        help="Path to input zarr dataset from run_inference",
    )
    parser.add_argument(
        "--output-gff", "-o", default=None, type=str, help="Path to output GFF file"
    )

    parser.add_argument(
        "--manifest",
        default=None,
        type=str,
        help="Manifest json for multi-chromosome runs. "
        "Key-value pairs 'chromosome_id', 'intervals_zarr' "
        "and 'raw_gff' are required. Required if "
        "--input-dir and --output-zarr are not specified.",
    )

    # TODO: if we have a separate filter step for this, why is it here now? Redundant
    parser.add_argument(
        "--min-transcript-length",
        type=int,
        default=0,
        help="Minimum transcript length (default: 0, no filtering)",
    )

    parser.add_argument(
        "--tqdm-position",
        type=int,
        default=None,
        help="Optional tqdm row to use for export when multiple jobs run in one terminal",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=1,
        help="CPU worker processes for transcript grouping (default: 1)",
    )

    args = parser.parse_args()

    if args.manifest is None:
        if (args.input_zarr is None) or (args.output_gff is None):
            logger.error(
                "Error: one of the following must be provided:\n"
                "--manifest\n OR \n --input-zarr and --output-gff"
            )
            raise RuntimeError

        export_gff(
            input_zarr=args.input_zarr,
            output=args.output_gff,
            tqdm_position=args.tqdm_position,
            min_transcript_length=args.min_transcript_length,
            cpu_workers=args.cpu_workers,
        )

    else:
        with open(args.manifest) as fh:
            entries = json.load(fh)

        for entry in entries:
            chromosome_id = entry["chromosome_id"]
            input_zarr = entry["intervals_zarr"]
            output_gff = entry["raw_gff"]

            logger.info(f"Exporting raw gff for chromosome {chromosome_id}")

            export_gff(
                input_zarr=input_zarr,
                output=output_gff,
                tqdm_position=args.tqdm_position,
                min_transcript_length=args.min_transcript_length,
                cpu_workers=args.cpu_workers,
            )


if __name__ == "__main__":
    main()
