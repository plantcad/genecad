"""GFF evaluation functions from https://github.com/maize-genetics/reelAnnote-dev/pull/2"""

import os
import logging
import warnings
import src.gff_parser as gff_parser
import src.evaluation as evaluation
from Bio import SeqRecord, Seq

logger = logging.getLogger(__name__)


def write_stats(
    master_stats: evaluation.Stats, out_handle: str, as_percentage: bool = True
):
    """
    Write a set of statistics to the given file.

    Stats are written as precision, recall, and F1 scores. By default, these values
    are multiplied by 100 to display as percentages for consistency with gffcompare
    output format.

    Args:
        master_stats: statistics
        out_handle: file name
        as_percentage: whether to display values as percentages (0-100) or decimals (0.0-1.0).
                      Defaults to True for consistency with gffcompare.
    """
    with open(out_handle, "w") as out_handle:
        out_handle.write("level\tprecision\trecall\tf1\n")

        out_handle.write("transcript_cds\t")
        out_handle.write(master_stats.transcript_cds.to_score_string(as_percentage))

        out_handle.write("transcript_intron\t")
        out_handle.write(master_stats.transcript_intron.to_score_string(as_percentage))

        out_handle.write("transcript\t")
        out_handle.write(master_stats.transcript.to_score_string(as_percentage))

        out_handle.write("exon_cds_longest_transcript_only\t")
        out_handle.write(master_stats.exon_cds_longest.to_score_string(as_percentage))

        out_handle.write("exon_longest_transcript_only\t")
        out_handle.write(master_stats.exon_longest.to_score_string(as_percentage))

        out_handle.write("intron_cds_longest_transcript_only\t")
        out_handle.write(master_stats.intron_cds_longest.to_score_string(as_percentage))

        out_handle.write("intron_longest_transcript_only\t")
        out_handle.write(master_stats.intron_longest.to_score_string(as_percentage))

        out_handle.write("exon_cds\t")
        out_handle.write(master_stats.exon_cds.to_score_string(as_percentage))

        out_handle.write("exon\t")
        out_handle.write(master_stats.exon.to_score_string(as_percentage))

        out_handle.write("intron_cds\t")
        out_handle.write(master_stats.intron_cds.to_score_string(as_percentage))

        out_handle.write("intron\t")
        out_handle.write(master_stats.intron.to_score_string(as_percentage))

        out_handle.write("base_cds\t")
        out_handle.write(master_stats.base_cds.to_score_string(as_percentage))

        out_handle.write("base_utr\t")
        out_handle.write(master_stats.base_utr.to_score_string(as_percentage))

        out_handle.write("base_exon\t")
        out_handle.write(master_stats.base_exon.to_score_string(as_percentage))


def calc_stats_for_contig_and_strand(
    pred_contig: SeqRecord,
    true_contig: SeqRecord,
    edge_tolerance: int = 0,
    strand: int = 1,
) -> evaluation.Stats:
    """
    Walk through a given pair of contigs and calculate the F1 stats for all genes on those contigs and the given strand

    Args:
        pred_contig: SeqRecord of the predicted genes for a contig
        true_contig: SeqRecord of the true genes for a contig
        edge_tolerance: tolerance to allow for matching transcript ends. Default 0
        strand: strand to evaluate. Default 1 (positive)

    Returns:
        Stats object for whole contig
    """

    master_stats = evaluation.blank_stats()

    next_pidx = 0
    next_tidx = 0

    while next_pidx < len(pred_contig) and next_tidx < len(true_contig):
        pred_indices, true_indices, next_pidx, next_tidx = (
            evaluation.get_next_overlap_set(
                pred_contig.features, true_contig.features, next_pidx, next_tidx, strand
            )
        )

        if len(pred_indices) == 0 and len(true_indices) == 0:
            break

        temp = evaluation.overlap_stats(
            pred_contig.features,
            true_contig.features,
            pred_indices,
            true_indices,
            edge_tolerance,
        )

        master_stats += temp

    return master_stats


def test_sorted(contig: SeqRecord) -> bool:
    for idx in range(len(contig.features) - 1):
        if (
            contig.features[idx].location.start
            > contig.features[idx + 1].location.start
        ):
            return False

    return True


def evaluate_gff(
    pred_handle, true_handle, edge_tolerance, ignore_unmatched=True
) -> evaluation.Stats:
    """
    Calculate precision, recall, and F1 scores on various levels for two GFF files.

    Args:
        pred_handle: filename of predicted gff file
        true_handle: filename of true gff file
        edge_tolerance: tolerance to allow for matching transcript ends
        ignore_unmatched: ignore contigs that aren't present in both gff files

    Returns:
        stats object for the whole genome
    """
    pred_contigs = {}
    true_contigs = {}

    logger.info("Parsing GFF files")
    with open(pred_handle) as pred_file:
        for contig in gff_parser.parse(pred_file):
            pred_contigs[contig.id] = contig

            if not test_sorted(contig):
                error_msg = f"Contig {contig.id} in predicted GFF is not sorted by gene start. Please sort the file before evaluation."
                raise ValueError(error_msg)

    with open(true_handle) as true_file:
        for contig in gff_parser.parse(true_file):
            true_contigs[contig.id] = contig

            if not test_sorted(contig):
                error_msg = f"Contig {contig.id} in reference GFF is not sorted by gene start. Please sort the file before evaluation."
                raise ValueError(error_msg)

    # check that contig names match
    pred_keys = set(pred_contigs.keys())
    true_keys = set(true_contigs.keys())

    if ignore_unmatched:
        use_keys = pred_keys.intersection(true_keys)
    else:
        use_keys = pred_keys.union(true_keys)

    if len(pred_keys.intersection(true_keys)) == 0:
        error_msg = f"Reference and predicted GFFs do not share any contigs. Check contig names. Reference contigs: {sorted(true_keys)}, Predicted contigs: {sorted(pred_keys)}"
        raise ValueError(error_msg)
    elif pred_keys != true_keys and not ignore_unmatched:
        contigs_only_in_reference = sorted(true_keys - pred_keys)
        contigs_only_in_predicted = sorted(pred_keys - true_keys)
        contigs_in_both = sorted(pred_keys.intersection(true_keys))

        warning_msg = (
            "Warning: reference and predicted GFFs have different contig names. "
            "These will still be evaluated. "
            "Use --ignore-unmatched-contigs to skip contigs that are not present in both files.\n"
            f"Contigs in both files ({len(contigs_in_both)}): {contigs_in_both}\n"
            f"Contigs only in reference ({len(contigs_only_in_reference)}): {contigs_only_in_reference}\n"
            f"Contigs only in predicted ({len(contigs_only_in_predicted)}): {contigs_only_in_predicted}"
        )
        warnings.warn(warning_msg)

    master_stats = evaluation.blank_stats()

    for contig_id in use_keys:
        logger.info(f"Processing contig {contig_id}")

        if contig_id in pred_contigs:
            pred_contig = pred_contigs[contig_id]
        else:
            pred_contig = SeqRecord.SeqRecord(
                seq=Seq.Seq("NNN"), id="None", features=[]
            )

        if contig_id in true_contigs:
            true_contig = true_contigs[contig_id]
        else:
            true_contig = SeqRecord.SeqRecord(
                seq=Seq.Seq("NNN"), id="None", features=[]
            )

        master_stats += calc_stats_for_contig_and_strand(
            pred_contig, true_contig, edge_tolerance, 1
        )

        master_stats += calc_stats_for_contig_and_strand(
            pred_contig, true_contig, edge_tolerance, -1
        )

    return master_stats


def run_gffeval(
    reference_path: str,
    input_path: str,
    output_dir: str,
    edge_tolerance: int = 0,
    ignore_unmatched: bool = True,
    as_percentage: bool = True,
) -> str:
    """Run GFF evaluation to compare input GFF against reference using internal evaluation functions.

    Stats are written as precision, recall, and F1 scores. By default, these values
    are multiplied by 100 to display as percentages for consistency with gffcompare
    output format.

    Parameters
    ----------
    reference_path : str
        Path to reference GFF file
    input_path : str
        Path to input/query GFF file
    output_dir : str
        Directory to store evaluation results
    edge_tolerance : int, default 0
        Tolerance to allow for matching transcript ends
    ignore_unmatched : bool, default True
        Whether to ignore contigs that aren't present in both GFF files
    as_percentage : bool, default True
        Whether to display values as percentages (0-100) or decimals (0.0-1.0).
        Defaults to True for consistency with gffcompare.

    Returns
    -------
    str
        Output prefix (path without extension) used for evaluation output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set up the output prefix for evaluation files
    output_prefix = os.path.join(output_dir, "gffeval")

    logger.info(
        f"Running GFF evaluation: {input_path} vs {reference_path} ({edge_tolerance=}, {ignore_unmatched=}, {as_percentage=})"
    )

    # Run the evaluation using the existing evaluate_gff function
    master_stats = evaluate_gff(
        pred_handle=input_path,
        true_handle=reference_path,
        edge_tolerance=edge_tolerance,
        ignore_unmatched=ignore_unmatched,
    )

    # Write stats to output file
    stats_file = f"{output_prefix}.stats.tsv"
    write_stats(master_stats, stats_file, as_percentage)

    logger.info(f"Evaluation complete. Results written to {stats_file}")

    return output_prefix
