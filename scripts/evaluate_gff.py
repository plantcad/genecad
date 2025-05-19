import argparse
import src.gff_parser as gff_parser
import src.evaluation as evaluation
import sys
from Bio import SeqRecord, Seq
import warnings


def parse_args():
    """
        Parse script arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ref-gff", required=True, type=str, help="Reference GFF file")
    parser.add_argument("-q", "--query-gff", required=True, type=str, help="Query GFF file")
    parser.add_argument("-o", "--output", required=True, type=str, help="output file")
    parser.add_argument("--tolerance", type=int, default=0, help="tolerance for transcript start/end")
    parser.add_argument("--ignore-unmatched-contigs", default=False, action="store_true",
                        help="ignore contigs that are not present in both gff files")
    return parser.parse_args()


def write_stats(master_stats: evaluation.Stats, out_handle: str):
    """
    Write a set of statistics to the given file

    Args:
        master_stats: statistics
        out_handle: file name
    """
    with open(out_handle, "w") as out_handle:
        out_handle.write("summary of comparison\n")

        out_handle.write("\ntranscript_cds:\n")
        out_handle.write(master_stats.transcript_cds.to_score_string())

        out_handle.write("\ntranscript_intron:\n")
        out_handle.write(master_stats.transcript_intron.to_score_string())

        out_handle.write("\ntranscript:\n")
        out_handle.write(master_stats.transcript.to_score_string())

        out_handle.write("\nexon_cds_longest_transcript_only:\n")
        out_handle.write(master_stats.exon_cds_longest.to_score_string())

        out_handle.write("\nexon_longest_transcript_only:\n")
        out_handle.write(master_stats.exon_longest.to_score_string())

        out_handle.write("\nintron_cds_longest_transcript_only:\n")
        out_handle.write(master_stats.intron_cds_longest.to_score_string())

        out_handle.write("\nintron_longest_transcript_only:\n")
        out_handle.write(master_stats.intron_longest.to_score_string())

        out_handle.write("\nexon_cds:\n")
        out_handle.write(master_stats.exon_cds.to_score_string())

        out_handle.write("\nexon:\n")
        out_handle.write(master_stats.exon.to_score_string())

        out_handle.write("\nintron_cds:\n")
        out_handle.write(master_stats.intron_cds.to_score_string())

        out_handle.write("\nintron:\n")
        out_handle.write(master_stats.intron.to_score_string())

        out_handle.write("\nbase_cds:\n")
        out_handle.write(master_stats.base_cds.to_score_string())

        out_handle.write("\nbase_utr:\n")
        out_handle.write(master_stats.base_utr.to_score_string())

        out_handle.write("\nbase_exon:\n")
        out_handle.write(master_stats.base_exon.to_score_string())


def calc_stats_for_contig_and_strand(pred_contig: SeqRecord, true_contig: SeqRecord,
                                     edge_tolerance: int = 0, strand: int = 1) -> evaluation.Stats:
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
        pred_indices, true_indices, next_pidx, next_tidx = evaluation.get_next_overlap_set(pred_contig.features,
                                                                                           true_contig.features,
                                                                                           next_pidx, next_tidx, strand)

        if len(pred_indices) == 0 and len(true_indices) == 0:
            break

        temp = evaluation.overlap_stats(pred_contig.features, true_contig.features, pred_indices, true_indices,
                                        edge_tolerance)

        master_stats += temp

    return master_stats


def test_sorted(contig: SeqRecord) -> bool:

    for idx in range(len(contig.features) - 1):
        if contig.features[idx].location.start > contig.features[idx+1].location.start:
            return False

    return True


def evaluate_gff(pred_handle, true_handle, edge_tolerance, ignore_unmatched=True) -> evaluation.Stats:
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

    print("Parsing GFF files")
    with open(pred_handle) as pred_file:
        for contig in gff_parser.parse(pred_file):
            pred_contigs[contig.id] = contig

            if not test_sorted(contig):
                warnings.warn("Contig " + contig.id + " is not sorted by gene start. Please sort file.")
                sys.exit()

    with open(true_handle) as true_file:
        for contig in gff_parser.parse(true_file):
            true_contigs[contig.id] = contig

            if not test_sorted(contig):
                warnings.warn("Contig " + contig.id + " is not sorted by gene start. Please sort file.")
                sys.exit()

    # check that contig names match
    pred_keys = set(pred_contigs.keys())
    true_keys = set(true_contigs.keys())

    if ignore_unmatched:
        use_keys = pred_keys.intersection(true_keys)
    else:
        use_keys = pred_keys.union(true_keys)

    if len(pred_keys.intersection(true_keys)) == 0:
        warnings.warn("Reference and predicted GFFs do not share contigs. Check contig names.")
        sys.exit(1)
    elif pred_keys != true_keys and not ignore_unmatched:
        warnings.warn("Warning: reference and predicted GFFs have different contig names. "
                       "These will still be evaluated. "
                       "Use --ignore-unmatched-contigs to skip contigs that are not present in both files")

    master_stats = evaluation.blank_stats()

    for contig_id in use_keys:
        print("Processing contig " + contig_id)

        if contig_id in pred_contigs:
            pred_contig = pred_contigs[contig_id]
        else:
            pred_contig = SeqRecord.SeqRecord(seq=Seq.Seq("NNN"), id="None", features=[])

        if contig_id in true_contigs:
            true_contig = true_contigs[contig_id]
        else:
            true_contig = SeqRecord.SeqRecord(seq=Seq.Seq("NNN"), id="None", features=[])

        master_stats += calc_stats_for_contig_and_strand(pred_contig, true_contig, edge_tolerance, 1)

        master_stats += calc_stats_for_contig_and_strand(pred_contig, true_contig, edge_tolerance, -1)

    return master_stats


if __name__ == "__main__":
    args = parse_args()
    pred_handle = args.query_gff
    true_handle = args.ref_gff
    out_handle = args.output
    edge_tolerance = args.tolerance
    ignore_unmatched = args.ignore_unmatched_contigs

    master_stats = evaluate_gff(pred_handle, true_handle, edge_tolerance, ignore_unmatched)

    print("Writing")
    write_stats(master_stats, out_handle)
    print("Finished!")
