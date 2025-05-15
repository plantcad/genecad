import argparse
import gff_parser
import evaluation
import sys
from Bio import SeqRecord, Seq
import logging

logger = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ref-gff", required=True, type=str, help="Reference GFF file")
    parser.add_argument("-q", "--query-gff", required=True, type=str, help="Query GFF file")
    parser.add_argument("-o", "--output", required=True, type=str, help="output file")
    parser.add_argument("--tolerance", type=int, default=0, help="tolerance for transcript start/end")
    parser.add_argument("--ignore-unmatched-contigs", default=False, action="store_true",
                        help="ignore contigs that are not present in both gff files")
    return parser.parse_args()


def write_stats(master_stats: evaluation.Stats, out_handle: str):
    with open(out_handle, "w") as out_handle:
        out_handle.write("summary of comparison\n")

        out_handle.write("\ntranscript_cds:\n")
        out_handle.write(master_stats.transcript_cds.to_f1_string())


        out_handle.write("\ntranscript_intron:\n")
        out_handle.write(master_stats.transcript_intron.to_f1_string())

        out_handle.write("\ntranscript:\n")
        out_handle.write(master_stats.transcript.to_f1_string())

        out_handle.write("\nexon_cds_longest_transcript_only:\n")
        out_handle.write(master_stats.exon_cds_longest.to_f1_string())

        out_handle.write("\nexon_longest_transcript_only:\n")
        out_handle.write(master_stats.exon_longest.to_f1_string())

        out_handle.write("\nintron_cds_longest_transcript_only:\n")
        out_handle.write(master_stats.intron_cds_longest.to_f1_string())

        out_handle.write("\nintron_longest_transcript_only:\n")
        out_handle.write(master_stats.intron_longest.to_f1_string())

        out_handle.write("\nexon_cds:\n")
        out_handle.write(master_stats.exon_cds.to_f1_string())

        out_handle.write("\nexon:\n")
        out_handle.write(master_stats.exon.to_f1_string())

        out_handle.write("\nintron_cds:\n")
        out_handle.write(master_stats.intron_cds.to_f1_string())

        out_handle.write("\nintron:\n")
        out_handle.write(master_stats.intron.to_f1_string())

        out_handle.write("\nbase_cds:\n")
        out_handle.write(master_stats.base_cds.to_f1_string())

        out_handle.write("\nbase_utr:\n")
        out_handle.write(master_stats.base_utr.to_f1_string())

        out_handle.write("\nbase_exon:\n")
        out_handle.write(master_stats.base_exon.to_f1_string())

def calculate_contig_and_strand(pred_contig, true_contig, edge_tolerance,  strand: int = 1):
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

def evaluate_gff(pred_handle, true_handle, edge_tolerance, ignore_unmatched=True):
    pred_contigs = {}
    true_contigs = {}

    logging.info("Parsing GFF files")
    with open(pred_handle) as pred_file:
        for contig in gff_parser.parse(pred_file):
            pred_contigs[contig.id] = contig

    with open(true_handle) as true_file:
        for contig in gff_parser.parse(true_file):
            true_contigs[contig.id] = contig

    # TODO check that GFFs are sorted

    # check that contig names match
    pred_keys = set(pred_contigs.keys())
    true_keys = set(true_contigs.keys())

    if ignore_unmatched:
        use_keys = pred_keys.intersection(true_keys)
    else:
        use_keys = pred_keys.union(true_keys)

    if len(pred_keys.intersection(true_keys)) == 0:
        logger.warning("Reference and predicted GFFs do not share contigs. Check contig names.")
        sys.exit(1)
    elif pred_keys != true_keys and not ignore_unmatched:
        logger.warning("Warning: reference and predicted GFFs have different contig names. "
                      "These will still be evaluated. "
                      "Use --ignore-unmatched-contigs to skip contigs that are not present in both files")

    master_stats = evaluation.blank_stats()

    for contig_id in use_keys:
        logger.info("Processing contig " + contig_id)

        if contig_id in pred_contigs:
            pred_contig = pred_contigs[contig_id]
        else:
            pred_contig = SeqRecord.SeqRecord(seq=Seq.Seq("NNN"), id="None", features=[])

        if contig_id in true_contigs:
            true_contig = true_contigs[contig_id]
        else:
            true_contig = SeqRecord.SeqRecord(seq=Seq.Seq("NNN"), id="None", features=[])

        master_stats += calculate_contig_and_strand(pred_contig, true_contig, edge_tolerance, 1)

        master_stats += calculate_contig_and_strand(pred_contig, true_contig, edge_tolerance, -1)

    return master_stats



if __name__ == "__main__":
    args = parse_args()
    pred_handle = args.query_gff
    true_handle = args.ref_gff
    out_handle = args.output
    edge_tolerance = args.tolerance
    ignore_unmatched = args.ignore_unmatched_contigs

    master_stats = evaluate_gff(pred_handle, true_handle, edge_tolerance, ignore_unmatched)

    logger.info("Writing")
    write_stats(master_stats, out_handle)
    logger.info("Finished!")

