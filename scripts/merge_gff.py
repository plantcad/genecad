#!/usr/bin/env python3
"""
Merge multiple per-chromosome GFF files into a single GFF file with unique,
chromosome-prefixed gene IDs.

Usage:
    python merge_gff.py --output merged.gff --inputs chr1.gff chr2.gff ...

ID transformation examples:
    seqid=1:   ID=gene_1   -> ID=1_gene_1
    seqid=chr1: ID=gene_1  -> ID=chr1_gene_1
"""

import argparse
import json
import re
import sys
from pathlib import Path


def prefix_ids_in_attributes(attributes: str, chrom_prefix: str) -> str:
    """Prefix all ID= and Parent= values in a GFF attributes string."""
    # Prefix ID=...
    attributes = re.sub(
        r"ID=([^;]+)",
        lambda m: f"ID={chrom_prefix}_{m.group(1)}",
        attributes,
    )
    # Prefix Parent=...
    attributes = re.sub(
        r"Parent=([^;]+)",
        lambda m: f"Parent={chrom_prefix}_{m.group(1)}",
        attributes,
    )
    return attributes


def merge_gff_files(input_files: list[str], output_file: str) -> None:
    """Merge multiple GFF files, prefixing IDs with chromosome name."""
    header_written = False
    total_genes = 0
    total_lines = 0

    with open(output_file, "w") as out_fh:
        for input_path in input_files:
            path = Path(input_path)
            if not path.exists():
                print(f"WARNING: Skipping missing file: {input_path}", file=sys.stderr)
                continue

            file_lines = 0
            with open(path, "r") as in_fh:
                for line in in_fh:
                    line = line.rstrip("\n")

                    # Handle header/comment lines
                    if line.startswith("#"):
                        if not header_written:
                            out_fh.write(line + "\n")
                        continue

                    # Skip empty lines
                    if not line.strip():
                        continue

                    parts = line.split("\t")
                    if len(parts) < 9:
                        # Non-standard line, write as-is
                        out_fh.write(line + "\n")
                        continue

                    # Extract chromosome name from seqid (column 1)
                    chrom = parts[0]

                    # Prefix IDs in the attributes column (column 9)
                    parts[8] = prefix_ids_in_attributes(parts[8], chrom)

                    out_fh.write("\t".join(parts) + "\n")
                    file_lines += 1

                    if parts[2] == "gene":
                        total_genes += 1

            header_written = True
            total_lines += file_lines
            print(f"Processed {input_path}: {file_lines} feature lines")

    print(f"\nMerge complete: {output_file}")
    print(f"  Total feature lines: {total_lines}")
    print(f"  Total genes: {total_genes}")
    print(f"  Input files: {len(input_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-chromosome GFF files with unique chromosome-prefixed IDs"
    )
    parser.add_argument(
        "--output-gff", "-o", required=True, help="Path to the merged output GFF file"
    )
    parser.add_argument(
        "--input-gffs",
        "-i",
        nargs="+",
        default=None,
        help="Paths to per-chromosome GFF files",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        type=str,
        help="Manifest json, alternative to --input-gffs. Key 'filtered_gff' is required, or "
        "'raw_gff' if --use-raw-gffs flag is set.",
    )

    parser.add_argument(
        "--use-raw-gffs",
        action="store_true",
        help="Set this flag to use the 'raw_gff' files in the manifest instead of 'filtered_gff'.",
    )

    args = parser.parse_args()

    if args.manifest is None:
        if args.input_gffs is None:
            print(
                "Error: one of the following must be provided:\n"
                "--manifest\n OR \n --input-gffs"
            )
            raise RuntimeError

        merge_gff_files(args.input_gffs, args.output_gff)

    else:
        with open(args.manifest) as fh:
            entries = json.load(fh)

        if args.use_raw_gffs:
            input_gffs = [entry["raw_gff"] for entry in entries]
        else:
            input_gffs = [entry["filtered_gff"] for entry in entries]

        merge_gff_files(input_gffs, args.output_gff)


if __name__ == "__main__":
    main()
