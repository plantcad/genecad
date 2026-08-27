# Detailed documentation for fix_orf.py

Step [7/8] of the prediction pipeline repairs CDS boundaries in GeneCAD predictions against the
genome sequence, so that each transcript's CDS is a valid, translatable ORF (starts on ATG, ends
on a stop codon, is a multiple of three, and contains no in-frame stop).

> [!NOTE]
> GeneCAD predicts per-base feature labels without reading the underlying nucleotide sequence, so
> a predicted CDS is not guaranteed to already be a valid ORF.

```
python fix_orf.py \
--input-gff genecad_raw.gff \
--input-fasta genome.fa \
--output-gff genecad_orf.gff \
--report genecad_orf_report.tsv
```

### Parameters

* `--input-gff`, `-i` - Input GFF3 file. Output from `merge_gff.py`
* `--input-fasta`, `-f` - Genome FASTA file
* `--output-gff`, `-o` - Output GFF3 file
* `--max-shift` - Maximum movement (nt, in spliced transcript coordinates) allowed for the TIS and
for the TTS. Bounds how far a repair may depart from the model's prediction, and prevents long CDS
being truncated to short ORFs. Default 300.
* `--min-protein-length` - Minimum protein length in residues (excluding the stop codon) for a
repair. Default 10.
* `--allow-noncanonical-introns` - Flag. Attempt repair even when the transcript has non-canonical
introns. Off by default: an ORF built on untrustworthy splice calls is not trustworthy.
* `--no-fix-weak-starts` - Flag. Disables Kozak-context re-ranking of already-valid but
suspiciously short first exons. On by default.
* `--weak-start-threshold` - First coding exon length (nt) below which alternative start codons are
considered, when weak-start fixing is not disabled. Default 9.
* `--kozak-margin` - Minimum Kozak log2-odds advantage an alternative start codon must have over
the original to trigger a switch. Used as the floor value that per-genome calibration raises from,
unless `--no-calibrate-kozak-margin` is set, in which case it is used as-is. Default 3.0.
* `--no-calibrate-kozak-margin` - Flag. Uses `--kozak-margin` as a fixed value for every genome
instead of raising it per genome from that genome's own confident (unambiguous) start codons. On
by default.
* `--weak-kozak-threshold` - Kozak log2-odds score below which even the best candidate start is
flagged (`orf_issue=weak_kozak_support`) rather than kept silently. Default 5.0.
* `--report` - Optional TSV path for per-transcript status output.

### A Note on Repair Constraints

The repair is deliberately constrained so that it cannot invent gene structure:

1. Exon structure is frozen. Every splice junction in the output is a junction the model itself
predicted; introns are never moved, split, merged, or created.
2. Only the TIS and TTS move, and only within the spliced mRNA — i.e. only across sequence the
model already called exonic. A repair can relabel predicted UTR as CDS or predicted CDS as UTR,
but it can never pull in intronic or intergenic sequence.
3. Internal introns must be canonical (GT-AG / GC-AG / AT-AC) for a repair to be attempted, unless
`--allow-noncanonical-introns` is set.
4. The repaired ORF must be fully valid: ATG start, length divisible by 3, no in-frame stop, stop
codon end.
5. Among valid solutions, the one closest to the original prediction wins (minimum total boundary
movement), capped by `--max-shift`.

Transcripts that cannot be repaired under these rules are passed through unchanged and flagged
(`partial=true`, `orf_issue=…`, plus GFF3 `start_range`/`end_range`) so downstream steps can filter
them.

### Next Step

`python refine.py` [Refine Documentation](refine.md)
