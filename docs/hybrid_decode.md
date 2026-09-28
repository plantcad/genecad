# Detailed documentation for hybrid_decode.py

With `--decoder hybrid` (the default), step [7/8] of the prediction pipeline continues after ORF repair
by repairing the remaining gene models with frame-aware decoding of local windows.

> [!NOTE]
> Frame-aware decoding of whole chromosomes guarantees every CDS is a valid ORF, but forcing a reading
> frame everywhere also creates new loci made of tiny ORFs and cuts long genes into several ORFs. The
> original 5-state Viterbi creates neither, but leaves some genes without a valid ORF. Hybrid decoding
> keeps the 5-state decode and uses frame-aware decoding only where a gene already exists, so it cannot
> create new loci.

Starting from the 5-state decode after `fix_orf.py` (partial transcripts flagged, not dropped):

1. **Rescue.** Every transcript `fix_orf.py` could not repair (`partial=true`) is decoded again,
frame-aware, in a window of `--flank` bases around it, on the model's cached feature logits. A decoded
transcript replaces it (keeping its gene ID, with `hybrid=rescued`) if it overlaps it, has more than
`--min-cds` nt of CDS and does not overlap a kept gene on the same strand. Partial genes that cannot be
rescued are dropped, or kept flagged with `--keep-partial`.
2. **Merge.** Each pair of consecutive same-strand genes at most `--max-gap` bp apart is decoded again,
frame-aware, with intergenic forbidden between them. If that yields one transcript spanning the CDS
midpoints of both, it replaces the pair (keeping the first gene's ID, with `hybrid=merged` and
`merged_from=`), and may then be merged with the next gene.

Genes that are not changed are written back unchanged.

```
python hybrid_decode.py \
--input-gff genecad_orf.gff \
--input-fasta genome.fa \
--predictions-root OUTPUT_DIR \
--output-gff genecad_hybrid.gff
```

### Parameters

* `--input-gff` - GFF3 after `fix_orf.py` without `--drop-partial`, from 5-state decoding
* `--input-fasta` - Genome FASTA file used for prediction
* `--predictions-root` - Directory holding `<CHR_ID>/predictions_<CHR_ID>/` for each sequence, as written by
`predict.sh`. Sequences without predictions are written back unchanged.
* `--output-gff` - Output GFF3 file
* `--domain` - `plant` or `animal`; selects the feature transition matrix. Default plant.
* `--flank` - Bases added on each side of a decoding window. Default 1000.
* `--max-gap` - Largest gap (bp) between consecutive same-strand genes for a merge to be tried. Default 20000.
* `--min-cds` - A rescued or merged transcript needs more than this many nt of CDS. Default 33.
* `--keep-partial` - Keep partial genes that cannot be rescued (flagged) instead of dropping them.
* `--workers` - Sequences processed in parallel. Default 1.
* `--min-intron-length`, `--min-coding-run-length`, `--exon-length-strictness`, `--allow-u12-introns` - Frame-aware
decoding settings, as for `detect_intervals.py`.

### Validation

On maize NAM (Ky21 chromosomes 1-9, Tzi8 and Ms71; 29 chromosomes, none used during development),
against the official annotations, compared with whole-chromosome frame-aware decoding:

| | predicted genes | tiny (CDS <= 33 nt) | split reference genes | fused | exact CDS recall | exact CDS precision |
|---|---|---|---|---|---|---|
| frame-aware (v0.5.0) | 140,549 | 26,585 | 4,213 | 1,879 | 57.6% | 48.5% |
| frame-aware, partial dropped | 114,114 | 175 | 3,936 | 1,874 | 57.6% | 59.8% |
| hybrid | 110,901 | 192 | 2,385 | 1,979 | 58.6% | 62.5% |

Hybrid decoding had higher recall and precision, and fewer split genes, on each of the 29 chromosomes.
