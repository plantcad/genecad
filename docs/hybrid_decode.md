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

* `--input-gff` - GFF3 after `fix_orf.py --keep-partial`, from 5-state decoding
* `--input-fasta` - Genome FASTA file used for prediction
* `--predictions-root` - Directory holding `<CHR_ID>/predictions_<CHR_ID>/` for each sequence, as written by
`predict.sh`. A sequence without prediction files stops the run with an error, unless
`--allow-missing-predictions` is given.
* `--output-gff` - Output GFF3 file
* `--domain` - `plant` or `animal`; selects the feature transition matrix. Default plant.
* `--flank` - Bases added on each side of a decoding window. Default 1000.
* `--max-gap` - Largest gap (bp) between consecutive same-strand genes for a merge to be tried. Default 20000.
* `--min-cds` - A rescued or merged transcript needs more than this many nt of CDS. Default 33.
* `--keep-partial` - Keep partial genes that cannot be rescued (flagged) instead of dropping them.
* `--workers` - Sequences processed in parallel. Default 1.
* `--allow-missing-predictions` - Write sequences whose prediction files are missing or damaged back unchanged,
instead of stopping.
* `--min-intron-length`, `--min-coding-run-length`, `--exon-length-strictness`, `--allow-u12-introns` - Frame-aware
decoding settings, as for `detect_intervals.py`.

If prediction files fail verification (for example because some of them were deleted), the script removes their
completion marker (`_SUCCESS.json`) and exits with status 3, unless `--allow-missing-predictions` is given.
`predict.sh` then predicts the damaged parts again and runs this step once more.

### Validation

On maize NAM (Ky21 chromosomes 1-9, Tzi8 and Ms71; 29 chromosomes, none used during development),
against the official annotations, compared with whole-chromosome frame-aware decoding:

| | predicted genes | tiny (CDS <= 33 nt) | split reference genes | fused | exact CDS recall | exact CDS precision |
|---|---|---|---|---|---|---|
| frame-aware (v0.5.0) | 140,549 | 26,585 | 4,213 | 1,879 | 57.6% | 48.5% |
| frame-aware, partial dropped | 114,114 | 175 | 3,936 | 1,874 | 57.6% | 59.8% |
| hybrid | 110,901 | 192 | 2,385 | 1,979 | 58.6% | 62.5% |

Hybrid decoding had higher recall and precision, and fewer split genes, on each of the 29 chromosomes.

The same comparison on 11 more species, against their reference annotations. Frame-aware here
is v0.5.0 with partial transcripts dropped, so both methods leave out transcripts without a
valid ORF. Recall and precision are for exact CDS chains, in percent. Split and fused are counted
as in the table above.

| species | reference genes | recall frame-aware | recall hybrid | precision frame-aware | precision hybrid | split frame-aware | split hybrid | fused frame-aware | fused hybrid |
|---|---|---|---|---|---|---|---|---|---|
| Athaliana | 27,655 | 82.8 | 83.1 | 87.6 | 88.5 | 896 | 794 | 703 | 709 |
| Bstricta | 27,297 | 71.4 | 71.5 | 73.7 | 74.6 | 912 | 644 | 326 | 345 |
| Crubella | 27,643 | 78.8 | 79.0 | 80.5 | 81.3 | 554 | 433 | 405 | 411 |
| Csativus | 21,235 | 59.1 | 59.6 | 61.0 | 62.1 | 342 | 243 | 390 | 427 |
| Esalsugineum | 26,340 | 68.0 | 68.1 | 68.4 | 69.3 | 600 | 441 | 262 | 284 |
| Fvesca | 34,000 | 56.7 | 57.0 | 69.3 | 71.3 | 1,211 | 983 | 832 | 873 |
| Mesculenta | 32,794 | 74.8 | 75.6 | 77.4 | 79.4 | 1,085 | 789 | 611 | 610 |
| Othomaeum | 28,440 | 25.4 | 25.3 | 27.1 | 27.6 | 1,277 | 1,144 | 1,813 | 1,832 |
| Ppersica | 26,868 | 69.9 | 70.2 | 74.7 | 76.0 | 699 | 535 | 489 | 504 |
| Spolyrhiza | 19,623 | 32.2 | 32.0 | 37.0 | 37.2 | 447 | 348 | 915 | 991 |
| Zmarina | 21,459 | 57.8 | 58.1 | 61.7 | 63.0 | 880 | 715 | 335 | 354 |

Hybrid decoding had higher precision and fewer split genes in all 11 species, and higher recall
in 9 (0.1 and 0.2 points lower in Othomaeum and Spolyrhiza). It gave more fused genes in 10 of
the 11 (1% to 10% more), which is the price of fewer split genes. Tiny genes fell from between
48 and 1,583 per species with v0.5.0 to between 4 and 28.
