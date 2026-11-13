# Detailed documentation for refine.py

Step [8/8] of the prediction pipeline uses ReelProtein to decide whether adjacent gene fragments that
together form a complete ORF should be merged, and, with `--filter-unmerged`, to remove gene models that
are unlikely to be functional protein coding genes.

By default every gene model is kept, so only merge candidates are scored. ORF repair and hybrid decoding
usually leave no fragment to merge; ProtT5 then does not run and the output equals the input.
`--filter-unmerged` scores every protein (21–76 minutes of GPU time per genome on the five test genomes). It raises
precision and lowers recall, and removes some conserved genes: on five test genomes it removed 6–10% of
gene models, raised CDS locus F1 by 0.5–2.5 points and lowered BUSCO completeness by 0.7–1.3 points.

> [!NOTE]
> Scoring proteins requires a CUDA GPU

```
python refine.py \
--input-gff genecad_orf.gff \
--input-fasta path/to/fasta.fa
--output-gff genecad_filtered.gff
```

### Parameters

* `--input-gff`, `-i` - Input GFF file. Output from `fix_orf.py`
* `--input-fasta` `-f` - Input FASTA file for the sample
* `--output-gff`, `-o` - Output GFF file
* `--filter-unmerged` - Also remove gene models whose protein ReelProtein scores below 0.5 (Default: off)
* `--reelprotein-model-path` - HuggingFace repository ID for the ReelProtein model used to
score gene models. Default is `plantcad/reelprotein`
* `--gpus` - Comma-separated list of GPU IDs to use for generating ProtT5 embeddings for ReelProtein analysis.
Default is 0 (Runs on the first GPU on the machine)
