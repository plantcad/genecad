# Detailed documentation for refine.py

Step [8/8] of the prediction pipeline refines GeneCAD predictions by removing gene models that are unlikely
to be functional protein coding genes using ReelProtein, and by attempting to merge fragmented gene models.

> [!NOTE]
> This script requires a CUDA GPU

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
* `--reelprotein-model-path` - HuggingFace repository ID for the ReelProtein model used to
score gene models. Default is `plantcad/reelprotein`
* `--gpus` - Comma-separated list of GPU IDs to use for generating ProtT5 embeddings for ReelProtein analysis.
Default is 0 (Runs on the first GPU on the machine)
