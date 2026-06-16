# Detailed documentation for extract_fasta.py

Step [1/7] of the prediction pipeline prepares fasta sequences for inference.

```
python extract_fasta.py \
--species-id species_name \
--input-fasta path/to/fasta.fa \
--model-path emarro/pcad2-200M-cnet-baseline \
--output-zarr path/to/output/sequence.zarr
```

### Parameters

* `--species-id` - The name of the species or sample.
This is used primarily in file names and metadata tags.
* `--input-fasta`, `-i` - Path to the fasta file to be processed.
* `--model-path` - Path to the base PlantCAD model being used.
Both absolute local paths and HuggingFace repository paths are supported.
For help determining which model to use, see [Available GeneCAD Models]
> [!NOTE]
> Use the same GeneCAD and PlantCAD models through all steps in the pipeline.
* `--output-zarr`, `-o` - path to the output sequence dataset file. The output
is in .zarr format and will appear as a directory.
* `--chrom-map` - (Optional) Map to rename chromosomes and/or select a
subset of chromosomes. If set, chromosomes will be renamed according to the map,
and chromosome names that do not appear in the map will be skipped. If not set,
all chromosomes will be prepared and will retain their original names. The map should
be formatted as a comma-separated list of key-value pairs, e.g.
`oldChr1:newChr1,oldChr2:newChr2,oldChr3:newChr3...`

### Next Step

Optional: If your fasta file has multiple contigs, use `create_manifest.py` to
process all contigs at once. Otherwise, you will have to run through the pipeline steps
independently for each contig. [Create Manifest Documentation](create_manifest.md)

If you are only processing one contig or would like to process contigs independently, move
on to `python predict.py` [Predict Documentation](predict.md)
