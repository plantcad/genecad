# Detailed documentation for extract.py

## extract_gff_features

TODO - not used for inference

## extract_fasta_sequences

TODO - not used for inference

## extract_fasta_file

Prepares fasta sequence for inference.

```
python extract.py extract_fasta_file \
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
[!NOTE]
Use the same GeneCAD and PlantCAD models through all steps in the pipeline.
* `--output-zarr`, `-o` - path to the output sequence dataset file. The output
is in .zarr format and will appear as a directory.
* `--chrom-map` - (Optional) Map to rename chromosomes and/or select a
subset of chromosomes. If set, chromosomes will be renamed according to the map,
and chromosome names that do not appear in the map will be skipped. If not set,
all chromosomes will be prepared and will retain their original names. The map should
be formatted as a comma-separated list of key-value pairs, e.g.
`oldChr1:newChr1,oldChr2:newChr2,oldChr3:newChr3...`


## validate_configs

TODO - not used for inference
