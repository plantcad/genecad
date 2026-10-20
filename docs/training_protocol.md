# Paired training comparisons

The lower-level Python scripts support comparisons in which training conditions
differ only in labels and loss masks. These options belong to `scripts/sample.py`,
`scripts/transform.py`, and `scripts/train.py`; they are not automatically forwarded
by the `genecad train` shell pipeline.

## Exclusion masks and fixed windows

Pass `--exclude-genes selection.tsv` to the `filter_features` subcommand of
`scripts/transform.py`. The tab-separated file must contain `species_id` and
`gene_id`. Matching retained genes are removed with reason `revision_selection`;
their original gene spans are recorded for downstream loss masking. Removing a
gene must not turn its span into a supervised intergenic example.

Generate a reference set of windows once. For each alternative label set, pass
`--windows-from /path/to/reference-windows.zarr` to the
`generate_training_windows` subcommand of `scripts/sample.py`. Reused windows:

- Keep reference strand/position order, including negative-strand orientation.
- Retain windows even when every position is masked.
- Require matching window length, valid bounds/strands, and identical DNA tokens.
- Fail if a chromosome containing reference windows is absent from the input.

Label creation currently enumerates chromosomes with retained features. If a
selection removes every gene from a chromosome, that chromosome can disappear
from the label input. Fixed-window sampling detects this and stops; reconstruct
its sequence labels and exclusion masks before proceeding. Do not remove its
windows from the reference merely to make the check pass.

Use the same species configuration, validation proportion, and split seed for
all conditions. Splitting sorts species/chromosome groups before consuming its
random-number stream, so filesystem listing order cannot change the result.
Check window coordinates, sample indices, and exclusion spans across the final
datasets before a full experiment.

## Shared initialization and optimizer updates

Export initial weights once per training seed from the reference condition:

```bash
python scripts/train.py \
  --train-dataset /path/to/reference-train.zarr \
  --val-dataset /path/to/reference-valid.zarr \
  --output-dir /path/to/export-log \
  --seed 11 --base-encoder-revision 5574ff5dd1795fe58ac929bd89a7f41a07e05997 \
  --export-initial-state /path/to/seed11.pt
```

Add the same architecture, window size, head, backbone, and freeze options used
for the experiment. Export exits before fitting and refuses to overwrite the
initialization artifact. Under a distributed launcher, only global rank zero
writes the artifact.

For every condition, use those same model options and training seed, plus:

```bash
python scripts/train.py \
  --train-dataset /path/to/condition-train.zarr \
  --val-dataset /path/to/condition-valid.zarr \
  --output-dir /path/to/condition-run \
  --seed 11 --base-encoder-revision 5574ff5dd1795fe58ac929bd89a7f41a07e05997 \
  --initial-state /path/to/seed11.pt --auto-class-weights no \
  --max-steps 1753
```

Loading checks the seed, tensor checksum, and relevant model configuration.
All conditions reuse the artifact's class frequencies. Shared initialization
cannot be combined with a checkpoint, per-condition auto class weights, or
`--torch-compile yes`. `initialization.json` records the starting state hash.

A positive `--max-steps` replaces the epoch limit and counts optimizer updates,
including gradient accumulation. An all-masked batch contributes a differentiable
zero loss, allowing backward and DDP synchronization to continue. It still
participates in the fixed batch/update schedule. Choose the final checkpoint
consistently when validation is intended only for monitoring.

## GPU validation scope

On 2026-10-20, H100 BF16 checks passed for shared initialization, all-masked
batches, accumulation, fixed steps across epochs, and checkpoint resumption.
Two H100s passed a small-classifier NCCL DDP test where one rank received only
masked batches; final parameters were identical across ranks. A separate
single-H100 check exercised the real PlantCAD2 backbone and eight-layer
FlashAttention head with both backbone and classifier updates.

The complete model's new training check used 128-base sequences and two optimizer
updates. It is a correctness smoke test, not a full-length training qualification
or scaling benchmark. Full-model multi-GPU training was not tested in that check.
For the separate native ARM64 GH200 plant-inference validation, see
[TACC Vista](tacc_vista.md). The training changes do not by themselves provide
new GPU architecture support or remove native CUDA dependency requirements.
