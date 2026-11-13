# Paired training comparisons

Use this protocol to test whether a change to the training data improves GeneCAD, for example removing
low-quality gene models. Each condition is trained on the same windows, from the same initial weights and
for the same number of optimizer updates, so the conditions differ only in which genes are labelled and
which positions count in the loss. Any difference in accuracy can then be attributed to the gene selection.

The steps below call the lower-level scripts `scripts/transform.py`, `scripts/sample.py` and
`scripts/train.py`. The plant pipeline (`train.sh` or `genecad train`) passes the same settings through
`--exclude-genes`, `--windows-from`, `--initial-state`, `--seed`, `--max-steps` and
`--base-encoder-revision`; its default `revision` preset uses the settings of the published model (see the
README).

## Overview

1. Write the genes to exclude for each condition to a table.
2. Mask those genes when the labels are built.
3. Sample windows once, from a reference condition, and reuse them for every other condition.
4. Export the initial weights once per training seed.
5. Train every condition from those weights for a fixed number of optimizer updates.

## 1. List the excluded genes

For each condition, write a tab-separated file with the columns `species_id` and `gene_id`, one row per gene
to remove. A condition that keeps every gene needs no file.

## 2. Mask the excluded genes

Pass the file to the `filter_features` subcommand of `scripts/transform.py`:

```bash
python scripts/transform.py filter_features ... --exclude-genes selection.tsv
```

Matching genes are removed with the reason `revision_selection`, and their spans are kept so that they can
be masked from the loss. An excluded gene is therefore ignored during training. It is not relabelled as
intergenic sequence.

## 3. Reuse one set of windows

Sample windows once, from the reference condition. For every other condition, pass those windows to the
`generate_training_windows` subcommand of `scripts/sample.py`:

```bash
python scripts/sample.py generate_training_windows ... --windows-from /path/to/reference-windows.zarr
```

Reused windows keep the reference order and strand, including the orientation of negative-strand windows, and
are kept even when every position in them is masked. Sampling stops with an error if a window has a
different length, invalid bounds or strand, or different DNA, or if a chromosome that has reference windows
is missing from the input.

The last case can happen when a selection removes every gene on a chromosome, because labels are only built
for chromosomes with retained genes. Rebuild the labels and masks of that chromosome and run again. Do not
delete its windows from the reference to get past the error.

Use the same species configuration, validation proportion and split seed for all conditions. The split
sorts species and chromosomes before drawing random numbers, so the order in which files are listed does
not change it. Before the full experiment, compare window coordinates, sample indices and excluded spans
across the final datasets.

## 4. Export the initial weights

Export the initial weights once per training seed, from the reference condition:

```bash
python scripts/train.py \
  --train-dataset /path/to/reference-train.zarr \
  --val-dataset /path/to/reference-valid.zarr \
  --output-dir /path/to/export-log \
  --seed 11 --base-encoder-revision 5574ff5dd1795fe58ac929bd89a7f41a07e05997 \
  --export-initial-state /path/to/seed11.pt
```

Add the architecture, window size, head, backbone and freezing options of the experiment. The command writes
the file and exits without training, and it will not overwrite an existing file. Under a distributed
launcher only rank zero writes it.

## 5. Train each condition

Train every condition with the same model options and seed, starting from the exported weights:

```bash
python scripts/train.py \
  --train-dataset /path/to/condition-train.zarr \
  --val-dataset /path/to/condition-valid.zarr \
  --output-dir /path/to/condition-run \
  --seed 11 --base-encoder-revision 5574ff5dd1795fe58ac929bd89a7f41a07e05997 \
  --initial-state /path/to/seed11.pt --auto-class-weights no \
  --max-steps 1753
```

When the weights are loaded, the seed, a checksum of the tensors and the relevant model settings are checked
against the export. All conditions use the class frequencies stored with the weights.
`initialization.json` in the run directory records the hash of the starting weights. `--initial-state`
cannot be combined with resuming from a checkpoint, `--auto-class-weights yes` or `--torch-compile yes`.

A positive `--max-steps` replaces the epoch limit and counts optimizer updates, including gradient
accumulation. A batch in which every position is masked contributes a zero loss: it still counts in the
update schedule, and multi-GPU training continues normally. If validation is used only for monitoring,
choose the final checkpoint the same way for every condition.

## What has been tested

Shared initialization, fully masked batches, gradient accumulation, fixed update counts across epochs and
resuming from a checkpoint were tested on H100 GPUs in BF16, including a two-GPU run in which one GPU received
only masked batches; both GPUs ended with identical parameters. The full model (PlantCAD2 backbone and
classifier head) was checked on short sequences on one GPU. These checks confirm correctness; they are not a
full-length training run or a multi-GPU run of the full model. For running GeneCAD on ARM64 GH200 nodes,
see [TACC Vista](tacc_vista.md).
