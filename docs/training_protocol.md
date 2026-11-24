# Paired training comparisons

This guide shows how to test whether a change to the training data improves GeneCAD, for example removing
low-quality gene models. The idea is to train each condition on the same windows, from the same initial weights
and for the same number of optimizer updates. The conditions then differ only in which genes are labelled and
which positions count in the loss, so any difference in accuracy comes from the gene selection.

The steps below use the lower-level scripts `scripts/transform.py`, `scripts/sample.py` and `scripts/train.py`.
If you use the plant pipeline (`train.sh` or `genecad train`), you can pass the same settings through
`--exclude-genes`, `--windows-from`, `--initial-state`, `--seed`, `--max-steps` and `--base-encoder-revision`.
Its default `revision` preset already uses the settings of the published model (see the README).

## Overview

1. List the genes you want to exclude in each condition.
2. Mask those genes when the labels are built.
3. Sample windows once, from a reference condition, and reuse them for the other conditions.
4. Export the initial weights once for each training seed.
5. Train every condition from those weights for a fixed number of optimizer updates.

## 1. List the excluded genes

For each condition, you need a tab-separated file with the columns `species_id` and `gene_id`, one row per gene
to remove. A condition that keeps every gene does not need a file.

## 2. Mask the excluded genes

The `filter_features` subcommand of `scripts/transform.py` takes this file:

```bash
python scripts/transform.py filter_features ... --exclude-genes selection.tsv
```

Matching genes are removed with the reason `revision_selection`, and their spans are kept so that they can be
masked from the loss. This way an excluded gene is simply ignored during training, rather than being relabelled
as intergenic sequence.

## 3. Reuse one set of windows

So that every condition sees exactly the same sequence, we sample windows once, from the reference condition, and
give them to the `generate_training_windows` subcommand of `scripts/sample.py` for every other condition:

```bash
python scripts/sample.py generate_training_windows ... --windows-from /path/to/reference-windows.zarr
```

Reused windows keep the reference order and strand, including the orientation of negative-strand windows, and
they are kept even when every position in them is masked. To catch mistakes early, sampling stops with an error
if a window has a different length, invalid bounds or strand, or different DNA, or if a chromosome that has
reference windows is missing from the input.

The last case can happen when a selection removes every gene on a chromosome, because labels are only built for
chromosomes with retained genes. In that case, rebuild the labels and masks of that chromosome and run again.
Deleting its windows from the reference would make the error go away, but the conditions would no longer share
the same windows.

We recommend using the same species configuration, validation proportion and split seed for all conditions. The
split sorts species and chromosomes before drawing random numbers, so the order in which files are listed does
not change it. Before starting the full experiment, it is worth comparing window coordinates, sample indices and
excluded spans across the final datasets.

## 4. Export the initial weights

Next, export the initial weights once for each training seed, from the reference condition:

```bash
python scripts/train.py \
  --train-dataset /path/to/reference-train.zarr \
  --val-dataset /path/to/reference-valid.zarr \
  --output-dir /path/to/export-log \
  --seed 11 --base-encoder-revision 5574ff5dd1795fe58ac929bd89a7f41a07e05997 \
  --export-initial-state /path/to/seed11.pt
```

Please add the architecture, window size, head, backbone and freezing options of your experiment. The command
writes the file and exits without training. It will not overwrite an existing file, and under a distributed
launcher only rank zero writes it.

## 5. Train each condition

Finally, train every condition with the same model options and seed, starting from the exported weights:

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
against the export, and all conditions use the class frequencies stored with the weights. You can find the hash
of the starting weights in `initialization.json` in the run directory. Note that `--initial-state` cannot be
combined with resuming from a checkpoint, `--auto-class-weights yes` or `--torch-compile yes`.

A positive `--max-steps` replaces the epoch limit and counts optimizer updates, including gradient accumulation.
A batch in which every position is masked contributes a zero loss: it still counts in the update schedule, and
multi-GPU training continues normally. If you use validation only for monitoring, it is best to choose the final
checkpoint the same way for every condition.

## What has been tested

Shared initialization, fully masked batches, gradient accumulation, fixed update counts across epochs and resuming
from a checkpoint were tested on H100 GPUs in BF16, including a two-GPU run in which one GPU received only masked
batches; both GPUs ended with identical parameters. The full model (PlantCAD2 backbone and classifier head) was
checked on short sequences on one GPU. These checks confirm correctness; they are not a full-length training run
or a multi-GPU run of the full model. For running GeneCAD on ARM64 GH200 nodes, see [TACC Vista](tacc_vista.md).
