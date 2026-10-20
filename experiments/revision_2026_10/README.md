# GPU08 revision training verification, 2026-10-20

The tested training implementation is commit `565be30a1ca2cb9ee312f8db6ac39ebeac770cfb`.
These files add reproducible verification tools and observations; they do not
change that training implementation. Tests ran on GPU08's idle H100 PCIe GPU1
with the backbone unfrozen and eight head encoder layers. Logging and model
loading used offline mode.

## Actual training entry point

`scripts/train.py` completed exactly two optimizer updates on 768 real, persisted
8192-bp Athaliana exclusion-fixture windows. Batch 4 and accumulation 96 give
effective batch 384, with 192 microbatches. The saved checkpoint has
`global_step=2`; 280 backbone tensors and 57 head tensors changed from the shared
seed-11 initialization. See `cli_fixture_verification.json`.

This fixture tests the actual CLI, DataLoader, mixed precision, optimizer,
checkpoint, shared-state loading, and fixed-step termination. It is not the
complete new eight-species training corpus. The two CSV loss observations are
optimizer-logging samples, not a claim about every microbatch or convergence.
All saved model and optimizer tensors are finite.

The relevant arguments are:

```bash
python scripts/train.py \
  --train-dataset "$EXCLUSION_FIXTURE/splits/train.zarr" \
  --val-dataset "$EXCLUSION_FIXTURE/splits/valid.zarr" \
  --output-dir "$VALIDATION_OUTPUT/cli_fixture" \
  --base-encoder-path emarro/pcad2-200M-cnet-baseline \
  --base-encoder-revision 5574ff5dd1795fe58ac929bd89a7f41a07e05997 \
  --base-encoder-frozen no --architecture all --head-encoder-layers 8 \
  --token-embedding-dim 256 --window-size 8192 \
  --batch-size 4 --accumulate-grad-batches 96 --max-steps 2 \
  --learning-rate 2e-4 --learning-rate-decay cosine \
  --num-workers 4 --prefetch-factor 2 --gpu 1 --strategy auto \
  --checkpoint-frequency 100000 --val-check-interval 1.0 \
  --limit-val-batches 0 --train-eval-frequency 100000 --log-frequency 1 \
  --enable-visualization no --torch-compile no --auto-class-weights no \
  --limit-train-examples 768 --seed 11 --initial-state "$SHARED_INITIAL_STATE" \
  --project-name genecad-revision-validation --run-name fullwindow_two_updates
```

Set `CUDA_VISIBLE_DEVICES` to an idle GPU, use `WANDB_MODE=offline`, and set
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` when the pinned models are cached.

## Real eight-species condition panel

`build_bounded_eight_species_panel.py` builds 22 fixed 8192-bp windows covering
both strands and all eight training species. Historical windows supply DNA
tokens and genomic coordinates only. Current U/B/M/R1 feature and filter
parquets supply labels and masks through production `stack_gff_features`,
`create_labels`, `extract_label_dataset`, and `_build_sample` functions. Small
chromosome prefixes provide complete genes and label context around the probes.
Boundary masks are explicitly upstream 944 bp and downstream 1240 bp, matching
the production CLI rather than the label function's different direct-call
defaults. The GPU verifier rejects a panel without these parameters. The final
artifacts supersede the initial panel report in commit `f75714c`, which used
the function defaults.

MMLR anchors are genes retained by B and removed by M. Additional basic-QC
anchors exercise B. The script checks equal DNA and coordinates, actual
exclusion-span masks, no reactivation of U-filtered bases, and equal labels on
jointly trainable bases. Substituting U's masks for another condition is detected.
The four full-corpus condition feature tables also have the same 107 chromosome groups.

This panel is deliberately enriched for excluded genes. Its retention fractions,
loss values, and update counts are not whole-genome estimates or condition-quality
comparisons. R1 is the random subset paired with training seed 11; R2/R3 and
training seeds 22/33 are outside this GPU check.

The data root must contain the verified `conditions/{U,B,M,R1}/transform/`
parquets and the earlier `gpu_smoke_v2/verification.json` bootstrap indices.
The historical source Zarr is passed explicitly. From the GeneCAD checkout:

```bash
export PYTHONPATH="$PWD"
python experiments/revision_2026_10/build_bounded_eight_species_panel.py \
  --data-root "$REVISION_DATA" --source "$HISTORICAL_TRAIN_ZARR" \
  --output "$VALIDATION_PANEL"
python experiments/revision_2026_10/verify_real_panel_gpu.py \
  --panel "$VALIDATION_PANEL" --initial "$SHARED_INITIAL_STATE" \
  --output "$VALIDATION_OUTPUT/real_conditions"
```

`verify_real_panel_gpu.py` uses the actual training sample transform, XarrayDataset,
DataLoader, production backbone/head and loss. Every condition restores the same
seed-11 state hash. Each condition ran 22 forward/backward probes and six AdamW
updates, with accumulation groups of four windows and a final two-window group.
The optimizer uses the production head/backbone learning-rate ratio and weight
decay, with no scheduler in this bounded probe. A separate all-masked 8192-bp
forward/backward check runs for every condition without an optimizer update.

All four conditions passed finite loss/gradient checks, zero masked-logit
gradients, loss invariance when masked labels/logits are perturbed, and actual
backbone/head updates. All-masked loss and parameter gradients were zero. That
means excluded positions do not contribute a direct supervised loss; DNA at
those positions remains available as sequence context. See
`panel_verification.json` and `gpu_verification.json`.

The JSON report records loop time after model setup and peak allocated CUDA
bytes. This uses a smaller batch than the CLI test and is not a clean hardware
throughput benchmark. Formal 12-run training,
whole-corpus window/split validation, and LS6 DDP/GPU checks remain separate work.
