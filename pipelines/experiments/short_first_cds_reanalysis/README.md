# Short first CDS reanalysis

Read the [findings](../../../docs/experiments/short_first_cds_reanalysis.md). This directory preserves scripts and results from the 2026-09-15 empirical audit. Scripts are offline diagnostics, not production pipeline components.

## Reproduce

Run from the repo root with its Python environment. Existing local `genecad_result` caches, historical per-locus table, and `data/Athaliana_TAIR12_chr4.fa` are required. Results are checked in here; use a separate output directory for a new run:

```bash
python pipelines/experiments/short_first_cds_reanalysis/audit_existing.py genecad_result/experiments/short_first_exon_validation/cross_species_decoder_redecode/per_locus.tsv /tmp/genecad-audit-repeat

.venv/bin/python pipelines/experiments/short_first_cds_reanalysis/replay.py /tmp/genecad-replay-repeat

.venv/bin/python pipelines/experiments/short_first_cds_reanalysis/positional_replay.py /tmp/genecad-positional-repeat

.venv/bin/python pipelines/experiments/short_first_cds_reanalysis/boundary_scores.py /tmp/genecad-boundary-repeat
```

`positional_replay.py` **deliberately uses incorrect row-based genomic lookup** as a negative control. Do not reuse it to read real prediction loci. `replay.py` reads actual sequence coordinates and also includes two deliberately incorrect orientation controls. Only rows with `mode=correct` in `replay_results.json` are the coordinate-aware baseline. All rows in `positional_results.json` use incorrect lookup despite that inherited orientation-mode label.

The scripts read legacy Zarr-v2 chunks directly using NumPy/numcodecs. They target the inspected cache schema, and the replay additionally imports the current frame graph/Numba. They do not support arbitrary Zarr layouts. Each replay checks crop coverage and duplicate positions; tag inspection checks class names and sampled aggregation differences.

## Artifacts

- `audit_existing.py`, `audit_summary.json`, `locus_audit.tsv`: recompute cohort counts and classify CDS-coordinate discrepancies.
- `replay.py`, `replay_results.json`: 11 Arabidopsis loci, actual coordinate lookup.
- `positional_replay.py`, `positional_results.json`: reproduce the historical chains with deliberately wrong lookup.
- `boundary_scores.py`, `boundary_summary.json`, `per_boundary.tsv`: 213 first-donor pairs and 108 exact-chain controls with available scores; the summary records cache paths and sampled aggregation differences.
- `manifest.json`: source revision, scope, and selected input hashes. Full cache/model hashes were not recomputed.

The data are exploratory and previously used in development. No training, learned score combination, whole-genome rerun, or independent confirmatory benchmark was performed. The original local-experiment generating script was not recovered, so matching its output identifies a reproducible failure mechanism rather than its exact faulty source line.
