# Archive

This directory contains code that is no longer part of the active GeneCAD pipeline.
Everything here is preserved for reference but is **not maintained**.

## Contents

| Path | Description |
|------|-------------|
| `experiments/01_emarro_hnet/` | Old single-chromosome and single-species scripts with hardcoded paths |
| `experiments/02_emarro_mlp/` | Abandoned MLP decoder experiment |
| `experiments/03_emarro_ar/` | Abandoned autoregressive decoder experiment |
| `scripts/` | Internal utility scripts not part of the user-facing pipeline |
| `examples/scripts/` | Old inference/evaluation scripts using `make` + `gffcompare` |
| `examples/configs/task.sky.yaml` | SkyPilot task config referencing old scripts |
| `pipelines/` | Makefile-based prediction pipeline (superseded by predict_all_chroms.sh) |
| `bin/` | TACC-specific HPC wrapper |
| `local/` | Internal scratch / logs |
| `docs/` | Internal run logs |
