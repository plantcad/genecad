# Intergenic Bias Sweep

See https://github.com/plantcad/genecad/pull/16.

## Overview

This directory contains an experiment for sweeping intergenic bias parameters during inference. It evaluates the impact of varying these bias values for species such as `jregia` and `zmays` (e.g. testing with or without UTR modes enabled) and analyzes the accuracy of the resulting annotations using `gffcompare`.

## Results Data
All generated result files, including the summary statistics (`.stats`), performance metric tables (`.tsv`), and visualizations (`.png`, `.pdf`), have been exported and uploaded to Hugging Face.

They can be accessed at the following dataset:
**Hugging Face Dataset:** [`plantcad/genecad_pr16_intergenic_bias_results`](https://huggingface.co/datasets/plantcad/genecad_pr16_intergenic_bias_results)
