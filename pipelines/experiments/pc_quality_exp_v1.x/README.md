## PC Quality Filter Experiment - v1.x

This experiment primarily assesses the impact of a zero-shot PlantCAD quality score used to remove low-quality, putative transcripts in ground-truth annotations.  This is assessed in both training data and final evaluations.  It also tests the impact of training dataset size, the use of Viterbi decoding, and the PlantCAD base model itself through a negative control model initialized from a random, frozen PlantCAD checkpoint.

The [main.sh](main.sh) script orchestrates the entire pipeline, including data preparation, training, sequence extraction, prediction generation, and evaluation with TACC.

See [Open-Athena/oa-cornell-dna](https://github.com/Open-Athena/oa-cornell-dna/issues/57) for more details and results.

### Training Runs

Correspondence between version numbers and training setup:

- `1.0`: Athaliana only (fresh start)
- `1.1`: Athaliana + Osativa (fresh start)
- `1.2`: Athaliana + Osativa + Gmax + Hvulgare + Ptrichocarpa (initialize from v1.1 checkpoint)
- `1.3`: Athaliana + Osativa with randomized base encoder (fresh start)

W&B runs:

- `1.0`: https://wandb.ai/eric-czech/pc-genome-annot/runs/pz5mvgqj
- `1.1`: https://wandb.ai/eric-czech/pc-genome-annot/runs/e3e5130k
- `1.2`: https://wandb.ai/eric-czech/pc-genome-annot/runs/9hk2itwe
- `1.3`: https://wandb.ai/eric-czech/pc-genome-annot/runs/5n6iemfh

### Model Configurations

For `v1.{0,1,2}`, here is the configuration used from [scripts/sweep.py](../../scripts/sweep.py):

```
Config  13 | cfg_013__rand_no__arch_all__frzn_yes__lr_1e-04
  randomize_base      : no
  architecture        : all
  learning_rate       : 0.0001
  base_encoder_frozen : yes
  token_embedding_dim : 128
  head_encoder_layers : 8
```

And for `v1.3`, this configuration is used instead:

```
Config  16 | cfg_016__rand_yes__arch_all__frzn_yes__lr_1e-04
  randomize_base      : yes
  architecture        : all
  learning_rate       : 0.0001
  base_encoder_frozen : yes
  token_embedding_dim : 128
  head_encoder_layers : 8
```

### Model Architecture

All variants of models trainined in this experiment have the following form:

```python
GeneClassifier(
  (criterion): CrossEntropyLoss()
  (classifier): MLP(
    (Wi): Linear(in_features=768, out_features=6144, bias=True)
    (drop): Dropout(p=0.1, inplace=False)
    (Wo): Linear(in_features=3072, out_features=17, bias=True)
  )
  (head_encoder): ModernBertModel(
    (embeddings): ModernBertEmbeddings(
      (tok_embeddings): Embedding(16, 768, padding_idx=0)
      (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (drop): Dropout(p=0.0, inplace=False)
    )
    (layers): ModuleList(
      (0): ModernBertEncoderLayer(
        (attn_norm): Identity()
        (attn): ModernBertAttention(
          (Wqkv): Linear(in_features=768, out_features=2304, bias=False)
          (rotary_emb): ModernBertRotaryEmbedding()
          (Wo): Linear(in_features=768, out_features=768, bias=False)
          (out_drop): Dropout(p=0.1, inplace=False)
        )
        (mlp_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): ModernBertMLP(
          (Wi): Linear(in_features=768, out_features=6144, bias=False)
          (act): GELUActivation()
          (drop): Dropout(p=0.1, inplace=False)
          (Wo): Linear(in_features=3072, out_features=768, bias=False)
        )
      )
      (1-7): 7 x ModernBertEncoderLayer(
        (attn_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): ModernBertAttention(
          (Wqkv): Linear(in_features=768, out_features=2304, bias=False)
          (rotary_emb): ModernBertRotaryEmbedding()
          (Wo): Linear(in_features=768, out_features=768, bias=False)
          (out_drop): Dropout(p=0.1, inplace=False)
        )
        (mlp_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): ModernBertMLP(
          (Wi): Linear(in_features=768, out_features=6144, bias=False)
          (act): GELUActivation()
          (drop): Dropout(p=0.1, inplace=False)
          (Wo): Linear(in_features=3072, out_features=768, bias=False)
        )
      )
    )
    (final_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (base_encoder): Caduceus(
    (backbone): CaduceusMixerModel(
      (embeddings): CaduceusEmbeddings(
        (word_embeddings): RCPSEmbedding(
          (embedding): Embedding(8, 768)
        )
      )
      (layers): ModuleList(
        (0-23): 24 x RCPSMambaBlock(
          (mixer): RCPSWrapper(
            (submodule): BiMambaWrapper(
              (mamba_fwd): Mamba2(
                (in_proj): Linear(in_features=768, out_features=3224, bias=False)
                (conv1d): Conv1d(1664, 1664, kernel_size=(4,), stride=(1,), padding=(3,), groups=1664)
                (act): SiLU()
                (norm): RMSNorm()
                (out_proj): Linear(in_features=1536, out_features=768, bias=False)
              )
              (mamba_rev): Mamba2(
                (in_proj): Linear(in_features=768, out_features=3224, bias=False)
                (conv1d): Conv1d(1664, 1664, kernel_size=(4,), stride=(1,), padding=(3,), groups=1664)
                (act): SiLU()
                (norm): RMSNorm()
                (out_proj): Linear(in_features=1536, out_features=768, bias=False)
              )
            )
          )
          (norm): RMSNorm()
        )
      )
      (norm_f): RMSNorm()
    )
  )
  (token_embedding): Embedding(16, 128)
  (embedding_projection): Linear(in_features=1536, out_features=640, bias=True)
)
```
