import math
import time
import pandas as pd
from typing import Literal, Optional
import lightning as L
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_f1_score
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.callbacks import Callback
from transformers import ModernBertConfig, ModernBertModel
from src.sequence import (
    N_BILUO_TAGS,
    create_entity_evaluation_intervals,
    convert_biluo_index_to_class_name,
    convert_to_entity_labels,
    get_evaluation_interval_metrics,
)
from src.schema import SEQUENCE_MODELING_FEATURES
from src.visualization import (
    visualize_entities,
    visualize_tokens,
)
from src.hub import load_base_model
from dataclasses import dataclass, field


# ------------------------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------------------------

IGNORE_INDEX = -100

class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, bias: Tensor | bool, dropout: float):
        super().__init__()
        self.Wi = nn.Linear(input_dim, hidden_dim * 2, bias=bias)
        self.act = F.gelu
        self.drop = nn.Dropout(dropout)
        self.Wo = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))
    
class ThroughputMonitor(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.total_batches = 0

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.total_batches = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.total_batches += 1
        elapsed_time = time.time() - self.start_time
        batch_per_sec = self.total_batches / elapsed_time
        sample_per_sec = batch_per_sec * batch["input_ids"].shape[0]
        token_per_sec = batch_per_sec * batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
        pl_module.log("throughput/train_batches_per_sec", batch_per_sec)
        pl_module.log("throughput/train_samples_per_sec", sample_per_sec)
        pl_module.log("throughput/train_tokens_per_sec", token_per_sec)


# ------------------------------------------------------------------------------------------------
# Gene Classifier
# ------------------------------------------------------------------------------------------------


@dataclass
class Source:
    split: str
    index: int

@dataclass
class Batch:
    logits: Tensor # (batch, sequence, num_labels)
    labels: Tensor # (batch, sequence)
    masks: Tensor  # (batch, sequence)
    index: Tensor  # (batch,)
    source: Source | None

    def name(self) -> str | None:
        if self.source is None:
            return None
        return f"{self.source.split}[{self.source.index}]"

TOKEN_ENTITY_NAMES = ["intron", "five_prime_utr", "cds", "three_prime_utr"]
TOKEN_SENTINEL_NAMES = ("mask", "intergenic")
NUM_LABELS = len(TOKEN_ENTITY_NAMES) * N_BILUO_TAGS + 1
TOKEN_CLASS_NAMES = [
    convert_biluo_index_to_class_name(i, entity_names=TOKEN_ENTITY_NAMES, sentinel_names=TOKEN_SENTINEL_NAMES)
    for i in range(NUM_LABELS)
]
TOKEN_CLASS_FREQUENCIES: dict[str, float] = {
    "intergenic": 7.531898e-01, # [0]
    "B-intron": 3.422340e-04, # [1]
    "I-intron": 1.071083e-01, # [2]
    "L-intron": 3.426275e-04, # [3]
    "U-intron": 2.781372e-08, # [4]
    "B-five_prime_utr": 9.035149e-05, # [5]
    "I-five_prime_utr": 1.633261e-02, # [6]
    "L-five_prime_utr": 9.067691e-05, # [7]
    "U-five_prime_utr": 2.447608e-07, # [8]
    "B-cds": 3.960827e-04, # [9]
    "I-cds": 9.466096e-02, # [10]
    "L-cds": 3.961328e-04, # [11]
    "U-cds": 2.781372e-08, # [12]
    "B-three_prime_utr": 8.391818e-05, # [13]
    "I-three_prime_utr": 2.688256e-02, # [14]
    "L-three_prime_utr": 8.319502e-05, # [15]
    "U-three_prime_utr": 2.072122e-07, # [16]
}
# Assert equality of pre-calculated frequency classes until support for dynamic configuration is necessary
assert TOKEN_CLASS_NAMES == list(TOKEN_CLASS_FREQUENCIES.keys())


@dataclass
class GeneClassifierConfig:
    vocab_size: int = 16
    num_labels: int = NUM_LABELS
    dropout: float = 0.1
    max_sequence_length: Optional[int] = None

    architecture: Literal["encoder-only", "sequence-only", "classifier-only", "all"] = "encoder-only"
    token_embedding_dim: int = 512
    head_encoder_layers: int = 4
    head_encoder_heads: int = 8
    base_encoder_dim: int = 1536
    base_encoder_path: str | None = None
    base_encoder_revision: str | None = None
    base_encoder_frozen: bool = True

    train_eval_frequency: Optional[int] = 250
    enable_visualization: bool = True
    token_entity_names: list[str] = field(default_factory=lambda: TOKEN_ENTITY_NAMES)
    token_class_names: list[str] = field(default_factory=lambda: TOKEN_CLASS_NAMES)
    token_class_frequencies: list[float] | None = field(default_factory=lambda: TOKEN_CLASS_FREQUENCIES)
    interval_entity_classes: list[list[int]] = field(default_factory=lambda: [
        # Map token entity classes to interval entity classes
        [1, 2, 3, 4], # transcript
        [2, 3, 4], # exon
        [1], # intron
        [2], # five_prime_utr
        [3], # cds
        [4], # three_prime_utr
    ])
    interval_entity_names: list[str] = field(default_factory=lambda: ["transcript", "exon"] + TOKEN_ENTITY_NAMES)
    sentinel_names: tuple[str, str] = field(default_factory=lambda: ("mask", "intergenic"))

    @property
    def hidden_size(self) -> int:
        if self.architecture in ["all"]:
            return min(self.base_encoder_dim, self.token_embedding_dim * 6)
        if self.architecture in ["classifier-only"]:
            return self.base_encoder_dim
        if self.architecture in ["sequence-only", "encoder-only"]:
            return self.token_embedding_dim
        raise ValueError(f"Invalid architecture: {self.architecture}")

    @property
    def use_base_encoder(self) -> bool:
        return self.architecture in ["all", "encoder-only", "classifier-only"]

    @property
    def use_head_encoder(self) -> bool:
        return self.architecture in ["all", "sequence-only", "encoder-only"]

    @property
    def use_token_embedding(self) -> bool:
        return self.architecture in ["all", "sequence-only"]
    
    @property
    def use_precomputed_base_encodings(self) -> bool:
        return self.use_base_encoder and self.base_encoder_path is None

    def interval_entity_name(self, entity: int) -> str:
        if 0 <= entity - 1 <= len(self.interval_entity_names) - 1:
            return self.interval_entity_names[entity - 1]
        else:
            return f"entity_{entity:02d}"

    def token_entity_names_with_background(self) -> list[str]:
        return [self.sentinel_names[1]] + self.token_entity_names
    
    def token_entity_name_map(self) -> dict[int, str]:
        return {
            **{
                i-1: self.sentinel_names[i]
                for i in range(len(self.sentinel_names))
            },
            **{
                i+1: self.token_entity_names[i]
                for i in range(len(self.token_entity_names))
            }
        }
    
    def token_label_name(self, label: int) -> str:
        return convert_biluo_index_to_class_name(
            label=label, 
            entity_names=self.token_entity_names, 
            sentinel_names=self.sentinel_names
        )
    
def validate_config(config: GeneClassifierConfig) -> None:
    if config.max_sequence_length is None:
        raise ValueError("max_sequence_length must be set")
    if (config.num_labels - 1) % N_BILUO_TAGS != 0:
        raise ValueError(
            "num_labels must be one more than a multiple of 4 for one background class and 4 BILUO classes per entity (B, I, L, U); "
            f"got {config.num_labels=}"
        )
    if len(config.token_class_names) != config.num_labels:
        raise ValueError(
            "token_class_names must be the same length as num_labels; "
            f"got {len(config.token_class_names)=} and {config.num_labels=}"
        )
    if config.token_embedding_dim > config.hidden_size:
        raise ValueError(
            "token_embedding_dim must be less than or equal to hidden_size; "
            f"got {config.token_embedding_dim=} and {config.hidden_size=}"
        )

class GeneClassifier(L.LightningModule):

    def __init__(self, config: GeneClassifierConfig, learning_rate: float = 8e-4, learning_rate_decay: str = "none", learning_rate_warmup: int = 25):
        super(GeneClassifier, self).__init__()
        validate_config(config)
        self.config = config
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_warmup = learning_rate_warmup
        self.num_labels = config.num_labels
        self.num_core_entities = len(config.token_entity_names)
        self.num_total_entities = self.num_core_entities + 1 # Entity count w/ background
        self.criterion = torch.nn.CrossEntropyLoss()

        self.classifier = MLP(
            input_dim=self.config.hidden_size,
            hidden_dim=4*self.config.hidden_size,
            output_dim=self.num_labels,
            bias=True,
            dropout=self.config.dropout
        )

        self.head_config = None
        self.head_encoder = None
        if config.use_head_encoder:
            self.head_config = ModernBertConfig(
                vocab_size=config.vocab_size,
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.hidden_size*4,
                num_hidden_layers=config.head_encoder_layers,
                num_attention_heads=config.head_encoder_heads,
                pad_token_id=0,
                max_position_embeddings=config.max_sequence_length,
                attention_dropout=self.config.dropout,
                mlp_dropout=self.config.dropout,
            )
            self.head_encoder = ModernBertModel(self.head_config)

        self.base_encoder = None
        if config.use_base_encoder:
            # If path is not provided, base encoder embeddings will be required as inputs
            if not config.use_precomputed_base_encodings:
                self.base_encoder = load_base_model(config.base_encoder_path, revision=config.base_encoder_revision)
                if config.base_encoder_frozen:
                    self.base_encoder = self.base_encoder.eval()
                    for p in self.base_encoder.parameters():
                        p.requires_grad = False
                else:
                    self.base_encoder = self.base_encoder.train()


        self.token_embedding = None
        if config.use_token_embedding:
            # With no base encoder, expand token embedding dim to hidden size
            token_embedding_dim = config.token_embedding_dim if config.use_base_encoder else config.hidden_size
            self.token_embedding = nn.Embedding(config.vocab_size, token_embedding_dim)

        self.embedding_projection = None
        self.embedding_projection_dim = config.hidden_size - (0 if self.token_embedding is None else self.token_embedding.embedding_dim)
        if config.use_base_encoder and config.base_encoder_dim != self.embedding_projection_dim:
            # Use projection to ensure that projected base embeddings concatenated with
            # token embeddings have length equal to hidden size
            self.embedding_projection = nn.Linear(config.base_encoder_dim, self.embedding_projection_dim)

        if self.config.token_class_frequencies is not None:
            # Clip class frequencies on low side to 1:1M
            freqs = np.clip(
                np.array([
                    self.config.token_class_frequencies[c] 
                    for c in self.config.token_class_names
                ]), 
                a_min=1e-6, a_max=1
            )
            self.bias = nn.Parameter(torch.tensor(np.log10(freqs), dtype=self.dtype))
        else:
            self.bias = nn.Parameter(torch.zeros(self.num_labels))

        self.save_hyperparameters()


    # -------------------------------------------------------------------------
    # Lightning Methods
    # -------------------------------------------------------------------------
    
    def on_before_optimizer_step(self, optimizer):
        # See: https://github.com/Lightning-AI/pytorch-lightning/pull/16745
        norms = {
            f"grad/{k}": v for k, v in grad_norm(self, norm_type=2).items() 
            if "norm_total" in k or "final_norm" in k
        }
        if norms:
            self.log_dict(norms)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # See: https://github.com/Lightning-AI/pytorch-lightning/issues/328#issuecomment-550114178
        for pg in optimizer.param_groups:
            pg['lr'] = self._learning_rate(self.trainer.global_step)
        optimizer.step(closure=optimizer_closure)

    def _learning_rate(self, step: int) -> float:
        if step < self.learning_rate_warmup:
            return min(1., float(step + 1) / self.learning_rate_warmup) * self.learning_rate
        return self.learning_rate

    def training_step(self, batch, batch_idx):
        batch = self._prepare_batch(batch, source=Source(split="train", index=batch_idx))
        loss = self._compute_loss(batch)
        self.log("train/loss", loss)
        for i in range(self.bias.shape[0]):
            self.log(f"bias/class_{i:02d}", self.bias[i])
            
        if (
            self.config.train_eval_frequency 
            and self.global_step % self.config.train_eval_frequency == 0
        ):
            visualize = self.config.enable_visualization and self.global_step > 0
            self._evaluate(batch, "train", visualize=visualize)
                
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self._prepare_batch(batch, source=Source(split="valid", index=batch_idx))
        loss = self._compute_loss(batch)
        self.log("valid/loss", loss, sync_dist=True)
        visualize = (
            self.config.enable_visualization 
            and self.global_step > 0
            and batch_idx < 3 # Visualize first 3 batches
        )
        self._evaluate(batch, "valid", visualize=visualize)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate(0))
        if self.learning_rate_decay == "none":
            return optimizer
        if self.learning_rate_decay != "cosine":
            raise ValueError(f"Invalid learning rate decay: {self.learning_rate_decay}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            eta_min=self.learning_rate * .01, 
            T_max=self.trainer.estimated_stepping_batches
        )
        scheduler_config = dict(
            scheduler=scheduler,
            interval='step',
            frequency=1
        )
        return [optimizer], [scheduler_config]


    # -------------------------------------------------------------------------
    # Model Methods
    # -------------------------------------------------------------------------

    def forward(self, input_ids: Tensor, inputs_embeds: Tensor | None = None) -> Tensor:
        B, S = input_ids.shape[0], input_ids.shape[1]
        assert S <= self.config.max_sequence_length
        assert inputs_embeds is None or inputs_embeds.shape == (B, S, self.config.base_encoder_dim)

        hidden_states = self._hidden_states(input_ids, inputs_embeds)

        logits = self._token_logits(hidden_states)

        if self.training:
            self.log("train/logit_mean", float(logits.mean()))

        return logits

    def _hidden_states(self, input_ids: Tensor, inputs_embeds: Tensor | None = None) -> Tensor:
        B, S, H = input_ids.shape[0], input_ids.shape[1], self.config.hidden_size

        # Compute token embeddings
        token_embedding = None
        if self.config.use_token_embedding:
            token_embedding = self.token_embedding(input_ids)

        # Compute base embedding
        base_embedding = None
        if self.config.use_base_encoder:
            if self.base_encoder is None and inputs_embeds is None:
                raise ValueError("inputs_embeds must be provided if use_base_encoder is True")
            if self.base_encoder is None:
                base_embedding = inputs_embeds
            else:
                base_embedding = self.base_encoder(input_ids=input_ids).last_hidden_state
            assert base_embedding.shape == (B, S, self.config.base_encoder_dim)

        # Project base embedding
        if self.embedding_projection is not None:
            base_embedding = self.embedding_projection(base_embedding)
            assert base_embedding.shape == (B, S, self.embedding_projection_dim)

        # Concatenate base and token embeddings
        hidden_states = torch.cat(tuple(e for e in [base_embedding, token_embedding] if e is not None), dim=-1)
        assert hidden_states.shape == (B, S, H)

        # Compute head encoding
        if self.config.use_head_encoder:
            hidden_states = self.head_encoder(inputs_embeds=hidden_states).last_hidden_state
            assert hidden_states.shape == (B, S, H)
        return hidden_states

    def _token_logits(self, hidden_states: Tensor) -> Tensor:
        B, S, C = hidden_states.shape[0], hidden_states.shape[1], self.num_labels
        logits = self.bias + self.classifier(hidden_states)
        assert logits.shape == (B, S, C)
        return logits
    
    def _compute_loss(self, batch: Batch) -> Tensor:
        labels = torch.where(batch.masks, batch.labels, IGNORE_INDEX)
        logits, labels = batch.logits.view(-1, self.num_labels), labels.view(-1)
        loss = self.criterion(logits, labels)
        return loss

    def _evaluate(self, batch: Batch, prefix: str, visualize: bool = False) -> None:
        self._evaluate_tokens(batch, prefix, visualize=visualize)
        self._evaluate_entities(batch, prefix, visualize=visualize)

    def _evaluate_tokens(self, batch: Batch, prefix: str, visualize: bool = False) -> None:
        if visualize:
            visualize_tokens(
                self, 
                logits=batch.logits, 
                labels=batch.labels, 
                masks=batch.masks, 
                prefix=prefix,
                batch_idx=batch.source.index,
            )
        logits, labels, masks = batch.logits.view(-1, self.num_labels), batch.labels.view(-1), batch.masks.view(-1)
        logits, labels = logits[masks], labels[masks]
        assert len(logits) == len(labels)
        if len(logits) == 0:
            return
        
        metrics = {}
        for metric in [
            ("f1", multiclass_f1_score),
        ]:
            name, func = metric
            values = func(logits, labels, num_classes=self.num_labels, average="none")
            for i in range(self.num_labels):
                label_name = self.config.token_label_name(i)
                metrics[f"{prefix}__token__classes/{name}/{i:02d}-{label_name}"] = values[i]
            metrics[f"{prefix}__token__overall/{name}"] = values.mean()
        self.log_dict(metrics, sync_dist=prefix == "valid")

    def aggregate_logits(self, logits: Tensor) -> Tensor:
        (B, S, _), E = logits.shape, self.num_core_entities
        # Create list of slices for each group of classes to aggregate
        slices = [
            slice(0, 1), # background class
            *[slice(1 + i*N_BILUO_TAGS, 1 + (i+1)*N_BILUO_TAGS) for i in range(E)]
        ]
        # Aggregate logits for each slice
        entity_logits = []
        for s in slices:
            entity_logits.append(torch.logsumexp(logits[:, :, s], dim=2, keepdim=True))
        entity_logits = torch.cat(entity_logits, dim=2)
        assert entity_logits.shape == (B, S, E+1)
        return entity_logits
    
    def _evaluate_entities(self, batch: Batch, prefix: str, visualize: bool = False) -> None:
        B, S = batch.labels.shape

        # Get true entity labels
        true_token_labels = batch.labels.detach().cpu().numpy()
        true_token_masks = batch.masks.detach().cpu().numpy()
        true_entity_labels = np.stack([
            convert_to_entity_labels(true_token_labels[sample])
            for sample in range(B)
        ])
        assert true_entity_labels.shape == (B, S)

        # Get predicted entity labels
        pred_entity_logits = self.aggregate_logits(batch.logits)
        pred_entity_labels = torch.argmax(pred_entity_logits, dim=2)
        pred_entity_labels = pred_entity_labels.detach().cpu().numpy()
        assert pred_entity_labels.shape == (B, S)

        # Ensure that masks are applied uniformly for both true and predicted labels
        true_entity_labels = np.where(true_token_masks, true_entity_labels, 0)
        pred_entity_labels = np.where(true_token_masks, pred_entity_labels, 0)

        # Create predicted and labeled intervals to evaluate
        eval_intervals = []
        for sample in range(B):
            intervals = create_entity_evaluation_intervals(
                true_labels=true_entity_labels[sample],
                pred_labels=pred_entity_labels[sample],
                mask=true_token_masks[sample],
                class_groups=self.config.interval_entity_classes,
            )
            eval_intervals.append(intervals.assign(sample_index=sample))
        eval_intervals = pd.concat(eval_intervals, axis=0, ignore_index=True)

        # Visualize entity predictions, if configured
        if visualize:
            # Choose a single sample in the batch to visualize
            sample_index = int(np.argmax((true_entity_labels > 0).sum(axis=1)))
            sample_name = f"dataset[{int(batch.index[sample_index])}]/{batch.name()}"
            pred_entity_logits = pred_entity_logits.detach().cpu().numpy()
            visualize_entities(
                self,
                true_entity_labels=true_entity_labels,
                pred_entity_labels=pred_entity_labels,
                pred_entity_logits=pred_entity_logits,
                true_token_masks=true_token_masks,
                eval_intervals=eval_intervals,
                prefix=prefix,
                sample_index=sample_index,
                sample_name=sample_name,
                batch_name=batch.name(),
                batch_idx=batch.source.index,
            )

        # Calculate metrics for each entity
        if len(eval_intervals) > 0:
            metrics = {}
            totals = {"f1": [], "precision": [], "recall": []} 
            for entity, group in eval_intervals.groupby("entity"):
                stats = get_evaluation_interval_metrics(group)
                entity_name = self.config.interval_entity_name(entity)
                for k, v in stats.items():
                    metrics[f"{prefix}__entity__classes/{k}/{entity:02d}-{entity_name}"] = v
                for k in totals.keys():
                    totals[k].append(stats[k])
            for k, v in totals.items():
                metrics[f"{prefix}__entity__overall/{k}"] = np.mean(v)
            self.log_dict(metrics, sync_dist=prefix == "valid")

    def _prepare_batch(self, batch: dict[str, Tensor], source: Source | None = None) -> Batch:
        (B, S), C = batch["input_ids"].shape, self.num_labels

        labels = batch["labels"]
        assert labels.shape == (B, S)
        assert (labels >= -1).all() and (labels < C).all()

        # label_mask and labels >= 0 are equivalent, but both are included here defensively
        masks = batch["label_mask"] & (labels >= 0)
        assert masks.shape == (B, S)

        # After all informating about masking has been move to the masks,
        # it can be removed from the labels from here on
        labels = torch.clamp(labels, min=0)

        logits = self.forward(input_ids=batch["input_ids"], inputs_embeds=batch.get("inputs_embeds"))
        assert logits.shape == (B, S, C)

        index = batch["sample_index"]
        assert index.shape == (B,)

        return Batch(
            source=source,
            logits=logits,
            labels=labels,
            masks=masks,
            index=index,
        )
    
# -------------------------------------------------------------------------
# CRF
# -------------------------------------------------------------------------

TOKEN_TRANSITION_PROBS = [
    # intergenic              intron                  five_prime_utr          cds                     three_prime_utr
    [9.9995043940833084e-01, 0.0000000000000000e+00, 4.9560591669216442e-05, 0.0000000000000000e+00, 0.0000000000000000e+00],  # intergenic
    [0.0000000000000000e+00, 9.9669866350105285e-01, 1.4053978186751904e-04, 3.0901254015455425e-03, 7.0671315534084214e-05],  # intron
    [0.0000000000000000e+00, 8.8466835423086916e-04, 9.9451467488120215e-01, 4.6006567645669990e-03, 0.0000000000000000e+00],  # five_prime_utr
    [0.0000000000000000e+00, 3.3405151589122515e-03, 0.0000000000000000e+00, 9.9585209719964018e-01, 8.0738764144757324e-04],  # cds
    [2.8711465041655749e-03, 2.6647086976677194e-04, 1.0282342852383601e-06, 0.0000000000000000e+00, 9.9686135439178236e-01],  # three_prime_utr
]

def token_transition_probs() -> pd.DataFrame:
    return pd.DataFrame(
        data=TOKEN_TRANSITION_PROBS,
        index=SEQUENCE_MODELING_FEATURES,
        columns=SEQUENCE_MODELING_FEATURES,
    )

def create_crf(config: GeneClassifierConfig, initialize: bool = True, freeze: bool = True):
    if (actual := config.token_entity_names_with_background()) != (expected := SEQUENCE_MODELING_FEATURES):
        raise ValueError(f"Token entity names must match those used to create transition probabilities; expected labels: {expected}, got: {actual}")
    from torchcrf import CRF
    labels = config.token_entity_names_with_background()
    crf = CRF(num_tags=len(labels), batch_first=True)
    if initialize:
        transition_probs = np.array(TOKEN_TRANSITION_PROBS, dtype=np.float32)
        log_transition_probs = np.log(transition_probs + np.finfo(transition_probs.dtype).eps)
        with torch.no_grad():
            crf.transitions.copy_(torch.tensor(log_transition_probs, dtype=torch.float))
            crf.start_transitions.zero_()
            crf.end_transitions.zero_()
        if freeze:
            crf.transitions.requires_grad = False
            crf.start_transitions.requires_grad = False
            crf.end_transitions.requires_grad = False
    return crf


# -------------------------------------------------------------------------
# UNET
# -------------------------------------------------------------------------
# From: https://huggingface.co/InstaDeepAI/segment_nt_multi_species/blob/main/modeling_segment_nt.py

class DownSample1D(nn.Module):
    """
    1D-UNET downsampling block with dilated convolutions for large receptive fields.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_layers: int = 2,
        base_dilation: int = 1,
    ):
        """
        Args:
            input_channels: number of input channels
            output_channels: number of output channels
            num_layers: number of convolution layers
            base_dilation: base dilation rate (will be multiplied by 2^layer_idx)
        """
        
        super().__init__()
        
        # Create layers with exponentially increasing dilation
        conv_layers = []
        for i in range(num_layers):
            dilation = base_dilation * (2 ** i)
            in_channels = input_channels if i == 0 else output_channels
            
            conv_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                dilation=dilation,
                padding=dilation,  # padding = dilation for same output size
            ))
        
        self.conv_layers = nn.ModuleList(conv_layers)
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        self.activation_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for conv_layer in self.conv_layers:
            x = self.activation_fn(conv_layer(x))

        hidden = x
        x = self.avg_pool(hidden)
        return x, hidden



class UpSample1D(nn.Module):
    """
    1D-UNET upsampling block with dilated convolutions for large receptive fields.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_layers: int = 2,
        base_dilation: int = 1,
    ):
        """
        Args:
            input_channels: number of input channels
            output_channels: number of output channels
            num_layers: number of convolution layers
            base_dilation: base dilation rate (will be multiplied by 2^layer_idx)
        """
        super().__init__()

        # Create layers with exponentially increasing dilation
        conv_layers = []
        for i in range(num_layers):
            dilation = base_dilation * (2 ** i)
            in_channels = input_channels if i == 0 else output_channels
            
            conv_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                dilation=dilation,
                padding=dilation,  # padding = dilation for same output size
            ))

        self.conv_layers = nn.ModuleList(conv_layers)
        self._activation_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_layer in self.conv_layers:
            x = self._activation_fn(conv_layer(x))      

        # Upsample by 2x using nearest neighbor interpolation
        x = nn.functional.interpolate(x, size=2 * x.shape[2], mode="nearest")
        return x



class FinalConv1D(nn.Module):
    """
    Final output block with dilated convolutions.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_layers: int = 2,
        base_dilation: int = 1,
    ):
        """
        Args:
            input_channels: number of input channels
            output_channels: number of output channels
            num_layers: number of convolution layers
            base_dilation: base dilation rate (will be multiplied by 2^layer_idx)
        """
        super().__init__()

        # Create layers with exponentially increasing dilation
        conv_layers = []
        for i in range(num_layers):
            dilation = base_dilation * (2 ** i)
            
            # Fix channel logic: all intermediate layers use input_channels, only final layer outputs output_channels
            if i == 0:
                in_channels = input_channels
            else:
                in_channels = input_channels  # All intermediate layers use input_channels
                
            if i == num_layers - 1:
                out_channels = output_channels  # Only final layer outputs to output_channels
            else:
                out_channels = input_channels  # All intermediate layers output input_channels
            
            conv_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                dilation=dilation,
                padding=dilation,  # padding = dilation for same output size
            ))

        self.conv_layers = nn.ModuleList(conv_layers)
        self._activation_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if i < len(self.conv_layers) - 1:
                x = self._activation_fn(x)
        return x


class UNET1DSegmentationHead(nn.Module):
    """
    1D-UNET with dilated convolutions for large receptive fields (64k+ bp).
    
    This version uses exponentially increasing dilation rates to achieve
    very large receptive fields without significantly increasing parameters.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        output_channels_list: tuple[int, ...] = (64, 128, 256, 512, 1024),
        num_conv_layers_per_block: int = 2,
        base_dilation: int = 1,
    ):
        """
        Args:
            embed_dim: input embedding dimension
            num_classes: number of classes to segment
            output_channels_list: list of the number of output channels at each level
            num_conv_layers_per_block: number of convolution layers per block
            base_dilation: base dilation rate for the first level
        """
        super().__init__()
        self._num_pooling_layers = len(output_channels_list)
        self._base_dilation = base_dilation
        self._num_conv_layers_per_block = num_conv_layers_per_block

        downsample_input_channels_list = (embed_dim, ) + output_channels_list[:-1]
        output_channels_list_reversed = tuple(reversed(output_channels_list))
        upsample_input_channels_list = (output_channels_list[-1],) + output_channels_list_reversed
        upsample_output_channels_list = output_channels_list_reversed

        # Create downsampling blocks with increasing dilation rates
        self._downsample_blocks = nn.ModuleList([
            DownSample1D(
                input_channels=input_channels,
                output_channels=output_channels,
                num_layers=num_conv_layers_per_block,
                base_dilation=base_dilation * (2 ** (2 * i)),  # Exponential increase per level
            )
            for i, (input_channels, output_channels) in enumerate(
                zip(downsample_input_channels_list, output_channels_list)
            )
        ])

        # Create upsampling blocks with decreasing dilation rates
        self._upsample_blocks = nn.ModuleList([
            UpSample1D(
                input_channels=input_channels,
                output_channels=output_channels,
                num_layers=num_conv_layers_per_block,
                base_dilation=base_dilation * (2 ** (2 * (len(output_channels_list) - 1 - i))),
            )
            for i, (input_channels, output_channels) in enumerate(
                zip(upsample_input_channels_list, upsample_output_channels_list)
            )
        ])

        # Final block with moderate dilation
        self.final_block = FinalConv1D(
            input_channels=output_channels_list[0],
            output_channels=num_classes,
            num_layers=num_conv_layers_per_block,
            base_dilation=base_dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (length := x.shape[2]) % (divisor := 2**self._num_pooling_layers):
            raise ValueError(
                f"Input length ({length}) must be divisible by 2 to the power of"
                f" number of pooling layers ({divisor})."
            )

        hiddens = []
        for downsample_block in self._downsample_blocks:
            x, hidden = downsample_block(x)
            hiddens.append(hidden)

        for upsample_block, hidden in zip(self._upsample_blocks, reversed(hiddens)):
            x = upsample_block(x) + hidden
            
        x = self.final_block(x)
        return x

class UnetClassifier(L.LightningModule):
    """Lightning module for UNET with large receptive field using dilated convolutions."""
    
    def __init__(
        self, 
        embed_dim: int, 
        num_classes: int, 
        learning_rate: float = 1e-4,
        output_channels_list: tuple[int, ...] = (64, 128, 256, 512, 1024),
        num_conv_layers_per_block: int = 2,
        base_dilation: int = 1,
        residual_mode: bool = True,
    ):
        super().__init__()
        self.model = UNET1DSegmentationHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            output_channels_list=output_channels_list,
            num_conv_layers_per_block=num_conv_layers_per_block,
            base_dilation=base_dilation,
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.residual_mode = residual_mode
        self.save_hyperparameters()
            
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got {x.ndim}D tensor; shape={x.shape}")
        B, S, C = x.shape # (batch, sequence, class)
        if C != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes in last dim, got {C}; shape={x.shape}")
        
        # Transpose for U-Net to (batch, class, sequence) and then back to (batch, sequence, class)
        corrections = self.model(x.transpose(1, 2)).transpose(1, 2)
        assert corrections.shape == (B, S, C)
        
        if self.residual_mode:
            # Add corrections to input logits (residual connection)
            output = x + corrections
        else:
            # Replace input logits with corrections
            output = corrections
            
        return output
    
    def _compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute cross entropy loss from batch data."""
        input_logits = batch['input_logits']
        target_labels = batch['target_labels']
        if (actual := target_labels.shape) != (expected := input_logits.shape):
            raise ValueError(f"Expected {expected} shape for target labels, got {actual}")
        target_class = target_labels.argmax(dim=-1)
        output_logits = self(input_logits)
        loss = self.criterion(output_logits.reshape(-1, self.num_classes), target_class.view(-1))
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('train/loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        expected_steps = self.trainer.estimated_stepping_batches
        warmup_steps = math.ceil(expected_steps * .05)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (expected_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(progress * math.pi))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scheduler_config = dict(scheduler=scheduler, interval="step", frequency=1)
        return [optimizer], [scheduler_config]
