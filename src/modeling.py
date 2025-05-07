"""
Utility related to pytorch-lightning.

Primarily models, but this is also the place for loss functions
"""
import time
import pandas as pd
from typing import Optional
import lightning as L
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torchcrf import CRF
from torch.utils import data
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_auroc
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
from src.visualization import (
    visualize_entities,
    visualize_tokens,
)
from dataclasses import dataclass, field

def worker_load_files(worker_id):
    """ Open a fresh file handle for each DataLoader worker """
    worker_info = data.get_worker_info()
    dataset = worker_info.dataset

    dataset.open_label_files()

# ------------------------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------------------------

IGNORE_INDEX = -100

class FocalLoss(nn.Module):
    """Focal Loss implementation from https://arxiv.org/abs/1708.02002.
    
    This is useful for down-weighting easily classified examples from, typically,
    more frequent clasess (e.g. intergenic, I-cds, I-intron, etc.).
    """

    def __init__(self, alpha, gamma=2.0, ignore_index: int = IGNORE_INDEX):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)
        loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        prob = torch.exp(-loss)
        indices = torch.clamp(targets, min=0)
        alpha = self.alpha.to(inputs.device)[indices]
        focal_loss = alpha * (1-prob)**self.gamma * loss
        mask = targets != self.ignore_index
        return (focal_loss * mask).sum() / mask.sum()

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
# Base Classifier
# ------------------------------------------------------------------------------------------------

def position_boundary_indices(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError(f"Input must be 2D, got {input_ids.shape=}")
    batch_size, sequence_length = input_ids.shape
    positions = torch.arange(sequence_length, device=input_ids.device)
    # Generate indicies like [0, 1, 1, ..., 1, 1, 2]
    boundary_indices = (positions > 0).long() + (positions == sequence_length-1).long()
    boundary_indices = boundary_indices.unsqueeze(0).expand(batch_size, -1)
    assert boundary_indices.shape == input_ids.shape
    return boundary_indices


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
TOKEN_CLASS_WEIGHTS: list[float] = [
    3.27254844e-07, # intergenic [0]
    7.20223719e-04, # B-intron [1]
    2.30126851e-06, # I-intron [2]
    7.19396424e-04, # L-intron [3]
    2.46485015e-01, # U-intron [4]
    2.72806808e-03, # B-five_prime_utr [5]
    1.50915858e-05, # I-five_prime_utr [6]
    2.71827761e-03, # L-five_prime_utr [7]
    2.46485015e-01, # U-five_prime_utr [8]
    6.22306916e-04, # B-cds [9]
    2.60387194e-06, # I-cds [10]
    6.22228267e-04, # L-cds [11]
    2.46485015e-01, # U-cds [12]
    2.93720655e-03, # B-three_prime_utr [13]
    9.16895564e-06, # I-three_prime_utr [14]
    2.96273766e-03, # L-three_prime_utr [15]
    2.46485015e-01, # U-three_prime_utr [16]
]
TOKEN_TRANSITION_PROBS = [
    # intergenic              B-intron                I-intron                L-intron                U-intron                B-five_prime_utr        I-five_prime_utr        L-five_prime_utr        U-five_prime_utr        B-cds                   I-cds                   L-cds                   U-cds                   B-three_prime_utr       I-three_prime_utr       L-three_prime_utr       U-three_prime_utr     
    [9.9986728674356296e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 1.3250703334417960e-04, 0.0000000000000000e+00, 0.0000000000000000e+00, 2.0622309290219395e-07, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],  # intergenic
    [0.0000000000000000e+00, 0.0000000000000000e+00, 1.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],  # B-intron
    [0.0000000000000000e+00, 0.0000000000000000e+00, 9.9668850440768986e-01, 3.3114955923101274e-03, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],  # I-intron
    [0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 4.2768390774200638e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, 3.7023625955596062e-04, 9.3457396384703950e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 5.6959424547070860e-05, 2.2230449694656800e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],  # L-intron
    [0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],  # U-intron
    [0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 9.9608745324506631e-01, 3.9125467549337217e-03, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],  # B-five_prime_utr
    [0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 9.9441337268119323e-01, 5.5866273188067746e-03, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],  # I-five_prime_utr
    [0.0000000000000000e+00, 1.6436565131445827e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 8.3554095911093129e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 9.3389574610487646e-05, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],  # L-five_prime_utr
    [0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 1.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],  # U-five_prime_utr
    [0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 9.9976382088457894e-01, 2.3617911542108623e-04, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],  # B-cds
    [0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 9.9579626722664549e-01, 4.2037327733545383e-03, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],  # I-cds
    [0.0000000000000000e+00, 8.0746419432388961e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 1.9201134799702924e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 5.2445767908117832e-04],  # L-cds
    [0.0000000000000000e+00, 2.9999999999999999e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 6.9999999999999996e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],  # U-cds
    [0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 9.9863362938030098e-01, 1.3663706196990653e-03, 0.0000000000000000e+00],  # B-three_prime_utr
    [0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 9.9692930636621491e-01, 3.0706936337850763e-03, 0.0000000000000000e+00],  # I-three_prime_utr
    [9.0914382564927843e-01, 9.0582253342692307e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 2.7392100802930953e-04, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],  # L-three_prime_utr
    [1.1564625850340136e-01, 8.8435374149659862e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],  # U-three_prime_utr
]


@dataclass
class SequenceSpanClassifierConfig:
    vocab_size: int = 16
    num_labels: int = NUM_LABELS
    n_layers: int = 8
    n_heads: int = 4
    token_embedding_dim: int = 32
    boundary_embedding_dim: int = 32
    embedding_input_dim: int = 1536
    use_input_projection: bool = True
    input_projection_dim: int = 768 - 32 * 2
    max_sequence_length: Optional[int] = None
    train_eval_frequency: Optional[int] = 250
    enable_visualization: bool = True
    use_base_model: bool = True
    dropout: float = 0.1
    use_focal_loss: bool = False
    use_crf: bool = False
    init_crf_transitions: bool = False
    freeze_crf: bool = False
    focal_loss_alpha: list[float] = field(default_factory=lambda: TOKEN_CLASS_WEIGHTS)
    focal_loss_gamma: float = 2.0
    token_entity_names: list[str] = field(default_factory=lambda: TOKEN_ENTITY_NAMES)
    token_class_names: list[str] = field(default_factory=lambda: TOKEN_CLASS_NAMES)
    token_transition_probs: list[list[float]] = field(default_factory=lambda: TOKEN_TRANSITION_PROBS)
    interval_entity_classes: list[list[int]] = field(default_factory=lambda: [
        # Maps token entity classes to interval entity classes
        [1, 2, 3, 4], # transcript
        [2, 3, 4], # exon
        [1], # intron
        [2], # five_prime_utr
        [3], # cds
        [4], # three_prime_utr
    ])
    interval_entity_names: list[str] = field(default_factory=lambda: ["transcript", "exon"] + TOKEN_ENTITY_NAMES)
    sentinel_names: tuple[str, str] = field(default_factory=lambda: ("mask", "intergenic"))

    def interval_entity_name(self, entity: int) -> str:
        if 0 <= entity - 1 <= len(self.interval_entity_names) - 1:
            return self.interval_entity_names[entity - 1]
        else:
            return f"entity_{entity:02d}"

    def token_entity_names_with_background(self) -> list[str]:
        return [self.sentinel_names[1]] + self.token_entity_names
    
    def token_label_name(self, label: int) -> str:
        return convert_biluo_index_to_class_name(
            label=label, 
            entity_names=self.token_entity_names, 
            sentinel_names=self.sentinel_names
        )
    
class SequenceSpanClassifier(L.LightningModule):

    def __init__(self, config: SequenceSpanClassifierConfig, learning_rate: float = 8e-4, learning_rate_decay: str = "none", learning_rate_warmup: int = 25):
        super(SequenceSpanClassifier, self).__init__()
        if config.max_sequence_length is None:
            raise ValueError("max_sequence_length must be set")
        if not config.use_input_projection and config.input_projection_dim != config.embedding_input_dim:
            raise ValueError(
                "input_projection_dim must be equal to embedding_input_dim if use_input_projection is False; "
                f"got {config.input_projection_dim=} and {config.embedding_input_dim=}"
            )
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
        if config.use_focal_loss and len(config.focal_loss_alpha) != config.num_labels:
            raise ValueError(
                "focal_loss_alpha must be the same length as num_labels; "
                f"got {len(config.focal_loss_alpha)=} and {config.num_labels=}"
            )
        self.config = config
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_warmup = learning_rate_warmup
        self.criterion = (
                FocalLoss(alpha=config.focal_loss_alpha, gamma=config.focal_loss_gamma)
                if config.use_focal_loss else
                torch.nn.CrossEntropyLoss()
        )
        self.token_embedding = nn.Embedding(config.vocab_size, config.token_embedding_dim)
        self.boundary_embedding = nn.Embedding(3, config.boundary_embedding_dim)
        self.embedding_projection = None
        if config.use_input_projection:
            self.embedding_projection = nn.Linear(config.embedding_input_dim, config.input_projection_dim)
        self.hidden_size = (
            config.input_projection_dim +
            config.token_embedding_dim + 
            config.boundary_embedding_dim
        )
        self.num_labels = config.num_labels
        self.num_core_entities = len(config.token_entity_names)
        self.num_total_entities = self.num_core_entities + 1 # Entity count w/ background
        self.mlp = MLP(
            input_dim=self.hidden_size,
            hidden_dim=4*self.hidden_size,
            output_dim=self.num_labels,
            bias=True,
            dropout=self.config.dropout
        )

        self.base_config = None
        self.base_model = None
        if config.use_base_model:
            self.base_config = ModernBertConfig(
                vocab_size=config.vocab_size,
                hidden_size=self.hidden_size,
                intermediate_size=self.hidden_size*4,
                num_hidden_layers=config.n_layers,
                num_attention_heads=config.n_heads,
                pad_token_id=0,
                max_position_embeddings=config.max_sequence_length,
                attention_dropout=self.config.dropout,
                mlp_dropout=self.config.dropout,
            )
            self.base_model = ModernBertModel(self.base_config)
        self.bias = nn.Parameter(torch.zeros(self.num_labels))

        self.crf = None
        if config.use_crf:
            self.crf = CRF(num_tags=self.num_labels, batch_first=True)
            if config.init_crf_transitions:
                transition_probs = np.array(config.token_transition_probs, dtype=np.float32)
                if transition_probs.shape != (self.num_labels, self.num_labels):
                    raise ValueError(f"Transition probabilities shape {transition_probs.shape} does not match expected shape {(self.num_labels, self.num_labels)}")
                log_transition_probs = np.log(transition_probs + np.finfo(transition_probs.dtype).eps)
                with torch.no_grad():
                    self.crf.transitions.copy_(torch.tensor(log_transition_probs, dtype=torch.float))
                    self.crf.start_transitions.zero_()
                    self.crf.end_transitions.zero_()
                self.crf.transitions.requires_grad = not config.freeze_crf
                self.crf.start_transitions.requires_grad = not config.freeze_crf
                self.crf.end_transitions.requires_grad = not config.freeze_crf

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
        if self.trainer.global_step < self.learning_rate_warmup:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.learning_rate_warmup)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.learning_rate

        optimizer.step(closure=optimizer_closure)

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
            if self.config.use_crf and visualize:
                from src.visualization import visualize_crf
                visualize_crf(self, prefix="train", batch_idx=batch_idx)
                
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.learning_rate_decay == "none":
            return optimizer
        if self.learning_rate_decay != "cosine":
            raise ValueError(f"Invalid learning rate decay: {self.learning_rate_decay}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            eta_min=self.learning_rate * .1, 
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

    def forward(self, input_ids: Tensor, inputs_embeds: Tensor) -> Tensor:
        B, S = input_ids.shape[0], input_ids.shape[1]
        assert S <= self.config.max_sequence_length
        assert inputs_embeds.shape == (B, S, self.config.embedding_input_dim)

        hidden_states = self._hidden_states(input_ids, inputs_embeds)

        logits = self._token_logits(hidden_states)

        if self.training:
            self.log("train/logit_mean", float(logits.mean()))

        return logits

    def _hidden_states(self, input_ids: Tensor, inputs_embeds: Tensor) -> Tensor:
        B, S, H = input_ids.shape[0], input_ids.shape[1], self.hidden_size
        # Compute token embeddings
        token_embedding = self.token_embedding(input_ids)
        assert token_embedding.shape == (B, S, self.config.token_embedding_dim)

        # Compute boundary embedding 
        boundary_indices = position_boundary_indices(input_ids)
        boundary_embedding = self.boundary_embedding(boundary_indices)
        assert boundary_embedding.shape == (B, S, self.config.boundary_embedding_dim)

        # Compute embedding projection
        if self.config.use_input_projection:
            inputs_embeds = self.embedding_projection(inputs_embeds)
            assert inputs_embeds.shape == (B, S, self.config.input_projection_dim)

        # Concatenate boundary embedding with input embeddings
        hidden_states = torch.cat((inputs_embeds, token_embedding, boundary_embedding), dim=-1)
        assert hidden_states.shape == (B, S, H)

        # Generate representations from base model
        if self.config.use_base_model:
            hidden_states = self.base_model(inputs_embeds=hidden_states).last_hidden_state
            assert hidden_states.shape == (B, S, H)
        return hidden_states

    def _token_logits(self, hidden_states: Tensor) -> Tensor:
        B, S, C = hidden_states.shape[0], hidden_states.shape[1], self.num_labels
        logits = self.bias + self.mlp(hidden_states)
        assert logits.shape == (B, S, C)
        return logits
    
    def _compute_loss(self, batch: Batch) -> Tensor:
        if self.config.use_crf:
            # Assign mask to True for first token across the batch
            mask = batch.masks | (torch.arange(batch.masks.shape[1], device=batch.masks.device) == 0)
            loss = -self.crf(emissions=batch.logits, tags=batch.labels, mask=mask, reduction="token_mean")
            return loss
        else:
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
        labels = torch.where(batch.masks, batch.labels, IGNORE_INDEX)
        logits, labels = batch.logits.view(-1, self.num_labels), labels.view(-1)
        
        metrics = {}
        for metric in [
            ("auroc", multiclass_auroc),
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
            totals = {"f1": [], "precision": []} 
            for entity, group in eval_intervals.groupby("entity"):
                stats = get_evaluation_interval_metrics(group)
                entity_name = self.config.interval_entity_name(entity)
                for k, v in stats.items():
                    metrics[f"{prefix}__entity__classes/{k}/{entity:02d}-{entity_name}"] = v
                totals["f1"].append(stats["f1"])
                totals["precision"].append(stats["precision"])
            for k, v in totals.items():
                metrics[f"{prefix}__entity__overall/{k}"] = np.mean(v)
            self.log_dict(metrics, sync_dist=prefix == "valid")

    def _prepare_batch(self, batch: dict[str, Tensor], source: Source | None = None) -> Batch:
        (B, S), C = batch["input_ids"].shape, self.num_labels

        labels = batch["labels"]
        assert labels.shape == (B, S)
        assert (labels >= -1).all() and (labels < C).all()

        # TODO: Evaluate inclusion of batch['soft_mask'];
        # label_mask and labels >= 0 are equivalent, but both are included here defensively
        masks = batch["label_mask"] & (labels >= 0)
        assert masks.shape == (B, S)

        # After all informating about masking has been move to the masks,
        # it can be removed from the labels from here on
        labels = torch.clamp(labels, min=0)

        logits = self.forward(input_ids=batch["input_ids"], inputs_embeds=batch["inputs_embeds"])
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
    
