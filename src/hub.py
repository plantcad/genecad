import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoModel, AutoConfig


@dataclass
class HiddenStateOutput:
    """Wrapper output that provides `last_hidden_state` for models that don't natively expose it."""
    last_hidden_state: torch.Tensor


class BaseEncoderWrapper(nn.Module):
    """Wraps a CausalLM model to extract hidden states before the LM head.

    Some models (like HNetForCausalLM) return CausalLMOutput with logits
    but no `last_hidden_state`. This wrapper intercepts the forward pass
    to return backbone hidden states instead.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, **kwargs) -> HiddenStateOutput:
        # Get embeddings from the model's embedding layer
        hidden_states = self.model.embeddings(input_ids)

        # Run through backbone (without the LM head)
        B, L, D = hidden_states.shape
        hidden_states = hidden_states.flatten(0, 1)
        cu_seqlens = torch.arange(B + 1, device=hidden_states.device) * L
        max_seqlen = torch.tensor(L, dtype=torch.int, device=hidden_states.device)
        num_tokens = torch.tensor([L] * B, device=hidden_states.device)

        hidden_states, _ = self.model.backbone(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=None,
            inference_params=None,
            num_tokens=num_tokens,
        )
        hidden_states = hidden_states.view(B, L, D)

        return HiddenStateOutput(last_hidden_state=hidden_states)


def _needs_wrapper(model: nn.Module) -> bool:
    """Check if the model needs wrapping (i.e., it's a CausalLM without last_hidden_state)."""
    return hasattr(model, 'lm_head') and hasattr(model, 'backbone') and hasattr(model, 'embeddings')


def load_base_model(
    path: str,
    revision: str | None = None,
    dtype=torch.bfloat16,
    randomize: bool = False,
) -> nn.Module:
    """Load a Hugging Face base encoder with optional random initialization.

    Parameters
    ----------
    path : str
        Repository identifier or local path of the model to load.
    revision : str or None, optional
        Specific revision of the model to load, by default ``None`` which
        selects the latest revision.
    dtype : torch.dtype, optional
        Torch data type to use for the returned model, by default
        ``torch.bfloat16``.
    randomize : bool, optional
        If ``True``, instantiate the model from configuration without
        pretrained weights. By default ``False``.

    Returns
    -------
    nn.Module
        The instantiated model ready for inference or fine-tuning.
        If the loaded model is a CausalLM (e.g., HNetForCausalLM), it will
        be wrapped to expose `last_hidden_state` from the backbone.
    """
    config = AutoConfig.from_pretrained(path, revision=revision, trust_remote_code=True)
    if randomize:
        base_model = AutoModel.from_config(config, dtype=dtype, trust_remote_code=True)
    else:
        base_model = AutoModel.from_pretrained(
            path, config=config, revision=revision, trust_remote_code=True, dtype=dtype
        )

    # Wrap CausalLM models to expose backbone hidden states
    if _needs_wrapper(base_model):
        base_model = BaseEncoderWrapper(base_model)

    return base_model
