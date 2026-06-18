import logging
import platform
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass
from types import ModuleType
from transformers import AutoModel, AutoConfig
import transformers.dynamic_module_utils as _dmu

logger = logging.getLogger(__name__)

# On ARM64, triton is not available (no wheels). Install a meta path finder
# that intercepts ALL triton.* imports and returns stub modules, so mamba_ssm
# can be imported. mamba_ssm falls back to compiled CUDA C++ kernels at runtime.
if platform.machine() == "aarch64":
    try:
        import triton  # noqa: F401
    except ImportError:
        import importlib.abc
        import importlib.machinery

        class _TritonConfig:
            def __init__(self, kwargs, num_warps=4, num_stages=2, num_ctas=1,
                         pre_hook=None, **kw):
                self.kwargs = kwargs
                self.num_warps = num_warps
                self.num_stages = num_stages
                self.num_ctas = num_ctas

        _STUB_PKGS = ("triton", "flash_attn")

        class _Stub(ModuleType):
            """Stub module/object: CapitalCase attrs → inheritable class; others → child stub."""
            def __getattr__(self, name):
                if name and name[0].isupper():
                    val = type(name, (), {"__init__": lambda s, *a, **kw: None})
                else:
                    val = _Stub(f"{self.__name__}.{name}")
                object.__setattr__(self, name, val)
                return val
            def __call__(self, *a, **kw):
                return None

        class _StubLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return _Stub(spec.name)
            def exec_module(self, module):
                pass

        class _StubFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                for pkg in _STUB_PKGS:
                    if fullname == pkg or fullname.startswith(pkg + "."):
                        return importlib.machinery.ModuleSpec(
                            fullname, _StubLoader(), is_package=True
                        )
                return None

        sys.meta_path.insert(0, _StubFinder())

        _triton = _Stub("triton")
        _triton.__version__ = "3.3.0"
        _triton.Config = _TritonConfig
        _triton.jit = lambda fn=None, **kw: ((lambda f: f) if fn is None else fn)
        _triton.autotune = lambda configs, key, **kw: (lambda fn: fn)
        _triton.heuristics = lambda values: (lambda fn: fn)
        _triton.cdiv = lambda a, b: (a + b - 1) // b
        sys.modules["triton"] = _triton

# flash_attn may be absent on ARM64 or CPU-only environments.
# transformers' check_imports raises ImportError before even loading the module,
# even when the remote file guards flash_attn with try/except. Wrap it to catch
# only flash_attn ImportErrors and still return the relative imports list that
# get_cached_module_file needs to download sibling .py files from HuggingFace.
_orig_check_imports = _dmu.check_imports

def _check_imports_allow_flash_attn(filename):
    try:
        return _orig_check_imports(filename)
    except ImportError as e:
        if "flash_attn" in str(e):
            from transformers.dynamic_module_utils import get_relative_imports
            return get_relative_imports(filename)
        raise

_dmu.check_imports = _check_imports_allow_flash_attn


def fix_and_register_ties(model, fwd_pattern="fwd", rev_pattern="rev"):
    """Ties fwd/rev weights in memory and generates the required _tied_weights_keys.

    Bidirectional models (like Caduceus-based HNet backbones) have mirrored
    forward/reverse parameters. After loading from a checkpoint, these may
    not be properly tied in memory, leading to ~2x parameter count and
    some parameters being uninitialized (zeros or random values).

    This function:
    1. Physically ties reverse parameters to forward parameters in memory
    2. Registers the tied keys so HuggingFace handles them correctly

    Parameters
    ----------
    model : nn.Module
        The model to fix weight tying for.
    fwd_pattern : str
        Pattern identifying forward-direction modules.
    rev_pattern : str
        Pattern identifying reverse-direction modules.

    Returns
    -------
    nn.Module
        The model with weight tying fixed.
    """
    tied_paths = []

    # Start with existing tied keys (like word embeddings) if present
    if hasattr(model, "_tied_weights_keys") and model._tied_weights_keys is not None:
        tied_paths.extend(model._tied_weights_keys)

    # Dictionary of all modules for quick lookup
    num_modules = dict(model.named_modules())
    ties_found = 0

    for name, module in num_modules.items():
        if fwd_pattern in name and "proj" in name:
            rev_name = name.replace(fwd_pattern, rev_pattern)

            if rev_name in num_modules:
                fwd_mod = module
                rev_mod = num_modules[rev_name]

                # Check if it's a layer with weights (Linear, Conv, etc.)
                if hasattr(fwd_mod, "weight"):
                    # Physical memory tie
                    rev_mod.weight = fwd_mod.weight
                    tied_paths.append(f"{rev_name}.weight")
                    ties_found += 1

                if hasattr(fwd_mod, "bias") and fwd_mod.bias is not None:
                    rev_mod.bias = fwd_mod.bias
                    tied_paths.append(f"{rev_name}.bias")

    # Deduplicate and assign to the magic HF attribute
    model._tied_weights_keys = list(set(tied_paths))

    if ties_found > 0:
        logger.info(f"Fixed {ties_found} weight ties (fwd/rev parameter pairs)")

    return model


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
    return (
        hasattr(model, "lm_head")
        and hasattr(model, "backbone")
        and hasattr(model, "embeddings")
    )


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

    # HF repository emarro/pcad2-200M-cnet-baseline has a bug in its config.json where it refers to
    # models_mixer_seq instead of mixer_seq. We patch this dynamically here to allow downloading from HF.
    if hasattr(config, "auto_map"):
        patched = False
        for k in list(config.auto_map.keys()):
            val = config.auto_map[k]
            if isinstance(val, str) and "models_mixer_seq." in val:
                config.auto_map[k] = val.replace("models_mixer_seq.", "mixer_seq.")
                patched = True
        if patched:
            logger.info(
                "Patched config.auto_map to fix HuggingFace repo typo (models_mixer_seq -> mixer_seq)"
            )

    if randomize:
        base_model = AutoModel.from_config(config, dtype=dtype, trust_remote_code=True)
    else:
        base_model = AutoModel.from_pretrained(
            path, config=config, revision=revision, trust_remote_code=True, dtype=dtype
        )

    # Fix weight tying for bidirectional models (fwd/rev parameter pairs)
    param_count_before = sum(p.numel() for p in base_model.parameters())
    base_model = fix_and_register_ties(base_model)
    param_count_after = sum(p.numel() for p in base_model.parameters())
    logger.info(
        f"Base model params: {param_count_before:,} (before tie fix) -> {param_count_after:,} (after tie fix)"
    )

    # Wrap CausalLM models to expose backbone hidden states
    if _needs_wrapper(base_model):
        base_model = BaseEncoderWrapper(base_model)

    return base_model
