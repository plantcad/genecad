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
import importlib.abc
import importlib.machinery

class _Stub(ModuleType):
    """Stub module: CapitalCase attrs → empty class; others → child stub."""
    _NONE_ATTRS = frozenset({"__file__", "__cached__", "__path__", "__package__"})
    def __getattr__(self, name):
        if name in self._NONE_ATTRS:
            return None
        if name and name[0].isupper():
            val = type(name, (), {"__init__": lambda s, *a, **kw: None})
        else:
            val = _Stub(f"{self.__name__}.{name}")
        object.__setattr__(self, name, val)
        return val
    def __call__(self, *a, **kw):
        return None

class _FlashAttnLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return _Stub(spec.name)
    def exec_module(self, module):
        if module.__name__ == "flash_attn":
            import torch as _torch
            import torch.nn.functional as _F

            def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None,
                                           causal=False, **kw):
                """qkv: (batch, seqlen, 3, nheads, headdim)"""
                q, k, v = qkv.unbind(2)
                return _F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    dropout_p=dropout_p, scale=softmax_scale, is_causal=causal,
                ).transpose(1, 2)

            def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen,
                                                  dropout_p=0.0, softmax_scale=None,
                                                  causal=False, **kw):
                """qkv: (total_tokens, 3, nheads, headdim)"""
                q, k, v = qkv.unbind(1)
                batch = len(cu_seqlens) - 1
                outs = []
                for i in range(batch):
                    s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                    qi = q[s:e].unsqueeze(0).transpose(1, 2)
                    ki = k[s:e].unsqueeze(0).transpose(1, 2)
                    vi = v[s:e].unsqueeze(0).transpose(1, 2)
                    oi = _F.scaled_dot_product_attention(
                        qi, ki, vi, dropout_p=dropout_p,
                        scale=softmax_scale, is_causal=causal,
                    )
                    outs.append(oi.squeeze(0).transpose(0, 1))
                return _torch.cat(outs, dim=0)

            def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None,
                                          causal=False, **kw):
                """kv: (batch, seqlen_k, 2, nheads, headdim)"""
                k, v = kv.unbind(2)
                return _F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    dropout_p=dropout_p, scale=softmax_scale, is_causal=causal,
                ).transpose(1, 2)

            def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k,
                                                 max_seqlen_q, max_seqlen_k,
                                                 dropout_p=0.0, softmax_scale=None,
                                                 causal=False, **kw):
                k, v = kv.unbind(2)
                return _F.scaled_dot_product_attention(
                    q.unsqueeze(0).transpose(1, 2),
                    k.unsqueeze(0).transpose(1, 2),
                    v.unsqueeze(0).transpose(1, 2),
                    dropout_p=dropout_p, scale=softmax_scale, is_causal=causal,
                ).transpose(1, 2).squeeze(0)

            def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None,
                                         softmax_scale=None, causal=False, **kw):
                k_use = k if k is not None else k_cache
                v_use = v if v is not None else v_cache
                return _F.scaled_dot_product_attention(
                    q.transpose(1, 2), k_use.transpose(1, 2), v_use.transpose(1, 2),
                    scale=softmax_scale, is_causal=causal,
                ).transpose(1, 2)

            module.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
            module.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
            module.flash_attn_kvpacked_func = flash_attn_kvpacked_func
            module.flash_attn_varlen_kvpacked_func = flash_attn_varlen_kvpacked_func
            module.flash_attn_with_kvcache = flash_attn_with_kvcache

        elif module.__name__ == "flash_attn.ops.activations":
            import torch.nn.functional as _F
            def swiglu(gate, x):
                return x * _F.silu(gate)
            module.swiglu = swiglu

        elif module.__name__ == "flash_attn.ops.triton.layer_norm":
            import torch as _torch
            import torch.nn as _nn

            class RMSNorm(_nn.Module):
                """Pure-PyTorch RMSNorm replacing flash_attn triton kernel on ARM64."""
                def __init__(self, hidden_size, eps=1e-5, dropout_p=0.0,
                             device=None, dtype=None, **kw):
                    super().__init__()
                    self.weight = _nn.Parameter(
                        _torch.ones(hidden_size, device=device, dtype=dtype)
                    )
                    self.eps = eps

                def forward(self, x, residual=None, prenorm=False,
                            residual_in_fp32=False, **kw):
                    if residual is not None:
                        x = x + residual.to(x.dtype)
                    res = x.float() if residual_in_fp32 else x
                    x_f = x.float()
                    x_norm = x_f * _torch.rsqrt(
                        x_f.pow(2).mean(-1, keepdim=True) + self.eps
                    )
                    out = self.weight * x_norm.to(self.weight.dtype)
                    return (out, res) if prenorm else out

            module.RMSNorm = RMSNorm

class _FlashAttnFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "flash_attn" or fullname.startswith("flash_attn."):
            return importlib.machinery.ModuleSpec(
                fullname, _FlashAttnLoader(), is_package=True
            )
        return None

try:
    import flash_attn  # noqa: F401
except ImportError:
    sys.meta_path.insert(0, _FlashAttnFinder())

if platform.machine() == "aarch64":
    try:
        import triton  # noqa: F401
        import triton.runtime.autotuner as _triton_autotuner

        # torch 2.7.1 expects triton.compiler.compiler.triton_key, which doesn't
        # exist in triton 3.5.0. Provide a shim so torch.inductor can JIT-compile
        # standard ops (attention, MLP) via triton on GH200.
        try:
            import triton.compiler.compiler as _tcc
            if not hasattr(_tcc, "triton_key"):
                import hashlib as _hashlib
                def _triton_key():
                    return _hashlib.sha256(triton.__version__.encode()).hexdigest()[:16]
                _tcc.triton_key = _triton_key
        except Exception:
            pass

        # GH200 (SM90a): mamba_ssm 2.2.4's default autotune configs all trigger
        # CUDA illegal memory access. Fix by:
        # 1. Replacing triton.autotune to emit only a single conservative config
        #    for any kernel decorated *after* this point (covers mamba_ssm imports
        #    that happen later, via model loading).
        # 2. Clearing configs on any mamba_ssm Autotuner objects already imported.
        # 3. Wrapping _bench to recover gracefully if a config still errors.
        _orig_autotune = triton.autotune
        def _conservative_autotune(configs, key, **kwargs):
            # Keep original constexpr kwargs (BLOCK_SIZE_H etc.) from the first
            # config but drop to a single, conservative warp/stage count.
            # An empty Config({}) loses required kernel constexprs on GH200.
            first = configs[0] if configs else triton.Config({})
            safe = triton.Config(
                dict(getattr(first, "kwargs", {}) or {}),
                num_warps=4,
                num_stages=1,
            )
            return _orig_autotune([safe], key, **kwargs)
        triton.autotune = _conservative_autotune

        _safe_cfg = triton.Config({}, num_warps=4, num_stages=1)

        _orig_bench = _triton_autotuner.Autotuner._bench
        def _safe_bench(self, *args, config=None, **kwargs):
            try:
                return _orig_bench(self, *args, config=config, **kwargs)
            except (RuntimeError, Exception) as _e:
                if any(x in str(_e) for x in ("CUDA", "illegal memory", "Triton Error")):
                    try:
                        import torch as _t
                        _t.cuda.synchronize()
                    except RuntimeError:
                        pass
                    return float("inf")
                raise
        _triton_autotuner.Autotuner._bench = _safe_bench

        def _repatch_existing_autotuners():
            for _mod in list(sys.modules.values()):
                if _mod is None:
                    continue
                _mname = getattr(_mod, "__name__", "") or ""
                if "mamba_ssm.ops.triton" not in _mname:
                    continue
                for _attr in dir(_mod):
                    _obj = getattr(_mod, _attr, None)
                    if isinstance(_obj, _triton_autotuner.Autotuner):
                        _obj.configs = [_safe_cfg]
                        if hasattr(_obj, "cache"):
                            _obj.cache.clear()
        _repatch_existing_autotuners()

        # GH200 (SM90a): mamba_ssm 2.2.4's triton kernels fail at load_binary
        # (wgmma / tl.dot paths can't be loaded). Mamba2.forward checks
        # self.use_mem_eff_path (not use_fast_path) to decide between the fused
        # triton path and the pure-PyTorch reference. Replace the fused function
        # in mamba2's module namespace so the call at forward:31 gets the ref.
        try:
            import inspect as _inspect
            import torch.nn.functional as _F
            import mamba_ssm.ops.triton.ssd_combined as _ssd_comb
            import mamba_ssm.ops.selective_scan_interface as _scan_iface
            import mamba_ssm.modules.mamba2 as _mamba2_mod

            # selective_scan_cuda works on GH200 — keep the CUDA ext (faster).
            # Only mamba2's triton-fused combined kernel fails; ref path uses CUDA ext.

            # layernorm_gated._layer_norm_fwd_1pass_kernel also fails at execution on
            # GH200. Replace rmsnorm_fn in ssd_combined's namespace with pure PyTorch.
            def _pure_rmsnorm_fn(x, weight, bias, z=None, eps=1e-5, group_size=None,
                                   norm_before_gate=True, is_rms_norm=True):
                _dtype = x.dtype
                x = x.float()
                if z is not None:
                    z = z.float()
                if norm_before_gate or z is None:
                    x_n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
                    if weight is not None:
                        x_n = x_n * weight.float()
                    if bias is not None:
                        x_n = x_n + bias.float()
                    if z is not None:
                        x_n = x_n * _F.silu(z)
                else:
                    x = x * _F.silu(z)
                    x_n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
                    if weight is not None:
                        x_n = x_n * weight.float()
                    if bias is not None:
                        x_n = x_n + bias.float()
                return x_n.to(_dtype)

            _ssd_comb.rmsnorm_fn = _pure_rmsnorm_fn

            _ref_fn = _ssd_comb.mamba_split_conv1d_scan_ref
            _ref_params = set(_inspect.signature(_ref_fn).parameters)

            def _compat_ref(*args, **kwargs):
                # dt_bias (positional arg 3) must be float32 for the ref impl
                args = list(args)
                if len(args) > 3 and args[3] is not None:
                    args[3] = args[3].float()
                return _ref_fn(*args, **{k: v for k, v in kwargs.items() if k in _ref_params})

            _mamba2_mod.mamba_split_conv1d_scan_combined = _compat_ref
            _ssd_comb.mamba_split_conv1d_scan_combined = _compat_ref
        except (ImportError, AttributeError):
            pass

    except ImportError:
        class _TritonConfig:
            def __init__(self, kwargs, num_warps=4, num_stages=2, num_ctas=1,
                         pre_hook=None, **kw):
                self.kwargs = kwargs
                self.num_warps = num_warps
                self.num_stages = num_stages
                self.num_ctas = num_ctas

        class _TritonLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return _Stub(spec.name)
            def exec_module(self, module):
                pass

        class _TritonFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "triton" or fullname.startswith("triton."):
                    return importlib.machinery.ModuleSpec(
                        fullname, _TritonLoader(), is_package=True
                    )
                return None

        sys.meta_path.insert(0, _TritonFinder())

        _triton = _Stub("triton")
        _triton.__version__ = "3.3.0"
        _triton.__spec__ = importlib.machinery.ModuleSpec(
            "triton", _TritonLoader(), is_package=True
        )
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
