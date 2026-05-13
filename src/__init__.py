import inspect
import transformers
import transformers.generation

# ── Generation output compat shim ────────────────────────────────────────────
# transformers 4.x renamed/removed the legacy GreedySearchDecoderOnlyOutput and
# SampleDecoderOnlyOutput classes.  We patch them back so that old remote model
# code can still reference them without crashing on import.
try:
    from transformers.generation import (  # pyrefly: ignore[missing-module-attribute]
        GreedySearchDecoderOnlyOutput,  # noqa: F401
        SampleDecoderOnlyOutput,  # noqa: F401
    )
except ImportError:
    try:
        from transformers.generation import GenerateDecoderOnlyOutput

        transformers.generation.GreedySearchDecoderOnlyOutput = (
            GenerateDecoderOnlyOutput
        )
        transformers.generation.SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput
    except ImportError:
        from dataclasses import dataclass

        @dataclass
        class DummyDecoderOnlyOutput:
            pass

        transformers.generation.GreedySearchDecoderOnlyOutput = DummyDecoderOnlyOutput
        transformers.generation.SampleDecoderOnlyOutput = DummyDecoderOnlyOutput

# ── Patch 1 & 2: PreTrainedModel compatibility ────────────────────────────────
# These patches require PyTorch.  In CPU-only CI environments, accessing
# transformers.PreTrainedModel raises ImportError ("requires the PyTorch
# library"), so we guard the entire block with try/except.
try:
    _PTM = transformers.PreTrainedModel  # triggers the PyTorch backend check
except ImportError:
    _PTM = None  # type: ignore[assignment]

if _PTM is not None:
    # Patch 1: transformers >= 4.51 renamed `_tied_weights_keys`
    # → `all_tied_weights_keys` and changed it from a list to a dict.
    # Old remote models (HNetForCausalLM) still use the old attribute, so we
    # bridge the two.
    if not hasattr(_PTM, "all_tied_weights_keys"):

        def _get_all_tied(self):
            if hasattr(self, "_all_tied_storage"):
                return self._all_tied_storage
            val = getattr(self, "_tied_weights_keys", None)
            if val is None:
                return {}
            if isinstance(val, list):
                return {k: None for k in val}
            return val

        def _set_all_tied(self, value):
            self._all_tied_storage = value

        setattr(
            _PTM,
            "all_tied_weights_keys",
            property(fget=_get_all_tied, fset=_set_all_tied),
        )

    # Patch 2: transformers >= 4.51 calls
    # tie_weights(missing_keys=..., recompute_mapping=False) but old remote
    # model classes define tie_weights() with NO keyword arguments.  We patch
    # _finalize_model_loading to filter kwargs for old model classes.
    if hasattr(_PTM, "_finalize_model_loading"):
        _orig_finalize = getattr(
            _PTM._finalize_model_loading, "__func__", _PTM._finalize_model_loading
        )

        def _safe_finalize_model_loading(cls, model, load_config, loading_info):
            orig_tw = model.tie_weights
            sig = inspect.signature(type(model).tie_weights)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if not has_var_kw:
                accepted = {k for k in sig.parameters if k != "self"}

                def _safe_tw(**kwargs):
                    return orig_tw(**{k: v for k, v in kwargs.items() if k in accepted})

                model.tie_weights = _safe_tw

            try:
                return _orig_finalize(  # pyrefly: ignore[bad-argument-count]
                    cls, model, load_config, loading_info
                )
            except TypeError:
                return _orig_finalize(model, load_config, loading_info)

        _PTM._finalize_model_loading = classmethod(  # pyrefly: ignore[bad-assignment]
            _safe_finalize_model_loading
        )
