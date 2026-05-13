import inspect
import transformers
import transformers.generation

# ── Generation output compat shim ────────────────────────────────────────────
# transformers 4.x renamed/removed GreedySearchDecoderOnlyOutput and
# SampleDecoderOnlyOutput.  Patch them back so old remote model code doesn't
# crash on import.
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
# transformers uses lazy backend proxies: the ImportError ("PreTrainedModel
# requires the PyTorch library") is NOT raised when you import the class name,
# but only when you call hasattr/getattr on it.  The only reliable guard is to
# wrap the entire patch block in try/except ImportError.
try:
    _has_all_tied = hasattr(transformers.PreTrainedModel, "all_tied_weights_keys")
    _has_finalize = hasattr(transformers.PreTrainedModel, "_finalize_model_loading")
except ImportError:
    # PyTorch not available (e.g., CPU-only CI runner) — skip all patches.
    _has_all_tied = False
    _has_finalize = False

if not _has_all_tied:
    try:

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
            transformers.PreTrainedModel,
            "all_tied_weights_keys",
            property(fget=_get_all_tied, fset=_set_all_tied),
        )
    except ImportError:
        pass

if _has_finalize:
    try:
        _orig_finalize = getattr(
            transformers.PreTrainedModel._finalize_model_loading,
            "__func__",
            transformers.PreTrainedModel._finalize_model_loading,
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

        transformers.PreTrainedModel._finalize_model_loading = (  # pyrefly: ignore[bad-assignment]
            classmethod(_safe_finalize_model_loading)
        )
    except ImportError:
        pass
