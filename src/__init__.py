import inspect
import transformers
import transformers.generation

try:
    from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
except ImportError:
    from transformers.generation import GenerateDecoderOnlyOutput
    transformers.generation.GreedySearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
    transformers.generation.SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput

# ── Patch 1 ──────────────────────────────────────────────────────────────────
# transformers >= 4.51 renamed `_tied_weights_keys` → `all_tied_weights_keys`
# and changed it from a list to a dict.  Old remote models (HNetForCausalLM)
# still use the old attribute, so we bridge the two.
if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
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
    setattr(transformers.PreTrainedModel, "all_tied_weights_keys",
            property(fget=_get_all_tied, fset=_set_all_tied))

# ── Patch 2 ──────────────────────────────────────────────────────────────────
# transformers >= 4.51 calls tie_weights(missing_keys=..., recompute_mapping=False)
# but old remote model classes (e.g. HNetForCausalLM from pcad2) define
# tie_weights() with NO keyword arguments.
#
# Patching PreTrainedModel.tie_weights doesn't work because HNetForCausalLM
# overrides it — Python's MRO finds the subclass method first, bypassing us.
#
# Instead we patch _finalize_model_loading (the call site inside transformers)
# to shadow the model's tie_weights via an instance attribute *before* the call
# is made, so kwargs are silently filtered out for old model classes.
if hasattr(transformers.PreTrainedModel, "_finalize_model_loading"):
    _orig = transformers.PreTrainedModel._finalize_model_loading
    _orig_finalize = getattr(_orig, "__func__", _orig)

    def _safe_finalize_model_loading(cls, model, load_config, loading_info):
        # Resolve tie_weights through the real class hierarchy
        orig_tw = model.tie_weights  # bound method (from HNetForCausalLM or wherever)
        sig = inspect.signature(type(model).tie_weights)
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not has_var_kw:
            # Old model: shadow via instance attribute with a kwargs-filtering wrapper
            accepted = {k for k in sig.parameters if k != "self"}
            def _safe_tw(**kwargs):
                return orig_tw(**{k: v for k, v in kwargs.items() if k in accepted})
            model.tie_weights = _safe_tw  # instance attribute shadows the class method

        try:
            return _orig_finalize(cls, model, load_config, loading_info)
        except TypeError:
            return _orig_finalize(model, load_config, loading_info)

    transformers.PreTrainedModel._finalize_model_loading = classmethod(_safe_finalize_model_loading)
