import sys
import types


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def install_reelprotein_dependency_stubs() -> None:
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
    )
    torch_stub.device = lambda name: name
    torch_stub.no_grad = lambda: _NoGrad()
    sys.modules.setdefault("torch", torch_stub)

    xgboost_stub = types.ModuleType("xgboost")
    xgboost_stub.XGBClassifier = object
    sys.modules.setdefault("xgboost", xgboost_stub)

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.T5EncoderModel = object
    transformers_stub.T5Tokenizer = object
    sys.modules.setdefault("transformers", transformers_stub)

    huggingface_hub_stub = types.ModuleType("huggingface_hub")
    huggingface_hub_stub.snapshot_download = lambda *args, **kwargs: ""
    sys.modules.setdefault("huggingface_hub", huggingface_hub_stub)
